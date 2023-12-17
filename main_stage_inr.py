import argparse
import math

import torch
import torch.distributed as dist
import wandb

import utils.dist as dist_utils

from model import create_model
from trainer import create_trainer, STAGE_META_INR_ARCH_TYPE
from dataset import create_dataset
from optimizer import create_optimizer, create_scheduler
from utils.utils import set_seed
from utils.profiler import Profiler
from utils.setup import setup
import yaml


def default_parser():
    parser = argparse.ArgumentParser()
    #parser.add_argument("-m", "--model-config", type=str, default="./configs/meta_learning/low_rank_modulated_meta/imagenette178_meta_low_rank.yaml")
    #parser.add_argument("-m", "--model-config", type=str,default="./configs/meta_learning/low_rank_modulated_meta/shapenet_meta.yaml")
    parser.add_argument("-m", "--model-config", type=str,default="./config/shapenet_transinr.yaml")
    parser.add_argument("-r", "--result-path", type=str, default="./exp_week8/")
    parser.add_argument("-t", "--task", type=str, default="tsdf0.3_PE0")

    #parser.add_argument("-l", "--load-path", type=str, default="/home/umaru/PycharmProjects/meta_shaope/exp_week7/shapenet_meta_sdf/tsdf0.3_PE0_2/epoch740_model.pt")
    parser.add_argument("-l", "--load-path", type=str,default="")
    parser.add_argument("-p", "--postfix", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--resume", action="store_true",default=False)
    return parser


def add_dist_arguments(parser):
    parser.add_argument("--world_size", default=0, type=int, help="number of nodes for distributed training")
    parser.add_argument("--local_rank", default=1, type=int, help="local rank for distributed training")
    parser.add_argument("--node_rank", default=0, type=int, help="node rank for distributed training")
    parser.add_argument("--nnodes", default=1, type=int)
    parser.add_argument("--nproc_per_node", default=1, type=int)
    parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument("--timeout", type=int, default=1, help="time limit (s) to wait for other nodes in DDP")
    return parser

def parse_args():
    parser = default_parser()
    parser = add_dist_arguments(parser)
    args, extra_args = parser.parse_known_args()
    return args, extra_args


if __name__ == "__main__":
    print(torch.cuda.is_available())
    args, extra_args = parse_args()
    set_seed(args.seed)
    config, logger, writer = setup(args, extra_args)
    
    #init wandb
    run = wandb.init(
        # Set the project where this run will be logged
        project = "ginr_simplified_2",
        # Identifier for the current run
        #notes = config.experiment.name,
        # Track hyperparameters and run metadata
        config = yaml.safe_load(open(args.model_config))
    )

    distenv = config.runtime.distenv
    profiler = Profiler(logger)

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda", distenv.local_rank)
    torch.cuda.set_device(device)

    dataset_trn, dataset_val = create_dataset(config, is_eval=args.eval, logger=logger)

    model = create_model(config.arch)
    model = model.to(device)

    if distenv.master:
        print(model)
        profiler.get_model_size(model)
        profiler.get_model_size(model, opt="trainable-only")

    # Checkpoint loading
    if not args.load_path == "":
        ckpt = torch.load(args.load_path, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"], strict=False)

        if distenv.master:
            logger.info(f"{args.load_path} model is loaded")
    else:
        ckpt = None
        if args.eval or args.resume:
            raise ValueError("--load-path must be specified in evaluation or resume mode")

    # Optimizer definition
    if args.eval:
        optimizer, scheduler, epoch_st = None, None, None

    else:
        steps_per_epoch = math.ceil(len(dataset_trn) / (config.experiment.batch_size * distenv.world_size))
        steps_per_epoch = steps_per_epoch // config.optimizer.grad_accm_steps


        # tmp patch

        if config.type == 'overfit' and config.arch.type == 'low_rank_modulated_transinr':
            model.init_factor_zero()
            loader_trn = torch.utils.data.DataLoader(
                dataset_trn,
                # sampler=self.sampler_trn,
                shuffle=True,
                pin_memory=True,
                batch_size=config.experiment.batch_size,
                # num_workers=num_workers,
            )
            for xt in loader_trn:
                if config.dataset.type == "shapenet":
                    if config.dataset.supervision == 'sdf' or config.dataset.supervision == 'occ':
                        coord_inputs = xt['coords'].to(device)

                    elif config.dataset.supervision == 'siren_sdf':
                        coords, xs = xt
                        coord_inputs = coords['coords'].to(device)
                model.init_factor(coord_inputs)
                break



        optimizer = create_optimizer(model, config)
        scheduler = create_scheduler(
            optimizer, config.optimizer.warmup, config.optimizer.warmup.step_size, config.experiment.epochs_cos, distenv
        )
        
        if distenv.master:
            print(optimizer)

        if args.resume:
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            epoch_st = ckpt["epoch"]

            if distenv.master:
                logger.info(f"Optimizer, scheduler, and epoch is resumed")
                logger.info(f"resuming from {epoch_st}..")
        else:
            epoch_st = 0        

    # Usual DDP setting
    static_graph = config.arch.type in STAGE_META_INR_ARCH_TYPE # use static_graph for high-order gradients in meta-learning
    #model = dist_utils.dataparallel_and_sync(distenv, model, static_graph=static_graph)

    trainer = create_trainer(config)
    trainer = trainer(model, dataset_trn, dataset_val, config, writer, device, distenv, wandb=run)

    if distenv.master:
        logger.info(f"Trainer created. type: {trainer.__class__}")

    run.watch(model, log_freq=config.experiment.test_freq, log="all")
    
    if args.eval:
        trainer.config.experiment.subsample_during_eval = False
        trainer.eval(valid=False, verbose=True)
        trainer.eval(valid=True, verbose=True)
    else:
        trainer.run_epoch(optimizer, scheduler, epoch_st)

    #dist.barrier()

    if distenv.master:
        writer.close()
