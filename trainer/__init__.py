from .trainer_stage_meta_inr import Trainer as TrainerMetaINR
STAGE_META_INR_ARCH_TYPE = ["meta_low_rank_modulated_inr"]

def create_trainer(config):
    if config.arch.type in STAGE_META_INR_ARCH_TYPE:
        return TrainerMetaINR

    else:
        raise ValueError("architecture type not supported")
