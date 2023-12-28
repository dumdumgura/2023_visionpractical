import torch

def create_inr_optimizer(model, config):
    optimizer_type = config.type.lower()
    if optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.init_lr, weight_decay=config.weight_decay, betas=config.betas
        )
    elif optimizer_type == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.init_lr, weight_decay=config.weight_decay, betas=config.betas
        )
    elif optimizer_type == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=config.init_lr, weight_decay=config.weight_decay, momentum=0.9
        )
    elif optimizer_type =='overfit':
        optimizer = torch.optim.Adam(
            model.factors.init_modulation_factors.parameters(),lr=config.init_lr, weight_decay=config.weight_decay, betas=config.betas
        )
    elif optimizer_type =='overfit_hyper':
        optimizer = torch.optim.SGD(
            model.specialized_factor.parameters(), lr=config.init_lr, weight_decay=config.weight_decay, momentum=0.9
        )
    elif optimizer_type =='adam_hyper':
        params = [
       # Adjust the learning rate as needed
            {'params': model.encoder.parameters(), 'lr': 0.0001}, #pointnet2
            {'params': model.hyponet.parameters(), 'lr': 0.0001}, #shared layer
            {'params': model.transformer.parameters(), 'lr': 0.00001},# Set a smaller learning rate for the transformer
            #{'params': model.weight_groups.parameters(), 'lr': 0.00001}, #shared initialization
        ]

        optimizer = torch.optim.Adam(
            params, weight_decay=config.weight_decay, betas=config.betas
        )

    else:
        raise ValueError(f"{optimizer_type} invalid..")
    return optimizer


def create_optimizer(model, config):
    arch_type = config.arch.type.lower()
    if "inr" in config.arch.type:
        optimizer = create_inr_optimizer(model, config.optimizer)
    else:
        raise ValueError(f"{arch_type} invalid..")
    return optimizer
