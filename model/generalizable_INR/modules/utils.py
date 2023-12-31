import math
import torch
import torch.nn as nn

from .layers import Sine


def convert_int_to_list(size, len_list=2):
    if isinstance(size, int):
        return [size] * len_list
    else:
        assert len(size) == len_list
        return size


def initialize_params(params, init_type, type,**kwargs):
    fan_in, fan_out = params.shape[0], params.shape[1]
    if init_type is None or init_type == "normal":
        nn.init.normal_(params)
    elif init_type == "kaiming_uniform":
        nn.init.kaiming_uniform_(params, a=math.sqrt(5))
    elif init_type == "uniform_fan_in":
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(params, -bound, bound)
    elif init_type == "zero":
        nn.init.zeros_(params)

    elif "siren" == init_type:
        assert "siren_w0" in kwargs.keys() and "is_first" in kwargs.keys()
        if type == 'weight':
            w0 = kwargs["siren_w0"]

            if kwargs["is_first"]:
                w_std = 1 / fan_in
            else:
                w_std = math.sqrt(6.0 / fan_in) / w0

            nn.init.uniform_(params, -w_std, w_std)



        elif type == 'bias':
            w0 = kwargs["siren_w0"]

            if kwargs["is_first"]:
                w_std = math.sqrt(1 / 3)
            else:
                w_std = math.sqrt(1.0 / 256)

            nn.init.uniform_(params, -w_std, w_std)

            print("-baselayer-")
            print(w_std)
            print(fan_in)
            print(fan_out)
            print("---")
            print(params.min())
            print(params.max())
            print(params.mean())
            print(params.var())
            #print(params.norm())

    else:
        raise NotImplementedError



def create_params_with_init(shape, init_type="normal", include_bias=False, bias_init_type="zero", **kwargs):
    if not include_bias:
        params = torch.empty([shape[0], shape[1]])
        initialize_params(params, init_type, **kwargs)
        return params
    else:
        params = torch.empty([shape[0] - 1, shape[1]])
        bias = torch.empty([1, shape[1]])

        initialize_params(params, init_type, 'weight', **kwargs)
        initialize_params(bias, bias_init_type, 'bias', **kwargs)
        return torch.cat([params, bias], dim=0)


def create_activation(config):
    if config.type == "relu":
        activation = nn.ReLU()
    elif config.type == "siren":
        activation = Sine(config.siren_w0)
    elif config.type == 'silu':
        activation = nn.SiLU()
    else:
        raise NotImplementedError
    return activation
