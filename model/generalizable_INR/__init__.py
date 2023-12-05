from .transinr import TransINR
from .meta_low_rank_modulated_inr import MetaLowRankModulatedINR

def transinr(config):
    return TransINR(config)


def meta_low_rank_modulated_inr(config):
    return MetaLowRankModulatedINR(config)
