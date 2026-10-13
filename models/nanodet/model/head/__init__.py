import copy 
from .axon_nanodet_head import AxonNanoDetHead
from .gfl_head import GFLHead

def build_head(config):
    head_config = copy.deepcopy(config)
    name = head_config.pop("name")

    if name == "GFLHead":
        head = GFLHead(**head_config)
    elif name == "AxonNanoDetHead":
        head = AxonNanoDetHead(**head_config)
    else:
        raise NotImplementedError

    return head