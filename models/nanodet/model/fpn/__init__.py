import copy
from .axon_fpn import AxonFPN

def build_fpn(config):
    fpn_config = copy.deepcopy(config)

    name = fpn_config.pop("name")

    if name == "AxonFPN":
        encoder = AxonFPN(**fpn_config)
    else:
        raise NotImplementedError

    return encoder