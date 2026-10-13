import copy
from .axon_backbone import AxonBackbone

def build_backbone(config):
    backbone_config = copy.deepcopy(config)
    name = backbone_config.pop("name")

    if name == "AxonBackbone":
        backbone = AxonBackbone(**backbone_config)

    else:
        raise NotImplementedError

    return backbone