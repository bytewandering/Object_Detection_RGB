import copy
from .nanodet.model.arch import build_model as build_nanodet
from .fomo import FOMO

def build_detector(config):
    detector_config = copy.deepcopy(config)
    name = detector_config.pop("name")

    if name == "Nanodet":
        detector = build_nanodet(detector_config)
    elif name == "FOMO":
        detector = FOMO(**detector_config)
    else:
        raise NotImplementedError

    return detector