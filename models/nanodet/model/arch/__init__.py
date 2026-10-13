import copy
from .one_stage_detector import OneStageDetector

def build_model(model_cfg):
    model_cfg = copy.deepcopy(model_cfg)
    name = model_cfg.arch.pop("name")
    if name == "OneStageDetector":
        model = OneStageDetector(
            model_cfg.arch.backbone, model_cfg.arch.fpn, model_cfg.arch.head
        )
    else:
        raise NotImplementedError
    return model