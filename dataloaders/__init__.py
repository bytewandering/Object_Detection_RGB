import copy
from .DareLab_Face.dataloader import DareLabFaceDataLoader


def build_dataloader(config, mode="train", pin_memory=None):
    dataloader_config = copy.deepcopy(config)
    name = dataloader_config.pop("name")

    if pin_memory is not None:
        dataloader_config["pin_memory"] = pin_memory

    if name in ("DareLabFaceDataset", "DareLabFaceDataLoader", "DareLab_Face"):
        dataloader = DareLabFaceDataLoader(**dataloader_config, mode=mode)
    else:
        raise NotImplementedError

    return dataloader
