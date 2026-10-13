"""torch.ao QAT configurations for numeric bit-width experiments on the detector.

Adapted from Adaptive_Compression; the model here is a bare OneStageDetector,
which enters the quantized domain through its own QuantStub.
INT16/24 use qint32 as the observer carrier, not 32-bit quantization ranges.
All modes here are fake-quant training; this is not an integer export backend.
"""

import torch
import torch.ao.quantization as aq


MODES = {"fp32": None, "int24": 24, "int16": 16, "int8": 8}

# Losses/postprocessing are floating-point; Integral is used only there.
FLOAT_HEAD_MODULES = ("distribution_project", "loss_qfl", "loss_dfl", "loss_bbox")


def make_qconfig(mode):
    if mode not in MODES:
        raise ValueError(f"mode must be one of {tuple(MODES)}, got {mode!r}")
    bits = MODES[mode]
    if bits is None:
        return None
    dtype = torch.qint8 if bits == 8 else torch.qint32
    limits = dict(quant_min=-(1 << (bits - 1)), quant_max=(1 << (bits - 1)) - 1,
                  dtype=dtype, reduce_range=False, eps=torch.finfo(torch.float32).tiny)
    return aq.QConfig(
        activation=aq.FakeQuantize.with_args(
            observer=aq.MovingAverageMinMaxObserver,
            qscheme=torch.per_tensor_symmetric, **limits),
        weight=aq.FakeQuantize.with_args(
            observer=aq.MovingAveragePerChannelMinMaxObserver,
            qscheme=torch.per_channel_symmetric, ch_axis=0, **limits),
    )


def prepare_model_for_qat(model, mode):
    qconfig = make_qconfig(mode)
    if qconfig is None:
        return model
    model.train()
    model.fuse_model(qat=True)
    model.qconfig = qconfig
    # NanoDet heads carry these; FOMO computes its loss outside the module tree.
    for name in FLOAT_HEAD_MODULES:
        if hasattr(model.head, name):
            getattr(model.head, name).qconfig = None
    aq.prepare_qat(model, inplace=True)
    return model


def set_quantization_mode(model, mode, reset_observers=True):
    """Change bit width in place; never replace Parameters or optimizer state.

    The model must have been prepared once, before optimizer construction.
    Python range/dtype attributes must also be restored when loading a checkpoint.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown quantization mode: {mode}")
    if mode == "fp32":
        model.apply(aq.disable_fake_quant)
        model.apply(aq.disable_observer)
        return
    bits = MODES[mode]
    low, high = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    dtype = torch.qint8 if bits == 8 else torch.qint32
    for fake in model.modules():
        if not isinstance(fake, aq.FakeQuantize):
            continue
        observer = fake.activation_post_process
        fake.quant_min = observer.quant_min = low
        fake.quant_max = observer.quant_max = high
        fake.dtype = observer.dtype = dtype
        if reset_observers:
            observer.reset_min_max_vals()
            # Some PyTorch versions recreate per-channel min/max buffers on CPU.
            observer.to(device=fake.scale.device)
            fake.scale.fill_(1.0)
            fake.zero_point.zero_()
        fake.enable_fake_quant()
        fake.enable_observer()
    config = make_qconfig(mode)
    for module in model.modules():
        if getattr(module, "qconfig", None) is not None:
            module.qconfig = config
