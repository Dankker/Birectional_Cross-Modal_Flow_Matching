"""Model construction kept intentionally specific to the released XL model."""

from libs.model.flowtok_t2i import FlowTok_XL


def build_field(config):
    if config.nnet.name != "flowtok-xl":
        raise ValueError(
            f"BVFM-image supports flowtok-xl, got {config.nnet.name!r}")
    return FlowTok_XL(config.nnet.model_args)
