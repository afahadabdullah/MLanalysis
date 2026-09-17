"""Compatibility shim mapping graphcast imports to weathernext."""

try:
    from weathernext.weathernext1_graph import graphcast as _model
    from weathernext.weathernext1_graph.graphcast import *
    from weathernext.utils import (
        autoregressive,
        casting,
        checkpoint,
        data_utils,
        normalization,
        rollout,
    )
except ImportError:
    pass
