from typing import NamedTuple, Any

import chex
import haiku as hk


class TrainingState(NamedTuple):
    params: hk.Params
    model_state: hk.State
    opt_state: Any


@chex.dataclass
class AuxiliaryLossData:
    var_loss: Any
    mc_estimate: Any
    mc_var_estimate: Any


def build_lr_schedule(base_lr, decay):
    return lambda t: base_lr / (1 + t / decay)
