from typing import Any

import optax
from jax import numpy as jnp

from utils import build_lr_schedule, AuxiliaryLossData


def build_optimizer(opt_config, initial_params):
    if opt_config.name == "adam":
        optimizer = optax.adam(
            build_lr_schedule(opt_config.learning_rate, opt_config.decay)
        )
        initial_opt_state = optimizer.init(initial_params)
    else:
        raise "Optimizer type not yet implemented"
    return optimizer, initial_opt_state


def build_loss_fn(solver):
    def _loss(params, model_state, rng, batch) -> Any:
        (y, _, g_t), state = solver(params, model_state, rng, *batch, True)

        diff = g_t - y
        var_loss, mc_var_estimate = jnp.var(diff), jnp.mean(diff)
        mc_estimate = jnp.mean(g_t)
        return var_loss, (
            state,
            AuxiliaryLossData(
                var_loss=var_loss,
                mc_estimate=mc_estimate,
                mc_var_estimate=mc_var_estimate,
            ),
        )

    return _loss
