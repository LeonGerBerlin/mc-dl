import os
import logging
import time
import yaml

from ml_collections import config_dict

import jax.numpy as jnp
import jax
import optax
import haiku as hk

from optimization import build_optimizer, build_loss_fn
from model import CVSolver
from utils import TrainingState


def build_solver(rng, model_config, pde_config, batch_size):
    x_init = init_x0(pde_config, 1)
    rng, rng_sampling = jax.random.split(rng, num=2)

    def forward(x, key, is_training):
        mdl = CVSolver(model_config, pde_config, batch_size)
        return mdl(x, key, is_training)

    model = hk.transform_with_state(forward)
    params, state = model.init(rng, x_init, rng_sampling, True)

    solver = lambda params, state, rng, x_init, rng_sampling, is_training: model.apply(
        params, state, rng, x_init, rng_sampling, is_training
    )
    return solver, params, state


def init_x0(pde_config, batch_size):
    if pde_config.name == "Heston":
        # Asset process
        ass = jnp.ones([batch_size, 1]) * pde_config.parameter.spot

        # Variance process
        var = jnp.ones([batch_size, 1]) * pde_config.parameter.v0

        # Combine both process x0
        x0 = jnp.concatenate([var, ass], axis=1)
    else:
        x0 = (
            jnp.ones([batch_size, pde_config.parameter.dim]) * pde_config.parameter.spot
        )
    return x0


if __name__ == "__main__":
    # Init
    rng = 12345
    rng, subkey = jax.random.split(jax.random.PRNGKey(rng))

    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler("debug.log"), logging.StreamHandler()],
    )

    logger = logging.getLogger()
    logger.info("Starting computation...")

    # Configs
    with open("configs/config.yaml", "r") as stream:
        cfg_dict = yaml.safe_load(stream)
    config = config_dict.FrozenConfigDict(cfg_dict)

    # Build model & initial option price
    solver, initial_params, initial_model_state = build_solver(
        subkey, config.model, config.pde, config.optimization.batch_size
    )
    x_init = init_x0(config.pde, config.optimization.batch_size)

    # Build optimizer/ jax stuff
    loss_fn = build_loss_fn(solver)
    optimizer, initial_opt_state = build_optimizer(
        config.optimization.optimizer, initial_params
    )

    training_state = TrainingState(
        initial_params, initial_model_state, initial_opt_state
    )

    # Single optimization step
    @jax.jit
    def _update_first_order(train_state, rng, batch):
        grads, (model_state, loss_data) = jax.grad(loss_fn, has_aux=True)(
            train_state.params, train_state.model_state, rng, batch
        )
        updates, opt_state = optimizer.update(grads, train_state.opt_state)
        params = optax.apply_updates(train_state.params, updates)
        return params, opt_state, model_state, loss_data

    def update(train_state: TrainingState, rng, batch):  # batch = (xinit, rng)
        params, opt_state, model_state, loss_data = _update_first_order(
            train_state, rng, batch
        )

        return TrainingState(params, model_state, opt_state), loss_data

    # Training loop
    for e in range(config.optimization.epochs):
        start = time.time()

        rng, subkey, subkey2 = jax.random.split(rng, num=3)
        training_state, loss_data = update(training_state, subkey, (x_init, subkey2))

        if e % config.logging.log_every == 0:
            logger.info(
                f"Nb. epoch: {e},"
                f" Time per epoch: {time.time() - start}, "
                f"Variance loss: {loss_data.var_loss}, "
                f"MC estimate: {loss_data.mc_estimate},"
            )
