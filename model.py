import jax
import jax.numpy as jnp

import jax
import haiku as hk

import pde


class SubNetwork(hk.Module):
    def __init__(self, config, dim=1, name=None) -> None:
        super().__init__(name=name)

        self.output_sizes = config.layer_size + (dim,)
        self.activation = dict(
            tanh=jnp.tanh, silu=jax.nn.silu, elu=jax.nn.elu, relu=jax.nn.relu
        )[config.activation]

        self.normalization = dict({})
        self.normalization.setdefault("eps", 1e-5)
        self.normalization.setdefault("decay_rate", 0.95)
        self.normalization.setdefault("create_scale", True)
        self.normalization.setdefault("create_offset", True)

    def __call__(self, x, is_training):
        for i, output_size in enumerate(self.output_sizes):
            x = hk.Linear(output_size=output_size)(x)
            x = hk.BatchNorm(**self.normalization)(x, is_training=is_training)
            x = self.activation(x)

        return x


class CVSolver(hk.Module):
    def __init__(
        self, model_config, pde_config, batch_size, name="MC_DL_Solver"
    ) -> None:
        super().__init__(name=name)
        self.model_conifg = model_config

        self.pde_config = pde_config

        self.sampler = getattr(pde, pde_config.name)(
            pde_config, batch_size
        )  # TODO: Remove batch size and get it from xold shape

        number_subnetworks = (
            self.pde_config.num_time_interval
            if self.model_conifg.subnetwork_per_timestep
            else 1
        )
        self.subnetworks = [
            SubNetwork(
                config=self.model_conifg,
                dim=pde_config.parameter.dim,
                name=f"subnet_{i}",
            )
            for i in range(number_subnetworks)
        ]

    def __call__(self, input, rng, is_training):
        # TODO: Switch to rng = hk.next_rng_key() instead of rng as input

        x_t = input
        y = jnp.zeros(
            x_t.shape[:1] + (1,)
        )  # hk.get_parameter("y_init", [1], init=jnp.ones)[..., None] * jnp.ones(x_t.shape[:1] + (1,))
        z = self.subnetworks[0](jnp.ones(x_t.shape), is_training=is_training)

        for i in range(self.pde_config.num_time_interval - 1):
            rng, subkey = jax.random.split(rng)
            x_t, dw, vol = self.sampler.sample(x_t, subkey)

            y = y + jnp.sum(z * vol, axis=1, keepdims=True)

            network_index = i + 1 if self.model_conifg.subnetwork_per_timestep else 0
            z = self.subnetworks[network_index](x_t, is_training=is_training)

        rng, subkey = jax.random.split(rng)
        x_t, dw, vol = self.sampler.sample(x_t, subkey)

        y = y + jnp.sum(z * vol, axis=1, keepdims=True)
        terminal = self.sampler.gt(None, x_t)
        return y, x_t, terminal
