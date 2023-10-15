import jax
import jax.numpy as jnp


class Equation(object):
    def __init__(self, config, batch_size) -> None:
        self.batch_size = batch_size
        self.dim = config.parameter.dim
        self.total_time = config.total_time
        self.num_time_interval = config.num_time_interval
        self.delta_t = self.total_time / self.num_time_interval
        self.sqrt_delta_t = jnp.sqrt(self.delta_t)
        self.y_init = None

    def ft(self, t, x):
        raise "Not implemented"

    def gt(self, t, x):
        raise "Not implemented"

    def sample(self, x, rng):
        raise "Not implemented"


class BlackScholes(Equation):
    def __init__(self, config, batch_size) -> None:
        super(BlackScholes, self).__init__(config, batch_size)
        self.spot = jnp.array([config.parameter.spot])
        self.sigma = jnp.array([config.parameter.sigmaV])
        self.rate = jnp.array([config.parameter.rate])
        self.strike = jnp.array([config.parameter.strike])

        self.batch_size = batch_size
        # exact price for T=1
        self.y_init = jnp.array([config.parameter.yinit])

    # Computes with the euler maruyama scheme next time step of discretization
    # based on previous time step. It gets X_t and returns X_t+1 and
    # vol which is sigma * X_t * dW_t.
    def sample(self, x_old, rng):
        dw = jax.random.normal(rng, [self.batch_size, 1]) * jnp.sqrt(self.delta_t)
        vol = self.sigma * x_old * dw
        x_new = x_old + vol

        return x_new, dw, vol

    # Put option
    def gt(self, t, x):
        return jnp.maximum(self.strike - x, 0)


class Heston(Equation):
    def __init__(self, config, batch_size):
        super(Heston, self).__init__(config, batch_size)
        self.spot = config.parameter.spot
        self.v0 = config.parameter.v0
        self.sigmaV = config.parameter.sigmaV
        self.rate = config.parameter.rate
        self.strike = config.parameter.strike
        self.rho = config.parameter.rho
        self.kappa = config.parameter.kappa
        self.theta = config.parameter.theta

        # exact price for 2 dim heston T=1 for a call option
        self.y_init = config.parameter.yinit

        self.batch_size = batch_size

    # Computes with euler maruyama scheme next time step of discretization
    # based on previous time step. It get X_t and V_t and returns X_t+1 and V_t+1 a
    # and an increment of the Brownian motion
    def sample(self, x_old, rng):
        batch_size = x_old.shape[0]
        dim = self.dim // 2
        dw1_sample = jax.random.normal(rng, [batch_size, dim])
        dw2_sample = jax.random.normal(rng, [batch_size, dim])

        # Brownian motion for asset. Correlated with rho to the Brownian motion of the variance processes
        dwS = self.rho * dw1_sample + jnp.sqrt(1 - self.rho**2) * dw2_sample
        zeros = jnp.zeros([batch_size, dim])

        # Variance processes
        a = (
            self.sigmaV
            * jnp.sqrt(jnp.maximum(x_old.at[:, :dim].get(), 0))
            * dw1_sample
            * jnp.sqrt(self.delta_t)
        )
        x_newV = (
            x_old.at[:, :dim].get()
            + self.kappa
            * (self.theta - jnp.maximum(x_old.at[:, :dim].get(), 0))
            * self.delta_t
            + a
        )

        # Asset process. It is the log(S_t) processs. Therefore we take the tf.math.exp function to get S_t
        b = (
            jnp.sqrt(jnp.maximum(x_old.at[:, :dim].get(), 0))
            * x_old.at[:, dim:].get()
            * dwS
            * jnp.sqrt(self.delta_t)
        )
        x_newA = x_old.at[:, dim:].get() * jnp.exp(
            (self.rate - 0.5 * jnp.maximum(x_old.at[:, :dim].get(), 0)) * self.delta_t
            + b
        )

        dw_sample = jnp.concatenate([dw1_sample, dwS], axis=1)
        dx_sample = jnp.concatenate([x_newV, x_newA], axis=1)

        # vol (volatility part)
        vol = jnp.concatenate([a, b], axis=1)

        return dx_sample, dw_sample, vol

    def gt(self, t, x):
        # 2 dim heston
        dim = self.dim // 2
        x_A = x.at[:, dim:].get()
        return jnp.maximum(x_A - self.strike, 0) * jnp.exp(-self.rate * self.total_time)
        # return jnp.maximum(x.at[:, self.dim // 2:].get() - self.strike, 0) * jnp.exp(-self.rate * self.total_time)
