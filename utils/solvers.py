import torch.nn as nn


def euler_update(h, dh, dt):
    return h + dh * dt


def euler_step(func, dt, state):
    return euler_update(state, func(state), dt)


def rk4_step(func, dt, state):
    k1 = func(state)
    k2 = func(euler_update(state, k1, dt / 2.))
    k3 = func(euler_update(state, k2, dt / 2.))
    k4 = func(euler_update(state, k3, dt))
    return state + dt * (k1 + 2. * k2 + 2. * k3 + k4) / 6.



def update_x(x, dxdt, delta_t):
    return x + dxdt * delta_t


class ode_class(nn.Module):
    def __init__(self, hidden_size):
        super(ode_class, self).__init__()
        self.temp_fn = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.LeakyReLU(),
                    nn.Linear(hidden_size, hidden_size))

    def forward(self, t, h):
        return self.temp_fn(h)
