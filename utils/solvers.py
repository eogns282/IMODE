import torch
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
    def __init__(self, hidden_size, activation):
        super(ode_class, self).__init__()
        self.temp_fn = nn.Sequential(
                       nn.Linear(hidden_size, hidden_size),
                       activation,
                       nn.Linear(hidden_size, hidden_size))

    def forward(self, t, h):
        return self.temp_fn(h)


class ode_function_for_adaptive(nn.Module):
    def __init__(self, z_x_size, z_a_size, h_size, activation):
        super(ode_function_for_adaptive, self).__init__()

        self.nfe = 0  # num forward evaluation

        self.z_x_size = z_x_size
        self.z_a_size = z_a_size
        self.h_size = h_size

        self.f_z_x = nn.Sequential(
                     nn.Linear(z_x_size, z_x_size),
                     activation,
                     nn.Linear(z_x_size, z_x_size))
        self.f_z_a = nn.Sequential(
                     nn.Linear(z_a_size, z_a_size),
                     activation,
                     nn.Linear(z_a_size, z_a_size))
        self.f_h = nn.Sequential(
                   nn.Linear(z_x_size + z_a_size + h_size, z_x_size),
                   activation,
                   nn.Linear(z_x_size, h_size))

    def reset_nfe(self):
        self.nfe = 0

    def forward(self, t, input_concat):
        self.nfe += 1

        z_x = input_concat[:, :self.z_x_size]
        z_a = input_concat[:, self.z_x_size:(self.z_x_size + self.z_a_size)]
        h = input_concat[:, (self.z_x_size + self.z_a_size):]

        d_z_x = self.f_z_x(z_x)
        d_z_a = self.f_z_a(z_a)
        d_h = self.f_h(torch.cat([z_x, z_a, h], axis=1))

        return torch.cat([d_z_x, d_z_a, d_h], axis=1)


class ode_function_for_adaptive_wo(nn.Module):
    def __init__(self, z_x_size, z_a_size, h_size, activation):
        super(ode_function_for_adaptive_wo, self).__init__()

        self.z_x_size = z_x_size
        self.z_a_size = z_a_size
        self.h_size = h_size

        self.f_z_x = nn.Sequential(
                     nn.Linear(z_x_size, z_x_size),
                     activation,
                     nn.Linear(z_x_size, z_x_size))
        self.f_z_a = nn.Sequential(
                     nn.Linear(z_a_size, z_a_size),
                     activation,
                     nn.Linear(z_a_size, z_a_size))
        self.f_h = nn.Sequential(
                   nn.Linear(z_x_size + z_a_size, z_x_size),
                   activation,
                   nn.Linear(z_x_size, h_size))

    def forward(self, t, input_concat):
        z_x = input_concat[:, :self.z_x_size]
        z_a = input_concat[:, self.z_x_size:(self.z_x_size + self.z_a_size)]
        # h = input_concat[:, (self.z_x_size + self.z_a_size):]

        d_z_x = self.f_z_x(z_x)
        d_z_a = self.f_z_a(z_a)
        d_h = self.f_h(torch.cat([z_x, z_a], axis=1))

        return torch.cat([d_z_x, d_z_a, d_h], axis=1)


class ode_function_for_cde(nn.Module):
    def __init__(self, z_x_size, z_a_size, h_size, activation):
        super(ode_function_for_cde, self).__init__()

        self.nfe = 0  # num forward evaluation

        self.z_x_size = z_x_size
        self.z_a_size = z_a_size
        self.h_size = h_size

        self.f_z_x = nn.Sequential(
                     nn.Linear(z_x_size, z_x_size),
                     activation,
                     nn.Linear(z_x_size, z_x_size))
        self.f_z_a = nn.Sequential(
                     nn.Linear(z_a_size, z_a_size),
                     activation,
                     nn.Linear(z_a_size, z_a_size))
        self.f_h = nn.Sequential(
                   nn.Linear(z_x_size + z_a_size + h_size, z_x_size),
                   activation,
                   nn.Linear(z_x_size, h_size))

        self.f_matrix = nn.Sequential(
                        nn.Linear(h_size, z_x_size),
                        activation,
                        nn.Linear(z_x_size, h_size * h_size))

    def forward(self, t, input_concat):
        z_x = input_concat[:, :self.z_x_size]
        z_a = input_concat[:, self.z_x_size:(self.z_x_size + self.z_a_size)]
        h = input_concat[:, (self.z_x_size + self.z_a_size):]

        d_z_x = self.f_z_x(z_x)
        d_z_a = self.f_z_a(z_a)
        d_h = self.f_h(torch.cat([z_x, z_a, h], axis=1))
        d_matrix = self.f_matrix(h)

        # d_h = torch.matmul(d_matrix, d_h)
        d_h = d_matrix.reshape(-1, self.h_size, self.h_size).matmul(d_h.unsqueeze(-1)).squeeze()
        return torch.cat([d_z_x, d_z_a, d_h], axis=1)