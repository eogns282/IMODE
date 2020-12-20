import torch
import torch.nn as nn


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