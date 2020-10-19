import torch
import torch.nn as nn

from torchdiffeq import odeint

from utils.solvers import ode_class, update_x


class ODEFunc(nn.Module):
    def __init__(self, args):
        super(ODEFunc, self).__init__()
        self.x_size, self.a_size = args.x_size, args.a_size
        self.x_hidden_size, self.a_hidden_size = args.x_hidden_size, args.a_hidden_size
        self.model_type = args.model_type
        self.device = torch.device(f"cuda:{args.gpu_num}")
        self.delta_t = args.ode_delta_t
        self.feed_first = args.feed_first
        if self.feed_first:
            self.feed_idx = int(args.feed_ratio * args.num_steps)
        self.use_rk4 = args.use_rk4
        self.exp_param = nn.Parameter(torch.Tensor([-1.]), requires_grad=True)

        if self.model_type == 'general':
            self.rnncell_a = nn.GRUCell(self.x_size + self.a_size, self.a_hidden_size)
            self.a_ode = ode_class(self.a_hidden_size)
            self.rnncell_x = nn.GRUCell(self.x_size, self.x_hidden_size)
            self.x_ode = ode_class(self.x_hidden_size)
            self.decoder_dx = nn.Sequential(
                nn.Linear(self.x_hidden_size + self.a_hidden_size, self.x_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(self.x_hidden_size, self.x_size),
            )
        elif self.model_type == 'decay_with_odernn':
            self.rnncell_a = nn.GRUCell(self.x_size + self.a_size, self.a_hidden_size)
            self.rnncell_x = nn.GRUCell(self.x_size, self.x_hidden_size)
            self.x_ode = ode_class(self.x_hidden_size)
            self.decoder_dx = nn.Sequential(
                nn.Linear(self.x_hidden_size + self.a_hidden_size, self.x_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(self.x_hidden_size, self.x_size),
            )
        elif self.model_type == 'decay_with_rnn':
            self.rnncell_a = nn.GRUCell(self.x_size + self.a_size, self.a_hidden_size)
            self.rnncell_x = nn.GRUCell(self.x_size, self.x_hidden_size)
            self.decoder_dx = nn.Sequential(
                nn.Linear(self.x_hidden_size + self.a_hidden_size, self.x_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(self.x_hidden_size, self.x_size),
            )
        elif self.model_type == 'switch':
            self.nn_x = nn.Sequential(
                nn.Linear(self.x_size, self.x_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(self.x_hidden_size, self.x_size),
            )
            self.nn_a = nn.Sequential(
                nn.Linear(self.x_size + self.a_size, self.a_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(self.a_hidden_size, self.x_size)
            )
        else:
            print('check the type of models')
            exit()

    def forward(self, x, y, time_steps):
        x_gt = x
        x0 = x[:, 0, :]

        x_list = [x0]
        x = x_list[0]

        z_a_list = []

        if self.model_type == 'general':
            z_x = torch.zeros(x.size(0), self.x_hidden_size).to(self.device)
            z_a = torch.zeros(x.size(0), self.a_hidden_size).to(self.device)

            z_a_list.append(z_a)

            for i, _ in enumerate(time_steps[:-1]):
                a_t = y[:, i, :]

                if self.feed_first:
                    if i < self.feed_idx:
                        x = x_gt[:, i, :]

                # flow
                if i != 0:
                    t_span = torch.Tensor([time_steps[i - 1], time_steps[i]]).unsqueeze(1)
                    z_a = odeint(self.a_ode, z_a, t_span, method='rk4', rtol=1e-10, atol=1e-12)[1]
                    z_x = odeint(self.x_ode, z_x, t_span, method='rk4', rtol=1e-10, atol=1e-12)[1]

                # jump
                z_a_new = self.rnncell_a(torch.cat([x, a_t], axis=1), z_a)
                mask = (a_t.sum(1) != 0)
                mask = mask.unsqueeze(-1)
                z_a = z_a * ~mask + z_a_new * mask
                z_a_list.append(z_a)
                z_x = self.rnncell_x(x, z_x)

                # prediction
                z = torch.cat([z_x, z_a], axis=1)
                dhdt_x = self.decoder_dx(z)
                x = update_x(x, dhdt_x, self.delta_t)
                x_list.append(x)

        elif self.model_type == 'decay_with_odernn':
            z_x = torch.zeros(x.size(0), self.x_hidden_size).to(self.device)
            z_a = torch.zeros(x.size(0), self.a_hidden_size).to(self.device)

            z_a_list.append(z_a)

            for i, _ in enumerate(time_steps[:-1]):
                a_t = y[:, i, :]

                if self.feed_first:
                    if i < self.feed_idx:
                        x = x_gt[:, i, :]

                # flow
                if i != 0:
                    z_a = z_a * torch.exp(min(torch.zeros([1], device=self.device), self.exp_param * self.delta_t))
                    t_span = torch.Tensor([time_steps[i - 1], time_steps[i]]).unsqueeze(1)
                    z_x = odeint(self.x_ode, z_x, t_span, method='rk4', rtol=1e-10, atol=1e-12)[1]

                # jump
                z_a_new = self.rnncell_a(torch.cat([x, a_t], axis=1), z_a)
                mask = (a_t.sum(1) != 0)
                mask = mask.unsqueeze(-1)
                z_a = z_a * ~mask + z_a_new * mask
                z_a_list.append(z_a)
                z_x = self.rnncell_x(x, z_x)

                # prediction
                z = torch.cat([z_x, z_a], axis=1)
                dhdt_x = self.decoder_dx(z)
                x = update_x(x, dhdt_x, self.delta_t)
                x_list.append(x)

        elif self.model_type == 'decay_with_rnn':
            z_x = torch.zeros(x.size(0), self.x_hidden_size).to(self.device)
            z_a = torch.zeros(x.size(0), self.a_hidden_size).to(self.device)

            for i, _ in enumerate(time_steps[:-1]):
                a_t = y[:, i, :]

                if self.feed_first:
                    if i < self.feed_idx:
                        x = x_gt[:, i, :]

                mask = (a_t.sum(1) != 0)
                mask = mask.unsqueeze(-1)

                # flow
                if i != 0:
                    z_a = z_a * torch.exp(min(torch.zeros([1], device=self.device), self.exp_param * self.delta_t))

                # jump
                z_a_new = self.rnncell_a(torch.cat([x, a_t], axis=1), z_a)
                z_a = z_a * ~mask + z_a_new * mask
                z_a_list.append(z_a)
                z_x = self.rnncell_x(x, z_x)

                # prediction
                z = torch.cat([z_x, z_a], axis=1)
                dhdt_x = self.decoder_dx(z)
                x = update_x(x, dhdt_x, self.delta_t)
                x_list.append(x)

        elif self.model_type == 'switch':
            for i, _ in enumerate(time_steps[:-1]):
                a_t = y[:, i, :]

                if self.feed_first:
                    if i < self.feed_idx:
                        x = x_gt[:, i, :]

                mask = (a_t.sum(1) != 0)
                mask = mask.unsqueeze(-1)

                # jump
                dhdt_x = self.nn_x(x)
                dhdt_a = self.nn_a(torch.cat([x, a_t], 1))
                dhdt_a = mask * dhdt_a
                z_a_list.append(dhdt_a)

                # prediction
                dhdt = dhdt_x + dhdt_a
                x = update_x(x, dhdt, self.delta_t)
                x_list.append(x)

        x_list = torch.stack(x_list[:], dim=0)
        x_list = x_list.permute(1, 0, 2)

        return x_list, z_a_list