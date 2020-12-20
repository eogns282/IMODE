import torch
import torch.nn as nn

from torchdiffeq import odeint
# /from torchdiffeq import odeint_adjoint as odeint

from utils.solvers import ode_function_for_adaptive


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
        if args.activation_fn == 'LeakyReLU':
            self.activation = nn.LeakyReLU()
        elif args.activation_fn == 'Softplus':
            self.activation = nn.Softplus()

        if self.model_type == 'adaptive':
            self.rnncell_a = nn.GRUCell(self.x_size + self.a_size, self.a_hidden_size)
            self.rnncell_x = nn.GRUCell(self.x_size, self.x_hidden_size)
            self.ode_fn = ode_function_for_adaptive(self.x_hidden_size, self.a_hidden_size,
                                                    self.x_size, self.activation)
            self.nfe_yes = 0
            self.nfe_no = 0
            self.inst_yes = 0
            self.inst_no = 0

        else:
            print('check the type of models')
            exit()

    def forward(self, x, y, time_steps):
        x_gt = x
        x0 = x[:, 0, :]

        x_list = [x0]
        x = x_list[0]

        z_a_list = []

        if self.model_type == 'adaptive':
            # init z vectors as zero vectors
            # TODO: init z vectors using first data point!
            z_x = torch.zeros(x.size(0), self.x_hidden_size).to(self.device)
            z_a = torch.zeros(x.size(0), self.a_hidden_size).to(self.device)


            z_a_list.append(z_a)
            for i, _ in enumerate(time_steps[:-1]):
                a_t = y[:, i, :]
                if self.feed_first:
                    if i < self.feed_idx:
                        x = x_gt[:, i, :]

                # jump
                z_a_new = self.rnncell_a(torch.cat([x, a_t], axis=1), z_a)
                mask = (a_t.sum(1) != 0)
                mask = mask.unsqueeze(-1)
                z_a = z_a * ~mask + z_a_new * mask
                z_a_list.append(z_a)
                z_x = self.rnncell_x(x, z_x)

                # flow
                input_concat = torch.cat([z_x, z_a, x], axis=1)
                t_span = torch.Tensor([time_steps[i], time_steps[i + 1]])
                if self.use_rk4:
                    new_input_concat = odeint(self.ode_fn, input_concat, t_span, method='rk4')[-1]
                else:
                    new_input_concat = odeint(self.ode_fn, input_concat, t_span, method='dopri5')[-1]

                if a_t.sum() != 0:
                    self.inst_yes += 1
                    self.nfe_yes += self.ode_fn.nfe
                else:
                    self.inst_no += 1
                    self.nfe_no += self.ode_fn.nfe

                self.ode_fn.reset_nfe()

                z_x = new_input_concat[:, :self.x_hidden_size]
                z_a = new_input_concat[:, self.x_hidden_size:(self.x_hidden_size + self.a_hidden_size)]
                x = new_input_concat[:, (self.x_hidden_size + self.a_hidden_size):]
                x_list.append(x)

        x_list = torch.stack(x_list[:], dim=0)
        x_list = x_list.permute(1, 0, 2)

        return x_list, z_a_list