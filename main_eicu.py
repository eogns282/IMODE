import argparse
import torch

from models.imode import ODEFunc
from trainer import Trainer_eicu
from evaluator import Evaluator_eicu

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

parser = argparse.ArgumentParser()
parser.add_argument('--exp-name', type=str, default='eicu_temp')
parser.add_argument('--model-type', type=str, default='adaptive',
                    choices=['adaptive'])
parser.add_argument('--epochs', type=int, default=1500)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--gpu-num', type=int, default=0)
parser.add_argument('--activation-fn', type=str, default='LeakyReLU', choices=['LeakyReLU', 'Softplus'])
parser.add_argument('--use-rk4', type=boolean_string, default=False)
parser.add_argument('--l2-coeff', type=float, default=0.0001)
parser.add_argument('--ode-delta-t', type=float, default=1.)
parser.add_argument('--cv-idx', type=int, default=4, choices=[0, 1, 2, 3, 4])
parser.add_argument('--feed-first', type=boolean_string, default=True)
parser.add_argument('--feed-ratio', type=float, default=0.2)
parser.add_argument('--x-size', type=int, default=3)
parser.add_argument('--a-size', type=int, default=5)
parser.add_argument('--x-hidden-size', type=int, default=20)
parser.add_argument('--a-hidden-size', type=int, default=10)
parser.add_argument('--num-steps', type=int, default=30)
parser.add_argument('--use-scheduler', type=boolean_string, default=True)
parser.add_argument('--test-phase', type=boolean_string, default=False)

args = parser.parse_args()

device = torch.device(f"cuda:{args.gpu_num}")
ode_func = ODEFunc(args).to(device)

if __name__ == '__main__':
    if args.test_phase:
        evaluator = Evaluator_eicu(args, ode_func)
        evaluator.run()
    else:
        trainer = Trainer_eicu(args, ode_func)
        trainer.run()