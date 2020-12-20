import numpy as np
import pandas as pd
import os
import torch
import tqdm

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets.exp_decay.decay_loader import decay_dataset, decay_collate
from datasets.collision.collision_loader import collision_dataset, collision_collate
from datasets.eICU.eicu_loader import eicu_dataset, eicu_collate
from utils.visualization import vis_decay_traj, vis_collision_traj


class Trainer_decay:
    def __init__(self, args, ode_func):
        self.exp_name = args.exp_name
        self.ode_func = ode_func
        self.num_epochs = args.epochs
        self.l2_coeff = args.l2_coeff
        self.cv_idx = args.cv_idx

        self._data_indexing(self.cv_idx)
        self._load_dataloader(args.batch_size)
        self._make_folder()

        self.optimizer = optim.RMSprop(nn.ParameterList(self.ode_func.parameters()), lr=1e-3)
        self.use_scheduler = args.use_scheduler
        if self.use_scheduler:
            self.lr_step = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                      T_max=self.num_epochs)
        self.writer = SummaryWriter("./runs/{}".format(self.exp_name))
        self.best_mse = float('inf')

    def _data_indexing(self, cv_idx):
        idx_list = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
        data_idx_num = idx_list[cv_idx]
        i = data_idx_num[0]
        j = data_idx_num[1]

        test_range = range(200 * i, 200 * (i + 1))
        val_range = range(200 * j, 200 * (j + 1))
        train_range = set(range(1000)) - set(test_range) - set(val_range)

        self.val_idx = np.array(val_range)
        self.train_idx = np.array(list(train_range))

    def _load_dataloader(self, batch_size):
        data_path = './datasets/exp_decay/synthetic_changed_itv_v_seed.csv'
        data_train = decay_dataset(data_path=data_path, idx=self.train_idx)
        data_val = decay_dataset(data_path=data_path, idx=self.val_idx)
        self.dl_train = DataLoader(dataset=data_train, shuffle=True, collate_fn=decay_collate,
                                   batch_size=batch_size)
        self.dl_val = DataLoader(dataset=data_val, shuffle=False, collate_fn=decay_collate,
                                 batch_size=len(data_val))

    def _make_folder(self):
        if not os.path.isdir("./save/{}".format(self.exp_name)):
            if not os.path.isdir("./save"):
                os.mkdir("./save")
            os.mkdir("./save/{}".format(self.exp_name))

        if not os.path.isdir("./results/{}".format(self.exp_name)):
            if not os.path.isdir("./results"):
                os.mkdir("./results")
            os.mkdir("./results/{}".format(self.exp_name))

    def _train_phase(self, epoch):
        print("Start training")
        self.ode_func.train()
        self.train_loss = 0.
        self.train_instance_num = 0

        for i, b in tqdm.tqdm(enumerate(self.dl_train)):
            t = b["t"].float()
            x = b["obs"].float().to(self.ode_func.device)
            a = b["itv"].float().to(self.ode_func.device)

            self.optimizer.zero_grad()

            pred_x, z_a_list = self.ode_func(x, a, t)
            loss = torch.mean(torch.pow(pred_x - x, 2))

            try:
                z_a_list = torch.stack(z_a_list)
                loss += self.l2_coeff * z_a_list.norm(2)
            except:
                pass

            loss.backward()
            self.optimizer.step()
            if self.use_scheduler:
                self.lr_step.step()

            inst_num = torch.ones_like(pred_x).sum().item()
            self.train_loss += loss.item() * inst_num
            self.train_instance_num += inst_num

        self.train_loss /= self.train_instance_num

        if (epoch + 1) % 5 == 0:
            vis_decay_traj(x[0, :, :2], pred_x[0, :, :2], a[0, :, :2], epoch, '/' + str(self.exp_name) + '/')

    def _validation_phase(self, epoch):
        print("Validation start")
        self.ode_func.eval()
        self.val_loss = 0.
        self.val_instance_num = 0

        with torch.no_grad():
            for i, b in tqdm.tqdm(enumerate(self.dl_val)):
                t = b["t"].float()
                x = b["obs"].float().to(self.ode_func.device)
                a = b["itv"].float().to(self.ode_func.device)
                pred_x, _ = self.ode_func(x, a, t)

                loss = torch.mean(torch.pow(pred_x - x, 2))
                inst_num = torch.ones_like(pred_x).sum().item()
                self.val_loss += loss.item() * inst_num
                self.val_instance_num += inst_num

            self.val_loss /= self.val_instance_num

        if self.val_loss < self.best_mse:
            self.best_mse = self.val_loss
            self.best_epoch = epoch
            print('save best model')
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.ode_func.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_mse': self.best_mse,
                'best_epoch': self.best_epoch
            }, "./save/{}/best_model".format(self.exp_name))

        # visualization
        if (epoch + 1) % 10 == 0:
            vis_decay_traj(x[0, :, :2], pred_x[0, :, :2], a[0, :, :2],
                           epoch, '/' + str(self.exp_name) + '/', True)

    def _logger_scalar(self, epoch):
        self.writer.add_scalar('training_MSE', self.train_loss, epoch)
        self.writer.add_scalar('validation_MSE', self.val_loss, epoch)

    def run(self):
        for epoch in range(self.num_epochs):
            self._train_phase(epoch)
            self._validation_phase(epoch)

            print(f"MSE at epoch {epoch}: train_MSE={self.train_loss:.6f}, val_MSE={self.val_loss:.7f}")
            print(f"Current best mse {self.best_mse:.6f}")

            self._logger_scalar(epoch)

            if (epoch + 1) % 10 == 0:
                print('save current model')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.ode_func.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_mse': self.val_loss,
                    'best_epoch': self.best_epoch
                }, "./save/{}/current_model".format(self.exp_name))

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.ode_func.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_mse': self.best_mse,
            'best_epoch': self.best_epoch
        }, "./save/{}/final_model".format(self.exp_name))

        df_file_name = "./results/{}/synthetic.csv".format(self.exp_name)
        df_res = pd.DataFrame(
            {"Name": [self.exp_name], "MSE": [self.best_mse], "best_epoch": [self.best_epoch],
             "cv-idx": [self.cv_idx]})
        if os.path.isfile(df_file_name):
            df = pd.read_csv(df_file_name)
            df = df.append(df_res)
            df.to_csv(df_file_name, index=False)
        else:
            df_res.to_csv(df_file_name, index=False)
        self.writer.close()


class Trainer_collision:
    def __init__(self, args, ode_func):
        self.exp_name = args.exp_name
        self.ode_func = ode_func
        self.num_epochs = args.epochs
        self.l2_coeff = args.l2_coeff
        self.cv_idx = args.cv_idx

        self._data_indexing(self.cv_idx)
        self._load_dataloader(args.batch_size)
        self._make_folder()

        self.optimizer = optim.RMSprop(nn.ParameterList(self.ode_func.parameters()), lr=1e-3)
        self.use_scheduler = args.use_scheduler
        if self.use_scheduler:
            self.lr_step = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                      T_max=self.num_epochs)
        self.writer = SummaryWriter("./runs/{}".format(self.exp_name))
        self.best_mse = float('inf')

    def _data_indexing(self, cv_idx):
        idx_list = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
        data_idx_num = idx_list[cv_idx]
        i = data_idx_num[0]
        j = data_idx_num[1]

        test_range = range(200 * i, 200 * (i + 1))
        val_range = range(200 * j, 200 * (j + 1))
        train_range = set(range(1000)) - set(test_range) - set(val_range)

        self.val_idx = np.array(val_range)
        self.train_idx = np.array(list(train_range))

    def _load_dataloader(self, batch_size):
        obs_path = './datasets/collision/dataset_observation_revised_0429_scale10.csv'
        itv_path = './datasets/collision/dataset_intervention_revised_0429_scale10.csv'
        data_train = collision_dataset(obs_csv_file=obs_path, itv_csv_file=itv_path, idx=self.train_idx)
        data_val = collision_dataset(obs_csv_file=obs_path, itv_csv_file=itv_path, idx=self.val_idx)
        self.dl_train = DataLoader(dataset=data_train, shuffle=True, collate_fn=collision_collate,
                                   batch_size=batch_size)
        self.dl_val = DataLoader(dataset=data_val, shuffle=False, collate_fn=collision_collate,
                                 batch_size=len(data_val))

    def _make_folder(self):
        if not os.path.isdir("./save/{}".format(self.exp_name)):
            if not os.path.isdir("./save"):
                os.mkdir("./save")
            os.mkdir("./save/{}".format(self.exp_name))

        if not os.path.isdir("./results/{}".format(self.exp_name)):
            if not os.path.isdir("./results"):
                os.mkdir("./results")
            os.mkdir("./results/{}".format(self.exp_name))

    def _train_phase(self, epoch):
        print("Start training")
        self.ode_func.train()
        self.train_loss = 0.
        self.train_instance_num = 0

        for i, b in tqdm.tqdm(enumerate(self.dl_train)):
            t = b["t"].float()
            x = b["obs"].float().to(self.ode_func.device)
            a = b["itv"].float().to(self.ode_func.device)

            self.optimizer.zero_grad()

            pred_x, z_a_list = self.ode_func(x, a, t)
            loss = torch.mean(torch.pow(pred_x - x, 2))

            try:
                z_a_list = torch.stack(z_a_list)
                loss += self.l2_coeff * z_a_list.norm(2)
            except:
                pass

            loss.backward()
            self.optimizer.step()
            if self.use_scheduler:
                self.lr_step.step()

            inst_num = torch.ones_like(pred_x).sum().item()
            self.train_loss += loss.item() * inst_num
            self.train_instance_num += inst_num

        self.train_loss /= self.train_instance_num

        if (epoch + 1) % 5 == 0:
            vis_collision_traj(x[0, :, :2], pred_x[0, :, :2], a[0, :, :2],
                               epoch, '/' + str(self.exp_name) + '/')

    def _validation_phase(self, epoch):
        print("Validation start")
        self.ode_func.eval()
        self.val_loss = 0.
        self.val_instance_num = 0

        with torch.no_grad():
            for i, b in tqdm.tqdm(enumerate(self.dl_val)):
                t = b["t"].float()
                x = b["obs"].float().to(self.ode_func.device)
                a = b["itv"].float().to(self.ode_func.device)
                pred_x, _ = self.ode_func(x, a, t)

                loss = torch.mean(torch.pow(pred_x - x, 2))
                inst_num = torch.ones_like(pred_x).sum().item()
                self.val_loss += loss.item() * inst_num
                self.val_instance_num += inst_num

            self.val_loss /= self.val_instance_num

        if self.val_loss < self.best_mse:
            self.best_mse = self.val_loss
            self.best_epoch = epoch
            print('save best model')
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.ode_func.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_mse': self.best_mse,
                'best_epoch': self.best_epoch
            }, "./save/{}/best_model".format(self.exp_name))

        # visualization
        if (epoch + 1) % 10 == 0:
            vis_collision_traj(x[0, :, :2], pred_x[0, :, :2], a[0, :, :2],
                               epoch, '/' + str(self.exp_name) + '/', True)

    def _logger_scalar(self, epoch):
        self.writer.add_scalar('training_MSE', self.train_loss, epoch)
        self.writer.add_scalar('validation_MSE', self.val_loss, epoch)

    def run(self):
        for epoch in range(self.num_epochs):
            self._train_phase(epoch)
            self._validation_phase(epoch)

            print(f"MSE at epoch {epoch}: train_MSE={self.train_loss:.6f}, val_MSE={self.val_loss:.7f}")
            print(f"Current best mse {self.best_mse:.6f}")

            self._logger_scalar(epoch)

            if (epoch + 1) % 10 == 0:
                print('save current model')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.ode_func.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_mse': self.val_loss,
                    'best_epoch': self.best_epoch
                }, "./save/{}/current_model".format(self.exp_name))

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.ode_func.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_mse': self.best_mse,
            'best_epoch': self.best_epoch
        }, "./save/{}/final_model".format(self.exp_name))

        df_file_name = "./results/{}/collision.csv".format(self.exp_name)
        df_res = pd.DataFrame(
            {"Name": [self.exp_name], "MSE": [self.best_mse], "best_epoch": [self.best_epoch],
             "cv-idx": [self.cv_idx]})
        if os.path.isfile(df_file_name):
            df = pd.read_csv(df_file_name)
            df = df.append(df_res)
            df.to_csv(df_file_name, index=False)
        else:
            df_res.to_csv(df_file_name, index=False)
        self.writer.close()


class Trainer_eicu:
    def __init__(self, args, ode_func):
        self.exp_name = args.exp_name
        self.ode_func = ode_func
        self.num_epochs = args.epochs
        self.l2_coeff = args.l2_coeff
        self.cv_idx = args.cv_idx

        self._data_indexing(self.cv_idx)
        self._load_dataloader(args.batch_size)
        self._make_folder()

        self.optimizer = optim.RMSprop(nn.ParameterList(self.ode_func.parameters()), lr=1e-3)
        self.use_scheduler = args.use_scheduler
        if self.use_scheduler:
            self.lr_step = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                      T_max=self.num_epochs)
        self.writer = SummaryWriter("./runs/{}".format(self.exp_name))
        self.best_mse = float('inf')

    def _data_indexing(self, cv_idx):
        idx_list = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
        data_idx_num = idx_list[cv_idx]
        i = data_idx_num[0]
        j = data_idx_num[1]

        rand_idx = np.load('datasets/eICU/eicu_index.npy')

        test_range = rand_idx[(50 * i):(50 * (i + 1))]
        val_range = rand_idx[(50 * j):(50 * (j + 1))]
        train_range = np.array(list(set(range(251)) - set(test_range) - set(val_range)))

        self.val_idx = np.array(val_range)
        self.train_idx = np.array(list(train_range))

    def _load_dataloader(self, batch_size):
        data_path = './datasets/eICU/final_patient_30_std.csv'
        data_train = eicu_dataset(data_path=data_path, idx=self.train_idx)
        data_val = eicu_dataset(data_path=data_path, idx=self.val_idx)
        self.dl_train = DataLoader(dataset=data_train, shuffle=True, collate_fn=eicu_collate,
                                   batch_size=batch_size)
        self.dl_val = DataLoader(dataset=data_val, shuffle=False, collate_fn=eicu_collate,
                                 batch_size=len(data_val))

    def _make_folder(self):
        if not os.path.isdir("./save/{}".format(self.exp_name)):
            if not os.path.isdir("./save"):
                os.mkdir("./save")
            os.mkdir("./save/{}".format(self.exp_name))

        if not os.path.isdir("./results/{}".format(self.exp_name)):
            if not os.path.isdir("./results"):
                os.mkdir("./results")
            os.mkdir("./results/{}".format(self.exp_name))

    def _train_phase(self, epoch):
        print("Start training")
        self.ode_func.train()
        self.train_loss = 0.
        self.train_instance_num = 0

        for i, b in tqdm.tqdm(enumerate(self.dl_train)):
            t = b["t"].float()
            x = b["obs"].float().to(self.ode_func.device)
            a = b["itv"].float().to(self.ode_func.device)

            self.optimizer.zero_grad()

            pred_x, z_a_list = self.ode_func(x, a, t)
            loss = torch.mean(torch.pow(pred_x - x, 2))

            try:
                z_a_list = torch.stack(z_a_list)
                loss += self.l2_coeff * z_a_list.norm(2)
            except:
                pass

            loss.backward()
            self.optimizer.step()
            if self.use_scheduler:
                self.lr_step.step()

            inst_num = torch.ones_like(pred_x).sum().item()
            self.train_loss += loss.item() * inst_num
            self.train_instance_num += inst_num

        self.train_loss /= self.train_instance_num

    def _validation_phase(self, epoch):
        print("Validation start")
        self.ode_func.eval()
        self.val_loss = 0.
        self.val_instance_num = 0

        with torch.no_grad():
            for i, b in tqdm.tqdm(enumerate(self.dl_val)):
                t = b["t"].float()
                x = b["obs"].float().to(self.ode_func.device)
                a = b["itv"].float().to(self.ode_func.device)
                pred_x, _ = self.ode_func(x, a, t)

                loss = torch.mean(torch.pow(pred_x - x, 2))
                inst_num = torch.ones_like(pred_x).sum().item()
                self.val_loss += loss.item() * inst_num
                self.val_instance_num += inst_num

            self.val_loss /= self.val_instance_num

        # save best model
        if self.val_loss < self.best_mse:
            self.best_mse = self.val_loss
            self.best_epoch = epoch
            print('save best model')
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.ode_func.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_mse': self.best_mse,
                'best_epoch': self.best_epoch
            }, "./save/{}/best_model".format(self.exp_name))


    def _logger_scalar(self, epoch):
        self.writer.add_scalar('training_MSE', self.train_loss, epoch)
        self.writer.add_scalar('validation_MSE', self.val_loss, epoch)

    def run(self):
        for epoch in range(self.num_epochs):
            self._train_phase(epoch)
            self._validation_phase(epoch)

            print(f"MSE at epoch {epoch}: train_MSE={self.train_loss:.5f}, val_MSE={self.val_loss:.7f}")
            print(f"Current best mse {self.best_mse:.4f}")

            self._logger_scalar(epoch)

            if (epoch + 1) % 10 == 0:
                print('save current model')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.ode_func.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_mse': self.val_loss,
                    'best_epoch': self.best_epoch
                }, "./save/{}/current_model".format(self.exp_name))

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.ode_func.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_mse': self.best_mse,
            'best_epoch': self.best_epoch
        }, "./save/{}/final_model".format(self.exp_name))

        df_file_name = "./results/{}/eicu.csv".format(self.exp_name)
        df_res = pd.DataFrame(
            {"Name": [self.exp_name], "MSE": [self.best_mse], "best_epoch": [self.best_epoch],
             "cv-idx": [self.cv_idx]})
        if os.path.isfile(df_file_name):
            df = pd.read_csv(df_file_name)
            df = df.append(df_res)
            df.to_csv(df_file_name, index=False)
        else:
            df_res.to_csv(df_file_name, index=False)
        self.writer.close()
