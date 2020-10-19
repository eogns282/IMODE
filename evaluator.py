import os
import numpy as np
import pandas as pd
import torch
import tqdm

from torch.utils.data import DataLoader

from datasets.exp_decay.decay_loader import decay_dataset, decay_collate
from datasets.collision.collision_loader import collision_dataset, collision_collate
from datasets.eICU.eicu_loader import eicu_dataset, eicu_collate

class Evaluator_decay:
    def __init__(self, args, ode_func):
        self.exp_name = args.exp_name
        self.ode_func = ode_func
        self.cv_idx = args.cv_idx

        self._data_indexing(self.cv_idx)
        self._load_dataloader(args.batch_size)
        self._load_checkpoint()
        self._make_folder()

    def _data_indexing(self, cv_idx):
        idx_list = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
        data_idx_num = idx_list[cv_idx]
        i = data_idx_num[0]
        j = data_idx_num[1]
        test_range = range(200 * i, 200 * (i + 1))
        val_range = range(200 * j, 200 * (j + 1))
        train_range = set(range(1000)) - set(test_range) - set(val_range)
        self.test_idx = np.array(test_range)
        self.val_idx = np.array(val_range)
        self.train_idx = np.array(list(train_range))

    def _load_dataloader(self, batch_size):
        data_path = './datasets/exp_decay/synthetic_changed_itv_v_seed.csv'
        data_train = decay_dataset(data_path=data_path, idx=self.train_idx)
        data_test = decay_dataset(data_path=data_path, idx=self.test_idx)
        data_val = decay_dataset(data_path=data_path, idx=self.val_idx)
        self.dl_val = DataLoader(dataset=data_val, shuffle=False, collate_fn=decay_collate,
                                 batch_size=len(data_val))
        self.dl_test = DataLoader(dataset=data_test, shuffle=False, collate_fn=decay_collate,
                                  batch_size=len(data_test))
        self.dl_train = DataLoader(dataset=data_train, shuffle=True, collate_fn=decay_collate,
                                   batch_size=batch_size)

    def _load_checkpoint(self):
        checkpoint = torch.load('./save/{}/best_model'.format(self.exp_name))
        self.ode_func.load_state_dict(checkpoint['model_state_dict'])
        self.best_epoch = checkpoint['best_epoch']

    def _make_folder(self):
        if not os.path.isdir("./test_results/{}".format(self.exp_name)):
            if not os.path.isdir("./test_results"):
                os.mkdir("./test_results")
            os.mkdir("./test_results/{}".format(self.exp_name))

    def _test_phase(self):
        print("Test start")
        self.ode_func.eval()
        self.test_loss = 0
        self.test_instance_num = 0

        with torch.no_grad():
            for i, b in tqdm.tqdm(enumerate(self.dl_test)):
                t = b["t"].float()
                x = b["obs"].float().to(self.ode_func.device)
                a = b["itv"].float().to(self.ode_func.device)
                pred_x, _ = self.ode_func(x, a, t)

                loss = torch.mean(torch.pow(pred_x - x, 2))
                inst_num = torch.ones_like(pred_x).sum().item()
                self.test_loss += loss.item() * inst_num
                self.test_instance_num += inst_num

            self.test_loss /= self.test_instance_num
            self.obs = x
            self.itv = a
            self.pred_x = pred_x

    def run(self):
        self._test_phase()

        print(f"MSE at epoch {self.best_epoch}: test_MSE={self.test_loss:.7f}")

        df_file_name = "./test_results/{}/synthetic.csv".format(self.exp_name)
        df_res = pd.DataFrame(
            {"Name": [self.exp_name], "MSE": [self.test_loss], "best_epoch": [self.best_epoch],
             "cv-idx": [self.cv_idx]})

        if os.path.isfile(df_file_name):
            df = pd.read_csv(df_file_name)
            df = df.append(df_res)
            df.to_csv(df_file_name, index=False)
        else:
            df_res.to_csv(df_file_name, index=False)


class Evaluator_collision:
    def __init__(self, args, ode_func):
        self.exp_name = args.exp_name
        self.ode_func = ode_func
        self.cv_idx = args.cv_idx

        self._data_indexing(self.cv_idx)
        self._load_dataloader(args.batch_size)
        self._load_checkpoint()
        self._make_folder()

    def _data_indexing(self, cv_idx):
        idx_list = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
        data_idx_num = idx_list[cv_idx]
        i = data_idx_num[0]
        j = data_idx_num[1]
        test_range = range(200 * i, 200 * (i + 1))
        val_range = range(200 * j, 200 * (j + 1))
        train_range = set(range(1000)) - set(test_range) - set(val_range)
        self.test_idx = np.array(test_range)
        self.val_idx = np.array(val_range)
        self.train_idx = np.array(list(train_range))

    def _load_dataloader(self, batch_size):
        obs_path = './datasets/collision/dataset_observation_revised_0429_scale10.csv'
        itv_path = './datasets/collision/dataset_intervention_revised_0429_scale10.csv'
        data_val = collision_dataset(obs_csv_file=obs_path, itv_csv_file=itv_path, idx=self.val_idx)
        data_test = collision_dataset(obs_csv_file=obs_path, itv_csv_file=itv_path, idx=self.test_idx)
        self.dl_val = DataLoader(dataset=data_val, shuffle=False, collate_fn=collision_collate,
                                 batch_size=len(data_test))
        self.dl_test = DataLoader(dataset=data_test, shuffle=False, collate_fn=collision_collate,
                                  batch_size=len(data_test))

    def _load_checkpoint(self):
        checkpoint = torch.load('./save/{}/best_model'.format(self.exp_name))
        self.ode_func.load_state_dict(checkpoint['model_state_dict'])
        self.best_epoch = checkpoint['best_epoch']

    def _make_folder(self):
        if not os.path.isdir("./test_results/{}".format(self.exp_name)):
            if not os.path.isdir("./test_results"):
                os.mkdir("./test_results")
            os.mkdir("./test_results/{}".format(self.exp_name))

    def _test_phase(self):
        print("Test start")
        self.ode_func.eval()
        self.test_loss = 0
        self.test_instance_num = 0

        with torch.no_grad():
            for i, b in tqdm.tqdm(enumerate(self.dl_test)):
                t = b["t"].float()
                x = b["obs"].float().to(self.ode_func.device)
                a = b["itv"].float().to(self.ode_func.device)
                pred_x, _ = self.ode_func(x, a, t)

                loss = torch.mean(torch.pow(pred_x - x, 2))
                inst_num = torch.ones_like(pred_x).sum().item()
                self.test_loss += loss.item() * inst_num
                self.test_instance_num += inst_num

            self.test_loss /= self.test_instance_num
            self.obs = x
            self.itv = a
            self.pred_x = pred_x

    def run(self):
        self._test_phase()

        print(f"MSE at epoch {self.best_epoch}: test_MSE={self.test_loss:.7f}")

        df_file_name = "./test_results/{}/collision.csv".format(self.exp_name)
        df_res = pd.DataFrame(
            {"Name": [self.exp_name], "MSE": [self.test_loss], "best_epoch": [self.best_epoch],
             "cv-idx": [self.cv_idx]})

        if os.path.isfile(df_file_name):
            df = pd.read_csv(df_file_name)
            df = df.append(df_res)
            df.to_csv(df_file_name, index=False)
        else:
            df_res.to_csv(df_file_name, index=False)


class Evaluator_eicu:
    def __init__(self, args, ode_func):
        self.exp_name = args.exp_name
        self.ode_func = ode_func
        self.cv_idx = args.cv_idx

        self._data_indexing(self.cv_idx)
        self._load_dataloader(args.batch_size)
        self._load_checkpoint()
        self._make_folder()

    def _data_indexing(self, cv_idx):
        idx_list = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
        data_idx_num = idx_list[cv_idx]
        i = data_idx_num[0]
        j = data_idx_num[1]

        rand_idx = np.load('datasets/eICU/eicu_index.npy')

        test_range = rand_idx[(50 * i):(50 * (i + 1))]
        self.test_idx = np.array(test_range)
        val_range = rand_idx[(50 * j):(50 * (j + 1))]
        self.val_idx = np.array(val_range)

    def _load_dataloader(self, batch_size):
        data_path = './datasets/eICU/final_patient_30_std.csv'
        data_val = eicu_dataset(data_path=data_path, idx=self.val_idx)
        data_test = eicu_dataset(data_path=data_path, idx=self.test_idx)
        self.dl_val = DataLoader(dataset=data_test, shuffle=False, collate_fn=eicu_collate,
                                 batch_size=len(data_val))
        self.dl_test = DataLoader(dataset=data_test, shuffle=False, collate_fn=eicu_collate,
                                  batch_size=len(data_test))

    def _load_checkpoint(self):
        checkpoint = torch.load('./save/{}/best_model'.format(self.exp_name))
        self.ode_func.load_state_dict(checkpoint['model_state_dict'])
        self.best_epoch = checkpoint['best_epoch']

    def _make_folder(self):
        if not os.path.isdir("./test_results/{}".format(self.exp_name)):
            if not os.path.isdir("./test_results"):
                os.mkdir("./test_results")
            os.mkdir("./test_results/{}".format(self.exp_name))

    def _test_phase(self):
        print("Test start")
        self.ode_func.eval()
        self.test_loss = 0
        self.test_instance_num = 0

        with torch.no_grad():
            for i, b in tqdm.tqdm(enumerate(self.dl_test)):
                t = b["t"].float()
                x = b["obs"].float().to(self.ode_func.device)
                a = b["itv"].float().to(self.ode_func.device)
                pred_x, _ = self.ode_func(x, a, t)

                loss = torch.mean(torch.pow(pred_x - x, 2))
                inst_num = torch.ones_like(pred_x).sum().item()
                self.test_loss += loss.item() * inst_num
                self.test_instance_num += inst_num

            self.test_loss /= self.test_instance_num
            self.obs = x
            self.itv = a
            self.pred_x = pred_x

    def run(self):
        self._test_phase()

        print(f"MSE at epoch {self.best_epoch}: test_MSE={self.test_loss:.7f}")

        df_file_name = "./test_results/{}/eicu.csv".format(self.exp_name)
        df_res = pd.DataFrame(
            {"Name": [self.exp_name], "MSE": [self.test_loss], "best_epoch": [self.best_epoch],
             "cv-idx": [self.cv_idx]})

        if os.path.isfile(df_file_name):
            df = pd.read_csv(df_file_name)
            df = df.append(df_res)
            df.to_csv(df_file_name, index=False)
        else:
            df_res.to_csv(df_file_name, index=False)