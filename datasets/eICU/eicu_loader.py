import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset


class eicu_dataset(Dataset):
    def __init__(self, data_path, idx):
        self.all_data = pd.read_csv(data_path)

        self.positions = self.all_data[['ID', 'Time', 'Systolic_BP', 'Diastolic_BP', 'Mean_BP']]
        self.interventions = self.all_data[['ID', 'Time', 'norepinephrine', 'vasopressin',
                                            'propofol', 'amiodarone', 'phenylephrine']]

        self.positions = self.positions.loc[self.positions["ID"].isin(idx)].copy()
        unique_position_id = self.positions["ID"].unique()
        map_dict = dict(zip(unique_position_id, np.arange(len(unique_position_id))))
        self.positions["ID"] = self.positions["ID"].map(map_dict)
        self.positions.set_index("ID", inplace=True)

        self.interventions = self.interventions.loc[self.interventions["ID"].isin(idx)].copy()
        unique_interventions_id = self.interventions["ID"].unique()
        map_dict = dict(zip(unique_interventions_id, np.arange(len(unique_interventions_id))))
        self.interventions["ID"] = self.interventions["ID"].map(map_dict)
        self.interventions.set_index("ID", inplace=True)

        self.length = len(unique_position_id)

    def __len__(self):
        """
        :return: the number of unique ID
        """
        return self.length

    def __getitem__(self, idx):
        obs = self.positions.loc[idx]
        itv = self.interventions.loc[idx]

        return {"idx": idx, "obs": obs, "itv": itv}


def eicu_collate(batch):
    t = torch.tensor(batch[0]["obs"]["Time"].values)
    positions_culumns = [False, True, True, True]
    positions = [torch.from_numpy(b["obs"].iloc[:, positions_culumns].values) for b in batch]
    positions = torch.stack(positions, dim=0)

    intervention_columns = [False, True, True, True, True, True]
    interventions = [torch.from_numpy(b["itv"].iloc[:, intervention_columns].values) for b in batch]
    interventions = torch.stack(interventions, dim=0)

    res = dict()
    res["t"] = t
    res["obs"] = positions
    res["itv"] = interventions

    return res