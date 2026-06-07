import os
import csv
from jarvis.core.atoms import Atoms
from tqdm import tqdm
import torch.nn as nn
import pandas as pd
import time
from pandarallel import pandarallel
import pdb
import torch
import numpy as np


def one_poscar_data(file_name):
    atoms = Atoms.from_poscar(file_name)
    info = {}
    info["atoms"] = atoms.to_dict()
    info["jid"] = file_name
    info["target"] = float(0.0)
    return info


def get_data(root_dir, file_format='poscar', debug=False):
    print("#############get data##################")
    t1 = time.time()
    pandarallel.initialize(nb_workers=32,progress_bar=True)
    id_prop_dat = os.path.join(root_dir, "id_prop_new.csv")
    df = pd.read_csv(id_prop_dat)
    # df = df.head(1000)
    
    n_outputs = np.zeros(len(df))
    dataset = df['ID'].parallel_apply(one_poscar_data).values
    
    t2 = time.time()
    print(f"#############get data done {t2-t1} s##################")
    return list(dataset), list(n_outputs)

class Criterion(nn.Module):
    def __init__(self, mask_ratio):
        super(Criterion, self).__init__()
        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.l2 = nn.L1Loss()
        self._samplenum = []
        self._mae = []
        self.mask_ratio = (mask_ratio is None)
        
        self.loss_step = []
    def loss_atoms(self, label_pred, label, mask):
        # pdb.set_trace()
        ce_loss_items = self.ce(label_pred, label)
        mean_loss = (ce_loss_items*mask).sum()/mask.sum()
        return mean_loss

    def reset(self):
        self._samplenum = []
        self._mae = []
            
    def update(self, output):
        y_pred, y_gt = output
        for k, value in y_pred.items():
            _samplenum = value.shape[0]
            break
        self._samplenum.append(_samplenum)
        self._mae.append(self.forward(y_pred, y_gt).item())
            
    def compute(self):
        return sum(w*v for w, v in zip(self._mae, self._samplenum)) / sum(self._samplenum)
                       
    def forward(self, y_pred, y_gt):
        
        all_loss = 0
        # pdb.set_trace()
        if "atoms" in y_pred.keys():
            atom_loss = self.loss_atoms(y_pred["atoms"], y_gt["atoms"], y_gt["mask"])
            all_loss += atom_loss
        if "positions" in y_pred.keys():
            position_loss = self.l2(y_pred["positions"].float(), y_gt["positions"].t().float())
            all_loss += position_loss
        if "lattice" in y_pred.keys():
            lattice_loss = self.l2(y_pred["lattice"].float(), y_gt["lattice"].t().float().view(-1,3,3))
            all_loss += lattice_loss

        loss_kl = y_pred["posterior"].kl()
        all_loss+=(0.00001*loss_kl).mean()  #KL divergence loss

        return all_loss
    
    
class CriterionLDM(nn.Module):
    def __init__(self):
        super(CriterionLDM, self).__init__()
        self._samplenum = []
        self._mae = []

        self.loss_step = []

    def reset(self):
        self._samplenum = []
        self._mae = []
            
    def update(self, output):
        # pdb.set_trace()
        y_pred, y_gt = output
        for k, value in y_pred.items():
            _samplenum = value.shape[0]
            break
        self._samplenum.append(_samplenum)
        self._mae.append(self.forward(y_pred['pred_x'], y_pred['noisy_dense_encoded_batch']).item())
            
    def compute(self):
        return sum(w*v for w, v in zip(self._mae, self._samplenum)) / sum(self._samplenum)
                       
    def forward(self, pred_x, noisy_dense_encoded_batch):
        gt_x_1 = noisy_dense_encoded_batch["x_1"]
        norm_scale = 1 - torch.min(noisy_dense_encoded_batch["t"].unsqueeze(-1), torch.tensor(0.9))
        x_error = (gt_x_1 - pred_x) / norm_scale
        loss_mask = (
            noisy_dense_encoded_batch["token_mask"] * noisy_dense_encoded_batch["diffuse_mask"]
        )
        loss_denom = torch.sum(loss_mask, dim=-1) * pred_x.size(-1)
        x_loss = torch.sum(x_error**2 * loss_mask[..., None], dim=(-1, -2)) / loss_denom
        loss_dict = {"loss": x_loss.mean(), "x_loss": x_loss}
        
        return loss_dict['loss']
