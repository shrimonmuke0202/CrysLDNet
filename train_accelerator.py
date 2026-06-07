from functools import partial

# from pathlib import Path
from typing import Any, Dict, Union

import ignite
import torch
from accelerate import Accelerator
# from accelerate import set_seed
from accelerate.logging import get_logger
from ignite.contrib.handlers import TensorboardLogger
try:
    from ignite.contrib.handlers.stores import EpochOutputStore
except Exception as exp:
    from ignite.handlers.stores import EpochOutputStore

    pass
from ignite.handlers import EarlyStopping
from ignite.contrib.handlers.tensorboard_logger import (
    global_step_from_engine,
)
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import (
    Events,
    create_supervised_evaluator,
    create_supervised_trainer,
)
from ignite.contrib.metrics import ROC_AUC, RocCurve
from ignite.metrics import (
    Accuracy,
    Precision,
    Recall,
    ConfusionMatrix,
)
import pickle as pk
import numpy as np
from ignite.handlers import Checkpoint, DiskSaver, TerminateOnNan
from ignite.metrics import Loss, MeanAbsoluteError
from torch import nn
from config import TrainingConfig


from jarvis.db.jsonutils import dumpjson
import json
import pprint

import os
import warnings
from ignite.metrics import Metric
from ignite.exceptions import NotComputableError
import time

# torch config
torch.set_default_dtype(torch.float32)

device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")



class CustomMetric_lattice(Metric):
    def __init__(self, output_transform=lambda x: x):
        self._samplenum = []
        self._mae = []
        self._MAE_loss = nn.L1Loss()
        # self._MAE_loss = nn.MSELoss()
        super(CustomMetric_lattice, self).__init__(output_transform=output_transform)
    def reset(self):
        self._samplenum = []
        self._mae = []
        
    def update(self, output):
        y_pred, y_gt = output
        self._samplenum.append(y_pred["lattice"].shape[0])
        self._mae.append(self._MAE_loss(y_pred["lattice"].float(), y_gt["lattice"].t().float().view(-1,3,3)).item())

    def compute(self):
        return sum(w*v for w, v in zip(self._mae, self._samplenum)) / sum(self._samplenum)


class CustomMetric_position(Metric):
    def __init__(self, output_transform=lambda x: x):
        self._samplenum = []
        self._mae = []
        self._MAE_loss = nn.L1Loss()
        # self._MAE_loss = nn.MSELoss()
        super(CustomMetric_position, self).__init__(output_transform=output_transform)
        
    def reset(self):
        self._samplenum = []
        self._mae = []
        
    def update(self, output):
        y_pred, y_gt = output
        self._samplenum.append(y_pred["positions"].shape[0])
        self._mae.append(self._MAE_loss(y_pred["positions"].float(), y_gt["positions"].t().float()).item())

    def compute(self):
        return sum(w*v for w, v in zip(self._mae, self._samplenum)) / sum(self._samplenum)

class CustomMetric_atom(Metric):
    def __init__(self, output_transform=lambda x: x):
        self._num_examples = 0
        self._correct =0
        super(CustomMetric_atom, self).__init__(output_transform=output_transform)

    def reset(self):
        self._correct = 0
        self._num_examples = 0

    def update(self, output):
        y_pred_dict, y_gt_dict = output
        y_pred = y_pred_dict["atoms"]
        label = y_gt_dict["atoms"]
        mask = y_gt_dict["mask"]
        _, y_pred_class = torch.max(y_pred, 1)
        self._correct += ((y_pred_class==label)*mask).sum().item()
        self._num_examples += mask.sum().item()
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('CustomMetric must have at least one example before it can be computed.')
        return self._correct / self._num_examples
    
    
def activated_output_transform(output):
    """Exponentiate output."""
    y_pred, y = output
    y_pred = torch.exp(y_pred)
    y_pred = y_pred[:, 1]
    return y_pred, y


def make_standard_scalar_and_pca(output):
    """Use standard scalar and PCS for multi-output data."""
    sc = pk.load(open(os.path.join(tmp_output_dir, "sc.pkl"), "rb"))
    y_pred, y = output
    y_pred = torch.tensor(sc.transform(y_pred.cpu().numpy()), device=device)
    y = torch.tensor(sc.transform(y.cpu().numpy()), device=device)
    return y_pred, y


def thresholded_output_transform(output):
    """Round off output."""
    y_pred, y = output
    y_pred = torch.round(torch.exp(y_pred))
    # print ('output',y_pred)
    return y_pred, y


def group_decay(model):
    """Omit weight decay from bias and batchnorm params."""
    decay, no_decay = [], []

    for name, p in model.named_parameters():
        if "bias" in name or "bn" in name or "norm" in name:
            no_decay.append(p)
        else:
            decay.append(p)

    return [
        {"params": decay},
        {"params": no_decay, "weight_decay": 0},
    ]


def setup_optimizer(params, config: TrainingConfig):
    """Set up optimizer for param groups."""
    if config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=config.weight_decay,
        )
    return optimizer