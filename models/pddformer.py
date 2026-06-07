"""Implementation based on the template of Matformer."""

from typing import Tuple
import math
import numpy as np
import torch
import torch.nn.functional as F
from pydantic.typing import Literal
from torch import nn
from .utils import RBFExpansion
from utils import BaseSettings
from features import angle_emb_mp
from torch_scatter import scatter
from .transformer import ComformerConv, ComformerConv_edge, ComformerConvEqui
import pdb

class DiagonalGaussianDistribution:
    """Diagonal Gaussian distribution with mean and logvar parameters.

    Adapted from: https://github.com/CompVis/latent-diffusion, with modifications for our tensors,
    which are of shape (N, d) instead of (B, H, W, d) for 2D images.
    """

    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=-1)  # split along channel dim
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=1
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=1,
                )

    def mode(self):
        return self.mean

    def __repr__(self):
        return f"DiagonalGaussianDistribution(mean={self.mean}, logvar={self.logvar})"


class iComformerConfig(BaseSettings):
    """Hyperparameter schema for jarvisdgl.models.cgcnn."""

    name: Literal["iComformer"]
    conv_layers: int = 4
    edge_layers: int = 1
    atom_input_features: int = 92
    edge_features: int = 256
    triplet_input_features: int = 256
    node_features: int = 256
    fc_layers: int = 1
    fc_features: int = 256
    output_features: int = 1
    node_layer_head: int = 1
    edge_layer_head: int = 1
    nn_based: bool = False

    link: Literal["identity", "log", "logit"] = "identity"
    zero_inflated: bool = False
    use_angle: bool = False
    angle_lattice: bool = False
    classification: bool = False
    pre_train: bool = False
    position_noise: float = None
    lattice_noise: float = None
    mask_ratio: float = None

    class Config:
        """Configure model settings behavior."""

        env_prefix = "jv_model"

class ZeoConv(nn.Module):
    def __init__(self, node_features):
        super().__init__()
        self.feature = node_features
        self.lin1 = nn.Linear(node_features, node_features * 2)
        self.lin2 = nn.Linear(node_features, node_features)
        self.lin3 = nn.Linear(node_features, node_features)
        self.bn = nn.BatchNorm1d(node_features)
        self.act = nn.GELU()
        self.drop = nn.Dropout(0.1)

    def forward(self, x, adj):
        adj = adj + x
        x1, x2 = self.lin1(self.bn(adj)).chunk(chunks=2, dim=1)
        x1 = self.lin2(x1)
        x2 = self.drop(self.act(x2))
        x = self.lin3(x1 * x2) + x
        return x, adj


class eComformerConfig(BaseSettings):
    """Hyperparameter schema for jarvisdgl.models.cgcnn."""

    name: Literal["eComformer"]
    conv_layers: int = 4
    edge_layers: int = 1
    atom_input_features: int = 92
    edge_features: int = 256
    triplet_input_features: int = 256
    node_features: int = 256
    fc_layers: int = 1
    fc_features: int = 256
    output_features: int = 1
    node_layer_head: int = 1
    edge_layer_head: int = 1
    nn_based: bool = False

    link: Literal["identity", "log", "logit"] = "identity"
    zero_inflated: bool = False
    use_angle: bool = False
    angle_lattice: bool = False
    classification: bool = False
    pre_train: bool = False
    position_noise: float = None
    lattice_noise: float = None
    mask_ratio: float = None

    class Config:
        """Configure model settings behavior."""

        env_prefix = "jv_model"


def bond_cosine(r1, r2):
    bond_cosine = torch.sum(r1 * r2, dim=-1) / (
        torch.norm(r1, dim=-1) * torch.norm(r2, dim=-1)
    )
    bond_cosine = torch.clamp(bond_cosine, -1, 1)
    return bond_cosine




class eComformer(nn.Module): # eComFormer
    """att pyg implementation."""

    def __init__(self, config: eComformerConfig = eComformerConfig(name="eComformer")):
        """Set up att modules."""
        super().__init__()
        self.classification = config.classification
        self.pre_train = config.pre_train
        self.mask_ratio = config.mask_ratio is not None
        self.position_noise = config.position_noise is not None
        self.lattice_noise = config.lattice_noise is not None
        self.use_angle = config.use_angle
        self.atom_embedding = nn.Linear(
            119, config.node_features
        )
        self.rbf = nn.Sequential(
            RBFExpansion(
                vmin=-4.0,
                vmax=0.0,
                bins=config.edge_features,
            ),
            nn.Linear(config.edge_features, config.node_features),
            nn.Softplus(),
        )

        self.att_layers = nn.ModuleList(
            [
                ComformerConv(in_channels=config.node_features, out_channels=config.node_features, heads=config.node_layer_head, edge_dim=config.node_features)
                for _ in range(config.conv_layers)
            ]
        )

        self.fc = nn.Sequential(
            nn.Linear(config.node_features, config.fc_features), nn.SiLU()
        )
        self.sigmoid = nn.Sigmoid()

        self.embed_adj = nn.Sequential(
            nn.Linear(93, config.node_features),
        )

        self.lin = nn.ModuleList(
            [
                ZeoConv(config.node_features)
                for _ in range(config.conv_layers-1)
            ]
        )
        
        if self.pre_train:
            pass

        if self.mask_ratio:
            self.mlm_pred = nn.Linear(
                config.node_features, 119
            )
            self.softmax_mlm = nn.LogSoftmax(dim=-1)
        
        self.position_mlp = nn.Linear(
            config.node_features, 3
        )
        
        self.lattice_mlp = nn.Linear(
            config.node_features, 9
        )

        self.quant_conv = torch.nn.Linear(config.node_features, 2 * config.node_features, bias=False)
        
        if self.classification:
            self.fc_out = nn.Linear(config.fc_features, 2)
            self.softmax = nn.LogSoftmax(dim=1)
        else:
            self.fc_out = nn.Linear(
                config.fc_features, config.output_features
            )

        self.link = None
        self.link_name = config.link
        if config.link == "identity":
            self.link = lambda x: x

    def forward(self, data,sample_posterior=True) -> torch.Tensor:
        # pdb.set_trace()
        data,ldat = data
        collect_dict = {}
        node_features = self.atom_embedding(data.x)
        n_nodes = node_features.shape[0]
        edge_feat = -0.75 / torch.norm(data.edge_attr, dim=1)
        num_edge = edge_feat.shape[0]
        edge_features = self.rbf(edge_feat)
        
        # pdb.set_trace()
        
        adj_feature = self.embed_adj(data.adj)

        node_features = self.att_layers[0](node_features, data.edge_index, edge_features)
        node_features, adj_feature = self.lin[0](node_features, adj_feature)
        node_features = self.att_layers[1](node_features, data.edge_index, edge_features)
        node_features, adj_feature = self.lin[1](node_features, adj_feature)
        node_features = self.att_layers[2](node_features, data.edge_index, edge_features)
        node_features, adj_feature = self.lin[2](node_features, adj_feature)
        node_features = self.att_layers[3](node_features, data.edge_index, edge_features)

        # crystal-level readout
        if self.pre_train:

            collect_dict["moments"] = self.quant_conv(node_features)
        
            collect_dict["posterior"] = DiagonalGaussianDistribution(collect_dict["moments"])
            
            if sample_posterior:
                node_features = collect_dict["posterior"].sample()
            else:
                node_features = collect_dict["posterior"].mode()

            if self.mask_ratio:
                atom_prob = self.softmax_mlm(self.mlm_pred(node_features))
                collect_dict["atoms"] = atom_prob
            
            position_pred = self.position_mlp(node_features)
            collect_dict["positions"] = position_pred
            
            crystal_features = scatter(node_features, data.batch, dim=0, reduce="mean")
            lattice_pred = self.lattice_mlp(crystal_features)
            collect_dict["lattice"] = lattice_pred.view(-1, 3, 3)
                
            return collect_dict
        else:
            return node_features