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
from pydantic import BaseSettings as PydanticBaseSettings

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
        self.use_angle = config.use_angle
        self.pre_train = config.pre_train
        self.mask_ratio = config.mask_ratio is not None
        self.position_noise = config.position_noise is not None
        self.lattice_noise = config.lattice_noise is not None
        self.atom_embedding = nn.Linear(
            config.atom_input_features, config.node_features
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

        self.equi_update = ComformerConvEqui(in_channels=config.node_features, out_channels=config.node_features, edge_dim=config.node_features, use_second_order_repr=True)

#         self.fc = nn.Sequential(
#             nn.Linear(config.node_features, config.fc_features), nn.SiLU()
#         )
#         self.sigmoid = nn.Sigmoid()

#         if self.classification:
#             self.fc_out = nn.Linear(config.fc_features, 2)
#             self.softmax = nn.LogSoftmax(dim=1)
#         else:
#             self.fc_out = nn.Linear(
#                 config.fc_features, config.output_features
#             )

        if self.mask_ratio:
            self.mlm_pred = nn.Linear(
                config.node_features, 119
            )
            self.softmax_mlm = nn.LogSoftmax(dim=-1)
        if self.position_noise:
            self.position_mlp = nn.Linear(
                config.node_features, 3
            )
        if self.lattice_noise:
            self.lattice_mlp = nn.Linear(
                config.node_features, 9
            )

        self.link = None
        self.link_name = config.link
        if config.link == "identity":
            self.link = lambda x: x

    def forward(self, data) -> torch.Tensor:
        data, _, _ = data
        node_features = self.atom_embedding(data.x)
        collect_dict = {}
        n_nodes = node_features.shape[0]
        edge_feat = -0.75 / torch.norm(data.edge_attr, dim=1)
        num_edge = edge_feat.shape[0]
        edge_features = self.rbf(edge_feat)

        node_features = self.att_layers[0](node_features, data.edge_index, edge_features)
        node_features = self.equi_update(data, node_features, data.edge_index, edge_features)
        node_features = self.att_layers[1](node_features, data.edge_index, edge_features) 
        node_features = self.att_layers[2](node_features, data.edge_index, edge_features)

        # crystal-level readout
        if self.pre_train:
            if self.mask_ratio:
                atom_prob = self.softmax_mlm(self.mlm_pred(node_features))
                collect_dict["atoms"] = atom_prob
                #collect_list.append(atom_prob)

            if self.position_noise:
                position_pred = self.position_mlp(node_features)
                collect_dict["positions"] = position_pred
                #collect_list.append(position_pred)

            if self.lattice_noise:
                crystal_features = scatter(node_features, data.batch, dim=0, reduce="mean")
                lattice_pred = self.lattice_mlp(crystal_features)
                collect_dict["lattice"] = lattice_pred.view(-1, 3, 3)
                #collect_list.append(lattice_pred.view(-1, 3, 3))
            #return [atom_prob, position_pred, lattice_pred.view(-1, 3, 3)]
            return collect_dict
        else:
            return node_features




class iComformer(nn.Module): # iComFormer
    """att pyg implementation."""

    def __init__(self, config: iComformerConfig = iComformerConfig(name="iComformer")):
        """Set up att modules."""
        super().__init__()
        self.classification = config.classification
        self.use_angle = config.use_angle
        self.pre_train = config.pre_train
        self.mask_ratio = config.mask_ratio is not None
        self.position_noise = config.position_noise is not None
        self.lattice_noise = config.lattice_noise is not None
        self.atom_embedding = nn.Linear(
            config.atom_input_features, config.node_features
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

        self.rbf_angle = nn.Sequential(
            RBFExpansion(
                vmin=-1.0,
                vmax=1.0,
                bins=config.triplet_input_features,
            ),
            nn.Linear(config.triplet_input_features, config.node_features),
            nn.Softplus(),
        )

        self.att_layers = nn.ModuleList(
            [
                ComformerConv(in_channels=config.node_features, out_channels=config.node_features, heads=config.node_layer_head, edge_dim=config.node_features)
                for _ in range(config.conv_layers)
            ]
        )

        self.edge_update_layer = ComformerConv_edge(in_channels=config.node_features, out_channels=config.node_features, heads=config.node_layer_head, edge_dim=config.node_features)

#         self.fc = nn.Sequential(
#             nn.Linear(config.node_features, config.fc_features), nn.SiLU()
#         )
#         self.sigmoid = nn.Sigmoid()

#         if self.classification:
#             self.fc_out = nn.Linear(config.fc_features, 2)
#             self.softmax = nn.LogSoftmax(dim=1)
#         else:
#             self.fc_out = nn.Linear(
#                 config.fc_features, config.output_features
#             )
        
        if self.mask_ratio:
            self.mlm_pred = nn.Linear(
                config.node_features, 119
            )
            self.softmax_mlm = nn.LogSoftmax(dim=-1)
        if self.position_noise:
            self.position_mlp = nn.Linear(
                config.node_features, 3
            )
        if self.lattice_noise:
            self.lattice_mlp = nn.Linear(
                config.node_features, 9
            )

        self.link = None
        self.link_name = config.link
        if config.link == "identity":
            self.link = lambda x: x

    def forward(self, data) -> torch.Tensor:
        data, ldata = data
        collect_dict = {}
        node_features = self.atom_embedding(data.x)
        edge_feat = -0.75 / torch.norm(data.edge_attr, dim=1) # [num_edges]
        edge_nei_len = -0.75 / torch.norm(data.edge_nei, dim=-1) # [num_edges, 3]
        edge_nei_angle = bond_cosine(data.edge_nei, data.edge_attr.unsqueeze(1).repeat(1, 3, 1)) # [num_edges, 3, 3] -> [num_edges, 3]
        num_edge = edge_feat.shape[0]
        edge_features = self.rbf(edge_feat)
        edge_nei_len = self.rbf(edge_nei_len.reshape(-1)).reshape(num_edge, 3, -1)
        edge_nei_angle = self.rbf_angle(edge_nei_angle.reshape(-1)).reshape(num_edge, 3, -1)

        node_features = self.att_layers[0](node_features, data.edge_index, edge_features) 
        edge_features = self.edge_update_layer(edge_features, edge_nei_len, edge_nei_angle)
        node_features = self.att_layers[1](node_features, data.edge_index, edge_features) 
        node_features = self.att_layers[2](node_features, data.edge_index, edge_features)
        node_features = self.att_layers[3](node_features, data.edge_index, edge_features)

        if self.pre_train:
            if self.mask_ratio:
                atom_prob = self.softmax_mlm(self.mlm_pred(node_features))
                collect_dict["atoms"] = atom_prob
                #collect_list.append(atom_prob)

            if self.position_noise:
                position_pred = self.position_mlp(node_features)
                collect_dict["positions"] = position_pred
                #collect_list.append(position_pred)

            if self.lattice_noise:
                crystal_features = scatter(node_features, data.batch, dim=0, reduce="mean")
                lattice_pred = self.lattice_mlp(crystal_features)
                collect_dict["lattice"] = lattice_pred.view(-1, 3, 3)
                #collect_list.append(lattice_pred.view(-1, 3, 3))
            #return [atom_prob, position_pred, lattice_pred.view(-1, 3, 3)]
            return collect_dict
        else:
            return node_features


