from torch.nn import ModuleDict
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch
from .dit import DiT
# from models_comformer.pyg_att import Matformer
# from models.pyg_att import MatformerConfig
#from .comformer import iComformerConfig, eComformerConfig
#from .comformer import iComformer, eComformer
from .pddformer import iComformerConfig, eComformerConfig
from .pddformer import iComformer, eComformer
from .flow_matching import FlowMatchingInterpolant
from torch import nn
import torch
import pdb

class LatentDiffusion(nn.Module):
    def __init__(
        self,
        autoencoder_ckpt: str,
        config: iComformerConfig = iComformerConfig(name="iComformer")
    ) -> None:
        super().__init__()
        self.autoencoder_ckpt = autoencoder_ckpt
        model = torch.load(autoencoder_ckpt)
        
        # self.net = Matformer(config)
        self.net = eComformer(config)
        pdb.set_trace()
        self.net.load_state_dict(model,strict=True)
        
        self.denoiser = DiT(d_x=256, d_model=768, nhead=12, num_layers=12, num_datasets=1)
        
        self.interpolent = FlowMatchingInterpolant(min_t=1e-2, corrupt=True, num_timesteps=100,self_condition=True, self_condition_prob=0.5)
    
    def compute_latent_variance(self, x_1):
        """
        x_1: [num_nodes, latent_dim]
        returns variance statistics across nodes
        """
        if x_1.size(0) < 2:
            dim_var = torch.zeros(x_1.size(1), device=x_1.device)
        else:
            dim_var = torch.var(x_1, dim=0, unbiased=False)  # [256]

        return {
            "mean_dim_var": dim_var.mean(),
            "total_var": dim_var.sum(),
            "dim_var": dim_var
        }

    def forward(self, batch: Data,return_latent_stats=False):
        collect_dict = {}
        
        # pdb.set_trace()

        data = batch[0]
        
        x_1 = self.net(batch)

        x_1, mask = to_dense_batch(x_1, data.batch)
        
        
        
        dense_encoded_batch = {"x_1": x_1, "token_mask": mask, "diffuse_mask": mask}
        
        

        # Corrupt batch using the interpolant
        
        self.interpolent.device = dense_encoded_batch["x_1"].device
        noisy_dense_encoded_batch = self.interpolent.corrupt_batch(dense_encoded_batch)

        x_sc = None

        # Run denoiser model
        pred_x = self.denoiser(
            x=noisy_dense_encoded_batch["x_t"],
            t=noisy_dense_encoded_batch["t"],
            dataset_idx=None,
            spacegroup=None,
            mask=mask,
            x_sc=x_sc,
        )

        collect_dict['pred_x']=pred_x
        collect_dict['noisy_dense_encoded_batch']=noisy_dense_encoded_batch

        return collect_dict
