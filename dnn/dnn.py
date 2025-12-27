import torch
import torch.nn.functional as F
import torch.nn as nn


from modules import SelfAttention, GatedMLP, DropPath, modulate

class SpaceTimeBlock(nn.Module):
    """
        A simple neural network block that implements factorized space time attention over video signals in R^d by first aggregating information across all pixels within frames with self attention;
        followed by aggregating temporal information for pixels across all frames again with self attention mechanism.


        args:
            dim : int - dimensionality of the neural vectorspace for each fragment of input signal
            y_dim : int - dimensionality of f(class_label, flow_time) conditioning vector
            norm_cls : norm layer type (LayerNorm)
            drop_path : dropout prob

            x : (B, T, HW, C) - input video signal in R^d
            y : (B, C) - aggregation of cts time variable and neural vectorspace representation of class conditioning for approximating the flow field
    """

    def __init__(self, dim: int, y_dim=None, norm_cls=nn.LayerNorm, drop_path=0.0):
        super().__init__()
        self.dim = dim
        y_dim = dim if y_dim is None else y_dim

        self.spatial_att = SelfAttention(dim, kqv_bias=True)
        self.temporal_att = SelfAttention(dim, kqv_bias=True)
        self.norm_s = norm_cls(dim)
        self.norm_t = norm_cls(dim)
        
        self.norm_mlp = norm_cls(dim)
        mlp_activation = lambda: nn.SiLU()
        self.mlp = GatedMLP(dim, activation=mlp_activation, bias=True)

        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(y_dim, 9*dim, bias=True))
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
    
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """
        Forward
        
        args:
        x : tensor (B, T, HW, C) representation from previous block
        y : tensor (B, C) f(flow_time, class_label) conditioning vector

        returns x
        """

        # do adaLN(spatial_attention(x)) then adaLN(temporal_attention(x)) then adaLN(temporal_MLP(x))

        scale_space_msa, gate_space_msa, shift_space_msa, scale_time_msa, gate_time_msa, shift_time_msa, scale_mlp, gate_mlp, shift_mlp = self.adaLN_modulation(y).chunk(9, dim=-1)

        # x (B, T, HW, C)
        x = x +  self.drop_path(gate_space_msa[:,None, None, :] * self.spatial_att(modulate(self.norm_s(x), shift_space_msa, scale_space_msa))) # (B, T, HW, C) information aggregation across all HW
        # prime the signal for temporal attention
        x = x.transpose(1, 2).contiguous() # x (B, HW, T, C)
        x = x + self.drop_path(gate_time_msa[:, None, None, :] * self.temporal_att(modulate(self.norm_t(x), shift_time_msa, scale_time_msa))) # (B, HW, T, C)
        # restructure signal in original format for next block 
        x = x.transpose(1, 2).contiguous() # x (B, T, HW, C)
        # mlp
        x = x + self.drop_path(gate_mlp[:, None, None, :] * self.mlp(modulate(self.norm_mlp(x), shift_mlp, scale_mlp))) # (B, T, HW, C)


        return x


                
            

            



        