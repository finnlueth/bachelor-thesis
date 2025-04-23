from bio2token_main.bio2token.models.fsq_ae import FSQ_AE
import torch
from einops import rearrange, pack, unpack


class Bio2TokenExtension(FSQ_AE):
    def __init__(self, config, *args, **kwargs):
        super(Bio2TokenExtension, self).__init__(config, *args, **kwargs)
    
    def forward(self, batch: dict, mode="inference"):
        return super().forward(batch, mode)

    def encode(self, batch: dict) -> torch.Tensor:
        input_mask = batch["atom_mask"].view(-1, self.config.max_len)
        x_input = batch["coords_groundtruth"].squeeze(1).view(-1, self.config.max_len, 3).clone()
        x_input[~input_mask] = 0
        feature_track = []
        if len(feature_track) > 0:
            feature_track = torch.stack(feature_track, dim=-1)
            x_input = torch.cat([x_input, feature_track], dim=-1)
        h_nodes = self.encoder(x_input).logits
        if self.config.use_fsq:
            h_nodes_cp, batch["indices"] = self.fsq(h_nodes_cp.view(-1, self.config.max_len, self.config.node_hidden_dims_s))
        if self.config.manual_masking:
            h_nodes_cp[~input_mask] = 0
        return h_nodes_cp
    
    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        