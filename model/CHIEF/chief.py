from turtle import forward
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from .transformer import TransformerEncoder, TransformerEncoderLayer
# from mamba_ssm import Mamba


class SelfAttention(nn.Module):
    def __init__(self, num_models=5,d_model=128,need_weights=False):
        super().__init__()
        if not need_weights:
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=4,
                    dim_feedforward=512,
                    batch_first=True
                ),
                num_layers=3
            )
        else:
            self.encoder = TransformerEncoder(
                TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=4,
                    dim_feedforward=512,
                    batch_first=True
                ),
                num_layers=3
            )
        self.inproj = nn.Linear(num_models*21,d_model)
        self.outproj = nn.Linear(d_model,21)
        self.linear = nn.Linear(num_models,1)
        self.positional_encoding = self.create_positional_encoding(d_model, 100000)
        self.encoder_output = None
        self.attention_weights = []
        
    def create_positional_encoding(self, d_model, max_len):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe
        
    def forward(self, log_probs,mask=None):
        probs = torch.exp(log_probs)
        h_probs = self.inproj(probs.view(probs.shape[0],probs.shape[1],-1)) #[B,L,21,5] -> [B,L,d_model]
        h_probs += self.positional_encoding[:,:h_probs.size(1),:].to(h_probs.device)
        h_probs = self.encoder(h_probs,src_key_padding_mask=mask) #[B,L,d_model]
        logits = self.outproj(h_probs) #[B,L,21]
        log_probs = F.log_softmax(logits,dim=-1)
        return log_probs
    
    def logits(self,log_probs,mask=None):
        probs = torch.exp(log_probs)
        h_probs = self.inproj(probs.view(probs.shape[0],probs.shape[1],-1)) #[B,L,21,5] -> [B,L,d_model]
        h_probs += self.positional_encoding[:,:h_probs.size(1), :].to(h_probs.device)
        h_probs = self.encoder(h_probs,src_key_padding_mask=mask) #[B,L,d_model]
        logits = self.outproj(h_probs) #[B,L,21]
        return logits
    
    def _encoder_hook(self, module, input, output):
        self.encoder_output = output
        
    def _attention_hook(self, module, input, output):
        if isinstance(module, nn.MultiheadAttention):
            self.attention_weights.append(output[1].detach())

    def get_attention(self, log_probs, mask):
        self.attention_weights = []
        # Register hooks temporarily
        encoder_hook_handle = self.encoder.register_forward_hook(self._encoder_hook)
        attention_hook_handles = [layer.self_attn.register_forward_hook(self._attention_hook) for layer in self.encoder.layers]
        self.forward(log_probs, mask)
        # Remove hooks after forward pass
        encoder_hook_handle.remove()
        for handle in attention_hook_handles:
            handle.remove()
        
        return self.attention_weights, self.encoder_output