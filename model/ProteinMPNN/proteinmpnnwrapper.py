import torch
from .protein_mpnn_utils import ProteinMPNN
import numpy as np


class ProteinMPNNWrapper():
    def __init__(self,state_dict,device=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.proteinmpnn = ProteinMPNN(node_features=128,
                                       edge_features=128,
                                       hidden_dim=128,
                                       num_decoder_layers=3,
                                       num_encoder_layers=3,
                                       k_neighbors=48,
                                       dropout=0.1,
                                       augment_eps=0.2,
                                       num_letters=21)
        state_dict = torch.load(state_dict)
        self.proteinmpnn.load_state_dict(state_dict['model_state_dict'])
        self.proteinmpnn.to(self.device)

    def forward(self,X, S, mask, chain_M, residue_idx, chain_encoding_all,train_mode=True):
        if train_mode:
            randn=torch.randn(chain_M.shape, device=X.device)
            log_probs = self.proteinmpnn(X, S, mask, chain_M, residue_idx, chain_encoding_all,randn)
        else:
            output_dict = self.sample_output(X, S, mask, chain_M, residue_idx, chain_encoding_all, omit_X=False)
            log_probs = torch.nn.functional.log_softmax(output_dict["logits"],dim=-1)
        return log_probs
        
    def logits(self,X, S, mask, chain_M, residue_idx, chain_encoding_all):
        output_dict = self.sample_output(X, S, mask, chain_M, residue_idx, chain_encoding_all, omit_X=False)
        logits = output_dict["logits"]
        return logits
    
    def sample(self,X, S, mask, chain_M, residue_idx, chain_encoding_all, temperature=1.0):
        output_dict = self.sample_output(X, S, mask, chain_M, residue_idx, chain_encoding_all, temperature)
        S_sampled = output_dict["S"]
        return S_sampled
        
    def sample_output(self,X, S, mask, chain_M, residue_idx, chain_encoding_all, temperature=1.0, omit_X=True):
        alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
        randn = torch.randn(chain_M.shape, device=X.device)
        if omit_X:
            omit_AAs_np = np.array([False]*20+[True]).astype(np.float32) # omit X
        else:
            omit_AAs_np = np.array([False]*21).astype(np.float32)
        bias_AAs_np = np.zeros(len(alphabet))
        bias_by_res = torch.zeros(X.shape[0],X.shape[1],len(alphabet)).to(self.device)
        output_dict = self.proteinmpnn.sample(
            X = X,
            randn = randn,
            S_true = S,
            chain_mask = chain_M,
            chain_encoding_all = chain_encoding_all,
            residue_idx = residue_idx,
            mask = mask,
            temperature = temperature,
            chain_M_pos = mask,
            omit_AAs_np = omit_AAs_np,
            bias_AAs_np = bias_AAs_np,
            bias_by_res = bias_by_res
        )
        return output_dict
        
    def score(self,X, S, S_test, mask, chain_M, residue_idx, chain_encoding_all):
        """ Positive log probabilities """
        mask_for_loss = mask * chain_M
        criterion = torch.nn.NLLLoss(reduction='none')
        log_probs = self.forward(X, S, mask, chain_M, residue_idx, chain_encoding_all)
        loss = criterion(
            log_probs.contiguous().view(-1,log_probs.size(-1)),
            S_test.contiguous().view(-1)
        ).view(S_test.size())
        scores = torch.sum(loss * mask_for_loss, dim=-1) / torch.sum(mask_for_loss, dim=-1)
        return -scores
        