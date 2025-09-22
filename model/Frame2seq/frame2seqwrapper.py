import os
from glob import glob
from time import time
from turtle import forward
import torch

from .frame2seq.model.Frame2seq import frame2seq
from .frame2seq.openfold.utils.tensor_utils import one_hot
from .frame2seq.utils.design import design
from .frame2seq.utils.score import score

import tqdm


class Frame2SeqWrapper():
    def __init__(self,device=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.models = []
        model_ckpts = ['./checkpoints/frame2seq/model1.ckpt',
                       './checkpoints/frame2seq/model2.ckpt',
                       './checkpoints/frame2seq/model3.ckpt']
        for ckpt_file in model_ckpts:
            # print(f"Loading {ckpt_file}...")
            self.models.append(
                frame2seq.load_from_checkpoint(ckpt_file, strict=False).eval().to(
                    self.device))
    
    @staticmethod
    def _add_Cb(X):
        """
        'N', 'CA', 'C', 'O' -> 'N', 'CA', 'C', 'CB', 'O'
        """
        B, L = X.shape[:2]
        X_Cb = torch.zeros(B,L,5,3)
        b = X[:,:,1,:] - X[:,:,0,:]
        c = X[:,:,2,:] - X[:,:,1,:]
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + X[:,:,1,:]
        Ca = X[:,:,1,:]
        N = X[:,:,0,:]
        C = X[:,:,2,:]
        O = X[:,:,3,:]
        X_Cb[:,:,0,:] = N
        X_Cb[:,:,1,:] = Ca
        X_Cb[:,:,2,:] = C
        X_Cb[:,:,3,:] = Cb
        X_Cb[:,:,4,:] = O
        return X_Cb
    
    def forward(self,X, S, mask, chain_M, residue_idx, chain_encoding_all):
        B, L = X.shape[:2]
        input_aatype_onehot = torch.zeros((B,L,21),device=self.device)
        input_aatype_onehot[:,:,20] = 1 
        seq_mask = mask.bool()
        logits = self.logits(X, S, mask, chain_M, residue_idx, chain_encoding_all)
        log_probs = torch.nn.functional.log_softmax(logits,dim=-1)
        return log_probs
    
    def logits(self,X, S, mask, chain_M, residue_idx, chain_encoding_all):
        B, L = X.shape[:2]
        input_aatype_onehot = torch.zeros((B,L,21),device=self.device)
        input_aatype_onehot[:,:,20] = 1 
        seq_mask = mask.bool()
        with torch.no_grad():
            X = self._add_Cb(X)
            pred_seq1 = self.models[0].forward(X, seq_mask, input_aatype_onehot) #[B,L,21] <mask>
            pred_seq2 = self.models[1].forward(X, seq_mask, input_aatype_onehot)
            pred_seq3 = self.models[2].forward(X, seq_mask, input_aatype_onehot)
            logits = (pred_seq1 + pred_seq2 + pred_seq3) / 3
        return logits


    def sample(self,X, S, mask, chain_M, residue_idx, chain_encoding_all,temperature=1.0):
        seq_mask = mask.bool()
        B,L = X.shape[:2]
        logits = self.logits(X, S, mask, chain_M, residue_idx, chain_encoding_all)

        logits = logits / temperature
        probs = torch.nn.functional.softmax(logits, dim=-1)
        probs[torch.isnan(probs)] = 1e-6
        probs_flat = probs.view(-1,probs.shape[-1])
        sampled_seq = torch.multinomial(probs_flat[:,:20],
                                        num_samples=1,
                                        replacement=True).squeeze(-1)
        return sampled_seq.view(B,L)
    
    def score(self,X, S, S_test,mask, chain_M, residue_idx, chain_encoding_all):
        log_probs = self.forward(X, S, mask, chain_M, residue_idx, chain_encoding_all)
        mask_for_loss = mask * chain_M
        criterion = torch.nn.NLLLoss(reduction='none')
        loss = criterion(
            log_probs.contiguous().view(-1,log_probs.size(-1)),
            S_test.contiguous().view(-1)
        ).view(S_test.size())
        scores = torch.sum(loss * mask_for_loss, dim=-1) / torch.sum(mask_for_loss, dim=-1)
        return -scores