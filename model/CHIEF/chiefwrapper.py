import torch
import torch.nn as nn
# from models.voting import SelfAttention
from .chief import SelfAttention

class ChiefWrapper(nn.Module):
    def __init__(self,model_dict,
                 base_models_list = ['ProteinMPNN-vanilla','ProteinMPNN-soluble','ESM-IF','Frame2seq','PiFold'],
                 state_dict = './checkpoints/chief/chief.pth',
                 need_weights = False,
                 device=None,train_mode=False):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.selfattention = SelfAttention(num_models=len(model_dict),d_model=128,need_weights=need_weights)
        self.base_models_list = base_models_list
        self.train_mode = train_mode
        if not self.train_mode:
            state_dict = torch.load(state_dict,map_location=self.device)
            self.selfattention.load_state_dict(state_dict)
        self.selfattention = self.selfattention.to(self.device)
        self.models = model_dict
        
    def forward(self,X, S, mask, chain_M, residue_idx, chain_encoding_all):
        log_probs_list = []
        with torch.no_grad():
            for model in self.base_models_list:
                log_probs_list.append(self.models[model].forward(X, S, mask, chain_M, residue_idx, chain_encoding_all).to(self.device))
        for i in log_probs_list:
            print(i.shape)
        log_probs = torch.stack(log_probs_list,dim=-1)
        log_probs = self.selfattention.forward(log_probs,mask=(1.-mask))
        return log_probs
    
    def sample(self,X, S, mask, chain_M, residue_idx, chain_encoding_all,temperature=1.0):
        with torch.no_grad():
            log_probs_list = []
            for model in self.base_models_list:
                log_probs_list.append(self.models[model].forward(X, S, mask, chain_M, residue_idx, chain_encoding_all))
            log_probs = torch.stack(log_probs_list,dim=-1)
            logits = self.selfattention.logits(log_probs,mask=(1.-mask).bool())
            B,L = X.shape[:2]
            logits_flat = logits.view(-1,logits.shape[-1])
            probs_flat = torch.nn.functional.softmax(logits_flat/temperature,dim=-1)
            probs_flat = probs_flat[:,:20]
            probs_flat[torch.isnan(probs_flat)] = 1e-6
            sampled_seq = torch.multinomial(probs_flat,
                                            num_samples=1,
                                            replacement=True)
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