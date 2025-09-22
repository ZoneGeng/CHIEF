import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .esm import pretrained
from .esm.inverse_folding.util import CoordBatchConverter
import warnings
warnings.filterwarnings("ignore")

class ESMIFWrapper(nn.Module):
    def __init__(self,state_dict = "./checkpoints/esmif/esm_if1_gvp4_t16_142M_UR50.pt",device=None):
        super(ESMIFWrapper, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        model_data = torch.load(state_dict)
        self.model, self.alphabet = pretrained.load_model_and_alphabet_core("esm_if1_gvp4_t16_142M_UR50", model_data, None)
        self.model.to(self.device)
    
    def esm2mpnn(self,logits):
        """
        [B,L,35] -> [B,L,21]
        """
        esm_toks = ('<null_0>','<pad>','<eos>','<unk>','L','A','G','V','S','E','R','T','I','D','P','K','Q','N','F','Y','M','H','W','C','X','B','U','Z','O','.','-','<null_1>','<mask>','<cath>','<af2>')
        mpnn_toks = tuple(list('ACDEFGHIKLMNPQRSTVWYX'))
        esm_to_mpnn_idx = [esm_toks.index(tok) for tok in mpnn_toks]
        logits = logits[:,:,esm_to_mpnn_idx]
        return logits

    def autoregressive_sample(self, X, S, mask, chain_M, residue_idx, chain_encoding_all, temperature=1.0):
        # convert coords to coords_list
        B, L = X.shape[:2]
        X = X[:,:,:3,:]
        coords_list = [X[i, residue_idx[i] != -100] for i in range(B)]

        self.model.eval()
        with torch.no_grad():
            batch_converter = CoordBatchConverter(self.model.decoder.dictionary)
            batch_coords, confidence, _, _, padding_mask = (
                batch_converter([(coord, None, None) for coord in coords_list], device=X[0].device)
            )
            # Start with prepend token
            mask_idx = self.model.decoder.dictionary.get_idx('<mask>')
            sampled_tokens = torch.full((B, 1+L), mask_idx, dtype=int)
            sampled_tokens[:, 0] = self.model.decoder.dictionary.get_idx('<cath>')
            incremental_state = dict()
            encoder_out = self.model.encoder(batch_coords, padding_mask, confidence)
        
            sampled_tokens = sampled_tokens.to(self.device)
            logits_list = []
            # Decode one token at a time
            for i in range(1, L+1):
                logits, _ = self.model.decoder(
                    sampled_tokens[:, :i], 
                    encoder_out,
                    incremental_state=incremental_state,
                )
                logits = logits.squeeze(-1)
                logits_list.append(logits)
                logits /= temperature
                probs = F.softmax(logits, dim=-1)
                if torch.all(sampled_tokens[:, i] == torch.full((B,), mask_idx).to(self.device)):
                    sampled_tokens[:, i] = torch.multinomial(probs, 1).squeeze(-1)
            sampled_seq = sampled_tokens[:, 1:]
        logits = torch.stack(logits_list, dim=0)
        # 调整维度从 [seq_len, batch_size, vocab_size] 到 [batch_size, seq_len, vocab_size]
        logits = logits.transpose(0, 1)
        logits = self.esm2mpnn(logits)
        seq_list = []
        for seq in sampled_seq:
            seq_list.append(''.join([self.model.decoder.dictionary.get_tok(a) for a in seq]))

        # Convert back to string via lookup
        return {
            'seq_list': seq_list,
            'logits': logits
        }
    
    def logits(self, X, S, mask, chain_M, residue_idx, chain_encoding_all):
        output = self.autoregressive_sample(X, S, mask, chain_M, residue_idx, chain_encoding_all)
        logits = output['logits']
        return logits
    
    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all):
        logits = self.logits(X, S, mask, chain_M, residue_idx, chain_encoding_all)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs
    
    def sample(self, X, S, mask, chain_M, residue_idx, chain_encoding_all, temperature=1.0):
        output = self.autoregressive_sample(X, S, mask, chain_M, residue_idx, chain_encoding_all, temperature)
        sequential_alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
        seq_index_list = []
        for seq in output['seq_list']:
            seq_index_list.append([sequential_alphabet.index(aa) for aa in seq])

        sampled_seq = torch.tensor(seq_index_list,device=self.device).long()
        return sampled_seq
    
    def score(self,X, S, S_test, mask, chain_M, residue_idx, chain_encoding_all):
        sequential_alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
        S_true_list = [''.join([sequential_alphabet[aa] for aa in seq]) for seq in S]
        S_test_list = [''.join([sequential_alphabet[aa] for aa in seq]) for seq in S_test]
        score_list = []
        
        for i, x in enumerate(X):
            _, score = score_sequence(self.model,self.alphabet,x,S_test_list[i])
            score_list.append(score)
        score_list = torch.tensor(score_list,device=self.device)
        return score_list
    
def get_sequence_loss(model, alphabet, coords, seq):
    device = next(model.parameters()).device
    batch_converter = CoordBatchConverter(alphabet)
    batch = [(coords, None, seq)]
    coords, confidence, strs, tokens, padding_mask = batch_converter(
        batch, device=device)

    prev_output_tokens = tokens[:, :-1].to(device)
    target = tokens[:, 1:]
    target_padding_mask = (target == alphabet.padding_idx)
    logits, _ = model.forward(coords, padding_mask, confidence, prev_output_tokens)
    loss = F.cross_entropy(logits, target, reduction='none')
    loss = loss[0].cpu().detach().numpy()
    target_padding_mask = target_padding_mask[0].cpu().numpy()
    return loss, target_padding_mask

def score_sequence(model, alphabet, coords, seq):
    loss, target_padding_mask = get_sequence_loss(model, alphabet, coords, seq)
    ll_fullseq = -np.sum(loss * ~target_padding_mask) / np.sum(~target_padding_mask)
    # Also calculate average when excluding masked portions
    coords_np = coords.cpu().detach().numpy()
    coord_mask = np.all(np.isfinite(coords_np), axis=(-1, -2))
    ll_withcoord = -np.sum(loss * coord_mask) / np.sum(coord_mask)
    return ll_fullseq, ll_withcoord