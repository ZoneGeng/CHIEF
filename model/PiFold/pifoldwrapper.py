import torch
from .methods.prodesign_model import ProDesign_Model

class PiFoldConfig():
    def __init__(
        self,
        node_features = 128,
        edge_features = 128,
        hidden_dim = 128,
        dropout = 0.1,
        num_encoder_layers = 10,
        k_neighbors = 30,
        num_rbf = 16,
        num_positional_embeddings = 16,
        node_dist = 1,
        node_direct = 1,
        node_angle = 1,
        edge_angle = 1,
        edge_dist = 1,
        edge_direct = 1,
        virtual_num = 3
    ):
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_encoder_layers = num_encoder_layers
        self.k_neighbors = k_neighbors
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings,
        self.node_dist = node_dist,
        self.node_angle = node_angle,
        self.node_direct = node_direct,
        self.edge_dist = edge_dist,
        self.edge_angle = edge_angle,
        self.edge_direct = edge_direct,
        self.virtual_num = virtual_num
        
class PiFoldWrapper():
    def __init__(self,args=PiFoldConfig(),state_dict = './checkpoints/pifold/pifold_weights.pth',device=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.model = ProDesign_Model(args)
        self.model = self.model.to(self.device)
        self.model.load_state_dict(torch.load(state_dict))
        
    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all):
        with torch.no_grad():
            X = X.to(self.device)
            S = S.to(self.device)
            mask = mask.to(self.device)
            B,L = X.shape[:2]
            score = None
            X, S, score, h_V, h_E, E_idx, batch_id, mask_bw, mask_fw, decoding_order = self.model._get_features(S, score, X=X, mask=mask)
            log_probs = self.model(h_V, h_E, E_idx, batch_id) # native log_probs
            # log_probs = torch.cat((log_probs,torch.full((log_probs.shape[0],1), float('-inf')).to(self.device)),dim=-1) # add -inf for X residue
            probs = torch.exp(log_probs)
            probs = torch.cat((probs,torch.full((probs.shape[0],1), 1e-10).to(self.device)),dim=-1) # add 0 for X residue
            log_probs = torch.log(probs)
            log_probs_full = torch.zeros(B*L,21).to(self.device)
            log_probs_full[mask.reshape(B*L).bool(),:] = log_probs
            log_probs_full = log_probs_full.reshape(B,L,21)
            return log_probs_full
        
    def logits(self, X, S, mask, chain_M, residue_idx, chain_encoding_all):
        with torch.no_grad():
            B,L = X.shape[:2]
            score = None
            X, S, score, h_V, h_E, E_idx, batch_id, mask_bw, mask_fw, decoding_order = self.model._get_features(S, score, X=X, mask=mask)
            _, logits = self.model(h_V, h_E, E_idx, batch_id,return_logit=True)
            return logits
        
    def sample(self, X, S, mask, chain_M, residue_idx, chain_encoding_all, temperature=1.0):
        with torch.no_grad():
            B,L = X.shape[:2]
            score = None
            logits = self.logits(X, S, mask, chain_M, residue_idx, chain_encoding_all)
            logits = logits / temperature
            probs = torch.nn.functional.softmax(logits,dim=-1)
            probs[torch.isnan(probs)] = 1e-6
            probs_flat = probs.view(-1,probs.shape[-1])

            sampled_seq = torch.multinomial(probs_flat[:,:20],num_samples=1,replacement=True).squeeze(-1)
            sampled_seq_full = torch.full((B * L,), 20, dtype=torch.long, device=sampled_seq.device)
            sampled_seq_full[mask.view(-1).bool()] = sampled_seq
            sampled_seq = sampled_seq_full.view(B,L)
        return sampled_seq
    
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