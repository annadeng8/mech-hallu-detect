import torch
import torch.nn.functional as F

class DoLaDecoder:
    def __init__(self, model, premature_layers=None):
        self.model = model
        # Default to a subset of early/mid layers for efficiency
        self.premature_layers = premature_layers or [2, 4, 8, 12, 16]

    def contrastive_step(self, hidden_states, final_logits, alpha=0.1):
        """
        Performs the full DoLa contrastive step for a single token.
        """
        # 1. Get logits only for the candidate premature layers
        candidate_logits = []
        for idx in self.premature_layers:
            h = hidden_states[idx][:, -1, :]
            candidate_logits.append(self.model.lm_head(h))
        
        # 2. Find the layer with maximum divergence from final_logits
        # We use Logit-space subtraction but JS-Divergence for selection
        best_j = self.select_premature(candidate_logits, final_logits)
        early_logits = candidate_logits[best_j]
        
        # 3. Contrast: L_final - alpha * L_early
        # Using a small alpha (0.1) prevents the distribution from collapsing
        contrasted_logits = final_logits - alpha * early_logits
        
        return contrasted_logits, self.premature_layers[best_j]

    def select_premature(self, candidate_logits, final_logits):
        best_j = 0
        max_div = -1
        
        p = F.softmax(final_logits, dim=-1)
        
        for i, q_logits in enumerate(candidate_logits):
            q = F.softmax(q_logits, dim=-1)
            # Simplified JS-like divergence or Max-Kullback
            # Often simple Jensen-Shannon helps find the layer that "knows the least"
            m = 0.5 * (p + q)
            div = 0.5 * (F.kl_div(p.log(), m, reduction='batchmean') + 
                         F.kl_div(q.log(), m, reduction='batchmean'))
            
            if div > max_div:
                max_div = div
                best_j = i
        return best_j