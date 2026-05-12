import torch

class TreeResampler:
    def __init__(self, tokenizer, k=5):
        self.tokenizer = tokenizer
        self.k = k

    def select(self, logits, scorer):

        # 1. Focus only on the top-k most likely tokens to maintain grammatical sanity
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_ids = torch.topk(probs, self.k)

        best_score = float('-inf')
        best_id = top_ids[0]

        for i, token_id in enumerate(top_ids):
            token_text = self.tokenizer.decode([token_id]).strip().lower()
            if token_text in ["the", "a", "is", "of"]:
                continue # Skip resampling for boring tokens
            
            # 2. Get the model's original confidence (log-prob)
            model_conf = torch.log(top_probs[i] + 1e-10).item()
            
            # 3. Get the mechanistic score (TSV/Lookback/SAE)
            # We assume scorer() returns a value where higher is better
            mech_score = scorer(int(token_id))
            
            # 4. Combined Score (Balance model fluency with truthfulness)
            # We use a weight here (e.g., 0.7 for truth, 0.3 for fluency)
            total_score = model_conf + mech_score
            
            if total_score > best_score:
                best_score = total_score
                best_id = token_id

        return best_id