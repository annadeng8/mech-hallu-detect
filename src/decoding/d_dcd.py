import time
import torch
import torch.nn.functional as F
from train_lookback_real import build_features

class DDCDDecoder:
    def __init__(self, model, tokenizer, tsv, sae, lookback, dola, resampler, config):
        self.model = model
        self.tokenizer = tokenizer
        self.tsv = tsv
        self.sae = sae
        self.lookback = lookback
        self.dola = dola
        self.resampler = resampler
        self.config = config
        self.logs = []

    def compute_risk(self, lookback_score, tsv_conflict, sae_score):
        w = self.config["weights"]
        # Standardizing risk: High value = High chance of hallucination
        return (
            w["w1"] * (1 - lookback_score) + 
            w["w2"] * tsv_conflict + 
            w["w3"] * sae_score
        )

    def select_tier(self, risk):
        tau1 = self.config["thresholds"]["tau1"]
        tau2 = self.config["thresholds"]["tau2"]
        if risk <= tau1:
            return 1
        elif risk <= tau2:
            return 2
        return 3

    @torch.no_grad()
    def generate(self, prompt):
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        generated = inputs["input_ids"]
        prompt_len = generated.shape[-1]
        
        # Initialize KV Cache for speed on Mac
        past_key_values = None

        for step in range(self.config["max_new_tokens"]):
            start = time.time()
            
            # Use cache for efficiency; only pass the last token if we have a cache
            model_inputs = generated if past_key_values is None else generated[:, -1:]
            
            outputs = self.model(
                input_ids=model_inputs,
                past_key_values=past_key_values,
                output_hidden_states=True,
                output_attentions=True,
                use_cache=True,
            )
            
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
            k = self.config.get("layer_avg_k", 4)
            pooling = self.config.get("pooling", "answer_layeravg")

            if pooling == "answer_layeravg":
                # Get the last k layers for the current token (index -1)
                # Each is [batch, 1, hidden_size] -> [hidden_size]
                layers = [outputs.hidden_states[i][:, -1, :] for i in range(-k, 0)]
                
                # Concatenate them exactly like the training code: last k layers joined together
                # Shape becomes [1, hidden_size * k]
                tsv_input_hidden = torch.cat(layers, dim=-1)
            else:
                # Fallback to standard hidden state
                tsv_input_hidden = outputs.hidden_states[-1][:, -1, :]

            # 1. Get the standard hidden state for the SAE
            # Shape: [1, 4096]
            standard_hidden = outputs.hidden_states[-1][:, -1, :]

            # 2. Get the concatenated hidden state for the TSV
            k = self.config.get("layer_avg_k", 4)
            layers = [outputs.hidden_states[i][:, -1, :] for i in range(-k, 0)]
            tsv_input_hidden = torch.cat(layers, dim=-1) # Shape: [1, 16384]

            # 3. USE DIFFERENT INPUTS FOR DIFFERENT MODULES
            # TSV gets the 16k vector
            tsv_conflict = float(self.tsv.conflict_score(tsv_input_hidden).item())

            # SAE gets the standard 4k vector
            _, features = self.sae(standard_hidden) # <--- Fix this line
            sae_score = float(features.max().item())
            # Normalize SAE score to [0, 1] based on expected activation ranges
            sae_score = min(sae_score / 10.0, 1.0)

            # 3. Lookback Attention Score
            lookback_score = 1.0
            if self.lookback is not None:
                # We need the full attention matrix for the lookback feature
                # Note: With KV Cache, outputs.attentions only contains attention for the NEW token
                # which is actually exactly what build_features expects!
                # 1. Stack all layer attentions: [32, batch, heads, seq, seq]
                # 2. Mean across heads: [32, batch, seq, seq]
                # 3. Select current batch (0): [32, seq, seq]
                attn_stacked = torch.stack(outputs.attentions).mean(dim=2).squeeze(1).cpu().numpy()

                # 4. Extract features
                features_lb = build_features(attn_stacked, prompt_len)

                # 5. features_lb is now [32], reshape to [1, 32] for sklearn
                lookback_score = self.lookback.predict_proba(features_lb.reshape(1, -1))[0, 1]

            # 4. Triage Logic
            risk = self.compute_risk(lookback_score, tsv_conflict, sae_score)
            tier = self.select_tier(risk)

            if tier == 1:
                next_token = logits.argmax(dim=-1)
                reason = "Grounded (Greedy)"
            elif tier == 2:
                # Trigger the custom DoLa contrastive logic
                contrasted_logits, layer_idx = self.dola.contrastive_step(
                    outputs.hidden_states, 
                    logits, 
                    alpha=self.config.get("dola_alpha", 0.1)
                )
                next_token = contrasted_logits.argmax(dim=-1)
                reason = f"DoLa (Contrast Layer {layer_idx})"
            else:
                # Trigger Tree-based Resampling
                next_token = self.resampler.select(
                    logits[0], 
                    # Scorer: higher is better. We penalize tokens using the risk
                    scorer=lambda token_id: float(logits[0, token_id].item()) - (risk * 5)
                )
                reason = "Resampling (High Risk)"

            # Update generation
            generated = torch.cat([generated, next_token.view(1, 1)], dim=-1)
            token_text = self.tokenizer.decode(next_token)

            self.logs.append({
                "step": step,
                "token": token_text,
                "tier": tier,
                "risk": round(risk, 4),
                "tsv": round(tsv_conflict, 4),
                "lb": round(lookback_score, 4),
                "reason": reason,
                "latency_ms": round((time.time() - start) * 1000, 2),
            })

            if next_token.item() == self.tokenizer.eos_token_id:
                break
            
            # Clean up MPS memory periodically
            if step % 5 == 0:
                torch.mps.empty_cache()

        input_len = inputs.input_ids.shape[1]
        answer_only = self.tokenizer.decode(generated[0][input_len:], skip_special_tokens=True).strip()

        return answer_only