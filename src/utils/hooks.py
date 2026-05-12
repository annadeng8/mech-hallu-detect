import torch


class ActivationCache:
    def __init__(self):
        self.hidden_states = {}
        self.attentions = {}

    def clear(self):
        self.hidden_states.clear()
        self.attentions.clear()


def register_hidden_hook(model, layer_idx, cache, name=None):
    module = model.model.layers[layer_idx]
    key = name or f"layer_{layer_idx}"

    def hook(_module, _inputs, outputs):
        if isinstance(outputs, tuple):
            cache.hidden_states[key] = outputs[0].detach()
        else:
            cache.hidden_states[key] = outputs.detach()

    return module.register_forward_hook(hook)
