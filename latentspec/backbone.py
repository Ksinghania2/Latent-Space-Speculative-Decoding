import torch


class Backbone:
    """Wrapper over a HF causal LM (Llama/Qwen/GPT-2 style) exposing logits, the intermediate
    hidden state (draft-head input) and the final post-norm hidden state."""

    def __init__(self, name="Qwen/Qwen2.5-0.5B", device=None, dtype=torch.float16, mid_layer=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cpu" and dtype == torch.float16:
            dtype = torch.float32
        self.dtype = dtype
        self.tok = AutoTokenizer.from_pretrained(name)
        try:
            m = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
        except TypeError:
            m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)
        self.model = m.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.n_layers, self.d = self.model.config.num_hidden_layers, self.model.config.hidden_size
        self.mid = mid_layer if mid_layer is not None else (2 * self.n_layers) // 3
        self.lm_head = self.model.get_output_embeddings()
        self.embed = self.model.get_input_embeddings()

    @torch.no_grad()
    def forward(self, ids, cache=None):
        out = self.model(input_ids=ids, past_key_values=cache, use_cache=True, output_hidden_states=True)
        hs = out.hidden_states  # hs[0]=embeddings ... hs[-1]=post-final-norm
        if hs is None:
            raise RuntimeError("backbone did not return hidden states")
        return out.logits, hs[self.mid], hs[-1], out.past_key_values

    @staticmethod
    def crop(cache, n):
        # Roll back KV cache to keep the first `n` tokens. Supports HF Cache objects
        # (transformers >=4.36, where crop removes tokens) and legacy tuple caches.
        if cache is None:
            return
        if hasattr(cache, "get_seq_length") and hasattr(cache, "crop"):
            cur = cache.get_seq_length()
            if n >= cur:
                return
            cache.crop(n - cur)  # negative => remove (cur - n) tokens
            return
        if hasattr(cache, "crop"):
            # Older transformers: crop(max_length) keeps first max_length.
            try:
                cache.crop(n)
                return
            except Exception:
                pass
        if hasattr(cache, "layers"):
            for layer in cache.layers:
                if hasattr(layer, "keys"):
                    layer.keys = layer.keys[..., :n, :].contiguous() if layer.keys is not None else None
                    layer.values = layer.values[..., :n, :].contiguous() if layer.values is not None else None
            return
        if hasattr(cache, "key_cache"):
            for i, (kc, vc) in enumerate(zip(cache.key_cache, cache.value_cache)):
                cache.key_cache[i] = kc[..., :n, :].contiguous() if kc is not None else None
                cache.value_cache[i] = vc[..., :n, :].contiguous() if vc is not None else None
            return
        if isinstance(cache, (tuple, list)):
            for i in range(len(cache)):
                kv = cache[i]
                if isinstance(kv, (tuple, list)) and len(kv) == 2:
                    k, v = kv
                    cache[i] = (k[..., :n, :].contiguous(), v[..., :n, :].contiguous())
