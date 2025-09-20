# Helper functions for clustering operations

import re
from collections import OrderedDict

def locate_original_weights(model):
    """
    Return a dict with:
      - 'W_E' and 'W_U' (if present)
      - 'layers': OrderedDict[int -> {'mlp_in': Tensor, 'mlp_out': Tensor}]
    pulled from model.patched_model.state_dict().
    """
    # The SPD ComponentModel carries the actual decomposed network under `patched_model`
    patched = getattr(model, "patched_model", model)
    sd = patched.state_dict()

    out = {}
    # Embedding matrices (if present in this model family)
    if "W_E" in sd:
        out["W_E"] = sd["W_E"]
    if "W_U" in sd:
        out["W_U"] = sd["W_U"]

    # Collect per-layer original weights
    layer_pat = re.compile(r"^layers\.(\d+)\.(mlp_in|mlp_out)\.original\.weight$")
    layers = OrderedDict()
    for k in sd.keys():
        m = layer_pat.match(k)
        if m:
            layer_idx = int(m.group(1))
            sub = m.group(2)  # 'mlp_in' or 'mlp_out'
            if layer_idx not in layers:
                layers[layer_idx] = {}
            layers[layer_idx][sub] = sd[k]

    out["layers"] = layers
    return out

def _tinfo(t):
    return f"shape={tuple(t.shape)}, dtype={t.dtype}"

def summarize_original_weights(weights_dict):
    print("[summary] original weights found:")
    # Embeddings
    for name in ("W_E", "W_U"):
        if name in weights_dict:
            print(f"  {name}: {_tinfo(weights_dict[name])}")
    # Layers
    n_entries = 0
    for layer_idx in sorted(weights_dict.get("layers", {}).keys()):
        for part in ("mlp_in", "mlp_out"):
            t = weights_dict["layers"][layer_idx].get(part)
            if t is not None:
                print(f"  layers.{layer_idx}.{part}.original.weight: {_tinfo(t)}")
                n_entries += 1
    print(f"[summary] total layer matrices: {n_entries}")