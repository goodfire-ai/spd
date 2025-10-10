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
    for k in sd:
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



import torch


def _gather_layer_modules_with_components(patched_sd_keys):
    """Return sorted list of module prefixes that have components.U/V."""
    mods = set()
    pat = re.compile(r"^(layers\.\d+\.(mlp_in|mlp_out))\.components\.(U|V)$")
    for k in patched_sd_keys:
        m = pat.match(k)
        if m:
            mods.add(m.group(1))  # e.g. 'layers.0.mlp_in'
    return sorted(mods, key=lambda s: (int(s.split('.')[1]), s.split('.')[-1]))

def _fetch_uvW_for_module(sd: dict[str, torch.Tensor], mod_prefix: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fetch U, V, and original weight W for a module prefix like 'layers.0.mlp_in'."""
    U = sd[f"{mod_prefix}.components.U"]
    V = sd[f"{mod_prefix}.components.V"]
    W = sd[f"{mod_prefix}.original.weight"]
    return U, V, W

def _matmul_candidates(U: torch.Tensor, V: torch.Tensor):
    """Yield (name, callable) candidates without forcing computation if shapes don't match."""
    # We define lazies so we can shape-check before matmul
    return [
        ("V@U",      lambda: V @ U,      V.shape, U.shape),
        ("U@V",      lambda: U @ V,      U.shape, V.shape),
        ("V@U.T",    lambda: V @ U.T,    V.shape, U.T.shape),
        ("U@V.T",    lambda: U @ V.T,    U.shape, V.T.shape),
        ("V.T@U",    lambda: V.T @ U,    V.T.shape, U.shape),
        ("U.T@V",    lambda: U.T @ V,    U.T.shape, V.shape),
        ("V.T@U.T",  lambda: V.T @ U.T,  V.T.shape, U.T.shape),
        ("U.T@V.T",  lambda: U.T @ V.T,  U.T.shape, V.T.shape),
    ]

def _choose_reconstruction(U: torch.Tensor, V: torch.Tensor, W: torch.Tensor) -> tuple[str, torch.Tensor, float]:
    """
    Pick the mathematically consistent formula that reproduces W's shape.
    If multiple match, choose the one with smallest Frobenius error.
    Returns (formula_name, W_hat, rel_error).
    """
    target_shape = W.shape
    eps = 1e-12
    best = None

    for name, thunk, lhs_shape, rhs_shape in _matmul_candidates(U, V):
        # shape-compatibility check for matmul
        if lhs_shape[-1] != rhs_shape[-2]:
            continue
        # output shape check
        out_shape = (lhs_shape[-2], rhs_shape[-1])
        if out_shape != target_shape:
            continue
        try:
            W_hat = thunk()
            num = torch.linalg.norm(W - W_hat).item()
            den = torch.linalg.norm(W).item() + eps
            rel = num / den
            if (best is None) or (rel < best[2]):
                best = (name, W_hat, rel)
        except Exception:
            # just skip invalid numerical combos
            continue

    if best is None:
        raise RuntimeError(
            f"No U/V multiplication matched target shape {tuple(target_shape)}. "
            f"U shape={tuple(U.shape)}, V shape={tuple(V.shape)}"
        )
    return best

def reconstruct_all_layers(model):
    patched = getattr(model, "patched_model", model)
    sd = patched.state_dict()
    mods = _gather_layer_modules_with_components(sd.keys())

    results = []
    print("[reconstruction] found componentized modules:")
    for m in mods:
        print(f"  - {m}")

    print("\n[reconstruction] per-module details:")
    for m in mods:
        U, V, W = _fetch_uvW_for_module(sd, m)
        name, W_hat, rel = _choose_reconstruction(U, V, W)

        def _ti(t): return f"{tuple(t.shape)}"
        print(f"  {m}:")
        print(f"    W shape    = {_ti(W)}")
        print(f"    U shape    = {_ti(U)}")
        print(f"    V shape    = {_ti(V)}")
        print(f"    formula    = {name}")
        print(f"    rel‖W-Ŵ‖F   = {rel:.10f}")

        results.append({
            "module": m,
            "W_shape": tuple(W.shape),
            "U_shape": tuple(U.shape),
            "V_shape": tuple(V.shape),
            "formula": name,
            "rel_error": rel,
        })

    # (Optional) Try embeddings too, if they have components (often they don't)
    for emb_name in ("W_E", "W_U"):
        U_key = f"{emb_name}.components.U"
        V_key = f"{emb_name}.components.V"
        W_key = emb_name
        if U_key in sd and V_key in sd and W_key in sd:
            U, V, W = sd[U_key], sd[V_key], sd[W_key]
            name, W_hat, rel = _choose_reconstruction(U, V, W)
            print(f"  {emb_name}:")
            print(f"    W shape    = {tuple(W.shape)}")
            print(f"    U shape    = {tuple(U.shape)}")
            print(f"    V shape    = {tuple(V.shape)}")
            print(f"    formula    = {name}")
            print(f"    rel‖W-Ŵ‖F   = {rel:.10f}")
            results.append({
                "module": emb_name,
                "W_shape": tuple(W.shape),
                "U_shape": tuple(U.shape),
                "V_shape": tuple(V.shape),
                "formula": name,
                "rel_error": rel,
            })

    # quick aggregate
    if results:
        mean_rel = sum(r["rel_error"] for r in results) / len(results)
        print(f"\n[reconstruction] mean relative error across {len(results)} modules: {mean_rel:.10f}")
    else:
        print("\n[reconstruction] no componentized modules found.")

    return results


# Minimal per-module SPD reconstruction & validation

def _mods_with_components(keys):
    pat = re.compile(r"^(layers\.\d+\.(mlp_in|mlp_out))\.components\.(U|V)$")
    mods = {pat.match(k).group(1) for k in keys if pat.match(k)}
    return sorted(mods, key=lambda s: (int(s.split('.')[1]), s.split('.')[-1]))

def _get(sd, key):
    # prefer original weight if present
    for k in (f"{key}.original.weight", f"{key}.weight"):
        if k in sd and sd[k].ndim == 2: return sd[k]
    raise KeyError(f"no weight for {key}")

def _rel_fro(A, B):
    return (torch.linalg.norm(A - B) / (torch.linalg.norm(B) + 1e-12)).item()

def _best_recon(U, V, W):
    # try sensible order first (paper form UV and its transpose-equivalent), then fallbacks
    cands = [
        ("U@V",      lambda: U @ V,      U.shape,      V.shape),
        ("U.T@V.T",  lambda: U.T @ V.T,  U.T.shape,    V.T.shape),
        ("U@V.T",    lambda: U @ V.T,    U.shape,      V.T.shape),
        ("U.T@V",    lambda: U.T @ V,    U.T.shape,    V.shape),
        ("V@U",      lambda: V @ U,      V.shape,      U.shape),
        ("V.T@U.T",  lambda: V.T @ U.T,  V.T.shape,    U.T.shape),
        ("V@U.T",    lambda: V @ U.T,    V.shape,      U.T.shape),
        ("V.T@U",    lambda: V.T @ U,    V.T.shape,    U.shape),
    ]
    target = W.shape
    best = None
    for name, thunk, L, R in cands:
        if L[-1] != R[-2]: continue                 # inner dims match
        if (L[-2], R[-1]) != target: continue      # output shape match
        try:
            What = thunk()
            rel  = _rel_fro(What, W)
            if (best is None) or (rel < best[2]): best = (name, rel)
        except Exception:
            pass
    if best is None:
        raise RuntimeError(f"No matching U/V product for target {tuple(target)}; "
                           f"U={tuple(U.shape)}, V={tuple(V.shape)}")
    return best

def validate_spd_reconstruction(model):
    patched = getattr(model, "patched_model", model)
    sd = patched.state_dict()
    mods = _mods_with_components(sd.keys())
    print("Module                 W         U         V         formula        rel‖W-Ŵ‖F")
    print("--------------------------------------------------------------------------------")
    rels = []
    for m in mods:
        U = sd[f"{m}.components.U"]
        V = sd[f"{m}.components.V"]
        W = _get(sd, f"{m}.original") if f"{m}.original.weight" in sd else _get(sd, m)
        name, rel = _best_recon(U, V, W)
        print(f"{m:<20} {tuple(W.shape)!s:<10} {tuple(U.shape)!s:<10} {tuple(V.shape)!s:<10} {name:<14} {rel:10.6f}")
        rels.append(rel)
    if rels:
        print(f"\nMean relative error across {len(rels)} modules: {sum(rels)/len(rels):.6f}")