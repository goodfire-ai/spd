#%%
from activation_store import ActivationsStore
from config import get_default_cfg, post_init_cfg
from sae import BatchTopKSAE, JumpReLUSAE, TopKSAE, VanillaSAE
from training import train_sae
from transformer_lens import HookedTransformer

for l1_coeff in [0.004, 0.0018, 0.0008]:
    cfg = get_default_cfg()
    cfg["sae_type"] = "jumprelu" # "vanilla", "topk", "batchtopk"
    cfg["model_name"] = "gpt2-small"
    cfg["layer"] = 8
    cfg["site"] = "resid_pre"
    cfg["dataset_path"] = "Skylion007/openwebtext"
    cfg["aux_penalty"] = (1/32)
    cfg["lr"] = 3e-4
    cfg["input_unit_norm"] = True
    cfg["top_k"] = 32
    cfg["dict_size"] = 768 * 16
    cfg['wandb_project'] = 'batchtopk_comparison'
    cfg['act_size'] = 768
    cfg['device'] = 'cuda'
    cfg['bandwidth'] = 0.001
    cfg['l1_coeff'] = l1_coeff

    if cfg["sae_type"] == "vanilla":
        sae = VanillaSAE(cfg)
    elif cfg["sae_type"] == "topk":
        sae = TopKSAE(cfg)
    elif cfg["sae_type"] == "batchtopk":
        sae = BatchTopKSAE(cfg)
    elif cfg["sae_type"] == 'jumprelu':
        sae = JumpReLUSAE(cfg)

    cfg = post_init_cfg(cfg)
                
    model = HookedTransformer.from_pretrained(cfg["model_name"]).to(cfg["dtype"]).to(cfg["device"])
    activations_store = ActivationsStore(model, cfg)
    train_sae(sae, activations_store, model, cfg)

for sae_type in ['topk', 'batchtopk']:
    for top_k in [16, 32, 64]:
        cfg = get_default_cfg()
        cfg["sae_type"] = sae_type
        cfg["model_name"] = "gpt2-small"
        cfg["layer"] = 8
        cfg["site"] = "resid_pre"
        cfg["dataset_path"] = "Skylion007/openwebtext"
        cfg["aux_penalty"] = (1/32)
        cfg["lr"] = 3e-4
        cfg["input_unit_norm"] = True
        cfg["top_k"] = 32
        cfg["dict_size"] = 768 * 16
        cfg['wandb_project'] = 'batchtopk_comparison'
        cfg['l1_coeff'] = 0.
        cfg['act_size'] = 768
        cfg['device'] = 'cuda'
        cfg['bandwidth'] = 0.001
        cfg['top_k'] = top_k

        if cfg["sae_type"] == "vanilla":
            sae = VanillaSAE(cfg)
        elif cfg["sae_type"] == "topk":
            sae = TopKSAE(cfg)
        elif cfg["sae_type"] == "batchtopk":
            sae = BatchTopKSAE(cfg)
        elif cfg["sae_type"] == 'jumprelu':
            sae = JumpReLUSAE(cfg)

        cfg = post_init_cfg(cfg)
                    
        model = HookedTransformer.from_pretrained(cfg["model_name"]).to(cfg["dtype"]).to(cfg["device"])
        activations_store = ActivationsStore(model, cfg)
        train_sae(sae, activations_store, model, cfg)


for sae_type in ['topk', 'batchtopk']:
    # don't retrain *16
    for dict_size in [768*4, 768*8, 768*32]:
        cfg = get_default_cfg()
        cfg["sae_type"] = sae_type
        cfg["model_name"] = "gpt2-small"
        cfg["layer"] = 8
        cfg["site"] = "resid_pre"
        cfg["dataset_path"] = "Skylion007/openwebtext"
        cfg["aux_penalty"] = (1/32)
        cfg["lr"] = 3e-4
        cfg["input_unit_norm"] = True
        cfg["top_k"] = 32
        cfg["dict_size"] = dict_size
        cfg['wandb_project'] = 'batchtopk_comparison'
        cfg['l1_coeff'] = 0.
        cfg['act_size'] = 768
        cfg['device'] = 'cuda'
        cfg['bandwidth'] = 0.001
        cfg['top_k'] = 32

        if cfg["sae_type"] == "vanilla":
            sae = VanillaSAE(cfg)
        elif cfg["sae_type"] == "topk":
            sae = TopKSAE(cfg)
        elif cfg["sae_type"] == "batchtopk":
            sae = BatchTopKSAE(cfg)
        elif cfg["sae_type"] == 'jumprelu':
            sae = JumpReLUSAE(cfg)

        cfg = post_init_cfg(cfg)
                    
        model = HookedTransformer.from_pretrained(cfg["model_name"]).to(cfg["dtype"]).to(cfg["device"])
        activations_store = ActivationsStore(model, cfg)
        train_sae(sae, activations_store, model, cfg)