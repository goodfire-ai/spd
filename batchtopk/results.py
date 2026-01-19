# %%
import pandas as pd
import wandb

api = wandb.Api()

project = "batchtopk_comparison"
entity = "patrickaaleask"

runs = api.runs(f"{entity}/{project}")
data = []

for run in runs:
    config_sae_type = run.config.get("sae_type", None)
    dict_size = run.config.get("dict_size", None)
    k = run.config.get("top_k", None)
    final_l0_norm = run.summary.get("l0_norm", None)
    final_l2_loss = run.summary.get("l2_loss", None)
    final_ce_degradation = run.summary.get("performance/ce_degradation", None)
    data.append({
        "config_sae_type": config_sae_type,
        "dictionary_size": dict_size,
        "k": k,
        "l0_norm": final_l0_norm,
        "normalized_mse": final_l2_loss,
        "ce_degradation": final_ce_degradation
    })

df = pd.DataFrame(data)
print(df.head())

# %%
import matplotlib.pyplot as plt
import pandas as pd

fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Plot 1: Normalized MSE vs Dictionary size (k=32)
for sae_type in ['batchtopk', 'topk']:
    data = df[(df['config_sae_type'] == sae_type) & (df['k'] == 32.)]
    data = data.sort_values(by='dictionary_size')
    axs[0, 0].plot(data['dictionary_size'], data['normalized_mse'], 
                   marker='o', linestyle='--', label=f"{sae_type} (k=32)")

axs[0, 0].set_title('Normalized MSE vs Dictionary size (k=32)')
axs[0, 0].set_xlabel('Dictionary size')
axs[0, 0].set_ylabel('Normalized MSE')
axs[0, 0].set_xscale('log')
axs[0, 0].legend()
axs[0, 0].grid(True)

# Plot 2: Normalized MSE vs k (Dict size = 12288)
for sae_type in ['batchtopk', 'topk', 'jumprelu']:
    data = df[(df['config_sae_type'] == sae_type) & (df['dictionary_size'] == 12288)]
    data = data.sort_values(by='dictionary_size')
    axs[0, 1].plot(data['l0_norm'], data['normalized_mse'], 
                   marker='o', linestyle='--', label=sae_type)

axs[0, 1].set_title('Normalized MSE vs k (Dict size = 12288)')
axs[0, 1].set_xlabel('k')
axs[0, 1].set_ylabel('Normalized MSE')
axs[0, 1].legend()
axs[0, 1].grid(True)

# Plot 3: CE degradation vs Dictionary size (k=32)
for sae_type in ['batchtopk', 'topk']:
    data = df[(df['config_sae_type'] == sae_type) & (df['k'] == 32)]
    axs[1, 0].plot(data['dictionary_size'], data['ce_degradation'], 
                   marker='o', linestyle='--', label=f"{sae_type} (k=32)")

axs[1, 0].set_title('CE degradation vs Dictionary size (k=32)')
axs[1, 0].set_xlabel('Dictionary size')
axs[1, 0].set_ylabel('CE degradation')
axs[1, 0].set_xscale('log')
axs[1, 0].legend()
axs[1, 0].grid(True)

# Plot 4: CE degradation vs k (Dict size = 12288)
for sae_type in ['batchtopk', 'topk', 'jumprelu']:
    data = df[(df['config_sae_type'] == sae_type) & (df['dictionary_size'] == 12288)]
    axs[1, 1].plot(data['l0_norm'], data['ce_degradation'], 
                   marker='o', linestyle='--', label=sae_type)

axs[1, 1].set_title('CE degradation vs k (Dict size = 12288)')
axs[1, 1].set_xlabel('k')
axs[1, 1].set_ylabel('CE degradation')
axs[1, 1].legend()
axs[1, 1].grid(True)

# Adjust layout and display the plot
plt.tight_layout()
plt.show()