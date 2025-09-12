#89831
spd-run --experiments ss_mlp_no_layerwise1 --dp 8

#89832
spd-run --experiments ss_mlp_no_layerwise2 --dp 4

spd-run --experiments ss_mlp_layerwise_act --sweep layerwise_l2_sweep.yaml --n_agents 12

spd-run --experiments ss_mlp --sweep ss_per_layer_decomp.yaml --n_agents 12


spd-run --experiments ss_mlp --sweep ss_per_layer_decomp.yaml --n_agents 12

#run_20250902_085927, 99734_*
spd-run --experiments ss_mlp_no_layerwise1 --sweep ss_recon_sweep.yaml --n_agents 3


# 100562, per layer for ensembeling with more
spd-run --experiments ss_mlp --sweep ss_per_layer_decomp_2.yaml --n_agents 12

# 100585, maybe we jsut need non-stoch recon loss
spd-run --experiments ss_mlp --sweep ss_recon_sweep.yaml --n_agents 3

# 100590, maybe we jsut need non-stoch recon loss
spd-run --experiments ss_mlp --sweep ss_stoch_recon_sweep.yaml --n_agents 3

spd-run --experiments ss_llama_all --dp 8 --no-create-snapshot


#control to compare runtime to from computing the subset loss 102582
spd-run --experiments ss_llama_single --no-create-snapshot

spd-run --experiments ss_llama_subset_recon --sweep subset_recon_sweep.yaml --n-agents 16

#102585_2 
spd-run --experiments ss_llama_l0_balance,ss_llama_subset_recon_harmonic --no-create-snapshot

#102587_1
spd-run --experiments ss_llama_l0_balance --sweep l0_bal_sweep.yaml --n-agents 8

spd-run --experiments ss_llama_subset_recon --sweep l0_bal_sweep.yaml --n-agents 5

spd-run --experiments ss_llama_subset_recon_0p25 --dp 4
