#103503
spd-run --experiments ss_llama_sans_base --dp 4

#103504
spd-run --experiments ss_llama_subset_sans --dp 4

#103514
spd-run --experiments ss_llama_sans_bal --dp 4

#103506
spd-run --experiments ss_llama_subset_bal_sans --dp 4

#103507
spd-run --experiments ss_llama_subset_bal_simple_sans --dp 4

#103508
spd-run --experiments ss_llama_subset_bal_sans_denser --dp 4

#103509
spd-run --experiments ss_llama_subset_bal_sans_sparser --dp 4




############

#103625
spd-run --experiments ss_llama_sans_base,ss_llama_subset_bal_sans,ss_llama_subset_sans  --sweep min_sweep_params.yaml  --dp 4 --n-agents 10

#103656
spd-run --experiments ss_llama_subset_sans_bal_simple,ss_llama_sans_bal  --sweep min_sweep_params.yaml  --dp 2 --n-agents 4

