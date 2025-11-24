#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=4

#SBATCH --partition=h200-reserved
#SBATCH --time=72:00:00
#SBATCH --job-name=spd-5h30m
#SBATCH --output=/mnt/polished-lake/home/oli/slurm_logs/slurm-%A_%a.out
#SBATCH --array=1-1

echo "Activating virtual environment"
source .venv/bin/activate

echo "Running..."
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

srun torchrun \
    --nnodes=2 \
    --nproc_per_node=8 \
    --rdzv_id=run_20251124_131448_0 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1):34844 \
    spd/experiments/lm/lm_decomposition.py \
    --config_json 'json:{"wandb_project": "spd", "wandb_run_name": null, "wandb_run_name_prefix": "", "seed": 0, "C": 1200, "n_mask_samples": 1, "ci_fn_type": "shared_mlp", "ci_fn_hidden_dims": [1000], "sampling": "continuous", "sigmoid_type": "leaky_hard", "target_module_patterns": ["h.*.mlp.c_fc", "h.*.mlp.down_proj", "h.*.attn.q_proj", "h.*.attn.k_proj", "h.*.attn.v_proj", "h.*.attn.o_proj"], "identity_module_patterns": null, "use_delta_component": true, "loss_metric_configs": [{"coeff": 0.004, "classname": "ImportanceMinimalityLoss", "pnorm": 2.0, "p_anneal_start_frac": 0.0, "p_anneal_final_p": 0.3, "p_anneal_end_frac": 1.0, "eps": 1e-12}, {"coeff": 1.0, "classname": "StochasticReconSubsetLoss"}], "output_loss_type": "kl", "lr": 0.0005, "steps": 20000, "batch_size": 256, "gradient_accumulation_steps": 1, "faithfulness_warmup_steps": 200, "faithfulness_warmup_lr": 0.01, "faithfulness_warmup_weight_decay": 0.1, "lr_schedule": "cosine", "lr_exponential_halflife": null, "lr_warmup_pct": 0.0, "out_dir": null, "train_log_freq": 200, "eval_freq": 1000, "eval_batch_size": 64, "slow_eval_freq": 2000, "n_eval_steps": 5, "slow_eval_on_first_step": true, "save_freq": null, "eval_metric_configs": [{"classname": "CIHistograms", "n_batches_accum": 5}, {"classname": "ComponentActivationDensity"}, {"classname": "CI_L0", "groups": {"total": ["*"], "layer_0": ["h.0.*"], "layer_1": ["h.1.*"], "layer_2": ["h.2.*"], "layer_3": ["h.3.*"]}}, {"classname": "CEandKLLosses", "rounding_threshold": 0.0}, {"classname": "CIMeanPerComponent"}, {"classname": "StochasticReconSubsetCEAndKL", "include_patterns": {"layer_0_only": ["h.0.*"], "layer_1_only": ["h.1.*"], "layer_2_only": ["h.2.*"], "layer_3_only": ["h.3.*"], "mlp_only": ["*.mlp.*"], "attention_only": ["*.attn.*"]}, "exclude_patterns": {"all_but_layer_0": ["h.0.*"], "all_but_layer_1": ["h.1.*"], "all_but_layer_2": ["h.2.*"], "all_but_layer_3": ["h.3.*"]}}, {"coeff": null, "classname": "StochasticHiddenActsReconLoss"}], "ci_alive_threshold": 0.0, "n_examples_until_dead": 1368400, "pretrained_model_class": "simple_stories_train.models.gpt2_simple.GPT2Simple", "pretrained_model_path": null, "pretrained_model_name": "wandb:goodfire/spd/runs/rvu66183", "pretrained_model_output_attr": "idx_0", "tokenizer_name": "SimpleStories/test-SimpleStories-gpt2-1.25M", "task_config": {"task_name": "lm", "max_seq_len": 512, "buffer_size": 1000, "dataset_name": "SimpleStories/SimpleStories", "column_name": "story", "train_data_split": "train", "eval_data_split": "test", "shuffle_each_epoch": true, "is_tokenized": false, "streaming": false}}' \
    --sweep_id run_20251124_131448 \
    --evals_id ss_gpt2_simple \
