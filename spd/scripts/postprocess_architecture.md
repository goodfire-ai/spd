# Postprocess Architecture

## Call Graph

```mermaid
graph TD
    subgraph "CLI Entry Points (defaults live here)"
        PP_CLI["spd-postprocess<br/><code>postprocess_cli.py</code>"]
        H_CLI["spd-harvest<br/><code>harvest/.../run_slurm_cli.py</code>"]
        A_CLI["spd-attributions<br/><code>dataset_attributions/.../run_slurm_cli.py</code>"]
        AI_CLI["spd-autointerp<br/><code>autointerp/.../run_slurm_cli.py</code>"]
    end

    subgraph "Config (defaults live here too)"
        PC["PostprocessConfig"]
        HSC["HarvestSlurmConfig"]
        ASC["AttributionsSlurmConfig"]
        AISC["AutointerpSlurmConfig"]
        HC["HarvestConfig"]
        DAC["DatasetAttributionConfig"]
        CSC["CompactSkepticalConfig"]
        AEC["AutointerpEvalConfig"]

        PC --> HSC & ASC & AISC
        HSC --> HC
        ASC --> DAC
        AISC --> CSC & AEC
    end

    subgraph "Orchestrator"
        PP["postprocess()"]
    end

    subgraph "SLURM Launchers (no defaults)"
        H_SLURM["harvest()"]
        A_SLURM["submit_attributions()"]
        AI_SLURM["launch_autointerp_pipeline()"]
    end

    subgraph "SLURM Workers"
        H_RUN["harvest/scripts/run.py"]
        A_RUN["dataset_attributions/scripts/run.py"]
        AI_RUN["autointerp/scripts/run_interpret.py"]
    end

    subgraph "Core Logic"
        H_CORE["harvest_activation_contexts()"]
        H_MERGE["merge_activation_contexts()"]
        A_CORE["harvest_attributions()"]
        A_MERGE["merge_attributions()"]
        AI_CORE["run_interpret()"]
    end

    PP_CLI -- "PostprocessConfig" --> PP
    PP --> H_SLURM & A_SLURM & AI_SLURM

    H_CLI -- "constructs HarvestConfig" --> H_SLURM
    A_CLI -- "constructs DatasetAttributionConfig" --> A_SLURM
    AI_CLI -- "constructs CompactSkepticalConfig" --> AI_SLURM

    H_SLURM -- "serializes config_json" --> H_RUN
    A_SLURM -- "serializes config_json" --> A_RUN
    AI_SLURM -- "serializes config_json" --> AI_RUN

    H_RUN -- "deserializes HarvestConfig" --> H_CORE & H_MERGE
    A_RUN -- "deserializes DatasetAttributionConfig" --> A_CORE & A_MERGE
    AI_RUN -- "deserializes CompactSkepticalConfig" --> AI_CORE
```

## SLURM Dependency Graph

```mermaid
graph LR
    subgraph "Harvest Unit"
        HW["harvest workers<br/>(GPU array)"]
        HM["harvest merge"]
        IE["intruder eval"]
        HW --> HM --> IE
    end

    subgraph "Autointerp Unit"
        INT["interpret<br/>(CPU, LLM)"]
        DET["detection<br/>(CPU)"]
        FUZ["fuzzing<br/>(CPU)"]
        INT --> DET & FUZ
    end

    subgraph "Attributions Unit"
        AW["attribution workers<br/>(GPU array)"]
        AM["attribution merge"]
        AW --> AM
    end

    HM --> INT
```

## Config Ownership

```mermaid
graph LR
    subgraph "Tuning Params (owned by config classes)"
        HC2["HarvestConfig<br/>n_batches, batch_size,<br/>ci_threshold, ..."]
        DAC2["DatasetAttributionConfig<br/>n_batches, batch_size,<br/>ci_threshold"]
        CSC2["CompactSkepticalConfig<br/>model, reasoning_effort,<br/>max_examples, ..."]
        AEC2["AutointerpEvalConfig<br/>eval_model, partition, time"]
    end

    subgraph "SLURM Params (owned by SlurmConfigs / CLIs)"
        S1["n_gpus, partition, time"]
    end

    subgraph "Runtime Args (not in any config)"
        R1["wandb_path"]
        R2["rank, world_size, merge"]
        R3["limit, cost_limit_usd"]
        R4["snapshot_branch, job_suffix"]
    end
```
