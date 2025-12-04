<script lang="ts">
    import type { LoadingState } from "./types";

    type Props = {
        state: LoadingState;
    };

    let { state }: Props = $props();
</script>

<div class="loading-overlay">
    <div class="stages">
        {#each state.stages as stage, i}
            {@const isCurrent = i === state.currentStage}
            {@const isComplete = i < state.currentStage}
            <div class="stage" class:current={isCurrent} class:complete={isComplete}>
                <div class="stage-header">
                    <span class="stage-number">{i + 1}</span>
                    <span class="stage-name">{stage.name}</span>
                    {#if isComplete}
                        <span class="stage-check">✓</span>
                    {/if}
                </div>
                {#if isCurrent}
                    <div class="progress-bar">
                        {#if stage.progress !== null}
                            <div class="progress-fill" style="width: {stage.progress * 100}%"></div>
                        {:else}
                            <div class="progress-fill indeterminate"></div>
                        {/if}
                    </div>
                {:else if isComplete}
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: 100%"></div>
                    </div>
                {:else}
                    <div class="progress-bar empty"></div>
                {/if}
            </div>
        {/each}
    </div>
</div>

<style>
    .loading-overlay {
        position: absolute;
        inset: 0;
        background: rgba(255, 255, 255, 0.95);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 100;
    }

    .stages {
        display: flex;
        flex-direction: column;
        gap: 1rem;
        width: 280px;
    }

    .stage {
        opacity: 0.4;
    }

    .stage.current,
    .stage.complete {
        opacity: 1;
    }

    .stage-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.35rem;
    }

    .stage-number {
        width: 20px;
        height: 20px;
        border-radius: 50%;
        background: #e0e0e0;
        color: #757575;
        font-size: 0.7rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .stage.current .stage-number {
        background: #2196f3;
        color: white;
    }

    .stage.complete .stage-number {
        background: #4caf50;
        color: white;
    }

    .stage-name {
        font-size: 0.85rem;
        color: #616161;
        font-weight: 500;
    }

    .stage.current .stage-name {
        color: #212121;
    }

    .stage-check {
        color: #4caf50;
        font-size: 0.85rem;
        margin-left: auto;
    }

    .progress-bar {
        height: 6px;
        background: #e0e0e0;
        border-radius: 3px;
        overflow: hidden;
        margin-left: 28px;
    }

    .progress-bar.empty {
        background: #eeeeee;
    }

    .progress-fill {
        height: 100%;
        background: #2196f3;
        border-radius: 3px;
        transition: width 0.15s ease-out;
    }

    .stage.complete .progress-fill {
        background: #4caf50;
    }

    .progress-fill.indeterminate {
        width: 30%;
        animation: indeterminate 1.2s ease-in-out infinite;
    }

    @keyframes indeterminate {
        0% {
            transform: translateX(-100%);
        }
        50% {
            transform: translateX(233%);
        }
        100% {
            transform: translateX(-100%);
        }
    }
</style>
