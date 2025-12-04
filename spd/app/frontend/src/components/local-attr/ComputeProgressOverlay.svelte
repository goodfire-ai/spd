<script lang="ts">
    import type { LoadingProgress } from "./types";

    type Props = {
        mode: "standard" | "optimized";
        progress: LoadingProgress | null;
    };

    let { mode, progress }: Props = $props();
</script>

<div class="loading-overlay">
    {#if progress}
        <div class="progress-container">
            <div class="progress-bar">
                <div
                    class="progress-fill"
                    style="width: {(progress.current / progress.total) * 100}%"
                ></div>
            </div>
            <span class="progress-text">
                Computing {progress.stage}... ({progress.current}/{progress.total})
            </span>
        </div>
    {:else}
        <div class="loading-spinner"></div>
        <span>{mode === "optimized" ? "Running optimization..." : "Computing graph..."}</span>
    {/if}
</div>

<style>
    .loading-overlay {
        position: absolute;
        inset: 0;
        background: rgba(255, 255, 255, 0.9);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 0.75rem;
        z-index: 100;
    }

    .loading-spinner {
        width: 28px;
        height: 28px;
        border: 3px solid #e0e0e0;
        border-top-color: #2196f3;
        border-radius: 50%;
        animation: spin 0.8s linear infinite;
    }

    @keyframes spin {
        to {
            transform: rotate(360deg);
        }
    }

    .progress-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0.5rem;
        width: 280px;
    }

    .progress-bar {
        width: 100%;
        height: 6px;
        background: #e0e0e0;
        border-radius: 3px;
        overflow: hidden;
    }

    .progress-fill {
        height: 100%;
        background: #2196f3;
        transition: width 0.15s ease-out;
    }

    .progress-text {
        font-size: 0.8rem;
        color: #757575;
    }
</style>
