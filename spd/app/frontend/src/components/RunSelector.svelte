<script lang="ts">
    import { CANONICAL_RUNS, formatRunIdForDisplay, type RegistryEntry } from "../lib/registry";

    type Props = {
        onSelect: (wandbPath: string, contextLength: number) => void;
        isLoading: boolean;
        username: string | null;
    };

    let { onSelect, isLoading, username }: Props = $props();

    let customPath = $state("");
    let contextLength = $state(512);

    function handleRegistrySelect(entry: RegistryEntry) {
        onSelect(entry.wandbRunId, contextLength);
    }

    function handleCustomSubmit(event: Event) {
        event.preventDefault();
        const path = customPath.trim();
        if (!path) return;
        onSelect(path, contextLength);
    }
</script>

<div class="selector-container">
    <div class="selector-content">
        {#if username}
            <p class="greeting">Hello, {username}</p>
        {/if}
        <h1 class="title">SPD Explorer</h1>
        <p class="subtitle">Select a run to explore component decompositions</p>

        <div class="context-length-section">
            <label for="context-length">Context Length:</label>
            <input
                type="number"
                id="context-length"
                bind:value={contextLength}
                disabled={isLoading}
                min="1"
                max="2048"
            />
        </div>

        <div class="runs-grid">
            {#each CANONICAL_RUNS as entry (entry.wandbRunId)}
                <button
                    class="run-card"
                    onclick={() => handleRegistrySelect(entry)}
                    disabled={isLoading}
                >
                    <span class="run-model">{entry.modelName}</span>
                    <span class="run-id">{formatRunIdForDisplay(entry.wandbRunId)}</span>
                    {#if entry.notes}
                        <span class="run-notes">{entry.notes}</span>
                    {/if}
                </button>
            {/each}
        </div>

        <div class="divider">
            <span>or enter a custom path</span>
        </div>

        <form class="custom-form" onsubmit={handleCustomSubmit}>
            <input
                type="text"
                placeholder="e.g. goodfire/spd/runs/33n6xjjt"
                bind:value={customPath}
                disabled={isLoading}
            />
            <button type="submit" disabled={isLoading || !customPath.trim()}>
                {isLoading ? "Loading..." : "Load"}
            </button>
        </form>
    </div>
</div>

<style>
    .selector-container {
        display: flex;
        align-items: center;
        justify-content: center;
        min-height: 100vh;
        background: var(--bg-base);
        padding: var(--space-4);
    }

    .selector-content {
        max-width: 720px;
        width: 100%;
    }

    .greeting {
        font-size: var(--text-sm);
        color: var(--text-muted);
        margin: 0 0 var(--space-4) 0;
        text-align: center;
        font-family: var(--font-sans);
    }

    .title {
        font-size: var(--text-3xl);
        font-weight: 600;
        color: var(--text-primary);
        margin: 0 0 var(--space-2) 0;
        text-align: center;
        font-family: var(--font-sans);
    }

    .subtitle {
        font-size: var(--text-base);
        color: var(--text-muted);
        margin: 0 0 var(--space-6) 0;
        text-align: center;
        font-family: var(--font-sans);
    }

    .context-length-section {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: var(--space-2);
        margin-bottom: var(--space-4);
    }

    .context-length-section label {
        font-size: var(--text-sm);
        color: var(--text-secondary);
        font-family: var(--font-sans);
        font-weight: 500;
    }

    .context-length-section input {
        width: 80px;
        padding: var(--space-1) var(--space-2);
        border: 1px solid var(--border-default);
        border-radius: var(--radius-sm);
        background: var(--bg-elevated);
        color: var(--text-primary);
        font-size: var(--text-sm);
        font-family: var(--font-mono);
    }

    .runs-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
        gap: var(--space-3);
        margin-bottom: var(--space-6);
    }

    .run-card {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        gap: var(--space-1);
        padding: var(--space-3);
        background: var(--bg-surface);
        border: 1px solid var(--border-default);
        border-radius: var(--radius-md);
        cursor: pointer;
        text-align: left;
        transition: border-color 0.15s, background 0.15s;
    }

    .run-card:hover:not(:disabled) {
        border-color: var(--accent-primary);
        background: var(--bg-elevated);
    }

    .run-card:disabled {
        opacity: 0.5;
        cursor: not-allowed;
    }

    .run-model {
        font-size: var(--text-sm);
        font-weight: 600;
        color: var(--text-primary);
        font-family: var(--font-sans);
    }

    .run-id {
        font-size: var(--text-xs);
        font-family: var(--font-mono);
        color: var(--accent-primary);
    }

    .run-notes {
        font-size: var(--text-xs);
        color: var(--text-muted);
        font-family: var(--font-sans);
    }

    .divider {
        display: flex;
        align-items: center;
        gap: var(--space-3);
        margin-bottom: var(--space-4);
    }

    .divider::before,
    .divider::after {
        content: "";
        flex: 1;
        height: 1px;
        background: var(--border-default);
    }

    .divider span {
        font-size: var(--text-sm);
        color: var(--text-muted);
        font-family: var(--font-sans);
    }

    .custom-form {
        display: flex;
        gap: var(--space-2);
    }

    .custom-form input[type="text"] {
        flex: 1;
        padding: var(--space-2) var(--space-3);
        border: 1px solid var(--border-default);
        border-radius: var(--radius-sm);
        background: var(--bg-elevated);
        color: var(--text-primary);
        font-size: var(--text-sm);
        font-family: var(--font-mono);
    }

    .custom-form input[type="text"]::placeholder {
        color: var(--text-muted);
    }

    .custom-form input[type="text"]:focus {
        outline: none;
        border-color: var(--accent-primary-dim);
    }

    .custom-form button {
        padding: var(--space-2) var(--space-4);
        background: var(--accent-primary);
        color: white;
        border: none;
        border-radius: var(--radius-sm);
        font-weight: 500;
        cursor: pointer;
        font-family: var(--font-sans);
    }

    .custom-form button:hover:not(:disabled) {
        opacity: 0.9;
    }

    .custom-form button:disabled {
        background: var(--border-default);
        color: var(--text-muted);
        cursor: not-allowed;
    }
</style>
