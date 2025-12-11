<script lang="ts">
    import {
        viewSettings,
        CORRELATION_STAT_LABELS,
        CORRELATION_STAT_DESCRIPTIONS,
        type CorrelationStatType,
    } from "../../lib/viewSettings.svelte";

    let showDropdown = $state(false);

    const statTypes: CorrelationStatType[] = ["pmi", "precision", "recall", "f1", "jaccard"];
</script>

<div
    class="settings-wrapper"
    role="group"
    onmouseenter={() => (showDropdown = true)}
    onmouseleave={() => (showDropdown = false)}
>
    <button type="button" class="settings-button">View Settings</button>
    {#if showDropdown}
        <div class="settings-dropdown">
            <div class="settings-section">
                <h4>Correlation Stats</h4>
                <p class="settings-hint">Select which correlation metrics to display</p>
                <div class="checkbox-list">
                    {#each statTypes as stat (stat)}
                        <label class="checkbox-item">
                            <input
                                type="checkbox"
                                checked={viewSettings.isCorrelationStatVisible(stat)}
                                onchange={() => viewSettings.toggleCorrelationStat(stat)}
                            />
                            <span class="stat-label">{CORRELATION_STAT_LABELS[stat]}</span>
                            <span class="stat-desc">{CORRELATION_STAT_DESCRIPTIONS[stat]}</span>
                        </label>
                    {/each}
                </div>
            </div>
        </div>
    {/if}
</div>

<style>
    .settings-wrapper {
        position: relative;
    }

    .settings-button {
        padding: var(--space-1) var(--space-2);
        background: var(--bg-elevated);
        border: 1px solid var(--border-default);
        border-radius: var(--radius-sm);
        cursor: pointer;
        font-size: var(--text-sm);
        font-family: var(--font-sans);
        color: var(--text-secondary);
        font-weight: 500;
    }

    .settings-button:hover {
        background: var(--bg-inset);
        color: var(--text-primary);
    }

    .settings-dropdown {
        position: absolute;
        top: 100%;
        right: 0;
        padding-top: var(--space-2);
        z-index: 1000;
        min-width: 280px;
    }

    .settings-section {
        background: var(--bg-elevated);
        border: 1px solid var(--border-strong);
        border-radius: var(--radius-md);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        padding: var(--space-3);
    }

    .settings-section h4 {
        margin: 0 0 var(--space-1) 0;
        font-size: var(--text-sm);
        font-weight: 600;
        color: var(--text-primary);
    }

    .settings-hint {
        margin: 0 0 var(--space-2) 0;
        font-size: var(--text-xs);
        color: var(--text-muted);
    }

    .checkbox-list {
        display: flex;
        flex-direction: column;
        gap: var(--space-2);
    }

    .checkbox-item {
        display: grid;
        grid-template-columns: auto 1fr;
        grid-template-rows: auto auto;
        gap: 0 var(--space-2);
        cursor: pointer;
        padding: var(--space-1);
        border-radius: var(--radius-sm);
    }

    .checkbox-item:hover {
        background: var(--bg-inset);
    }

    .checkbox-item input {
        grid-row: span 2;
        margin: 0;
        cursor: pointer;
        accent-color: var(--accent-primary);
    }

    .stat-label {
        font-size: var(--text-sm);
        font-weight: 500;
        color: var(--text-primary);
    }

    .stat-desc {
        font-size: var(--text-xs);
        color: var(--text-muted);
        font-family: var(--font-mono);
    }
</style>
