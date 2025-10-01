<script lang="ts">
    import { onMount } from "svelte";
    import { api } from "../api";
    import type { ActivationContext } from "../api";

    export let componentId: number;
    export let layer: string;
    // export let maxExamples: number = 5;
    // export let contextSize: number = 10;
    // export let threshold: number = 0.01;
    export let compact: boolean = false; // For modal view vs full tab view

    let contexts: ActivationContext[] = [];
    let loading = false;
    let error: string | null = null;

    async function loadContexts() {
        loading = true;
        error = null;
        try {
            const result = await api.getComponentActivationContexts(
                componentId,
                layer,
                // maxExamples,
                // contextSize,
                // threshold
            );
            contexts = result.examples;
        } catch (e: any) {
            error = e.message || "Failed to load activation contexts";
            console.error("Failed to load activation contexts:", e);
        }
        loading = false;
    }

    onMount(() => {
        loadContexts();
    });

    function getHighlightColor(importance: number): string {
        const importanceNorm = Math.min(Math.max(importance, 0), 1);
        const opacity = 0.15 + importanceNorm * 0.35;
        return `rgba(0, 200, 0, ${opacity})`;
    }

    function renderTokenWithHighlight(
        rawText: string,
        offsetMapping: [number, number][],
        tokenCiValues: number[],
        activePosition: number
    ): string {
        let htmlChunks: string[] = [];
        let cursor = 0;

        for (let idx = 0; idx < offsetMapping.length; idx++) {
            const [start, end] = offsetMapping[idx];

            // Add text between tokens
            if (cursor < start) {
                htmlChunks.push(escapeHtml(rawText.substring(cursor, start)));
            }

            const escapedText = escapeHtml(rawText.substring(start, end));
            const ciValue = tokenCiValues[idx];

            if (ciValue > 0) {
                const bgColor = getHighlightColor(ciValue);
                const borderStyle =
                    idx === activePosition
                        ? "border: 2px solid rgba(255,100,0,0.6);"
                        : "";
                htmlChunks.push(
                    `<span style="background-color:${bgColor}; padding: 2px 4px; border-radius: 3px; ${borderStyle}" title="Importance: ${ciValue.toFixed(3)}">${escapedText}</span>`
                );
            } else {
                htmlChunks.push(escapedText);
            }

            cursor = end;
        }

        // Add remaining text
        if (cursor < rawText.length) {
            htmlChunks.push(escapeHtml(rawText.substring(cursor)));
        }

        return htmlChunks.join("");
    }

    function escapeHtml(text: string): string {
        const div = document.createElement("div");
        div.textContent = text;
        return div.innerHTML;
    }
</script>

<div class="activation-contexts" class:compact>
    {#if loading}
        <div class="loading">Loading activation examples...</div>
    {:else if error}
        <div class="error">{error}</div>
    {:else if contexts.length === 0}
        <div class="empty">No activation examples found above threshold.</div>
    {:else}
        {#if !compact}
            <div class="header">
                <h4>Component {componentId}</h4>
                <span class="example-count">{contexts.length} examples</span>
            </div>
        {/if}

        <div class="examples-container" class:compact-container={compact}>
            {#each contexts as context, i}
                <div class="example-item">
                    {#if !compact}
                        <strong>{i + 1}.</strong>
                    {/if}
                    {@html renderTokenWithHighlight(
                        context.raw_text,
                        context.offset_mapping,
                        context.token_ci_values,
                        context.active_position
                    )}
                </div>
            {/each}
        </div>
    {/if}
</div>

<style>
    .activation-contexts {
        padding: 1rem;
        background: var(--background-color, #fff);
        border-radius: 6px;
        border: 1px solid #e0e0e0;
    }

    .activation-contexts.compact {
        padding: 0.5rem;
        border: none;
        background: transparent;
    }

    .header {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 0.75rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #e0e0e0;
    }

    .header h4 {
        margin: 0;
        font-size: 1rem;
        color: #333;
    }

    .example-count {
        font-size: 0.85rem;
        color: #999;
        margin-left: auto;
    }

    .examples-container {
        display: flex;
        flex-direction: column;
        gap: 0.75rem;
        overflow-y: auto;
    }

    .compact-container {
        gap: 0.5rem;
    }

    .example-item {
        font-family: monospace;
        font-size: 14px;
        line-height: 1.8;
        color: #333;
        padding: 0.5rem;
        background: #f8f9fa;
        border-radius: 4px;
        border: 1px solid #e9ecef;
    }

    .compact .example-item {
        font-size: 13px;
        padding: 0.4rem;
    }

    .loading,
    .error,
    .empty {
        padding: 1rem;
        text-align: center;
        color: #666;
        font-style: italic;
    }

    .error {
        color: #dc3545;
    }

    /* Tooltip styles for highlighted tokens */
    :global(.activation-contexts span[title]) {
        position: relative;
        cursor: help;
    }

    :global(.activation-contexts span[title]:hover::after) {
        content: attr(title);
        position: absolute;
        bottom: 100%;
        left: 50%;
        transform: translateX(-50%);
        background-color: rgba(40, 40, 40, 0.95);
        color: rgba(255, 255, 255, 1);
        padding: 4px 8px;
        border-radius: 3px;
        font-size: 0.75em;
        white-space: nowrap;
        z-index: 10000;
        pointer-events: none;
        margin-bottom: 5px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3);
        font-weight: 500;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    :global(.activation-contexts span[title]:hover::before) {
        content: "";
        position: absolute;
        bottom: 100%;
        left: 50%;
        transform: translateX(-50%);
        border: 4px solid transparent;
        border-top-color: rgba(40, 40, 40, 0.95);
        z-index: 10000;
        pointer-events: none;
        margin-bottom: 1px;
    }
</style>