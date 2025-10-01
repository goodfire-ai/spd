<script lang="ts">
    import type { ActivationContext } from "$lib/api";

    export let component_idx: number;
    export let examples: ActivationContext[];

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
                    idx === activePosition ? "border: 2px solid rgba(255,100,0,0.6);" : "";
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

<div class="activation-contexts">
    <div class="header">
        <h4>Component {component_idx}</h4>
        <span class="example-count">{examples.length} examples</span>
    </div>

    <div class="examples-container">
        {#each examples as context, i}
            <div class="example-item">
                <strong>{i + 1}.</strong>
                {@html renderTokenWithHighlight(
                    context.raw_text,
                    context.offset_mapping,
                    context.token_ci_values,
                    context.active_position
                )}
            </div>
        {/each}
    </div>
</div>

<style>
    .activation-contexts {
        padding: 1rem;
        background: var(--background-color, #fff);
        border-radius: 6px;
        border: 1px solid #e0e0e0;
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
