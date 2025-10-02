<script lang="ts">
    import type { ActivationContext } from "$lib/api";

    export let example: ActivationContext;

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

<div class="example-item">
    {@html renderTokenWithHighlight(
        example.raw_text,
        example.offset_mapping,
        example.token_ci_values,
        example.active_position
    )}
</div>

<style>
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
