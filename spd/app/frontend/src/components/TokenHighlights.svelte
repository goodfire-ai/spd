<script lang="ts">
    import { getTokenHighlightBg, getInnerActUnderlineColor } from "../lib/colors";
    import type { ExampleColorMode } from "../lib/displaySettings.svelte";

    interface Props {
        tokenStrings: string[];
        tokenCi: number[]; // CI values (0-1)
        tokenInnerActs: number[]; // Inner activations (can be negative)
        colorMode: ExampleColorMode;
        maxAbsInnerAct: number; // For normalizing inner act colors
    }

    let { tokenStrings, tokenCi, tokenInnerActs, colorMode, maxAbsInnerAct }: Props = $props();

    function getTooltipText(ci: number, innerAct: number): string {
        if (colorMode === "ci") {
            return `CI: ${ci.toFixed(3)}`;
        } else if (colorMode === "inner_act") {
            return `Act: ${innerAct.toFixed(3)}`;
        } else {
            return `CI: ${ci.toFixed(3)} | Act: ${innerAct.toFixed(3)}`;
        }
    }

    function getBgColor(ci: number, innerAct: number): string {
        if (colorMode === "inner_act") {
            // Inner act mode: use inner act for background coloring (normalized)
            const normalizedAbs = Math.abs(innerAct) / maxAbsInnerAct;
            if (innerAct >= 0) {
                return `rgba(37, 99, 235, ${normalizedAbs})`; // blue
            } else {
                return `rgba(220, 38, 38, ${normalizedAbs})`; // red
            }
        } else {
            // CI or Both mode: use CI for background
            return getTokenHighlightBg(ci);
        }
    }

    function getUnderlineStyle(innerAct: number): string {
        if (colorMode !== "both") return "none";
        const normalizedAbs = Math.abs(innerAct) / maxAbsInnerAct;
        return `3px solid ${getInnerActUnderlineColor(innerAct, normalizedAbs)}`;
    }
</script>

<span class="token-highlights">
    {#each tokenStrings as tok, i (i)}
        <span
            class="token-highlight"
            style="background-color:{getBgColor(tokenCi[i], tokenInnerActs[i])};border-bottom:{getUnderlineStyle(
                tokenInnerActs[i],
            )}"
            data-tooltip={getTooltipText(tokenCi[i], tokenInnerActs[i])}
        >
            {tok}
        </span>
    {/each}
</span>

<style>
    .token-highlights {
        display: inline;
        white-space: pre-wrap;
        font-family: var(--font-mono);
    }

    .token-highlight {
        display: inline;
        padding: 1px 0 4px 0;
        margin-right: 1px;
        border-right: 1px solid var(--border-subtle);
        position: relative;
        white-space: pre;
    }

    .token-highlight::after {
        content: attr(data-tooltip);
        position: absolute;
        top: calc(100% + 4px);
        left: 0;
        background: var(--bg-elevated);
        border: 1px solid var(--border-strong);
        color: var(--text-primary);
        padding: var(--space-1) var(--space-2);
        font-size: var(--text-xs);
        font-family: var(--font-mono);
        white-space: nowrap;
        opacity: 0;
        pointer-events: none;
        z-index: 1000;
    }

    .token-highlight:hover::after {
        opacity: 1;
    }
</style>
