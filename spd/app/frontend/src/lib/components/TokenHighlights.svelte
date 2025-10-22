<!-- svelte-ignore a11y_click_events_have_key_events -->
<!-- svelte-ignore a11y_no_static_element_interactions -->
<script lang="ts">
    export let rawText: string;
    export let offsetMapping: [number, number][];
    export let tokenCiValues: number[];
    export let activePosition: number = -1;
    export let precision: number = 3;

    type Segment = {
        text: string;
        ciValue: number;
        isActive: boolean;
    };

    const defaultHighlightColor = (importance: number): string => {
        const clamped = Math.min(Math.max(importance, 0), 1);
        const opacity = 0.15 + clamped * 0.35;
        return `rgba(0, 200, 0, ${opacity})`;
    };

    export let getHighlightColor: (importance: number) => string = defaultHighlightColor;

    $: segments = buildSegments(rawText, offsetMapping, tokenCiValues, activePosition);

    function buildSegments(
        text: string,
        offsets: [number, number][],
        ciValues: number[],
        activeIdx: number
    ): Segment[] {
        const result: Segment[] = [];
        let cursor = 0;

        offsets.forEach(([start, end], idx) => {
            if (cursor < start) {
                result.push({ text: text.slice(cursor, start), ciValue: 0, isActive: false });
            }

            const tokenText = text.slice(start, end);
            const ciValue = ciValues[idx] ?? 0;
            result.push({ text: tokenText, ciValue, isActive: idx === activeIdx });

            cursor = end;
        });

        if (cursor < text.length) {
            result.push({ text: text.slice(cursor), ciValue: 0, isActive: false });
        }

        return result.filter((segment) => segment.text.length > 0);
    }
</script>

<span class="token-highlights">
    {#each segments as segment, idx}
        {#if segment.ciValue > 0}
            <span
                class="token-highlight"
                class:active-token={segment.isActive}
                style={`background-color:${getHighlightColor(segment.ciValue)};`}
                data-ci={`CI: ${segment.ciValue.toFixed(precision)}`}>{segment.text}</span
            >
        {:else}
            {segment.text}
        {/if}
    {/each}
</span>

<style>
    .token-highlights {
        display: inline;
        white-space: pre-wrap;
    }

    .token-highlight {
        display: inline;
        padding: 2px 4px;
        border-radius: 3px;
        position: relative;
    }

    .token-highlight::after {
        content: attr(data-ci);
        position: absolute;
        bottom: 100%;
        left: 50%;
        transform: translateX(-50%);
        background: rgba(0, 0, 0, 0.9);
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
        white-space: nowrap;
        opacity: 0;
        pointer-events: none;
        transition: opacity 0s;
        margin-bottom: 4px;
        z-index: 1000;
    }

    .token-highlight:hover::after {
        opacity: 1;
    }

    .token-highlight.active-token {
        border: 2px solid rgba(255, 100, 0, 0.6);
    }
</style>
