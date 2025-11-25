<script lang="ts">
    import TokenHighlights from "./TokenHighlights.svelte";

    interface Props {
        tokenStrings: string[];
        tokenCi: number[];  // CI values scaled by 100
        activePosition: number;
    }

    let { tokenStrings, tokenCi, activePosition }: Props = $props();
</script>

<div class="example-item">
    <TokenHighlights {tokenStrings} {tokenCi} {activePosition} />
</div>

<style>
    .example-item {
        font-family: monospace;
        font-size: 14px;
        line-height: 1.8;
        color: #333;
        padding: 0.5rem;
        /* background: white; */
        /* border-radius: 4px;
        border: 1px solid #e9ecef; */
        overflow: visible;
    }
</style>
