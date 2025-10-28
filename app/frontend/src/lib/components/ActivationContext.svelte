<script lang="ts">
    import type { ActivationContext } from "../api";
    import TokenHighlights from "./TokenHighlights.svelte";

    export let example: ActivationContext;
</script>

<div class="example-item">
    <TokenHighlights
        tokenStrings={example.token_strings}
        tokenCiValues={example.token_ci_values}
        activePosition={example.active_position}
    />
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
        overflow: visible;
    }
</style>
