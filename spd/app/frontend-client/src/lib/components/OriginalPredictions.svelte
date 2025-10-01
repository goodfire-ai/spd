<script lang="ts">
    import TokenPredictions from "./TokenPredictions.svelte";
    import type { OutputTokenLogit } from "../api";

    export let tokenLogits: OutputTokenLogit[][];
    export let promptTokens: string[];
    export let title: string = "Original Model Predictions";
</script>

<div class="predictions-section">
    <h2>{@html title}</h2>
    <TokenPredictions
        {tokenLogits}
        {promptTokens}
        containerClass="original"
    />
</div>

<style>
    .predictions-section {
        padding: 1rem;
        border: 2px solid #4caf50;
        border-radius: 8px;
        background-color: #f8fff8;
        margin-bottom: 1rem;
    }

    .predictions-section h2 {
        margin: 0 0 0.5rem 0;
        color: #2e7d2e;
    }
</style>