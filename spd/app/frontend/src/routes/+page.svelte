<script lang="ts">
    import { onMount } from "svelte";
    import { api } from "$lib/api";
    import type { RunPromptResponse, ComponentMask, StatusDTO, SparseVector } from "$lib/api";
    import { runAblation, popupData, ablationResults } from "$lib/stores/componentState";

    import RunSelector from "$lib/components/RunSelector.svelte";
    import PromptInput from "$lib/components/PromptInput.svelte";
    import ComponentHeatmap from "$lib/components/ComponentHeatmap.svelte";
    import DisabledComponentsPanel from "$lib/components/DisabledComponentsPanel.svelte";
    import ComponentDetailModal from "$lib/components/ComponentDetailModal.svelte";
    import OriginalPredictions from "$lib/components/OriginalPredictions.svelte";
    import AblationPredictions from "$lib/components/AblationPredictions.svelte";

    let isLoading = false;
    let result: RunPromptResponse | null = null;
    let wandbRunId: string | null = null;
    let loadingRun = false;

    async function runRandomPrompt() {
        isLoading = true;
        try {
            result = await api.runRandomPrompt();
            $ablationResults = [];
        } catch (error: any) {
            console.error(`Error: ${error.message}`);
        }
        isLoading = false;
    }

    function initializeRunAblation() {
        if (!result) return;
        const newMask: ComponentMask = {};
        for (const layer of result.layer_cis) {
            newMask[layer.module] = result.prompt_tokens.map(() => []);
        }
        $runAblation = newMask;
    }

    function toggleComponentDisabled(layerName: string, tokenIdx: number, componentIdx: number) {
        if (!$runAblation[layerName]) {
            $runAblation[layerName] = result!.prompt_tokens.map(() => []);
        }

        const disabledComponents = $runAblation[layerName][tokenIdx];
        const existingIdx = disabledComponents.indexOf(componentIdx);

        if (existingIdx === -1) {
            disabledComponents.push(componentIdx);
        } else {
            disabledComponents.splice(existingIdx, 1);
        }

        $runAblation = { ...$runAblation };
    }

    function isComponentDisabled(
        layerName: string,
        tokenIdx: number,
        componentIdx: number
    ): boolean {
        return $runAblation[layerName][tokenIdx].includes(componentIdx)
    }

    async function sendAblation() {
        if (!result) return;

        isLoading = true;
        try {
            const data = await api.ablateComponents($runAblation);

            const deepCopyMask: ComponentMask = {};
            for (const [layerName, tokenArrays] of Object.entries($runAblation)) {
                deepCopyMask[layerName] = tokenArrays.map((tokenMask) => [...tokenMask]);
            }

            $ablationResults = [
                ...$ablationResults,
                {
                    tokenLogits: data.token_logits,
                    applied_mask: deepCopyMask,
                    id: Date.now()
                }
            ];
        } catch (error: any) {
            console.error(`Error: ${error.message}`);
        }
        isLoading = false;
    }

    function openPopup(
        token: string,
        tokenIdx: number,
        layer: string,
        layerIdx: number,
        tokenCis: SparseVector,
    ) {
        $popupData = { token, tokenIdx, layer, layerIdx, tokenCis };
    }

    function closePopup() {
        $popupData = null;
    }

    $: if (result) {
        initializeRunAblation();
    }

    let status: StatusDTO | null = null;
    async function getStatus() {
        status = await api.getStatus();
    }

    $: wandbRunId = status?.run_id ?? null;

    onMount(() => getStatus());
</script>

<main>
    <div class="container">
        <div class="main-layout">
            <!-- Left side: Static heatmap and controls -->
            <div class="left-panel">
                <RunSelector
                    bind:loadingRun
                    bind:wandbRunId
                    {isLoading}
                    onRunLoaded={runRandomPrompt}
                />

                <!-- <PromptInput bind:prompt {isLoading} onInput={runPrompt}
                onRandomPrompt={runRandomPrompt} /> -->
                <button on:click={runRandomPrompt}>Random Prompt</button>

                {#if result}
                    <ComponentHeatmap {result} onCellClick={openPopup} />

                    <DisabledComponentsPanel
                        promptTokens={result.prompt_tokens}
                        {isLoading}
                        onSendAblation={sendAblation}
                        onToggleComponent={toggleComponentDisabled}
                    />
                {/if}
            </div>

            <!-- Right side: Scrollable predictions -->
            <div class="right-panel">
                {#if result && result.full_run_token_logits}
                    <OriginalPredictions
                        tokenLogits={result.full_run_token_logits}
                        promptTokens={result.prompt_tokens}
                        title="Original Model Predictions"
                    />
                {/if}

                {#if result && result.ci_masked_token_logits}
                    <OriginalPredictions
                        tokenLogits={result.ci_masked_token_logits}
                        promptTokens={result.prompt_tokens}
                        title="Original <strong>CI Masked</strong> Model Predictions"
                    />
                {/if}

                {#if result}
                    {#each $ablationResults as ablationResult}
                        <AblationPredictions
                            tokenLogits={ablationResult.tokenLogits}
                            promptTokens={result.prompt_tokens}
                            appliedMask={ablationResult.applied_mask}
                        />
                    {/each}
                {/if}
            </div>
        </div>

        <ComponentDetailModal
            onClose={closePopup}
            onToggleComponent={toggleComponentDisabled}
            {isComponentDisabled}
        />
    </div>
</main>

<style>
    main {
        padding: 0.5rem;
        font-family:
            system-ui,
            -apple-system,
            sans-serif;
    }

    .container {
        max-width: none;
        margin: 0;
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }

    .main-layout {
        display: flex;
        gap: 1rem;
        min-height: 80vh;
    }

    .left-panel {
        padding: 0.5rem;
        flex: 0 0 50%;
        max-width: 50%;
        position: sticky;
        top: 1rem;
        align-self: flex-start;
        max-height: calc(100vh - 2rem);
        overflow-y: auto;
    }

    .right-panel {
        flex: 1;
        overflow-y: auto;
        padding-right: 1rem;
    }
</style>
