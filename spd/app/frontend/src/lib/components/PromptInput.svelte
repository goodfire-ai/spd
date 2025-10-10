<script lang="ts">
    export let prompt: string;
    export let isLoading: boolean;
    export let onInput: () => void;
    export let onRandomPrompt: () => void;
</script>

<div class="prompt-section">
    <textarea
        id="prompt"
        bind:value={prompt}
        disabled={isLoading}
        placeholder="Enter your prompt here..."
        rows="2"
        on:input={onInput}
    ></textarea>

    <button on:click={onRandomPrompt}>Random Prompt</button>
</div>

<style>
    .prompt-section {
        display: flex;
        flex-direction: row;
        gap: 0.5rem;
    }

    textarea {
        flex: 1;
        padding: 0.5rem;
        border: 1px solid #ccc;
        border-radius: 4px;
        font-size: 1rem;
        resize: vertical;
    }

    button {
        padding: 0.5rem 1rem;
        background-color: #4a90e2;
        color: white;
        border: none;
        border-radius: 4px;
        font-size: 1rem;
        cursor: pointer;
        white-space: nowrap;
    }

    button:hover:not(:disabled) {
        background-color: #357abd;
    }

    button:disabled {
        background-color: #ccc;
        cursor: not-allowed;
    }
</style>