<script lang="ts">
  import type { Status } from "./lib/api";
  import * as api from "./lib/api";

  import ActivationContextsTab from "./components/ActivationContextsTab.svelte";
  import { parseWandbRunPath } from "./lib";

  let loadingTrainRun = $state<boolean>(false);

  /** can be a wandb run path, or id. we sanitize this on sumbit */
  let trainWandbRunEntry = $state<string | null>(null);

  let status = $state<Status>({ train_run: null });

  async function loadStatus() {
    if (loadingTrainRun) return;
    console.log("getting status");
    try {
      status = await api.getStatus();
      loadingTrainRun = false;
      if (!status.train_run) return;
      trainWandbRunEntry = status.train_run.wandb_path;
    } catch (error) {
      console.error("error loading status", error);
    }
  }

  async function loadRun(event: Event) {
    event.preventDefault();
    const input = trainWandbRunEntry?.trim();
    if (!input) return;
    try {
      loadingTrainRun = true;
      status = { train_run: null };
      const wandbRunPath = parseWandbRunPath(input);
      console.log("loading run", wandbRunPath);
      trainWandbRunEntry = wandbRunPath;
      await api.loadRun(wandbRunPath);
      await loadStatus();
    } catch (error) {
      console.error("error loading run", error);
    } finally {
      loadingTrainRun = false;
    }
  }

  //   when the page loads, and every 5 seconds thereafter, load the status
  $effect(() => {
    loadStatus();
    const interval = setInterval(loadStatus, 5000);
    // return cleanup function to stop the polling
    return () => clearInterval(interval);
  });

  let activeTab = $state<"activation-contexts" | null>(null);
  $inspect(activeTab);

  let actsHidden = $derived(activeTab !== "activation-contexts");
  $inspect(actsHidden);
</script>

<div class="app-layout">
  <aside class="sidebar">
    <div class="run-selector">
      <div class="section-heading">W&B Run ID</div>
      <form onsubmit={loadRun} class="input-group">
        <input
          type="text"
          id="wandb-run-id"
          list="run-options"
          bind:value={trainWandbRunEntry}
          disabled={loadingTrainRun}
          placeholder="Select or enter run ID"
        />
        <button
          type="submit"
          disabled={loadingTrainRun || !trainWandbRunEntry?.trim()}
        >
          {loadingTrainRun ? "Loading..." : "Load Run"}
        </button>
      </form>
    </div>
    <div class="tab-navigation">
      {#if status.train_run}
        <button
          class="tab-button"
          class:active={activeTab === "activation-contexts"}
          onclick={() => {
            console.log("clicking activation contexts tab");
            activeTab = "activation-contexts";
          }}
        >
          Activation Contexts
        </button>
      {/if}
    </div>
    {#if status.train_run}
      <div class="config">
        <div class="section-heading">Config</div>
        <pre>{status.train_run?.config_yaml}</pre>
      </div>
    {/if}
  </aside>

  <div class="main-content">
    {#if status.train_run && !actsHidden}
      <ActivationContextsTab />
    {/if}
  </div>
</div>

<style>
  .app-layout {
    display: flex;
    min-height: 100vh;
  }

  .sidebar {
    background: #f8f9fa;
    border-right: 1px solid #dee2e6;
    padding: 1.5rem;
    display: flex;
    flex-direction: column;
    position: sticky;
    top: 0;
    height: 100vh;
    overflow-y: auto;
  }

  .main-content {
    flex: 1;
    min-width: 0;
    padding: 2rem;
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .tab-navigation {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .tab-button {
    padding: 0.75rem 1rem;
    background: white;
    border: 1px solid #dee2e6;
    border-radius: 6px;
    cursor: pointer;
    font-size: 0.9rem;
    font-weight: 500;
    color: #495057;
    transition: all 0.15s ease;
    text-align: left;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  .tab-button:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  .tab-button:hover {
    color: #007bff;
    background: #f8f9fa;
    border-color: #007bff;
  }

  .tab-button.active {
    color: white;
    background: #007bff;
    border-color: #007bff;
    box-shadow: 0 2px 4px rgba(0, 123, 255, 0.2);
  }

  .run-selector {
    margin-bottom: 1rem;
  }

  .section-heading {
    margin: 0 0 0.5rem 0;
    font-weight: 600;
    color: #333;
    font-size: 0.9rem;
  }

  .input-group {
    display: flex;
    gap: 0.5rem;
  }

  .input-group input[type="text"] {
    flex: 1;
    padding: 0.5rem;
    border: 1px solid #ddd;
    border-radius: 4px;
    font-size: 1rem;
  }

  .input-group input[type="text"]:focus {
    outline: none;
    border-color: #4a90e2;
    box-shadow: 0 0 0 2px rgba(74, 144, 226, 0.1);
  }

  .input-group button {
    padding: 0.5rem 1rem;
    background-color: #4a90e2;
    color: white;
    border: none;
    border-radius: 4px;
    font-size: 1rem;
    cursor: pointer;
    white-space: nowrap;
  }

  .input-group button:hover:not(:disabled) {
    background-color: #357abd;
  }

  .input-group button:disabled {
    background-color: #ccc;
    cursor: not-allowed;
  }

  .config {
    margin-top: 1rem;
  }

  .config pre {
    margin: 0;
    font-size: 0.8rem;
  }
</style>
