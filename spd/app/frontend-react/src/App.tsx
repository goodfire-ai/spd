import { useState, useEffect } from "react";
import "./App.css";
import { getStatus, loadRun, loadClusterRun } from "./api";
import type { Status } from "./api";
import { ComponentStateProvider } from "./stores/componentState";
import { ActivationContextsTab } from "./components/ActivationContextsTab";
import { InterventionsTab } from "./components/InterventionsTab";
import { ClusterDashboardTab } from "./components/ClusterDashboardTab";

function App() {
  const [inputTrainWandbRunId, setInputTrainWandbRunId] = useState<string>("");
  const [loadingTrainRun, setLoadingTrainRun] = useState(false);

  const [status, setStatus] = useState<Status | null>(null);

  const [clusterWandbRunPath, setClusterWandbRunPath] = useState<string | null>(
    null
  );
  const [clusterIteration, setClusterIteration] = useState<number | null>(null);
  const [loadingClusterRun, setLoadingClusterRun] = useState(false);

  const [activeTab, setActiveTab] = useState<string | null>(null);

  const loadStatus = async () => {
    const statusData = await getStatus();
    setStatus(statusData);
    if (!statusData.train_run) {
      return;
    }
    if (!statusData.cluster_run) {
      return;
    }
    setClusterWandbRunPath(statusData.cluster_run.wandb_path);
  };

  useEffect(() => {
    setInterval(loadStatus, 3000);
  }, []);

  const handleLoadRun = async () => {
    if (!inputTrainWandbRunId?.trim()) return;

    setStatus(null);
    setLoadingTrainRun(true);
    try {
      await loadRun(inputTrainWandbRunId);
    } catch (error) {
      console.error("Error loading run", error);
    } finally {
      setLoadingTrainRun(false);
    }
    await loadStatus();
  };

  const handleLoadClusterRun = async () => {
    console.log("loading cluster run", clusterWandbRunPath, clusterIteration);
    const canLoadCluster =
      clusterWandbRunPath !== null && clusterIteration !== null;
    if (!canLoadCluster) {
      console.log("cannot submit cluster settings", {
        clusterWandbRunPath,
        clusterIteration,
      });
      return;
    }

    setLoadingClusterRun(true);
    await loadClusterRun(
      clusterWandbRunPath!.split("/").pop()!,
      clusterIteration!
    );
    setLoadingClusterRun(false);

    await loadStatus();
  };

  useEffect(() => {
    loadStatus();
  }, []);

  return (
    <ComponentStateProvider>
      <div className="app-layout">
        <aside className="sidebar">
          <div className="run-selector">
            <label htmlFor="wandb-run-id">W&B Run ID</label>
            <div className="input-group">
              <input
                type="text"
                id="wandb-run-id"
                value={inputTrainWandbRunId}
                onChange={(e) => setInputTrainWandbRunId(e.target.value)}
                disabled={loadingTrainRun}
                placeholder="Select or enter run ID"
              />
              <button
                onClick={handleLoadRun}
                disabled={loadingTrainRun || !inputTrainWandbRunId?.trim()}
              >
                {loadingTrainRun ? "Loading..." : "Load Run"}
              </button>
            </div>
          </div>

          <div className="tab-navigation">
            {status?.train_run && (
              <>
                <button
                  className={`tab-button ${
                    activeTab === "activation-contexts" ? "active" : ""
                  }`}
                  onClick={() => setActiveTab("activation-contexts")}
                >
                  Activation Contexts
                </button>

                <div className="cluster-settings">
                  <h4>Cluster Settings</h4>
                  <form
                    onSubmit={(e) => {
                      e.preventDefault();
                      handleLoadClusterRun();
                    }}
                  >
                    <label>
                      Clustering Run
                      <select
                        value={clusterWandbRunPath || ""}
                        onChange={(e) => setClusterWandbRunPath(e.target.value)}
                      >
                        {status?.train_run?.available_cluster_runs?.map((run) => (
                          <option key={run} value={run}>
                            {run}
                          </option>
                        ))}
                      </select>
                    </label>
                    <div className="settings-grid">
                      <label>
                        Iteration
                        <input
                          type="number"
                          value={clusterIteration || ""}
                          onChange={(e) =>
                            setClusterIteration(
                              e.target.value ? parseInt(e.target.value) : null
                            )
                          }
                        />
                      </label>
                    </div>
                    <button
                      className="cluster-load"
                      type="submit"
                      disabled={loadingClusterRun}
                    >
                      Load Cluster Run
                    </button>
                  </form>
                </div>

                <button
                  className={`tab-button ${
                    activeTab === "ablation" ? "active" : ""
                  }`}
                  disabled={!status?.cluster_run}
                  onClick={() => setActiveTab("ablation")}
                >
                  Component Ablation
                  {loadingClusterRun && <div className="spinner"></div>}
                </button>

                <button
                  className={`tab-button ${
                    activeTab === "cluster-dashboard" ? "active" : ""
                  }`}
                  disabled={!status?.cluster_run}
                  onClick={() => setActiveTab("cluster-dashboard")}
                >
                  Cluster Dashboard
                  {loadingClusterRun && <div className="spinner"></div>}
                </button>
              </>
            )}
          </div>
        </aside>
        <div className="main-content">
          {status?.train_run && (
            <div
              style={{
                display: activeTab === "activation-contexts" ? "block" : "none",
              }}
            >
              <ActivationContextsTab
                availableComponentLayers={status.train_run.component_layers}
              />
            </div>
          )}
          {status?.train_run && activeTab === "ablation" && (
            <div>
              {status?.cluster_run && clusterIteration !== null ? (
                <InterventionsTab cluster_run={status.cluster_run} iteration={clusterIteration} />
              ) : (
                <div className="status">No cluster run selected.</div>
              )}
            </div>
          )}
          {status?.train_run && activeTab === "cluster-dashboard" && (
            <div>
              {status?.cluster_run && clusterIteration !== null ? (
                <ClusterDashboardTab iteration={clusterIteration} />
              ) : (
                <div className="status">No cluster run selected.</div>
              )}
            </div>
          )}
        </div>
      </div>
    </ComponentStateProvider>
  );
}

export default App;
