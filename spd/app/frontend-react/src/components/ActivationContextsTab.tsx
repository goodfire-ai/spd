import React, { useState } from "react";
import type {
  ActivationContextsConfig,
  SubcomponentActivationContexts,
} from "../api";
import { getLayerActivationContexts } from "../api";
import { ActivationContext } from "./ActivationContext";

interface ActivationContextsTabProps {
  availableComponentLayers: string[];
}

export const ActivationContextsTab: React.FC<ActivationContextsTabProps> = ({
  availableComponentLayers,
}) => {
  if (availableComponentLayers.length === 0) {
    throw new Error(
      `No component layers available: ${availableComponentLayers}`
    );
  }

  const [selectedLayer, setSelectedLayer] = useState<string>(
    availableComponentLayers[0]
  );
  const [subcomponentsActivationContexts, setSubcomponentsActivationContexts] =
    useState<SubcomponentActivationContexts[] | null>(null);

  const [loading, setLoading] = useState(false);
  const [currentPage, setCurrentPage] = useState(0);

  // Configuration parameters
  const [importanceThreshold, setImportanceThreshold] = useState(0.01);
  const [maxExamplesPerSubcomponent, setMaxExamplesPerSubcomponent] =
    useState(100);
  const [nBatches, setNBatches] = useState(1);
  const [nTokensEitherSide, setNTokensEitherSide] = useState(10);
  const [batchSize, setBatchSize] = useState(32);

  const totalPages = subcomponentsActivationContexts?.length ?? 0;
  const currentItem = subcomponentsActivationContexts?.[currentPage] ?? null;

  const loadContexts = async () => {
    setLoading(true);
    try {
      console.log(`loading contexts for layer ${selectedLayer}`);
      const config: ActivationContextsConfig = {
        importance_threshold: importanceThreshold,
        max_examples_per_subcomponent: maxExamplesPerSubcomponent,
        n_batches: nBatches,
        batch_size: batchSize,
        n_tokens_either_side: nTokensEitherSide,
      };
      const data = await getLayerActivationContexts(selectedLayer, config);
      data.sort((a, b) => b.examples.length - a.examples.length);
      for (const d of data) {
        d.examples = d.examples.slice(0, 100);
      }
      setSubcomponentsActivationContexts(data);
      setCurrentPage(0);
    } catch (e: any) {
      if (e?.name !== "AbortError") {
        console.error(e);
      }
    } finally {
      setLoading(false);
    }
  };

  const previousPage = () => {
    if (currentPage > 0) setCurrentPage(currentPage - 1);
  };

  const nextPage = () => {
    if (currentPage < totalPages - 1) setCurrentPage(currentPage + 1);
  };

  return (
    <div className="tab-content">
      <div className="controls">
        <div className="control-row">
          <label htmlFor="layer-select">Layer:</label>
          <select
            id="layer-select"
            value={selectedLayer}
            onChange={(e) => {
              setSelectedLayer(e.target.value);
              loadContexts();
            }}
          >
            {availableComponentLayers.map((layer) => (
              <option key={layer} value={layer}>
                {layer}
              </option>
            ))}
          </select>
        </div>

        <div className="config-section">
          <h4>Configuration</h4>
          <div className="config-grid">
            <div className="config-item">
              <label htmlFor="importance-threshold">
                Importance Threshold:
              </label>
              <input
                id="importance-threshold"
                type="number"
                step="0.001"
                min="0"
                max="1"
                value={importanceThreshold}
                onChange={(e) =>
                  setImportanceThreshold(parseFloat(e.target.value))
                }
              />
            </div>

            <div className="config-item">
              <label htmlFor="max-examples">
                Max Examples per Subcomponent:
              </label>
              <input
                id="max-examples"
                type="number"
                step="1"
                min="1"
                value={maxExamplesPerSubcomponent}
                onChange={(e) =>
                  setMaxExamplesPerSubcomponent(parseInt(e.target.value))
                }
              />
            </div>

            <div className="config-item">
              <label htmlFor="n-steps">Number of Batches:</label>
              <input
                id="n-steps"
                type="number"
                step="1"
                min="1"
                value={nBatches}
                onChange={(e) => setNBatches(parseInt(e.target.value))}
              />
            </div>

            <div className="config-item">
              <label htmlFor="batch-size">Batch Size:</label>
              <input
                id="batch-size"
                type="number"
                step="1"
                min="1"
                value={batchSize}
                onChange={(e) => setBatchSize(parseInt(e.target.value))}
              />
            </div>

            <div className="config-item">
              <label htmlFor="n-tokens">Context Tokens Either Side:</label>
              <input
                id="n-tokens"
                type="number"
                step="1"
                min="0"
                value={nTokensEitherSide}
                onChange={(e) => setNTokensEitherSide(parseInt(e.target.value))}
              />
            </div>
          </div>
          <button
            className="load-button"
            onClick={loadContexts}
            disabled={loading}
          >
            {loading ? "Loading..." : "Load Contexts"}
          </button>
        </div>
      </div>

      {loading && <div className="loading">Loading...</div>}

      {currentItem && (
        <>
          <div className="pagination-controls">
            <button onClick={previousPage} disabled={currentPage === 0}>
              &lt;
            </button>
            <input
              type="number"
              min="0"
              max={totalPages - 1}
              value={currentPage}
              onChange={(e) => setCurrentPage(parseInt(e.target.value))}
              className="page-input"
            />
            <span>of {totalPages - 1}</span>
            <button
              onClick={nextPage}
              disabled={currentPage === totalPages - 1}
            >
              &gt;
            </button>
          </div>

          <div className="subcomponent-section-header">
            <h4>Subcomponent {currentItem.subcomponent_idx}</h4>

            {currentItem.token_densities &&
              currentItem.token_densities.length > 0 && (
                <div className="token-densities">
                  <h5>Token Activation Densities (top 20)</h5>
                  <div className="densities-grid">
                    {currentItem.token_densities
                      .slice(0, 20)
                      .map(({ token, density }) => (
                        <div className="density-item" key={token}>
                          <span className="token">{token}</span>
                          <div className="density-bar-container">
                            <div
                              className="density-bar"
                              style={{ width: `${density * 100}%` }}
                            ></div>
                          </div>
                          <span className="density-value">
                            {(density * 100).toFixed(1)}%
                          </span>
                        </div>
                      ))}
                  </div>
                </div>
              )}

            <div className="subcomponent-section">
              {currentItem.examples.map((example, idx) => (
                <ActivationContext key={idx} example={example} />
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
};
