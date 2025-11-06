/**
 * Utility for loading sample activations from ZANJ activation data.
 * Shared between component.html and trees.html.
 */

/**
 * Load sample activations for a specific component from the full activations data.
 *
 * @param {Object} fullData - The complete ZANJ data object
 * @param {string} componentLabel - Component label in format "module.name:index"
 * @param {number} maxSamples - Maximum number of samples to load (default: 10)
 * @returns {Promise<Array>} Array of {token_strs, activations} objects
 */
async function loadComponentSamples(fullData, componentLabel, maxSamples = 10) {
    try {
        // Parse component label
        const [moduleName, componentIndexStr] = componentLabel.split(':');
        const componentIndex = parseInt(componentIndexStr);

        // Load activations
        const activations = await fullData.activations;
        const moduleData = await activations.data[moduleName];
        const componentLabels = await activations.component_labels[moduleName];
        const tokenData = await activations.token_data;
        const tokens = await tokenData.tokens;

        // Find component position
        const componentPos = componentLabels.indexOf(componentLabel);
        if (componentPos === -1) {
            console.warn(`Component ${componentLabel} not found in module`);
            return [];
        }

        // Extract activations using flat indexing for NumPy arrays
        const actualData = moduleData.data || moduleData;
        const [nSeqs, nCtx, nComponents] = moduleData.shape;

        const samples = [];
        for (let seq = 0; seq < Math.min(nSeqs, maxSamples); seq++) {
            const seqActivations = [];
            for (let pos = 0; pos < nCtx; pos++) {
                // NumPy arrays are stored in row-major (C) order
                const flatIdx = seq * (nCtx * nComponents) + pos * nComponents + componentPos;
                seqActivations.push(actualData[flatIdx]);
            }
            samples.push({
                token_strs: tokens[seq],
                activations: seqActivations
            });
        }
        return samples;
    } catch (error) {
        console.error(`Error loading samples for ${componentLabel}:`, error);
        return [];
    }
}
