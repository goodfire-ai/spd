
// histogram.js - Histogram utility functions
// origin: https://github.com/mivanit/js-dev-toolkit
// license: GPLv3

/**
 * Calculate bin centers from histogram bin edges
 * @param {Array} binEdges - Array of bin edges
 * @returns {Array} Array of bin centers
 */
function calculateBinCenters(binEdges) {
    const binCenters = [];
    for (let i = 0; i < binEdges.length - 1; i++) {
        binCenters.push((binEdges[i] + binEdges[i + 1]) / 2);
    }
    return binCenters;
}

/**
 * Calculate mean from histogram data
 * @param {Object} histData - Object with counts and edges (or bin_counts and bin_edges for backwards compatibility)
 * @returns {number} Weighted mean
 */
function calculateHistogramMean(histData) {
    // Support both old and new property names
    const counts = histData.counts || histData.bin_counts;
    const edges = histData.edges || histData.bin_edges;

    let sum = 0;
    let count = 0;

    for (let i = 0; i < counts.length; i++) {
        const binCenter = (edges[i] + edges[i + 1]) / 2;
        sum += binCenter * counts[i];
        count += counts[i];
    }

    return count > 0 ? sum / count : 0;
}

/**
 * Calculate median from histogram data (approximate)
 * @param {Object} histData - Object with counts and edges (or bin_counts and bin_edges for backwards compatibility)
 * @returns {number} Approximate median
 */
function calculateHistogramMedian(histData) {
    // Support both old and new property names
    const counts = histData.counts || histData.bin_counts;
    const edges = histData.edges || histData.bin_edges;

    const totalCount = counts.reduce((a, b) => a + b, 0);
    const halfCount = totalCount / 2;

    let cumulativeCount = 0;
    for (let i = 0; i < counts.length; i++) {
        cumulativeCount += counts[i];
        if (cumulativeCount >= halfCount) {
            return (edges[i] + edges[i + 1]) / 2;
        }
    }

    return 0;
}
