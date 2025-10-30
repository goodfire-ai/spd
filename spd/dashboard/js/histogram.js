
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
 * @param {Object} histData - Object with bin_counts and bin_edges
 * @returns {number} Weighted mean
 */
function calculateHistogramMean(histData) {
    const { bin_counts, bin_edges } = histData;
    let sum = 0;
    let count = 0;

    for (let i = 0; i < bin_counts.length; i++) {
        const binCenter = (bin_edges[i] + bin_edges[i + 1]) / 2;
        sum += binCenter * bin_counts[i];
        count += bin_counts[i];
    }

    return count > 0 ? sum / count : 0;
}

/**
 * Calculate median from histogram data (approximate)
 * @param {Object} histData - Object with bin_counts and bin_edges
 * @returns {number} Approximate median
 */
function calculateHistogramMedian(histData) {
    const { bin_counts, bin_edges } = histData;
    const totalCount = bin_counts.reduce((a, b) => a + b, 0);
    const halfCount = totalCount / 2;

    let cumulativeCount = 0;
    for (let i = 0; i < bin_counts.length; i++) {
        cumulativeCount += bin_counts[i];
        if (cumulativeCount >= halfCount) {
            return (bin_edges[i] + bin_edges[i + 1]) / 2;
        }
    }

    return 0;
}
