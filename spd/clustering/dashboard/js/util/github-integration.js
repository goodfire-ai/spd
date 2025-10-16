// github-integration.js - Utilities for GitHub issue creation and explanation tracking

/**
 * Generate GitHub issue URL for cluster explanation submission
 * @param {number} clusterId - Cluster ID
 * @param {object} clusterData - Cluster data object
 * @param {object} modelInfo - Model information
 * @returns {string} GitHub issue creation URL with pre-filled template
 */
function generateGitHubIssueURL(clusterId, clusterData, modelInfo) {
    if (!CONFIG.github.enabled) {
        console.warn('GitHub integration is disabled in config');
        return null;
    }

    const { owner, repo, labels, runMetadata } = CONFIG.github;

    // Generate issue title
    const title = `Cluster ${clusterId} Explanation`;

    // Generate issue body with template
    const body = generateIssueBody(clusterId, clusterData, modelInfo, runMetadata);

    // Construct GitHub issue URL with query parameters
    const baseURL = `https://github.com/${owner}/${repo}/issues/new`;
    const params = new URLSearchParams({
        title: title,
        body: body,
        labels: labels.join(',')
    });

    return `${baseURL}?${params.toString()}`;
}

/**
 * Generate the issue body template
 * @param {number} clusterId - Cluster ID
 * @param {object} clusterData - Cluster data
 * @param {object} modelInfo - Model information
 * @param {object} runMetadata - Run metadata from config
 * @returns {string} Issue body with YAML frontmatter
 */
function generateIssueBody(clusterId, clusterData, modelInfo, runMetadata) {
    // Extract cluster statistics
    const componentCount = clusterData.components.length;
    const sampleCount = clusterData.samples.length;

    // Get unique modules
    const modules = [...new Set(clusterData.components.map(c => c.module))];

    // Calculate activation statistics
    const allActivations = [];
    clusterData.samples.forEach(sample => {
        sample.activations.forEach(act => {
            if (act > 0) allActivations.push(act);
        });
    });

    const maxActivation = Math.max(...allActivations);
    const meanActivation = allActivations.reduce((a, b) => a + b, 0) / allActivations.length;

    // Get component indices grouped by module
    const componentsByModule = {};
    clusterData.components.forEach(comp => {
        if (!componentsByModule[comp.module]) {
            componentsByModule[comp.module] = [];
        }
        componentsByModule[comp.module].push(comp.index);
    });

    // Build metadata object
    const metadata = {
        cluster_id: clusterId,
        model: runMetadata.model,
        wandb_project: runMetadata.wandbProject,
        wandb_run: runMetadata.wandbRun,
        decomp_run: runMetadata.decompRun,
        iteration: runMetadata.iteration,
        num_components: componentCount,
        num_samples: sampleCount,
        max_activation: parseFloat(maxActivation.toFixed(4)),
        mean_activation: parseFloat(meanActivation.toFixed(4)),
        modules: modules,
        components_by_module: componentsByModule,
        dashboard_link: `${window.location.origin}${window.location.pathname.replace('cluster.html', 'cluster.html')}?id=${clusterId}`
    };

    // Generate issue body with YAML frontmatter
    const body = `${createYAMLFrontmatter(metadata)}

## Proposed Explanation

<!-- Please describe what you think this cluster represents -->

**Hypothesis:**


**Evidence:**


**Confidence Level:** (Low/Medium/High)


## Additional Notes

<!-- Any additional observations, questions, or context -->

`;

    return body;
}

/**
 * Fetch existing cluster explanation issues from GitHub
 * @returns {Promise<object>} Map of cluster IDs to issue metadata
 */
async function fetchGitHubExplanations() {
    if (!CONFIG.github.enabled) {
        throw new Error('GitHub integration is disabled in config');
    }

    const { owner, repo, labels } = CONFIG.github;
    const clusterExplanations = {};

    // Fetch all issues with the cluster-explanation label
    const issues = await fetchAllIssues(owner, repo, labels);

    // Parse issue titles to extract cluster IDs
    // Expected format: "Cluster {id} Explanation"
    const clusterPattern = /Cluster\s+(\d+)/i;

    issues.forEach(issue => {
        const match = issue.title.match(clusterPattern);
        if (match) {
            const clusterId = match[1];
            clusterExplanations[clusterId] = {
                issueNumber: issue.number,
                issueUrl: issue.html_url,
                author: issue.user.login
            };
        }
    });

    console.log(`Loaded ${Object.keys(clusterExplanations).length} cluster explanations from GitHub`);
    return clusterExplanations;
}

/**
 * Fetch all issues from a GitHub repository with pagination
 * @param {string} owner - Repository owner
 * @param {string} repo - Repository name
 * @param {Array<string>} labels - Labels to filter by
 * @returns {Promise<Array>} Array of issue objects
 */
async function fetchAllIssues(owner, repo, labels = []) {
    const allIssues = [];
    let page = 1;
    const perPage = 100;

    while (true) {
        const url = new URL(`https://api.github.com/repos/${owner}/${repo}/issues`);
        url.searchParams.set('state', 'all');
        url.searchParams.set('per_page', perPage.toString());
        url.searchParams.set('page', page.toString());
        if (labels.length > 0) {
            url.searchParams.set('labels', labels.join(','));
        }

        const response = await fetch(url.toString(), {
            headers: {
                'Accept': 'application/vnd.github.v3+json',
                // Note: For unauthenticated requests, GitHub allows 60 req/hour
                // For authenticated requests, include: 'Authorization': 'Bearer YOUR_TOKEN'
            }
        });

        if (!response.ok) {
            throw new Error(`GitHub API error: ${response.status} ${response.statusText}`);
        }

        const issues = await response.json();

        // Filter out pull requests (they appear in issues API but have pull_request field)
        const actualIssues = issues.filter(issue => !issue.pull_request);

        allIssues.push(...actualIssues);

        // Check if there are more pages
        if (issues.length < perPage) {
            break;
        }

        page++;
    }

    return allIssues;
}


/**
 * Create explanation badge HTML
 * @param {number} clusterId - Cluster ID
 * @param {object} explanations - Map of cluster IDs to explanation metadata
 * @returns {HTMLElement} Badge element
 */
function createExplanationBadge(clusterId, explanations) {
    const explanation = explanations[clusterId];
    const badge = document.createElement('span');
    badge.className = 'explanation-badge';
    badge.style.cssText = `
        display: inline-block;
        padding: 3px 8px;
        border-radius: 3px;
        font-size: 11px;
        font-weight: 600;
    `;

    if (explanation) {
        // Has explanation - show link to issue
        badge.style.backgroundColor = '#28a74522';
        badge.style.color = '#28a745';
        badge.style.border = '1px solid #28a745';

        const link = document.createElement('a');
        link.href = explanation.issueUrl;
        link.target = '_blank';
        link.style.cssText = 'color: #28a745; text-decoration: none;';
        link.textContent = `✓ #${explanation.issueNumber}`;
        link.title = `Explanation by ${explanation.author}`;
        badge.appendChild(link);
    } else {
        // No explanation yet
        badge.style.backgroundColor = '#99999922';
        badge.style.color = '#999';
        badge.style.border = '1px solid #999';
        badge.textContent = '—';
        badge.title = 'Not yet explained';
    }

    return badge;
}
