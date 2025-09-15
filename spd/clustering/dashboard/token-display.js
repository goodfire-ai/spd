// Token display utilities

function createTokenVisualization(tokens, activations, maxPosition) {
    const tokenContainer = document.createElement('div');
    tokenContainer.className = 'token-container';
    
    tokens.forEach((token, idx) => {
        const span = document.createElement('span');
        span.className = 'token';
        
        // Handle subword tokens
        if (token.startsWith('##')) {
            span.textContent = token.substring(2);
        } else {
            span.textContent = token;
        }
        
        // Color based on activation
        const activation = activations[idx];
        const normalizedAct = Math.min(Math.max(activation, 0), 1);
        const intensity = Math.floor((1 - normalizedAct) * 255);
        span.style.backgroundColor = `rgb(255, ${intensity}, ${intensity})`;
        
        // Mark max position
        if (idx === maxPosition) {
            span.style.border = '2px solid blue';
            span.style.fontWeight = 'bold';
        }
        
        // Add tooltip
        span.title = `${token}: ${activation.toFixed(6)}`;
        
        tokenContainer.appendChild(span);
        
        // Add space after token (unless it's a subword)
        if (!tokens[idx + 1]?.startsWith('##')) {
            tokenContainer.appendChild(document.createTextNode(' '));
        }
    });
    
    return tokenContainer;
}

function createTokenVisualizationWithTooltip(tokens, activations, maxPosition) {
    const tokenContainer = document.createElement('div');
    tokenContainer.className = 'token-container';
    tokenContainer.style.position = 'relative';
    
    tokens.forEach((token, idx) => {
        const span = document.createElement('span');
        span.className = 'token';
        span.style.position = 'relative';
        
        // Handle subword tokens
        if (token.startsWith('##')) {
            span.textContent = token.substring(2);
        } else {
            span.textContent = token;
        }
        
        // Color based on activation
        const activation = activations[idx];
        const normalizedAct = Math.min(Math.max(activation, 0), 1);
        const intensity = Math.floor((1 - normalizedAct) * 255);
        span.style.backgroundColor = `rgb(255, ${intensity}, ${intensity})`;
        
        // Mark max position
        if (idx === maxPosition) {
            span.style.border = '2px solid blue';
            span.style.fontWeight = 'bold';
        }
        
        // Create tooltip div
        const tooltip = document.createElement('div');
        tooltip.className = 'token-tooltip';
        tooltip.textContent = `${token}: ${activation.toFixed(6)}`;
        tooltip.style.display = 'none';
        span.appendChild(tooltip);
        
        // Show/hide tooltip on hover
        span.addEventListener('mouseenter', () => {
            tooltip.style.display = 'block';
        });
        span.addEventListener('mouseleave', () => {
            tooltip.style.display = 'none';
        });
        
        tokenContainer.appendChild(span);
        
        // Add space after token (unless it's a subword)
        if (!tokens[idx + 1]?.startsWith('##')) {
            tokenContainer.appendChild(document.createTextNode(' '));
        }
    });
    
    return tokenContainer;
}