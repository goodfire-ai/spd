/**
 * Generate an SVG sparkline chart from data
 *
 * Creates a small inline chart suitable for embedding in tables, dashboards, or text.
 * The sparkline automatically scales to fit the data range and supports various
 * visual customizations including gradients, markers, and optional axes.
 *
 * # Parameters:
 *  - `values : number[]`
 *     Array of values to plot. If yvalues is null, these are y-values with auto x-values (0..n-1).
 *     If yvalues is provided, these are x-values.
 *  - `yvalues : number[]|null`
 *     Array of y-values (defaults to null). When provided, values becomes x-values.
 *  - `options : object`
 *     Configuration object with the following properties:
 *     - `width : number` - SVG width in pixels (defaults to `120`)
 *     - `height : number` - SVG height in pixels (defaults to `40`)
 *     - `color : string` - Color for line, markers, and fill (defaults to `"#4169E1"`)
 *     - `shading : boolean|number` - Fill under the line:
 *       - `false` for no shading
 *       - `true` for solid fill
 *       - `0.0-1.0` for gradient with specified opacity (defaults to `0.3`)
 *     - `lineWidth : number` - Stroke width of the line (defaults to `2`)
 *     - `markers : string` - Data point markers:
 *       - `""` for no markers
 *       - `"."` for small dots
 *       - `"o"` for large dots (defaults to `"."`)
 *     - `margin : number` - Base margin around the chart (defaults to `5`)
 *     - `xAxis : object` - X-axis configuration:
 *       - `line : boolean` - Draw axis line (defaults to `false`)
 *       - `ticks : boolean` - Show tick labels (defaults to `false`)
 *     - `yAxis : object` - Y-axis configuration:
 *       - `line : boolean` - Draw axis line (defaults to `false`) 
 *       - `ticks : boolean` - Show tick labels (defaults to `false`)
 * 
 * # Returns:
 *  - `string`
 *     Complete SVG element as an HTML string
 * 
 * # Usage:
 *
 * ```javascript
 * // Basic sparkline with auto x-values (0, 1, 2, 3, 4) - matplotlib style
 * document.getElementById('chart').innerHTML = sparkline([1, 5, 2, 8, 3]);
 *
 * // Sparkline with custom x and y values - matplotlib style
 * document.getElementById('chart').innerHTML = sparkline([0, 2, 4, 6, 8], [1, 5, 2, 8, 3]);
 *
 * // Customized sparkline with axes
 * document.getElementById('chart').innerHTML = sparkline([10, 25, 15, 30, 20], null, {
 *     width: 200,
 *     height: 60,
 *     color: '#22c55e',
 *     shading: 0.5,
 *     markers: 'o',
 *     xAxis: { line: true, ticks: true },
 *     yAxis: { line: true, ticks: true }
 * });
 * ```
 */
function sparkline(values, yvalues = null, options = {}) {
	// Validate inputs
	if (!Array.isArray(values)) {
		throw new Error(`First parameter must be an array, got: ${typeof values}`);
	}

	if (values.length === 0) {
		throw new Error('Values array cannot be empty');
	}

	// Default options
	const opts = {
		width: 120,
		height: 40,
		color: '#4169E1',
		shading: 0.3,
		lineWidth: 2,
		markers: '.',
		margin: 5,
		xAxis: { line: false, ticks: false },
		yAxis: { line: false, ticks: false },
		...options
	};

	// Handle matplotlib-style parameter interpretation
	let xvalues, yvals;

	if (yvalues === null) {
		// values is y-values, generate x-values as 0..n-1
		yvals = values;
		xvalues = Array.from({ length: values.length }, (_, i) => i);
	} else {
		// values is x-values, yvalues is y-values
		if (!Array.isArray(yvalues)) {
			throw new Error(`Second parameter must be an array or null, got: ${typeof yvalues}`);
		}
		xvalues = values;
		yvals = yvalues;
	}

	// Validate that x and y arrays have same length
	if (xvalues.length !== yvals.length) {
		throw new Error(`x-values length (${xvalues.length}) must match y-values length (${yvals ? yvals.length : 'undefined'})`);
	}

	// Calculate margins based on what's shown
	const needsLeftMargin = opts.yAxis.ticks;
	const needsBottomMargin = opts.xAxis.ticks;
	const leftMargin = needsLeftMargin ? opts.margin + 20 : opts.margin;
	const rightMargin = opts.margin;
	const topMargin = opts.margin;
	const bottomMargin = needsBottomMargin ? opts.margin + 15 : opts.margin;

	const ymin = Math.min(...yvals);
	const ymax = Math.max(...yvals);
	const yrange = ymax - ymin || 1;

	const xmin = Math.min(...xvalues);
	const xmax = Math.max(...xvalues);
	const xrange = xmax - xmin || 1;

	const chartWidth = opts.width - leftMargin - rightMargin;
	const chartHeight = opts.height - topMargin - bottomMargin;

	// Build path
	let path = '';
	let dots = '';
	const points = [];

	yvals.forEach((yval, i) => {
		const xval = xvalues[i];
		const x = leftMargin + ((xval - xmin) / xrange) * chartWidth;
		const y = topMargin + (1 - (yval - ymin) / yrange) * chartHeight;
		points.push({ x, y, xval, yval });

		path += `${i === 0 ? 'M' : 'L'} ${x} ${y} `;

		// Add markers
		if (opts.markers === '.') {
			dots += `<circle cx="${x}" cy="${y}" r="2" fill="${opts.color}"/>`;
		} else if (opts.markers === 'o') {
			dots += `<circle cx="${x}" cy="${y}" r="3.5" fill="${opts.color}"/>`;
		}
	});

	// Build SVG
	let svg = `<svg width="${opts.width}" height="${opts.height}">`;

	// Add shading if requested
	if (opts.shading !== false) {
		if (opts.shading === true) {
			// Solid fill
			for (let i = 0; i < points.length - 1; i++) {
				const p1 = points[i];
				const p2 = points[i + 1];
				const segmentPath = `M ${p1.x} ${p1.y} L ${p2.x} ${p2.y} L ${p2.x} ${opts.height - bottomMargin} L ${p1.x} ${opts.height - bottomMargin} Z`;
				svg += `<path d="${segmentPath}" fill="${opts.color}"/>`;
			}
		} else {
			// Gradient fill
			const gradId = `grad${Date.now()}${Math.random()}`;
			const opacity = parseFloat(opts.shading);
			svg += `
            <defs>
                <linearGradient id="${gradId}" x1="0%" y1="0%" x2="0%" y2="100%">
                    <stop offset="0%" style="stop-color:${opts.color};stop-opacity:${opacity}"/>
                    <stop offset="100%" style="stop-color:${opts.color};stop-opacity:0"/>
                </linearGradient>
            </defs>`;

			for (let i = 0; i < points.length - 1; i++) {
				const p1 = points[i];
				const p2 = points[i + 1];
				const segmentPath = `M ${p1.x} ${p1.y} L ${p2.x} ${p2.y} L ${p2.x} ${opts.height - bottomMargin} L ${p1.x} ${opts.height - bottomMargin} Z`;
				svg += `<path d="${segmentPath}" fill="url(#${gradId})"/>`;
			}
		}
	}

	// Add y-axis
	if (opts.yAxis.line) {
		svg += `<line x1="${leftMargin}" y1="${topMargin}" x2="${leftMargin}" y2="${opts.height - bottomMargin}" 
                      stroke="#ccc" stroke-width="1"/>`;
	}
	if (opts.yAxis.ticks) {
		svg += `<text x="${leftMargin - 3}" y="${topMargin + 3}" font-size="9" fill="#666" text-anchor="end">${ymax}</text>`;
		svg += `<text x="${leftMargin - 3}" y="${opts.height - bottomMargin + 3}" font-size="9" fill="#666" text-anchor="end">${ymin}</text>`;
	}

	// Add x-axis
	if (opts.xAxis.line) {
		svg += `<line x1="${leftMargin}" y1="${opts.height - bottomMargin}" x2="${opts.width - rightMargin}" y2="${opts.height - bottomMargin}" 
                      stroke="#ccc" stroke-width="1"/>`;
	}
	if (opts.xAxis.ticks && yvals.length > 0) {
		svg += `<text x="${leftMargin}" y="${opts.height - bottomMargin + 12}" font-size="9" fill="#666" text-anchor="middle">${xmin}</text>`;
		svg += `<text x="${opts.width - rightMargin}" y="${opts.height - bottomMargin + 12}" font-size="9" fill="#666" text-anchor="middle">${xmax}</text>`;
	}

	// Add the line
	svg += `<path d="${path}" stroke="${opts.color}" stroke-width="${opts.lineWidth}" fill="none" stroke-linecap="round" stroke-linejoin="round"/>`;

	// Add dots
	svg += dots;

	svg += '</svg>';

	return svg;
}