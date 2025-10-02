// sparklines.js - Generate SVG sparklines and sparkbars from data arrays
// origin: https://github.com/mivanit/js-dev-toolkit
// license: GPLv3

const _PLOT_OPTS_DEFAULT = {
	// Chart dimensions and basic styling
	width: 160,  // number: SVG width in pixels
	height: 80,  // number: SVG height in pixels
	color: '#4169E1',  // string: CSS color for line/bars/markers
	shading: 0.3,  // boolean|number: false=none, true=solid fill, 0-1=gradient opacity
	lineWidth: 2,  // number: stroke width for line charts in pixels
	markers: null,  // null|number: null=no markers, positive number=circle radius
	margin: 5,  // number: base margin around chart content in pixels
	style: 'line',  // string: 'line' or 'bar' (internal, set by sparkline/sparkbars)
	barWidthRatio: 1,  // number: 0-1 ratio, 1=touching bars, <1=gaps between bars
	logScale: false,  // boolean: use logarithmic scale for y-axis

	// X-axis configuration
	xAxis: {
		line: false,  // boolean: show axis line
		ticks: false,  // boolean: show tick labels (min/max values)
		text_offset: 10,  // number: pixels below axis for tick labels
		text_y_position: 0,  // number: additional Y offset for labels (positive = down)
		label_margin: 5,  // number: extra margin in pixels when ticks shown
		limits_length: 2  // number: expected array length for xlims (internal)
	},

	// Y-axis configuration
	yAxis: {
		line: false,  // boolean: show axis line
		ticks: false,  // boolean: show tick labels (min/max values)
		text_offset: 2,  // number: pixels left of axis for tick labels
		text_y_offset: 0,  // number: pixels down from top edge for max label to prevent cutoff
		label_margin: 10,  // number: extra margin in pixels when ticks shown
		limits_length: 2  // number: expected array length for ylims (internal)
	},

	xlims: null,  // null|array: [min, max] numbers, null values, or null for auto from data
	ylims: null,  // null|array: [min, max] numbers, null values, or null for auto from data

	// Bar chart specific settings
	bar: {
		opacity: 0.8,  // number: 0-1 opacity of bar rectangles
		y_axis_shift_ratio: 0.6,  // number: Y-axis left shift as ratio of bar width
		y_axis_reduction_factor: 1  // number: width reduction factor for Y-axis labels
	},

	// Axis styling (applies to both X and Y)
	axis_style: {
		color: '#ccc',  // string: CSS color for axis lines
		width: 1,  // number: axis line stroke width in pixels
		font_size: 10,  // number: tick label font size in pixels
		text_color: '#666'  // string: CSS color for tick labels
	},

	// Gradient fill configuration (when shading is 0-1)
	gradient: {
		start_offset: '0%',  // string: SVG gradient start position
		end_offset: '100%',  // string: SVG gradient end position
		end_opacity: 0  // number: 0-1 opacity at gradient end
	},

	min_range: 1  // number: minimum range when max equals min in data
};
/**
 * Deep merge options with defaults
 */
function merge_options(options = {}) {
	return {
		..._PLOT_OPTS_DEFAULT,
		...options,
		xAxis: { ..._PLOT_OPTS_DEFAULT.xAxis, ...(options.xAxis || {}) },
		yAxis: { ..._PLOT_OPTS_DEFAULT.yAxis, ...(options.yAxis || {}) },
		bar: { ..._PLOT_OPTS_DEFAULT.bar, ...(options.bar || {}) },
		axis_style: { ..._PLOT_OPTS_DEFAULT.axis_style, ...(options.axis_style || {}) },
		gradient: { ..._PLOT_OPTS_DEFAULT.gradient, ...(options.gradient || {}) }
	};
}

/**
 * Process and validate input values
 * Returns { xvalues, yvalues } arrays
 */
function process_values(values, yvalues = null) {
	// Validate inputs
	if (!Array.isArray(values)) {
		throw new Error(`First parameter must be an array, got: ${typeof values}`);
	}

	if (values.length === 0) {
		throw new Error('Values array cannot be empty');
	}

	let xvals, yvals;

	if (yvalues === null) {
		// values is y-values, generate x-values as 0..n-1
		yvals = values;
		xvals = Array.from({ length: values.length }, (_, i) => i);
	} else {
		// values is x-values, yvalues is y-values
		if (!Array.isArray(yvalues)) {
			throw new Error(`Second parameter must be an array or null, got: ${typeof yvalues}`);
		}
		xvals = values;
		yvals = yvalues;
	}

	// Validate that x and y arrays have same length
	if (xvals.length !== yvals.length) {
		throw new Error(`x-values length (${xvals.length}) must match y-values length (${yvals.length})`);
	}

	return { xvalues: xvals, yvalues: yvals };
}

function plot(values, yvalues = null, options = {}) {
	const opts = merge_options(options);
	const { xvalues, yvalues: yvals } = process_values(values, yvalues);

	// Calculate margins based on what's shown
	const needsLeftMargin = opts.yAxis.ticks;
	const needsBottomMargin = opts.xAxis.ticks;
	// For bar charts with Y-axis, we need extra left margin since axis is shifted left
	const barChartExtraMargin = opts.style === 'bar' && (opts.yAxis.line || opts.yAxis.ticks) ?
		(opts.width - 2 * opts.margin) / yvals.length * opts.bar.y_axis_shift_ratio : 0;
	const leftMargin = (needsLeftMargin ? opts.margin + opts.yAxis.label_margin : opts.margin) + barChartExtraMargin;
	const bottomMargin = needsBottomMargin ? opts.margin + opts.xAxis.label_margin : opts.margin;

	// Calculate axis limits - use custom limits if provided, otherwise data range
	const dataYmin = Math.min(...yvals);
	const dataYmax = Math.max(...yvals);
	let ymin = opts.ylims && opts.ylims[0] !== null ? opts.ylims[0] : dataYmin;
	let ymax = opts.ylims && opts.ylims[1] !== null ? opts.ylims[1] : dataYmax;

	// For log scale, assert all values >= 0
	if (opts.logScale) {
		if (yvals.some(v => v < 0) || ymin < 0 || ymax < 0) {
			console.error('y-values:', yvals);
			console.error('ymin:', ymin, 'ymax:', ymax);
			throw new Error('Log scale requires all values >= 0');
		}
	}

	// Transform to log space if needed, handling zeros
	// For log scale, we use log10(y) for y > 0, and leave 0 as special case (will skip bars)
	const transformY = opts.logScale
		? (y) => y > 0 ? Math.log10(y) : 0
		: (y) => y;

	// For axis limits, only use positive values in log scale
	const positiveYvals = opts.logScale ? yvals.filter(v => v > 0) : yvals;
	const yminForScale = opts.logScale && positiveYvals.length > 0 ? Math.min(...positiveYvals) : ymin;
	const ymaxForScale = opts.logScale && positiveYvals.length > 0 ? Math.max(...positiveYvals) : ymax;

	const yminTransformed = transformY(yminForScale);
	const ymaxTransformed = transformY(ymaxForScale);
	const yrange = ymaxTransformed - yminTransformed || opts.min_range;

	const dataXmin = Math.min(...xvalues);
	const dataXmax = Math.max(...xvalues);
	const xmin = opts.xlims && opts.xlims[0] !== null ? opts.xlims[0] : dataXmin;
	const xmax = opts.xlims && opts.xlims[1] !== null ? opts.xlims[1] : dataXmax;
	const xrange = xmax - xmin || opts.min_range;

	const chartWidth = opts.width - leftMargin - opts.margin;
	const chartHeight = opts.height - opts.margin - bottomMargin;

	// Build SVG

	// this setup is messy, but syntax highlighting gets confused with template strings??
	let svg = "<svg " + `width="${opts.width}" height="${opts.height}"` + ">";

	if (opts.style === 'bar') {
		// Bar chart rendering
		const baseY = opts.height - bottomMargin;
		// For bars, we position them at centers, with the first at 0 and last at (n-1)/n of the width
		// This matches how the X-axis is drawn (ending at center of last bar)
		const adjustedChartWidth = chartWidth - (chartWidth / yvals.length / 2);

		// Calculate bar spacing based on actual x-values
		let barSpacing;
		if (yvals.length === 1) {
			// Single bar: use full available width
			barSpacing = adjustedChartWidth;
		} else {
			// Multiple bars: find minimum spacing between consecutive x-values
			const sortedIndices = [...Array(xvalues.length).keys()].sort((a, b) => xvalues[a] - xvalues[b]);
			let minSpacing = Infinity;
			for (let j = 1; j < sortedIndices.length; j++) {
				const prevX = xvalues[sortedIndices[j - 1]];
				const currX = xvalues[sortedIndices[j]];
				const spacing = ((currX - prevX) / xrange) * adjustedChartWidth;
				if (spacing < minSpacing) minSpacing = spacing;
			}
			// If all x-values are the same, fall back to dividing by count
			barSpacing = minSpacing === Infinity ? adjustedChartWidth / yvals.length : minSpacing;
		}

		// Bar width is the spacing times the ratio (1 = touching, <1 = gaps)
		const actualBarWidth = barSpacing * opts.barWidthRatio;

		yvals.forEach((yval, i) => {
			// Skip bars with zero values in log scale
			if (opts.logScale && yval === 0) {
				return;
			}

			const xval = xvalues[i];
			// Position at bar center
			const x = leftMargin + ((xval - xmin) / xrange) * adjustedChartWidth;
			const yvalTransformed = transformY(yval);
			const barHeight = Math.abs((yvalTransformed - yminTransformed) / yrange) * chartHeight;
			const barY = baseY - barHeight;

			// Calculate bar x position (offset by half width to center)
			const barX = x - actualBarWidth / 2;

			svg += `<rect x="${barX}" y="${barY}" width="${actualBarWidth}" height="${barHeight}" fill="${opts.color}" opacity="${opts.bar.opacity}"/>`;
		});
	} else {
		// Line chart rendering (existing code)
		let path = '';
		let dots = '';
		const points = [];

		yvals.forEach((yval, i) => {
			const xval = xvalues[i];
			const x = leftMargin + ((xval - xmin) / xrange) * chartWidth;

			// Skip zero values in log scale for line charts (break the line)
			if (opts.logScale && yval === 0) {
				// Break the line by moving to next point
				return;
			}

			const yvalTransformed = transformY(yval);
			const y = opts.margin + (1 - (yvalTransformed - yminTransformed) / yrange) * chartHeight;
			points.push({ x, y, xval, yval });

			// Start new path segment if previous was skipped
			const moveCmd = (i === 0 || (opts.logScale && i > 0 && yvals[i-1] === 0)) ? 'M' : 'L';
			path += `${moveCmd} ${x} ${y} `;

			// Add markers if specified
			if (opts.markers !== null && opts.markers > 0) {
				dots += `<circle cx="${x}" cy="${y}" r="${opts.markers}" fill="${opts.color}"/>`;
			}
		});

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
	                <linearGradient id="${gradId}" x1="${opts.gradient.start_offset}" y1="${opts.gradient.start_offset}" x2="${opts.gradient.start_offset}" y2="${opts.gradient.end_offset}">
	                    <stop offset="${opts.gradient.start_offset}" style="stop-color:${opts.color};stop-opacity:${opacity}"/>
	                    <stop offset="${opts.gradient.end_offset}" style="stop-color:${opts.color};stop-opacity:${opts.gradient.end_opacity}"/>
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

		// Add the line
		svg += `<path d="${path}" stroke="${opts.color}" stroke-width="${opts.lineWidth}" fill="none" stroke-linecap="round" stroke-linejoin="round"/>`;

		// Add dots
		svg += dots;
	}

	// Format y-axis tick labels (use scientific notation for log scale)
	const formatYTick = (value) => {
		if (!opts.logScale) {
			return value;
		}
		// For log scale, use scientific notation unless value is 0
		if (value === 0) {
			return '0';
		}
		// Convert to scientific notation with 1 decimal place
		return value.toExponential(1);
	};

	// Add y-axis
	if (opts.yAxis.line || opts.yAxis.ticks) {
		// For bar charts, shift Y-axis left by 0.75 bar widths to avoid overlap
		const barShift = opts.style === 'bar' ? (chartWidth / yvals.length * opts.bar.y_axis_shift_ratio) : 0;
		const yAxisX = leftMargin - barShift;

		if (opts.yAxis.line) {
			svg += `<line x1="${yAxisX}" y1="${opts.margin}" x2="${yAxisX}" y2="${opts.height - bottomMargin}"
                      stroke="${opts.axis_style.color}" stroke-width="${opts.axis_style.width}"/>`;
		}
		if (opts.yAxis.ticks) {
			svg += `<text x="${yAxisX - opts.yAxis.text_offset}" y="${opts.margin + opts.yAxis.text_y_offset + opts.axis_style.font_size}" font-size="${opts.axis_style.font_size}" fill="${opts.axis_style.text_color}" text-anchor="end">${formatYTick(ymaxForScale)}</text>`;
			// For log scale with ymin=0, always show "0" for the lower bound
			const lowerLabel = (opts.logScale && ymin === 0) ? '0' : formatYTick(yminForScale);
			svg += `<text x="${yAxisX - opts.yAxis.text_offset}" y="${opts.height - bottomMargin + opts.yAxis.text_offset}" font-size="${opts.axis_style.font_size}" fill="${opts.axis_style.text_color}" text-anchor="end">${lowerLabel}</text>`;
		}
	}

	// Add x-axis
	if (opts.xAxis.line) {
		// For bar charts, end X-axis at center of last bar, not full width
		const xAxisEnd = opts.style === 'bar' ?
			leftMargin + chartWidth - (chartWidth / yvals.length / 2) :
			opts.width - opts.margin;
		svg += `<line x1="${leftMargin}" y1="${opts.height - bottomMargin}" x2="${xAxisEnd}" y2="${opts.height - bottomMargin}"
                      stroke="${opts.axis_style.color}" stroke-width="${opts.axis_style.width}"/>`;
	}
	if (opts.xAxis.ticks && yvals.length > 0) {
		// For bar charts, position end tick at center of last bar
		const xTickEnd = opts.style === 'bar' ?
			leftMargin + chartWidth - (chartWidth / yvals.length / 2) :
			opts.width - opts.margin;
		const xLabelY = opts.height - bottomMargin + opts.xAxis.text_offset + opts.xAxis.text_y_position;
		svg += `<text x="${leftMargin}" y="${xLabelY}" font-size="${opts.axis_style.font_size}" fill="${opts.axis_style.text_color}" text-anchor="middle">${xmin}</text>`;
		svg += `<text x="${xTickEnd}" y="${xLabelY}" font-size="${opts.axis_style.font_size}" fill="${opts.axis_style.text_color}" text-anchor="middle">${xmax}</text>`;
	}

	svg += '</svg>';

	return svg;
}

/**
 * Generate an SVG sparkline (line chart) from data
 * @param {number[]} values - Y-values or X-values if yvalues provided
 * @param {number[]|null} yvalues - Y-values (when values becomes x-values)
 * @param {object} options - Chart options (xlims/ylims can be [min,max] with null for auto)
 * @returns {string} SVG element as HTML string
 */
function sparkline(values, yvalues = null, options = {}) {
	return plot(values, yvalues, { ...options, style: 'line' });
}

/**
 * Generate an SVG sparkbar chart (bar chart) from data
 * @param {number[]} values - Y-values or X-values if yvalues provided
 * @param {number[]|null} yvalues - Y-values (when values becomes x-values)
 * @param {object} options - Chart options (xlims/ylims can be [min,max] with null for auto)
 * @returns {string} SVG element as HTML string
 */
function sparkbars(values, yvalues = null, options = {}) {
	return plot(values, yvalues, { ...options, style: 'bar' });
}