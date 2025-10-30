// ColorUtil.js - Color manipulation utilities
// origin: https://github.com/mivanit/js-dev-toolkit
// license: GPLv3

function hslToHex(h, s, l) {
	s /= 100;
	l /= 100;
	const c = (1 - Math.abs(2 * l - 1)) * s,
		x = c * (1 - Math.abs(((h / 60) % 2) - 1)),
		m = l - c / 2;
	let r, g, b;
	if (h < 60) [r, g, b] = [c, x, 0];
	else if (h < 120) [r, g, b] = [x, c, 0];
	else if (h < 180) [r, g, b] = [0, c, x];
	else if (h < 240) [r, g, b] = [0, x, c];
	else if (h < 300) [r, g, b] = [x, 0, c];
	else [r, g, b] = [c, 0, x];
	const toHex = (v) => {
		const h = Math.round((v + m) * 255).toString(16);
		return h.length === 1 ? "0" + h : h;
	};
	return "#" + toHex(r) + toHex(g) + toHex(b);
}

function generateDistinctColors(n) {
	const out = [];
	for (let i = 0; i < n; i++) {
		const h = Math.floor(Math.random() * 360),
			s = Math.floor(50 + Math.random() * 50),
			l = Math.floor(40 + Math.random() * 20);
		out.push(hslToHex(h, s, l));
	}
	return out;
}

const COLORMAPS = {
	blues: {
		r: [247, 198, 107, 33],
		g: [251, 219, 174, 113],
		b: [255, 239, 214, 181],
	},
	reds: {
		r: [255, 252, 203, 103],
		g: [245, 174, 24, 0],
		b: [240, 145, 29, 13],
	},
	viridis: {
		r: [68, 59, 33, 253],
		g: [1, 82, 144, 231],
		b: [84, 139, 140, 37],
	},
	plasma: {
		r: [13, 126, 204, 240],
		g: [8, 3, 187, 249],
		b: [135, 192, 10, 33],
	},
};

/**
 * Get color for a value based on colormap and range
 * @param {number} value - The value to map to a color
 * @param {number[]|{min: number, max: number}} range - Either [min, max] array or {min, max} object
 * @param {string} colormap - Name of colormap (blues, reds, viridis, plasma)
 * @returns {string} RGB color string
 */
function getColorForValue(value, range, colormap = "blues") {
	let min, max;

	if (Array.isArray(range)) {
		[min, max] = range;
	} else if (typeof range === "object" && range !== null) {
		min = range.min;
		max = range.max;
	} else {
		// Legacy: single number treated as max with min=0
		min = 0;
		max = range;
	}

	if (max === min) return "#f5f5f5";

	const intensity = Math.max(0, Math.min(1, (value - min) / (max - min)));
	const colors = COLORMAPS[colormap] || COLORMAPS.blues;
	const pos = intensity * (colors.r.length - 1);
	const i = Math.floor(pos);
	const f = pos - i;

	if (i >= colors.r.length - 1) {
		return `rgb(${colors.r[colors.r.length - 1]}, ${colors.g[colors.g.length - 1]}, ${colors.b[colors.b.length - 1]})`;
	}

	const r = Math.round(colors.r[i] + f * (colors.r[i + 1] - colors.r[i]));
	const g = Math.round(colors.g[i] + f * (colors.g[i + 1] - colors.g[i]));
	const b = Math.round(colors.b[i] + f * (colors.b[i + 1] - colors.b[i]));

	return `rgb(${r}, ${g}, ${b})`;
}

/**
 * Map an array of values to colors
 * @param {number[]} values - Array of values to map
 * @param {string} colormap - Name of colormap
 * @returns {string[]} Array of RGB color strings
 */
function getColorsForValues(values, colormap = "blues") {
	if (!values || values.length === 0) return [];

	const min = Math.min(...values);
	const max = Math.max(...values);

	return values.map((v) => getColorForValue(v, { min, max }, colormap));
}
