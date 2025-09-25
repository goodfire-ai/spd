function hslToHex(h, s, l) {
	s /= 100; l /= 100;
	const c = (1 - Math.abs(2 * l - 1)) * s, x = c * (1 - Math.abs((h / 60) % 2 - 1)), m = l - c / 2;
	let r, g, b;
	if (h < 60) [r, g, b] = [c, x, 0];
	else if (h < 120) [r, g, b] = [x, c, 0];
	else if (h < 180) [r, g, b] = [0, c, x];
	else if (h < 240) [r, g, b] = [0, x, c];
	else if (h < 300) [r, g, b] = [x, 0, c];
	else[r, g, b] = [c, 0, x];
	const toHex = v => { const h = Math.round((v + m) * 255).toString(16); return h.length === 1 ? '0' + h : h };
	return '#' + toHex(r) + toHex(g) + toHex(b);
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
