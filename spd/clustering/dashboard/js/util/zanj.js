// zanj.js
// Minimal ZANJ loader with transparent lazy loading.
// Supports refs to npy, json, and jsonl. Requires array.js (NDArray).

// Must match muutils.json_serialize.util.{_REF_KEY, _FORMAT_KEY}
// _FORMAT_KEY is defined in array.js
const _REF_KEY = "$ref";

function joinUrl(base, rel) {
	const b = base.replace(/\/+$/, "");
	const r = String(rel || "").replace(/^\/+/, "");
	return b + "/" + r;
}

class ZanjLoader {
	constructor(path, opts) {
		this.path = path;
		this.fetchInit = (opts && opts.fetchInit) || undefined;
		this._cache = new Map();
	}

	async read() {
		const url = joinUrl(this.path, "__zanj__.json");
		const res = await fetch(url, this.fetchInit);
		if (!res.ok) throw new Error("Failed to fetch " + url + ": " + res.status);
		const root = await res.json();
		return this._makeLazy(root);
	}

	_makeLazy(node) {
		if (node == null) return node;
		if (Array.isArray(node)) return node.map(v => this._makeLazy(v));

		if (typeof node === "object") {
			// Check for external reference ($ref)
			if (Object.prototype.hasOwnProperty.call(node, _REF_KEY)) {
				const path = String(node[_REF_KEY]);
				const fmt = String(node.format || this._inferFormat(path));
				if (!["npy", "json", "jsonl"].includes(fmt))
					throw new Error("Unsupported ref format: " + fmt);
				return this._makeLazyRef(fmt, path);
			}

			// Check for inline array formats
			const arrayFormat = NDArray.inferFormat(node);
			if (arrayFormat) {
				// Inline array - deserialize immediately
				return NDArray.fromJSON(node, arrayFormat);
			}

			// Recursively wrap objects
			const out = {};
			for (const k in node)
				if (Object.prototype.hasOwnProperty.call(node, k))
					out[k] = this._makeLazy(node[k]);
			return this._proxify(out);
		}

		return node;
	}

	_inferFormat(path) {
		const p = path.toLowerCase();
		if (p.endsWith(".npy")) return "npy";
		if (p.endsWith(".json")) return "json";
		if (p.endsWith(".jsonl")) return "jsonl";
		throw new Error("Cannot infer format from path: " + path);
	}

	_makeLazyRef(fmt, path) {
		let value = null;
		let loaded = false;
		const key = fmt + ":" + path;

		const load = async () => {
			if (this._cache.has(key)) return this._cache.get(key);
			let p;
			if (fmt === "npy") {
				p = NDArray.load(joinUrl(this.path, path), undefined, this.fetchInit);
			} else if (fmt === "json") {
				p = fetch(joinUrl(this.path, path), this.fetchInit).then(r => {
					if (!r.ok) throw new Error("Failed to fetch " + path + ": " + r.status);
					return r.json();
				});
			} else if (fmt === "jsonl") {
				p = fetch(joinUrl(this.path, path), this.fetchInit)
					.then(r => {
						if (!r.ok) throw new Error("Failed to fetch " + path + ": " + r.status);
						return r.text();
					})
					.then(t => t.split(/\r?\n/).filter(Boolean).map(line => JSON.parse(line)));
			} else {
				throw new Error("Unsupported format: " + fmt);
			}
			this._cache.set(key, p);
			return p;
		};

		// Return a proxy object that resolves itself on first access
		return new Proxy(
			{},
			{
				get(_, prop) {
					// Handle 'then' property for await support
					// This must be checked BEFORE the loaded check to handle multiple awaits
					if (prop === "then") {
						if (loaded) {
							// Already loaded - return a thenable that resolves to the value
							// This ensures 'await proxy' returns the actual value, not the proxy
							return (resolve) => resolve(value);
						}
						// Not loaded yet - return a thenable that loads first
						return (resolve, reject) => {
							load().then(resolve, reject);
						};
					}

					// if already loaded, just return property from value
					if (loaded) return value[prop];

					// start loading if not started
					if (!value) {
						// kick off loading once, async
						load().then(v => {
							value = v;
							loaded = true;
						});
					}

					// Before loaded: can't access properties
					throw new Error(`Accessing property '${prop}' before ${path} loaded; use 'await data.${path.split(".").pop()}' or wait a bit.`);
				},
			}
		);
	}

	_proxify(obj) {
		// Return a Proxy that auto-awaits lazy refs when you do `await data.x`
		return new Proxy(obj, {
			get: (target, prop, receiver) => {
				const val = Reflect.get(target, prop, receiver);
				return val;
			},
		});
	}
}
