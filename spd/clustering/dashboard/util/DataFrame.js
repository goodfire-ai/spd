class DataFrame {
	/**
	 * Creates a DataFrame from a list of row objects
	 * 
	 * @param {Array<Object>} data - Array of row objects
	 * @param {Array<string>} [columns] - Optional column names (will be inferred from data if not provided)
	 */
	constructor(data = [], columns = null) {
		this.data = data;

		// Infer columns from data if not provided
		if (!columns && data.length > 0) {
			this.columns = Object.keys(data[0]);
		} else {
			this.columns = columns || [];
		}
	}

	/**
	 * Returns the values for a specific column
	 * 
	 * @param {string} name - Column name
	 * @returns {Array} - Values in the column
	 */
	col(name) {
		if (!this.columns.includes(name)) {
			throw new Error(`Column '${name}' not found in columns: ${this.columns.join(', ')}`);
		}

		return this.data.map(row => row[name]);
	}

	/**
	 * Returns the values for a specific cell
	 * @param {row_idx} row_idx - Row index
	 * @param {col_name} col_name - Column name
	 * @returns {any} - Value in the cell
	 * @throws {Error} - If row index or column name is invalid
	 */
	get(row_idx, col_name) {
		if (row_idx < 0 || row_idx >= this.data.length) {
			throw new Error(`Row index ${row_idx} out of bounds (0 to ${this.data.length - 1})`);
		}

		if (!this.columns.includes(col_name)) {
			throw new Error(`Column '${col_name}' not found in columns: ${this.columns.join(', ')}`);
		}

		return this.data[row_idx][col_name];
	}

	/**
	 * Returns a specific row as an object
	 * 
	 * @param {number} rowIdx - Row index
	 * @returns {Object} - Row as an object
	 */
	row(rowIdx) {
		if (rowIdx < 0 || rowIdx >= this.data.length) {
			throw new Error(`Row index ${rowIdx} out of bounds (0 to ${this.data.length - 1})`);
		}

		return this.data[rowIdx];
	}

	/**
	 * Returns unique values in a specific column
	 * @param {string} name - Column name
	 * @returns {Set} - Unique values in the column
	 */
	col_unique(name) {
		return new Set(this.col(name));
	}

	/**
	 * Apply a function to transform values in a specific column
	 * @param {string} name - Column name
	 * @param {Function} fn - Function to apply to each value (value, rowIndex) => newValue
	 */
	col_apply(name, fn) {
		if (!this.columns.includes(name)) {
			throw new Error(`Column '${name}' not found in columns: ${this.columns.join(', ')}`);
		}

		for (let i = 0; i < this.data.length; i++) {
			this.data[i][name] = fn(this.data[i][name], i);
		}
	}

	/**
	 * Parses a CSV string into a DataFrame
	 * 
	 * @param {string} text - CSV text content
	 * @returns {DataFrame} - New DataFrame instance
	 */
	static from_csv(text) {
		// Split text into lines and filter out empty lines
		const lines = text.split('\n').filter(line => line.trim().length > 0);

		if (lines.length === 0) {
			return new DataFrame();
		}

		// First line is the header
		const header = lines[0].split(',').map(col => col.trim());

		// Parse each line into a row object
		const data = [];
		for (let i = 1; i < lines.length; i++) {
			const values = lines[i].split(',').map(val => val.trim());

			// Create row object mapping column names to values
			const row = {};
			for (let j = 0; j < header.length; j++) {
				// Try to convert to number if possible
				const value = values[j];
				if (value === undefined) continue;

				// Handle quoted strings
				if (value.startsWith('"') && value.endsWith('"')) {
					row[header[j]] = value.slice(1, -1);
				}
				// Handle null values
				else if (value === '' || value.toLowerCase() === 'null') {
					row[header[j]] = null;
				}
				// Try converting to number
				else if (!isNaN(value)) {
					row[header[j]] = Number(value);
				}
				// Otherwise keep as string
				else {
					row[header[j]] = value;
				}
			}

			data.push(row);
		}

		return new DataFrame(data, header);
	}

	/**
	 * Converts DataFrame to CSV string
	 * 
	 * @returns {string} - CSV representation of the DataFrame
	 */
	to_csv() {
		if (this.data.length === 0) {
			return this.columns.join(',');
		}

		// Start with the header
		const lines = [this.columns.join(',')];

		// Add each row
		for (const row of this.data) {
			const values = this.columns.map(col => {
				const val = row[col];

				if (val === null || val === undefined) {
					return '';
				} else if (typeof val === 'string' && (val.includes(',') || val.includes('"'))) {
					// Escape quotes and wrap in quotes
					return `"${val.replace(/"/g, '""')}"`;
				} else {
					return String(val);
				}
			});

			lines.push(values.join(','));
		}

		return lines.join('\n');
	}

	/**
	 * Parses a JSONL string into a DataFrame
	 * 
	 * @param {string} text - JSONL text content
	 * @returns {DataFrame} - New DataFrame instance
	 */
	static from_jsonl(text) {
		// Parse each line as JSON
		const data = text.trim().split('\n').map(line => JSON.parse(line));

		// Extract all unique column names from all rows
		const allColumns = new Set();
		for (const row of data) {
			Object.keys(row).forEach(key => allColumns.add(key));
		}

		return new DataFrame(data, Array.from(allColumns));
	}

	/**
	 * Converts DataFrame to JSONL string
	 * 
	 * @returns {string} - JSONL representation of the DataFrame
	 */
	to_jsonl() {
		return this.data.map(row => JSON.stringify(row)).join('\n');
	}

	/**
	 * Returns the number of rows in the DataFrame
	 * 
	 * @returns {number} - Number of rows
	 */
	get length() {
		return this.data.length;
	}

	/**
	 * Returns a string representation of the DataFrame
	 *
	 * @returns {string} - String representation
	 */
	toString() {
		if (this.data.length === 0) {
			return 'Empty DataFrame';
		}

		return `DataFrame with ${this.data.length} rows and ${this.columns.length} columns: ${this.columns.join(', ')}`;
	}

	/**
	 * Returns HTML code for displaying the DataFrame as a simple table
	 *
	 * @returns {string} - HTML table representation
	 * @throws {Error} - If the DataFrame is empty
	 */
	display_simple() {
		if (this.data.length === 0) {
			throw new Error('Cannot display empty DataFrame');
		}

		let html = '<table>\n';

		html += '  <thead>\n    <tr>\n';
		for (const column of this.columns) {
			html += `      <th>${column}</th>\n`;
		}
		html += '    </tr>\n  </thead>\n';

		html += '  <tbody>\n';
		for (let i = 0; i < this.data.length; i++) {
			const row = this.data[i];
			html += '    <tr>\n';

			for (const column of this.columns) {
				const value = row[column];
				const displayValue = value === null || value === undefined ? '' : String(value);
				html += `      <td>${displayValue}</td>\n`;
			}

			html += '    </tr>\n';
		}
		html += '  </tbody>\n';

		html += '</table>';

		return html;
	}
}
