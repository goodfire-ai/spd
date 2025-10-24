// yaml.js - Simple YAML serialization util

/**
 * Convert a JavaScript object to YAML format
 * Simple implementation - handles basic types, arrays, and nested objects
 * @param {object} obj - Object to convert
 * @param {number} indent - Current indentation level
 * @returns {string} YAML formatted string
 */
function toYAML(obj, indent = 0) {
    const lines = [];
    const indentStr = '  '.repeat(indent);

    for (const [key, value] of Object.entries(obj)) {
        if (value === null || value === undefined) {
            lines.push(`${indentStr}${key}: null`);
        } else if (Array.isArray(value)) {
            if (value.length === 0) {
                lines.push(`${indentStr}${key}: []`);
            } else if (value.every(item => typeof item !== 'object')) {
                // Simple array - inline format
                lines.push(`${indentStr}${key}: [${value.map(v => JSON.stringify(v)).join(', ')}]`);
            } else {
                // Complex array - multiline format
                lines.push(`${indentStr}${key}:`);
                value.forEach(item => {
                    if (typeof item === 'object' && item !== null) {
                        lines.push(`${indentStr}  -`);
                        const subYaml = toYAML(item, indent + 2);
                        lines.push(subYaml.split('\n').map(line => '  ' + line).join('\n'));
                    } else {
                        lines.push(`${indentStr}  - ${JSON.stringify(item)}`);
                    }
                });
            }
        } else if (typeof value === 'object' && value !== null) {
            lines.push(`${indentStr}${key}:`);
            lines.push(toYAML(value, indent + 1));
        } else if (typeof value === 'string') {
            // Always quote strings
            lines.push(`${indentStr}${key}: ${JSON.stringify(value)}`);
        } else {
            // Number, boolean
            lines.push(`${indentStr}${key}: ${value}`);
        }
    }

    return lines.join('\n');
}

/**
 * Create YAML frontmatter block
 * @param {object} obj - Object to convert to YAML
 * @returns {string} YAML frontmatter with delimiters
 */
function createYAMLFrontmatter(obj) {
    return `---\n${toYAML(obj)}\n---`;
}
