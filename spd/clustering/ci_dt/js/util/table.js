
// table.js - Interactive data table component
// origin: https://github.com/mivanit/js-dev-toolkit
// license: GPLv3

// Global constants dictionary
const _TABLE_CONSTS = {
    // CSS Class prefixes
    CSS_PREFIX: 'tablejs-',

    // Default styling values
    COLORS: {
        BORDER: '#ccc',
        HEADER_BG: '#f5f5f5',
        FILTER_BG: '#f5f5f5',
        FOOTER_BG: '#f5f5f5',
        HOVER_BG: '#f0f0f0',
        ERROR_BG: '#ffcccc',
        SUCCESS_COLOR: 'green',
        MUTED_TEXT: '#999'
    },

    // Spacing and sizing
    SPACING: {
        PADDING_CELL: '8px',
        PADDING_FILTER: '4px 8px',
        PADDING_BUTTON: '4px 8px',
        MARGIN_ICON: '4px',
        BORDER_WIDTH: '1px',
        MIN_COLUMN_WIDTH: '2em',
        RESIZE_HANDLE_WIDTH: '4px'
    },

    // Pagination defaults
    PAGINATION: {
        DEFAULT_PAGE_SIZES: [10, 25, 50, 100],
        MAX_VISIBLE_PAGES: 10,
        BUTTON_MIN_WIDTH: '30px'
    },

    // Animation and feedback
    FEEDBACK: {
        SUCCESS_TIMEOUT: 1500,
        ICON_OPACITY_ACTIVE: '1',
        ICON_OPACITY_INACTIVE: '0.6'
    },

    // SVG Icons
    ICONS: {
        sort: '<svg width="16" height="16" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M12 4 L7 9 L10 9 L10 15 L7 15 L12 20 L17 15 L14 15 L14 9 L17 9 Z" fill="currentColor" /></svg>',
        sortUp: '<svg width="16" height="16" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M12 4 L6 12 L10 12 L10 20 L14 20 L14 12 L18 12 Z" fill="currentColor" /></svg>',
        sortDown: '<svg width="16" height="16" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M12 20 L6 12 L10 12 L10 4 L14 4 L14 12 L18 12 Z" fill="currentColor" /></svg>',
        filter: '<svg width="16" height="16" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><rect x="4" y="5" width="16" height="2" rx="1" fill="currentColor" /><rect x="7" y="10" width="10" height="2" rx="1" fill="currentColor" /><rect x="10" y="15" width="4" height="2" rx="1" fill="currentColor" /></svg>',
        download: '<svg width="16" height="16" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M12 3 L12 14 M12 14 L7 9 M12 14 L17 9" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" /><path d="M5 16 L5 20 L19 20 L19 16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" /></svg>',
        copy: '<svg width="16" height="16" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><rect x="5" y="9" width="10" height="10" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" /><rect x="9" y="5" width="10" height="10" fill="white" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" /></svg>',
        prev: '‹',
        next: '›',
        clear: '×'
    },

    // Default messages
    MESSAGES: {
        NO_DATA: 'No data to display',
        COPY_SUCCESS: 'Copied!',
        COPY_TITLE: 'Copy to clipboard',
        CLEAR_FILTER: 'Clear filter',
        EXPORT_CSV: 'Export CSV',
        CLEAR_FILTERS: 'Clear Filters',
        DOWNLOAD_FILENAME: 'table-data.csv'
    },

    // Filter tooltips
    FILTER_TOOLTIPS: {
        NUMBER: 'Use operators: >, <, >=, <=, ==, != (e.g., >50, <=100)',
        DEFAULT: 'Type to filter. Use * for wildcards (e.g., foo*, *bar)'
    }
};

class DataTable {
    constructor(container, config = {}) {
        this.container = typeof container === 'string' ? document.querySelector(container) : container;
        if (!this.container) {
            throw new Error(`DataTable container not found: ${container}`);
        }
        this.data = config.data || [];
        this.columns = config.columns || [];
        this.pageSizeOptions = config.pageSizeOptions || _TABLE_CONSTS.PAGINATION.DEFAULT_PAGE_SIZES;
        this.pageSize = config.pageSize || this.pageSizeOptions[0];
        this.showFilters = config.showFilters !== false; // Default to true
        this.showInfo = config.showInfo === true; // Default to false
        this.minColumnWidth = config.minColumnWidth || _TABLE_CONSTS.SPACING.MIN_COLUMN_WIDTH;
        this.currentPage = 1;
        this.sortColumnId = null;
        this.sortDirection = null;
        this.filters = {};
        this.filteredData = [];
        this.paginationListeners = []; // Store listeners for cleanup
        if (this.columns.length === 0 && this.data.length > 0) {
            this.columns = Object.keys(this.data[0]).map(key => ({
                key: key,
                label: key.charAt(0).toUpperCase() + key.slice(1),
                type: this.inferType(this.data[0][key]),
                filterable: true
            }));
        }

        this.init();
    }

    inferType(value) {
        if (value === null || value === undefined) return 'string';
        if (typeof value === 'number') return 'number';
        if (typeof value === 'boolean') return 'boolean';
        if (value instanceof Date) return 'date';
        return 'string';
    }

    getNestedValue(obj, key) {
        // Support dot notation for nested keys (e.g., 'stats.entropy')
        if (!key.includes('.')) {
            return obj[key];
        }

        const keys = key.split('.');
        let value = obj;
        for (const k of keys) {
            if (value === null || value === undefined) {
                return undefined;
            }
            value = value[k];
        }
        return value;
    }

    getColumnId(col) {
        // Use id if provided, otherwise fall back to key
        return col.id || col.key;
    }

    cssClass(className) {
        return _TABLE_CONSTS.CSS_PREFIX + className;
    }

    createStyledElement(tagName, className, styles = {}) {
        const element = document.createElement(tagName);
        if (className) {
            element.className = this.cssClass(className);
        }
        Object.assign(element.style, styles);
        return element;
    }

    getMinColumnWidthInPixels() {
        // Create a temporary element to measure the minimum width
        const tempDiv = document.createElement('div');
        tempDiv.style.position = 'absolute';
        tempDiv.style.visibility = 'hidden';
        tempDiv.style.width = this.minColumnWidth;
        tempDiv.style.fontSize = window.getComputedStyle(this.container).fontSize;
        document.body.appendChild(tempDiv);
        const pixelWidth = tempDiv.offsetWidth;
        document.body.removeChild(tempDiv);
        return pixelWidth;
    }

    init() {
        this.createTableStructure();
        this.applyFiltersAndSort();
        this.render();
    }

    createTableStructure() {
        this.container.innerHTML = '';

        // Create wrapper
        const wrapper = this.createStyledElement('div', 'wrapper', {});

        // Create table container
        const tableContainer = this.createStyledElement('div', 'container');

        const table = this.createStyledElement('table', 'table');

        // Create header
        const thead = this.createStyledElement('thead', 'thead');

        // Header row with column names, sort, and resize handles
        const headerRow = this.createStyledElement('tr', 'header-row', {
            backgroundColor: _TABLE_CONSTS.COLORS.HEADER_BG
        });
        this.columns.forEach((col, index) => {
            const thStyles = {
                position: 'relative',
                padding: _TABLE_CONSTS.SPACING.PADDING_CELL,
                borderBottom: `${_TABLE_CONSTS.SPACING.BORDER_WIDTH} solid ${_TABLE_CONSTS.COLORS.BORDER}`
                // Don't set minWidth here - it will be enforced during resize
            };

            if (col.width) {
                thStyles.width = col.width;
            }

            if (col.align) {
                thStyles.textAlign = col.align;
            }

            const th = this.createStyledElement('th', 'header-cell', thStyles);

            const headerContent = this.createStyledElement('div', 'header-content', {
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                minWidth: '0' // allow flex children to shrink
            });

            const label = this.createStyledElement('span', 'header-label', {
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap'
            });
            label.textContent = col.label || col.key;
            headerContent.appendChild(label);

            const sortIcon = this.createStyledElement('span', 'sort-icon', {
                marginLeft: _TABLE_CONSTS.SPACING.MARGIN_ICON,
                opacity: _TABLE_CONSTS.FEEDBACK.ICON_OPACITY_INACTIVE
            });
            sortIcon.innerHTML = _TABLE_CONSTS.ICONS.sort;
            headerContent.appendChild(sortIcon);

            headerContent.onclick = () => this.handleSort(this.getColumnId(col));

            // Add resize handle
            const resizeHandle = this.createStyledElement('div', 'resize-handle', {
                position: 'absolute',
                right: '0',
                top: '0',
                bottom: '0',
                width: _TABLE_CONSTS.SPACING.RESIZE_HANDLE_WIDTH,
                cursor: 'col-resize',
                background: 'transparent',
                borderRight: `2px solid transparent`,
                transition: 'border-color 0.2s'
            });

            // Add hover effect
            resizeHandle.addEventListener('mouseenter', () => {
                resizeHandle.style.borderRightColor = _TABLE_CONSTS.COLORS.BORDER;
            });
            resizeHandle.addEventListener('mouseleave', () => {
                resizeHandle.style.borderRightColor = 'transparent';
            });

            this.addResizeListener(resizeHandle, th, index);

            th.appendChild(headerContent);
            th.appendChild(resizeHandle);
            headerRow.appendChild(th);
        });
        thead.appendChild(headerRow);

        // Info row (conditional)
        if (this.showInfo) {
            const infoRow = this.createStyledElement('tr', 'info-row', {
                backgroundColor: _TABLE_CONSTS.COLORS.HEADER_BG,
                borderTop: `${_TABLE_CONSTS.SPACING.BORDER_WIDTH} solid ${_TABLE_CONSTS.COLORS.BORDER}`
            });

            this.columns.forEach(col => {
                const tdStyles = {
                    padding: _TABLE_CONSTS.SPACING.PADDING_CELL,
                    backgroundColor: _TABLE_CONSTS.COLORS.HEADER_BG,
                    fontSize: '0.9em',
                    fontStyle: 'italic',
                    color: _TABLE_CONSTS.COLORS.MUTED_TEXT
                };

                if (col.align) {
                    tdStyles.textAlign = col.align;
                }

                const td = this.createStyledElement('td', 'info-cell', tdStyles);
                td.innerHTML = this.calculateColumnInfo(col);
                infoRow.appendChild(td);
            });
            thead.appendChild(infoRow);
            this.infoRow = infoRow;
        }

        // Filter row (conditional)
        if (this.showFilters) {
            const filterRow = this.createStyledElement('tr', 'filter-row', {
                backgroundColor: _TABLE_CONSTS.COLORS.FILTER_BG,
                borderTop: `${_TABLE_CONSTS.SPACING.BORDER_WIDTH} solid ${_TABLE_CONSTS.COLORS.BORDER}`
            });

            this.columns.forEach(col => {
                const tdStyles = {
                    padding: _TABLE_CONSTS.SPACING.PADDING_FILTER,
                    backgroundColor: _TABLE_CONSTS.COLORS.FILTER_BG
                };

                if (col.align) {
                    tdStyles.textAlign = col.align;
                }

                const td = this.createStyledElement('td', 'filter-cell', tdStyles);

                // Check if column is filterable
                const isFilterable = col.filterable !== false;

                if (isFilterable) {
                    const filterContainer = this.createStyledElement('div', 'filter-container', {
                        display: 'flex',
                        alignItems: 'center',
                        minWidth: '0' // allow this flex container to shrink inside the cell
                    });

                    // Add filter icon
                    const filterIcon = this.createStyledElement('span', 'filter-icon', {
                        marginRight: _TABLE_CONSTS.SPACING.MARGIN_ICON,
                        opacity: '0.5',
                        display: 'flex',
                        alignItems: 'center'
                    });
                    filterIcon.innerHTML = _TABLE_CONSTS.ICONS.filter;

                    const input = this.createStyledElement('input', 'filter-input', {
                        flex: '1 1 0',     // allow shrinking and growing; 0 basis avoids intrinsic width bias
                        minWidth: '0',     // critical: override flexbox min-width:auto
                        width: '0',        // ensures the flex-basis drives width, not the default input width
                        border: `${_TABLE_CONSTS.SPACING.BORDER_WIDTH} solid ${_TABLE_CONSTS.COLORS.BORDER}`,
                        padding: '2px 4px',
                        borderRadius: '3px',
                        overflow: 'hidden' // optional: clip when very narrow
                    });
                    input.type = 'text';
                    input.placeholder = '';

                    // Add tooltip for filter help
                    const tooltipText = this.getFilterTooltip(col);
                    input.title = tooltipText;
                    filterIcon.title = tooltipText;

                    const clearBtn = this.createStyledElement('button', 'filter-clear-btn', {
                        border: 'none',
                        background: 'none',
                        cursor: 'pointer',
                        padding: '2px 8px',
                        fontSize: '16px',
                        color: _TABLE_CONSTS.COLORS.MUTED_TEXT,
                        marginLeft: _TABLE_CONSTS.SPACING.MARGIN_ICON
                    });
                    clearBtn.textContent = _TABLE_CONSTS.ICONS.clear;
                    clearBtn.title = _TABLE_CONSTS.MESSAGES.CLEAR_FILTER;

                    input.addEventListener('input', (e) => {
                        this.handleFilter(this.getColumnId(col), col.key, e.target.value, col.type, input, col);
                    });

                    clearBtn.addEventListener('click', () => {
                        this.clearFilter(this.getColumnId(col), input);
                    });

                    filterContainer.appendChild(filterIcon);
                    filterContainer.appendChild(input);
                    filterContainer.appendChild(clearBtn);
                    td.appendChild(filterContainer);
                }
                filterRow.appendChild(td);
            });
            thead.appendChild(filterRow);
        }

        table.appendChild(thead);

        // Create body
        const tbody = this.createStyledElement('tbody', 'tbody');
        table.appendChild(tbody);

        tableContainer.appendChild(table);
        wrapper.appendChild(tableContainer);

        // Create bottom pagination with page size selector
        const paginationBottom = this.createStyledElement('div', 'pagination-bottom', {
            backgroundColor: _TABLE_CONSTS.COLORS.FOOTER_BG,
            borderTop: `${_TABLE_CONSTS.SPACING.BORDER_WIDTH} solid ${_TABLE_CONSTS.COLORS.BORDER}`,
            padding: _TABLE_CONSTS.SPACING.PADDING_CELL
        });
        wrapper.appendChild(paginationBottom);

        this.container.appendChild(wrapper);

        // Store references
        this.tbody = tbody;
        this.paginationBottom = paginationBottom;
        this.headerRow = headerRow;

        // Store table reference on container for button access
        this.container.table = this;
    }

    addResizeListener(handle, th, columnIndex) {
        let startX, startWidth;

        handle.addEventListener('mousedown', (e) => {
            e.preventDefault(); // Prevent text selection
            e.stopPropagation(); // Prevent header click events
            startX = e.clientX;
            startWidth = parseInt(document.defaultView.getComputedStyle(th).width, 10);
            document.body.style.cursor = 'col-resize';
            document.body.style.userSelect = 'none'; // Prevent text selection during drag
            document.addEventListener('mousemove', doDrag);
            document.addEventListener('mouseup', stopDrag);
        });

        const doDrag = (e) => {
            const minWidthPx = this.getMinColumnWidthInPixels();
            const newWidth = startWidth + e.clientX - startX;
            const width = Math.max(minWidthPx, newWidth);
            th.style.width = width + 'px';
            th.style.minWidth = width + 'px';  // Must update minWidth to allow shrinking below current size
            // Update column width in config
            this.columns[columnIndex].width = width + 'px';
        };

        const stopDrag = () => {
            document.body.style.cursor = ''; // Reset cursor
            document.body.style.userSelect = ''; // Reset text selection
            document.removeEventListener('mousemove', doDrag);
            document.removeEventListener('mouseup', stopDrag);
        };
    }

    handleSort(columnId) {
        if (this.sortColumnId === columnId) {
            // Cycle through: asc -> desc -> none
            if (this.sortDirection === 'asc') {
                this.sortDirection = 'desc';
            } else if (this.sortDirection === 'desc') {
                this.sortColumnId = null;
                this.sortDirection = null;
            }
        } else {
            this.sortColumnId = columnId;
            this.sortDirection = 'asc';
        }

        this.applyFiltersAndSort();
        this.render();
    }

    parseNumericFilter(value) {
        const trimmed = value.trim();
        if (!trimmed) return null;

        const match = trimmed.match(/^(==|!=|>=|<=|>|<)?\s*(-?\d+\.?\d*)$/);
        if (!match) return null;

        const operator = match[1] || '==';
        const number = parseFloat(match[2]);

        if (isNaN(number)) return null;

        return { operator, value: number };
    }

    applyNumericFilter(cellValue, filter) {
        if (!filter) return true;

        const value = parseFloat(cellValue);
        if (isNaN(value)) return false;

        switch (filter.operator) {
            case '==': return value === filter.value;
            case '!=': return value !== filter.value;
            case '>': return value > filter.value;
            case '<': return value < filter.value;
            case '>=': return value >= filter.value;
            case '<=': return value <= filter.value;
            default: return true;
        }
    }

    getFilterTooltip(col) {
        // Check for custom tooltip in column definition
        if (col.filterTooltip) {
            return col.filterTooltip;
        }

        // Default tooltips based on type
        if (col.type === 'number') {
            return _TABLE_CONSTS.FILTER_TOOLTIPS.NUMBER;
        }
        return _TABLE_CONSTS.FILTER_TOOLTIPS.DEFAULT;
    }

    calculateColumnInfo(col) {
        // Check for custom info function in column definition
        if (col.infoFunction) {
            const columnData = this.data.map(row => row[col.key]).filter(val => val !== null && val !== undefined);
            const result = col.infoFunction(columnData, col);
            return typeof result === 'string' ? result : result.outerHTML || '';
        }

        // Default statistical calculations
        const columnData = this.data.map(row => row[col.key]).filter(val => val !== null && val !== undefined);

        if (columnData.length === 0) {
            return 'No data';
        }

        if (col.type === 'number') {
            const numbers = columnData.filter(val => typeof val === 'number');
            if (numbers.length === 0) return 'No numeric data';

            const min = Math.min(...numbers);
            const max = Math.max(...numbers);
            const sum = numbers.reduce((a, b) => a + b, 0);
            const mean = sum / numbers.length;

            // Calculate median
            const sorted = [...numbers].sort((a, b) => a - b);
            const median = sorted.length % 2 === 0
                ? (sorted[sorted.length / 2 - 1] + sorted[sorted.length / 2]) / 2
                : sorted[Math.floor(sorted.length / 2)];

            return `R=[${min.toLocaleString()}, ${max.toLocaleString()}] μ=${mean.toFixed(1)} x̃=${median.toLocaleString()}`;
        } else {
            // For string/other types, show unique count
            const uniqueValues = new Set(columnData.map(val => String(val)));
            const count = uniqueValues.size;
            return `${count} unique value${count === 1 ? '' : 's'}`;
        }
    }

    handleFilter(columnId, columnKey, filterValue, type, inputElement, col) {
        if (!filterValue) {
            delete this.filters[columnId];
            inputElement.style.backgroundColor = '';
        } else {
            let isValid = true;
            let customFilter = null;

            // col is already passed in
            if (col?.filterFunction) {
                try {
                    customFilter = col.filterFunction(filterValue);
                    // If custom function returns null, fall back to default
                    if (customFilter === null) {
                        customFilter = null;
                        if (type === 'number') {
                            const numFilter = this.parseNumericFilter(filterValue);
                            isValid = numFilter !== null;
                        }
                    } else {
                        isValid = typeof customFilter === 'function';
                    }
                } catch (e) {
                    isValid = false;
                }
            } else if (type === 'number') {
                const numFilter = this.parseNumericFilter(filterValue);
                isValid = numFilter !== null;
            }

            this.filters[columnId] = {
                key: columnKey,  // Store the key for data access
                value: filterValue,
                type: type,
                valid: isValid,
                customFilter: customFilter
            };

            inputElement.style.backgroundColor = isValid ? '' : _TABLE_CONSTS.COLORS.ERROR_BG;
        }

        this.currentPage = 1;
        this.applyFiltersAndSort();
        this.validateCurrentPage();
        this.render();
    }

    clearFilter(columnId, inputElement) {
        inputElement.value = '';
        delete this.filters[columnId];
        inputElement.style.backgroundColor = '';
        this.currentPage = 1;
        this.applyFiltersAndSort();
        this.validateCurrentPage();
        this.render();
    }

    applyFiltersAndSort() {
        // Start with all data
        let filtered = [...this.data];

        filtered = filtered.filter(row => {
            for (const [columnId, filter] of Object.entries(this.filters)) {
                if (!filter.valid) continue;

                // Get cellValue using the stored key
                const cellValue = row[filter.key];

                // Apply custom filter if available
                if (filter.customFilter) {
                    try {
                        if (!filter.customFilter(cellValue)) {
                            return false;
                        }
                    } catch (e) {
                        // If custom filter fails, treat as no match
                        return false;
                    }
                } else if (filter.type === 'number') {
                    const numFilter = this.parseNumericFilter(filter.value);
                    if (numFilter && !this.applyNumericFilter(cellValue, numFilter)) {
                        return false;
                    }
                } else {
                    if (cellValue === null || cellValue === undefined) return false;
                    const strValue = String(cellValue).toLowerCase();
                    const filterStr = filter.value.toLowerCase();

                    if (filterStr.includes('*')) {
                        const regex = new RegExp('^' + filterStr.replace(/\*/g, '.*') + '$');
                        if (!regex.test(strValue)) return false;
                    } else {
                        if (!strValue.includes(filterStr)) return false;
                    }
                }
            }
            return true;
        });

        // Apply sorting
        if (this.sortColumnId) {
            // Find the column config for custom sort function
            const columnConfig = this.columns.find(col => this.getColumnId(col) === this.sortColumnId);
            const sortFunction = columnConfig?.sortFunction;

            filtered.sort((a, b) => {
                // Get values using nested key support (use column's key, not id)
                let aVal = this.getNestedValue(a, columnConfig.key);
                let bVal = this.getNestedValue(b, columnConfig.key);

                // Apply custom sort function if provided
                if (sortFunction) {
                    aVal = sortFunction(aVal, a);
                    bVal = sortFunction(bVal, b);
                }

                // Handle nulls
                if (aVal === null || aVal === undefined) return 1;
                if (bVal === null || bVal === undefined) return -1;

                let comparison = 0;
                if (typeof aVal === 'number' && typeof bVal === 'number') {
                    comparison = aVal - bVal;
                } else {
                    comparison = String(aVal).localeCompare(String(bVal));
                }

                return this.sortDirection === 'desc' ? -comparison : comparison;
            });
        }

        this.filteredData = filtered;
    }


    getPageData() {
        const start = (this.currentPage - 1) * this.pageSize;
        const end = start + this.pageSize;
        return this.filteredData.slice(start, end);
    }

    getTotalPages() {
        return Math.ceil(this.filteredData.length / this.pageSize);
    }

    validateCurrentPage() {
        const totalPages = this.getTotalPages();
        if (this.currentPage > totalPages && totalPages > 0) {
            this.currentPage = totalPages;
        } else if (this.currentPage < 1) {
            this.currentPage = 1;
        }
    }

    render() {
        this.renderTable();
        this.renderPagination();
        this.updateSortIcons();
        this.updateInfoRow();
    }

    updateSortIcons() {
        const headers = this.headerRow.querySelectorAll('th');
        headers.forEach((th, index) => {
            const col = this.columns[index];
            const sortIcon = th.querySelector(`.${this.cssClass('sort-icon')}`);
            if (!sortIcon) return;

            if (this.sortColumnId === this.getColumnId(col)) {
                if (this.sortDirection === 'asc') {
                    sortIcon.innerHTML = _TABLE_CONSTS.ICONS.sortUp;
                    sortIcon.style.opacity = _TABLE_CONSTS.FEEDBACK.ICON_OPACITY_ACTIVE;
                } else if (this.sortDirection === 'desc') {
                    sortIcon.innerHTML = _TABLE_CONSTS.ICONS.sortDown;
                    sortIcon.style.opacity = _TABLE_CONSTS.FEEDBACK.ICON_OPACITY_ACTIVE;
                }
            } else {
                sortIcon.innerHTML = _TABLE_CONSTS.ICONS.sort;
                sortIcon.style.opacity = _TABLE_CONSTS.FEEDBACK.ICON_OPACITY_INACTIVE;
            }
        });
    }

    updateInfoRow() {
        if (!this.showInfo || !this.infoRow) return;

        const infoCells = this.infoRow.querySelectorAll('td');
        infoCells.forEach((cell, index) => {
            const col = this.columns[index];
            cell.innerHTML = this.calculateColumnInfo(col);
        });
    }

    renderTable() {
        this.tbody.innerHTML = '';

        const pageData = this.getPageData();

        if (pageData.length === 0) {
            const tr = this.createStyledElement('tr', 'empty-row');
            const td = this.createStyledElement('td', 'empty-cell', {
                textAlign: 'center',
                padding: _TABLE_CONSTS.SPACING.PADDING_CELL,
                fontStyle: 'italic',
                color: _TABLE_CONSTS.COLORS.MUTED_TEXT
            });
            td.colSpan = this.columns.length;
            td.textContent = _TABLE_CONSTS.MESSAGES.NO_DATA;
            tr.appendChild(td);
            this.tbody.appendChild(tr);
            return;
        }

        pageData.forEach(row => {
            const tr = this.createStyledElement('tr', 'data-row');

            this.columns.forEach(col => {
                const tdStyles = {
                    padding: _TABLE_CONSTS.SPACING.PADDING_CELL,
                    borderBottom: `${_TABLE_CONSTS.SPACING.BORDER_WIDTH} solid ${_TABLE_CONSTS.COLORS.BORDER}`
                };

                if (col.align) {
                    tdStyles.textAlign = col.align;
                }

                const td = this.createStyledElement('td', 'data-cell', tdStyles);
                const value = this.getNestedValue(row, col.key);

                // Check for custom renderer in column definition
                if (col.renderer) {
                    const rendered = col.renderer(value, row, col);
                    if (typeof rendered === 'string') {
                        td.innerHTML = rendered;
                    } else {
                        td.appendChild(rendered);
                    }
                } else {
                    // Default rendering
                    if (value === null || value === undefined) {
                        td.textContent = '';
                    } else if (col.type === 'number' && typeof value === 'number') {
                        td.textContent = value.toLocaleString();
                    } else {
                        td.textContent = String(value);
                    }
                }

                tr.appendChild(td);
            });

            this.tbody.appendChild(tr);
        });
    }

    renderPagination() {
        const totalPages = this.getTotalPages();
        const paginationHTML = this.createPaginationHTML(totalPages);

        this.paginationBottom.innerHTML = paginationHTML;

        // Clear old listeners to prevent memory leaks
        this.paginationListeners = [];

        // Add page button listeners
        this.paginationBottom.querySelectorAll('button').forEach(btn => {
            const handler = (e) => {
                const page = parseInt(e.target.dataset.page);
                if (!isNaN(page)) {
                    this.currentPage = page;
                    this.render();
                }
            };
            btn.addEventListener('click', handler);
            this.paginationListeners.push({ element: btn, event: 'click', handler });
        });

        // Add page size selector listener
        const pageSizeSelect = this.paginationBottom.querySelector(`.${this.cssClass('page-size-select')}`);
        if (pageSizeSelect) {
            const handler = (e) => {
                this.setPageSize(parseInt(e.target.value));
            };
            pageSizeSelect.addEventListener('change', handler);
            this.paginationListeners.push({ element: pageSizeSelect, event: 'change', handler });
        }

        // Add export CSV button listeners
        const exportDownloadBtn = this.paginationBottom.querySelector(`.${this.cssClass('export-download-btn')}`);
        if (exportDownloadBtn) {
            const handler = () => {
                this.exportAndDownloadCSV();
            };
            exportDownloadBtn.addEventListener('click', handler);
            this.paginationListeners.push({ element: exportDownloadBtn, event: 'click', handler });
        }

        const exportCopyBtn = this.paginationBottom.querySelector(`.${this.cssClass('export-copy-btn')}`);
        if (exportCopyBtn) {
            const handler = () => {
                this.copyCSVToClipboard();
            };
            exportCopyBtn.addEventListener('click', handler);
            this.paginationListeners.push({ element: exportCopyBtn, event: 'click', handler });
        }

        // Add clear filters button listener
        const clearBtn = this.paginationBottom.querySelector(`.${this.cssClass('clear-filters-btn')}`);
        if (clearBtn) {
            const handler = () => {
                this.clearAllFilters();
            };
            clearBtn.addEventListener('click', handler);
            this.paginationListeners.push({ element: clearBtn, event: 'click', handler });
        }
    }

    createPaginationHTML(totalPages) {
        let html = '<div style="display: flex; justify-content: space-between; align-items: center;">';

        // Left side - controls
        html += '<div style="display: flex; align-items: center; gap: 10px;">';
        html += this.createExportCSVButton();
        if (this.showFilters) {
            const clearBtnStyle = `border: ${_TABLE_CONSTS.SPACING.BORDER_WIDTH} solid ${_TABLE_CONSTS.COLORS.BORDER}; background: white; padding: ${_TABLE_CONSTS.SPACING.PADDING_BUTTON}; cursor: pointer; border-radius: 3px;`;
            html += `<button class="${this.cssClass('clear-filters-btn')}" style="${clearBtnStyle}">${_TABLE_CONSTS.MESSAGES.CLEAR_FILTERS}</button>`;
        }
        html += this.createPageSizeSelector();
        html += '</div>';

        if (totalPages <= 1) {
            html += '</div>';
            return html;
        }

        // Right side - pagination controls
        html += '<div style="display: flex; align-items: center; gap: 2px;">';

        // Previous button
        const prevBtnStyle = `min-width: ${_TABLE_CONSTS.PAGINATION.BUTTON_MIN_WIDTH}; padding: ${_TABLE_CONSTS.SPACING.PADDING_BUTTON}; border: ${_TABLE_CONSTS.SPACING.BORDER_WIDTH} solid ${_TABLE_CONSTS.COLORS.BORDER}; background: white; cursor: pointer;`;
        html += `<button class="${this.cssClass('pagination-btn')}" style="${prevBtnStyle}" ${this.currentPage === 1 ? 'disabled' : ''} data-page="${this.currentPage - 1}">${_TABLE_CONSTS.ICONS.prev}</button>`;

        // Page buttons - always show 10 elements for consistent width
        const pageButtons = this.generatePageButtons(totalPages);
        pageButtons.forEach(item => {
            if (item.type === 'page') {
                const isActive = item.page === this.currentPage;
                const activeBg = isActive ? _TABLE_CONSTS.COLORS.BORDER : 'white';
                const pageBtnStyle = `min-width: ${_TABLE_CONSTS.PAGINATION.BUTTON_MIN_WIDTH}; padding: ${_TABLE_CONSTS.SPACING.PADDING_BUTTON}; border: ${_TABLE_CONSTS.SPACING.BORDER_WIDTH} solid ${_TABLE_CONSTS.COLORS.BORDER}; background: ${activeBg}; cursor: pointer;`;
                html += `<button class="${this.cssClass('pagination-btn')}" style="${pageBtnStyle}" ${isActive ? 'disabled' : ''} data-page="${item.page}">${item.page}</button>`;
            } else if (item.type === 'dots') {
                html += `<span class="${this.cssClass('pagination-dots')}" style="min-width: ${_TABLE_CONSTS.PAGINATION.BUTTON_MIN_WIDTH}; padding: ${_TABLE_CONSTS.SPACING.PADDING_BUTTON}; text-align: center; display: inline-block;">…</span>`;
            }
        });

        // Next button
        const nextBtnStyle = `min-width: ${_TABLE_CONSTS.PAGINATION.BUTTON_MIN_WIDTH}; padding: ${_TABLE_CONSTS.SPACING.PADDING_BUTTON}; border: ${_TABLE_CONSTS.SPACING.BORDER_WIDTH} solid ${_TABLE_CONSTS.COLORS.BORDER}; background: white; cursor: pointer;`;
        html += `<button class="${this.cssClass('pagination-btn')}" style="${nextBtnStyle}" ${this.currentPage === totalPages ? 'disabled' : ''} data-page="${this.currentPage + 1}">${_TABLE_CONSTS.ICONS.next}</button>`;

        html += '</div>'; // End pagination controls
        html += '</div>'; // End main container

        return html;
    }

    generatePageButtons(totalPages) {
        if (totalPages <= 10) {
            // Show all pages
            return Array.from({ length: totalPages }, (_, i) => ({ type: 'page', page: i + 1 }));
        }

        // Always show exactly 10 elements (pages + dots)
        const items = [];

        // Always show page 1
        items.push({ type: 'page', page: 1 });

        // Determine the window around current page
        let start = Math.max(2, this.currentPage - 3);
        let end = Math.min(totalPages - 1, this.currentPage + 3);

        // Add dots after 1 if needed
        if (start > 2) {
            items.push({ type: 'dots' });
            // Adjust to maintain 10 elements total
            if (end === totalPages - 1) start = Math.max(start, totalPages - 7);
        } else {
            start = 2;
        }

        // Add middle pages
        for (let i = start; i <= end && items.length < 9; i++) {
            items.push({ type: 'page', page: i });
        }

        // Add dots before last page if needed
        if (end < totalPages - 1) {
            items.push({ type: 'dots' });
        }

        // Always show last page (if different from 1)
        if (totalPages > 1) {
            items.push({ type: 'page', page: totalPages });
        }

        // Pad with pages if we have less than 10 items
        while (items.length < 10 && totalPages > items.length) {
            const initialLength = items.length;

            // Find gaps and fill them
            for (let i = 1; i < items.length; i++) {
                if (items[i].type === 'page' && items[i - 1].type === 'page' &&
                    items[i].page - items[i - 1].page > 1) {
                    items.splice(i, 0, { type: 'page', page: items[i - 1].page + 1 });
                    break;
                }
            }

            // Safety check: if no progress was made, break to avoid infinite loop
            if (items.length === initialLength) {
                break;
            }
        }

        return items.slice(0, 10); // Ensure exactly 10 elements
    }

    createExportCSVButton() {
        const containerStyle = `display: flex; align-items: center; gap: 4px; border: ${_TABLE_CONSTS.SPACING.BORDER_WIDTH} solid ${_TABLE_CONSTS.COLORS.BORDER}; border-radius: 4px; padding: 2px; margin: 3px;`;
        const btnStyle = `border: none; background: none; padding: ${_TABLE_CONSTS.SPACING.PADDING_BUTTON}; cursor: pointer; display: flex; align-items: center; gap: 4px;`;
        const copyBtnStyle = `border: none; background: none; padding: ${_TABLE_CONSTS.SPACING.PADDING_BUTTON}; cursor: pointer; display: flex; align-items: center;`;
        const separatorStyle = `border-left: ${_TABLE_CONSTS.SPACING.BORDER_WIDTH} solid ${_TABLE_CONSTS.COLORS.BORDER}; height: 20px;`;

        return `<div class="${this.cssClass('export-container')}" style="${containerStyle}">
            <button class="${this.cssClass('export-download-btn')}" style="${btnStyle}">
                ${_TABLE_CONSTS.MESSAGES.EXPORT_CSV} ${_TABLE_CONSTS.ICONS.download}
            </button>
            <div style="${separatorStyle}"></div>
            <button class="${this.cssClass('export-copy-btn')}" style="${copyBtnStyle}" title="${_TABLE_CONSTS.MESSAGES.COPY_TITLE}">
                ${_TABLE_CONSTS.ICONS.copy}
            </button>
        </div>`;
    }

    createPageSizeSelector() {
        const options = this.pageSizeOptions.map(size =>
            `<option value="${size}" ${this.pageSize === size ? 'selected' : ''}>${size}</option>`
        ).join('');

        const selectStyle = `border: ${_TABLE_CONSTS.SPACING.BORDER_WIDTH} solid ${_TABLE_CONSTS.COLORS.BORDER}; padding: 2px 4px; margin: 0 4px;`;

        return `<label class="${this.cssClass('page-size-label')}">Show
            <select class="${this.cssClass('page-size-select')}" style="${selectStyle}">
                ${options}
            </select>
            entries</label>`;
    }

    // Public methods for data manipulation
    setData(data) {
        this.data = data;
        this.currentPage = 1;
        this.applyFiltersAndSort();
        this.validateCurrentPage();
        this.render();
    }

    addRow(row) {
        this.data.push(row);
        this.applyFiltersAndSort();
        this.validateCurrentPage();
        this.render();
    }

    setPageSize(size) {
        this.pageSize = size;
        this.currentPage = 1;
        this.applyFiltersAndSort();
        this.validateCurrentPage();
        this.render();
    }

    exportCSV() {
        const csv = [];

        // Headers
        csv.push(this.columns.map(col => col.label || col.key).join(','));

        // Data
        this.filteredData.forEach(row => {
            const values = this.columns.map(col => {
                const val = row[col.key];
                if (val === null || val === undefined) return '';
                let str = String(val);

                // Escape newlines in cell content
                str = str.replace(/\n/g, '\\n');

                // Escape if contains comma or quotes
                if (str.includes(',') || str.includes('"')) {
                    return '"' + str.replace(/"/g, '""') + '"';
                }
                return str;
            });
            csv.push(values.join(','));
        });

        return csv.join('\n');
    }

    exportAndDownloadCSV() {
        const csv = this.exportCSV();
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = _TABLE_CONSTS.MESSAGES.DOWNLOAD_FILENAME;
        a.click();
        URL.revokeObjectURL(url);
    }

    async copyCSVToClipboard() {
        const csv = this.exportCSV();
        try {
            await navigator.clipboard.writeText(csv);
            // Provide visual feedback
            const copyBtn = this.paginationBottom.querySelector(`.${this.cssClass('export-copy-btn')}`);
            if (copyBtn) {
                const originalTitle = copyBtn.title;
                copyBtn.title = _TABLE_CONSTS.MESSAGES.COPY_SUCCESS;
                copyBtn.style.color = _TABLE_CONSTS.COLORS.SUCCESS_COLOR;
                setTimeout(() => {
                    copyBtn.title = originalTitle;
                    copyBtn.style.color = '';
                }, _TABLE_CONSTS.FEEDBACK.SUCCESS_TIMEOUT);
            }
        } catch (err) {
            console.error('Failed to copy CSV to clipboard:', err);
            // Fallback for older browsers
            this.fallbackCopyToClipboard(csv);
        }
    }

    fallbackCopyToClipboard(text) {
        const textArea = document.createElement('textarea');
        textArea.value = text;
        textArea.style.position = 'fixed';
        textArea.style.left = '-999999px';
        textArea.style.top = '-999999px';
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();
        try {
            document.execCommand('copy');
            const copyBtn = this.paginationBottom.querySelector(`.${this.cssClass('export-copy-btn')}`);
            if (copyBtn) {
                const originalTitle = copyBtn.title;
                copyBtn.title = _TABLE_CONSTS.MESSAGES.COPY_SUCCESS;
                copyBtn.style.color = _TABLE_CONSTS.COLORS.SUCCESS_COLOR;
                setTimeout(() => {
                    copyBtn.title = originalTitle;
                    copyBtn.style.color = '';
                }, _TABLE_CONSTS.FEEDBACK.SUCCESS_TIMEOUT);
            }
        } catch (err) {
            console.error('Fallback copy failed:', err);
        }
        document.body.removeChild(textArea);
    }

    clearAllFilters() {
        this.filters = {};
        this.container.querySelectorAll(`.${this.cssClass('filter-input')}`).forEach(input => {
            input.value = '';
            input.style.backgroundColor = '';
        });
        this.currentPage = 1;
        this.applyFiltersAndSort();
        this.render();
    }
}