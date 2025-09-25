
class DataTable {
    constructor(container, config = {}) {
        this.container = typeof container === 'string' ? document.querySelector(container) : container;
        if (!this.container) {
            throw new Error(`DataTable container not found: ${container}`);
        }
        this.data = config.data || [];
        this.columns = config.columns || [];
        this.pageSizeOptions = config.pageSizeOptions || [10, 25, 50, 100];
        this.pageSize = config.pageSize || this.pageSizeOptions[0];
        this.showFilters = config.showFilters !== false; // Default to true
        this.currentPage = 1;
        this.sortColumn = null;
        this.sortDirection = null;
        this.filters = {};
        this.filteredData = [];
        this.paginationListeners = []; // Store listeners for cleanup
        if (this.columns.length === 0 && this.data.length > 0) {
            this.columns = Object.keys(this.data[0]).map(key => ({
                key: key,
                label: key.charAt(0).toUpperCase() + key.slice(1),
                type: this.inferType(this.data[0][key])
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

    init() {
        this.createTableStructure();
        this.applyFiltersAndSort();
        this.render();
    }

    createTableStructure() {
        this.container.innerHTML = '';

        // Create wrapper with border
        const wrapper = document.createElement('div');
        wrapper.className = 'datatable-wrapper';
        wrapper.style.border = '1px solid #ccc';

        // Create table container
        const tableContainer = document.createElement('div');
        tableContainer.className = 'datatable-container';

        const table = document.createElement('table');
        table.className = 'datatable';

        // Create header
        const thead = document.createElement('thead');

        // Header row with column names, sort, and resize handles
        const headerRow = document.createElement('tr');
        this.columns.forEach((col, index) => {
            const th = document.createElement('th');
            th.className = 'datatable-header';
            th.style.position = 'relative';
            if (col.width) {
                th.style.width = col.width;
                th.style.minWidth = col.width;
            } else {
                th.style.minWidth = '10px';
            }

            // Apply column alignment
            if (col.align) {
                th.style.textAlign = col.align;
            }

            const headerContent = document.createElement('div');
            headerContent.className = 'datatable-header-content';
            headerContent.style.display = 'flex';
            headerContent.style.justifyContent = 'space-between';
            headerContent.style.alignItems = 'center';

            const label = document.createElement('span');
            label.textContent = col.label || col.key;
            headerContent.appendChild(label);

            const sortIcon = document.createElement('span');
            sortIcon.className = 'datatable-sort-icon';
            sortIcon.textContent = ' ↕';
            headerContent.appendChild(sortIcon);

            headerContent.onclick = () => this.handleSort(col.key);

            // Add resize handle
            const resizeHandle = document.createElement('div');
            resizeHandle.style.position = 'absolute';
            resizeHandle.style.right = '0';
            resizeHandle.style.top = '0';
            resizeHandle.style.bottom = '0';
            resizeHandle.style.width = '4px';
            resizeHandle.style.cursor = 'col-resize';
            resizeHandle.style.background = 'transparent';

            this.addResizeListener(resizeHandle, th, index);

            th.appendChild(headerContent);
            th.appendChild(resizeHandle);
            headerRow.appendChild(th);
        });
        thead.appendChild(headerRow);

        // Filter row (conditional)
        if (this.showFilters) {
            const filterRow = document.createElement('tr');
            filterRow.className = 'datatable-filter-row';

            this.columns.forEach(col => {
                const td = document.createElement('td');
                td.className = 'datatable-filter-cell';

                // Apply column alignment to filter cell
                if (col.align) {
                    td.style.textAlign = col.align;
                }

                const filterContainer = document.createElement('div');
                filterContainer.style.display = 'flex';
                filterContainer.style.alignItems = 'center';

                const input = document.createElement('input');
                input.className = 'datatable-filter-input';
                input.type = 'text';
                input.placeholder = col.type === 'number' ? 'e.g. >50' : 'Filter...';
                input.style.flex = '1';

                const clearBtn = document.createElement('button');
                clearBtn.textContent = '×';
                clearBtn.style.border = 'none';
                clearBtn.style.background = 'none';
                clearBtn.style.cursor = 'pointer';
                clearBtn.style.padding = '2px 8px';
                clearBtn.style.fontSize = '16px';
                clearBtn.style.color = '#999';
                clearBtn.style.marginLeft = '4px';

                input.addEventListener('input', (e) => {
                    this.handleFilter(col.key, e.target.value, col.type, input);
                });

                clearBtn.addEventListener('click', () => {
                    this.clearFilter(col.key, input);
                });

                filterContainer.appendChild(input);
                filterContainer.appendChild(clearBtn);
                td.appendChild(filterContainer);
                filterRow.appendChild(td);
            });
            thead.appendChild(filterRow);
        }

        table.appendChild(thead);

        // Create body
        const tbody = document.createElement('tbody');
        tbody.className = 'datatable-body';
        table.appendChild(tbody);

        tableContainer.appendChild(table);
        wrapper.appendChild(tableContainer);

        // Create bottom pagination with page size selector
        const paginationBottom = document.createElement('div');
        paginationBottom.className = 'datatable-pagination datatable-pagination-bottom';
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
            startX = e.clientX;
            startWidth = parseInt(document.defaultView.getComputedStyle(th).width, 10);
            document.addEventListener('mousemove', doDrag);
            document.addEventListener('mouseup', stopDrag);
        });

        const doDrag = (e) => {
            const width = startWidth + e.clientX - startX;
            th.style.width = width + 'px';
            th.style.minWidth = width + 'px';
            // Update column width in config
            this.columns[columnIndex].width = width + 'px';
        };

        const stopDrag = () => {
            document.removeEventListener('mousemove', doDrag);
            document.removeEventListener('mouseup', stopDrag);
        };
    }

    handleSort(columnKey) {
        if (this.sortColumn === columnKey) {
            // Cycle through: asc -> desc -> none
            if (this.sortDirection === 'asc') {
                this.sortDirection = 'desc';
            } else if (this.sortDirection === 'desc') {
                this.sortColumn = null;
                this.sortDirection = null;
            }
        } else {
            this.sortColumn = columnKey;
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

    handleFilter(columnKey, filterValue, type, inputElement) {
        if (!filterValue) {
            delete this.filters[columnKey];
            inputElement.style.backgroundColor = '';
        } else {
            let isValid = true;

            if (type === 'number') {
                const numFilter = this.parseNumericFilter(filterValue);
                isValid = numFilter !== null;
            }

            this.filters[columnKey] = {
                value: filterValue,
                type: type,
                valid: isValid
            };

            inputElement.style.backgroundColor = isValid ? '' : '#ffcccc';
        }

        this.currentPage = 1;
        this.applyFiltersAndSort();
        this.validateCurrentPage();
        this.render();
    }

    clearFilter(columnKey, inputElement) {
        inputElement.value = '';
        delete this.filters[columnKey];
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
            for (const [key, filter] of Object.entries(this.filters)) {
                if (!filter.valid) continue;

                const cellValue = row[key];

                if (filter.type === 'number') {
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
        if (this.sortColumn) {
            filtered.sort((a, b) => {
                const aVal = a[this.sortColumn];
                const bVal = b[this.sortColumn];

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
    }

    updateSortIcons() {
        const headers = this.headerRow.querySelectorAll('th');
        headers.forEach((th, index) => {
            const col = this.columns[index];
            const sortIcon = th.querySelector('.datatable-sort-icon');
            if (!sortIcon) return;

            if (this.sortColumn === col.key) {
                if (this.sortDirection === 'asc') {
                    sortIcon.textContent = ' ↑';
                } else if (this.sortDirection === 'desc') {
                    sortIcon.textContent = ' ↓';
                }
            } else {
                sortIcon.textContent = ' ↕';
            }
        });
    }

    renderTable() {
        this.tbody.innerHTML = '';

        const pageData = this.getPageData();

        if (pageData.length === 0) {
            const tr = document.createElement('tr');
            const td = document.createElement('td');
            td.colSpan = this.columns.length;
            td.className = 'datatable-empty';
            td.textContent = 'No data to display';
            tr.appendChild(td);
            this.tbody.appendChild(tr);
            return;
        }

        pageData.forEach(row => {
            const tr = document.createElement('tr');

            this.columns.forEach(col => {
                const td = document.createElement('td');
                const value = row[col.key];

                // Apply column alignment
                if (col.align) {
                    td.style.textAlign = col.align;
                }

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
        const pageSizeSelect = this.paginationBottom.querySelector('.page-size-select');
        if (pageSizeSelect) {
            const handler = (e) => {
                this.setPageSize(parseInt(e.target.value));
            };
            pageSizeSelect.addEventListener('change', handler);
            this.paginationListeners.push({ element: pageSizeSelect, event: 'change', handler });
        }

        // Add export CSV button listener
        const exportBtn = this.paginationBottom.querySelector('.export-csv-btn');
        if (exportBtn) {
            const handler = () => {
                this.exportAndDownloadCSV();
            };
            exportBtn.addEventListener('click', handler);
            this.paginationListeners.push({ element: exportBtn, event: 'click', handler });
        }

        // Add clear filters button listener
        const clearBtn = this.paginationBottom.querySelector('.clear-filters-btn');
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
        html += `<button class="export-csv-btn">Export CSV</button>`;
        if (this.showFilters) {
            html += `<button class="clear-filters-btn">Clear Filters</button>`;
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
        html += `<button style="min-width: 30px; padding: 4px 8px;" ${this.currentPage === 1 ? 'disabled' : ''} data-page="${this.currentPage - 1}">‹</button>`;

        // Page buttons - always show 10 elements for consistent width
        const pageButtons = this.generatePageButtons(totalPages);
        pageButtons.forEach(item => {
            if (item.type === 'page') {
                const isActive = item.page === this.currentPage;
                html += `<button style="min-width: 30px; padding: 4px 8px; ${isActive ? 'background: #ccc;' : ''}" ${isActive ? 'disabled' : ''} data-page="${item.page}">${item.page}</button>`;
            } else if (item.type === 'dots') {
                html += '<span style="min-width: 30px; padding: 4px 8px; text-align: center; display: inline-block;">…</span>';
            }
        });

        // Next button
        html += `<button style="min-width: 30px; padding: 4px 8px;" ${this.currentPage === totalPages ? 'disabled' : ''} data-page="${this.currentPage + 1}">›</button>`;

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

    createPageSizeSelector() {
        const options = this.pageSizeOptions.map(size =>
            `<option value="${size}" ${this.pageSize === size ? 'selected' : ''}>${size}</option>`
        ).join('');

        return `<label>Show
            <select class="page-size-select">
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
        a.download = 'table-data.csv';
        a.click();
        URL.revokeObjectURL(url);
    }

    clearAllFilters() {
        this.filters = {};
        this.container.querySelectorAll('.datatable-filter-input').forEach(input => {
            input.value = '';
            input.style.backgroundColor = '';
        });
        this.currentPage = 1;
        this.applyFiltersAndSort();
        this.render();
    }
}