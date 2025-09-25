/* notif.js - Standalone notification system */
class NotificationManager {
    constructor(options = {}) {
        // Timeout settings
        this.defaultTimeout = options.defaultTimeout || 4000;
        this.successTimeout = options.successTimeout || 2000;
        this.fadeTimeout = options.fadeTimeout || 300;

        // Layout settings
        this.topOffset = options.topOffset || 20;
        this.spacing = options.spacing || 60;
        this.slideOffset = options.slideOffset || -20;

        this.notifications = new Map(); // Map of id -> notification data
        this.nextId = 0;
        this.container = null;
        this.isReady = false;

        // Initialize when DOM is ready
        this._initWhenReady();
    }

    _initWhenReady() {
        if (document.body) {
            this._createContainer();
            this.isReady = true;
        } else if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => {
                this._createContainer();
                this.isReady = true;
            });
        } else {
            // DOM is ready but body might not be parsed yet
            setTimeout(() => this._initWhenReady(), 10);
        }
    }

    _createContainer() {
        if (this.container) return; // Already created

        this.container = document.createElement('div');
        this.container.className = 'notification-container';
        document.body.appendChild(this.container);
    }

    _ensureReady() {
        if (!this.isReady) {
            console.warn('NotificationManager not ready yet, initializing synchronously');
            this._createContainer();
            this.isReady = true;
        }
    }

    _createNotificationElement(id, type, message) {
        this._ensureReady();

        const element = document.createElement('div');
        element.className = `notification-indicator visible ${type}`;
        element.dataset.id = id;

        if (type === 'spinner') {
            element.innerHTML = `
                <span class="notification-spinner"></span>
                <span class="notification-text">${message}</span>
            `;
        } else if (type === 'pbar') {
            element.innerHTML = `
                <div class="notification-text">${message}</div>
                <div class="notification-progress-container">
                    <div class="notification-progress-bar" style="width: 0%"></div>
                </div>
            `;
        } else {
            element.innerHTML = `
                <span class="notification-text">${message}</span>
            `;
        }

        this.container.appendChild(element);
        this._updatePositions();
        return element;
    }

    _updatePositions() {
        if (!this.container) return;

        const notifications = Array.from(this.container.children);
        notifications.forEach((el, index) => {
            el.style.top = `${this.topOffset + index * this.spacing}px`;
        });
    }

    _removeNotification(id) {
        const notifData = this.notifications.get(id);
        if (!notifData) return;

        // Clear timeout if exists
        if (notifData.timeout) {
            clearTimeout(notifData.timeout);
        }

        // Remove element with fade animation
        notifData.element.style.opacity = '0';
        notifData.element.style.transform = `translateX(-50%) translateY(${this.slideOffset}px)`;

        setTimeout(() => {
            if (notifData.element.parentNode) {
                notifData.element.parentNode.removeChild(notifData.element);
            }
            this.notifications.delete(id);
            this._updatePositions();
        }, this.fadeTimeout);
    }

    /**
     * Show a regular notification message
     * @param {string} message - Message to display
     * @param {number} timeout - Auto-hide timeout in ms (default from constructor)
     */
    show(message, timeout = null) {
        console.log(message);

        const id = this.nextId++;
        const element = this._createNotificationElement(id, 'show', message);

        const hideTimeout = timeout || this.defaultTimeout;
        const timeoutId = setTimeout(() => {
            this._removeNotification(id);
        }, hideTimeout);

        this.notifications.set(id, {
            element,
            type: 'show',
            timeout: timeoutId
        });
    }

    /**
     * Show a persistent spinner with message
     * @param {string} message - Loading message to display
     * @returns {object} Spinner control object with .complete() method
     */
    spinner(message) {
        console.log(message);

        const id = this.nextId++;
        const element = this._createNotificationElement(id, 'spinner', message);

        this.notifications.set(id, {
            element,
            type: 'spinner',
            timeout: null
        });

        return {
            complete: () => {
                this._removeNotification(id);
            }
        };
    }

    /**
     * Show a persistent progress bar with message
     * @param {string} message - Loading message to display
     * @returns {object} Progress bar control object with .progress(value) method
     */
    pbar(message) {
        console.log(message);

        const id = this.nextId++;
        const element = this._createNotificationElement(id, 'pbar', message);

        this.notifications.set(id, {
            element,
            type: 'pbar',
            timeout: null
        });

        return {
            progress: (value) => {
                const progressBar = element.querySelector('.notification-progress-bar');
                if (progressBar) {
                    const percentage = Math.max(0, Math.min(100, value * 100));
                    progressBar.style.width = `${percentage}%`;
                }
            },
            complete: () => {
                this._removeNotification(id);
            }
        };
    }

    /**
     * Show success message
     * @param {string} message - Success message
     * @param {number} timeout - Auto-hide timeout in ms (default from constructor)
     */
    success(message, timeout = null) {
        console.log(message);

        const id = this.nextId++;
        const element = this._createNotificationElement(id, 'success', message);

        const hideTimeout = timeout || this.successTimeout;
        const timeoutId = setTimeout(() => {
            this._removeNotification(id);
        }, hideTimeout);

        this.notifications.set(id, {
            element,
            type: 'success',
            timeout: timeoutId
        });
    }

    /**
     * Show error message
     * @param {string} message - Error message
     * @param {Error} err - Error object to log
     * @param {number} timeout - Auto-hide timeout in ms (default from constructor)
     */
    error(message, err = null, timeout = null) {
        if (err) {
            console.error(err);
        } else {
            console.error(message);
        }

        const id = this.nextId++;
        const element = this._createNotificationElement(id, 'error', message);

        const hideTimeout = timeout || this.defaultTimeout;
        const timeoutId = setTimeout(() => {
            this._removeNotification(id);
        }, hideTimeout);

        this.notifications.set(id, {
            element,
            type: 'error',
            timeout: timeoutId
        });
    }

    /**
     * Hide all notifications
     */
    clear() {
        for (const [id] of this.notifications) {
            this._removeNotification(id);
        }
    }
}

// Initialize global instance when DOM is ready
let NOTIF;

// Initialize immediately if DOM is already ready, otherwise wait
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        NOTIF = new NotificationManager();
    });
} else {
    // DOM already loaded
    NOTIF = new NotificationManager();
}