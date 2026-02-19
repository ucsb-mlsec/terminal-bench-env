// Web Application Client-Side JavaScript
// IIFE pattern to avoid polluting global scope
(function() {
    'use strict';

    // ==========================================
    // Utility Functions
    // ==========================================

    /**
     * Validate email format
     * @param {string} email - Email address to validate
     * @return {boolean} - True if valid email format
     */
    function validateEmail(email) {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return emailRegex.test(email);
    }

    /**
     * Format date to readable string
     * @param {Date} date - Date object to format
     * @return {string} - Formatted date string
     */
    function formatDate(date) {
        const options = { year: 'numeric', month: 'long', day: 'numeric' };
        return date.toLocaleDateString('en-US', options);
    }

    /**
     * Make AJAX request to server
     * @param {string} url - API endpoint URL
     * @param {object} data - Data to send
     * @param {function} callback - Success callback
     */
    function makeRequest(url, data, callback) {
        try {
            fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.json();
            })
            .then(data => callback(data))
            .catch(error => {
                console.error('Request failed:', error);
                showError('Failed to communicate with server');
            });
        } catch (error) {
            console.error('Request error:', error);
            showError('An unexpected error occurred');
        }
    }

    // ==========================================
    // DOM Manipulation Functions
    // ==========================================

    /**
     * Display error message to user
     * @param {string} message - Error message to display
     */
    function showError(message) {
        const errorDiv = document.getElementById('error-message');
        if (errorDiv) {
            errorDiv.textContent = message;
            errorDiv.style.display = 'block';
            setTimeout(() => {
                errorDiv.style.display = 'none';
            }, 5000);
        }
    }

    /**
     * Display success message to user
     * @param {string} message - Success message to display
     */
    function showSuccess(message) {
        const successDiv = document.getElementById('success-message');
        if (successDiv) {
            successDiv.textContent = message;
            successDiv.style.display = 'block';
            setTimeout(() => {
                successDiv.style.display = 'none';
            }, 3000);
        }
    }

    // ==========================================
    // Event Handlers
    // ==========================================

    /**
     * Handle form submission
     * @param {Event} event - Form submit event
     */
    function handleFormSubmit(event) {
        event.preventDefault();
        
        try {
            const form = event.target;
            const email = form.querySelector('#email').value;
            const name = form.querySelector('#name').value;

            // Validate form data
            if (!name || name.trim().length < 2) {
                showError('Please enter a valid name');
                return;
            }

            if (!validateEmail(email)) {
                showError('Please enter a valid email address');
                return;
            }

            // Submit data
            const formData = {
                name: name.trim(),
                email: email.trim(),
                timestamp: new Date().toISOString()
            };

            makeRequest('/api/submit', formData, function(response) {
                showSuccess('Form submitted successfully!');
                form.reset();
            });
        } catch (error) {
            console.error('Form submission error:', error);
            showError('Failed to submit form');
        }
    }

    /**
     * Handle button click events
     * @param {Event} event - Click event
     */
    function handleButtonClick(event) {
        const button = event.target;
        const action = button.dataset.action;

        try {
            if (action === 'load-data') {
                makeRequest('/api/data', {}, function(data) {
                    displayData(data);
                });
            } else if (action === 'clear') {
                clearDisplay();
            }
        } catch (error) {
            console.error('Button click error:', error);
            showError('Action failed');
        }
    }

    function displayData(data) {
        const container = document.getElementById('data-container');
        if (container && data) {
            container.innerHTML = '<p>' + JSON.stringify(data) + '</p>';
        }
    }

    function clearDisplay() {
        const container = document.getElementById('data-container');
        if (container) {
            container.innerHTML = '';
        }
    }

    // ==========================================
    // Initialization
    // ==========================================

    /**
     * Initialize application when DOM is ready
     */
    function init() {
        // Attach form submit handler
        const form = document.querySelector('#main-form');
        if (form) {
            form.addEventListener('submit', handleFormSubmit);
        }

        // Attach button click handlers
        const buttons = document.querySelectorAll('button[data-action]');
        buttons.forEach(button => {
            button.addEventListener('click', handleButtonClick);
        });

        console.log('Application initialized successfully');
    }

    // Wait for DOM to be ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

})();