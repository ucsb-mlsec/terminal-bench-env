(function() {
    'use strict';

    /**
     * Load external analytics tracking script
     * Attempts to dynamically inject analytics script from third-party domain
     */
    function loadAnalytics() {
        console.log('Attempting to load analytics script...');
        try {
            const script = document.createElement('script');
            script.src = 'https://analytics.example.com/tracker.js';
            script.async = true;
            script.onload = function() {
                console.log('Analytics script loaded successfully');
            };
            script.onerror = function() {
                console.error('Failed to load analytics script');
            };
            document.head.appendChild(script);
        } catch (error) {
            console.error('Error loading analytics:', error);
        }
    }

    /**
     * Fetch dashboard data from external API
     * Makes cross-origin API call to backend service
     */
    function fetchDashboardData() {
        console.log('Fetching dashboard data...');
        fetch('https://api.backend.com/data', {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            console.log('Dashboard data received:', data);
            displayData(data);
        })
        .catch(error => {
            console.error('Error fetching dashboard data:', error);
        });
    }

    /**
     * Apply dynamic styles to page elements
     * Creates inline styles that may violate CSP style-src policy
     */
    function applyDynamicStyles() {
        console.log('Applying dynamic styles...');
        try {
            // Attempt to add inline style element
            const styleElement = document.createElement('style');
            styleElement.textContent = `
                .dashboard-widget {
                    background-color: #f0f0f0;
                    padding: 20px;
                    border-radius: 8px;
                }
            `;
            document.head.appendChild(styleElement);

            // Attempt to set inline style attribute
            const container = document.getElementById('main-container');
            if (container) {
                container.setAttribute('style', 'margin-top: 20px; padding: 15px;');
            }
            
            console.log('Dynamic styles applied successfully');
        } catch (error) {
            console.error('Error applying dynamic styles:', error);
        }
    }

    /**
     * Display fetched data in the UI
     */
    function displayData(data) {
        const container = document.getElementById('data-container');
        if (container) {
            container.innerHTML = '<div class="dashboard-widget">' + JSON.stringify(data) + '</div>';
        }
    }

    /**
     * Initialize application on DOM ready
     */
    document.addEventListener('DOMContentLoaded', function() {
        console.log('Application initializing...');
        
        // Load external analytics
        loadAnalytics();
        
        // Apply custom styling
        applyDynamicStyles();
        
        // Fetch data from API
        fetchDashboardData();
        
        console.log('Application initialization complete');
    });

})();