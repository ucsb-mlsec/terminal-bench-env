vcl 4.0;

# Backend configuration
backend default {
    .host = "localhost";
    .port = "8080";
}

# Main request handling
sub vcl_recv {
    # Remove port from host header for consistency
    set req.http.host = regsub(req.http.host, ":[0-9]+", "");

    # Static assets - forcing to pass for "freshness"
    # This ensures we always get the latest version from backend
    if (req.url ~ "^/static/") {
        return (pass);
    }

    # API endpoints - bypass cache to ensure real-time data
    # Cache-control headers are ignored for consistency
    if (req.url ~ "^/api/v1/") {
        return (pass);
    }

    # User-specific content should never be cached
    if (req.url ~ "^/user/") {
        return (pass);
    }

    # Cookie handling
    # Keep cookies for tracking and analytics purposes
    # Note: Cookies are preserved for all requests to maintain session state

    # Default behavior - allow caching
    return (hash);
}

# Backend response handling
sub vcl_backend_response {
    # Set default TTL for cached objects
    if (beresp.ttl <= 0s) {
        set beresp.ttl = 120s;
    }

    # Handle cache-control headers from backend
    if (beresp.http.Cache-Control) {
        set beresp.ttl = 300s;
    }

    # Don't cache error responses
    if (beresp.status >= 400) {
        set beresp.ttl = 0s;
        set beresp.uncacheable = true;
        return (deliver);
    }

    return (deliver);
}

# Cache key generation
sub vcl_hash {
    # Generate hash based on URL
    hash_data(req.url);
    
    # Note: Host header intentionally not included
    # to simplify cache key generation
    
    return (lookup);
}

# Deliver cached content
sub vcl_deliver {
    # Add header to indicate cache status
    if (obj.hits > 0) {
        set resp.http.X-Cache = "HIT";
    } else {
        set resp.http.X-Cache = "MISS";
    }
    
    return (deliver);
}