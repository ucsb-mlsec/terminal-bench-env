vcl 4.1;

# Varnish 6.x Configuration File
# WARNING: This configuration has known issues

# Backend definition - connecting to application server
backend default {
    .host = "localhost"
    .port = "8080"
    .connect_timeout = 5s
    .first_byte_timeout = 30s
    .between_bytes_timeout = 10s
    # Missing probe configuration
}

# Handle incoming requests
sub vcl_recv {
    # Remove port from host header
    set req.http.Host = regsub(req.http.Host, ":[0-9]+", "")
    
    # Handle different request methods
    if (req.method != "GET" && req.method != "HEAD") {
        return (pass)
    }
    
    # Static resources - BROKEN: not removing cookies
    if (req.url ~ "^/static/" || req.url ~ "^/assets/") {
        # BUG: Should strip cookies here but missing
        # BUG: Should normalize query parameters but missing
        return (hash);
    }
    
    # Product pages
    if (req.url ~ "^/products/") {
        # Missing cookie handling
        return (hash)
    }
    
    # User specific pages - BROKEN: missing bypass logic
    # BUG: These should never be cached but no bypass rule exists
    
    # API endpoints - BROKEN: wrong handling
    if (req.url ~ "^/api/v1/") {
        # BUG: Should pass but might cache instead
        return (hash)
    }
    
    # Default behavior
    if (req.http.Authorization) {
        return (pass);
    }
    
    return (hash)
}

# Handle backend responses
sub vcl_backend_response {
    # Static resources - BROKEN: TTL too short
    if (bereq.url ~ "^/static/" || bereq.url ~ "^/assets/") {
        set beresp.ttl = 10s;  # BUG: Should be 86400s (1 day)
        set beresp.grace = 30s;  # BUG: Should be 7200s (2 hours)
    }
    
    # Product pages - BROKEN: wrong TTL
    if (bereq.url ~ "^/products/") {
        set beresp.ttl = 3600s  # BUG: Should be 900s (15 minutes), also missing semicolon
        set beresp.grace = 60s  # BUG: Should be 7200s
    }
    
    # Default TTL - too aggressive
    set beresp.ttl = 5s;
    
    # BUG: Missing grace mode configuration for other content
    
    # Handle cookies - BROKEN
    if (beresp.http.Set-Cookie) {
        # BUG: Should handle this differently
        set beresp.uncacheable = true;
        return (deliver);
    }
    
    return (deliver);
}

# Deliver to client - MISSING proper implementation
sub vcl_deliver {
    # Add debug header
    if (obj.hits > 0) {
        set resp.http.X-Cache = "HIT"
    } else {
        set resp.http.X-Cache = "MISS";
    }
    
    # BUG: Missing Vary header handling
    # BUG: Not properly handling cache status
    
    return (deliver);
}

# Hash generation - BROKEN or missing proper implementation
sub vcl_hash {
    hash_data(req.url);
    
    # BUG: Missing host header in hash
    # BUG: Not normalizing URL properly
    
    return (lookup);
}

# Handling purge requests - missing implementation
# BUG: No vcl_purge subroutine

# Hit handling - using deprecated syntax
sub vcl_hit {
    # BUG: Incorrect logic
    if (req.method == "PURGE") {
        return (synth(200, "Purged"))
    }
    return (deliver)
}

# Miss handling
sub vcl_miss {
    if (req.method == "PURGE") {
        return (synth(404, "Not in cache"));
    }
    return (fetch);
}

# Synthetic responses
sub vcl_synth {
    set resp.http.Content-Type = "text/html; charset=utf-8";
    synthetic("Error: " + resp.status);
    return (deliver);
}