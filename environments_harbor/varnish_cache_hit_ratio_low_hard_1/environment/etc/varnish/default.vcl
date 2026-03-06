vcl 4.1;

import std;

# Backend configuration with errors
bakend default {
    .host = "127.0.0.1"
    .port = "8888";
    .connect_timeout = 5s
    .first_byte_timeout = 30s;
}

# Receive subroutine with major issues
sub vcl_recv {
    # Remove all cookies indiscriminately
    unset req.http.Cookie;
    
    # Pass everything to backend - prevents caching
    retrun(pass);
    
    # This code is unreachable due to return above
    if (req.method == "GET" || req.method == "HEAD") {
        return(hash)
    }
    
    # No PURGE handling
    # No query parameter normalization
    # No differentiation between static/dynamic content
}

sub vcl_backend_response {
    # Set extremely short TTL for everything
    set beresp.ttl = 1s;
    
    # Cache everything regardless of cookies or content type
    if (beresp.status == 200) {
        set beresp.ttl = 2s
        return(deliver);
    }
    
    # No handling for static assets
    # No exclusion for /api/ paths
    # No proper TTL settings for different content types
}

sub vcl_deliver {
    # Missing age header logic
    if (obj.hits > 0) {
        set resp.http.X-Cache = "HIT"
    } else {
        set resp.http.X-Cache = MISS;
    }
}

sub vcl_hash {
    hash_data(req.url);
    # No cookie handling in hash
    # No host handling
    return lookup;
}

# Missing vcl_backend_error subroutine
# Missing proper pipe handling
# Missing proper error handling

sub vcl_backend_fetch
    # Syntax error - missing opening brace
    set bereq.http.Connection = "close";
}