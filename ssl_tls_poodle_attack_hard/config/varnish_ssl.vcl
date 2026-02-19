vcl 4.0;

# Varnish Cache SSL termination configuration
# SSL is terminated at backend - Varnish handles cached content

backend default {
    .host = "127.0.0.1";
    .port = "8080";
    .connect_timeout = 600s;
    .first_byte_timeout = 600s;
    .between_bytes_timeout = 600s;
}

backend ssl_backend {
    .host = "10.0.1.15";
    .port = "443";
    .connect_timeout = 300s;
}

sub vcl_recv {
    # Remove cookies for static content
    if (req.url ~ "\.(jpg|jpeg|gif|png|ico|css|zip|tgz|gz|rar|bz2|pdf|txt|tar|wav|bmp|rtf|js|flv|swf|html|htm)$") {
        unset req.http.Cookie;
    }
    
    # Force lookup for specific URLs
    if (req.url ~ "^/admin") {
        return (pass);
    }
    
    # Handle POST requests
    if (req.method == "POST") {
        return (pass);
    }
    
    return (hash);
}

sub vcl_backend_response {
    # Cache static content for 1 hour
    if (bereq.url ~ "\.(jpg|jpeg|gif|png|ico|css|js)$") {
        set beresp.ttl = 3600s;
        unset beresp.http.Set-Cookie;
    }
    
    # Don't cache admin pages
    if (bereq.url ~ "^/admin") {
        set beresp.ttl = 0s;
        set beresp.uncacheable = true;
        return (deliver);
    }
    
    return (deliver);
}

sub vcl_deliver {
    # Add cache hit/miss header for debugging
    if (obj.hits > 0) {
        set resp.http.X-Cache = "HIT";
    } else {
        set resp.http.X-Cache = "MISS";
    }
    
    # Remove backend headers
    unset resp.http.X-Varnish;
    unset resp.http.Via;
    
    return (deliver);
}