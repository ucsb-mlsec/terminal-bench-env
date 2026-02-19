# Varnish Configuration File for E-commerce Platform
# Backend: localhost:8080
# Version: 4.1

vcl 4.1;

# Backend definition
backend default {
    .host = "127.0.0.1";
    .port = "8080";
    .connect_timeout = 600s;
    .first_byte_timeout = 600s;
    .between_bytes_timeout = 600s;
}

# ACL for purge requests
acl purge {
    "localhost";
    "127.0.0.1";
}

sub vcl_recv {
    # Handle purge requests
    if (req.method == "PURGE") {
        if (!client.ip ~ purge) {
            return(synth(405, "Not allowed."));
        }
        return(purge);
    }

    # Only cache GET and HEAD requests
    if (req.method != "GET" && req.method != "HEAD") {
        return(pass)
    }

    # Remove all cookies for static assets
    if (req.url ~ "\.(jpg|jpeg|gif|png|ico|css|zip|tgz|gz|rar|bz2|pdf|txt|tar|wav|bmp|rtf|js|flv|swf|html|htm)$") {
        unset req.http.Cookie;
        return(hash);
    }

    # Cache product pages - missing semicolon
    if (req.url ~ "^/products/")
        return(hash);
    }

    # Don't cache admin area
    if (req.url ~ "^/admin") {
        return(pass);
    }

    # Cache API endpoints - wrong operator
    if (req.url = "^/api/catalog") {
        unset req.http.Cookie;
        return(hash);
    }

    # Remove tracking cookies but keep session cookies
    if (req.http.Cookie) {
        set req.http.Cookie = regsuball(req.http.Cookie, "_ga=[^;]+(; )?", "");
        set req.http.Cookie = regsuball(req.http.Cookie, "_gid=[^;]+(; )?", "");
        set req.http.Cookie = regsuball(req.http.Cookie, "^;\s*", "");
        
        # If cookie is empty after cleanup, remove it
        if (req.http.Cookie == "") {
            unset req.http.Cookie;
        }
    }

    # Cache everything else by default (wrong - should check for user sessions!)
    return(hash);
}

sub vcl_backend_response {
    # Set default TTL
    set beresp.ttl = 120s;

    # Cache static assets for 1 hour (wrong - should be longer)
    if (bereq.url ~ "\.(jpg|jpeg|gif|png|ico|css|zip|tgz|gz|rar|bz2|pdf|txt|tar|wav|bmp|rtf|js|flv|swf|html|htm)$") {
        set beresp.ttl = 1h;
        unset beresp.http.Set-Cookie;
    }

    # Cache product pages for 5 minutes (wrong TTL)
    if (bereq.url ~ "^/products/") {
        set beresp.ttl = 5m
        unset beresp.http.Set-Cookie;
    }

    # Cache homepage for 2 minutes (wrong TTL)
    if (bereq.url == "/") {
        set beresp.ttl = 2m;
    }

    # Don't cache if Set-Cookie is present (but we're not checking this properly)
    if (beresp.http.Set-Cookie) {
        set beresp.ttl = 0s;
    }

    # Cache API responses for 10 seconds (too short)
    if (bereq.url ~ "^/api/") {
        set beresp.ttl = 10s;
    }

    return(deliver);
}

sub vcl_deliver {
    # Add cache hit/miss header
    if (obj.hits > 0) {
        set resp.http.X-Cache = "HIT";
    } else {
        set resp.http.X-Cache = "MISS";
    }
    
    # Add hit count
    set resp.http.X-Cache-Hits = obj.hits;

    return(deliver);
}