#!/usr/bin/awk -f

# Log Correlation Analysis Script
# Purpose: Track request flows across microservices and detect anomalies
# Usage: awk -f correlate_logs.awk auth.log api.log database.log

BEGIN {
    # Initialize counters and tracking arrays
    FS = " "
    
    # Track correlation IDs and their details
    # corr_services[corr_id] = list of services seen
    # corr_status[corr_id, service] = status for that service
    # corr_lines[corr_id, service] = line number where seen
    # corr_files[corr_id, service] = filename where seen
    
    print "Starting log correlation analysis..." > "/dev/stderr"
}

{
    # Skip empty lines or malformed entries
    if (NF < 5) next
    
    # Parse log entry fields
    timestamp = $1
    corr_id = $2
    service = $3
    status = $4
    message = $5
    
    # Track line number for this entry
    # Store the current line number for reference
    line_num = NR  # Track which line this entry is on
    
    # Store the filename for cross-reference
    current_file = FILENAME
    
    # Initialize services list if this is first time seeing this correlation ID
    if (!(corr_id in seen_corr_ids)) {
        seen_corr_ids[corr_id] = 1
        corr_services[corr_id] = ""
        corr_count[corr_id] = 0
    }
    
    # Track which services this correlation ID has appeared in
    service_key = corr_id SUBSEP service
    if (!(service_key in service_seen)) {
        service_seen[service_key] = 1
        if (corr_services[corr_id] == "") {
            corr_services[corr_id] = service
        } else {
            corr_services[corr_id] = corr_services[corr_id] "," service
        }
        corr_count[corr_id]++
    }
    
    # Store status for this correlation ID and service
    corr_status[corr_id, service] = status
    
    # Store line number and file for debugging
    corr_lines[corr_id, service] = line_num
    corr_files[corr_id, service] = current_file
    
    # Track all unique services we've seen
    all_services[service] = 1
}

END {
    # Analysis phase - identify anomalies
    print "Analysis complete. Processing anomalies..." > "/dev/stderr"
    
    # Expected services for a complete flow
    expected_services["auth"] = 1
    expected_services["api"] = 1
    expected_services["database"] = 1
    
    # Check each correlation ID for anomalies
    for (corr_id in seen_corr_ids) {
        
        # Check for incomplete flows (not in all three services)
        services_present = 0
        has_auth = 0
        has_api = 0
        has_database = 0
        
        if ((corr_id, "auth") in corr_status) {
            has_auth = 1
            services_present++
        }
        if ((corr_id, "api") in corr_status) {
            has_api = 1
            services_present++
        }
        if ((corr_id, "database") in corr_status) {
            has_database = 1
            services_present++
        }
        
        # Detect incomplete flows
        if (services_present < 3) {
            anomalies[corr_id] = "INCOMPLETE_FLOW|" corr_services[corr_id]
            continue
        }
        
        # Check for status mismatches
        # All services should have SUCCESS status for normal flow
        auth_status = corr_status[corr_id, "auth"]
        api_status = corr_status[corr_id, "api"]
        db_status = corr_status[corr_id, "database"]
        
        if (auth_status != api_status || api_status != db_status || auth_status != db_status) {
            # Status mismatch detected
            anomalies[corr_id] = "STATUS_MISMATCH|" corr_services[corr_id]
            continue
        }
        
        # Check for pattern violations (e.g., all failures)
        if (auth_status == "FAILURE" && api_status == "FAILURE" && db_status == "FAILURE") {
            anomalies[corr_id] = "PATTERN_VIOLATION|" corr_services[corr_id]
        }
    }
    
    # Output anomalies in sorted order
    n = asorti(anomalies, sorted_ids)
    for (i = 1; i <= n; i++) {
        corr_id = sorted_ids[i]
        print corr_id "|" anomalies[corr_id]
    }
}