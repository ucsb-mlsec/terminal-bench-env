# AWS Route53 Health Check Best Practices

## Timeout Configuration

The timeout value determines how long Route53 will wait for a response from your health check endpoint before considering it a failure. This is one of the most critical settings to configure correctly.

**Key Recommendations:**
- Set timeout based on your application's P95 or P99 response times from actual production metrics
- Timeout should be at least 2x the P95 response time to account for normal variance and avoid false positives
- AWS allows timeout values between 2 and 60 seconds
- Timeout must be less than the interval setting (AWS enforces this constraint)
- For applications with variable response times, err on the side of longer timeouts

**Example:** If your health endpoint P95 response time is 250ms, set timeout to at least 500ms (round up to 1 second for safety margin).

## Interval Configuration

The interval determines how frequently Route53 performs health checks against your endpoint. AWS offers two options:

**Available Options:**
- **10 seconds (fast interval):** Provides rapid detection of failures but generates more traffic and is more sensitive to transient issues
- **30 seconds (standard interval):** Recommended for most applications, provides good balance between detection speed and stability

**Recommendations:**
- Use 30-second intervals for most production applications to reduce false positives from transient network issues
- Only use 10-second intervals for mission-critical services where rapid failover (under 1 minute) is essential
- Consider that more frequent checks generate more load on your application
- Standard 30-second interval is sufficient for most SLAs and reduces unnecessary failovers

## Failure Threshold

The failure threshold determines how many consecutive failed health checks must occur before Route53 marks the endpoint as unhealthy.

**Key Considerations:**
- Set threshold to 3 or higher to avoid transient failures causing unnecessary failovers
- Calculate total detection time: threshold × interval = time before failover
- Higher thresholds provide more stability but slower failure detection
- Account for occasional performance spikes that might cause single check failures

**Example Calculations:**
- Threshold=2, Interval=30s → 60 seconds to detect failure (too sensitive)
- Threshold=3, Interval=30s → 90 seconds to detect failure (recommended)
- Threshold=3, Interval=10s → 30 seconds to detect failure (for critical services)

**Recommendations:**
- Use threshold of 3 as a minimum for production stability
- Consider threshold of 4-5 for applications with occasional performance variability
- Balance between avoiding false positives and acceptable recovery time for your SLA

## Health Endpoint Design

Your health check endpoint implementation significantly impacts the reliability of Route53 monitoring.

**Best Practices:**
- Keep health checks lightweight with target response time under 100ms
- Avoid expensive operations like database queries, external API calls, or complex computations
- Implement caching for health status checks that require external validation
- Add internal timeouts in your health check code (should be less than Route53 timeout)
- Return appropriate HTTP status codes (200 for healthy, 5xx for unhealthy)
- Include minimal response body to reduce network overhead

**Anti-Patterns to Avoid:**
- Deep database connection pool checks on every request
- Calling downstream services synchronously
- Complex validation logic that takes seconds to execute
- Loading large amounts of data to verify system state

## General Recommendations

- Monitor Route53 health check metrics in CloudWatch to track failure patterns
- Test health check configuration under realistic load conditions
- Review health check failures regularly to distinguish between true failures and false positives
- Document your health check strategy and configuration decisions
- Consider using calculated health checks to combine multiple endpoint checks
- Implement gradual degradation rather than binary healthy/unhealthy states when possible
- Plan for the trade-off between detection sensitivity and operational stability
- Update health check configuration as application performance characteristics change