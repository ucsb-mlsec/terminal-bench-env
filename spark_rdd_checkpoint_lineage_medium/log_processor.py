from pyspark import SparkContext

sc = SparkContext("local[2]", "LogProcessor")

# Load data
logs = sc.textFile("/workspace/data/access.log")

# Transformation 1: Parse log lines into structured tuples
def parse_log(line):
    try:
        parts = line.split("|")
        if len(parts) == 6:
            timestamp, ip, method, url, status, response_time = parts
            return (timestamp, ip, method, url, int(status), int(response_time))
        return None
    except:
        return None

parsed_logs = logs.map(parse_log)

# Transformation 2: Filter out invalid/None entries
valid_logs = parsed_logs.filter(lambda x: x is not None)

# Transformation 3: Extract key metrics - map to (url, (status, response_time))
url_metrics = valid_logs.map(lambda x: (x[3], (x[4], x[5])))

# Transformation 4: Group by URL to aggregate metrics
grouped_by_url = url_metrics.groupByKey()

# Transformation 5: Calculate statistics per URL (count, avg response time, error count)
def calculate_stats(url_data):
    url, metrics_iter = url_data
    metrics = list(metrics_iter)
    total_requests = len(metrics)
    total_response_time = sum(m[1] for m in metrics)
    avg_response_time = total_response_time / total_requests if total_requests > 0 else 0
    error_count = sum(1 for m in metrics if m[0] >= 400)
    return (url, (total_requests, avg_response_time, error_count))

url_stats = grouped_by_url.map(calculate_stats)

# Transformation 6: Format output for readability
formatted_output = url_stats.map(lambda x: f"{x[0]}|requests={x[1][0]}|avg_time={x[1][1]:.2f}ms|errors={x[1][2]}")

# Save results
formatted_output.saveAsTextFile("/workspace/output/")

sc.stop()