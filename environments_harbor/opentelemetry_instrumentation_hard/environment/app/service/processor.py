#!/usr/bin/env python3

import os
import json

# Read the original processor.py
with open('/app/service/processor.py', 'r') as f:
    original_code = f.read()

# Create the instrumented version
instrumented_code = '''import time
import json
import os
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
import atexit

class JSONFileSpanExporter(SpanExporter):
    """Custom exporter that writes spans to a JSON file."""
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.spans = []
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    def export(self, spans):
        """Export spans to the file."""
        for span in spans:
            span_dict = {
                "name": span.name,
                "context": {
                    "trace_id": format(span.context.trace_id, '032x'),
                    "span_id": format(span.context.span_id, '016x'),
                    "trace_state": str(span.context.trace_state)
                },
                "kind": str(span.kind),
                "parent_id": format(span.parent.span_id, '016x') if span.parent else None,
                "start_time": span.start_time,
                "end_time": span.end_time,
                "status": {
                    "status_code": str(span.status.status_code)
                },
                "attributes": dict(span.attributes) if span.attributes else {},
                "events": [
                    {
                        "name": event.name,
                        "timestamp": event.timestamp,
                        "attributes": dict(event.attributes) if event.attributes else {}
                    }
                    for event in span.events
                ] if span.events else [],
                "resource": dict(span.resource.attributes) if span.resource else {}
            }
            self.spans.append(span_dict)
        
        return SpanExportResult.SUCCESS
    
    def shutdown(self):
        """Write all spans to file on shutdown."""
        self.force_flush()
    
    def force_flush(self, timeout_millis=30000):
        """Force flush all spans to file."""
        output = {
            "resource_spans": self.spans,
            "spans_exported": len(self.spans)
        }
        
        with open(self.file_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        return True

# Initialize OpenTelemetry
resource = Resource.create({"service.name": "data-processor"})
provider = TracerProvider(resource=resource)

# Create and configure the JSON file exporter
exporter = JSONFileSpanExporter('/app/output/traces.json')
processor = SimpleSpanProcessor(exporter)
provider.add_span_processor(processor)

# Set the global tracer provider
trace.set_tracer_provider(provider)

# Get a tracer
tracer = trace.get_tracer(__name__)

# Register shutdown hook to ensure spans are flushed
def shutdown_tracer():
    provider.force_flush()
    provider.shutdown()

atexit.register(shutdown_tracer)

def process_request(data):
    """Process incoming data and return result."""
    with tracer.start_as_current_span("process_request") as span:
        span.set_attribute("data.type", type(data).__name__)
        
        # Simulate some processing work
        time.sleep(0.1)
        
        # Parse and validate data
        parsed = validate_data(data)
        
        # Transform the data
        result = transform_data(parsed)
        
        span.set_attribute("processing.status", "success")
        
        return result

def validate_data(data):
    """Validate input data."""
    with tracer.start_as_current_span("validate_data") as span:
        time.sleep(0.05)
        if not isinstance(data, dict):
            span.set_attribute("validation.error", "Data must be a dictionary")
            raise ValueError("Data must be a dictionary")
        
        span.set_attribute("validation.status", "valid")
        return data

def transform_data(data):
    """Transform data to output format."""
    with tracer.start_as_current_span("transform_data") as span:
        time.sleep(0.05)
        result = {
            "status": "processed",
            "original": data,
            "timestamp": time.time()
        }
        span.set_attribute("transform.status", "completed")
        return result

if __name__ == "__main__":
    # Test the service
    test_data = {"user_id": "12345", "action": "login"}
    result = process_request(test_data)
    print(json.dumps(result, indent=2))
    
    # Ensure traces are flushed
    shutdown_tracer()
'''

# Write the instrumented code back to the file
with open('/app/service/processor.py', 'w') as f:
    f.write(instrumented_code)

print("Instrumentation complete. Modified /app/service/processor.py with OpenTelemetry tracing.")