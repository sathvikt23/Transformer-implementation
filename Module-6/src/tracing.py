import os
import json
import time
from contextlib import contextmanager
from typing import Generator, Optional, Dict, Any

# OpenTelemetry imports guarded by availability & flags
_OTEL_AVAILABLE = False
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False

_TRACING_ENABLED = True
_TRACE_FILE = "otel_traces.jsonl"


def set_tracing_enabled(enabled: bool):
    global _TRACING_ENABLED
    _TRACING_ENABLED = enabled


def set_trace_file(file_path: str):
    global _TRACE_FILE
    _TRACE_FILE = file_path


class FileSpanExporter:
    """
    In-process local file exporter writing OpenTelemetry-style span records to JSONL.
    No network dependency or external collector required.
    """

    def __init__(self, filepath: str):
        self.filepath = filepath

    def export(self, span_data: Dict[str, Any]):
        if not _TRACING_ENABLED:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.filepath)), exist_ok=True)
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(span_data) + "\n")


_FILE_EXPORTER: Optional[FileSpanExporter] = None


def init_tracer(output_filepath: str = "otel_traces.jsonl"):
    global _FILE_EXPORTER
    _FILE_EXPORTER = FileSpanExporter(output_filepath)


@contextmanager
def span(name: str, **attributes: Any) -> Generator[Dict[str, Optional[str]], None, None]:
    """
    Thin, no-op-safe span wrapper for OpenTelemetry tracing.
    Returns dict containing 'trace_id' and 'span_id' strings for non-authoritative ledger attribution.
    """
    if not _TRACING_ENABLED:
        yield {"trace_id": None, "span_id": None}
        return

    import uuid
    trace_id = uuid.uuid4().hex
    span_id = uuid.uuid4().hex[:16]

    start_time = time.time()

    span_ctx = {
        "trace_id": trace_id,
        "span_id": span_id,
    }

    try:
        yield span_ctx
    finally:
        end_time = time.time()
        span_record = {
            "trace_id": trace_id,
            "span_id": span_id,
            "parent_span_id": None,
            "name": name,
            "start": start_time,
            "end": end_time,
            "duration_ms": (end_time - start_time) * 1000.0,
            "attributes": attributes,
        }
        if _FILE_EXPORTER:
            _FILE_EXPORTER.export(span_record)
