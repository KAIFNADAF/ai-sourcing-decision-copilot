# src/trace_logger.py

from __future__ import annotations

from datetime import datetime
from typing import Dict, Any

import pandas as pd

from src.workflow_store import WorkflowStore


class TraceLogger:
    """
    Lightweight wrapper around WorkflowStore for writing workflow trace events.
    Keeps trace creation consistent and avoids duplicating event-writing logic
    across validation, approval, orchestration, and recommendation steps.
    """

    def __init__(self, store: WorkflowStore) -> None:
        self.store = store

    def _next_trace_id(self) -> str:
        trace_df = self.store.load_trace()

        if trace_df.empty:
            return "TRC-00001"

        existing_ids = trace_df["trace_id"].dropna().astype(str).tolist()
        numeric_parts = []

        for trace_id in existing_ids:
            if trace_id.startswith("TRC-"):
                try:
                    numeric_parts.append(int(trace_id.split("-")[1]))
                except (IndexError, ValueError):
                    continue

        next_num = max(numeric_parts, default=0) + 1
        return f"TRC-{next_num:05d}"

    def log_event(
        self,
        workflow_run_id: str,
        request_id: str,
        stage_name: str,
        status: str,
        event_type: str,
        event_message: str,
        timestamp: str | None = None,
    ) -> Dict[str, Any]:
        """
        Append a trace event and return the created event payload.
        """
        event = {
            "trace_id": self._next_trace_id(),
            "workflow_run_id": workflow_run_id,
            "request_id": request_id,
            "stage_name": stage_name,
            "status": status,
            "event_type": event_type,
            "event_message": event_message,
            "timestamp": timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        self.store.append_trace_event(event)
        return event

    def log_stage_started(
        self,
        workflow_run_id: str,
        request_id: str,
        stage_name: str,
        message: str | None = None,
    ) -> Dict[str, Any]:
        return self.log_event(
            workflow_run_id=workflow_run_id,
            request_id=request_id,
            stage_name=stage_name,
            status="In Progress",
            event_type=f"{stage_name}_started",
            event_message=message or f"{stage_name} stage started.",
        )

    def log_stage_success(
        self,
        workflow_run_id: str,
        request_id: str,
        stage_name: str,
        message: str | None = None,
    ) -> Dict[str, Any]:
        return self.log_event(
            workflow_run_id=workflow_run_id,
            request_id=request_id,
            stage_name=stage_name,
            status="Success",
            event_type=f"{stage_name}_completed",
            event_message=message or f"{stage_name} stage completed successfully.",
        )

    def log_stage_failure(
        self,
        workflow_run_id: str,
        request_id: str,
        stage_name: str,
        message: str | None = None,
    ) -> Dict[str, Any]:
        return self.log_event(
            workflow_run_id=workflow_run_id,
            request_id=request_id,
            stage_name=stage_name,
            status="Fail",
            event_type=f"{stage_name}_failed",
            event_message=message or f"{stage_name} stage failed.",
        )

    def log_stage_pending(
        self,
        workflow_run_id: str,
        request_id: str,
        stage_name: str,
        message: str | None = None,
    ) -> Dict[str, Any]:
        return self.log_event(
            workflow_run_id=workflow_run_id,
            request_id=request_id,
            stage_name=stage_name,
            status="Pending",
            event_type=f"{stage_name}_pending",
            event_message=message or f"{stage_name} stage is pending.",
        )


if __name__ == "__main__":
    store = WorkflowStore(data_dir="data")
    logger = TraceLogger(store=store)

    sample_run = store.load_runs().iloc[0].to_dict()

    created = logger.log_event(
        workflow_run_id=sample_run["workflow_run_id"],
        request_id=sample_run["request_id"],
        stage_name="test_stage",
        status="Success",
        event_type="test_event",
        event_message="Trace logger smoke test.",
    )

    print("Created trace event:")
    print(created)