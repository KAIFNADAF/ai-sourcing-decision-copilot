from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd


class WorkflowStore:
    """
    CSV-backed persistence layer for the orchestrator system.
    This keeps file access, row lookup, inserts, and updates separate
    from workflow logic.
    """

    def __init__(self, data_dir: str = "data") -> None:
        self.data_dir = Path(data_dir)

        self.requests_path = self.data_dir / "sourcing_requests.csv"
        self.runs_path = self.data_dir / "workflow_runs.csv"
        self.validation_path = self.data_dir / "validation_results.csv"
        self.approvals_path = self.data_dir / "approvals.csv"
        self.trace_path = self.data_dir / "workflow_trace.csv"
        self.artifacts_path = self.data_dir / "artifacts.csv"
        self.recommendations_path = self.data_dir / "recommendation_outputs.csv"

        self._validate_paths()

    def _validate_paths(self) -> None:
        required_paths = [
            self.requests_path,
            self.runs_path,
            self.validation_path,
            self.approvals_path,
            self.trace_path,
            self.artifacts_path,
            self.recommendations_path,
        ]

        missing = [str(path) for path in required_paths if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing required workflow dataset files:\n" + "\n".join(missing)
            )

    def _read_csv(self, path: Path) -> pd.DataFrame:
        return pd.read_csv(path)

    def _write_csv(self, df: pd.DataFrame, path: Path) -> None:
        df.to_csv(path, index=False)

    # -------------------------
    # Load full tables
    # -------------------------
    def load_requests(self) -> pd.DataFrame:
        return self._read_csv(self.requests_path)

    def load_runs(self) -> pd.DataFrame:
        return self._read_csv(self.runs_path)

    def load_validation_results(self) -> pd.DataFrame:
        return self._read_csv(self.validation_path)

    def load_approvals(self) -> pd.DataFrame:
        return self._read_csv(self.approvals_path)

    def load_trace(self) -> pd.DataFrame:
        return self._read_csv(self.trace_path)

    def load_artifacts(self) -> pd.DataFrame:
        return self._read_csv(self.artifacts_path)

    def load_recommendations(self) -> pd.DataFrame:
        return self._read_csv(self.recommendations_path)

    # -------------------------
    # Single-record getters
    # -------------------------
    def get_request(self, request_id: str) -> Dict[str, Any]:
        df = self.load_requests()
        match = df[df["request_id"] == request_id]
        if match.empty:
            raise ValueError(f"Request not found: {request_id}")
        return match.iloc[0].to_dict()

    def get_run(self, workflow_run_id: str) -> Dict[str, Any]:
        df = self.load_runs()
        match = df[df["workflow_run_id"] == workflow_run_id]
        if match.empty:
            raise ValueError(f"Workflow run not found: {workflow_run_id}")
        return match.iloc[0].to_dict()

    def get_latest_run_for_request(self, request_id: str) -> Dict[str, Any]:
        df = self.load_runs()
        match = df[df["request_id"] == request_id].sort_values("run_version")
        if match.empty:
            raise ValueError(f"No workflow runs found for request: {request_id}")
        return match.iloc[-1].to_dict()

    def get_validation_for_run(self, workflow_run_id: str) -> Dict[str, Any]:
        df = self.load_validation_results()
        match = df[df["workflow_run_id"] == workflow_run_id]
        if match.empty:
            raise ValueError(f"Validation result not found for run: {workflow_run_id}")
        return match.iloc[0].to_dict()

    def get_approval_for_run(self, workflow_run_id: str) -> Dict[str, Any]:
        df = self.load_approvals()
        match = df[df["workflow_run_id"] == workflow_run_id]
        if match.empty:
            raise ValueError(f"Approval record not found for run: {workflow_run_id}")
        return match.iloc[0].to_dict()

    def get_recommendation_for_run(self, workflow_run_id: str) -> Dict[str, Any]:
        df = self.load_recommendations()
        match = df[df["workflow_run_id"] == workflow_run_id]
        if match.empty:
            raise ValueError(f"Recommendation output not found for run: {workflow_run_id}")
        return match.iloc[0].to_dict()

    def get_trace_for_run(self, workflow_run_id: str) -> pd.DataFrame:
        df = self.load_trace()
        return df[df["workflow_run_id"] == workflow_run_id].sort_values("timestamp")

    def get_artifacts_for_run(self, workflow_run_id: str) -> pd.DataFrame:
        df = self.load_artifacts()
        return df[df["workflow_run_id"] == workflow_run_id].sort_values("created_at")

    # -------------------------
    # Update helpers
    # -------------------------
    def update_request_status(self, request_id: str, new_status: str) -> None:
        df = self.load_requests()
        mask = df["request_id"] == request_id
        if not mask.any():
            raise ValueError(f"Request not found: {request_id}")
        df.loc[mask, "status"] = new_status
        self._write_csv(df, self.requests_path)

    def update_run_stage_and_status(
        self,
        workflow_run_id: str,
        current_stage: str,
        status: str,
        last_updated_at: Optional[str] = None,
    ) -> None:
        df = self.load_runs()
        mask = df["workflow_run_id"] == workflow_run_id
        if not mask.any():
            raise ValueError(f"Workflow run not found: {workflow_run_id}")

        df.loc[mask, "current_stage"] = current_stage
        df.loc[mask, "status"] = status

        if last_updated_at is not None:
            df.loc[mask, "last_updated_at"] = last_updated_at

        self._write_csv(df, self.runs_path)

    # -------------------------
    # Append helpers
    # -------------------------
    def append_trace_event(self, event: Dict[str, Any]) -> None:
        df = self.load_trace()
        df = pd.concat([df, pd.DataFrame([event])], ignore_index=True)
        self._write_csv(df, self.trace_path)

    def append_artifact_record(self, artifact: Dict[str, Any]) -> None:
        df = self.load_artifacts()
        df = pd.concat([df, pd.DataFrame([artifact])], ignore_index=True)
        self._write_csv(df, self.artifacts_path)