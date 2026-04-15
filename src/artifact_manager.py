from __future__ import annotations

from datetime import datetime
from typing import Dict, Any, Optional

import pandas as pd

from src.workflow_store import WorkflowStore


class ArtifactManager:
    """
    Handles artifact registration for workflow runs.

    Key behavior:
    - idempotent per (workflow_run_id, artifact_type, version_no)
    - can clean existing duplicate artifact rows
    """

    def __init__(self, store: WorkflowStore) -> None:
        self.store = store

    def _build_artifact_id(self, request_id: str, run_version: int, sequence_no: int) -> str:
        request_num = int(request_id.split("-")[1])
        return f"ART-{request_num:04d}-{run_version:02d}-{sequence_no:02d}"

    def _next_artifact_sequence_for_run(self, workflow_run_id: str) -> int:
        artifacts_df = self.store.load_artifacts()
        run_rows = artifacts_df[artifacts_df["workflow_run_id"] == workflow_run_id]

        if run_rows.empty:
            return 1

        sequence_numbers = []
        for artifact_id in run_rows["artifact_id"].dropna().astype(str).tolist():
            parts = artifact_id.split("-")
            if len(parts) == 4 and parts[0] == "ART":
                try:
                    sequence_numbers.append(int(parts[3]))
                except ValueError:
                    continue

        return max(sequence_numbers, default=0) + 1

    def _find_existing_artifact(
        self,
        workflow_run_id: str,
        artifact_type: str,
        version_no: int,
    ) -> Optional[Dict[str, Any]]:
        artifacts_df = self.store.load_artifacts()

        match = artifacts_df[
            (artifacts_df["workflow_run_id"] == workflow_run_id)
            & (artifacts_df["artifact_type"] == artifact_type)
            & (artifacts_df["version_no"] == int(version_no))
        ].sort_values("created_at")

        if match.empty:
            return None

        return match.iloc[-1].to_dict()

    def deduplicate_artifacts(self) -> int:
        """
        One-time cleanup:
        keep only the latest row for each
        (workflow_run_id, artifact_type, version_no)
        """
        df = self.store.load_artifacts()
        if df.empty:
            return 0

        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
        df = df.sort_values("created_at")

        deduped = df.drop_duplicates(
            subset=["workflow_run_id", "artifact_type", "version_no"],
            keep="last",
        ).copy()

        deduped["is_latest_version"] = 1
        deduped = deduped.sort_values(["workflow_run_id", "created_at"]).reset_index(drop=True)
        deduped.to_csv(self.store.artifacts_path, index=False)

        return len(df) - len(deduped)

    def register_artifact(
        self,
        workflow_run_id: str,
        request_id: str,
        artifact_type: str,
        version_no: int,
        created_by: str = "system",
        storage_uri: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Idempotent registration:
        if the same artifact already exists for this run/version/type,
        reuse it instead of appending a duplicate.
        """
        existing = self._find_existing_artifact(
            workflow_run_id=workflow_run_id,
            artifact_type=artifact_type,
            version_no=version_no,
        )
        if existing is not None:
            return existing

        sequence_no = self._next_artifact_sequence_for_run(workflow_run_id)
        artifact_id = self._build_artifact_id(
            request_id=request_id,
            run_version=version_no,
            sequence_no=sequence_no,
        )

        if storage_uri is None:
            storage_uri = (
                f"artifacts/{request_id}/{workflow_run_id}/"
                f"{artifact_type}_v{version_no}.json"
            )

        artifact = {
            "artifact_id": artifact_id,
            "workflow_run_id": workflow_run_id,
            "request_id": request_id,
            "artifact_type": artifact_type,
            "version_no": int(version_no),
            "storage_uri": storage_uri,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "created_by": created_by,
            "is_latest_version": 1,
        }

        self.store.append_artifact_record(artifact)
        return artifact

    def register_stage_artifacts(
        self,
        workflow_run_id: str,
        request_id: str,
        run_version: int,
        stage_name: str,
    ) -> list[Dict[str, Any]]:
        stage_artifact_map = {
            "parse": ["parsed_scenario"],
            "validate": ["validation_report"],
            "execute": ["optimization_result", "decision_audit", "sensitivity_report"],
            "recommend": ["recommendation_packet"],
        }

        artifact_types = stage_artifact_map.get(stage_name, [])
        created = []

        for artifact_type in artifact_types:
            created.append(
                self.register_artifact(
                    workflow_run_id=workflow_run_id,
                    request_id=request_id,
                    artifact_type=artifact_type,
                    version_no=run_version,
                )
            )

        return created

    def get_latest_artifacts_for_run(self, workflow_run_id: str) -> pd.DataFrame:
        artifacts_df = self.store.get_artifacts_for_run(workflow_run_id)
        if artifacts_df.empty:
            return artifacts_df

        deduped = (
            artifacts_df.sort_values("created_at")
            .drop_duplicates(subset=["artifact_type", "version_no"], keep="last")
            .sort_values("created_at")
        )
        return deduped.reset_index(drop=True)

    def get_artifacts_by_type(
        self,
        workflow_run_id: str,
        artifact_type: str,
    ) -> pd.DataFrame:
        artifacts_df = self.store.get_artifacts_for_run(workflow_run_id)
        if artifacts_df.empty:
            return artifacts_df

        filtered = artifacts_df[artifacts_df["artifact_type"] == artifact_type].sort_values("created_at")
        return filtered.reset_index(drop=True)