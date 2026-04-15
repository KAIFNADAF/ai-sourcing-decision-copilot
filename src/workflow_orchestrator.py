from __future__ import annotations

from datetime import datetime
from typing import Dict, Any, List

import pandas as pd

from src.workflow_store import WorkflowStore
from src.trace_logger import TraceLogger
from src.validation_agent import ValidationAgent
from src.artifact_manager import ArtifactManager
from src.recommendation_agent import RecommendationAgent

from src.optimizer import load_suppliers, optimize_supplier_allocation
from src.audit import generate_decision_audit
from src.sensitivity import run_sensitivity_analysis


class WorkflowOrchestrator:
    """
    Main workflow controller for the sourcing orchestration system.
    """

    TERMINAL_OR_PAUSED_STATUSES = {
        "Completed",
        "Rejected",
        "Awaiting Approval",
    }

    def __init__(
        self,
        data_dir: str = "data",
        suppliers_path: str = "data/suppliers_dataset.csv",
    ) -> None:
        self.store = WorkflowStore(data_dir=data_dir)
        self.trace_logger = TraceLogger(store=self.store)
        self.validation_agent = ValidationAgent(
            store=self.store,
            suppliers_path=suppliers_path,
        )
        self.artifact_manager = ArtifactManager(store=self.store)
        self.recommendation_agent = RecommendationAgent()
        self.suppliers_path = suppliers_path

    def _now(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _parse_blocked_regions(self, raw_value: str) -> List[str]:
        if pd.isna(raw_value) or str(raw_value).strip() == "":
            return []
        return [x.strip() for x in str(raw_value).split("|") if x.strip()]

    # ---------------------------------------------------
    # ID / version helpers
    # ---------------------------------------------------
    def _request_numeric_part(self, request_id: str) -> str:
        parts = str(request_id).split("-")
        if len(parts) < 2 or not parts[1]:
            raise ValueError(f"Unexpected request_id format: {request_id}")
        return parts[1]

    def _build_run_id(self, request_id: str, run_version: int) -> str:
        request_num = self._request_numeric_part(request_id)
        return f"RUN-{request_num}-{int(run_version):02d}"

    def _build_validation_id(self, request_id: str, run_version: int) -> str:
        request_num = self._request_numeric_part(request_id)
        return f"VAL-{request_num}-{int(run_version):02d}"

    def _build_approval_id(self, request_id: str, run_version: int) -> str:
        request_num = self._request_numeric_part(request_id)
        return f"APR-{request_num}-{int(run_version):02d}"

    def _build_recommendation_id(self, request_id: str, run_version: int) -> str:
        request_num = self._request_numeric_part(request_id)
        return f"REC-{request_num}-{int(run_version):02d}"

    def _next_run_version_for_request(self, request_id: str) -> int:
        runs_df = self.store.load_runs()
        request_runs = runs_df[runs_df["request_id"] == request_id]

        if request_runs.empty:
            return 1

        return int(request_runs["run_version"].max()) + 1

    # ---------------------------------------------------
    # Scenario normalization helpers
    # ---------------------------------------------------
    def _build_normalized_scenario_from_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build the canonical executable scenario payload for this workflow run.

        This is intentionally deterministic and derived from the structured request row.
        It is not the natural-language parser. The goal is to make the parse stage
        a real normalization step instead of just a placeholder.
        """
        blocked_regions = self._parse_blocked_regions(request.get("blocked_regions_raw", ""))

        parsed_scenario = {
            "total_demand": int(request["requested_demand_units"]),
            "max_supplier_share": float(request["requested_max_supplier_share"]),
            "min_avg_esg": int(request["requested_min_avg_esg"]),
            "max_avg_risk": int(request["requested_max_avg_risk"]),
            "min_suppliers": int(request["requested_min_suppliers"]),
            "blocked_regions": blocked_regions,
            "w_cost": 0.65,
            "w_risk": 0.20,
            "w_esg": 0.10,
            "supplier_selection_penalty": 0.02,
        }

        field_sources = {
            "total_demand": "request",
            "max_supplier_share": "request",
            "min_avg_esg": "request",
            "max_avg_risk": "request",
            "min_suppliers": "request",
            "blocked_regions": "request",
            "w_cost": "default",
            "w_risk": "default",
            "w_esg": "default",
            "supplier_selection_penalty": "default",
        }

        assumptions = [
            "Workflow normalization converted the structured request into an executable scenario payload.",
            "Objective weights and supplier selection penalty used workflow defaults because they are not stored as request-level fields.",
        ]

        explicit_fields = [
            "total_demand",
            "max_supplier_share",
            "min_avg_esg",
            "max_avg_risk",
            "min_suppliers",
            "blocked_regions",
        ]

        missing_fields = [
            "w_cost",
            "w_risk",
            "w_esg",
            "supplier_selection_penalty",
        ]

        return {
            "parsed_scenario": parsed_scenario,
            "field_sources": field_sources,
            "assumptions": assumptions,
            "explicit_fields": explicit_fields,
            "missing_fields": missing_fields,
        }

    # ---------------------------------------------------
    # CSV append helpers for new run versions
    # ---------------------------------------------------
    def _append_run_row(self, row: Dict[str, Any]) -> None:
        df = self.store.load_runs()
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(self.store.runs_path, index=False)

    def _append_validation_row(self, row: Dict[str, Any]) -> None:
        df = self.store.load_validation_results()
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(self.store.validation_path, index=False)

    def _append_approval_row(self, row: Dict[str, Any]) -> None:
        df = self.store.load_approvals()
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(self.store.approvals_path, index=False)

    def _append_recommendation_row(self, row: Dict[str, Any]) -> None:
        df = self.store.load_recommendations()
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(self.store.recommendations_path, index=False)

    def create_run_version(
        self,
        request_id: str,
        source_workflow_run_id: str | None = None,
        trigger_type: str = "manual_rerun",
    ) -> Dict[str, Any]:
        """
        Create a new workflow run version for an existing request.
        """
        request = self.store.get_request(request_id)

        source_run = None
        if source_workflow_run_id:
            source_run = self.store.get_run(source_workflow_run_id)
            if source_run["request_id"] != request_id:
                raise ValueError(
                    f"Run {source_workflow_run_id} does not belong to request {request_id}."
                )

        new_run_version = self._next_run_version_for_request(request_id)
        new_workflow_run_id = self._build_run_id(request_id, new_run_version)
        now = self._now()

        source_approval = None
        if source_workflow_run_id:
            try:
                source_approval = self.store.get_approval_for_run(source_workflow_run_id)
            except Exception:
                source_approval = None

        new_run_row = {
            "workflow_run_id": new_workflow_run_id,
            "request_id": request_id,
            "run_version": int(new_run_version),
            "current_stage": "request_intake",
            "status": "Submitted",
            "started_at": now,
            "last_updated_at": now,
            "trigger_type": trigger_type,
            "orchestrator_version": "0.2.0",
        }

        new_validation_row = {
            "validation_id": self._build_validation_id(request_id, new_run_version),
            "workflow_run_id": new_workflow_run_id,
            "request_id": request_id,
            "validation_status": "pending",
            "schema_check_passed": 0,
            "business_rules_passed": 0,
            "warning_count": 0,
            "failure_count": 0,
            "validation_summary": "Validation not yet executed for this run version.",
            "recommended_action": "Run workflow validation.",
            "checked_at": "",
        }

        new_approval_row = {
            "approval_id": self._build_approval_id(request_id, new_run_version),
            "workflow_run_id": new_workflow_run_id,
            "request_id": request_id,
            "approval_required": int(request.get("manual_approval_required", 1)),
            "decision": "Pending",
            "reviewer_name": (
                source_approval["reviewer_name"]
                if source_approval and "reviewer_name" in source_approval
                else "Unassigned"
            ),
            "review_comment": "Awaiting review for new run version.",
            "reviewed_at": "",
            "sla_hours": (
                float(source_approval["sla_hours"])
                if source_approval and "sla_hours" in source_approval and pd.notna(source_approval["sla_hours"])
                else 24.0
            ),
        }

        new_recommendation_row = {
            "recommendation_id": self._build_recommendation_id(request_id, new_run_version),
            "workflow_run_id": new_workflow_run_id,
            "request_id": request_id,
            "recommendation_status": "not_generated",
            "headline": "Recommendation not produced yet.",
            "detail": "This run version has been created but not executed.",
            "selected_suppliers": None,
            "estimated_total_cost": None,
            "generated_at": "",
        }

        self._append_run_row(new_run_row)
        self._append_validation_row(new_validation_row)
        self._append_approval_row(new_approval_row)
        self._append_recommendation_row(new_recommendation_row)

        self.store.update_request_status(request_id, "submitted")

        self.trace_logger.log_event(
            workflow_run_id=new_workflow_run_id,
            request_id=request_id,
            stage_name="request_intake",
            status="Success",
            event_type="run_version_created",
            event_message=(
                f"Created run version v{int(new_run_version)}"
                + (
                    f" from source run {source_workflow_run_id}."
                    if source_workflow_run_id
                    else "."
                )
            ),
            timestamp=now,
        )

        return new_run_row

    def _current_run_snapshot(self, workflow_run_id: str) -> Dict[str, Any]:
        run = self.store.get_run(workflow_run_id)
        request = self.store.get_request(run["request_id"])
        validation = self.store.get_validation_for_run(workflow_run_id)
        approval = self.store.get_approval_for_run(workflow_run_id)
        recommendation = self.store.get_recommendation_for_run(workflow_run_id)

        final_status = run["status"]
        reason = f"Run is already in status: {run['status']}."

        if run["status"] == "Awaiting Approval":
            reason = "Approval is pending."
        elif run["status"] == "Rejected":
            reason = "Approval was rejected."
        elif run["status"] == "Completed":
            reason = "Workflow already completed successfully."
        elif validation["validation_status"] == "failed":
            final_status = "Failed"
            reason = "Validation failed."

        return {
            "workflow_run_id": workflow_run_id,
            "request_id": request["request_id"],
            "final_status": final_status,
            "reason": reason,
            "run": run,
            "request": request,
            "validation_result": validation,
            "approval_result": approval,
            "recommendation": recommendation,
        }

    def _update_request_status_from_run_status(
        self,
        request_id: str,
        run_status: str,
        current_stage: str,
    ) -> None:
        if run_status == "Failed":
            self.store.update_request_status(request_id, "rejected")
        elif run_status == "Rejected":
            self.store.update_request_status(request_id, "rejected")
        elif run_status == "Awaiting Approval":
            self.store.update_request_status(request_id, "awaiting_approval")
        elif run_status == "Completed":
            if current_stage == "recommend":
                self.store.update_request_status(request_id, "completed")
            else:
                self.store.update_request_status(request_id, "approved")
        else:
            stage_to_request_status = {
                "request_intake": "submitted",
                "parse": "parsed",
                "validate": "validated",
                "approve": "awaiting_approval",
                "execute": "approved",
                "recommend": "completed",
            }
            self.store.update_request_status(
                request_id,
                stage_to_request_status.get(current_stage, "submitted"),
            )

    def _write_validation_result_back(self, validation_result: Dict[str, Any]) -> None:
        df = self.store.load_validation_results()
        mask = df["workflow_run_id"] == validation_result["workflow_run_id"]

        if not mask.any():
            raise ValueError(
                f"No validation row found for workflow_run_id={validation_result['workflow_run_id']}"
            )

        allowed_cols = [
            "validation_id",
            "workflow_run_id",
            "request_id",
            "validation_status",
            "schema_check_passed",
            "business_rules_passed",
            "warning_count",
            "failure_count",
            "validation_summary",
            "recommended_action",
            "checked_at",
        ]

        for col in allowed_cols:
            df.loc[mask, col] = validation_result[col]

        df.to_csv(self.store.validation_path, index=False)

    def _write_recommendation_result_back(self, recommendation: Dict[str, Any]) -> None:
        df = self.store.load_recommendations()
        mask = df["workflow_run_id"] == recommendation["workflow_run_id"]

        if not mask.any():
            raise ValueError(
                f"No recommendation row found for workflow_run_id={recommendation['workflow_run_id']}"
            )

        if not recommendation.get("recommendation_id"):
            run = self.store.get_run(recommendation["workflow_run_id"])
            recommendation["recommendation_id"] = self._build_recommendation_id(
                request_id=recommendation["request_id"],
                run_version=int(run["run_version"]),
            )

        allowed_cols = [
            "recommendation_id",
            "workflow_run_id",
            "request_id",
            "recommendation_status",
            "headline",
            "detail",
            "selected_suppliers",
            "estimated_total_cost",
            "generated_at",
        ]

        for col in allowed_cols:
            df.loc[mask, col] = recommendation[col]

        df.to_csv(self.store.recommendations_path, index=False)

    def run_parse_stage(self, workflow_run_id: str) -> Dict[str, Any]:
        run = self.store.get_run(workflow_run_id)
        request = self.store.get_request(run["request_id"])

        self.store.update_run_stage_and_status(
            workflow_run_id=workflow_run_id,
            current_stage="parse",
            status="In Progress",
            last_updated_at=self._now(),
        )
        self._update_request_status_from_run_status(
            request_id=request["request_id"],
            run_status="In Progress",
            current_stage="parse",
        )

        self.trace_logger.log_stage_started(
            workflow_run_id=workflow_run_id,
            request_id=request["request_id"],
            stage_name="parse",
            message="Request normalization stage started.",
        )

        normalized = self._build_normalized_scenario_from_request(request)

        self.trace_logger.log_stage_success(
            workflow_run_id=workflow_run_id,
            request_id=request["request_id"],
            stage_name="parse",
            message="Structured request normalized into executable scenario payload.",
        )

        self.artifact_manager.register_stage_artifacts(
            workflow_run_id=workflow_run_id,
            request_id=request["request_id"],
            run_version=int(run["run_version"]),
            stage_name="parse",
        )

        return {
            "workflow_run_id": workflow_run_id,
            "request_id": request["request_id"],
            "stage": "parse",
            "status": "Success",
            "parsed_scenario": normalized["parsed_scenario"],
            "field_sources": normalized["field_sources"],
            "assumptions": normalized["assumptions"],
            "explicit_fields": normalized["explicit_fields"],
            "missing_fields": normalized["missing_fields"],
        }

    def run_validation_stage(
        self,
        workflow_run_id: str,
        parse_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        run = self.store.get_run(workflow_run_id)
        request = self.store.get_request(run["request_id"])

        self.store.update_run_stage_and_status(
            workflow_run_id=workflow_run_id,
            current_stage="validate",
            status="In Progress",
            last_updated_at=self._now(),
        )
        self._update_request_status_from_run_status(
            request_id=request["request_id"],
            run_status="In Progress",
            current_stage="validate",
        )

        self.trace_logger.log_stage_started(
            workflow_run_id=workflow_run_id,
            request_id=request["request_id"],
            stage_name="validate",
            message="Validation stage started using normalized scenario payload.",
        )

        validation_result = self.validation_agent.evaluate_run(
            workflow_run_id=workflow_run_id,
            scenario=parse_result["parsed_scenario"],
        )
        self._write_validation_result_back(validation_result)

        if validation_result["validation_status"] == "failed":
            self.trace_logger.log_stage_failure(
                workflow_run_id=workflow_run_id,
                request_id=request["request_id"],
                stage_name="validate",
                message=validation_result["validation_summary"],
            )

            self.store.update_run_stage_and_status(
                workflow_run_id=workflow_run_id,
                current_stage="validate",
                status="Failed",
                last_updated_at=self._now(),
            )
            self._update_request_status_from_run_status(
                request_id=request["request_id"],
                run_status="Failed",
                current_stage="validate",
            )
        else:
            self.trace_logger.log_stage_success(
                workflow_run_id=workflow_run_id,
                request_id=request["request_id"],
                stage_name="validate",
                message=validation_result["validation_summary"],
            )

        self.artifact_manager.register_stage_artifacts(
            workflow_run_id=workflow_run_id,
            request_id=request["request_id"],
            run_version=int(run["run_version"]),
            stage_name="validate",
        )

        return validation_result

    def run_approval_stage(self, workflow_run_id: str) -> Dict[str, Any]:
        run = self.store.get_run(workflow_run_id)
        request = self.store.get_request(run["request_id"])
        approval = self.store.get_approval_for_run(workflow_run_id)

        self.store.update_run_stage_and_status(
            workflow_run_id=workflow_run_id,
            current_stage="approve",
            status="In Progress",
            last_updated_at=self._now(),
        )

        self.trace_logger.log_stage_started(
            workflow_run_id=workflow_run_id,
            request_id=request["request_id"],
            stage_name="approve",
            message="Approval stage started.",
        )

        if approval["decision"] == "Pending":
            self.trace_logger.log_stage_pending(
                workflow_run_id=workflow_run_id,
                request_id=request["request_id"],
                stage_name="approve",
                message=approval["review_comment"],
            )
            self.store.update_run_stage_and_status(
                workflow_run_id=workflow_run_id,
                current_stage="approve",
                status="Awaiting Approval",
                last_updated_at=self._now(),
            )
            self._update_request_status_from_run_status(
                request_id=request["request_id"],
                run_status="Awaiting Approval",
                current_stage="approve",
            )
        elif approval["decision"] == "Rejected":
            self.trace_logger.log_stage_failure(
                workflow_run_id=workflow_run_id,
                request_id=request["request_id"],
                stage_name="approve",
                message=approval["review_comment"],
            )
            self.store.update_run_stage_and_status(
                workflow_run_id=workflow_run_id,
                current_stage="approve",
                status="Rejected",
                last_updated_at=self._now(),
            )
            self._update_request_status_from_run_status(
                request_id=request["request_id"],
                run_status="Rejected",
                current_stage="approve",
            )
        else:
            self.trace_logger.log_stage_success(
                workflow_run_id=workflow_run_id,
                request_id=request["request_id"],
                stage_name="approve",
                message=approval["review_comment"],
            )

        return approval

    def run_execution_stage(
        self,
        workflow_run_id: str,
        parsed_scenario: Dict[str, Any],
    ) -> Dict[str, Any]:
        run = self.store.get_run(workflow_run_id)
        request = self.store.get_request(run["request_id"])
        validation = self.store.get_validation_for_run(workflow_run_id)
        approval = self.store.get_approval_for_run(workflow_run_id)

        if validation["validation_status"] == "failed":
            raise ValueError("Execution blocked because validation failed.")

        if approval["decision"] in {"Rejected", "Pending"}:
            raise ValueError(
                f"Execution blocked because approval decision is {approval['decision']}."
            )

        self.store.update_run_stage_and_status(
            workflow_run_id=workflow_run_id,
            current_stage="execute",
            status="In Progress",
            last_updated_at=self._now(),
        )

        self.trace_logger.log_stage_started(
            workflow_run_id=workflow_run_id,
            request_id=request["request_id"],
            stage_name="execute",
            message="Execution stage started from normalized scenario payload.",
        )

        suppliers_df = load_suppliers(self.suppliers_path)
        base_params = parsed_scenario

        result = optimize_supplier_allocation(
            df=suppliers_df,
            total_demand=base_params["total_demand"],
            max_supplier_share=base_params["max_supplier_share"],
            min_avg_esg=base_params["min_avg_esg"],
            max_avg_risk=base_params["max_avg_risk"],
            min_suppliers=base_params["min_suppliers"],
            blocked_regions=base_params["blocked_regions"],
            w_cost=base_params["w_cost"],
            w_risk=base_params["w_risk"],
            w_esg=base_params["w_esg"],
            supplier_selection_penalty=base_params["supplier_selection_penalty"],
        )

        audit = generate_decision_audit(result)
        sensitivity_output = run_sensitivity_analysis(
            df=suppliers_df,
            base_params=base_params,
        )

        if result.get("status") != "Optimal":
            self.trace_logger.log_stage_failure(
                workflow_run_id=workflow_run_id,
                request_id=request["request_id"],
                stage_name="execute",
                message=result.get("message", "Execution failed."),
            )
            self.store.update_run_stage_and_status(
                workflow_run_id=workflow_run_id,
                current_stage="execute",
                status="Failed",
                last_updated_at=self._now(),
            )
            self._update_request_status_from_run_status(
                request_id=request["request_id"],
                run_status="Failed",
                current_stage="execute",
            )
        else:
            self.trace_logger.log_stage_success(
                workflow_run_id=workflow_run_id,
                request_id=request["request_id"],
                stage_name="execute",
                message="Optimization, audit, and sensitivity analysis completed successfully.",
            )

        self.artifact_manager.register_stage_artifacts(
            workflow_run_id=workflow_run_id,
            request_id=request["request_id"],
            run_version=int(run["run_version"]),
            stage_name="execute",
        )

        return {
            "result": result,
            "audit": audit,
            "sensitivity_output": sensitivity_output,
            "parsed_scenario": parsed_scenario,
        }

    def run_recommendation_stage(
        self,
        workflow_run_id: str,
        execution_bundle: Dict[str, Any],
    ) -> Dict[str, Any]:
        run = self.store.get_run(workflow_run_id)
        request = self.store.get_request(run["request_id"])

        result = execution_bundle["result"]
        audit = execution_bundle["audit"]
        sensitivity_output = execution_bundle["sensitivity_output"]

        self.store.update_run_stage_and_status(
            workflow_run_id=workflow_run_id,
            current_stage="recommend",
            status="In Progress",
            last_updated_at=self._now(),
        )

        self.trace_logger.log_stage_started(
            workflow_run_id=workflow_run_id,
            request_id=request["request_id"],
            stage_name="recommend",
            message="Recommendation stage started.",
        )

        recommendation = self.recommendation_agent.build_recommendation(
            workflow_run_id=workflow_run_id,
            request_id=request["request_id"],
            result=result,
            audit=audit,
            sensitivity_output=sensitivity_output,
        )

        if not recommendation.get("recommendation_id"):
            recommendation["recommendation_id"] = self._build_recommendation_id(
                request_id=request["request_id"],
                run_version=int(run["run_version"]),
            )

        self._write_recommendation_result_back(recommendation)

        if recommendation["recommendation_status"] == "generated":
            self.trace_logger.log_stage_success(
                workflow_run_id=workflow_run_id,
                request_id=request["request_id"],
                stage_name="recommend",
                message=recommendation["headline"],
            )
            self.store.update_run_stage_and_status(
                workflow_run_id=workflow_run_id,
                current_stage="recommend",
                status="Completed",
                last_updated_at=self._now(),
            )
            self._update_request_status_from_run_status(
                request_id=request["request_id"],
                run_status="Completed",
                current_stage="recommend",
            )
        else:
            self.trace_logger.log_stage_failure(
                workflow_run_id=workflow_run_id,
                request_id=request["request_id"],
                stage_name="recommend",
                message=recommendation["detail"],
            )
            self.store.update_run_stage_and_status(
                workflow_run_id=workflow_run_id,
                current_stage="recommend",
                status="Failed",
                last_updated_at=self._now(),
            )
            self._update_request_status_from_run_status(
                request_id=request["request_id"],
                run_status="Failed",
                current_stage="recommend",
            )

        self.artifact_manager.register_stage_artifacts(
            workflow_run_id=workflow_run_id,
            request_id=request["request_id"],
            run_version=int(run["run_version"]),
            stage_name="recommend",
        )

        return recommendation

    def run_workflow(self, workflow_run_id: str) -> Dict[str, Any]:
        """
        Execute the governed workflow for a single run.

        Important behavior:
        - do NOT rerun terminal or paused runs in place
        - return the current snapshot instead of mutating history
        """
        run = self.store.get_run(workflow_run_id)

        if run["status"] in self.TERMINAL_OR_PAUSED_STATUSES:
            return self._current_run_snapshot(workflow_run_id)

        request = self.store.get_request(run["request_id"])

        self.trace_logger.log_event(
            workflow_run_id=workflow_run_id,
            request_id=request["request_id"],
            stage_name="request_intake",
            status="Success",
            event_type="workflow_started",
            event_message="Workflow execution started by orchestrator.",
            timestamp=self._now(),
        )

        parse_result = self.run_parse_stage(workflow_run_id)
        validation_result = self.run_validation_stage(
            workflow_run_id=workflow_run_id,
            parse_result=parse_result,
        )

        if validation_result["validation_status"] == "failed":
            return {
                "workflow_run_id": workflow_run_id,
                "request_id": request["request_id"],
                "final_status": "Failed",
                "reason": "Validation failed.",
                "parse_result": parse_result,
                "validation_result": validation_result,
            }

        approval_result = self.run_approval_stage(workflow_run_id)

        if approval_result["decision"] == "Pending":
            return {
                "workflow_run_id": workflow_run_id,
                "request_id": request["request_id"],
                "final_status": "Awaiting Approval",
                "reason": "Approval is pending.",
                "parse_result": parse_result,
                "validation_result": validation_result,
                "approval_result": approval_result,
            }

        if approval_result["decision"] == "Rejected":
            return {
                "workflow_run_id": workflow_run_id,
                "request_id": request["request_id"],
                "final_status": "Rejected",
                "reason": "Approval was rejected.",
                "parse_result": parse_result,
                "validation_result": validation_result,
                "approval_result": approval_result,
            }

        execution_bundle = self.run_execution_stage(
            workflow_run_id=workflow_run_id,
            parsed_scenario=parse_result["parsed_scenario"],
        )

        if execution_bundle["result"].get("status") != "Optimal":
            return {
                "workflow_run_id": workflow_run_id,
                "request_id": request["request_id"],
                "final_status": "Failed",
                "reason": execution_bundle["result"].get("message", "Execution failed."),
                "parse_result": parse_result,
                "validation_result": validation_result,
                "approval_result": approval_result,
                "execution_result": execution_bundle["result"],
                "parsed_scenario": execution_bundle["parsed_scenario"],
            }

        recommendation = self.run_recommendation_stage(
            workflow_run_id=workflow_run_id,
            execution_bundle=execution_bundle,
        )

        return {
            "workflow_run_id": workflow_run_id,
            "request_id": request["request_id"],
            "final_status": "Completed",
            "parse_result": parse_result,
            "validation_result": validation_result,
            "approval_result": approval_result,
            "execution_result": execution_bundle["result"],
            "audit_result": execution_bundle["audit"],
            "sensitivity_result": execution_bundle["sensitivity_output"],
            "recommendation": recommendation,
            "parsed_scenario": execution_bundle["parsed_scenario"],
        }