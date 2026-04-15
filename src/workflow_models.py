from dataclasses import dataclass
from typing import Optional


@dataclass
class SourcingRequest:
    request_id: str
    request_text: str
    requester_name: str
    business_unit: str
    priority: str
    status: str
    submitted_at: str
    due_by: str
    requested_demand_units: int
    requested_min_avg_esg: int
    requested_max_avg_risk: int
    requested_min_suppliers: int
    requested_max_supplier_share: float
    blocked_regions_raw: str
    manual_approval_required: int


@dataclass
class WorkflowRun:
    workflow_run_id: str
    request_id: str
    run_version: int
    current_stage: str
    status: str
    started_at: str
    last_updated_at: str
    trigger_type: str
    orchestrator_version: str


@dataclass
class ValidationResult:
    validation_id: str
    workflow_run_id: str
    request_id: str
    validation_status: str
    schema_check_passed: int
    business_rules_passed: int
    warning_count: int
    failure_count: int
    validation_summary: str
    recommended_action: str
    checked_at: str


@dataclass
class ApprovalRecord:
    approval_id: str
    workflow_run_id: str
    request_id: str
    approval_required: int
    decision: str
    reviewer_name: str
    review_comment: str
    reviewed_at: str
    sla_hours: float


@dataclass
class WorkflowTraceEvent:
    trace_id: str
    workflow_run_id: str
    request_id: str
    stage_name: str
    status: str
    event_type: str
    event_message: str
    timestamp: str


@dataclass
class ArtifactRecord:
    artifact_id: str
    workflow_run_id: str
    request_id: str
    artifact_type: str
    version_no: int
    storage_uri: str
    created_at: str
    created_by: str
    is_latest_version: int


@dataclass
class RecommendationOutput:
    recommendation_id: str
    workflow_run_id: str
    request_id: str
    recommendation_status: str
    headline: str
    detail: str
    selected_suppliers: Optional[float]
    estimated_total_cost: Optional[float]
    generated_at: str