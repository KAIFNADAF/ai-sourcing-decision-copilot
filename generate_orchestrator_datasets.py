from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta
import random
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np


SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# -----------------------------
# Configuration
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
SUPPLIER_FILE = DATA_DIR / "suppliers_dataset.csv"

# exact row target for validation_results.csv
REQUEST_COUNT = 2000

# keep one run per request so validation rows = workflow run rows = 2000
MIN_RERUNS = 0
MAX_RERUNS = 0

# scenario mix
FAILED_REQUEST_SHARE = 0.12
WARNING_REQUEST_SHARE = 0.28


# -----------------------------
# Required existing supplier schema
# -----------------------------
REQUIRED_SUPPLIER_COLUMNS = {
    "supplier_id",
    "supplier_name",
    "region",
    "unit_cost",
    "max_capacity",
    "risk_score",
    "esg_score",
    "compliance_flag",
}


# -----------------------------
# Helpers
# -----------------------------
def iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def rand_dt(start: datetime, end: datetime) -> datetime:
    delta = end - start
    total_minutes = int(delta.total_seconds() // 60)
    offset_minutes = random.randint(0, total_minutes)
    return start + timedelta(minutes=offset_minutes)


def weighted_choice(options: List[Tuple[str, float]]) -> str:
    labels = [x[0] for x in options]
    weights = [x[1] for x in options]
    return random.choices(labels, weights=weights, k=1)[0]


def ensure_data_dir() -> None:
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")


def load_supplier_data() -> pd.DataFrame:
    if not SUPPLIER_FILE.exists():
        raise FileNotFoundError(f"Supplier dataset not found: {SUPPLIER_FILE}")

    df = pd.read_csv(SUPPLIER_FILE)

    missing = REQUIRED_SUPPLIER_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            "Supplier dataset is missing required columns: "
            + ", ".join(sorted(missing))
        )

    if len(df) < 100:
        raise ValueError(
            f"Supplier dataset looks unexpectedly small ({len(df)} rows). "
            "Expected a full synthetic supplier master."
        )

    if df["supplier_id"].duplicated().any():
        raise ValueError("supplier_id must be unique in suppliers_dataset.csv")

    return df


def build_request_text(
    demand: int,
    blocked_regions: List[str],
    min_esg: int,
    max_risk: int,
    min_suppliers: int,
    max_supplier_share: float,
    style: str,
) -> str:
    region_text = ", ".join(blocked_regions) if blocked_regions else "none"
    share_pct = int(round(max_supplier_share * 100))

    if style == "explicit":
        return (
            f"Allocate {demand} units. Block {region_text}. "
            f"Keep average ESG at least {min_esg}, average risk at most {max_risk}, "
            f"use at least {min_suppliers} suppliers, and cap each supplier at {share_pct}%."
        )

    if style == "semi_explicit":
        return (
            f"We need to source {demand} units for the next cycle. "
            f"Avoid {region_text}. Keep ESG above {min_esg} and risk under {max_risk}. "
            f"Please make sure the allocation is not too concentrated."
        )

    return (
        f"Need coverage for {demand} units. "
        f"Try to avoid {region_text}. We want strong ESG, low risk, "
        f"and a reasonably diversified supplier base."
    )


def classify_request_status(
    manual_approval_required: int,
    validation_status: str,
    final_decision: str,
) -> str:
    if validation_status == "failed":
        return "rejected"
    if final_decision == "Pending":
        return "awaiting_approval"
    if final_decision == "Rejected":
        return "rejected"
    if final_decision in {"Approved", "Auto-Approved"}:
        return random.choice(["approved", "completed"])
    return "submitted"


# -----------------------------
# Dataset builders
# -----------------------------
def build_sourcing_requests(suppliers_df: pd.DataFrame) -> pd.DataFrame:
    requester_names = [
        "Priya Nair", "Liam Carter", "Sofia Martinez", "Ethan Brooks",
        "Ava Turner", "Noah Bennett", "Mia Lopez", "Lucas Hall",
    ]
    business_units = [
        "Consumer Electronics", "Industrial", "Healthcare",
        "Automotive", "Retail", "Energy",
    ]
    priorities = ["Low", "Medium", "High", "Critical"]

    all_regions = sorted(suppliers_df["region"].dropna().unique().tolist())
    compliant_df = suppliers_df[suppliers_df["compliance_flag"] == 1].copy()
    total_compliant_capacity = float(compliant_df["max_capacity"].sum())

    start = datetime(2026, 1, 5, 9, 0, 0)
    end = datetime(2026, 4, 12, 18, 0, 0)

    rows = []

    failed_count = int(REQUEST_COUNT * FAILED_REQUEST_SHARE)
    warning_count = int(REQUEST_COUNT * WARNING_REQUEST_SHARE)
    normal_count = REQUEST_COUNT - failed_count - warning_count

    scenario_types = (
        ["failed"] * failed_count +
        ["warning"] * warning_count +
        ["normal"] * normal_count
    )
    random.shuffle(scenario_types)

    for i, scenario_type in enumerate(scenario_types, start=1):
        request_id = f"REQ-{i:04d}"


        submitted_at = rand_dt(start, end)
        due_by = submitted_at + timedelta(days=random.randint(2, 10))
        priority = weighted_choice(
            [("Low", 0.10), ("Medium", 0.35), ("High", 0.35), ("Critical", 0.20)]
        )

        if scenario_type == "failed":
            demand = int(total_compliant_capacity * random.uniform(1.05, 1.20))
            min_esg = random.choice([80, 82, 85])
            max_risk = random.choice([25, 30])
            min_suppliers = random.choice([5, 6])
            max_supplier_share = 0.25
            blocked_regions = []
            manual_approval_required = 1

        elif scenario_type == "warning":
            demand = random.choice([120000, 135000, 150000, 170000])
            min_esg = random.choice([78, 80, 82])
            max_risk = random.choice([30, 35])
            min_suppliers = random.choice([4, 5])
            max_supplier_share = random.choice([0.25, 0.30])
            blocked_regions = random.sample(all_regions, k=random.choice([1, 2]))
            manual_approval_required = 1

        else:
            demand = random.choice([40000, 55000, 70000, 90000, 110000])
            min_esg = random.choice([68, 70, 72, 75])
            max_risk = random.choice([40, 45, 50])
            min_suppliers = random.choice([2, 3, 4])
            max_supplier_share = random.choice([0.30, 0.35, 0.40])
            blocked_regions = random.sample(all_regions, k=random.choice([0, 1])) if random.random() < 0.5 else []
            manual_approval_required = int(
                priority in {"High", "Critical"} or min_esg >= 80 or max_risk <= 35
            )

        request_text = build_request_text(
            demand=demand,
            blocked_regions=blocked_regions,
            min_esg=min_esg,
            max_risk=max_risk,
            min_suppliers=min_suppliers,
            max_supplier_share=max_supplier_share,
            style=random.choice(["explicit", "semi_explicit", "vague"]),
        )

        rows.append(
            {
                "request_id": request_id,
                "request_text": request_text,
                "requester_name": random.choice(requester_names),
                "business_unit": random.choice(business_units),
                "priority": priority,
                "status": "submitted",
                "submitted_at": iso(submitted_at),
                "due_by": iso(due_by),
                "requested_demand_units": demand,
                "requested_min_avg_esg": min_esg,
                "requested_max_avg_risk": max_risk,
                "requested_min_suppliers": min_suppliers,
                "requested_max_supplier_share": max_supplier_share,
                "blocked_regions_raw": "|".join(blocked_regions),
                "manual_approval_required": manual_approval_required,
            }
        )

    requests_df = pd.DataFrame(rows)

    if requests_df["request_id"].duplicated().any():
        raise ValueError("request_id must be unique in sourcing_requests.csv")

    return requests_df


def build_workflow_runs(requests_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, req in requests_df.iterrows():
        run_version = 1
        workflow_run_id = f"RUN-{req['request_id'].split('-')[1]}-{run_version:02d}"
        started_at = pd.to_datetime(req["submitted_at"]) + timedelta(minutes=10)

        rows.append(
            {
                "workflow_run_id": workflow_run_id,
                "request_id": req["request_id"],
                "run_version": run_version,
                "current_stage": "request_intake",
                "status": "In Progress",
                "started_at": iso(started_at),
                "last_updated_at": iso(started_at + timedelta(hours=random.randint(1, 36))),
                "trigger_type": "initial_submission",
                "orchestrator_version": random.choice(["0.1.0", "0.1.1", "0.2.0"]),
            }
        )

    runs_df = pd.DataFrame(rows)

    if runs_df["workflow_run_id"].duplicated().any():
        raise ValueError("workflow_run_id must be unique in workflow_runs.csv")

    return runs_df


def build_validation_results(
    suppliers_df: pd.DataFrame,
    requests_df: pd.DataFrame,
    workflow_runs_df: pd.DataFrame,
) -> pd.DataFrame:
    eligible_suppliers = suppliers_df[suppliers_df["compliance_flag"] == 1].copy()
    eligible_capacity = float(eligible_suppliers["max_capacity"].sum())

    rows = []

    for _, run in workflow_runs_df.iterrows():
        req = requests_df.loc[requests_df["request_id"] == run["request_id"]].iloc[0]

        warnings = []
        failures = []

        demand = int(req["requested_demand_units"])
        min_suppliers = int(req["requested_min_suppliers"])
        min_esg = int(req["requested_min_avg_esg"])
        max_risk = int(req["requested_max_avg_risk"])
        max_share = float(req["requested_max_supplier_share"])

        blocked_regions = [
            x for x in str(req["blocked_regions_raw"]).split("|") if x and x != "nan"
        ]

        filtered_df = eligible_suppliers[~eligible_suppliers["region"].isin(blocked_regions)].copy()
        filtered_capacity = float(filtered_df["max_capacity"].sum())
        eligible_count = len(filtered_df)

        if filtered_capacity < demand:
            failures.append("Eligible capacity after region filters is below requested demand.")
        if eligible_count < min_suppliers:
            failures.append("Eligible supplier count is below the minimum supplier requirement.")
        if demand > eligible_capacity * 0.55:
            warnings.append("Demand is large relative to the current eligible capacity pool.")
        if max_share <= 0.25:
            warnings.append("Tight supplier share cap may increase infeasibility risk.")
        if min_esg >= 82:
            warnings.append("High ESG target may increase cost or reduce feasibility.")
        if max_risk <= 30:
            warnings.append("Low risk ceiling may reduce the feasible supplier set.")
        if len(blocked_regions) >= 2:
            warnings.append("Multiple blocked regions narrow the supplier pool.")

        if failures:
            validation_status = "failed"
            business_rules_passed = 0
            summary = "Scenario failed business-rule validation."
            action = "Relax demand, blocked regions, or portfolio constraints before execution."
        elif len(warnings) >= 2:
            validation_status = "warning"
            business_rules_passed = 1
            summary = "Scenario passed validation with elevated feasibility risk."
            action = "Proceed to approval with reviewer attention on tight constraints."
        else:
            validation_status = "passed"
            business_rules_passed = 1
            summary = "Scenario passed schema and business-rule validation."
            action = "Proceed to execution."

        checked_at = pd.to_datetime(run["started_at"]) + timedelta(minutes=20)

        rows.append(
            {
                "validation_id": f"VAL-{run['workflow_run_id'].split('-')[1]}-{run['run_version']:02d}",
                "workflow_run_id": run["workflow_run_id"],
                "request_id": run["request_id"],
                "validation_status": validation_status,
                "schema_check_passed": 1,
                "business_rules_passed": business_rules_passed,
                "warning_count": len(warnings),
                "failure_count": len(failures),
                "validation_summary": summary,
                "recommended_action": action,
                "checked_at": iso(checked_at),
            }
        )

    return pd.DataFrame(rows)


def build_approvals(
    requests_df: pd.DataFrame,
    workflow_runs_df: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> pd.DataFrame:
    reviewers = ["A. Shah", "M. Chen", "R. Patel", "S. Walker", "J. Morgan"]
    rows = []

    for _, run in workflow_runs_df.iterrows():
        req = requests_df.loc[requests_df["request_id"] == run["request_id"]].iloc[0]
        val = validation_df.loc[validation_df["workflow_run_id"] == run["workflow_run_id"]].iloc[0]

        approval_required = int(
            int(req["manual_approval_required"]) == 1
            or val["validation_status"] in {"warning", "failed"}
        )

        reviewer_name = "System"
        decision = "Auto-Approved"
        review_comment = "Request met auto-approval thresholds."

        if approval_required:
            reviewer_name = random.choice(reviewers)

            if val["validation_status"] == "failed":
                decision = "Rejected"
                review_comment = "Rejected because validation found likely infeasibility."
            else:
                decision = weighted_choice(
                    [("Approved", 0.65), ("Pending", 0.20), ("Rejected", 0.15)]
                )
                if decision == "Approved":
                    review_comment = "Approved after review of trade-offs and feasibility warnings."
                elif decision == "Pending":
                    review_comment = "Pending sourcing manager review."
                else:
                    review_comment = "Rejected due to policy or portfolio concerns."

        reviewed_at = pd.to_datetime(run["started_at"]) + timedelta(hours=random.randint(1, 24))
        sla_hours = round((reviewed_at - pd.to_datetime(run["started_at"])).total_seconds() / 3600, 2)

        rows.append(
            {
                "approval_id": f"APR-{run['workflow_run_id'].split('-')[1]}-{run['run_version']:02d}",
                "workflow_run_id": run["workflow_run_id"],
                "request_id": run["request_id"],
                "approval_required": approval_required,
                "decision": decision,
                "reviewer_name": reviewer_name,
                "review_comment": review_comment,
                "reviewed_at": iso(reviewed_at),
                "sla_hours": sla_hours,
            }
        )

    return pd.DataFrame(rows)


def build_recommendation_outputs(
    requests_df: pd.DataFrame,
    workflow_runs_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    approvals_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for _, run in workflow_runs_df.iterrows():
        req = requests_df.loc[requests_df["request_id"] == run["request_id"]].iloc[0]
        val = validation_df.loc[validation_df["workflow_run_id"] == run["workflow_run_id"]].iloc[0]
        appr = approvals_df.loc[approvals_df["workflow_run_id"] == run["workflow_run_id"]].iloc[0]

        can_generate = (
            val["validation_status"] != "failed"
            and appr["decision"] not in {"Rejected", "Pending"}
        )

        generated_at = pd.to_datetime(run["last_updated_at"]) + timedelta(minutes=20)

        if can_generate:
            selected_suppliers = random.choice([3, 4, 5, 6])
            estimated_total_cost = round(
                float(req["requested_demand_units"]) * random.uniform(8.5, 15.5), 2
            )
            headline = (
                f"Allocate {int(req['requested_demand_units']):,} units across "
                f"{selected_suppliers} suppliers."
            )
            detail = random.choice(
                [
                    "Cost remains the main driver with acceptable ESG and risk balance.",
                    "Portfolio is diversified but slightly more expensive due to tighter ESG requirements.",
                    "Recommended allocation prioritizes lower-risk supply while preserving coverage.",
                ]
            )
            recommendation_status = "generated"
        else:
            selected_suppliers = np.nan
            estimated_total_cost = np.nan
            headline = "Recommendation not produced."
            detail = "Workflow stopped before execution due to approval or validation state."
            recommendation_status = "not_generated"

        rows.append(
            {
                "recommendation_id": f"REC-{run['workflow_run_id'].split('-')[1]}-{run['run_version']:02d}",
                "workflow_run_id": run["workflow_run_id"],
                "request_id": run["request_id"],
                "recommendation_status": recommendation_status,
                "headline": headline,
                "detail": detail,
                "selected_suppliers": selected_suppliers,
                "estimated_total_cost": estimated_total_cost,
                "generated_at": iso(generated_at),
            }
        )

    return pd.DataFrame(rows)


def build_artifacts(
    workflow_runs_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    approvals_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for _, run in workflow_runs_df.iterrows():
        val = validation_df.loc[validation_df["workflow_run_id"] == run["workflow_run_id"]].iloc[0]
        appr = approvals_df.loc[approvals_df["workflow_run_id"] == run["workflow_run_id"]].iloc[0]

        artifact_types = ["parsed_scenario", "validation_report"]

        can_execute = (
            val["validation_status"] != "failed"
            and appr["decision"] not in {"Rejected", "Pending"}
        )
        if can_execute:
            artifact_types.extend(
                [
                    "optimization_result",
                    "decision_audit",
                    "sensitivity_report",
                    "recommendation_packet",
                ]
            )

        base_created_at = pd.to_datetime(run["started_at"])

        for i, artifact_type in enumerate(artifact_types, start=1):
            rows.append(
                {
                    "artifact_id": f"ART-{run['workflow_run_id'].split('-')[1]}-{run['run_version']:02d}-{i:02d}",
                    "workflow_run_id": run["workflow_run_id"],
                    "request_id": run["request_id"],
                    "artifact_type": artifact_type,
                    "version_no": int(run["run_version"]),
                    "storage_uri": (
                        f"artifacts/{run['request_id']}/{run['workflow_run_id']}/"
                        f"{artifact_type}_v{int(run['run_version'])}.json"
                    ),
                    "created_at": iso(base_created_at + timedelta(minutes=10 * i)),
                    "created_by": "system",
                    "is_latest_version": 1,
                }
            )

    return pd.DataFrame(rows)


def build_workflow_trace(
    workflow_runs_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    approvals_df: pd.DataFrame,
    recommendations_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    trace_counter = 1

    for _, run in workflow_runs_df.iterrows():
        val = validation_df.loc[validation_df["workflow_run_id"] == run["workflow_run_id"]].iloc[0]
        appr = approvals_df.loc[approvals_df["workflow_run_id"] == run["workflow_run_id"]].iloc[0]
        rec = recommendations_df.loc[recommendations_df["workflow_run_id"] == run["workflow_run_id"]].iloc[0]

        started = pd.to_datetime(run["started_at"])

        stage_events = [
            (
                "request_intake",
                "Success",
                "request_received",
                "Request recorded and assigned workflow run.",
                started,
            ),
            (
                "parse",
                "Success",
                "scenario_parsed",
                "Natural-language request converted to structured scenario fields.",
                started + timedelta(minutes=10),
            ),
            (
                "validate",
                "Fail" if val["validation_status"] == "failed" else "Success",
                "validation_completed",
                val["validation_summary"],
                started + timedelta(minutes=20),
            ),
            (
                "approve",
                "Fail" if appr["decision"] == "Rejected" else (
                    "Pending" if appr["decision"] == "Pending" else "Success"
                ),
                "approval_decision",
                appr["review_comment"],
                started + timedelta(minutes=40),
            ),
        ]

        can_execute = (
            val["validation_status"] != "failed"
            and appr["decision"] not in {"Rejected", "Pending"}
        )

        if can_execute:
            stage_events.extend(
                [
                    (
                        "execute",
                        "Success",
                        "optimization_completed",
                        "Optimization, audit, and sensitivity analysis completed.",
                        started + timedelta(hours=2),
                    ),
                    (
                        "recommend",
                        "Success",
                        "recommendation_generated",
                        rec["headline"],
                        started + timedelta(hours=2, minutes=20),
                    ),
                ]
            )

        for stage_name, status, event_type, event_message, ts in stage_events:
            rows.append(
                {
                    "trace_id": f"TRC-{trace_counter:05d}",
                    "workflow_run_id": run["workflow_run_id"],
                    "request_id": run["request_id"],
                    "stage_name": stage_name,
                    "status": status,
                    "event_type": event_type,
                    "event_message": event_message,
                    "timestamp": iso(ts),
                }
            )
            trace_counter += 1

    return pd.DataFrame(rows)


def build_data_dictionary(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    purpose_map = {
        "sourcing_requests.csv": "Raw sourcing requests entering the orchestrator.",
        "workflow_runs.csv": "Execution instances for each request, including reruns.",
        "validation_results.csv": "Validation-agent output for each workflow run.",
        "approvals.csv": "Human or system approval outcomes.",
        "workflow_trace.csv": "Append-only stage-level trace log for each workflow run.",
        "artifacts.csv": "Versioned artifact registry for each workflow run.",
        "recommendation_outputs.csv": "Final recommendation payload for executable runs.",
    }

    rows = []

    for filename, df in datasets.items():
        for col in df.columns:
            rows.append(
                {
                    "dataset_name": filename,
                    "column_name": col,
                    "dtype_example": str(df[col].dtype),
                    "sample_value": "" if df.empty else str(df[col].iloc[0]),
                    "dataset_purpose": purpose_map[filename],
                }
            )

    return pd.DataFrame(rows)


# -----------------------------
# Cross-checks / validations
# -----------------------------
def validate_outputs(
    suppliers_df: pd.DataFrame,
    requests_df: pd.DataFrame,
    workflow_runs_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    approvals_df: pd.DataFrame,
    trace_df: pd.DataFrame,
    artifacts_df: pd.DataFrame,
    recommendations_df: pd.DataFrame,
) -> None:
    # basic cardinality
    if len(suppliers_df) != 150:
        raise ValueError(
            f"Expected existing supplier dataset to have 150 rows, found {len(suppliers_df)}."
        )

    if requests_df["request_id"].duplicated().any():
        raise ValueError("Duplicate request_id found.")

    if workflow_runs_df["workflow_run_id"].duplicated().any():
        raise ValueError("Duplicate workflow_run_id found.")

    # FK checks
    request_ids = set(requests_df["request_id"])
    run_ids = set(workflow_runs_df["workflow_run_id"])

    for df_name, df, col in [
        ("workflow_runs", workflow_runs_df, "request_id"),
        ("validation_results", validation_df, "request_id"),
        ("approvals", approvals_df, "request_id"),
        ("workflow_trace", trace_df, "request_id"),
        ("artifacts", artifacts_df, "request_id"),
        ("recommendation_outputs", recommendations_df, "request_id"),
    ]:
        missing = set(df[col]) - request_ids
        if missing:
            raise ValueError(f"{df_name}.{col} contains unknown request_id values: {sorted(missing)[:5]}")

    for df_name, df, col in [
        ("validation_results", validation_df, "workflow_run_id"),
        ("approvals", approvals_df, "workflow_run_id"),
        ("workflow_trace", trace_df, "workflow_run_id"),
        ("artifacts", artifacts_df, "workflow_run_id"),
        ("recommendation_outputs", recommendations_df, "workflow_run_id"),
    ]:
        missing = set(df[col]) - run_ids
        if missing:
            raise ValueError(f"{df_name}.{col} contains unknown workflow_run_id values: {sorted(missing)[:5]}")

    # 1:1 checks
    if validation_df["workflow_run_id"].duplicated().any():
        raise ValueError("validation_results must have one row per workflow_run_id.")

    if approvals_df["workflow_run_id"].duplicated().any():
        raise ValueError("approvals must have one row per workflow_run_id.")

    if recommendations_df["workflow_run_id"].duplicated().any():
        raise ValueError("recommendation_outputs must have one row per workflow_run_id.")

    # recommendation generation logic
    merged = (
        workflow_runs_df[["workflow_run_id", "request_id"]]
        .merge(validation_df[["workflow_run_id", "validation_status"]], on="workflow_run_id")
        .merge(approvals_df[["workflow_run_id", "decision"]], on="workflow_run_id")
        .merge(recommendations_df[["workflow_run_id", "recommendation_status"]], on="workflow_run_id")
    )

    bad_rows = merged[
        (
            (merged["validation_status"] == "failed")
            | (merged["decision"].isin(["Rejected", "Pending"]))
        )
        & (merged["recommendation_status"] == "generated")
    ]
    if not bad_rows.empty:
        raise ValueError("Found generated recommendations for blocked workflow runs.")

    # region alignment with real supplier file
    supplier_regions = set(suppliers_df["region"].dropna().unique().tolist())
    request_regions = set()
    for raw in requests_df["blocked_regions_raw"]:
        parts = [x for x in str(raw).split("|") if x and x != "nan"]
        request_regions.update(parts)

    if not request_regions.issubset(supplier_regions):
        raise ValueError(
            "Some blocked regions in sourcing_requests.csv do not exist in suppliers_dataset.csv."
        )

    # trace stage ordering
    stage_order = {
        "request_intake": 1,
        "parse": 2,
        "validate": 3,
        "approve": 4,
        "execute": 5,
        "recommend": 6,
    }

    for run_id, grp in trace_df.groupby("workflow_run_id"):
        seq = grp.sort_values("timestamp")["stage_name"].map(stage_order).tolist()
        if seq != sorted(seq):
            raise ValueError(f"Trace stage order is invalid for {run_id}.")


# -----------------------------
# Save outputs
# -----------------------------
def save_csv(df: pd.DataFrame, filename: str) -> None:
    out_path = DATA_DIR / filename
    df.to_csv(out_path, index=False)


def main() -> None:
    ensure_data_dir()
    suppliers_df = load_supplier_data()

    requests_df = build_sourcing_requests(suppliers_df)
    workflow_runs_df = build_workflow_runs(requests_df)
    validation_df = build_validation_results(suppliers_df, requests_df, workflow_runs_df)
    approvals_df = build_approvals(requests_df, workflow_runs_df, validation_df)
    recommendations_df = build_recommendation_outputs(
        requests_df, workflow_runs_df, validation_df, approvals_df
    )
    artifacts_df = build_artifacts(workflow_runs_df, validation_df, approvals_df)
    trace_df = build_workflow_trace(
        workflow_runs_df, validation_df, approvals_df, recommendations_df
    )

    # backfill request status from downstream state
    request_status_map = {}
    for request_id, grp in workflow_runs_df.groupby("request_id"):
        latest_run = grp.sort_values("run_version").iloc[-1]
        val = validation_df.loc[validation_df["workflow_run_id"] == latest_run["workflow_run_id"]].iloc[0]
        appr = approvals_df.loc[approvals_df["workflow_run_id"] == latest_run["workflow_run_id"]].iloc[0]

        request_status_map[request_id] = classify_request_status(
            manual_approval_required=int(
                requests_df.loc[requests_df["request_id"] == request_id, "manual_approval_required"].iloc[0]
            ),
            validation_status=val["validation_status"],
            final_decision=appr["decision"],
        )

    requests_df["status"] = requests_df["request_id"].map(request_status_map)

    # update workflow current_stage / status from downstream state
    updated_current_stage = []
    updated_status = []

    for _, run in workflow_runs_df.iterrows():
        val = validation_df.loc[validation_df["workflow_run_id"] == run["workflow_run_id"]].iloc[0]
        appr = approvals_df.loc[approvals_df["workflow_run_id"] == run["workflow_run_id"]].iloc[0]
        rec = recommendations_df.loc[recommendations_df["workflow_run_id"] == run["workflow_run_id"]].iloc[0]

        if val["validation_status"] == "failed":
            updated_current_stage.append("validate")
            updated_status.append("Failed")
        elif appr["decision"] == "Pending":
            updated_current_stage.append("approve")
            updated_status.append("Awaiting Approval")
        elif appr["decision"] == "Rejected":
            updated_current_stage.append("approve")
            updated_status.append("Rejected")
        elif rec["recommendation_status"] == "generated":
            updated_current_stage.append("recommend")
            updated_status.append("Completed")
        else:
            updated_current_stage.append("execute")
            updated_status.append("In Progress")

    workflow_runs_df["current_stage"] = updated_current_stage
    workflow_runs_df["status"] = updated_status

    datasets = {
        "sourcing_requests.csv": requests_df,
        "workflow_runs.csv": workflow_runs_df,
        "validation_results.csv": validation_df,
        "approvals.csv": approvals_df,
        "workflow_trace.csv": trace_df,
        "artifacts.csv": artifacts_df,
        "recommendation_outputs.csv": recommendations_df,
    }

    data_dictionary_df = build_data_dictionary(datasets)

    validate_outputs(
        suppliers_df=suppliers_df,
        requests_df=requests_df,
        workflow_runs_df=workflow_runs_df,
        validation_df=validation_df,
        approvals_df=approvals_df,
        trace_df=trace_df,
        artifacts_df=artifacts_df,
        recommendations_df=recommendations_df,
    )

    for filename, df in datasets.items():
        save_csv(df, filename)
    save_csv(data_dictionary_df, "data_dictionary.csv")

    print("\nGenerated datasets successfully in:")
    print(DATA_DIR)
    print("\nExisting supplier file kept unchanged:")
    print(f"- suppliers_dataset.csv ({len(suppliers_df)} rows)")

    print("\nNew files:")
    for filename, df in datasets.items():
        print(f"- {filename}: {len(df)} rows")
    print(f"- data_dictionary.csv: {len(data_dictionary_df)} rows")

    print("\nSupplier regions detected from existing file:")
    for region in sorted(suppliers_df["region"].dropna().unique().tolist()):
        print(f"- {region}")

    print(
        "\nImportant follow-up: update parser.py ALLOWED_REGIONS to match these real regions, "
        "otherwise parsing and blocked-region logic will stay misaligned."
    )


if __name__ == "__main__":
    main()