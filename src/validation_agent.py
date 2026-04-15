from __future__ import annotations

from datetime import datetime
from typing import Dict, Any, List, Optional

import pandas as pd

from src.workflow_store import WorkflowStore


class ValidationAgent:
    """
    Applies lightweight business-rule validation to a workflow run
    using either:
    - the normalized scenario payload supplied by the orchestrator, or
    - the request inputs as a fallback.

    This keeps validation aligned with execution when a parsed/normalized
    scenario is available.
    """

    def __init__(self, store: WorkflowStore, suppliers_path: str = "data/suppliers_dataset.csv") -> None:
        self.store = store
        self.suppliers_path = suppliers_path

    def _load_suppliers(self) -> pd.DataFrame:
        df = pd.read_csv(self.suppliers_path)

        required_columns = {
            "supplier_id",
            "supplier_name",
            "region",
            "unit_cost",
            "max_capacity",
            "risk_score",
            "esg_score",
            "compliance_flag",
        }
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(
                "Supplier dataset is missing required validation columns: "
                + ", ".join(sorted(missing))
            )

        return df

    def _parse_blocked_regions(self, raw_value: str) -> List[str]:
        if pd.isna(raw_value) or str(raw_value).strip() == "":
            return []
        return [x.strip() for x in str(raw_value).split("|") if x.strip()]

    def _build_scenario_from_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fallback behavior for older callers that do not supply a normalized scenario.
        """
        return {
            "total_demand": int(request["requested_demand_units"]),
            "max_supplier_share": float(request["requested_max_supplier_share"]),
            "min_avg_esg": int(request["requested_min_avg_esg"]),
            "max_avg_risk": int(request["requested_max_avg_risk"]),
            "min_suppliers": int(request["requested_min_suppliers"]),
            "blocked_regions": self._parse_blocked_regions(request["blocked_regions_raw"]),
            "w_cost": 0.65,
            "w_risk": 0.20,
            "w_esg": 0.10,
            "supplier_selection_penalty": 0.02,
        }

    def evaluate_run(
        self,
        workflow_run_id: str,
        scenario: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        run = self.store.get_run(workflow_run_id)
        request = self.store.get_request(run["request_id"])
        suppliers_df = self._load_suppliers()

        effective_scenario = scenario or self._build_scenario_from_request(request)

        eligible_df = suppliers_df[suppliers_df["compliance_flag"] == 1].copy()

        warnings: List[str] = []
        failures: List[str] = []

        demand = int(effective_scenario["total_demand"])
        min_suppliers = int(effective_scenario["min_suppliers"])
        min_esg = int(effective_scenario["min_avg_esg"])
        max_risk = int(effective_scenario["max_avg_risk"])
        max_share = float(effective_scenario["max_supplier_share"])
        blocked_regions = effective_scenario.get("blocked_regions", []) or []

        if blocked_regions:
            eligible_df = eligible_df[~eligible_df["region"].isin(blocked_regions)].copy()

        eligible_capacity = float(eligible_df["max_capacity"].sum()) if not eligible_df.empty else 0.0
        eligible_supplier_count = int(len(eligible_df))

        if eligible_supplier_count == 0:
            failures.append("No eligible compliant suppliers remain after blocked-region filtering.")

        if eligible_capacity < demand:
            failures.append("Eligible capacity after region filters is below requested demand.")

        if eligible_supplier_count < min_suppliers:
            failures.append("Eligible supplier count is below the minimum supplier requirement.")

        if demand > eligible_capacity * 0.55 and eligible_capacity > 0:
            warnings.append("Demand is large relative to the current eligible capacity pool.")

        if max_share <= 0.25:
            warnings.append("Tight supplier share cap may increase infeasibility risk.")

        if min_esg >= 82:
            warnings.append("High ESG target may increase cost or reduce feasibility.")

        if max_risk <= 30:
            warnings.append("Low risk ceiling may reduce the feasible supplier set.")

        if len(blocked_regions) >= 2:
            warnings.append("Multiple blocked regions narrow the supplier pool.")

        if min_suppliers >= 5:
            warnings.append("Higher minimum supplier count may reduce optimizer flexibility.")

        if failures:
            validation_status = "failed"
            schema_check_passed = 1
            business_rules_passed = 0
            validation_summary = "Scenario failed business-rule validation."
            recommended_action = (
                "Relax demand, blocked regions, or portfolio constraints before execution."
            )
        elif len(warnings) >= 2:
            validation_status = "warning"
            schema_check_passed = 1
            business_rules_passed = 1
            validation_summary = "Scenario passed validation with elevated feasibility risk."
            recommended_action = (
                "Proceed to approval with reviewer attention on tight constraints."
            )
        else:
            validation_status = "passed"
            schema_check_passed = 1
            business_rules_passed = 1
            validation_summary = "Scenario passed schema and business-rule validation."
            recommended_action = "Proceed to execution."

        checked_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        scenario_source = "parsed_scenario" if scenario is not None else "request_fallback"

        return {
            "validation_id": f"VAL-{workflow_run_id.split('-')[1]}-{int(run['run_version']):02d}",
            "workflow_run_id": workflow_run_id,
            "request_id": run["request_id"],
            "validation_status": validation_status,
            "schema_check_passed": schema_check_passed,
            "business_rules_passed": business_rules_passed,
            "warning_count": len(warnings),
            "failure_count": len(failures),
            "validation_summary": validation_summary,
            "recommended_action": recommended_action,
            "checked_at": checked_at,
            "debug_context": {
                "scenario_source": scenario_source,
                "eligible_supplier_count": eligible_supplier_count,
                "eligible_capacity": eligible_capacity,
                "blocked_regions": blocked_regions,
                "warnings": warnings,
                "failures": failures,
                "validated_scenario": effective_scenario,
            },
        }