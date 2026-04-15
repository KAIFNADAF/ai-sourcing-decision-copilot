# src/recommendation_agent.py

from __future__ import annotations

from datetime import datetime
from typing import Dict, Any, Optional, List

import pandas as pd


class RecommendationAgent:
    """
    Builds a workflow-friendly recommendation payload from optimizer,
    audit, and sensitivity outputs.

    This layer does not decide whether the workflow should proceed.
    It assumes the orchestrator has already decided the run is allowed
    to execute.
    """

    def __init__(self) -> None:
        pass

    def _safe_top_suppliers(self, allocations: pd.DataFrame, top_n: int = 3) -> List[str]:
        if allocations is None or allocations.empty:
            return []

        return (
            allocations.sort_values(by="allocation_qty", ascending=False)["supplier_name"]
            .head(top_n)
            .astype(str)
            .tolist()
        )

    def _build_headline(self, result: Dict[str, Any]) -> str:
        summary = result.get("summary", {})
        demand = summary.get("total_demand")
        selected_suppliers = summary.get("selected_suppliers")

        if demand is None or selected_suppliers is None:
            return "Recommendation generated, but summary details are incomplete."

        return f"Allocate {int(demand):,} units across {int(selected_suppliers)} suppliers."

    def _build_detail(
        self,
        result: Dict[str, Any],
        audit: Dict[str, Any],
        sensitivity_output: Dict[str, Any],
    ) -> str:
        if result.get("status") != "Optimal":
            return "No executable recommendation is available because the scenario is not feasible."

        summary = result.get("summary", {})
        portfolio_checks = audit.get("portfolio_checks", {}) if audit else {}
        insights = sensitivity_output.get("insights", []) if sensitivity_output else []

        weighted_avg_esg = summary.get("weighted_avg_esg")
        weighted_avg_risk = summary.get("weighted_avg_risk")
        total_cost = summary.get("total_cost")

        tradeoff_driver = audit.get("decision_summary", {}).get("tradeoff_driver")
        top_2_concentration = portfolio_checks.get("top_2_supplier_concentration_pct")

        parts: List[str] = []

        if total_cost is not None:
            parts.append(f"Estimated total cost is {float(total_cost):,.2f}.")

        if weighted_avg_esg is not None and weighted_avg_risk is not None:
            parts.append(
                f"Portfolio average ESG is {float(weighted_avg_esg):.2f} and average risk is {float(weighted_avg_risk):.2f}."
            )

        if tradeoff_driver:
            parts.append(tradeoff_driver)

        if top_2_concentration is not None:
            parts.append(
                f"The top two suppliers account for {float(top_2_concentration) * 100:.2f}% of awarded volume."
            )

        if insights:
            parts.append(f"Scenario testing suggests: {insights[0]}")

        if not parts:
            return "Recommendation generated successfully."

        return " ".join(parts)

    def _build_risk_flags(
        self,
        audit: Dict[str, Any],
        sensitivity_output: Dict[str, Any],
    ) -> List[str]:
        flags: List[str] = []

        portfolio_checks = audit.get("portfolio_checks", {}) if audit else {}
        insights = sensitivity_output.get("insights", []) if sensitivity_output else []

        concentration = portfolio_checks.get("top_2_supplier_concentration_pct")
        esg_tightness = portfolio_checks.get("esg_tightness")
        risk_tightness = portfolio_checks.get("risk_tightness")

        if concentration is not None and float(concentration) >= 0.70:
            flags.append("High supplier concentration")

        if esg_tightness in {"binding_or_nearly_binding", "tight"}:
            flags.append("ESG boundary is tight")

        if risk_tightness in {"binding_or_nearly_binding", "tight"}:
            flags.append("Risk ceiling is tight")

        for item in insights:
            lowered = str(item).lower()
            if "infeasible" in lowered:
                flags.append(item)
            elif "increased cost" in lowered:
                flags.append(item)
            elif "reduced concentration" in lowered:
                flags.append(item)

        deduped: List[str] = []
        seen = set()
        for flag in flags:
            if flag not in seen:
                deduped.append(flag)
                seen.add(flag)

        return deduped[:5]

    def _build_next_step(
        self,
        result: Dict[str, Any],
        audit: Dict[str, Any],
        sensitivity_output: Dict[str, Any],
    ) -> str:
        if result.get("status") != "Optimal":
            return "Relax one or more constraints and rerun the workflow."

        portfolio_checks = audit.get("portfolio_checks", {}) if audit else {}
        concentration = portfolio_checks.get("top_2_supplier_concentration_pct")
        esg_tightness = portfolio_checks.get("esg_tightness")

        comparison_table = (
            sensitivity_output.get("comparison_table", pd.DataFrame())
            if sensitivity_output else pd.DataFrame()
        )

        if concentration is not None and float(concentration) >= 0.70:
            return "Test a lower supplier share cap if resilience matters more than concentration risk."

        if esg_tightness in {"binding_or_nearly_binding", "tight"}:
            return "Test a slightly higher ESG floor before changing sourcing priorities."

        if not comparison_table.empty:
            blocked_row = comparison_table[
                comparison_table["scenario_name"] == "block_eastern_europe"
            ]
            if not blocked_row.empty and blocked_row.iloc[0]["status"] != "Optimal":
                return "Add more non-Eastern-Europe capacity before applying stricter regional exclusions."

        return "Use this recommendation as the baseline and test one tighter business rule at a time."

    def build_recommendation(
        self,
        workflow_run_id: str,
        request_id: str,
        result: Dict[str, Any],
        audit: Dict[str, Any],
        sensitivity_output: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build a recommendation payload suitable for recommendation_outputs.csv
        or future artifact storage.
        """
        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if result.get("status") != "Optimal":
            return {
                "recommendation_id": None,
                "workflow_run_id": workflow_run_id,
                "request_id": request_id,
                "recommendation_status": "not_generated",
                "headline": "Recommendation not produced.",
                "detail": "The workflow did not produce an executable sourcing recommendation.",
                "selected_suppliers": None,
                "estimated_total_cost": None,
                "generated_at": generated_at,
                "top_suppliers": [],
                "risk_flags": [],
                "next_step": "Relax one or more constraints and rerun the workflow.",
            }

        summary = result.get("summary", {})
        allocations = result.get("allocations", pd.DataFrame())

        recommendation = {
            "recommendation_id": None,  # assigned later if needed by orchestrator/store layer
            "workflow_run_id": workflow_run_id,
            "request_id": request_id,
            "recommendation_status": "generated",
            "headline": self._build_headline(result),
            "detail": self._build_detail(result, audit, sensitivity_output),
            "selected_suppliers": summary.get("selected_suppliers"),
            "estimated_total_cost": summary.get("total_cost"),
            "generated_at": generated_at,
            "top_suppliers": self._safe_top_suppliers(allocations),
            "risk_flags": self._build_risk_flags(audit, sensitivity_output),
            "next_step": self._build_next_step(result, audit, sensitivity_output),
        }

        return recommendation


if __name__ == "__main__":
    from src.workflow_store import WorkflowStore
    from src.optimizer import load_suppliers, optimize_supplier_allocation
    from src.audit import generate_decision_audit
    from src.sensitivity import run_sensitivity_analysis

    store = WorkflowStore(data_dir="data")
    request = store.load_requests().iloc[0].to_dict()
    run = store.get_latest_run_for_request(request["request_id"])

    suppliers_df = load_suppliers("data/suppliers_dataset.csv")

    blocked_regions = [
        x for x in str(request["blocked_regions_raw"]).split("|")
        if x and x != "nan"
    ]

    base_params = {
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
    sensitivity_output = run_sensitivity_analysis(df=suppliers_df, base_params=base_params)

    agent = RecommendationAgent()
    recommendation = agent.build_recommendation(
        workflow_run_id=run["workflow_run_id"],
        request_id=request["request_id"],
        result=result,
        audit=audit,
        sensitivity_output=sensitivity_output,
    )

    print("Recommendation payload:")
    for key, value in recommendation.items():
        print(f"{key}: {value}")