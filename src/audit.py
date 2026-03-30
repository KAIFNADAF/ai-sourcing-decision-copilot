from typing import Dict, Any, List

import pandas as pd

from src.optimizer import load_suppliers, optimize_supplier_allocation


def _safe_pct(value: float) -> float:
    return round(value * 100, 2)


def _classify_tightness(actual: float, limit: float) -> str:
    if limit == 0:
        return "unknown"

    gap_ratio = abs(actual - limit) / abs(limit)

    if gap_ratio <= 0.01:
        return "binding_or_nearly_binding"
    if gap_ratio <= 0.05:
        return "tight"
    if gap_ratio <= 0.15:
        return "moderately_loose"
    return "loose"


def _build_supplier_flags(
    allocations_df: pd.DataFrame,
    scenario: Dict[str, Any],
) -> List[Dict[str, Any]]:
    flags = []
    max_supplier_share = scenario["max_supplier_share"]
    min_avg_esg = scenario["min_avg_esg"]
    max_avg_risk = scenario["max_avg_risk"]

    for _, row in allocations_df.iterrows():
        allocation_pct = float(row["allocation_pct"])
        supplier_flags = []

        if abs(allocation_pct - max_supplier_share) <= 0.01:
            supplier_flags.append("near_share_cap")

        if row["esg_score"] >= min_avg_esg + 10:
            supplier_flags.append("strong_esg")
        elif row["esg_score"] < min_avg_esg:
            supplier_flags.append("below_target_esg")

        if row["risk_score"] >= max_avg_risk - 5:
            supplier_flags.append("higher_risk")
        elif row["risk_score"] <= max_avg_risk - 15:
            supplier_flags.append("lower_risk")

        flags.append(
            {
                "supplier_id": row["supplier_id"],
                "supplier_name": row["supplier_name"],
                "region": row["region"],
                "allocation_qty": float(row["allocation_qty"]),
                "allocation_pct": round(allocation_pct, 4),
                "unit_cost": float(row["unit_cost"]),
                "risk_score": float(row["risk_score"]),
                "esg_score": float(row["esg_score"]),
                "flags": supplier_flags,
            }
        )

    return flags


def generate_decision_audit(result: Dict[str, Any]) -> Dict[str, Any]:
    status = result.get("status")
    scenario = result.get("scenario", {})
    metadata = result.get("metadata", {})
    allocations_df = result.get("allocations", pd.DataFrame())
    summary = result.get("summary", {})

    if status != "Optimal":
        return {
            "status": status,
            "message": result.get("message", "Optimization did not return an optimal solution."),
            "scenario_overview": scenario,
            "portfolio_checks": {
                "feasible_solution_found": False,
                "eligible_supplier_count": metadata.get("eligible_supplier_count"),
                "total_eligible_capacity": metadata.get("total_eligible_capacity"),
            },
            "decision_summary": {
                "headline": "No feasible sourcing allocation was found.",
                "detail": result.get("message", ""),
            },
            "plain_english_explanation": (
                "The model could not find a workable allocation under the current demand, "
                "capacity, share, ESG, risk, and region rules."
            ),
        }

    if allocations_df.empty:
        return {
            "status": "Optimal",
            "message": "Optimal status returned but no positive allocations were found.",
            "scenario_overview": scenario,
            "portfolio_checks": {},
            "decision_summary": {
                "headline": "The solver returned an empty allocation.",
                "detail": "This looks like an unexpected result edge case.",
            },
            "plain_english_explanation": (
                "The optimization finished, but no positive supplier allocations were returned."
            ),
        }

    total_demand = float(summary["total_demand"])
    selected_suppliers = int(summary["selected_suppliers"])
    min_suppliers_required = int(summary["min_suppliers_required"])
    weighted_avg_esg = float(summary["weighted_avg_esg"])
    weighted_avg_risk = float(summary["weighted_avg_risk"])
    max_supplier_share = float(summary["max_supplier_share"])
    min_avg_esg = float(scenario["min_avg_esg"])
    max_avg_risk = float(scenario["max_avg_risk"])

    total_allocated = float(allocations_df["allocation_qty"].sum())
    demand_fulfilled = abs(total_allocated - total_demand) <= 1e-3

    esg_tightness = _classify_tightness(weighted_avg_esg, min_avg_esg)
    risk_tightness = _classify_tightness(weighted_avg_risk, max_avg_risk)

    suppliers_at_or_near_share_cap = allocations_df[
        (max_supplier_share - allocations_df["allocation_pct"]).abs() <= 0.01
    ][["supplier_id", "supplier_name", "allocation_pct"]]

    concentration_ratio_top_2 = float(
        allocations_df.sort_values(by="allocation_qty", ascending=False)
        .head(2)["allocation_pct"]
        .sum()
    )

    if esg_tightness in {"binding_or_nearly_binding", "tight"} and risk_tightness in {
        "binding_or_nearly_binding",
        "tight",
    }:
        tradeoff_driver = "Both ESG and risk are actively shaping the portfolio."
    elif esg_tightness in {"binding_or_nearly_binding", "tight"}:
        tradeoff_driver = "The ESG floor is one of the main portfolio constraints."
    elif risk_tightness in {"binding_or_nearly_binding", "tight"}:
        tradeoff_driver = "The risk cap is one of the main portfolio constraints."
    elif concentration_ratio_top_2 >= 0.70:
        tradeoff_driver = "Share limits matter because the allocation is still concentrated."
    else:
        tradeoff_driver = "Cost appears to be the main driver."

    diversification_comment = (
        "The allocation goes beyond the minimum supplier count."
        if selected_suppliers > min_suppliers_required
        else "The allocation is close to the minimum supplier count."
    )

    supplier_flags = _build_supplier_flags(allocations_df, scenario)

    scenario_overview = {
        "total_demand": total_demand,
        "input_supplier_count": metadata.get("input_supplier_count"),
        "compliant_supplier_count": metadata.get("compliant_supplier_count"),
        "eligible_supplier_count": metadata.get("eligible_supplier_count"),
        "total_eligible_capacity": metadata.get("total_eligible_capacity"),
        "blocked_regions": scenario.get("blocked_regions", []),
        "min_suppliers_required": min_suppliers_required,
        "max_supplier_share": max_supplier_share,
        "weights": scenario.get("weights", {}),
        "supplier_selection_penalty": scenario.get("supplier_selection_penalty"),
    }

    portfolio_checks = {
        "feasible_solution_found": True,
        "demand_fulfilled": demand_fulfilled,
        "total_allocated": round(total_allocated, 2),
        "selected_suppliers": selected_suppliers,
        "selected_vs_required": f"{selected_suppliers} selected vs {min_suppliers_required} required",
        "weighted_avg_esg": weighted_avg_esg,
        "min_avg_esg_required": min_avg_esg,
        "esg_tightness": esg_tightness,
        "weighted_avg_risk": weighted_avg_risk,
        "max_avg_risk_allowed": max_avg_risk,
        "risk_tightness": risk_tightness,
        "top_2_supplier_concentration_pct": round(concentration_ratio_top_2, 4),
        "suppliers_at_or_near_share_cap": suppliers_at_or_near_share_cap.to_dict(orient="records"),
    }

    decision_summary = {
        "headline": (
            f"The model allocated {int(total_demand):,} units across {selected_suppliers} suppliers."
        ),
        "detail": (
            f"ESG finished at {weighted_avg_esg:.2f} against a minimum of {min_avg_esg:.2f}. "
            f"Risk finished at {weighted_avg_risk:.2f} against a maximum of {max_avg_risk:.2f}. "
            f"{tradeoff_driver} {diversification_comment}"
        ),
        "tradeoff_driver": tradeoff_driver,
        "diversification_comment": diversification_comment,
    }

    plain_english_explanation = (
        f"The model found a feasible allocation for {int(total_demand):,} units. "
        f"It used {selected_suppliers} suppliers, versus a minimum requirement of {min_suppliers_required}. "
        f"The top two suppliers account for {_safe_pct(concentration_ratio_top_2)}% of awarded volume. "
        f"Average ESG finished at {weighted_avg_esg:.2f} and average risk finished at {weighted_avg_risk:.2f}. "
        f"In simple terms, this looks like a workable portfolio, but the main story still depends on concentration and how tight the ESG and risk rules are."
    )

    return {
        "status": "Optimal",
        "scenario_overview": scenario_overview,
        "portfolio_checks": portfolio_checks,
        "supplier_flags": supplier_flags,
        "decision_summary": decision_summary,
        "plain_english_explanation": plain_english_explanation,
    }


if __name__ == "__main__":
    suppliers_df = load_suppliers()

    result = optimize_supplier_allocation(
        df=suppliers_df,
        total_demand=120000,
        max_supplier_share=0.4,
        min_avg_esg=70,
        max_avg_risk=45,
        min_suppliers=3,
        blocked_regions=[],
        w_cost=0.65,
        w_risk=0.20,
        w_esg=0.10,
        supplier_selection_penalty=0.02,
    )

    audit = generate_decision_audit(result)

    print("\nAUDIT STATUS:", audit["status"])
    if audit["status"] == "Optimal":
        print(audit["decision_summary"])
        print(audit["plain_english_explanation"])