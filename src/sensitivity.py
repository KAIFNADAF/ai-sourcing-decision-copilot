from typing import Dict, Any, List

import pandas as pd

from src.optimizer import load_suppliers, optimize_supplier_allocation


def _extract_selected_supplier_names(result: Dict[str, Any], top_n: int = 5) -> List[str]:
    allocations = result.get("allocations", pd.DataFrame())

    if allocations.empty:
        return []

    return (
        allocations.sort_values(by="allocation_qty", ascending=False)["supplier_name"]
        .head(top_n)
        .tolist()
    )


def _extract_top_supplier(result: Dict[str, Any]) -> str:
    allocations = result.get("allocations", pd.DataFrame())

    if allocations.empty:
        return "None"

    top_row = allocations.sort_values(by="allocation_qty", ascending=False).iloc[0]
    return str(top_row["supplier_name"])


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


def _build_scenario_row(
    scenario_name: str,
    scenario_params: Dict[str, Any],
    result: Dict[str, Any],
) -> Dict[str, Any]:
    row = {
        "scenario_name": scenario_name,
        "status": result.get("status"),
        "message": result.get("message", ""),
        "total_demand": scenario_params.get("total_demand"),
        "max_supplier_share": scenario_params.get("max_supplier_share"),
        "min_avg_esg": scenario_params.get("min_avg_esg"),
        "max_avg_risk": scenario_params.get("max_avg_risk"),
        "min_suppliers": scenario_params.get("min_suppliers"),
        "blocked_regions": ", ".join(scenario_params.get("blocked_regions", [])),
    }

    if result.get("status") == "Optimal":
        summary = result.get("summary", {})
        allocations = result.get("allocations", pd.DataFrame())

        weighted_avg_esg = summary.get("weighted_avg_esg")
        weighted_avg_risk = summary.get("weighted_avg_risk")

        row.update(
            {
                "selected_suppliers": summary.get("selected_suppliers"),
                "total_cost": summary.get("total_cost"),
                "weighted_avg_esg": weighted_avg_esg,
                "weighted_avg_risk": weighted_avg_risk,
                "esg_tightness": _classify_tightness(
                    float(weighted_avg_esg), float(scenario_params["min_avg_esg"])
                )
                if weighted_avg_esg is not None
                else None,
                "risk_tightness": _classify_tightness(
                    float(weighted_avg_risk), float(scenario_params["max_avg_risk"])
                )
                if weighted_avg_risk is not None
                else None,
                "top_supplier": _extract_top_supplier(result),
                "top_2_concentration_pct": round(
                    float(
                        allocations.sort_values(by="allocation_qty", ascending=False)
                        .head(2)["allocation_pct"]
                        .sum()
                    ),
                    4,
                )
                if not allocations.empty
                else None,
                "selected_supplier_names": ", ".join(_extract_selected_supplier_names(result)),
            }
        )
    else:
        row.update(
            {
                "selected_suppliers": None,
                "total_cost": None,
                "weighted_avg_esg": None,
                "weighted_avg_risk": None,
                "esg_tightness": None,
                "risk_tightness": None,
                "top_supplier": None,
                "top_2_concentration_pct": None,
                "selected_supplier_names": None,
            }
        )

    return row


def _generate_sensitivity_insights(
    comparison_df: pd.DataFrame,
    base_result: Dict[str, Any],
) -> List[str]:
    insights = []

    if comparison_df.empty:
        return ["No scenarios were available for comparison."]

    base_row_df = comparison_df[comparison_df["scenario_name"] == "base_case"]
    if base_row_df.empty:
        return ["Base case was not found in the scenario comparison output."]

    base_row = base_row_df.iloc[0]
    optimal_rows = comparison_df[comparison_df["status"] == "Optimal"].copy()

    if optimal_rows.empty:
        return ["None of the tested scenarios produced an optimal solution."]

    feasible_count = len(optimal_rows)
    total_count = len(comparison_df)

    if feasible_count == total_count:
        insights.append("The current setup stayed feasible across all tested scenarios.")
    else:
        insights.append(f"{feasible_count} of {total_count} tested scenarios remained feasible.")

    stricter_esg_df = comparison_df[comparison_df["scenario_name"] == "stricter_esg"]
    if not stricter_esg_df.empty:
        stricter_esg_row = stricter_esg_df.iloc[0]
        if stricter_esg_row["status"] == "Optimal":
            if pd.notna(base_row["total_cost"]) and pd.notna(stricter_esg_row["total_cost"]):
                cost_diff = round(float(stricter_esg_row["total_cost"]) - float(base_row["total_cost"]), 2)
                if abs(cost_diff) >= 1.0:
                    if cost_diff > 0:
                        insights.append(f"Raising the ESG floor increased cost by {cost_diff}.")
                    else:
                        insights.append(f"Raising the ESG floor reduced cost by {abs(cost_diff)}.")
        else:
            insights.append("Raising the ESG floor made the case infeasible.")

    tighter_share_df = comparison_df[comparison_df["scenario_name"] == "tighter_supplier_share"]
    if not tighter_share_df.empty:
        tighter_share_row = tighter_share_df.iloc[0]
        if tighter_share_row["status"] == "Optimal":
            if pd.notna(base_row["top_2_concentration_pct"]) and pd.notna(tighter_share_row["top_2_concentration_pct"]):
                concentration_diff = round(
                    float(tighter_share_row["top_2_concentration_pct"]) - float(base_row["top_2_concentration_pct"]),
                    4,
                )
                if concentration_diff < -0.001:
                    insights.append("Tightening the share cap reduced concentration.")
                elif concentration_diff > 0.001:
                    insights.append("Tightening the share cap increased concentration, which should be reviewed.")

    blocked_region_df = comparison_df[comparison_df["scenario_name"] == "block_eastern_europe"]
    if not blocked_region_df.empty:
        blocked_region_row = blocked_region_df.iloc[0]
        if blocked_region_row["status"] == "Optimal":
            if pd.notna(base_row["total_cost"]) and pd.notna(blocked_region_row["total_cost"]):
                cost_diff = round(float(blocked_region_row["total_cost"]) - float(base_row["total_cost"]), 2)
                if abs(cost_diff) >= 1.0:
                    if cost_diff > 0:
                        insights.append(
                            f"Removing Eastern Europe kept the case feasible but increased cost by {cost_diff}."
                        )
                    else:
                        insights.append(
                            f"Removing Eastern Europe kept the case feasible and reduced cost by {abs(cost_diff)}."
                        )
        else:
            insights.append("Removing Eastern Europe made the case infeasible.")

    higher_demand_df = comparison_df[comparison_df["scenario_name"] == "higher_demand"]
    if not higher_demand_df.empty:
        higher_demand_row = higher_demand_df.iloc[0]
        if higher_demand_row["status"] != "Optimal":
            insights.append("The case failed when demand increased by 25%.")
        elif pd.notna(base_row["total_cost"]) and pd.notna(higher_demand_row["total_cost"]):
            cost_diff = round(float(higher_demand_row["total_cost"]) - float(base_row["total_cost"]), 2)
            if abs(cost_diff) >= 1.0:
                if cost_diff > 0:
                    insights.append(f"Raising demand by 25% increased cost by {cost_diff}.")
                else:
                    insights.append(f"Raising demand by 25% reduced cost by {abs(cost_diff)}.")

    if not optimal_rows.empty:
        lowest_cost_row = optimal_rows.loc[optimal_rows["total_cost"].idxmin()]
        highest_esg_row = optimal_rows.loc[optimal_rows["weighted_avg_esg"].idxmax()]

        if lowest_cost_row["scenario_name"] != "base_case":
            insights.append(f"Lowest-cost feasible case: {lowest_cost_row['scenario_name']}.")
        if highest_esg_row["scenario_name"] != "base_case":
            insights.append(f"Highest-ESG feasible case: {highest_esg_row['scenario_name']}.")

    cleaned = []
    seen = set()
    for insight in insights:
        if insight not in seen:
            cleaned.append(insight)
            seen.add(insight)

    return cleaned[:4]


def run_sensitivity_analysis(
    df: pd.DataFrame,
    base_params: Dict[str, Any],
) -> Dict[str, Any]:
    scenarios = {
        "base_case": dict(base_params),
        "stricter_esg": {
            **base_params,
            "min_avg_esg": base_params["min_avg_esg"] + 5,
        },
        "tighter_supplier_share": {
            **base_params,
            "max_supplier_share": 0.30,
        },
        "higher_min_suppliers": {
            **base_params,
            "min_suppliers": base_params["min_suppliers"] + 1,
        },
        "block_eastern_europe": {
            **base_params,
            "blocked_regions": ["Eastern Europe"],
        },
        "higher_demand": {
            **base_params,
            "total_demand": int(base_params["total_demand"] * 1.25),
        },
    }

    results_by_scenario = {}
    summary_rows = []

    for scenario_name, params in scenarios.items():
        result = optimize_supplier_allocation(
            df=df,
            total_demand=params["total_demand"],
            max_supplier_share=params["max_supplier_share"],
            min_avg_esg=params["min_avg_esg"],
            max_avg_risk=params["max_avg_risk"],
            min_suppliers=params["min_suppliers"],
            blocked_regions=params.get("blocked_regions", []),
            w_cost=params["w_cost"],
            w_risk=params["w_risk"],
            w_esg=params["w_esg"],
            supplier_selection_penalty=params["supplier_selection_penalty"],
        )

        results_by_scenario[scenario_name] = result
        summary_rows.append(_build_scenario_row(scenario_name, params, result))

    comparison_df = pd.DataFrame(summary_rows)

    base_result = results_by_scenario["base_case"]
    scenario_insights = _generate_sensitivity_insights(comparison_df, base_result)

    return {
        "comparison_table": comparison_df,
        "scenario_results": results_by_scenario,
        "insights": scenario_insights,
    }


if __name__ == "__main__":
    suppliers_df = load_suppliers()

    base_params = {
        "total_demand": 120000,
        "max_supplier_share": 0.40,
        "min_avg_esg": 70,
        "max_avg_risk": 45,
        "min_suppliers": 3,
        "blocked_regions": [],
        "w_cost": 0.65,
        "w_risk": 0.20,
        "w_esg": 0.10,
        "supplier_selection_penalty": 0.02,
    }

    sensitivity_output = run_sensitivity_analysis(
        df=suppliers_df,
        base_params=base_params,
    )

    print(sensitivity_output["comparison_table"].to_string(index=False))
    print("\nKEY INSIGHTS:")
    for insight in sensitivity_output["insights"]:
        print("-", insight)