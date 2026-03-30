from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
import pulp


def load_suppliers(csv_path: str = "data/suppliers_dataset.csv") -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found at: {path.resolve()}")
    return pd.read_csv(path)


def min_max_normalize(series: pd.Series) -> pd.Series:
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series([0.0] * len(series), index=series.index)
    return (series - min_val) / (max_val - min_val)


def optimize_supplier_allocation(
    df: pd.DataFrame,
    total_demand: float,
    max_supplier_share: float,
    min_avg_esg: float,
    max_avg_risk: float,
    min_suppliers: int = 2,
    blocked_regions: Optional[List[str]] = None,
    w_cost: float = 0.65,
    w_risk: float = 0.20,
    w_esg: float = 0.10,
    supplier_selection_penalty: float = 0.02,
) -> Dict[str, Any]:
    """
    Product-style sourcing optimization:
    - cost remains the main optimization driver
    - risk and ESG act as both portfolio controls and light trade-off signals
    - diversification is supported through min supplier count
    - blocked regions are supported
    - solver output includes richer metadata for audit and explainability
    """

    blocked_regions = blocked_regions or []

    input_supplier_count = len(df)

    # Step 1: keep only compliant suppliers
    working_df = df[df["compliance_flag"] == 1].copy()
    compliant_supplier_count = len(working_df)

    if working_df.empty:
        return {
            "status": "Infeasible",
            "message": "No compliant suppliers available after compliance filtering.",
            "scenario": {
                "total_demand": total_demand,
                "max_supplier_share": max_supplier_share,
                "min_avg_esg": min_avg_esg,
                "max_avg_risk": max_avg_risk,
                "min_suppliers": min_suppliers,
                "blocked_regions": blocked_regions,
                "weights": {
                    "cost": w_cost,
                    "risk": w_risk,
                    "esg": w_esg,
                },
                "supplier_selection_penalty": supplier_selection_penalty,
            },
            "metadata": {
                "input_supplier_count": input_supplier_count,
                "compliant_supplier_count": compliant_supplier_count,
                "eligible_supplier_count": 0,
                "total_eligible_capacity": 0,
            },
        }

    # Step 2: remove blocked regions
    if blocked_regions:
        working_df = working_df[~working_df["region"].isin(blocked_regions)].copy()

    eligible_supplier_count = len(working_df)

    if working_df.empty:
        return {
            "status": "Infeasible",
            "message": "No suppliers available after applying blocked region filters.",
            "scenario": {
                "total_demand": total_demand,
                "max_supplier_share": max_supplier_share,
                "min_avg_esg": min_avg_esg,
                "max_avg_risk": max_avg_risk,
                "min_suppliers": min_suppliers,
                "blocked_regions": blocked_regions,
                "weights": {
                    "cost": w_cost,
                    "risk": w_risk,
                    "esg": w_esg,
                },
                "supplier_selection_penalty": supplier_selection_penalty,
            },
            "metadata": {
                "input_supplier_count": input_supplier_count,
                "compliant_supplier_count": compliant_supplier_count,
                "eligible_supplier_count": 0,
                "total_eligible_capacity": 0,
            },
        }

    # Step 3: feasibility pre-checks
    total_eligible_capacity = float(working_df["max_capacity"].sum())

    if total_eligible_capacity < total_demand:
        return {
            "status": "Infeasible",
            "message": (
                f"Total eligible capacity ({total_eligible_capacity}) is below "
                f"required demand ({total_demand})."
            ),
            "scenario": {
                "total_demand": total_demand,
                "max_supplier_share": max_supplier_share,
                "min_avg_esg": min_avg_esg,
                "max_avg_risk": max_avg_risk,
                "min_suppliers": min_suppliers,
                "blocked_regions": blocked_regions,
                "weights": {
                    "cost": w_cost,
                    "risk": w_risk,
                    "esg": w_esg,
                },
                "supplier_selection_penalty": supplier_selection_penalty,
            },
            "metadata": {
                "input_supplier_count": input_supplier_count,
                "compliant_supplier_count": compliant_supplier_count,
                "eligible_supplier_count": eligible_supplier_count,
                "total_eligible_capacity": total_eligible_capacity,
            },
        }

    if min_suppliers > eligible_supplier_count:
        return {
            "status": "Infeasible",
            "message": (
                f"Minimum suppliers requested ({min_suppliers}) exceeds available "
                f"eligible suppliers ({eligible_supplier_count})."
            ),
            "scenario": {
                "total_demand": total_demand,
                "max_supplier_share": max_supplier_share,
                "min_avg_esg": min_avg_esg,
                "max_avg_risk": max_avg_risk,
                "min_suppliers": min_suppliers,
                "blocked_regions": blocked_regions,
                "weights": {
                    "cost": w_cost,
                    "risk": w_risk,
                    "esg": w_esg,
                },
                "supplier_selection_penalty": supplier_selection_penalty,
            },
            "metadata": {
                "input_supplier_count": input_supplier_count,
                "compliant_supplier_count": compliant_supplier_count,
                "eligible_supplier_count": eligible_supplier_count,
                "total_eligible_capacity": total_eligible_capacity,
            },
        }

    # Step 4: normalized columns for objective only
    working_df["norm_cost"] = min_max_normalize(working_df["unit_cost"])
    working_df["norm_risk"] = min_max_normalize(working_df["risk_score"])
    working_df["norm_esg"] = min_max_normalize(working_df["esg_score"])

    # Step 5: optimization model
    model = pulp.LpProblem("Supplier_Allocation", pulp.LpMinimize)

    x = {
        i: pulp.LpVariable(f"alloc_{i}", lowBound=0, cat="Continuous")
        for i in working_df.index
    }

    y = {
        i: pulp.LpVariable(f"select_{i}", cat="Binary")
        for i in working_df.index
    }

    # Objective:
    # cost is primary
    # risk is a light penalty
    # ESG is a light reward
    # selection penalty discourages unnecessary fragmentation
    model += (
        pulp.lpSum(
            (
                w_cost * working_df.loc[i, "norm_cost"]
                + w_risk * working_df.loc[i, "norm_risk"]
                - w_esg * working_df.loc[i, "norm_esg"]
            ) * x[i]
            for i in working_df.index
        )
        + pulp.lpSum(supplier_selection_penalty * y[i] for i in working_df.index)
    )

    # Step 6: constraints

    # Demand must be met exactly
    model += pulp.lpSum(x[i] for i in working_df.index) == total_demand, "Demand_Fulfillment"

    for i in working_df.index:
        capacity = working_df.loc[i, "max_capacity"]

        # Capacity limit
        model += x[i] <= capacity, f"Capacity_{i}"

        # Max portfolio share
        model += x[i] <= max_supplier_share * total_demand, f"MaxShare_{i}"

        # Link allocation to selection
        model += x[i] <= capacity * y[i], f"SelectionLink_{i}"

    # Minimum number of selected suppliers
    model += pulp.lpSum(y[i] for i in working_df.index) >= min_suppliers, "MinSuppliers"

    # Portfolio average ESG threshold
    model += (
        pulp.lpSum(working_df.loc[i, "esg_score"] * x[i] for i in working_df.index)
        >= min_avg_esg * total_demand
    ), "MinAvgESG"

    # Portfolio average risk cap
    model += (
        pulp.lpSum(working_df.loc[i, "risk_score"] * x[i] for i in working_df.index)
        <= max_avg_risk * total_demand
    ), "MaxAvgRisk"

    # Step 7: solve
    status = model.solve(pulp.PULP_CBC_CMD(msg=False))
    status_str = pulp.LpStatus[status]

    scenario = {
        "total_demand": total_demand,
        "max_supplier_share": max_supplier_share,
        "min_avg_esg": min_avg_esg,
        "max_avg_risk": max_avg_risk,
        "min_suppliers": min_suppliers,
        "blocked_regions": blocked_regions,
        "weights": {
            "cost": w_cost,
            "risk": w_risk,
            "esg": w_esg,
        },
        "supplier_selection_penalty": supplier_selection_penalty,
    }

    metadata = {
        "input_supplier_count": input_supplier_count,
        "compliant_supplier_count": compliant_supplier_count,
        "eligible_supplier_count": eligible_supplier_count,
        "total_eligible_capacity": total_eligible_capacity,
    }

    if status_str != "Optimal":
        return {
            "status": status_str,
            "message": (
                "No feasible optimal solution found. This may be caused by demand, "
                "capacity, max-share, min-suppliers, ESG, risk, or blocked-region restrictions."
            ),
            "scenario": scenario,
            "metadata": metadata,
        }

    # Step 8: build result table
    results = []
    for i in working_df.index:
        alloc = x[i].value()
        selected = y[i].value()

        if alloc is not None and alloc > 0:
            results.append(
                {
                    "supplier_id": working_df.loc[i, "supplier_id"],
                    "supplier_name": working_df.loc[i, "supplier_name"],
                    "region": working_df.loc[i, "region"],
                    "unit_cost": working_df.loc[i, "unit_cost"],
                    "risk_score": working_df.loc[i, "risk_score"],
                    "esg_score": working_df.loc[i, "esg_score"],
                    "max_capacity": working_df.loc[i, "max_capacity"],
                    "allocation_qty": round(alloc, 2),
                    "selected": int(selected) if selected is not None else 0,
                    "allocation_pct": round(alloc / total_demand, 4),
                }
            )

    result_df = pd.DataFrame(results)

    if result_df.empty:
        return {
            "status": "Optimal",
            "message": "Solver returned an optimal status but no positive allocations were found.",
            "allocations": result_df,
            "summary": {},
            "scenario": scenario,
            "metadata": metadata,
        }

    total_cost = float((result_df["allocation_qty"] * result_df["unit_cost"]).sum())
    weighted_avg_esg = float(
        (result_df["allocation_qty"] * result_df["esg_score"]).sum() / total_demand
    )
    weighted_avg_risk = float(
        (result_df["allocation_qty"] * result_df["risk_score"]).sum() / total_demand
    )

    summary = {
        "total_demand": total_demand,
        "selected_suppliers": int(result_df["selected"].sum()),
        "total_cost": round(total_cost, 2),
        "weighted_avg_esg": round(weighted_avg_esg, 2),
        "weighted_avg_risk": round(weighted_avg_risk, 2),
        "blocked_regions": blocked_regions,
        "min_suppliers_required": min_suppliers,
        "max_supplier_share": max_supplier_share,
    }

    return {
        "status": "Optimal",
        "allocations": result_df.sort_values(by="allocation_qty", ascending=False),
        "summary": summary,
        "scenario": scenario,
        "metadata": metadata,
        "eligible_suppliers": working_df[
            [
                "supplier_id",
                "supplier_name",
                "region",
                "unit_cost",
                "risk_score",
                "esg_score",
                "max_capacity",
                "compliance_flag",
            ]
        ].sort_values(by="unit_cost", ascending=True).reset_index(drop=True),
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

    print("\nOptimization Status:", result["status"])

    if result["status"] == "Optimal":
        print("\nSummary:")
        for key, value in result["summary"].items():
            print(f"{key}: {value}")

        print("\nMetadata:")
        for key, value in result["metadata"].items():
            print(f"{key}: {value}")

        print("\nTop Allocations:")
        print(result["allocations"].head(10))
    else:
        print("\nMessage:")
        print(result["message"])

        print("\nMetadata:")
        for key, value in result["metadata"].items():
            print(f"{key}: {value}")