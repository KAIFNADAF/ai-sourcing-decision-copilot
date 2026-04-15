import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Ensuring project root is importable when running: streamlit run app/main.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_loader import load_suppliers
from src.optimizer import optimize_supplier_allocation
from src.audit import generate_decision_audit
from src.sensitivity import run_sensitivity_analysis
from src.parser import parse_sourcing_request
from src.workflow_store import WorkflowStore
from src.workflow_orchestrator import WorkflowOrchestrator


st.set_page_config(
    page_title="AI-Assisted Sourcing Decision Copilot",
    page_icon="📦",
    layout="wide",
)


# ---------------------------------------------------
# Cached loaders
# ---------------------------------------------------
@st.cache_data
def get_supplier_data() -> pd.DataFrame:
    return load_suppliers("data/suppliers_dataset.csv")


@st.cache_data
def load_workflow_table(table_name: str) -> pd.DataFrame:
    store = WorkflowStore(data_dir="data")

    if table_name == "requests":
        return store.load_requests()
    if table_name == "runs":
        return store.load_runs()
    if table_name == "validation":
        return store.load_validation_results()
    if table_name == "approvals":
        return store.load_approvals()
    if table_name == "trace":
        return store.load_trace()
    if table_name == "artifacts":
        return store.load_artifacts()
    if table_name == "recommendations":
        return store.load_recommendations()

    raise ValueError(f"Unknown table name: {table_name}")


def clear_workflow_caches() -> None:
    load_workflow_table.clear()


# ---------------------------------------------------
# Formatting helpers
# ---------------------------------------------------
def _format_currency(value: float) -> str:
    return f"{value:,.2f}"


def _safe_pipe_list(value: str) -> list[str]:
    if pd.isna(value) or str(value).strip() == "":
        return []
    return [x.strip() for x in str(value).split("|") if x.strip()]


def _display_stage_name(stage: str) -> str:
    mapping = {
        "request_intake": "Request Intake",
        "parse": "Normalize Request",
        "validate": "Validate",
        "approve": "Approve",
        "execute": "Execute",
        "recommend": "Recommend",
    }
    return mapping.get(str(stage), str(stage).replace("_", " ").title())


def _display_validation_status(status: str) -> str:
    status = str(status).strip().lower()
    mapping = {
        "passed": "Passed",
        "warning": "Passed with Warnings",
        "failed": "Failed",
        "pending": "Pending",
    }
    return mapping.get(status, str(status).title())


def _display_approval_status(decision: str) -> str:
    decision = str(decision).strip().lower()
    mapping = {
        "approved": "Approved",
        "rejected": "Rejected",
        "pending": "Pending",
    }
    return mapping.get(decision, str(decision).title())


def _display_recommendation_status(status: str) -> str:
    status = str(status).strip().lower()
    mapping = {
        "generated": "Ready",
        "not_generated": "Not Ready",
    }
    return mapping.get(status, str(status).replace("_", " ").title())


def _derive_case_status(row: pd.Series) -> str:
    request_status = str(row.get("status", "")).lower()
    run_status = str(row.get("status_run", ""))
    validation_status = str(row.get("validation_status", "")).lower()
    approval_decision = str(row.get("decision", "")).lower()

    if validation_status == "failed":
        return "Validation Failed"
    if approval_decision == "rejected" or request_status == "rejected":
        return "Rejected"
    if approval_decision == "pending" or run_status == "Awaiting Approval":
        return "Awaiting Approval"
    if request_status == "completed" or run_status == "Completed":
        return "Completed"
    return "Needs Review"


def _build_request_browser_df(
    requests_df: pd.DataFrame,
    runs_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    approvals_df: pd.DataFrame,
) -> pd.DataFrame:
    latest_runs = (
        runs_df.sort_values(["request_id", "run_version"])
        .drop_duplicates(subset=["request_id"], keep="last")
        .copy()
    )

    merged = (
        requests_df.merge(
            latest_runs[
                ["request_id", "workflow_run_id", "run_version", "current_stage", "status"]
            ],
            on="request_id",
            how="left",
            suffixes=("", "_run"),
        )
        .merge(
            validation_df[["workflow_run_id", "validation_status"]],
            on="workflow_run_id",
            how="left",
        )
        .merge(
            approvals_df[["workflow_run_id", "decision"]],
            on="workflow_run_id",
            how="left",
        )
    )

    merged["blocked_regions_display"] = merged["blocked_regions_raw"].apply(
        lambda x: ", ".join(_safe_pipe_list(x)) if _safe_pipe_list(x) else "None"
    )
    merged["demand_display"] = merged["requested_demand_units"].apply(lambda x: f"{int(x):,}")
    merged["case_status"] = merged.apply(_derive_case_status, axis=1)
    merged["stage_display"] = merged["current_stage"].apply(_display_stage_name)

    merged["request_label"] = merged.apply(
        lambda row: f"{row['request_id']} • {row['business_unit']} • {row['case_status']}",
        axis=1,
    )

    return merged.sort_values(
        ["priority", "business_unit", "requested_demand_units"],
        ascending=[False, True, False],
    )


def _build_run_label(
    run_row: pd.Series,
    validation_row: pd.Series | None,
    approval_row: pd.Series | None,
) -> str:
    validation_status = (
        _display_validation_status(validation_row["validation_status"])
        if validation_row is not None
        else "n/a"
    )
    approval_decision = (
        _display_approval_status(approval_row["decision"])
        if approval_row is not None
        else "n/a"
    )

    return (
        f"v{int(run_row['run_version'])} • {_display_stage_name(run_row['current_stage'])} • "
        f"{run_row['status']} • Validation: {validation_status} • Approval: {approval_decision}"
    )


def _safe_list_from_pipe(value: Any) -> List[str]:
    if pd.isna(value) or str(value).strip() == "":
        return []
    return [x.strip() for x in str(value).split("|") if x.strip()]


def _build_base_params(
    total_demand: int,
    max_supplier_share: float,
    min_avg_esg: int,
    max_avg_risk: int,
    min_suppliers: int,
    blocked_regions: List[str],
    w_cost: float,
    w_risk: float,
    w_esg: float,
    supplier_selection_penalty: float,
) -> dict:
    return {
        "total_demand": total_demand,
        "max_supplier_share": max_supplier_share,
        "min_avg_esg": min_avg_esg,
        "max_avg_risk": max_avg_risk,
        "min_suppliers": min_suppliers,
        "blocked_regions": blocked_regions,
        "w_cost": w_cost,
        "w_risk": w_risk,
        "w_esg": w_esg,
        "supplier_selection_penalty": supplier_selection_penalty,
    }


def _get_parser_context() -> dict:
    parsed_output = st.session_state.get("parsed_output", {})
    if not parsed_output:
        return {
            "used": False,
            "has_defaults": False,
            "has_heuristics": False,
            "missing_fields": [],
            "heuristic_fields": [],
            "explicit_fields": [],
        }

    missing_fields = parsed_output.get("missing_fields", [])
    heuristic_fields = parsed_output.get("heuristic_fields", [])
    explicit_fields = parsed_output.get("explicit_fields", [])

    return {
        "used": st.session_state.get("use_parsed_scenario", False),
        "has_defaults": len(missing_fields) > 0,
        "has_heuristics": len(heuristic_fields) > 0,
        "missing_fields": missing_fields,
        "heuristic_fields": heuristic_fields,
        "explicit_fields": explicit_fields,
    }


def _needs_soft_language(parser_context: dict) -> bool:
    return parser_context["used"] and (
        parser_context["has_defaults"] or parser_context["has_heuristics"]
    )


def _build_short_parser_notes(parsed_output: dict) -> list[str]:
    notes = []

    heuristic_fields = parsed_output.get("heuristic_fields", [])

    if "min_avg_esg" in heuristic_fields:
        value = parsed_output["parsed_scenario"].get("min_avg_esg")
        notes.append(f"ESG was interpreted as {value}.")

    if "max_avg_risk" in heuristic_fields:
        value = parsed_output["parsed_scenario"].get("max_avg_risk")
        notes.append(f"Risk was interpreted as {value}.")

    if "max_supplier_share" in heuristic_fields:
        value = parsed_output["parsed_scenario"].get("max_supplier_share")
        notes.append(f"Max supplier share was inferred as {value:.2f}.")

    if any(field in heuristic_fields for field in ["w_cost", "w_risk", "w_esg"]):
        notes.append("Some trade-off weights were inferred from the wording.")

    if parsed_output.get("missing_fields"):
        notes.append("Some fields still used defaults. Review them before treating this as final.")

    deduped = []
    seen = set()
    for note in notes:
        if note not in seen:
            deduped.append(note)
            seen.add(note)

    return deduped[:3]


# ---------------------------------------------------
# Core sourcing analysis UI
# ---------------------------------------------------
def render_header(df: pd.DataFrame) -> None:
    st.title("AI-Assisted Sourcing Decision Copilot")
    st.caption(
        "A sourcing decision workspace for scenario analysis and governed workflow execution."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Suppliers", len(df))
    with col2:
        compliant_count = int((df["compliance_flag"] == 1).sum())
        st.metric("Compliant Suppliers", compliant_count)
    with col3:
        st.metric("Regions", df["region"].nunique())


def render_nl_parser_section() -> None:
    st.subheader("Scenario Input")
    st.write(
        "Describe a sourcing case in plain English. The system converts it into structured inputs, "
        "while the optimizer still makes the final allocation decision."
    )

    default_prompt = (
        "Allocate 120000 units, avoid Eastern Europe, keep ESG at least 75, "
        "risk under 40, and use at least 4 suppliers."
    )

    user_request = st.text_area(
        "Describe your sourcing scenario",
        value=st.session_state.get("nl_request", default_prompt),
        height=120,
        key="nl_request",
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        parse_clicked = st.button("Interpret Scenario", use_container_width=True)
    with col2:
        clear_clicked = st.button("Clear Scenario", use_container_width=True)

    if clear_clicked:
        st.session_state.pop("parsed_output", None)
        st.session_state.pop("parsed_scenario", None)
        st.session_state.pop("use_parsed_scenario", None)
        st.rerun()

    if parse_clicked:
        if not user_request.strip():
            st.warning("Enter a sourcing request before interpreting it.")
        else:
            with st.spinner("Interpreting scenario..."):
                try:
                    parsed_output = parse_sourcing_request(user_request.strip())
                    st.session_state["parsed_output"] = parsed_output
                    st.session_state["parsed_scenario"] = parsed_output["parsed_scenario"]
                    st.session_state["use_parsed_scenario"] = True
                except Exception as exc:
                    st.error(f"Scenario interpretation failed: {exc}")

    parsed_output = st.session_state.get("parsed_output")

    if parsed_output:
        st.info("Review the interpreted scenario before running the analysis.")

        with st.expander("Scenario Review", expanded=True):
            left, right = st.columns(2)

            with left:
                st.dataframe(
                    pd.DataFrame([parsed_output["parsed_scenario"]]),
                    use_container_width=True,
                    hide_index=True,
                )

            with right:
                st.write(f"**Summary:** {parsed_output['interpretation']}")

                notes = _build_short_parser_notes(parsed_output)
                if notes:
                    st.write("**Notes**")
                    for item in notes:
                        st.markdown(f"- {item}")

                explicit_fields = parsed_output.get("explicit_fields", [])
                heuristic_fields = parsed_output.get("heuristic_fields", [])
                missing_fields = parsed_output.get("missing_fields", [])

                if explicit_fields:
                    st.caption("Explicit fields")
                    st.markdown(", ".join([f"`{x}`" for x in explicit_fields]))

                if heuristic_fields:
                    st.caption("Interpreted fields")
                    st.markdown(", ".join([f"`{x}`" for x in heuristic_fields]))

                if missing_fields:
                    st.caption("Defaulted fields")
                    st.markdown(", ".join([f"`{x}`" for x in missing_fields]))

            use_parsed = st.checkbox(
                "Use this interpreted scenario",
                value=st.session_state.get("use_parsed_scenario", True),
                key="use_parsed_checkbox",
            )
            st.session_state["use_parsed_scenario"] = use_parsed

            with st.expander("Raw Parser Output", expanded=False):
                st.code(parsed_output.get("raw_model_output", ""), language="json")


def render_sidebar(df: pd.DataFrame) -> dict:
    st.sidebar.header("Manual Inputs")

    parsed_scenario = st.session_state.get("parsed_scenario")
    use_parsed_scenario = st.session_state.get("use_parsed_scenario", False)

    manual_defaults = {
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

    active_defaults = parsed_scenario if (parsed_scenario and use_parsed_scenario) else manual_defaults

    total_demand = st.sidebar.number_input(
        "Total Demand",
        min_value=1000,
        max_value=1_000_000,
        value=int(active_defaults["total_demand"]),
        step=5000,
    )

    max_supplier_share = st.sidebar.slider(
        "Max Supplier Share",
        min_value=0.10,
        max_value=0.80,
        value=float(active_defaults["max_supplier_share"]),
        step=0.05,
    )

    min_avg_esg = st.sidebar.slider(
        "Minimum Average ESG",
        min_value=0,
        max_value=100,
        value=int(active_defaults["min_avg_esg"]),
        step=1,
    )

    max_avg_risk = st.sidebar.slider(
        "Maximum Average Risk",
        min_value=0,
        max_value=100,
        value=int(active_defaults["max_avg_risk"]),
        step=1,
    )

    min_suppliers = st.sidebar.number_input(
        "Minimum Suppliers",
        min_value=1,
        max_value=20,
        value=int(active_defaults["min_suppliers"]),
        step=1,
    )

    st.sidebar.markdown("### Region Rules")
    available_regions = sorted(df["region"].dropna().unique().tolist())
    blocked_regions = st.sidebar.multiselect(
        "Blocked Regions",
        options=available_regions,
        default=[
            region for region in active_defaults["blocked_regions"]
            if region in available_regions
        ],
    )

    st.sidebar.markdown("### Objective Weights")
    w_cost = st.sidebar.slider(
        "Cost Weight",
        min_value=0.0,
        max_value=1.0,
        value=float(active_defaults["w_cost"]),
        step=0.05,
    )
    w_risk = st.sidebar.slider(
        "Risk Weight",
        min_value=0.0,
        max_value=1.0,
        value=float(active_defaults["w_risk"]),
        step=0.05,
    )
    w_esg = st.sidebar.slider(
        "ESG Weight",
        min_value=0.0,
        max_value=1.0,
        value=float(active_defaults["w_esg"]),
        step=0.05,
    )

    supplier_selection_penalty = st.sidebar.slider(
        "Selection Penalty",
        min_value=0.0,
        max_value=0.20,
        value=float(active_defaults["supplier_selection_penalty"]),
        step=0.01,
    )

    if parsed_scenario and use_parsed_scenario:
        st.sidebar.success("Inputs are prefilled from the interpreted scenario.")

    return _build_base_params(
        total_demand=total_demand,
        max_supplier_share=max_supplier_share,
        min_avg_esg=min_avg_esg,
        max_avg_risk=max_avg_risk,
        min_suppliers=min_suppliers,
        blocked_regions=blocked_regions,
        w_cost=w_cost,
        w_risk=w_risk,
        w_esg=w_esg,
        supplier_selection_penalty=supplier_selection_penalty,
    )


def render_scenario_snapshot(base_params: dict) -> None:
    with st.expander("Scenario Snapshot", expanded=False):
        snapshot_df = pd.DataFrame(
            [
                {
                    "Total Demand": base_params["total_demand"],
                    "Max Supplier Share": base_params["max_supplier_share"],
                    "Min Avg ESG": base_params["min_avg_esg"],
                    "Max Avg Risk": base_params["max_avg_risk"],
                    "Min Suppliers": base_params["min_suppliers"],
                    "Blocked Regions": ", ".join(base_params["blocked_regions"]) or "None",
                    "Cost Weight": base_params["w_cost"],
                    "Risk Weight": base_params["w_risk"],
                    "ESG Weight": base_params["w_esg"],
                    "Selection Penalty": base_params["supplier_selection_penalty"],
                }
            ]
        )
        st.dataframe(snapshot_df, use_container_width=True, hide_index=True)


def _get_main_driver(audit: dict, result: dict, parser_context: dict) -> str:
    if result["status"] != "Optimal":
        return "No driver is available because the scenario is not feasible."

    portfolio_checks = audit.get("portfolio_checks", {})
    esg_tightness = portfolio_checks.get("esg_tightness")
    risk_tightness = portfolio_checks.get("risk_tightness")
    concentration = portfolio_checks.get("top_2_supplier_concentration_pct", 0)

    prefix = "Under the interpreted scenario, " if _needs_soft_language(parser_context) else ""

    if esg_tightness in {"binding_or_nearly_binding", "tight"} and risk_tightness in {
        "binding_or_nearly_binding",
        "tight",
    }:
        return f"{prefix}Both ESG and risk are shaping the portfolio."
    if esg_tightness in {"binding_or_nearly_binding", "tight"}:
        return f"{prefix}The ESG floor is shaping the portfolio."
    if risk_tightness in {"binding_or_nearly_binding", "tight"}:
        return f"{prefix}The risk cap is shaping the portfolio."
    if concentration >= 0.70:
        return f"{prefix}Share limits are shaping the allocation because volume is still concentrated."

    return f"{prefix}Cost appears to be the main driver."


def _get_current_weakness(audit: dict, sensitivity_output: dict, result: dict, parser_context: dict) -> str:
    if result["status"] != "Optimal":
        return "The main issue is feasibility under the current rules."

    portfolio_checks = audit.get("portfolio_checks", {})
    comparison_table = sensitivity_output.get("comparison_table", pd.DataFrame())

    concentration = portfolio_checks.get("top_2_supplier_concentration_pct")
    esg_tightness = portfolio_checks.get("esg_tightness")

    prefix = "Under the interpreted scenario, " if _needs_soft_language(parser_context) else ""

    if concentration is not None and concentration >= 0.70:
        return f"{prefix}The portfolio is still too concentrated in the top suppliers."

    if esg_tightness in {"binding_or_nearly_binding", "tight"}:
        return f"{prefix}The portfolio is close to the ESG boundary."

    blocked_row = comparison_table[comparison_table["scenario_name"] == "block_eastern_europe"]
    if not blocked_row.empty and blocked_row.iloc[0]["status"] != "Optimal":
        return f"{prefix}The portfolio depends too much on Eastern Europe."

    higher_demand_row = comparison_table[comparison_table["scenario_name"] == "higher_demand"]
    if not higher_demand_row.empty and higher_demand_row.iloc[0]["status"] != "Optimal":
        return f"{prefix}The portfolio may struggle under higher demand."

    return f"{prefix}No single weakness stands out strongly."


def _get_recommended_next_step(audit: dict, sensitivity_output: dict, result: dict, parser_context: dict) -> str:
    if result["status"] != "Optimal":
        return "Relax one or more constraints or widen the eligible supplier pool."

    portfolio_checks = audit.get("portfolio_checks", {})
    comparison_table = sensitivity_output.get("comparison_table", pd.DataFrame())

    concentration = portfolio_checks.get("top_2_supplier_concentration_pct")
    esg_tightness = portfolio_checks.get("esg_tightness")

    if concentration is not None and concentration >= 0.70:
        return "Lower the max supplier share and rerun."

    if esg_tightness in {"binding_or_nearly_binding", "tight"}:
        stricter_row = comparison_table[comparison_table["scenario_name"] == "stricter_esg"]
        if not stricter_row.empty and stricter_row.iloc[0]["status"] == "Optimal":
            return "Test a higher ESG floor before changing the target."

    blocked_row = comparison_table[comparison_table["scenario_name"] == "block_eastern_europe"]
    if not blocked_row.empty and blocked_row.iloc[0]["status"] != "Optimal":
        return "Add more non-Eastern-Europe supply before using that region block."

    if _needs_soft_language(parser_context):
        return "Replace vague business preferences with numeric targets and rerun."

    return "Use this result as the baseline and tighten one rule at a time."


def _get_main_tradeoff(audit: dict, sensitivity_output: dict, result: dict, parser_context: dict) -> str:
    if result["status"] != "Optimal":
        return "No trade-off is available because the current scenario is not feasible."

    comparison_table = sensitivity_output.get("comparison_table", pd.DataFrame())
    if comparison_table.empty:
        return "No trade-off pattern is available from scenario testing."

    base_row_df = comparison_table[comparison_table["scenario_name"] == "base_case"]
    if base_row_df.empty or base_row_df.iloc[0]["status"] != "Optimal":
        return "No reliable base case was available for trade-off analysis."

    base_row = base_row_df.iloc[0]

    stricter_esg_row = comparison_table[comparison_table["scenario_name"] == "stricter_esg"]
    tighter_share_row = comparison_table[comparison_table["scenario_name"] == "tighter_supplier_share"]

    prefix = "Under the interpreted scenario, " if _needs_soft_language(parser_context) else ""

    if not stricter_esg_row.empty and stricter_esg_row.iloc[0]["status"] == "Optimal":
        cost_diff = None
        if pd.notna(base_row["total_cost"]) and pd.notna(stricter_esg_row.iloc[0]["total_cost"]):
            cost_diff = float(stricter_esg_row.iloc[0]["total_cost"]) - float(base_row["total_cost"])

        if cost_diff is not None and cost_diff > 1:
            return f"{prefix}A higher ESG target is possible, but it comes with a higher cost."

    if not tighter_share_row.empty and tighter_share_row.iloc[0]["status"] == "Optimal":
        base_conc = base_row.get("top_2_concentration_pct")
        new_conc = tighter_share_row.iloc[0].get("top_2_concentration_pct")
        if pd.notna(base_conc) and pd.notna(new_conc) and float(new_conc) < float(base_conc):
            return (
                f"{prefix}Better diversification is possible, but it may require tighter share limits "
                f"and a less cost-efficient plan."
            )

    return f"{prefix}No major trade-off stands out across the tested scenarios."


def _get_risk_if_priorities_shift(audit: dict, sensitivity_output: dict, result: dict, parser_context: dict) -> str:
    if result["status"] != "Optimal":
        return "The plan is fragile because it is not feasible under the current rules."

    comparison_table = sensitivity_output.get("comparison_table", pd.DataFrame())
    if comparison_table.empty:
        return "No risk signal is available from scenario testing."

    prefix = "Under the interpreted scenario, " if _needs_soft_language(parser_context) else ""

    higher_demand_row = comparison_table[comparison_table["scenario_name"] == "higher_demand"]
    if not higher_demand_row.empty and higher_demand_row.iloc[0]["status"] != "Optimal":
        return f"{prefix}The main risk is demand scaling. The plan does not hold when demand increases by 25%."

    blocked_row = comparison_table[comparison_table["scenario_name"] == "block_eastern_europe"]
    if not blocked_row.empty and blocked_row.iloc[0]["status"] != "Optimal":
        return (
            f"{prefix}The plan appears dependent on Eastern Europe. "
            f"Blocking that region makes it infeasible."
        )

    portfolio_checks = audit.get("portfolio_checks", {})
    concentration = portfolio_checks.get("top_2_supplier_concentration_pct")
    if concentration is not None and concentration >= 0.70:
        return f"{prefix}The main risk is concentration. Too much volume still sits with the top suppliers."

    return f"{prefix}The plan looks reasonably stable across the tested scenario changes."


def _get_what_to_test_next(audit: dict, sensitivity_output: dict, result: dict, parser_context: dict) -> str:
    if result["status"] != "Optimal":
        return "Relax one or more constraints or widen the eligible supplier pool."

    comparison_table = sensitivity_output.get("comparison_table", pd.DataFrame())
    portfolio_checks = audit.get("portfolio_checks", {})
    concentration = portfolio_checks.get("top_2_supplier_concentration_pct")
    esg_tightness = portfolio_checks.get("esg_tightness")

    blocked_row = comparison_table[comparison_table["scenario_name"] == "block_eastern_europe"]
    higher_demand_row = comparison_table[comparison_table["scenario_name"] == "higher_demand"]
    stricter_esg_row = comparison_table[comparison_table["scenario_name"] == "stricter_esg"]

    if not blocked_row.empty and blocked_row.iloc[0]["status"] != "Optimal":
        return "Expand supplier capacity outside Eastern Europe before applying a regional block."

    if not higher_demand_row.empty and higher_demand_row.iloc[0]["status"] != "Optimal":
        return "Add more capacity or broaden the supplier pool before planning for higher demand."

    if concentration is not None and concentration >= 0.70:
        return "Lower the supplier share cap if resilience and diversification matter most."

    if esg_tightness in {"binding_or_nearly_binding", "tight"}:
        if not stricter_esg_row.empty and stricter_esg_row.iloc[0]["status"] == "Optimal":
            return "Test a higher ESG target if sustainability is becoming a stronger priority."

    if _needs_soft_language(parser_context):
        return "Replace vague business preferences with numeric targets and rerun."

    return "Use this result as the baseline and tighten one business priority at a time."


def render_decision_summary_panel(result: dict, audit: dict, sensitivity_output: dict, parser_context: dict) -> None:
    st.subheader("Decision Summary")

    if result["status"] != "Optimal":
        st.error(result.get("message", "No feasible recommendation available."))
        st.info("Outcome: The current scenario is not feasible.")
        st.warning("Next step: Relax one or more constraints and rerun.")
        return

    summary = result["summary"]

    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.metric("Selected Suppliers", summary["selected_suppliers"])
    with metric_cols[1]:
        st.metric("Total Cost", _format_currency(summary["total_cost"]))
    with metric_cols[2]:
        st.metric("Weighted Avg ESG", summary["weighted_avg_esg"])
    with metric_cols[3]:
        st.metric("Weighted Avg Risk", summary["weighted_avg_risk"])

    if _needs_soft_language(parser_context):
        st.caption(
            "This result uses a mix of parsed, inferred, and defaulted inputs. Review the interpreted scenario before treating it as final."
        )

    insight_cols = st.columns(3)

    with insight_cols[0]:
        st.info(f"**Main Driver**\n\n{_get_main_driver(audit, result, parser_context)}")

    with insight_cols[1]:
        st.warning(
            f"**Current Weakness**\n\n{_get_current_weakness(audit, sensitivity_output, result, parser_context)}"
        )

    with insight_cols[2]:
        st.success(
            f"**Recommended Next Step**\n\n{_get_recommended_next_step(audit, sensitivity_output, result, parser_context)}"
        )


def render_tradeoff_guidance(audit: dict, sensitivity_output: dict, result: dict, parser_context: dict) -> None:
    st.subheader("What Changes If Priorities Shift")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(
            f"**Main Trade-off**\n\n"
            f"{_get_main_tradeoff(audit, sensitivity_output, result, parser_context)}"
        )

    with col2:
        st.warning(
            f"**Main Risk If You Push Further**\n\n"
            f"{_get_risk_if_priorities_shift(audit, sensitivity_output, result, parser_context)}"
        )

    with col3:
        st.success(
            f"**What To Test Next**\n\n"
            f"{_get_what_to_test_next(audit, sensitivity_output, result, parser_context)}"
        )


def _build_portfolio_flags(
    audit: dict,
    sensitivity_output: dict,
    result: dict,
    parser_context: dict,
) -> list[dict]:
    flags = []

    if result["status"] != "Optimal":
        return [
            {
                "flag": "Scenario infeasible",
                "severity": "High",
                "why_it_matters": "The current combination of rules does not allow a valid allocation.",
                "recommended_action": "Relax one or more constraints or widen the supplier pool.",
                "priority": 100,
            }
        ]

    portfolio_checks = audit.get("portfolio_checks", {})
    comparison_table = sensitivity_output.get("comparison_table", pd.DataFrame())

    concentration = portfolio_checks.get("top_2_supplier_concentration_pct")
    esg_tightness = portfolio_checks.get("esg_tightness")
    risk_tightness = portfolio_checks.get("risk_tightness")

    current_weakness_text = _get_current_weakness(
        audit, sensitivity_output, result, parser_context
    ).lower()
    shift_risk_text = _get_risk_if_priorities_shift(
        audit, sensitivity_output, result, parser_context
    ).lower()

    concentration_already_covered = "concentrat" in current_weakness_text and "concentrat" in shift_risk_text
    esg_already_covered = "esg" in current_weakness_text or "esg" in shift_risk_text
    regional_already_covered = "eastern europe" in current_weakness_text or "eastern europe" in shift_risk_text
    demand_already_covered = "demand" in current_weakness_text or "demand" in shift_risk_text

    if concentration is not None and concentration >= 0.70 and not concentration_already_covered:
        flags.append(
            {
                "flag": "High concentration",
                "severity": "High",
                "why_it_matters": "Too much volume sits with the top two suppliers.",
                "recommended_action": "Lower max supplier share and test the case again.",
                "priority": 95,
            }
        )

    if esg_tightness in {"binding_or_nearly_binding", "tight"} and not esg_already_covered:
        flags.append(
            {
                "flag": "ESG is tight",
                "severity": "Medium",
                "why_it_matters": "The portfolio is close to the ESG floor.",
                "recommended_action": "Test a higher ESG floor before changing the target.",
                "priority": 80,
            }
        )

    if risk_tightness in {"binding_or_nearly_binding", "tight"}:
        flags.append(
            {
                "flag": "Risk is tight",
                "severity": "Medium",
                "why_it_matters": "The portfolio is close to the maximum allowed risk.",
                "recommended_action": "Look for lower-risk supply if you need more flexibility.",
                "priority": 75,
            }
        )

    blocked_row = comparison_table[comparison_table["scenario_name"] == "block_eastern_europe"]
    if not blocked_row.empty and blocked_row.iloc[0]["status"] != "Optimal" and not regional_already_covered:
        flags.append(
            {
                "flag": "Regional dependence",
                "severity": "High",
                "why_it_matters": "Blocking Eastern Europe makes this case infeasible.",
                "recommended_action": "Add more supply outside Eastern Europe before using that rule.",
                "priority": 90,
            }
        )

    higher_demand_row = comparison_table[comparison_table["scenario_name"] == "higher_demand"]
    if not higher_demand_row.empty and higher_demand_row.iloc[0]["status"] != "Optimal" and not demand_already_covered:
        flags.append(
            {
                "flag": "Demand scaling risk",
                "severity": "High",
                "why_it_matters": "The case fails when demand increases.",
                "recommended_action": "Check spare capacity before committing to higher demand.",
                "priority": 85,
            }
        )

    flags = sorted(flags, key=lambda x: x["priority"], reverse=True)
    return flags[:2]


def render_portfolio_risk_flags(audit: dict, sensitivity_output: dict, result: dict, parser_context: dict) -> None:
    flags = _build_portfolio_flags(audit, sensitivity_output, result, parser_context)

    if not flags:
        return

    st.subheader("Additional Risk Flags")

    for flag in flags:
        severity = flag["severity"]
        text = (
            f"**{flag['flag']}**  \n"
            f"Why it matters: {flag['why_it_matters']}  \n"
            f"Next step: {flag['recommended_action']}"
        )

        if severity == "High":
            st.error(text)
        elif severity == "Medium":
            st.warning(text)
        else:
            st.success(text)


def render_concentration_chart(result: dict) -> None:
    st.subheader("Supplier Concentration")

    if result["status"] != "Optimal":
        st.info("Concentration view is available only for feasible solutions.")
        return

    allocations = result.get("allocations", pd.DataFrame())
    if allocations.empty:
        st.info("No allocations available for concentration analysis.")
        return

    ranked = allocations.sort_values(by="allocation_qty", ascending=False).copy()
    ranked["cumulative_pct"] = ranked["allocation_pct"].cumsum()

    chart_df = ranked.head(5)[["supplier_name", "allocation_pct", "cumulative_pct"]].copy()
    chart_df["allocation_pct"] = chart_df["allocation_pct"] * 100
    chart_df["cumulative_pct"] = chart_df["cumulative_pct"] * 100

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(chart_df["supplier_name"], chart_df["allocation_pct"], label="Individual Share (%)")
    ax.plot(chart_df["supplier_name"], chart_df["cumulative_pct"], marker="o", label="Cumulative Share (%)")
    ax.set_ylabel("Awarded Volume (%)")
    ax.set_title("Concentration Across Top Awarded Suppliers")
    ax.legend()
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    st.pyplot(fig)

    st.caption("This view shows whether awarded volume is spread well or concentrated in the top suppliers.")


def render_optimizer_output(result: dict) -> None:
    st.subheader("Optimization Results")

    if result["status"] != "Optimal":
        st.error(result.get("message", "No feasible optimal solution found."))
        metadata = result.get("metadata", {})
        if metadata:
            with st.expander("Feasibility Context", expanded=True):
                st.dataframe(pd.DataFrame([metadata]), use_container_width=True, hide_index=True)
        return

    summary = result["summary"]
    allocations = result["allocations"]
    metadata = result.get("metadata", {})

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Selected Suppliers", summary["selected_suppliers"])
    with col2:
        st.metric("Total Cost", _format_currency(summary["total_cost"]))
    with col3:
        st.metric("Weighted Avg ESG", summary["weighted_avg_esg"])
    with col4:
        st.metric("Weighted Avg Risk", summary["weighted_avg_risk"])

    st.markdown("### Allocation Distribution")
    chart_df = allocations[["supplier_name", "allocation_qty"]].sort_values(
        by="allocation_qty",
        ascending=False,
    )
    st.bar_chart(chart_df.set_index("supplier_name"))

    with st.expander("Allocation Details", expanded=True):
        st.dataframe(allocations, use_container_width=True, hide_index=True)

    with st.expander("Optimization Metadata", expanded=False):
        st.dataframe(pd.DataFrame([metadata]), use_container_width=True, hide_index=True)

    with st.expander("Eligible Supplier Pool", expanded=False):
        eligible_df = result.get("eligible_suppliers", pd.DataFrame())
        if not eligible_df.empty:
            st.dataframe(eligible_df, use_container_width=True, hide_index=True)
        else:
            st.info("No eligible supplier table available.")


def render_audit_output(audit: dict) -> None:
    st.subheader("Decision Audit")

    if audit["status"] != "Optimal":
        st.warning(audit.get("plain_english_explanation", "Audit unavailable."))
        return

    decision_summary = audit["decision_summary"]
    portfolio_checks = audit["portfolio_checks"]
    scenario_overview = audit["scenario_overview"]
    supplier_flags = audit["supplier_flags"]

    st.markdown("### Summary")
    st.write(f"**Headline:** {decision_summary['headline']}")
    st.write(f"**Detail:** {decision_summary['detail']}")

    with st.expander("Plain-English Explanation", expanded=True):
        st.info(audit["plain_english_explanation"])

    with st.expander("Portfolio Checks", expanded=False):
        st.dataframe(
            pd.DataFrame([portfolio_checks]),
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Scenario Overview", expanded=False):
        st.dataframe(
            pd.DataFrame([scenario_overview]),
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Supplier Flags", expanded=False):
        st.dataframe(pd.DataFrame(supplier_flags), use_container_width=True, hide_index=True)


def render_scenario_delta_chart(sensitivity_output: dict) -> None:
    comparison_table = sensitivity_output["comparison_table"].copy()
    if comparison_table.empty:
        st.info("No sensitivity scenarios available.")
        return

    base_row_df = comparison_table[comparison_table["scenario_name"] == "base_case"]
    if base_row_df.empty:
        st.info("Base case not found in sensitivity output.")
        return

    base_row = base_row_df.iloc[0]
    delta_rows = []

    for _, row in comparison_table.iterrows():
        if row["scenario_name"] == "base_case":
            continue
        if row["status"] != "Optimal" or base_row["status"] != "Optimal":
            continue
        if pd.isna(row["total_cost"]) or pd.isna(base_row["total_cost"]):
            continue

        cost_delta = round(float(row["total_cost"]) - float(base_row["total_cost"]), 2)
        if abs(cost_delta) < 1.0:
            continue

        delta_rows.append(
            {
                "scenario_name": row["scenario_name"],
                "cost_delta": cost_delta,
            }
        )

    delta_df = pd.DataFrame(delta_rows)
    if delta_df.empty:
        st.info("No meaningful cost changes across the tested scenarios.")
        return

    st.markdown("### Cost Impact vs Base Case")
    st.bar_chart(delta_df.set_index("scenario_name"))

    with st.expander("Scenario Delta Table", expanded=False):
        st.dataframe(delta_df, use_container_width=True, hide_index=True)


def _build_scenario_implication(row: pd.Series, base_row: pd.Series) -> str:
    if row["status"] != "Optimal":
        return "This scenario is not feasible."

    implications = []

    if pd.notna(row["total_cost"]) and pd.notna(base_row["total_cost"]):
        cost_diff = round(float(row["total_cost"]) - float(base_row["total_cost"]), 2)
        if abs(cost_diff) >= 1.0:
            if cost_diff > 0:
                implications.append(f"Cost increased by {_format_currency(cost_diff)}.")
            else:
                implications.append(f"Cost decreased by {_format_currency(abs(cost_diff))}.")

    if pd.notna(row["selected_suppliers"]) and pd.notna(base_row["selected_suppliers"]):
        supplier_diff = int(row["selected_suppliers"] - base_row["selected_suppliers"])
        if supplier_diff > 0:
            implications.append(f"It used {supplier_diff} more supplier(s).")
        elif supplier_diff < 0:
            implications.append(f"It used {abs(supplier_diff)} fewer supplier(s).")

    if not implications:
        return "This scenario did not materially change the result."

    return " ".join(implications)


def render_scenario_implication_cards(sensitivity_output: dict) -> None:
    st.markdown("### Scenario Notes")

    comparison_table = sensitivity_output["comparison_table"].copy()
    if comparison_table.empty:
        st.info("No scenario notes available.")
        return

    base_row_df = comparison_table[comparison_table["scenario_name"] == "base_case"]
    if base_row_df.empty:
        st.info("Base case not found in sensitivity output.")
        return

    base_row = base_row_df.iloc[0]

    for _, row in comparison_table.iterrows():
        if row["scenario_name"] == "base_case":
            continue

        scenario_name = str(row["scenario_name"]).replace("_", " ").title()

        change_text = ""
        if row["scenario_name"] == "stricter_esg":
            change_text = "Raised the minimum ESG requirement."
        elif row["scenario_name"] == "tighter_supplier_share":
            change_text = "Tightened the max supplier share cap."
        elif row["scenario_name"] == "higher_min_suppliers":
            change_text = "Raised the minimum supplier count."
        elif row["scenario_name"] == "block_eastern_europe":
            change_text = "Blocked Eastern Europe."
        elif row["scenario_name"] == "higher_demand":
            change_text = "Raised demand by 25%."

        implication_text = _build_scenario_implication(row, base_row)

        with st.expander(scenario_name, expanded=False):
            st.write(f"**Change:** {change_text}")
            st.write(f"**Result:** {implication_text}")


def render_sensitivity_output(sensitivity_output: dict) -> None:
    st.subheader("Sensitivity Analysis")

    insights = sensitivity_output["insights"]

    st.markdown("### Top Insights")
    if insights:
        for insight in insights[:3]:
            st.success(insight)
    else:
        st.info("No standout insight from the tested scenarios.")

    render_scenario_delta_chart(sensitivity_output)
    render_scenario_implication_cards(sensitivity_output)

    with st.expander("Scenario Comparison Table", expanded=False):
        st.dataframe(
            sensitivity_output["comparison_table"],
            use_container_width=True,
            hide_index=True,
        )


def run_existing_analysis_flow(df: pd.DataFrame) -> None:
    render_nl_parser_section()
    base_params = render_sidebar(df)
    render_scenario_snapshot(base_params)

    run_button = st.button("Run Sourcing Analysis", type="primary", use_container_width=True)

    if not run_button:
        st.markdown("### How to Use")
        st.write(
            "Adjust the scenario manually in the sidebar or describe it in plain English above. "
            "If you use the parser, review the interpreted fields before running the analysis."
        )
        return

    with st.spinner("Running optimization, audit, and scenario analysis..."):
        result = optimize_supplier_allocation(
            df=df,
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
        sensitivity_output = run_sensitivity_analysis(df=df, base_params=base_params)

    parser_context = _get_parser_context()

    render_decision_summary_panel(result, audit, sensitivity_output, parser_context)
    render_tradeoff_guidance(audit, sensitivity_output, result, parser_context)
    render_portfolio_risk_flags(audit, sensitivity_output, result, parser_context)

    st.markdown("---")
    render_concentration_chart(result)

    tab1, tab2, tab3 = st.tabs(
        ["Optimization", "Audit", "Sensitivity"]
    )

    with tab1:
        render_optimizer_output(result)

    with tab2:
        render_audit_output(audit)

    with tab3:
        render_sensitivity_output(sensitivity_output)


# ---------------------------------------------------
# Workflow orchestration UI
# ---------------------------------------------------
def render_workflow_overview() -> None:
    requests_df = load_workflow_table("requests")
    runs_df = load_workflow_table("runs")
    approvals_df = load_workflow_table("approvals")
    trace_df = load_workflow_table("trace")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Requests", len(requests_df))
    with col2:
        st.metric("Workflow Runs", len(runs_df))
    with col3:
        pending_approvals = int((approvals_df["decision"] == "Pending").sum())
        st.metric("Pending Approvals", pending_approvals)
    with col4:
        st.metric("Trace Events", len(trace_df))


def render_case_summary_cards(request_row: Dict[str, Any]) -> None:
    blocked_regions = _safe_list_from_pipe(request_row.get("blocked_regions_raw", ""))

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Demand", f"{int(request_row['requested_demand_units']):,}")
    with col2:
        st.metric("Minimum ESG", int(request_row["requested_min_avg_esg"]))
    with col3:
        st.metric("Maximum Risk", int(request_row["requested_max_avg_risk"]))
    with col4:
        st.metric("Minimum Suppliers", int(request_row["requested_min_suppliers"]))

    st.caption(
        f"Max Supplier Share: {float(request_row['requested_max_supplier_share']):.2f} • "
        f"Blocked Regions: {', '.join(blocked_regions) if blocked_regions else 'None'}"
    )


def render_request_detail(request_row: Dict[str, Any]) -> None:
    blocked_regions = _safe_list_from_pipe(request_row.get("blocked_regions_raw", ""))

    detail_df = pd.DataFrame(
        [
            {
                "Request ID": request_row["request_id"],
                "Requester": request_row["requester_name"],
                "Business Unit": request_row["business_unit"],
                "Priority": request_row["priority"],
                "Status": request_row["status"],
                "Demand": request_row["requested_demand_units"],
                "Min ESG": request_row["requested_min_avg_esg"],
                "Max Risk": request_row["requested_max_avg_risk"],
                "Min Suppliers": request_row["requested_min_suppliers"],
                "Max Supplier Share": request_row["requested_max_supplier_share"],
                "Blocked Regions": ", ".join(blocked_regions) if blocked_regions else "None",
                "Manual Approval Required": request_row["manual_approval_required"],
                "Submitted At": request_row["submitted_at"],
                "Due By": request_row["due_by"],
            }
        ]
    )
    st.dataframe(detail_df, use_container_width=True, hide_index=True)

    with st.expander("Raw Request Text", expanded=False):
        st.write(request_row["request_text"])


def render_run_detail(run_row: Dict[str, Any]) -> None:
    display_row = dict(run_row)
    display_row["current_stage"] = _display_stage_name(display_row["current_stage"])
    run_df = pd.DataFrame([display_row])
    st.dataframe(run_df, use_container_width=True, hide_index=True)


def render_run_summary_panel(
    selected_run: Dict[str, Any],
    validation_row: Dict[str, Any],
    approval_row: Dict[str, Any],
    recommendation_row: Dict[str, Any],
) -> None:
    validation_status = _display_validation_status(validation_row.get("validation_status", "pending"))
    approval_status = _display_approval_status(approval_row.get("decision", "pending"))
    recommendation_status = _display_recommendation_status(
        recommendation_row.get("recommendation_status", "not_generated")
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Run Status", selected_run.get("status", "n/a"))
    with col2:
        st.metric("Validation", validation_status)
    with col3:
        st.metric("Approval", approval_status)
    with col4:
        st.metric("Recommendation", recommendation_status)

    headline = recommendation_row.get("headline", "")
    if headline and str(headline).strip():
        st.caption(f"Recommendation: {headline}")


def render_workflow_execution_panel() -> None:
    st.subheader("Workflow Orchestrator")
    st.write("Manage sourcing requests step by step: review, approve, run, and track outcomes.")

    requests_df = load_workflow_table("requests")
    runs_df = load_workflow_table("runs")
    validation_df = load_workflow_table("validation")
    approvals_df = load_workflow_table("approvals")
    recommendations_df = load_workflow_table("recommendations")

    browser_df = _build_request_browser_df(
        requests_df=requests_df,
        runs_df=runs_df,
        validation_df=validation_df,
        approvals_df=approvals_df,
    )

    status_bucket = st.selectbox(
        "Show cases by status",
        options=[
            "Needs review",
            "Pending approval",
            "Rejected",
            "Completed",
            "Validation failed",
            "All",
        ],
    )

    filtered_df = browser_df.copy()

    if status_bucket != "All":
        status_map = {
            "Needs review": "Needs Review",
            "Pending approval": "Awaiting Approval",
            "Rejected": "Rejected",
            "Completed": "Completed",
            "Validation failed": "Validation Failed",
        }
        filtered_df = filtered_df[filtered_df["case_status"] == status_map[status_bucket]]

    if filtered_df.empty:
        st.info("No cases match the selected filter.")
        return

    request_options = {
        row["request_label"]: row["request_id"]
        for _, row in filtered_df.iterrows()
    }

    selected_request_display = st.selectbox(
        "Select a case",
        options=list(request_options.keys()),
    )

    selected_request_id = request_options[selected_request_display]

    request_row = requests_df[requests_df["request_id"] == selected_request_id].iloc[0].to_dict()
    request_runs = runs_df[runs_df["request_id"] == selected_request_id].sort_values("run_version")

    st.markdown("### Case Overview")
    render_case_summary_cards(request_row)

    with st.expander("View Full Case Details", expanded=False):
        render_request_detail(request_row)

    run_options = {}
    for _, run_row in request_runs.iterrows():
        validation_row = validation_df[validation_df["workflow_run_id"] == run_row["workflow_run_id"]]
        approval_row = approvals_df[approvals_df["workflow_run_id"] == run_row["workflow_run_id"]]

        validation_series = validation_row.iloc[0] if not validation_row.empty else None
        approval_series = approval_row.iloc[0] if not approval_row.empty else None

        display = f"{run_row['workflow_run_id']} • {_build_run_label(run_row, validation_series, approval_series)}"
        run_options[display] = run_row["workflow_run_id"]

    run_display_options = list(run_options.keys())

    default_run_index = 0
    preferred_run_id = st.session_state.get("selected_workflow_run_id")
    if preferred_run_id:
        for idx, display in enumerate(run_display_options):
            if run_options[display] == preferred_run_id:
                default_run_index = idx
                break

    selected_run_display = st.selectbox(
        "Select a workflow run",
        options=run_display_options,
        index=default_run_index,
    )

    selected_run_id = run_options[selected_run_display]
    selected_run = request_runs[request_runs["workflow_run_id"] == selected_run_id].iloc[0].to_dict()

    st.session_state["selected_workflow_run_id"] = selected_run_id

    validation_match = validation_df[validation_df["workflow_run_id"] == selected_run_id]
    approval_match = approvals_df[approvals_df["workflow_run_id"] == selected_run_id]
    recommendation_match = recommendations_df[recommendations_df["workflow_run_id"] == selected_run_id]

    validation_row = validation_match.iloc[0].to_dict() if not validation_match.empty else {}
    approval_row = approval_match.iloc[0].to_dict() if not approval_match.empty else {}
    recommendation_row = recommendation_match.iloc[0].to_dict() if not recommendation_match.empty else {}

    st.markdown("### Workflow Status")
    render_run_summary_panel(selected_run, validation_row, approval_row, recommendation_row)

    detail_tab_1, detail_tab_2, detail_tab_3, detail_tab_4 = st.tabs(
        ["Run Details", "Validation", "Approval", "Recommendation"]
    )

    with detail_tab_1:
        render_run_detail(selected_run)

    with detail_tab_2:
        st.caption(
            "Pre-execution business-rule screening. This reduces obvious bad runs, "
            "but does not guarantee solver feasibility."
        )
        if validation_row:
            st.dataframe(pd.DataFrame([validation_row]), use_container_width=True, hide_index=True)
        else:
            st.info("No validation row found for this run.")

    with detail_tab_3:
        if approval_row:
            st.dataframe(pd.DataFrame([approval_row]), use_container_width=True, hide_index=True)
        else:
            st.info("No approval row found for this run.")

    with detail_tab_4:
        if recommendation_row:
            st.dataframe(pd.DataFrame([recommendation_row]), use_container_width=True, hide_index=True)
        else:
            st.info("No recommendation row found for this run.")

    st.markdown("### Workflow Action")

    selected_status = str(selected_run.get("status", "")).strip()
    run_is_immutable = selected_status in {"Completed", "Rejected", "Awaiting Approval"}

    orchestrator = WorkflowOrchestrator(
        data_dir="data",
        suppliers_path="data/suppliers_dataset.csv",
    )

    if selected_status == "Completed":
        st.info("This run is completed and kept as a historical record. Create a new run to continue work on this case.")
        action_label = "Create New Run"

    elif selected_status == "Rejected":
        st.info("This run was rejected and kept as a historical record. Create a new run if the case needs to be revisited.")
        action_label = "Create New Run"

    elif selected_status == "Awaiting Approval":
        st.warning("This run is waiting for approval and cannot proceed until it is reviewed. Create a new run only if you want to start a separate version of the case.")
        action_label = "Create New Run"

    else:
        st.success("This run is ready to execute.")
        action_label = "Run Workflow"

    run_workflow_clicked = st.button(
        action_label,
        type="primary",
        use_container_width=True,
    )

    if run_workflow_clicked:
        with st.spinner("Processing workflow..."):
            try:
                selected_status = str(selected_run.get("status", ""))

                if selected_status in {"Completed", "Rejected", "Awaiting Approval"}:
                    new_run = orchestrator.create_run_version(
                        request_id=selected_request_id,
                        source_workflow_run_id=selected_run_id,
                        trigger_type="auto_rerun",
                    )

                    new_run_id = new_run["workflow_run_id"]
                    workflow_output = orchestrator.run_workflow(new_run_id)

                    clear_workflow_caches()

                    st.session_state["last_workflow_output"] = workflow_output
                    st.session_state["last_created_run_version"] = new_run
                    st.session_state["selected_workflow_run_id"] = new_run_id

                    st.success(
                        f"Created {new_run_id} and started a new workflow run."
                    )
                    st.rerun()

                else:
                    workflow_output = orchestrator.run_workflow(selected_run_id)

                    clear_workflow_caches()

                    st.session_state["last_workflow_output"] = workflow_output
                    st.session_state["selected_workflow_run_id"] = selected_run_id

                    st.success(
                        f"Workflow finished with status: {workflow_output['final_status']}"
                    )

            except Exception as exc:
                st.error(f"Workflow execution failed: {exc}")

    latest_created_run = st.session_state.get("last_created_run_version")
    if latest_created_run and latest_created_run.get("request_id") == selected_request_id:
        with st.expander("Latest Run Created", expanded=False):
            st.dataframe(pd.DataFrame([latest_created_run]), use_container_width=True, hide_index=True)

    workflow_output = st.session_state.get("last_workflow_output")

    if workflow_output:
        active_output_run_id = workflow_output.get("workflow_run_id")
        if active_output_run_id == st.session_state.get("selected_workflow_run_id"):
            with st.expander("Latest Workflow Output", expanded=False):
                display_output = {}
                for key, value in workflow_output.items():
                    if isinstance(value, dict):
                        display_output[key] = f"<dict: {', '.join(list(value.keys())[:8])}>"
                    else:
                        display_output[key] = value
                st.dataframe(pd.DataFrame([display_output]), use_container_width=True, hide_index=True)


def render_trace_and_artifacts_panel() -> None:
    st.subheader("Run History")
    st.write("Review what happened in a run and what outputs were generated.")

    runs_df = load_workflow_table("runs")
    trace_df = load_workflow_table("trace")
    artifacts_df = load_workflow_table("artifacts")

    if runs_df.empty:
        st.warning("No workflow runs available.")
        return

    run_options = runs_df.sort_values("workflow_run_id").copy()
    run_options["label"] = run_options.apply(
        lambda row: (
            f"{row['workflow_run_id']} • {row['request_id']} • "
            f"{_display_stage_name(row['current_stage'])} • {row['status']}"
        ),
        axis=1,
    )

    selected_run_label = st.selectbox(
        "Select a run",
        options=run_options["label"].tolist(),
        key="trace_artifact_run_select",
    )

    selected_run_id = selected_run_label.split(" • ")[0]

    run_trace = trace_df[trace_df["workflow_run_id"] == selected_run_id].sort_values("timestamp")
    run_artifacts = artifacts_df[artifacts_df["workflow_run_id"] == selected_run_id].sort_values("created_at")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Workflow Steps")
        if run_trace.empty:
            st.info("No trace events found for this run.")
        else:
            display_trace = run_trace.copy()
            if "stage_name" in display_trace.columns:
                display_trace["stage_name"] = display_trace["stage_name"].apply(_display_stage_name)
            st.dataframe(display_trace, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("### Generated Outputs")
        if run_artifacts.empty:
            st.info("No artifacts found for this run.")
        else:
            st.dataframe(run_artifacts, use_container_width=True, hide_index=True)


def render_dataset_browser() -> None:
    st.subheader("Dataset Browser")
    st.write("Browse the workflow data tables used by the orchestration layer.")

    dataset_name = st.selectbox(
        "Select a dataset",
        options=[
            "requests",
            "runs",
            "validation",
            "approvals",
            "trace",
            "artifacts",
            "recommendations",
        ],
    )

    df = load_workflow_table(dataset_name)
    st.dataframe(df, use_container_width=True, hide_index=True)


# ---------------------------------------------------
# Main app
# ---------------------------------------------------
def main() -> None:
    try:
        df = get_supplier_data()
    except Exception as exc:
        st.error(f"Failed to load supplier data: {exc}")
        st.stop()

    render_header(df)

    mode = st.radio(
        "Choose workspace",
        options=["Core Sourcing Analysis", "Workflow Orchestrator"],
        horizontal=True,
    )

    if mode == "Core Sourcing Analysis":
        st.caption(
            "Use this workspace for direct scenario analysis: manual inputs or natural-language parsing, "
            "then optimization, audit, and sensitivity analysis."
        )
        run_existing_analysis_flow(df)
        return

    st.caption(
        "Use this workspace to manage sourcing requests through review, approval, execution, "
        "recommendation, and run history."
    )

    render_workflow_overview()

    tab1, tab2, tab3 = st.tabs(
        ["Workflow Execution", "Run History", "Dataset Browser"]
    )

    with tab1:
        render_workflow_execution_panel()

    with tab2:
        render_trace_and_artifacts_panel()

    with tab3:
        render_dataset_browser()


if __name__ == "__main__":
    main()