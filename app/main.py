import sys
from pathlib import Path

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


st.set_page_config(
    page_title="AI-Assisted Sourcing Decision Copilot",
    page_icon="📦",
    layout="wide",
)


@st.cache_data
def get_supplier_data() -> pd.DataFrame:
    return load_suppliers("data/suppliers_dataset.csv")


def _format_currency(value: float) -> str:
    return f"{value:,.2f}"


def _build_base_params(
    total_demand: int,
    max_supplier_share: float,
    min_avg_esg: int,
    max_avg_risk: int,
    min_suppliers: int,
    blocked_regions: list[str],
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
    """
    Keeps parser notes short and user-facing.
    """
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


def render_header(df: pd.DataFrame) -> None:
    st.title("AI-Assisted Sourcing Decision Copilot")
    st.caption(
        "A sourcing decision tool that combines optimization, audit, sensitivity analysis, "
        "and controlled natural-language input parsing."
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
    st.subheader("Natural Language Scenario Input")
    st.write(
        "Describe a sourcing case in plain English. The parser turns it into structured inputs, "
        "but the optimizer still makes the allocation decision."
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
        parse_clicked = st.button("Parse Natural Language Request", use_container_width=True)
    with col2:
        clear_clicked = st.button("Clear Parsed Scenario", use_container_width=True)

    if clear_clicked:
        st.session_state.pop("parsed_output", None)
        st.session_state.pop("parsed_scenario", None)
        st.session_state.pop("use_parsed_scenario", None)
        st.rerun()

    if parse_clicked:
        if not user_request.strip():
            st.warning("Enter a sourcing request before parsing.")
        else:
            with st.spinner("Parsing sourcing request..."):
                try:
                    parsed_output = parse_sourcing_request(user_request.strip())
                    st.session_state["parsed_output"] = parsed_output
                    st.session_state["parsed_scenario"] = parsed_output["parsed_scenario"]
                    st.session_state["use_parsed_scenario"] = True
                except Exception as exc:
                    st.error(f"Parser failed: {exc}")

    parsed_output = st.session_state.get("parsed_output")

    if parsed_output:
        st.warning("Review the parsed scenario before running analysis.")

        with st.expander("Parsed Scenario", expanded=True):
            left, right = st.columns(2)

            with left:
                st.dataframe(
                    pd.DataFrame([parsed_output["parsed_scenario"]]),
                    use_container_width=True,
                    hide_index=True,
                )

                explicit_fields = parsed_output.get("explicit_fields", [])
                heuristic_fields = parsed_output.get("heuristic_fields", [])
                missing_fields = parsed_output.get("missing_fields", [])

                st.write("**Field status**")
                if explicit_fields:
                    st.caption("Explicitly parsed")
                    st.markdown(", ".join([f"`{x}`" for x in explicit_fields]))

                if heuristic_fields:
                    st.caption("Heuristically interpreted")
                    st.markdown(", ".join([f"`{x}`" for x in heuristic_fields]))

                if missing_fields:
                    st.caption("Defaulted")
                    st.markdown(", ".join([f"`{x}`" for x in missing_fields]))

            with right:
                st.write(f"**Interpretation:** {parsed_output['interpretation']}")

                notes = _build_short_parser_notes(parsed_output)
                if notes:
                    st.write("**Notes**")
                    for item in notes:
                        st.markdown(f"- {item}")

            use_parsed = st.checkbox(
                "Use parsed scenario for analysis",
                value=st.session_state.get("use_parsed_scenario", True),
                key="use_parsed_checkbox",
            )
            st.session_state["use_parsed_scenario"] = use_parsed

            with st.expander("Show Raw Model Output", expanded=False):
                st.code(parsed_output.get("raw_model_output", ""), language="json")


def render_sidebar(df: pd.DataFrame) -> dict:
    st.sidebar.header("Manual Scenario Inputs")

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

    st.sidebar.markdown("### Region Filter")
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
        "ESG Reward Weight",
        min_value=0.0,
        max_value=1.0,
        value=float(active_defaults["w_esg"]),
        step=0.05,
    )

    supplier_selection_penalty = st.sidebar.slider(
        "Supplier Selection Penalty",
        min_value=0.0,
        max_value=0.20,
        value=float(active_defaults["supplier_selection_penalty"]),
        step=0.01,
    )

    st.sidebar.caption("Higher cost weight favors cheaper suppliers.")
    st.sidebar.caption("Higher risk weight penalizes higher-risk suppliers.")
    st.sidebar.caption("Higher ESG weight rewards stronger ESG suppliers.")

    if parsed_scenario and use_parsed_scenario:
        st.sidebar.success("Inputs are prefilled from the parsed scenario.")

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
        return f"{prefix}ESG and risk are both shaping the portfolio."
    if esg_tightness in {"binding_or_nearly_binding", "tight"}:
        return f"{prefix}the ESG floor is shaping the portfolio."
    if risk_tightness in {"binding_or_nearly_binding", "tight"}:
        return f"{prefix}the risk cap is shaping the portfolio."
    if concentration >= 0.70:
        return f"{prefix}share limits are shaping the allocation because volume is still concentrated."

    return f"{prefix}cost appears to be the main driver."


def _get_main_vulnerability(audit: dict, sensitivity_output: dict, result: dict, parser_context: dict) -> str:
    if result["status"] != "Optimal":
        return "The main issue is feasibility under the current rules."

    portfolio_checks = audit.get("portfolio_checks", {})
    comparison_table = sensitivity_output.get("comparison_table", pd.DataFrame())

    concentration = portfolio_checks.get("top_2_supplier_concentration_pct")
    esg_tightness = portfolio_checks.get("esg_tightness")

    prefix = "Under the interpreted scenario, " if _needs_soft_language(parser_context) else ""

    if concentration is not None and concentration >= 0.70:
        return f"{prefix}the portfolio is still too concentrated in the top suppliers."

    if esg_tightness in {"binding_or_nearly_binding", "tight"}:
        return f"{prefix}the portfolio is close to the ESG boundary."

    blocked_row = comparison_table[comparison_table["scenario_name"] == "block_eastern_europe"]
    if not blocked_row.empty and blocked_row.iloc[0]["status"] != "Optimal":
        return f"{prefix}the portfolio depends too much on Eastern Europe."

    higher_demand_row = comparison_table[comparison_table["scenario_name"] == "higher_demand"]
    if not higher_demand_row.empty and higher_demand_row.iloc[0]["status"] != "Optimal":
        return f"{prefix}the portfolio may struggle under higher demand."

    return f"{prefix}no single weakness stands out strongly."


def _get_priority_action(audit: dict, sensitivity_output: dict, result: dict, parser_context: dict) -> str:
    if result["status"] != "Optimal":
        return "Relax one or more constraints or widen the eligible supplier pool."

    portfolio_checks = audit.get("portfolio_checks", {})
    comparison_table = sensitivity_output.get("comparison_table", pd.DataFrame())

    concentration = portfolio_checks.get("top_2_supplier_concentration_pct")
    esg_tightness = portfolio_checks.get("esg_tightness")

    if concentration is not None and concentration >= 0.70:
        return "Lower max supplier share to 25–30% and rerun."

    if esg_tightness in {"binding_or_nearly_binding", "tight"}:
        stricter_row = comparison_table[comparison_table["scenario_name"] == "stricter_esg"]
        if not stricter_row.empty and stricter_row.iloc[0]["status"] == "Optimal":
            return "Test a higher ESG floor before changing the target."

    blocked_row = comparison_table[comparison_table["scenario_name"] == "block_eastern_europe"]
    if not blocked_row.empty and blocked_row.iloc[0]["status"] != "Optimal":
        return "Add more non-Eastern-Europe supply before using that region block."

    if _needs_soft_language(parser_context):
        return "Replace vague terms with numeric targets and rerun."

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
            return (
                f"{prefix}the clearest trade-off is between cost and sustainability, "
                f"because a higher ESG requirement stays feasible but increases cost."
            )

    if not tighter_share_row.empty and tighter_share_row.iloc[0]["status"] == "Optimal":
        base_conc = base_row.get("top_2_concentration_pct")
        new_conc = tighter_share_row.iloc[0].get("top_2_concentration_pct")
        if pd.notna(base_conc) and pd.notna(new_conc) and float(new_conc) < float(base_conc):
            return (
                f"{prefix}the clearest trade-off is between cost efficiency and diversification, "
                f"because tighter share limits reduce concentration."
            )

    return f"{prefix}no single trade-off dominates strongly across the tested scenarios."


def _get_plan_fragility(audit: dict, sensitivity_output: dict, result: dict, parser_context: dict) -> str:
    if result["status"] != "Optimal":
        return "The plan is fragile because it is not feasible under the current rules."

    comparison_table = sensitivity_output.get("comparison_table", pd.DataFrame())
    if comparison_table.empty:
        return "No fragility signal is available from scenario testing."

    prefix = "Under the interpreted scenario, " if _needs_soft_language(parser_context) else ""

    higher_demand_row = comparison_table[comparison_table["scenario_name"] == "higher_demand"]
    if not higher_demand_row.empty and higher_demand_row.iloc[0]["status"] != "Optimal":
        return f"{prefix}the main fragility is demand scaling, because the plan fails when demand increases by 25%."

    blocked_row = comparison_table[comparison_table["scenario_name"] == "block_eastern_europe"]
    if not blocked_row.empty and blocked_row.iloc[0]["status"] != "Optimal":
        return f"{prefix}the plan appears region-dependent, because blocking Eastern Europe makes it infeasible."

    portfolio_checks = audit.get("portfolio_checks", {})
    concentration = portfolio_checks.get("top_2_supplier_concentration_pct")
    if concentration is not None and concentration >= 0.70:
        return f"{prefix}the main fragility is supplier concentration, because too much volume still sits with the top suppliers."

    return f"{prefix}the plan looks reasonably stable across the tested scenario changes."


def _get_best_next_lever(audit: dict, sensitivity_output: dict, result: dict, parser_context: dict) -> str:
    if result["status"] != "Optimal":
        return "Best next lever: relax one or more constraints or widen the eligible supplier pool."

    comparison_table = sensitivity_output.get("comparison_table", pd.DataFrame())
    portfolio_checks = audit.get("portfolio_checks", {})
    concentration = portfolio_checks.get("top_2_supplier_concentration_pct")
    esg_tightness = portfolio_checks.get("esg_tightness")

    blocked_row = comparison_table[comparison_table["scenario_name"] == "block_eastern_europe"]
    higher_demand_row = comparison_table[comparison_table["scenario_name"] == "higher_demand"]
    stricter_esg_row = comparison_table[comparison_table["scenario_name"] == "stricter_esg"]

    if not blocked_row.empty and blocked_row.iloc[0]["status"] != "Optimal":
        return "Best next lever: expand supplier capacity outside Eastern Europe before applying a regional exclusion."

    if not higher_demand_row.empty and higher_demand_row.iloc[0]["status"] != "Optimal":
        return "Best next lever: test additional capacity or a broader supplier pool before planning for higher demand."

    if concentration is not None and concentration >= 0.70:
        return "Best next lever: tighten the supplier share cap if resilience and diversification matter most."

    if esg_tightness in {"binding_or_nearly_binding", "tight"}:
        if not stricter_esg_row.empty and stricter_esg_row.iloc[0]["status"] == "Optimal":
            return "Best next lever: test a higher ESG target if sustainability is becoming a stronger business priority."

    if _needs_soft_language(parser_context):
        return "Best next lever: replace vague business preferences with numeric targets and rerun the scenario."

    return "Best next lever: use the current result as the baseline and tighten one business priority at a time."


def render_decision_summary_panel(result: dict, audit: dict, sensitivity_output: dict, parser_context: dict) -> None:
    st.subheader("Decision Summary")

    if result["status"] != "Optimal":
        st.error(result.get("message", "No feasible recommendation available."))
        st.info("Outcome: the current scenario is not feasible.")
        st.warning("Next step: relax one or more constraints and rerun.")
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
            "This result uses a mix of parsed, inferred, and defaulted inputs. Review the parsed scenario before treating it as final."
        )

    insight_cols = st.columns(3)

    with insight_cols[0]:
        st.info(f"**Main Driver**\n\n{_get_main_driver(audit, result, parser_context)}")

    with insight_cols[1]:
        st.warning(f"**Main Vulnerability**\n\n{_get_main_vulnerability(audit, sensitivity_output, result, parser_context)}")

    with insight_cols[2]:
        st.success(f"**Priority Action**\n\n{_get_priority_action(audit, sensitivity_output, result, parser_context)}")


def render_tradeoff_guidance(audit: dict, sensitivity_output: dict, result: dict, parser_context: dict) -> None:
    st.subheader("Trade-off Guidance")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(
            f"**Main Trade-off**\n\n"
            f"{_get_main_tradeoff(audit, sensitivity_output, result, parser_context)}"
        )

    with col2:
        st.warning(
            f"**Plan Fragility**\n\n"
            f"{_get_plan_fragility(audit, sensitivity_output, result, parser_context)}"
        )

    with col3:
        st.success(
            f"**Best Next Lever**\n\n"
            f"{_get_best_next_lever(audit, sensitivity_output, result, parser_context)}"
        )


def _build_portfolio_flags(audit: dict, sensitivity_output: dict, result: dict, parser_context: dict) -> list[dict]:
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

    if concentration is not None and concentration >= 0.70:
        flags.append(
            {
                "flag": "High concentration",
                "severity": "High",
                "why_it_matters": "Too much volume sits with the top two suppliers.",
                "recommended_action": "Lower max supplier share and test the case again.",
                "priority": 95,
            }
        )

    if esg_tightness in {"binding_or_nearly_binding", "tight"}:
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
    if not blocked_row.empty and blocked_row.iloc[0]["status"] != "Optimal":
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
    if not higher_demand_row.empty and higher_demand_row.iloc[0]["status"] != "Optimal":
        flags.append(
            {
                "flag": "Demand scaling risk",
                "severity": "High",
                "why_it_matters": "The case fails when demand increases.",
                "recommended_action": "Check spare capacity before committing to higher demand.",
                "priority": 85,
            }
        )

    if not flags:
        soft_text = "under the interpreted inputs" if _needs_soft_language(parser_context) else "in this case"
        flags.append(
            {
                "flag": "No major stress signal",
                "severity": "Low",
                "why_it_matters": f"The portfolio looks reasonably balanced {soft_text}.",
                "recommended_action": "Use this result as a baseline and test one tighter rule next.",
                "priority": 10,
            }
        )

    flags = sorted(flags, key=lambda x: x["priority"], reverse=True)
    return flags[:2]


def render_portfolio_risk_flags(audit: dict, sensitivity_output: dict, result: dict, parser_context: dict) -> None:
    st.subheader("Portfolio Risk Flags")

    flags = _build_portfolio_flags(audit, sensitivity_output, result, parser_context)

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

    st.caption("This shows whether awarded volume is well spread or still concentrated in the top suppliers.")


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

    with st.expander("Allocation Table", expanded=True):
        st.dataframe(allocations, use_container_width=True, hide_index=True)

    with st.expander("Optimization Metadata", expanded=False):
        st.dataframe(pd.DataFrame([metadata]), use_container_width=True, hide_index=True)

    with st.expander("Eligible Supplier Pool Used in Optimization", expanded=False):
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

    with st.expander("Simple explanation", expanded=True):
        st.info(audit["plain_english_explanation"])

    col1, col2 = st.columns(2)
    with col1:
        with st.expander("Portfolio Checks", expanded=True):
            st.dataframe(
                pd.DataFrame([portfolio_checks]),
                use_container_width=True,
                hide_index=True,
            )
    with col2:
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

    st.markdown("### Scenario Cost Impact vs Base Case")
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


def main() -> None:
    try:
        df = get_supplier_data()
    except Exception as exc:
        st.error(f"Failed to load supplier data: {exc}")
        st.stop()

    render_header(df)
    render_nl_parser_section()
    base_params = render_sidebar(df)
    render_scenario_snapshot(base_params)

    run_button = st.button("Run Sourcing Analysis", type="primary", use_container_width=True)

    if not run_button:
        st.markdown("### How to use")
        st.write(
            "You can adjust the scenario manually in the sidebar or describe it in plain English above. "
            "If you use the parser, review the parsed fields before running the analysis."
        )
        st.stop()

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
        ["Optimization Results", "Decision Audit", "Sensitivity Analysis"]
    )

    with tab1:
        render_optimizer_output(result)

    with tab2:
        render_audit_output(audit)

    with tab3:
        render_sensitivity_output(sensitivity_output)


if __name__ == "__main__":
    main()