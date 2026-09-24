from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import nbformat as nbf
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from jinja2 import Template
from nbclient import NotebookClient
from plotly.offline import get_plotlyjs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "personal_finance_tracker_dataset.csv"
OUTPUT_DIR = BASE_DIR / "analysis_outputs"
NOTEBOOK_FILE = BASE_DIR / "personal_finance_analysis.ipynb"
DASHBOARD_FILE = BASE_DIR / "finance_story_dashboard.html"


STRESS_MAP = {"Low": 1, "Medium": 2, "High": 3}
SCENARIO_ORDER = ["normal", "inflation", "recession"]
SEGMENT_PROFILES = {
    "Resilient investors": {
        "meaning": (
            "Users with the strongest overall savings profile. They combine healthy savings rates, "
            "better emergency coverage, and relatively manageable debt pressure."
        ),
        "standard": (
            "Benchmark segment. Keep savings consistent, preserve liquidity, and deepen investment "
            "planning without increasing avoidable debt."
        ),
        "bank_action": (
            "Offer high-yield savings, automated investing, portfolio review, and pre-approved "
            "credit only when repayment capacity remains healthy."
        ),
        "advisor_action": (
            "Shift the conversation from budgeting to wealth building: asset allocation, tax-aware "
            "investing, insurance coverage, and retirement milestones."
        ),
    },
    "Goal-focused savers": {
        "meaning": (
            "Users who are most likely to hit savings goals even if their average savings rate is "
            "not the highest. They show discipline around target-based saving."
        ),
        "standard": (
            "Goal-achievement standard. Protect the habit with automatic transfers, clear target "
            "dates, and alerts when spending threatens the goal."
        ),
        "bank_action": (
            "Provide goal pockets, round-up savings, bonus rates for streaks, and personalized "
            "goal-progress nudges."
        ),
        "advisor_action": (
            "Translate goals into a plan: emergency fund target, debt payoff schedule, investment "
            "contribution ladder, and review cadence."
        ),
    },
    "Debt-pressure spenders": {
        "meaning": (
            "Users with the highest debt burden and lower goal achievement. Cash flow is present, "
            "but debt payments and expense pressure limit progress."
        ),
        "standard": (
            "Risk-reduction standard. Bring debt-to-income down first, then rebuild savings goals "
            "after repayment pressure eases."
        ),
        "bank_action": (
            "Prioritize refinance checks, debt consolidation options, payment reminders, spending "
            "limits, and hardship or restructuring support where suitable."
        ),
        "advisor_action": (
            "Build a debt-first plan: rank balances by rate, cap discretionary leakage, protect "
            "minimum emergency cash, and set a realistic payoff milestone."
        ),
    },
    "Cash-flow stretched": {
        "meaning": (
            "Users whose expenses are close to or above income, leaving little room for actual "
            "savings despite moderate savings-rate signals."
        ),
        "standard": (
            "Stabilization standard. Create positive monthly cash flow before pushing aggressive "
            "investment or long-term saving targets."
        ),
        "bank_action": (
            "Use low-balance alerts, bill timing tools, subscription review, overdraft prevention, "
            "and small emergency savings automation."
        ),
        "advisor_action": (
            "Start with cash-flow repair: separate essential and discretionary spending, reset the "
            "budget, renegotiate recurring costs, and define a small starter emergency fund."
        ),
    },
}


def pct(value: float) -> str:
    return f"{value:.1%}"


def money(value: float) -> str:
    return f"${value:,.0f}"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE, parse_dates=["date"])
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["expense_to_income"] = np.divide(
        out["monthly_expense_total"],
        out["monthly_income"],
        out=np.zeros(len(out)),
        where=out["monthly_income"].ne(0),
    )
    out["discretionary_share"] = np.divide(
        out["discretionary_spending"],
        out["monthly_expense_total"],
        out=np.zeros(len(out)),
        where=out["monthly_expense_total"].ne(0),
    )
    out["essential_share"] = np.divide(
        out["essential_spending"],
        out["monthly_expense_total"],
        out=np.zeros(len(out)),
        where=out["monthly_expense_total"].ne(0),
    )
    out["investment_rate"] = np.divide(
        out["investment_amount"],
        out["monthly_income"],
        out=np.zeros(len(out)),
        where=out["monthly_income"].ne(0),
    )
    out["emergency_fund_months"] = np.divide(
        out["emergency_fund"],
        out["monthly_expense_total"],
        out=np.zeros(len(out)),
        where=out["monthly_expense_total"].ne(0),
    )
    out["stress_score"] = out["financial_stress_level"].map(STRESS_MAP)
    return out


def make_category_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("category", as_index=False)
        .agg(
            records=("category", "size"),
            avg_income=("monthly_income", "mean"),
            avg_expense=("monthly_expense_total", "mean"),
            avg_actual_savings=("actual_savings", "mean"),
            avg_savings_rate=("savings_rate", "mean"),
            goal_met_rate=("savings_goal_met", "mean"),
            avg_debt_to_income=("debt_to_income_ratio", "mean"),
            avg_discretionary_share=("discretionary_share", "mean"),
            avg_essential_share=("essential_share", "mean"),
            avg_stress_score=("stress_score", "mean"),
            fraud_rate=("fraud_flag", "mean"),
        )
        .sort_values("avg_expense", ascending=False)
    )
    total_records = summary["records"].sum()
    summary["record_share"] = summary["records"] / total_records
    return summary


def make_monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("month", as_index=False)
        .agg(
            avg_income=("monthly_income", "mean"),
            avg_expense=("monthly_expense_total", "mean"),
            avg_actual_savings=("actual_savings", "mean"),
            avg_savings_rate=("savings_rate", "mean"),
            goal_met_rate=("savings_goal_met", "mean"),
            avg_debt_to_income=("debt_to_income_ratio", "mean"),
        )
        .sort_values("month")
    )


def label_clusters(cluster_summary: pd.DataFrame) -> dict[int, str]:
    score_frame = cluster_summary.copy()
    metrics = [
        "avg_savings_rate",
        "goal_met_rate",
        "avg_emergency_fund_months",
        "avg_investment_rate",
        "avg_debt_to_income",
        "avg_expense_to_income",
    ]
    z = score_frame[metrics].apply(
        lambda s: (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) else 1)
    )
    score_frame["strength_score"] = (
        z["avg_savings_rate"]
        + z["goal_met_rate"]
        + z["avg_emergency_fund_months"]
        + z["avg_investment_rate"]
        - z["avg_debt_to_income"]
        - z["avg_expense_to_income"]
    )

    labels: dict[int, str] = {}
    available = set(score_frame["cluster"].tolist())

    strongest = int(score_frame.loc[score_frame["strength_score"].idxmax(), "cluster"])
    labels[strongest] = "Resilient investors"
    available.remove(strongest)

    if available:
        goal_cluster = int(
            score_frame[score_frame["cluster"].isin(available)]
            .sort_values(["goal_met_rate", "avg_savings_rate"], ascending=False)
            .iloc[0]["cluster"]
        )
        labels[goal_cluster] = "Goal-focused savers"
        available.remove(goal_cluster)

    if available:
        debt_cluster = int(
            score_frame[score_frame["cluster"].isin(available)]
            .sort_values(["avg_debt_to_income", "avg_expense_to_income"], ascending=False)
            .iloc[0]["cluster"]
        )
        labels[debt_cluster] = "Debt-pressure spenders"
        available.remove(debt_cluster)

    for cluster_id in available:
        labels[int(cluster_id)] = "Cash-flow stretched"

    return labels


def make_user_segments(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    user = (
        df.groupby("user_id", as_index=False)
        .agg(
            observations=("user_id", "size"),
            avg_income=("monthly_income", "mean"),
            avg_expense=("monthly_expense_total", "mean"),
            avg_savings_rate=("savings_rate", "mean"),
            avg_actual_savings=("actual_savings", "mean"),
            goal_met_rate=("savings_goal_met", "mean"),
            avg_debt_to_income=("debt_to_income_ratio", "mean"),
            avg_loan_payment=("loan_payment", "mean"),
            avg_investment_amount=("investment_amount", "mean"),
            avg_subscription_services=("subscription_services", "mean"),
            avg_emergency_fund=("emergency_fund", "mean"),
            avg_transaction_count=("transaction_count", "mean"),
            avg_financial_advice_score=("financial_advice_score", "mean"),
            high_stress_rate=("financial_stress_level", lambda s: (s == "High").mean()),
            avg_discretionary_share=("discretionary_share", "mean"),
            avg_essential_share=("essential_share", "mean"),
            avg_investment_rate=("investment_rate", "mean"),
            avg_emergency_fund_months=("emergency_fund_months", "mean"),
            avg_expense_to_income=("expense_to_income", "mean"),
        )
    )

    features = [
        "avg_income",
        "avg_expense",
        "avg_savings_rate",
        "avg_actual_savings",
        "goal_met_rate",
        "avg_debt_to_income",
        "avg_loan_payment",
        "avg_investment_rate",
        "avg_emergency_fund_months",
        "avg_discretionary_share",
        "high_stress_rate",
        "avg_expense_to_income",
    ]
    scaled = StandardScaler().fit_transform(user[features])
    model = KMeans(n_clusters=4, n_init=25, random_state=42)
    user["cluster"] = model.fit_predict(scaled)

    pre_summary = (
        user.groupby("cluster", as_index=False)
        .agg(
            users=("user_id", "size"),
            avg_income=("avg_income", "mean"),
            avg_expense=("avg_expense", "mean"),
            avg_savings_rate=("avg_savings_rate", "mean"),
            avg_actual_savings=("avg_actual_savings", "mean"),
            goal_met_rate=("goal_met_rate", "mean"),
            avg_debt_to_income=("avg_debt_to_income", "mean"),
            avg_investment_rate=("avg_investment_rate", "mean"),
            avg_emergency_fund_months=("avg_emergency_fund_months", "mean"),
            avg_discretionary_share=("avg_discretionary_share", "mean"),
            high_stress_rate=("high_stress_rate", "mean"),
            avg_expense_to_income=("avg_expense_to_income", "mean"),
        )
    )
    labels = label_clusters(pre_summary)
    user["savings_segment"] = user["cluster"].map(labels)

    segment_summary = (
        user.groupby("savings_segment", as_index=False)
        .agg(
            users=("user_id", "size"),
            avg_income=("avg_income", "mean"),
            avg_expense=("avg_expense", "mean"),
            avg_savings_rate=("avg_savings_rate", "mean"),
            avg_actual_savings=("avg_actual_savings", "mean"),
            goal_met_rate=("goal_met_rate", "mean"),
            avg_debt_to_income=("avg_debt_to_income", "mean"),
            avg_loan_payment=("avg_loan_payment", "mean"),
            avg_investment_rate=("avg_investment_rate", "mean"),
            avg_emergency_fund_months=("avg_emergency_fund_months", "mean"),
            avg_discretionary_share=("avg_discretionary_share", "mean"),
            high_stress_rate=("high_stress_rate", "mean"),
            avg_expense_to_income=("avg_expense_to_income", "mean"),
        )
        .sort_values("users", ascending=False)
    )
    segment_summary["user_share"] = segment_summary["users"] / segment_summary["users"].sum()
    return user, segment_summary


def make_segment_guidance(segment_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    summary_lookup = {
        row["savings_segment"]: row for row in segment_summary.to_dict("records")
    }
    for segment, profile in SEGMENT_PROFILES.items():
        if segment not in summary_lookup:
            continue
        row = summary_lookup[segment]
        rows.append(
            {
                "savings_segment": segment,
                "users": int(row["users"]),
                "user_share": row["user_share"],
                "meaning": profile["meaning"],
                "standard": profile["standard"],
                "data_standard": (
                    f"{pct(row['avg_savings_rate'])} avg savings rate; "
                    f"{pct(row['goal_met_rate'])} goal-met rate; "
                    f"{pct(row['avg_debt_to_income'])} debt-to-income; "
                    f"{row['avg_emergency_fund_months']:.2f} months emergency coverage."
                ),
                "bank_action": profile["bank_action"],
                "advisor_action": profile["advisor_action"],
            }
        )
    return pd.DataFrame(rows)


def make_figures(
    df: pd.DataFrame,
    category_summary: pd.DataFrame,
    monthly_summary: pd.DataFrame,
    user_segments: pd.DataFrame,
    segment_summary: pd.DataFrame,
) -> dict[str, go.Figure]:
    template = "plotly_white"
    colors = {
        "Resilient investors": "#197278",
        "Goal-focused savers": "#2f80ed",
        "Debt-pressure spenders": "#c44536",
        "Cash-flow stretched": "#7b61ff",
    }

    cat_plot = category_summary.sort_values("avg_expense", ascending=True)
    fig_category_spend = px.bar(
        cat_plot,
        x="avg_expense",
        y="category",
        color="avg_savings_rate",
        color_continuous_scale=["#f5b7a7", "#f6d365", "#82c0cc"],
        template=template,
        labels={
            "avg_expense": "Average monthly expense",
            "category": "",
            "avg_savings_rate": "Savings rate",
        },
        title="Average Expense by Category",
    )
    fig_category_spend.update_layout(height=440, margin=dict(l=10, r=20, t=60, b=30))
    fig_category_spend.update_xaxes(tickprefix="$", separatethousands=True)

    fig_goal = px.bar(
        category_summary.sort_values("goal_met_rate", ascending=False),
        x="category",
        y="goal_met_rate",
        color="avg_debt_to_income",
        color_continuous_scale=["#7fc97f", "#fdc086", "#beaed4"],
        template=template,
        labels={
            "category": "",
            "goal_met_rate": "Savings goal met rate",
            "avg_debt_to_income": "Debt-to-income",
        },
        title="Goal Achievement by Category",
    )
    fig_goal.update_layout(height=390, margin=dict(l=20, r=20, t=60, b=90))
    fig_goal.update_yaxes(tickformat=".0%")

    fig_segments = px.scatter(
        user_segments,
        x="avg_debt_to_income",
        y="avg_savings_rate",
        size="avg_income",
        color="savings_segment",
        color_discrete_map=colors,
        hover_data={
            "user_id": True,
            "avg_income": ":$,.0f",
            "avg_expense": ":$,.0f",
            "goal_met_rate": ":.0%",
            "avg_emergency_fund_months": ":.2f",
        },
        template=template,
        labels={
            "avg_debt_to_income": "Average debt-to-income",
            "avg_savings_rate": "Average savings rate",
            "savings_segment": "Segment",
        },
        title="Savings Behavior Segments",
    )
    fig_segments.update_layout(height=500, margin=dict(l=20, r=20, t=60, b=40))
    fig_segments.update_xaxes(tickformat=".0%")
    fig_segments.update_yaxes(tickformat=".0%")

    fig_segment_bars = go.Figure()
    ordered = segment_summary.sort_values("avg_savings_rate", ascending=False)
    fig_segment_bars.add_trace(
        go.Bar(
            name="Savings rate",
            x=ordered["savings_segment"],
            y=ordered["avg_savings_rate"],
            marker_color="#197278",
        )
    )
    fig_segment_bars.add_trace(
        go.Bar(
            name="Goal met rate",
            x=ordered["savings_segment"],
            y=ordered["goal_met_rate"],
            marker_color="#f2a541",
        )
    )
    fig_segment_bars.add_trace(
        go.Bar(
            name="High stress rate",
            x=ordered["savings_segment"],
            y=ordered["high_stress_rate"],
            marker_color="#c44536",
        )
    )
    fig_segment_bars.update_layout(
        barmode="group",
        template=template,
        height=410,
        title="Segment Health Signals",
        margin=dict(l=20, r=20, t=60, b=90),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig_segment_bars.update_yaxes(tickformat=".0%")

    fig_monthly = go.Figure()
    fig_monthly.add_trace(
        go.Scatter(
            x=monthly_summary["month"],
            y=monthly_summary["avg_savings_rate"],
            mode="lines",
            line=dict(color="#197278", width=3),
            name="Savings rate",
        )
    )
    fig_monthly.add_trace(
        go.Scatter(
            x=monthly_summary["month"],
            y=monthly_summary["goal_met_rate"],
            mode="lines",
            line=dict(color="#f2a541", width=3),
            name="Goal met rate",
        )
    )
    fig_monthly.update_layout(
        template=template,
        height=390,
        title="Savings Discipline Over Time",
        margin=dict(l=20, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig_monthly.update_yaxes(tickformat=".0%")

    scenario_category = (
        df.groupby(["financial_scenario", "category"], as_index=False)
        .agg(avg_savings_rate=("savings_rate", "mean"))
    )
    scenario_category["financial_scenario"] = pd.Categorical(
        scenario_category["financial_scenario"], categories=SCENARIO_ORDER, ordered=True
    )
    heat = scenario_category.pivot(
        index="financial_scenario", columns="category", values="avg_savings_rate"
    ).sort_index()
    fig_heatmap = px.imshow(
        heat,
        color_continuous_scale=["#c44536", "#f2a541", "#197278"],
        aspect="auto",
        template=template,
        labels=dict(color="Savings rate", x="", y=""),
        title="Savings Rate by Category and Scenario",
    )
    fig_heatmap.update_layout(height=340, margin=dict(l=20, r=20, t=60, b=70))
    fig_heatmap.update_coloraxes(colorbar_tickformat=".0%")

    return {
        "category_spend": fig_category_spend,
        "goal": fig_goal,
        "segments": fig_segments,
        "segment_bars": fig_segment_bars,
        "monthly": fig_monthly,
        "heatmap": fig_heatmap,
    }


def write_outputs(
    df: pd.DataFrame,
    category_summary: pd.DataFrame,
    monthly_summary: pd.DataFrame,
    user_segments: pd.DataFrame,
    segment_summary: pd.DataFrame,
    segment_guidance: pd.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    df.to_csv(OUTPUT_DIR / "finance_dataset_enriched.csv", index=False)
    category_summary.to_csv(OUTPUT_DIR / "category_summary.csv", index=False)
    monthly_summary.to_csv(OUTPUT_DIR / "monthly_summary.csv", index=False)
    user_segments.to_csv(OUTPUT_DIR / "user_savings_segments.csv", index=False)
    segment_summary.to_csv(OUTPUT_DIR / "segment_summary.csv", index=False)
    segment_guidance.to_csv(OUTPUT_DIR / "segment_guidance.csv", index=False)


def top_insights(
    df: pd.DataFrame,
    category_summary: pd.DataFrame,
    segment_summary: pd.DataFrame,
) -> dict[str, str]:
    highest_expense = category_summary.iloc[0]
    best_goal = category_summary.sort_values("goal_met_rate", ascending=False).iloc[0]
    weakest_goal = category_summary.sort_values("goal_met_rate", ascending=True).iloc[0]
    largest_segment = segment_summary.sort_values("users", ascending=False).iloc[0]
    saver_segment = segment_summary.sort_values("avg_savings_rate", ascending=False).iloc[0]
    pressure_segment = segment_summary.sort_values("avg_debt_to_income", ascending=False).iloc[0]

    return {
        "period": f"{df['date'].min():%b %Y} to {df['date'].max():%b %Y}",
        "records": f"{len(df):,}",
        "users": f"{df['user_id'].nunique():,}",
        "avg_income": money(df["monthly_income"].mean()),
        "avg_expense": money(df["monthly_expense_total"].mean()),
        "avg_savings": money(df["actual_savings"].mean()),
        "avg_savings_rate": pct(df["savings_rate"].mean()),
        "goal_met_rate": pct(df["savings_goal_met"].mean()),
        "highest_expense_category": str(highest_expense["category"]),
        "highest_expense_value": money(highest_expense["avg_expense"]),
        "best_goal_category": str(best_goal["category"]),
        "best_goal_rate": pct(best_goal["goal_met_rate"]),
        "weakest_goal_category": str(weakest_goal["category"]),
        "weakest_goal_rate": pct(weakest_goal["goal_met_rate"]),
        "largest_segment": str(largest_segment["savings_segment"]),
        "largest_segment_share": pct(largest_segment["user_share"]),
        "saver_segment": str(saver_segment["savings_segment"]),
        "saver_segment_rate": pct(saver_segment["avg_savings_rate"]),
        "pressure_segment": str(pressure_segment["savings_segment"]),
        "pressure_segment_dti": pct(pressure_segment["avg_debt_to_income"]),
        "fraud_rate": pct(df["fraud_flag"].mean()),
    }


def render_dashboard(
    figures: dict[str, go.Figure],
    category_summary: pd.DataFrame,
    segment_summary: pd.DataFrame,
    segment_guidance: pd.DataFrame,
    insights: dict[str, str],
) -> None:
    chart_html = {
        name: fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})
        for name, fig in figures.items()
    }
    segment_rows = []
    for row in segment_summary.sort_values("avg_savings_rate", ascending=False).to_dict("records"):
        segment_rows.append(
            {
                "segment": row["savings_segment"],
                "users": f"{int(row['users']):,}",
                "share": pct(row["user_share"]),
                "savings_rate": pct(row["avg_savings_rate"]),
                "goal_met": pct(row["goal_met_rate"]),
                "dti": pct(row["avg_debt_to_income"]),
                "emergency": f"{row['avg_emergency_fund_months']:.2f} mo",
            }
        )

    category_rows = []
    for row in category_summary.sort_values("avg_expense", ascending=False).to_dict("records"):
        category_rows.append(
            {
                "category": row["category"],
                "avg_expense": money(row["avg_expense"]),
                "savings_rate": pct(row["avg_savings_rate"]),
                "goal_met": pct(row["goal_met_rate"]),
                "dti": pct(row["avg_debt_to_income"]),
                "stress": f"{row['avg_stress_score']:.2f}",
            }
        )

    guidance_rows = []
    for row in segment_guidance.to_dict("records"):
        guidance_rows.append(
            {
                "segment": row["savings_segment"],
                "users": f"{int(row['users']):,}",
                "share": pct(row["user_share"]),
                "meaning": row["meaning"],
                "standard": row["standard"],
                "data_standard": row["data_standard"],
                "bank_action": row["bank_action"],
                "advisor_action": row["advisor_action"],
            }
        )

    template = Template(
        dedent(
            """
            <!doctype html>
            <html lang="en">
            <head>
              <meta charset="utf-8">
              <meta name="viewport" content="width=device-width, initial-scale=1">
              <title>Personal Finance Tracker Story Dashboard</title>
              <script>{{ plotly_js | safe }}</script>
              <style>
                :root {
                  color-scheme: light;
                  --ink: #15222a;
                  --muted: #64727d;
                  --line: #d9e1e5;
                  --panel: #ffffff;
                  --paper: #f5f7f8;
                  --green: #197278;
                  --amber: #f2a541;
                  --red: #c44536;
                  --blue: #2f80ed;
                }
                * { box-sizing: border-box; }
                body {
                  margin: 0;
                  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                  color: var(--ink);
                  background: var(--paper);
                }
                header {
                  padding: 28px 32px 18px;
                  border-bottom: 1px solid var(--line);
                  background: #fbfcfc;
                }
                main { padding: 24px 32px 36px; }
                h1 {
                  margin: 0 0 8px;
                  font-size: clamp(28px, 4vw, 44px);
                  letter-spacing: 0;
                  line-height: 1.05;
                }
                h2 {
                  margin: 0 0 14px;
                  font-size: 20px;
                  letter-spacing: 0;
                }
                p { color: var(--muted); line-height: 1.55; margin: 0; }
                .subtitle { max-width: 980px; font-size: 16px; }
                .kpis {
                  display: grid;
                  grid-template-columns: repeat(4, minmax(160px, 1fr));
                  gap: 12px;
                  margin-top: 22px;
                }
                .kpi {
                  background: var(--panel);
                  border: 1px solid var(--line);
                  border-radius: 8px;
                  padding: 14px 16px;
                }
                .kpi span {
                  display: block;
                  color: var(--muted);
                  font-size: 12px;
                  text-transform: uppercase;
                  letter-spacing: .08em;
                }
                .kpi strong {
                  display: block;
                  margin-top: 8px;
                  font-size: 24px;
                  letter-spacing: 0;
                }
                .story {
                  display: grid;
                  grid-template-columns: 1.1fr .9fr;
                  gap: 18px;
                  margin-bottom: 22px;
                }
                .panel {
                  background: var(--panel);
                  border: 1px solid var(--line);
                  border-radius: 8px;
                  padding: 18px;
                }
                .chart-grid {
                  display: grid;
                  grid-template-columns: repeat(2, minmax(0, 1fr));
                  gap: 18px;
                }
                .wide { grid-column: 1 / -1; }
                .callouts {
                  display: grid;
                  gap: 12px;
                }
                .callout {
                  border-left: 4px solid var(--green);
                  padding-left: 12px;
                }
                .callout:nth-child(2) { border-left-color: var(--amber); }
                .callout:nth-child(3) { border-left-color: var(--red); }
                .callout strong { display: block; margin-bottom: 4px; }
                .explain-grid {
                  display: grid;
                  grid-template-columns: repeat(2, minmax(0, 1fr));
                  gap: 14px;
                  margin-bottom: 18px;
                }
                .segment-card {
                  background: var(--panel);
                  border: 1px solid var(--line);
                  border-radius: 8px;
                  padding: 16px;
                }
                .segment-card h3 {
                  margin: 0 0 8px;
                  font-size: 17px;
                  letter-spacing: 0;
                }
                .segment-card .meta {
                  display: flex;
                  flex-wrap: wrap;
                  gap: 8px;
                  margin: 0 0 12px;
                }
                .pill {
                  display: inline-flex;
                  align-items: center;
                  min-height: 26px;
                  border-radius: 999px;
                  border: 1px solid var(--line);
                  background: #f8fafb;
                  color: var(--muted);
                  font-size: 12px;
                  font-weight: 700;
                  padding: 4px 9px;
                  white-space: nowrap;
                }
                .segment-card .label {
                  display: block;
                  margin-top: 12px;
                  margin-bottom: 4px;
                  color: var(--ink);
                  font-size: 12px;
                  font-weight: 800;
                  text-transform: uppercase;
                  letter-spacing: .06em;
                }
                table {
                  width: 100%;
                  border-collapse: collapse;
                  font-size: 14px;
                }
                th, td {
                  padding: 10px 8px;
                  border-bottom: 1px solid var(--line);
                  text-align: right;
                  white-space: nowrap;
                }
                th:first-child, td:first-child { text-align: left; }
                th {
                  color: var(--muted);
                  font-weight: 700;
                  font-size: 12px;
                  text-transform: uppercase;
                  letter-spacing: .06em;
                }
                .tables {
                  display: grid;
                  grid-template-columns: 1fr 1fr;
                  gap: 18px;
                  margin-top: 18px;
                }
                .playbook-table th,
                .playbook-table td {
                  white-space: normal;
                  vertical-align: top;
                  text-align: left;
                  min-width: 180px;
                }
                footer {
                  margin-top: 18px;
                  color: var(--muted);
                  font-size: 13px;
                }
                @media (max-width: 980px) {
                  header, main { padding-left: 18px; padding-right: 18px; }
                  .kpis, .story, .chart-grid, .tables, .explain-grid { grid-template-columns: 1fr; }
                  .wide { grid-column: auto; }
                }
              </style>
            </head>
            <body>
              <header>
                <h1>Personal Finance Tracker</h1>
                <p class="subtitle">
                  A category and savings behavior view across {{ insights.records }} records, {{ insights.users }} users,
                  and the period {{ insights.period }}.
                </p>
                <section class="kpis" aria-label="Key metrics">
                  <div class="kpi"><span>Average income</span><strong>{{ insights.avg_income }}</strong></div>
                  <div class="kpi"><span>Average expense</span><strong>{{ insights.avg_expense }}</strong></div>
                  <div class="kpi"><span>Average savings</span><strong>{{ insights.avg_savings }}</strong></div>
                  <div class="kpi"><span>Goal met rate</span><strong>{{ insights.goal_met_rate }}</strong></div>
                </section>
              </header>
              <main>
                <section class="story">
                  <div class="panel">
                    <h2>Storyline</h2>
                    <p>
                      The data points to a savings challenge more than an income challenge. Average savings rate is
                      {{ insights.avg_savings_rate }}, but only {{ insights.goal_met_rate }} of records meet the savings goal.
                      {{ insights.highest_expense_category }} carries the highest average expense at
                      {{ insights.highest_expense_value }}, while {{ insights.best_goal_category }} shows the strongest goal
                      performance at {{ insights.best_goal_rate }}.
                    </p>
                  </div>
                  <div class="panel callouts">
                    <div class="callout">
                      <strong>Largest segment: {{ insights.largest_segment }}</strong>
                      <p>{{ insights.largest_segment_share }} of users land here, making it the main audience for financial guidance.</p>
                    </div>
                    <div class="callout">
                      <strong>Best savings behavior: {{ insights.saver_segment }}</strong>
                      <p>This group averages a {{ insights.saver_segment_rate }} savings rate and sets the benchmark.</p>
                    </div>
                    <div class="callout">
                      <strong>Highest debt pressure: {{ insights.pressure_segment }}</strong>
                      <p>Average debt-to-income reaches {{ insights.pressure_segment_dti }}, so cash-flow relief matters most.</p>
                    </div>
                  </div>
                </section>

                <section class="chart-grid">
                  <div class="panel">{{ charts.category_spend | safe }}</div>
                  <div class="panel">{{ charts.goal | safe }}</div>
                  <div class="panel wide">{{ charts.segments | safe }}</div>
                  <div class="panel">{{ charts.segment_bars | safe }}</div>
                  <div class="panel">{{ charts.monthly | safe }}</div>
                  <div class="panel wide">{{ charts.heatmap | safe }}</div>
                </section>

                <section class="panel" style="margin-top: 18px;">
                  <h2>Segment Meaning and Standards</h2>
                  <p>
                    The segment names are readable labels for KMeans clusters. Each standard below combines the
                    business interpretation with the current data profile for that segment.
                  </p>
                </section>
                <section class="explain-grid">
                  {% for row in guidance_rows %}
                  <article class="segment-card">
                    <h3>{{ row.segment }}</h3>
                    <div class="meta">
                      <span class="pill">{{ row.users }} users</span>
                      <span class="pill">{{ row.share }} share</span>
                    </div>
                    <p>{{ row.meaning }}</p>
                    <span class="label">Standard</span>
                    <p>{{ row.standard }}</p>
                    <span class="label">Current data profile</span>
                    <p>{{ row.data_standard }}</p>
                  </article>
                  {% endfor %}
                </section>

                <section class="panel" style="margin-top: 18px;">
                  <h2>Bank and Advisor Playbook</h2>
                  <table class="playbook-table">
                    <thead>
                      <tr>
                        <th>Segment</th><th>What banks can do</th><th>What advisors can do</th>
                      </tr>
                    </thead>
                    <tbody>
                      {% for row in guidance_rows %}
                      <tr>
                        <td>{{ row.segment }}</td><td>{{ row.bank_action }}</td><td>{{ row.advisor_action }}</td>
                      </tr>
                      {% endfor %}
                    </tbody>
                  </table>
                </section>

                <section class="tables">
                  <div class="panel">
                    <h2>Segment Summary</h2>
                    <table>
                      <thead>
                        <tr>
                          <th>Segment</th><th>Users</th><th>Share</th><th>Savings</th><th>Goal met</th><th>DTI</th><th>Emergency</th>
                        </tr>
                      </thead>
                      <tbody>
                        {% for row in segment_rows %}
                        <tr>
                          <td>{{ row.segment }}</td><td>{{ row.users }}</td><td>{{ row.share }}</td><td>{{ row.savings_rate }}</td>
                          <td>{{ row.goal_met }}</td><td>{{ row.dti }}</td><td>{{ row.emergency }}</td>
                        </tr>
                        {% endfor %}
                      </tbody>
                    </table>
                  </div>
                  <div class="panel">
                    <h2>Category Pattern Summary</h2>
                    <table>
                      <thead>
                        <tr>
                          <th>Category</th><th>Avg expense</th><th>Savings</th><th>Goal met</th><th>DTI</th><th>Stress</th>
                        </tr>
                      </thead>
                      <tbody>
                        {% for row in category_rows %}
                        <tr>
                          <td>{{ row.category }}</td><td>{{ row.avg_expense }}</td><td>{{ row.savings_rate }}</td>
                          <td>{{ row.goal_met }}</td><td>{{ row.dti }}</td><td>{{ row.stress }}</td>
                        </tr>
                        {% endfor %}
                      </tbody>
                    </table>
                  </div>
                </section>
                <footer>
                  Generated from personal_finance_tracker_dataset.csv. Segment labels are derived from KMeans clusters
                  interpreted by savings strength, goal achievement, emergency coverage, investment rate, debt pressure,
                  and expense-to-income ratio.
                </footer>
              </main>
            </body>
            </html>
            """
        )
    )
    DASHBOARD_FILE.write_text(
        template.render(
            charts=chart_html,
            insights=insights,
            segment_rows=segment_rows,
            category_rows=category_rows,
            guidance_rows=guidance_rows,
            plotly_js=get_plotlyjs(),
        ),
        encoding="utf-8",
    )


def make_notebook() -> None:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    cells = [
        nbf.v4.new_markdown_cell(
            dedent(
                """
                # Personal Finance Tracker Analysis

                This notebook analyzes category-level spending patterns and segments users by savings behavior.
                The same data model is used by `finance_story_dashboard.html`.
                """
            ).strip()
        ),
        nbf.v4.new_code_cell(
            dedent(
                """
                from pathlib import Path
                import numpy as np
                import pandas as pd
                import plotly.express as px
                import plotly.graph_objects as go
                from sklearn.cluster import KMeans
                from sklearn.preprocessing import StandardScaler

                pd.set_option("display.max_columns", 50)
                DATA_FILE = Path("personal_finance_tracker_dataset.csv")
                OUTPUT_DIR = Path("analysis_outputs")
                OUTPUT_DIR.mkdir(exist_ok=True)
                """
            ).strip()
        ),
        nbf.v4.new_markdown_cell("## 1. Load and Inspect the Dataset"),
        nbf.v4.new_code_cell(
            dedent(
                """
                df = pd.read_csv(DATA_FILE, parse_dates=["date"])
                df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
                print(f"Rows: {len(df):,}")
                print(f"Users: {df['user_id'].nunique():,}")
                print(f"Period: {df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d}")
                display(df.head())
                display(df.isna().sum().to_frame("missing_values"))
                """
            ).strip()
        ),
        nbf.v4.new_markdown_cell("## 2. Feature Engineering"),
        nbf.v4.new_code_cell(
            dedent(
                """
                stress_map = {"Low": 1, "Medium": 2, "High": 3}

                df["expense_to_income"] = df["monthly_expense_total"] / df["monthly_income"]
                df["discretionary_share"] = df["discretionary_spending"] / df["monthly_expense_total"]
                df["essential_share"] = df["essential_spending"] / df["monthly_expense_total"]
                df["investment_rate"] = df["investment_amount"] / df["monthly_income"]
                df["emergency_fund_months"] = df["emergency_fund"] / df["monthly_expense_total"]
                df["stress_score"] = df["financial_stress_level"].map(stress_map)

                df.replace([np.inf, -np.inf], np.nan, inplace=True)
                ratio_cols = [
                    "expense_to_income", "discretionary_share", "essential_share",
                    "investment_rate", "emergency_fund_months"
                ]
                df[ratio_cols] = df[ratio_cols].fillna(0)
                display(df[ratio_cols + ["stress_score"]].describe().T)
                """
            ).strip()
        ),
        nbf.v4.new_markdown_cell("## 3. Pattern Analysis by Category"),
        nbf.v4.new_code_cell(
            dedent(
                """
                category_summary = (
                    df.groupby("category", as_index=False)
                    .agg(
                        records=("category", "size"),
                        avg_income=("monthly_income", "mean"),
                        avg_expense=("monthly_expense_total", "mean"),
                        avg_actual_savings=("actual_savings", "mean"),
                        avg_savings_rate=("savings_rate", "mean"),
                        goal_met_rate=("savings_goal_met", "mean"),
                        avg_debt_to_income=("debt_to_income_ratio", "mean"),
                        avg_discretionary_share=("discretionary_share", "mean"),
                        avg_essential_share=("essential_share", "mean"),
                        avg_stress_score=("stress_score", "mean"),
                        fraud_rate=("fraud_flag", "mean"),
                    )
                    .sort_values("avg_expense", ascending=False)
                )
                category_summary["record_share"] = category_summary["records"] / category_summary["records"].sum()
                category_summary.to_csv(OUTPUT_DIR / "category_summary.csv", index=False)

                display(category_summary.style.format({
                    "avg_income": "${:,.0f}",
                    "avg_expense": "${:,.0f}",
                    "avg_actual_savings": "${:,.0f}",
                    "avg_savings_rate": "{:.1%}",
                    "goal_met_rate": "{:.1%}",
                    "avg_debt_to_income": "{:.1%}",
                    "avg_discretionary_share": "{:.1%}",
                    "avg_essential_share": "{:.1%}",
                    "fraud_rate": "{:.1%}",
                    "record_share": "{:.1%}",
                }))
                """
            ).strip()
        ),
        nbf.v4.new_code_cell(
            dedent(
                """
                fig = px.bar(
                    category_summary.sort_values("avg_expense"),
                    x="avg_expense",
                    y="category",
                    color="avg_savings_rate",
                    color_continuous_scale=["#f5b7a7", "#f6d365", "#82c0cc"],
                    labels={
                        "avg_expense": "Average monthly expense",
                        "category": "",
                        "avg_savings_rate": "Savings rate",
                    },
                    title="Average Monthly Expense by Category"
                )
                fig.update_xaxes(tickprefix="$", separatethousands=True)
                fig.show()
                """
            ).strip()
        ),
        nbf.v4.new_code_cell(
            dedent(
                """
                fig = px.bar(
                    category_summary.sort_values("goal_met_rate", ascending=False),
                    x="category",
                    y="goal_met_rate",
                    color="avg_debt_to_income",
                    color_continuous_scale=["#7fc97f", "#fdc086", "#beaed4"],
                    labels={
                        "category": "",
                        "goal_met_rate": "Savings goal met rate",
                        "avg_debt_to_income": "Debt-to-income",
                    },
                    title="Savings Goal Achievement by Category"
                )
                fig.update_yaxes(tickformat=".0%")
                fig.show()
                """
            ).strip()
        ),
        nbf.v4.new_markdown_cell("## 4. Savings Behavior Segmentation"),
        nbf.v4.new_code_cell(
            dedent(
                """
                user = (
                    df.groupby("user_id", as_index=False)
                    .agg(
                        observations=("user_id", "size"),
                        avg_income=("monthly_income", "mean"),
                        avg_expense=("monthly_expense_total", "mean"),
                        avg_savings_rate=("savings_rate", "mean"),
                        avg_actual_savings=("actual_savings", "mean"),
                        goal_met_rate=("savings_goal_met", "mean"),
                        avg_debt_to_income=("debt_to_income_ratio", "mean"),
                        avg_loan_payment=("loan_payment", "mean"),
                        avg_investment_rate=("investment_rate", "mean"),
                        avg_emergency_fund_months=("emergency_fund_months", "mean"),
                        avg_discretionary_share=("discretionary_share", "mean"),
                        avg_expense_to_income=("expense_to_income", "mean"),
                        high_stress_rate=("financial_stress_level", lambda s: (s == "High").mean()),
                    )
                )

                segment_features = [
                    "avg_income", "avg_expense", "avg_savings_rate", "avg_actual_savings",
                    "goal_met_rate", "avg_debt_to_income", "avg_loan_payment",
                    "avg_investment_rate", "avg_emergency_fund_months",
                    "avg_discretionary_share", "high_stress_rate", "avg_expense_to_income"
                ]
                scaled = StandardScaler().fit_transform(user[segment_features])
                user["cluster"] = KMeans(n_clusters=4, n_init=25, random_state=42).fit_predict(scaled)
                """
            ).strip()
        ),
        nbf.v4.new_code_cell(
            dedent(
                """
                cluster_summary = (
                    user.groupby("cluster", as_index=False)
                    .agg(
                        users=("user_id", "size"),
                        avg_income=("avg_income", "mean"),
                        avg_expense=("avg_expense", "mean"),
                        avg_savings_rate=("avg_savings_rate", "mean"),
                        avg_actual_savings=("avg_actual_savings", "mean"),
                        goal_met_rate=("goal_met_rate", "mean"),
                        avg_debt_to_income=("avg_debt_to_income", "mean"),
                        avg_investment_rate=("avg_investment_rate", "mean"),
                        avg_emergency_fund_months=("avg_emergency_fund_months", "mean"),
                        avg_discretionary_share=("avg_discretionary_share", "mean"),
                        high_stress_rate=("high_stress_rate", "mean"),
                        avg_expense_to_income=("avg_expense_to_income", "mean"),
                    )
                )

                metrics = [
                    "avg_savings_rate", "goal_met_rate", "avg_emergency_fund_months",
                    "avg_investment_rate", "avg_debt_to_income", "avg_expense_to_income"
                ]
                z = cluster_summary[metrics].apply(lambda s: (s - s.mean()) / (s.std(ddof=0) or 1))
                cluster_summary["strength_score"] = (
                    z["avg_savings_rate"] + z["goal_met_rate"] + z["avg_emergency_fund_months"]
                    + z["avg_investment_rate"] - z["avg_debt_to_income"] - z["avg_expense_to_income"]
                )

                labels = {}
                available = set(cluster_summary["cluster"])
                strongest = int(cluster_summary.loc[cluster_summary["strength_score"].idxmax(), "cluster"])
                labels[strongest] = "Resilient investors"
                available.remove(strongest)
                goal_cluster = int(cluster_summary[cluster_summary["cluster"].isin(available)].sort_values(
                    ["goal_met_rate", "avg_savings_rate"], ascending=False
                ).iloc[0]["cluster"])
                labels[goal_cluster] = "Goal-focused savers"
                available.remove(goal_cluster)
                debt_cluster = int(cluster_summary[cluster_summary["cluster"].isin(available)].sort_values(
                    ["avg_debt_to_income", "avg_expense_to_income"], ascending=False
                ).iloc[0]["cluster"])
                labels[debt_cluster] = "Debt-pressure spenders"
                available.remove(debt_cluster)
                for cluster in available:
                    labels[int(cluster)] = "Cash-flow stretched"

                user["savings_segment"] = user["cluster"].map(labels)
                segment_summary = (
                    user.groupby("savings_segment", as_index=False)
                    .agg(
                        users=("user_id", "size"),
                        avg_income=("avg_income", "mean"),
                        avg_expense=("avg_expense", "mean"),
                        avg_savings_rate=("avg_savings_rate", "mean"),
                        avg_actual_savings=("avg_actual_savings", "mean"),
                        goal_met_rate=("goal_met_rate", "mean"),
                        avg_debt_to_income=("avg_debt_to_income", "mean"),
                        avg_investment_rate=("avg_investment_rate", "mean"),
                        avg_emergency_fund_months=("avg_emergency_fund_months", "mean"),
                        avg_discretionary_share=("avg_discretionary_share", "mean"),
                        high_stress_rate=("high_stress_rate", "mean"),
                        avg_expense_to_income=("avg_expense_to_income", "mean"),
                    )
                    .sort_values("users", ascending=False)
                )
                segment_summary["user_share"] = segment_summary["users"] / segment_summary["users"].sum()

                user.to_csv(OUTPUT_DIR / "user_savings_segments.csv", index=False)
                segment_summary.to_csv(OUTPUT_DIR / "segment_summary.csv", index=False)
                display(segment_summary.style.format({
                    "avg_income": "${:,.0f}",
                    "avg_expense": "${:,.0f}",
                    "avg_actual_savings": "${:,.0f}",
                    "avg_savings_rate": "{:.1%}",
                    "goal_met_rate": "{:.1%}",
                    "avg_debt_to_income": "{:.1%}",
                    "avg_investment_rate": "{:.1%}",
                    "avg_emergency_fund_months": "{:.2f}",
                    "avg_discretionary_share": "{:.1%}",
                    "high_stress_rate": "{:.1%}",
                    "avg_expense_to_income": "{:.1%}",
                    "user_share": "{:.1%}",
                }))
                """
            ).strip()
        ),
        nbf.v4.new_markdown_cell(
            dedent(
                """
                ### Segment Meaning, Standards, and Recommended Actions

                Each segment name is an interpretation of a KMeans cluster. The standard combines a practical benchmark
                for how the segment should be managed with the actual data profile observed in this dataset.
                """
            ).strip()
        ),
        nbf.v4.new_code_cell(
            dedent(
                """
                segment_profiles = {
                    "Resilient investors": {
                        "meaning": "Strongest overall savings profile, with healthier savings, emergency coverage, and manageable debt pressure.",
                        "standard": "Benchmark segment: keep savings consistent, preserve liquidity, and deepen investment planning without adding avoidable debt.",
                        "bank_action": "Offer high-yield savings, automated investing, portfolio review, and pre-approved credit only when repayment capacity remains healthy.",
                        "advisor_action": "Focus on wealth building: asset allocation, tax-aware investing, insurance coverage, and retirement milestones.",
                    },
                    "Goal-focused savers": {
                        "meaning": "Most likely to hit savings goals, showing target-based discipline even when average savings rate is not the highest.",
                        "standard": "Goal-achievement standard: protect the habit with automatic transfers, target dates, and alerts when spending threatens the goal.",
                        "bank_action": "Provide goal pockets, round-up savings, bonus rates for streaks, and personalized goal-progress nudges.",
                        "advisor_action": "Translate goals into a plan: emergency fund target, debt payoff schedule, investment contribution ladder, and review cadence.",
                    },
                    "Debt-pressure spenders": {
                        "meaning": "Highest debt burden and lower goal achievement. Cash flow exists, but debt payments and expenses limit progress.",
                        "standard": "Risk-reduction standard: bring debt-to-income down first, then rebuild savings goals after repayment pressure eases.",
                        "bank_action": "Prioritize refinance checks, debt consolidation options, payment reminders, spending limits, and restructuring support where suitable.",
                        "advisor_action": "Build a debt-first plan: rank balances by rate, cap discretionary leakage, protect minimum emergency cash, and set payoff milestones.",
                    },
                    "Cash-flow stretched": {
                        "meaning": "Expenses are close to or above income, leaving little room for actual savings despite moderate savings-rate signals.",
                        "standard": "Stabilization standard: create positive monthly cash flow before pushing aggressive investment or long-term saving targets.",
                        "bank_action": "Use low-balance alerts, bill timing tools, subscription review, overdraft prevention, and small emergency savings automation.",
                        "advisor_action": "Start with cash-flow repair: separate essential and discretionary spending, reset the budget, renegotiate recurring costs, and define a starter emergency fund.",
                    },
                }

                guidance_rows = []
                summary_lookup = {
                    row["savings_segment"]: row for row in segment_summary.to_dict("records")
                }
                for segment_name, profile in segment_profiles.items():
                    if segment_name not in summary_lookup:
                        continue
                    row = summary_lookup[segment_name]
                    guidance_rows.append({
                        "savings_segment": segment_name,
                        "users": int(row["users"]),
                        "user_share": row["user_share"],
                        "meaning": profile["meaning"],
                        "standard": profile["standard"],
                        "data_standard": (
                            f"{row['avg_savings_rate']:.1%} avg savings rate; "
                            f"{row['goal_met_rate']:.1%} goal-met rate; "
                            f"{row['avg_debt_to_income']:.1%} debt-to-income; "
                            f"{row['avg_emergency_fund_months']:.2f} months emergency coverage."
                        ),
                        "bank_action": profile["bank_action"],
                        "advisor_action": profile["advisor_action"],
                    })

                segment_guidance = pd.DataFrame(guidance_rows)
                segment_guidance.to_csv(OUTPUT_DIR / "segment_guidance.csv", index=False)
                display(segment_guidance[[
                    "savings_segment", "users", "user_share", "meaning", "standard", "data_standard"
                ]].style.format({"user_share": "{:.1%}"}))
                """
            ).strip()
        ),
        nbf.v4.new_markdown_cell("### What Banks and Personal Finance Advisors Can Do"),
        nbf.v4.new_code_cell(
            dedent(
                """
                display(segment_guidance[[
                    "savings_segment", "bank_action", "advisor_action"
                ]])
                """
            ).strip()
        ),
        nbf.v4.new_code_cell(
            dedent(
                """
                colors = {
                    "Resilient investors": "#197278",
                    "Goal-focused savers": "#2f80ed",
                    "Debt-pressure spenders": "#c44536",
                    "Cash-flow stretched": "#7b61ff",
                }
                fig = px.scatter(
                    user,
                    x="avg_debt_to_income",
                    y="avg_savings_rate",
                    size="avg_income",
                    color="savings_segment",
                    color_discrete_map=colors,
                    hover_data={
                        "user_id": True,
                        "avg_income": ":$,.0f",
                        "avg_expense": ":$,.0f",
                        "goal_met_rate": ":.0%",
                        "avg_emergency_fund_months": ":.2f",
                    },
                    labels={
                        "avg_debt_to_income": "Average debt-to-income",
                        "avg_savings_rate": "Average savings rate",
                        "savings_segment": "Segment",
                    },
                    title="User Savings Behavior Segments"
                )
                fig.update_xaxes(tickformat=".0%")
                fig.update_yaxes(tickformat=".0%")
                fig.show()
                """
            ).strip()
        ),
        nbf.v4.new_markdown_cell("## 5. Trend and Scenario Context"),
        nbf.v4.new_code_cell(
            dedent(
                """
                monthly_summary = (
                    df.groupby("month", as_index=False)
                    .agg(
                        avg_income=("monthly_income", "mean"),
                        avg_expense=("monthly_expense_total", "mean"),
                        avg_actual_savings=("actual_savings", "mean"),
                        avg_savings_rate=("savings_rate", "mean"),
                        goal_met_rate=("savings_goal_met", "mean"),
                        avg_debt_to_income=("debt_to_income_ratio", "mean"),
                    )
                    .sort_values("month")
                )
                monthly_summary.to_csv(OUTPUT_DIR / "monthly_summary.csv", index=False)

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=monthly_summary["month"], y=monthly_summary["avg_savings_rate"],
                    mode="lines", name="Savings rate", line=dict(color="#197278", width=3)
                ))
                fig.add_trace(go.Scatter(
                    x=monthly_summary["month"], y=monthly_summary["goal_met_rate"],
                    mode="lines", name="Goal met rate", line=dict(color="#f2a541", width=3)
                ))
                fig.update_layout(title="Savings Discipline Over Time", legend=dict(orientation="h"))
                fig.update_yaxes(tickformat=".0%")
                fig.show()
                """
            ).strip()
        ),
        nbf.v4.new_code_cell(
            dedent(
                """
                scenario_category = (
                    df.groupby(["financial_scenario", "category"], as_index=False)
                    .agg(avg_savings_rate=("savings_rate", "mean"))
                )
                heat = scenario_category.pivot(
                    index="financial_scenario", columns="category", values="avg_savings_rate"
                )
                fig = px.imshow(
                    heat,
                    color_continuous_scale=["#c44536", "#f2a541", "#197278"],
                    aspect="auto",
                    labels=dict(color="Savings rate", x="", y=""),
                    title="Savings Rate by Category and Financial Scenario"
                )
                fig.update_coloraxes(colorbar_tickformat=".0%")
                fig.show()
                """
            ).strip()
        ),
        nbf.v4.new_markdown_cell("## 6. Executive Takeaways"),
        nbf.v4.new_code_cell(
            dedent(
                """
                highest_expense = category_summary.iloc[0]
                best_goal = category_summary.sort_values("goal_met_rate", ascending=False).iloc[0]
                weakest_goal = category_summary.sort_values("goal_met_rate").iloc[0]
                largest_segment = segment_summary.sort_values("users", ascending=False).iloc[0]
                strongest_segment = segment_summary.sort_values("avg_savings_rate", ascending=False).iloc[0]
                debt_segment = segment_summary.sort_values("avg_debt_to_income", ascending=False).iloc[0]

                takeaways = [
                    f"{highest_expense['category']} has the highest average monthly expense at ${highest_expense['avg_expense']:,.0f}.",
                    f"{best_goal['category']} has the strongest savings goal performance at {best_goal['goal_met_rate']:.1%}; {weakest_goal['category']} is lowest at {weakest_goal['goal_met_rate']:.1%}.",
                    f"The largest savings segment is {largest_segment['savings_segment']}, covering {largest_segment['user_share']:.1%} of users.",
                    f"{strongest_segment['savings_segment']} is the benchmark segment with a {strongest_segment['avg_savings_rate']:.1%} average savings rate.",
                    f"{debt_segment['savings_segment']} carries the highest average debt-to-income ratio at {debt_segment['avg_debt_to_income']:.1%}.",
                ]
                for item in takeaways:
                    print("-", item)
                """
            ).strip()
        ),
    ]
    nb["cells"] = cells
    nbf.write(nb, NOTEBOOK_FILE)
    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    client.execute()
    nbf.write(nb, NOTEBOOK_FILE)


def main() -> None:
    df = add_features(load_data())
    category_summary = make_category_summary(df)
    monthly_summary = make_monthly_summary(df)
    user_segments, segment_summary = make_user_segments(df)
    segment_guidance = make_segment_guidance(segment_summary)
    write_outputs(df, category_summary, monthly_summary, user_segments, segment_summary, segment_guidance)
    figures = make_figures(df, category_summary, monthly_summary, user_segments, segment_summary)
    insights = top_insights(df, category_summary, segment_summary)
    render_dashboard(figures, category_summary, segment_summary, segment_guidance, insights)
    make_notebook()
    print(json.dumps({
        "notebook": str(NOTEBOOK_FILE),
        "dashboard": str(DASHBOARD_FILE),
        "outputs": str(OUTPUT_DIR),
        "insights": insights,
    }, indent=2))


if __name__ == "__main__":
    main()
