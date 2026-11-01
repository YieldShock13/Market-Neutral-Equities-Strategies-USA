
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(
    page_title="Meridian | Model X", page_icon="◉",
    layout="wide"
)

ROOT = Path(__file__).resolve().parent
PIPE = ROOT / "pipeline"

@st.cache_data(ttl=300)
def load_data():
    signals = pd.read_csv(
        PIPE / "signals_v3/basket_signals.csv"
    )
    audit = pd.read_csv(
        PIPE / "final_audit/basket_audit.csv"
    )
    positions = pd.read_csv(
        PIPE / "final_audit/dollar_positions.csv"
    )

    data = signals.merge(
        audit[
            [
                "Horizon",
                "BasketID",
                "TrainSpecificationRobust",
                "TrainRankDet0",
                "TrainRankDetMinus1"
            ]
        ],
        on=["Horizon", "BasketID"],
        how="left"
    )

    return data, positions

st.markdown("""
<style>
.block-container {max-width:1480px;padding-top:2.2rem;}
[data-testid="stSidebar"] {border-right:1px solid #29404b;}
[data-testid="stMetric"] {background:#14212b;border:1px solid #29404b;border-radius:12px;padding:18px;}
[data-testid="stMetricLabel"] {text-transform:uppercase;letter-spacing:.12em;font-size:.7rem;}
h1,h2,h3 {font-weight:400 !important;letter-spacing:.025em;}
.meridian-header {position:relative;overflow:hidden;border:1px solid #36505c;background:linear-gradient(115deg,#172c37,#0c141b);padding:36px 40px;border-radius:16px;margin-bottom:24px;}
.meridian-header:after {content:"";position:absolute;width:330px;height:330px;right:60px;top:-110px;border:1px solid #bda37166;border-radius:50%;box-shadow:0 0 0 35px #bda37109,0 0 0 70px #bda37108,0 0 0 105px #bda37106;pointer-events:none;}
.brand {font-family:Georgia,serif;font-size:clamp(36px,6vw,66px);letter-spacing:.18em;color:#e9edf0;line-height:1.2;}
.model {color:#bda371;font-size:15px;letter-spacing:.32em;text-transform:uppercase;margin-top:12px;}
.descriptor {color:#a5b7c0;margin-top:18px;font-size:14px;}
</style>
<div class="meridian-header"><div class="brand">MERIDIAN</div><div class="model">Model X</div><div class="descriptor">Market-neutral equities · Relative value research</div></div>
""", unsafe_allow_html=True)
import json
status_path = PIPE / "refresh_status.json"
if status_path.exists():
    refresh = json.loads(status_path.read_text())
    st.caption(f"Price date: {refresh['price_date']} · Updated: {refresh['completed_at']} · {refresh['eligible_issuers']:,} issuers")
else:
    st.info("Historical snapshot · First automated refresh pending")

try:
    data, positions = load_data()
except Exception as exc:
    st.error(f"Unable to load research files: {exc}")
    st.stop()

data = data[data["Status"] == "SIGNAL_COMPUTED"].copy()
# Legacy baskets must be recalculated, never relabelled as equivalent exposures.
import re
legacy_duplicate = data["Basket"].astype(str).map(lambda value: bool(re.search(r"(?<![A-Z])ZG(?![A-Z])", value)))
if legacy_duplicate.any():
    st.caption(f"{int(legacy_duplicate.sum())} legacy baskets excluded pending issuer-deduplicated refresh.")
data = data.loc[~legacy_duplicate].copy()

data["Stability"] = np.where(
    data["TrainSpecificationRobust"].fillna(False),
    "PASS",
    "FLAGGED"
)

with st.sidebar:
    st.markdown("### Meridian")
    st.caption("MODEL X / UNIVERSE FILTERS")

    horizons = st.multiselect(
        "Estimation horizon",
        ["1Y", "3Y", "5Y", "FULL"],
        default=["1Y", "3Y", "5Y", "FULL"]
    )

    stability = st.multiselect(
        "Historical stability",
        ["PASS", "FLAGGED"],
        default=["PASS", "FLAGGED"]
    )

    minimum = st.slider(
        "Minimum mispricing",
        min_value=0.0,
        max_value=5.0,
        value=0.0,
        step=0.1
    )

filtered = data[
    data["Horizon"].isin(horizons)
    & data["Stability"].isin(stability)
    & (data["JointMispricing"] >= minimum)
].sort_values(
    "JointMispricing", ascending=False
)

c1, c2, c3 = st.columns(3)

c1.metric("Baskets", len(filtered))
c2.metric(
    "Stability passes",
    int((filtered["Stability"] == "PASS").sum())
)
c3.metric(
    "Highest mispricing",
    f"{filtered['JointMispricing'].max():.2f}"
    if len(filtered) else "—"
)

st.subheader("Opportunity Monitor")

columns = [
    "Horizon",
    "Basket",
    "JointMispricing",
    "Stability",
    "PortfolioBeta",
    "MarketExplainedVariancePct",
    "NetDollarExposurePct"
]

st.dataframe(
    filtered[columns].rename(columns={
        "JointMispricing": "Absolute Z",
        "PortfolioBeta": "Portfolio Beta",
        "MarketExplainedVariancePct": "Market R² (%)",
        "NetDollarExposurePct": "Net Exposure (%)"
    }),
    hide_index=True,
    use_container_width=True
)

if filtered.empty:
    st.info("No baskets match the selected filters.")
    st.stop()



st.subheader("Basket Analytics")

choices = {
    f"{r.Horizon} | {r.Basket} | {r.JointMispricing:.2f}σ":
    (str(r.Horizon), str(r.BasketID))
    for r in filtered.itertuples()
}

selected = st.selectbox("Select basket", list(choices))
horizon, bid = choices[selected]

basket = filtered[
    filtered["Horizon"].astype(str).eq(horizon)
    & filtered["BasketID"].astype(str).eq(bid)
].iloc[0]

@st.cache_data(ttl=300)
def analytics():
    base = PIPE / "dashboard_analytics"
    files = {
        "history": "spread_history.csv",
        "rolling": "rolling_diagnostics.csv",
        "matrix": "transition_matrix.csv",
        "modes": "eigenmodes.csv",
        "summary": "analytics_summary.csv"
    }
    result = {}
    for key, filename in files.items():
        frame = pd.read_csv(base / filename)
        frame["Horizon"] = frame["Horizon"].astype(str)
        frame["BasketID"] = frame["BasketID"].astype(str)
        if "Date" in frame:
            frame["Date"] = pd.to_datetime(frame["Date"])
        result[key] = frame
    return result

try:
    datasets = analytics()
except Exception as exc:
    st.error(f"Historical analytics unavailable: {exc}")
    st.stop()

def subset(name):
    df = datasets[name]
    return df[
        df["Horizon"].eq(horizon)
        & df["BasketID"].eq(bid)
    ].copy()

hist = subset("history")
rolling = subset("rolling")
matrix = subset("matrix")
modes = subset("modes")
summary = subset("summary")

if hist.empty:
    st.warning("No historical analytics for this basket.")
    st.stop()

hist = hist.sort_values(["Component", "Date"])
rolling = rolling.sort_values(["Component", "Date"])

import plotly.graph_objects as go
import plotly.express as px

COLORS = {
    "RELIABLE": "rgba(28,160,95,0.16)",
    "WATCH": "rgba(240,173,45,0.18)",
    "FLAGGED": "rgba(220,65,65,0.17)"
}

st.caption(
    "Historical reliability colours are diagnostic classifications "
    "from trailing windows, not verified market regimes."
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Current mispricing", f"{basket.JointMispricing:.3f}σ")
m2.metric("Historical audit", basket.Stability)
m3.metric("Market beta", f"{basket.PortfolioBeta:.3f}")
m4.metric("Net dollar exposure", f"{basket.NetDollarExposurePct:.2f}%")

components = sorted(hist["Component"].unique().tolist())
component = st.selectbox("Stationary component", components)

h = hist[hist["Component"].eq(component)].copy()
r = rolling[rolling["Component"].eq(component)].copy()

def shade_reliability(fig, rolling_frame, first_date):
    """Each colour represents the most recent completed rolling diagnostic."""
    rr = rolling_frame.sort_values("Date").reset_index(drop=True)
    if rr.empty:
        return
    for i, record in rr.iterrows():
        left = record["Date"]
        right = (
            rr.iloc[i+1]["Date"]
            if i+1 < len(rr)
            else h["Date"].max()
        )
        if right > left:
            fig.add_vrect(
                x0=left, x1=right,
                fillcolor=COLORS.get(record["Reliability"], "rgba(0,0,0,0)"),
                opacity=1,
                line_width=0,
                layer="below"
            )

tab1, tab2, tab3, tab4 = st.tabs([
    "Spread & Mispricing",
    "Reliability & Half-life",
    "MVOU Matrix & Convergence",
    "Portfolio Exposures"
])

with tab1:
    st.markdown("#### Historical cointegrating spread")

    fig = go.Figure()
    shade_reliability(fig, r, h["Date"].min())

    h = h.copy()
    h["Upper1"] = h["Equilibrium"] + h["SpreadSD"]
    h["Lower1"] = h["Equilibrium"] - h["SpreadSD"]
    h["Upper2"] = h["Equilibrium"] + 2*h["SpreadSD"]
    h["Lower2"] = h["Equilibrium"] - 2*h["SpreadSD"]
    h["Upper3"] = h["Equilibrium"] + 3*h["SpreadSD"]
    h["Lower3"] = h["Equilibrium"] - 3*h["SpreadSD"]

    fig.add_trace(go.Scatter(
        x=h["Date"], y=h["Spread"],
        name="Observed spread",
        line=dict(color="#3378d4", width=2)
    ))
    fig.add_trace(go.Scatter(
        x=h["Date"], y=h["Equilibrium"],
        name="Fitted equilibrium",
        line=dict(color="#222222", dash="dash")
    ))

    for n, dash in [(1, "dot"), (2, "dash"), (3, "longdash")]:
        for sign, side in [("+", "Upper"), ("-", "Lower")]:
            fig.add_trace(go.Scatter(
                x=h["Date"], y=h[f"{side}{n}"],
                name=f"{sign}{n}σ",
                line=dict(color="#999999", dash=dash, width=1),
                legendgroup=f"band{n}",
                showlegend=(sign == "+")
            ))

    fig.update_layout(
        height=480,
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title="Log-price cointegrating spread",
        legend=dict(orientation="h")
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Historical standardized deviation")

    zfig = go.Figure()
    shade_reliability(zfig, r, h["Date"].min())
    zfig.add_trace(go.Scatter(
        x=h["Date"], y=h["ZScore"],
        name="Spread z-score",
        line=dict(color="#3378d4")
    ))

    for level in [-3, -2, -1, 0, 1, 2, 3]:
        zfig.add_hline(
            y=level,
            line_dash="solid" if level == 0 else "dot",
            line_color="#888888",
            line_width=1
        )

    zfig.update_layout(
        height=380,
        yaxis_title="Standard deviations",
        hovermode="x unified"
    )
    st.plotly_chart(zfig, use_container_width=True)

    st.metric("Latest component z-score", f"{h['ZScore'].iloc[-1]:+.3f}")
    st.caption(
        "For a one-component model, absolute z-score equals joint "
        "Mahalanobis mispricing. The sign depends on the stored "
        "cointegrating-vector orientation."
    )

with tab2:
    st.markdown("#### Historical reliability classification")

    if r.empty:
        st.info("No rolling diagnostics available.")
    else:
        counts = r["Reliability"].value_counts()
        st.dataframe(
            pd.DataFrame({
                "Classification": counts.index,
                "Completed windows": counts.values
            }),
            hide_index=True
        )

        st.markdown("#### Rolling mean-reversion half-life")
        fig = px.line(
            r, x="Date", y="HalfLifeDays",
            markers=True,
            labels={"HalfLifeDays": "Trading days"}
        )
        shade_reliability(fig, r, h["Date"].min())
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0c141b", plot_bgcolor="#0c141b", height=360)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Equilibrium stability")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=r["Date"],
            y=r["EquilibriumDriftSigma"],
            name="Equilibrium drift (σ)"
        ))
        fig.add_trace(go.Scatter(
            x=r["Date"],
            y=r["DispersionRatio"],
            name="Dispersion ratio"
        ))
        shade_reliability(fig, r, h["Date"].min())
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0c141b", plot_bgcolor="#0c141b", height=360, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Rolling stationarity diagnostics")
        fig = px.line(
            r, x="Date", y="ADFp",
            labels={"ADFp": "ADF p-value"}
        )
        fig.add_hline(y=0.05, line_dash="dash")
        fig.add_hline(y=0.10, line_dash="dot")
        shade_reliability(fig, r, h["Date"].min())
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0c141b", plot_bgcolor="#0c141b", height=340)
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            r[[
                "Date", "HalfLifeDays", "ADFp",
                "EquilibriumDriftSigma",
                "DispersionRatio", "Reliability", "Reason"
            ]].sort_values("Date", ascending=False),
            hide_index=True,
            use_container_width=True
        )

with tab3:
    st.markdown("#### MVOU transition matrix")

    if not matrix.empty:
        pivot = matrix.pivot(
            index="Row", columns="Column", values="A"
        )
        fig = px.imshow(
            pivot,
            text_auto=".4f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            color_continuous_midpoint=0
        )
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0c141b", plot_bgcolor="#0c141b", height=360)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Stationary eigenmodes")

    if not modes.empty:
        st.dataframe(
            modes[[
                "Mode", "EigenvalueReal", "EigenvalueImag",
                "Magnitude", "HalfLifeDays", "Stable"
            ]],
            hide_index=True,
            use_container_width=True
        )

        st.markdown("#### Mean-reversion decay by eigenmode")
        fig = go.Figure()
        days = np.arange(0, 121)

        for _, mode in modes.iterrows():
            lam = complex(
                mode["EigenvalueReal"],
                mode["EigenvalueImag"]
            )
            if abs(lam) < 1:
                decay = np.abs(lam) ** days
                fig.add_trace(go.Scatter(
                    x=days, y=decay,
                    name=f"Mode {int(mode['Mode'])}"
                ))

        fig.add_hline(y=0.5, line_dash="dash")
        fig.update_layout(
            height=360,
            xaxis_title="Trading days",
            yaxis_title="Remaining deviation magnitude"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "These are fitted stationary-spread dynamics. "
        "They are not stock-level return forecasts. "
        "A one-component model has a 1×1 transition matrix."
    )

with tab4:
    p = positions[
        positions["Horizon"].astype(str).eq(horizon)
        & positions["BasketID"].astype(str).eq(bid)
    ].copy()

    st.markdown("#### Long / short exposure")
    st.dataframe(
        p[[
            "Ticker", "Side", "DollarWeight",
            "LatestAdjustedPrice", "SharesPerDollarGross"
        ]],
        hide_index=True,
        use_container_width=True
    )

    fig = px.bar(
        p, x="Ticker", y="DollarWeight",
        color="Side",
        color_discrete_map={
            "LONG": "#15996c",
            "SHORT": "#dc5555"
        }
    )
    fig.update_layout(template="plotly_dark", paper_bgcolor="#0c141b", plot_bgcolor="#0c141b", height=360)
    st.plotly_chart(fig, use_container_width=True)

    st.write(
        f"**Market-explained variance:** "
        f"{basket.MarketExplainedVariancePct:.3f}%"
    )
    st.caption(
        "Research exposures only. Adjusted prices are not "
        "live executable market prices."
    )

st.caption(
    "Meridian · Model X | Historical diagnostics use "
    "fixed fitted cointegrating vectors and trailing windows. "
    "The colour thresholds are heuristic, not probabilities "
    "of future convergence."
)

