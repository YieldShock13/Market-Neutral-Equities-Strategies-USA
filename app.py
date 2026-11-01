
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

import base64
LOGO = base64.b64encode((ROOT / "assets/meridian-mark.svg").read_bytes()).decode()
st.markdown("""
<style>
.block-container {max-width:1460px;padding:2.5rem 3.2rem 4rem;}
.stApp {background:radial-gradient(ellipse at 85% 0%,#eeeee8 0%,#f6f3eb 52%);}
[data-testid="stHeader"] {background:transparent;}
[data-testid="stSidebar"] {background:#ebe8df;border-right:1px solid #d0d2cc;}
[data-testid="stSidebar"] .block-container {padding-top:2rem;}
[data-testid="stMetric"] {background:linear-gradient(135deg,#fbfaf6,#f0eee7);border:1px solid #d4d7d2;border-radius:3px;padding:23px 25px;box-shadow:0 4px 16px #343e4405;}
[data-testid="stMetricLabel"] {color:#657078;text-transform:uppercase;letter-spacing:.14em;font-size:.68rem;}
[data-testid="stMetricValue"] {font-family:Georgia,serif;font-weight:400;color:#35434c;}
h1,h2,h3 {font-family:Georgia,serif !important;font-weight:400 !important;letter-spacing:.015em;color:#35434c;}
h3 {font-size:1.5rem !important;margin-top:1.6rem !important;}
.meridian-header {position:relative;overflow:hidden;padding:38px 40px 34px;border-top:1px solid #aeb7ba;border-bottom:1px solid #bfc5c4;margin-bottom:22px;}
.meridian-header:after {content:"";position:absolute;width:420px;height:420px;right:-40px;top:-175px;border:1px solid #a2adb340;border-radius:50%;box-shadow:0 0 0 38px #a2adb310,0 0 0 76px #a2adb30c,0 0 0 114px #a2adb308;pointer-events:none;}
.identity {display:flex;align-items:center;gap:24px;position:relative;z-index:1;}
.identity img {width:104px;height:104px;}
.brand {font-family:Georgia,'Times New Roman',serif;font-size:clamp(30px,4.5vw,58px);letter-spacing:.19em;color:#35434c;line-height:1.15;}
.model {color:#737f86;font-size:11px;letter-spacing:.4em;text-transform:uppercase;margin-top:12px;}
.descriptor {color:#727a7d;margin-top:25px;font-size:12px;letter-spacing:.08em;}
.edition {position:absolute;right:40px;bottom:34px;font-size:10px;color:#849096;letter-spacing:.22em;text-transform:uppercase;}
[data-baseweb="tab-list"] {gap:26px;border-bottom:1px solid #cdd1cd;}
[data-baseweb="tab"] {font-size:12px;letter-spacing:.035em;}
[data-baseweb="tab-highlight"] {background:#7e8d96;height:2px;}
[data-testid="stDataFrame"] {border:1px solid #d4d7d2;border-radius:3px;}
[data-testid="stCaptionContainer"] {color:#778187;}
[data-testid="stAlert"] {border-radius:3px;background:#eceee9;}
@media(max-width:700px) {.block-container{padding:1.5rem 1rem;}.meridian-header{padding:24px 12px;}.identity{gap:12px;}.identity img{width:68px;height:68px;}.brand{letter-spacing:.1em;}.edition{display:none;}.descriptor{font-size:10px;}}
</style>
""", unsafe_allow_html=True)
from audit_tools import snapshot_audit, duplicate_issuer_basket
from formulas import FORMULAS

@st.dialog("Meridian · Audit & formulas", width="large")
def show_audit_dialog():
    quality, maths, backend = st.tabs(["Data integrity", "All formulas", "Backend review"])
    with quality:
        report = snapshot_audit(PIPE)
        st.caption(report["scope"])
        st.dataframe(pd.DataFrame(report["checks"]), hide_index=True, use_container_width=True)
        raw_audit = PIPE / "data_audit.csv"
        if raw_audit.exists():
            st.markdown("#### Raw-price and sample exclusions")
            st.dataframe(pd.read_csv(raw_audit), hide_index=True, use_container_width=True)
        probe = PIPE / "legacy_price_probe.csv"
        if probe.exists():
            st.markdown("#### Fresh vendor comparison with legacy snapshot")
            st.caption("Re-downloaded today; vendor revisions can differ from the original download.")
            st.dataframe(pd.read_csv(probe), hide_index=True, use_container_width=True)
        st.markdown("#### Missing and infinite values")
        if report["missing"]:
            st.dataframe(pd.DataFrame(report["missing"]), hide_index=True, use_container_width=True)
            st.caption("Includes optional diagnostics; missing values are disclosed rather than filled.")
        else:
            st.write("No missing or infinite cells found in the readable published CSVs.")
        st.download_button("Download snapshot audit", json.dumps(report, indent=2), "meridian_snapshot_audit.json", "application/json")
    with maths:
        for title, formula, explanation in FORMULAS:
            with st.expander(title):
                st.latex(formula)
                st.write(explanation)
    with backend:
        report_path = ROOT / "AUDIT.md"
        st.markdown(report_path.read_text() if report_path.exists() else "Backend report unavailable.")
        proof = PIPE / "backend_audit.json"
        if proof.exists():
            st.download_button("Download backend evidence", proof.read_bytes(), "meridian_backend_audit.json", "application/json")

import json
_, audit_control = st.columns([5, 1])
with audit_control:
    if st.button("Audit & formulas", use_container_width=True):
        show_audit_dialog()

st.markdown(f"""
<div class="meridian-header"><div class="identity"><img src="data:image/svg+xml;base64,{LOGO}" alt="Meridian globe and M monogram"/><div><div class="brand">MERIDIAN</div><div class="model">Model X</div></div></div><div class="descriptor">Market-neutral equities &nbsp; / &nbsp; Relative value research</div><div class="edition">Equities / United States</div></div>
""", unsafe_allow_html=True)
import json
status_path = PIPE / "refresh_status.json"
if status_path.exists():
    refresh = json.loads(status_path.read_text())
    st.caption(f"Price date: {refresh['price_date']} · Updated: {refresh['completed_at']} · {refresh['eligible_issuers']:,} issuers")

try:
    data, positions = load_data()
except Exception as exc:
    st.error(f"Unable to load research files: {exc}")
    st.stop()

data = data[data["Status"] == "SIGNAL_COMPUTED"].copy()
# Legacy exports have no proof of the mandatory rolling full-history gate.
# Do not show those baskets as currently eligible after tightening the rule.
verified_tickers = set()
proof_path = PIPE / "data_audit.csv"
if status_path.exists() and proof_path.exists():
    manifest = json.loads(status_path.read_text())
    if manifest.get("eligibility_policy") == "rolling_1260_complete_sessions_v1":
        proof = pd.read_csv(proof_path)
        if "Eligible5TradingYears" in proof and "Ticker" in proof:
            passed = proof["Eligible5TradingYears"].astype(str).str.lower().eq("true")
            passed &= proof["RequiredSessions"].eq(1260)
            passed &= proof["WindowEnd"].eq(manifest.get("window_end"))
            verified_tickers = set(proof.loc[passed, "Ticker"])
data = data.loc[data["Basket"].map(lambda basket: set(str(basket).split("|")).issubset(verified_tickers))].copy()

# Legacy baskets must be recalculated, never relabelled as equivalent exposures.
import re
legacy_duplicate = data["Basket"].map(duplicate_issuer_basket)
data = data.loc[~legacy_duplicate].copy()

data["Stability"] = np.where(
    data["TrainSpecificationRobust"].map(lambda v: str(v).lower() == "true"),
    "PASS",
    "FLAGGED"
)

with st.sidebar:
    st.image(str(ROOT / "assets/meridian-mark.svg"), width=64)
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
        "JointMispricing": "Joint distance",
        "PortfolioBeta": "Portfolio Beta",
        "MarketExplainedVariancePct": "Market R² (%)",
        "NetDollarExposurePct": "Net Exposure (%)"
    }),
    hide_index=True,
    use_container_width=True
)

if filtered.empty:
    st.info("No baskets have verified complete five-trading-year data under the selected filters.")
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
import plotly.io as pio
pio.templates["meridian"] = go.layout.Template(
    layout=dict(paper_bgcolor="#f6f3eb", plot_bgcolor="#fbfaf6",
                font=dict(family="Arial, sans-serif", color="#46545c", size=12),
                colorway=["#657c8b", "#a5aeb1", "#9b907b", "#445b67"],
                xaxis=dict(gridcolor="#e2e3dc", zerolinecolor="#c7cecc"),
                yaxis=dict(gridcolor="#e2e3dc", zerolinecolor="#c7cecc"),
                margin=dict(l=24,r=24,t=32,b=32),
                hoverlabel=dict(bgcolor="#fbfaf6",font_color="#35434c")))
pio.templates.default = "plotly_white+meridian"


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
        line=dict(color="#657c8b", width=2)
    ))
    fig.add_trace(go.Scatter(
        x=h["Date"], y=h["Equilibrium"],
        name="Fitted equilibrium",
        line=dict(color="#525e65", dash="dash")
    ))

    for n, dash in [(1, "dot"), (2, "dash"), (3, "longdash")]:
        for sign, side in [("+", "Upper"), ("-", "Lower")]:
            fig.add_trace(go.Scatter(
                x=h["Date"], y=h[f"{side}{n}"],
                name=f"{sign}{n}σ",
                line=dict(color="#a2aaad", dash=dash, width=1),
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
    st.plotly_chart(fig, use_container_width=True, theme=None)

    st.markdown("#### Historical standardized deviation")

    zfig = go.Figure()
    shade_reliability(zfig, r, h["Date"].min())
    zfig.add_trace(go.Scatter(
        x=h["Date"], y=h["ZScore"],
        name="Spread z-score",
        line=dict(color="#657c8b")
    ))

    for level in [-3, -2, -1, 0, 1, 2, 3]:
        zfig.add_hline(
            y=level,
            line_dash="solid" if level == 0 else "dot",
            line_color="#99a3a8",
            line_width=1
        )

    zfig.update_layout(
        height=380,
        yaxis_title="Standard deviations",
        hovermode="x unified"
    )
    st.plotly_chart(zfig, use_container_width=True, theme=None)

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
        fig.update_layout(template="plotly_white+meridian", paper_bgcolor="#f6f3eb", plot_bgcolor="#fbfaf6", height=360)
        st.plotly_chart(fig, use_container_width=True, theme=None)

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
        fig.update_layout(template="plotly_white+meridian", paper_bgcolor="#f6f3eb", plot_bgcolor="#fbfaf6", height=360, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True, theme=None)

        st.markdown("#### Rolling stationarity diagnostics")
        fig = px.line(
            r, x="Date", y="ADFp",
            labels={"ADFp": "ADF p-value"}
        )
        fig.add_hline(y=0.05, line_dash="dash")
        fig.add_hline(y=0.10, line_dash="dot")
        shade_reliability(fig, r, h["Date"].min())
        fig.update_layout(template="plotly_white+meridian", paper_bgcolor="#f6f3eb", plot_bgcolor="#fbfaf6", height=340)
        st.plotly_chart(fig, use_container_width=True, theme=None)

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
            color_continuous_scale=["#748a99", "#f6f3eb", "#a29383"],
            color_continuous_midpoint=0
        )
        fig.update_layout(template="plotly_white+meridian", paper_bgcolor="#f6f3eb", plot_bgcolor="#fbfaf6", height=360)
        st.plotly_chart(fig, use_container_width=True, theme=None)

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
        st.plotly_chart(fig, use_container_width=True, theme=None)

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
            "LONG": "#627d76",
            "SHORT": "#a1877e"
        }
    )
    fig.update_layout(template="plotly_white+meridian", paper_bgcolor="#f6f3eb", plot_bgcolor="#fbfaf6", height=360)
    st.plotly_chart(fig, use_container_width=True, theme=None)

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

