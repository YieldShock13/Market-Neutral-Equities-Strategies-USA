
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(
    page_title="MVOU Relative Value",
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

st.title("MVOU Relative Value Scanner")
st.caption(
    "Cointegration • Mean reversion • Current mispricing"
)

try:
    data, positions = load_data()
except Exception as exc:
    st.error(f"Unable to load research files: {exc}")
    st.stop()

data = data[
    data["Status"] == "SIGNAL_COMPUTED"
].copy()

data["Stability"] = np.where(
    data["TrainSpecificationRobust"].fillna(False),
    "PASS",
    "FLAGGED"
)

with st.sidebar:
    st.header("Filters")

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

st.subheader("Relative Mispricing Rankings")

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

st.subheader("Basket Detail")

choices = {
    f"{r.Horizon} | {r.Basket} | {r.JointMispricing:.2f}":
        (r.Horizon, r.BasketID)
    for r in filtered.itertuples()
}

selected = st.selectbox(
    "Select basket",
    list(choices.keys())
)

horizon, basket_id = choices[selected]

basket = filtered[
    (filtered["Horizon"] == horizon)
    & (filtered["BasketID"] == basket_id)
].iloc[0]

st.write(
    f"**Mispricing:** {basket.JointMispricing:.3f}σ"
    f"  |  **Stability:** {basket.Stability}"
    f"  |  **Beta:** {basket.PortfolioBeta:.3f}"
)

p = positions[
    (positions["Horizon"] == horizon)
    & (positions["BasketID"] == basket_id)
].copy()

st.dataframe(
    p[
        [
            "Ticker",
            "Side",
            "DollarWeight",
            "LatestAdjustedPrice",
            "SharesPerDollarGross"
        ]
    ],
    hide_index=True,
    use_container_width=True
)

st.bar_chart(
    p.set_index("Ticker")["DollarWeight"]
)

st.caption(
    "Weights are normalized statistical exposures, "
    "not live execution instructions. "
    "Historical stability is diagnostic, not a guarantee."
)
