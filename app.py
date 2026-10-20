
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Climate Intelligence Dashboard",
    page_icon="🌡️",
    layout="wide"
)

BASE_DIR = Path(__file__).parent
ASSET_DIR = BASE_DIR / "climate_dashboard_assets"

MODEL_PATH = ASSET_DIR / "heat_index_model.joblib"
FEATURES_PATH = ASSET_DIR / "heat_index_features.joblib"
DATA_PATH = ASSET_DIR / "climate_dashboard_data.csv"
METRICS_PATH = ASSET_DIR / "model_evaluation.csv"
IMPORTANCE_PATH = ASSET_DIR / "feature_importance.csv"

# --------------------------------------------------
# LOAD SAVED ASSETS
# --------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    return model, features

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    data["Date"] = pd.to_datetime(
        data["Date"], dayfirst=True, errors="coerce"
    )
    return data

try:
    model, feature_cols = load_model()
    df = load_data()
    metrics = pd.read_csv(METRICS_PATH, index_col=0)
    importance = pd.read_csv(IMPORTANCE_PATH)
except Exception as e:
    st.error(f"Could not load the dashboard assets: {e}")
    st.info(
        "Check that all exported files are inside "
        "climate_dashboard_assets/ beside app.py."
    )
    st.stop()

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.title("🌡️ Climate Intelligence Dashboard")
st.caption(
    "Interactive climate analysis, Heat Index estimation, "
    "and data-driven insights for India."
)

st.warning(
    "Model scope: estimates Heat Index from climate measurements "
    "for the same record. It is not a validated future heatwave "
    "forecast or an official public warning system."
)

# --------------------------------------------------
# SIDEBAR FILTERS
# --------------------------------------------------
st.sidebar.header("Dashboard Filters")

states = sorted(df["State"].dropna().unique().tolist())
selected_states = st.sidebar.multiselect(
    "Select state(s)",
    states,
    default=states
)

state_data = df[df["State"].isin(selected_states)]

cities = sorted(state_data["City"].dropna().unique().tolist())
selected_cities = st.sidebar.multiselect(
    "Select city/cities",
    cities,
    default=cities
)

filtered = state_data[state_data["City"].isin(selected_cities)].copy()

valid_dates = filtered["Date"].dropna()
if not valid_dates.empty:
    min_date = valid_dates.min().date()
    max_date = valid_dates.max().date()

    date_range = st.sidebar.date_input(
        "Select date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start_date, end_date = date_range
        filtered = filtered[
            filtered["Date"].dt.date.between(start_date, end_date)
        ]
else:
    st.sidebar.info("No valid dates available for filtering.")

if filtered.empty:
    st.warning("No records match the selected filters. Change your filters.")
    st.stop()

# --------------------------------------------------
# RISK LABELS
# --------------------------------------------------
def risk_label(value):
    if value < 39:
        return "Low"
    elif value < 44:
        return "Medium"
    return "High"

filtered["Dashboard_Risk"] = filtered["Heat_Index"].apply(risk_label)

# --------------------------------------------------
# KPI CARDS
# --------------------------------------------------
st.subheader("Climate Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Records", f"{len(filtered):,}")
c2.metric(
    "Average Heat Index",
    f"{filtered['Heat_Index'].mean():.2f}"
)
c3.metric(
    "Maximum Heat Index",
    f"{filtered['Heat_Index'].max():.2f}"
)
high_pct = (filtered["Dashboard_Risk"] == "High").mean() * 100
c4.metric("High-risk records", f"{high_pct:.1f}%")

# --------------------------------------------------
# TABS
# --------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "🌍 City & State",
    "🤖 AI Prediction",
    "💡 Automatic Insights",
    "📈 Model Evaluation"
])

# --------------------------------------------------
# TAB 1: OVERVIEW
# --------------------------------------------------
with tab1:
    st.subheader("Heat Index Distribution")

    fig_hist = px.histogram(
        filtered,
        x="Heat_Index",
        nbins=35,
        color="Dashboard_Risk",
        title="Distribution of Heat Index",
        labels={"Heat_Index": "Heat Index"}
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    left, right = st.columns(2)

    with left:
        risk_counts = (
            filtered["Dashboard_Risk"]
            .value_counts()
            .reindex(["Low", "Medium", "High"], fill_value=0)
            .rename_axis("Risk")
            .reset_index(name="Records")
        )
        fig_risk = px.pie(
            risk_counts,
            names="Risk",
            values="Records",
            title="Risk Category Distribution",
            hole=0.4
        )
        st.plotly_chart(fig_risk, use_container_width=True)

    with right:
        daily = (
            filtered.dropna(subset=["Date"])
            .groupby("Date", as_index=False)["Heat_Index"]
            .mean()
        )
        fig_trend = px.line(
            daily,
            x="Date",
            y="Heat_Index",
            title="Average Heat Index Over Time"
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    st.subheader("Filtered Data Preview")
    st.dataframe(filtered.head(100), use_container_width=True)

    st.download_button(
        "Download filtered data as CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_climate_data.csv",
        mime="text/csv"
    )

# --------------------------------------------------
# TAB 2: CITY AND STATE
# --------------------------------------------------
with tab2:
    st.subheader("Compare Cities and States")

    city_summary = (
        filtered.groupby(["State", "City"], as_index=False)
        .agg(
            Average_Heat_Index=("Heat_Index", "mean"),
            Maximum_Heat_Index=("Heat_Index", "max"),
            Records=("Heat_Index", "size")
        )
        .sort_values("Average_Heat_Index", ascending=False)
    )

    fig_city = px.bar(
        city_summary.head(20),
        x="City",
        y="Average_Heat_Index",
        color="State",
        title="Top 20 Cities by Average Heat Index",
        hover_data=["Maximum_Heat_Index", "Records"]
    )
    st.plotly_chart(fig_city, use_container_width=True)

    st.dataframe(city_summary, use_container_width=True)

    fig_scatter = px.scatter(
        filtered,
        x="Day_Temperature_C",
        y="Heat_Index",
        color="Dashboard_Risk",
        hover_data=["City", "State", "Date"],
        title="Day Temperature vs Heat Index"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# --------------------------------------------------
# TAB 3: AI PREDICTION
# --------------------------------------------------
with tab3:
    st.subheader("Estimate Heat Index from Climate Inputs")
    st.write(
        "Enter climate measurements to estimate Heat Index. "
        "The prediction applies to the entered conditions, not a future date."
    )

    # Ranges are based on observed dataset values.
    numeric_ranges = {}
    for col in feature_cols:
        numeric_ranges[col] = (
            float(df[col].min()),
            float(df[col].max()),
            float(df[col].median())
        )

    with st.form("prediction_form"):
        values = {}
        cols = st.columns(2)

        for i, feature in enumerate(feature_cols):
            low, high, median = numeric_ranges[feature]
            if not np.isfinite(low) or not np.isfinite(high):
                low, high, median = 0.0, 100.0, 50.0
            if low == high:
                values[feature] = low
                continue

            with cols[i % 2]:
                values[feature] = st.number_input(
                    feature.replace("_", " "),
                    min_value=low,
                    max_value=high,
                    value=float(np.clip(median, low, high)),
                    key=f"input_{feature}"
                )

        submitted = st.form_submit_button("Estimate Heat Index")

    if submitted:
        input_df = pd.DataFrame(
            [[values[col] for col in feature_cols]],
            columns=feature_cols
        )
        predicted = float(model.predict(input_df)[0])
        predicted_risk = risk_label(predicted)

        st.success("Estimation complete")
        p1, p2 = st.columns(2)
        p1.metric("Estimated Heat Index", f"{predicted:.2f}")
        p2.metric("Indicative risk category", predicted_risk)

        if predicted_risk == "High":
            st.error(
                "The estimated Heat Index falls in the High category "
                "under the dataset's thresholds."
            )
        elif predicted_risk == "Medium":
            st.warning(
                "The estimated Heat Index falls in the Medium category "
                "under the dataset's thresholds."
            )
        else:
            st.info(
                "The estimated Heat Index falls in the Low category "
                "under the dataset's thresholds."
            )

        st.caption(
            "These categories are derived from this dataset's label "
            "ranges and are not official health or weather alerts."
        )

# --------------------------------------------------
# TAB 4: AUTOMATIC INSIGHTS
# --------------------------------------------------
with tab4:
    st.subheader("Data-Driven Automatic Insights")

    avg_heat = filtered["Heat_Index"].mean()
    max_heat = filtered["Heat_Index"].max()
    avg_temp = filtered["Day_Temperature_C"].mean()
    avg_humidity = filtered["Humidity_Percent"].mean()
    high_count = (filtered["Dashboard_Risk"] == "High").sum()

    st.markdown(
        f"- **Average Heat Index:** {avg_heat:.2f} across "
        f"{len(filtered):,} filtered records."
    )
    st.markdown(
        f"- **Maximum recorded Heat Index:** {max_heat:.2f}."
    )
    st.markdown(
        f"- **Average daytime temperature:** {avg_temp:.2f} °C."
    )
    st.markdown(
        f"- **Average humidity:** {avg_humidity:.1f}%."
    )
    st.markdown(
        f"- **High-category records:** {high_count:,} "
        f"({high_pct:.1f}% of filtered records)."
    )

    if not city_summary.empty:
        hottest = city_summary.iloc[0]
        st.markdown(
            f"- **Highest average among displayed city groups:** "
            f"{hottest['City']}, {hottest['State']} "
            f"({hottest['Average_Heat_Index']:.2f})."
        )

    corr = filtered[
        ["Day_Temperature_C", "Humidity_Percent", "Heat_Index"]
    ].corr()["Heat_Index"]

    st.markdown(
        f"- **Correlation with Heat Index:** daytime temperature "
        f"{corr['Day_Temperature_C']:.3f}; humidity "
        f"{corr['Humidity_Percent']:.3f}."
    )

    st.caption(
        "Insights are calculated from the selected data using "
        "statistical summaries and rules; they are not generated "
        "by a separate generative AI service."
    )

# --------------------------------------------------
# TAB 5: MODEL EVALUATION
# --------------------------------------------------
with tab5:
    st.subheader("Held-Out Test Set Performance")
    st.write(
        "Metrics reported here are from the saved evaluation "
        "performed during model preparation."
    )

    st.dataframe(metrics.round(4), use_container_width=True)

    st.subheader("Feature Importance")
    fig_imp = px.bar(
        importance.sort_values("Importance", ascending=True),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Random Forest Feature Importance"
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    st.info(
        "The model's very high test score may reflect the strong "
        "relationship between daytime temperature and Heat Index "
        "in this dataset. Performance on independent real-world "
        "data has not been established."
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()
st.caption(
    "DSVL Experiment 7 | Climate Intelligence Dashboard | "
    "Educational analysis, not an official warning service."
)
