
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
import os

# --------------------------------------------------
# Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="SupplyChain Action Copilot",
    page_icon="📦",
    layout="wide",
)

DATA_PATH = Path("data/orders_modeling.csv")
# MODEL_PATH = Path("models/late_risk_model_temporal.joblib")
MODEL_PATH = Path("models/late_risk_model_compact.joblib")
ENABLE_LOCAL_LLM = os.getenv("ENABLE_LOCAL_LLM", "true").lower() == "true"

# NUMERIC_FEATURES = [
#     "scheduled_shipping_days",
#     "order_month",
#     "order_day_of_week",
#     "item_count",
#     "total_quantity",
#     "total_sales",
#     "total_item_value",
#     "total_discount",
#     "mean_discount_rate",
#     "mean_product_price",
#     "mean_benefit_per_order",
# ]

# CATEGORICAL_FEATURES = [
#     "order_type",
#     "customer_segment",
#     "market",
#     "order_region",
#     "shipping_mode",
#     "customer_country",
#     "order_country",
# ]

# FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

FEATURES = [
    "scheduled_shipping_days",
    "order_month",
    "shipping_mode",
    "order_region",
    "order_type",
]


# --------------------------------------------------
# LLM recommendations
# --------------------------------------------------

def generate_ai_recommendations(order):
    import ollama
    order_context = {
        "predicted_late_delivery_risk": round(
            float(order["risk_score"]), 4
        ),
        "shipping_mode": str(order["shipping_mode"]),
        "region": str(order["order_region"]),
        "scheduled_shipping_days": int(
            order["scheduled_shipping_days"]
        ),
        "number_of_items": int(order["item_count"]),
        "total_quantity": int(order["total_quantity"]),
        "total_sales": round(float(order["total_sales"]), 2),
    }

    system_prompt = """
    You are an AI supply chain decision-support assistant.

    Interpret machine learning risk predictions and suggest
    practical actions for supply chain managers.

    RULES:
    1. Only use information explicitly provided.
    2. Never invent carrier delays, route congestion, inventory
       shortages, delivery guarantees, or operational conditions.
    3. A predicted risk score is a model estimate, not confirmation
       that an order will be late.
    4. Do not assume that a shipping mode is unreliable.
    5. Distinguish known information from hypotheses requiring
       further investigation.
    6. Recommendations must be practical and relevant to the
       information available.
    7. Do not recommend actions that require assuming unavailable
       operational capabilities.
    8. Explain why each recommendation is useful.
    9. If information is missing, explicitly identify what
       should be verified.

    FORMAT YOUR RESPONSE:

    RISK INTERPRETATION:
    Explain what the prediction means.

    RECOMMENDED ACTIONS:
    Provide three actionable recommendations.

    INFORMATION TO VERIFY:
    Identify important missing operational information.

    Keep the response concise and professional.
    """

    response = ollama.chat(
        model="qwen3.5:4b",
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": (
                    "Analyze this selected order:\n"
                    f"{order_context}"
                ),
            },
        ],
        options={"temperature": 0.2},
    )

    return response.message.content



# --------------------------------------------------
# Load and score data
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_scored_orders():
    df = pd.read_csv(DATA_PATH, parse_dates=["order_date"])
    df = df.dropna(subset=["order_date", "Late_delivery_risk"])
    df = df.sort_values("order_date").reset_index(drop=True)

    # Reproduce the temporal validation split.
    split_at = int(len(df) * 0.80)
    test = df.iloc[split_at:].copy()

    model = load_model()
    test["risk_score"] = model.predict_proba(
        test[FEATURES]
    )[:, 1]

    return test


if not DATA_PATH.exists():
    st.error("Could not find data/orders_modeling.csv.")
    st.stop()

if not MODEL_PATH.exists():
    st.error("Could not find models/late_risk_model_compact.joblib.")
    st.stop()

orders = load_scored_orders()


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("SupplyChain Action Copilot")
st.sidebar.caption("Historical shipment-risk analysis")

threshold = st.sidebar.slider(
    "Risk threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.50,
    step=0.05,
)

show_actual = st.sidebar.checkbox(
    "Show actual historical outcomes (evaluation only)",
    value=False,
)

st.sidebar.divider()

st.sidebar.caption(
    "This prototype uses the later 20% of the dataset "
    "as a historical holdout. Actual outcomes are hidden "
    "from the default risk queue."
)


# --------------------------------------------------
# Filter and calculate metrics
# --------------------------------------------------

orders["risk_level"] = pd.cut(
    orders["risk_score"],
    bins=[-0.001, 0.40, 0.70, 1.001],
    labels=["Low", "Medium", "High"],
)

flagged = orders[orders["risk_score"] >= threshold].copy()

total_orders = len(orders)
flagged_orders = len(flagged)
avg_risk = orders["risk_score"].mean()

# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("📦 SupplyChain Action Copilot")
st.write(
    "An order-prioritization dashboard that uses machine "
    "learning to identify orders with elevated late-delivery risk."
)

st.info(
    f"Historical holdout period: "
    f"{orders['order_date'].min():%Y-%m-%d} to "
    f"{orders['order_date'].max():%Y-%m-%d}. "
    "Predictions are generated using a model trained on earlier orders."
)

# --------------------------------------------------
# KPI cards
# --------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

col1.metric("Orders evaluated", f"{total_orders:,}")
col2.metric("Orders flagged", f"{flagged_orders:,}")
col3.metric(
    "Flagged percentage",
    f"{flagged_orders / total_orders:.1%}",
)
col4.metric("Average predicted risk", f"{avg_risk:.1%}")

st.divider()

# --------------------------------------------------
# Risk distribution
# --------------------------------------------------

left, right = st.columns(2)

with left:
    st.subheader("Predicted risk distribution")

    fig = px.histogram(
        orders,
        x="risk_score",
        nbins=30,
        title="Distribution of predicted risk scores",
        labels={"risk_score": "Predicted late-risk probability"},
    )

    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Orders by risk category")

    risk_counts = (
        orders["risk_level"]
        .value_counts()
        .reindex(["Low", "Medium", "High"])
        .fillna(0)
        .reset_index()
    )

    risk_counts.columns = ["Risk category", "Orders"]

    fig = px.bar(
        risk_counts,
        x="Risk category",
        y="Orders",
        color="Risk category",
        title="Order count by predicted risk category",
    )

    st.plotly_chart(fig, use_container_width=True)

st.divider()

# --------------------------------------------------
# Priority queue
# --------------------------------------------------

st.subheader("🚨 Order priority queue")

st.write(
    f"Showing {flagged_orders:,} orders with predicted risk "
    f"at or above {threshold:.0%}."
)

display_columns = [
    "Order Id",
    "order_date",
    "risk_score",
    "risk_level",
    "shipping_mode",
    "order_region",
    "scheduled_shipping_days",
    "item_count",
    "total_quantity",
    "total_sales",
]

if show_actual:
    display_columns.append("Late_delivery_risk")

display = flagged[display_columns].copy()

display = display.rename(
    columns={
        "Order Id": "Order ID",
        "order_date": "Order date",
        "risk_score": "Predicted risk",
        "risk_level": "Risk category",
        "shipping_mode": "Shipping mode",
        "order_region": "Region",
        "scheduled_shipping_days": "Scheduled days",
        "item_count": "Number of items",
        "total_quantity": "Total quantity",
        "total_sales": "Total sales",
        "Late_delivery_risk": "Actual late-risk label",
    }
)

display["Predicted risk"] = display["Predicted risk"].map(
    lambda value: f"{value:.1%}"
)

display = display.sort_values(
    "Predicted risk",
    ascending=False,
    key=lambda column: column.str.rstrip("%").astype(float),
)

st.dataframe(
    display,
    use_container_width=True,
    hide_index=True,
)

# --------------------------------------------------
# Export
# --------------------------------------------------

csv_data = flagged.sort_values(
    "risk_score", ascending=False
).to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download priority queue as CSV",
    data=csv_data,
    file_name="supplychain_priority_queue.csv",
    mime="text/csv",
)

st.divider()

# --------------------------------------------------
# Selected order details
# --------------------------------------------------

st.subheader("🔎 Order details")

if flagged.empty:
    st.info("No orders currently meet the selected risk threshold.")
else:
    selected_id = st.selectbox(
        "Select an order",
        flagged.sort_values("risk_score", ascending=False)["Order Id"],
    )

    selected = flagged.loc[
        flagged["Order Id"] == selected_id
    ].iloc[0]

    detail1, detail2, detail3 = st.columns(3)

    detail1.metric("Order ID", str(selected_id))
    detail2.metric("Predicted risk", f"{selected['risk_score']:.1%}")
    detail3.metric("Risk category", str(selected["risk_level"]))

    st.write("**Order characteristics**")

    st.json({
        "Order date": str(selected["order_date"]),
        "Shipping mode": selected["shipping_mode"],
        "Region": selected["order_region"],
        "Scheduled shipping days": int(
            selected["scheduled_shipping_days"]
        ),
        "Number of items": int(selected["item_count"]),
        "Total quantity": int(selected["total_quantity"]),
        "Total sales": round(float(selected["total_sales"]), 2),
    })

    # Transparent, illustrative review guidance.
    if selected["risk_score"] >= 0.70:
        st.warning(
            "Suggested review: prioritize checking shipment readiness, "
            "inventory availability, and carrier pickup status."
        )
    elif selected["risk_score"] >= 0.40:
        st.info(
            "Suggested review: monitor shipment preparation and "
            "confirm the planned dispatch schedule."
        )
    else:
        st.success(
            "Suggested review: continue standard shipment monitoring."
        )

    st.caption(
        "These are illustrative, rule-based review suggestions. "
        "They are not validated interventions or LLM-generated advice."
    )

    # --------------------------------------------------
    # AI-generated recommendations
    # --------------------------------------------------

    st.divider()

    st.subheader("🤖 AI Supply Chain Recommendations")

if ENABLE_LOCAL_LLM:
    st.write(
        "Generate an AI-assisted analysis of this order "
        "using its predicted risk and available characteristics."
    )

    if st.button(
        "✨ Generate AI Recommendations",
        type="primary",
        use_container_width=True,
    ):
        try:
            with st.spinner(
                "Analyzing order and generating recommendations..."
            ):
                recommendation = generate_ai_recommendations(selected)

                st.session_state["ai_recommendation"] = recommendation
                st.session_state["ai_recommendation_order_id"] = str(
                    selected_id
                )
                st.session_state.pop("ai_error", None)

        except Exception as error:
            st.session_state["ai_error"] = str(error)
            st.session_state.pop("ai_recommendation", None)

    if st.session_state.get("ai_error"):
        st.error(
            "Could not generate recommendations. "
            "Make sure Ollama is running and Qwen 3.5 4B is installed."
        )
        st.caption(st.session_state["ai_error"])

    if (
        st.session_state.get("ai_recommendation_order_id")
        == str(selected_id)
        and st.session_state.get("ai_recommendation")
    ):
        st.markdown(st.session_state["ai_recommendation"])

else:
    st.info(
        "Local AI recommendations are disabled in this deployment. "
        "The ML risk predictions and dashboard features remain available."
    )

    st.write(
        "Generate an AI-assisted analysis of this order "
        "using its predicted risk and available characteristics."
    )

    if st.button(
        "✨ Generate AI Recommendations",
        type="primary",
        use_container_width=True,
    ):
        try:
            with st.spinner(
                "Analyzing order and generating recommendations..."
            ):
                recommendation = generate_ai_recommendations(selected)

                st.session_state["ai_recommendation"] = recommendation
                st.session_state["ai_recommendation_order_id"] = str(
                    selected_id
                )
                st.session_state.pop("ai_error", None)

        except Exception as error:
            st.session_state["ai_error"] = str(error)
            st.session_state.pop("ai_recommendation", None)

    if st.session_state.get("ai_error"):
        st.error(
            "Could not generate recommendations. "
            "Make sure Ollama is running and the model is installed."
        )
        st.caption(st.session_state["ai_error"])

    if (
        st.session_state.get("ai_recommendation_order_id")
        == str(selected_id)
        and st.session_state.get("ai_recommendation")
    ):
        st.markdown("### AI-generated analysis")

        st.markdown(
            st.session_state["ai_recommendation"]
        )

        st.caption(
            "Generated locally using Qwen 3.5 4B through Ollama. "
            "Recommendations are AI-generated suggestions, not "
            "verified operational instructions."
        )

