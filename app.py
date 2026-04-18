import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from model import train_model, predict_future

# -----------------------------
# CONFIG (STANDARD LAYOUT)
# -----------------------------
st.set_page_config(
    page_title="Expense Tracker",
    layout="centered",
    initial_sidebar_state="expanded"
)

DATA_PATH = "data/expenses.csv"
os.makedirs("data", exist_ok=True)

# -----------------------------
# SAFE LOAD
# -----------------------------
def load_data():
    if os.path.exists(DATA_PATH):
        try:
            df = pd.read_csv(DATA_PATH)
            if df.empty:
                raise ValueError
        except:
            df = pd.DataFrame(columns=["date", "category", "amount", "payment_method"])
    else:
        df = pd.DataFrame(columns=["date", "category", "amount", "payment_method"])
    return df


def save_data(df):
    df.to_csv(DATA_PATH, index=False)


df = load_data()

# -----------------------------
# SIDEBAR INPUT
# -----------------------------
st.sidebar.header("Add Expense")

date = st.sidebar.date_input("Date")
category = st.sidebar.selectbox(
    "Category", ["Food", "Travel", "Rent", "Shopping", "Bills", "Entertainment"]
)
amount = st.sidebar.number_input("Amount", min_value=0.0)
payment = st.sidebar.selectbox("Payment Method", ["Cash", "UPI", "Card"])

if st.sidebar.button("Add Expense"):
    new_row = pd.DataFrame(
        [[date, category, amount, payment]],
        columns=["date", "category", "amount", "payment_method"]
    )
    df = pd.concat([df, new_row], ignore_index=True)
    save_data(df)
    st.sidebar.success("Expense Added")

# -----------------------------
# MAIN UI
# -----------------------------
st.markdown(
    "<h1 style='text-align: center;'>💰 Expense Tracker Dashboard</h1>",
    unsafe_allow_html=True
)

if df.empty:
    st.warning("No data available. Add expenses.")
else:
    # -----------------------------
    # CLEANING
    # -----------------------------
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    df = df.dropna()

    df['month'] = df['date'].dt.to_period('M').astype(str)

    # -----------------------------
    # KPIs
    # -----------------------------
    total_spent = df['amount'].sum()
    avg_spent = df['amount'].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Spending", f"₹{total_spent:.2f}")
    col2.metric("Average Spending", f"₹{avg_spent:.2f}")
    col3.metric("Transactions", len(df))

    st.markdown("---")

    # -----------------------------
    # CATEGORY CHART
    # -----------------------------
    st.subheader("Category-wise Spending")

    cat = df.groupby('category')['amount'].sum()

    fig1, ax1 = plt.subplots(figsize=(5, 5))
    cat.plot(kind='pie', autopct='%1.1f%%', ax=ax1)
    ax1.set_ylabel("")
    st.pyplot(fig1, use_container_width=True)

    st.markdown("---")

    # -----------------------------
    # MONTHLY TREND
    # -----------------------------
    st.subheader("Monthly Spending Trend")

    monthly = df.groupby('month')['amount'].sum()

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    monthly.plot(marker='o', ax=ax2)
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Amount")
    plt.xticks(rotation=45)
    st.pyplot(fig2, use_container_width=True)

    st.markdown("---")

    # -----------------------------
    # ML PREDICTION
    # -----------------------------
    st.subheader("🔮 Future Expense Prediction")

    model = train_model(df)

    if model is None:
        st.warning("Add at least 5 valid records for prediction.")
    else:
        preds = predict_future(model, 7)

        pred_df = pd.DataFrame({
            "Day": list(range(1, 8)),
            "Predicted Expense": preds
        })

        st.dataframe(pred_df)

        fig3, ax3 = plt.subplots(figsize=(6, 4))
        ax3.plot(pred_df["Day"], pred_df["Predicted Expense"], marker='o')
        ax3.set_title("Next 7 Days Prediction")
        ax3.set_xlabel("Day")
        ax3.set_ylabel("Expense")
        st.pyplot(fig3, use_container_width=True)

    # -----------------------------
    # ALERT
    # -----------------------------
    if df['amount'].max() > 10000:
        st.error("⚠️ High spending detected!")