from datetime import date

from ml_model import train_full_model, predict_category
import streamlit as st
import pandas as pd
import plotly.express as px
from anomaly_detection import detect_unusual_expenses

from database import (
    initialize_database,
    initialize_budgets,
    add_transaction,
    get_transactions,
    save_budget,
    get_budgets,
    delete_transaction,
)

st.set_page_config(
    page_title="AI Expense Analyzer",
    page_icon="💰",
    layout="wide",
)

initialize_database()
initialize_budgets()
@st.cache_resource
def load_category_model():
    return train_full_model()


category_model = load_category_model()
st.title("💰 AI Expense Analyzer")
st.caption("Track income, analyze spending, and manage your budget.")

# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.header("Add Transaction")

    with st.form("transaction_form"):
        date = st.date_input("Transaction date")
        description = st.text_input("Description")

        transaction_type = st.selectbox(
            "Transaction type", ["Expense", "Income"]
        )

        if transaction_type == "Income":
            categories = [
                "Salary", "Allowance", "Freelance", "Other"
            ]
        else:
            categories = [
    "Food",
    "Transport",
    "Shopping",
    "Entertainment",
    "Bills",
    "Education",
    "Healthcare",
    "Groceries",
    "Other",
    "Auto-detect with AI",
]

        category = st.selectbox("Category", categories)

        amount = st.number_input(
            "Amount (₹)", min_value=0.0, step=10.0
        )

        submitted = st.form_submit_button(
            "Save transaction", type="primary"
        )

    
if submitted:
    if not description.strip() or amount <= 0:
        st.error("Enter a description and a positive amount.")

    else:
        saved_category = category
        prediction_message = None

        if (
            transaction_type == "Expense"
            and category == "Auto-detect with AI"
        ):
            saved_category, confidence = predict_category(
                category_model,
                description,
            )

            prediction_message = (
                f"AI predicted **{saved_category}** "
                f"(model score: {confidence:.1%}). "
                "Please verify the category."
            )

        add_transaction(
            date,
            description,
            transaction_type,
            saved_category,
            amount,
        )

        if prediction_message:
            st.success("Transaction saved using AI categorization.")
            st.info(prediction_message)
        else:
            st.success("Transaction saved.")

        st.rerun()


# ---------------- DASHBOARD ----------------

df = get_transactions()

st.divider()
st.header("Monthly Budget Planner")

budget_date = st.date_input(
    "Select budget month",
    value=date.today().replace(day=1),
    key="budget_month",
)
month_key = budget_date.strftime("%Y-%m")
expense_categories = [
    "Food", "Transport", "Shopping", "Entertainment", "Bills",
    "Education", "Healthcare", "Groceries", "Other",
]

with st.form("budget_form"):
    budget_category = st.selectbox(
        "Expense category", expense_categories
    )
    budget_limit = st.number_input(
        "Monthly limit (₹)",
        min_value=1.0,
        value=2000.0,
        step=100.0,
    )
    budget_submitted = st.form_submit_button("Save budget")

if budget_submitted:
    save_budget(month_key, budget_category, budget_limit)
    st.success("Budget saved successfully.")
    st.rerun()

budgets = get_budgets(month_key)
if not budgets.empty:
    month_expenses = df[df["Type"] == "Expense"].copy()
    if not month_expenses.empty:
        month_expenses["Month"] = pd.to_datetime(
            month_expenses["Date"], errors="coerce"
        ).dt.strftime("%Y-%m")
        month_expenses = month_expenses[
            month_expenses["Month"] == month_key
        ]
        spent = month_expenses.groupby("Category")["Amount"].sum()
    else:
        spent = pd.Series(dtype="float64")

    for row in budgets.itertuples(index=False):
        category = row.Category
        limit_amount = float(row.Budget)
        total_spent = float(spent.get(category, 0))
        remaining = limit_amount - total_spent
        progress = total_spent / limit_amount

        st.subheader(category)
        st.write(f"Budget: ₹{limit_amount:,.2f}")
        st.write(f"Spent: ₹{total_spent:,.2f}")
        st.progress(min(max(progress, 0.0), 1.0))
        st.write(f"Remaining: ₹{remaining:,.2f}")

        if remaining < 0:
            st.error("Budget exceeded!")
        elif progress >= 0.8:
            st.warning("You have used at least 80% of this budget.")
        else:
            st.success("You are within your budget.")
else:
    st.info("Save a budget to start tracking monthly spending.")

st.divider()
st.header("Unusual Expense Detection")
unusual = detect_unusual_expenses(df)
if unusual.empty:
    st.info(
        "No unusual expenses detected with the current rule. "
        "More transaction history may be needed."
    )
else:
    st.warning(f"{len(unusual)} potentially unusual expense(s) detected.")
    st.dataframe(unusual, use_container_width=True, hide_index=True)

if df.empty:
    st.info(
        "No transactions yet. Add your first transaction "
        "using the sidebar."
    )
    st.stop()

income = df.loc[df["Type"] == "Income", "Amount"].sum()
expenses = df.loc[df["Type"] == "Expense", "Amount"].sum()
balance = income - expenses

col1, col2, col3 = st.columns(3)
col1.metric("Total Income", f"₹{income:,.2f}")
col2.metric("Total Expenses", f"₹{expenses:,.2f}")
col3.metric("Net Balance", f"₹{balance:,.2f}")

# ---------------- FILTERS ----------------

st.subheader("Transaction History")

search = st.text_input("Search descriptions")

categories = sorted(df["Category"].unique().tolist())
selected_categories = st.multiselect(
    "Filter by category",
    categories,
    default=categories,
)

filtered = df[
    df["Category"].isin(selected_categories)
].copy()

if search.strip():
    filtered = filtered[
        filtered["Description"].str.contains(
            search.strip(), case=False, na=False
        )
    ]

st.dataframe(
    filtered.drop(columns=["id"]),
    use_container_width=True,
    hide_index=True,
)

# ---------------- CHARTS ----------------

expense_df = df[df["Type"] == "Expense"].copy()

if not expense_df.empty:
    left, right = st.columns(2)

    with left:
        category_totals = (
            expense_df.groupby("Category", as_index=False)["Amount"]
            .sum()
            .sort_values("Amount", ascending=False)
        )

        fig = px.pie(
            category_totals,
            names="Category",
            values="Amount",
            title="Expenses by Category",
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        expense_df["Month"] = pd.to_datetime(
            expense_df["Date"]
        ).dt.to_period("M").astype(str)

        monthly = (
            expense_df.groupby("Month", as_index=False)["Amount"]
            .sum()
            .sort_values("Month")
        )

        fig = px.bar(
            monthly,
            x="Month",
            y="Amount",
            title="Monthly Expense Trend",
        )
        st.plotly_chart(fig, use_container_width=True)

# ---------------- DELETE ----------------

st.subheader("Delete a Transaction")

transaction_options = {
    f'#{row.id} — {row.Description} — ₹{row.Amount:.2f}':
        int(row.id)
    for row in df.itertuples(index=False)
}

selected_transaction = st.selectbox(
    "Select a transaction",
    list(transaction_options.keys()),
)

if st.button("Delete selected transaction"):
    delete_transaction(transaction_options[selected_transaction])
    st.success("Transaction deleted.")
    st.rerun()

# ---------------- EXPORT ----------------

st.subheader("Export Data")

csv_data = df.drop(columns=["id"]).to_csv(index=False).encode("utf-8")

st.download_button(
    "Download transactions as CSV",
    data=csv_data,
    file_name="expense_report.csv",
    mime="text/csv",
)
