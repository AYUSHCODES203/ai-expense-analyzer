import pandas as pd


def detect_unusual_expenses(df):
    result_columns = list(df.columns) + [
        "Category Median",
        "Detection Reason",
    ]
    required_columns = {"Date", "Type", "Category", "Amount"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            "Transactions must include Date, Type, Category, and Amount."
        )

    expenses = df.loc[df["Type"] == "Expense"].copy()
    expenses["Amount"] = pd.to_numeric(
        expenses["Amount"], errors="coerce"
    )
    expenses["_parsed_date"] = pd.to_datetime(
        expenses["Date"], errors="coerce"
    )
    expenses = expenses.dropna(
        subset=["Amount", "_parsed_date", "Category"]
    )
    if expenses.empty:
        return pd.DataFrame(columns=result_columns)

    expenses["_row_order"] = range(len(expenses))
    expenses = expenses.sort_values(
        ["_parsed_date", "_row_order"], kind="stable"
    )

    results = []
    for _, group in expenses.groupby("Category", sort=False):
        history = []
        for _, row in group.iterrows():
            amount = float(row["Amount"])
            if len(history) >= 5:
                median_amount = float(pd.Series(history).median())
                if median_amount > 0 and amount > 3 * median_amount:
                    record = row.drop(
                        labels=["_parsed_date", "_row_order"]
                    ).to_dict()
                    record["Category Median"] = median_amount
                    record["Detection Reason"] = (
                        "Amount exceeds 3 times the historical "
                        "category median"
                    )
                    results.append(record)
            history.append(amount)

    return pd.DataFrame(results, columns=result_columns)