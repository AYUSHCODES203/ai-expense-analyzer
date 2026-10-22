
import sqlite3
import pandas as pd

DB_NAME = "expenses.db"


def get_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def initialize_database():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_date TEXT NOT NULL,
                description TEXT NOT NULL,
                transaction_type TEXT NOT NULL
                    CHECK(transaction_type IN ('Income', 'Expense')),
                category TEXT NOT NULL,
                amount REAL NOT NULL CHECK(amount > 0)
            )
        """)


def initialize_budgets():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS budgets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                month TEXT NOT NULL,
                category TEXT NOT NULL,
                limit_amount REAL NOT NULL CHECK(limit_amount > 0),
                UNIQUE(month, category)
            )
        """)


def add_transaction(date, description, transaction_type,
                    category, amount):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO transactions (
                transaction_date, description,
                transaction_type, category, amount
            )
            VALUES (?, ?, ?, ?, ?)
        """, (
            str(date), description.strip(),
            transaction_type, category, float(amount)
        ))


def get_transactions():
    with get_connection() as conn:
        return pd.read_sql_query("""
            SELECT id, transaction_date AS Date,
                   description AS Description,
                   transaction_type AS Type,
                   category AS Category,
                   amount AS Amount
            FROM transactions
            ORDER BY transaction_date DESC, id DESC
        """, conn)


def save_budget(month, category, limit_amount):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO budgets (month, category, limit_amount)
            VALUES (?, ?, ?)
            ON CONFLICT(month, category)
            DO UPDATE SET limit_amount = excluded.limit_amount
        """, (str(month), str(category), float(limit_amount)))


def get_budgets(month):
    with get_connection() as conn:
        return pd.read_sql_query("""
            SELECT category AS Category,
                   limit_amount AS Budget
            FROM budgets
            WHERE month = ?
            ORDER BY category
        """, conn, params=(str(month),))


def delete_transaction(transaction_id):
    with get_connection() as conn:
        conn.execute(
            "DELETE FROM transactions WHERE id = ?",
            (int(transaction_id),)
        )
