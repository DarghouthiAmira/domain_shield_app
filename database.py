import sqlite3
from datetime import datetime

DB_NAME = "app_history.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS domain_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            domain TEXT NOT NULL,
            risk_score REAL NOT NULL,
            status TEXT NOT NULL,
            entropy REAL NOT NULL,
            analyzed_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def save_analysis(domain: str, risk_score: float, status: str, entropy: float):
    conn = sqlite3.connect(DB_NAME)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        "INSERT INTO domain_logs (domain, risk_score, status, entropy, analyzed_at) VALUES (?, ?, ?, ?, ?)",
        (domain, round(risk_score, 2), status, round(entropy, 3), timestamp)
    )
    conn.commit()
    conn.close()

def get_history(limit: int = 50):
    conn = sqlite3.connect(DB_NAME)
    rows = conn.execute(
        "SELECT id, domain, risk_score, status, entropy, analyzed_at FROM domain_logs ORDER BY id DESC LIMIT ?",
        (limit,)
    ).fetchall()
    conn.close()
    return rows