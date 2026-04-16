"""
Reset the Admin user's password in the production database.
Usage: python _reset_admin_password.py
"""
import sqlite3
import bcrypt
import os
import getpass

NEW_PASSWORD = getpass.getpass("Enter new admin password: ")
if not NEW_PASSWORD:
    print("ERROR: Password cannot be empty.")
    raise SystemExit(1)
BCRYPT_ROUNDS = 12

# Use DATABASE_PATH env var if set, otherwise default to local dev db
db_path = os.environ.get("DATABASE_PATH", os.path.join(os.path.dirname(__file__), "..", "dev_portfolio.db"))

hashed = bcrypt.hashpw(NEW_PASSWORD.encode("utf-8"), bcrypt.gensalt(rounds=BCRYPT_ROUNDS)).decode("utf-8")

conn = sqlite3.connect(db_path)
cur = conn.cursor()

cur.execute("SELECT id, username FROM users WHERE username = 'Admin'")
row = cur.fetchone()

if not row:
    print("ERROR: No user with username 'Admin' found in the database.")
    conn.close()
    raise SystemExit(1)

cur.execute("UPDATE users SET password_hash = ? WHERE username = 'Admin'", (hashed,))
conn.commit()
print(f"Password updated for user '{row[1]}' (id={row[0]})")
conn.close()
