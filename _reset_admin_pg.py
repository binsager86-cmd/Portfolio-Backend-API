"""
Reset the Admin user's password — works on BOTH SQLite and PostgreSQL
because it goes through SQLAlchemy / the app's User model.

Usage (DigitalOcean App Platform Console):
    python _reset_admin_pg.py

Reads DATABASE_URL (Postgres) or DATABASE_PATH (SQLite) from env, same
as the running app.
"""
import getpass
import sys

import bcrypt

from app.core.database import SessionLocal
from app.models.user import User

USERNAME = "Admin"
BCRYPT_ROUNDS = 12

new_password = getpass.getpass(f"Enter new password for user '{USERNAME}': ")
if not new_password:
    print("ERROR: Password cannot be empty.")
    sys.exit(1)
confirm = getpass.getpass("Confirm password: ")
if new_password != confirm:
    print("ERROR: Passwords do not match.")
    sys.exit(1)

hashed = bcrypt.hashpw(
    new_password.encode("utf-8"),
    bcrypt.gensalt(rounds=BCRYPT_ROUNDS),
).decode("utf-8")

db = SessionLocal()
try:
    user = db.query(User).filter(User.username == USERNAME).first()
    if not user:
        print(f"ERROR: No user with username '{USERNAME}' found.")
        sys.exit(1)
    user.password_hash = hashed
    db.commit()
    print(f"Password updated for user '{user.username}' (id={user.id})")
finally:
    db.close()
