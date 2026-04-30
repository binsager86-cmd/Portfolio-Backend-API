"""
Reset the Admin user's password via SQLAlchemy (supports SQLite + PostgreSQL).

Usage:
    python -m scripts.reset_admin_password [--username Admin]

Environment variables read (same as the running app):
    DATABASE_URL   — PostgreSQL connection string (production)
    DATABASE_PATH  — SQLite file path (development, default: ../dev_portfolio.db)
"""

from __future__ import annotations

import argparse
import getpass
import sys

import bcrypt

# Allow running as `python -m scripts.reset_admin_password` from backend-api/
if __name__ == "__main__" and __package__ is None:
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.database import SessionLocal
from app.models.user import User

BCRYPT_ROUNDS = 12


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reset an admin user's password.")
    parser.add_argument(
        "--username",
        default="Admin",
        help="Username whose password to reset (default: Admin)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    new_password = getpass.getpass(f"New password for '{args.username}': ")
    if not new_password:
        print("ERROR: Password cannot be empty.", file=sys.stderr)
        sys.exit(1)

    confirm = getpass.getpass("Confirm password: ")
    if new_password != confirm:
        print("ERROR: Passwords do not match.", file=sys.stderr)
        sys.exit(1)

    hashed = bcrypt.hashpw(
        new_password.encode("utf-8"),
        bcrypt.gensalt(rounds=BCRYPT_ROUNDS),
    ).decode("utf-8")

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == args.username).first()
        if not user:
            print(
                f"ERROR: No user with username '{args.username}' found in the database.",
                file=sys.stderr,
            )
            sys.exit(1)
        user.password_hash = hashed
        db.commit()
        print(f"Password updated for user '{user.username}' (id={user.id})")
    finally:
        db.close()


if __name__ == "__main__":
    main()
