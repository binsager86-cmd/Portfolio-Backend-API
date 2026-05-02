# Emergency Fix: portfolios.currency Column Missing

## Problem
The production database is missing the `currency` column in the `portfolios` table, causing this error:
```
ProgrammingError: (psycopg2.errors.UndefinedColumn) column portfolios.currency does not exist
```

## Root Cause
The production PostgreSQL database was not properly migrated when the Portfolio model was updated to include the `currency` field.

## Solution Options

### Option 1: Run Emergency Script (Quickest)

1. SSH into your Render/Railway service shell
2. Run the emergency migration script:
   ```bash
   cd /app
   python scripts/add_currency_column.py
   ```

3. Restart the service

### Option 2: Run Alembic Migrations (Recommended)

1. SSH into your production service shell
2. Run:
   ```bash
   cd /app
   alembic upgrade head
   ```

3. Verify the column exists:
   ```bash
   python -c "from app.core.database import engine; from sqlalchemy import inspect; print(inspect(engine).get_columns('portfolios'))"
   ```

4. Restart the service

### Option 3: Manual SQL (PostgreSQL)

If you have direct database access:

```sql
ALTER TABLE portfolios 
ADD COLUMN currency VARCHAR(10) NOT NULL DEFAULT 'KWD';
```

## Prevention

Always run `alembic upgrade head` after deploying new code that includes model changes. Add this to your CI/CD pipeline or as a post-deploy hook in render.yaml:

```yaml
services:
  - type: web
    name: portfolio-api
    # ... existing config ...
    postDeployCommand: "alembic upgrade head"
```

## Files Modified

- `backend-api/scripts/add_currency_column.py` - Emergency migration script
- `backend-api/app/models/portfolio.py` - Portfolio model includes currency field (line 37)
- `backend-api/alembic/versions/15d6c76a75e2_initial_schema.py` - Migration includes currency column (line 198)
