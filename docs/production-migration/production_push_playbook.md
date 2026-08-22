# Production Push Playbook (PostgreSQL)

This playbook is for pushing backend-api to production with PostgreSQL as the source of truth.

## 1. Set production environment

Create `.env.production` on the server (do not commit secrets):

```env
ENVIRONMENT=production
DATABASE_URL=postgresql://USER:PASSWORD@HOST:5432/DBNAME
SECRET_KEY=<long-random-secret>
CRON_SECRET_KEY=<long-random-secret>
CORS_ORIGINS=https://your-real-frontend-domain
LEGACY_PLAINTEXT_LOGIN=false
```

## 2. Pre-deploy checks (local or staging)

From backend-api root:

```bash
python scripts/production_preflight.py
```

Expected result: `Preflight passed: required checks are green.`

If you are intentionally testing in SQLite mode only, use:

```bash
python scripts/production_preflight.py --allow-sqlite
```

## 3. Database migration discipline

Run schema migration before serving traffic:

```bash
alembic upgrade head
```

Verify migration state:

```bash
alembic current
```

## 4. Deploy

Use your platform deploy flow (Render blueprint or VPS process manager).

For Render, `render.yaml` already includes:
- managed PostgreSQL database wiring
- `postDeployCommand: alembic upgrade head`
- health check path `/health`

## 5. Post-deploy verification

Run these checks immediately after deployment:

1. `GET /health` returns success
2. login endpoint returns success for known test user
3. fundamental statement endpoint returns data (spot-check AMZN/NKE)
4. scheduler starts without DB connectivity errors

## 6. Rollback rule

If required checks fail or post-deploy checks fail:

1. Roll back to previous release
2. Re-run `python scripts/production_preflight.py`
3. Fix root cause before re-deploy

## 7. Non-negotiable production rules

1. Never deploy with placeholder secrets.
2. Never run production in SQLite mode.
3. Never bypass Alembic for schema changes.
4. Never push without preflight + post-deploy health verification.
