# Local Postgres Development Setup

Eagle Eye dev uses a dedicated Postgres container on port **5433** (avoids conflict with any local Postgres on 5432).

## Start the container

```bash
# From the backend-api/ directory
docker compose up -d postgres-dev
```

## Check it's running

```bash
docker ps | grep eagle_eye_postgres_dev
# Should show: Up ... 0.0.0.0:5433->5432/tcp
```

## Connect with psql (optional)

```bash
docker exec -it eagle_eye_postgres_dev psql -U eagle_eye_dev -d eagle_eye_dev
```

## Connection string (add to your .env)

```
DATABASE_URL=postgresql://eagle_eye_dev:eagle_eye_dev_password@localhost:5433/eagle_eye_dev
```

## Stop (keeps data)

```bash
docker compose stop postgres-dev
```

## Reset (drops all data — fresh schema on next start)

```bash
docker compose down postgres-dev
docker volume rm backend-api_eagle_eye_pgdata
docker compose up -d postgres-dev
```

## Switching between SQLite and Postgres

| Mode     | `.env` setting                                                               |
|----------|------------------------------------------------------------------------------|
| SQLite   | `DATABASE_PATH=../dev_portfolio.db` (comment out `DATABASE_URL`)             |
| Postgres | `DATABASE_URL=postgresql://eagle_eye_dev:eagle_eye_dev_password@localhost:5433/eagle_eye_dev` |

The app reads `DATABASE_URL` first. If it starts with `postgresql://`, Postgres mode is active. Otherwise SQLite is used.
