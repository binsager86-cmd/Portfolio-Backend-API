Backfill guard: YES — enforced in app/services/eagle_eye/simulator.py and tools/backfill_simulator.py via ratings_history.created_at <= simulated date.
Recompute-raises: YES — missing/late ratings_history rows raise instead of using ee_ratings_cache or recomputing past ratings.
Daily snapshot to ratings_history: YES — app/services/eagle_eye/scheduler_service.py calls snapshot_ratings_history(run_date) after R11 ratings refresh.
Append-only trigger: YES — ratings_history update/delete blocked in app/services/eagle_eye/store.py.
Live-from date: 2026-07-19 forward; earlier dates only run if an honest ratings_history row already exists with created_at on-or-before that date.
