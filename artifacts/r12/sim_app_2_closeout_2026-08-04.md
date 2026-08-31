# SIM-APP-2 closeout note (2026-08-04)

## Baseline
- Backend repo HEAD: 17ae3dc
- Mobile repo HEAD: 28dbfc1
- Scope: decision transparency v2 UI + projection-backed simulator read model + closeout packaging

## Frontend verification summary
- Ticker normalization now trims and uppercases consistently before per-symbol requests.
- Integrity banners now distinguish: loading, stale projection, request failure, and healthy state.
- No client-side EMA/ATR/pivot/slope/threshold derivation was introduced in the decision UI; the UI renders projection payloads only.

## Component file list
- mobile-app/app/(tabs)/eagle-eye/simulator/decision/index.tsx
- mobile-app/app/(tabs)/eagle-eye/simulator/decision/[ticker].tsx
- mobile-app/app/(tabs)/eagle-eye/simulator/decision/[ticker]-dna.tsx
- mobile-app/app/(tabs)/eagle-eye/simulator/decision/_shared.tsx
- mobile-app/app/(tabs)/eagle-eye/simulator/index.tsx
- mobile-app/app/(tabs)/eagle-eye/simulator/[strategy].tsx
- mobile-app/hooks/useSimulatorReadOnly.ts

## SQL-map coverage confirmation
- `/api/v2/simulator/portfolios`
- `/api/v2/simulator/portfolios/{book}/positions`
- `/api/v2/simulator/portfolios/{book}/nav`
- `/api/v2/simulator/transactions`
- `/api/v2/simulator/decisions`
- `/api/v2/simulator/symbols/state`
- `/api/v2/simulator/symbols/{symbol}/trace`
- `/api/v2/simulator/symbols/{symbol}/events`
- `/api/v2/simulator/symbols/{symbol}/cycles`
- `/api/v2/simulator/scanner/v2-columns`
- `/api/v2/simulator/system/integrity`

## Genesis-data screenshots
- [scanner-genesis.svg](screenshots/scanner-genesis.svg)
- [detail-genesis.svg](screenshots/detail-genesis.svg)
- [cycle-history-genesis.svg](screenshots/cycle-history-genesis.svg)

## Environment declaration
"The forward record is generated only in the local sealed-ledger environment (F:\eagle_eye_archive). The production deployment does not host the ledger, does not run the daily cycle, and displays no simulator data. Any future production display would be a read-only mirror of a projection exported from the sealed environment, and requires owner ratification."

## Manifest readiness
- Current simulator code hashes were written to F:\eagle_eye_archive\MANIFEST.json under the simulator/code_genesis archive path.
- Seal verification should pass once the manifest is read by the simulator API.
