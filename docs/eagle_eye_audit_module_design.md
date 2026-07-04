# Eagle Eye Audit Module Design

## Objective

Provide a full governance layer for Eagle Eye so teams can:
- Record immutable audit events for operations/config/model changes
- Manage concept changes through a review lifecycle
- Preserve status transition history for compliance and investigations
- Produce operational summaries for audit reporting

## Architecture

The module is split into three layers:

1. API Layer
- File: `app/api/v1/eagle_eye_audit.py`
- Responsibility: endpoint contracts, auth/authorization, response envelopes

2. Service Layer
- File: `app/services/eagle_eye/audit_service.py`
- Responsibility: schema bootstrap, business rules, transitions, aggregation

3. Schema Layer
- File: `app/schemas/eagle_eye_audit.py`
- Responsibility: request payload validation and strict field semantics

## Data Design

### Table: ee_audit_events
Append-only event journal for Eagle Eye actions.

Key fields:
- id (PK)
- event_time (unix timestamp)
- actor_user_id / actor_username
- action
- entity_type / entity_id
- change_type (operation/config/model/workflow/data)
- before_state / after_state (JSON string)
- rationale
- risk_level (low/medium/high/critical)
- trace_id
- source (api/scheduler/manual/system)
- metadata_json
- concept_version
- requires_follow_up (0/1)

Indexes:
- idx_ee_audit_event_time
- idx_ee_audit_entity
- idx_ee_audit_action

### Table: ee_change_requests
Primary workflow record for concept changes.

Key fields:
- id (PK)
- created_at / updated_at
- requested_by_user_id / requested_by_username
- title / description
- target_area
- change_category
- proposed_payload_json
- status
- reviewed_by_user_id / reviewed_by_username
- review_notes
- approved_at / rejected_at
- effective_from / effective_to
- supersedes_request_id

Indexes:
- idx_ee_change_status
- idx_ee_change_created

### Table: ee_change_status_history
Immutable workflow transition history.

Key fields:
- id (PK)
- request_id
- changed_at
- changed_by_user_id / changed_by_username
- old_status / new_status
- note

Index:
- idx_ee_change_hist_req

## Lifecycle Rules

Statuses:
- draft
- proposed
- needs_changes
- approved
- rejected
- implemented
- cancelled

Allowed transitions:
- draft -> proposed, cancelled
- proposed -> needs_changes, approved, rejected, cancelled
- needs_changes -> proposed, cancelled
- approved -> implemented, cancelled
- rejected -> (terminal)
- implemented -> (terminal)
- cancelled -> (terminal)

Authorization:
- Authenticated users can create/list/view
- Admin required for review endpoint
- Admin required for implemented/cancelled transitions from transition endpoint

## API Endpoints

Base prefix: `/api/v1/eagle-eye/audit`

- GET `/design`
- POST `/events`
- GET `/events`
- POST `/change-requests`
- GET `/change-requests`
- GET `/change-requests/{request_id}`
- POST `/change-requests/{request_id}/review`
- POST `/change-requests/{request_id}/transition`
- GET `/summary`

## Audit/Compliance Notes

- Service initializes schema lazily (`ensure_schema`) to support fresh dev/prod DBs.
- Status transitions are validated centrally in service layer.
- History row is written for every accepted status change.
- Summary endpoint includes:
  - total events in window
  - counts by risk level
  - counts by request status
  - recent high/critical events

## Change Management Concept

Recommended internal policy:
1. Author creates request in `draft`.
2. Author promotes to `proposed` when scope/impact is complete.
3. Reviewer sets `approved`, `needs_changes`, or `rejected`.
4. Implementation owner moves `approved` to `implemented` after deployment.
5. Every key operation writes an `ee_audit_events` entry referencing request id in metadata.

## Integration Checklist

- Router is included by `app/api/v1/__init__.py`.
- Endpoint contracts are visible in OpenAPI docs.
- Existing auth model (`TokenData`) is reused.
- No migration runner dependency required; tables are created if missing.
