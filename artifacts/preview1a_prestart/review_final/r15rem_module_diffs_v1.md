# R15-REM Module Diffs v1

Source: `git diff 7b0471c^ 7b0471c -- app/services/eagle_eye_v2/lifecycle_intent_router.py app/services/eagle_eye_v2/staged_position_policy.py app/services/eagle_eye_v2/adaptive_base_geometry.py`

Note: `staged_position_policy.py` had no diff in Part B; the pilot re-entry guard was implemented at the owning router boundary where same-symbol open-position feedback is available.

```diff
diff --git a/app/services/eagle_eye_v2/adaptive_base_geometry.py b/app/services/eagle_eye_v2/adaptive_base_geometry.py
index 3dae326..7aa9d12 100644
--- a/app/services/eagle_eye_v2/adaptive_base_geometry.py
+++ b/app/services/eagle_eye_v2/adaptive_base_geometry.py
@@ -15,6 +15,7 @@ BASE_REFERENCE_ADVANCE_OK = "BASE_REFERENCE_ADVANCE_OK"
 BASE_MIN_SESSIONS = "base_min_sessions"
 BASE_MAX_WIDTH_PCT = "base_max_width_pct"
 ATR_SQUEEZE_PCTILE = "atr_squeeze_pctile"
+UPWARD_RETIREMENT_MFE_THRESHOLD = "UPWARD_RETIREMENT_MFE_THRESHOLD"
 
 RULE_CLOSE_BELOW_BASE_LOW_N = "CLOSE_BELOW_BASE_LOW_N"
 RULE_CLOSE_BELOW_BASE_LOW_BY_ATR_X_N = "CLOSE_BELOW_BASE_LOW_BY_ATR_X_N"
@@ -116,7 +117,24 @@ class AdaptiveBaseGeometry:
                 transition["no_freeze_reason"] = ",".join(reasons) if reasons else "UNSPECIFIED"
         else:
             validity_state = str(base_reference.get("base_validity_state") or "").upper()
-            if validity_state == "RETIRED":
+            if validity_state == "RETIRED" and str(base_reference.get("base_retirement_reason") or "").startswith("RETIRED_SUPERSEDED_BY_MARKUP"):
+                base_reference = None
+                if freeze_eligible:
+                    base_reference = {
+                        "base_reference_id": f"{normalized_day_payload['symbol']}::{normalized_day_payload['trade_date']}::BASE01",
+                        "base_high_ref": float(high_ref),
+                        "base_low_ref": float(low_ref),
+                        "base_origin_date": normalized_day_payload["trade_date"],
+                        "base_validity_state": "VALID",
+                        "base_retirement_reason": "NONE",
+                        "invalidation_rule_form": invalidation_rule_form,
+                        "invalidation_rule_state": {},
+                    }
+                    base_state = "BASE_FROZEN"
+                    transition["base_freeze_event"] = "BASE_FROZEN"
+                else:
+                    base_state = "BASE_FORMING"
+            elif validity_state == "RETIRED":
                 base_state = "BASE_RETIRED"
             else:
                 base_reference["base_validity_state"] = "VALID"
@@ -137,6 +155,15 @@ class AdaptiveBaseGeometry:
                     atr_value=float(volatility_regime_state.get("atr_value") or 0.0),
                     flow_confirmed_progress=flow_confirmed_progress,
                 )
+                if not retire:
+                    upward_retirement = self._evaluate_upward_retirement(
+                        rule_params=volatility_regime_state,
+                        rule_state=next_rule_state,
+                        high_px=float(normalized_day_payload.get("high") or close_px),
+                        base_high_ref=float(base_reference.get("base_high_ref") or 0.0),
+                    )
+                    if upward_retirement is not None:
+                        retire, retire_reason, next_rule_state = upward_retirement
                 base_reference["invalidation_rule_state"] = next_rule_state
                 if retire:
                     base_reference["base_validity_state"] = "RETIRED"
@@ -278,6 +305,31 @@ class AdaptiveBaseGeometry:
         reason = f"{RULE_CLOSE_BELOW_BASE_LOW_N}(n={n_sessions})"
         return retire, reason, state
 
+    @staticmethod
+    def _evaluate_upward_retirement(
+        *,
+        rule_params: dict[str, Any],
+        rule_state: dict[str, Any],
+        high_px: float,
+        base_high_ref: float,
+    ) -> tuple[bool, str, dict[str, Any]] | None:
+        if UPWARD_RETIREMENT_MFE_THRESHOLD not in rule_params:
+            return None
+        threshold = float(rule_params[UPWARD_RETIREMENT_MFE_THRESHOLD])
+        state = dict(rule_state)
+        upward = dict(state.get("upward_retirement") or {})
+        age = int(upward.get("age_sessions") or 0) + 1
+        mfe = 0.0 if base_high_ref <= 0.0 else max(float(upward.get("mfe") or 0.0), (high_px / base_high_ref) - 1.0)
+        upward["age_sessions"] = age
+        upward["mfe"] = mfe
+        upward["threshold"] = threshold
+        state["upward_retirement"] = upward
+        retire = age <= 120 and mfe >= threshold
+        if not retire:
+            return False, "NONE", state
+        reason = f"RETIRED_SUPERSEDED_BY_MARKUP({UPWARD_RETIREMENT_MFE_THRESHOLD}={threshold},sessions<=120)"
+        return True, reason, state
+
     def _append_base_predicate(
         self,
         *,
diff --git a/app/services/eagle_eye_v2/lifecycle_intent_router.py b/app/services/eagle_eye_v2/lifecycle_intent_router.py
index 052db44..711da05 100644
--- a/app/services/eagle_eye_v2/lifecycle_intent_router.py
+++ b/app/services/eagle_eye_v2/lifecycle_intent_router.py
@@ -65,6 +65,7 @@ class LifecycleIntentRouter:
         candidate_state = str(candidate_intent.get("intent_state") or "INTENT_NONE")
         base_valid = str(base_state.get("base_state") or "").upper() in {"BASE_VALID", "BASE_FROZEN"}
         confirmed = str(confirmation_state.get("confirmation_state") or "").upper() == "CONFIRMED"
+        position_open = bool(current_state.get("active")) and str(current_state.get("state") or "").upper() == "POSITION_OPEN"
 
         deferred_active = False
         deferred_expiry_ok = True
@@ -105,7 +106,7 @@ class LifecycleIntentRouter:
             extension_pct=extension_pct,
         )
 
-        early_active = deferred_active and staged.get("cap_ok", True)
+        early_active = deferred_active and staged.get("cap_ok", True) and not position_open
         early_scale_ready = bool(staged.get("scale_action", {}).get("scale_ready"))
         confirmed_direct_ready = bool(
             candidate_state == "INTENT_FORMED"
@@ -137,6 +138,9 @@ class LifecycleIntentRouter:
             "time_stop_sessions": None if execution_state == "EXECUTE_CONFIRMED_DIRECT" else staged.get("early_entry", {}).get("time_stop_sessions", 60),
             "chase_advisory": staged.get("chase_band", {}),
         }
+        if position_open and deferred_active and execution_state == "NONE" and not veto_record.get("veto"):
+            execution_intent["no_path_reason"] = "POSITION_ALREADY_OPEN_FEEDBACK_SUPPRESSED_PILOT"
+            execution_intent["disposition_state"] = "NO_PATH_EXPLICIT"
 
         lifecycle_terms = {
             DEFERRED_INTENT_ACTIVE: deferred_active,
```
