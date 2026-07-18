# R14-C Module (c) Closure Pin v1

{
  "generated_at_utc": "2026-07-13T18:40:37Z",
  "module_c_status": "PASSED",
  "notes": [
    "No module (d) content is adjudicated here.",
    "Module (c) closure pins bytes and registers gate finding only."
  ],
  "parameter_gate_evidence_base": {
    "artifact": "r14c_invalidation_rule_candidates_v1.json",
    "ex_set_b_symbol_count": 134,
    "explicit_non_optimization_statement": "No invalidation rule form is chosen by optimizing SANAM May-2025 owner window. SANAM owner-window rows are reported descriptively alongside every other symbol.",
    "scope": "EX_SET_B full evidence base (all symbols except Set B; Set A included within this scope)",
    "selection_status": "NO_CANDIDATE_SELECTED_IN_THIS_ARTIFACT"
  },
  "registered_parameter_gate_finding": {
    "evidence": {
      "module_c_test_evidence": "r14c_module_c_test_evidence_v4.json",
      "tijara_final_state": "RETIRED",
      "tijara_retire_count": 1
    },
    "finding_id": "R14C_FINDING_TIJARA_RETIRE_UNDER_DEFAULT",
    "statement": "TIJARA retired under module (c) default invalidation form; invalidation rule is load-bearing and deferred to parameter gate."
  },
  "reviewed_bytes": {
    "adaptive_base_geometry_py": {
      "path": "app/services/eagle_eye_v2/adaptive_base_geometry.py",
      "sha256": "c37d36afbb0b484cb06577793592d437708d973fc8ab3d6660bab6accce9256f"
    },
    "warmup_readiness_engine_py": {
      "note": "post-carry-in-fix bytes pinned",
      "path": "app/services/eagle_eye_v2/warmup_readiness_engine.py",
      "sha256": "52cbf36955ea1592f80ab6cedbdc07d3518c5e4ccc64d12c72c096199a4950b2"
    }
  },
  "version_id": "R14C_MODULE_C_CLOSURE_PIN_V1"
}
