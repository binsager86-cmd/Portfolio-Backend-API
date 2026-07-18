# R14-B Module (a) v1 Defect Note v1

{
  "affected_artifacts": [
    "r14b_module_a_implementation_report_v1.md",
    "r14b_module_a_interface_conformance_v1.json",
    "r14b_module_a_test_evidence_v1.json"
  ],
  "defect_record": {
    "classification": "MODERATE/REPRODUCIBILITY",
    "fact": "In-place edit occurred on a previously sealed generator script (v1).",
    "script_path": "scripts/r14b_module_a_write_path_harness_v1.py",
    "script_sha256_current": "da1402b0f72a642e4329e91626151abe9fe6ac5ef3bb6db4e9da066e29e2d3a1",
    "v1_evidence_version_id": "R14B_MODULE_A_TEST_EVIDENCE_V1",
    "v1_harness_lines": [
      "R14B_MODULE_A_HARNESS_START",
      "DDL_APPLIED count=16 dialect=sqlite",
      "WRITE_READ_OK",
      "APPEND_ONLY_TRIGGER_CHECK_COMPLETE",
      "SIDECAR_CHAIN_EMITTED trade_date=2026-07-13 chain_hash=c15b571c17f094c921f2be923a0075a12efafdea9be3fb5f397355aa532071a0",
      "INTERFACE_CONFORMANCE pass=True"
    ]
  },
  "frozen_history_notice": "r14b_module_a_write_path_harness_v1.py was edited in place after initial seal; do not revert history.",
  "generated_at_utc": "2026-07-13T18:08:32Z",
  "remediation_instruction": "Supersede with r14b_module_a_write_path_harness_v2.py and emit fresh _v2 boundary artifacts.",
  "version_id": "R14B_MODULE_A_V1_DEFECT_NOTE_V1"
}
