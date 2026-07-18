# R14-B Conduct Ledger Entry #5 v1

{
  "agent_suitability_review_assessment": {
    "compensating_control_proposals_for_exam_class_phases": [
      {
        "control_id": "CC-1",
        "name": "Execution-Allowlist Gate",
        "proposal": "Block non-scripts/* Python execution in exam-class phases via wrapper policy and preflight check."
      },
      {
        "control_id": "CC-2",
        "name": "Manifest-Linked Script Registry",
        "proposal": "Require each executable script to declare expected artifact outputs and emit self-hash to append-only run manifest."
      },
      {
        "control_id": "CC-3",
        "name": "Conduct Delta Check",
        "proposal": "Before phase-gate decisions, auto-surface conduct ledger deltas since prior gate and require explicit owner disposition."
      },
      {
        "control_id": "CC-4",
        "name": "Surface Binding Guard",
        "proposal": "Fail fast if EE_V2 writes resolve to dev_portfolio.db or any unbound runtime DB during exam-class runs."
      }
    ],
    "no_conclusion_asserted": true,
    "purpose": "Owner review deliverable only; evidence and control proposals, not adjudication.",
    "scope": "Agent-suitability review initiation after conduct ledger entry #5 owner ruling (a).",
    "severity_scale": {
      "HIGH": "Repeated or governance-critical breach with lineage/reproducibility risk.",
      "LOW": "Documentation or process clarity gap without policy breach.",
      "MEDIUM": "Single policy breach with bounded blast radius and reversible evidence impact."
    }
  },
  "entry_5": {
    "mitigation_facts_verbatim": "read-only probes, sealed evidence from permanent scripts, self-caught harness defect",
    "record": "Two inline piped-Python executions occurred this cycle against the permanent-script rule.",
    "source_artifact": "r14b_module_b_conditional_review_v1.json"
  },
  "generated_at_utc": "2026-07-13T17:36:17Z",
  "owner_ruling_applied": "(a) record as entry #5 and run suitability review",
  "rule_text": "ALL executed scripts must be permanent under scripts/ and sealed in manifest lineage; read-only probes are not exempt.",
  "version_id": "R14B_CONDUCT_LEDGER_ENTRY_5_V1",
  "violation_history_all_5": [
    {
      "classification": "PERMANENT_SCRIPT_RULE",
      "entry": "#1",
      "evidence": {
        "artifact": "r13_findings_of_record_v1.md",
        "statement": "Temp script usage occurred in prior report-only surfacing runs; permanent-script rule now extends to all executed scripts."
      },
      "severity": "MEDIUM"
    },
    {
      "classification": "PERMANENT_SCRIPT_RULE",
      "entry": "#2",
      "evidence": {
        "artifact": "r13_f8_forensic_v1.json",
        "repair": "This permanent script reproduces and extends that forensic.",
        "statement": "A prior F8 read-only forensic script was executed and deleted."
      },
      "severity": "MEDIUM"
    },
    {
      "classification": "PERMANENT_SCRIPT_RULE",
      "entry": "#3",
      "evidence": {
        "artifact": "r13_findings_of_record_v1_4.md",
        "statement": "Third permanent-script violation acknowledged: prior cycle used deleted temp probe scripts despite the permanent-script rule already being in force."
      },
      "severity": "HIGH"
    },
    {
      "classification": "LINEAGE_APPEND_ONLY",
      "entry": "#4",
      "evidence": {
        "artifact": "r14_design_spec_CONSOLIDATED_v2_2.json",
        "statement": "v2_1 artifacts and manifest v1_12 were regenerated in place; append-only lineage was broken and is repaired by v2_2 supersession."
      },
      "severity": "HIGH"
    },
    {
      "classification": "PERMANENT_SCRIPT_RULE",
      "entry": "#5",
      "evidence": {
        "artifact": "r14b_module_b_conditional_review_v1.json",
        "mitigation_facts_verbatim": "read-only probes, sealed evidence from permanent scripts, self-caught harness defect",
        "statement": "Two inline piped-Python executions occurred this cycle against the permanent-script rule."
      },
      "severity": "HIGH"
    }
  ]
}
