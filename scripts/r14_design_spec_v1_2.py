from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "artifacts" / "preview1a_prestart" / "review_final"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    spec = read_json(REVIEW / "r14_design_spec_v1_1.json")
    m5 = read_json(REVIEW / "r13_m5_liquidity_forensic_v1.json")

    spec['version_id'] = 'R14_DESIGN_SPEC_V1_2'
    spec['supersedes'] = 'R14_DESIGN_SPEC_V1_1'
    spec['governing_constraints']['chase_policy'] = 'TOLERANT_WITH_ADVISORY'
    spec['governing_constraints']['advisory_extension_band_parameter_only'] = True
    spec['architecture_blueprint']['confirmation_core'] = 'Proposal A accumulation-window flow confirmation with tolerant-with-advisory chase policy'
    spec['state_machine']['named_predicate_terms']['confirmation'] = spec['state_machine']['named_predicate_terms']['confirmation'] + ['CHASE_EXTENSION_ADVISORY_ONLY', 'CURRENT_DAY_LIQUIDITY_OK', 'LIQUIDITY_CONTEXT_OK']
    spec['telemetry_schema']['daily_term_row'] = spec['telemetry_schema']['daily_term_row'] + ['extension_pct_vs_current_valid_reference', 'chase_advisory_flag', 'current_day_value_kwd', 'trailing_liquidity_context_value']
    spec['telemetry_schema']['execution_outcome_row'] = spec['telemetry_schema']['execution_outcome_row'] + ['chase_advisory_emitted', 'chase_advisory_extension_pct']
    spec['finding_response_map']['F9'] = 'Solved by treating current and arriving liquidity as decisive execution evidence, with trailing baseline used as context rather than sole veto authority.'
    spec['governing_constraints']['trailing_liquidity_never_sole_veto_when_current_day_exceeds_execution_size_parameter'] = True
    spec['r15_acceptance_criteria']['SANAM'] = spec['r15_acceptance_criteria']['SANAM'] + [
        'The 2025-05-18 session must produce a confirmed entry across all layers.',
        'No Set A high-volume day may be blocked by trailing-liquidity alone when same-day value exceeds the execution-size parameter.',
    ]
    spec['r15_acceptance_criteria']['BPCC'] = spec['r15_acceptance_criteria']['BPCC'] + [
        'Extended entries beyond the current valid reference must emit advisory telemetry rather than hard rejection when flow confirmation holds.',
    ]
    spec['m5_forensic_note'] = {
        'source': 'artifacts/preview1a_prestart/review_final/r13_m5_liquidity_forensic_v1.json',
        'status': m5['f9']['status'],
        'statement': m5['f9']['statement'],
    }

    md = []
    md.append('# R14 Design Spec v1.2')
    md.append('')
    md.append('Supersedes: R14 Design Spec v1.1')
    md.append('')
    md.append('Amendment focus:')
    md.append('- Encode owner ruling CHASE_POLICY = TOLERANT_WITH_ADVISORY.')
    md.append('- Entry beyond the chase reference is permitted when flow confirmation holds; extended entries emit advisory telemetry only.')
    md.append('- Encode the M5/F9 result: current and arriving liquidity must be weighted directly; trailing baseline is context, never sole veto authority.')
    md.append('')
    md.append('New/strengthened design principles:')
    md.append('- Chase policy is tolerant with advisory, not hard rejection, when confirmation core is satisfied.')
    md.append('- Advisory telemetry records extension_pct versus the current valid reference and references a named advisory-band parameter.')
    md.append('- Execution-liquidity assessment must combine current-day liquidity, short-horizon liquidity trend, and trailing context; trailing context cannot be sole veto authority.')
    md.append('')
    md.append('State-machine additions:')
    md.append(f"- confirmation terms: {spec['state_machine']['named_predicate_terms']['confirmation']}")
    md.append('')
    md.append('Telemetry additions:')
    md.append(f"- daily_term_row: {spec['telemetry_schema']['daily_term_row']}")
    md.append(f"- execution_outcome_row: {spec['telemetry_schema']['execution_outcome_row']}")
    md.append('')
    md.append('Updated finding-response map:')
    md.append(f"- F9: {spec['finding_response_map']['F9']}")
    md.append('')
    md.append('M5 forensic note:')
    md.append(f"- {spec['m5_forensic_note']['status']}: {spec['m5_forensic_note']['statement']}")
    md.append('')
    md.append('Updated R15 acceptance criteria:')
    for sym in ['SANAM','BPCC']:
        md.append(f"- {sym}: {spec['r15_acceptance_criteria'][sym]}")
    md.append('')
    md.append('R14-B and R15 remain NOT AUTHORIZED.')
    md.append('')

    out_json = REVIEW / 'r14_design_spec_v1_2.json'
    out_md = REVIEW / 'r14_design_spec_v1_2.md'
    out_json.write_text(json.dumps(spec, ensure_ascii=True, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    out_md.write_text('\n'.join(md), encoding='utf-8')
    print('R14_DESIGN_SPEC_V1_2_COMPLETE')
    print('json_sha256', sha256_file(out_json))
    print('md_sha256', sha256_file(out_md))


if __name__ == '__main__':
    main()
