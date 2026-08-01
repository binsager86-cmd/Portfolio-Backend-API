from __future__ import annotations

import json

from app.services.eagle_eye_v2.simulator.projection import POSTGRES_GRANTS, project_simulator_ledger


def main() -> None:
    result = project_simulator_ledger()
    print("SIM_PROJECTION_VERIFY|" + json.dumps(result.to_dict(), sort_keys=True))
    print("SIM_PROJECTION_GRANTS|" + POSTGRES_GRANTS.replace("\n", " "))


if __name__ == "__main__":
    main()
