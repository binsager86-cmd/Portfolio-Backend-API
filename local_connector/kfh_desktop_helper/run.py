"""Run the KFH desktop helper.

Usage:
    python -m local_connector.kfh_desktop_helper.run

On first run this generates a pairing code and prints it once. Paste that
code into Saham's KFH connection settings (production or local) - it pairs
this specific helper installation with your browser and is never sent
anywhere except back to this helper, over loopback, on your own machine.

Configure the site this helper will accept requests from with the
KFH_HELPER_ALLOWED_ORIGIN environment variable (defaults to the production
Saham web app). Point it at http://localhost:8081 instead when testing
against a local Saham dev server.
"""

from __future__ import annotations

import os

from local_connector.kfh_desktop_helper.app import (
    LOCAL_HOST,
    LOCAL_PORT,
    create_desktop_helper_app,
)
from local_connector.kfh_desktop_helper.pairing import load_or_create_pairing_nonce

DEFAULT_ALLOWED_ORIGIN = "https://portfolioproapp.com"


def main() -> None:
    import uvicorn

    allowed_origin = os.environ.get("KFH_HELPER_ALLOWED_ORIGIN", DEFAULT_ALLOWED_ORIGIN)
    nonce = load_or_create_pairing_nonce()
    app = create_desktop_helper_app(nonce=nonce, allowed_origin=allowed_origin)

    print("=" * 64)
    print("Saham KFH Desktop Helper")
    print(f"Accepting requests from: {allowed_origin}")
    print("Paste this pairing code into Saham's KFH connection settings:")
    print()
    print(f"  {nonce}")
    print()
    print("Keep this window open while syncing your KFH statement.")
    print("=" * 64)

    uvicorn.run(app, host=LOCAL_HOST, port=LOCAL_PORT, access_log=False)


if __name__ == "__main__":
    main()
