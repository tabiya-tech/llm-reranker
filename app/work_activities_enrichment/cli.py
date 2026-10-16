"""Set ``onet_work_activities`` on ranked jobs (the batch's enrichment post-step, run on its own).

    python -m app.work_activities_enrichment.cli

Configured by the ``ENRICH_WA_*`` settings (``Settings.enrich_wa_*`` in ``app/config.py``).
"""

from __future__ import annotations

from app.config import get_settings
from app.logging_config import configure_logging
from app.server_dependencies.db_dependencies import standalone_db
from app.work_activities_enrichment.get_work_activities_enrichment_service import (
    build_work_activities_enrichment_service,
)


def main() -> int:
    configure_logging()
    settings = get_settings()
    with standalone_db(settings) as db:
        build_work_activities_enrichment_service(settings, db).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
