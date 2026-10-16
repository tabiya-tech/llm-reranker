"""Smoke tests: modules can be imported without runtime errors.

These tests do not require MongoDB, Gemini API, or any external services.
They simply verify the codebase loads cleanly so CI can run before a fuller
test suite is added.
"""


def test_import_app_config():
    from app import config  # noqa: F401


def test_import_use_case_modules():
    from app.batch import service as batch_service  # noqa: F401
    from app.job_ranking import service as job_ranking_service  # noqa: F401


def test_import_app_ranked_document():
    from app.job_ranking import ranked_document

    assert ranked_document.PIPELINE_SCHEMA_VERSION == "32"


def test_import_main():
    import main  # noqa: F401


def test_import_api():
    import api  # noqa: F401


def test_import_utils_classifier_linked_entities():
    from utils import classifier_linked_entities  # noqa: F401
