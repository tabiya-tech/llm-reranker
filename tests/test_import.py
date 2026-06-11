"""Smoke tests: modules can be imported without runtime errors.

These tests do not require MongoDB, Gemini API, or any external services.
They simply verify the codebase loads cleanly so CI can run before a fuller
test suite is added.
"""


def test_import_app_config():
    from app import config  # noqa: F401


def test_import_app_pipeline():
    from app import pipeline  # noqa: F401


def test_import_app_ranked_job_schema():
    from app import ranked_job_schema  # noqa: F401

    assert ranked_job_schema.PIPELINE_SCHEMA_VERSION == "32"


def test_import_api():
    import api  # noqa: F401


def test_import_utils_classifier_linked_entities():
    from utils import classifier_linked_entities  # noqa: F401
