"""Guards for the Argentine Spanish taxonomy shipped under ``resources/argentina/``.

Argentina reuses every code path — only ``TARGET_COUNTRY`` and the four taxonomy CSV
env vars change (see the Argentina block in ``.env.example``). These tests keep that
configuration honest: the files must be present, resolve through ``Settings``, and
carry the columns the loaders in ``utils/`` read by name. A truncated or wrongly
exported drop then fails in CI instead of halfway through a Cloud Run Job.

Only header rows are read, so this stays fast despite the CSVs being tens of MB.
"""

import csv

import pytest

from app.config import repo_root

pytestmark = pytest.mark.unit

ARGENTINA_DIR = repo_root() / "resources" / "argentina"

# Columns each loader looks up by name.
REQUIRED_COLUMNS = {
    # utils/occupation_to_skills_mapping.load_occupations_lookup
    "occupations.csv": {"ID", "ORIGINURI", "UUIDHISTORY", "PREFERREDLABEL"},
    # utils/uuid_to_tabiya_id.load_skills_uuid_tabiya_map
    "skills.csv": {"ID", "ORIGINURI", "UUIDHISTORY", "PREFERREDLABEL"},
    # utils/occupation_to_skills_mapping.load_occupation_skill_relations
    "occupation_to_skill_relations.csv": {
        "OCCUPATIONTYPE",
        "OCCUPATIONID",
        "RELATIONTYPE",
        "SKILLID",
        "SIGNALLINGVALUELABEL",
    },
    # utils/skill_groups._load_skill_to_groups_map
    "skill_hierarchy.csv": {
        "PARENTOBJECTTYPE",
        "CHILDOBJECTTYPE",
        "PARENTID",
        "CHILDID",
    },
}

# Env var -> resources-relative path, as documented in .env.example.
ENV_OVERRIDES = {
    "OCCUPATIONS_CSV": "argentina/occupations.csv",
    "SKILLS_CSV": "argentina/skills.csv",
    "OCCUPATION_TO_SKILL_RELATIONS_CSV": "argentina/occupation_to_skill_relations.csv",
    "SKILL_HIERARCHY_CSV": "argentina/skill_hierarchy.csv",
}


def _header(filename: str) -> list[str]:
    with (ARGENTINA_DIR / filename).open(encoding="utf-8", newline="") as f:
        return next(csv.reader(f))


@pytest.mark.parametrize("filename", sorted(REQUIRED_COLUMNS))
def test_taxonomy_file_is_present_and_non_trivial(filename):
    path = ARGENTINA_DIR / filename
    assert path.is_file(), f"missing {path}"
    # Anything this small is a failed export, not a taxonomy.
    assert path.stat().st_size > 100_000, f"{filename} looks truncated"


@pytest.mark.parametrize(("filename", "columns"), sorted(REQUIRED_COLUMNS.items()))
def test_required_columns_are_present(filename, columns):
    header = set(_header(filename))
    missing = columns - header
    assert not missing, f"{filename} is missing column(s) {sorted(missing)}"


def test_model_info_declares_the_argentine_spanish_locale():
    """The drop must be the AR-es translation, not an accidental copy of the default."""
    with (ARGENTINA_DIR / "model_info.csv").open(encoding="utf-8", newline="") as f:
        row = next(csv.DictReader(f))
    assert row["LOCALE"] == "AR-es"


def test_settings_resolve_the_argentina_overrides(make_settings):
    s = make_settings(TARGET_COUNTRY="argentina", **ENV_OVERRIDES)
    assert s.target_country == "argentina"
    assert s.country_name == "Argentina"
    for attr in (
        "occupations_csv_path",
        "skills_csv_path",
        "occupation_to_skill_relations_path",
        "skill_hierarchy_csv_path",
    ):
        path = getattr(s, attr)
        assert path.is_file(), f"{attr} did not resolve to a file: {path}"
        assert path.parent.name == "argentina", f"{attr} resolved outside resources/argentina"


def test_relations_graph_matches_the_default_esco_columns():
    """The ESCO half of the relations graph is shared, so the columns must line up.

    Argentina's file differs from the default only in its Spanish labels and its
    ``localoccupation`` rows; if the two headers diverge, one of the two drops came
    from a different taxonomy release and the reranker would silently lose columns.
    """
    default_header = None
    with (repo_root() / "resources" / "occupation_to_skill_relations.csv").open(
        encoding="utf-8", newline=""
    ) as f:
        default_header = next(csv.reader(f))
    argentina_header = _header("occupation_to_skill_relations.csv")
    assert set(default_header) <= set(argentina_header)
