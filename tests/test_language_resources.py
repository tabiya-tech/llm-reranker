"""Guards for the per-language resource packs under ``resources/<language>/``.

Every registered language ships a pack; the reranker builds LLM1's skill menu and LLM2's
occupation line out of its CSVs. These tests keep the packs honest: files present, columns
the loaders read by name, the declared locale, and resolution through ``Settings``. A
truncated or wrongly exported drop then fails in CI instead of halfway through a Cloud Run
Job.

Only header rows are read, so this stays fast despite the CSVs being tens of MB.

Replaces the earlier ``test_argentina_resources.py``: the Spanish pack moved from
``resources/argentina/`` to ``resources/es/`` when language packs replaced the
country-specific directory.
"""

import csv

import pytest

from app.config import repo_root
from app.languages import LANGUAGE_REGISTRY, get_language_config

pytestmark = pytest.mark.unit

csv.field_size_limit(10_000_000)

RESOURCES = repo_root() / "resources"

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

PACK_CASES = [
    (language, filename) for language in LANGUAGE_REGISTRY for filename in sorted(REQUIRED_COLUMNS)
]


def _pack_dir(language: str):
    subdir = get_language_config(language).get("resources_subdir") or language
    return RESOURCES / subdir


def _header(path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as f:
        return next(csv.reader(f))


@pytest.mark.parametrize(("language", "filename"), PACK_CASES)
def test_taxonomy_file_is_present_and_non_trivial(language, filename):
    path = _pack_dir(language) / filename
    assert path.is_file(), f"missing {path}"
    # Anything this small is a failed export, not a taxonomy.
    assert path.stat().st_size > 100_000, f"{language}/{filename} looks truncated"


@pytest.mark.parametrize(("language", "filename"), PACK_CASES)
def test_required_columns_are_present(language, filename):
    header = set(_header(_pack_dir(language) / filename))
    missing = REQUIRED_COLUMNS[filename] - header
    assert not missing, f"{language}/{filename} is missing column(s) {sorted(missing)}"


def test_spanish_pack_declares_the_argentine_spanish_locale():
    """The drop must be the AR-es translation, not an accidental copy of the English one."""
    with (_pack_dir("es") / "model_info.csv").open(encoding="utf-8", newline="") as f:
        row = next(csv.DictReader(f))
    assert row["LOCALE"] == get_language_config("es")["taxonomy_locale"] == "AR-es"


@pytest.mark.parametrize("language", LANGUAGE_REGISTRY)
def test_settings_resolve_inside_the_language_pack(language, make_settings):
    s = make_settings(RERANKER_LANGUAGE=language)
    assert s.language == language
    subdir = get_language_config(language).get("resources_subdir")
    for attr in (
        "occupations_csv_path",
        "skills_csv_path",
        "occupation_to_skill_relations_path",
        "skill_hierarchy_csv_path",
        "job_attributes_schema_path",
    ):
        path = getattr(s, attr)
        assert path.is_file(), f"{language}: {attr} did not resolve to a file: {path}"
        assert path.parent.name == subdir, (
            f"{language}: {attr} resolved outside resources/{subdir} ({path})"
        )


def test_argentina_country_selects_the_spanish_pack(make_settings):
    """The GCP Workflow sets TARGET_COUNTRY; the language follows from it."""
    s = make_settings(TARGET_COUNTRY="argentina")
    assert s.target_country == "argentina"
    assert s.country_name == "Argentina"
    assert s.language == "es"
    assert s.occupations_csv_path.parent.name == "es"


def test_relations_graph_matches_the_english_esco_columns():
    """The ESCO half of the relations graph is shared, so the columns must line up.

    A translated pack differs from English only in its label text and its
    ``localoccupation`` rows; if the two headers diverge, one of the two drops came
    from a different taxonomy release and the reranker would silently lose columns.
    """
    english_header = set(_header(_pack_dir("en") / "occupation_to_skill_relations.csv"))
    for language in LANGUAGE_REGISTRY:
        if language == "en":
            continue
        header = set(_header(_pack_dir(language) / "occupation_to_skill_relations.csv"))
        assert english_header <= header, f"{language} relations file is missing English columns"


def test_language_packs_share_one_skill_id_space():
    """Every pack must describe the same skills, so ids stay comparable downstream.

    ``UUIDHISTORY``'s oldest entry is stable across taxonomy locales; the matching
    service and the classifier both rely on that to resolve a Spanish-linked skill to the
    same entity as an English one. If a pack is built from a different taxonomy release
    this diverges, and cross-language matching silently degrades.
    """

    def oldest_uuids(language: str) -> set[str]:
        path = _pack_dir(language) / "skills.csv"
        with path.open(encoding="utf-8", newline="") as f:
            return {
                (row["UUIDHISTORY"] or "").strip().split("\n")[-1].strip()
                for row in csv.DictReader(f)
            }

    english = oldest_uuids("en")
    assert len(english) > 10_000
    for language in LANGUAGE_REGISTRY:
        if language == "en":
            continue
        other = oldest_uuids(language)
        assert other == english, (
            f"{language} skills.csv does not share the English skill id space "
            f"({len(other - english)} extra, {len(english - other)} missing)"
        )
