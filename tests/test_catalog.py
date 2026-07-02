"""Tests for app.catalog: lookups, canonical recommendation building, and the
anti-hallucination post-filter (CLAUDE.md §4, §9)."""

from __future__ import annotations

from app.catalog import Catalog

RECORDS = [
    {
        "id": "core-java-advanced-level-new",
        "name": "Core Java (Advanced Level) (New)",
        "url": "https://www.shl.com/products/product-catalog/view/core-java-advanced-level-new/",
        "test_type": "K",
        "test_types": ["K"],
        "keys": ["Knowledge & Skills"],
        "description": "Advanced Java knowledge test.",
        "job_levels": ["Mid-Professional"],
        "duration": "13 minutes",
    },
    {
        "id": "microsoft-excel-365-new",
        "name": "Microsoft Excel 365 (New)",
        "url": "https://www.shl.com/products/product-catalog/view/microsoft-excel-365-new/",
        "test_type": "K",
        "test_types": ["K", "S"],
        "keys": ["Knowledge & Skills", "Simulations"],
        "description": "Excel simulation + knowledge.",
    },
]


def make() -> Catalog:
    return Catalog(records=[dict(r) for r in RECORDS])


def test_get_and_has():
    cat = make()
    assert cat.has("core-java-advanced-level-new")
    assert not cat.has("nope")
    assert cat.get("core-java-advanced-level-new")["name"].startswith("Core Java")


def test_url_and_name_for():
    cat = make()
    assert cat.url_for("microsoft-excel-365-new").endswith("microsoft-excel-365-new/")
    assert cat.name_for("microsoft-excel-365-new") == "Microsoft Excel 365 (New)"
    assert cat.url_for("nope") is None


def test_test_type_joins_multi_key():
    cat = make()
    assert cat.test_type_for("microsoft-excel-365-new") == "K,S"
    assert cat.test_type_for("core-java-advanced-level-new") == "K"


def test_to_recommendation_is_canonical():
    cat = make()
    rec = cat.to_recommendation("microsoft-excel-365-new")
    assert set(rec) == {"name", "url", "test_type"}
    assert rec["test_type"] == "K,S"
    assert cat.to_recommendation("nope") is None


def test_recommendations_for_drops_unknown_and_dedupes():
    cat = make()
    recs = cat.recommendations_for(
        ["core-java-advanced-level-new", "ghost", "core-java-advanced-level-new"]
    )
    assert len(recs) == 1
    assert recs[0]["name"].startswith("Core Java")


def test_filter_valid_by_id():
    cat = make()
    items = [
        {"id": "core-java-advanced-level-new"},
        {"id": "hallucinated-id"},
    ]
    kept = cat.filter_valid(items)
    assert kept == [{"id": "core-java-advanced-level-new"}]


def test_filter_valid_drops_id_url_mismatch():
    cat = make()
    items = [
        {
            "id": "core-java-advanced-level-new",
            "url": "https://www.shl.com/products/product-catalog/view/FAKE/",
        }
    ]
    assert cat.filter_valid(items) == []


def test_filter_valid_keeps_known_url_without_id():
    cat = make()
    items = [{"url": RECORDS[0]["url"]}]
    assert cat.filter_valid(items) == items


def test_record_text_includes_name_and_description():
    cat = make()
    text = cat.record_text("core-java-advanced-level-new")
    assert "Core Java" in text and "Advanced Java knowledge" in text
    assert cat.record_text("nope") == ""
