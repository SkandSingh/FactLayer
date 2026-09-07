"""Tests for app/matching.py.

`find_candidates` is a pure-code, no-LLM prefilter: it narrows the
existing fact store down to a short list of plausibly-related facts
before the expensive LLM comparison step. Recall matters more than
precision, so these tests check both that real matches are NOT missed
(the failure mode that actually matters) and that a clearly unrelated
pair is excluded (so the prefilter isn't a no-op that returns
everything).
"""
from app.matching import find_candidates
from app.models import Fact, TimeScope


def _make_fact(
    id: int,
    entity_name: str,
    entity_id: str | None,
    attribute: str,
    raw_value: str = "100",
) -> Fact:
    """Build a minimal real `Fact` for tests -- only entity_name,
    entity_id, and attribute matter to `find_candidates`, but `Fact`
    requires the rest of its fields too."""
    return Fact(
        id=id,
        document_id=1,
        entity_name=entity_name,
        entity_id=entity_id,
        attribute=attribute,
        raw_value=raw_value,
        unit=None,
        time_scope=TimeScope(period_type="unknown"),
        entity_scope=None,
        verbatim_quote=raw_value,
        confidence=0.9,
        normalized_value=None,
        normalized_unit=None,
        section_path="",
        char_offset=0,
        page_number=1,
        created_at="2026-01-01T00:00:00+00:00",
    )


class TestExactIdentifierMatch:
    def test_exact_entity_id_match(self):
        new_fact = _make_fact(1, "Acme Corp", "ABC123", "revenue")
        existing = _make_fact(2, "Acme Corp", "ABC123", "revenue")
        assert find_candidates(new_fact, [existing]) == [existing]

    def test_cin_u_vs_l_prefix_matches_via_normalize_identifier(self):
        """CINs differing only in the U (unlisted) vs L (listed)
        listing-status prefix should match via criterion 1, since
        normalize_identifier collapses that distinction."""
        new_fact = _make_fact(
            1, "Delhivery Limited", "U63090DL2011PLC221234", "total_revenue"
        )
        existing = _make_fact(
            2, "Delhivery Limited", "L63090DL2011PLC221234", "total_revenue"
        )
        candidates = find_candidates(new_fact, [existing])
        assert candidates == [existing]


class TestFuzzyEntityNameMatch:
    def test_delhivery_limited_vs_delhivery_ltd_no_entity_id(self):
        """No entity_id and no attribute overlap at all -- only the
        fuzzy company-name match should surface this as a candidate."""
        new_fact = _make_fact(1, "Delhivery Limited", None, "warehouse_count")
        existing = _make_fact(2, "Delhivery Ltd.", None, "employee_headcount")
        candidates = find_candidates(new_fact, [existing])
        assert candidates == [existing]

    def test_unrelated_short_names_do_not_fuzzy_match(self):
        new_fact = _make_fact(1, "Delhivery Limited", None, "warehouse_count")
        existing = _make_fact(2, "Zomato Limited", None, "employee_headcount")
        candidates = find_candidates(new_fact, [existing])
        assert existing not in candidates


class TestAttributeKeywordOverlap:
    def test_revenue_from_operations_vs_revenue_from_customers(self):
        """Same entity, attributes share 'revenue' (a meaningful token)
        even though they also share the stopword 'from'."""
        new_fact = _make_fact(1, "Acme Corp", "ID1", "revenue_from_operations")
        existing = _make_fact(2, "Acme Corp", "ID1", "revenue_from_customers")
        candidates = find_candidates(new_fact, [existing])
        assert candidates == [existing]

    def test_shared_stopword_only_does_not_match(self):
        """Two attributes that only share a trivial connector word
        ('from') should NOT be considered overlapping."""
        new_fact = _make_fact(1, "Acme Corp", "ID1", "revenue_from_operations")
        existing = _make_fact(2, "Zeta Inc", "ID2", "profit_from_sale")
        candidates = find_candidates(new_fact, [existing])
        assert existing not in candidates


class TestTrueNegative:
    def test_clearly_different_companies_and_attributes_excluded(self):
        new_fact = _make_fact(1, "Delhivery Limited", "U63090DL2011PLC221234", "total_revenue")
        existing = _make_fact(
            2, "Reserve Bank of India", "RBI001", "repo_rate"
        )
        candidates = find_candidates(new_fact, [existing])
        assert candidates == []


class TestSelfExclusion:
    def test_never_matches_fact_against_itself(self):
        fact = _make_fact(1, "Acme Corp", "ID1", "revenue")
        candidates = find_candidates(fact, [fact])
        assert candidates == []


class TestMixedStore:
    def test_narrows_a_larger_store_to_the_plausible_subset(self):
        new_fact = _make_fact(1, "Delhivery Limited", "U63090DL2011PLC221234", "total_revenue")
        exact_id_match = _make_fact(
            2, "Delhivery Limited", "L63090DL2011PLC221234", "total_revenue"
        )
        fuzzy_name_match = _make_fact(3, "Delhivery Ltd.", None, "warehouse_count")
        attribute_overlap_match = _make_fact(
            4, "Some Other Company", None, "total_revenue_reported"
        )
        unrelated = _make_fact(5, "Reserve Bank of India", "RBI001", "repo_rate")

        candidates = find_candidates(
            new_fact,
            [exact_id_match, fuzzy_name_match, attribute_overlap_match, unrelated],
        )

        assert exact_id_match in candidates
        assert fuzzy_name_match in candidates
        assert attribute_overlap_match in candidates
        assert unrelated not in candidates
