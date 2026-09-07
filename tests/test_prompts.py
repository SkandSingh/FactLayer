"""Tests for app.llm.prompts prompt-template builders.

Run: python -m pytest tests/test_prompts.py -v

These tests only check string interpolation and structure — they never
make a network call to Gemini (no API key is configured in this
environment).
"""
from app.llm.prompts import build_comparison_prompt, build_extraction_prompt


def test_build_extraction_prompt_interpolates_section_text():
    section_text = "Revenue from operations was Rs 1,234 Million in FY24."
    prompt = build_extraction_prompt(section_text)

    assert section_text in prompt


def test_build_extraction_prompt_contains_key_markers():
    prompt = build_extraction_prompt("some section text")

    assert "Respond with a JSON array of fact objects only." in prompt
    assert "entity_name" in prompt
    assert "entity_id" in prompt
    assert "attribute" in prompt
    assert "raw_value" in prompt
    assert "unit" in prompt
    assert "time_scope" in prompt
    assert "entity_scope" in prompt
    assert "verbatim_quote" in prompt
    assert "confidence" in prompt


def test_build_extraction_prompt_preserves_time_scope_braces():
    # The time_scope field description uses a literal `{...}` structure in
    # the template; make sure .format() didn't mangle it or leave stray
    # unescaped braces from the templating mechanism itself.
    prompt = build_extraction_prompt("text")

    assert '{period_type: "point_in_time"' in prompt


def test_build_extraction_prompt_different_inputs_produce_different_prompts():
    prompt_a = build_extraction_prompt("Section A text")
    prompt_b = build_extraction_prompt("Section B text")

    assert prompt_a != prompt_b
    assert "Section A text" in prompt_a
    assert "Section A text" not in prompt_b


def test_build_comparison_prompt_interpolates_fact_digest_json():
    fact_digest_json = '[{"fact_id": "f1", "entity_name": "Acme Corp"}]'
    prompt = build_comparison_prompt(fact_digest_json)

    assert fact_digest_json in prompt


def test_build_comparison_prompt_contains_key_markers():
    prompt = build_comparison_prompt("[]")

    assert "Respond with a JSON array of relationship objects only." in prompt
    assert "corroborates" in prompt
    assert "contradicts" in prompt
    assert "reconciled_context" in prompt
    assert "fact_id_a" in prompt
    assert "fact_id_b" in prompt
    assert "relation_type" in prompt
    assert "reconciled_dimension" in prompt
    assert "reasoning_text" in prompt
    assert "confidence" in prompt


def test_build_comparison_prompt_handles_empty_digest():
    prompt = build_comparison_prompt("[]")

    assert prompt.endswith("Respond with a JSON array of relationship objects only.\n")
