"""Templated shortlist replies and COMPARE formatting.

Responsibilities (CLAUDE.md §4, "Reply text: hybrid"):
- Generate the schema-critical shortlist reply from a deterministic template
  (e.g. "Here are N assessments for a {level} {role}...") so the response never
  depends on LLM formatting.
- Format COMPARE answers from catalog evidence.

CLARIFY and COMPARE prose come from the LLM (fuzzy, needs fluency); the shortlist
reply text is templated here.
"""
