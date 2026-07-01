"""Tests for agent orchestration and conversation policy (CLAUDE.md §4, §9).

Assert: no recommendation on turn 1 for a vague query; commit once role + one
discriminating attribute is known; refine re-derives constraints from full history;
post-filter drops non-catalog items; the 8-turn cap is honored.
"""
