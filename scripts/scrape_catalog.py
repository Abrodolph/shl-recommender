"""Build-time: scrape/load the SHL catalog into ``data/catalog.json``.

Responsibilities (CLAUDE.md §2):
- Source the SHL product catalog (Individual Test Solutions ONLY; pre-packaged Job
  Solutions are out of scope and excluded). Prefer SHL's downloadable catalog file
  if available; otherwise scrape the public catalog with httpx + BeautifulSoup.
- For each assessment capture (as available): id (stable slug from the URL), name,
  url (canonical), test_type (single-letter code — confirm the exact legend from
  the catalog page), description, job_levels, duration/remote_testing/adaptive_irt
  flags, competencies/keywords.
- Persist a single ``data/catalog.json`` (checked into the repo).

Build-time only — NOT a runtime dependency of the API.
"""
