# Transformers movie transcripts

Full-dialogue transcripts (subtitle-derived) for all nine Transformers films,
one file per movie, named `YEAR_title.txt`.

- Source: https://www.springfieldspringfield.co.uk (movie_script.php pages)
- Downloaded: 2026-07-08 via `code/springfield.py`
- Cleanup applied: HTML stripped, entities unescaped, per-line whitespace
  trimmed, subtitle-index / subber-credit artifacts removed.
- ~522K characters total. Dialogue only — no scene descriptions or speaker
  labels, since these come from subtitles rather than shooting scripts.

For classroom/educational use in a text-generation exercise.
