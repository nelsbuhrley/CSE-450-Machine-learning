---
# This frontmatter prevents Jekyll from rendering this file as a site page.
# Remove it if you ever want TEMPLATE.md to appear in the site.
sitemap: false
permalink: false
---

# Project Page Template

How to add a new project (module_1, module_3, ...) to the site with the same
three-page structure as `module_2-bank`. Keep this short; if anything diverges
from what's in `module_2-bank`, treat that as the source of truth.

## Folder layout

For a project with slug `<slug>` (e.g. `module_3-foo`):

```
<slug>/
  page/
    index.md         → /<slug>/page/             "Overview"
    summary.md       → /<slug>/page/summary/     "Executive Summary"
    technical.md     → /<slug>/page/technical/   "Technical Deep-Dive"
    assets/
      <slug>-executive-summary.pdf
```

The page files live inside the project's own top-level folder (e.g.
`module_2-bank/page/`), alongside the raw data, code, and charts. This keeps
everything for a project in one place.

## Required frontmatter

Every page in a project must declare these four keys so the shared sub-nav
strip (`_includes/project-nav.html`) renders correctly:

```yaml
layout: default
project: <slug>                # e.g. module_2-bank
project_title: "Module 2 · Bank Marketing"
page_type: overview            # or "summary" or "technical"
permalink: /<slug>/page/    # adjust path per page (see above)
```

The `technical.md` page uses GitHub links for code excerpts. By default those
links resolve using the site-wide values in `_config.yml`, so you only need
to set overrides when a project lives in a different repo or branch:

```yaml
github_repo: "nelsbuhrley/CSE-450-Machine-learning"
github_branch: "main"
```

## What goes on each page

**`index.md` — Overview.** Academic framing: course, term, team, problem
statement, why it's an interesting ML problem, what concepts the project
demonstrates, headline result, links to the other two pages. Audience: a
peer or grader skimming the project.

**`summary.md` — Executive Summary.** Stakeholder-facing one-pager. Open with
an `<section class="exec-hero">` block: eyebrow, title, italic subtitle, a
"Download PDF" button and a "Technical Deep-Dive →" ghost-button. Body uses
`<div class="kpi-strip">` for headline numbers and embeds the marketing
chart variants from the project's `visualisation/output/marketing/white_bars/`.
No code, no jargon.

**`technical.md` — Technical Deep-Dive.** Methodology, evaluation, model
walk-throughs with verbatim code excerpts. Each code block is preceded by a
`<div class="code-meta">` row showing the file path and a "View on GitHub →"
link. Pull excerpts directly from the source files — don't paraphrase. Use
the `detailed/` chart variants from `visualisation/output/detailed/` for any
technical figures. Close with a "Code, data, and reproduction" section
linking each top-level folder of the project to GitHub.

## Reusable HTML patterns

These class names are styled by `ui/css/custom.css`. Use them verbatim — do
not invent variants.

```html
<!-- Hero block at the top of summary.md -->
<section class="exec-hero">
  <p class="exec-hero__eyebrow">EYEBROW</p>
  <h1 class="exec-hero__title">Title</h1>
  <p class="exec-hero__subtitle">Subtitle</p>
  <div class="exec-hero__actions">
    <a class="btn-pill" href="..." download>⬇  Download PDF</a>
    <a class="btn-pill btn-pill--ghost" href="...">Technical Deep-Dive →</a>
  </div>
</section>

<!-- KPI strip — repeat the cell block per metric -->
<div class="kpi-strip">
  <div class="kpi-strip__cell">
    <span class="kpi-strip__label">METRIC LABEL</span>
    <span class="kpi-strip__value kpi-strip__value--gain">$15.58</span>
    <!-- modifiers: --gain (green), --loss (red), omit for default -->
  </div>
  <!-- ... more cells ... -->
</div>

<!-- Code meta row — sits directly above a fenced code block -->
<div class="code-meta">
  <span class="code-meta__path">module_X/models/foo.py · L10–L25</span>
  <a class="code-meta__link"
  href="https://github.com/{{ page.github_repo | default: site.github_repo }}/blob/{{ page.github_branch | default: site.github_branch }}/module_X/models/foo.py#L10-L25"
     target="_blank" rel="noopener">View on GitHub →</a>
</div>

```python
# ... actual code from the file ...
```
```

For figure captions, use kramdown's block-attribute syntax to attach the
`figure-caption` class to the paragraph immediately below an image:

```markdown
![Alt text]({{ '/path/to/chart.png' | relative_url }})
{: .figure-caption }
*Caption text.*
```

If a chart has a dark-mode variant, add a `data-dark-src` attribute to the
image so the theme switcher can swap it automatically:

```markdown
![Alt text]({{ '/path/to/chart_light.png' | relative_url }}){: data-dark-src="{{ '/path/to/chart_dark.png' | relative_url }}" }
```

## Adding a new project — checklist

1. Create `<slug>/page/` and the three `.md` files above with the
   required frontmatter.
2. Generate or copy the executive-summary PDF into
   `<slug>/page/assets/<slug>-executive-summary.pdf`.
3. Make sure the project's image folder is **not** excluded by `_config.yml`.
   PNGs at `<slug>/visualisation/...` are served as-is.
4. Optionally update the corner-menu nav in `ui/js/nav.js` if you want the
  project to appear in the top-right "Menu" panel.
5. The project sub-nav automatically renders a **Repo Folder** tab pointing
  at `https://github.com/<repo>/tree/<branch>/<slug>` using `_config.yml`
  defaults. If a project lives in a different repo or branch, set
  `github_repo` and `github_branch` in all three page frontmatters.
6. Push to `main`; the GitHub Pages workflow rebuilds the site.

## Conventions worth keeping

- Chart variants: **`white_bars/`** for the executive summary,
  **`detailed/`** for the technical deep-dive.
- Code excerpts are **verbatim**, never paraphrased. If the file changes,
  re-paste, don't edit the snippet by hand.
- Numbers, dollar values, and percentages in the executive summary must
  match the source charts and the corresponding numbers in the technical
  deep-dive. If they disagree, the technical page is the source of truth.
- "Em dashes — like this" are real em dashes (U+2014), not double hyphens.
- Page titles use a "·" middle-dot separator: "Module N · Project Name —
  Section". The middle dot is U+00B7.
