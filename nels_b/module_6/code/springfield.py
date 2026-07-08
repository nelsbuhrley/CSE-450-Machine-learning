import re, html, time, pathlib, urllib.request

UA = "Mozilla/5.0 (X11; Linux x86_64; rv:127.0) Gecko/20100101 Firefox/127.0"
OUT = pathlib.Path("/home/nbuhrley/CSE-450-Machine-learning/nels_b/module_6/data/transformers")
OUT.mkdir(parents=True, exist_ok=True)

MOVIES = [
    ("the-transformers-the-movie",           "1986_the_transformers_the_movie.txt"),
    ("transformers",                         "2007_transformers.txt"),
    ("transformers-revenge-of-the-fallen",   "2009_transformers_revenge_of_the_fallen.txt"),
    ("transformers-dark-of-the-moon",        "2011_transformers_dark_of_the_moon.txt"),
    ("transformers-age-of-extinction",       "2014_transformers_age_of_extinction.txt"),
    ("transformers-the-last-knight",         "2017_transformers_the_last_knight.txt"),
    ("bumblebee",                            "2018_bumblebee.txt"),
    ("transformers-rise-of-the-beasts",      "2023_transformers_rise_of_the_beasts.txt"),
    ("transformers-one",                     "2024_transformers_one.txt"),
]

# subtitle-rip artifacts, not movie dialogue
JUNK = re.compile(
    r"^(1|Lovingly and VERY difficultly|cooked by \w+|"
    r"(Sub(title)?s? (by|ripped|synced).*)|(www\.\S+))$",
    re.I,
)

def clean(txt: str) -> str:
    lines = [ln.strip() for ln in txt.split("\n")]
    while lines and (not lines[0] or JUNK.match(lines[0])):
        lines.pop(0)
    body = "\n".join(lines)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    return body + "\n"

for slug, fname in MOVIES:
    url = f"https://www.springfieldspringfield.co.uk/movie_script.php?movie={slug}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        page = r.read().decode("utf-8", "replace")
    m = re.search(r'class="scrolling-script-container"(.*?)</div>', page, re.S)
    if not m:
        print(f"FAILED: {slug} (no script container)")
        continue
    txt = re.sub(r"<br\s*/?>", "\n", m.group(1))
    txt = re.sub(r"<[^>]+>", "", txt)
    txt = html.unescape(txt).lstrip(" >\n")
    txt = clean(txt)
    (OUT / fname).write_text(txt, encoding="utf-8")
    print(f"saved {fname}: {len(txt):,} chars")
    time.sleep(2)
