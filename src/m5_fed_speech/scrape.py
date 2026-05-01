"""Scrape Federal Reserve speeches (2015-2024) into data/speeches.csv.

Source: https://www.federalreserve.gov/newsevents/speeches.htm
- Per-year index pages list links to individual speech pages.
- Polite: 1 req/sec, custom User-Agent.
- Falls back to a tiny synthetic dataset if scraping fails (network, structure
  change, CI). The fallback is documented and clearly marked in the CSV.
"""
from __future__ import annotations

import argparse
import csv
import io
import logging
import random
import time
from pathlib import Path
from typing import Iterable

import requests
from bs4 import BeautifulSoup

BASE = "https://www.federalreserve.gov"
INDEX_TMPL = BASE + "/newsevents/speech/{year}-speeches.htm"
HEADERS = {
    "User-Agent": (
        "m5-fed-speech research bot (academic; contact: yashpatel06050@gmail.com)"
    )
}
DELAY_SEC = 1.0
TIMEOUT = 20

log = logging.getLogger("scrape")


def _get(url: str) -> str | None:
    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        if r.status_code == 200:
            return r.text
        log.warning("GET %s -> %s", url, r.status_code)
    except requests.RequestException as e:
        log.warning("GET %s failed: %s", url, e)
    return None


def list_speech_links(year: int) -> list[tuple[str, str, str]]:
    """Return [(date_iso, title, url), ...] for one year's index page."""
    html = _get(INDEX_TMPL.format(year=year))
    if not html:
        return []
    soup = BeautifulSoup(html, "lxml")
    out: list[tuple[str, str, str]] = []
    for row in soup.select("div.row.eventlist .col-xs-12, div.eventlist .row, .itemTitle"):
        # Defensive: site markup changes. Try multiple selectors below.
        pass
    # Generic: every <a> under content whose href contains "/newsevents/speech/" and ends .htm
    for a in soup.select("a[href*='/newsevents/speech/']"):
        href = a.get("href", "")
        if not href.endswith(".htm") or "speeches.htm" in href:
            continue
        url = href if href.startswith("http") else BASE + href
        title = a.get_text(strip=True)
        if not title:
            continue
        # Date often appears as sibling text "Month DD, YYYY"; fallback to year
        date = f"{year}-01-01"
        parent = a.find_parent()
        if parent:
            txt = parent.get_text(" ", strip=True)
            import re as _re
            m = _re.search(r"([A-Z][a-z]+ \d{1,2}, \d{4})", txt)
            if m:
                from datetime import datetime as _dt
                try:
                    date = _dt.strptime(m.group(1), "%B %d, %Y").date().isoformat()
                except ValueError:
                    pass
        out.append((date, title, url))
    # Dedup by url
    seen: set[str] = set()
    uniq = []
    for d, t, u in out:
        if u in seen:
            continue
        seen.add(u)
        uniq.append((d, t, u))
    return uniq


def fetch_speech(url: str) -> str:
    html = _get(url) or ""
    if not html:
        return ""
    soup = BeautifulSoup(html, "lxml")
    main = soup.select_one("#article, .col-xs-12.col-sm-8.col-md-8, article, main")
    text = (main or soup).get_text(" ", strip=True)
    return text


def scrape_years(years: Iterable[int], out_csv: Path, max_per_year: int | None = None) -> int:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "title", "url", "text", "source"])
        for y in years:
            links = list_speech_links(y)
            if max_per_year:
                links = links[:max_per_year]
            log.info("year=%s links=%s", y, len(links))
            for date, title, url in links:
                time.sleep(DELAY_SEC + random.uniform(0, 0.25))
                text = fetch_speech(url)
                if len(text) < 500:
                    continue
                w.writerow([date, title, url, text, "federalreserve.gov"])
                n += 1
    return n


# ---------- Synthetic fallback ----------
SYNTH_HAWK = (
    "Inflation risk remains elevated and the Committee judges that further "
    "tightening may be appropriate. Policy will need to remain restrictive "
    "to bring inflation back to target. We are prepared to raise rates if "
    "the economy shows signs of overheating."
)
SYNTH_DOVE = (
    "Downside risk to growth has increased. The Committee remains patient "
    "and supportive of the recovery, and stands ready to ease policy and "
    "lower rates if conditions warrant additional accommodation."
)


def write_synthetic(out_csv: Path, n: int = 60, seed: int = 42) -> int:
    import random as _r
    from datetime import date, timedelta
    _r.seed(seed)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    start = date(2015, 1, 5)
    rows = []
    for i in range(n):
        d = (start + timedelta(days=i * 45)).isoformat()
        body = SYNTH_HAWK if _r.random() > 0.5 else SYNTH_DOVE
        # add filler so each "speech" is realistic length
        filler = " The economic outlook is uncertain and data dependent." * 30
        rows.append((d, f"Synthetic speech {i}", f"synthetic://{i}", body + filler, "synthetic"))
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "title", "url", "text", "source"])
        w.writerows(rows)
    return n


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/speeches.csv")
    p.add_argument("--start", type=int, default=2015)
    p.add_argument("--end", type=int, default=2024)
    p.add_argument("--max-per-year", type=int, default=None)
    p.add_argument("--synthetic", action="store_true",
                   help="Skip network; write a synthetic dataset (documented in README).")
    args = p.parse_args()
    out = Path(args.out)
    if args.synthetic:
        n = write_synthetic(out)
    else:
        try:
            n = scrape_years(range(args.start, args.end + 1), out, args.max_per_year)
            if n == 0:
                log.warning("Live scrape returned 0 rows; writing synthetic fallback.")
                n = write_synthetic(out)
        except Exception as e:
            log.warning("Scrape failed (%s); writing synthetic fallback.", e)
            n = write_synthetic(out)
    log.info("Wrote %s rows to %s", n, out)


if __name__ == "__main__":
    main()
