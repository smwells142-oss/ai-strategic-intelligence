#!/usr/bin/env python3
"""
Daily AI Strategic Intelligence Digest Generator

Uses the Anthropic Claude API to research and generate a structured digest
of AI developments relevant to CTSI (Clinical and Translational Science Institute).
Appends the new entry to digests.json.
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

import re
from urllib.parse import urlparse

import anthropic
import requests


DIGEST_SCHEMA = """\
Return a JSON object (not wrapped in markdown) with this exact structure:
{
  "date": "YYYY-MM-DD",
  "domains": [
    {
      "name": "<domain name>",
      "items": [
        {
          "headline": "<concise headline>",
          "source": "<source name>",
          "sourceUrl": "<URL>",
          "sourceDate": "<human-readable date>",
          "priority": "high" | "medium" | "low",
          "summary": "<2-4 sentence factual summary>",
          "whyItMatters": "<2-4 sentences on relevance to CTSI/UMN>",
          "actionImplication": "<1-2 sentences, or null if none>"
        }
      ]
    }
  ],
  "sourcesConsulted": ["<source descriptions>"],
  "confidenceNotes": ["<caveats about sourcing or verification>"]
}
"""

SYSTEM_PROMPT = """\
You are an AI strategic intelligence analyst for the University of Minnesota's \
Clinical and Translational Science Institute (CTSI). Your job is to produce a \
daily intelligence digest covering AI developments relevant to academic medical \
centers, translational research, and CTSI specifically.

Cover these domains (include at least 2, ideally 4-5, with 2-4 items each):
1. Federal Policy Signals - federal AI legislation, NIH/NSF/FDA policy, funding
2. Institutional Strategy - higher ed AI adoption, workforce, peer institutions
3. Responsible AI & Ethics - governance, regulation, compliance (EU AI Act, state bills)
4. AI in Clinical & Translational Research - clinical AI tools, trial design, IRB frameworks
5. AI Tool & Capability Developments - major model releases, infrastructure, MCP/agents

Priority guidelines:
- HIGH: Direct CTSI operational impact, major federal policy shifts, funding deadlines
- MEDIUM: Peer institution benchmarks, emerging frameworks, relevant research
- LOW: Background context, market trends, incremental capability updates

Quality standards:
- Every item must have a real, verifiable source URL
- Summaries must be factual and specific, not vague
- "Why it matters" must connect specifically to CTSI, UMN, or translational science
- Action implications should be concrete and addressable
- Include confidence notes for any items where sourcing is indirect

Output ONLY the JSON object. No markdown wrapping, no commentary.
"""


def generate_digest(today: str) -> dict:
    """Call Claude to generate today's digest."""
    client = anthropic.Anthropic()

    user_prompt = f"""\
Generate the AI Strategic Intelligence Digest for {today}.

Research and compile the most significant AI developments from the past 24-48 hours \
relevant to CTSI and academic medical centers. Focus on:

- New federal policy actions, guidance, or funding announcements related to AI
- University/institutional AI strategy moves by peer institutions
- AI governance and ethics developments (state, federal, international)
- Clinical and translational research AI tools, frameworks, or publications
- Notable AI capability releases with research implications

Today's date: {today}

{DIGEST_SCHEMA}
"""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=8000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    response_text = message.content[0].text.strip()

    # Handle potential markdown wrapping
    if response_text.startswith("```"):
        lines = response_text.split("\n")
        # Remove first and last lines (```json and ```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        response_text = "\n".join(lines)

    return json.loads(response_text)


BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}


def validate_url(url: str, retries: int = 2) -> tuple[bool, int | None, str]:
    """
    Check if a URL is reachable.
    Returns (is_valid, status_code, reason).
    Treats 2xx and 403 as valid (403 = bot-blocked but page exists).
    """
    for attempt in range(retries):
        try:
            resp = requests.head(
                url, headers=BROWSER_HEADERS, timeout=15, allow_redirects=True
            )
            # Some servers reject HEAD; fall back to GET
            if resp.status_code == 405:
                resp = requests.get(
                    url, headers=BROWSER_HEADERS, timeout=15,
                    allow_redirects=True, stream=True,
                )
            code = resp.status_code
            # 2xx = works, 403 = bot-blocked but page exists, 3xx handled by redirects
            if code < 400 or code == 403:
                return True, code, "OK" if code < 400 else "OK (bot-protected)"
            return False, code, f"HTTP {code}"
        except requests.exceptions.SSLError:
            return False, None, "SSL error"
        except requests.exceptions.ConnectionError:
            if attempt < retries - 1:
                time.sleep(2)
                continue
            return False, None, "Connection failed"
        except requests.exceptions.Timeout:
            if attempt < retries - 1:
                time.sleep(2)
                continue
            return False, None, "Timeout"
        except requests.exceptions.RequestException as e:
            return False, None, str(e)[:80]
    return False, None, "Max retries exceeded"


def get_source_homepage(url: str) -> str | None:
    """Extract the homepage URL from a full URL (e.g., https://www.nih.gov/some/path -> https://www.nih.gov)."""
    try:
        parsed = urlparse(url)
        if parsed.scheme and parsed.netloc:
            return f"{parsed.scheme}://{parsed.netloc}"
    except Exception:
        pass
    return None


def search_for_replacement_url(headline: str, source: str) -> str | None:
    """
    Search DuckDuckGo for the headline + source to find a real URL.
    Returns the first relevant result URL, or None.
    """
    query = f"{source} {headline}"
    search_url = "https://html.duckduckgo.com/html/"
    try:
        resp = requests.post(
            search_url,
            data={"q": query},
            headers=BROWSER_HEADERS,
            timeout=15,
        )
        if resp.status_code != 200:
            return None

        # Extract URLs from DuckDuckGo result links
        # DDG HTML results contain links in <a class="result__a" href="...">
        urls = re.findall(r'class="result__a"[^>]*href="([^"]+)"', resp.text)
        if not urls:
            # Fallback: try extracting from uddg= redirect params
            urls = re.findall(r'uddg=([^&"]+)', resp.text)

        for found_url in urls[:5]:
            # URL-decode if needed
            found_url = requests.utils.unquote(found_url)
            # Skip DuckDuckGo internal links
            if "duckduckgo.com" in found_url:
                continue
            # Validate the found URL actually works
            is_valid, code, reason = validate_url(found_url, retries=1)
            if is_valid:
                return found_url

    except Exception as e:
        print(f"    Search failed: {e}")
    return None


def validate_digest_links(digest: dict) -> dict:
    """
    Validate all sourceUrls in a digest. For broken URLs:
    1. Search the web for a replacement URL matching the headline + source.
    2. If no replacement found, fall back to the source's homepage.
    3. Never remove items — always keep the content.
    Flags modified URLs in confidence notes.
    """
    ok_count = 0
    replaced_count = 0
    fallback_count = 0
    modified_items = []

    for domain in digest.get("domains", []):
        for item in domain.get("items", []):
            url = item.get("sourceUrl", "")
            if not url:
                continue

            is_valid, code, reason = validate_url(url)
            headline = item.get("headline", "Unknown")[:70]
            source = item.get("source", "")

            if is_valid:
                ok_count += 1
                status = f"OK ({code})" if code else "OK"
                print(f"  {status:20s} {headline}")
                continue

            # URL is broken — try to find a replacement
            print(f"  BROKEN ({reason:10s}) {headline}")
            print(f"    Original URL: {url}")

            # Step 1: Search for the correct URL
            print(f"    Searching for replacement...")
            replacement = search_for_replacement_url(headline, source)

            if replacement:
                replaced_count += 1
                item["sourceUrl"] = replacement
                print(f"    REPLACED with: {replacement}")
                modified_items.append(
                    f'"{headline}" — original URL broken ({reason}), '
                    f"replaced via web search"
                )
            else:
                # Step 2: Fall back to source homepage
                homepage = get_source_homepage(url)
                if homepage:
                    fallback_count += 1
                    item["sourceUrl"] = homepage
                    print(f"    FALLBACK to:   {homepage}")
                    modified_items.append(
                        f'"{headline}" — original URL broken ({reason}), '
                        f"linked to source homepage"
                    )
                else:
                    # Last resort: keep the original broken URL but flag it
                    fallback_count += 1
                    print(f"    KEPT original (no replacement found)")
                    modified_items.append(
                        f'"{headline}" — source URL could not be verified ({reason})'
                    )

    # Add notes about modified links to confidence notes
    if modified_items:
        note = (
            f"Link validation: {len(modified_items)} source URL(s) required "
            f"correction ({replaced_count} replaced via search, "
            f"{fallback_count} fell back to homepage/kept as-is): "
            + "; ".join(modified_items)
        )
        digest.setdefault("confidenceNotes", []).append(note)

    total = ok_count + replaced_count + fallback_count
    print(
        f"\nLink validation complete: {total} total, {ok_count} valid, "
        f"{replaced_count} replaced, {fallback_count} fallback."
    )
    return digest


def update_digests_file(new_digest: dict, filepath: str = "digests.json"):
    """Load existing digests, prepend the new one, and save."""
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            digests = json.load(f)
    else:
        digests = []

    # Check if today's digest already exists
    existing_dates = {d["date"] for d in digests}
    if new_digest["date"] in existing_dates:
        print(f"Digest for {new_digest['date']} already exists. Replacing.")
        digests = [d for d in digests if d["date"] != new_digest["date"]]

    # Prepend new digest
    digests.insert(0, new_digest)

    # Sort by date descending
    digests.sort(key=lambda d: d["date"], reverse=True)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(digests, f, indent=2, ensure_ascii=False)

    print(f"Updated {filepath} — now contains {len(digests)} digest(s).")


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    print(f"Generating digest for {today}...")

    digest = generate_digest(today)

    # Validate structure
    assert "date" in digest, "Missing 'date' field"
    assert "domains" in digest, "Missing 'domains' field"
    assert len(digest["domains"]) >= 2, "Need at least 2 domains"

    # Ensure date matches
    digest["date"] = today

    # Validate all source URLs — fix broken links, never remove content
    print("\nValidating source URLs...")
    digest = validate_digest_links(digest)

    update_digests_file(digest)
    print("Done.")


if __name__ == "__main__":
    main()
