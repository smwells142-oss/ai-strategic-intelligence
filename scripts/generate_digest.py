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
from datetime import datetime, timezone

import anthropic


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

    update_digests_file(digest)
    print("Done.")


if __name__ == "__main__":
    main()
