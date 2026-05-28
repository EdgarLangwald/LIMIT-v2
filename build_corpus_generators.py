"""
Evaluates every JSON in dariusk/corpora and generates sentence-family dataclasses for
usable people-profile attributes. Calls Claude Haiku per file concurrently.

Usage:
    python build_corpus_generators.py          # evaluate + assemble
    python build_corpus_generators.py --assemble-only   # skip eval, just reassemble
"""

import asyncio
import json
import re
import sys
import textwrap
from pathlib import Path

import httpx
from anthropic import AsyncAnthropic

CORPORA_TREE_URL = (
    "https://api.github.com/repos/dariusk/corpora/git/trees/master?recursive=1"
)
RAW_BASE = "https://raw.githubusercontent.com/dariusk/corpora/master/"
EVAL_DIR = Path("corpus_evaluations")
OUTPUT_FILE = Path("profile_sentences.py")
CONCURRENCY = 15
MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = """\
You are helping build a synthetic US-person profile generator.
Your job: evaluate a JSON file from the "corpora" collection and decide whether its \
items make natural, interesting attributes for a fictional person.
Always respond with a single JSON object and nothing else.\
"""


def user_prompt(filename: str, raw: str) -> str:
    preview = raw[:4000]
    return f"""\
Filename: {filename}
Content (first 4000 chars):
{preview}

DECISION CRITERIA
Accept if items fit naturally into sentences like "He/she has/likes/plays/drives X":
  GOOD: books, cars, instruments, academic_subjects, dog_breeds, foods, sports, hobbies,
        board_games, tv_shows, beers, cocktails, languages, programming_languages,
        occupations, car_makes, music_genres, phobias, flowers, vegetables
  BAD:  primes, zodiac, commercial_aircraft, airport_codes, math constants,
        lists where items are abstract/metaphorical (e.g. "settings" containing "heaven"),
        lists with fewer than 10 usable items
  ALWAYS REJECT: personality_test.json, anything purely technical or numerical

SENTENCE TEMPLATES (if accepted)
- 5 to 10 varied templates using {{item}} as the placeholder
- Vary structure, tense, and context (childhood memory, current habit, aspiration, etc.)
- Every template must make sense with *every* item in the list (no misfits!)
- Bad: "He keeps a {{item}} in his yard." — works for "dog" but not for "bass guitar"

Return ONLY this JSON (no markdown, no commentary):
{{
  "accepted": true,
  "reason": "one sentence",
  "category_name": "cars",
  "description": "Types of cars / car brands",
  "templates": ["He drives a {{item}}.", "He always wanted to own a {{item}}."],
  "items": ["Ford Mustang", "Bugatti", ...]
}}

Put ALL usable items in "items" (skip items that would break the templates).
If rejected, omit templates and items.\
"""


async def fetch_json_paths(client: httpx.AsyncClient) -> list[str]:
    resp = await client.get(CORPORA_TREE_URL, timeout=30)
    resp.raise_for_status()
    tree = resp.json()["tree"]
    return [
        entry["path"]
        for entry in tree
        if entry["path"].endswith(".json") and entry["path"].startswith("data/")
    ]


async def fetch_content(client: httpx.AsyncClient, path: str) -> str:
    resp = await client.get(RAW_BASE + path, timeout=30)
    resp.raise_for_status()
    return resp.text


def parse_haiku_response(text: str) -> dict:
    """Extract JSON from Haiku's response, tolerating markdown fences."""
    text = text.strip()
    # Strip ```json ... ``` if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


async def evaluate_file(
    sem: asyncio.Semaphore,
    client: httpx.AsyncClient,
    anthropic: AsyncAnthropic,
    path: str,
) -> dict | None:
    filename = Path(path).name
    out_path = EVAL_DIR / (path.replace("/", "__") + ".json")

    if out_path.exists():
        print(f"  [skip] {filename} (cached)")
        return json.loads(out_path.read_text())

    async with sem:
        try:
            raw = await fetch_content(client, path)
        except Exception as e:
            print(f"  [fetch error] {filename}: {e}")
            return None

        try:
            msg = await anthropic.messages.create(
                model=MODEL,
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt(filename, raw)}],
            )
            result_text = msg.content[0].text
        except Exception as e:
            print(f"  [api error] {filename}: {e}")
            return None

        try:
            result = parse_haiku_response(result_text)
        except json.JSONDecodeError as e:
            print(f"  [parse error] {filename}: {e}\nRaw: {result_text[:200]}")
            result = {"accepted": False, "reason": f"parse error: {e}", "raw": result_text}

        result["_path"] = path
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        status = "ACCEPT" if result.get("accepted") else "reject"
        print(f"  [{status}] {filename}: {result.get('reason', '')}")
        return result


def python_identifier(s: str) -> str:
    s = re.sub(r"[^a-z0-9_]", "_", s.lower())
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def assemble_module(results: list[dict]) -> str:
    accepted = [r for r in results if r.get("accepted") and r.get("templates")]
    accepted.sort(key=lambda r: r.get("category_name", ""))

    lines = [
        '"""',
        "Auto-generated profile sentence families.",
        f"Total categories: {len(accepted)}",
        '"""',
        "",
        "from __future__ import annotations",
        "import random",
        "from dataclasses import dataclass, field",
        "from typing import ClassVar",
        "",
        "",
    ]

    class_names = []
    for r in accepted:
        cat = python_identifier(r.get("category_name", "unknown"))
        class_name = "".join(w.title() for w in cat.split("_")) + "Generator"
        class_names.append((cat, class_name))
        items = r.get("items", [])
        templates = r.get("templates", [])
        desc = r.get("description", cat)

        avg_len = (
            sum(len(t) for t in templates) / len(templates) if templates else 0.0
        )

        items_repr = json.dumps(items, ensure_ascii=False)
        templates_repr = json.dumps(templates, ensure_ascii=False)

        lines += [
            f"@dataclass",
            f"class {class_name}:",
            f'    """{desc}"""',
            f"    pool: ClassVar[list[str]] = {items_repr}",
            f"    templates: ClassVar[list[str]] = {templates_repr}",
            f"    pool_size: ClassVar[int] = {len(items)}",
            f"    avg_template_length: ClassVar[float] = {avg_len:.1f}",
            "",
            "    def __call__(self, seed: int | None = None) -> dict:",
            "        rng = random.Random(seed)",
            "        item = rng.choice(self.pool)",
            "        template = rng.choice(self.templates)",
            '        sentence = template.replace("{item}", item)',
            f'        return {{"sentence": sentence, "facts": {{"category": "{cat}", "item": item}}}}',
            "",
            "",
        ]

    # Generate class
    lines += ["class _Generate:"]
    for cat, class_name in class_names:
        lines.append(f"    {cat}: {class_name} = field(default_factory={class_name})")
    lines += [
        "",
        "    def __post_init__(self) -> None:",
    ]
    for cat, class_name in class_names:
        lines.append(f"        self.{cat} = {class_name}()")
    lines += [
        "",
        "",
        "generate = _Generate()",
        "",
    ]

    # Fix: _Generate needs to be a dataclass too
    for i, line in enumerate(lines):
        if line == "class _Generate:":
            lines[i] = "@dataclass\nclass _Generate:"
            break

    return "\n".join(lines)


async def main(assemble_only: bool = False) -> None:
    EVAL_DIR.mkdir(exist_ok=True)

    if not assemble_only:
        print("Fetching corpora file list…")
        async with httpx.AsyncClient(
            headers={"User-Agent": "LIMIT-v2-corpus-builder"}
        ) as http_client:
            paths = await fetch_json_paths(http_client)

        print(f"Found {len(paths)} JSON files. Evaluating with Haiku…\n")
        sem = asyncio.Semaphore(CONCURRENCY)
        anthropic = AsyncAnthropic()

        async with httpx.AsyncClient(
            headers={"User-Agent": "LIMIT-v2-corpus-builder"}
        ) as http_client:
            tasks = [
                evaluate_file(sem, http_client, anthropic, path) for path in paths
            ]
            results = await asyncio.gather(*tasks)
    else:
        results = [
            json.loads(p.read_text()) for p in EVAL_DIR.glob("*.json")
        ]

    accepted = [r for r in results if r and r.get("accepted")]
    print(f"\n{len(accepted)} categories accepted out of {len([r for r in results if r])} evaluated.")

    module_src = assemble_module([r for r in results if r])
    OUTPUT_FILE.write_text(module_src, encoding="utf-8")
    print(f"Written: {OUTPUT_FILE}")


if __name__ == "__main__":
    assemble_only = "--assemble-only" in sys.argv
    asyncio.run(main(assemble_only=assemble_only))
