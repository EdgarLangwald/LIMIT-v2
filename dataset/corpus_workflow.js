// THIS IS A SCRIPT FOR CLAUDE CODE'S "DYNAMIC WORKFLOW" TOOL (requires Opus 4.7+)
//
// Build src/profile_sentences.py from dariusk/corpora via 200+ Haiku agents.
//
// Run from repo root:
//   Workflow({ scriptPath: "dataset/corpus_workflow.js" })
//
// Intermediate results are cached in corpus_evaluations/ (one JSON per corpora file).
// If interrupted, resume with: Workflow({ scriptPath: "dataset/corpus_workflow.js", resumeFromRunId: "wf_..." })
// After: src/profile_sentences.py is written and ready to use.

export const meta = {
  name: 'build-corpus',
  description: 'Evaluate dariusk/corpora JSONs with Haiku and write src/profile_sentences.py',
  phases: [
    { title: 'Scout', detail: 'Fetch corpora file list from GitHub, find already-cached evaluations' },
    { title: 'Evaluate', detail: 'One Haiku agent per corpora JSON: fetch content, evaluate, cache result' },
    { title: 'Assemble', detail: 'Concatenate accepted categories into src/profile_sentences.py' },
  ],
}

const CORPORA_TREE_URL = 'https://api.github.com/repos/dariusk/corpora/git/trees/master?recursive=1'
const RAW_BASE = 'https://raw.githubusercontent.com/dariusk/corpora/master/'
const EVAL_DIR = 'corpus_evaluations'

// ---------------------------------------------------------------------------
// Phase 1: Scout
// ---------------------------------------------------------------------------
phase('Scout')

const scoutResult = await agent(
  `You have two tasks:

1. Fetch this URL using WebFetch: ${CORPORA_TREE_URL}
   From the JSON response, collect all "path" values in the "tree" array where:
   - path starts with "data/"
   - path ends with ".json"
   Return them as the "all_paths" list.

2. Use Glob with pattern "${EVAL_DIR}/*.json" to list all already-cached evaluation files.
   For each filename (e.g. "data__people__famous_scientists.json.json"), convert it back to a
   corpora path by: replacing "__" with "/" and removing the trailing ".json".
   Return these as the "done_paths" list.

Return JSON: { "all_paths": [...], "done_paths": [...] }`,
  {
    label: 'scout',
    schema: {
      type: 'object',
      properties: {
        all_paths: { type: 'array', items: { type: 'string' } },
        done_paths: { type: 'array', items: { type: 'string' } },
      },
      required: ['all_paths', 'done_paths'],
    },
  }
)

const doneSet = new Set(scoutResult.done_paths)
const todo = scoutResult.all_paths.filter(p => !doneSet.has(p))
log(`${scoutResult.all_paths.length} corpora files total | ${scoutResult.done_paths.length} cached | ${todo.length} to evaluate`)

// ---------------------------------------------------------------------------
// Phase 2: Evaluate
// ---------------------------------------------------------------------------
phase('Evaluate')

const EVAL_SCHEMA = {
  type: 'object',
  properties: {
    accepted: { type: 'boolean' },
    reason: { type: 'string' },
    category_name: { type: 'string' },
    description: { type: 'string' },
    frequency: { type: 'integer', minimum: 0, maximum: 100 },
    items: { type: 'array', items: { type: 'string' } },
    templates: { type: 'array', items: { type: 'string' } },
    python_class: { type: 'string' },
  },
  required: ['accepted', 'reason'],
}

await pipeline(
  todo,
  async (path, _orig, idx) => {
    const rawUrl = `${RAW_BASE}${path}`
    const cacheKey = path.replace(/\//g, '__')
    const cacheFile = `${EVAL_DIR}/${cacheKey}.json`

    const result = await agent(
      `You are evaluating a JSON dataset from dariusk/corpora to decide if it can serve as a
sentence category for a synthetic US-person profile generator.

Step 1 — Fetch the file:
  Use WebFetch to fetch: ${rawUrl}

Step 2 — Evaluate. BE AGGRESSIVE — when in doubt, reject.

  ACCEPT only if ALL of these are true:
  - The dataset contains a list of concrete items a fictional US person might individually
    have as a preference, hobby, or trait (cars, instruments, foods, sports, dog breeds, etc.)
  - Items naturally complete sentences like "He likes/drives/plays/collects/reads {item}."
  - A typical US adult would recognise most items without needing specialist knowledge.

  REJECT outright if any of these apply:
  - Geographic data, place names, country/city lists (even if "interesting")
  - Lists of people or names
  - Abstract, metaphorical, or numerical content (primes, math constants, zodiac symbols)
  - Technical/specialist knowledge most US adults lack (airport codes, chemical compounds, etc.)
  - Fewer than 10 usable items remain after filtering
  - Items describe groups or concepts, not things a single person owns/likes/does

  PROBLEM 1 — Too many obscure items: if the list is otherwise good but half the items are
  unknown to a typical US adult, choose one of:
    a) Reject entirely if the category itself is too niche (e.g. StreetFighterCharacters,
       LeagueOfLegendsChampions, EnglishPlaces, HotPeppers, SculptureMaterials)
    b) Filter down to the 5–10 most well-known items and assign a weight of 5 if the
       category is broadly relevant but the full list is too obscure
       (e.g. Scents → keep rose/vanilla/lavender/...; Gemstones → keep diamond/ruby/emerald/...)

  PROBLEM 2 — Items not obviously associated with the category: if reading an item in a
  sentence would confuse a reader (e.g. "Jake" as a dog name, "Plastic" as a building material,
  "Phoenix" as a US state capital), you must either:
    a) Filter those items out, OR
    b) Rewrite the templates to include enough context that the sentence is self-explanatory
       (e.g. "His favourite US state capital to visit is {item}." instead of "He lives in {item}.")

Step 3 — If ACCEPTED, produce:
  - category_name: short snake_case name (e.g. "music_genres", "favorite_vegetables")
  - description: one sentence describing the category
  - frequency: a sampling weight (integer, not a percentage — categories are normalised at
      sampling time, so only the relative values matter). Use these tiers as a guide:
        60 → nearly universal trait, almost every adult has a clear answer (music_genres, car_brand, favorite_books, religions, work_industry)
        40 → very common, most people engage with it (tv_shows, movie_genres, us_cities, board_games)
        20 → fairly common, a solid minority of people (sports, fruits, moods, favorite_color, nfl_teams)
        10 → uncommon but mainstream (cat_breeds, famous_duos, wine_tasting_descriptors, scents)
         5 → niche or specialist interest (programming_languages, scotch_whisky, shakespeare_plays, norse_gods, cannabis_strains)
         0 → keep in file but never sample (internal use only)
      Filtered-down borderline categories should be ≤ 5.
  - items: the final filtered list of usable items only.
  - templates: 5–10 varied sentence templates. Rules:
      * Use these placeholders: {He} {he} {His} {his} {Him} {him} {Himself} {himself} {item}
        ({He}/{he} = He/She, {His}/{his} = His/Her, {Him}/{him} = Him/Her, {Himself}/{himself} = Himself/Herself)
      * EVERY template must work grammatically and naturally with EVERY item in the final list.
        Bad: "He keeps a {item} in his yard." — works for "dog" but not "jazz guitar".
      * Vary structure, tense, and context: current habit, childhood memory, aspiration, quirk, etc.
      * Good variety example for books: "His favourite book is {item}.", "{item} was {his} favourite
        book growing up.", "He has read {item} four times already.", "He always recommends {item}
        to anyone who asks.", "Last week he couldn't put {item} down."
  - python_class: the full Python dataclass code block for this category, using this exact format:

@dataclass
class _Cat_<PascalCaseName>:
    pool: ClassVar[list] = [<items as a JSON-style Python list, one item per line, each quoted>]
    templates: ClassVar[list] = [<templates list>]
    pool_size: ClassVar[int] = <len(items)>
    avg_template_length: ClassVar[float] = <mean char length of templates, 1 decimal>
    frequency: ClassVar[int] = <frequency>
    description: ClassVar[str] = '<description>'
    category_name: ClassVar[str] = '<category_name>'

    def __call__(self, seed=None, gender='male'):
        rng = random.Random(seed)
        pronouns = _PRONOUNS[gender]
        item = rng.choice(self.pool)
        template = rng.choice(self.templates)
        return {"sentence": template.format(item=item, **pronouns),
                "facts": {"category": self.category_name, "item": item}}

Step 4 — Write result to file:
  Use the Write tool to write the evaluation result as JSON to: ${cacheFile}
  The JSON should include all fields above (accepted, reason, category_name, description,
  frequency, items, templates, python_class, and add "_path": "${path}").

Return the evaluation result as structured output.`,
      {
        label: path,
        model: 'haiku',
        schema: EVAL_SCHEMA,
        phase: 'Evaluate',
      }
    )
    return result
  }
)

// ---------------------------------------------------------------------------
// Phase 3: Assemble
// ---------------------------------------------------------------------------
phase('Assemble')

await agent(
  `Read all JSON files matching the glob pattern "${EVAL_DIR}/*.json".
For each file, parse the JSON. Filter to only those where "accepted" is true AND "python_class" is a non-empty string.
Sort the accepted ones alphabetically by "category_name".

Then write the file "src/profile_sentences.py" with this exact structure:

--- FILE START ---
"""Auto-generated by dataset/corpus_workflow.js from dariusk/corpora."""
from dataclasses import dataclass, field
from typing import ClassVar
import random
import csv as _csv
import os as _os

def _load_name_csv(filename):
    path = _os.path.join(_os.path.dirname(__file__), "pools", filename)
    with open(path, encoding='utf-8', newline='') as f:
        return [row['name'] for row in _csv.DictReader(f)]

_PRONOUNS = {
    'male':   {'He': 'He',  'he': 'he',  'His': 'His', 'his': 'his',
               'Him': 'Him', 'him': 'him', 'Himself': 'Himself', 'himself': 'himself',
               'HisP': 'his', 'hisP': 'his'},
    'female': {'He': 'She', 'he': 'she', 'His': 'Her', 'his': 'her',
               'Him': 'Her', 'him': 'her', 'Himself': 'Herself', 'himself': 'herself',
               'HisP': 'Hers', 'hisP': 'hers'},
}


<INSERT EACH python_class BLOCK HERE, separated by a blank line>


@dataclass
class _Cat_MaleName:
    pool: ClassVar[list] = _load_name_csv('male_names.csv')
    pool_size: ClassVar[int] = 0
    frequency: ClassVar[int] = 0
    description: ClassVar[str] = 'Male first names'
    category_name: ClassVar[str] = 'male_name'

    def __call__(self, seed=None, gender='male'):
        rng = random.Random(seed)
        return {"sentence": "", "facts": {"category": "male_name", "item": rng.choice(self.pool)}}

@dataclass
class _Cat_FemaleName:
    pool: ClassVar[list] = _load_name_csv('female_names.csv')
    pool_size: ClassVar[int] = 0
    frequency: ClassVar[int] = 0
    description: ClassVar[str] = 'Female first names'
    category_name: ClassVar[str] = 'female_name'

    def __call__(self, seed=None, gender='male'):
        rng = random.Random(seed)
        return {"sentence": "", "facts": {"category": "female_name", "item": rng.choice(self.pool)}}

@dataclass
class _Cat_FamilyName:
    pool: ClassVar[list] = _load_name_csv('family_names.csv')
    pool_size: ClassVar[int] = 0
    frequency: ClassVar[int] = 0
    description: ClassVar[str] = 'Family surnames'
    category_name: ClassVar[str] = 'family_name'

    def __call__(self, seed=None, gender='male'):
        rng = random.Random(seed)
        return {"sentence": "", "facts": {"category": "family_name", "item": rng.choice(self.pool)}}


@dataclass
class _Generate:
    <one line per accepted corpus category: "    <category_name>: _Cat_<PascalCase> = field(default_factory=_Cat_<PascalCase>)">
    male_name: _Cat_MaleName = field(default_factory=_Cat_MaleName)
    female_name: _Cat_FemaleName = field(default_factory=_Cat_FemaleName)
    family_name: _Cat_FamilyName = field(default_factory=_Cat_FamilyName)


generate = _Generate()
--- FILE END ---

Important:
- The accepted corpus category classes go BEFORE the name pool classes.
- The _Generate fields list ALL categories (corpus + 3 name pools).
- Use the Write tool to write the file.
- Return how many categories were written.`,
  { label: 'assemble' }
)

log('Done. src/profile_sentences.py written.')
