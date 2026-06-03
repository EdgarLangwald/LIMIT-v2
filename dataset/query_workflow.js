// THIS IS A SCRIPT FOR CLAUDE CODE'S "DYNAMIC WORKFLOW" TOOL (requires Opus 4.7+)
//
// LIMIT-v2 query generation workflow (run via the Workflow tool, from the repo root):
//   Workflow({ scriptPath: "dataset/query_workflow.js", args: { seed: 42, n_units: 100 } })
//
// Prereq: `python dataset/build_query_workflow_input.py --dump-specs --seed S --n N` must have
//         written dataset/query_workflow_input/unit_NNN.json. Also `mkdir dataset/query_workflow_output`.
// After:  `python dataset/build_query_workflow_input.py --render --seed S` renders the final dataset.
//
// 100 Haiku agents, one per unit. Each unit = 2 draws of 3 categories. Per draw the agent
// picks 2 DISTINCT template indices per category and writes a SPECIFIC and a BROAD query —
// both must (a) describe each item without naming it, (b) be SELECTIVE (the broad query
// narrows to a sub-family, never just the bare category), (c) be true under BOTH chosen
// templates of all 3 categories. Result -> dataset/query_workflow_output/unit_NNN.json.

export const meta = {
  name: 'query-gen',
  description: 'Generate selective specific+broad queries + 2-template picks for 100 unit specs (100 Haiku agents)',
  phases: [{ title: 'Generate', detail: '100 Haiku agents, one per unit spec; each writes its result file' }],
}

const SEED = (args && args.seed) || 42
const N = (args && args.n_units) || 100
const SPEC_DIR = `dataset/query_workflow_input`
const RES_DIR = `dataset/query_workflow_output`

const pad = (n) => String(n).padStart(3, '0')

function buildPrompt(u) {
  const specPath = `${SPEC_DIR}/unit_${pad(u)}.json`
  const resPath = `${RES_DIR}/unit_${pad(u)}.json`
  return `You are writing evaluation queries for an information-retrieval benchmark that stress-tests
  embedding models over MILLIONS of person-profile documents. A query must be SELECTIVE: only the handful
  of profiles that genuinely match should rank highly. Be precise and follow the output contract exactly.

STEP 1 - Read the spec.
Read the file "${specPath}". It is JSON with "unit_idx" and a "draws" array of exactly 2 draws. For each
draw use ONLY draws[i].slots, a list of 3 categories. Each slot has:
  - "category_name": string
  - "item": the specific trait value (e.g. "bagpipe")
  - "templates": a 0-indexed list of template sentences containing {item} (the trait value) and pronoun
  placeholders like {He}, {his}, {him}. To read a template, mentally substitute the item and a he/his
  pronoun.
Ignore every other field (targets_meta, fillers) - a separate script renders those later.

STEP 2 - For EACH category in EACH draw, pick the templates and craft two descriptors.
  (a) Pick EXACTLY 2 DISTINCT template indices (0-based, valid for that list). The two MUST be mutually
  consistent - they describe the same kind of relationship to the trait - because all combinations are
  rendered into profiles, and a query must be true of both.
  (b) Write a BROAD descriptor of the item: a recognizable SUB-FAMILY / SUB-TYPE the item clearly belongs
  to. It must be NARROWER than the whole category (it excludes most other items in the category) but wider
  than a unique description. NEVER write just "a/an <category>" or "a particular/specific/certain
  <category>" - that matches almost everyone and is useless.
  (c) Write a SPECIFIC descriptor of the item: a precise description, using distinguishing attributes,
  that points to (almost) this exact item among its category.
  NEITHER descriptor may contain the item's name, nor any proper noun or distinctive identifying word
  from it (brand/place/person/species names, e.g. "Honda", "Orlando", "Batista", "Mandarin"). Generic
  category words ("duck", "play", "breed", "fish") are fine.
  Both descriptors must be TRUE under BOTH chosen templates.

STEP 3 - Assemble the two queries (per draw).
  - query_specific: ONE fluent, natural question describing a person with all 3 SPECIFIC descriptors woven
  together.
  - query_broad: the SAME person with all 3 BROAD descriptors woven together. Clearly less specific than
  query_specific, but still SELECTIVE (each trait narrowed to a sub-family).
  No names, no item strings, not a bare comma list.

WORKED EXAMPLES - these define the quality bar (study the GOOD vs BAD contrast):

  instruments, item "bagpipe" (two "plays since youth" templates chosen):
    BROAD    GOOD: "a woodwind"
    BROAD    BAD:  "a musical instrument"  (matches everyone)
    SPECIFIC GOOD: "a loud Scottish reed instrument they have played since childhood"
    SPECIFIC BAD : "the bagpipe"                     (names the item - FORBIDDEN)

  shakespeare_plays, item "Much Ado About Nothing":
    BROAD    GOOD: "a Shakespeare comedy"
    BROAD    BAD:  "a particular Shakespeare play"  (just the category)
    SPECIFIC GOOD: "a Shakespeare comedy of witty sparring lovers and a slandered bride"
    SPECIFIC BAD : "Much Ado About Nothing"         (names the item - FORBIDDEN)

  cat_breeds, item "Savannah":
    BROAD    GOOD: "a large exotic hybrid cat breed"
    BROAD    BAD: "a specific cat breed"  (just the category)
    SPECIFIC GOOD: "a tall, spotted cat breed descended from a wild African ancestor"
    SPECIFIC BAD : "a Savannah cat"                  (names the item - FORBIDDEN)

  Assembled for a draw with those three:
    query_broad   : "Who plays a woodwind, loves a Shakespeare comedy, and keeps a large exotic hybrid cat breed?"
    query_specific: "Who has played a loud Scottish reed instrument since childhood, is devoted to a
    Shakespeare comedy of witty sparring lovers and a slandered bride, and keeps a tall spotted cat
    breed descended from a wild African ancestor?"
  Both are selective (few items match) AND name no item AND stay true to the chosen templates.

  Contrast - a BAD broad query (do NOT produce this): "Who is fond of a specific Shakespeare play,
  prefers a particular cat breed, and enjoys a favorite sandwich?" -> every profile with those three
  categories matches; worthless.

STEP 4 - Self-review BEFORE writing. For each draw and category confirm ALL of:
  1. both indices are integers in [0, len(templates)-1] and the two are DISTINCT;
  2. both chosen templates, rendered with the item, are mutually consistent;
  3. query_specific AND query_broad are each TRUE of BOTH chosen templates;
  4. neither query contains the item name or a distinctive proper-noun word from it (scan both queries
  against each item, case-insensitively);
  5. each BROAD descriptor is a real sub-family, NOT a bare category mention - a stranger reading it
  could exclude most items in that category;
  6. query_specific is clearly more specific than query_broad;
  7. output category order matches slot order in the spec.
Fix anything that fails, then proceed.

STEP 5 - Write the result file. Write EXACTLY this JSON (no markdown fences, no commentary) to "${resPath}":
{
  "unit_idx": ${u},
  "draws": [
    {
      "draw_idx": 0,
      "query_specific": "...",
      "query_broad": "...",
      "categories": [
        {"category_name": "<slot 0 name>", "template_indices": [i, j]},
        {"category_name": "<slot 1 name>", "template_indices": [i, j]},
        {"category_name": "<slot 2 name>", "template_indices": [i, j]}
      ]
    },
    { "draw_idx": 1, "query_specific": "...", "query_broad": "...", "categories": [ ...same shape... ] }
  ]
}
The "category_name" values MUST equal the slot category names in spec order.

STEP 6 - Return ONLY a one-line status: "unit ${u} OK" if the file was written, or "unit ${u} FAIL:
<reason>". Your final message IS the return value - nothing else.`
}

phase('Generate')
const statuses = await parallel(
  Array.from({ length: N }, (_, u) => () =>
    agent(buildPrompt(u), { label: `unit ${pad(u)}`, phase: 'Generate', model: 'haiku' })
  )
)

const fails = statuses
  .map((s, u) => ({ u, s }))
  .filter(({ s }) => !(typeof s === 'string' && /\bOK\b/.test(s)))
  .map(({ u, s }) => ({ unit: u, status: s }))

log(`OK ${N - fails.length}/${N}; failures: ${fails.length}`)
return { n_units: N, ok: N - fails.length, fails }