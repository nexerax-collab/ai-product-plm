"""
Multi-Agent PLM System for AI-Driven Product Configuration
==========================================================
Architecture:
  User Prompt
  → Orchestrator
  → Product Family Agent (defines product line: features, options, constraints, variants)
  → Configurator Agent   (selects valid configuration from family + builds BOM)
  → Evaluator Agent      (scores dimensions defined by the product family)
  → Optimizer Agent      (fixes critical issues first, then improves scores)
  → PLM Agent            (persists parts + BOM to Airtable)
  → CAD Agent            (maps BOM → parametric geometry → Onshape via MCP)

Setup:
  1. Copy .env.example to .env and fill in your API keys.
  2. pip install -r requirements.txt
  3. python plm_agents.py
"""

import json
import os
import re
from dataclasses import dataclass

# Load .env file if present (python-dotenv)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv optional — use shell env vars or set values directly below

import anthropic
import requests

# ─────────────────────────────────────────────────────────────
# CONFIGURATION  —  set via .env or environment variables
# ─────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
AIRTABLE_TOKEN    = os.getenv("AIRTABLE_TOKEN",    "")
AIRTABLE_BASE_ID  = os.getenv("AIRTABLE_BASE_ID",  "")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY",    "")

CLAUDE_MODEL       = "claude-opus-4-6"          # CAD reasoning (adaptive thinking)
CLAUDE_MODEL_MID   = "claude-sonnet-4-6"        # config / eval / optimize
MAX_ITER           = 5   # hard cap on iterations


# ─────────────────────────────────────────────────────────────
# INTENT  –  what the user actually wants
# ─────────────────────────────────────────────────────────────

@dataclass
class Intent:
    """
    Describes what to optimise and what hard constraints must hold.

    goal        : what to maximise / focus on
                  e.g. "maximum flight time", "best FPV racing performance"
    constraints : hard limits that must never be violated
                  e.g. ["cost < €500", "total weight < 1kg", "waterproof IP65"]
    context     : optional extra info (use case, environment, user level)
                  e.g. "used outdoors in wind, operator is a beginner"

    Example:
        Intent(
            goal        = "maximum flight time",
            constraints = ["cost below €500", "frame size max 5 inch"],
            context     = "recreational weekend flying, beginner pilot",
        )
    """
    goal:        str
    constraints: list[str]
    context:     str = ""

    def as_prompt_block(self) -> str:
        """Format for injection into agent prompts."""
        lines = [f"Goal: {self.goal}"]
        if self.constraints:
            lines.append("Hard constraints (must ALL be satisfied):")
            for c in self.constraints:
                lines.append(f"  - {c}")
        if self.context:
            lines.append(f"Context: {self.context}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return f"Goal={self.goal!r} | Constraints={self.constraints} | Context={self.context!r}"

_AIRTABLE_SCHEMA = [
    {"name": "Product Families", "fields": [
        {"name": "Name",         "type": "singleLineText"},
        {"name": "Product Type", "type": "singleLineText"},
        {"name": "Description",  "type": "multilineText"},
    ]},
    {"name": "Features", "fields": [
        {"name": "Name",   "type": "singleLineText"},
        {"name": "Type",   "type": "singleLineText"},
        {"name": "Family", "type": "singleLineText"},
    ]},
    {"name": "Feature Options", "fields": [
        {"name": "Feature", "type": "singleLineText"},
        {"name": "Value",   "type": "singleLineText"},
        {"name": "Family",  "type": "singleLineText"},
    ]},
    {"name": "Constraints", "fields": [
        {"name": "Rule",   "type": "singleLineText"},
        {"name": "Family", "type": "singleLineText"},
    ]},
    {"name": "Parts", "fields": [
        {"name": "part_number", "type": "singleLineText"},
        {"name": "name",        "type": "singleLineText"},
        {"name": "category",    "type": "singleLineText"},
        {"name": "active",      "type": "checkbox", "options": {"color": "greenBright", "icon": "check"}},
    ]},
    {"name": "BOM", "fields": [
        {"name": "parent",      "type": "singleLineText"},
        {"name": "quantity",    "type": "number", "options": {"precision": 0}},
        {"name": "level",       "type": "number", "options": {"precision": 0}},
        {"name": "notes",       "type": "singleLineText"},
    ]},
]


def setup_airtable() -> None:
    """Create any missing Airtable tables. Safe to run on every startup."""
    url     = f"https://api.airtable.com/v0/meta/bases/{AIRTABLE_BASE_ID}/tables"
    r       = requests.get(url, headers=AIRTABLE_HEADERS)
    if not r.ok:
        print(f"  ⚠  Could not read Airtable schema: {r.status_code} — skipping table setup.")
        return

    existing = {t["name"] for t in r.json().get("tables", [])}
    created  = []

    for table in _AIRTABLE_SCHEMA:
        if table["name"] in existing:
            continue
        resp = requests.post(url, headers=AIRTABLE_HEADERS, json=table)
        if resp.ok:
            created.append(table["name"])
        else:
            print(f"  ⚠  Could not create table '{table['name']}': {resp.text[:120]}")

    if created:
        print(f"  ✓ Airtable tables created: {', '.join(created)}")


def _check_config() -> None:
    """Fail fast with a clear message if required keys are missing."""
    required = {
        "ANTHROPIC_API_KEY": ANTHROPIC_API_KEY,
        "AIRTABLE_TOKEN":    AIRTABLE_TOKEN,
        "AIRTABLE_BASE_ID":  AIRTABLE_BASE_ID,
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        print("\n  ERROR: Missing required environment variables:")
        for k in missing:
            print(f"    • {k}")
        print("\n  Copy .env.example → .env and fill in your keys.")
        print("  Or export them in your shell before running.\n")
        raise SystemExit(1)

    # Onshape keys are only required when CAD is used — warn, don't abort
    onshape = {
        "ONSHAPE_DID": _ONSHAPE_DID, "ONSHAPE_WID": _ONSHAPE_WID,
        "ONSHAPE_EID": _ONSHAPE_EID, "ONSHAPE_ACCESS_KEY": _ONSHAPE_ACCESS_KEY,
        "ONSHAPE_SECRET_KEY": _ONSHAPE_SECRET_KEY,
    }
    missing_os = [k for k, v in onshape.items() if not v]
    if missing_os:
        print(f"  ⚠  Onshape keys not set ({', '.join(missing_os)}) — CAD agent will be unavailable.")

    if not OPENAI_API_KEY:
        print(f"  ⚠  OPENAI_API_KEY not set — image generation will be unavailable.")


claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

AIRTABLE_HEADERS = {
    "Authorization": f"Bearer {AIRTABLE_TOKEN}",
    "Content-Type":  "application/json",
}


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def call_claude(prompt: str, system: str = "",
                model: str = CLAUDE_MODEL_MID, max_tokens: int = 2048,
                cache_system: bool = False) -> str:
    """
    Call Claude and return the text response.

    cache_system=True marks the system prompt for Anthropic prompt caching
    (ephemeral, 5-minute TTL). Use for agents called repeatedly in a loop
    (evaluator, optimizer) to pay ~10% of input cost on cache hits.
    """
    kwargs = dict(
        model      = model,
        max_tokens = max_tokens,
        messages   = [{"role": "user", "content": prompt}],
    )
    if system:
        if cache_system:
            kwargs["system"] = [{"type": "text", "text": system,
                                 "cache_control": {"type": "ephemeral"}}]
        else:
            kwargs["system"] = system

    response = claude.messages.create(**kwargs)

    for block in response.content:
        if block.type == "text":
            return block.text

    print(f"  ⚠ No text block. Block types: {[b.type for b in response.content]}")
    return ""


def extract_json(text: str) -> dict | list:
    """Extract the first JSON object or array from a Claude reply."""
    if not text or not text.strip():
        raise ValueError("Claude returned an empty response")

    # Prefer fenced ```json ... ``` block
    match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\}|\[[\s\S]*?\])\s*```", text)
    if match:
        return json.loads(match.group(1))

    # Fallback: find outermost braces/brackets
    brace   = text.find("{")
    bracket = text.find("[")

    if brace == -1 and bracket == -1:
        print(f"\n  ⚠ No JSON found in response:\n{text[:400]}")
        raise ValueError("No JSON found in Claude response")

    if brace == -1:
        start = bracket
    elif bracket == -1:
        start = brace
    else:
        start = min(brace, bracket)

    end = max(text.rfind("}"), text.rfind("]")) + 1
    if end <= start:
        print(f"\n  ⚠ Malformed JSON in response:\n{text[:400]}")
        raise ValueError("Malformed JSON in Claude response")

    return json.loads(text[start:end])


def separator(title: str):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")


def has_critical_issues(evaluation: dict) -> bool:
    """Return True if any issue is marked critical."""
    return any(
        i.get("type") == "critical"
        for i in evaluation.get("issues", [])
    )


def _primary_metric(intent: Intent, family: dict | None = None) -> str:
    """
    Infer the most important scoring dimension.
    Checks the intent goal against family scoring_dimension names/descriptions.
    Falls back to the first dimension, or 'quality' if no family.
    """
    dims = (family or {}).get("scoring_dimensions", [])
    if not dims:
        return "quality"

    goal = intent.goal.lower() if intent else ""
    # Score each dimension by how many of its words appear in the goal
    best, best_score = dims[0]["name"], 0
    for d in dims:
        keywords = (d["name"] + " " + d.get("description", "")).lower().split()
        score = sum(1 for w in keywords if len(w) > 3 and w in goal)
        if score > best_score:
            best, best_score = d["name"], score
    return best


def should_stop(evaluation: dict, iteration: int,
                intent: Intent | None = None, family: dict | None = None) -> bool:
    """
    Stop when: no critical issues AND the primary scoring dimension >= 8.
    Always run at least 2 iterations.
    """
    if iteration < 2:
        return False
    if has_critical_issues(evaluation):
        return False
    scores = evaluation.get("scores", {})
    metric = _primary_metric(intent, family)
    return scores.get(metric, 0) >= 8


# ─────────────────────────────────────────────────────────────
# AGENT 0 – PRODUCT FAMILY
# Defines the product line: features, options, constraints, variants
# Persists to Airtable: Product Families / Features / Feature Options / Constraints
# ─────────────────────────────────────────────────────────────

_FAMILY_SYSTEM = (
    "You are a product portfolio and variant management expert. "
    "Define practical product families like in Configit or pure::variants. "
    "Think like a product line engineer — keep it concrete and usable. "
    "Output JSON only."
)

_FAMILY_AT_TABLES = {
    "families":    "Product Families",
    "features":    "Features",
    "options":     "Feature Options",
    "constraints": "Constraints",
}


def _persist_family(family_def: dict) -> dict:
    """Write a family definition to Airtable. Returns counts per table."""
    family      = family_def.get("family", {})
    family_name = family.get("name", "Unknown")
    counts      = {k: 0 for k in _FAMILY_AT_TABLES}

    # Product Families — one record
    url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{_FAMILY_AT_TABLES['families']}"
    r = requests.post(url, headers=AIRTABLE_HEADERS, json={
        "records": [{"fields": {
            "Name":         family_name,
            "Product Type": family.get("product_type", ""),
            "Description":  family.get("description", ""),
        }}]
    })
    if r.ok:
        counts["families"] = 1
    else:
        print(f"  ⚠ Product Families write: {r.status_code} {r.text[:150]}")

    # Features — one per feature
    feat_fields = [
        {"Name": f.get("name", ""), "Type": f.get("type", "enum"), "Family": family_name}
        for f in family_def.get("features", [])
    ]
    if feat_fields:
        # reuse _batch_create — defined later, resolved at call-time
        _batch_create(_FAMILY_AT_TABLES["features"], feat_fields)
        counts["features"] = len(feat_fields)

    # Feature Options — one row per (feature, value) pair
    opt_fields = []
    for opt in family_def.get("options", []):
        feat = opt.get("feature", "")
        for val in opt.get("values", []):
            opt_fields.append({"Feature": feat, "Value": str(val), "Family": family_name})
    if opt_fields:
        _batch_create(_FAMILY_AT_TABLES["options"], opt_fields)
        counts["options"] = len(opt_fields)

    # Constraints — plain text rules
    con_fields = [
        {"Rule": c if isinstance(c, str) else str(c), "Family": family_name}
        for c in family_def.get("constraints", [])
    ]
    if con_fields:
        _batch_create(_FAMILY_AT_TABLES["constraints"], con_fields)
        counts["constraints"] = len(con_fields)

    return counts


def product_family_agent(product_idea: str) -> dict:
    """
    Generate a product family definition from a free-form product idea.

    Returns:
    {
        "family":             {"name", "product_type", "description"},
        "features":           [{"name", "type"}, ...],
        "options":            [{"feature", "values": [...]}, ...],
        "constraints":        ["plain-English rule", ...],
        "variants":           [{"name", "description", "configuration": {...}}, ...],
        "scoring_dimensions": [{"name", "description"}, ...]   ← used by evaluator/optimizer
    }
    """
    separator("PRODUCT FAMILY AGENT")
    print(f"  Product idea: {product_idea}")

    prompt = f"""
Define a practical product family for this idea.

Product idea: {product_idea}

Return exactly this JSON structure:

{{
  "family": {{
    "name":         "string",
    "product_type": "string",
    "description":  "1-2 sentence description"
  }},
  "features": [
    {{"name": "string", "type": "enum|boolean|numeric"}}
  ],
  "options": [
    {{"feature": "<feature name>", "values": ["opt1", "opt2", ...]}}
  ],
  "constraints": [
    "plain English rule, e.g. 'large motor requires reinforced frame'"
  ],
  "variants": [
    {{
      "name": "string (e.g. entry-level / standard / pro)",
      "description": "string",
      "configuration": {{"<feature_name>": "<selected_option>"}}
    }}
  ],
  "scoring_dimensions": [
    {{"name": "string (snake_case, e.g. range_km)", "description": "what this measures"}}
  ]
}}

Rules:
- 5-10 features, 2-6 options per enum feature, 3-6 constraints, 2-3 variants.
- scoring_dimensions: exactly 3 dimensions that matter most for this product type.
  Examples for a bicycle: range_km, comfort, cost. For a camera: image_quality, portability, cost.
- All options must be real-world, specific values (not vague like "good battery").
- Variants must each satisfy all constraints.
- Output JSON only, no markdown outside the block.
""".strip()

    raw        = call_claude(prompt, system=_FAMILY_SYSTEM, max_tokens=2500)
    family_def = extract_json(raw)

    f     = family_def.get("family", {})
    feats = [x["name"] for x in family_def.get("features", [])]
    vars_ = [v["name"] for v in family_def.get("variants", [])]
    dims  = [d["name"] for d in family_def.get("scoring_dimensions", [])]

    print(f"\n  ✓ Family     : {f.get('name')} ({f.get('product_type')})")
    print(f"  ✓ Features   : {feats}")
    print(f"  ✓ Constraints: {len(family_def.get('constraints', []))} rules")
    print(f"  ✓ Variants   : {vars_}")
    print(f"  ✓ Scoring    : {dims}")

    # Persist to Airtable
    counts = _persist_family(family_def)
    print(f"  ✓ Airtable   : {counts['families']} family | "
          f"{counts['features']} features | "
          f"{counts['options']} options | "
          f"{counts['constraints']} constraints")

    return family_def


# ─────────────────────────────────────────────────────────────
# AGENT 1 – CONFIGURATOR
# ─────────────────────────────────────────────────────────────

def configurator_agent(intent: Intent, family: dict | None = None) -> dict:
    """
    Generate a feature model, valid configuration, and initial BOM.

    Returns:
    {
        "features":      { feature: [option, ...] },
        "configuration": { feature: selected_option },
        "constraints":   ["C1: ...", ...],
        "bom":           [{"part_number", "name", "category", "quantity"}, ...]
    }
    """
    separator("CONFIGURATOR AGENT")
    print(f"  {intent}")

    # Build optional family context block
    family_block = ""
    if family:
        feat_lines  = "\n".join(f"  - {f['name']} ({f['type']})"
                                for f in family.get("features", []))
        opt_lines   = "\n".join(f"  - {o['feature']}: {o['values']}"
                                for o in family.get("options", []))
        con_lines   = "\n".join(f"  - {c}" for c in family.get("constraints", []))
        family_name = family.get("family", {}).get("name", "")
        family_block = f"""
Product family: {family_name}

Available features and options — you MUST select from these:
{feat_lines}

Options per feature:
{opt_lines}

Family constraints (must all be satisfied):
{con_lines}
""".strip()

    product_type = family.get("family", {}).get("product_type", "product") if family else "product"

    prompt = f"""
You are a PLM configurator for: {product_type}.
Output ONLY a valid JSON object with these keys:

"features"      : dict mapping feature name → list of possible options
"configuration" : dict mapping feature name → chosen option (must satisfy all constraints)
"constraints"   : list of constraint strings
"bom"           : list of parts, each with:
    "part_number" (unique string, e.g. "STR-001"),
    "name"        (string),
    "category"    (appropriate category for this product type),
    "quantity"    (integer)

{family_block + chr(10) if family_block else ""}User intent:
{intent.as_prompt_block()}

Rules:
- ALL hard constraints must be satisfied — they are non-negotiable.
- Optimise the configuration toward the stated goal.
- Use realistic real-world parts or components with appropriate specs.
- Include at least 8 parts in the BOM.
- Output JSON only — no explanation, no markdown outside the JSON.
""".strip()

    system = "You are a PLM configurator. Output JSON only."
    for attempt in range(1, 3):
        raw = call_claude(prompt, system=system, max_tokens=3000)
        try:
            result = extract_json(raw)
            break
        except Exception as e:
            if attempt == 1:
                print(f"  ⚠ Configurator JSON parse error — retrying...")
                prompt += "\n\nIMPORTANT: Output ONLY the JSON object. No prose, no markdown, no trailing text."
            else:
                raise RuntimeError(f"Configurator failed to return valid JSON: {e}") from e
    result["_intent"] = intent  # carry intent forward for other agents
    result["_family"] = family  # carry family forward for evaluator/optimizer

    print(f"\n  ✓ Features : {list(result.get('features', {}).keys())}")
    print(f"  ✓ Config   : {result.get('configuration', {})}")
    print(f"  ✓ BOM      : {len(result.get('bom', []))} parts")
    return result


# ─────────────────────────────────────────────────────────────
# AGENT 2 – EVALUATOR
# ─────────────────────────────────────────────────────────────

def evaluator_agent(config: dict) -> dict:
    """
    Score the configuration against the family's scoring dimensions and identify issues.

    Returns:
    {
        "scores":  {<dim_name>: 0-10, ...},   ← keys match family scoring_dimensions
        "issues":  [{"type": "critical"|"normal", "text": "..."}, ...],
        "summary": "one-sentence verdict"
    }
    """
    separator("EVALUATOR AGENT")
    intent: Intent = config.get("_intent")
    family: dict   = config.get("_family") or {}

    config_data = {
        "configuration": config.get("configuration", {}),
        "bom":           config.get("bom", []),
        "constraints":   config.get("constraints", []),
    }

    dims         = family.get("scoring_dimensions", [])
    product_type = family.get("family", {}).get("product_type", "product")
    intent_block = intent.as_prompt_block() if intent else ""

    # Build score schema from family dimensions
    score_schema = "\n".join(
        f'    "{d["name"]}": <integer 0-10>   // {d["description"]}'
        for d in dims
    ) or '    "quality": <integer 0-10>'

    prompt = f"""
You are a product engineering evaluator for: {product_type}.
Evaluate the configuration below. Output ONLY a valid JSON object with these keys:

"scores": {{
{score_schema}
}}
(10 = excellent, 0 = terrible. Be realistic and critical. Score each dimension independently.)

"issues": list of issue objects, each with:
    "type": "critical" (hard-constraint violation, safety risk, or fundamental incompatibility)
         or "normal"   (suboptimal but acceptable)
    "text": short description of the issue

"summary": one sentence overall verdict

User intent:
{intent_block}

Product configuration:
{json.dumps(config_data)}

Output JSON only.
""".strip()

    system = "You are a product engineering evaluator. Output JSON only."
    for attempt in range(1, 3):
        raw = call_claude(prompt, system=system, max_tokens=2048, cache_system=True)
        try:
            result = extract_json(raw)
            break
        except (ValueError, Exception) as e:
            if attempt == 1:
                print(f"  ⚠ Evaluator JSON parse error — retrying with stricter prompt...")
                prompt += "\n\nIMPORTANT: Output ONLY the JSON object. No prose, no markdown, no trailing text."
            else:
                print(f"  ⚠ Evaluator failed twice ({e}) — returning empty evaluation.")
                result = {"scores": {}, "issues": [], "summary": "evaluation failed"}

    scores   = result.get("scores", {})
    issues   = result.get("issues", [])
    critical = [i["text"] for i in issues if i.get("type") == "critical"]
    normal   = [i["text"] for i in issues if i.get("type") == "normal"]

    score_str = " | ".join(f"{k}: {v}/10" for k, v in scores.items()) or "n/a"
    print(f"\n  Scores  → {score_str}")
    if critical:
        print(f"  Critical: {critical}")
    if normal:
        print(f"  Normal  : {normal}")
    print(f"  Summary → {result.get('summary', '')}")
    return result


# ─────────────────────────────────────────────────────────────
# AGENT 3 – OPTIMIZER
# ─────────────────────────────────────────────────────────────

def optimizer_agent(config: dict, evaluation: dict) -> dict:
    """
    Improve the configuration by fixing critical issues first,
    then improving scores.

    Returns:
    {
        "configuration": { ... },
        "bom":           [ ... ],
        "changes":       ["what changed and why", ...]
    }
    """
    separator("OPTIMIZER AGENT")

    critical = [i["text"] for i in evaluation.get("issues", []) if i.get("type") == "critical"]
    normal   = [i["text"] for i in evaluation.get("issues", []) if i.get("type") == "normal"]

    if critical:
        print(f"  Priority: fix {len(critical)} critical issue(s) first")
    else:
        print(f"  No critical issues — improving scores")

    intent: Intent = config.get("_intent")
    intent_block   = intent.as_prompt_block() if intent else ""
    # Only send what the optimizer needs — skip the full feature option lists
    config_data = {
        "configuration": config.get("configuration", {}),
        "bom":           config.get("bom", []),
        "constraints":   config.get("constraints", []),
    }

    family: dict   = config.get("_family") or {}
    product_type   = family.get("family", {}).get("product_type", "product")
    dims           = family.get("scoring_dimensions", [])
    dim_labels = ", ".join(d["name"] for d in dims) if dims else "quality"

    prompt = f"""
You are a product design optimizer for: {product_type}.
Improve the configuration below to fix issues and raise scores.

User intent:
{intent_block}

Priority order:
1. Fix ALL critical issues — especially any hard constraint violations.
2. Then address normal issues where possible.
3. Then improve scores toward the goal, focusing on: {dim_labels}
4. Never violate a hard constraint while optimising.

Output ONLY a valid JSON object with these keys:
"configuration": dict of feature → selected option (updated)
"bom":           full updated parts list (same schema: part_number, name, category, quantity)
"changes":       list of short strings explaining what changed and why

Current configuration:
{json.dumps(config_data)}

Evaluation:
- Scores: {json.dumps(evaluation.get('scores', {}))}
- Critical issues: {json.dumps(critical)}
- Normal issues:   {json.dumps(normal)}

Rules:
- Every critical issue must be resolved.
- All hard constraints must remain satisfied after changes.
- Keep the BOM realistic and self-consistent.
- Output JSON only.
""".strip()

    system = "You are a product design optimizer. Output JSON only."
    for attempt in range(1, 3):
        raw = call_claude(prompt, system=system, max_tokens=3000, cache_system=True)
        try:
            result = extract_json(raw)
            break
        except Exception as e:
            if attempt == 1:
                print(f"  ⚠ Optimizer JSON parse error — retrying...")
                prompt += "\n\nIMPORTANT: Output ONLY the JSON object. No prose, no markdown, no trailing text."
            else:
                raise RuntimeError(f"Optimizer failed to return valid JSON: {e}") from e
    result["_intent"] = config.get("_intent")  # carry intent forward
    result["_family"] = config.get("_family")  # carry family forward

    print(f"\n  Changes:")
    for change in result.get("changes", []):
        print(f"    • {change}")
    print(f"  Updated BOM: {len(result.get('bom', []))} parts")
    return result


# ─────────────────────────────────────────────────────────────
# AGENT 4 – PLM AGENT  (Airtable)
# ─────────────────────────────────────────────────────────────

_AIRTABLE_BATCH = 10   # Airtable max records per request


def _batch_create(table: str, records: list[dict]) -> list[str | None]:
    """
    POST up to _AIRTABLE_BATCH records at once.
    Returns a list of record IDs (None for failed entries).
    """
    url  = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{table}"
    ids  = []
    for i in range(0, len(records), _AIRTABLE_BATCH):
        chunk = records[i : i + _AIRTABLE_BATCH]
        r = requests.post(url, headers=AIRTABLE_HEADERS,
                          json={"records": [{"fields": f} for f in chunk]})
        if r.ok:
            ids.extend(rec.get("id") for rec in r.json().get("records", []))
        else:
            print(f"    ✗ Batch error [{r.status_code}]: {r.text[:120]}")
            ids.extend([None] * len(chunk))
    return ids


def plm_agent(bom: list, parent_name: str) -> dict:
    """Write all parts and BOM entries to Airtable using batched requests."""
    separator("PLM AGENT  →  Airtable")
    print(f"  Writing {len(bom)} parts for: '{parent_name}'")

    # ── Batch 1: create all parts ──────────────────────────────
    part_fields = [
        {"part_number": p.get("part_number", f"UNK-{i:03d}"),
         "name":        p.get("name", ""),
         "category":    p.get("category", ""),
         "active":      True}
        for i, p in enumerate(bom)
    ]
    part_ids = _batch_create("Parts", part_fields)
    parts_created = sum(1 for pid in part_ids if pid)
    print(f"  ✓ {parts_created}/{len(bom)} parts created")

    # ── Batch 2: create BOM entries for successful parts ───────
    bom_fields = [
        {"parent":      parent_name,
         "part_number": [pid],
         "quantity":    bom[i].get("quantity", 1),
         "level":       1,
         "notes":       f"PLM agent | qty={bom[i].get('quantity', 1)}"}
        for i, pid in enumerate(part_ids) if pid
    ]
    bom_ids = _batch_create("BOM", bom_fields)
    bom_created = sum(1 for bid in bom_ids if bid)
    errors = (len(bom) - parts_created) + (parts_created - bom_created)

    summary = {"parts_created": parts_created, "bom_created": bom_created,
               "errors": errors, "parent": parent_name}
    print(f"  ✓ {bom_created} BOM entries created | errors: {errors}")
    return summary


# ─────────────────────────────────────────────────────────────
# AGENT 5 – IMAGE AGENT  (DALL-E 3 product render)
# ─────────────────────────────────────────────────────────────

def image_agent(bom: list, family: dict, intent: Intent) -> dict:
    """
    Generate a product render via DALL-E 3.
    1. Claude writes a detailed image prompt from the family + BOM + intent.
    2. DALL-E 3 generates a 1792×1024 render.
    3. Image is downloaded and saved as render_<timestamp>.png.
    """
    import base64 as _b64
    import time as _time

    separator("IMAGE AGENT  →  DALL-E 3")

    if not OPENAI_API_KEY:
        print("  ✗ OPENAI_API_KEY not set — skipping image generation.")
        return {"status": "skipped — no OpenAI key", "file": None}

    product_name = family.get("family", {}).get("name", "product")
    product_type = family.get("family", {}).get("product_type", "product")
    parts        = [p.get("name", "") for p in bom if p.get("name")]
    variants     = [v["name"] for v in family.get("variants", [])]

    # ── Step 1: Claude writes the image prompt ────────────────
    print("  Generating render prompt...")
    prompt_request = f"""
Write a detailed DALL-E 3 image prompt for a professional product render of this item.

Product     : {product_name} ({product_type})
Goal        : {intent.goal}
Constraints : {', '.join(intent.constraints) if intent.constraints else 'none'}
Key parts   : {', '.join(parts[:12])}
Variants    : {', '.join(variants)}

Rules:
- Describe the product as a real, physical object with realistic materials and finish.
- Specify a clean studio background (white, light grey, or gradient).
- Use product photography language: "studio lighting", "soft shadows", "high detail", "8K render".
- Mention the most important physical features based on the parts list.
- Keep it under 200 words.
- Output the prompt text only — no explanation, no quotes around it.
""".strip()

    image_prompt = call_claude(prompt_request,
                               system="You write precise, vivid DALL-E 3 image prompts for product renders.",
                               max_tokens=300)
    print(f"\n  Prompt: {image_prompt[:120]}...")

    # ── Step 2: Call DALL-E 3 ─────────────────────────────────
    print("\n  Calling DALL-E 3...")
    r = requests.post(
        "https://api.openai.com/v1/images/generations",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}",
                 "Content-Type": "application/json"},
        json={
            "model":   "dall-e-3",
            "prompt":  image_prompt,
            "size":    "1792x1024",
            "quality": "hd",
            "n":       1,
        },
        timeout=60,
    )

    if not r.ok:
        print(f"  ✗ DALL-E 3 error [{r.status_code}]: {r.text[:200]}")
        return {"status": f"error {r.status_code}", "file": None}

    image_url = r.json()["data"][0]["url"]

    # ── Step 3: Download and save ─────────────────────────────
    img_data  = requests.get(image_url, timeout=30).content
    timestamp = _time.strftime("%Y%m%d_%H%M%S")
    filename  = os.path.join(os.path.dirname(__file__), f"render_{timestamp}.png")
    with open(filename, "wb") as f:
        f.write(img_data)

    print(f"  ✓ Saved → {filename}")

    # Try to open it in the default image viewer
    try:
        import subprocess
        subprocess.Popen(["explorer", filename])
    except Exception:
        pass

    return {"status": "ok", "file": filename, "prompt": image_prompt}


# ─────────────────────────────────────────────────────────────
# AGENT 6 – CAD AGENT
# Pipeline: BOM → Claude plan → onshape-mcp builders → Onshape
# ─────────────────────────────────────────────────────────────

# ── Import onshape-mcp directly from the local install ────────
_MCP_ROOT = r"C:\Users\f.boehme\onshape-mcp"
_MCP_VENV = os.path.join(_MCP_ROOT, "venv", "Lib", "site-packages")
import sys as _sys
for _p in [_MCP_ROOT, _MCP_VENV]:
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

try:
    import asyncio as _asyncio
    from onshape_mcp.api.client import OnshapeClient, OnshapeCredentials
    from onshape_mcp.api.partstudio import PartStudioManager
    from onshape_mcp.api.variables import VariableManager
    from onshape_mcp.builders.sketch import SketchBuilder, SketchPlane
    from onshape_mcp.builders.extrude import ExtrudeBuilder, ExtrudeType
    from onshape_mcp.builders.revolve import RevolveBuilder, RevolveType
    from onshape_mcp.builders.pattern import CircularPatternBuilder
    _MCP_AVAILABLE = True
except ImportError as _e:
    _MCP_AVAILABLE = False
    print(f"  ⚠ onshape-mcp import failed: {_e}")
    print(f"    CAD will run in simulation mode.")

# Onshape document IDs (copy from your document URL) + API credentials
_ONSHAPE_DID        = os.getenv("ONSHAPE_DID",        "")
_ONSHAPE_WID        = os.getenv("ONSHAPE_WID",        "")
_ONSHAPE_EID        = os.getenv("ONSHAPE_EID",        "")
_ONSHAPE_ACCESS_KEY = os.getenv("ONSHAPE_ACCESS_KEY", "")
_ONSHAPE_SECRET_KEY = os.getenv("ONSHAPE_SECRET_KEY", "")

CAD_MAX_STEPS = 40   # variables + geometry steps combined


def call_claude_thinking(prompt: str, system: str = "",
                         model: str = CLAUDE_MODEL,
                         max_tokens: int = 12000,
                         retries: int = 2) -> tuple[str, str]:
    """Call Claude with adaptive thinking. Returns (thinking_text, reply_text).
    Uses streaming (required by Anthropic for high max_tokens).
    Retries on connection drops."""
    import time as _time

    kwargs = dict(
        model      = model,
        max_tokens = max_tokens,
        thinking   = {"type": "adaptive"},
        messages   = [{"role": "user", "content": prompt}],
    )
    if system:
        kwargs["system"] = system

    last_exc = None
    for attempt in range(1, retries + 2):
        try:
            thinking = ""
            reply    = ""
            with claude.messages.stream(**kwargs) as stream:
                for _ in stream:
                    pass
                response = stream.get_final_message()

            for block in response.content:
                if block.type == "thinking":
                    thinking = block.thinking
                elif block.type == "text":
                    reply = block.text

            if not reply.strip() and thinking:
                m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", thinking)
                if m:
                    reply = m.group(1)

            return thinking, reply

        except Exception as e:
            last_exc = e
            if attempt <= retries:
                wait = 5 * attempt
                print(f"\n  ⚠ Stream error (attempt {attempt}/{retries+1}): {e}")
                print(f"    Retrying in {wait}s...")
                _time.sleep(wait)
            else:
                raise last_exc


# ── Phase 1: Plan (Opus thinking → MCP tool steps) ───────────

def _cad_plan(bom: list, family: dict | None = None) -> dict:
    """
    Opus thinks through the BOM and plans a realistic parametric model
    using the onshape-mcp builder vocabulary. Works for any product type.
    """
    separator("CAD AGENT  Phase 1 — Plan")

    bom_summary = [
        {"pn": p.get("part_number"), "name": p.get("name"),
         "cat": p.get("category"),   "qty":  p.get("quantity", 1)}
        for p in bom
    ]

    product_type = (family or {}).get("family", {}).get("product_type", "product")
    product_name = (family or {}).get("family", {}).get("name", "Product")
    constraints  = (family or {}).get("constraints", [])
    con_block    = ("\n".join(f"  - {c}" for c in constraints)
                    if constraints else "  (none)")

    prompt = f"""
You are an expert parametric CAD engineer. Build a realistic, complete 3D model
of a {product_type} in Onshape using the tools below.
ALL measurements must be in INCHES (1 inch = 25.4 mm).

Product: {product_name}
BOM to model:
{json.dumps(bom_summary)}

Product constraints:
{con_block}

═══ AVAILABLE TOOLS ═══

set_variable
  {{"tool":"set_variable","name":"body_len","expression":"12.0 in","description":"main body length"}}

rect_sketch   — rectangle on a plane
  {{"tool":"rect_sketch","ref":"body_sk","name":"Body","plane":"Top","corner1":[-1,-2],"corner2":[1,2]}}

circle_sketch — circle on a plane
  {{"tool":"circle_sketch","ref":"wheel_sk","name":"Wheel","plane":"Front","centerX":0,"centerY":0,"radius":1.0}}

polygon_sketch — regular polygon
  {{"tool":"polygon_sketch","ref":"hub_sk","name":"Hub","plane":"Top","centerX":0,"centerY":0,"sides":6,"radius":0.5}}

line_polygon   — arbitrary closed polygon (min 3 [x,y] vertices)
  {{"tool":"line_polygon","ref":"bracket_sk","name":"Bracket","plane":"Front","vertices":[[0,0],[1,0],[0.5,1]]}}

extrude
  {{"tool":"extrude","ref":"body","name":"Body","sketch_ref":"body_sk","depth":0.5,"operationType":"NEW"}}
  operationType: "NEW" | "ADD" (merge) | "REMOVE" (cut)

revolve        — revolve profile around axis
  {{"tool":"revolve","ref":"shaft","name":"Shaft","sketch_ref":"shaft_sk","axis":"Y","angle":360,"operationType":"NEW"}}

circular_pattern — repeat feature N times around Z axis
  {{"tool":"circular_pattern","ref":"spokes_pat","name":"Spokes","feature_ref":"spoke_extrude","count":8,"axis":"Z"}}
  feature_ref must be the "ref" of a preceding extrude or revolve.

═══ MODELLING GUIDE ═══

Think carefully about what this product physically looks like.
Map each BOM item to appropriate geometry. Use these principles:

1. VARIABLES FIRST — define key dimensions as set_variable steps before any geometry.
   Name them clearly (e.g. body_len, wheel_r, wall_t). 8–14 variables is typical.

2. STRUCTURE — start with the main structural body/frame/chassis as the foundation
   (operationType NEW). Build all other parts relative to it.

3. EACH BOM COMPONENT → GEOMETRY — for every part in the BOM, create at least one
   sketch + extrude/revolve that represents it visually. Group repeated parts using
   circular_pattern where appropriate (wheels, bolts, symmetrical features).

4. REALISTIC PROPORTIONS — use real-world scale. Think about how the product would
   actually look assembled. Position components in their correct spatial relationship.

5. PLANES:
   - Top    = XY plane (footprint / top-down view)
   - Front  = XZ plane (front face / profile view)
   - Right  = YZ plane (side view)

6. OPERATIONTYPES:
   - NEW    = first body of a new separate part
   - ADD    = merge into an existing body (structural connections)
   - REMOVE = cut/subtract (holes, slots, pockets)

═══ RULES ═══
- Max {CAD_MAX_STEPS} total steps. Prioritise the most visually important components.
- All set_variable steps MUST come before any geometry step.
- sketch_ref must match the "ref" of a preceding sketch step exactly.
- feature_ref in circular_pattern must match the "ref" of a preceding extrude/revolve.
- extrude depth = plain positive number (inches). No units string in depth field.
- All XY coordinates in inches from model origin.
- Output JSON only.

Output:
{{"steps":[...]}}
""".strip()

    print(f"\n  Opus is reasoning through {product_type} geometry...")
    thinking, raw = call_claude_thinking(
        prompt,
        system=f"You are an expert parametric CAD engineer building a complete, realistic {product_type}. Think carefully about every component's geometry, position, and proportions in inches. Output JSON only.",
        model=CLAUDE_MODEL,
        max_tokens=12000,
    )

    if thinking:
        lines = [l.strip() for l in thinking.splitlines() if l.strip()]
        preview = "\n    ".join(lines[:14])
        print(f"\n  ┌─ Thinking (first 14 lines):\n    {preview}")
        if len(lines) > 14:
            print(f"    ... ({len(lines) - 14} more lines)")
        print("  └─")

    if not raw.strip():
        print("\n  Thinking complete. Generating JSON step list from reasoning...")
        thinking_summary = "\n".join(
            [l.strip() for l in thinking.splitlines() if l.strip()][:200]
        ) if thinking else "No thinking available."

        gen_prompt = f"""
You are a parametric CAD engineer. You have reasoned through a complete {product_type}
geometry plan. Now output ONLY the JSON step list based on your reasoning.

Your reasoning summary:
{thinking_summary}

Available tools and their JSON format:
set_variable:    {{"tool":"set_variable","name":"x","expression":"1.0 in"}}
rect_sketch:     {{"tool":"rect_sketch","ref":"id","name":"Label","plane":"Top","corner1":[x,y],"corner2":[x,y]}}
circle_sketch:   {{"tool":"circle_sketch","ref":"id","name":"Label","plane":"Top","centerX":0,"centerY":0,"radius":1.0}}
line_polygon:    {{"tool":"line_polygon","ref":"id","name":"Label","plane":"Top","vertices":[[x,y],...]}}
polygon_sketch:  {{"tool":"polygon_sketch","ref":"id","name":"Label","plane":"Top","centerX":0,"centerY":0,"sides":6,"radius":0.2}}
extrude:         {{"tool":"extrude","ref":"id","name":"Label","sketch_ref":"sketch_ref","depth":0.1,"operationType":"NEW"|"ADD"|"REMOVE"}}
revolve:         {{"tool":"revolve","ref":"id","name":"Label","sketch_ref":"sketch_ref","axis":"Y","angle":360,"operationType":"NEW"}}
circular_pattern:{{"tool":"circular_pattern","ref":"id","name":"Label","feature_ref":"feature_ref","count":4,"axis":"Z"}}

Output ONLY: {{"steps":[...]}}
""".strip()
        raw = call_claude(gen_prompt,
                          system="Output a JSON object with a 'steps' array only.",
                          model=CLAUDE_MODEL,
                          max_tokens=8000)

    plan  = extract_json(raw)
    steps = plan.get("steps", [])
    nvars = sum(1 for s in steps if s.get("tool") == "set_variable")
    ngeo  = len(steps) - nvars
    print(f"\n  ✓ Variables : {nvars}  |  Geometry steps : {ngeo}  |  Total : {len(steps)}")
    return plan


# ── Phase 1b: Self-check + fix loop ──────────────────────────

_VERIFY_RULES = """
Check ALL of the following rules and fix every violation:

1. VARIABLE REFS — variableWidth, variableHeight, variableDepth must each match the
   "name" of a set_variable step that appears EARLIER in the list.
2. SKETCH REFS — extrude "sketch_ref" and revolve "sketch_ref" must match the "ref"
   of a rect_sketch, circle_sketch, polygon_sketch, or line_polygon step EARLIER in the list.
3. FEATURE REFS — circular_pattern "feature_ref" must match the "ref" of an extrude
   or revolve step that appears EARLIER in the list.
4. ORDERING — set_variable before any reference; sketches before their extrude/revolve;
   extrude/revolve before circular_pattern that references them.
5. UNITS — set_variable expressions must end in " in" (e.g. "2.5 in"). No bare mm numbers.
6. DEPTH — extrude "depth" must be a plain positive number (inches). variableDepth if
   used must match a set_variable name defined earlier.
7. CORNERS — rect_sketch corner1 and corner2 must each be a 2-element [x, y] array.
8. VERTICES — line_polygon "vertices" must be a list of at least 3 [x, y] pairs,
   forming a closed polygon (first and last point are automatically connected).
9. TOOL NAMES — only: set_variable, rect_sketch, circle_sketch, polygon_sketch,
   line_polygon, extrude, revolve, circular_pattern.
10. REQUIRED FIELDS per tool:
    set_variable:     name, expression
    rect_sketch:      ref, name, plane, corner1, corner2
    circle_sketch:    ref, name, plane, centerX, centerY, radius
    polygon_sketch:   ref, name, plane, centerX, centerY, sides, radius
    line_polygon:     ref, name, plane, vertices
    extrude:          ref, name, sketch_ref, depth, operationType
    revolve:          ref, name, sketch_ref, axis, angle, operationType
    circular_pattern: ref, name, feature_ref, count, axis
""".strip()

_MAX_VERIFY_ROUNDS = 3


def _cad_verify_and_fix(plan: dict) -> dict:
    """
    Opus re-reads the generated plan and self-checks it against the
    consistency rules. If issues are found it fixes them and re-checks.
    Loops up to _MAX_VERIFY_ROUNDS times.

    This is the core of the reasoning loop — Claude acts as its own reviewer,
    catching variable/ref mismatches, ordering violations, and unit errors
    before a single API call is made.
    """
    separator("CAD AGENT  Phase 1b — Self-Check & Fix Loop")

    current = plan
    all_issues: list[list] = []

    for round_n in range(1, _MAX_VERIFY_ROUNDS + 1):
        print(f"\n  Round {round_n}/{_MAX_VERIFY_ROUNDS} — verifying plan...")

        verify_prompt = f"""
You are a strict CAD plan verifier.

{_VERIFY_RULES}

Current plan:
{json.dumps(current)}

First, think through every rule against every step.
Then output ONLY a JSON object:
{{
  "issues": [
    {{"step_ref": "<ref or name>", "rule": "<which rule>", "problem": "<what is wrong>", "fix": "<exact correction>"}}
  ],
  "fixed_plan": {{
    "steps": [ ... corrected step list ... ]
  }}
}}

If there are no issues, output:
{{"issues": [], "fixed_plan": {json.dumps(current)}}}
""".strip()

        thinking, raw = call_claude_thinking(
            verify_prompt,
            system="You are a CAD plan verifier. Find every rule violation, fix it, output JSON only.",
            model=CLAUDE_MODEL,
            max_tokens=12000,
        )

        if thinking:
            lines = [l.strip() for l in thinking.splitlines() if l.strip()]
            preview = "\n    ".join(lines[:8])
            print(f"\n  ┌─ Reasoning (first 8 lines):\n    {preview}")
            if len(lines) > 8:
                print(f"    ... ({len(lines) - 8} more lines)")
            print("  └─")

        try:
            result  = extract_json(raw)
            issues  = result.get("issues", [])
            fixed   = result.get("fixed_plan", current)
            all_issues.append(issues)
        except Exception as e:
            print(f"  ⚠ Verify round {round_n} parse error: {e} — keeping current plan")
            break

        if issues:
            print(f"\n  Issues found in round {round_n}:")
            for iss in issues:
                print(f"    🔧 [{iss.get('step_ref')}] {iss.get('problem')}  →  {iss.get('fix')}")
            current = fixed
        else:
            print(f"  ✓ Round {round_n}: no issues — plan is consistent")
            break

    total_fixed = sum(len(r) for r in all_issues)
    print(f"\n  Self-check complete — {total_fixed} issue(s) found and fixed across {round_n} round(s)")
    return current


# ── Phase 2: Execute via onshape-mcp builders ─────────────────

import math as _math


def _rotate_vertices(verts: list, angle_deg: float) -> list:
    """Rotate a list of [x,y] vertices around origin by angle_deg."""
    a = _math.radians(angle_deg)
    cos_a, sin_a = _math.cos(a), _math.sin(a)
    return [[v[0]*cos_a - v[1]*sin_a, v[0]*sin_a + v[1]*cos_a] for v in verts]


def _rotate_point(cx: float, cy: float, angle_deg: float) -> tuple:
    """Rotate a single point around origin."""
    a = _math.radians(angle_deg)
    return (cx*_math.cos(a) - cy*_math.sin(a),
            cx*_math.sin(a) + cy*_math.cos(a))


async def _execute_async(plan: dict) -> list[dict]:
    """Execute the MCP tool plan against Onshape using the onshape-mcp builders."""
    did, wid, eid = _ONSHAPE_DID, _ONSHAPE_WID, _ONSHAPE_EID
    creds = OnshapeCredentials(access_key=_ONSHAPE_ACCESS_KEY,
                               secret_key=_ONSHAPE_SECRET_KEY)

    _plane_id_cache: dict = {}

    async def get_plane(ps: PartStudioManager, plane_name: str) -> str:
        if plane_name not in _plane_id_cache:
            _plane_id_cache[plane_name] = await ps.get_plane_id(did, wid, eid, plane_name)
        return _plane_id_cache[plane_name]

    def check_status(r: dict, label: str) -> str:
        """Extract featureId and warn if Onshape reports an error state."""
        fid = r.get("feature", {}).get("featureId", "")
        status = r.get("featureState", {}).get("featureStatus", "OK")
        if status != "OK":
            print(f"    ⚠ Onshape reports {status} for '{label}'")
        return fid

    async with OnshapeClient(creds) as client:
        ps = PartStudioManager(client)

        results       = []
        ref_to_fid    = {}   # ref → Onshape featureId
        ref_to_step   = {}   # ref → original step dict (for pattern fallback)

        def _f(v) -> float:
            """Coerce any coordinate value to float.
            Handles: plain numbers, numeric strings, '#varname' expressions (→ 0.0 fallback),
            and single-element lists (Claude sometimes wraps scalars in a list).
            """
            if isinstance(v, list):
                v = v[0] if v else 0
            if isinstance(v, str):
                # strip variable references like '#desk_w' or expressions — use 0 as fallback
                try:
                    return float(v.replace("#", "").replace(" in", "").strip())
                except ValueError:
                    return 0.0
            try:
                return float(v)
            except (TypeError, ValueError):
                return 0.0

        def _fxy(pair, fallback=(0.0, 0.0)) -> tuple:
            """Coerce a [x, y] pair to (float, float)."""
            if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                return (_f(pair[0]), _f(pair[1]))
            return fallback

        async def _make_sketch(name: str, plane_name: str, step: dict) -> tuple[str, str]:
            """Build and post a sketch; returns (featureId, plane_name)."""
            plane    = SketchPlane[plane_name.upper()]
            plane_id = await get_plane(ps, plane_name)
            sketch   = SketchBuilder(name=name, plane=plane, plane_id=plane_id)
            tool     = step.get("tool")

            if tool == "rect_sketch":
                sketch.add_rectangle(
                    corner1=_fxy(step["corner1"]),
                    corner2=_fxy(step["corner2"]),
                    variable_width=step.get("variableWidth"),
                    variable_height=step.get("variableHeight"),
                )
            elif tool == "circle_sketch":
                sketch.add_circle(
                    center=(_f(step.get("centerX", 0)), _f(step.get("centerY", 0))),
                    radius=_f(step.get("radius", 1.0)),
                )
            elif tool == "polygon_sketch":
                sketch.add_polygon(
                    center=(_f(step.get("centerX", 0)), _f(step.get("centerY", 0))),
                    sides=int(step.get("sides", 6)),
                    radius=_f(step.get("radius", 0.2)),
                )
            elif tool == "line_polygon":
                verts = [_fxy(v) for v in step.get("vertices", [])]
                if len(verts) < 3:
                    raise ValueError("line_polygon needs at least 3 vertices")
                for i in range(len(verts)):
                    sketch.add_line(
                        start=verts[i],
                        end=verts[(i + 1) % len(verts)],
                    )

            r   = await ps.add_feature(did, wid, eid, sketch.build())
            fid = check_status(r, name)
            return fid, plane_name

        async def _make_extrude(name: str, sketch_fid: str, depth: float,
                                op_str: str, var_depth: str | None) -> str:
            op  = ExtrudeType[op_str]
            ext = ExtrudeBuilder(name=name, sketch_feature_id=sketch_fid,
                                 operation_type=op)
            ext.set_depth(depth, variable_name=var_depth)
            r   = await ps.add_feature(did, wid, eid, ext.build())
            return check_status(r, name)

        for step in plan.get("steps", []):
            tool = step.get("tool")
            ref  = step.get("ref", "")
            name = step.get("name", ref or tool)
            ref_to_step[ref] = step

            try:
                # ── set_variable: skip — Part Studio element doesn't support variables ──
                if tool == "set_variable":
                    print(f"    ~ var   {step.get('name')} = {step.get('expression')}  (skipped — use numbers directly)")
                    results.append({"tool": tool, "ok": True, "skipped": True})

                # ── sketch tools ──────────────────────────────────────────────────────
                elif tool in ("rect_sketch", "circle_sketch", "polygon_sketch", "line_polygon"):
                    plane_name = step.get("plane", "Top")
                    fid, _     = await _make_sketch(name, plane_name, step)
                    if ref:
                        ref_to_fid[ref] = fid
                    print(f"    ✓ {tool:<16} '{name}' → {fid}")
                    results.append({"tool": tool, "ref": ref, "feature_id": fid, "ok": True})

                # ── extrude ───────────────────────────────────────────────────────────
                elif tool == "extrude":
                    sketch_fid = ref_to_fid.get(step.get("sketch_ref", ""))
                    if not sketch_fid:
                        raise ValueError(f"sketch_ref '{step.get('sketch_ref')}' not resolved")
                    fid = await _make_extrude(
                        name, sketch_fid, step["depth"],
                        step.get("operationType", "NEW"), step.get("variableDepth")
                    )
                    if ref:
                        ref_to_fid[ref] = fid
                    print(f"    ✓ extrude          '{name}' [{step.get('operationType','NEW')}] → {fid}")
                    results.append({"tool": tool, "ref": ref, "feature_id": fid, "ok": True})

                # ── revolve ───────────────────────────────────────────────────────────
                elif tool == "revolve":
                    sketch_fid = ref_to_fid.get(step.get("sketch_ref", ""))
                    if not sketch_fid:
                        raise ValueError(f"sketch_ref '{step.get('sketch_ref')}' not resolved")
                    op  = RevolveType[step.get("operationType", "NEW")]
                    rev = RevolveBuilder(name=name, sketch_feature_id=sketch_fid,
                                        axis=step.get("axis", "Y"),
                                        angle=step.get("angle", 360),
                                        operation_type=op)
                    r   = await ps.add_feature(did, wid, eid, rev.build())
                    fid = check_status(r, name)
                    if ref:
                        ref_to_fid[ref] = fid
                    print(f"    ✓ revolve          '{name}' → {fid}")
                    results.append({"tool": tool, "ref": ref, "feature_id": fid, "ok": True})

                # ── circular_pattern: implemented in Python via vertex rotation ───────
                # Onshape's REST circular pattern requires a model edge as axis which
                # is unreliable. We rotate the original sketch's geometry ourselves
                # and create explicit copies — guaranteed correct.
                elif tool == "circular_pattern":
                    feature_ref = step.get("feature_ref", "")
                    orig_step   = ref_to_step.get(feature_ref, {})
                    count       = step.get("count", 4)
                    angle_step  = 360.0 / count

                    # Find the sketch step that the feature_ref extrude used
                    if orig_step.get("tool") == "extrude":
                        sk_ref      = orig_step.get("sketch_ref", "")
                        sk_step     = ref_to_step.get(sk_ref, {})
                        depth       = orig_step.get("depth", 0.098)
                        var_depth   = orig_step.get("variableDepth")
                        op_str      = orig_step.get("operationType", "ADD")
                        plane_name  = sk_step.get("plane", "Top")
                        sk_tool     = sk_step.get("tool", "")

                        copy_names = [f"Copy {i+1}" for i in range(count - 1)]

                        for i, cname in enumerate(copy_names, 1):
                            angle   = angle_step * i
                            cp_name = f"{name} {cname}"

                            # Build rotated step
                            rotated_step = dict(sk_step)
                            if sk_tool == "line_polygon":
                                rotated_step["vertices"] = _rotate_vertices(
                                    sk_step.get("vertices", []), angle)
                            elif sk_tool in ("circle_sketch", "polygon_sketch"):
                                cx, cy = _rotate_point(
                                    _f(sk_step.get("centerX", 0)),
                                    _f(sk_step.get("centerY", 0)), angle)
                                rotated_step = dict(sk_step, centerX=cx, centerY=cy)
                            elif sk_tool == "rect_sketch":
                                c1 = _rotate_vertices([_fxy(sk_step["corner1"])], angle)[0]
                                c2 = _rotate_vertices([_fxy(sk_step["corner2"])], angle)[0]
                                rotated_step = dict(sk_step, corner1=c1, corner2=c2)

                            sk_fid, _ = await _make_sketch(
                                f"{cp_name} Sketch", plane_name, rotated_step)
                            ex_fid = await _make_extrude(
                                cp_name, sk_fid, depth, op_str, var_depth)
                            print(f"    ✓ pattern copy     '{cp_name}' → {ex_fid}")
                            results.append({"tool": "extrude", "ref": f"{ref}_{i}",
                                            "feature_id": ex_fid, "ok": True})
                    else:
                        print(f"    ⚠ circular_pattern '{name}': feature_ref must point to an extrude — skipping")
                        results.append({"tool": tool, "ok": False, "error": "feature_ref not an extrude"})
                        continue

                    results.append({"tool": tool, "ref": ref, "ok": True,
                                    "method": "python_rotation", "copies": count - 1})
                    print(f"    ✓ circular_pattern '{name}' ×{count} (Python rotation)")

                else:
                    print(f"    ⚠ unknown tool '{tool}' — skipping")
                    results.append({"tool": tool, "ok": False, "error": "unknown tool"})

            except Exception as e:
                print(f"    ✗ {tool} '{name}': {e}")
                results.append({"tool": tool, "ref": ref, "ok": False, "error": str(e)})

    return results


def _execute_plan(plan: dict) -> list[dict]:
    """Synchronous wrapper around the async execution."""
    separator("CAD AGENT  Phase 2 — Execute via onshape-mcp")

    steps = plan.get("steps", [])
    if not steps:
        print("  ⚠ No steps to execute")
        return []

    # Enforce step cap — always keep all set_variable steps, then fill remaining slots with geometry
    if len(steps) > CAD_MAX_STEPS:
        vars_  = [s for s in steps if s.get("tool") == "set_variable"]
        geo    = [s for s in steps if s.get("tool") != "set_variable"]
        steps  = vars_ + geo[:CAD_MAX_STEPS - len(vars_)]
        plan   = {**plan, "steps": steps}
        print(f"  ⚠ Plan truncated to {len(steps)} steps (cap={CAD_MAX_STEPS})")

    if not _MCP_AVAILABLE:
        print("  ⚠ onshape-mcp not available — simulation mode")
        for s in steps:
            print(f"    [sim] {s.get('tool')}  {s.get('name', s.get('ref', ''))}")
        return [{"ok": True, "simulated": True} for _ in steps]

    print(f"\n  Connected : https://cad.onshape.com/documents/{_ONSHAPE_DID}")
    print(f"  Executing {len(steps)} steps...\n")
    return _asyncio.run(_execute_async(plan))


def cad_agent(bom: list, family: dict | None = None) -> dict:
    """Full CAD pipeline: BOM → Claude plan → onshape-mcp → Onshape."""
    product_type = (family or {}).get("family", {}).get("product_type", "product")
    separator(f"CAD AGENT  →  Onshape  [{product_type}]")
    print(f"  BOM input: {len(bom)} parts")

    plan    = _cad_plan(bom, family)    # Phase 1  — Opus thinks, generates step list
    plan    = _cad_verify_and_fix(plan) # Phase 1b — Opus self-checks & fixes in a loop
    results = _execute_plan(plan)       # Phase 2  — execute step-by-step via MCP builders

    ok = sum(1 for r in results if r.get("ok"))
    separator("CAD AGENT  DONE")
    print(f"  {ok}/{len(results)} operations succeeded")
    print(f"  View: https://cad.onshape.com/documents/{_ONSHAPE_DID}")

    return {
        "cad_steps": [s.get("name", s.get("ref")) for s in plan.get("steps", [])],
        "results":   results,
        "status":    f"{ok}/{len(results)} features created in Onshape",
    }


# ─────────────────────────────────────────────────────────────
# ORCHESTRATOR
# ─────────────────────────────────────────────────────────────

def orchestrator(intent: Intent, family: dict) -> dict:
    """
    Engineering loop (family already defined before calling):
      1. Configure (once, guided by family)
      2. Evaluate → check stop condition → Optimize  (repeat)
      3. Persist final BOM to Airtable

    Stop logic:
      - Always run at least 2 iterations
      - Continue if any critical issue exists (incl. constraint violations)
      - Stop when primary metric >= 8
      - Hard cap at MAX_ITER
    """
    separator("ORCHESTRATOR  –  START")
    family_name = family.get("family", {}).get("name", "")
    print(f"\n  Family      : {family_name}")
    print(f"  Goal        : {intent.goal}")
    print(f"  Constraints : {intent.constraints}")
    if intent.context:
        print(f"  Context     : {intent.context}")
    print(f"  Max iters   : {MAX_ITER}")

    # Step 1 — initial configuration (guided by family)
    config    = configurator_agent(intent, family=family)
    best_bom  = config.get("bom", [])
    last_eval = {}

    # Step 2 — evaluate / optimize loop
    for iteration in range(1, MAX_ITER + 1):
        separator(f"ITERATION {iteration} / {MAX_ITER}")

        last_eval = evaluator_agent(config)

        if should_stop(last_eval, iteration, intent, family):
            metric = _primary_metric(intent, family)
            scores = last_eval.get("scores", {})
            print(f"\n  ✅ Done — no critical issues, "
                  f"{metric}={scores.get(metric)}/10")
            break

        if iteration == MAX_ITER:
            print(f"\n  ⏹  Hard cap reached ({MAX_ITER} iterations).")
            break

        # Explain why we continue
        if iteration < 2:
            print(f"\n  ↺  Minimum 2 iterations — continuing.")
        elif has_critical_issues(last_eval):
            n = sum(1 for i in last_eval.get("issues", []) if i.get("type") == "critical")
            print(f"\n  ↺  {n} critical issue(s) remain — optimizing.")
        else:
            s      = last_eval.get("scores", {})
            metric = _primary_metric(intent, family)
            print(f"\n  ↺  {metric}={s.get(metric)}/10 — not yet ≥ 8, optimizing.")

        optimized = optimizer_agent(config, last_eval)
        config = {
            **config,
            "configuration": optimized.get("configuration", config.get("configuration")),
            "bom":           optimized.get("bom", best_bom),
        }
        best_bom = config.get("bom", best_bom)

    # Save session (BOM + family) so the user can skip straight to CAD next run
    _save_last_bom(best_bom, family)

    # Step 3 — persist to Airtable
    product_type  = family.get("family", {}).get("product_type", "Product")
    assembly_name = f"AI-{product_type}: {intent.goal[:50]}"
    plm_result    = plm_agent(best_bom, parent_name=assembly_name)

    # Step 4 — visualise / build
    print(f"\n  BOM ready. {len(best_bom)} parts configured.")
    print("""
  What would you like to do next?
    1  Generate 3D model in Onshape
    2  Generate product image  (DALL-E 3)
    3  Skip
""")
    vis_choice = input("  Choose [1/2/3]: ").strip()

    image_result = {"status": "skipped", "file": None}
    cad_result   = {"status": "skipped", "cad_steps": [], "results": []}

    if vis_choice == "1":
        cad_result = cad_agent(best_bom, family=family)
    elif vis_choice == "2":
        image_result = image_agent(best_bom, family, intent)
    else:
        print("  → Visualisation skipped.")

    # Step 5 — final report
    separator("FINAL REPORT")
    s = last_eval.get("scores", {})
    remaining   = [i["text"] for i in last_eval.get("issues", []) if i.get("type") == "critical"]
    family_name = family.get("family", {}).get("name", "n/a")
    variants    = [v["name"] for v in family.get("variants", [])]
    scores_str  = " | ".join(f"{k}={v}/10" for k, v in s.items()) if s else "n/a"
    image_line  = image_result["file"] if image_result.get("file") else image_result.get("status", "skipped")
    print(f"""
  Goal        : {intent.goal}
  Constraints : {intent.constraints}
  Family      : {family_name}  ({', '.join(variants)})
  Final BOM   : {len(best_bom)} parts
  Scores      : {scores_str}
  Critical    : {remaining if remaining else 'none'}
  Airtable    : {plm_result['parts_created']} parts + {plm_result['bom_created']} BOM entries written
  CAD         : {cad_result['status']}
  Image       : {image_line}
""")

    outcome = {
        "intent":       intent,
        "family":       family,
        "final_config": config,
        "evaluation":   last_eval,
        "plm_result":   plm_result,
        "cad_result":   cad_result,
        "image_result": image_result,
    }
    _save_html_report(outcome)
    return outcome


# ─────────────────────────────────────────────────────────────
# SESSION PERSISTENCE  (BOM + family saved together)
# ─────────────────────────────────────────────────────────────

_LAST_SESSION_FILE = os.path.join(os.path.dirname(__file__), ".last_session.json")
_LAST_BOM_FILE     = os.path.join(os.path.dirname(__file__), ".last_bom.json")  # legacy


def _save_last_bom(bom: list, family: dict | None = None) -> None:
    try:
        with open(_LAST_SESSION_FILE, "w") as f:
            json.dump({"bom": bom, "family": family or {}}, f, indent=2)
    except Exception as e:
        print(f"  ⚠ Could not save session: {e}")


def _load_last_session() -> tuple[list | None, dict | None]:
    """Returns (bom, family). Falls back to legacy .last_bom.json if needed."""
    for path in [_LAST_SESSION_FILE, _LAST_BOM_FILE]:
        if not os.path.exists(path):
            continue
        try:
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, list):          # legacy format
                return data, None
            return data.get("bom"), data.get("family")
        except Exception:
            continue
    return None, None


# ─────────────────────────────────────────────────────────────
# HTML REPORT
# ─────────────────────────────────────────────────────────────

def _save_html_report(outcome: dict) -> None:
    """Generate and save a self-contained HTML report. Opens in browser."""
    import time as _time, base64 as _b64, webbrowser as _wb

    intent      = outcome["intent"]
    family      = outcome["family"]
    config      = outcome["final_config"]
    evaluation  = outcome["evaluation"]
    plm_result  = outcome["plm_result"]
    cad_result  = outcome["cad_result"]
    image_result= outcome["image_result"]

    f_info      = family.get("family", {})
    scores      = evaluation.get("scores", {})
    issues      = evaluation.get("issues", [])
    bom         = config.get("bom", [])
    variants    = family.get("variants", [])
    dims        = family.get("scoring_dimensions", [])
    configuration = config.get("configuration", {})

    # Score bars
    score_bars = ""
    for dim in dims:
        name = dim["name"]
        val  = scores.get(name, 0)
        pct  = val * 10
        color = "#22c55e" if val >= 8 else "#f59e0b" if val >= 5 else "#ef4444"
        score_bars += f"""
        <div class="score-row">
          <span class="score-label">{name}</span>
          <div class="score-bar-bg">
            <div class="score-bar" style="width:{pct}%;background:{color}"></div>
          </div>
          <span class="score-val">{val}/10</span>
        </div>"""

    # BOM table rows
    bom_rows = "".join(
        f"<tr><td>{p.get('part_number','')}</td><td>{p.get('name','')}</td>"
        f"<td>{p.get('category','')}</td><td>{p.get('quantity',1)}</td></tr>"
        for p in bom
    )

    # Configuration rows
    cfg_rows = "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>"
        for k, v in configuration.items()
    )

    # Issues
    issue_html = ""
    for iss in issues:
        cls  = "critical" if iss.get("type") == "critical" else "normal"
        icon = "⚠" if cls == "critical" else "•"
        issue_html += f'<div class="issue {cls}">{icon} {iss.get("text","")}</div>'
    if not issue_html:
        issue_html = '<div class="issue ok">No critical issues</div>'

    # Variants
    variant_html = ""
    for v in variants:
        cfg_preview = ", ".join(f"{k}={val}" for k, val in list(v.get("configuration", {}).items())[:4])
        variant_html += f"<div class='variant'><strong>{v['name']}</strong> — {v.get('description','')} <br><small>{cfg_preview}</small></div>"

    # Image embed
    img_html = ""
    img_file = image_result.get("file")
    if img_file and os.path.exists(img_file):
        with open(img_file, "rb") as fh:
            b64 = _b64.b64encode(fh.read()).decode()
        img_html = f'<div class="section"><h2>Product Render</h2><img src="data:image/png;base64,{b64}" style="max-width:100%;border-radius:8px;"></div>'

    timestamp = _time.strftime("%Y-%m-%d %H:%M")
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{f_info.get('name','Product')} — Design Report</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 0; background: #f8fafc; color: #1e293b; }}
  .header {{ background: linear-gradient(135deg,#1e293b,#334155); color: white; padding: 2rem; }}
  .header h1 {{ margin: 0 0 .4rem; font-size: 1.8rem; }}
  .header p  {{ margin: 0; opacity: .7; }}
  .body {{ max-width: 960px; margin: 2rem auto; padding: 0 1rem; }}
  .section {{ background: white; border-radius: 10px; padding: 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 1px 4px rgba(0,0,0,.07); }}
  h2 {{ margin: 0 0 1rem; font-size: 1.1rem; color: #475569; text-transform: uppercase; letter-spacing: .05em; }}
  table {{ width: 100%; border-collapse: collapse; font-size: .9rem; }}
  th {{ background: #f1f5f9; text-align: left; padding: .5rem .75rem; }}
  td {{ padding: .45rem .75rem; border-bottom: 1px solid #f1f5f9; }}
  .score-row {{ display:flex; align-items:center; margin-bottom:.6rem; gap:.75rem; }}
  .score-label {{ width:200px; font-size:.85rem; color:#64748b; }}
  .score-bar-bg {{ flex:1; background:#e2e8f0; border-radius:4px; height:12px; }}
  .score-bar {{ height:12px; border-radius:4px; transition:width .3s; }}
  .score-val {{ width:40px; font-weight:600; font-size:.85rem; }}
  .issue {{ padding:.4rem .75rem; margin-bottom:.4rem; border-radius:6px; font-size:.9rem; }}
  .issue.critical {{ background:#fef2f2; color:#991b1b; }}
  .issue.normal   {{ background:#fffbeb; color:#92400e; }}
  .issue.ok       {{ background:#f0fdf4; color:#166534; }}
  .variant {{ background:#f8fafc; border-left:3px solid #6366f1; padding:.6rem 1rem; margin-bottom:.6rem; border-radius:0 6px 6px 0; font-size:.9rem; }}
  .meta {{ display:flex; gap:1.5rem; flex-wrap:wrap; font-size:.85rem; color:#64748b; margin-top:.5rem; }}
  .meta span b {{ color:#1e293b; }}
  .badge {{ display:inline-block; background:#e0e7ff; color:#3730a3; padding:.2rem .6rem; border-radius:999px; font-size:.75rem; margin-right:.3rem; }}
</style>
</head>
<body>
<div class="header">
  <h1>{f_info.get('name','Product')} — Design Report</h1>
  <p>{f_info.get('description','')} &nbsp;·&nbsp; Generated {timestamp}</p>
</div>
<div class="body">

  <div class="section">
    <h2>Design Goal</h2>
    <p style="font-size:1.1rem;margin:0 0 .5rem"><strong>{intent.goal}</strong></p>
    <div class="meta">
      <span>Constraints: <b>{', '.join(intent.constraints) if intent.constraints else 'none'}</b></span>
      {'<span>Context: <b>' + intent.context + '</b></span>' if intent.context else ''}
    </div>
  </div>

  <div class="section">
    <h2>Scores</h2>
    {score_bars}
    <p style="margin:.75rem 0 0;font-size:.85rem;color:#64748b">{evaluation.get('summary','')}</p>
  </div>

  <div class="section">
    <h2>Issues</h2>
    {issue_html}
  </div>

  <div class="section">
    <h2>Selected Configuration</h2>
    <table><tr><th>Feature</th><th>Selected Option</th></tr>{cfg_rows}</table>
  </div>

  <div class="section">
    <h2>Bill of Materials ({len(bom)} parts)</h2>
    <table>
      <tr><th>Part No.</th><th>Name</th><th>Category</th><th>Qty</th></tr>
      {bom_rows}
    </table>
    <div class="meta" style="margin-top:.75rem">
      <span>Airtable: <b>{plm_result.get('parts_created',0)} parts + {plm_result.get('bom_created',0)} BOM entries</b></span>
      <span>CAD: <b>{cad_result.get('status','skipped')}</b></span>
    </div>
  </div>

  {img_html}

  <div class="section">
    <h2>Product Variants</h2>
    {variant_html}
  </div>

</div>
</body>
</html>"""

    ts       = _time.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(os.path.dirname(__file__), f"report_{ts}.html")
    with open(filename, "w", encoding="utf-8") as fh:
        fh.write(html)
    separator("REPORT")
    print(f"  Saved → {filename}")
    try:
        _wb.open(f"file:///{filename.replace(os.sep, '/')}")
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

def ask_product_idea() -> str:
    """Ask the user what product they want to design."""
    print("\n" + "═"*60)
    print("  AI PRODUCT PLM AGENT")
    print("═"*60)
    print("""
What product do you want to design?

Examples:
  - electric mountain bike
  - professional inspection robot
  - espresso machine for home use
  - industrial inspection robot
  - portable solar power station
""")
    idea = input("  Product idea: ").strip()
    while not idea:
        idea = input("  (required) Product idea: ").strip()
    return idea


def ask_intent(family: dict) -> Intent:
    """Ask for design intent, informed by the product family definition."""
    f        = family.get("family", {})
    dims     = family.get("scoring_dimensions", [])
    variants = family.get("variants", [])
    features = family.get("features", [])

    print("\n" + "═"*60)
    print(f"  DEFINE YOUR INTENT  —  {f.get('name', '')}")
    print("═"*60)

    if dims:
        print("\nThis product is evaluated on:")
        for d in dims:
            print(f"  • {d['name']}: {d['description']}")

    # ── Variant picker ────────────────────────────────────────
    goal = ""
    constraints_preset: list[str] = []
    if variants:
        print(f"\nHow do you want to proceed?\n")
        print(f"  0  Auto — let the system recommend the best starting point")
        for i, v in enumerate(variants, 1):
            cfg_str = ", ".join(f"{k}={val}" for k, val in list(v.get("configuration", {}).items())[:3])
            print(f"  {i}  {v['name']} — {v.get('description', '')}  [{cfg_str}]")
        print(f"  {len(variants)+1}  Custom (type your own goal)")
        print()
        pick = input(f"  Choose [0-{len(variants)+1}]: ").strip()

        if pick == "0":
            # Claude reads the family and recommends the best intent
            print("\n  Analysing product family to recommend best intent...")
            auto_prompt = f"""
You are a product strategist. Given this product family, recommend the single most
compelling and balanced design intent — the one that would create the best product
for the broadest real-world use case.

Product family:
{json.dumps({"family": family.get("family"), "features": family.get("features"),
             "options": family.get("options"), "constraints": family.get("constraints"),
             "variants": family.get("variants"), "scoring_dimensions": family.get("scoring_dimensions")}, indent=2)}

Return a JSON object:
{{
  "goal": "concise goal statement, e.g. 'maximum range with balanced cost'",
  "constraints": ["hard constraint 1", "hard constraint 2"],
  "reasoning": "one sentence explaining why this is the best starting point"
}}

Output JSON only.
""".strip()
            raw  = call_claude(auto_prompt,
                               system="You are a product strategist. Output JSON only.",
                               max_tokens=512)
            rec  = extract_json(raw)
            goal              = rec.get("goal", "")
            constraints_preset = rec.get("constraints", [])
            print(f"\n  Recommended goal       : {goal}")
            print(f"  Recommended constraints: {constraints_preset}")
            print(f"  Reasoning              : {rec.get('reasoning', '')}")

        elif pick.isdigit() and 1 <= int(pick) <= len(variants):
            chosen = variants[int(pick) - 1]
            goal   = f"{chosen['name']}: {chosen.get('description', '')}"
            print(f"\n  Starting from: {chosen['name']}")

    if not goal:
        print()
        goal = input("  Your goal (what to optimise for): ").strip()
        while not goal:
            goal = input("  (required) Your goal: ").strip()

    # Show key features as constraint hints
    if features:
        feat_names = [ft["name"] for ft in features[:5]]
        print(f"\nKey features: {', '.join(feat_names)}")
    print("""
Hard constraints — things that must never be violated.
Enter one per line. Press Enter on an empty line when done.
""")
    constraints = list(constraints_preset)
    if constraints:
        print(f"  (pre-filled from recommendation:)")
        for c in constraints:
            print(f"    • {c}")
        print()
    while True:
        c = input(f"  Constraint {len(constraints)+1} (or Enter to finish): ").strip()
        if not c:
            break
        constraints.append(c)

    print("\nAny extra context? (use case, environment, user profile, etc.) — Enter to skip.")
    context = input("  Context: ").strip()

    intent = Intent(goal=goal, constraints=constraints, context=context)

    print(f"\n  ✓ Goal        : {intent.goal}")
    print(f"  ✓ Constraints : {intent.constraints or 'none'}")
    print(f"  ✓ Context     : {intent.context or 'none'}")
    input("\n  Start design? (Enter to confirm, Ctrl+C to cancel): ")

    return intent


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AI Product Agents")
    parser.add_argument("--setup", action="store_true",
                        help="Verify config and create Airtable tables, then exit")
    parser.add_argument("--idea",  type=str, default="",
                        help="Product idea (skips the interactive prompt)")
    args = parser.parse_args()

    _check_config()
    setup_airtable()

    if args.setup:
        print("\n  Setup complete. You are ready to run: python plm_agents.py")
        raise SystemExit(0)

    last_bom, last_family = _load_last_session()

    if last_bom:
        print("\n" + "═"*60)
        print("  AI PRODUCT AGENTS")
        print("═"*60)
        family_name = (last_family or {}).get("family", {}).get("name", "")
        print(f"\n  Last design found — {len(last_bom)} parts"
              + (f"  ({family_name})" if family_name else ""))
        for p in last_bom[:5]:
            print(f"    • {p.get('part_number')}  {p.get('name')}")
        if len(last_bom) > 5:
            print(f"    ... and {len(last_bom)-5} more")
        print("""
  Options:
    1  Full pipeline   (new product idea → family → configure → evaluate → optimise → report)
    2  CAD only        (Onshape 3D model from last design)
    3  Image only      (DALL-E 3 render from last design)
""")
        choice = input("  Choose [1/2/3]: ").strip()
    else:
        choice = "1"

    if choice == "2" and last_bom:
        separator("CAD ONLY — using last design")
        cad_agent(last_bom, family=last_family)
    elif choice == "3" and last_bom:
        separator("IMAGE ONLY — using last design")
        if not last_family:
            print("  ⚠ No family context saved — using BOM names only for image prompt.")
            last_family = {"family": {"name": "Product", "product_type": "product"},
                           "scoring_dimensions": [], "variants": []}
        fake_intent = Intent(goal="product render", constraints=[])
        image_agent(last_bom, last_family, fake_intent)
    else:
        product_idea = args.idea or ask_product_idea()
        family       = product_family_agent(product_idea)
        intent       = ask_intent(family)
        orchestrator(intent, family)
