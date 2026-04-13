"""
Design to Intent — Multi-Agent Product Design System
=====================================================
Architecture:
  Product Idea
  → Product Family Agent   (defines product line: features, options, constraints, variants)
  → Intent Definition      (goal + constraints + context — manually or auto from market gap)
  → Configurator Agent     (selects valid configuration from family + builds BOM)
  → Evaluator Agent        (scores dimensions defined by the product family)
  → Optimizer Agent        (fixes critical issues first, then improves scores toward intent)
  → Builder Agent          (persists parts + BOM to Airtable)
  → CAD Agent (optional)   (maps BOM → parametric geometry → Onshape via MCP)
  → Image Agent (optional) (DALL-E 3 product render)
  → HTML Report            (radar chart, journey chart, BOM)

Setup:
  1. Copy .env.example to .env and fill in your API keys.
  2. pip install -r requirements.txt
  3. python plm_agents.py        (CLI)
     python gui.py               (desktop GUI)
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
# DOMAIN KNOWLEDGE  (RAG + KG)  — optional, graceful fallback
# ─────────────────────────────────────────────────────────────
try:
    from domain_knowledge import (
        DomainContext, get_domain_agent
    )
    _DOMAIN_KNOWLEDGE_AVAILABLE = True
except ImportError as _dk_err:
    print(f"  ⚠ domain_knowledge module unavailable: {_dk_err}")
    _DOMAIN_KNOWLEDGE_AVAILABLE = False

    class DomainContext:  # type: ignore[no-redef]
        """Stub when domain_knowledge.py is absent."""
        rag_chunks: list = []
        graph_triples: list = []
        validation_trace: dict = {}
        company_sources: list = []
        def prompt_block(self) -> str: return ""
        def is_empty(self) -> bool: return True
        def sources_for_report(self) -> list: return []
        def vv_coverage(self) -> dict:
            return {"company_sourced": 0, "graph": 0, "rag": 0, "llm_reasoned": 0}

# ─────────────────────────────────────────────────────────────
# INTENT ELICITATION  — optional, graceful fallback
# ─────────────────────────────────────────────────────────────
try:
    from intent_elicitation import (
        IntentContext, intent_elicitation_agent
    )
    _INTENT_ELICITATION_AVAILABLE = True
except ImportError as _ie_err:
    _INTENT_ELICITATION_AVAILABLE = False

    class IntentContext:  # type: ignore[no-redef]
        """Stub when intent_elicitation.py is absent."""
        original_input: str = ""
        depth: str = "vague"
        detected_domain: str = "default"
        clarifications: list = []
        enriched_goal: str = ""
        enriched_constraints: str = ""
        enriched_context: str = ""
        def summary(self) -> str: return ""

    def intent_elicitation_agent(product_idea, existing_goal="",  # type: ignore[misc]
                                  existing_constraints="", existing_context="",
                                  _interactive=True, **_kw):
        ctx = IntentContext()
        ctx.original_input = product_idea
        ctx.enriched_goal = existing_goal
        ctx.enriched_constraints = existing_constraints
        ctx.enriched_context = existing_context
        return ctx

# ─────────────────────────────────────────────────────────────
# CONFIGURATION  —  set via .env or environment variables
# ─────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
AIRTABLE_TOKEN    = os.getenv("AIRTABLE_TOKEN",    "")
AIRTABLE_BASE_ID  = os.getenv("AIRTABLE_BASE_ID",  "")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY",    "")

CLAUDE_MODEL       = "claude-opus-4-6"           # CAD reasoning (adaptive thinking)
CLAUDE_MODEL_MID   = "claude-sonnet-4-6"         # configurator / evaluator / optimizer
CLAUDE_MODEL_FAST  = "claude-haiku-4-5-20251001" # family agent (structured data, no deep reasoning)
MAX_ITER           = 3   # hard cap on iterations (most products converge in 2-3)


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

    Retries automatically on 529 overload errors with exponential backoff.
    """
    import time as _time
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

    for attempt in range(5):
        try:
            response = claude.messages.create(**kwargs)
            for block in response.content:
                if block.type == "text":
                    return block.text
            print(f"  ⚠ No text block. Block types: {[b.type for b in response.content]}")
            return ""
        except Exception as e:
            if "overloaded" in str(e).lower() or "529" in str(e):
                wait = 15 * (2 ** attempt)   # 15s, 30s, 60s, 120s, 240s
                print(f"  ⚠ API overloaded — waiting {wait}s before retry ({attempt+1}/5)...")
                _time.sleep(wait)
            else:
                raise
    raise RuntimeError("Claude API still overloaded after 5 retries — try again later.")


def _clean_json_text(text: str) -> str:
    """
    Light pre-processing to fix common Claude JSON mistakes:
    - Strip // line comments
    - Strip /* block comments */
    - Remove trailing commas before } or ]
    """
    # Remove // line comments (not inside strings — good-enough heuristic)
    text = re.sub(r'//[^\n"]*\n', '\n', text)
    # Remove /* ... */ block comments
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    # Remove trailing commas before closing brace/bracket
    text = re.sub(r',\s*([}\]])', r'\1', text)
    return text


def extract_json(text: str) -> dict | list:
    """Extract the first JSON object or array from a Claude reply."""
    if not text or not text.strip():
        raise ValueError("Claude returned an empty response")

    # Prefer fenced ```json ... ``` block
    match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\}|\[[\s\S]*?\])\s*```", text)
    if match:
        candidate = _clean_json_text(match.group(1))
        return json.loads(candidate)

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

    candidate = _clean_json_text(text[start:end])
    return json.loads(candidate)


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
                intent: Intent | None = None, family: dict | None = None,
                score_history: list | None = None) -> bool:
    """
    Stop when any of:
    - Primary metric >= 8 and no critical issues
    - Scores didn't improve at all vs previous iteration (diminishing returns)
    Always run at least 2 iterations.
    """
    if iteration < 2:
        return False
    if has_critical_issues(evaluation):
        return False
    scores = evaluation.get("scores", {})
    metric = _primary_metric(intent, family)
    if scores.get(metric, 0) >= 8:
        return True
    # Diminishing returns: stop if no score improved since last iteration
    if score_history and len(score_history) >= 2:
        prev = score_history[-2]["scores"]
        curr = score_history[-1]["scores"]
        if all(curr.get(k, 0) <= prev.get(k, 0) for k in curr):
            print(f"\n  ↩  No score improvement vs previous iteration — stopping early.")
            return True
    return False


# ─────────────────────────────────────────────────────────────
# DOMAIN KNOWLEDGE AGENT  (runs before Product Family)
# ─────────────────────────────────────────────────────────────

def domain_knowledge_agent(product_idea: str,
                            product_type: str = "",
                            intent_goal: str = "",
                            detected_domain: str = "") -> "DomainContext":
    """
    Retrieve RAG chunks + KG triples relevant to this product.
    detected_domain: if provided by IntentElicitationAgent, skip re-detection.
    Returns a DomainContext (empty stub if domain_knowledge unavailable).
    Always succeeds — errors are logged, never raised.
    """
    separator("DOMAIN KNOWLEDGE AGENT")
    if not _DOMAIN_KNOWLEDGE_AVAILABLE:
        print("  ⚠ Skipped — install llama-index, chromadb, kuzu for full context")
        return DomainContext()
    try:
        agent = get_domain_agent()
        ctx   = agent.run(product_idea, product_type, intent_goal,
                          detected_domain=detected_domain)
        trace = ctx.validation_trace
        print(f"  ✓ RAG     : {trace.get('rag_chunks', 0)} chunks "
              f"({trace.get('stale_chunks', 0)} stale)")
        print(f"  ✓ Graph   : {trace.get('graph_triples', 0)} triples "
              f"({trace.get('sourced', 0)} sourced, "
              f"{trace.get('llm_reasoned', 0)} llm_reasoned)")
        if trace.get("sources"):
            for src in trace["sources"]:
                print(f"    • {src}")
        return ctx
    except Exception as e:
        print(f"  ⚠ Domain knowledge error: {e} — continuing without context")
        return DomainContext()


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


def _load_company_constraints(product_idea: str) -> tuple[list[str], list[str], str]:
    """
    Lightweight loader — reads company_knowledge_{slug}.json without starting RAG/KG.
    Returns (constraints, decisions, files_summary).
    Called before product_family_agent so company constraints can inform the family.
    """
    slug = re.sub(r"[^a-z0-9]+", "_", product_idea.lower()).strip("_")[:50]
    path = os.path.join(os.path.dirname(__file__), f"company_knowledge_{slug}.json")
    if not os.path.exists(path):
        return [], [], ""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        constraints: list[str] = []
        decisions:   list[str] = []
        files: set[str] = set()
        for r in data.get("records", []):
            sf = r.get("source_file", "")
            if sf:
                files.add(sf)
            for c in r.get("constraints", []):
                if isinstance(c, str) and c.strip():
                    constraints.append(c.strip())
            for d in r.get("decisions", []):
                if isinstance(d, str) and d.strip():
                    decisions.append(d.strip())
        summary = f"{len(data.get('records', []))} records from: {', '.join(sorted(files))}"
        return constraints, decisions, summary
    except Exception as _e:
        print(f"  ⚠ Could not read company knowledge for family agent: {_e}")
        return [], [], ""


def product_family_agent(product_idea: str,
                         company_constraints: list[str] | None = None) -> dict:
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

    # Build company constraints block if provided
    company_block = ""
    if company_constraints:
        lines = ["COMPANY-VERIFIED CONSTRAINTS (must all be included — non-negotiable):"]
        for c in company_constraints:
            lines.append(f"  - {c}")
        company_block = "\n".join(lines) + "\n\n"
        print(f"  Company constraints loaded: {len(company_constraints)}")

    prompt = f"""
Define a practical product family for this idea.

Product idea: {product_idea}

{company_block}

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
  Choose dimensions that are DIRECTLY MEASURABLE on the actual product being designed.
  Examples: bicycle → range_km, comfort_score, cost_usd.
            camera  → image_quality_score, portability_score, cost_usd.
            PCB dev board → compute_performance_mhz, developer_accessibility_score, cost_usd.
            energy storage → usable_capacity_kwh, round_trip_efficiency_pct, cost_per_kwh.
  NEVER choose a dimension that the product structurally cannot achieve
  (e.g. network_range_km for a USB-only device, or speed_kmh for a stationary product).
- All options must be real-world, specific values (not vague like "good battery").
- Variants must each satisfy all constraints.
- Output JSON only, no markdown outside the block.
""".strip()

    raw        = call_claude(prompt, system=_FAMILY_SYSTEM, max_tokens=2500, model=CLAUDE_MODEL_FAST)
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

def configurator_agent(intent: Intent, family: dict | None = None,
                       domain_ctx: "DomainContext | None" = None) -> dict:
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

    # Domain knowledge context (RAG + graph)
    domain_block = ""
    if domain_ctx and not domain_ctx.is_empty():
        domain_block = f"\n{domain_ctx.prompt_block()}\n"

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

{family_block + chr(10) if family_block else ""}{domain_block}User intent:
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
    result["_intent"]         = intent       # carry intent forward for other agents
    result["_family"]         = family       # carry family forward for evaluator/optimizer
    result["_domain_context"] = domain_ctx  # carry domain context forward

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
    intent:     Intent        = config.get("_intent")
    family:     dict          = config.get("_family") or {}
    domain_ctx: DomainContext = config.get("_domain_context")  # type: ignore[assignment]

    config_data = {
        "configuration": config.get("configuration", {}),
        "bom":           config.get("bom", []),
        "constraints":   config.get("constraints", []),
    }

    dims         = family.get("scoring_dimensions", [])
    product_type = family.get("family", {}).get("product_type", "product")
    intent_block = intent.as_prompt_block() if intent else ""

    # Company-sourced constraints — injected before all other context
    company_block = ""
    if domain_ctx and getattr(domain_ctx, "company_sources", []):
        lines = ["COMPANY-VERIFIED CONSTRAINTS (treat as hard constraints — highest confidence):"]
        for r in domain_ctx.company_sources:
            sf = r.get("source_file", "unknown")
            sd = r.get("source_date", "")
            for c in r.get("constraints", []):
                if isinstance(c, str) and c.strip():
                    src = f"[{sf}" + (f" {sd}" if sd else "") + "]"
                    lines.append(f"  - {c.strip()}  {src}")
            for d in r.get("decisions", []):
                if isinstance(d, str) and d.strip():
                    src = f"[{sf}" + (f" {sd}" if sd else "") + "]"
                    lines.append(f"  • VERIFIED DECISION: {d.strip()}  {src}")
        company_block = "\n".join(lines) + "\n"

    # Query knowledge graph for relevant constraints before scoring
    graph_block = ""
    if domain_ctx and not domain_ctx.is_empty():
        bom_names  = [p.get("name", "") for p in config.get("bom", [])]
        cfg_values = [str(v) for v in config.get("configuration", {}).values()]
        kw_source  = " ".join(bom_names[:8] + cfg_values[:4])
        from domain_knowledge import _extract_keywords
        kw = _extract_keywords(kw_source)
        extra_triples = domain_ctx._kg.query(kw) if hasattr(domain_ctx, "_kg") else []
        # Merge with existing triples (no duplicates)
        existing_keys = {(t.subject, t.relation, t.object)
                         for t in domain_ctx.graph_triples}
        new_triples   = [t for t in extra_triples
                         if (t.subject, t.relation, t.object) not in existing_keys]
        all_triples   = domain_ctx.graph_triples + new_triples
        if all_triples:
            graph_block = (
                "\nKnowledge graph constraints to consider during scoring:\n"
                + "\n".join(t.as_context_line() for t in all_triples[:15])
                + "\n"
            )

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
{company_block}{graph_block}
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

    # Validation trace — record which context types informed this evaluation
    if domain_ctx:
        cov = domain_ctx.vv_coverage()
        result["_validation_trace"] = {
            "company_sourced":    cov.get("company_sourced", 0),
            "graph":              cov["graph"],
            "rag":                cov["rag"],
            "llm_reasoned":       cov["llm_reasoned"],
            "graph_block_used":   bool(graph_block),
            "company_block_used": bool(company_block),
            "company_files":      domain_ctx.validation_trace.get("company_files_used", []),
        }
        if company_block:
            n_files = len(domain_ctx.validation_trace.get("company_files_used", []))
            print(f"  ✓ Company knowledge applied ({n_files} file(s), "
                  f"{len(getattr(domain_ctx, 'company_sources', []))} records)")
        if graph_block:
            print(f"  ✓ Graph context applied ({cov['graph']} sourced, "
                  f"{cov['llm_reasoned']} llm_reasoned triples)")

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
    # Send trimmed BOM (part_number + name only) to reduce input tokens.
    # The optimizer rebuilds the full BOM in its output anyway.
    bom_trimmed = [{"part_number": p.get("part_number"), "name": p.get("name"),
                    "category": p.get("category"), "quantity": p.get("quantity", 1)}
                   for p in config.get("bom", [])]
    config_data = {
        "configuration": config.get("configuration", {}),
        "bom":           bom_trimmed,
        "constraints":   config.get("constraints", []),
    }

    family: dict          = config.get("_family") or {}
    domain_ctx: DomainContext = config.get("_domain_context")  # type: ignore[assignment]
    product_type   = family.get("family", {}).get("product_type", "product")
    dims           = family.get("scoring_dimensions", [])
    dim_labels     = ", ".join(d["name"] for d in dims) if dims else "quality"

    domain_block = ""
    if domain_ctx and not domain_ctx.is_empty():
        domain_block = f"\n{domain_ctx.prompt_block()}\n"

    # Company constraints block for optimizer — never allowed to violate these
    opt_company_block = ""
    if domain_ctx and getattr(domain_ctx, "company_sources", []):
        lines = ["COMPANY-VERIFIED CONSTRAINTS (absolute — never violate these):"]
        for r in domain_ctx.company_sources:
            sf = r.get("source_file", "unknown")
            sd = r.get("source_date", "")
            for c in r.get("constraints", []):
                if isinstance(c, str) and c.strip():
                    src = f"[{sf}" + (f" {sd}" if sd else "") + "]"
                    lines.append(f"  - {c.strip()}  {src}")
        opt_company_block = "\n".join(lines) + "\n\n"

    prompt = f"""
You are a product design optimizer for: {product_type}.
Improve the configuration below to fix issues and raise scores.

User intent:
{intent_block}
{opt_company_block}{domain_block}

Priority order:
1. Fix ALL critical issues — especially any hard constraint violations.
2. Then address normal issues where possible.
3. Then improve scores toward the goal, focusing on: {dim_labels}
4. Never violate a hard constraint while optimising.
5. Company-verified constraints listed above are absolute — they can NEVER be relaxed.

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

    system = "You are a product design optimizer. Output JSON only. No comments, no trailing text."
    for attempt in range(1, 4):
        raw = call_claude(prompt, system=system, max_tokens=4096, cache_system=True)
        try:
            result = extract_json(raw)
            break
        except Exception as e:
            if attempt < 3:
                print(f"  ⚠ Optimizer JSON parse error (attempt {attempt}) — retrying...")
                prompt += (
                    "\n\nCRITICAL: Your last response contained invalid JSON. "
                    "Output ONLY a valid JSON object. No comments (// or /* */), "
                    "no trailing commas, no prose before or after the JSON. "
                    "Keep BOM part names short (under 40 chars). "
                    "Keep changes list to 5 items max.")
            else:
                raise RuntimeError(f"Optimizer failed to return valid JSON after 3 attempts: {e}") from e
    result["_intent"]         = config.get("_intent")         # carry intent forward
    result["_family"]         = config.get("_family")         # carry family forward
    result["_domain_context"] = config.get("_domain_context") # carry domain context forward

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
    for _attempt in range(2):
        try:
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
                timeout=120,
            )
            break
        except requests.exceptions.ReadTimeout:
            if _attempt == 0:
                print("  ⚠ DALL-E 3 timed out, retrying...")
            else:
                print("  ✗ DALL-E 3 timed out twice — skipping image.")
                return {"status": "timeout", "file": None}

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

def orchestrator(intent: Intent, family: dict,
                  intent_ctx: "IntentContext | None" = None) -> dict:
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

    # Step 1 — domain knowledge (RAG + KG) — runs before configurator
    # Use detected_domain from elicitation agent to skip re-detection inside domain agent
    product_type_str  = family.get("family", {}).get("product_type", "")
    detected_domain   = intent_ctx.detected_domain if intent_ctx else ""
    domain_ctx        = domain_knowledge_agent(
        product_idea    = family.get("family", {}).get("name", ""),
        product_type    = product_type_str,
        intent_goal     = intent.goal,
        detected_domain = detected_domain,
    )

    # Step 2 — initial configuration (guided by family + domain context)
    config          = configurator_agent(intent, family=family, domain_ctx=domain_ctx)
    best_bom        = config.get("bom", [])
    last_eval       = {}
    score_history   = []   # [{iteration, scores}] for the journey chart

    # Step 3 — evaluate / optimize loop
    for iteration in range(1, MAX_ITER + 1):
        separator(f"ITERATION {iteration} / {MAX_ITER}")

        last_eval = evaluator_agent(config)
        score_history.append({"iteration": iteration,
                               "scores": dict(last_eval.get("scores", {}))})

        if should_stop(last_eval, iteration, intent, family, score_history):
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
            "configuration":  optimized.get("configuration", config.get("configuration")),
            "bom":            optimized.get("bom", best_bom),
            "_domain_context": domain_ctx,  # ensure ctx never drops off
        }
        best_bom = config.get("bom", best_bom)

    # Save session (BOM + family) so the user can skip straight to CAD next run
    _save_last_bom(best_bom, family)

    # Step 4 — persist to Airtable
    product_type  = family.get("family", {}).get("product_type", "Product")
    assembly_name = f"AI-{product_type}: {intent.goal[:50]}"
    plm_result    = plm_agent(best_bom, parent_name=assembly_name)

    # Step 5 — visualise / build
    print(f"\n  BOM ready. {len(best_bom)} parts configured.")
    # DTI_VIS_CHOICE is set by the GUI to avoid an interactive prompt
    vis_choice = os.getenv("DTI_VIS_CHOICE", "")
    if not vis_choice:
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
        "intent":        intent,
        "intent_ctx":    intent_ctx,
        "family":        family,
        "final_config":  config,
        "evaluation":    last_eval,
        "score_history": score_history,
        "plm_result":    plm_result,
        "cad_result":    cad_result,
        "image_result":  image_result,
        "domain_ctx":    domain_ctx,
    }
    _save_html_report(outcome)

    # Step 8 — requirements document
    rm_data = requirements_agent(intent, family, config, last_eval)
    _save_rm_document(rm_data, outcome)

    return outcome


# ─────────────────────────────────────────────────────────────
# REQUIREMENTS AGENT
# ─────────────────────────────────────────────────────────────

def requirements_agent(intent: Intent, family: dict,
                       config: dict, evaluation: dict) -> dict:
    """
    Derive a structured requirements document from the design outcome.

    Returns:
    {
      "stakeholder_requirements": [
        {"id": "SR-001", "text": "...", "priority": "shall|should|may",
         "source": "intent goal|constraint|context"}
      ],
      "system_requirements": [
        {"id": "SYS-001", "text": "...", "category": "functional|performance|interface|environmental",
         "priority": "shall|should|may", "derived_from": "SR-001"}
      ],
      "verification": [
        {"req_id": "SYS-001", "method": "test|analysis|inspection|demonstration",
         "acceptance_criteria": "..."}
      ],
      "traceability": [
        {"req_id": "SYS-001", "bom_parts": ["PN-001"]}
      ]
    }
    """
    separator("REQUIREMENTS AGENT")

    f_info       = family.get("family", {})
    dims         = family.get("scoring_dimensions", [])
    constraints  = family.get("constraints", [])
    bom          = config.get("bom", [])
    scores       = evaluation.get("scores", {})
    issues       = [i for i in evaluation.get("issues", []) if i.get("type") == "critical"]

    bom_summary  = [{"part_number": p.get("part_number"), "name": p.get("name"),
                     "category": p.get("category")} for p in bom]

    prompt = f"""
You are a systems engineer generating a concise, traceable requirements document.
Derive requirements from the following product design outcome.

Product      : {f_info.get('name')} ({f_info.get('product_type')})
Description  : {f_info.get('description', '')}

Design intent:
  Goal        : {intent.goal}
  Constraints : {intent.constraints}
  Context     : {intent.context or 'none'}

Family constraints (rules): {json.dumps(constraints)}

Scoring dimensions (what was optimised):
{json.dumps([{"name": d["name"], "description": d["description"],
               "final_score": scores.get(d["name"], "n/a")} for d in dims], indent=2)}

Final BOM ({len(bom)} parts):
{json.dumps(bom_summary, indent=2)}

Unresolved critical issues: {json.dumps([i["text"] for i in issues]) if issues else "none"}

Generate a requirements document with these four sections:

1. stakeholder_requirements — 4-8 high-level "shall" statements derived from the intent goal,
   constraints and context. Each must be testable and unambiguous.

2. system_requirements — 8-15 lower-level requirements derived from the family constraints,
   scoring dimensions and BOM. Categorise each as functional / performance / interface /
   environmental. Each traces to a stakeholder requirement.

3. verification — for every system requirement, one verification entry:
   method (test | analysis | inspection | demonstration) and a concrete acceptance criterion.

4. traceability — map each system requirement to the BOM part(s) that implement it.

Return exactly this JSON structure:
{{
  "stakeholder_requirements": [
    {{"id": "SR-001", "text": "...", "priority": "shall", "source": "intent goal"}}
  ],
  "system_requirements": [
    {{"id": "SYS-001", "text": "...", "category": "functional",
      "priority": "shall", "derived_from": "SR-001"}}
  ],
  "verification": [
    {{"req_id": "SYS-001", "method": "test", "acceptance_criteria": "..."}}
  ],
  "traceability": [
    {{"req_id": "SYS-001", "bom_parts": ["PN-001"]}}
  ]
}}

Rules:
- Requirements must be specific and measurable — no vague "shall be good".
- Performance requirements must include a numeric threshold where possible.
- Output JSON only. No markdown, no prose outside the object.
""".strip()

    raw = call_claude(prompt,
                      system="You are a systems engineer. Output JSON only.",
                      max_tokens=8192)
    try:
        rm = extract_json(raw)
    except Exception as e:
        print(f"  ⚠ Requirements parse error: {e} — returning empty document.")
        rm = {"stakeholder_requirements": [], "system_requirements": [],
              "verification": [], "traceability": []}

    sr  = len(rm.get("stakeholder_requirements", []))
    sys = len(rm.get("system_requirements", []))
    print(f"  ✓ {sr} stakeholder requirements, {sys} system requirements")
    print(f"  ✓ {len(rm.get('verification', []))} verification entries")
    return rm


def _save_rm_document(rm: dict, outcome: dict) -> None:
    """Render the requirements document as a styled HTML file."""
    import time as _time, webbrowser as _wb

    intent   = outcome["intent"]
    family   = outcome["family"]
    f_info   = family.get("family", {})
    timestamp = _time.strftime("%Y-%m-%d %H:%M")

    sr_list  = rm.get("stakeholder_requirements", [])
    sys_list = rm.get("system_requirements", [])
    ver_map  = {v["req_id"]: v for v in rm.get("verification", [])}
    tra_map  = {t["req_id"]: t.get("bom_parts", []) for t in rm.get("traceability", [])}

    # Priority badge colours
    def badge(priority):
        colours = {"shall": ("#1e40af", "#dbeafe"),
                   "should": ("#92400e", "#fef3c7"),
                   "may": ("#374151", "#f3f4f6")}
        bg, fg = colours.get(priority, ("#374151", "#f3f4f6"))
        return (f'<span style="background:{fg};color:{bg};padding:.15rem .5rem;'
                f'border-radius:999px;font-size:.72rem;font-weight:600;'
                f'text-transform:uppercase">{priority}</span>')

    def cat_badge(cat):
        colours = {"functional": "#6366f1", "performance": "#0891b2",
                   "interface": "#059669", "environmental": "#d97706"}
        c = colours.get(cat, "#64748b")
        return (f'<span style="background:{c}22;color:{c};padding:.15rem .5rem;'
                f'border-radius:4px;font-size:.72rem;font-weight:600">{cat}</span>')

    # ── Stakeholder requirements table ────────────────────────
    sr_rows = ""
    for r in sr_list:
        sr_rows += (f"<tr><td style='font-weight:600;color:#1e40af'>{r['id']}</td>"
                    f"<td>{r['text']}</td>"
                    f"<td>{badge(r.get('priority','shall'))}</td>"
                    f"<td style='color:#64748b;font-size:.85rem'>{r.get('source','')}</td></tr>")

    # ── System requirements + verification + traceability ─────
    sys_rows = ""
    for r in sys_list:
        rid  = r["id"]
        ver  = ver_map.get(rid, {})
        parts = ", ".join(tra_map.get(rid, [])) or "—"
        sys_rows += (
            f"<tr>"
            f"<td style='font-weight:600;color:#0f172a'>{rid}</td>"
            f"<td>{r['text']}</td>"
            f"<td>{cat_badge(r.get('category',''))}</td>"
            f"<td>{badge(r.get('priority','shall'))}</td>"
            f"<td style='color:#475569;font-size:.82rem'>{r.get('derived_from','')}</td>"
            f"<td style='font-size:.82rem'><strong>{ver.get('method','')}</strong><br>"
            f"<span style='color:#64748b'>{ver.get('acceptance_criteria','')}</span></td>"
            f"<td style='color:#6366f1;font-size:.82rem'>{parts}</td>"
            f"</tr>")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Requirements — {f_info.get('name','Product')}</title>
<style>
  *{{box-sizing:border-box}}
  body{{font-family:system-ui,sans-serif;margin:0;background:#f1f5f9;color:#1e293b}}
  .header{{background:linear-gradient(135deg,#0f172a 0%,#1e3a8a 60%,#2563eb 100%);
           color:white;padding:2.5rem 2rem 2rem}}
  .header-brand{{font-size:.72rem;letter-spacing:.18em;text-transform:uppercase;
                 opacity:.55;font-weight:600;margin-bottom:.6rem}}
  .header h1{{margin:0 0 .35rem;font-size:1.8rem;font-weight:700}}
  .header p{{margin:0;opacity:.6;font-size:.9rem}}
  .body{{max-width:1100px;margin:2rem auto;padding:0 1rem}}
  .section{{background:white;border-radius:12px;padding:1.5rem;
            box-shadow:0 1px 6px rgba(0,0,0,.07);margin-bottom:1.5rem}}
  h2{{margin:0 0 1.2rem;font-size:.78rem;color:#64748b;text-transform:uppercase;
      letter-spacing:.1em;font-weight:600}}
  table{{width:100%;border-collapse:collapse;font-size:.85rem}}
  th{{background:#f8fafc;text-align:left;padding:.5rem .75rem;font-weight:600;
      color:#475569;border-bottom:2px solid #e2e8f0;white-space:nowrap}}
  td{{padding:.5rem .75rem;border-bottom:1px solid #f1f5f9;vertical-align:top}}
  tr:last-child td{{border-bottom:none}}
  tr:hover td{{background:#fafafa}}
  .intent-block{{background:#f8fafc;border-left:3px solid #6366f1;
                 padding:1rem 1.25rem;border-radius:0 8px 8px 0;
                 font-size:.88rem;margin-bottom:0}}
  .intent-block strong{{display:block;margin-bottom:.25rem;color:#1e293b}}
  .intent-block span{{color:#475569}}
  footer{{text-align:center;padding:2rem 1rem 3rem;font-size:.78rem;color:#94a3b8}}
</style>
</head>
<body>
<div class="header">
  <div class="header-brand">Design to Intent</div>
  <h1>{f_info.get('name','Product')} — Requirements</h1>
  <p>{f_info.get('product_type','')} &nbsp;·&nbsp; {timestamp}</p>
</div>
<div class="body">

  <div class="section">
    <h2>Design Intent</h2>
    <div class="intent-block">
      <strong>Goal</strong><span>{intent.goal}</span>
    </div>
    <div class="intent-block" style="margin-top:.75rem">
      <strong>Hard Constraints</strong>
      <span>{' &nbsp;·&nbsp; '.join(intent.constraints) if intent.constraints else 'none'}</span>
    </div>
    {'<div class="intent-block" style="margin-top:.75rem"><strong>Context</strong><span>' + intent.context + '</span></div>' if intent.context else ''}
  </div>

  <div class="section">
    <h2>Stakeholder Requirements ({len(sr_list)})</h2>
    <table>
      <tr><th>ID</th><th>Requirement</th><th>Priority</th><th>Source</th></tr>
      {sr_rows if sr_rows else '<tr><td colspan="4" style="color:#94a3b8">No requirements generated.</td></tr>'}
    </table>
  </div>

  <div class="section">
    <h2>System Requirements — Verification — Traceability ({len(sys_list)})</h2>
    <table>
      <tr><th>ID</th><th>Requirement</th><th>Category</th><th>Priority</th>
          <th>Derives from</th><th>Verification</th><th>BOM parts</th></tr>
      {sys_rows if sys_rows else '<tr><td colspan="7" style="color:#94a3b8">No requirements generated.</td></tr>'}
    </table>
  </div>

</div>
<footer>Design to Intent &nbsp;·&nbsp; {timestamp} &nbsp;·&nbsp; {f_info.get('product_type','')}</footer>
</body>
</html>"""

    ts       = _time.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(os.path.dirname(__file__), f"requirements_{ts}.html")
    with open(filename, "w", encoding="utf-8") as fh:
        fh.write(html)
    separator("REQUIREMENTS DOCUMENT")
    print(f"  ✓ {len(sr_list)} stakeholder + {len(sys_list)} system requirements")
    print(f"  Saved → {filename}")
    try:
        _wb.open(f"file:///{filename.replace(os.sep, '/')}")
    except Exception:
        pass


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
    """Generate and save a self-contained HTML report with charts. Opens in browser."""
    import time as _time, base64 as _b64, webbrowser as _wb

    intent        = outcome["intent"]
    family        = outcome["family"]
    config        = outcome["final_config"]
    evaluation    = outcome["evaluation"]
    score_history = outcome.get("score_history", [])
    plm_result    = outcome["plm_result"]
    cad_result    = outcome["cad_result"]
    image_result  = outcome["image_result"]

    f_info        = family.get("family", {})
    scores        = evaluation.get("scores", {})
    issues        = evaluation.get("issues", [])
    bom           = config.get("bom", [])
    variants      = family.get("variants", [])
    dims          = family.get("scoring_dimensions", [])
    configuration = config.get("configuration", {})

    dim_labels  = [d["name"] for d in dims] or list(scores.keys())
    dim_values  = [scores.get(d, 0) for d in dim_labels]

    # ── Radar chart data ──────────────────────────────────────
    radar_labels = json.dumps(dim_labels)
    radar_data   = json.dumps(dim_values)

    # ── Journey line chart data ───────────────────────────────
    journey_labels  = json.dumps([f"Iter {h['iteration']}" for h in score_history])
    journey_datasets = []
    colors = ["#6366f1","#22c55e","#f59e0b","#ef4444","#06b6d4"]
    for i, dim in enumerate(dim_labels):
        vals = [h["scores"].get(dim, 0) for h in score_history]
        journey_datasets.append({
            "label": dim,
            "data": vals,
            "borderColor": colors[i % len(colors)],
            "backgroundColor": colors[i % len(colors)] + "22",
            "tension": 0.3,
            "fill": False,
        })
    journey_datasets_json = json.dumps(journey_datasets)
    show_journey = "true" if len(score_history) > 1 else "false"

    # ── BOM table + CSV export ────────────────────────────────
    bom_rows = "".join(
        f"<tr><td>{p.get('part_number','')}</td><td>{p.get('name','')}</td>"
        f"<td>{p.get('category','')}</td><td>{p.get('quantity',1)}</td></tr>"
        for p in bom
    )
    csv_data = "part_number,name,category,quantity\\n" + "\\n".join(
        f"{p.get('part_number','')},{p.get('name','').replace(',','')},{p.get('category','')},{p.get('quantity',1)}"
        for p in bom
    )

    # ── Configuration rows ────────────────────────────────────
    cfg_rows = "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>"
        for k, v in configuration.items()
    )

    # ── Issues ────────────────────────────────────────────────
    issue_html = ""
    for iss in issues:
        cls  = "critical" if iss.get("type") == "critical" else "normal"
        icon = "⚠" if cls == "critical" else "•"
        issue_html += f'<div class="issue {cls}">{icon} {iss.get("text","")}</div>'
    if not issue_html:
        issue_html = '<div class="issue ok">✓ No critical issues</div>'

    # ── Variants ──────────────────────────────────────────────
    variant_html = ""
    for v in variants:
        cfg_preview = ", ".join(f"{k}={val}" for k, val in list(v.get("configuration", {}).items())[:4])
        variant_html += (f"<div class='variant'><strong>{v['name']}</strong>"
                         f" — {v.get('description','')} <br><small>{cfg_preview}</small></div>")


    # ── Intent Elicitation ────────────────────────────────────
    intent_ctx   = outcome.get("intent_ctx")
    elicit_html  = ""
    if intent_ctx:
        _depth_colors = {
            "technical": ("#dcfce7", "#166534"),
            "mixed":     ("#fef9c3", "#854d0e"),
            "vague":     ("#fee2e2", "#991b1b"),
        }
        _dc, _tc = _depth_colors.get(intent_ctx.depth, ("#f1f5f9", "#334155"))
        _domain_color = "#dbeafe" if intent_ctx.detected_domain != "default" else "#f1f5f9"
        _domain_text  = "#1e40af" if intent_ctx.detected_domain != "default" else "#475569"

        qa_rows = ""
        for clar in getattr(intent_ctx, "clarifications", []):
            qa_rows += (
                f"<tr><td style='color:#64748b;font-style:italic'>{clar.question}</td>"
                f"<td>{clar.answer}</td></tr>"
            )

        elicit_html = f"""
  <div class="section full">
    <h2>Intent Elicitation</h2>
    <div style="display:flex;gap:.75rem;flex-wrap:wrap;margin-bottom:1rem">
      <span style="background:{_dc};color:{_tc};padding:.2rem .65rem;border-radius:999px;
                   font-size:.75rem;font-weight:700">depth: {intent_ctx.depth}</span>
      <span style="background:{_domain_color};color:{_domain_text};padding:.2rem .65rem;
                   border-radius:999px;font-size:.75rem;font-weight:700">
        domain: {intent_ctx.detected_domain}</span>
    </div>
    <p style="margin:.25rem 0;font-size:.88rem"><strong>Original input:</strong>
      {intent_ctx.original_input}</p>
    {f'<p style="margin:.25rem 0;font-size:.88rem"><strong>Enriched goal:</strong> {intent_ctx.enriched_goal}</p>' if intent_ctx.enriched_goal else ''}
    {f'<p style="margin:.25rem 0;font-size:.88rem"><strong>Enriched constraints:</strong> {intent_ctx.enriched_constraints}</p>' if intent_ctx.enriched_constraints else ''}
    {f'<p style="margin:.25rem 0;font-size:.88rem"><strong>Context:</strong> {intent_ctx.enriched_context}</p>' if intent_ctx.enriched_context else ''}
    {f'<h3 style="margin:1rem 0 .5rem;font-size:.88rem;color:#475569">Clarification trail</h3><table><tr><th>Question</th><th>Answer</th></tr>{qa_rows}</table>' if qa_rows else ''}
  </div>"""

    # ── Domain Knowledge: sources + V&V coverage ─────────────
    domain_ctx = outcome.get("domain_ctx")
    sources_html  = ""
    vv_html       = ""
    company_html  = ""
    if domain_ctx and not domain_ctx.is_empty():
        sources   = domain_ctx.sources_for_report()
        cov       = domain_ctx.vv_coverage()
        val_trace = evaluation.get("_validation_trace", {})

        # ── Company Knowledge section ─────────────────────────
        company_records = getattr(domain_ctx, "company_sources", [])
        if company_records:
            ck_files_used = val_trace.get("company_files", [])
            ck_rows = ""
            for r in company_records:
                sf  = r.get("source_file", "unknown")
                sd  = r.get("source_date", "")
                cons = r.get("constraints", [])
                decs = r.get("decisions", [])
                used_badge = (
                    ' <span style="background:#dcfce7;color:#166534;padding:.1rem .4rem;'
                    'border-radius:4px;font-size:.7rem;font-weight:600">applied</span>'
                    if sf in ck_files_used else ""
                )
                cons_html = "".join(
                    f"<li style='color:#1e40af;font-size:.82rem'>{c}</li>"
                    for c in cons if isinstance(c, str) and c.strip()
                )
                decs_html = "".join(
                    f"<li style='color:#0f766e;font-size:.82rem'>{d}</li>"
                    for d in decs if isinstance(d, str) and d.strip()
                )
                ck_rows += (
                    f"<tr>"
                    f"<td style='font-weight:600;font-size:.84rem'>{sf}{used_badge}</td>"
                    f"<td style='white-space:nowrap;color:#64748b'>{sd}</td>"
                    f"<td><ul style='margin:0;padding-left:1.2rem'>{cons_html}</ul></td>"
                    f"<td><ul style='margin:0;padding-left:1.2rem'>{decs_html}</ul></td>"
                    f"</tr>"
                )
            company_html = f"""
  <div class="section full" style="border-left:4px solid #1e40af">
    <h2>Company Knowledge ({len(company_records)} records)</h2>
    <p style="font-size:.82rem;color:#475569;margin:-.5rem 0 1rem">
      Source: <code>company_knowledge_*.json</code> &nbsp;·&nbsp;
      Applied to evaluator + optimizer as highest-priority constraints</p>
    <table>
      <tr><th>File</th><th>Date</th><th>Constraints used</th><th>Decisions used</th></tr>
      {ck_rows}
    </table>
  </div>"""

        # Sources table rows — include source_tag badge
        _tag_styles = {
            "company_sourced": ("background:#dbeafe;color:#1e40af",  "company"),
            "domain_specific": ("background:#ede9fe;color:#5b21b6",  "domain"),
            "general":         ("background:#f1f5f9;color:#475569",   "general"),
            "llm_reasoned":    ("background:#fef3c7;color:#92400e",   "llm"),
        }
        src_rows = ""
        for s in sources:
            stale_badge = (' <span style="background:#fef2f2;color:#991b1b;padding:.1rem .4rem;'
                           'border-radius:4px;font-size:.7rem;font-weight:600">STALE</span>'
                           if s.get("stale") else "")
            tag        = s.get("source_tag", "general")
            _ts, _tl   = _tag_styles.get(tag, ("background:#f1f5f9;color:#475569", tag))
            tag_badge  = (f' <span style="{_ts};padding:.1rem .4rem;border-radius:4px;'
                          f'font-size:.7rem;font-weight:600">{_tl}</span>')
            src_rows += (
                f"<tr><td style='font-size:.82rem;word-break:break-all'>"
                f"<a href='{s['url']}' target='_blank' style='color:#2563eb'>{s['url']}</a>"
                f"{stale_badge}{tag_badge}</td>"
                f"<td style='white-space:nowrap'>{s['date']}</td>"
                f"<td style='text-align:center'>{s['chunks']}</td></tr>"
            )

        sources_html = f"""
  <div class="section full">
    <h2>Domain Knowledge Sources ({len(sources)} sources, {len(domain_ctx.rag_chunks)} chunks)</h2>
    <table>
      <tr><th>Source URL</th><th>Date</th><th>Chunks</th></tr>
      {src_rows if src_rows else '<tr><td colspan="3" style="color:#94a3b8">No sources retrieved.</td></tr>'}
    </table>
  </div>"""

        # V&V coverage indicator — now includes company_sourced segment
        n_company  = cov.get("company_sourced", 0)
        total = (n_company + cov["graph"] + cov["rag"] + cov["llm_reasoned"]) or 1
        pct_c  = round(n_company           / total * 100)
        pct_g  = round(cov["graph"]        / total * 100)
        pct_r  = round(cov["rag"]          / total * 100)
        pct_l  = round(cov["llm_reasoned"] / total * 100)
        graph_used   = "yes" if val_trace.get("graph_block_used")   else "no"
        company_used = "yes" if val_trace.get("company_block_used") else "no"
        vv_html = f"""
  <div class="section full">
    <h2>V&amp;V Coverage Indicator</h2>
    <div style="display:flex;gap:1.5rem;flex-wrap:wrap;align-items:center;margin-bottom:.75rem">
      {f'<div style="display:flex;align-items:center;gap:.4rem"><div style="width:12px;height:12px;background:#1e40af;border-radius:2px"></div><span style="font-size:.82rem">Company sourced: <b>{n_company}</b></span></div>' if n_company else ''}
      <div style="display:flex;align-items:center;gap:.4rem">
        <div style="width:12px;height:12px;background:#6366f1;border-radius:2px"></div>
        <span style="font-size:.82rem">Graph sourced: <b>{cov['graph']}</b></span>
      </div>
      <div style="display:flex;align-items:center;gap:.4rem">
        <div style="width:12px;height:12px;background:#22c55e;border-radius:2px"></div>
        <span style="font-size:.82rem">RAG chunks: <b>{cov['rag']}</b></span>
      </div>
      <div style="display:flex;align-items:center;gap:.4rem">
        <div style="width:12px;height:12px;background:#f59e0b;border-radius:2px"></div>
        <span style="font-size:.82rem">LLM-reasoned: <b>{cov['llm_reasoned']}</b></span>
      </div>
      <span style="font-size:.82rem;color:#64748b">Graph: <b>{graph_used}</b> &nbsp;·&nbsp; Company: <b>{company_used}</b></span>
    </div>
    <div style="height:14px;border-radius:7px;overflow:hidden;display:flex;background:#f1f5f9">
      <div style="width:{pct_c}%;background:#1e40af" title="Company sourced"></div>
      <div style="width:{pct_g}%;background:#6366f1" title="Graph sourced"></div>
      <div style="width:{pct_r}%;background:#22c55e" title="RAG"></div>
      <div style="width:{pct_l}%;background:#f59e0b" title="LLM-reasoned"></div>
    </div>
    <p style="margin:.5rem 0 0;font-size:.78rem;color:#94a3b8">
      Company-verified: {pct_c}% &nbsp;·&nbsp;
      Sourced knowledge: {pct_c + pct_g + pct_r}% &nbsp;·&nbsp;
      LLM-inferred: {pct_l}% &nbsp;·&nbsp;
      Stale sources: {sum(1 for c in domain_ctx.rag_chunks if c.stale)}
    </p>
  </div>"""

    # ── Image embed ───────────────────────────────────────────
    img_html = ""
    img_file = image_result.get("file")
    if img_file and os.path.exists(img_file):
        with open(img_file, "rb") as fh:
            b64 = _b64.b64encode(fh.read()).decode()
        img_html = f'<div class="section"><h2>Product Render</h2><img src="data:image/png;base64,{b64}" style="max-width:100%;border-radius:8px;box-shadow:0 4px 20px rgba(0,0,0,.15)"></div>'

    timestamp = _time.strftime("%Y-%m-%d %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Design to Intent — {f_info.get('name','Product')}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  *{{box-sizing:border-box}}
  body{{font-family:system-ui,sans-serif;margin:0;background:#f1f5f9;color:#1e293b}}
  .header{{background:linear-gradient(135deg,#0f172a 0%,#1e3a8a 60%,#2563eb 100%);color:white;padding:2.5rem 2rem 2rem}}
  .header-brand{{font-size:.72rem;letter-spacing:.18em;text-transform:uppercase;opacity:.55;font-weight:600;margin-bottom:.6rem}}
  .header h1{{margin:0 0 .35rem;font-size:2rem;font-weight:700;letter-spacing:-.02em}}
  .header p{{margin:0;opacity:.6;font-size:.9rem}}
  .body{{max-width:1000px;margin:2rem auto;padding:0 1rem}}
  .grid{{display:grid;grid-template-columns:1fr 1fr;gap:1.5rem;margin-bottom:1.5rem}}
  .section{{background:white;border-radius:12px;padding:1.5rem;box-shadow:0 1px 6px rgba(0,0,0,.07)}}
  .section.full{{grid-column:1/-1}}
  h2{{margin:0 0 1.2rem;font-size:.8rem;color:#64748b;text-transform:uppercase;letter-spacing:.08em;font-weight:600}}
  table{{width:100%;border-collapse:collapse;font-size:.88rem}}
  th{{background:#f8fafc;text-align:left;padding:.5rem .75rem;font-weight:600;color:#475569;border-bottom:2px solid #e2e8f0}}
  td{{padding:.45rem .75rem;border-bottom:1px solid #f1f5f9}}
  tr:last-child td{{border-bottom:none}}
  .issue{{padding:.45rem .8rem;margin-bottom:.4rem;border-radius:6px;font-size:.88rem}}
  .issue.critical{{background:#fef2f2;color:#991b1b;border-left:3px solid #ef4444}}
  .issue.normal{{background:#fffbeb;color:#92400e;border-left:3px solid #f59e0b}}
  .issue.ok{{background:#f0fdf4;color:#166534;border-left:3px solid #22c55e}}
  .variant{{background:#f8fafc;border-left:3px solid #6366f1;padding:.6rem 1rem;margin-bottom:.6rem;border-radius:0 8px 8px 0;font-size:.88rem}}
  .meta{{display:flex;gap:1.5rem;flex-wrap:wrap;font-size:.82rem;color:#64748b;margin-top:.75rem}}
  .meta b{{color:#1e293b}}
  .summary{{margin:.75rem 0 0;font-size:.88rem;color:#64748b;font-style:italic}}
  .chart-wrap{{position:relative;height:260px}}
  .btn{{display:inline-block;margin-top:1rem;padding:.45rem 1rem;background:#1e40af;color:white;border:none;border-radius:6px;font-size:.82rem;cursor:pointer;text-decoration:none}}
  .btn:hover{{background:#1d4ed8}}
</style>
</head>
<body>
<div class="header">
  <div class="header-brand">Design to Intent</div>
  <h1>{f_info.get('name','Product')}</h1>
  <p>{f_info.get('description','')} &nbsp;·&nbsp; {timestamp}</p>
</div>
<div class="body">

  <div class="section full">
    <h2>Design Goal</h2>
    <p style="font-size:1.05rem;margin:0 0 .5rem;font-weight:600">{intent.goal}</p>
    <div class="meta">
      <span>Constraints: <b>{', '.join(intent.constraints) if intent.constraints else 'none'}</b></span>
      {'<span>Context: <b>' + intent.context + '</b></span>' if intent.context else ''}
      <span>BOM: <b>{len(bom)} parts</b></span>
      <span>Airtable: <b>{plm_result.get('parts_created',0)} parts written</b></span>
    </div>
  </div>

  <div class="grid">
    <div class="section">
      <h2>Score Radar</h2>
      <div class="chart-wrap">
        <canvas id="radarChart"></canvas>
      </div>
      <p class="summary">{evaluation.get('summary','')}</p>
    </div>

    <div class="section">
      <h2>Optimization Journey</h2>
      <div class="chart-wrap">
        <canvas id="journeyChart"></canvas>
      </div>
      {'<p class="summary">Single iteration — no journey to show.</p>' if not show_journey == "true" else ''}
    </div>
  </div>

  <div class="section full">
    <h2>Issues</h2>
    {issue_html}
  </div>

  <div class="grid">
    <div class="section">
      <h2>Selected Configuration</h2>
      <table><tr><th>Feature</th><th>Option</th></tr>{cfg_rows}</table>
    </div>

    <div class="section">
      <h2>Product Variants</h2>
      {variant_html}
    </div>
  </div>

  {elicit_html}

  {company_html}

  {sources_html}

  {vv_html}

  <div class="section full">
    <h2>Bill of Materials ({len(bom)} parts)</h2>
    <table>
      <tr><th>Part No.</th><th>Name</th><th>Category</th><th>Qty</th></tr>
      {bom_rows}
    </table>
    <button class="btn" onclick="downloadCSV()">Export BOM as CSV</button>
  </div>

  {img_html}

</div>
<div style="text-align:center;padding:2rem 1rem 3rem;font-size:.78rem;color:#94a3b8">
  Design to Intent &nbsp;·&nbsp; {timestamp} &nbsp;·&nbsp; {f_info.get('product_type','Product')}
</div>

<script>
// ── Radar chart ──────────────────────────────────────────────
new Chart(document.getElementById('radarChart'), {{
  type: 'radar',
  data: {{
    labels: {radar_labels},
    datasets: [{{
      label: 'Final Scores',
      data: {radar_data},
      backgroundColor: 'rgba(99,102,241,0.15)',
      borderColor: '#6366f1',
      borderWidth: 2,
      pointBackgroundColor: '#6366f1',
      pointRadius: 4,
    }}]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    scales: {{ r: {{ min:0, max:10, ticks: {{ stepSize:2, font:{{size:10}} }},
      pointLabels: {{ font:{{size:11}} }} }} }},
    plugins: {{ legend: {{ display: false }} }}
  }}
}});

// ── Journey line chart ────────────────────────────────────────
const journeyCtx = document.getElementById('journeyChart');
if ({show_journey}) {{
  new Chart(journeyCtx, {{
    type: 'line',
    data: {{
      labels: {journey_labels},
      datasets: {journey_datasets_json}
    }},
    options: {{
      responsive: true,
      maintainAspectRatio: false,
      scales: {{
        y: {{ min:0, max:10, ticks:{{ stepSize:2 }} }},
        x: {{ grid:{{ display:false }} }}
      }},
      plugins: {{ legend: {{ position:'bottom', labels:{{ boxWidth:10, font:{{size:10}} }} }} }}
    }}
  }});
}} else {{
  journeyCtx.style.display = 'none';
}}

// ── CSV export ────────────────────────────────────────────────
function downloadCSV() {{
  const csv = `{csv_data}`;
  const a   = document.createElement('a');
  a.href    = 'data:text/csv;charset=utf-8,' + encodeURIComponent(csv);
  a.download = '{f_info.get("name","bom").replace(" ","_")}_bom.csv';
  a.click();
}}
</script>
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
    print("  DESIGN TO INTENT")
    print("  AI-driven product design — from idea to optimised BOM")
    print("═"*60)
    print("""
What product do you want to design?

Examples:
  - modular home energy storage system
  - electric mountain bike
  - professional inspection robot
  - espresso machine for home use
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
    context_preset = ""
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
  "goal": "concise goal statement, e.g. 'maximum range at under-market price'",
  "constraints": ["hard constraint 1", "hard constraint 2"],
  "context": "use case, target user, environment — e.g. 'residential installer, EU market, garage wall-mount'",
  "reasoning": "one sentence explaining what market gap this targets and why"
}}

Output JSON only.
""".strip()
            raw  = call_claude(auto_prompt,
                               system="You are a product strategist. Output JSON only.",
                               max_tokens=512)
            rec  = extract_json(raw)
            goal               = rec.get("goal", "")
            constraints_preset = rec.get("constraints", [])
            context_preset     = rec.get("context", "")
            print(f"\n  Recommended goal       : {goal}")
            print(f"  Recommended constraints: {constraints_preset}")
            print(f"  Recommended context    : {context_preset}")
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
        print(f"  Auto-filled constraints:")
        for c in constraints:
            print(f"    • {c}")
        extra = input("  Add more constraints (or Enter to accept): ").strip()
        if extra:
            constraints.append(extra)
            while True:
                c = input(f"  Constraint {len(constraints)+1} (or Enter to finish): ").strip()
                if not c:
                    break
                constraints.append(c)
    else:
        while True:
            c = input(f"  Constraint {len(constraints)+1} (or Enter to finish): ").strip()
            if not c:
                break
            constraints.append(c)

    if context_preset:
        print(f"\nContext (auto-filled): {context_preset}")
        override = input("  Override? (Enter to accept, or type replacement): ").strip()
        context = override or context_preset
    else:
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
    parser = argparse.ArgumentParser(description="Design to Intent — AI Product Design")
    parser.add_argument("--setup",       action="store_true",
                        help="Verify config and create Airtable tables, then exit")
    parser.add_argument("--idea",        type=str, default="",
                        help="Product idea (skips the interactive prompt)")
    parser.add_argument("--goal",        type=str, default="",
                        help="Design goal (skips variant picker / auto-intent prompt)")
    parser.add_argument("--constraints", type=str, default="",
                        help="Hard constraints, comma-separated")
    parser.add_argument("--context",     type=str, default="",
                        help="Extra context (use case, environment, user profile)")
    args = parser.parse_args()

    _check_config()
    setup_airtable()

    if args.setup:
        print("\n  Setup complete. Run: python plm_agents.py  or  python gui.py")
        raise SystemExit(0)

    last_bom, last_family = _load_last_session()

    if last_bom:
        print("\n" + "═"*60)
        print("  DESIGN TO INTENT")
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

        # ── Intent Elicitation (runs before family agent) ─────────
        # Non-interactive when --goal is provided (GUI path / scripted)
        _interactive_mode = not bool(args.goal)
        intent_ctx = intent_elicitation_agent(
            product_idea        = product_idea,
            existing_goal       = args.goal,
            existing_constraints= args.constraints,
            existing_context    = args.context,
            interactive         = _interactive_mode,
        )

        # Load company constraints before family agent so they inform the family definition
        _ck_constraints, _ck_decisions, _ck_summary = _load_company_constraints(product_idea)
        if _ck_constraints:
            print(f"\n  Company knowledge detected: {_ck_summary}")

        family = product_family_agent(product_idea,
                                      company_constraints=_ck_constraints or None)

        # Non-interactive mode when --goal is supplied (e.g. from GUI)
        if args.goal:
            # Use enriched values from elicitation if synthesis produced something,
            # otherwise fall back to raw CLI args
            goal_str  = intent_ctx.enriched_goal or args.goal
            cons_str  = intent_ctx.enriched_constraints or args.constraints
            ctx_str   = intent_ctx.enriched_context or args.context
            constraints = [c.strip() for c in cons_str.split(",") if c.strip()]
            intent = Intent(goal=goal_str, constraints=constraints, context=ctx_str)
            print(f"\n  Intent (from CLI args):")
            print(f"    Goal        : {intent.goal}")
            print(f"    Constraints : {intent.constraints}")
            print(f"    Context     : {intent.context or 'none'}")
        else:
            intent = ask_intent(family)
        orchestrator(intent, family, intent_ctx=intent_ctx)
