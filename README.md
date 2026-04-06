# AI Product PLM Agent

A multi-agent system for AI-driven product configuration, bill of materials generation, and optional CAD model creation — for **any product type**.

**Pipeline:**
```
Product Idea  (you type: "electric mountain bike", "inspection drone", etc.)
  → Product Family Agent   — defines features, options, constraints, variants
  → Configurator Agent     — selects a valid configuration and builds a BOM
  → Evaluator Agent        — scores the design across product-specific dimensions
  → Optimizer Agent        — fixes issues and improves scores iteratively
  → PLM Agent              — persists BOM to Airtable
  → CAD Agent (optional)   — generates a parametric 3D model in Onshape
```

Powered by Claude claude-sonnet-4-6 / Opus 4.6 and connected to Airtable for PLM data and Onshape for CAD.

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/nexerax-collab/ai-product-plm.git
cd ai-product-plm
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
```

Open `.env` and fill in:

| Variable | Where to get it |
|---|---|
| `ANTHROPIC_API_KEY` | [console.anthropic.com/keys](https://console.anthropic.com/keys) |
| `AIRTABLE_TOKEN` | [airtable.com/create/tokens](https://airtable.com/create/tokens) — scopes: `data.records:write`, `schema.bases:read` |
| `AIRTABLE_BASE_ID` | Your base URL: `airtable.com/<BASE_ID>/...` |
| `ONSHAPE_ACCESS_KEY` / `ONSHAPE_SECRET_KEY` | [dev-portal.onshape.com/keys](https://dev-portal.onshape.com/keys) _(CAD only)_ |
| `ONSHAPE_DID` / `ONSHAPE_WID` / `ONSHAPE_EID` | Your Onshape document URL _(CAD only)_ |

Onshape keys are optional — skip the CAD step and the rest of the pipeline works fine.

### 3. Airtable base structure

Create a base with these tables:

| Table | Fields |
|---|---|
| **Product Families** | Name, Product Type, Description |
| **Features** | Name, Type, Family |
| **Feature Options** | Feature, Value, Family |
| **Constraints** | Rule, Family |
| **Parts** | part_number, name, category, active |
| **BOM** | parent, part_number (linked), quantity, level, notes |

### 4. Run

```bash
python drone_plm_agents.py
```

You will be prompted for:
1. **Product idea** — anything: `inspection drone`, `electric mountain bike`, `espresso machine`
2. **Design intent** — your goal and hard constraints, shown after the product family is generated

---

## How it works

### Product Family Agent
Takes your product idea and defines a structured feature model — like Configit or pure::variants. Returns features (enum/boolean/numeric), options per feature, cross-feature constraints, 2–3 predefined variants, and the scoring dimensions used by the evaluator.

### Configurator Agent
Uses the family definition to select a valid configuration and generate a realistic BOM with part numbers, names, categories, and quantities.

### Evaluator + Optimizer loop
Scores the configuration against the product-specific dimensions (e.g. `range_km`, `trail_performance`, `cost` for an e-bike). Runs up to 5 iterations, stopping when the primary metric hits ≥ 8/10 with no critical issues.

### PLM Agent
Writes all parts and BOM entries to Airtable in batched requests.

### CAD Agent _(drone-specific)_
Generates a parametric drone model in Onshape using the onshape-mcp library. Prompted after PLM — you choose whether to run it.

---

## Models

| Agent | Model |
|---|---|
| Product Family, Configurator, Evaluator, Optimizer, PLM | `claude-sonnet-4-6` |
| CAD planning | `claude-opus-4-6` with extended thinking |

---

## Notes

- **Product-agnostic** except for CAD — evaluator/optimizer dimensions come from the product family, not hardcoded logic.
- **Last design cached** in `.last_bom.json` (gitignored) — next run offers to skip straight to CAD.
- **Prompt caching** on evaluator/optimizer system prompts cuts API cost in long optimization loops.

---

## Environment variables

See [`.env.example`](.env.example) for the full list with descriptions.
