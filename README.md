# AI Product Agents

A multi-agent system that takes a product idea and a design intent, then configures, evaluates, and builds it — automatically.

```
Your idea  →  "electric mountain bike"
Your intent →  "maximum range, cost under €2000"

  Product Family Agent   — defines features, options, constraints, variants
  Configurator Agent     — selects a valid configuration and builds a BOM
  Evaluator Agent        — scores the design against what matters for this product
  Optimizer Agent        — fixes issues and improves scores iteratively
  Builder Agent          — writes the result to Airtable
  CAD Agent (optional)   — generates a parametric 3D model in Onshape
```

Works for any product — bikes, robots, cameras, machines, drones, appliances.

---

## How it works

**1. You give it a product idea.**
Anything: `inspection robot`, `portable solar station`, `espresso machine for cafes`.

**2. The system defines the product family.**
It creates a structured feature model — features, options per feature, cross-feature constraints, and 2–3 predefined variants. Like a configurator in Configit or pure::variants, but generated from scratch.

**3. You define your intent.**
Based on the product family it just built, the system shows you the relevant features and scoring dimensions, then asks:
- What do you want to optimise for?
- What are your hard constraints?

**4. The agents run.**
Configuration → Evaluation → Optimization loop → Airtable → optional CAD in Onshape.

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/config-collab/ai-product-agents.git
cd ai-product-agents
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

Onshape keys are optional — skip the CAD step and everything else still works.

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
python plm_agents.py
```

---

## Models

| Agent | Model |
|---|---|
| Product Family, Configurator, Evaluator, Optimizer | `claude-sonnet-4-6` |
| CAD planning | `claude-opus-4-6` with extended thinking |

---

## Notes

- **Product-agnostic** — scoring dimensions, features, and CAD geometry all come from the product family the system defines, not hardcoded rules.
- **Last design cached** in `.last_bom.json` (gitignored) — next run offers to skip straight to CAD.
- **Prompt caching** on evaluator/optimizer system prompts cuts API cost during optimization loops.

---

See [`.env.example`](.env.example) for the full environment variable reference.
