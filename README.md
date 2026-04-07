# Design to Intent

AI-driven product design — from idea to optimised BOM, competitive landscape, and report.

---

## How it works

```
  You type: "modular home energy storage system"
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│  PRODUCT FAMILY AGENT                                           │
│  Generates the product's feature model from scratch             │
│                                                                 │
│  → Features: cell_chemistry, capacity_kwh, inverter_type, ...  │
│  → Options:  LFP | NMC | LTO  /  5kWh | 10kWh | 15kWh  / ...  │
│  → Constraints: "LTO requires active cooling"                   │
│  → Variants: entry-level / standard / pro                       │
│  → Scoring: usable_capacity, efficiency, install_cost           │
│                                                                 │
│  Writes family to Airtable                                      │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  COMPETITIVE ANALYSIS AGENT                                     │
│  Identifies real market competitors using Claude's knowledge    │
│                                                                 │
│  → Tesla Powerwall 3   [€9,500]  — best ecosystem integration   │
│  → Sonnen Eco          [€12,000] — premium, long warranty       │
│  → BYD Battery-Box     [€5,800]  — value, modular expansion     │
│  → Enphase IQ Battery  [€7,200]  — AC-coupled, easy retrofit    │
│                                                                 │
│  Market gap: nothing strong at €4k–6k with open protocols       │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  INTENT DEFINITION                                              │
│                                                                 │
│  Option 0: Auto — Claude recommends based on market gap         │
│    → Goal: "best value at under €5,000 with open BMS"           │
│    → Constraints: ["grid-tie capable", "IP55 outdoor rating"]   │
│    → Context: "residential installer, EU market"                │
│                                                                 │
│  Or pick a predefined variant, or type your own goal            │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  CONFIGURATOR AGENT                                             │
│  Selects valid options, builds initial BOM                      │
│                                                                 │
│  Configuration:  LFP cells / 10kWh / hybrid inverter / IP55    │
│  BOM: 14 parts — cell modules, BMS, inverter, enclosure, ...    │
└──────────────────────────────┬──────────────────────────────────┘
                               │
              ┌────────────────┘
              │   up to 3 iterations
              ▼
┌─────────────────────────────────────────────────────────────────┐
│  EVALUATOR AGENT              →    OPTIMIZER AGENT              │
│                                                                 │
│  Scores each dimension:            Fixes critical issues first  │
│  usable_capacity  7/10             then improves toward intent  │
│  efficiency       6/10      →                                   │
│  install_cost     8/10             Adjusts config + BOM         │
│                                    Stops early if no progress   │
│  Issues:                                                        │
│  ⚠ BMS lacks CAN bus (critical)                                 │
│  ⚠ no surge protection (critical)                               │
└──────────────────────────────┬──────────────────────────────────┘
                               │  scores ≥ 8 or no improvement
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  BUILDER AGENT (Airtable)                                       │
│  Writes final parts + BOM to your Airtable base                 │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
             ┌─────────────────┴─────────────────┐
             │                                   │
             ▼                                   ▼
┌────────────────────────┐          ┌────────────────────────────┐
│  CAD AGENT (optional)  │          │  IMAGE AGENT (optional)    │
│  Onshape parametric    │          │  DALL-E 3 product render   │
│  3D model via API      │          │  1792×1024 HD image        │
└────────────┬───────────┘          └─────────────┬──────────────┘
             └─────────────────┬─────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  HTML REPORT  (auto-opens in browser)                           │
│  Radar chart · Optimization journey · Competitive landscape     │
│  BOM table with CSV export · Configuration · Issues · Render   │
└─────────────────────────────────────────────────────────────────┘
```

See **[example_report.html](example_report.html)** for a full sample output.

---

## A different relationship to product development

Traditional product development puts requirements first:

```
Market research → Requirements → Design → Verify against requirements
```

Design to Intent inverts this. You express *what you're trying to achieve*, and the system reasons about what matters, explores the solution space, and verifies against your intent continuously — not at a gate at the end.

**Your role shifts from specification writer to product strategist.** You define the goal, set the hard limits, judge the trade-offs. The agents handle the feature model, the BOM, the scoring, the iteration.

**What replaces requirements management:**

| Traditional RM | Design to Intent equivalent |
|---|---|
| Requirements document | Intent (goal + constraints + context) |
| Feature tree / product breakdown | Product family (auto-generated) |
| Shall statements | Scoring dimensions (measurable, 0–10) |
| Verification matrix | Evaluation loop — continuous, not a gate |
| BOM as output | BOM as the primary truth artifact |

The constraints and scoring dimensions that emerge from the pipeline *are* a lightweight requirements spec — they just derive from reasoning about the product rather than being written upfront by a requirements engineer.

**Do you still need formal requirements management?**
For regulated industries (medical, aerospace, automotive) — yes, legal traceability obligations don't go away. For everything else — concept work, startups, innovation — intent is enough, and significantly faster.

---

## Running

### CLI
```bash
python plm_agents.py
python plm_agents.py --idea "ergonomic standing desk"
python plm_agents.py --idea "espresso machine" --goal "maximum extraction quality" --constraints "cost under €800, domestic voltage"
```

### Desktop GUI
```bash
python gui.py
```

The GUI collects your product idea, optional intent fields, and visualisation choice, then streams all agent output live into a log window. When the pipeline finishes, a **View Report** button opens the HTML report.

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
| `OPENAI_API_KEY` | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) _(image generation only)_ |
| `ONSHAPE_ACCESS_KEY` / `ONSHAPE_SECRET_KEY` | [dev-portal.onshape.com/keys](https://dev-portal.onshape.com/keys) _(CAD only)_ |
| `ONSHAPE_DID` / `ONSHAPE_WID` / `ONSHAPE_EID` | Your Onshape document URL _(CAD only)_ |

Onshape and OpenAI keys are optional.

### 3. First-time setup

```bash
python plm_agents.py --setup
```

Creates all required Airtable tables automatically and verifies your API keys.

---

## Models

| Agent | Model | Reason |
|---|---|---|
| Product Family, Competitive Analysis | `claude-haiku-4-5` | Structured data generation — fast and cheap |
| Configurator, Evaluator, Optimizer | `claude-sonnet-4-6` | Reasoning quality matters here |
| CAD planning | `claude-opus-4-6` | Extended thinking for geometry |
| Image generation | DALL-E 3 (1792×1024 HD) | |

---

## Notes

- **Product-agnostic** — scoring dimensions, features, and CAD geometry all come from the product family the system defines, not hardcoded rules. Works for any physical product.
- **Competitive analysis** — real competitors identified before you set your intent; auto-intent uses the market gap to recommend a differentiated goal.
- **Diminishing returns** — the optimizer loop stops early if scores don't improve iteration-over-iteration, saving API calls.
- **Session saved** after each run — next run offers CAD-only or image-only from the last design, with full family context preserved.
- **Fully non-interactive** — pass `--idea`, `--goal`, `--constraints`, `--context` to skip all prompts (used by the GUI).

---

See [`.env.example`](.env.example) for the full environment variable reference.
