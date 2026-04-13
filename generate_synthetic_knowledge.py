"""
Generate Synthetic Knowledge
=============================
Standalone test script — no connection to the agent pipeline.

Generates 50 synthetic engineering reasoning records for a product,
covering intent, constraints, trade-offs, alternatives rejected.
All records tagged 'synthetic_pending' for human review before use.

Usage:
    python generate_synthetic_knowledge.py
    python generate_synthetic_knowledge.py --product "espresso machine"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import anthropic as _anthropic
    _client = _anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
    _MODEL  = "claude-sonnet-4-6"
except Exception as e:
    print(f"  ✗ Cannot import anthropic: {e}")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _slug(name: str) -> str:
    """Convert product name to a safe filename slug."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:50]


def _call(prompt: str, max_tokens: int = 4096) -> str:
    rsp = _client.messages.create(
        model=_MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return rsp.content[0].text.strip()


def _extract_json(text: str):
    """Extract first JSON array or object from text."""
    # Fenced block
    m = re.search(r"```(?:json)?\s*(\[[\s\S]*?\]|\{[\s\S]*?\})\s*```", text)
    if m:
        raw = m.group(1)
    else:
        start = min(
            (text.find("[") if text.find("[") != -1 else len(text)),
            (text.find("{") if text.find("{") != -1 else len(text)),
        )
        end = max(text.rfind("]"), text.rfind("}")) + 1
        raw = text[start:end]
    # Strip trailing commas
    raw = re.sub(r",\s*([}\]])", r"\1", raw)
    return json.loads(raw)


# ─────────────────────────────────────────────────────────────────────────────
# GENERATION
# ─────────────────────────────────────────────────────────────────────────────

_BATCH_SIZE = 10   # records per Claude call (50 total = 5 calls)
_TOTAL      = 50


def _generate_batch(product: str, batch_num: int, start_id: int) -> list[dict]:
    """Generate _BATCH_SIZE synthetic reasoning records in one Claude call."""
    prompt = f"""You are a senior systems engineer with 20 years of product development experience.

Generate exactly {_BATCH_SIZE} synthetic engineering reasoning records for a: {product}

Each record must be a realistic, plausible engineering decision scenario —
the kind of reasoning captured in design reviews, FMEA sessions, or trade studies.

Return a JSON array of exactly {_BATCH_SIZE} objects. Each object:
{{
  "record_id":            "SK-{start_id + i:03d}",   // sequential
  "intent":               "string — what the engineer was trying to achieve",
  "constraints":          ["string", ...],  // 2-4 hard constraints that applied
  "trade_offs":           "string — what was gained vs. lost in the decision",
  "alternatives_rejected": ["string", ...],  // 2-3 options that were considered but rejected, each with a brief reason
  "decision":             "string — what was actually chosen and why",
  "confidence":           "plausible",
  "domain_area":          "string — e.g. thermal, structural, electrical, software, regulatory, cost"
}}

IMPORTANT:
- All records must be specific and realistic for a: {product}
- Use real standards, materials, part families, and engineering units where relevant
- Records should cover a diverse range of domain areas: thermal, structural, electrical,
  software, manufacturing, regulatory, supply chain, cost, reliability, safety
- Do NOT repeat the same scenario — vary the decision context significantly
- Output only the JSON array, no prose before or after.

This is batch {batch_num} of 5. Generate records SK-{start_id:03d} through SK-{start_id + _BATCH_SIZE - 1:03d}.
""".strip()

    raw     = _call(prompt)
    records = _extract_json(raw)
    if not isinstance(records, list):
        raise ValueError(f"Expected list, got {type(records)}")
    return records


def generate_synthetic_knowledge(product: str) -> list[dict]:
    """Generate all 50 records across 5 batches."""
    all_records = []
    for batch in range(1, (_TOTAL // _BATCH_SIZE) + 1):
        start_id = (batch - 1) * _BATCH_SIZE + 1
        print(f"  Generating batch {batch}/5 (records SK-{start_id:03d}–SK-{start_id + _BATCH_SIZE - 1:03d})...",
              end=" ", flush=True)
        try:
            records = _generate_batch(product, batch, start_id)
            # Ensure tag and flatten to text for RAG ingestion
            for r in records:
                r["tag"]       = "synthetic_pending"
                r["product"]   = product
                r["text"]      = _flatten_record(r)
            all_records.extend(records)
            print(f"✓ {len(records)} records")
        except Exception as e:
            print(f"✗ failed: {e}")
    return all_records


def _flatten_record(r: dict) -> str:
    """Flatten a record into a single text blob for RAG ingestion."""
    parts = [
        f"Product: {r.get('product', '')}",
        f"Intent: {r.get('intent', '')}",
        f"Domain area: {r.get('domain_area', '')}",
    ]
    if r.get("constraints"):
        parts.append("Constraints: " + "; ".join(r["constraints"]))
    if r.get("trade_offs"):
        parts.append(f"Trade-offs: {r['trade_offs']}")
    if r.get("alternatives_rejected"):
        parts.append("Alternatives rejected: " + "; ".join(r["alternatives_rejected"]))
    if r.get("decision"):
        parts.append(f"Decision: {r['decision']}")
    parts.append(f"Confidence: {r.get('confidence', 'plausible')}")
    parts.append(f"Tag: {r.get('tag', 'synthetic_pending')}")
    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic engineering knowledge")
    parser.add_argument("--product", type=str, default="",
                        help="Product name (skips interactive prompt)")
    args = parser.parse_args()

    print("\n" + "═" * 60)
    print("  SYNTHETIC KNOWLEDGE GENERATOR")
    print("═" * 60)
    print("  Generates 50 synthetic engineering reasoning records.")
    print("  All records tagged 'synthetic_pending' — review before use.\n")

    product = args.product.strip() or input("  What product? > ").strip()
    if not product:
        print("  ✗ No product specified.")
        sys.exit(1)

    slug     = _slug(product)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             f"synthetic_{slug}_knowledge.json")

    print(f"\n  Product : {product}")
    print(f"  Output  : {out_path}")
    print(f"  Records : {_TOTAL}\n")

    records = generate_synthetic_knowledge(product)

    output = {
        "product":      product,
        "generated_at": datetime.now().isoformat(),
        "model":        _MODEL,
        "tag":          "synthetic_pending",
        "total":        len(records),
        "records":      records,
    }

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)

    print(f"\n  ✓ {len(records)} records saved → {out_path}")
    print("""
  Next steps:
    1. Review records — edit or delete any that are implausible
    2. Promote to company knowledge:  python ingest_company_knowledge.py
    3. Or use directly by renaming:   company_knowledge_{slug}.json
       (and changing tag from 'synthetic_pending' to 'company_sourced')
""")


if __name__ == "__main__":
    main()
