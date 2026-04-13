"""
Design to Intent — Intent Elicitation Agent
============================================
Runs before the Domain Knowledge Agent.

Responsibilities:
1. Score input depth: technical | mixed | vague
2. If vague: ask up to 3 clarifying questions (informed by detected domain)
3. Log intent_context: original input + clarifications + detected domain
4. Return enriched goal / constraints / context for downstream agents

Usage:
    from intent_elicitation import IntentContext, intent_elicitation_agent
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

# ── Optional Anthropic client (graceful fallback) ─────────────────
try:
    import anthropic as _anthropic
    _CLIENT = _anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
    _HAIKU  = "claude-haiku-4-5-20251001"
    _HAS_ANTHROPIC = True
except Exception:
    _CLIENT = None
    _HAS_ANTHROPIC = False


# ─────────────────────────────────────────────────────────────────
# Domain keyword registry (mirrors domain_knowledge.py — kept local
# so this module has no hard dependency on that file)
# ─────────────────────────────────────────────────────────────────
_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "drone": [
        "drone", "uav", "quadcopter", "multirotor", "fpv", "hexacopter",
        "octocopter", "fixed-wing", "vtol", "rotor", "flight controller",
        "ardupilot", "pixhawk", "betaflight",
    ],
    "energy_storage": [
        "battery", "energy storage", "bms", "lfp", "nmc", "lto", "lifepo4",
        "cell", "pack", "inverter", "solar", "grid", "kwh", "charge",
        "discharge", "thermal management",
    ],
    "automotive": [
        "car", "vehicle", "automotive", "ev", "autosar", "misra",
        "iso 26262", "powertrain", "adas", "lidar", "radar", "can bus",
        "obd", "ecu", "brake", "steering",
    ],
    "aerospace": [
        "spacecraft", "satellite", "rocket", "ecss", "nasa", "orbit",
        "launch", "propulsion", "attitude control", "thermal control",
        "radiation", "cubesat", "starlink",
    ],
    "medical": [
        "medical", "device", "implant", "fda", "510k", "iso 13485",
        "iso 14971", "iec 60601", "biocompatibility", "sterilization",
        "clinical", "patient", "wearable", "diagnostic",
    ],
    "electronics": [
        "pcb", "microcontroller", "mcu", "embedded", "arduino", "raspberry pi",
        "rp2040", "esp32", "stm32", "fpga", "soc", "gpio", "uart", "spi", "i2c",
        "jtag", "swd", "bootloader", "firmware", "kicad", "breadboard",
        "dev board", "development board", "single board", "micropython",
        "circuitpython", "bare metal", "rtos", "schematic",
    ],
}

# Domain-specific clarifying question banks
_DOMAIN_QUESTIONS: dict[str, list[str]] = {
    "drone": [
        "What is the primary use case — photography, inspection, delivery, or racing?",
        "What regulatory environment applies — FAA Part 107 (US), EASA UAS (EU), or other?",
        "What is the target flight time and maximum payload weight?",
        "Will the drone operate beyond visual line of sight (BVLOS)?",
        "What environment — indoor, outdoor urban, or remote/rural?",
    ],
    "energy_storage": [
        "Is this a stationary (grid/home) or mobile (EV/portable) application?",
        "What is the target energy capacity in kWh and peak discharge rate?",
        "Are there specific certifications required — UL 9540, IEC 62619, UN 38.3?",
        "What thermal management approach is preferred — passive, active air, or liquid cooling?",
        "Will this be grid-tied, off-grid, or hybrid?",
    ],
    "automotive": [
        "Is this for passenger, commercial, or off-road vehicle application?",
        "What ASIL safety integrity level is required (ASIL A–D or QM)?",
        "Is this an EV, hybrid, or ICE platform?",
        "What communication protocols are required — CAN, LIN, Ethernet, FlexRay?",
        "Does the design need to comply with a specific regional homologation (EU, US, China)?",
    ],
    "aerospace": [
        "What orbit or mission profile — LEO, GEO, deep space, suborbital?",
        "What launch vehicle and associated interface requirements apply?",
        "What radiation tolerance level is required (total ionizing dose)?",
        "Is this a commercial, government, or academic mission?",
        "What redundancy level is required — cold standby, hot standby, or TMR?",
    ],
    "medical": [
        "What device classification applies — Class I, II, or III (FDA) / Class I, IIa, IIb, III (EU MDR)?",
        "Is this an implantable, wearable, or capital equipment device?",
        "What is the intended patient population and clinical environment?",
        "Is the device life-sustaining or life-supporting?",
        "What sterilization method will be used — EtO, gamma, steam, or e-beam?",
    ],
    "electronics": [
        "What microcontroller or SoC family is preferred — RP2040, ESP32, STM32, or open?",
        "What connectivity is required — USB only, Wi-Fi, Bluetooth, LoRa, Ethernet?",
        "What is the target BOM cost per unit and expected production volume?",
        "What programming environment must be supported — MicroPython, Arduino, C/C++ SDK?",
        "Are there specific certifications needed — FCC, CE, RoHS, or OSHWA open hardware?",
    ],
    "default": [
        "What is the primary end-user and their technical expertise level?",
        "Are there specific regulatory standards or certifications required?",
        "What is the target unit cost and production volume?",
        "What environmental conditions must the product withstand?",
        "What is the most critical performance metric for success?",
    ],
}


# ─────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────

@dataclass
class Clarification:
    question: str
    answer:   str


@dataclass
class IntentContext:
    original_input:       str
    depth:                str   = "vague"          # technical | mixed | vague
    detected_domain:      str   = "default"
    clarifications:       list[Clarification] = field(default_factory=list)
    enriched_goal:        str   = ""
    enriched_constraints: str   = ""
    enriched_context:     str   = ""

    def summary(self) -> str:
        """Single-string summary for prompt injection."""
        lines = [f"Original input: {self.original_input}",
                 f"Depth: {self.depth}  |  Domain: {self.detected_domain}"]
        if self.clarifications:
            lines.append("Clarifications:")
            for c in self.clarifications:
                lines.append(f"  Q: {c.question}")
                lines.append(f"  A: {c.answer}")
        if self.enriched_goal:
            lines.append(f"Enriched goal: {self.enriched_goal}")
        if self.enriched_constraints:
            lines.append(f"Enriched constraints: {self.enriched_constraints}")
        if self.enriched_context:
            lines.append(f"Enriched context: {self.enriched_context}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# Domain detection (local copy — no import dependency)
# ─────────────────────────────────────────────────────────────────

def _detect_domain(product_idea: str) -> str:
    text = product_idea.lower()
    scores: dict[str, int] = {d: 0 for d in _DOMAIN_KEYWORDS}
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                scores[domain] += 1
    best = max(scores, key=scores.__getitem__)
    return best if scores[best] > 0 else "default"


# ─────────────────────────────────────────────────────────────────
# Depth scoring (heuristic, then LLM confirmation if available)
# ─────────────────────────────────────────────────────────────────

_TECHNICAL_SIGNALS = [
    r"\b(iso|iec|astm|ansi|mil-std|ecss|faa|easa|fda)\b",
    r"\b(asil|sil|ral|ip\d{2}|ik\d{2})\b",
    r"\b(kwh|wh|mah|v\b|a\b|hz|rpm|nm|mpa|gpa|db)\b",
    r"\b(protocol|api|firmware|redundancy|failsafe|bom|bms|can\s+bus|uart|spi|i2c)\b",
    r"\b(tolerance|clearance|creepage|isolation|emc|emi|esd)\b",
    r"\b(thermal|cooling|heat\s+dissipation|thermal\s+runaway)\b",
]

_VAGUE_SIGNALS = [
    r"^(a\s+)?(cool|nice|good|great|smart|innovative|modern)",
    r"\b(something|thing|stuff|device|gadget|product)\b",
    r"^[\w\s]{1,25}$",   # very short with no domain words
]


def _heuristic_depth(text: str) -> str:
    """Fast keyword-based depth estimate (no API call)."""
    low = text.lower()
    tech_hits = sum(1 for pat in _TECHNICAL_SIGNALS if re.search(pat, low))
    vague_hits = sum(1 for pat in _VAGUE_SIGNALS if re.search(pat, low))
    words = len(text.split())

    if tech_hits >= 2 or (tech_hits >= 1 and words >= 10):
        return "technical"
    if vague_hits >= 2 or words <= 4:
        return "vague"
    return "mixed"


def _llm_depth(product_idea: str) -> str:
    """Confirm depth with Haiku (fast + cheap). Returns technical|mixed|vague."""
    if not _HAS_ANTHROPIC:
        return _heuristic_depth(product_idea)
    prompt = (
        "Classify the specificity of this product idea as exactly one of: "
        "technical, mixed, vague.\n\n"
        "Rules:\n"
        "- technical: contains domain-specific terms, standards, metrics, or constraints\n"
        "- mixed: some specifics but key dimensions are unspecified\n"
        "- vague: generic description with no engineering detail\n\n"
        f"Product idea: \"{product_idea}\"\n\n"
        "Reply with a single word: technical, mixed, or vague."
    )
    try:
        rsp = _CLIENT.messages.create(
            model=_HAIKU,
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}],
        )
        word = rsp.content[0].text.strip().lower()
        if word in ("technical", "mixed", "vague"):
            return word
    except Exception:
        pass
    return _heuristic_depth(product_idea)


# ─────────────────────────────────────────────────────────────────
# Question selection
# ─────────────────────────────────────────────────────────────────

def _select_questions(domain: str, depth: str, n: int = 3) -> list[str]:
    """Pick the most relevant clarifying questions for this domain + depth."""
    bank = _DOMAIN_QUESTIONS.get(domain, _DOMAIN_QUESTIONS["default"])
    # For mixed depth, limit to 1–2 questions; vague gets up to n
    limit = n if depth == "vague" else min(n, 2) if depth == "mixed" else 0
    return bank[:limit]


# ─────────────────────────────────────────────────────────────────
# Intent synthesis
# ─────────────────────────────────────────────────────────────────

def _synthesize_intent(
    product_idea: str,
    domain: str,
    clarifications: list[Clarification],
    existing_goal: str = "",
    existing_constraints: str = "",
    existing_context: str = "",
) -> tuple[str, str, str]:
    """
    Merge original input + clarifications into enriched goal/constraints/context.
    Returns (enriched_goal, enriched_constraints, enriched_context).
    Falls back gracefully if no Anthropic client.
    """
    if not _HAS_ANTHROPIC:
        return existing_goal, existing_constraints, existing_context

    qa_block = ""
    if clarifications:
        qa_block = "\n".join(
            f"Q: {c.question}\nA: {c.answer}" for c in clarifications
        )

    prompt = f"""You are a product strategist. Synthesize the information below into:
1. A concise goal sentence (what the product must achieve)
2. A comma-separated list of hard constraints (measurable where possible)
3. A brief context sentence (user, environment, market)

Product idea: {product_idea}
Domain: {domain}
{"Existing goal: " + existing_goal if existing_goal else ""}
{"Existing constraints: " + existing_constraints if existing_constraints else ""}
{"Existing context: " + existing_context if existing_context else ""}
{"Clarifications:\n" + qa_block if qa_block else ""}

Reply in this exact format:
GOAL: <one sentence>
CONSTRAINTS: <comma-separated list>
CONTEXT: <one sentence>"""

    try:
        rsp = _CLIENT.messages.create(
            model=_HAIKU,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        text = rsp.content[0].text.strip()
        goal = constraints = context = ""
        for line in text.splitlines():
            if line.startswith("GOAL:"):
                goal = line[5:].strip()
            elif line.startswith("CONSTRAINTS:"):
                constraints = line[12:].strip()
            elif line.startswith("CONTEXT:"):
                context = line[8:].strip()
        return (
            goal        or existing_goal,
            constraints or existing_constraints,
            context     or existing_context,
        )
    except Exception:
        return existing_goal, existing_constraints, existing_context


# ─────────────────────────────────────────────────────────────────
# Main agent class
# ─────────────────────────────────────────────────────────────────

class IntentElicitationAgent:
    """
    Scores input depth, asks clarifying questions (interactive mode only),
    and enriches the intent for downstream agents.
    """

    def run(
        self,
        product_idea: str,
        existing_goal: str         = "",
        existing_constraints: str  = "",
        existing_context: str      = "",
        interactive: bool          = True,
    ) -> IntentContext:
        """
        Parameters
        ----------
        product_idea        : raw user input
        existing_goal       : --goal flag (may be empty)
        existing_constraints: --constraints flag (may be empty)
        existing_context    : --context flag (may be empty)
        interactive         : False skips all questions (GUI / non-interactive CLI)
        """
        domain = _detect_domain(product_idea)
        depth  = _llm_depth(product_idea)

        ctx = IntentContext(
            original_input  = product_idea,
            depth           = depth,
            detected_domain = domain,
        )

        # If goal already provided, treat as at least mixed
        if existing_goal and depth == "vague":
            depth = "mixed"
            ctx.depth = "mixed"

        # Ask clarifying questions only in interactive mode for vague/mixed
        if interactive and depth in ("vague", "mixed"):
            questions = _select_questions(domain, depth)
            for q in questions:
                try:
                    answer = input(f"\n  {q}\n  > ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if answer:
                    ctx.clarifications.append(Clarification(question=q, answer=answer))

        # Synthesize enriched intent
        ctx.enriched_goal, ctx.enriched_constraints, ctx.enriched_context = (
            _synthesize_intent(
                product_idea,
                domain,
                ctx.clarifications,
                existing_goal,
                existing_constraints,
                existing_context,
            )
        )

        return ctx


# ─────────────────────────────────────────────────────────────────
# Module-level singleton + wrapper
# ─────────────────────────────────────────────────────────────────

_agent: IntentElicitationAgent | None = None


def get_elicitation_agent() -> IntentElicitationAgent:
    global _agent
    if _agent is None:
        _agent = IntentElicitationAgent()
    return _agent


def intent_elicitation_agent(
    product_idea: str,
    existing_goal: str        = "",
    existing_constraints: str = "",
    existing_context: str     = "",
    interactive: bool         = True,
) -> IntentContext:
    """
    Convenience wrapper used by plm_agents.py.
    Returns an IntentContext; never raises — failures produce a minimal stub.
    """
    try:
        agent = get_elicitation_agent()
        return agent.run(
            product_idea        = product_idea,
            existing_goal       = existing_goal,
            existing_constraints= existing_constraints,
            existing_context    = existing_context,
            interactive         = interactive,
        )
    except Exception as exc:
        # Graceful fallback: return a minimal context with original input
        return IntentContext(
            original_input  = product_idea,
            depth           = "vague",
            detected_domain = _detect_domain(product_idea),
            enriched_goal        = existing_goal,
            enriched_constraints = existing_constraints,
            enriched_context     = existing_context,
        )
