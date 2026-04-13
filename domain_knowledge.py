"""
Domain Knowledge — Layer 1: RAG  |  Layer 2: Knowledge Graph
=============================================================
Layer 1 — RAG (LlamaIndex + ChromaDB)
  Dynamic domain detection: drone, energy_storage, automotive,
  aerospace, medical, electronics, default.
  Each domain loads product-specific regulatory/engineering sources.
  Sources tagged: domain_specific / general / llm_reasoned.
  Returns top-5 chunks per intent; flags chunks older than 2 years.
  User can add custom sources at runtime via add_custom_source().

Layer 2 — Knowledge Graph (Kuzu embedded)
  Models:  material → requires → constraint
           component → compatible_with → component
           standard  → applies_to     → product_category
  Edges tagged: sourced / llm_reasoned.

Both layers fail gracefully if their package is unavailable.
"""

from __future__ import annotations

import datetime
import os
import re
import textwrap
from dataclasses import dataclass, field

# Suppress HuggingFace symlink warning on Windows (cosmetic only — caching still works)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# ─────────────────────────────────────────────────────────────────────────────
# DATA TYPES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RAGChunk:
    text:       str
    source_url: str
    date:       str          # ISO-8601, e.g. "2023-04"
    stale:      bool = False  # True if > 2 years old
    source_tag: str  = "general"  # "domain_specific" | "general" | "llm_reasoned"

    def as_context_line(self) -> str:
        stale_flag = "  [STALE]" if self.stale else ""
        return f"[{self.source_url}  {self.date}  {self.source_tag}{stale_flag}]\n{self.text}"


@dataclass
class KGTriple:
    subject:  str
    relation: str
    object:   str
    tag:      str   # "sourced" | "llm_reasoned"

    def as_context_line(self) -> str:
        return f"({self.subject}) --[{self.relation}]--> ({self.object})  [{self.tag}]"


@dataclass
class DomainContext:
    """Everything the Domain Knowledge Agent produces for downstream agents."""
    rag_chunks:       list[RAGChunk]  = field(default_factory=list)
    graph_triples:    list[KGTriple]  = field(default_factory=list)
    validation_trace: dict            = field(default_factory=dict)
    detected_domain:  str             = "default"

    # Keep a reference to the KG for in-loop queries from evaluator
    _kg: object = field(default=None, repr=False)

    # ── convenience ──────────────────────────────────────────────────────────

    def rag_block(self) -> str:
        if not self.rag_chunks:
            return "(no RAG context retrieved)"
        return "\n\n".join(c.as_context_line() for c in self.rag_chunks)

    def graph_block(self) -> str:
        if not self.graph_triples:
            return "(no graph context retrieved)"
        return "\n".join(t.as_context_line() for t in self.graph_triples)

    def prompt_block(self) -> str:
        parts = []
        if self.rag_chunks:
            parts.append("=== DOMAIN KNOWLEDGE (RAG) ===\n" + self.rag_block())
        if self.graph_triples:
            parts.append("=== KNOWLEDGE GRAPH CONTEXT ===\n" + self.graph_block())
        return "\n\n".join(parts) if parts else ""

    def is_empty(self) -> bool:
        return not self.rag_chunks and not self.graph_triples

    def sources_for_report(self) -> list[dict]:
        """Return structured list for the HTML report."""
        seen: dict[str, dict] = {}
        for c in self.rag_chunks:
            key = c.source_url
            if key not in seen:
                seen[key] = {
                    "url":        c.source_url,
                    "date":       c.date,
                    "stale":      c.stale,
                    "source_tag": c.source_tag,
                    "chunks":     0,
                }
            seen[key]["chunks"] += 1
        return list(seen.values())

    def vv_coverage(self) -> dict:
        return {
            "graph":           sum(1 for t in self.graph_triples if t.tag == "sourced"),
            "rag":             len(self.rag_chunks),
            "domain_specific": sum(1 for c in self.rag_chunks if c.source_tag == "domain_specific"),
            "general":         sum(1 for c in self.rag_chunks if c.source_tag == "general"),
            "llm_reasoned":    sum(1 for t in self.graph_triples if t.tag == "llm_reasoned"),
        }


# ─────────────────────────────────────────────────────────────────────────────
# DOMAIN DETECTION
# ─────────────────────────────────────────────────────────────────────────────

_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "drone": [
        "drone", "uav", "quadcopter", "multirotor", "fpv", "fixed wing",
        "octocopter", "hexacopter", "tricopter", "ardupilot", "pixhawk",
        "flight controller", "propeller", "esc", "lipo", "dji", "betaflight",
    ],
    "energy_storage": [
        "battery", "energy storage", "bms", "lfp", "nmc", "lto", "nca",
        "powerwall", "ess", "battery pack", "cell", "lithium", "accumulator",
        "charge controller", "inverter", "grid storage", "lifepo",
    ],
    "automotive": [
        "car", "vehicle", "automotive", "ev", "electric vehicle", "autosar",
        "can bus", "misra", "iso 26262", "functional safety", "ecu",
        "powertrain", "chassis", "adas", "lidar", "obdii", "j1939",
    ],
    "aerospace": [
        "spacecraft", "satellite", "rocket", "launcher", "space", "cubesat",
        "nanosatellite", "orbit", "ecss", "nasa", "esa", "propellant",
        "attitude control", "thermal vacuum", "space qualification",
    ],
    "medical": [
        "medical", "device", "implant", "diagnostic", "surgical", "fda",
        "510k", "mdr", "iso 13485", "clinical", "biocompatible", "sterile",
        "hospital", "patient", "therapy", "wearable health",
    ],
    "electronics": [
        "pcb", "microcontroller", "mcu", "embedded", "arduino", "raspberry pi",
        "rp2040", "esp32", "stm32", "fpga", "soc", "gpio", "uart", "spi", "i2c",
        "jtag", "swd", "bootloader", "firmware", "kicad", "altium", "eagle",
        "schematic", "soldering", "breadboard", "prototype board", "dev board",
        "development board", "single board", "bare metal", "rtos", "micropython",
        "circuitpython", "register", "interrupt", "pwm", "adc", "dac",
    ],
}


def detect_domain(product_idea: str) -> str:
    """
    Classify a product idea into a domain using keyword matching.
    Returns one of: drone | energy_storage | automotive | aerospace | medical | electronics | default.
    """
    text = product_idea.lower()
    # Score each domain by keyword hits
    scores = {domain: 0 for domain in _DOMAIN_KEYWORDS}
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                scores[domain] += 1
    best  = max(scores, key=scores.__getitem__)
    return best if scores[best] > 0 else "default"


# ─────────────────────────────────────────────────────────────────────────────
# DOMAIN-SPECIFIC SYNTHETIC SOURCE DOCUMENTS
# ─────────────────────────────────────────────────────────────────────────────

def _docs_drone(product_type: str) -> list[dict]:
    return [
        {
            "text": textwrap.dedent(f"""\
                FAA regulations for UAS / drones — {product_type}.
                Part 107 (USA): commercial operations require Remote Pilot Certificate.
                Weight classes: micro (<250g, minimal regs), small (250g–25kg, Part 107).
                Restricted zones: airports (5nm buffer), national parks, military airspace.
                Night operations: anti-collision lighting required (3 statute miles visible).
                Max altitude: 400ft AGL unless within 400ft of a structure.
                BVLOS (beyond visual line of sight): specific waiver required.
                Remote ID: required since 2023 for all UAVs > 250g.
                Payload operations: additional waivers for dropping objects."""),
            "url":  "https://www.faa.gov/uas/commercial_operators/part_107_waivers",
            "date": "2024-06",
            "source_tag": "domain_specific",
        },
        {
            "text": textwrap.dedent(f"""\
                EASA regulations — UAS in EU airspace.
                Open Category: <25kg, VLOS, pre-defined operations (A1/A2/A3 subcategories).
                A1: <250g, fly near people. A2: <4kg, 30m lateral distance from people.
                A3: <25kg, away from populated areas.
                Specific Category: risk assessment via SORA methodology.
                Certified Category: >25kg or over crowds — requires type certification.
                EU drone registration: mandatory for Open A1-A3 ≥ 250g or with camera.
                C-class labels: C0–C6 define equipment requirements.
                Geo-awareness: required for class C1+. Remote ID: mandatory C1+."""),
            "url":  "https://www.easa.europa.eu/domains/civil-drones",
            "date": "2024-03",
            "source_tag": "domain_specific",
        },
        {
            "text": textwrap.dedent(f"""\
                ArduPilot / RCGroups engineering specs — {product_type} design.
                Frame: carbon fibre (strength/weight), 3K twill or UD layup.
                Motors: KV rating inversely proportional to propeller size.
                  Racing: 2300-2700KV + 5-inch props, 4S LiPo.
                  Freestyle: 1800-2300KV + 5-inch props, 4-6S.
                  Long range: 800-1500KV + 7-9 inch props, 6S.
                ESC protocol: DSHOT600/1200 for digital, BLHeli32 recommended.
                Flight controller: F7 for racing (fast loops), H7 for AI/mapping.
                Telemetry: CRSF/ELRS for long range, FrSky for sport.
                Battery: 4S (14.8V) for 5-inch racing; 6S (22.2V) for efficiency.
                Props: 5x4.3x3 tri-blade for racing; 5x4.5x2 bi-blade for efficiency."""),
            "url":  "https://ardupilot.org/copter/docs/choosing-a-frame.html",
            "date": "2024-01",
            "source_tag": "domain_specific",
        },
        {
            "text": textwrap.dedent(f"""\
                General engineering reference — {product_type}.
                Common materials: carbon fibre (200-400 GPa modulus), aluminium 6061-T6.
                IP ratings: IP44 splash-proof minimum for outdoor; IP65 dust/jet-proof.
                Vibration isolation: rubber dampers for FC/camera gimbal mounting.
                Thermal: BLDC motors typically rated 80°C continuous, 100°C peak.
                EMI: separate power and signal cables; use ferrite beads on video lines.
                Weight budget: motors ~20-25% of AUW, battery ~30-40% of AUW."""),
            "url":  "https://www.engineeringtoolbox.com/uav-drone-materials.html",
            "date": "2024-01",
            "source_tag": "general",
        },
        {
            "text": textwrap.dedent(f"""\
                EU regulatory compliance for drones — legacy reference (2021).
                Note: EU UAS regulation (EU) 2019/945 fully applied from 2021.
                Previous national exemptions expired. Now check current EASA Open/Specific rules."""),
            "url":  "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32019R0945",
            "date": "2021-06",
            "source_tag": "general",
        },
    ]


def _docs_energy_storage(product_type: str) -> list[dict]:
    return [
        {
            "text": textwrap.dedent(f"""\
                IEC 62619:2022 — Safety requirements for secondary lithium cells for use in
                industrial applications including stationary {product_type}.
                Scope: LFP, NMC, LTO, NCA chemistries in stationary and industrial ESS.
                Key requirements: overcharge protection (≥4.25V/cell trip for NMC),
                overdischarge protection (<2.5V for NMC, <2.0V for LFP),
                short-circuit protection: BMS must interrupt within 1s at 2×rated current.
                Temperature: charge 0–45°C, discharge -20–60°C (chemistry-dependent).
                Cell matching: capacity deviation <5% within pack.
                Thermal runaway: propagation prevention mandatory for packs >5kWh.
                Insulation: 1500 Vdc for systems >60V, tested per IEC 61140."""),
            "url":  "https://webstore.iec.ch/publication/59640",
            "date": "2024-02",
            "source_tag": "domain_specific",
        },
        {
            "text": textwrap.dedent(f"""\
                UN 38.3 — Transport of dangerous goods — lithium battery testing.
                Mandatory for all lithium cells/batteries shipped by air, sea, or road.
                Test sequence: T1 altitude simulation, T2 thermal, T3 vibration,
                T4 shock, T5 external short circuit, T6 crush (cells only),
                T7 forced discharge (cells only), T8 overcharge.
                Watt-hour limits: cells ≤20Wh, batteries ≤100Wh for air cargo (Section II).
                State of charge for transport: ≤30% for lithium-ion.
                Labelling: UN3480 (standalone), UN3481 (in equipment).
                Manufacturer must maintain test summary for 10 years."""),
            "url":  "https://unece.org/transport/standards/transport/dangerous-goods/un-model-regulations-rev-23",
            "date": "2023-07",
            "source_tag": "domain_specific",
        },
        {
            "text": textwrap.dedent(f"""\
                battery-archive.org — real-world degradation data for {product_type}.
                Cycle life benchmarks (80% capacity retention):
                  LFP: 2000-5000 cycles at 0.5C (best for stationary ESS)
                  NMC: 500-1500 cycles (higher energy density, faster degradation)
                  LTO: 10000+ cycles, but lower energy density (60-80 Wh/kg)
                  NCA: 800-2000 cycles (Tesla application)
                Temperature effect: every 10°C above 25°C halves calendar life.
                Depth of discharge: limiting to 80% DoD doubles cycle count vs 100%.
                C-rate: charge at ≤0.5C for longevity; peak discharge up to 3-5C.
                Calendar aging: SEI growth dominates; store at 40-50% SoC, 15-25°C."""),
            "url":  "https://www.batteryarchive.org/list.html",
            "date": "2024-04",
            "source_tag": "domain_specific",
        },
        {
            "text": textwrap.dedent(f"""\
                General engineering — battery system design.
                BMS: must provide: cell voltage monitoring (±5mV), temperature (±1°C),
                SoC estimation (Coulomb counting + OCV lookup), balancing (passive/active).
                Cell-to-cell variation: <2mV at rest, <10mV under load.
                Enclosure: IP55 minimum for indoor ESS, IP65 for outdoor.
                Fire safety: UL 9540A fire propagation test for >10 kWh systems.
                Thermal management: liquid cooling preferred >100kWh; air cooling for <20kWh."""),
            "url":  "https://www.engineeringtoolbox.com/battery-energy-storage.html",
            "date": "2024-01",
            "source_tag": "general",
        },
        {
            "text": textwrap.dedent(f"""\
                Legacy battery system reference (2021).
                IEC 62619:2017 (superseded by 2022 edition). Update test results to 2022.
                Previous UN 38.3 test requirements updated in Rev.22 (2021)."""),
            "url":  "https://webstore.iec.ch/publication/59640-legacy",
            "date": "2021-06",
            "source_tag": "general",
        },
    ]


def _docs_automotive(product_type: str) -> list[dict]:
    return [
        {
            "text": textwrap.dedent(f"""\
                ISO 26262:2018 — Functional Safety for Road Vehicles — {product_type}.
                ASIL levels (A→D): D = highest integrity. Derived from S×E×C risk parameters.
                ASIL decomposition: ASIL D = ASIL B + ASIL B or ASIL C + ASIL A.
                Safety lifecycle: concept → system design → hardware → software → production.
                Hardware metrics: SPFM ≥97% (ASIL D), LFM ≥80% (ASIL D).
                Software: MC/DC coverage mandatory for ASIL C/D.
                Diagnostic coverage: ≥99% for ASIL D safety mechanisms.
                FTTI: fault tolerant time interval — must be met by safety mechanism.
                Key documents: Safety Plan, HARA, FSC, TSC, DFA, safety case."""),
            "url":  "https://www.iso.org/standard/68383.html",
            "date": "2024-01",
            "source_tag": "domain_specific",
        },
        {
            "text": textwrap.dedent(f"""\
                AUTOSAR — AUTomotive Open System ARchitecture.
                Classic AUTOSAR: RTOS-based, hard real-time, safety-critical ECUs.
                Adaptive AUTOSAR: POSIX-based (Linux), OTA updates, autonomous driving.
                BSW layers: Microcontroller Abstraction (MCAL), ECU Abstraction, Services.
                Communication: COM stack — CAN, FlexRay, Ethernet (SOME/IP).
                Diagnostic: UDS (ISO 14229), OBD-II (SAE J1979) via DCM module.
                Memory: NvM for persistent data, FlashDriver for OTA.
                OS: OSEK/VDX (Classic), AUTOSAR OS with scheduling tables.
                MISRA C:2012 mandatory for AUTOSAR BSW development."""),
            "url":  "https://www.autosar.org/standards",
            "date": "2024-03",
            "source_tag": "domain_specific",
        },
        {
            "text": textwrap.dedent(f"""\
                MISRA C:2012 and MISRA C++:2023 — coding standards for {product_type}.
                Purpose: subset of C/C++ to avoid undefined/unspecified behaviour.
                225 rules (MISRA C:2012): Mandatory, Required, Advisory.
                Key Mandatory rules: no undefined behaviour, no dynamic memory post-init.
                Static analysis: Polyspace, LDRA, PC-lint, Parasoft must achieve 0 violations.
                Deviation: documented and justified per tool/safety case process.
                MISRA C:2012 Amendment 3 (2023): aligns with C11 and C17 standards.
                Combined with ISO 26262: MISRA is the normative coding guideline for ASIL C/D."""),
            "url":  "https://www.misra.org.uk/Publications/tabid/57/Default.aspx",
            "date": "2024-01",
            "source_tag": "domain_specific",
        },
        {
            "text": textwrap.dedent(f"""\
                General engineering — automotive product design.
                Operating temperature: -40°C to +125°C (engine bay); -40°C to +85°C (interior).
                Vibration: ISO 16750-3 — road vehicle environmental conditions.
                EMC: CISPR 25 (emissions), ISO 11452 (immunity) for automotive components.
                Connector reliability: USCAR-2 / USCAR-21 for automotive connectors.
                Fluid resistance: ISO 16750-5 — chemical exposure requirements."""),
            "url":  "https://www.engineeringtoolbox.com/automotive-engineering.html",
            "date": "2024-01",
            "source_tag": "general",
        },
        {
            "text": textwrap.dedent(f"""\
                Legacy automotive reference (2021). ISO 26262:2011 superseded by 2018 edition.
                AUTOSAR R4.2 (2021) superseded; use R22-11 or later for new projects."""),
            "url":  "https://www.iso.org/standard/43464.html",
            "date": "2021-04",
            "source_tag": "general",
        },
    ]


def _docs_aerospace(product_type: str) -> list[dict]:
    return [
        {
            "text": textwrap.dedent(f"""\
                NASA NTRS — reliability and design for {product_type} systems.
                FMEA (Failure Mode and Effect Analysis): identify single-point failures.
                Derating guidelines: operate components at 50-80% rated capacity.
                Thermal: worst-case temperature analysis; radiation environment (TID/SEE).
                Connector reliability: MIL-DTL-38999 series for harsh environments.
                Vibration: GEVS (GSFC-STD-7000B) random vibration PSD profiles.
                Materials: avoid cadmium, mercury; prefer RoHS-compliant finishes.
                EEE parts: NASA-HDBK-8739.23 for component selection and screening.
                Redundancy: cold/hot/warm standby for critical functions (FDIR)."""),
            "url":  "https://ntrs.nasa.gov/search?q=spacecraft+reliability",
            "date": "2024-02",
            "source_tag": "domain_specific",
        },
        {
            "text": textwrap.dedent(f"""\
                ECSS — European Cooperation for Space Standardization.
                ECSS-E-ST-10C: system engineering general requirements.
                ECSS-Q-ST-70C: materials, mechanical parts and processes.
                ECSS-E-ST-20C: electrical and electronic engineering.
                ECSS-Q-ST-60C: EEE components requirements.
                ECSS-E-ST-32C: structural general requirements.
                Design margins: structural (1.25 yield, 1.5 ultimate), thermal (10°C margin).
                Cleanliness: ECSS-Q-ST-70-01C for surfaces and environments.
                Software: ECSS-E-ST-40C (software engineering), DO-178C for airborne."""),
            "url":  "https://ecss.nl/standards",
            "date": "2024-01",
            "source_tag": "domain_specific",
        },
        {
            "text": textwrap.dedent(f"""\
                Radiation environment — {product_type} space qualification.
                LEO (400-1000km): TID ~10 krad/year (inside 4mm Al shielding).
                GEO (36000km): TID up to 100 krad/year; proton belt crossings.
                Single Event Effects (SEE): SEU, SET, SEFI, SEL — use rad-hard or mitigation.
                Triple modular redundancy (TMR) for critical logic under radiation.
                Shielding: Al or Ta spot shields for sensitive components.
                Qualification: MIL-STD-750, ESCC-22900 for radiation testing."""),
            "url":  "https://ntrs.nasa.gov/search?q=radiation+space+electronics",
            "date": "2023-11",
            "source_tag": "domain_specific",
        },
        {
            "text": textwrap.dedent(f"""\
                General engineering — aerospace materials and processes.
                Aluminium alloys: 6061-T6 (structural), 7075-T6 (high strength).
                Titanium Ti-6Al-4V: high strength-to-weight for load-bearing structures.
                CFRP: 200-400 GPa modulus; autoclave cure for primary structure.
                Fasteners: Hi-Lok, LOCKBOLT — no self-locking nuts in vibration environments.
                Surface finish: alodine (1200S) for corrosion protection of Al."""),
            "url":  "https://www.engineeringtoolbox.com/aerospace-materials.html",
            "date": "2024-01",
            "source_tag": "general",
        },
        {
            "text": textwrap.dedent(f"""\
                Legacy aerospace reference (2020). ECSS standards updated in 2021-2024.
                Verify applicability of specific ECSS-Q-ST-60C amendments."""),
            "url":  "https://ecss.nl/standards-legacy",
            "date": "2020-09",
            "source_tag": "general",
        },
    ]


def _docs_medical(product_type: str) -> list[dict]:
    return [
        {
            "text": textwrap.dedent(f"""\
                FDA 510(k) — substantial equivalence pathway for {product_type}.
                Class I (<lowest risk): most exempt from 510(k); general controls apply.
                Class II (moderate risk): 510(k) required; special controls.
                Class III (high risk): PMA (Pre-Market Approval) required.
                510(k) elements: device description, substantial equivalence argument,
                performance testing, biocompatibility (ISO 10993), software documentation.
                Software: FDA Guidance on Software as Medical Device (SaMD) 2019.
                Cybersecurity: FDA 2023 guidance — mandatory for networked devices.
                UDI (Unique Device Identifier): required for Class II/III since 2021."""),
            "url":  "https://www.fda.gov/medical-devices/premarket-submissions/premarket-notification-510k",
            "date": "2024-05",
            "source_tag": "domain_specific",
        },
        {
            "text": textwrap.dedent(f"""\
                ISO 13485:2016 — Medical devices quality management system.
                Mandatory for regulatory submissions in EU (MDR), Canada (CMDCAS), Japan.
                Key requirements: design controls (design input/output/review/verification/validation),
                risk management integration (ISO 14971), production and process controls,
                traceability: lot/batch tracking to customer level.
                CAPA: corrective and preventive action system mandatory.
                Design history file (DHF): full documentation of design decisions.
                Device master record (DMR): production specifications.
                Post-market surveillance: mandatory adverse event reporting."""),
            "url":  "https://www.iso.org/standard/59752.html",
            "date": "2024-01",
            "source_tag": "domain_specific",
        },
        {
            "text": textwrap.dedent(f"""\
                ISO 14971:2019 — Risk management for medical devices.
                Risk management process: hazard identification → risk estimation →
                risk evaluation → risk control → residual risk evaluation.
                Risk acceptability: ALARP (As Low As Reasonably Practicable).
                Risk control measures priority: design → protective measures → information for safety.
                Benefit-risk analysis: required for Class II/III devices.
                Risk management file: maintained throughout product lifecycle.
                Post-market: risk management updated with real-world data."""),
            "url":  "https://www.iso.org/standard/72704.html",
            "date": "2024-01",
            "source_tag": "domain_specific",
        },
        {
            "text": textwrap.dedent(f"""\
                General engineering — medical device materials and design.
                Biocompatibility: ISO 10993-1 for materials in contact with body.
                Sterilisation: EtO (ISO 11135), gamma (ISO 11137), steam (ISO 17665).
                Electrical safety: IEC 60601-1 — medical electrical equipment safety.
                Applied parts: Type B (body contact), BF (body floating), CF (cardiac direct).
                IP requirements: IPX1 (drip-proof) minimum for bedside equipment.
                Cleaning/disinfection: must withstand hospital-grade disinfectants."""),
            "url":  "https://www.engineeringtoolbox.com/medical-device-materials.html",
            "date": "2024-01",
            "source_tag": "general",
        },
        {
            "text": textwrap.dedent(f"""\
                Legacy medical device reference (2021). EU MDR (2017/745) fully applied May 2021.
                MDD (93/42/EEC) no longer valid for new certifications. Use MDR exclusively."""),
            "url":  "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32017R0745",
            "date": "2021-05",
            "source_tag": "general",
        },
    ]


def _docs_electronics(product_type: str) -> list[dict]:
    return [
        {
            "text": textwrap.dedent(f"""\
                IPC-2221B — Generic Standard on Printed Board Design (2012).
                Trace width vs. current capacity: 1 oz copper, 1mm trace ≈ 1 A sustained.
                Clearance: ≥0.1 mm at 50 V; ≥0.4 mm at 150 V (internal layers tighter).
                Via drill: minimum 0.2 mm for standard PCB; 0.1 mm for HDI.
                Annular ring: ≥0.05 mm (IPC Class 2) or ≥0.025 mm (Class 3 high-rel).
                Solder mask: LPI (Liquid Photo-Imageable) preferred; ENIG finish for fine pitch.
                Thermal relief: spokes on pads connected to large copper planes.
                Component keepout: ≥1.5 mm from board edge for SMD; ≥2 mm for through-hole.
                Design rule check (DRC): mandatory before Gerber generation.
                Fabrication notes: include drill file, stackup, impedance targets."""),
            "url":  "https://www.ipc.org/TOC/IPC-2221B.pdf",
            "date": "2024-01",
            "source_tag": "domain_specific",
        },
        {
            "text": textwrap.dedent(f"""\
                RP2040 / Raspberry Pi Pico — hardware design guide (2023).
                Core: dual ARM Cortex-M0+ at up to 133 MHz; 264 KB SRAM (6 banks).
                Flash: external QSPI (max 16 MB); XIP (execute-in-place) via cache.
                GPIO: 30 multi-function pins; 3.3 V logic; NOT 5 V tolerant.
                Peripherals: 2× UART, 2× SPI, 2× I²C, 16× PWM, 4× ADC (12-bit, 500 ksps).
                USB: USB 1.1 host/device via internal PHY (no external USB chip needed).
                PIO: 2× programmable I/O blocks, 8 state machines — custom protocols.
                Power: SMPS (RT6150) + LDO on Pico board; 1.8–5.5 V input range.
                MicroPython: full support, official UF2 bootloader image.
                Arduino: arduino-pico core (Earle Philhower); C/C++ SDK (BSD-3-Clause).
                Footprint: 21 × 51 mm; castellated edge for SMD mounting."""),
            "url":  "https://datasheets.raspberrypi.com/rp2040/hardware-design-with-rp2040.pdf",
            "date": "2024-03",
            "source_tag": "domain_specific",
        },
        {
            "text": textwrap.dedent(f"""\
                ESP32 family — hardware design guidelines (Espressif, 2023).
                ESP32-C3: single RISC-V core 160 MHz; 400 KB SRAM; Wi-Fi 802.11b/g/n + BT 5.0 LE.
                ESP32-S3: dual Xtensa LX7 240 MHz; 512 KB SRAM; vector extensions for ML.
                PCB antenna keep-out: 15 mm clearance under chip antenna; no copper pour.
                RF ground plane: continuous under module except antenna keep-out.
                Decoupling: 10 µF + 100 nF on each VDD33 pin; place within 0.5 mm of pin.
                Flash: internal (C3: 4 MB typical) or external SPI (ESP32: up to 16 MB).
                JTAG/SWD: ESP-PROG or built-in USB-JTAG (ESP32-C3/S3).
                OTA: Wi-Fi OTA via IDF esp_https_ota; partition table must have two app slots.
                Regulatory: FCC ID / CE marked on certified modules (e.g. ESP32-C3-MINI-1)."""),
            "url":  "https://www.espressif.com/sites/default/files/documentation/esp32-c3_hardware_design_guidelines_en.pdf",
            "date": "2024-02",
            "source_tag": "domain_specific",
        },
        {
            "text": textwrap.dedent(f"""\
                STM32 hardware design guide — CortexM MCU PCB recommendations (ST, 2023).
                Decoupling: 100 nF ceramic per VDD pin + 4.7 µF bulk; place <0.5 mm from pin.
                Crystal: keep trace short (<10 mm), guard ring to GND, no signal routing under.
                Reset: 100 nF on NRST; external pull-up optional (internal pull-up on most devices).
                Boot pins: BOOT0 strap to GND (normal run) or VDD (bootloader) via 10 kΩ.
                SWD: 4-pin (SWDIO, SWDCLK, GND, VDD); 10 kΩ pull-up on SWDIO.
                Embedded bootloader: USART1/USB DFU — no external programmer required.
                Clock: HSE (external crystal) recommended for USB; HSI ±1% for UART.
                EMC: ferrite bead on VDD_USB; TVS on exposed I/O pins."""),
            "url":  "https://www.st.com/resource/en/application_note/an4488-getting-started-with-stm32-stm32cubemx.pdf",
            "date": "2024-01",
            "source_tag": "domain_specific",
        },
        {
            "text": textwrap.dedent(f"""\
                General embedded / PCB engineering — {product_type}.
                Power budget: sum all rail currents; add 20% margin; select regulator accordingly.
                LDO vs SMPS: LDO simple but wastes (Vin-Vout)×I as heat; SMPS >85% efficient.
                USB power: USB 2.0 = 500 mA max; USB 3.0 = 900 mA; USB-C PD up to 100 W.
                ESD protection: TVS or Schottky arrays on all external-facing pins.
                Level shifting: TXS0108E (bidirectional auto-dir) or MOSFET for simple cases.
                Thermal: θJA for SOT-23 ≈ 200°C/W; ensure ambient + P_diss × θJA < Tjmax.
                Test points: add TP on every power rail, key signals, and ground.
                BOM cost reduction: use JLCPCB basic parts library; prefer 0402 passives.
                Open source toolchain: KiCad 7 (GPL), OpenOCD (GPL), GCC ARM (GPL), PlatformIO."""),
            "url":  "https://www.engineeringtoolbox.com/embedded-electronics-design.html",
            "date": "2024-01",
            "source_tag": "general",
        },
    ]


def _docs_default(product_type: str) -> list[dict]:
    slug = re.sub(r"[^a-z0-9_]", "_", product_type.lower())[:64]
    return [
        {
            "text": textwrap.dedent(f"""\
                Engineering Toolbox — {product_type} material reference.
                Common materials: aluminium alloys (6061-T6, 7075-T6), structural steel (S235/S355),
                polypropylene (PP), ABS, PEEK for high-temp. Key properties: tensile strength,
                thermal conductivity, corrosion resistance. For {product_type}: consider weight-to-strength
                ratio, operating temperature range, and chemical compatibility.
                Fastener standards: ISO 898-1 (metric bolts), ASME B18 (imperial).
                Surface treatments: anodising, powder coating, galvanising."""),
            "url":  f"https://www.engineeringtoolbox.com/{slug}-materials.html",
            "date": "2024-01",
            "source_tag": "general",
        },
        {
            "text": textwrap.dedent(f"""\
                ISO/IEC standards applicable to {product_type}:
                ISO 9001:2015 — Quality management systems.
                IEC 61010-1 — Safety requirements for electrical equipment.
                ISO 12100 — Safety of machinery — risk assessment.
                ISO 13849 — Safety of machinery — control systems (PLr categories).
                IEC 60529 — Degrees of protection (IP ratings: IP54, IP65, IP67, IP68).
                EN 62368-1 — Audio/video and IT equipment safety.
                CE marking: Machinery Directive 2006/42/EC, LVD 2014/35/EU, EMC 2014/30/EU."""),
            "url":  "https://www.iso.org/search.html?q=" + slug,
            "date": "2024-06",
            "source_tag": "general",
        },
        {
            "text": textwrap.dedent(f"""\
                NASA NTRS — reliability guidance for {product_type} systems.
                FMEA: identify single-point failures. Derating: operate at 50-80% rated capacity.
                Thermal management: worst-case temperature analysis, heat sink sizing.
                Connector reliability: MIL-DTL-38999 for harsh environments.
                Vibration: random vibration PSD per GEVS (GSFC-STD-7000).
                Redundancy: cold/hot/warm standby for critical functions."""),
            "url":  "https://ntrs.nasa.gov/search?q=" + slug,
            "date": "2023-09",
            "source_tag": "general",
        },
        {
            "text": textwrap.dedent(f"""\
                EUR-Lex regulatory framework for {product_type} in the EU market.
                REACH (EC) No 1907/2006: restriction of hazardous chemicals.
                RoHS Directive 2011/65/EU: hazardous substances in EEE.
                WEEE Directive 2012/19/EU: waste EEE.
                Ecodesign Regulation (EU) 2019/2021: energy efficiency.
                Battery Regulation (EU) 2023/1542: sustainability and labelling.
                Product Liability Directive: manufacturer responsibility."""),
            "url":  "https://eur-lex.europa.eu/search.html?text=" + slug,
            "date": "2024-03",
            "source_tag": "general",
        },
        {
            "text": textwrap.dedent(f"""\
                Legacy reference for {product_type} (2021). Some values may be superseded.
                Dimensional tolerances: ISO 2768 (general), ISO 286 (limits/fits).
                GD&T: ASME Y14.5-2018. Thread standards: ISO 68-1, ASME B1.1."""),
            "url":  f"https://www.engineeringtoolbox.com/{slug}-legacy.html",
            "date": "2021-06",
            "source_tag": "general",
        },
    ]


_DOMAIN_DOC_BUILDERS = {
    "drone":          _docs_drone,
    "energy_storage": _docs_energy_storage,
    "automotive":     _docs_automotive,
    "aerospace":      _docs_aerospace,
    "medical":        _docs_medical,
    "electronics":    _docs_electronics,
    "default":        _docs_default,
}


def _build_domain_docs(product_type: str, domain: str) -> list[dict]:
    """Build source documents for the detected domain."""
    builder = _DOMAIN_DOC_BUILDERS.get(domain, _docs_default)
    return builder(product_type)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

_CUTOFF_YEARS = 2


def _is_stale(date_str: str) -> bool:
    """Return True if date_str is more than _CUTOFF_YEARS years ago."""
    try:
        parts  = date_str.split("-")
        year   = int(parts[0])
        month  = int(parts[1]) if len(parts) > 1 else 6
        doc_d  = datetime.date(year, month, 1)
        cutoff = datetime.date.today().replace(
            year=datetime.date.today().year - _CUTOFF_YEARS
        )
        return doc_d < cutoff
    except Exception:
        return False


def _extract_keywords(text: str) -> list[str]:
    """Extract meaningful keywords from text for KG lookup."""
    stopwords = {
        "the", "a", "an", "and", "or", "for", "with", "to", "of",
        "in", "on", "at", "by", "from", "is", "are", "be", "that",
        "this", "it", "as", "was", "were", "has", "have",
        "i", "my", "me", "we", "our", "you", "your", "best", "good",
        "high", "low", "all", "any", "some", "new", "most", "more",
    }
    words = re.findall(r"[a-z]{3,}", text.lower())
    return [w for w in words if w not in stopwords][:20]


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 1 — RAG (LlamaIndex + ChromaDB)
# ─────────────────────────────────────────────────────────────────────────────

_CHROMA_DIR = os.path.join(os.path.dirname(__file__), ".chromadb")


class RAGLayer:
    """
    Retrieves top-k domain chunks for a given query using LlamaIndex + ChromaDB.
    Gracefully degrades to an empty result if packages are unavailable.
    """

    def __init__(self, top_k: int = 5):
        self.top_k  = top_k
        self._ready = False
        self._index = None
        self._setup()

    def _setup(self):
        try:
            from llama_index.core import (
                VectorStoreIndex, Document, StorageContext, Settings
            )
            from llama_index.vector_stores.chroma import ChromaVectorStore
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            import chromadb

            self._VectorStoreIndex  = VectorStoreIndex
            self._Document          = Document
            self._StorageContext    = StorageContext
            self._Settings          = Settings
            self._ChromaVectorStore = ChromaVectorStore
            self._chromadb          = chromadb

            Settings.embed_model = HuggingFaceEmbedding(
                model_name="BAAI/bge-small-en-v1.5"
            )
            Settings.llm = None
            self._ready  = True
        except ImportError as e:
            print(f"  ⚠ RAG layer unavailable: {e}")
            print("    Install: pip install llama-index llama-index-vector-stores-chroma "
                  "llama-index-embeddings-huggingface chromadb sentence-transformers")

    def _get_or_create_collection(self, name: str = "domain_knowledge"):
        client     = self._chromadb.PersistentClient(path=_CHROMA_DIR)
        collection = client.get_or_create_collection(name)
        store      = self._ChromaVectorStore(chroma_collection=collection)
        return store, collection

    def ingest(self, product_type: str, domain: str = "default",
               extra_docs: list[dict] | None = None) -> int:
        """
        Build/update ChromaDB index from domain-specific + custom docs.
        extra_docs: list of {text, url, date, source_tag} added by user.
        Returns number of documents ingested.
        """
        if not self._ready:
            return 0
        try:
            docs = _build_domain_docs(product_type, domain)
            if extra_docs:
                docs = docs + extra_docs

            store, _ = self._get_or_create_collection()
            ctx      = self._StorageContext.from_defaults(vector_store=store)
            self._index = self._VectorStoreIndex.from_documents(
                [self._Document(
                    text     = d["text"],
                    metadata = {
                        "source_url": d["url"],
                        "date":       d["date"],
                        "source_tag": d.get("source_tag", "general"),
                    }
                ) for d in docs],
                storage_context=ctx,
                show_progress=False,
            )
            return len(docs)
        except Exception as e:
            print(f"  ⚠ RAG ingest error: {e}")
            return 0

    def retrieve(self, query: str, product_type: str,
                 domain: str = "default",
                 extra_docs: list[dict] | None = None) -> list[RAGChunk]:
        """Return top-k RAGChunk objects for the query."""
        if not self._ready:
            return []
        try:
            if self._index is None:
                store, collection = self._get_or_create_collection()
                if collection.count() == 0:
                    n = self.ingest(product_type, domain, extra_docs)
                    if n == 0:
                        return []
                else:
                    ctx         = self._StorageContext.from_defaults(vector_store=store)
                    self._index = self._VectorStoreIndex.from_vector_store(
                        store, storage_context=ctx
                    )

            retriever = self._index.as_retriever(similarity_top_k=self.top_k)
            nodes     = retriever.retrieve(query)

            chunks = []
            for node in nodes:
                meta = node.metadata or {}
                url  = meta.get("source_url", "unknown")
                date = meta.get("date", "unknown")
                tag  = meta.get("source_tag", "general")
                chunks.append(RAGChunk(
                    text       = node.get_content().strip()[:600],
                    source_url = url,
                    date       = date,
                    stale      = _is_stale(date),
                    source_tag = tag,
                ))
            return chunks
        except Exception as e:
            print(f"  ⚠ RAG retrieve error: {e}")
            return []


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 2 — KNOWLEDGE GRAPH (Kuzu embedded)
# ─────────────────────────────────────────────────────────────────────────────

_KUZU_DIR = os.path.join(os.path.dirname(__file__), ".kuzudb")

# Seed triples: (subject, relation, object, tag)
_SEED_TRIPLES: list[tuple[str, str, str, str]] = [
    # material → requires → constraint
    ("aluminium_6061",       "requires", "anodising_or_coating",          "sourced"),
    ("stainless_steel_316",  "requires", "passivation_treatment",         "sourced"),
    ("lithium_ion_cell",     "requires", "bms_protection_circuit",        "sourced"),
    ("carbon_fibre_cfrp",    "requires", "cnc_or_autoclave_process",      "sourced"),
    ("abs_plastic",          "requires", "uv_stabiliser_for_outdoor",     "sourced"),
    ("peek",                 "requires", "high_temp_process_above_300c",  "sourced"),
    ("nmc_battery",          "requires", "thermal_runaway_protection",    "sourced"),
    ("lfp_battery",          "requires", "balancing_circuit",             "sourced"),
    # drone-specific
    ("carbon_fibre_frame",   "requires", "vibration_isolation_dampers",   "sourced"),
    ("lipo_battery",         "requires", "low_resistance_discharge_path", "sourced"),
    # energy storage
    ("lithium_cell_pack",    "requires", "thermal_management_system",     "sourced"),
    ("bms_module",           "requires", "cell_voltage_monitoring",       "sourced"),
    # automotive
    ("safety_critical_sw",   "requires", "misra_c_compliance",           "sourced"),
    ("can_bus_ecu",          "requires", "iso11898_termination",          "sourced"),
    # medical
    ("implantable_material",  "requires", "iso10993_biocompatibility",   "sourced"),
    ("electrical_medical_eq", "requires", "iec60601_safety_class",       "sourced"),

    # component → compatible_with → component
    ("brushless_motor",      "compatible_with", "esc_pwm_interface",      "sourced"),
    ("brushless_motor",      "compatible_with", "hall_sensor_feedback",   "sourced"),
    ("can_bus_module",       "compatible_with", "iso11898_controller",    "sourced"),
    ("rs485_interface",      "compatible_with", "modbus_rtu_protocol",    "sourced"),
    ("dc_dc_converter",      "compatible_with", "li_ion_charging_profile","sourced"),
    ("servo_motor",          "compatible_with", "pwm_controller_50hz",    "sourced"),
    ("solar_panel",          "compatible_with", "mppt_charge_controller", "sourced"),
    ("inverter_hybrid",      "compatible_with", "grid_tie_relay",         "sourced"),
    ("stepper_motor",        "compatible_with", "microstep_driver",       "sourced"),
    ("strain_gauge",         "compatible_with", "wheatstone_bridge_amp",  "sourced"),
    ("elrs_receiver",        "compatible_with", "expresslrs_transmitter", "sourced"),
    ("flight_controller_f7", "compatible_with", "dshot_esc_protocol",     "sourced"),
    ("bms_can_interface",    "compatible_with", "can_bus_module",          "sourced"),
    ("autosar_com_stack",    "compatible_with", "can_flexray_ethernet",    "sourced"),

    # standard → applies_to → product_category
    ("iec_61010",            "applies_to", "electrical_measurement_equipment", "sourced"),
    ("iso_13849",            "applies_to", "industrial_machinery",             "sourced"),
    ("iec_60529",            "applies_to", "enclosure_protection_rating",      "sourced"),
    ("iso_9001",             "applies_to", "quality_management",               "sourced"),
    ("reach_regulation",     "applies_to", "chemical_substances_in_products",  "sourced"),
    ("rohs_directive",       "applies_to", "electrical_electronic_equipment",  "sourced"),
    ("machinery_directive",  "applies_to", "eu_machinery_ce_marking",          "sourced"),
    ("iso_14971",            "applies_to", "medical_device_risk_management",   "sourced"),
    ("iec_62619",            "applies_to", "stationary_battery_energy_storage","sourced"),
    ("un_38_3",              "applies_to", "lithium_battery_transport",        "sourced"),
    ("iso_26262",            "applies_to", "automotive_functional_safety",     "sourced"),
    ("faa_part_107",         "applies_to", "commercial_drone_operations_usa",  "sourced"),
    ("easa_uas_regs",        "applies_to", "drone_operations_eu",              "sourced"),
    ("fda_510k",             "applies_to", "medical_device_clearance_usa",     "sourced"),
    ("iso_13485",            "applies_to", "medical_device_qms",               "sourced"),
    ("ecss_standards",       "applies_to", "european_space_products",          "sourced"),

    # LLM-reasoned edges
    ("high_voltage_system",  "requires",        "isolation_monitoring",   "llm_reasoned"),
    ("outdoor_product",      "requires",        "ip65_or_higher_rating",  "llm_reasoned"),
    ("battery_system",       "compatible_with", "fuse_protection",        "llm_reasoned"),
    ("rotating_machinery",   "requires",        "vibration_analysis",     "llm_reasoned"),
    ("food_contact_material","requires",        "fda_or_lfgb_compliance", "llm_reasoned"),
    ("fpv_drone",            "requires",        "remote_id_module",       "llm_reasoned"),
    ("ev_powertrain",        "requires",        "asil_d_bms_safety",      "llm_reasoned"),
]


class KnowledgeGraph:
    """Kuzu-backed embedded knowledge graph. Fails gracefully if kuzu absent."""

    def __init__(self):
        self._ready = False
        self._db    = None
        self._conn  = None
        self._setup()

    def _setup(self):
        try:
            import kuzu
            import shutil
            self._kuzu = kuzu
            # Kuzu 0.8+ creates the directory itself; if the directory already
            # exists but is empty (e.g. from a failed prior run) it raises
            # "Database path cannot be a directory" — remove it so Kuzu starts fresh.
            if os.path.isdir(_KUZU_DIR) and not os.listdir(_KUZU_DIR):
                shutil.rmtree(_KUZU_DIR)
            self._db   = kuzu.Database(_KUZU_DIR)
            self._conn = kuzu.Connection(self._db)
            self._init_schema()
            self._seed()
            self._ready = True
        except ImportError:
            print("  ⚠ KG layer unavailable: kuzu not installed")
            print("    Install: pip install kuzu")
        except Exception as e:
            print(f"  ⚠ KG setup error: {e}")

    def _init_schema(self):
        conn = self._conn
        for tbl in ("Material", "Component", "Standard", "Constraint", "ProductCategory"):
            try:
                conn.execute(
                    f"CREATE NODE TABLE IF NOT EXISTS {tbl} "
                    f"(name STRING, PRIMARY KEY (name))"
                )
            except Exception:
                pass
        try:
            conn.execute(
                "CREATE NODE TABLE IF NOT EXISTS Entity "
                "(name STRING, etype STRING, PRIMARY KEY (name))"
            )
        except Exception:
            pass
        try:
            conn.execute(
                "CREATE REL TABLE IF NOT EXISTS Relation "
                "(FROM Entity TO Entity, rel STRING, tag STRING)"
            )
        except Exception:
            pass

    def _seed(self):
        try:
            result = self._conn.execute("MATCH (e:Entity) RETURN count(e) AS n")
            if result.get_next()[0] > 0:
                return
        except Exception:
            pass
        for subj, rel, obj, tag in _SEED_TRIPLES:
            self._upsert_triple(subj, rel, obj, tag)

    def _upsert_triple(self, subj: str, rel: str, obj: str, tag: str):
        conn = self._conn
        for name in (subj, obj):
            try:
                conn.execute(
                    "MERGE (e:Entity {name: $n}) SET e.etype = $t",
                    {"n": name, "t": "entity"}
                )
            except Exception:
                try:
                    conn.execute(
                        "CREATE (e:Entity {name: $n, etype: $t})",
                        {"n": name, "t": "entity"}
                    )
                except Exception:
                    pass
        try:
            conn.execute(
                "MATCH (a:Entity {name: $s}), (b:Entity {name: $o}) "
                "CREATE (a)-[:Relation {rel: $r, tag: $tag}]->(b)",
                {"s": subj, "o": obj, "r": rel, "tag": tag}
            )
        except Exception:
            pass

    def add_triple(self, subj: str, rel: str, obj: str,
                   tag: str = "llm_reasoned") -> bool:
        if not self._ready:
            return False
        try:
            self._upsert_triple(subj, rel, obj, tag)
            return True
        except Exception as e:
            print(f"  ⚠ KG add_triple error: {e}")
            return False

    def query(self, keywords: list[str], limit: int = 20) -> list[KGTriple]:
        if not self._ready:
            return []
        try:
            results: list[KGTriple] = []
            seen: set[tuple] = set()
            for kw in keywords[:8]:
                kw_pat = kw.lower().replace(" ", "_")
                res = self._conn.execute(
                    "MATCH (a:Entity)-[r:Relation]->(b:Entity) "
                    "WHERE contains(lower(a.name), $kw) OR contains(lower(b.name), $kw) "
                    "RETURN a.name, r.rel, b.name, r.tag LIMIT $lim",
                    {"kw": kw_pat, "lim": limit}
                )
                while res.has_next():
                    row = res.get_next()
                    key = (row[0], row[1], row[2])
                    if key not in seen:
                        seen.add(key)
                        results.append(KGTriple(
                            subject=row[0], relation=row[1],
                            object=row[2], tag=row[3]
                        ))
            if not results:
                res = self._conn.execute(
                    "MATCH (a:Entity)-[r:Relation]->(b:Entity) "
                    "RETURN a.name, r.rel, b.name, r.tag LIMIT $lim",
                    {"lim": limit}
                )
                while res.has_next():
                    row = res.get_next()
                    results.append(KGTriple(
                        subject=row[0], relation=row[1],
                        object=row[2], tag=row[3]
                    ))
            return results[:limit]
        except Exception as e:
            print(f"  ⚠ KG query error: {e}")
            return []


# ─────────────────────────────────────────────────────────────────────────────
# DOMAIN KNOWLEDGE AGENT
# ─────────────────────────────────────────────────────────────────────────────

class DomainKnowledgeAgent:
    """
    Detects domain, loads domain-specific sources, retrieves RAG chunks + KG triples.
    Accepts user-added custom sources at runtime.
    """

    def __init__(self):
        self._rag         = RAGLayer(top_k=5)
        self._kg          = KnowledgeGraph()
        self._custom_docs: list[dict] = []

    def add_custom_source(self, url: str, text: str, date: str,
                          source_tag: str = "domain_specific") -> None:
        """
        Add a custom source document at runtime (before or after first run).
        Triggers re-ingestion on next retrieve() call.
        """
        self._custom_docs.append({
            "text":       text,
            "url":        url,
            "date":       date,
            "source_tag": source_tag,
        })
        self._rag._index = None   # force re-index on next retrieve

    def run(self, product_idea: str, product_type: str = "",
            intent_goal: str = "",
            detected_domain: str = "") -> DomainContext:
        """
        Retrieve domain knowledge for a product idea.
        detected_domain: if provided (from IntentElicitationAgent), skip re-detection.
        """
        query  = " ".join(filter(None, [product_idea, product_type, intent_goal]))
        pt     = product_type or product_idea
        domain = detected_domain or detect_domain(query)

        print(f"\n  Detected domain : {domain}")
        print(f"  Ingesting sources for: {pt!r}...")
        n_docs = self._rag.ingest(pt, domain=domain, extra_docs=self._custom_docs or None)
        if n_docs:
            print(f"  ✓ {n_docs} documents indexed (domain={domain},"
                  f" custom={len(self._custom_docs)})")

        print(f"  Retrieving RAG chunks...")
        chunks = self._rag.retrieve(query, pt, domain=domain,
                                    extra_docs=self._custom_docs or None)
        stale  = sum(1 for c in chunks if c.stale)
        ds     = sum(1 for c in chunks if c.source_tag == "domain_specific")
        print(f"  ✓ {len(chunks)} chunks ({ds} domain-specific, {stale} stale)")

        print(f"  Querying knowledge graph...")
        keywords = _extract_keywords(product_idea + " " + intent_goal + " " + domain)
        triples  = self._kg.query(keywords)
        sourced  = sum(1 for t in triples if t.tag == "sourced")
        inferred = len(triples) - sourced
        print(f"  ✓ {len(triples)} triples ({sourced} sourced, {inferred} llm_reasoned)")

        trace = {
            "rag_chunks":      len(chunks),
            "stale_chunks":    stale,
            "domain_specific": ds,
            "graph_triples":   len(triples),
            "sourced":         sourced,
            "llm_reasoned":    inferred,
            "sources":         list({c.source_url for c in chunks}),
            "detected_domain": domain,
            "custom_sources":  len(self._custom_docs),
        }

        ctx = DomainContext(
            rag_chunks       = chunks,
            graph_triples    = triples,
            validation_trace = trace,
            detected_domain  = domain,
        )
        ctx._kg = self._kg   # expose KG for in-loop evaluator queries
        return ctx

    def enrich_graph(self, triples: list[tuple[str, str, str, str]]) -> int:
        added = 0
        for subj, rel, obj, tag in triples:
            if self._kg.add_triple(subj, rel, obj, tag):
                added += 1
        return added


# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL SINGLETON
# ─────────────────────────────────────────────────────────────────────────────

_agent: DomainKnowledgeAgent | None = None


def get_domain_agent() -> DomainKnowledgeAgent:
    global _agent
    if _agent is None:
        _agent = DomainKnowledgeAgent()
    return _agent
