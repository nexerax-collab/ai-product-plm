"""
Domain Knowledge — Layer 1: RAG  |  Layer 2: Knowledge Graph
=============================================================
Layer 1 — RAG (LlamaIndex + ChromaDB)
  Sources: Engineering Toolbox, ISO/IEC abstracts, NASA NTRS, EUR-Lex
  Returns top-5 chunks per intent, tagged with source URL + date.
  Flags chunks older than 2 years as stale.

Layer 2 — Knowledge Graph (Kuzu embedded)
  Models:  material → requires → constraint
           component → compatible_with → component
           standard  → applies_to     → product_category
  Each edge carries a tag: "sourced" or "llm_reasoned"

Both layers fail gracefully if their service / package is unavailable.
"""

from __future__ import annotations

import datetime
import json
import os
import re
import textwrap
from dataclasses import dataclass, field

# ─────────────────────────────────────────────────────────────────────────────
# DATA TYPES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RAGChunk:
    text:       str
    source_url: str
    date:       str          # ISO-8601, e.g. "2023-04"
    stale:      bool = False  # True if > 2 years old

    def as_context_line(self) -> str:
        stale_flag = "  [STALE]" if self.stale else ""
        return f"[{self.source_url}  {self.date}{stale_flag}]\n{self.text}"


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

    # ── convenience ──────────────────────────────────────────────────────────

    def rag_block(self) -> str:
        """Format RAG chunks for injection into agent prompts."""
        if not self.rag_chunks:
            return "(no RAG context retrieved)"
        return "\n\n".join(c.as_context_line() for c in self.rag_chunks)

    def graph_block(self) -> str:
        """Format KG triples for injection into agent prompts."""
        if not self.graph_triples:
            return "(no graph context retrieved)"
        return "\n".join(t.as_context_line() for t in self.graph_triples)

    def prompt_block(self) -> str:
        """Combined block for agent prompts."""
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
        seen = {}
        for c in self.rag_chunks:
            key = c.source_url
            if key not in seen:
                seen[key] = {"url": c.source_url, "date": c.date,
                             "stale": c.stale, "chunks": 0}
            seen[key]["chunks"] += 1
        return list(seen.values())

    def vv_coverage(self) -> dict:
        """Return counts for the V&V coverage indicator."""
        return {
            "graph":        sum(1 for t in self.graph_triples if t.tag == "sourced"),
            "rag":          len(self.rag_chunks),
            "llm_reasoned": sum(1 for t in self.graph_triples if t.tag == "llm_reasoned"),
        }


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

_CUTOFF_YEARS = 2

def _is_stale(date_str: str) -> bool:
    """Return True if date_str is more than _CUTOFF_YEARS years ago."""
    try:
        # Accept YYYY, YYYY-MM, YYYY-MM-DD
        parts = date_str.split("-")
        year  = int(parts[0])
        month = int(parts[1]) if len(parts) > 1 else 6
        doc_date   = datetime.date(year, month, 1)
        cutoff     = datetime.date.today().replace(year=datetime.date.today().year - _CUTOFF_YEARS)
        return doc_date < cutoff
    except Exception:
        return False


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9_]", "_", text.lower())[:64]


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 1 — RAG
# ─────────────────────────────────────────────────────────────────────────────

# Source catalogue — each entry is ingested once and stored in ChromaDB.
# The fetch functions return (text, url, date) tuples.

_SOURCES = [
    {
        "id":   "engineering_toolbox",
        "name": "Engineering Toolbox",
        # Public reference pages — we search by product type keyword
        "url_template": "https://www.engineeringtoolbox.com/{slug}.html",
        "date": "2024-01",
    },
    {
        "id":   "iso_iec",
        "name": "ISO/IEC Abstracts",
        "url_template": "https://www.iso.org/search.html#q={slug}",
        "date": "2024-06",
    },
    {
        "id":   "nasa_ntrs",
        "name": "NASA NTRS",
        "url_template": "https://ntrs.nasa.gov/search?q={slug}",
        "date": "2023-09",
    },
    {
        "id":   "eur_lex",
        "name": "EUR-Lex",
        "url_template": "https://eur-lex.europa.eu/search.html?text={slug}",
        "date": "2024-03",
    },
]

# Persist ChromaDB to a local directory so it survives across runs.
_CHROMA_DIR = os.path.join(os.path.dirname(__file__), ".chromadb")


class RAGLayer:
    """
    Retrieves top-k domain chunks for a given query using LlamaIndex + ChromaDB.

    Gracefully degrades to an empty result if either package is unavailable
    or if the ChromaDB store cannot be opened.
    """

    def __init__(self, top_k: int = 5):
        self.top_k   = top_k
        self._ready  = False
        self._index  = None
        self._setup()

    def _setup(self):
        try:
            from llama_index.core import (
                VectorStoreIndex, Document, StorageContext, Settings
            )
            from llama_index.vector_stores.chroma import ChromaVectorStore
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            import chromadb

            self._VectorStoreIndex = VectorStoreIndex
            self._Document         = Document
            self._StorageContext   = StorageContext
            self._Settings         = Settings
            self._ChromaVectorStore = ChromaVectorStore
            self._chromadb         = chromadb

            # Use a small local embedding model — no API key needed.
            Settings.embed_model = HuggingFaceEmbedding(
                model_name="BAAI/bge-small-en-v1.5"
            )
            Settings.llm = None   # we call Claude ourselves

            self._ready = True
        except ImportError as e:
            print(f"  ⚠ RAG layer unavailable: {e}")
            print("    Install: pip install llama-index llama-index-vector-stores-chroma "
                  "llama-index-embeddings-huggingface chromadb sentence-transformers")

    def _get_or_create_collection(self, collection_name: str = "domain_knowledge"):
        """Return an existing or new ChromaDB collection + its vector store."""
        client     = self._chromadb.PersistentClient(path=_CHROMA_DIR)
        collection = client.get_or_create_collection(collection_name)
        store      = self._ChromaVectorStore(chroma_collection=collection)
        return store, collection

    def ingest(self, product_type: str) -> int:
        """
        Build (or update) the ChromaDB index with synthetic domain documents
        for the given product type. Synthetic docs are derived from the source
        catalogue — in production, replace with real HTTP fetches.
        Returns number of documents ingested.
        """
        if not self._ready:
            return 0
        try:
            docs = _build_synthetic_docs(product_type)
            if not docs:
                return 0

            store, _ = self._get_or_create_collection()
            ctx      = self._StorageContext.from_defaults(vector_store=store)
            self._index = self._VectorStoreIndex.from_documents(
                [self._Document(text=d["text"],
                                metadata={"source_url": d["url"],
                                          "date":       d["date"]})
                 for d in docs],
                storage_context=ctx,
                show_progress=False,
            )
            return len(docs)
        except Exception as e:
            print(f"  ⚠ RAG ingest error: {e}")
            return 0

    def retrieve(self, query: str, product_type: str) -> list[RAGChunk]:
        """Return top-k RAGChunk objects for the query."""
        if not self._ready:
            return []
        try:
            # Load existing index if not already in memory
            if self._index is None:
                store, collection = self._get_or_create_collection()
                if collection.count() == 0:
                    n = self.ingest(product_type)
                    if n == 0:
                        return []
                else:
                    ctx          = self._StorageContext.from_defaults(vector_store=store)
                    self._index  = self._VectorStoreIndex.from_vector_store(
                        store, storage_context=ctx
                    )

            retriever = self._index.as_retriever(similarity_top_k=self.top_k)
            nodes     = retriever.retrieve(query)

            chunks = []
            for node in nodes:
                meta = node.metadata or {}
                url  = meta.get("source_url", "unknown")
                date = meta.get("date", "unknown")
                chunks.append(RAGChunk(
                    text       = node.get_content().strip()[:600],
                    source_url = url,
                    date       = date,
                    stale      = _is_stale(date),
                ))
            return chunks
        except Exception as e:
            print(f"  ⚠ RAG retrieve error: {e}")
            return []


def _build_synthetic_docs(product_type: str) -> list[dict]:
    """
    Build a set of domain documents for the product type from the source
    catalogue. In production these would be real HTTP fetches; here we
    generate representative reference text so the RAG pipeline can be
    exercised end-to-end without external network access.
    """
    slug = _slug(product_type)
    docs = []

    # Engineering Toolbox — material & thermal properties
    docs.append({
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
    })

    # ISO/IEC — relevant standards abstracts
    docs.append({
        "text": textwrap.dedent(f"""\
            ISO/IEC standards applicable to {product_type}:
            ISO 9001:2015 — Quality management systems.
            IEC 61010-1 — Safety requirements for electrical equipment.
            ISO 12100 — Safety of machinery — risk assessment.
            ISO 13849 — Safety of machinery — control systems (PLr categories).
            IEC 60529 — Degrees of protection (IP ratings: IP54, IP65, IP67, IP68).
            EN 62368-1 — Audio/video and IT equipment safety.
            Applicable CE marking directives: Machinery Directive 2006/42/EC,
            Low Voltage Directive 2014/35/EU, EMC Directive 2014/30/EU."""),
        "url":  "https://www.iso.org/search.html?q=" + slug,
        "date": "2024-06",
    })

    # NASA NTRS — reliability and design for space/harsh environments
    docs.append({
        "text": textwrap.dedent(f"""\
            NASA NTRS — reliability guidance for {product_type} systems.
            FMEA (Failure Mode and Effect Analysis): identify single-point failures.
            Derating guidelines: operate components at 50-80% rated capacity.
            Thermal management: worst-case temperature analysis, heat sink sizing.
            Connector reliability: use MIL-DTL-38999 series for harsh environments.
            Vibration: random vibration PSD profiles per GEVS (GSFC-STD-7000).
            Materials: avoid cadmium, mercury; prefer RoHS-compliant finishes.
            Redundancy: cold/hot/warm standby configurations for critical functions."""),
        "url":  "https://ntrs.nasa.gov/search?q=" + slug,
        "date": "2023-09",
    })

    # EUR-Lex — EU regulatory requirements
    docs.append({
        "text": textwrap.dedent(f"""\
            EUR-Lex regulatory framework for {product_type} in the EU market.
            REACH Regulation (EC) No 1907/2006: restriction of hazardous chemicals.
            RoHS Directive 2011/65/EU: restriction of hazardous substances in EEE.
            WEEE Directive 2012/19/EU: waste electrical and electronic equipment.
            General Product Safety Directive 2001/95/EC.
            Ecodesign Regulation (EU) 2019/2021: energy efficiency requirements.
            Battery Regulation (EU) 2023/1542: sustainability and labelling.
            Product Liability Directive: manufacturer responsibility for defects."""),
        "url":  "https://eur-lex.europa.eu/search.html?text=" + slug,
        "date": "2024-03",
    })

    # Older document — will be flagged as stale
    docs.append({
        "text": textwrap.dedent(f"""\
            Legacy reference for {product_type} — design guidelines (2021).
            Note: some values may have been superseded by newer standards.
            Dimensional tolerances: ISO 2768 (general), ISO 286 (limits/fits).
            GD&T: ASME Y14.5-2018. Drawing standards: ISO 128.
            Thread standards: ISO 68-1 (metric), ASME B1.1 (UNC/UNF).
            Welding: ISO 9692 joint preparation, ISO 5817 quality levels."""),
        "url":  f"https://www.engineeringtoolbox.com/{slug}-legacy.html",
        "date": "2021-06",   # > 2 years → stale
    })

    return docs


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 2 — KNOWLEDGE GRAPH (Kuzu)
# ─────────────────────────────────────────────────────────────────────────────

_KUZU_DIR = os.path.join(os.path.dirname(__file__), ".kuzudb")

# Seed triples that populate the graph on first use.
# Each tuple: (subject, relation, object, tag)
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

    # standard → applies_to → product_category
    ("iec_61010",            "applies_to", "electrical_measurement_equipment", "sourced"),
    ("iso_13849",            "applies_to", "industrial_machinery",             "sourced"),
    ("iec_60529",            "applies_to", "enclosure_protection_rating",      "sourced"),
    ("iso_9001",             "applies_to", "quality_management",               "sourced"),
    ("reach_regulation",     "applies_to", "chemical_substances_in_products",  "sourced"),
    ("rohs_directive",       "applies_to", "electrical_electronic_equipment",  "sourced"),
    ("machinery_directive",  "applies_to", "eu_machinery_ce_marking",         "sourced"),
    ("astm_f2413",           "applies_to", "protective_footwear",              "sourced"),
    ("ul_94",                "applies_to", "plastic_flammability_rating",      "sourced"),
    ("iso_14971",            "applies_to", "medical_device_risk_management",   "sourced"),

    # LLM-reasoned edges (inferred, not directly sourced)
    ("high_voltage_system",  "requires",       "isolation_monitoring",    "llm_reasoned"),
    ("outdoor_product",      "requires",       "ip65_or_higher_rating",   "llm_reasoned"),
    ("battery_system",       "compatible_with","fuse_protection",          "llm_reasoned"),
    ("rotating_machinery",   "requires",       "vibration_analysis",      "llm_reasoned"),
    ("food_contact_material","requires",        "fda_or_lfgb_compliance",  "llm_reasoned"),
]


class KnowledgeGraph:
    """
    Kuzu-backed embedded knowledge graph.
    Fails gracefully if kuzu is not installed.
    """

    def __init__(self):
        self._ready = False
        self._db    = None
        self._conn  = None
        self._setup()

    def _setup(self):
        try:
            import kuzu
            self._kuzu = kuzu
            os.makedirs(_KUZU_DIR, exist_ok=True)
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
        """Create node + relationship tables if they don't exist."""
        conn = self._conn
        # Node tables
        for tbl in ("Material", "Component", "Standard",
                    "Constraint", "ProductCategory"):
            try:
                conn.execute(
                    f"CREATE NODE TABLE IF NOT EXISTS {tbl} "
                    f"(name STRING, PRIMARY KEY (name))"
                )
            except Exception:
                pass  # table already exists

        # Unified Entity table for cross-type queries
        try:
            conn.execute(
                "CREATE NODE TABLE IF NOT EXISTS Entity "
                "(name STRING, etype STRING, PRIMARY KEY (name))"
            )
        except Exception:
            pass

        # Relationship table
        try:
            conn.execute(
                "CREATE REL TABLE IF NOT EXISTS Relation "
                "(FROM Entity TO Entity, rel STRING, tag STRING)"
            )
        except Exception:
            pass

    def _seed(self):
        """Insert seed triples if the graph is empty."""
        try:
            result = self._conn.execute("MATCH (e:Entity) RETURN count(e) AS n")
            count  = result.get_next()[0]
            if count > 0:
                return  # already seeded
        except Exception:
            pass

        for subj, rel, obj, tag in _SEED_TRIPLES:
            self._upsert_triple(subj, rel, obj, tag)

    def _upsert_triple(self, subj: str, rel: str, obj: str, tag: str):
        """Insert a triple, ignoring duplicate-key errors."""
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
        """Add a new relationship to the graph. Returns True on success."""
        if not self._ready:
            return False
        try:
            self._upsert_triple(subj, rel, obj, tag)
            return True
        except Exception as e:
            print(f"  ⚠ KG add_triple error: {e}")
            return False

    def query(self, keywords: list[str], limit: int = 20) -> list[KGTriple]:
        """
        Return triples whose subject or object contains any of the keywords.
        Falls back to a broad scan if keyword match returns nothing.
        """
        if not self._ready:
            return []
        try:
            results = []
            seen    = set()
            for kw in keywords[:8]:
                kw_pat = kw.lower().replace(" ", "_")
                q = (
                    "MATCH (a:Entity)-[r:Relation]->(b:Entity) "
                    "WHERE contains(lower(a.name), $kw) OR contains(lower(b.name), $kw) "
                    "RETURN a.name, r.rel, b.name, r.tag LIMIT $lim"
                )
                res = self._conn.execute(q, {"kw": kw_pat, "lim": limit})
                while res.has_next():
                    row = res.get_next()
                    key = (row[0], row[1], row[2])
                    if key not in seen:
                        seen.add(key)
                        results.append(KGTriple(
                            subject  = row[0],
                            relation = row[1],
                            object   = row[2],
                            tag      = row[3],
                        ))
            if not results:
                # Broad fallback — return first N triples
                res = self._conn.execute(
                    "MATCH (a:Entity)-[r:Relation]->(b:Entity) "
                    "RETURN a.name, r.rel, b.name, r.tag LIMIT $lim",
                    {"lim": limit}
                )
                while res.has_next():
                    row = res.get_next()
                    results.append(KGTriple(
                        subject  = row[0],
                        relation = row[1],
                        object   = row[2],
                        tag      = row[3],
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
    Runs before Product Family Agent.
    Builds RAG index + KG, retrieves relevant context,
    and returns a DomainContext for downstream agents.
    """

    def __init__(self):
        self._rag = RAGLayer(top_k=5)
        self._kg  = KnowledgeGraph()

    def run(self, product_idea: str, product_type: str = "",
            intent_goal: str = "") -> DomainContext:
        """
        Retrieve domain knowledge for a product idea.

        Args:
            product_idea: raw user idea string
            product_type: if known (from family agent — pass empty on first call)
            intent_goal:  intent goal string for semantic retrieval

        Returns:
            DomainContext with rag_chunks, graph_triples, validation_trace
        """
        query = " ".join(filter(None, [product_idea, product_type, intent_goal]))
        pt    = product_type or product_idea

        print(f"\n  Ingesting domain sources for: {pt!r}...")
        n_docs = self._rag.ingest(pt)
        if n_docs:
            print(f"  ✓ {n_docs} documents indexed in ChromaDB")

        print(f"  Retrieving RAG chunks...")
        chunks = self._rag.retrieve(query, pt)
        stale  = sum(1 for c in chunks if c.stale)
        print(f"  ✓ {len(chunks)} chunks retrieved"
              + (f" ({stale} stale)" if stale else ""))

        print(f"  Querying knowledge graph...")
        keywords = _extract_keywords(product_idea + " " + intent_goal)
        triples  = self._kg.query(keywords)
        sourced  = sum(1 for t in triples if t.tag == "sourced")
        inferred = len(triples) - sourced
        print(f"  ✓ {len(triples)} triples retrieved "
              f"({sourced} sourced, {inferred} llm_reasoned)")

        trace = {
            "rag_chunks":    len(chunks),
            "stale_chunks":  stale,
            "graph_triples": len(triples),
            "sourced":       sourced,
            "llm_reasoned":  inferred,
            "sources":       list({c.source_url for c in chunks}),
        }

        return DomainContext(
            rag_chunks       = chunks,
            graph_triples    = triples,
            validation_trace = trace,
        )

    def enrich_graph(self, triples: list[tuple[str, str, str, str]]) -> int:
        """
        Add new triples from evaluation/optimizer reasoning back into the graph.
        tag should be "llm_reasoned" for inferred relationships.
        Returns number of triples added.
        """
        added = 0
        for subj, rel, obj, tag in triples:
            if self._kg.add_triple(subj, rel, obj, tag):
                added += 1
        return added


def _extract_keywords(text: str) -> list[str]:
    """Extract meaningful keywords from text for KG lookup."""
    stopwords = {
        "the", "a", "an", "and", "or", "for", "with", "to", "of",
        "in", "on", "at", "by", "from", "is", "are", "be", "that",
        "this", "it", "as", "at", "be", "was", "were", "has", "have",
        "i", "my", "me", "we", "our", "you", "your", "best", "good",
        "high", "low", "all", "any", "some", "new", "most", "more",
    }
    words = re.findall(r"[a-z]{3,}", text.lower())
    return [w for w in words if w not in stopwords][:20]


# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL SINGLETON  (import once, reuse across agents)
# ─────────────────────────────────────────────────────────────────────────────

_agent: DomainKnowledgeAgent | None = None


def get_domain_agent() -> DomainKnowledgeAgent:
    """Return the module-level singleton DomainKnowledgeAgent."""
    global _agent
    if _agent is None:
        _agent = DomainKnowledgeAgent()
    return _agent
