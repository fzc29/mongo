"""
Multi-Agent Portfolio Analysis System — MongoDB Backend
"""

import re
import time
import asyncio
import os
from datetime import datetime, timezone
from typing import Any, Dict

from dotenv import load_dotenv
from anthropic import Anthropic

from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient

# Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings


# ============================================================
# MongoDB Helper
# ============================================================

def get_vector_store(collection_name: str, index_name: str, embedding) -> MongoDBAtlasVectorSearch:
    """
    Returns a LangChain-compatible MongoDB vector store.
    Replaces FAISS.load_local() entirely.
    """
    client = MongoClient(os.getenv("MONGO_URI_USER"))
    collection = client[os.getenv("MONGO_DB_NAME", "portfolio_rag")][collection_name]

    return MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embedding,
        index_name=index_name,
        text_key="text",
        embedding_key="embedding",
    )


# ============================================================
# Base Agent
# ============================================================

class BaseAgent:
    def __init__(self, embedding, anthropic_client: Anthropic, context_k: int = 30):
        self.embedding = embedding
        self.client = anthropic_client
        self.context_k = context_k

        self.model      = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-latest")
        self.max_tokens = int(os.getenv("CLAUDE_MAX_OUTPUT_TOKENS", "2048"))
        self.temperature = float(os.getenv("CLAUDE_TEMPERATURE", "0"))

    def _search(self, store: MongoDBAtlasVectorSearch, query: str, k: int = 20):
        """
        Replaces hybrid_search() + rerank().
        MongoDB Atlas Vector Search handles similarity directly.
        """
        return store.similarity_search(query, k=k)

    def _call_claude(self, system_prompt: str, user_prompt: str) -> str:
        """Call Claude with automatic retry on rate limit errors (429)."""
        wait = 15  # seconds before first retry
        for attempt in range(4):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                return "".join(
                    block.text for block in response.content if hasattr(block, "text")
                ).strip()
            except Exception as e:
                if "rate_limit" in str(e).lower() and attempt < 3:
                    print(f"  Rate limit hit — waiting {wait}s before retry {attempt + 1}/3...")
                    time.sleep(wait)
                    wait *= 2  # exponential backoff: 15 → 30 → 60s
                else:
                    raise

    def _critique(self, draft: str, rubric: str) -> str:
        """
        Ask the model to critique a draft against a rubric.
        Returns a short critique — either 'PASS' or a list of specific issues.
        Uses the base model (haiku) to keep costs low.
        """
        response = self.client.messages.create(
            model=os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001"),
            max_tokens=512,
            temperature=0.0,
            system=(
                "You are a strict editorial reviewer. "
                "Evaluate the draft against the rubric provided. "
                "If the draft meets all criteria, respond with exactly: PASS\n"
                "If it fails any criteria, respond with a numbered list of specific issues only. "
                "Be concise — maximum 5 issues. Do not rewrite the draft."
            ),
            messages=[{"role": "user", "content": f"RUBRIC:\n{rubric}\n\nDRAFT:\n{draft}"}],
        )
        return "".join(
            block.text for block in response.content if hasattr(block, "text")
        ).strip()

    def _call_claude_with_critique(
        self,
        system_prompt: str,
        user_prompt: str,
        rubric: str,
        max_retries: int = 2,
    ) -> tuple[str, list[str]]:
        """
        Generate a response, critique it, and revise if needed.
        Returns (final_output, list_of_critique_rounds).
        Falls back to the original draft if retries are exhausted.

        max_retries=0 disables critique and behaves identically to _call_claude().
        """
        critique_log = []
        draft = self._call_claude(system_prompt, user_prompt)

        for attempt in range(max_retries):
            critique = self._critique(draft, rubric)
            critique_log.append(critique)

            if critique.strip().upper() == "PASS":
                break

            # Revise — feed the critique back into the same system prompt context
            revision_prompt = (
                f"{user_prompt}\n\n"
                f"---\n"
                f"Your previous draft had the following issues:\n{critique}\n\n"
                f"Please rewrite addressing each issue. Keep everything that was correct."
            )
            draft = self._call_claude(system_prompt, revision_prompt)

        return draft, critique_log


# ============================================================
# Market Context Agent
# ============================================================

class MarketContextAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.context_store = get_vector_store(
            "context_vectors",
            "vector_index",
            self.embedding,
        )
        # Override model for this agent via env var if a stronger model is available.
        # e.g. set CLAUDE_CONTEXT_MODEL=claude-3-5-sonnet-20241022 in .env
        context_model = os.getenv("CLAUDE_CONTEXT_MODEL")
        if context_model:
            self.model = context_model

    def _build_macro_query(self, question: str) -> str:
        """Reframe the question as a macro-focused vector search query."""
        period = _extract_period(question)
        if period:
            # Convert YYYY-MM to "Month YYYY" for readable query
            month_names = {v: k.capitalize() for k, v in _MONTH_MAP.items() if len(k) > 3}
            mm = period.split("-")[1]
            yyyy = period.split("-")[0]
            month_label = month_names.get(mm, mm)
            return (
                f"{month_label} {yyyy} macro market events rates FX equities "
                f"central bank Fed monetary policy inflation commodities"
            )
        return "macro market events rates FX equities central bank monetary policy"

    def analyze(self, question: str, feedback: str = "") -> Dict[str, Any]:
        period = _extract_period(question)
        macro_query = self._build_macro_query(question)

        # Fetch extra docs so we have room to filter by period
        fetch_k = self.context_k * 3
        all_docs = self._search(self.context_store, macro_query, k=fetch_k)

        # Post-filter to the detected period if metadata exists on the chunks
        if period:
            period_docs = [d for d in all_docs if d.metadata.get("report_period") == period]
            docs = period_docs[:self.context_k] if len(period_docs) >= 5 else all_docs[:self.context_k]
        else:
            docs = all_docs[:self.context_k]

        month_label = macro_query.split(" macro")[0] if period else "the relevant period"

        system_prompt = (
            "You are a macro market analyst for a hedge fund. "
            "You will be given a set of source documents. "
            "Your ONLY job is to summarize what those documents say. "
            "Every single claim you make must be directly supported by the provided documents. "
            "Do NOT use any knowledge from your training data. "
            "Do NOT invent events, figures, yields, price moves, or central bank actions. "
            "If a topic is not covered in the documents, do not mention it. "
            "If the documents are insufficient, say exactly which topics are missing."
        )

        user_prompt = f"""
Source documents from {month_label}:

{chr(10).join(f'---{chr(10)}{doc.page_content}' for doc in docs)}

---
Based ONLY on the source documents above, summarize:
1. The key macro events and market-moving developments
2. Which asset classes were affected and how (rates, FX, equities, commodities)
3. Any central bank actions or policy shifts mentioned

Do not add anything not stated in the documents above.
{f'''
---
REVISION FEEDBACK FROM USER:
{feedback}
Address this feedback in your revised response. Keep everything that was correct.''' if feedback else ''}
"""

        analysis = self._call_claude(system_prompt, user_prompt)

        return {
            "agent": "MarketContextAgent",
            "analysis": analysis,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# ============================================================
# Portfolio Performance Agent
# ============================================================

_MONTH_MAP = {
    "january": "01",  "jan": "01",
    "february": "02", "feb": "02",
    "march": "03",    "mar": "03",
    "april": "04",    "apr": "04",
    "may": "05",
    "june": "06",     "jun": "06",
    "july": "07",     "jul": "07",
    "august": "08",   "aug": "08",
    "september": "09","sep": "09",  "sept": "09",
    "october": "10",  "oct": "10",
    "november": "11", "nov": "11",
    "december": "12", "dec": "12",
}


def _extract_period(question: str) -> str | None:
    """Extract a YYYY-MM period string from a free-text question."""
    q = question.lower()
    for name, num in _MONTH_MAP.items():
        m = re.search(rf'\b{name}\b[^a-z]*\b(20\d{{2}})\b', q)
        if m:
            return f"{m.group(1)}-{num}"
    m = re.search(r'\b(20\d{2})-(\d{2})\b', q)
    if m:
        return m.group(0)
    return None


class PortfolioPerformanceAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Direct MongoDB connection — no vector store needed for structured PnL
        self._mongo_client = MongoClient(os.getenv("MONGO_URI_USER"))

    def _pnl_col(self):
        return self._mongo_client[os.getenv("MONGO_DB_NAME", "portfolio_rag")]["pnl_table"]

    def _summary_col(self):
        return self._mongo_client[os.getenv("MONGO_DB_NAME", "portfolio_rag")]["pnl_summary"]

    def _get_pnl_summary(self, period: str) -> dict | None:
        doc = self._summary_col().find_one({"report_period": period}, {"_id": 0})
        return doc

    def _extract_period(self, question: str) -> str | None:
        return _extract_period(question)

    def _get_available_periods(self) -> list[str]:
        return sorted(self._pnl_col().distinct("report_period"))

    def _format_as_table(self, rows: list[dict]) -> str:
        """Render row dicts as a markdown table for the prompt."""
        skip = {"report_period", "source_file", "uploaded_by", "uploaded_at", "_id"}
        cols = [k for k in rows[0].keys() if k not in skip]
        header = "| " + " | ".join(cols) + " |"
        sep    = "| " + " | ".join("---" for _ in cols) + " |"
        lines  = [header, sep]
        for row in rows:
            lines.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
        return "\n".join(lines)

    def analyze(self, question: str, feedback: str = "") -> Dict[str, Any]:
        period = self._extract_period(question)
        available = self._get_available_periods()

        if not period:
            period = available[-1] if available else None

        if not period:
            return {
                "agent": "PortfolioPerformanceAgent",
                "analysis": "No PnL data available in the database.",
                "period": None,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        rows = list(self._pnl_col().find({"report_period": period}, {"_id": 0}))

        if not rows:
            return {
                "agent": "PortfolioPerformanceAgent",
                "analysis": (
                    f"No PnL data found for period '{period}'. "
                    f"Available periods: {available}"
                ),
                "period": period,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        # Fetch pre-computed AUM summary for accurate return figure
        pnl_summary = self._get_pnl_summary(period)
        if pnl_summary and pnl_summary.get("return_pct") is not None:
            ret_pct = pnl_summary["return_pct"]
            start_aum = pnl_summary.get("start_aum", 0)
            end_aum = pnl_summary.get("end_aum", 0)
            total_pnl = pnl_summary.get("total_pnl", 0)
            summary_line = (
                f"PORTFOLIO RETURN FOR {period}: {ret_pct:+.2f}%\n"
                f"Start AUM: ${start_aum:,.0f}  |  End AUM: ${end_aum:,.0f}  |  "
                f"Total P&L: ${total_pnl:,.0f}\n"
                f"This return figure is mathematically exact — use it verbatim."
            )
        else:
            summary_line = ""

        system_prompt = (
            "You are a portfolio performance analyst for a macro hedge fund. "
            "Your job is to produce a thematic analysis of the portfolio — NOT a line-by-line attribution table. "
            "\n\n"
            "GROUP positions by theme: (1) Rates & Duration (include TIPS, swaptions, SOFR futures, Gilts, JGBs together), "
            "(2) Foreign Exchange, (3) Equities, (4) Commodities & Precious Metals, (5) Digital Assets. "
            "Within each theme, lead with total P&L impact, then explain the market dynamics, then describe any trading activity from the Trading Notes. "
            "\n\n"
            "For every major P&L item (>$200k in either direction), articulate the FULL causal chain: "
            "(1) what happened in the market with specific dates, "
            "(2) why it happened — the catalyst, "
            "(3) HOW that move transmitted into the position's P&L — the mechanism through rates, vol, FX, or beta, "
            "(4) any action taken per the Trading Notes and its rationale. "
            "\n\n"
            "Treat linked positions as single trades (e.g. long Gilts + short US 10y = one RV trade). "
            "Distinguish between organic position changes (delta drift, roll) and active trading decisions. "
            "Use exact P&L figures from the data — never round or estimate. "
            "Omit positions with P&L below $200k unless they are strategically relevant."
        )

        user_prompt = f"""
Question: {question}

{summary_line}

P&L Data for period {period}:
{self._format_as_table(rows)}
{f'''
---
REVISION FEEDBACK FROM USER:
{feedback}
Address this feedback in your revised response. Keep everything that was correct.''' if feedback else ''}
"""

        rubric = (
            "1. Positions are grouped by theme (Rates & Duration, FX, Equities, Commodities, Digital Assets) — not listed individually.\n"
            "2. Every P&L figure above $200k includes a full causal chain: catalyst → market move → transmission mechanism → position impact.\n"
            "3. Linked positions (e.g. Gilts + US 10y RV, TIPS curve) are described as single trades, not separately.\n"
            "4. Trading Notes from the data are reflected with their rationale.\n"
            "5. P&L figures are exact — not rounded or estimated.\n"
            "6. Positions below $200k P&L are omitted unless strategically relevant."
        )

        analysis, critique_log = self._call_claude_with_critique(
            system_prompt, user_prompt, rubric, max_retries=1
        )

        return {
            "agent": "PortfolioPerformanceAgent",
            "analysis": analysis,
            "period": period,
            "pnl_summary": pnl_summary,
            "critique_log": critique_log,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# ============================================================
# Weekly Market Data Agent
# ============================================================

class WeeklyMarketDataAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weekly_store = get_vector_store(
            "weekly_vectors",
            "vector_index",
            self.embedding,
        )

    def analyze(self, question: str) -> Dict[str, Any]:
        # Append a date-anchored suffix so the vector search targets the right
        # time window within multi-month files rather than finding an older
        # period with similar vocabulary.
        search_query = f"{question} 2026 weekly market data"
        docs = self._search(self.weekly_store, search_query, k=self.context_k)

        system_prompt = (
            "You are a market data analyst for a hedge fund. "
            "Summarize the key weekly trends in the data, focusing on: (1) significant moves in rates, "
            "FX, equities, and commodities, (2) inflection points or trend breaks, and (3) how the "
            "week's data fits into the broader macro narrative. "
            "Connect data points to each other — e.g. how a rates move influenced FX or equity positioning. "
            "Strictly use only the provided context."
        )

        user_prompt = f"""
Question: {question}

Weekly Market Data:
{''.join(doc.page_content for doc in docs[:20])}
"""

        analysis = self._call_claude(system_prompt, user_prompt)

        return {
            "agent": "WeeklyMarketDataAgent",
            "analysis": analysis,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# ============================================================
# Risk Agent
# ============================================================

class RiskAnalystAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mongo_client = MongoClient(os.getenv("MONGO_URI_USER"))

    def _pnl_col(self):
        return self._mongo_client[os.getenv("MONGO_DB_NAME", "portfolio_rag")]["pnl_table"]

    def _summary_col(self):
        return self._mongo_client[os.getenv("MONGO_DB_NAME", "portfolio_rag")]["pnl_summary"]

    @staticmethod
    def _parse_val(s) -> float | None:
        """Parse a stored value string like '$-2,538,840' or '95000.0' → float."""
        if s is None:
            return None
        s = str(s).strip().replace("$", "").replace(",", "")
        s = re.sub(r"^\((.+)\)$", r"-\1", s)   # (1234) → -1234
        try:
            return float(s)
        except (ValueError, TypeError):
            return None

    def _compute_risk_metrics(self, period: str) -> str:
        """
        Query pnl_table for the period and return a structured metrics block:
          - P&L attribution by asset class ($ and % of total)
          - Rates DV01 exposure (beginning vs ending)
          - Top 3 contributors and top 3 detractors
          - Notional concentration (largest long and short positions)

        Returns a formatted string ready to inject into the risk agent prompt.
        """
        rows = list(self._pnl_col().find({"report_period": period}, {"_id": 0}))
        if not rows:
            return ""

        summary_doc = self._summary_col().find_one(
            {"report_period": period}, {"_id": 0}
        )
        total_pnl_abs = abs(self._parse_val(
            (summary_doc or {}).get("total_pnl", 0)
        ) or 1)  # avoid div/0

        # ── detect column names (they include the month, e.g. trading_notes_feb_2026)
        sample = rows[0]
        pnl_col    = next((k for k in sample if "p_l" in k or "pnl" in k.replace("_","")), None)
        beg_col    = next((k for k in sample if "beginning" in k), None)
        end_col    = next((k for k in sample if "ending" in k), None)
        type_col   = next((k for k in sample if "position_type" in k or k == "type"), None)
        class_col  = next((k for k in sample if "asset_class" in k or k == "class"), None)
        pos_col    = next((k for k in sample if k == "positions"), None)

        # ── P&L by asset class
        theme_map = {
            "rate":      "Rates & Duration",
            "rates":     "Rates & Duration",
            "equity":    "Equities",
            "fx":        "Foreign Exchange",
            "commodity": "Commodities",
            "crypto":    "Digital Assets",
            "":          "Other",
        }
        theme_pnl: dict[str, float] = {}
        position_pnls: list[tuple[str, str, float]] = []  # (name, theme, pnl)

        for row in rows:
            raw_class = str(row.get(class_col, "") or "").strip().lower()
            theme     = theme_map.get(raw_class, "Other")
            pnl_val   = self._parse_val(row.get(pnl_col)) if pnl_col else None
            pos_name  = str(row.get(pos_col, "Unknown") or "Unknown").strip()

            if pnl_val is not None:
                theme_pnl[theme] = theme_pnl.get(theme, 0.0) + pnl_val
                position_pnls.append((pos_name, theme, pnl_val))

        # ── Rates DV01 (beginning and ending)
        dv01_beg = 0.0
        dv01_end = 0.0
        if beg_col and end_col and type_col:
            for row in rows:
                if str(row.get(type_col, "")).strip().upper() == "DV01":
                    b = self._parse_val(row.get(beg_col))
                    e = self._parse_val(row.get(end_col))
                    if b is not None:
                        dv01_beg += b
                    if e is not None:
                        dv01_end += e

        # ── Notional positions (top 3 longs and shorts by absolute ending notional)
        notional_rows: list[tuple[str, float]] = []
        if end_col and type_col:
            for row in rows:
                if str(row.get(type_col, "")).strip().upper() == "NOTIONAL":
                    e = self._parse_val(row.get(end_col))
                    name = str(row.get(pos_col, "?") or "?").strip()
                    if e is not None and e != 0:
                        notional_rows.append((name, e))
        notional_rows.sort(key=lambda x: x[1], reverse=True)
        top_longs  = [(n, v) for n, v in notional_rows if v > 0][:3]
        top_shorts = [(n, v) for n, v in notional_rows if v < 0][:3]

        # ── Top contributors / detractors
        position_pnls.sort(key=lambda x: x[2], reverse=True)
        contributors = position_pnls[:3]
        detractors   = position_pnls[-3:][::-1]

        # ── Format output
        lines = [f"PORTFOLIO RISK METRICS — {period}", ""]

        # Section 1: P&L attribution
        lines.append("── P&L ATTRIBUTION BY THEME ──")
        lines.append(f"{'Theme':<25} {'P&L':>14}  {'% of Total':>10}  Direction")
        lines.append("-" * 60)
        for theme, val in sorted(theme_pnl.items(), key=lambda x: -abs(x[1])):
            pct = (val / total_pnl_abs * 100) if total_pnl_abs else 0
            direction = "▲ Long" if val > 0 else "▼ Short"
            lines.append(f"{theme:<25} ${val:>13,.0f}  {pct:>+9.1f}%  {direction}")
        total_computed = sum(theme_pnl.values())
        lines.append("-" * 60)
        lines.append(f"{'TOTAL (computed)':<25} ${total_computed:>13,.0f}")
        lines.append("")

        # Section 2: Rates DV01
        if dv01_beg or dv01_end:
            lines.append("── RATES DV01 EXPOSURE ──")
            lines.append(f"  Beginning of period:  ${dv01_beg:>12,.0f}")
            lines.append(f"  End of period:        ${dv01_end:>12,.0f}")
            change = dv01_end - dv01_beg
            lines.append(f"  Change (net add/cut): ${change:>+12,.0f}")
            lines.append("")

        # Section 3: Notional concentration
        if top_longs or top_shorts:
            lines.append("── LARGEST NOTIONAL POSITIONS (ending) ──")
            if top_longs:
                lines.append("  Top longs:")
                for name, val in top_longs:
                    lines.append(f"    {name:<35} ${val:>14,.0f}")
            if top_shorts:
                lines.append("  Top shorts:")
                for name, val in top_shorts:
                    lines.append(f"    {name:<35} ${val:>14,.0f}")
            lines.append("")

        # Section 4: Contributors / detractors
        lines.append("── TOP P&L CONTRIBUTORS ──")
        for name, theme, val in contributors:
            lines.append(f"  {name:<35} ${val:>+14,.0f}  ({theme})")
        lines.append("")
        lines.append("── TOP P&L DETRACTORS ──")
        for name, theme, val in detractors:
            lines.append(f"  {name:<35} ${val:>+14,.0f}  ({theme})")
        lines.append("")

        # Section 5: Concentration flag
        top3_pnl = sum(abs(p[2]) for p in contributors)
        concentration_pct = top3_pnl / total_pnl_abs * 100 if total_pnl_abs else 0
        lines.append(f"── CONCENTRATION ──")
        lines.append(f"  Top 3 positions account for {concentration_pct:.0f}% of gross P&L")

        return "\n".join(lines)

    def analyze(self, question: str, market_context: str, portfolio_performance: str, feedback: str = "") -> Dict[str, Any]:
        period = _extract_period(question)

        # Pre-compute structured metrics from pnl_table
        metrics_block = ""
        if period:
            try:
                metrics_block = self._compute_risk_metrics(period)
            except Exception as e:
                print(f"  RiskAnalystAgent: metrics computation failed — {e}")

        system_prompt = (
            "You are a senior portfolio risk analyst for a global macro hedge fund. "
            "You will be given structured portfolio metrics, market context, and performance analysis. "
            "Produce a rigorous, forward-looking risk report organized into EXACTLY FOUR sections:\n\n"

            "SECTION 1 — MACRO DRIVERS\n"
            "Identify the 2–3 most influential macro forces that drove portfolio performance this period. "
            "For each driver: name the specific force (e.g. 'US real rate decline', 'CNH devaluation pressure'), "
            "explain which positions it affected and through what transmission mechanism, "
            "and identify cross-asset correlations — which positions were driven by the same underlying force "
            "and which moved independently. "
            "Use the P&L attribution table to quantify how much each driver contributed.\n\n"

            "SECTION 2 — DIVERSIFICATION OPPORTUNITIES\n"
            "Given the current position concentrations shown in the metrics, screen for 2–3 trade ideas "
            "that are structurally uncorrelated or negatively correlated with the dominant exposures. "
            "For each idea: name the instrument or theme, explain WHY it is uncorrelated under the current regime, "
            "and describe what macro scenario it hedges against. "
            "Do not recommend adding more of what already worked — focus on genuine diversifiers.\n\n"

            "SECTION 3 — POSITION ADJUSTMENTS\n"
            "Propose 2–3 specific adjustments to improve the portfolio's risk-adjusted return profile. "
            "Reference the concentration data and the top detractors directly. "
            "For each adjustment: state what to do (add, reduce, restructure), name the position, "
            "explain the risk-return rationale, and identify what would trigger the adjustment. "
            "Be concrete — avoid generic 'reduce risk' language.\n\n"

            "SECTION 4 — STRESS TESTS\n"
            "Run three stress tests: two historical analogs and one forward-looking hypothetical. "
            "For each test: name the scenario, describe the key market moves (rates, FX, equities, vol), "
            "identify which specific positions are most exposed using the DV01 and notional data provided, "
            "estimate the approximate P&L impact direction and magnitude for each major theme, "
            "and note any natural hedges that would offset the shock. "
            "Historical analogs should be chosen for their relevance to the CURRENT portfolio composition — "
            "not just the most famous crises. The hypothetical should reflect a plausible tail risk "
            "in the current macro environment.\n\n"

            "STANDARDS: Use exact figures from the metrics table. Never invent data. "
            "Every claim about a position must reference the actual position name from the data. "
            "Avoid generic risk language — every sentence should be specific to this portfolio."
        )

        user_prompt = f"""
Question: {question}

{f"PORTFOLIO RISK METRICS:{chr(10)}{metrics_block}{chr(10)}" if metrics_block else ""}
Market Context:
{market_context}

Portfolio Performance:
{portfolio_performance}
{f'''
---
REVISION FEEDBACK FROM USER:
{feedback}
Address this feedback in your revised response. Keep everything that was correct.''' if feedback else ''}
"""

        analysis = self._call_claude(system_prompt, user_prompt)

        return {
            "agent": "RiskAnalystAgent",
            "analysis": analysis,
            "metrics": metrics_block,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# ============================================================
# Newsletter Writer
# ============================================================

class NewsletterWriterAgent(BaseAgent):
    def write(
        self,
        question: str,
        market_context: str,
        portfolio_performance: str,
        risk_analysis: str,
        weekly_market_data: str = "",
        pnl_summary: dict | None = None,
        feedback: str = "",
    ) -> Dict[str, Any]:
        # Build a return fact line to anchor the opening paragraph
        if pnl_summary and pnl_summary.get("return_pct") is not None:
            ret_pct = pnl_summary["return_pct"]
            total_pnl = pnl_summary.get("total_pnl", 0)
            return_fact = (
                f"EXACT MONTHLY RETURN: {ret_pct:+.2f}%  (Total P&L: ${total_pnl:,.0f})\n"
                f"You MUST use this figure in the opening paragraph. Do not calculate or estimate the return yourself."
            )
        else:
            return_fact = ""

        system_prompt = (
            "You are writing a monthly investor letter for a global macro hedge fund. "
            "Target length: 1,000–1,500 words. Tone: direct, candid, analytical — not defensive or euphemistic. "
            "\n\n"
            "STRUCTURE:\n"
            "1. Opening paragraph: state the monthly return and 2-3 sentence macro summary of what drove the month.\n"
            "2. Body: organized by THEME not by position. Suggested sections: Precious Metals & Miners, "
            "Rates & Duration, Foreign Exchange, Equities, Digital Assets. "
            "Within each section: lead with P&L impact, then market dynamics, then trading activity.\n"
            "3. Outlook: connect current positioning to specific forward catalysts. "
            "Name what needs to happen for the portfolio to benefit AND what would make the thesis wrong.\n"
            "4. Closing: one short paragraph.\n"
            "\n\n"
            "ANALYTICAL STANDARDS:\n"
            "- Explain transmission mechanisms, not just correlations. "
            "Don't say 'oil rose and gold fell' — explain the full chain: why oil moved, how that repriced inflation/rates, how that mechanism hit the position.\n"
            "- Surface cross-asset linkages: which positions were driven by the same underlying force? Which provided natural offsets?\n"
            "- Organize around REGIME PHASES within the month (e.g. 'Early Feb: gold volatility phase, Feb 2–7'), not day-by-day recitation.\n"
            "- Use exact P&L figures. Never round.\n"
            "- Omit positions below ~$200k P&L impact unless strategically relevant.\n"
            "- Do NOT list every line item. This is a narrative, not an attribution table.\n"
            "- Never assert macro facts (data releases, yield levels) not present in the provided context."
        )

        user_prompt = f"""
Question: {question}

{return_fact}

Market Context:
{market_context}

Weekly Market Data:
{weekly_market_data}

Portfolio Performance:
{portfolio_performance}

Risk Analysis:
{risk_analysis}

Write a professional monthly newsletter.
{f'''
---
REVISION FEEDBACK FROM USER:
{feedback}
Address this feedback in your revised response. Keep everything that was correct.''' if feedback else ''}
"""

        rubric = (
            "1. Opens with monthly return figure and 2-3 sentence macro summary.\n"
            "2. Body is organized by theme/asset class — NOT a position-by-position list.\n"
            "3. Each thematic section leads with P&L impact, then market dynamics, then trading activity.\n"
            "4. Transmission mechanisms are explained (full causal chain), not just correlations stated.\n"
            "5. Cross-asset linkages are surfaced (which positions were driven by the same force).\n"
            "6. Month is organized around regime phases, not a chronological day-by-day walkthrough.\n"
            "7. Outlook names specific catalysts and acknowledges what would make the thesis wrong.\n"
            "8. No macro facts asserted without support from the provided context.\n"
            "9. Length is approximately 1,000–1,500 words."
        )

        newsletter, critique_log = self._call_claude_with_critique(
            system_prompt, user_prompt, rubric, max_retries=1
        )

        return {
            "agent": "NewsletterWriterAgent",
            "newsletter": newsletter,
            "critique_log": critique_log,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# ============================================================
# Orchestrator
# ============================================================

class OrchestratorAgent:
    def __init__(self, embedding, anthropic_client: Anthropic, context_k: int = 30):
        # base_dir removed — no longer needed without local FAISS files
        self.market      = MarketContextAgent(embedding, anthropic_client, context_k)
        self.performance = PortfolioPerformanceAgent(embedding, anthropic_client, context_k)
        self.weekly      = WeeklyMarketDataAgent(embedding, anthropic_client, context_k)
        self.risk        = RiskAnalystAgent(embedding, anthropic_client, context_k)
        self.writer      = NewsletterWriterAgent(embedding, anthropic_client, context_k)

    async def run_parallel(self, question: str) -> Dict[str, Any]:
        # Market, performance, and weekly run in parallel
        market_result, perf_result, weekly_result = await asyncio.gather(
            asyncio.to_thread(self.market.analyze, question),
            asyncio.to_thread(self.performance.analyze, question),
            asyncio.to_thread(self.weekly.analyze, question),
        )

        # Risk waits for market + performance
        risk_result = await asyncio.to_thread(
            self.risk.analyze,
            question,
            market_result["analysis"],
            perf_result["analysis"],
        )

        # Newsletter waits for everything
        writer_result = await asyncio.to_thread(
            self.writer.write,
            question,
            market_result["analysis"],
            perf_result["analysis"],
            risk_result["analysis"],
            weekly_result["analysis"],
            perf_result.get("pnl_summary"),
        )

        return {
            "question":    question,
            "market":      market_result,
            "performance": perf_result,
            "weekly":      weekly_result,
            "risk":        risk_result,
            "newsletter":  writer_result,
            "timestamp":   datetime.now(timezone.utc).isoformat(),
        }

    async def revise_section(
        self,
        section: str,
        feedback: str,
        current_result: dict,
    ) -> dict:
        """
        Re-run a single section with user feedback, then cascade to newsletter.

        section options: "newsletter", "risk", "performance", "market"

        All revisions cascade to newsletter so the final output stays coherent.
        Other sections (weekly, market←→risk cascade) are left unchanged to
        keep revision fast — user can do a full re-run if needed.
        """
        import copy
        result = copy.deepcopy(current_result)
        question = result["question"]

        if section == "newsletter":
            writer_result = await asyncio.to_thread(
                self.writer.write,
                question,
                result["market"]["analysis"],
                result["performance"]["analysis"],
                result["risk"]["analysis"],
                result["weekly"]["analysis"],
                result["performance"].get("pnl_summary"),
                feedback,
            )
            result["newsletter"] = writer_result

        elif section == "risk":
            risk_result = await asyncio.to_thread(
                self.risk.analyze,
                question,
                result["market"]["analysis"],
                result["performance"]["analysis"],
                feedback,
            )
            result["risk"] = risk_result
            # cascade to newsletter
            writer_result = await asyncio.to_thread(
                self.writer.write,
                question,
                result["market"]["analysis"],
                result["performance"]["analysis"],
                risk_result["analysis"],
                result["weekly"]["analysis"],
                result["performance"].get("pnl_summary"),
            )
            result["newsletter"] = writer_result

        elif section == "performance":
            perf_result = await asyncio.to_thread(
                self.performance.analyze,
                question,
                feedback,
            )
            result["performance"] = perf_result
            # cascade to newsletter
            writer_result = await asyncio.to_thread(
                self.writer.write,
                question,
                result["market"]["analysis"],
                perf_result["analysis"],
                result["risk"]["analysis"],
                result["weekly"]["analysis"],
                perf_result.get("pnl_summary"),
            )
            result["newsletter"] = writer_result

        elif section == "market":
            market_result = await asyncio.to_thread(
                self.market.analyze,
                question,
                feedback,
            )
            result["market"] = market_result
            # cascade to newsletter
            writer_result = await asyncio.to_thread(
                self.writer.write,
                question,
                market_result["analysis"],
                result["performance"]["analysis"],
                result["risk"]["analysis"],
                result["weekly"]["analysis"],
                result["performance"].get("pnl_summary"),
            )
            result["newsletter"] = writer_result

        # track revision in result metadata
        if "revisions" not in result:
            result["revisions"] = []
        result["revisions"].append({
            "section":   section,
            "feedback":  feedback,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        return result


# ============================================================
# Factory
# ============================================================

def build_agent_system() -> OrchestratorAgent:
    load_dotenv()

    provider = os.getenv("EMBEDDING_PROVIDER", "gemini").lower()

    if provider == "gemini":
        embedding = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=os.getenv("GEMINI_API_KEY"),
        )
    elif provider == "openai":
        embedding = OpenAIEmbeddings(
            model="text-embedding-3-large",
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")

    anthropic_client = Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))

    return OrchestratorAgent(embedding, anthropic_client)


# # ============================================================
# # Entry Point
# # ============================================================

# if __name__ == "__main__":
#     orchestrator = build_agent_system()

#     question = "Analyze October 2025 portfolio performance and produce a newsletter."

#     result = asyncio.run(orchestrator.run_parallel(question))

#     print("\nNEWSLETTER\n")
#     print(result["newsletter"]["newsletter"])