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

    def analyze(self, question: str) -> Dict[str, Any]:
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

    def analyze(self, question: str) -> Dict[str, Any]:
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

P&L Data for period {period}:
{self._format_as_table(rows)}
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
    def analyze(self, question: str, market_context: str, portfolio_performance: str) -> Dict[str, Any]:
        system_prompt = (
            "You are a portfolio risk analyst for a macro hedge fund. "
            "Construct three forward-looking scenarios: base, upside, and downside. "
            "\n\n"
            "For each scenario:\n"
            "(1) Describe the specific macro conditions — name the catalysts, not just 'if things improve'.\n"
            "(2) Explain which positions benefit or suffer and WHY — trace the transmission mechanism.\n"
            "(3) Note where the portfolio has convexity or asymmetric payoff.\n"
            "(4) Assign a probability.\n"
            "\n\n"
            "Also address: what needs to happen for the portfolio to recover? What would make the core thesis wrong? "
            "Identify natural hedges within the portfolio and explain how they interact. "
            "Ground everything in the actual positions and macro dynamics provided — no generic risk statements."
        )

        user_prompt = f"""
Question: {question}

Market Context:
{market_context}

Portfolio Performance:
{portfolio_performance}

Provide base, upside, and downside scenarios with probabilities.
"""

        analysis = self._call_claude(system_prompt, user_prompt)

        return {
            "agent": "RiskAnalystAgent",
            "analysis": analysis,
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
    ) -> Dict[str, Any]:
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

Market Context:
{market_context}

Weekly Market Data:
{weekly_market_data}

Portfolio Performance:
{portfolio_performance}

Risk Analysis:
{risk_analysis}

Write a professional monthly newsletter.
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