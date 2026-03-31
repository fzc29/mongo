"""
Multi-Agent Portfolio Analysis System — MongoDB Backend
"""

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


# ============================================================
# Market Context Agent
# ============================================================

class MarketContextAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # context_vectors uses a different index name — see build_index.py
        self.context_store = get_vector_store(
            "context_vectors",
            "vector_index_context",
            self.embedding,
        )

    def analyze(self, question: str) -> Dict[str, Any]:
        docs = self._search(self.context_store, question, k=self.context_k)

        system_prompt = (
            "You are a macro market analyst. Extract key events and impacts "
            "strictly from provided context."
        )

        user_prompt = f"""
Question: {question}

Context:
{''.join(doc.page_content for doc in docs[:20])}
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

class PortfolioPerformanceAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pnl_store = get_vector_store(
            "pnl_vectors",
            "vector_index",
            self.embedding,
        )

    def analyze(self, question: str) -> Dict[str, Any]:
        docs = self._search(
            self.pnl_store,
            f"{question} PnL attribution portfolio positions",
            k=self.context_k,
        )

        system_prompt = "You are a portfolio performance analyst."

        user_prompt = f"""
Question: {question}

P&L Data:
{''.join(doc.page_content for doc in docs[:20])}
"""

        analysis = self._call_claude(system_prompt, user_prompt)

        return {
            "agent": "PortfolioPerformanceAgent",
            "analysis": analysis,
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
        docs = self._search(self.weekly_store, question, k=self.context_k)

        system_prompt = (
            "You are a market data analyst. Extract key weekly trends and "
            "data points strictly from the provided context."
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
        system_prompt = "You are a portfolio risk analyst."

        user_prompt = f"""
Question: {question}

Market Context:
{market_context}

Portfolio Performance:
{portfolio_performance}

Provide base, upside, downside scenarios with probabilities.
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
        system_prompt = "You are a hedge fund newsletter writer."

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

        newsletter = self._call_claude(system_prompt, user_prompt)

        return {
            "agent": "NewsletterWriterAgent",
            "newsletter": newsletter,
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