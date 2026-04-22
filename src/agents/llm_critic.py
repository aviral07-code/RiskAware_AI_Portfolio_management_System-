import json
import os
import re
import numpy as np
from typing import Dict, Tuple

from langchain_community.vectorstores import FAISS
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

try:
    from langchain_ollama import ChatOllama
except ImportError:
    from langchain_community.chat_models import ChatOllama
    
from langchain_core.prompts import ChatPromptTemplate

# --- NEW: Import the Sentiment module ---
from src.utils.sentiment import FinBertSentiment

class LLMCritic:
    def __init__(
        self,
        vector_db_path: str = None,
        model_name: str = "llama3.1",
        temperature: float = 0.5,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        if vector_db_path is None:
            this_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(this_dir))
            vector_db_path = os.path.join(project_root, "data", "vector_db", "faiss_index")

        # CPU forced to avoid macOS Metal memory constraints
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'}
        )
        
        try:
            self.db = FAISS.load_local(
                vector_db_path,
                embeddings,
                allow_dangerous_deserialization=True,
            )
        except Exception as e:
            print(f"⚠️ LLMCritic Warning: RAG disabled. Error: {e}")
            self.db = None

        self.llm = ChatOllama(model=model_name, temperature=temperature)

        # --- NEW: Initialize FinBERT for secondary validation (Forced to CPU) ---
        try:
            print("⏳ Loading FinBERT Sentiment Analyzer...")
            self.sentiment_analyzer = FinBertSentiment(device=-1)
        except Exception as e:
            print(f"⚠️ FinBERT Load Error: {e}")
            self.sentiment_analyzer = None

        # --- UPDATED PROMPT: Now accepts FinBERT input ---
        self.prompt = ChatPromptTemplate.from_template(
            """
You are a senior portfolio risk manager using Llama 3.
You are evaluating a portfolio with the following top holdings (Ticker: Weight): 
{sector_weights}

For your reference, here is the sector map for these tickers:
Technology: AAPL, MSFT, NVDA, INTC, CSCO, ADBE, CRM, ORCL, AVGO, ACN, AMD, TXN, QCOM
Financials: JPM, BAC, WFC, V, MA, PYPL, C, GS, BLK
Healthcare: UNH, JNJ, PFE, MRK, ABT
Energy: XOM, CVX, COP
Consumer Staples: PG, KO, PEP, COST
Consumer Discretionary: AMZN, TSLA, HD, LOW, MCD, NKE
Communications: GOOGL, META, DIS, NFLX, T, VZ, CMCSA
Industrials: HON, CAT

Context from financial news/reports:
{context}

Quantitative Sentiment Validation (from FinBERT model):
The overall statistical sentiment of the news context above is: {finbert_sentiment}

Task:
1. Analyze the holdings based ONLY on the sector map provided above. 
2. Assign a **Risk Score** (0.0 to 1.0) based on volatility, the provided context, and the FinBERT sentiment.
3. Provide a specific **Action Plan** to balance the portfolio. Suggest specific tickers to buy/sell based on the context.

Output strictly valid JSON with these keys:
- "regime_alert": A short headline about the market state (e.g., "High Volatility in Tech").
- "bias_warning": Specific critique of the current holdings.
- "actionable_advice": A direct suggestion on how to fix it (e.g., "Consider reducing AAPL by 5% and adding defensive stocks like PG or KO.").
- "risk_factor": A float between 0.0 and 1.0.

Do NOT output markdown blocks like ```json. Just the raw JSON string.
"""
        )

    def _retrieve(self, query: str) -> str:
        if not self.db:
            return "No external context available."
        try:
            docs = self.db.similarity_search(query, k=3)
            return "\n".join(d.page_content for d in docs)
        except Exception:
            return "Error retrieving documents."

    def _clean_json_string(self, json_str: str) -> str:
        json_str = re.sub(r"```json", "", json_str)
        json_str = re.sub(r"```", "", json_str)
        return json_str.strip()

    def evaluate(self, sector_weights: Dict[str, float], query_hint: str = "") -> Tuple[str, str, float]:
        if not sector_weights:
            return "No holdings", "Portfolio is empty.", 0.0

        dominant_sector = max(sector_weights, key=sector_weights.get)
        q = f"market outlook for {dominant_sector} and recession risks"
        context = self._retrieve(q)

        # --- NEW: Execute FinBERT Scoring ---
        finbert_score_str = "Neutral"
        if self.sentiment_analyzer and context != "No external context available.":
            # Score the first 3 sentences of the context
            snippets = context.split('.')[:3]
            scores = self.sentiment_analyzer.score_headlines(snippets)
            avg_score = np.mean(scores) if scores else 0.0
            
            if avg_score > 0.15:
                finbert_score_str = f"Positive (+{avg_score:.2f})"
            elif avg_score < -0.15:
                finbert_score_str = f"Negative ({avg_score:.2f})"
        # ------------------------------------

        chain = self.prompt | self.llm
        resp = chain.invoke({
            "context": context[:2000], 
            "sector_weights": sector_weights,
            "finbert_sentiment": finbert_score_str
        })
        
        try:
            clean_content = self._clean_json_string(resp.content)
            data = json.loads(clean_content)
            alert = data.get("regime_alert", "Market Normal")
            warning = f"{data.get('bias_warning', '')} \n\n👉 **Suggestion:** {data.get('actionable_advice', 'Diversify holdings.')}"
            risk_factor = float(data.get("risk_factor", 0.5))
        except Exception as e:
            print(f"LLM Parse Error: {e}")
            alert = "Analysis Failed"
            warning = f"Raw Output: {resp.content[:100]}..."
            risk_factor = 0.5
            
        return alert, warning, risk_factor
