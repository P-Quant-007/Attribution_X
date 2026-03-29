import os
import json
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct
)
from fastembed import TextEmbedding
#from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd

def _get_company_name(isin: str) -> str:
    """Look up company name from first 7 chars of ISIN."""
    try:
        map_path = Path(__file__).parent / "isin_company_map.csv"
        df = pd.read_csv(map_path)
        prefix = isin[:7].upper()
        match = df[df["ISIN_7_PREFIX"] == prefix]
        if not match.empty:
            return match.iloc[0]["Company"]
    except Exception:
        pass
    return "Unknown Issuer"


load_dotenv()

COLLECTION_NAME = "credit_events"
#EMBEDDING_MODEL  = "all-MiniLM-L6-v2"  # fast, 384-dim, works offline
EMBEDDING_MODEL  = "BAAI/bge-small-en-v1.5"  # fast ONNX model, no torch needed
VECTOR_SIZE      = 384

_model_cache = None

def get_embedding_model() -> TextEmbedding:
    global _model_cache
    if _model_cache is None:
        _model_cache = TextEmbedding(EMBEDDING_MODEL)
    return _model_cache

# ── Credit event corpus ───────────────────────────────────────────────────
# 20 carefully written events covering Indian credit stress 2015-2024
CREDIT_EVENTS = [
    {
        "id": 1,
        "date": "2018-09-21",
        "issuer": "IL&FS",
        "tag": "ILFS",
        "title": "IL&FS defaults on inter-corporate deposits",
        "text": (
            "Infrastructure Leasing & Financial Services (IL&FS) defaulted on "
            "short-term borrowings and inter-corporate deposits in September 2018, "
            "triggering a severe liquidity crisis across Indian credit markets. "
            "The group had accumulated over ₹91,000 crore in debt. The default "
            "caused a sharp widening of spreads across NBFCs and HFCs, with "
            "investors pulling back from the entire sector. DHFL, Reliance Capital, "
            "and other leveraged NBFCs saw immediate contagion effects with YTM "
            "spikes of 100-200 bps within days."
        ),
    },
    {
        "id": 2,
        "date": "2019-01-31",
        "issuer": "DHFL",
        "tag": "DHFL",
        "title": "Cobrapost expose triggers DHFL liquidity crisis",
        "text": (
            "A January 2019 investigative report alleged fund diversion at Dewan "
            "Housing Finance Corporation (DHFL), triggering a collapse in investor "
            "confidence. DHFL bonds, which had been trading at 9-10% YTM, saw "
            "yields spike to 12-15% within weeks as mutual funds and insurers "
            "rushed to exit positions. The company struggled to roll over commercial "
            "paper and NCDs, marking the beginning of a sustained liquidity crisis."
        ),
    },
    {
        "id": 3,
        "date": "2019-04-18",
        "issuer": "DHFL",
        "tag": "DHFL",
        "title": "DHFL halts repayments — acute default phase begins",
        "text": (
            "In April 2019, DHFL officially delayed interest payments on its bonds, "
            "triggering rating downgrades to default category by CRISIL and ICRA. "
            "Bond yields surged to 20-29% as the secondary market effectively "
            "ceased to function. Mutual funds holding DHFL paper were forced to "
            "side-pocket investments, impacting retail investors. This represents "
            "the peak stress period visible in CBRICS trade data with spreads "
            "exceeding 1,500-2,000 bps above GOI benchmark."
        ),
    },
    {
        "id": 4,
        "date": "2019-11-29",
        "issuer": "DHFL",
        "tag": "DHFL",
        "title": "DHFL admitted to insolvency under IBC",
        "text": (
            "DHFL became the first financial services company to be resolved under "
            "the Insolvency and Bankruptcy Code (IBC) in November 2019. The RBI "
            "superseded its board and appointed an administrator. Total creditor "
            "claims exceeded ₹87,000 crore. Piramal Capital eventually acquired "
            "DHFL in a resolution plan approved in 2021, with creditors receiving "
            "significant haircuts on their exposures."
        ),
    },
    {
        "id": 5,
        "date": "2020-03-05",
        "issuer": "Yes Bank",
        "tag": "YES_BANK",
        "title": "RBI places Yes Bank under moratorium",
        "text": (
            "The Reserve Bank of India placed Yes Bank under a moratorium on "
            "March 5, 2020, restricting withdrawals to ₹50,000 per depositor. "
            "The bank had accumulated ₹32,000+ crore in stressed assets. "
            "Yes Bank bonds saw YTMs spike to 20-29% as the moratorium was "
            "announced. A rescue plan led by SBI and other banks was executed "
            "within days, preventing a complete collapse, but AT1 bondholders "
            "faced full write-down of ₹8,415 crore."
        ),
    },
    {
        "id": 6,
        "date": "2019-09-30",
        "issuer": "Reliance Capital",
        "tag": "RELIANCE_CAP",
        "title": "Reliance Capital misses NCD payment",
        "text": (
            "Reliance Capital, part of the Anil Ambani group, began missing "
            "scheduled NCD repayments in late 2019. The company had significant "
            "exposure to group companies and illiquid assets. Rating agencies "
            "downgraded Reliance Capital paper to default, causing bond yields "
            "to widen sharply. The stress was compounded by regulatory action "
            "and eventually led to IBC proceedings in 2021."
        ),
    },
    {
        "id": 7,
        "date": "2021-11-29",
        "issuer": "Reliance Capital",
        "tag": "RELIANCE_CAP",
        "title": "RBI supersedes Reliance Capital board",
        "text": (
            "The RBI superseded the board of Reliance Capital in November 2021 "
            "and initiated insolvency proceedings under IBC. Total debt exceeded "
            "₹40,000 crore. This was the third large NBFC after IL&FS and DHFL "
            "to face RBI intervention. Bond yields remained elevated at 25-28% "
            "through the resolution process."
        ),
    },
    {
        "id": 8,
        "date": "2021-10-04",
        "issuer": "SREI Infrastructure",
        "tag": "SREI",
        "title": "RBI supersedes SREI Infrastructure board",
        "text": (
            "SREI Infrastructure Finance and SREI Equipment Finance had their "
            "boards superseded by the RBI in October 2021. The twin NBFCs had "
            "combined debt of approximately ₹32,000 crore. Significant exposure "
            "to stressed infrastructure projects and related-party transactions "
            "contributed to the collapse. Bond yields spiked to 12-15% range "
            "as the stress became apparent in trading data."
        ),
    },
    {
        "id": 9,
        "date": "2018-10-01",
        "issuer": "NBFC sector",
        "tag": "ILFS",
        "title": "NBFC liquidity crisis — IL&FS contagion spreads",
        "text": (
            "Following the IL&FS default, Indian credit markets experienced a "
            "broad-based liquidity crisis in October 2018. Mutual funds reduced "
            "exposure to NBFC commercial paper, causing short-term funding costs "
            "to spike. The RBI conducted open market operations to inject "
            "liquidity. NBFCs with high reliance on market borrowings including "
            "DHFL, Indiabulls, and Reliance Capital faced acute refinancing risk. "
            "Spreads over g-sec widened by 150-300 bps across the NBFC sector."
        ),
    },
    {
        "id": 10,
        "date": "2020-04-01",
        "issuer": "COVID-19",
        "tag": None,
        "title": "COVID-19 market dislocation — RBI emergency measures",
        "text": (
            "The COVID-19 pandemic caused severe market dislocation in March-April "
            "2020. RBI announced emergency rate cuts, LTRO, and moratorium on "
            "loan repayments. Credit spreads widened sharply across all rating "
            "categories. Already-stressed issuers like DHFL saw final trades at "
            "distressed levels. Mutual fund redemptions accelerated, with Franklin "
            "Templeton winding up six credit funds in April 2020 citing illiquidity."
        ),
    },
    {
        "id": 11,
        "date": "2021-04-23",
        "issuer": "Franklin Templeton",
        "tag": None,
        "title": "Franklin Templeton wind-up — credit fund contagion",
        "text": (
            "Franklin Templeton wound up six Indian debt mutual fund schemes in "
            "April 2020 citing inability to manage redemptions due to illiquid "
            "credit holdings. The affected schemes had combined AUM of ₹25,000 "
            "crore. Major holdings included DHFL, Yes Bank, Vodafone, and other "
            "stressed credits. This event accelerated the re-pricing of credit "
            "risk across the Indian bond market and triggered regulatory reforms "
            "to mutual fund liquidity management."
        ),
    },
    {
        "id": 12,
        "date": "2022-05-04",
        "issuer": "RBI",
        "tag": None,
        "title": "RBI emergency rate hike cycle begins",
        "text": (
            "The RBI began an aggressive rate hiking cycle in May 2022 with an "
            "off-cycle 40 bps repo rate hike, followed by further increases "
            "totalling 250 bps by February 2023. GOI benchmark yields rose from "
            "6.8% to 7.5%+, causing mark-to-market losses across bond portfolios. "
            "Credit spreads initially widened but subsequently compressed as "
            "higher-quality issuers attracted demand. The hiking cycle created "
            "duration risk and impacted NAV of fixed income funds significantly."
        ),
    },
    {
        "id": 13,
        "date": "2019-07-05",
        "issuer": "DHFL",
        "tag": "DHFL",
        "title": "DHFL bond trades at 25%+ — deep distress",
        "text": (
            "By July 2019, DHFL bonds were trading at yields exceeding 25% in "
            "sporadic secondary market transactions, reflecting near-zero recovery "
            "expectations. The company had failed to repay deposits and NCDs worth "
            "thousands of crores. Trades in CBRICS data for this period show "
            "extreme illiquidity with very few prints (trades) and enormous spread "
            "to benchmark of 1,800+ bps. The Isolation Forest model flags these "
            "as maximum anomaly score events."
        ),
    },
    {
        "id": 14,
        "date": "2018-08-01",
        "issuer": "DHFL",
        "tag": "DHFL",
        "title": "DHFL pre-stress — YTM begins rising",
        "text": (
            "Prior to the IL&FS shock, DHFL bonds were already showing early "
            "stress indicators in mid-2018. YTMs were rising from 9% toward 9.5% "
            "as concerns about the NBFC sector's asset quality grew. Short selling "
            "pressure and negative reports from foreign brokerages were circulating. "
            "This period shows d1 and z_score features beginning to drift positive "
            "in feature engineering, an early warning signal before the acute "
            "stress phase triggered by IL&FS."
        ),
    },
    {
        "id": 15,
        "date": "2019-03-01",
        "issuer": "DHFL",
        "tag": "DHFL",
        "title": "DHFL rating downgrade cascade",
        "text": (
            "Between January and March 2019, all major rating agencies downgraded "
            "DHFL paper multiple notches. CRISIL cut DHFL's long-term rating from "
            "AA to AA-, then to A+. ICRA and CARE followed. Each downgrade "
            "triggered forced selling by mandate-constrained investors including "
            "insurance companies and provident funds. Bond yields jumped 200-300 "
            "bps on each downgrade announcement, clearly visible as large d1 "
            "values in the feature engineering output."
        ),
    },
    {
        "id": 16,
        "date": "2016-01-01",
        "issuer": "RBI",
        "tag": None,
        "title": "RBI Asset Quality Review — NPA recognition cycle",
        "text": (
            "The RBI's Asset Quality Review of 2015-16 forced banks to recognise "
            "stressed assets as NPAs, leading to a broad reassessment of credit "
            "risk in the Indian market. While primarily affecting bank bonds, the "
            "AQR indirectly increased investor scrutiny of all leveraged borrowers "
            "including NBFCs. Spreads on lower-rated corporate bonds widened "
            "modestly during this period as the credit cycle turned."
        ),
    },
    {
        "id": 17,
        "date": "2020-05-26",
        "issuer": "DHFL",
        "tag": "DHFL",
        "title": "DHFL final trades before complete illiquidity",
        "text": (
            "The last recorded DHFL bond trades in CBRICS data appear in May 2020 "
            "at YTMs of 20%+, reflecting deep distress valuations by the few "
            "counterparties still willing to transact. By this point DHFL was "
            "under IBC proceedings with an administrator appointed. These trades "
            "represent forced sellers transacting at distressed prices, explaining "
            "the extreme anomaly scores assigned by the Isolation Forest model "
            "for this date."
        ),
    },
    {
        "id": 18,
        "date": "2018-09-01",
        "issuer": "Vodafone Idea",
        "tag": "VODAFONE",
        "title": "Vodafone Idea AGR dues — telecom sector stress",
        "text": (
            "Vodafone Idea faced existential stress following the Supreme Court's "
            "2019 ruling on Adjusted Gross Revenue (AGR) dues of ₹58,000+ crore. "
            "Bond yields on Vodafone Idea paper spiked to 25-29% as the market "
            "priced in near-certain default. The company repeatedly sought "
            "government relief and underwent multiple rounds of fundraising. "
            "This represents a sector-specific stress event distinct from NBFC "
            "contagion, driven by regulatory and legal risk."
        ),
    },
    {
        "id": 19,
        "date": "2021-01-01",
        "issuer": "Future Group",
        "tag": "FUTURE",
        "title": "Future Retail defaults on bond payments",
        "text": (
            "Future Retail and Future Enterprises defaulted on NCD payments in "
            "2021 following the collapse of the Amazon-Reliance deal dispute. "
            "The group had significant retail debt that could not be serviced "
            "as operations deteriorated during COVID-19. Bond yields spiked "
            "sharply and trading became illiquid. The episode demonstrates how "
            "operational stress combined with legal disputes can rapidly translate "
            "into credit events visible in bond market data."
        ),
    },
    {
        "id": 20,
        "date": "2023-03-01",
        "issuer": "Adani Group",
        "tag": None,
        "title": "Hindenburg report triggers Adani bond volatility",
        "text": (
            "The Hindenburg Research report in January 2023 alleged accounting "
            "fraud at Adani Group companies, triggering sharp selling in Adani "
            "bonds and equities. While Adani bonds did not default, spreads "
            "widened significantly and some overseas bonds traded at distressed "
            "levels briefly. The episode demonstrates how reputational and "
            "governance risk can rapidly translate to bond market stress, even "
            "for investment-grade issuers, and the importance of monitoring "
            "spread movements as early warning signals."
        ),
    },
]


def get_qdrant_client() -> QdrantClient:
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    if not url:
        raise ValueError("QDRANT_URL not set in .env")
    return QdrantClient(url=url, api_key=api_key)


#_model_cache = None

#def get_embedding_model() -> SentenceTransformer:
#    global _model_cache
#    if _model_cache is None:
#        _model_cache = SentenceTransformer(EMBEDDING_MODEL)
#    return _model_cache


def setup_collection(client: QdrantClient):
    """Create Qdrant collection and payload indexes if they don't exist."""
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )
        print(f"[qdrant] Collection '{COLLECTION_NAME}' created.")
    else:
        print(f"[qdrant] Collection '{COLLECTION_NAME}' already exists.")

    # Create payload index on 'tag' field for filtered search
    from qdrant_client.models import PayloadSchemaType
    try:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="tag",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        print(f"[qdrant] Payload index created on 'tag' field.")
    except Exception:
        pass  # Index may already exist


#def seed_events(client: QdrantClient, model: SentenceTransformer):
def seed_events(client: QdrantClient, model: TextEmbedding):
    """Embed and upload all credit events to Qdrant."""
    texts = [e["text"] for e in CREDIT_EVENTS]
    print(f"[qdrant] Embedding {len(texts)} events...")
    #embeddings = model.encode(texts, show_progress_bar=False)
    embeddings = list(model.embed(texts))

    points = []
    for event, embedding in zip(CREDIT_EVENTS, embeddings):
        points.append(PointStruct(
            id=event["id"],
            vector=embedding.tolist(),
            payload={
                "title":   event["title"],
                "text":    event["text"],
                "date":    event["date"],
                "issuer":  event["issuer"],
                "tag":     event["tag"],
            }
        ))

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"[qdrant] {len(points)} events seeded successfully.")


def retrieve_evidence(
    query: str,
    client: QdrantClient,
    #model: SentenceTransformer,
    model: TextEmbedding,
    top_k: int = 3,
    tag_filter: str = None,
) -> list[dict]:
    """
    Retrieve top-k most relevant events for a given anomaly query.

    query:      natural language description of the anomaly
    tag_filter: optional issuer tag to bias results (e.g. 'DHFL')
    """
    #query_vector = model.encode(query).tolist()
    query_vector = list(model.embed([query]))[0].tolist()

    from qdrant_client.models import Filter, FieldCondition, MatchValue

    search_filter = None
    if tag_filter:
        search_filter = Filter(
            must=[FieldCondition(
                key="tag",
                match=MatchValue(value=tag_filter)
            )]
        )

    try:
        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k,
            query_filter=search_filter,
            with_payload=True,
        )
    except Exception:
        # Fall back to unfiltered if index not ready
        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
        )

    # Fall back to unfiltered if tag filter returns nothing
    if not results and tag_filter:
        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
        )

    return [
        {
            "title":     r.payload["title"],
            "text":      r.payload["text"],
            "date":      r.payload["date"],
            "issuer":    r.payload["issuer"],
            "score":     round(r.score, 4),
        }
        for r in results
    ]

def compute_confidence_score(row: dict) -> float:
    """
    Compute a confidence score for an anomaly row.
    Formula: 0.4 × prints_norm + 0.3 × vol_norm + 0.3 × persistence_flag
    All inputs normalised to [0, 1].
    Returns score in [0, 1].
    """
    # Prints component — normalise against typical max of 20 trades/day
    prints = float(row.get("prints", 0) or 0)
    prints_norm = min(prints / 20.0, 1.0)

    # Volume component — vol_log typically 0-25, normalise against 20
    vol_log = float(row.get("vol_log", 0) or 0)
    vol_norm = min(vol_log / 20.0, 1.0)

    # Persistence flag — 1 if confirmed, 0 if not
    persistence = float(row.get("confirmed_anomaly", 0) or 0)

    score = (0.4 * prints_norm) + (0.3 * vol_norm) + (0.3 * persistence)
    return round(min(score, 1.0), 4)


def explain_anomaly(
    anomaly: dict,
    client: QdrantClient,
    #model: SentenceTransformer,
    model: TextEmbedding,
    anthropic_client=None,
) -> dict:
    """
    Generate a full explanation for an anomaly using:
    1. Evidence retrieved from Qdrant (always)
    2. LLM narrative from Anthropic Claude (if API key available)

    Returns dict with: evidence, confidence_score, explanation
    """
    # Build query from anomaly features
    isin       = anomaly.get("isin", "")
    date       = str(anomaly.get("date", ""))[:10]
    company    = _get_company_name(isin)
    
    avg_ytm    = anomaly.get("avg_ytm", 0)
    spread_bps = anomaly.get("spread_bps", 0)
    d1         = anomaly.get("d1", 0)
    z_score    = anomaly.get("z_score_21d", 0)

    query = (
    f"Issuer: {company} | ISIN {isin} on {date} showed YTM of {avg_ytm:.1f}% "
    f"with spread of {spread_bps:.0f}bps above benchmark, "
    f"1-day change of {d1:.0f}bps and z-score of {z_score:.1f}. "
    f"Indian credit market stress event default NBFC."
)

    # Retrieve evidence from Qdrant
    # Derive tag from ISIN prefix if not explicitly set
    # INE202 = DHFL, INE535 = IL&FS, INE528 = Yes Bank
    isin_tag_map = {
        "INE202": "DHFL",
        "INE535": "ILFS",
        "INE528": "YES_BANK",
        "INE013": "RELIANCE_CAP",
        "INE564": "DHFL",
    }
    tag = anomaly.get("stress_tag") or anomaly.get("tag")
    if not tag:
        for prefix, t in isin_tag_map.items():
            if isin.startswith(prefix):
                tag = t
                break
    evidence = retrieve_evidence(
        query, client, model,
        top_k=3,
        tag_filter=tag if tag else None
    )

    # Confidence score
    confidence = compute_confidence_score(anomaly)

    # LLM explanation
    explanation, ai_generated = _generate_llm_explanation(
        anomaly, evidence, anthropic_client, company=company
    )

    return {
        "isin":             isin,
        "date":             date,
        "avg_ytm":          avg_ytm,
        "spread_bps":       spread_bps,
        "confidence_score": confidence,
        "evidence":         evidence,
        "explanation":      explanation,
        "ai_generated":     ai_generated,
    }


def _generate_llm_explanation(
    anomaly: dict,
    evidence: list,
    anthropic_client=None,
    company: str = "Unknown Issuer",
) -> str:
    """
    Generate a concise explanation using Claude.
    Falls back to a deterministic template if no API key.
    """
    date       = str(anomaly.get("date", ""))[:10]
    avg_ytm    = anomaly.get("avg_ytm", 0)
    spread_bps = anomaly.get("spread_bps", 0)
    d1         = anomaly.get("d1", 0)
    z_score    = anomaly.get("z_score_21d", 0)
    confirmed  = anomaly.get("confirmed_anomaly", 0)
    cusum      = anomaly.get("cusum_signal", 0)

    evidence_text = "\n".join([
        f"- {e['title']} ({e['date']}): {e['text'][:200]}..."
        for e in evidence[:2]
    ])

    # Try Claude API
    if anthropic_client:
        try:
            prompt = f"""You are an expert Indian fixed income analyst.

An anomaly was detected in bond trading data:
- Issuer: {company}
- ISIN: {anomaly.get("isin", "")}
- Date: {date}
- YTM: {avg_ytm:.2f}%
- Spread to benchmark: {spread_bps:.0f} bps
- 1-day YTM change: {d1:.0f} bps
- 21-day z-score: {z_score:.2f}
- Confirmed by persistence filter: {"Yes" if confirmed else "No"}
- CUSUM regime shift: {"Yes" if cusum else "No"}

Relevant historical context:
{evidence_text}

Write a 3-sentence explanation of this anomaly for a fund manager. 
Be specific about the likely cause, the severity, and the market context.
Do not use bullet points. Be direct and factual."""

            #message = anthropic_client.messages.create(
            #    model="claude-haiku-4-5-20251001",
            message = anthropic_client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}]
            )
            return (message.content[0].text.strip(), True)
        except Exception as e:
            print(f"[evidence] LLM call failed: {e}, using template.")

    # Deterministic fallback — no API key needed
    severity = "extreme" if avg_ytm > 20 else "severe" if avg_ytm > 15 else "elevated"
    confirmed_text = "confirmed by the persistence filter" if confirmed else "flagged as a single-day event"
    cusum_text = " A CUSUM regime shift was also detected, indicating a sustained change in yield behaviour." if cusum else ""

    top_event = evidence[0]["title"] if evidence else "an Indian credit market stress event"

    return (
        f"Issuer: {company}. This anomaly represents {severity} credit stress with YTM of {avg_ytm:.1f}% "
        f"and a spread of {spread_bps:.0f} bps above the GOI benchmark, {confirmed_text}. "
        f"The most likely market context is: {top_event}.{cusum_text}",
        False
    )