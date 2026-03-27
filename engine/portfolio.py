import pandas as pd
import numpy as np
from pathlib import Path
from datetime import date


# ── Valid rating categories ────────────────────────────────────────────────
VALID_RATINGS = {
    "AAA", "AA+", "AA", "AA-", "A+", "A", "A-",
    "BBB+", "BBB", "BBB-", "BB+", "BB", "BB-",
    "B+", "B", "B-", "CCC", "CC", "C", "D",
    "NR", "UNRATED", "DEFAULT"
}

# ── Stress tag lookup by ISIN prefix ──────────────────────────────────────
ISIN_STRESS_MAP = {
    "INE202": "DHFL",
    "INE535": "ILFS",
    "INE020": "ILFS",
    "INE528": "YES_BANK",
    "INE013": "RELIANCE_CAP",
    "INE564": "DHFL",
}

REQUIRED_COLUMNS = {"isin", "coupon", "maturity_date", "face_value"}

COLUMN_ALIASES = {
    "ISIN": "isin",
    "Isin": "isin",
    "IssuerName": "issuer_name",
    "Issuer": "issuer_name",
    "Issuer Name": "issuer_name",
    "Coupon": "coupon",
    "Coupon Rate": "coupon",
    "MaturityDate": "maturity_date",
    "Maturity": "maturity_date",
    "Maturity Date": "maturity_date",
    "FaceValue": "face_value",
    "Face Value": "face_value",
    "Notional": "face_value",
    "Holding": "face_value",
    "Rating": "rating",
    "CallDate": "call_date",
    "Call Date": "call_date",
    "PutDate": "put_date",
    "Put Date": "put_date",
}


def load_portfolio(filepath: str | Path) -> pd.DataFrame:
    """
    Load and validate a portfolio file (CSV or XLSX).

    Required columns: isin, coupon, maturity_date, face_value
    Optional columns: issuer_name, rating, call_date, put_date

    Returns clean portfolio DataFrame with computed fields:
    - weight: % allocation by face value
    - years_to_maturity: from today
    - stress_tag: if ISIN belongs to a known stress issuer
    - is_anomaly_issuer: bool flag
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Portfolio file not found: {filepath}")

    suffix = filepath.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(filepath)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    # Normalise column names
    df = df.rename(columns=COLUMN_ALIASES)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Validate required columns
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"File has: {list(df.columns)}"
        )

    # Add optional columns if missing
    for col in ["issuer_name", "rating", "call_date", "put_date"]:
        if col not in df.columns:
            df[col] = None

    df = df.copy()

    # ── Clean ISIN ─────────────────────────────────────────────────────────
    df["isin"] = df["isin"].astype(str).str.strip().str.upper()
    invalid_isin = ~df["isin"].str.match(r"^[A-Z]{2}[A-Z0-9]{10}$")
    if invalid_isin.any():
        bad = df.loc[invalid_isin, "isin"].tolist()
        raise ValueError(f"Invalid ISINs found: {bad}")

    # ── Clean coupon ───────────────────────────────────────────────────────
    df["coupon"] = pd.to_numeric(df["coupon"], errors="coerce")
    if df["coupon"].isna().any():
        raise ValueError("Coupon must be numeric for all rows.")
    if not df["coupon"].between(0, 30).all():
        raise ValueError("Coupon values must be between 0 and 30%.")

    # ── Clean maturity date ────────────────────────────────────────────────
    df["maturity_date"] = pd.to_datetime(
        df["maturity_date"], format="mixed", dayfirst=True, errors="coerce"
    )
    if df["maturity_date"].isna().any():
        raise ValueError("maturity_date could not be parsed for all rows.")

    # ── Clean face value ───────────────────────────────────────────────────
    df["face_value"] = pd.to_numeric(df["face_value"], errors="coerce")
    if df["face_value"].isna().any() or (df["face_value"] <= 0).any():
        raise ValueError("face_value must be positive numeric for all rows.")

    # ── Clean optional dates ───────────────────────────────────────────────
    for col in ["call_date", "put_date"]:
        df[col] = pd.to_datetime(df[col], format="mixed", dayfirst=True, errors="coerce")

    # ── Clean rating ───────────────────────────────────────────────────────
    df["rating"] = df["rating"].fillna("NR").astype(str).str.strip().str.upper()

    # ── Clean issuer name ──────────────────────────────────────────────────
    df["issuer_name"] = (
        df["issuer_name"]
        .fillna("UNKNOWN")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # ── Computed fields ────────────────────────────────────────────────────
    today = pd.Timestamp.today().normalize()

    df["years_to_maturity"] = (
        (df["maturity_date"] - today).dt.days / 365.25
    ).round(4)

    # Warn on matured bonds
    matured = df["years_to_maturity"] <= 0
    if matured.any():
        print(f"[portfolio] Warning: {matured.sum()} bond(s) have already matured.")

    # Portfolio weight by face value
    total_fv = df["face_value"].sum()
    df["weight"] = (df["face_value"] / total_fv * 100).round(4)

    # Stress tag from ISIN prefix
    def get_stress_tag(isin: str) -> str | None:
        for prefix, tag in ISIN_STRESS_MAP.items():
            if isin.startswith(prefix):
                return tag
        return None

    df["stress_tag"] = df["isin"].apply(get_stress_tag)
    df["is_stress_issuer"] = df["stress_tag"].notna()

    # Sort by face value descending
    df = df.sort_values("face_value", ascending=False).reset_index(drop=True)

    print(f"[portfolio] Loaded {len(df)} holdings | "
          f"Total AUM: ₹{total_fv:,.0f} Lacs | "
          f"Stress issuers: {df['is_stress_issuer'].sum()}")

    return df


def portfolio_summary(df: pd.DataFrame) -> dict:
    """Return a summary dict for API response and display."""
    return {
        "total_holdings":     len(df),
        "total_aum_lacs":     round(float(df["face_value"].sum()), 2),
        "stress_holdings":    int(df["is_stress_issuer"].sum()),
        "stress_aum_pct":     round(
            float(df.loc[df["is_stress_issuer"], "face_value"].sum()
                  / df["face_value"].sum() * 100), 2
        ),
        "avg_coupon":         round(float(df["coupon"].mean()), 4),
        "avg_maturity_years": round(float(df["years_to_maturity"].mean()), 2),
        "rating_breakdown":   df["rating"].value_counts().to_dict(),
        "holdings": df[[
            "isin", "issuer_name", "coupon", "maturity_date",
            "face_value", "weight", "rating",
            "years_to_maturity", "stress_tag", "is_stress_issuer"
        ]].assign(
            maturity_date=df["maturity_date"].astype(str)
        ).to_dict(orient="records"),
    }