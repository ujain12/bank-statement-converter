import streamlit as st
import pdfplumber
import pandas as pd
import json
import base64
import io
import re
import subprocess
import os
import time
from groq import Groq
from PIL import Image

# ── Config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Bank Statement → QuickBooks", layout="wide")

TEXT_MODEL = "llama-3.3-70b-versatile"
VISION_MODEL = "llama-4-scout-17b-16e-instruct"

EXTRACTION_PROMPT = """You are a bank statement parser. Extract ALL transactions from this bank statement.

Return ONLY a valid JSON array. Each transaction object must have:
- "date": string in MM/DD/YYYY format (infer the year from context if not shown; use the statement period year)
- "description": string (the merchant/payee/description as shown)
- "amount": number (NEGATIVE for debits/withdrawals/purchases/fees, POSITIVE for credits/deposits/interest)
- "check_number": string or null (if a check number is mentioned)
- "type": string — one of "Deposit", "Withdrawal", "Check", "Fee", "Interest", "Transfer", "Other"

Rules:
1. Extract EVERY transaction. Do not skip any.
2. POS PURCHASE, ATM WITHDRAWAL, CHECK, SERVICE CHARGE = negative amounts (debits)
3. PREAUTHORIZED CREDIT, INTEREST CREDIT, DEPOSIT = positive amounts (credits)
4. If the statement shows separate Debit/Credit columns, use those to determine sign.
5. Do NOT include summary lines (beginning balance, ending balance, averages).
6. Return ONLY the JSON array — no markdown, no explanation, no code fences."""


def extract_text_from_pdf(pdf_file) -> str:
    text_parts = []
    with pdfplumber.open(pdf_file) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text:
                text_parts.append(f"--- Page {i+1} ---\n{page_text}")
    return "\n\n".join(text_parts)


def extract_tables_from_pdf(pdf_file) -> str:
    table_parts = []
    pdf_file.seek(0)
    with pdfplumber.open(pdf_file) as pdf:
        for i, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            for j, table in enumerate(tables):
                if table:
                    table_str = f"Table {j+1} on Page {i+1}:\n"
                    for row in table:
                        table_str += " | ".join(str(c) if c else "" for c in row) + "\n"
                    table_parts.append(table_str)
    return "\n\n".join(table_parts)


def pdf_to_base64_images(pdf_bytes: bytes) -> list[str]:
    """Convert PDF pages to base64 JPEG strings."""
    tmp_dir = "/tmp/stmt_pages"
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_pdf = os.path.join(tmp_dir, "input.pdf")
    with open(tmp_pdf, "wb") as f:
        f.write(pdf_bytes)
    subprocess.run(
        ["pdftoppm", "-jpeg", "-r", "200", tmp_pdf, os.path.join(tmp_dir, "page")],
        check=True, capture_output=True,
    )
    images_b64 = []
    page_files = sorted(
        f for f in os.listdir(tmp_dir) if f.startswith("page") and f.endswith(".jpg")
    )
    for pf in page_files:
        path = os.path.join(tmp_dir, pf)
        with open(path, "rb") as img:
            images_b64.append(base64.standard_b64encode(img.read()).decode())
        os.remove(path)
    os.remove(tmp_pdf)
    return images_b64


def parse_with_groq_text(text: str, table_text: str, api_key: str) -> list[dict]:
    """Parse transactions from extracted text via Groq."""
    client = Groq(api_key=api_key)
    combined = f"EXTRACTED TEXT:\n{text}"
    if table_text.strip():
        combined += f"\n\nEXTRACTED TABLES:\n{table_text}"

    response = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[
            {"role": "user", "content": f"{EXTRACTION_PROMPT}\n\n{combined}"}
        ],
        temperature=0,
        max_tokens=4096,
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


def parse_with_groq_vision(page_images_b64: list[str], api_key: str) -> list[dict]:
    """Parse transactions from page images via Groq vision (Llama 4 Scout)."""
    client = Groq(api_key=api_key)

    content = []
    for i, img_b64 in enumerate(page_images_b64):
        content.append({"type": "text", "text": f"Page {i+1} of the bank statement:"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
        })
    content.append({"type": "text", "text": EXTRACTION_PROMPT})

    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[{"role": "user", "content": content}],
        temperature=0,
        max_tokens=4096,
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


def to_quickbooks_df(transactions: list[dict], detailed: bool = False) -> pd.DataFrame:
    rows = []
    for t in transactions:
        row = {
            "Date": t.get("date", ""),
            "Description": t.get("description", ""),
            "Amount": t.get("amount", 0),
        }
        if detailed:
            row["Check Number"] = t.get("check_number", "") or ""
            row["Type"] = t.get("type", "Other")
        rows.append(row)
    cols = ["Date", "Description", "Amount", "Check Number", "Type"] if detailed else ["Date", "Description", "Amount"]
    return pd.DataFrame(rows, columns=cols)


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Transactions")
        ws = writer.sheets["Transactions"]
        for col_cells in ws.columns:
            max_len = max(len(str(cell.value or "")) for cell in col_cells) + 2
            ws.column_dimensions[col_cells[0].column_letter].width = min(max_len, 50)
    return buf.getvalue()


# ── UI ──────────────────────────────────────────────────────────────────
st.title("🏦 Bank Statement → QuickBooks Converter")
st.caption("Upload any bank statement PDF — text-based or scanned. Powered by Groq (free, no credit card).")

with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Get free at https://console.groq.com/keys",
    )
    detailed_export = st.checkbox("Include Check # and Type columns", value=True)
    force_vision = st.checkbox(
        "Force vision mode (use for scanned PDFs)",
        value=False,
        help="Bypasses text extraction and sends page images to Llama 4 Scout",
    )
    st.divider()
    st.markdown("""
    **💡 Free API Key**  
    1. Go to [console.groq.com](https://console.groq.com)  
    2. Sign up (Google/GitHub login)  
    3. Go to **API Keys** → **Create**  
    4. Paste it here  
    
    **Free limits:** 30 req/min, 14,400 req/day  
    
    **Models used:**  
    - Text PDFs → Llama 3.3 70B  
    - Scanned PDFs → Llama 4 Scout (vision)  
    """)
    st.divider()
    st.markdown("""
    **QuickBooks Import Steps**  
    1. **Banking → Upload Transactions**  
    2. Select your bank account  
    3. Upload the `.xlsx` or `.csv`  
    4. Map columns if prompted  
    5. Review & accept transactions
    """)

uploaded = st.file_uploader("Upload Bank Statement PDF", type=["pdf"])

if uploaded and api_key:
    pdf_bytes = uploaded.read()

    # Step 1: Try text extraction
    text_mode = False
    pdf_text, table_text = "", ""
    if not force_vision:
        with st.spinner("Extracting text from PDF..."):
            pdf_text = extract_text_from_pdf(io.BytesIO(pdf_bytes))
            table_text = extract_tables_from_pdf(io.BytesIO(pdf_bytes))
            text_mode = len(pdf_text.strip()) > 50

    if text_mode:
        st.info(f"📄 Text-based PDF detected — using **{TEXT_MODEL}** for parsing.")
    else:
        st.info(f"🖼️ Scanned/image PDF detected — using **{VISION_MODEL}** (vision) to read pages.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Source Preview")
        if text_mode:
            st.text_area(
                "Extracted text",
                pdf_text[:3000] + ("..." if len(pdf_text) > 3000 else ""),
                height=300,
            )
        else:
            with st.spinner("Rasterizing PDF pages..."):
                page_images_b64 = pdf_to_base64_images(pdf_bytes)
            for i, img in enumerate(page_images_b64):
                st.image(base64.b64decode(img), caption=f"Page {i+1}", use_container_width=True)

    with col2:
        st.subheader("Parsed Transactions")
        with st.spinner("Parsing transactions..."):
            try:
                if text_mode:
                    transactions = parse_with_groq_text(pdf_text, table_text, api_key)
                else:
                    transactions = parse_with_groq_vision(page_images_b64, api_key)
                st.success(f"Extracted **{len(transactions)}** transactions")
            except json.JSONDecodeError as e:
                st.error(f"Failed to parse response as JSON: {e}")
                st.stop()
            except Exception as e:
                st.error(f"API error: {e}")
                st.stop()

        df = to_quickbooks_df(transactions, detailed=detailed_export)
        st.dataframe(df, use_container_width=True, height=300)

    # Summary
    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    total_credits = df[df["Amount"] > 0]["Amount"].sum()
    total_debits = df[df["Amount"] < 0]["Amount"].sum()
    c1.metric("Total Credits", f"${total_credits:,.2f}")
    c2.metric("Total Debits", f"${abs(total_debits):,.2f}")
    c3.metric("Net Change", f"${total_credits + total_debits:,.2f}")
    c4.metric("Transactions", len(df))

    # Downloads
    st.divider()
    dcol1, dcol2 = st.columns(2)
    with dcol1:
        st.download_button(
            "⬇️ Download Excel (.xlsx)",
            data=to_excel_bytes(df),
            file_name="quickbooks_import.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    with dcol2:
        st.download_button(
            "⬇️ Download CSV",
            data=df.to_csv(index=False).encode(),
            file_name="quickbooks_import.csv",
            mime="text/csv",
        )

elif uploaded and not api_key:
    st.warning("Enter your Groq API key in the sidebar. Get one free at console.groq.com")
