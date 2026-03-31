import streamlit as st
import pdfplumber
import pandas as pd
import json
import base64
import io
import re
import subprocess
import os
import pytesseract
from PIL import Image
from groq import Groq

# ── Config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Bank Statement → QuickBooks", layout="wide")

TEXT_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are a bank statement transaction extractor. Your ONLY job is to read bank statement data and return a JSON object.

You must return a JSON object with this EXACT structure:
{
  "opening_balance": <number or null>,
  "closing_balance": <number or null>,
  "statement_date_range": "<string like 'January 2026' or null>",
  "transactions": [
    {
      "date": "MM/DD/YYYY",
      "description": "...",
      "amount": <number>,
      "check_number": "<string or null>",
      "type": "Deposit|Withdrawal|Check|Fee|Interest|Transfer|Other"
    }
  ]
}

CRITICAL RULES FOR AMOUNTS:
- Every debit (purchase, withdrawal, check paid, fee, charge, DR) must be NEGATIVE
- Every credit (deposit, interest, refund, transfer in, CR) must be POSITIVE
- Read numbers VERY carefully. Count every digit. 80000000.00 is NOT the same as 8000000.00
- If a column says "WITHDRAWAL (DR)" or "Debit", the amount must be NEGATIVE
- If a column says "DEPOSIT (CR)" or "Credit", the amount must be POSITIVE

CRITICAL RULES FOR BALANCES:
- Opening Balance / Beginning Balance = opening_balance (always positive)
- Closing Balance / Ending Balance = closing_balance (always positive)
- Do NOT include Opening Balance or Closing Balance as transactions

EXAMPLE:
Given this statement:
  Opening Balance: 84626827.19 Cr
  01-01-2026  TO 01/10074-DBJAYA  WITHDRAWAL: 80000000.00
  Closing Balance: 4626827.19 Cr

Return:
{
  "opening_balance": 84626827.19,
  "closing_balance": 4626827.19,
  "statement_date_range": "January 2026",
  "transactions": [
    {"date": "01/01/2026", "description": "TO 01/10074-DBJAYA", "amount": -80000000.00, "check_number": null, "type": "Withdrawal"}
  ]
}

Verify: 84626827.19 + (-80000000.00) = 4626827.19 ✓

Return ONLY the JSON object. No markdown, no code fences, no explanation."""


def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from text-based PDF using pdfplumber."""
    text_parts = []
    with pdfplumber.open(pdf_file) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text:
                text_parts.append(f"--- Page {i+1} ---\n{page_text}")
    return "\n\n".join(text_parts)


def extract_tables_from_pdf(pdf_file) -> str:
    """Extract tables from PDF."""
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


def pdf_to_images(pdf_bytes: bytes) -> list[str]:
    """Convert PDF pages to image files, return file paths."""
    tmp_dir = "/tmp/stmt_pages"
    os.makedirs(tmp_dir, exist_ok=True)
    for f in os.listdir(tmp_dir):
        os.remove(os.path.join(tmp_dir, f))
    tmp_pdf = os.path.join(tmp_dir, "input.pdf")
    with open(tmp_pdf, "wb") as f:
        f.write(pdf_bytes)
    subprocess.run(
        ["pdftoppm", "-jpeg", "-r", "300", tmp_pdf, os.path.join(tmp_dir, "page")],
        check=True, capture_output=True,
    )
    page_files = sorted(
        os.path.join(tmp_dir, f)
        for f in os.listdir(tmp_dir)
        if f.startswith("page") and f.endswith(".jpg")
    )
    return page_files


def ocr_images(image_paths: list[str]) -> str:
    """Run OCR on page images to extract text."""
    text_parts = []
    for i, path in enumerate(image_paths):
        img = Image.open(path)
        # Use psm 6 (uniform block of text) for better table extraction
        ocr_text = pytesseract.image_to_string(img, config="--psm 6")
        if ocr_text.strip():
            text_parts.append(f"--- Page {i+1} (OCR) ---\n{ocr_text}")
    return "\n\n".join(text_parts)


def extract_json_from_response(raw: str) -> dict:
    """Robustly extract JSON object from model response."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*\n?", "", raw)
    raw = re.sub(r"\n?\s*```$", "", raw)
    raw = raw.strip()

    # Try direct parse
    try:
        result = json.loads(raw)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Find outermost JSON object
    brace_start = raw.find("{")
    brace_end = raw.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        try:
            result = json.loads(raw[brace_start : brace_end + 1])
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    # Fallback: look for JSON array (old format)
    bracket_start = raw.find("[")
    bracket_end = raw.rfind("]")
    if bracket_start != -1 and bracket_end != -1 and bracket_end > bracket_start:
        try:
            arr = json.loads(raw[bracket_start : bracket_end + 1])
            if isinstance(arr, list):
                return {"opening_balance": None, "closing_balance": None, "transactions": arr}
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not find valid JSON in response. Raw:\n{raw[:500]}")


def parse_with_groq(text: str, api_key: str) -> dict:
    """Send text to Llama 3.3 70B for parsing."""
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Extract all transactions from this bank statement. Return ONLY the JSON object.\n\n{text}"},
        ],
        temperature=0,
        max_tokens=8000,
    )
    return extract_json_from_response(response.choices[0].message.content)


def validate_balance(
    transactions: list[dict],
    opening_balance: float | None,
    closing_balance: float | None,
) -> list[str]:
    """Validate that opening + sum(transactions) = closing balance."""
    warnings = []

    if opening_balance is None or closing_balance is None:
        warnings.append("Could not extract opening/closing balance — skipping balance validation.")
        return warnings

    total = sum(t.get("amount", 0) for t in transactions)
    calculated_closing = opening_balance + total
    diff = abs(calculated_closing - closing_balance)

    if diff < 0.02:  # within rounding tolerance
        return []

    warnings.append(
        f"**Balance mismatch detected!**\n"
        f"- Opening Balance: **{opening_balance:,.2f}**\n"
        f"- Sum of Transactions: **{total:,.2f}**\n"
        f"- Expected Closing: **{calculated_closing:,.2f}**\n"
        f"- Actual Closing: **{closing_balance:,.2f}**\n"
        f"- Difference: **{diff:,.2f}**\n\n"
        f"This usually means an amount was misread (wrong number of digits) or a transaction was skipped."
    )
    return warnings


def validate_transactions(transactions: list[dict]) -> list[str]:
    """Check individual transactions for obvious issues."""
    warnings = []
    for i, t in enumerate(transactions):
        amt = t.get("amount", 0)
        desc = (t.get("description") or "").upper()
        date = t.get("date", "")

        debit_kw = ["PURCHASE", "WITHDRAWAL", "CHECK", "CHARGE", "FEE", "DEBIT", "ATM", "PAID"]
        credit_kw = ["CREDIT", "DEPOSIT", "INTEREST", "REFUND"]

        if any(k in desc for k in debit_kw) and amt > 0:
            warnings.append(f"Row {i+1}: \"{t.get('description')}\" looks like a debit but amount is positive ({amt:,.2f})")
        if any(k in desc for k in credit_kw) and "SERVICE" not in desc and amt < 0:
            warnings.append(f"Row {i+1}: \"{t.get('description')}\" looks like a credit but amount is negative ({amt:,.2f})")

        if date and not re.match(r"\d{2}/\d{2}/\d{4}", date):
            warnings.append(f"Row {i+1}: Date \"{date}\" not in MM/DD/YYYY format")

        if amt == 0:
            warnings.append(f"Row {i+1}: \"{t.get('description')}\" has zero amount")

    return warnings


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
st.caption("Upload any bank statement PDF. Uses OCR + AI parsing + balance validation for accurate results.")

with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Get free at https://console.groq.com/keys",
    )
    detailed_export = st.checkbox("Include Check # and Type columns", value=True)
    st.divider()
    st.markdown("""
    **How it works**  
    1. PDF → images (pdftoppm at 300 DPI)  
    2. Images → text (Tesseract OCR)  
    3. Text → structured JSON (Llama 3.3 70B)  
    4. Balance validation catches digit errors  
    5. You review & edit before export  
    """)
    st.divider()
    st.markdown("""
    **💡 Free Groq API Key**  
    [console.groq.com](https://console.groq.com) → Sign up → API Keys → Create  
    """)
    st.divider()
    st.markdown("""
    **QuickBooks Import**  
    Banking → Upload Transactions → Upload `.xlsx` or `.csv`
    """)

uploaded = st.file_uploader("Upload Bank Statement PDF", type=["pdf"])

if uploaded and api_key:
    pdf_bytes = uploaded.read()

    # ── Step 1: Extract text (pdfplumber first, then OCR fallback) ──
    with st.spinner("Step 1/3: Extracting text from PDF..."):
        pdf_text = extract_text_from_pdf(io.BytesIO(pdf_bytes))
        table_text = extract_tables_from_pdf(io.BytesIO(pdf_bytes))
        has_text = len(pdf_text.strip()) > 50

        # Always do OCR for scanned PDFs, optionally combine for text PDFs
        page_image_paths = pdf_to_images(pdf_bytes)
        ocr_text = ocr_images(page_image_paths)

    # Combine all extracted text for best results
    combined_text = ""
    if has_text:
        combined_text = f"=== PDFPLUMBER TEXT ===\n{pdf_text}\n\n"
        if table_text.strip():
            combined_text += f"=== PDFPLUMBER TABLES ===\n{table_text}\n\n"
        st.info(f"📄 Text-based PDF. Using pdfplumber text + OCR verification → **{TEXT_MODEL}**")
    else:
        st.info(f"🖼️ Scanned PDF. Using Tesseract OCR → **{TEXT_MODEL}**")

    if ocr_text.strip():
        combined_text += f"=== OCR TEXT ===\n{ocr_text}"

    # Show source
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Source Preview")
        tab1, tab2 = st.tabs(["📄 Extracted Text", "🖼️ Page Images"])
        with tab1:
            st.text_area("Combined text sent to AI", combined_text[:4000] + ("..." if len(combined_text) > 4000 else ""), height=300)
        with tab2:
            for i, path in enumerate(page_image_paths):
                st.image(path, caption=f"Page {i+1}", use_container_width=True)

    # ── Step 2: Parse with Llama ──
    with col2:
        st.subheader("Parsed Transactions")
        with st.spinner("Step 2/3: AI is parsing transactions..."):
            try:
                result = parse_with_groq(combined_text, api_key)
                transactions = result.get("transactions", [])
                opening_bal = result.get("opening_balance")
                closing_bal = result.get("closing_balance")
                date_range = result.get("statement_date_range", "")

                st.success(f"Extracted **{len(transactions)}** transactions")
                if opening_bal is not None:
                    st.caption(f"Opening: **{opening_bal:,.2f}** → Closing: **{closing_bal:,.2f}**" + (f" | Period: {date_range}" if date_range else ""))
            except (json.JSONDecodeError, ValueError) as e:
                st.error(f"Failed to parse response: {e}")
                st.stop()
            except Exception as e:
                if "429" in str(e):
                    st.error("Rate limit hit. Wait 30 seconds and retry.")
                else:
                    st.error(f"API error: {e}")
                st.stop()

    # ── Step 3: Validation ──
    balance_warnings = validate_balance(transactions, opening_bal, closing_bal)
    txn_warnings = validate_transactions(transactions)
    all_warnings = balance_warnings + txn_warnings

    if balance_warnings:
        st.error("🔴 Balance Validation Failed")
        for w in balance_warnings:
            st.markdown(w)
    elif opening_bal is not None and closing_bal is not None:
        st.success("✅ Balance validation passed — opening + transactions = closing balance")

    if txn_warnings:
        with st.expander(f"⚠️ {len(txn_warnings)} transaction warning(s)", expanded=False):
            for w in txn_warnings:
                st.warning(w)

    # ── Step 4: Editable table ──
    rows = []
    for t in transactions:
        row = {
            "Date": t.get("date", ""),
            "Description": t.get("description", ""),
            "Amount": float(t.get("amount", 0)),
        }
        if detailed_export:
            row["Check Number"] = str(t.get("check_number", "") or "")
            row["Type"] = t.get("type", "Other")
        rows.append(row)

    cols = ["Date", "Description", "Amount"]
    if detailed_export:
        cols += ["Check Number", "Type"]
    df = pd.DataFrame(rows, columns=cols)

    st.subheader("✏️ Review & Edit Transactions")
    st.caption("Fix any errors below. Add/delete rows as needed. The export uses YOUR edits.")

    col_config = {
        "Date": st.column_config.TextColumn("Date", help="MM/DD/YYYY"),
        "Description": st.column_config.TextColumn("Description", width="large"),
        "Amount": st.column_config.NumberColumn("Amount", help="Negative = debit, Positive = credit", format="%.2f"),
    }
    if detailed_export:
        col_config["Type"] = st.column_config.SelectboxColumn(
            "Type",
            options=["Deposit", "Withdrawal", "Check", "Fee", "Interest", "Transfer", "Other"],
        )

    edited_df = st.data_editor(df, use_container_width=True, height=400, num_rows="dynamic", column_config=col_config)

    # ── Step 5: Summary (from edited data) ──
    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    total_credits = edited_df[edited_df["Amount"] > 0]["Amount"].sum()
    total_debits = edited_df[edited_df["Amount"] < 0]["Amount"].sum()
    c1.metric("Total Credits", f"${total_credits:,.2f}")
    c2.metric("Total Debits", f"${abs(total_debits):,.2f}")
    c3.metric("Net Change", f"${total_credits + total_debits:,.2f}")
    c4.metric("Transactions", len(edited_df))

    # Re-validate with edits
    if opening_bal is not None and closing_bal is not None:
        edited_total = edited_df["Amount"].sum()
        edited_closing = opening_bal + edited_total
        diff = abs(edited_closing - closing_bal)
        if diff < 0.02:
            st.success(f"✅ After edits: Opening ({opening_bal:,.2f}) + Transactions ({edited_total:,.2f}) = {edited_closing:,.2f} matches Closing ({closing_bal:,.2f})")
        else:
            st.warning(f"⚠️ After edits: Opening ({opening_bal:,.2f}) + Transactions ({edited_total:,.2f}) = {edited_closing:,.2f} — expected {closing_bal:,.2f} (diff: {diff:,.2f})")

    # ── Step 6: Downloads ──
    st.divider()
    dcol1, dcol2 = st.columns(2)
    with dcol1:
        st.download_button(
            "⬇️ Download Excel (.xlsx)",
            data=to_excel_bytes(edited_df),
            file_name="quickbooks_import.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    with dcol2:
        st.download_button(
            "⬇️ Download CSV",
            data=edited_df.to_csv(index=False).encode(),
            file_name="quickbooks_import.csv",
            mime="text/csv",
        )

elif uploaded and not api_key:
    st.warning("Enter your Groq API key in the sidebar. Get one free at console.groq.com")
