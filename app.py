import streamlit as st
import pdfplumber
import pandas as pd
import json
import base64
import io
import re
import subprocess
import os
from groq import Groq

# ── Config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Bank Statement → QuickBooks", layout="wide")

TEXT_MODEL = "llama-3.3-70b-versatile"
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

SYSTEM_PROMPT = """You are a bank statement transaction extractor. Your ONLY job is to read bank statements and return a JSON array of transactions.

CRITICAL RULES:
- Return ONLY a JSON array starting with [ and ending with ]
- No explanations, no markdown, no code fences, no text before or after the JSON
- Every debit (purchase, withdrawal, check, fee, charge) must be NEGATIVE
- Every credit (deposit, interest, refund, transfer in) must be POSITIVE
- Use MM/DD/YYYY date format
- Skip summary/balance lines — only extract individual transactions

Here is an example. Given this bank statement text:

Date  Description           Debit    Credit   Balance
10/02 POS PURCHASE          4.23              65.73
10/03 PREAUTHORIZED CREDIT           763.01   828.74
10/05 CHECK 1234            9.98              807.08
11/09 INTEREST CREDIT                .26      598.71
11/09 SERVICE CHARGE        12.00             586.71

You must return:
[
  {"date": "10/02/2009", "description": "POS PURCHASE", "amount": -4.23, "check_number": null, "type": "Withdrawal"},
  {"date": "10/03/2009", "description": "PREAUTHORIZED CREDIT", "amount": 763.01, "check_number": null, "type": "Deposit"},
  {"date": "10/05/2009", "description": "CHECK 1234", "amount": -9.98, "check_number": "1234", "type": "Check"},
  {"date": "11/09/2009", "description": "INTEREST CREDIT", "amount": 0.26, "check_number": null, "type": "Interest"},
  {"date": "11/09/2009", "description": "SERVICE CHARGE", "amount": -12.00, "check_number": null, "type": "Fee"}
]

IMPORTANT:
- Debits/purchases/withdrawals/checks/fees → NEGATIVE amount (e.g., -4.23)
- Credits/deposits/interest → POSITIVE amount (e.g., 763.01)
- If a check number appears in the description like "CHECK 1234", extract "1234" as check_number
- type must be one of: "Deposit", "Withdrawal", "Check", "Fee", "Interest", "Transfer", "Other"
- Do NOT include beginning balance, ending balance, or summary rows
- Extract ALL transactions, do not skip any"""


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
    tmp_dir = "/tmp/stmt_pages"
    os.makedirs(tmp_dir, exist_ok=True)
    # Clean up any leftover files
    for f in os.listdir(tmp_dir):
        os.remove(os.path.join(tmp_dir, f))
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
    return images_b64


def extract_json_from_response(raw: str) -> list[dict]:
    """Robustly extract JSON array from model response, even if wrapped in text."""
    raw = raw.strip()
    # Strip markdown code fences
    raw = re.sub(r"^```(?:json)?\s*\n?", "", raw)
    raw = re.sub(r"\n?\s*```$", "", raw)
    raw = raw.strip()

    # Try direct parse first
    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Find the outermost JSON array in the response
    bracket_start = raw.find("[")
    bracket_end = raw.rfind("]")
    if bracket_start != -1 and bracket_end != -1 and bracket_end > bracket_start:
        try:
            result = json.loads(raw[bracket_start : bracket_end + 1])
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not find valid JSON array in response. Raw output:\n{raw[:500]}")


def parse_with_groq_text(text: str, table_text: str, api_key: str) -> list[dict]:
    client = Groq(api_key=api_key)
    user_msg = f"Extract all transactions from this bank statement and return ONLY a JSON array:\n\n{text}"
    if table_text.strip():
        user_msg += f"\n\nTABLE DATA:\n{table_text}"

    response = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
        max_tokens=8000,
    )
    return extract_json_from_response(response.choices[0].message.content)


def parse_with_groq_vision(page_images_b64: list[str], api_key: str) -> list[dict]:
    client = Groq(api_key=api_key)

    content = []
    for i, img_b64 in enumerate(page_images_b64):
        content.append({"type": "text", "text": f"Page {i+1}:"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
        })
    content.append({
        "type": "text",
        "text": "Extract ALL transactions from this bank statement. Return ONLY a JSON array. Every debit/purchase/withdrawal/check/fee must be NEGATIVE. Every credit/deposit/interest must be POSITIVE.",
    })

    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        temperature=0,
        max_tokens=8000,
    )
    return extract_json_from_response(response.choices[0].message.content)


def validate_transactions(transactions: list[dict]) -> list[str]:
    """Return list of warning messages for suspicious data."""
    warnings = []
    if not transactions:
        warnings.append("No transactions were extracted.")
        return warnings

    for i, t in enumerate(transactions):
        amt = t.get("amount", 0)
        desc = t.get("description", "").upper()
        date = t.get("date", "")

        # Check sign correctness
        debit_keywords = ["PURCHASE", "WITHDRAWAL", "CHECK", "CHARGE", "FEE", "DEBIT", "ATM"]
        credit_keywords = ["CREDIT", "DEPOSIT", "INTEREST", "REFUND"]

        if any(k in desc for k in debit_keywords) and amt > 0:
            warnings.append(f"Row {i+1}: \"{t.get('description')}\" looks like a debit but amount is positive ({amt})")
        if any(k in desc for k in credit_keywords) and "SERVICE" not in desc and amt < 0:
            warnings.append(f"Row {i+1}: \"{t.get('description')}\" looks like a credit but amount is negative ({amt})")

        # Check date format
        if date and not re.match(r"\d{2}/\d{2}/\d{4}", date):
            warnings.append(f"Row {i+1}: Date \"{date}\" is not in MM/DD/YYYY format")

        # Check zero amounts
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
st.caption("Upload any bank statement PDF — text-based or scanned. Powered by Groq (free).")

with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Get free at https://console.groq.com/keys",
    )
    detailed_export = st.checkbox("Include Check # and Type columns", value=True)
    force_vision = st.checkbox(
        "Force vision mode (for scanned PDFs)",
        value=False,
    )
    st.divider()
    st.markdown("""
    **💡 Free API Key**  
    1. Go to [console.groq.com](https://console.groq.com)  
    2. Sign up (Google/GitHub login)  
    3. **API Keys** → **Create**  
    4. Paste it here  
    """)
    st.divider()
    st.markdown("""
    **QuickBooks Import**  
    1. **Banking → Upload Transactions**  
    2. Select your bank account  
    3. Upload the `.xlsx` or `.csv`  
    4. Map columns if prompted  
    5. Review & accept
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
        st.info(f"📄 Text-based PDF — using **{TEXT_MODEL}**")
    else:
        st.info(f"🖼️ Scanned PDF — using **{VISION_MODEL}** (vision)")

    # Step 2: Show source + parse
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Source Preview")
        if text_mode:
            st.text_area("Extracted text", pdf_text[:3000] + ("..." if len(pdf_text) > 3000 else ""), height=300)
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
            except (json.JSONDecodeError, ValueError) as e:
                st.error(f"Failed to parse response: {e}")
                st.stop()
            except Exception as e:
                if "429" in str(e):
                    st.error("Rate limit hit. Wait 30 seconds and try again.")
                else:
                    st.error(f"API error: {e}")
                st.stop()

    # Step 3: Validation warnings
    warnings = validate_transactions(transactions)
    if warnings:
        with st.expander(f"⚠️ {len(warnings)} potential issue(s) detected — click to review", expanded=True):
            for w in warnings:
                st.warning(w)

    # Step 4: Build editable dataframe
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

    if detailed_export:
        df = pd.DataFrame(rows, columns=["Date", "Description", "Amount", "Check Number", "Type"])
    else:
        df = pd.DataFrame(rows, columns=["Date", "Description", "Amount"])

    st.subheader("✏️ Review & Edit Transactions")
    st.caption("Fix any errors directly in the table below before exporting.")

    edited_df = st.data_editor(
        df,
        use_container_width=True,
        height=400,
        num_rows="dynamic",
        column_config={
            "Date": st.column_config.TextColumn("Date", help="MM/DD/YYYY"),
            "Description": st.column_config.TextColumn("Description", width="large"),
            "Amount": st.column_config.NumberColumn("Amount", help="Negative = debit, Positive = credit", format="%.2f"),
            "Type": st.column_config.SelectboxColumn(
                "Type",
                options=["Deposit", "Withdrawal", "Check", "Fee", "Interest", "Transfer", "Other"],
            ) if detailed_export else None,
        },
    )

    # Step 5: Summary
    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    total_credits = edited_df[edited_df["Amount"] > 0]["Amount"].sum()
    total_debits = edited_df[edited_df["Amount"] < 0]["Amount"].sum()
    c1.metric("Total Credits", f"${total_credits:,.2f}")
    c2.metric("Total Debits", f"${abs(total_debits):,.2f}")
    c3.metric("Net Change", f"${total_credits + total_debits:,.2f}")
    c4.metric("Transactions", len(edited_df))

    # Step 6: Downloads
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
