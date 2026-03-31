# Bank Statement → QuickBooks Converter

Streamlit app that converts any bank statement PDF into QuickBooks-ready Excel/CSV using Google Gemini Flash (free).

## Local Setup

```bash
pip install -r requirements.txt
# macOS: brew install poppler | Ubuntu: sudo apt install poppler-utils
streamlit run app.py
```

## Deploy to Streamlit Community Cloud (Free)

### Step 1: Push to GitHub

```bash
git init
git add .
git commit -m "bank statement converter"
git remote add origin https://github.com/YOUR_USERNAME/bank-statement-converter.git
git branch -M main
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click **New app**
4. Select your repo → branch `main` → file `app.py`
5. Click **Deploy**

That's it. Streamlit Cloud auto-installs `requirements.txt` (Python) and `packages.txt` (system deps like poppler-utils).

### Step 3: Get Gemini API Key

1. Go to [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
2. Click **Create API Key**
3. Paste it into the app sidebar

Free tier gives you 15 requests/min and 1M tokens/day — plenty for personal use.

## How It Works

1. PDF uploaded → `pdfplumber` tries text extraction
2. If scanned (no text), pages are rasterized to images via `pdftoppm`
3. Text or images sent to Gemini Flash for structured transaction parsing
4. Output: QuickBooks-compatible Excel/CSV with Date, Description, Amount columns
