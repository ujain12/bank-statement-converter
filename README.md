Bank Statement → QuickBooks Converter
Streamlit app that converts any bank statement PDF into QuickBooks-ready Excel/CSV using Groq (free, no credit card needed).
Local Setup
bashpip install -r requirements.txt
# macOS: brew install poppler | Ubuntu: sudo apt install poppler-utils
streamlit run app.py
Deploy to Streamlit Community Cloud (Free)

Push this folder to a GitHub repo
Go to share.streamlit.io → New app → select repo → app.py → Deploy
Get free Groq key at console.groq.com → API Keys → Create

Free limits: 30 req/min, 14,400 req/day.
How It Works

PDF uploaded → pdfplumber tries text extraction
If scanned (no text), pages rasterized to images via pdftoppm
Text PDFs → Llama 3.3 70B | Scanned PDFs → Llama 4 Scout (vision)
Output: QuickBooks-compatible Excel/CSV (Date, Description, Amount)
