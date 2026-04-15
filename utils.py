import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from ddgs import DDGS


# =========================
# WEB SEARCH (DYNAMIC + FIXED)
# =========================

def web_search(query):
    urls = []

    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=3)

            for r in results:
                if "href" in r:
                    urls.append(r["href"])

    except Exception as e:
        print("Search error:", e)

    # 🔥 FALLBACK (IMPORTANT — NEVER EMPTY)
    if not urls:
        urls = [
            f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}"
        ]

    return urls


# =========================
# WEBSITE SCRAPER (ROBUST)
# =========================

def scrape_website(url):
    try:
        response = requests.get(
            url,
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"}
        )

        # if blocked or failed
        if response.status_code != 200:
            return ""

        soup = BeautifulSoup(response.text, "html.parser")

        # remove unwanted tags
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)

        return text[:2000]  # 🔥 limit for speed

    except Exception as e:
        print("Scraping error:", e)
        return ""


# =========================
# PDF READER
# =========================

def read_pdf(file):
    try:
        reader = PdfReader(file)
        text = ""

        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted

        return text[:3000]  # 🔥 limit

    except Exception as e:
        print("PDF error:", e)
        return ""