import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader
from ddgs import DDGS


# =========================
# WEB SEARCH (ROBUST + NEVER FAILS)
# =========================

def web_search(query, num_results=5):
    urls = []

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=num_results))

            for r in results:
                link = r.get("href") or r.get("url")
                if link and link.startswith("http"):
                    urls.append(link)

    except Exception as e:
        print("❌ Search error:", e)

    # 🔥 STRONG FALLBACK (VERY IMPORTANT)
    if not urls:
        print("⚠️ Using fallback URLs")

        urls = [
            f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}",
            f"https://www.britannica.com/search?query={query}",
            f"https://www.google.com/search?q={query}"
        ]

    return urls[:num_results]


# =========================
# WEBSITE SCRAPER (STRONG + CLEAN)
# =========================

def scrape_website(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }

        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code != 200:
            return ""

        soup = BeautifulSoup(response.text, "html.parser")

        # ❌ Remove noise
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        # ✅ Extract only meaningful text
        paragraphs = soup.find_all("p")

        text = " ".join([p.get_text(strip=True) for p in paragraphs])

        # 🔥 Safety check
        if len(text) < 100:
            return ""

        return text[:3000]

    except Exception as e:
        print("❌ Scraping error:", e)
        return ""


# =========================
# PDF READER (SAFE)
# =========================

def read_pdf(file):
    try:
        reader = PdfReader(file)
        text = ""

        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"

        if len(text.strip()) == 0:
            return ""

        return text[:4000]

    except Exception as e:
        print("❌ PDF error:", e)
        return ""
