import requests

HEADERS = {'User-Agent': 'GeorgianDictionaryBot/1.0'}


def fetch_wiki_content(title: str) -> str:
    url = "https://ka.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
        "exlimit": 1
    }
    try:
        response = requests.get(url, params=params, headers=HEADERS, timeout=10)
        data = response.json()
        pages = data.get("query", {}).get("pages", {})
        for page_data in pages.values():
            if "extract" in page_data:
                return page_data["extract"]
    except Exception as e:
        print(f"Error fetching page '{title}': {e}")
    return ""


def parse_wikipedia() -> None:
    """fetch wikipedia pages and extract texts"""
    seed_titles = [
        "საქართველო", "თბილისი", "საქართველოს_ისტორია",
        "ფიზიკა", "მათემატიკა", "ქიმია", "ბიოლოგია",
        "ლიტერატურა", "ხელოვნება", "მუსიკა", "სპორტი",
        "ეკონომიკა", "პოლიტიკა", "გეოგრაფია", "ფილოსოფია"
    ]

    print(f"Fetching {len(seed_titles)} seed pages...")
    for title in seed_titles:
        text = fetch_wiki_content(title)
        if text:
            with open(f"{title}.txt", "w", encoding="utf-8") as f:
                f.write(text)


if __name__ == "__main__":
    parse_wikipedia()
