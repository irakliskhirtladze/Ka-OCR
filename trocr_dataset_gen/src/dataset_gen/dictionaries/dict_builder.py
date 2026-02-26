import re
import json
import requests
from pathlib import Path
from collections import Counter
import pymupdf as fitz
from docx import Document
from dataset_gen.utils import BASE_DIR

GEORGIAN_PATTERN = re.compile(r'[ა-ჰ]')
MIN_WORD_LENGTH = 2
MAX_WORD_LENGTH = 32
HEADERS = {'User-Agent': 'GeorgianDictionaryBot/1.0'}


def has_georgian(text: str) -> bool:
    return bool(GEORGIAN_PATTERN.search(text))


def extract_words(text: str) -> Counter:
    # split by spaces and keep words with at least one georgian char
    words = text.split()
    counter = Counter()
    for word in words:
        word = word.strip()
        if has_georgian(word) and MIN_WORD_LENGTH <= len(word) <= MAX_WORD_LENGTH:
            counter[word] += 1
    return counter


def extract_text_from_pdf(pdf_path: Path) -> str:
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
        doc.close()
    except Exception as e:
        print(f"Error reading PDF {pdf_path.name}: {e}")
    return text


def extract_text_from_docx(docx_path: Path) -> str:
    text = ""
    try:
        doc = Document(docx_path)
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
    except Exception as e:
        print(f"Error reading DOCX {docx_path.name}: {e}")
    return text


def process_documents(docs_dir: Path) -> Counter:
    # process all PDF and DOCX files in directory
    word_counter = Counter()
    pdf_files = list(docs_dir.glob("*.pdf"))
    docx_files = list(docs_dir.glob("*.docx"))

    print(f"Found {len(pdf_files)} PDF and {len(docx_files)} DOCX files")

    for pdf_path in pdf_files:
        print(f"  Processing: {pdf_path.name}")
        text = extract_text_from_pdf(pdf_path)
        if text:
            word_counter.update(extract_words(text))

    for docx_path in docx_files:
        print(f"  Processing: {docx_path.name}")
        text = extract_text_from_docx(docx_path)
        if text:
            word_counter.update(extract_words(text))

    return word_counter


def fetch_page_content(title: str) -> str:
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


def fetch_random_page() -> str:
    url = "https://ka.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "list": "random",
        "rnnamespace": 0,
        "rnlimit": 1
    }
    try:
        response = requests.get(url, params=params, headers=HEADERS, timeout=10)
        data = response.json()
        random_pages = data.get("query", {}).get("random", [])
        if random_pages:
            return fetch_page_content(random_pages[0]["title"])
    except Exception as e:
        print(f"Error fetching random page: {e}")
    return ""


def process_wikipedia(num_pages: int = 100) -> Counter:
    # fetch wikipedia pages and extract words
    word_counter = Counter()

    seed_titles = [
        "საქართველო", "თბილისი", "საქართველოს_ისტორია",
        "ფიზიკა", "მათემატიკა", "ქიმია", "ბიოლოგია",
        "ლიტერატურა", "ხელოვნება", "მუსიკა", "სპორტი",
        "ეკონომიკა", "პოლიტიკა", "გეოგრაფია", "ფილოსოფია"
    ]

    print(f"Fetching {len(seed_titles)} seed pages...")
    for title in seed_titles:
        text = fetch_page_content(title)
        if text:
            word_counter.update(extract_words(text))

    remaining = num_pages - len(seed_titles)
    if remaining > 0:
        print(f"Fetching {remaining} random pages...")
        for i in range(remaining):
            text = fetch_random_page()
            if text:
                word_counter.update(extract_words(text))
            if (i + 1) % 10 == 0:
                print(f"  Fetched {i + 1}/{remaining} random pages")

    return word_counter


def build_dictionary(word_counter: Counter, min_frequency: int = 1) -> dict:
    # filter by frequency and build dictionary structure
    filtered = {w: c for w, c in word_counter.items() if c >= min_frequency}
    total = sum(filtered.values())

    word_list = [
        {"word": w, "frequency": c, "weight": c / total, "length": len(w)}
        for w, c in filtered.items()
    ]
    word_list.sort(key=lambda x: x["frequency"], reverse=True)

    return {
        "words": word_list,
        "total_unique": len(word_list),
        "total_occurrences": total,
        "metadata": {"min_frequency": min_frequency}
    }


def save_dictionary(dictionary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # json with metadata
    json_path = output_dir / "ka_dictionary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dictionary, f, ensure_ascii=False, indent=4)
    print(f"Saved: {json_path}")

    # plain text
    txt_path = output_dir / "ka_dictionary.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        for item in dictionary["words"]:
            f.write(f"{item['word']}\n")
    print(f"Saved: {txt_path}")

    # weighted format
    weighted_path = output_dir / "ka_dictionary_weighted.txt"
    with open(weighted_path, "w", encoding="utf-8") as f:
        for item in dictionary["words"]:
            f.write(f"{item['word']}\t{item['frequency']}\n")
    print(f"Saved: {weighted_path}")


if __name__ == "__main__":
    docs_dir = BASE_DIR / "data" / "source_docs"
    output_dir = BASE_DIR / "generator" / "dictionaries"

    print("=== Step 1: Processing Documents ===")
    doc_words = process_documents(docs_dir)
    print(f"Found {len(doc_words)} unique words from documents\n")

    print("=== Step 2: Processing Wikipedia ===")
    wiki_words = process_wikipedia(num_pages=1000)
    print(f"Found {len(wiki_words)} unique words from Wikipedia\n")

    print("=== Step 3: Combining Results ===")
    combined = doc_words + wiki_words
    print(f"Combined: {len(combined)} unique words")

    dictionary = build_dictionary(combined, min_frequency=1)
    save_dictionary(dictionary, output_dir)

    print(f"\nTotal unique words: {dictionary['total_unique']}")
    print(f"Total occurrences: {dictionary['total_occurrences']}")
    print("\nTop 20 words:")
    for i, item in enumerate(dictionary['words'][:20], 1):
        print(f"  {i}. {item['word']} ({item['frequency']})")