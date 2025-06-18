import os
import json
from datetime import datetime
from playwright.sync_api import sync_playwright, TimeoutError
from bs4 import BeautifulSoup

# === CONFIG ===
BASE_URL = "https://discourse.onlinedegree.iitm.ac.in"
CATEGORY_ID = 34
CATEGORY_JSON_URL = f"{BASE_URL}/c/courses/tds-kb/{CATEGORY_ID}.json"
AUTH_STATE_FILE = "auth.json"
DATE_FROM = datetime(2025, 1, 1)
DATE_TO = datetime(2025, 4, 14)
OUTPUT_DIR = "discourse_json"  # Directory to save topic-wise JSON files

def parse_date(date_str):
    try:
        return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError:
        return datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")

def login_and_save_auth(playwright):
    print("🔐 No auth found. Launching browser for manual login...")
    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()
    page.goto(f"{BASE_URL}/login")
    print("🌐 Please log in manually using Google. Then press ▶️ (Resume) in Playwright bar.")
    page.pause()
    context.storage_state(path=AUTH_STATE_FILE)
    print("✅ Login state saved.")
    browser.close()

def is_authenticated(page):
    try:
        page.goto(CATEGORY_JSON_URL, timeout=10000)
        page.wait_for_selector("pre", timeout=5000)
        json.loads(page.inner_text("pre"))
        return True
    except (TimeoutError, json.JSONDecodeError):
        return False

def scrape_posts(playwright):
    print("🔍 Starting scrape using saved session...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context(storage_state=AUTH_STATE_FILE)
    page = context.new_page()
    all_topics = []
    page_num = 0
    while True:
        paginated_url = f"{CATEGORY_JSON_URL}?page={page_num}"
        print(f"📦 Fetching page {page_num}...")
        page.goto(paginated_url)
        try:
            data = json.loads(page.inner_text("pre"))
        except:
            data = json.loads(page.content())
        topics = data.get("topic_list", {}).get("topics", [])
        if not topics:
            break
        all_topics.extend(topics)
        page_num += 1
    print(f"📄 Found {len(all_topics)} total topics across all pages")
    num_saved = 0
    for topic in all_topics:
        created_at = parse_date(topic["created_at"])
        if DATE_FROM <= created_at <= DATE_TO:
            topic_url = f"{BASE_URL}/t/{topic['slug']}/{topic['id']}.json"
            page.goto(topic_url)
            try:
                topic_data = json.loads(page.inner_text("pre"))
            except:
                topic_data = json.loads(page.content())
            # Clean up all posts' cooked HTML to plain text and add images
            posts = topic_data.get("post_stream", {}).get("posts", [])
            for post in posts:
                soup = BeautifulSoup(post["cooked"], "html.parser")
                # Extract image URLs
                images = []
                for img in soup.find_all("img"):
                    src = img.get("src")
                    if src and not src.startswith("data:"):
                        if src.startswith("/"):
                            src = BASE_URL + src
                        images.append(src)
                post["content"] = soup.get_text()
                post["images"] = images
            # Save topic as topic_<topic_id>.json
            topic_file = os.path.join(OUTPUT_DIR, f"topic_{topic['id']}.json")
            with open(topic_file, "w", encoding="utf-8") as f:
                json.dump(topic_data, f, indent=2, ensure_ascii=False)
            num_saved += 1
    print(f"✅ Saved {num_saved} topic files in {OUTPUT_DIR}")
    browser.close()

def main():
    with sync_playwright() as p:
        if not os.path.exists(AUTH_STATE_FILE):
            login_and_save_auth(p)
        else:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(storage_state=AUTH_STATE_FILE)
            page = context.new_page()
            if not is_authenticated(page):
                print("⚠️ Session invalid. Re-authenticating...")
                browser.close()
                login_and_save_auth(p)
            else:
                print("✅ Using existing authenticated session.")
                browser.close()
        scrape_posts(p)

if __name__ == "__main__":
    main()
