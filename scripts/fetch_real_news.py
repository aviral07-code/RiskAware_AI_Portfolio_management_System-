import pandas as pd
import json
import os
import time
import random
from gnews import GNews
from newspaper import Article, Config
from datetime import datetime, timedelta

# --- CONFIGURATION ---
# Use absolute paths to be safe
BASE_DIR = "/Users/aviralgarg/AI_Portfolio_Manager"
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
OUTPUT_FILE = os.path.join(DATA_DIR, "news_articles.jsonl")

START_DATE = "2020-01-01"
END_DATE = "2020-12-31"
TOPICS = ["Stock Market Crash", "Federal Reserve", "COVID-19 Economy", "Tech Sector", "Recession"]

# Configure Newspaper3k
user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
config = Config()
config.browser_user_agent = user_agent
config.request_timeout = 10

def fetch_full_text(url):
    """Downloads and parses full article text."""
    try:
        article = Article(url, config=config)
        article.download()
        article.parse()
        return article.title, article.text
    except Exception:
        return None, None

def generate_fallback_article(date_str, topic):
    """Generates a dummy entry if scraping fails, to keep pipeline moving."""
    return {
        "date": date_str,
        "topic": topic,
        "headline": f"Market update regarding {topic}",
        "content": f"Market analysts discuss the impact of {topic} on global trends. " * 5,
        "url": "http://manual-fallback.com"
    }

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Clear old file to avoid appending to bad data
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
        
    # Initialize GNews
    google_news = GNews(language='en', country='US', max_results=5)
    
    start_dt = datetime.strptime(START_DATE, "%Y-%m-%d")
    end_dt = datetime.strptime(END_DATE, "%Y-%m-%d")
    current_dt = start_dt
    
    print(f"Fetching news from {START_DATE} to {END_DATE}...")
    
    with open(OUTPUT_FILE, 'a') as f_out:
        while current_dt <= end_dt:
            date_str = current_dt.strftime("%Y-%m-%d")
            print(f"Processing {date_str}...")
            
            # --- FIX: GNews requires a date RANGE, not same day ---
            # We set the range from [Today] to [Tomorrow] to capture 24h
            next_day = current_dt + timedelta(days=1)
            google_news.start_date = (current_dt.year, current_dt.month, current_dt.day)
            google_news.end_date = (next_day.year, next_day.month, next_day.day)
            
            articles_found_today = 0
            
            for topic in TOPICS:
                try:
                    news_items = google_news.get_news(topic)
                    
                    if news_items:
                        for item in news_items:
                            url = item.get('url')
                            full_title, full_text = fetch_full_text(url)
                            
                            # Use scraped text, or fallback to description, or fallback to title
                            content = full_text if full_text and len(full_text) > 50 else item.get('description', '')
                            if len(content) < 20: content = item.get('title') * 3

                            entry = {
                                "date": date_str,
                                "topic": topic,
                                "headline": full_title or item.get('title'),
                                "content": content,
                                "url": url
                            }
                            f_out.write(json.dumps(entry) + "\n")
                            articles_found_today += 1
                    
                    time.sleep(1) # Be gentle with rate limits
                    
                except Exception as e:
                    print(f"  Error on {topic}: {e}")

            # --- FALLBACK: If GNews returned NOTHING for the day, inject dummy data ---
            # This ensures your RL agent doesn't crash on empty dates
            if articles_found_today == 0:
                print(f"  ⚠️ No articles found for {date_str}. Injecting fallback data.")
                dummy = generate_fallback_article(date_str, "General Market")
                f_out.write(json.dumps(dummy) + "\n")

            current_dt += timedelta(days=1)

    print(f"✅ Done! Articles saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
