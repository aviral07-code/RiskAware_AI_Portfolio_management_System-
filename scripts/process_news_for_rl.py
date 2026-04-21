import os
import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

def main():
    # Setup Paths
    BASE_DIR = "/Users/aviralgarg/AI_Portfolio_Manager"
    jsonl_path = os.path.join(BASE_DIR, "data", "raw", "news_articles.jsonl")
    output_path = os.path.join(BASE_DIR, "data", "raw", "news_embeddings.csv")
    
    if not os.path.exists(jsonl_path):
        print(f"❌ Error: {jsonl_path} not found.")
        print("Run 'python scripts/fetch_real_news.py' first!")
        return

    # Check if file has size
    if os.path.getsize(jsonl_path) == 0:
        print("❌ Error: JSONL file is empty. Fetch script failed to write data.")
        return

    print("Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
    
    print("Reading news articles...")
    daily_news = {}
    total_lines = 0
    
    try:
        with open(jsonl_path, 'r') as f:
            for line in f:
                total_lines += 1
                try:
                    data = json.loads(line)
                    date = data['date']
                    
                    # Robust content extraction
                    headline = data.get('headline', '')
                    content = data.get('content', '')
                    if content is None: content = ""
                    
                    # Combine Headline + Snippet
                    text = f"{headline}. {content[:200]}"
                    
                    if date not in daily_news:
                        daily_news[date] = []
                    daily_news[date].append(text)
                except Exception as e:
                    print(f"Skipping bad line: {e}")
                    continue
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    print(f"Read {total_lines} lines. Found news for {len(daily_news)} unique days.")
    
    if len(daily_news) == 0:
        print("❌ No valid news data found to embed.")
        return

    print("Computing embeddings...")
    daily_embeddings = {}
    
    # Compute Average Embedding per Day
    for date, texts in daily_news.items():
        if not texts:
            continue
        vectors = model.encode(texts)
        avg_vector = np.mean(vectors, axis=0)
        daily_embeddings[date] = avg_vector
        
    # Convert to DataFrame and Save
    df = pd.DataFrame.from_dict(daily_embeddings, orient='index')
    df.index.name = 'date'
    df.sort_index(inplace=True)
    
    df.to_csv(output_path)
    print(f"✅ Success! Saved daily news embeddings to: {output_path}")
    print(f"   Shape: {df.shape} (Rows=Days, Cols=384 dimensions)")

if __name__ == "__main__":
    main()
