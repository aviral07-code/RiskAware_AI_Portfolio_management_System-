import os
import json
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document 
from langchain_text_splitters import RecursiveCharacterTextSplitter

def main():
    # Setup Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    jsonl_path = os.path.join(project_root, "data", "raw", "news_articles.jsonl")
    vector_dir = os.path.join(project_root, "data", "vector_db")
    
    if not os.path.exists(jsonl_path):
        print(f"⚠️ No news_articles.jsonl found at {jsonl_path}")
        print("Run 'scripts/fetch_real_news.py' first.")
        return

    print("Loading articles...")
    documents = []
    
    # Read JSONL file
    try:
        with open(jsonl_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # Create LangChain Document
                    doc = Document(
                        page_content=data.get('content', '') or data.get('headline', ''),
                        metadata={
                            "source": data.get('url', 'unknown'), 
                            "date": data.get('date', 'unknown'), 
                            "headline": data.get('headline', 'unknown')
                        }
                    )
                    documents.append(doc)
                except Exception:
                    continue
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    if not documents:
        print("⚠️ No valid documents found. Check your JSONL file.")
        return
    
    print(f"Loaded {len(documents)} articles. Splitting text...")
    
    # Split text into chunks (LLMs have context limits)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs_split = splitter.split_documents(documents)
    
    print(f"Generated {len(docs_split)} chunks. Embedding...")
    
    # --- FIX: Force CPU to prevent Mac Crash ---
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'} 
    )
    # -------------------------------------------
    
    vectordb = FAISS.from_documents(docs_split, embeddings)
    
    # Ensure directory exists
    os.makedirs(vector_dir, exist_ok=True)
    
    # Save index
    save_path = os.path.join(vector_dir, "faiss_index")
    vectordb.save_local(save_path)
    
    print(f"✅ Vector DB saved to {save_path}")

if __name__ == "__main__":
    main()
