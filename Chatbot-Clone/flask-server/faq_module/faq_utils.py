#The multicore usage version of knowledge.py
import json
import chromadb.config
from sentence_transformers import SentenceTransformer
import chromadb
from concurrent.futures import ThreadPoolExecutor
import requests
import os
from dotenv import load_dotenv



embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
client = chromadb.Client(chromadb.config.Settings(persist_directory="./db"))  #db inside venv folder > chroma > db
collection = client.create_collection("delta_faq")


def process_line(line):
    return json.loads(line)

def load_faq() -> list:
    
    faq_data = []
    with open("delta_faq.jsonl", "r", encoding="utf-8") as f:
        lines = f.readlines()

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(process_line, lines))  #executor.map returns a lazy iterator, need list() to return a full executed list.

    faq_data.extend(results)
    print(faq_data[:2])
    return faq_data

# -----------------   Embedding into vector database -------------------#



def embed_faq(faq_data):
    
    faq_embeddings = []
    for item in faq_data:
        text_to_embed = item["question"] + " " + item["answer"]   #question + answer to be embedded together.
        vector = embedder.encode(text_to_embed).tolist()          #float embeddings stored in a list.
        faq_embeddings.append({
            "vector": vector,
            "metadata": item
        })
    #faq_embeddings = [{"vector: []"}]

    collection.add(
        documents=[item["metadata"]["answer"] for item in faq_embeddings],
        metadatas=[item["metadata"] for item in faq_embeddings],
        embeddings=[item["vector"] for item in faq_embeddings],
        ids=[str(i) for i in range(len(faq_embeddings))]
    )

def query_faq(user_question = "What does “Shape” and “Pattern” functions of the Machine Vision System DMV Series mean") -> dict:

    query_vector = embedder.encode(user_question).tolist() #easier and friendlier to embed than numpy array
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=3
    )
    top_faqs = []
    for i, (faq, distance) in enumerate(zip(results["metadatas"][0], results["distances"][0])): 
        #faq is dict, distance is float

        faq['similarity_distance'] = distance
        top_faqs.append(faq)

        print(f"Match{i+1}:")
        print("Q", faq["question"])
        print("A", faq["answer"])
        print("Similarity Distance:", faq['similarity_distance'] )
        
        print("------")

        #---------------- GEMINI LLM RAG --------------------------
    load_dotenv()
    VITE_API_URL = os.getenv("VITE_API_URL")


    contents=[
            {
                "role": "user",
                "parts": [
                    {"text": f'''
                            Remember that the customer cannot see the RAG retrieved context, so don't make any mention about it at all.
                            Context: {top_faqs}\n\n
                            User Question: {user_question}\n\nAnswer:
                     '''}
                ]
            }
        ]

    headers = {"Content-Type": "application/json"}
    payload = {"contents": contents}

    response = requests.post(VITE_API_URL, headers=headers, json=payload)
    result = response.json()
    bot_message = result["candidates"][0]["content"]["parts"][0]["text"]

    print(result)
    print("BOT MESSAGE:", bot_message)
    return(bot_message)

faq_data = load_faq()
embed_faq(faq_data)




 