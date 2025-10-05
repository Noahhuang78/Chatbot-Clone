#The multicore usage version of knowledge.py
import json
import chromadb.config
from sentence_transformers import SentenceTransformer
import chromadb
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import time

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
client = chromadb.Client(chromadb.config.Settings(persist_directory="./db"))  #db inside venv folder > chroma > db
collection = client.create_collection("FAQ")


faq_data = []

def process_line(line):
    return json.loads(line)

def load_faq() -> list:
    
    with open("../delta_faq.jsonl", "r", encoding="utf-8") as f:
        lines = f.readlines()

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(process_line, lines))  #executor.map returns a lazy iterator, need list() to return a full executed list.

    faq_data.extend(results)
    # print("First 3 FAQ DATA:" + "\n" + str(faq_data[:2]))


# -----------------   Embedding into vector database -------------------#


def embed_faq():

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


def query_faq(user_question) -> dict:
   
        query_vector = embedder.encode(user_question).tolist() #easier and friendlier to embed than numpy array
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=1
        )
        top_faqs = []
        for i, (faq, distance) in enumerate(zip(results["metadatas"][0], results["distances"][0])): 
            #faq is dict, distance is float
            faq['similarity_distance'] = distance
            top_faqs.append(faq)

        return distance

def load_questions():
    with open("questions.md", 'r', encoding="utf-8") as f:
        return f.readlines()
    

load_faq()
embed_faq()
results = {}
questions = load_questions()


start_time = time.perf_counter()
with ThreadPoolExecutor(max_workers=8) as executor:
    results = list(executor.map(query_faq, questions)) 
end_time = time.perf_counter()
elapsed_time = end_time - start_time

with open('perf_time.jsonl', 'a', encoding='utf-8') as f:
    json.dump({'name': "sentence-transformers/all-MiniLM-L6-v2 ", 'perf_time': elapsed_time}, f , ensure_ascii=False)
    f.writes("\n")


with open(f'MiniLM_similarities.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False)


df = pd.DataFrame(results)
print(f"My DF:\n {df}")
avg_simlarity = df.iloc[:, 0].mean()     #[row number, column number]      .loc[row name, column name]

print(f'Average similarity distance: {avg_simlarity}')
#average for MiniLM is 0.880526
