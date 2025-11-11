import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Set
import time

MODEL_NAME = "BAAI/bge-base-en-v1.5"
similarity_distances = []
query_time = []
with open("corpus.jsonl", 'r', encoding='utf-8') as f:
    print(f"Length of Corpus:{len(f.readlines())}")
class FAISSEvaluator:
    def __init__(self, corpus_path: str, eval_path: str):
        """
        Initialize the evaluator with corpus and evaluation data.
        
        Args:
            corpus_path: Path to corpus.jsonl file
            eval_path: Path to eval_set_test.jsonl file
        """
        self.corpus_path = corpus_path
        self.eval_path = eval_path
        
        # Initialize embedding model
        self.model = SentenceTransformer(MODEL_NAME)
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        # Load data
        self.corpus = self._load_corpus()
        self.eval_queries = self._load_eval_queries()
        
        # Create index and ID mapping
        self.index, self.id_to_passage, self.passage_to_id = self._create_index()
        
    def _load_corpus(self) -> Dict[str, Dict]:
        """Load corpus from JSONL file."""
        corpus = {}
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            
            for line in f:
                doc = json.loads(line)
                corpus[doc['passage_id']] = doc
        return corpus
    
    def _load_eval_queries(self) -> List[Dict]:
        """Load evaluation queries from JSONL file."""
        queries = []
        with open(self.eval_path, 'r', encoding='utf-8') as f:
            for line in f:
                query = json.loads(line)
                if query.get('relevant_passages'):
                    queries.append(query)
        return queries
    
    def _create_index(self):
        """Create FAISS index and add corpus documents."""
        # Create FAISS index (using cosine similarity via inner product on normalized vectors)
        index = faiss.IndexFlatIP(self.dimension)
        
        # Create ID mappings (FAISS uses integer IDs)
        id_to_passage = {}
        passage_to_id = {}
        
        # Prepare embeddings
        print(f"Embedding {len(self.corpus)} documents...")
        embeddings = []
        
        for idx, (passage_id, doc) in enumerate(self.corpus.items()):
            embedding = self.model.encode(doc['text'])
            # Normalize for cosine similarity
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)
            
            id_to_passage[idx] = passage_id
            passage_to_id[passage_id] = idx
        
        # Convert to numpy array and add to index
        embeddings_array = np.array(embeddings).astype('float32')
        index.add(embeddings_array)
        
        print(f"Added {len(self.corpus)} documents to FAISS index")
        return index, id_to_passage, passage_to_id
    
    def retrieve(self, query: str, k: int = 10) -> tuple[List[str], float]:
        """
        Retrieve top-k passages for a query.
        
        Returns:
            Tuple of (List of passage IDs, retrieval time in seconds)
        """
      
       
        query_embedding = self.model.encode(query)
        # Normalize for cosine similarity
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_embedding = np.array([query_embedding]).astype('float32')
        
        # Search/Retrieval Time
        start_time = time.time()
        distances, indices = self.index.search(query_embedding, k)
        similarity_distances.extend(distances)
        retrieval_time = (time.time() - start_time)/10
        query_time.append(retrieval_time)
        # Convert FAISS indices to passage IDs
        return [self.id_to_passage[idx] for idx in indices[0]]
    
    def precision_at_k(self, retrieved: List[str], relevant: Set[str], k: int) -> float:
        """Calculate Precision@K."""
        retrieved_at_k = retrieved[:k]
        relevant_retrieved = len(set(retrieved_at_k) & relevant)
        return relevant_retrieved / k if k > 0 else 0.0
    
    def recall_at_k(self, retrieved: List[str], relevant: Set[str], k: int) -> float:
        """Calculate Recall@K."""
        if len(relevant) == 0:
            return 0.0
        retrieved_at_k = retrieved[:k]
        relevant_retrieved = len(set(retrieved_at_k) & relevant)
        return relevant_retrieved / len(relevant)
    
    def average_precision(self, retrieved: List[str], relevant: Set[str]) -> float:
        """Calculate Average Precision for a single query."""
        if len(relevant) == 0:
            return 0.0
        
        score = 0.0
        num_relevant_retrieved = 0
        
        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant:
                num_relevant_retrieved += 1
                precision_at_i = num_relevant_retrieved / (i + 1)
                score += precision_at_i
        
        return score / len(relevant)
    
    def reciprocal_rank(self, retrieved: List[str], relevant: Set[str]) -> float:
        """Calculate Reciprocal Rank for a single query."""
        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant:
                return 1.0 / (i + 1)
        return 0.0
    
    def dcg_at_k(self, retrieved: List[str], relevant: Set[str], k: int) -> float:
        """Calculate Discounted Cumulative Gain@K."""
        dcg = 0.0
        for i, doc_id in enumerate(retrieved[:k]):
            if doc_id in relevant:
                dcg += 1.0 / np.log2(i + 2)
        return dcg
    
    def ndcg_at_k(self, retrieved: List[str], relevant: Set[str], k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain@K."""
        dcg = self.dcg_at_k(retrieved, relevant, k)
        ideal_retrieved = list(relevant) + [doc for doc in retrieved if doc not in relevant]
        idcg = self.dcg_at_k(ideal_retrieved, relevant, k)
        return dcg / idcg if idcg > 0 else 0.0
    
    def r_precision(self, retrieved: List[str], relevant: Set[str]) -> float:
        """Calculate R-Precision."""
        r = len(relevant)
        if r == 0:
            return 0.0
        return self.precision_at_k(retrieved, relevant, r)
    
    def f1_at_k(self, retrieved: List[str], relevant: Set[str], k: int) -> float:
        """Calculate F1 Score@K."""
        precision = self.precision_at_k(retrieved, relevant, k)
        recall = self.recall_at_k(retrieved, relevant, k)
        
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    def evaluate(self, k_values: List[int] = [1, 3, 5, 10]) -> Dict:
        """Evaluate retrieval performance across all queries."""
        results = {
            'precision_at_k': {k: [] for k in k_values},
            'recall_at_k': {k: [] for k in k_values},
            'map': {},
            'mrr': {},
            'ndcg': {k: [] for k in k_values},
            'r_precision': {},
            'f1_score': {k: [] for k in k_values},
            'mean_retrieval_time': {},
            'mean_similarity_distance': {},
        }
        
        ap_scores = []
        rr_scores = []
        r_prec_scores = []
        
        print(f"\nEvaluating {len(self.eval_queries)} queries...")
        
        for idx, query_data in enumerate(self.eval_queries):
            query = query_data['query']
            relevant = set(query_data['relevant_passages'])
            
            max_k = max(k_values)
            retrieved = self.retrieve(query, k=max_k)
            
            for k in k_values:
                results['precision_at_k'][k].append(
                    self.precision_at_k(retrieved, relevant, k)
                )
                results['recall_at_k'][k].append(
                    self.recall_at_k(retrieved, relevant, k)
                )
                results['ndcg'][k].append(
                    self.ndcg_at_k(retrieved, relevant, k)
                )
                results['f1_score'][k].append(
                    self.f1_at_k(retrieved, relevant, k)
                )
            
            ap_scores.append(self.average_precision(retrieved, relevant))
            rr_scores.append(self.reciprocal_rank(retrieved, relevant))
            r_prec_scores.append(self.r_precision(retrieved, relevant))
            
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(self.eval_queries)} queries")
        
        for k in k_values:
            results['precision_at_k'][k] = np.mean(results['precision_at_k'][k])
            results['recall_at_k'][k] = np.mean(results['recall_at_k'][k])
            results['ndcg'][k] = np.mean(results['ndcg'][k])
            results['f1_score'][k] = np.mean(results['f1_score'][k])
        
        results['map'] = np.mean(ap_scores)
        results['mrr'] = np.mean(rr_scores)
        results['r_precision'] = np.mean(r_prec_scores)
        results['mean_retrieval_time'] = float(np.mean(query_time))
        results['mean_similarity_distance'] = float(np.mean(similarity_distances))
        return results
    
    def print_results(self, results: Dict):
        """Pretty print evaluation results."""
        print("\n" + "="*60)
        print(f"{MODEL_NAME.replace('/','_')}RETRIEVAL EVALUATION RESULTS")
        print("="*60)
        
        print("\n📊 Precision@K:")
        for k, score in results['precision_at_k'].items():
            print(f"  P@{k:2d}: {score:.4f}")
        
        print("\n📊 Recall@K:")
        for k, score in results['recall_at_k'].items():
            print(f"  R@{k:2d}: {score:.4f}")
        
        print("\n📊 NDCG@K:")
        for k, score in results['ndcg'].items():
            print(f"  NDCG@{k:2d}: {score:.4f}")
        
        print("\n📊 F1 Score@K:")
        for k, score in results['f1_score'].items():
            print(f"  F1@{k:2d}: {score:.4f}")
        
        print("\n📊 Single-Value Metrics:")
        print(f"  MAP (Mean Average Precision): {results['map']:.4f}")
        print(f"  MRR (Mean Reciprocal Rank):   {results['mrr']:.4f}")
        print(f"  R-Precision:                  {results['r_precision']:.4f}")
        print(f"  Mean Retrieval Time:           {np.mean(query_time):.10f} ")
        print(f"  Mean Similarity Distance:           {np.mean(similarity_distances):.4f} ")
        print("\n" + "="*60)

       

# Example usage
if __name__ == "__main__":
    evaluator = FAISSEvaluator(
        corpus_path="corpus.jsonl",
        eval_path="eval_set_test.jsonl"
    )
    
    results = evaluator.evaluate(k_values=[1, 3, 5, 10])
    evaluator.print_results(results)
    
    with open(f'{MODEL_NAME.replace("/","_")}_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to {MODEL_NAME.replace('/','_')}_results.json")