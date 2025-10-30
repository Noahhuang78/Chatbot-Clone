# IMPORTANT: Save as 'evaluate_qdrant.py' (not 'qdrant.py')

import json
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Set
import time

query_time = []
similarity_distances = []
class QdrantEvaluator:
    def __init__(self, corpus_path: str, eval_path: str, collection_name: str = "eval_collection", url: str = "localhost", port: int = 6333):
        """
        Initialize the evaluator with corpus and evaluation data.
        
        Args:
            corpus_path: Path to corpus.jsonl file
            eval_path: Path to eval_set_test.jsonl file
            collection_name: Name for the Qdrant collection
            url: Qdrant server URL
            port: Qdrant server port
        """
        self.corpus_path = corpus_path
        self.eval_path = eval_path
        self.collection_name = collection_name
        
        # Initialize Qdrant client
        self.client = QdrantClient(host=url, port=port)
        
        # Initialize embedding model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = 384  # MiniLM-L6-v2 dimension
        
        # Load data
        self.corpus = self._load_corpus()
        self.eval_queries = self._load_eval_queries()
        
        # Create and populate collection
        self._create_collection()
        
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
    
    def _create_collection(self):
        """Create Qdrant collection and add corpus documents."""
        # Delete collection if it exists
        try:
            self.client.delete_collection(collection_name=self.collection_name)
        except:
            pass
        
        # Create new collection
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE)
        )
        
        # Prepare data for batch insertion
        print(f"Embedding {len(self.corpus)} documents...")
        points = []
        
        for idx, (passage_id, doc) in enumerate(self.corpus.items()):
            embedding = self.model.encode(doc['text']).tolist()
            
            point = PointStruct(
                id=idx,
                vector=embedding,
                payload={
                    "passage_id": passage_id,
                    "text": doc['text'],
                    "metadata": doc.get('meta', {})
                }
            )
            points.append(point)
            
            # Upload in batches of 100
            if len(points) >= 100:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                points = []
        
        # Upload remaining points
        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
        
        print(f"Added {len(self.corpus)} documents to Qdrant collection")
    
    def retrieve(self, query: str, k: int = 10) -> List[str]:
        """Retrieve top-k passages for a query."""
        start = time.time()
        query_embedding = self.model.encode(query).tolist()
        
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=k
        )
        retrieval_time = (time.time() - start)/10
        query_time.append(retrieval_time)
        for hit in results:
            similarity_distances.append(hit.score)
        return [hit.payload['passage_id'] for hit in results]
    
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
            'mean_similarity_distance': {}
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
        results['mean_retrieval_time'] = np.mean(query_time),
        results['mean_similarity_distance'] = np.mean(similarity_distances)
        return results
    
    def print_results(self, results: Dict):
        """Pretty print evaluation results."""
        print("\n" + "="*60)
        print("QDRANT RETRIEVAL EVALUATION RESULTS")
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
        print(f"  Mean Retrieval Time:           {np.mean(query_time):.4f} ")
        print(f"  Mean Similarity Distance:           {np.mean(similarity_distances):.4f} ")
        
        print("\n" + "="*60)
    
    def cleanup(self):
        """Delete the collection."""
        self.client.delete_collection(collection_name=self.collection_name)
        print(f"\n🗑️  Deleted collection: {self.collection_name}")


# Example usage
if __name__ == "__main__":
    # Make sure Qdrant is running locally on port 6333
    # docker run -d -p 6333:6333 qdrant/qdrant
    
    evaluator = QdrantEvaluator(
        corpus_path="corpus.jsonl",
        eval_path="eval_set_test.jsonl",
        collection_name="eval_collection",
        url="localhost",
        port=6333
    )
    
    results = evaluator.evaluate(k_values=[1, 3, 5, 10])
    evaluator.print_results(results)
    
    with open('qdrant_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n✅ Results saved to qdrant_results.json")
    
    # Cleanup
    evaluator.cleanup()