# IMPORTANT: Save as 'evaluate_weaviate.py' (not 'weaviate.py')
# Requires: pip install weaviate-client>=4.0.0

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Set

try:
    import weaviate
    from weaviate.classes.config import Configure, Property, DataType
    from weaviate.classes.query import MetadataQuery
    WEAVIATE_V4 = True
except ImportError:
    # Fallback for Weaviate v3
    import weaviate
    WEAVIATE_V4 = False
    print("⚠️  Warning: Using Weaviate client v3. Please upgrade: pip install weaviate-client>=4.0.0")

class WeaviateEvaluator:
    def __init__(self, corpus_path: str, eval_path: str, url: str = "http://localhost:8080", collection_name: str = "EvalCollection"):
        """
        Initialize the evaluator with corpus and evaluation data.
        
        Args:
            corpus_path: Path to corpus.jsonl file
            eval_path: Path to eval_set_test.jsonl file
            url: Weaviate instance URL
            collection_name: Name for the Weaviate collection
        """
        self.corpus_path = corpus_path
        self.eval_path = eval_path
        self.collection_name = collection_name
        
        # Initialize Weaviate client based on version
        if WEAVIATE_V4:
            try:
                # Try with gRPC first
                self.client = weaviate.connect_to_local(
                    host="localhost", 
                    port=8080,
                    grpc_port=50051
                )
            except Exception as e:
                print(f"⚠️  gRPC connection failed, trying without it...")
                # Fallback: skip gRPC checks
                self.client = weaviate.connect_to_local(
                    host="localhost", 
                    port=8080,
                    skip_init_checks=True
                )
            self.is_v4 = True
        else:
            self.client = weaviate.Client(url)
            self.is_v4 = False
        
        # Initialize embedding model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load data
        self.corpus = self._load_corpus()
        self.eval_queries = self._load_eval_queries()
        
        # Create and populate collection
        if self.is_v4:
            self.collection = self._create_collection_v4()
        else:
            self._create_collection_v3()
        
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
        """Create Weaviate collection and add corpus documents (V4 API)."""
        # Delete collection if it exists
        if self.client.collections.exists(self.collection_name):
            self.client.collections.delete(self.collection_name)
        
        # Create new collection
        collection = self.client.collections.create(
            name=self.collection_name,
            properties=[
                Property(name="passage_id", data_type=DataType.TEXT),
                Property(name="text", data_type=DataType.TEXT),
                Property(name="metadata", data_type=DataType.TEXT),
            ],
            vectorizer_config=Configure.Vectorizer.none()
        )
        
        # Prepare data for batch insertion
        print(f"Embedding {len(self.corpus)} documents...")
        
        with collection.batch.dynamic() as batch:
            for passage_id, doc in self.corpus.items():
                embedding = self.model.encode(doc['text']).tolist()
                batch.add_object(
                    properties={
                        "passage_id": passage_id,
                        "text": doc['text'],
                        "metadata": json.dumps(doc.get('meta', {}))
                    },
                    vector=embedding
                )
        
        print(f"Added {len(self.corpus)} documents to Weaviate collection")
        return collection
    
    _create_collection_v4 = _create_collection  # Alias for clarity
    
    def _create_collection_v3(self):
        """Create Weaviate collection and add corpus documents (V3 API)."""
        # Delete class if exists
        try:
            self.client.schema.delete_class(self.collection_name)
        except:
            pass
        
        # Create schema
        class_obj = {
            "class": self.collection_name,
            "vectorizer": "none",
            "properties": [
                {"name": "passage_id", "dataType": ["text"]},
                {"name": "text", "dataType": ["text"]},
                {"name": "metadata", "dataType": ["text"]},
            ]
        }
        self.client.schema.create_class(class_obj)
        
        # Add documents with batch
        print(f"Embedding {len(self.corpus)} documents...")
        
        with self.client.batch as batch:
            batch.batch_size = 100
            for passage_id, doc in self.corpus.items():
                embedding = self.model.encode(doc['text']).tolist()
                properties = {
                    "passage_id": passage_id,
                    "text": doc['text'],
                    "metadata": json.dumps(doc.get('meta', {}))
                }
                batch.add_data_object(
                    data_object=properties,
                    class_name=self.collection_name,
                    vector=embedding
                )
        
        print(f"Added {len(self.corpus)} documents to Weaviate collection")
    
    def retrieve(self, query: str, k: int = 10) -> List[str]:
        """Retrieve top-k passages for a query."""
        query_embedding = self.model.encode(query).tolist()
        
        if self.is_v4:
            response = self.collection.query.near_vector(
                near_vector=query_embedding,
                limit=k,
                return_metadata=MetadataQuery(distance=True)
            )
            return [obj.properties['passage_id'] for obj in response.objects]
        else:
            # V3 API
            result = self.client.query.get(
                self.collection_name, 
                ["passage_id"]
            ).with_near_vector({
                "vector": query_embedding
            }).with_limit(k).do()
            
            if result.get('data') and result['data'].get('Get', {}).get(self.collection_name):
                return [obj['passage_id'] for obj in result['data']['Get'][self.collection_name]]
            return []
    
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
            'f1_score': {k: [] for k in k_values}
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
        
        return results
    
    def print_results(self, results: Dict):
        """Pretty print evaluation results."""
        print("\n" + "="*60)
        print("WEAVIATE RETRIEVAL EVALUATION RESULTS")
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
        
        print("\n" + "="*60)
    
    def cleanup(self):
        """Delete the collection and close client."""
        if self.is_v4:
            self.client.collections.delete(self.collection_name)
            self.client.close()
        else:
            self.client.schema.delete_class(self.collection_name)
        print(f"\n🗑️  Deleted collection: {self.collection_name}")


# Example usage
if __name__ == "__main__":
    # Make sure Weaviate is running locally on port 8080
    # docker run -d -p 8080:8080 semitechnologies/weaviate:latest
    
    evaluator = WeaviateEvaluator(
        corpus_path="corpus.jsonl",
        eval_path="eval_set_test.jsonl",
        url="http://localhost:8080",
        collection_name="EvalCollection"
    )
    
    results = evaluator.evaluate(k_values=[1, 3, 5, 10])
    evaluator.print_results(results)
    
    with open('weaviate_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n✅ Results saved to weaviate_results.json")
    
    # Cleanup
    evaluator.cleanup()