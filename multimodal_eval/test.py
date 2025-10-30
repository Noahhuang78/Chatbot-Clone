"""
Multimodal Embedding Model Evaluation Framework with FAISS
Tests CLIP, SigLIP, and ColPali models for image-text retrieval tasks
"""

import torch
import numpy as np
from PIL import Image
from transformers import (
    CLIPProcessor, CLIPModel,
    AutoProcessor, AutoModel
)
from typing import List, Dict, Tuple, Optional
import time
from dataclasses import dataclass
from pathlib import Path
import json
import faiss

# Install requirements:
# pip install torch transformers pillow numpy scikit-learn sentencepiece protobuf faiss-cpu
# or for GPU support: pip install faiss-gpu
# 
# For ColPali (optional):
# pip install colpali-engine

@dataclass
class ModelConfig:
    name: str
    model_id: str
    description: str

@dataclass
class QueryResult:
    qid: str
    query: str
    relevant_images: List[str]  # Ground truth
    retrieved_images: List[str]  # Model predictions
    scores: List[float]
    latency_ms: float

class MultimodalEvaluator:
    """Evaluates multimodal embedding models for image-text retrieval with FAISS"""
    
    MODELS = {
        'clip_vit_b32': ModelConfig(
            'CLIP ViT-B/32',
            'openai/clip-vit-base-patch32',
            'Fast baseline'
        ),
        'clip_vit_l14': ModelConfig(
            'CLIP ViT-L/14',
            'openai/clip-vit-large-patch14',
            'More accurate, slower'
        ),
        'siglip_base': ModelConfig(
            'SigLIP base-patch16',
            'google/siglip-base-patch16-224',
            'Strong cross-lingual performance'
        ),
        'siglip_large': ModelConfig(
            'SigLIP large',
            'google/siglip-large-patch16-384',
            'Most accurate; best for technical diagrams'
        ),
        'colpali': ModelConfig(
        'ColPali',
        'vidore/colpali',   # official HF model repository id
        'State-of-the-art document retrieval (PDF/page embeddings)'
        ),
    }
    
    def __init__(self, device: str = None, faiss_index_type: str = 'flat'):
        """
        Initialize evaluator with FAISS support
        
        Args:
            device: torch device ('cuda' or 'cpu')
            faiss_index_type: 'flat' (exact, slower) or 'ivf' (approximate, faster)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.current_model = None
        self.current_processor = None
        self.current_model_name = None
        self.faiss_index_type = faiss_index_type
        self.faiss_index = None
        self.image_paths_indexed = None
        
    def load_model(self, model_key: str):
        """Load a specific model and processor"""
        config = self.MODELS[model_key]
        print(f"Loading {config.name}...")
        
        try:
            if 'colpali' in model_key:
                # ColPali uses different architecture
                try:
                    from colpali_engine.models import ColPali
                except ImportError:
                    raise ImportError(
                        "ColPali requires colpali-engine. Install with:\n"
                        "  pip install colpali-engine"
                    )
                self.current_model = ColPali.from_pretrained(config.model_id).to(self.device)
                self.current_processor = AutoProcessor.from_pretrained(config.model_id)
            elif 'siglip' in model_key:
                try:
                    import sentencepiece
                except ImportError:
                    raise ImportError(
                        "SigLIP models require sentencepiece. Install with:\n"
                        "  pip install sentencepiece protobuf"
                    )
                self.current_model = AutoModel.from_pretrained(config.model_id).to(self.device)
                self.current_processor = AutoProcessor.from_pretrained(config.model_id)
            else:  # CLIP models
                self.current_model = CLIPModel.from_pretrained(config.model_id).to(self.device)
                self.current_processor = CLIPProcessor.from_pretrained(config.model_id)
            
            self.current_model_name = model_key
            print(f"✓ Loaded {config.name}")
            
        except Exception as e:
            print(f"✗ Error loading {config.name}: {e}")
            raise
    
    def encode_images(self, image_paths: List[str]) -> torch.Tensor:
        """Encode a batch of images to embeddings"""
        images = [Image.open(path).convert('RGB') for path in image_paths]
        
        if 'colpali' in self.current_model_name:
            # ColPali requires processing images one at a time with empty prompts
            all_embeddings = []
            for img in images:
                # Use empty string as prompt for image-only encoding
                inputs = self.current_processor(
                    text="",
                    images=img,
                    return_tensors="pt"
                ).to(self.device)
                with torch.no_grad():
                    outputs = self.current_model(**inputs)
                    # ColPali returns last_hidden_state, pool it
                    embeddings = outputs.last_hidden_state.mean(dim=1)
                    all_embeddings.append(embeddings)
            embeddings = torch.cat(all_embeddings, dim=0)
        elif 'siglip' in self.current_model_name:
            inputs = self.current_processor(images=images, return_tensors="pt").to(self.device)
            with torch.no_grad():
                embeddings = self.current_model.get_image_features(**inputs)
        else:  # CLIP
            inputs = self.current_processor(images=images, return_tensors="pt").to(self.device)
            with torch.no_grad():
                embeddings = self.current_model.get_image_features(**inputs)
        
        # Normalize embeddings
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings
    
    def encode_text(self, queries: List[str]) -> torch.Tensor:
        """Encode text queries to embeddings"""
        if 'colpali' in self.current_model_name:
            inputs = self.current_processor(text=queries, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                embeddings = self.current_model(**inputs).text_embeds
        elif 'siglip' in self.current_model_name:
            inputs = self.current_processor(text=queries, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                embeddings = self.current_model.get_text_features(**inputs)
        else:  # CLIP
            inputs = self.current_processor(text=queries, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                embeddings = self.current_model.get_text_features(**inputs)
        
        # Normalize embeddings
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings
    
    def build_faiss_index(self, image_embeddings: torch.Tensor, image_paths: List[str]):
        """
        Build FAISS index from image embeddings
        
        Args:
            image_embeddings: Normalized embeddings tensor
            image_paths: Corresponding image paths
        """
        # Convert to numpy and ensure float32
        embeddings_np = image_embeddings.cpu().numpy().astype('float32')
        dimension = embeddings_np.shape[1]
        n_images = embeddings_np.shape[0]
        
        print(f"Building FAISS index ({self.faiss_index_type}) with {n_images} images, {dimension}D embeddings...")
        
        if self.faiss_index_type == 'flat':
            # Exact search with inner product (since embeddings are normalized, this is cosine similarity)
            self.faiss_index = faiss.IndexFlatIP(dimension)
        elif self.faiss_index_type == 'ivf':
            # Approximate search with IVF
            nlist = min(100, n_images // 10)  # Number of clusters
            quantizer = faiss.IndexFlatIP(dimension)
            self.faiss_index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            
            # Train the index
            print("Training IVF index...")
            self.faiss_index.train(embeddings_np)
            self.faiss_index.nprobe = 10  # Number of clusters to search
        else:
            raise ValueError(f"Unknown FAISS index type: {self.faiss_index_type}")
        
        # Add embeddings to index
        self.faiss_index.add(embeddings_np)
        self.image_paths_indexed = image_paths
        
        print(f"✓ FAISS index built with {self.faiss_index.ntotal} vectors")
    
    def search_faiss(self, query: str, top_k: int = 10) -> Tuple[List[str], List[float], float]:
        """
        Search for top-k most relevant images using FAISS
        
        Args:
            query: Text query
            top_k: Number of results to return
            
        Returns:
            retrieved_paths, retrieved_scores, latency_ms
        """
        if self.faiss_index is None:
            raise ValueError("FAISS index not built. Call build_faiss_index() first.")
        
        start_time = time.time()
        
        # Encode query
        query_embedding = self.encode_text([query])
        query_np = query_embedding.cpu().numpy().astype('float32')
        
        # Search FAISS index
        scores, indices = self.faiss_index.search(query_np, top_k)
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Convert to lists
        scores = scores[0].tolist()
        indices = indices[0].tolist()
        
        # Get corresponding image paths
        retrieved_paths = [self.image_paths_indexed[i] for i in indices]
        
        return retrieved_paths, scores, latency_ms
    
    @staticmethod
    def recall_at_k(relevant: List[str], retrieved: List[str], k: int) -> float:
        """Calculate Recall@K"""
        retrieved_k = set(retrieved[:k])
        relevant_set = set(relevant)
        
        if len(relevant_set) == 0:
            return 0.0
        
        hits = len(retrieved_k & relevant_set)
        return hits / len(relevant_set)
    
    @staticmethod
    def mrr_at_k(relevant: List[str], retrieved: List[str], k: int) -> float:
        """Calculate Mean Reciprocal Rank@K"""
        relevant_set = set(relevant)
        
        for i, img in enumerate(retrieved[:k], 1):
            if img in relevant_set:
                return 1.0 / i
        return 0.0
    
    @staticmethod
    def ndcg_at_k(relevant: List[str], retrieved: List[str], scores: List[float], k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain@K"""
        relevant_set = set(relevant)
        
        # DCG
        dcg = 0.0
        for i, (img, score) in enumerate(zip(retrieved[:k], scores[:k]), 1):
            rel = 1.0 if img in relevant_set else 0.0
            dcg += rel / np.log2(i + 1)
        
        # IDCG (ideal DCG)
        ideal_rels = sorted([1.0] * len(relevant_set) + [0.0] * (k - len(relevant_set)), reverse=True)
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_rels[:k]))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def evaluate_queries(self, queries: List[Dict], qrels: List[Dict], 
                        image_paths: List[str], 
                        k_values: List[int] = [1, 3, 5, 10]) -> Tuple[Dict, List[QueryResult]]:
        """
        Evaluate model on a set of queries with qrels using FAISS
        
        Args:
            queries: List of dicts with 'qid' and 'query'
            qrels: List of dicts with 'qid' and 'relevant_images'
            image_paths: List of all image paths in knowledge base
            k_values: K values for metrics
        """
        # Create qrels lookup
        qrels_dict = {q['qid']: q['relevant_images'] for q in qrels}
        
        print(f"\nIndexing {len(image_paths)} images...")
        image_embeddings = self.encode_images(image_paths)
        
        # Extract just filenames from full paths for matching
        image_filenames = [Path(p).name for p in image_paths]
        
        # Build FAISS index
        self.build_faiss_index(image_embeddings, image_filenames)
        
        results = []
        
        print(f"\nEvaluating {len(queries)} queries with FAISS...")
        for query_obj in queries:
            qid = query_obj['qid']
            query_text = query_obj['query']
            
            if qid not in qrels_dict:
                print(f"Warning: No qrels found for {qid}")
                continue
            
            relevant_images = qrels_dict[qid]
            
            # Use FAISS search instead of direct similarity
            retrieved, scores, latency = self.search_faiss(query_text, top_k=max(k_values))
            
            results.append(QueryResult(
                qid=qid,
                query=query_text,
                relevant_images=relevant_images,
                retrieved_images=retrieved,
                scores=scores,
                latency_ms=latency
            ))
        
        # Compute metrics
        metrics = {f'recall@{k}': [] for k in k_values}
        metrics.update({f'mrr@{k}': [] for k in k_values})
        metrics.update({f'ndcg@{k}': [] for k in k_values})
        metrics['latency_ms'] = []
        
        for result in results:
            for k in k_values:
                metrics[f'recall@{k}'].append(
                    self.recall_at_k(result.relevant_images, result.retrieved_images, k)
                )
                metrics[f'mrr@{k}'].append(
                    self.mrr_at_k(result.relevant_images, result.retrieved_images, k)
                )
                metrics[f'ndcg@{k}'].append(
                    self.ndcg_at_k(result.relevant_images, result.retrieved_images, result.scores, k)
                )
            metrics['latency_ms'].append(result.latency_ms)
        
        # Average metrics
        summary = {}
        for metric_name, values in metrics.items():
            if metric_name == 'latency_ms':
                summary[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
            else:
                summary[metric_name] = np.mean(values) * 100  # Convert to percentage
        
        # Get embedding size
        summary['embedding_size'] = {
            'dimensions': image_embeddings.shape[1],
            'dtype': str(image_embeddings.dtype),
            'memory_per_image_kb': (image_embeddings[0].element_size() * image_embeddings.shape[1]) / 1024
        }
        
        # Add FAISS info
        summary['faiss_info'] = {
            'index_type': self.faiss_index_type,
            'num_vectors': self.faiss_index.ntotal
        }
        
        return summary, results


def load_jsonl(filepath: str) -> List[Dict]:
    """Load JSONL file"""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def run_evaluation(image_dir: str, queries_file: str, qrels_file: str, 
                   output_dir: str = 'results', faiss_index_type: str = 'flat'):
    """
    Run complete evaluation across all models with FAISS
    
    Args:
        image_dir: Directory containing images
        queries_file: Path to queries.jsonl
        qrels_file: Path to qrels.jsonl
        output_dir: Where to save results
        faiss_index_type: 'flat' (exact) or 'ivf' (approximate, faster)
    """
    # Load queries and qrels
    queries = load_jsonl(queries_file)
    qrels = load_jsonl(qrels_file)
    
    # Get all images
    image_paths = set()
    for ext in ['jpg', 'jpeg', 'png']:
        image_paths.update([str(p) for p in Path(image_dir).glob(f'*.{ext}')])
        image_paths.update([str(p) for p in Path(image_dir).glob(f'*.{ext.upper()}')])
    image_paths = sorted(list(image_paths))
    
    print(f"Found {len(image_paths)} images")
    print(f"Loaded {len(queries)} queries")
    print(f"Loaded {len(qrels)} qrels")
    print(f"Using FAISS index type: {faiss_index_type}")
    
    evaluator = MultimodalEvaluator(faiss_index_type=faiss_index_type)
    all_results = {}
    
    # Test each model
    failed_models = []
    for model_key in evaluator.MODELS.keys():
        print(f"\n{'='*60}")
        print(f"Testing: {evaluator.MODELS[model_key].name}")
        print(f"{'='*60}")
        
        try:
            evaluator.load_model(model_key)
            summary, detailed_results = evaluator.evaluate_queries(
                queries, qrels, image_paths
            )
            
            all_results[model_key] = {
                'config': evaluator.MODELS[model_key].__dict__,
                'metrics': summary,
                'detailed_results': [
                    {
                        'qid': r.qid,
                        'query': r.query,
                        'relevant': r.relevant_images,
                        'retrieved_top5': r.retrieved_images[:5],
                        'scores_top5': r.scores[:5],
                        'latency_ms': r.latency_ms,
                        'hit@1': r.retrieved_images[0] in r.relevant_images if len(r.retrieved_images) > 0 else False,
                        'hit@5': any(img in r.relevant_images for img in r.retrieved_images[:5])
                    }
                    for r in detailed_results
                ]
            }
            
            # Print summary
            print(f"\n{evaluator.MODELS[model_key].name} Results:")
            print(f"  Recall@1:  {summary['recall@1']:.2f}%")
            print(f"  Recall@3:  {summary['recall@3']:.2f}%")
            print(f"  Recall@5:  {summary['recall@5']:.2f}%")
            print(f"  Recall@10: {summary['recall@10']:.2f}%")
            print(f"  MRR@10:    {summary['mrr@10']:.2f}%")
            print(f"  nDCG@10:   {summary['ndcg@10']:.2f}%")
            print(f"  Latency:   {summary['latency_ms']['mean']:.1f} ± {summary['latency_ms']['std']:.1f} ms")
            print(f"  Embedding: {summary['embedding_size']['dimensions']}D, {summary['embedding_size']['memory_per_image_kb']:.2f} KB/image")
            
        except Exception as e:
            error_msg = f"Failed to evaluate {model_key}: {e}"
            print(f"\n❌ {error_msg}")
            failed_models.append((model_key, str(e)))
            import traceback
            traceback.print_exc()
            continue
    
    # Save results
    Path(output_dir).mkdir(exist_ok=True)
    
    # Save main results
    with open(f"{output_dir}/evaluation_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save failed models log
    if failed_models:
        with open(f"{output_dir}/failed_models.log", 'w') as f:
            for model_key, error in failed_models:
                f.write(f"{model_key}: {error}\n\n")
    
    # Generate comparison table
    if not all_results:
        print("\n❌ No models successfully evaluated!")
        return
    
    print(f"\n{'='*80}")
    print("FINAL COMPARISON (with FAISS retrieval)")
    print(f"{'='*80}")
    
    if failed_models:
        print(f"\n⚠️  Warning: {len(failed_models)} model(s) failed to evaluate:")
        for model_key, error in failed_models:
            model_name = evaluator.MODELS[model_key].name
            print(f"   • {model_name}: {error[:80]}...")
        print()
    
    print(f"{'Model':<30} {'Embed':<8} {'R@1':<6} {'R@3':<6} {'R@5':<6} {'R@10':<6} {'MRR@10':<7} {'nDCG@10':<7} {'Latency (ms)':<15}")
    print("-" * 110)

    for model_key, results in all_results.items():
        name = results['config']['name']
        metrics = results['metrics']
        embed_dim = metrics['embedding_size']['dimensions']
        print(f"{name:<30} "
            f"{embed_dim:>5}D  "
            f"{metrics['recall@1']:>5.1f}% "
            f"{metrics['recall@3']:>5.1f}% "
            f"{metrics['recall@5']:>5.1f}% "
            f"{metrics['recall@10']:>5.1f}% "
            f"{metrics['mrr@10']:>6.2f}% "
            f"{metrics['ndcg@10']:>6.2f}% "
            f"{metrics['latency_ms']['mean']:>7.1f} ± {metrics['latency_ms']['std']:.1f}")
    
    # Generate per-query analysis
    print(f"\n{'='*80}")
    print("PER-QUERY ANALYSIS (Top model performance)")
    print(f"{'='*80}")
    
    # Find best performing model
    best_model = max(all_results.items(), key=lambda x: x[1]['metrics']['recall@1'])
    best_model_key, best_model_results = best_model
    
    print(f"Best Model: {best_model_results['config']['name']}")
    print(f"\n{'QID':<6} {'Hit@1':<7} {'Hit@5':<7} {'Query':<50}")
    print("-" * 70)
    
    for result in best_model_results['detailed_results']:
        query = result['query'][::]
        print(f"{result['qid']:<6} "
              f"{'✓' if result['hit@1'] else '✗':<7} "
              f"{'✓' if result['hit@5'] else '✗':<7} "
              f"{query}")


if __name__ == "__main__":
    # Example usage
    run_evaluation(
        image_dir="./images",
        queries_file="queries.jsonl",
        qrels_file="qrels.jsonl",
        output_dir="results",
        faiss_index_type="flat"  # Use 'ivf' for faster approximate search
    )