#!/usr/bin/env python3
import argparse, json, math, re, time
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# ----- Optional deps -----
try:
    import faiss
    FAISS_OK = True
except Exception:
    FAISS_OK = False

try:
    from sentence_transformers import SentenceTransformer
    ST_OK = True
except Exception:
    ST_OK = False

try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    HF_OK = True
except Exception:
    HF_OK = False

try:
    from FlagEmbedding import BGEM3FlagModel
    FLAG_OK = True
except Exception:
    FLAG_OK = False

try:
    from rank_bm25 import BM25Okapi
    BM25_OK = True
except Exception:
    BM25_OK = False

DEFAULT_KSET = [1, 3, 5, 10, 15, 20, 50]


# ---------- IO ----------
def safe_name(model_name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]+', '-', model_name)

def load_chunks(path: str) -> Tuple[List[str], List[str]]:
    pids, texts = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            pid = r.get("pid") or r.get("passage_id") or r.get("id") or r.get("doc_id") or r.get("docid")
            if pid is None:
                raise ValueError("Each chunk row must have an id key (pid/passage_id/id/doc_id/docid).")
            txt = r.get("text") or r.get("content") or r.get("body")
            if txt is None:
                raise ValueError("Each chunk row must have text/content/body.")
            pids.append(str(pid)); texts.append(str(txt))
    return pids, texts

def load_gold(path: str):
    qids, queries, gold, behaviors = [], [], {}, {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            qid = str(r.get("qid") or r.get("id") or r.get("query_id"))
            q = r.get("query") or r.get("question") or r.get("text")
            rel = r.get("relevant_passages") or r.get("relevant_pids") or r.get("positives") or r.get("labels") or []
            beh = r.get("expected_behavior", "answer" if rel else "abstain")
            if qid is None or q is None:
                raise ValueError("Gold rows need qid and query.")
            qids.append(qid); queries.append(str(q)); gold[qid] = [str(x) for x in rel]
            behaviors[qid] = beh
    return qids, queries, gold, behaviors

# ---------- Metrics ----------
def _dcg(rels): return sum((rel / math.log2(i + 2)) for i, rel in enumerate(rels))
def ndcg_at_k(rank_list, relevant, k):
    hits = [1 if pid in relevant else 0 for pid in rank_list[:k]]
    ideal = sorted(hits, reverse=True); idcg = _dcg(ideal)
    return (_dcg(hits) / idcg) if idcg > 0 else 0.0

def mrr_at_k(rank_list, relevant, k):
    for i, pid in enumerate(rank_list[:k], start=1):
        if pid in relevant: return 1.0 / i
    return 0.0

def recall_at_k(rank_list, relevant, k):
    return 1.0 if any(pid in relevant for pid in rank_list[:k]) else 0.0

def map_at_k(rank_list, relevant, k):
    hits, score = 0, 0.0
    for i, pid in enumerate(rank_list[:k], start=1):
        if pid in relevant:
            hits += 1; score += hits / i
    denom = min(len(relevant), k)
    return score / denom if denom > 0 else 0.0


# ---------- Search ----------
def build_faiss_ip_index(doc_embs: np.ndarray):
    dim = doc_embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(doc_embs.astype(np.float32))
    return index

def search_numpy_cosine(doc_embs: np.ndarray, query_embs: np.ndarray, topk: int):
    def l2norm(x):
        n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        return x / n
    d = l2norm(doc_embs); q = l2norm(query_embs)
    sims = q @ d.T
    k = min(topk, sims.shape[1])
    idx = np.argpartition(-sims, kth=k-1, axis=1)[:, :k]
    part = np.take_along_axis(sims, idx, axis=1)
    order = np.argsort(-part, axis=1)
    idx = np.take_along_axis(idx, order, axis=1)
    scores = np.take_along_axis(sims, idx, axis=1)
    return scores, idx


# ---------- Embedding adapters ----------
class BaseEmbedder:
    def encode_passages(self, texts: List[str]): raise NotImplementedError
    def encode_queries(self, texts: List[str]):  raise NotImplementedError

class STEmbedder(BaseEmbedder):
    def __init__(self, model_name: str, device: Optional[str], batch_size: int, max_length: int, normalize: bool=True):
        if not ST_OK:
            raise RuntimeError("sentence-transformers not installed")
        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = max_length
        self.batch = batch_size
        self.normalize = normalize

    def _enc(self, texts: List[str]) -> np.ndarray:
        embs = self.model.encode(texts, batch_size=self.batch, convert_to_numpy=True,
                                 show_progress_bar=False, normalize_embeddings=self.normalize)
        return embs.astype("float32")

    def encode_passages(self, texts: List[str]): return {"dense_vecs": self._enc(texts)}
    def encode_queries(self, texts: List[str]):  return {"dense_vecs": self._enc(texts)}

class HFMeanPoolEmbedder(BaseEmbedder):
    def __init__(self, model_name: str, device: Optional[str], batch_size: int, max_length: int,
                 fp16: bool=True, trust_remote_code: bool=True, prefix_q: str="", prefix_p: str=""):
        if not HF_OK:
            raise RuntimeError("transformers/torch not installed")
        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        self.model.eval()
        self.device = device if device else ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.batch = batch_size
        self.max_length = max_length
        self.normalize = True
        self.fp16 = fp16 and str(self.device).startswith("cuda")
        self.prefix_q, self.prefix_p = prefix_q, prefix_p
        if self.fp16:
            try:
                self.model = self.model.half()
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

    @staticmethod
    def _mean_pool(last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def _encode(self, texts: List[str]) -> np.ndarray:
        outs = []
        for i in range(0, len(texts), self.batch):
            batch = texts[i:i+self.batch]
            with torch.no_grad():
                toks = self.tok(batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
                if self.fp16:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        out = self.model(**toks, output_hidden_states=False)
                else:
                    out = self.model(**toks, output_hidden_states=False)
                reps = self._mean_pool(out.last_hidden_state, toks["attention_mask"])
                reps = torch.nn.functional.normalize(reps, p=2, dim=1)
            outs.append(reps.detach().float().cpu().numpy())
        return np.vstack(outs).astype("float32")

    def encode_passages(self, texts: List[str]):
        return {"dense_vecs": self._encode([self.prefix_p + t for t in texts])}

    def encode_queries(self, texts: List[str]):
        return {"dense_vecs": self._encode([self.prefix_q + t for t in texts])}

class BGEM3FlagAdapter(BaseEmbedder):
    def __init__(self, model_name: str, batch_size: int, max_length: int, fp16: bool=True,
                 prefix_q: str="", prefix_p: str=""):
        if not FLAG_OK:
            raise RuntimeError("FlagEmbedding not installed")
        self.model = BGEM3FlagModel(model_name, use_fp16=fp16)
        self.batch = batch_size
        self.max_length = max_length
        self.prefix_q, self.prefix_p = prefix_q, prefix_p

    def encode_passages(self, texts: List[str]):
        return self.model.encode([self.prefix_p + t for t in texts], batch_size=self.batch, max_length=self.max_length)

    def encode_queries(self, texts: List[str]):
        return self.model.encode_queries([self.prefix_q + t for t in texts], batch_size=self.batch, max_length=self.max_length)

    def hybrid_scores_topk(self, q_emb, p_emb, topk=50, w_dense=0.5, w_sparse=0.5, w_colbert=0.0):
        scores = self.model.compute_score(
            query_embeddings=q_emb,
            corpus_embeddings=p_emb,
            weight_dense=w_dense, weight_sparse=w_sparse, weight_colbert=w_colbert,
            normalize=True
        )
        k = min(topk, scores.shape[1])
        idx = np.argpartition(-scores, kth=k-1, axis=1)[:, :k]
        part = np.take_along_axis(scores, idx, axis=1)
        order = np.argsort(-part, axis=1)
        idx = np.take_along_axis(idx, order, axis=1)
        top_scores = np.take_along_axis(scores, idx, axis=1)
        return top_scores.astype(np.float32), idx.astype(np.int32)

class BM25Adapter(BaseEmbedder):
    def __init__(self, passages: List[str]):
        if not BM25_OK:
            raise RuntimeError("rank_bm25 not installed")
        tokenized = [p.split() for p in passages]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, queries: List[str], topk: int):
        idx_all, scores_all = [], []
        for q in tqdm(queries, desc="[bm25] search"):
            scores = self.bm25.get_scores(q.split()).astype(np.float32)
            k = min(topk, scores.shape[0])
            idx = np.argpartition(-scores, kth=k-1)[:k]
            part = scores[idx]
            order = np.argsort(-part)
            idx = idx[order]
            idx_all.append(idx)
            scores_all.append(scores[idx])
        return np.stack(scores_all), np.stack(idx_all)

# --- EmbeddingGemma (Sentence-Transformers) ---
class EmbeddingGemmaAdapter(BaseEmbedder):
    def __init__(self, model_name: str, device: Optional[str], batch_size: int, max_length: int):
        if not ST_OK:
            raise RuntimeError("sentence-transformers not installed")
        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = max_length
        self.batch = batch_size

    def encode_passages(self, texts: List[str]):
        embs = self.model.encode_document(
            texts, batch_size=self.batch, convert_to_numpy=True, show_progress_bar=False
        ).astype("float32")
        return {"dense_vecs": embs}

    def encode_queries(self, texts: List[str]):
        embs = self.model.encode_query(
            texts, batch_size=self.batch, convert_to_numpy=True, show_progress_bar=False
        ).astype("float32")
        return {"dense_vecs": embs}

# ------------- Registry -------------
def build_embedder(model: str, backend: str, device: Optional[str], batch: int, max_length: int, mode: str):
    m = model.lower()
    if backend == "bm25" or m == "bm25":
        return "bm25"
    
    if "embeddinggemma" in m or m == "google/embeddinggemma-300m":
        return EmbeddingGemmaAdapter(model_name=model, device=device, batch_size=batch, max_length=max_length)
    
    if "bge-m3" in m:
        if mode in ("hybrid", "auto") and FLAG_OK:
            return BGEM3FlagAdapter(model_name=model, batch_size=batch, max_length=max_length, fp16=True)
        return HFMeanPoolEmbedder(model_name=model, device=device, batch_size=batch, max_length=max_length, fp16=True, trust_remote_code=True)
    
    if "minilm" in m or "sentence-transformers" in m:
        return STEmbedder(model_name=model, device=device, batch_size=batch, max_length=max_length, normalize=True)
    
    # generic HF (GTE, Qwen, E5, etc.)
    return HFMeanPoolEmbedder(model_name=model, device=device, batch_size=batch, max_length=max_length, fp16=True, trust_remote_code=True)


# ---------- Evaluation ----------
def evaluate(
    corpus_path: str,
    gold_path: str,
    out_dir: str,
    model_name: str,
    device: Optional[str],
    topk: int,
    index: str,              # faiss | numpy | hybrid | bm25
    batch_size: int,
    max_length: int,
    mode: str = "auto",      # for bge-m3: auto | hybrid | dense
    w_dense: float = 0.5,
    w_sparse: float = 0.5,
    w_colbert: float = 0.0,
    sim_threshold: float = 0.28,
    query_prefix: str = "",
    passage_prefix: str = "",
    sweep_thresholds: str = "",                
    sweep_objective: str = "balanced_accuracy"  
):
    # --- small helper (inline) ---
    def parse_thresholds(spec: str):
        """Accept '0.2,0.24,0.28' or 'start:end:step' like '0.18:0.36:0.02'."""
        if not spec:
            return []
        spec = spec.strip()
        if ":" in spec:
            a, b, c = spec.split(":")
            start, end, step = float(a), float(b), float(c)
            n = int(round((end - start) / step)) + 1
            return [round(start + i * step, 6) for i in range(n)]
        return [float(x) for x in spec.split(",")]

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    pids, passages = load_chunks(corpus_path)
    qids, queries, gold, behaviors = load_gold(gold_path)

    tag = safe_name(model_name)
    retrieval_path = out_dir / f"{tag}_retrieval.jsonl"

    # BM25 short-circuit
    if index == "bm25" or model_name.lower() == "bm25":
        if not BM25_OK:
            raise RuntimeError("Install rank_bm25 to run lexical BM25 (pip install rank-bm25)")
        bm = BM25Adapter(passages)
        scores, idx = bm.search(queries, topk)
        per_query_ms = [0.0] * len(queries)
        passage_encode_s = 0.0
        index_kind = "bm25"
    else:
        emb = build_embedder(model_name, backend=index, device=device, batch=batch_size, max_length=max_length, mode=mode)

        # Encode passages
        t0 = time.time()
        if isinstance(emb, BGEM3FlagAdapter):
            p_emb = emb.encode_passages(passages)
            dense_mat = p_emb.get("dense_vecs")
        else:
            if isinstance(emb, HFMeanPoolEmbedder):
                emb.prefix_q, emb.prefix_p = query_prefix, passage_prefix
            p_emb = emb.encode_passages(passages)
            dense_mat = p_emb["dense_vecs"]
        passage_encode_s = time.time() - t0

        # Build index
        index_kind = "hybrid" if (isinstance(emb, BGEM3FlagAdapter) and index in ("hybrid", "auto")) else ("faiss_ip" if (index == "faiss" and FAISS_OK) else "numpy_cosine")
        if index_kind == "faiss_ip":
            faiss_index = build_faiss_ip_index(dense_mat)

        # Encode queries (batched) and search
        per_query_ms, scores_list, idx_list = [], [], []
        for i in tqdm(range(0, len(queries), batch_size), desc=f"[{tag}] encode/search"):
            batch = queries[i:i+batch_size]
            t = time.time()
            q_emb = emb.encode_queries(batch)
            dt = (time.time() - t) / max(1, len(batch)) * 1000.0
            per_query_ms.extend([dt] * len(batch))
            if index_kind == "hybrid":
                s, ix = emb.hybrid_scores_topk(q_emb, p_emb, topk, w_dense, w_sparse, w_colbert)
            elif index_kind == "faiss_ip":
                s, ix = faiss_index.search(q_emb["dense_vecs"], topk)
            else:
                s, ix = search_numpy_cosine(dense_mat, q_emb["dense_vecs"], topk)
            scores_list.append(s); idx_list.append(ix)
        scores = np.vstack(scores_list); idx = np.vstack(idx_list)

    # Retrieval metrics
    Kset = [k for k in DEFAULT_KSET if k <= topk]
    agg = {f"Recall@{k}": 0.0 for k in Kset}
    agg.update({f"MRR@{k}": 0.0 for k in Kset})
    agg.update({f"nDCG@{k}": 0.0 for k in Kset})
    agg.update({f"MAP@{k}": 0.0 for k in Kset})

    # Abstain counters
    total = len(qids)
    should_abstain = sum(1 for qid in qids if behaviors.get(qid, "answer") == "abstain")
    should_answer = total - should_abstain
    pred_abstain = 0
    tn_correct = 0   # correct abstains on no-answer
    fn_on_answer = 0 # wrongly abstained when should answer

    # For sweep
    top_scores_all: List[float] = []
    should_flags: List[bool] = []

    with open(retrieval_path, "w", encoding="utf-8") as fout:
        for qi in range(len(qids)):
            qid = qids[qi]
            cand_idx = idx[qi].tolist()
            cand_scores = scores[qi].tolist()
            ranked_pids = [pids[j] for j in cand_idx]
            relset = set(gold.get(qid, []))
            top_score = float(cand_scores[0]) if len(cand_scores) > 0 else -1e9

            # Retrieval metrics
            for k in Kset:
                agg[f"Recall@{k}"] += recall_at_k(ranked_pids, relset, k)
                agg[f"MRR@{k}"] += mrr_at_k(ranked_pids, relset, k)
                agg[f"nDCG@{k}"] += ndcg_at_k(ranked_pids, relset, k)
                agg[f"MAP@{k}"] += map_at_k(ranked_pids, relset, k)

            # Abstain prediction at provided sim_threshold
            pred_is_abstain = (top_score < sim_threshold) or (len(cand_scores) == 0)
            if pred_is_abstain:
                pred_abstain += 1

            should = behaviors.get(qid, "answer")
            if should == "abstain" and pred_is_abstain:
                tn_correct += 1
            elif should == "answer" and pred_is_abstain:
                fn_on_answer += 1

            # Collect for sweep
            top_scores_all.append(top_score)
            should_flags.append(should == "abstain")

            fout.write(json.dumps({
                "qid": qid,
                "query": queries[qi],
                "candidates": [{"pid": pids[j], "score": float(cand_scores[r])} for r, j in enumerate(cand_idx)],
                "relevant_pids": list(relset),
                "expected_behavior": should,
                "top_score": top_score,
                "pred_abstain": bool(pred_is_abstain)
            }, ensure_ascii=False) + "\n")

    n = max(1, len(qids))
    for k in list(agg.keys()):
        agg[k] = round(agg[k] / n, 6)

    # Abstain metrics at provided sim_threshold
    abstain_accuracy = (tn_correct / should_abstain) if should_abstain > 0 else None
    false_answer_rate = (1 - abstain_accuracy) if abstain_accuracy is not None else None
    answer_false_abstain_rate = (fn_on_answer / should_answer) if should_answer > 0 else None
    fp_on_abstain = should_abstain - tn_correct

    # -------- Threshold sweep (no re-encoding) --------
    def _eval_thr(thr: float):
        TP = TN = FP = FN = 0
        for ts, should_abst in zip(top_scores_all, should_flags):
            pred_abst = (ts < thr)
            if  pred_abst and  should_abst: TP += 1
            if (not pred_abst) and (not should_abst): TN += 1
            if  pred_abst and (not should_abst): FP += 1
            if (not pred_abst) and  should_abst: FN += 1

        total_pos = TP + FN
        total_neg = TN + FP
        abst_acc = (TP / total_pos) if total_pos > 0 else None       # accuracy on abstain class
        answ_acc = (TN / total_neg) if total_neg > 0 else None       # accuracy on answer class
        bal_acc  = (0.5 * (abst_acc + answ_acc)) if (abst_acc is not None and answ_acc is not None) else None
        prec     = (TP / (TP + FP)) if (TP + FP) > 0 else None
        rec      = (TP / (TP + FN)) if (TP + FN) > 0 else None
        f1_abs   = (2*prec*rec/(prec+rec)) if (prec is not None and rec is not None and (prec+rec)>0) else None

        return {
            "thr": thr, "TP": TP, "TN": TN, "FP": FP, "FN": FN,
            "abstain_accuracy": abst_acc,
            "answer_accuracy": answ_acc,
            "balanced_accuracy": bal_acc,
            "precision_abstain": prec,
            "recall_abstain": rec,
            "f1_abstain": f1_abs
        }

    sweep_list = parse_thresholds(sweep_thresholds)
    best_thr = None
    best_score = -1.0
    sweep_results = []

    if sweep_list:
        for t in sweep_list:
            r = _eval_thr(t)
            sweep_results.append(r)
            metric = {
                "abstain_accuracy": r["abstain_accuracy"] if r["abstain_accuracy"] is not None else -1,
                "balanced_accuracy": r["balanced_accuracy"] if r["balanced_accuracy"] is not None else -1,
                "f1_abstain": r["f1_abstain"] if r["f1_abstain"] is not None else -1
            }[sweep_objective]
            if metric > best_score:
                best_score = metric
                best_thr = r

        # save sweep table
        sweep_df = pd.DataFrame(sweep_results)
        sweep_csv = out_dir / f"{safe_name(model_name)}_sweep.csv"
        sweep_df.to_csv(sweep_csv, index=False)
        print(f"[SWEEP] Saved {sweep_csv}")
        print(f"[SWEEP] Best {sweep_objective}: {best_score:.6f} at threshold={best_thr['thr']}")

    # Summary row
    row = {
        "model": model_name,
        "index": index_kind if index != "bm25" else "bm25",
        "topk": topk,
        "num_passages": len(pids),
        "num_queries": len(qids),
        "Recall@1": agg.get("Recall@1", 0.0),
        "Recall@3": agg.get("Recall@3", 0.0),
        "Recall@5": agg.get("Recall@5", 0.0),
        "Recall@10": agg.get("Recall@10", 0.0),
        "Recall@15": agg.get("Recall@15", 0.0),
        "Recall@20": agg.get("Recall@20", 0.0),
        "Recall@50": agg.get("Recall@50", 0.0),
        "MRR@1": agg.get("MRR@1", 0.0),
        "MRR@3": agg.get("MRR@3", 0.0),
        "MRR@5": agg.get("MRR@5", 0.0),
        "MRR@10": agg.get("MRR@10", 0.0),
        "MRR@15": agg.get("MRR@15", 0.0),
        "MRR@20": agg.get("MRR@20", 0.0),
        "MRR@50": agg.get("MRR@50", 0.0),
        "nDCG@1": agg.get("nDCG@1", 0.0),
        "nDCG@3": agg.get("nDCG@3", 0.0),
        "nDCG@5": agg.get("nDCG@5", 0.0),
        "nDCG@10": agg.get("nDCG@10", 0.0),
        "nDCG@15": agg.get("nDCG@15", 0.0),
        "nDCG@20": agg.get("nDCG@20", 0.0),
        "nDCG@50": agg.get("nDCG@50", 0.0),
        "MAP@1": agg.get("MAP@1", 0.0),
        "MAP@3": agg.get("MAP@3", 0.0),
        "MAP@5": agg.get("MAP@5", 0.0),
        "MAP@10": agg.get("MAP@10", 0.0),
        "MAP@15": agg.get("MAP@15", 0.0),
        "MAP@20": agg.get("MAP@20", 0.0),
        "MAP@50": agg.get("MAP@50", 0.0),
        "sim_threshold": sim_threshold,
        "should_abstain": should_abstain,
        "pred_abstain": pred_abstain,
        "tn_correct": tn_correct,                   
        "fp_on_abstain": fp_on_abstain,              
        "fn_on_answer": fn_on_answer,                
        "abstain_accuracy": round(abstain_accuracy, 6) if abstain_accuracy is not None else "",
        "false_answer_rate": round(false_answer_rate, 6) if false_answer_rate is not None else "",
        "answer_false_abstain_rate": round(answer_false_abstain_rate, 6) if answer_false_abstain_rate is not None else "",
        "passage_encode_s": round(passage_encode_s, 6),
        "avg_query_encode_ms": round(float(np.mean(per_query_ms)), 6) if per_query_ms else "",
        
    }

    if sweep_list and best_thr is not None:
        row.update({
            "sweep_objective": sweep_objective,
            "sweep_best_threshold": best_thr["thr"],
            "sweep_best_score": round(best_score, 6),
            "sweep_best_abstain_acc": round(best_thr["abstain_accuracy"], 6) if best_thr["abstain_accuracy"] is not None else "",
            "sweep_best_answer_acc": round(best_thr["answer_accuracy"], 6) if best_thr["answer_accuracy"] is not None else "",
            "sweep_best_f1_abstain": round(best_thr["f1_abstain"], 6) if best_thr["f1_abstain"] is not None else ""
        })

    cols = list(row.keys())
    df = pd.DataFrame([[row.get(c, "") for c in cols]], columns=cols)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{safe_name(model_name)}_summary.csv"
    df.to_csv(out_csv, index=False)

    print(df.to_string(index=False))
    print("→", out_csv)
    print("→", retrieval_path)


# ---------- CLI ----------
def main():
    p = argparse.ArgumentParser(description="Unified evaluator with abstain metrics")
    p.add_argument("--corpus", default="corpus.jsonl", type=str)
    p.add_argument("--gold", default="eval_set_silver_plus.jsonl", type=str)
    p.add_argument("--out", default="Outputs", type=str)
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", type=str,
                   help="HF or ST model id (or 'bm25')")
    p.add_argument("--backend", choices=["auto","faiss","numpy","hybrid","bm25"], default="faiss",
                   help="faiss|numpy; 'hybrid' for bge-m3; 'bm25' for lexical")
    p.add_argument("--mode", choices=["auto","hybrid","dense"], default="auto",
                   help="For bge-m3: auto prefers hybrid if available")
    p.add_argument("--topk", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--device", type=str, default=None, help="cpu | cuda | cuda:0")
    p.add_argument("--w-dense", type=float, default=0.5)
    p.add_argument("--w-sparse", type=float, default=0.5)
    p.add_argument("--w-colbert", type=float, default=0.0)
    p.add_argument("--sim-threshold", type=float, default=0.28,
                   help="Threshold on top score to abstain (tune on a dev split)")
    p.add_argument("--query-prefix", type=str, default="")
    p.add_argument("--passage-prefix", type=str, default="")
    p.add_argument("--sweep-thresholds", type=str, default="",
               help="Sweep thresholds to auto-tune abstain. Formats: '0.18,0.20,0.22' or 'start:end:step' (e.g. '0.18:0.36:0.02').")
    p.add_argument("--sweep-objective", type=str, default="abstain_accuracy",
               choices=["abstain_accuracy", "balanced_accuracy", "f1_abstain"],
               help="Metric to maximize when choosing threshold.")

    args = p.parse_args()

    evaluate(
        corpus_path=args.corpus,
        gold_path=args.gold,
        out_dir=args.out,
        model_name=args.model,
        device=args.device,
        topk=args.topk,
        index=args.backend,
        batch_size=args.batch_size,
        max_length=args.max_length,
        mode=args.mode,
        w_dense=args.w_dense, w_sparse=args.w_sparse, w_colbert=args.w_colbert,
        sim_threshold=args.sim_threshold,
        query_prefix=args.query_prefix, passage_prefix=args.passage_prefix,
        sweep_thresholds=args.sweep_thresholds,
        sweep_objective=args.sweep_objective
    )

if __name__ == "__main__":
    main()

"""
============================
Usage Examples
============================

# ---- Lexical baseline ----
python .\Embedding\eval_unified_plus.py --model bm25 --backend bm25 --corpus corpus.jsonl --gold eval_set_test.jsonl --out Outputs/Outputs_Test/BM25 --topk 50 --sim-threshold (-1e9)

# ---- Sentence-Transformers (MiniLM) ----
python .\Embedding\eval_unified_plus.py --corpus corpus.jsonl --gold eval_set_dev.jsonl --out Outputs/Outputs_Dev --model sentence-transformers/all-MiniLM-L6-v2 --backend faiss --sweep-thresholds 0.15:0.60:0.01 --sweep-objective balanced_accuracy
python .\Embedding\eval_unified_plus.py --corpus corpus.jsonl --gold eval_set_test.jsonl --out Outputs/Outputs_Test --model sentence-transformers/all-MiniLM-L6-v2 --backend faiss --sim-threshold 0.43

python .\Embedding\eval_unified_plus.py --corpus corpus.jsonl --gold eval_set_dev.jsonl --out Outputs/Outputs_Dev --model sentence-transformers/all-MiniLM-L12-v2 --backend faiss --sweep-thresholds 0.0:1.0:0.01 --sweep-objective balanced_accuracy
python .\Embedding\eval_unified_plus.py --corpus corpus.jsonl --gold eval_set_test.jsonl --out Outputs/Outputs_Test --model sentence-transformers/all-MiniLM-L12-v2 --backend faiss --sim-threshold 0.48


# ---- EmbeddingGemma ----
python .\Embedding\eval_unified_plus.py --corpus corpus.jsonl --gold eval_set_dev.jsonl --out Outputs/Outputs_Dev/embeddinggemma-300m --model google/embeddinggemma-300m --backend faiss --device cuda:0 --topk 50 --max-length 256 --query-prefix "query: " --passage-prefix "passage: " --sweep-thresholds 0.00:0.60:0.01 --sweep-objective balanced_accuracy
python .\Embedding\eval_unified_plus.py --corpus corpus.jsonl --gold eval_set_test.jsonl --out Outputs/Outputs_Test/embeddinggemma-300m --model google/embeddinggemma-300m --backend faiss --device cuda:0 --topk 50 --max-length 256 --sim-threshold 0.41


# ---- BGE-M3 (Dense) ----
python .\Embedding\eval_unified_plus.py --corpus corpus.jsonl --gold eval_set_dev.jsonl --out Outputs/Outputs_Dev/bge-m3 --model BAAI/bge-m3 --backend faiss --mode dense --device cuda:0 --topk 50 --max-length 256 --sweep-thresholds 0.0:1.0:0.01 --sweep-objective balanced_accuracy
python .\Embedding\eval_unified_plus.py --corpus corpus.jsonl --gold eval_set_test.jsonl --out Outputs/Outputs_Test/bge-m3 --model BAAI/bge-m3 --backend faiss --mode dense --device cuda:0 --topk 50 --max-length 256 --sim-threshold 0.77


# ---- GTE Multilingual ----
python .\Embedding\eval_unified_plus.py --corpus corpus.jsonl --gold eval_set_dev.jsonl --out Outputs/Outputs_Dev/gte-multilingual-base --model Alibaba-NLP/gte-multilingual-base --backend faiss --device cuda:0 --topk 50 --max-length 256 --sweep-thresholds 0.70:1.00:0.01 --sweep-objective balanced_accuracy
python .\Embedding\eval_unified_plus.py --corpus corpus.jsonl --gold eval_set_test.jsonl --out Outputs/Outputs_Test/gte-multilingual-base --model Alibaba-NLP/gte-multilingual-base --backend faiss --device cuda:0 --topk 50 --max-length 256 --sim-threshold 0.9


# ---- Qwen embeddings ----
python .\Embedding\eval_unified_plus.py --corpus corpus.jsonl --gold eval_set_dev.jsonl --out Outputs/Outputs_Dev/Qwen --model Qwen/Qwen3-Embedding-0.6B --backend faiss --device cuda:0 --topk 50 --max-length 256 --sweep-thresholds 0.0:1.0:0.01 --sweep-objective balanced_accuracy
python .\Embedding\eval_unified_plus.py --corpus corpus.jsonl --gold eval_set_test.jsonl --out Outputs/Outputs_Test/Qwen --model Qwen/Qwen3-Embedding-0.6B --backend faiss --device cuda:0 --topk 50 --max-length 256 --sim-threshold 0.76

"""
