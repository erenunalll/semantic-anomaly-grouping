import json
import time
import numpy as np

import os
import psutil

from semantic_match import get_embeddings, build_index, search_neighbors, top_k_pairs
from sanitize import sanitize

MODELS = ["all-minilm", "nomic-embed-text", "bge-m3"]
MODEL_SERVER_URL = r"http://localhost:11434/v1/embeddings"
DATASET_PATH = r"data/eval_dataset.json"


def load_dataset(path, threshold=0.7, top_k=2):
    with open(path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    ids = []
    texts = []
    true_groups = {}

    counter = 0
    for group in dataset["ground_truth_groups"]:
        for record in group["records"]:
            record_id = f"record_{counter}"
            ids.append(record_id)
            texts.append(sanitize(record))
            true_groups[record_id] = group["group_id"]
            counter += 1

    return {
        "ids": ids,
        "texts": texts,
        "true_groups": true_groups,
        "threshold": threshold,
        "top_k": top_k,
    }


def get_embed_index_metadata(model_name, model_server_url, texts, ids):
    # warming call
    get_embeddings([texts[0]], model_name, model_server_url)

    process = psutil.Process(os.getpid())
    ram_before = process.memory_info().rss / (1024 * 1024)

    t_before_emb = time.perf_counter()
    vectors = get_embeddings(texts, model_name, model_server_url)
    t_elapsed_emb = (time.perf_counter() - t_before_emb) * 1000

    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    t_before_index = time.perf_counter()
    index, id_map = build_index(vectors, ids)
    t_elapsed_index = (time.perf_counter() - t_before_index) * 1000

    ram_footprint = (process.memory_info().rss / (1024 * 1024)) - ram_before
    return {
        "index":index,
        "id_map":id_map,
        "embeddings":vectors,
        "embed_latency":t_elapsed_emb,
        "vector_dim":vectors.shape[1],
        "index_latency":t_elapsed_index,
        "ram_usage":ram_footprint
    }


def compute_confusion_counts(pairs, ids, true_groups):
    TP = 0
    FP = 0
    for left_id, right_id in pairs:
        if true_groups[left_id] == true_groups[right_id]:
            TP += 1
        else:
            FP += 1

    found_pairs = set(pairs.keys())
    FN = 0
    TN = 0
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            id_1, id_2 = ids[i], ids[j]
            if id_1 > id_2:
                id_1, id_2 = id_2, id_1
            pair = (id_1, id_2)
            is_found = pair in found_pairs
            same_group = true_groups[id_1] == true_groups[id_2]

            if same_group and not is_found:
                FN += 1
            elif not same_group and not is_found:
                TN += 1

    return TP, FP, FN, TN


def compute_ann_recall(index, embeddings, top_k):
    similarity_matrix = embeddings @ embeddings.T
    ann_recalls = []

    for i in range(len(embeddings)):
        matches = index.search(embeddings[i], top_k + 1)
        ann_neighbors = set()
        for idx in matches.keys:
            if idx != i:
                ann_neighbors.add(idx)
            if len(ann_neighbors) == top_k:
                break

        sorted_indices = np.argsort(similarity_matrix[i])[::-1]
        exact_neighbors = set()
        for idx in sorted_indices:
            if idx != i:
                exact_neighbors.add(idx)
            if len(exact_neighbors) == top_k:
                break

        intersection = len(ann_neighbors & exact_neighbors)
        exact_count = len(exact_neighbors)
        ann_recalls.append(intersection / exact_count)

    return np.mean(ann_recalls)


def run_benchmark(model_name, dataset):
    ids = dataset["ids"]
    texts = dataset["texts"]
    true_groups = dataset["true_groups"]
    threshold = dataset["threshold"]
    top_k = dataset["top_k"]
    candidate_k = min(len(texts) - 1, max(32, top_k*4))

    emb_idx_metadata = get_embed_index_metadata(model_name, MODEL_SERVER_URL, texts, ids)

    t_before_query = time.perf_counter()
    neighbors = search_neighbors(
        emb_idx_metadata["index"],
        emb_idx_metadata["embeddings"],
        ids,
        emb_idx_metadata["id_map"],
        candidate_k,
        threshold
    )
    t_elapsed_query = (time.perf_counter() - t_before_query) * 1000

    pairs = top_k_pairs(neighbors, ids, top_k)

    TP, FP, FN, TN = compute_confusion_counts(pairs, ids, true_groups)
    accuracy = (TP + TN)/(TP + TN + FN + FP)
    ann_recall = compute_ann_recall(emb_idx_metadata["index"], emb_idx_metadata["embeddings"], top_k)

    return {
        "model_name":model_name,
        "top_k":top_k,
        "embed_latency":(emb_idx_metadata["embed_latency"]/len(texts))*100,
        "vector_dim":emb_idx_metadata["vector_dim"],
        "index_latency":emb_idx_metadata["index_latency"],
        "query_latency":t_elapsed_query,
        "ann_recall":ann_recall*100,
        "ram_usage":emb_idx_metadata["ram_usage"],
        "accuracy":accuracy*100
    }


def print_result(result):
    print(
        f"                   Model: {result['model_name']}\n",
        f"       Vector Dimension: {result['vector_dim']}\n",
        f"      Embedding Latency: {result['embed_latency']:.2f} ms per 100 items.\n",
        f"    Index Build Latency: {result['index_latency']:.2f} ms\n",
        f"          Query Latency: {result['query_latency']:.2f} ms\n"
        f"    ANN Recall @ top_k={result['top_k']}: {result['ann_recall']:.2f}%\n",
        f"Semantic Match Accuracy: {result['accuracy']:.2f}%\n",
        f"              RAM Usage: {result['ram_usage']:.2f} MB\n"
    )


def main(trial_count=1):
    dataset = load_dataset(DATASET_PATH, 0.7, 2)

    for model in MODELS:

        embed_latencs = []
        index_latencs = []
        query_latencs = []
        ram_usages = []

        for trial in range(trial_count):
            result = run_benchmark(model, dataset)
            embed_latencs.append(result["embed_latency"])
            index_latencs.append(result["index_latency"])
            query_latencs.append(result["query_latency"])
            ram_usages.append(result["ram_usage"])

        result["embed_latency"] = np.mean(embed_latencs)
        result["index_latency"] = np.mean(index_latencs)
        result["query_latency"] = np.mean(query_latencs)
        result["ram_usage"] = np.mean(ram_usages)
        
        print_result(result)


if __name__ == "__main__":
    main()