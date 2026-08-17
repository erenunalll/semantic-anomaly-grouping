import json
import numpy as np

from sanitize import sanitize
from benchmark import compute_confusion_counts
from semantic_match import get_embeddings, build_index, search_neighbors, top_k_pairs

DATASET_PATH = r"data/eval_dataset.json"

MODEL_NAME = "bge-m3"
MODEL_SERVER_URL = r"http://localhost:11434/v1/embeddings"


def compare_preprocessing_impact(dataset_path, model_name, model_server_url, threshold, top_k):
    with open(dataset_path, encoding="utf-8") as f:
        raw_data = json.load(f)

    ids, raw_texts, sanitized_texts, true_groups = [], [], [], {}
    counter = 0
    for group in raw_data["ground_truth_groups"]:
        for record in group["records"]:
            record_id = f"record_{counter}"
            ids.append(record_id)
            raw_texts.append(record)                  
            sanitized_texts.append(sanitize(record))     
            true_groups[record_id] = group["group_id"]
            counter += 1

    candidate_k = min(len(ids) - 1, max(32, top_k * 4))

    results = {}
    for label, texts in [("raw", raw_texts), ("sanitized", sanitized_texts)]:
        vectors = get_embeddings(texts, model_name, model_server_url)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        index, id_map = build_index(vectors, ids)
        neighbors = search_neighbors(index, vectors, ids, id_map, candidate_k, threshold)
        pairs = top_k_pairs(neighbors, ids, top_k)

        TP, FP, FN, TN = compute_confusion_counts(pairs, ids, true_groups)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results[label] = {
            "precision": precision*100, 
            "recall": recall*100, 
            "f1": f1*100, 
            "TP": TP, 
            "FP": FP, 
            "FN": FN
        }

    return results


def print_preprocess_result(results):
    for label in results.keys():
        print(
            f"{label}\n",
            f"Precision: {results[label]['precision']:.2f}%\n",
            f"Recall: {results[label]['recall']:.2f}%\n",
            f"F1-Score: {results[label]['f1']:.2f}%\n",
            f"TP: {results[label]['TP']}\n",
            f"FP: {results[label]['FP']}\n",
            f"FN: {results[label]['FN']}\n"
        )


def main():
    results = compare_preprocessing_impact(DATASET_PATH, MODEL_NAME, MODEL_SERVER_URL, 0.7, 2)
    print_preprocess_result(results)


if __name__ == "__main__":
    main()