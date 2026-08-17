import numpy as np
import matplotlib.pyplot as plt

from semantic_match import get_embeddings, build_index, search_neighbors, top_k_pairs
from benchmark import load_dataset, compute_confusion_counts

MODEL_NAME = "bge-m3"
MODEL_SERVER_URL = r"http://localhost:11434/v1/embeddings"
THRESHOLDS = np.round(np.linspace(0.50, 0.95, 10), 2)
DATASET_PATH = r"data/eval_dataset.json"


def metrics(index, vectors, ids, id_map, texts, top_k, true_groups):
    precisions =  []
    recalls = []
    f1s = []
    for threshold in THRESHOLDS:
            neighbors = search_neighbors(index, vectors, ids, id_map, min(len(texts)-1, max(32, top_k*4)), threshold)
            pairs = top_k_pairs(neighbors, ids, top_k)
            TP, FP, FN, TN = compute_confusion_counts(pairs, ids, true_groups)

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = 2*((precision*recall)/(precision+recall)) if precision+recall != 0 else 0

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

    return  precisions, recalls, f1s


def plot_metrics_thresholds(precisions, recalls, f1s):
    plt.figure(figsize=(8,5))

    plt.plot(THRESHOLDS, precisions, linewidth=2, label="Precision")
    plt.plot(THRESHOLDS, recalls, linewidth=2, label="Recall")
    plt.plot(THRESHOLDS, f1s, linewidth=2, label="F1-Score")

    best_idx = np.argmax(f1s)
    best_threshold = THRESHOLDS[best_idx]
    best_f1 = f1s[best_idx]

    plt.axvline(
        best_threshold,
        linestyle="--",
        linewidth=1,
        c="red"
    )

    plt.text(
        best_threshold,
        best_f1,
        f"  Highest F1 ({best_threshold:.2f},{best_f1:.2f})"
    )

    plt.xlabel("Similarity Threshold")
    plt.ylabel("Score")
    plt.title("Precision, Recall and F1-Score vs Similarity Threshold")

    plt.xticks(THRESHOLDS)
    plt.ylim(0, 1.05)

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/threshold_metrics.png", dpi=300, bbox_inches="tight")
    plt.show()


def main():
    dataset = load_dataset(DATASET_PATH)
    texts = dataset["texts"]
    ids = dataset["ids"]
    true_groups = dataset["true_groups"]
    top_k = 100000 # top_k cap shouldn't be involved in calculations to get all the real matches
     
    vectors = get_embeddings(texts, MODEL_NAME, MODEL_SERVER_URL)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
     
    index, id_map = build_index(vectors, ids)
    precisions, recalls, f1s = metrics(index, vectors, ids, id_map, texts, top_k, true_groups)

    plot_metrics_thresholds(precisions, recalls, f1s)


if __name__ == "__main__":
    main()