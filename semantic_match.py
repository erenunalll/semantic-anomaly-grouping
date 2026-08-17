import sys
import os
import json
import requests
import numpy as np

from usearch.index import Index
from sanitize import sanitize

MODEL_SERVER_URL = os.environ.get("MODEL_SERVER_URL", "http://localhost:11434/v1/embeddings")
MODEL_NAME = os.environ.get("MODEL_NAME", "bge-m3")


class ModelServerError(Exception):
    def __init__(self, message, exit_code):
        super().__init__(message)
        self.exit_code = exit_code


def build_input(data):
    model_server_url = data.get("model_server_url", MODEL_SERVER_URL)

    model_name = data.get("model_name", MODEL_NAME)

    items = data["items"]
    sim_threshold = data["threshold"]
    top_k = data["top_k"]

    id_to_count = {}

    for item in items:
        id_to_count[item["id"]] = item.get("total_count",0)

    candidate_k = min(len(items)-1, max(32, top_k*4))

    texts = [sanitize(item["text"]) for item in items]
    raw_texts = [item["text"] for item in items]

    ids = [item["id"] for item in items]

    id_to_rawtext = dict(zip(ids, raw_texts))

    return {
        "model_name": model_name,
        "model_server_url": model_server_url,
        "sim_threshold": sim_threshold,
        "candidate_k": candidate_k,
        "top_k": top_k,
        "texts": texts,
        "ids": ids,
        "id_to_count": id_to_count,
        "id_to_rawtext": id_to_rawtext
    }


def read_input():
    raw_text = sys.stdin.read().lstrip("\ufeff")
    data = json.loads(raw_text)

    return build_input(data)


def get_embeddings(texts, model_name, model_server_url):
    payload = {
        "model":model_name,
        "input":texts
    }

    try:
        response = requests.post(
            model_server_url,
            json=payload,
            timeout=120
        )
    except requests.RequestException as e:
        raise ModelServerError(f"Model server unreachable: {e}", exit_code=2)

    if response.status_code != 200:
        raise ModelServerError(f"Model server returned non-200 status {response.status_code} - {response.text}", exit_code=2)

    try:
        result = response.json()
        return np.array([item["embedding"] for item in result["data"]], dtype=np.float32)
    except (KeyError, ValueError) as e:
        raise ModelServerError(f"Malformed response from model server: {e}", exit_code=1)


def build_index(vectors, ids):
    index = Index(
        ndim = vectors.shape[1],
        metric = "cos"
    )

    # to preserve originial ids
    id_map = {}
    for idx, (original_id, embedding) in enumerate(zip(ids, vectors)):
        id_map[idx] = original_id
        index.add(idx, embedding)

    return index, id_map


def search_neighbors(index, vectors, ids, id_map, candidate_k, sim_threshold):
    neighbors = {item_id: [] for item_id in ids}
    for idx in range(len(ids)):
        matches = index.search(vectors[idx], candidate_k + 1)

        for match_id, distance in zip(matches.keys, matches.distances):
            if idx == match_id:
                continue

            similarity = 1 - distance

            if similarity >= sim_threshold:
                left_id = id_map[idx]
                right_id = id_map[match_id]
                neighbors[left_id].append((right_id, similarity))
                neighbors[right_id].append((left_id, similarity))

    # sorting
    for item_id in neighbors:
        neighbors[item_id].sort(key=lambda x: (-x[1], x[0]))

    return neighbors


def top_k_pairs(neighbors, ids, top_k):
    pairs = {}
    accepted_count = {item_id: 0 for item_id in ids}
    for item_id in sorted(ids):
        for neighbor_id, similarity in neighbors[item_id]:
            if accepted_count[item_id] >= top_k:
                break

            if accepted_count[neighbor_id] >= top_k:
                continue

            left_id = item_id
            right_id = neighbor_id

            if right_id < left_id:
                left_id, right_id = right_id, left_id

            pair = (left_id, right_id)

            if pair not in pairs or similarity > pairs[pair]:
                pairs[pair] = similarity
                accepted_count[item_id] += 1
                accepted_count[neighbor_id] += 1
    
    return pairs


def run_semantic_match(input):
    model_name = input["model_name"]
    model_server_url = input["model_server_url"]
    
    texts = input["texts"]
    ids = input["ids"]

    if len(ids) < 2:
        input.update({"id_map": {}, "vectors": None})
        return {}, input
    
    embs = get_embeddings(texts, model_name, model_server_url)
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)
    
    index, id_map = build_index(embs, ids)
    neighbors = search_neighbors(index, embs, ids, id_map, input["candidate_k"], input["sim_threshold"])
    pairs = top_k_pairs(neighbors, ids, input["top_k"])

    input.update(
        {
        "id_map":id_map,
        "vectors":embs
        }
    )
    return pairs, input


def build_output(pairs):
    candidate_pairs = []
    for (left_id, right_id), similarity in pairs.items():
        candidate_pairs.append(
            {
                "left_id":left_id,
                "right_id":right_id,
                "similarity":round(float(similarity), 4)
            }
        )

    return {
        "candidate_pairs": candidate_pairs
    }   


def similarity_main(input):
    pairs, _ = run_semantic_match(input)
    output = build_output(pairs)

    return output


if __name__ == "__main__":
    try:
        input = read_input()
        print(json.dumps(similarity_main(input)))
    except json.JSONDecodeError as e:
        print(f"Malformed JSON input: {e}", file=sys.stderr)
        sys.exit(1)
    except ModelServerError as e:
        print(str(e), file=sys.stderr)
        sys.exit(e.exit_code)
        