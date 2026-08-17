# Similarity Graph Grouping & Metadata Selection Analysis
The semantic matching stage produces candidate pairs whose cosine similarity meets or exceeds the similarity threshold. These pairs are treated as edges in a similarity graph, where each log message represents a node. A `Union-Find` structure is then used to identify connected components.

## 1. Graph Grouping Algorithm
**Union-Find Algorithm** 

 The algorithm maintains a `parent` dictionary that stores the parent of each node. Initially, every node is its own parent, meaning none of the nodes are connected at the start. The Union-Find has two methods: `find(x)`, which returns the root node of `x` by traversing the parent nodes and `union(x, y)` that merges two sets by connecting the root of one set as the parent of another.
```python
class UnionFind():
    def __init__(self, ids):
        self.parent = {id:id for id in ids}

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, x, y):
        self.parent[self.find(x)] = self.find(y)
```

**Building Groups**

1. Applying `top_k` cap returns a dictionary that contains pairs. To build the groups, firstly every pair that exceeds the similarity threshold is connected using the `union(left_id, right_id)` function. This updates the `parent` dictionary so that all connected nodes eventually share the same root.
```python
def build_groups(pairs, ids):
    uf = UnionFind(ids)

        for left_id, right_id in pairs.keys():
            uf.union(left_id, right_id)
```

2. After all unions are completed, every log ID is assigned to its root using `find()`. A dictionary maps each root to the list of nodes (sub-graphs) belonging to that connected component.
```python
    roots_groups = {}
        for id in ids:
            root = uf.find(id)
            if root not in roots_groups:
                roots_groups[root] = []   # Initialize group
            roots_groups[root].append(id)
``` 

3. Lastly, the groups that have less than two members are filtered and other groups are returned.
```python
    groups = [members for members in roots_groups.values() if len(members) >= 2]

    return groups
```

## 2. Chain-Link Mitigation Analysis
After creating sub-graphs, some members may only be indirectly connected to the selected canonical item. For example, `log A` may be similar to `log B`, and `log B` may be similar to `log C`, while `log A` and `log C` are not similar enough. As a result, connected components may group semantically different logs into the same sub-graph. This problem is known as the chain-link effect.
```
A-----0.91-----B-----0.90-----C

        A-----0.42-----C
```
Although A, B and C belong to the same sub-graph, C is removed because its similarity to the canonical item does not satisfy the similarity threshold.

To mitigate this issue, each group is processed independently. A canonical log is first selected according to the metadata selection policy (see Section 3). Every other member is compared directly with the canonical log. If the pair exists in the `pairs` dictionary, the similarity is retrieved and compared with the threshold. If the pair is absent in the `pairs` dictionary (this may happen due to top_k cap), the cosine similarity is recomputed using the stored embedding vectors.

**Implementation Chain-Link Mitigation**
```python
def chain_link_mitigation(group, pairs, canonical_id, sim_threshold, vectors, reverse_id_map):
    accepted = [canonical_id]
    rejected=[]

    for member in group:
        if member == canonical_id:
            continue

        left_id = canonical_id
        right_id = member

        if member < canonical_id:
            left_id, right_id = right_id, left_id

        if (left_id, right_id) in pairs:
            similarity = pairs[(left_id, right_id)]
        else:
            similarity = cosine_sim(vectors, reverse_id_map, left_id, right_id)

        if similarity >= sim_threshold:
            accepted.append(member)
        else:
            rejected.append(member)

    return accepted, rejected
```
Members whose similarity is below the desired threshold are removed from the current group and placed into a different list called rejected. The rejected members are recursively processed using the same procedure until there are no splits required. This process allows rejected members to form new groups if their similarity is high enough, preventing valid clusters from being discarded because they do not match the current canonical node.

**Implementation of Recursive Grouping**
```python
def recursive_grouping(group, pairs, vectors, sim_threshold, reverse_id_map, id_to_count, id_to_text):
    if len(group) < 2:
        return []

    canonical_id = select_canonical(group, id_to_count, id_to_text)
    accepted, rejected = chain_link_mitigation(group, pairs, canonical_id, sim_threshold, vectors, reverse_id_map)

    result = []
    if len(accepted) >= 2:
        result.append(accepted)

    result.extend(recursive_grouping(rejected, pairs, vectors, sim_threshold, reverse_id_map, id_to_count, id_to_text))
    return result
```

## 3. Canonical Title Rule Definition
The selection policy first considers the frequency (`total_count`). The log with the highest `total_count` is selected as the canonical item. If multiple logs have the same frequency, the shorter log message is preferred. If a tie still remains, the lexicographically smaller log ID is selected to guarantee that the same input always produces the same canonical title.

**Implementation of Canonical Title/Id Selection**
```python
def select_canonical(group, id_to_count, id_to_text):
    best_id = group[0]

    for member in group:

        if id_to_count[member] > id_to_count[best_id]:
            best_id = member
        elif id_to_count[member] == id_to_count[best_id]:
            if len(id_to_text[member]) < len(id_to_text[best_id]):
                best_id = member
            elif len(id_to_text[member]) == len(id_to_text[best_id]):
                if member < best_id:
                    best_id = member

    return best_id

# canonical title is retrieved using canonical_id while building the output
"canonical_title":id_to_rawtext[canonical_id]
```

---