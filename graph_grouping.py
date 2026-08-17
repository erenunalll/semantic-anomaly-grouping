import json

from semantic_match import read_input, run_semantic_match, build_output

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


def build_groups(pairs, ids):
    uf = UnionFind(ids)

    for left_id, right_id in pairs.keys():
        uf.union(left_id, right_id)

    roots_groups = {}
    for id in ids:
        root = uf.find(id)
        if root not in roots_groups:
            roots_groups[root] = []
        roots_groups[root].append(id)

    groups = [members for members in roots_groups.values() if len(members) >= 2]

    return groups


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


def cosine_sim(vectors, reverse_id_map, id_1, id_2):
    idx_1 = reverse_id_map[id_1]
    idx_2 = reverse_id_map[id_2]

    return float(vectors[idx_1] @ vectors[idx_2])
  

def select_canonical(group, id_to_count, id_to_text):
    best_id = group[0]

    for member in group:

        if id_to_count[member] > id_to_count[best_id]:   # large count
            best_id = member
        elif id_to_count[member] == id_to_count[best_id]:
            if len(id_to_text[member]) < len(id_to_text[best_id]):   # small length
                best_id = member
            elif len(id_to_text[member]) == len(id_to_text[best_id]):
                if member < best_id:   # lexicographically small
                    best_id = member

    return best_id


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


def cluster_output(final_groups, id_to_rawtext, id_to_count):
    out_groups = []
    for group in final_groups:
        canonical_id = select_canonical(group, id_to_count, id_to_rawtext)
        sorted_members = sorted(group)
        total_count = sum(id_to_count[m] for m in group)

        out_group = {
            "group_id":sorted_members[0],
            "canonical_id":canonical_id,
            "canonical_title":id_to_rawtext[canonical_id],
            "member_ids":sorted_members,
            "total_count":total_count
        }
        out_groups.append(out_group)

    return {
        "groups":out_groups
        }


def cluster_main(input_in):
    pairs, input = run_semantic_match(input_in)
    groups = build_groups(pairs, input["ids"])

    reverse_id_map = {v: k for k, v in input["id_map"].items()}
    id_to_count = input["id_to_count"]
    id_to_rawtext = input["id_to_rawtext"]

    vectors = input["vectors"]
    
    final_groups = []
    for group in groups:
        sub_groups = recursive_grouping(group, pairs, vectors, input["sim_threshold"], reverse_id_map, id_to_count, id_to_rawtext)
        final_groups.extend(sub_groups)

    return build_output(pairs) | cluster_output(final_groups, id_to_rawtext, id_to_count)


if __name__ == "__main__":
    input = read_input()
    print(json.dumps(cluster_main(input)))