from libsemigroups_pybind11.transf import Transf
from libsemigroups_pybind11 import FroidurePin
from collections import defaultdict


# =============================================================================
# COMPUTE GENERATORS AND SEMIGROUP ELEMENTS AS
# TRANSFORMATIONS VIA FROIDURE-PIN ALGORITHM
# =============================================================================
def mul(f, g):
    return tuple(g[f[q]] for q in range(len(f)))


def build_letter_generators(min_dfa):
    """Returns alphabet, and dict letter -> letter transformation."""
    states = min_dfa.states
    state_index = {q: i for i, q in enumerate(states)}
    alphabet = sorted(min_dfa._input_symbols, key=str)

    # build transformations for each letter
    letter_transf = {}
    for a in alphabet:
        images = []
        for q in states:
            targets = min_dfa._transition_function(q, a)
            tgt = targets[0]
            images.append(state_index[tgt])

        letter_transf[a] = Transf(images)
    return alphabet, letter_transf


def build_fp_and_reps(min_dfa):
    alphabet, letter_transf = build_letter_generators(min_dfa)
    gens = [letter_transf[a] for a in alphabet]
    fp = FroidurePin(gens)
    fp.run()

    def factorize(j):
        factors_list = []
        while fp.length(j) > 1:
            factors_list.append(fp.final_letter(j))
            j = fp.prefix(j)
        # j is a generator
        factors_list.append(fp.final_letter(j))
        factors_list.reverse()
        return factors_list

    reps = {}
    for i, x in enumerate(fp):
        elem = tuple(list(x))
        factors = factorize(i)
        word = "".join(str(alphabet[g]) for g in factors)
        reps[elem] = word
    return reps, fp, alphabet


# =============================================================================
# FIND GREEN'S CLASSES VIA STRONGLY CONNECTED COMPONENTS OF
# CAYLEY GRAPHS AND ADJACENCY BUILDING
# =============================================================================
class _SimpleCayleyGraph:
    def __init__(self, n_nodes, adj_with_labels):
        # adj_with_labels: dict[node] -> list[(label, target)]
        self._n = n_nodes
        self._adj = adj_with_labels

    def nodes(self):
        return range(self._n)

    def labels_and_targets(self, node):
        return self._adj.get(node, [])


def find(x, parent):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def union(x, y, parent):
    parent[find(x, parent)] = find(y, parent)


def kosaraju_sccs(n, adj):
    """Returns list of SCCs as sets of node indices."""
    visited = set()
    order = []

    def dfs1(a):
        stack = [(a, iter(adj.get(a, [])))]
        while stack:
            node, children = stack[-1]
            try:
                child = next(children)
                if child not in visited:
                    visited.add(child)
                    stack.append((child, iter(adj.get(child, []))))
            except StopIteration:
                order.append(node)
                stack.pop()

    for v in range(n):
        if v not in visited:
            visited.add(v)
            dfs1(v)

    # reverse graph
    radj = defaultdict(set)
    for u, vs in adj.items():
        for v in vs:
            radj[v].add(u)

    visited2 = set()
    sccs = []

    def dfs2(b):
        second_dfs_scc = set()
        stack = [b]
        while stack:
            node = stack.pop()
            if node in visited2:
                continue
            visited2.add(node)
            second_dfs_scc.add(node)
            for nb in radj.get(node, []):
                if nb not in visited2:
                    stack.append(nb)
        return second_dfs_scc

    for v in reversed(order):
        if v not in visited2:
            scc = dfs2(v)
            sccs.append(scc)

    return sccs


def build_adj(wg):
    adj = defaultdict(set)
    for s in wg.nodes():
        for _label, target in wg.labels_and_targets(s):
            if target is not None:
                adj[s].add(target)
    return adj

def minimal_gens_of_stable(reps, stable):
    stable_set = set(stable)
    decomposable = set()
    for s in stable_set:
        for t in stable_set:
            st = mul(s, t)
            if st in stable_set and st != s and st != t:
                decomposable.add(st)
    gens = set(stable_set - decomposable)

    def closure(generators):
        current = set(generators)
        while True:
            new = {mul(a, b) for a in current for b in current} | current
            if new == current:
                return current
            current = new

    closed = closure(gens)
    for s in stable_set:
        if s in closed:
            continue
        gens.add(s)
        closed = closure(gens)
        if closed == stable_set:
            break

    return sorted(gens, key=lambda e: (len(reps[e]), reps[e]))


def build_stable_cayley_adj(min_dfa):
    """Compute stable semigroup and build its right and left Cayley graph adjacency."""
    reps, _alphabet = compute_syntactic_semigroup(min_dfa)
    stable = compute_stable_semigroup(reps)
    reps_stable = {e: w for e, w in reps.items() if e in stable}
    gens = minimal_gens_of_stable(reps_stable, stable)
    elements = list(reps_stable.keys())
    index_of = {e: i for i, e in enumerate(elements)}
    n = len(elements)

    right_adj = {i: [] for i in range(n)}
    left_adj = {i: [] for i in range(n)}
    for i, x in enumerate(elements):
        for g_idx, g in enumerate(gens):
            xg = mul(x, g)
            if xg in index_of:
                right_adj[i].append((g_idx, index_of[xg]))
            gx = mul(g, x)
            if gx in index_of:
                left_adj[i].append((g_idx, index_of[gx]))
    return reps_stable, elements, index_of, gens, right_adj, left_adj


def map_nodes_to_classes(n, lcg, rcg):
    # SCCs for R-classes (right Cayley) and L-classes (left Cayley)
    l_adj = build_adj(lcg)
    l_sccs = kosaraju_sccs(n, l_adj)

    r_adj = build_adj(rcg)
    r_sccs = kosaraju_sccs(n, r_adj)

    node_to_r = {}
    for r_id, scc in enumerate(r_sccs):
        for node in scc:
            node_to_r[node] = r_id

    node_to_l = {}
    for l_id, scc in enumerate(l_sccs):
        for node in scc:
            node_to_l[node] = l_id

    # D-classes: union-find over R-classes that share an L-class
    parent = list(range(len(r_sccs)))

    # group R-class ids by L-class
    l_to_r_classes = defaultdict(set)
    for node in range(n):
        l_to_r_classes[node_to_l[node]].add(node_to_r[node])

    for r_ids in l_to_r_classes.values():
        r_list = list(r_ids)
        for i in range(1, len(r_list)):
            union(r_list[0], r_list[i], parent)

    node_to_d = {node: find(node_to_r[node], parent) for node in range(n)}
    return node_to_r, node_to_l, node_to_d


# =============================================================================
# COMPUTE SYNTACTIC SEMIGROUP ELEMENTS AND GREEN'S CLASSES
# =============================================================================
def compute_syntactic_semigroup(min_dfa):
    reps, _fp, _alphabet = build_fp_and_reps(min_dfa)
    return reps, _alphabet


def compute_green_classes_semigroup(min_dfa):
    reps, fp, _alphabet = build_fp_and_reps(min_dfa)
    n = fp.size()

    rcg = fp.right_cayley_graph()
    lcg = fp.left_cayley_graph()

    node_to_r, node_to_l, node_to_d = map_nodes_to_classes(n, lcg, rcg)
    return _alphabet, fp, reps, node_to_r, node_to_l, node_to_d, lcg, rcg


# =============================================================================
# COMPUTE SYNTACTIC MONOID ELEMENTS AND GREEN'S CLASSES
# =============================================================================
def compute_syntactic_monoid(min_dfa):
    reps, _alphabet = compute_syntactic_semigroup(min_dfa)

    # add identity to reps if not already present
    n_states = len(next(iter(reps.keys())))
    identity = tuple(range(n_states))
    if identity not in reps:
        reps[identity] = ""
    return reps


def compute_green_classes_monoid(min_dfa):
    _alphabet, _fp, reps, node_to_r, node_to_l, node_to_d, _lcg, _rcg = compute_green_classes_semigroup(min_dfa)

    fp_elems = list(reps.keys())
    n_states = len(fp_elems[0])
    identity = tuple(range(n_states))

    injected_identity_d = None

    # only add identity if the semigroup doesn't already contain it
    if identity not in set(fp_elems):
        n = len(fp_elems)
        fp_elems.append(identity)
        # update reps so the egg-box can label the identity cell as ε
        reps[identity] = ""

        existing = set(node_to_r.values()) | set(node_to_l.values()) | set(node_to_d.values())
        new_id = max(existing) + 1 if existing else 0

        node_to_r[n] = new_id
        node_to_l[n] = new_id
        node_to_d[n] = new_id

        injected_identity_d = new_id
    return _alphabet, _fp, reps, node_to_r, node_to_l, node_to_d, _lcg, _rcg, injected_identity_d


# =============================================================================
# COMPUTE STABLE SEMIGROUP AND GREEN'S CLASSES
# =============================================================================
def compute_stable_semigroup(reps):
    generators = {t for t, w in reps.items() if len(w) == 1}
    current = set(generators)
    while True:
        squared = {mul(a, b) for a in current for b in current}
        if current == squared:
            return current
        current = {mul(a, b) for a in current for b in generators}


def compute_green_classes_stable_semigroup(min_dfa):
    reps_stable, elements, index_of, gens, right_adj, left_adj = build_stable_cayley_adj(min_dfa)
    n = len(elements)
    reps = reps_stable

    rcg = _SimpleCayleyGraph(n, right_adj)
    lcg = _SimpleCayleyGraph(n, left_adj)

    node_to_r, node_to_l, node_to_d = map_nodes_to_classes(n, lcg, rcg)
    return reps, node_to_r, node_to_l, node_to_d, lcg, rcg
