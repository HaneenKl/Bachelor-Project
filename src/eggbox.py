from graphviz import Digraph
from collections import deque
from collections import defaultdict
from semigroup import mul, compute_green_classes_semigroup, compute_green_classes_monoid, \
    compute_green_classes_stable_semigroup, compute_stable_semigroup


# =============================================================================
# COMPUTE J-ORDER COVERS VIA COMBINING LEFT AND RIGHT CAYLEY GRAPHS
# =============================================================================
# build adjacency of the union of right and left Cayley graphs
def build_two_sided_adj(lcg, rcg):
    adj = defaultdict(set)
    for s in rcg.nodes():
        for _label, target in rcg.labels_and_targets(s):
            if target is not None:
                adj[s].add(target)
        for _label, target in lcg.labels_and_targets(s):
            if target is not None:
                adj[s].add(target)
    return adj


def compute_j_order_covers_via_graph(lcg, rcg, node_to_d):
    adj = build_two_sided_adj(lcg, rcg)

    # build adjacency on D-classes
    d_adj = defaultdict(set)
    all_d_ids = set(node_to_d.values())
    for d in all_d_ids:
        d_adj.setdefault(d, set())  # ensure isolated D-classes appear

    for u, neighbors in adj.items():
        du = node_to_d[u]
        for v in neighbors:
            dv = node_to_d[v]
            if du != dv:
                d_adj[du].add(dv)
    covers = transitive_reduction(d_adj)

    return covers


def _topological_sort_from_covers(d_ids, covers):
    above_count = {d: 0 for d in d_ids}
    for (i, j) in covers:
        above_count[j] += 1

    children = defaultdict(list)
    for (i, j) in covers:
        children[i].append(j)

    queue = deque(d for d in d_ids if above_count[d] == 0)
    d_order = []
    while queue:
        d = queue.popleft()
        d_order.append(d)
        for j in children[d]:
            above_count[j] -= 1
            if above_count[j] == 0:
                queue.append(j)

    for d in d_ids:
        if d not in d_order:
            d_order.append(d)
    return d_order


def transitive_reduction(d_adj):
    reachable = {d: set() for d in d_adj}

    for start in d_adj:
        visited = set()
        stack = [start]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            for next_node in d_adj.get(node, ()):
                if next_node not in visited:
                    stack.append(next_node)
        reachable[start] = visited
    covers = []
    for u in d_adj:
        for v in d_adj[u]:
            if not any(v in reachable[w] for w in d_adj[u] if w != v):
                covers.append((u, v))
    return covers


# =============================================================================
# BUILD EGGBOX SVG
# =============================================================================
def build_eggbox_svg(reps, node_to_r, node_to_l, node_to_d, stable, stable_only, lcg, rcg, top_d_id=None):
    fp_elems = list(reps.keys())
    n = len(fp_elems)

    idempotents = {e for e in fp_elems if mul(e, e) == e}

    n_states = len(fp_elems[0])
    identity = tuple(range(n_states))

    def word(t):
        return reps.get(t, "?")

    # group nodes by D-class
    d_groups = defaultdict(list)
    for i in range(n):
        d_groups[node_to_d[i]].append(i)

    # compute J-order covers for layout
    covers = compute_j_order_covers_via_graph(lcg, rcg, node_to_d)

    if top_d_id is not None:
        covered = {lo for (_, lo) in covers}
        current_roots = [d for d in d_groups if d != top_d_id and d not in covered]
        for root in current_roots:
            covers.append((top_d_id, root))

    d_order = _topological_sort_from_covers(list(d_groups.keys()), covers)

    d_id_to_box_idx = {d_id: idx for idx, d_id in enumerate(d_order)}

    eggboxes = []
    for d_id in d_order:
        nodes = d_groups[d_id]
        # collect R and L class ids within this D-class
        r_ids = sorted({node_to_r[i] for i in nodes})
        l_ids = sorted({node_to_l[i] for i in nodes})

        r_index = {r: j for j, r in enumerate(r_ids)}
        l_index = {l: j for j, l in enumerate(l_ids)}

        # H-class cells: (l_row, r_col) -> list of words
        cells = defaultdict(list)
        for i in nodes:
            elem = fp_elems[i]
            is_stable = stable is not None and elem in stable

            # For stable_only mode, skip non-stable elements entirely
            if stable_only and not is_stable:
                continue

            r = r_index[node_to_r[i]]
            l = l_index[node_to_l[i]]
            w = word(elem)
            is_idempotent = idempotents is not None and elem in idempotents
            if w == "":
                display_word = "ε"
                ###change later maybe: to show "aa=ε" for example
            elif elem == identity:
                display_word = "ε"
            else:
                display_word = w

            cells[(r, l)].append((display_word, is_stable, is_idempotent))

        # sort words within each H-class
        for key in cells:
            cells[key].sort(key=lambda triple: (triple[0] != "ε", len(triple[0]), triple[0]))

        # If stable_only is True and this D-class has no stable elements, skip it
        if stable_only and not cells:
            continue

        eggboxes.append({
            "n_rows": len(r_ids),
            "n_cols": len(l_ids),
            "cells": cells,
        })
    cover_edges = [(d_id_to_box_idx[hi], d_id_to_box_idx[lo]) for (hi, lo) in covers]

    return eggboxes, cover_edges


def plot_eggbox_svg(eggboxes, cover_edges=None):
    dot = Digraph(format="svg")
    dot.attr(rankdir="TB")
    dot.attr("node", shape="plaintext")

    dot.attr(newrank="true")

    for i, box in enumerate(eggboxes):
        n_rows = box["n_rows"]
        n_cols = box["n_cols"]
        cells = box["cells"]

        table = ['<table border="1" cellborder="1" cellspacing="0">', ]

        for r in range(n_rows):
            table.append("<tr>")
            for c in range(n_cols):
                entries = cells.get((r, c), [])
                if not entries:
                    table.append("<td width='100'> </td>")
                else:
                    def fmt_entry(w, s, idem):
                        label = w + ("*" if idem else "")
                        return f"<font color='green'>{label}</font>" if s else label

                    display = ", ".join(fmt_entry(w, s, idem) for w, s, idem in entries)
                    table.append(f"<td width='100' align='center'><b>{display}</b></td>")
            table.append("</tr>")

        table.append("</table>")
        dot.node(f"D{i}", label="<" + "".join(table) + ">")
    # connect D-classes only along J-order covering relations.
    # incomparable D-classes get no edge between them and so are placed
    # side by side rather than stacked vertically.
    if cover_edges:
        for hi, lo in cover_edges:
            dot.edge(f"D{hi}", f"D{lo}", style="invis")
    n_boxes = len(eggboxes)
    parents = defaultdict(list)
    if cover_edges:
        for hi, lo in cover_edges:
            parents[lo].append(hi)

    rank = {}

    def depth_up(u):
        if u in rank:
            return rank[u]
        ps = parents.get(u, [])
        if not ps:
            rank[u] = 0
        else:
            rank[u] = 1 + max(depth_up(p) for p in ps)
        return rank[u]

    for i in range(n_boxes):
        depth_up(i)

    return dot.pipe(format="svg").decode("utf-8")


# =============================================================================
# PLOT SEMIGROUP EGGBOX
# =============================================================================
def visualize_syntactic_semigroup(min_dfa):
    alphabet, fp, reps, node_to_r, node_to_l, node_to_d, lcg, rcg = compute_green_classes_semigroup(min_dfa)
    stable = compute_stable_semigroup(reps)
    eggboxes, cover_edges = build_eggbox_svg(reps, node_to_r, node_to_l, node_to_d, stable, stable_only=False, lcg=lcg,
                                             rcg=rcg)
    return plot_eggbox_svg(eggboxes, cover_edges)


# =============================================================================
# PLOT MONOID EGGBOX
# =============================================================================
def visualize_syntactic_monoid(min_dfa):
    alphabet, fp, reps, node_to_r, node_to_l, node_to_d, lcg, rcg, injected_identity_d = compute_green_classes_monoid(
        min_dfa)
    stable = compute_stable_semigroup(reps)
    eggboxes, cover_edges = build_eggbox_svg(reps, node_to_r, node_to_l, node_to_d, stable, stable_only=False, lcg=lcg,
                                             rcg=rcg, top_d_id=injected_identity_d)
    return plot_eggbox_svg(eggboxes, cover_edges)


# =============================================================================
# PLOT STABLE SEMIGROUP EGGBOX
# =============================================================================
def visualize_syntactic_stable_semigroup(min_dfa):
    reps, node_to_r, node_to_l, node_to_d, lcg, rcg = compute_green_classes_stable_semigroup(min_dfa)

    eggboxes, cover_edges = build_eggbox_svg(reps, node_to_r, node_to_l, node_to_d, stable=None, stable_only=False,
                                             lcg=lcg, rcg=rcg)
    return plot_eggbox_svg(eggboxes, cover_edges)
