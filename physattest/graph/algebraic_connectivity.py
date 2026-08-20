"""
Algebraic connectivity analysis and Theorem 1 (k-compromise resilience).

The algebraic connectivity λ₂ is the second-smallest eigenvalue of the
graph Laplacian. It measures how well-connected the coupling graph is.

THEOREM 1: For coupling graph G with algebraic connectivity λ₂ and
sensor noise level σ, an attacker spoofing fewer than
    k = ⌈λ₂ / σ_noise⌉
sensors produces residual ‖r‖ > τ with probability ≥ 1 - δ.

Plain English: The attacker must compromise at least k sensors
simultaneously to avoid detection. One lie is easy to catch because
neighbours disagree. Many coordinated lies are needed to fool the
physics-based observer, and k grows with graph connectivity.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional


def compute_graph_laplacian(G: nx.Graph) -> np.ndarray:
    """
    Compute the weighted graph Laplacian L = D - W.

    D = diagonal matrix of weighted degrees
    W = weighted adjacency matrix

    The Laplacian encodes the coupling structure. Its eigenvalues
    tell us how connected the graph is.
    """
    return nx.laplacian_matrix(G, weight="weight").toarray().astype(float)


def algebraic_connectivity(G: nx.Graph) -> float:
    """
    Compute λ₂ — the algebraic connectivity (Fiedler value).

    λ₂ = 0  → graph is disconnected (attacker can isolate sensors)
    λ₂ > 0  → graph is connected
    λ₂ large → graph is well-connected (harder to attack)

    For security: higher λ₂ = better. Each edge you add (including
    GNN-discovered ones) can only increase λ₂.
    """
    return nx.algebraic_connectivity(G, weight="weight")


def fiedler_vector(G: nx.Graph) -> np.ndarray:
    """
    The Fiedler vector — eigenvector corresponding to λ₂.

    Nodes with similar Fiedler values are in the same "cluster".
    The sign boundary identifies the weakest cut in the graph.
    This tells you where the graph is most vulnerable to partitioning.

    For security: sensors near the sign boundary are the attacker's
    best targets — compromising them can isolate a subgraph.

    If the graph is disconnected, uses the largest connected component
    and assigns 0 to isolated nodes.
    """
    if nx.is_connected(G):
        return nx.fiedler_vector(G, weight="weight")

    nodes = list(G.nodes())
    fv = np.zeros(len(nodes))
    largest_cc = max(nx.connected_components(G), key=len)
    subgraph = G.subgraph(largest_cc)
    if len(largest_cc) >= 2:
        sub_fv = nx.fiedler_vector(subgraph, weight="weight")
        sub_nodes = list(subgraph.nodes())
        for i, node in enumerate(sub_nodes):
            idx = nodes.index(node)
            fv[idx] = sub_fv[i]
    return fv


def k_resilience(G: nx.Graph, sigma_noise: float) -> int:
    """
    THEOREM 1: k-compromise resilience.

    k = ⌈λ₂ / σ_noise⌉

    An attacker spoofing fewer than k sensors produces a detectable
    residual with high probability.

    Parameters:
        G: coupling graph
        sigma_noise: measurement noise standard deviation (in normalized units)

    Returns:
        k: minimum number of sensors attacker must compromise simultaneously
    """
    lambda2 = algebraic_connectivity(G)
    k = int(np.ceil(lambda2 / sigma_noise))
    return max(k, 1)


def vulnerability_analysis(G: nx.Graph) -> Dict:
    """
    Identify the most and least protected sensors in the graph.

    Combines:
    - Node degree (more connections = harder to spoof)
    - Betweenness centrality (high = critical for information flow)
    - Fiedler vector position (near sign boundary = vulnerable)
    - Per-domain coupling count

    Returns a dictionary with rankings and recommendations.
    """
    nodes = list(G.nodes())

    # Weighted degree: sum of edge weights
    degrees = dict(G.degree(weight="weight"))

    # Betweenness centrality
    betweenness = nx.betweenness_centrality(G, weight="weight")

    # Fiedler vector
    try:
        fv = fiedler_vector(G)
        fiedler_map = {node: fv[i] for i, node in enumerate(nodes)}
    except nx.NetworkXError:
        fiedler_map = {node: 0.0 for node in nodes}

    # Per-domain coupling count
    domain_counts = {}
    for node in nodes:
        counts = {"physics": 0, "chemistry": 0, "math": 0}
        for _, _, data in G.edges(node, data=True):
            domain = data.get("domain", "physics")
            counts[domain] += 1
        domain_counts[node] = counts

    # Composite vulnerability score (lower = more vulnerable)
    # Score = weighted_degree × (1 + multi_domain_bonus) / (1 + fiedler_boundary_proximity)
    vulnerability = {}
    for node in nodes:
        n_domains = sum(1 for d, c in domain_counts[node].items() if c > 0)
        multi_domain_bonus = (n_domains - 1) * 0.3
        fiedler_boundary = abs(fiedler_map[node])
        score = degrees[node] * (1 + multi_domain_bonus) / (1 + 1.0 / (fiedler_boundary + 0.01))
        vulnerability[node] = score

    # Sort: lowest score = most vulnerable
    ranked = sorted(vulnerability.items(), key=lambda x: x[1])

    return {
        "lambda2": algebraic_connectivity(G),
        "degrees": degrees,
        "betweenness": betweenness,
        "fiedler": fiedler_map,
        "domain_counts": domain_counts,
        "vulnerability_scores": vulnerability,
        "most_vulnerable": [name for name, _ in ranked[:5]],
        "most_protected": [name for name, _ in ranked[-5:]],
    }


def compute_k_for_subgraphs(G: nx.Graph, sigma_noise: float) -> Dict[str, int]:
    """
    Compute k-resilience for each stage subgraph.

    Some stages may be internally well-connected but weakly coupled
    to other stages. This identifies where inter-stage edges are
    most needed.
    """
    stage_k = {}

    # Group nodes by stage
    stages = {}
    for node, data in G.nodes(data=True):
        stage = data.get("stage", 0)
        if stage not in stages:
            stages[stage] = []
        stages[stage].append(node)

    for stage, nodes in sorted(stages.items()):
        subgraph = G.subgraph(nodes)
        if len(nodes) < 2:
            stage_k[f"Stage_{stage}"] = 1
            continue
        if not nx.is_connected(subgraph):
            stage_k[f"Stage_{stage}"] = 0
            continue
        stage_k[f"Stage_{stage}"] = k_resilience(subgraph, sigma_noise)

    return stage_k


def suggest_edges_to_add(G: nx.Graph, top_n: int = 5) -> List[Tuple[str, str, float]]:
    """
    Suggest edges that would most increase algebraic connectivity.

    Uses the Fiedler vector heuristic: adding edges between nodes on
    opposite sides of the Fiedler sign boundary increases λ₂ most.

    These suggestions go to Member 2 as targets for GNN discovery —
    if the GNN finds evidence for these couplings in the data, adding
    them tightens the security guarantee.
    """
    nodes = list(G.nodes())
    fv = fiedler_vector(G)

    # Find nodes on opposite sides of the sign boundary
    positive = [(nodes[i], fv[i]) for i in range(len(nodes)) if fv[i] > 0]
    negative = [(nodes[i], fv[i]) for i in range(len(nodes)) if fv[i] <= 0]

    # Score potential edges by how much they'd bridge the cut
    candidates = []
    for n_pos, v_pos in positive:
        for n_neg, v_neg in negative:
            if not G.has_edge(n_pos, n_neg):
                score = abs(v_pos - v_neg)
                candidates.append((n_pos, n_neg, score))

    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates[:top_n]


def add_discovered_edges(G: nx.Graph, discovered: List[Tuple[str, str, float]]):
    """
    Interface for Member 2's GNN: add discovered coupling edges to the graph.

    Each discovered edge must pass a physical plausibility check:
    the coupling must be explainable by some physical mechanism
    (vibration transfer, thermal radiation, electromagnetic interference, etc.)

    Parameters:
        G: the coupling graph (modified in place)
        discovered: list of (source, target, confidence) from GNN
    """
    for source, target, confidence in discovered:
        if source not in G.nodes() or target not in G.nodes():
            continue
        if confidence < 0.3:
            continue
        G.add_edge(
            source, target,
            weight=confidence * 0.5,
            coupling_type=CouplingType.DISCOVERED.value,
            domain=CouplingDomain.PHYSICS.value,
            equation=f"GNN-discovered coupling (confidence={confidence:.2f})",
        )


# Need to import CouplingType for add_discovered_edges
from .coupling_graph import CouplingType, CouplingDomain
