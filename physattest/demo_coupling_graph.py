"""
Demo: Coupling Graph construction, Theorem 1 analysis, and visualization.

Run: python demo_coupling_graph.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from graph.coupling_graph import build_swat_coupling_graph, CouplingType, CouplingDomain
from graph.algebraic_connectivity import (
    algebraic_connectivity, k_resilience, vulnerability_analysis,
    compute_k_for_subgraphs, fiedler_vector, suggest_edges_to_add,
)


def run_demo():
    print("=" * 70)
    print("PhysAttest Coupling Graph — Demo")
    print("=" * 70)

    # --- Build the graph ---
    G = build_swat_coupling_graph()
    print(f"\nGraph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # --- Basic stats ---
    print(f"\nConnected: {nx.is_connected(G)}")
    print(f"Diameter: {nx.diameter(G) if nx.is_connected(G) else 'N/A (disconnected)'}")
    print(f"Average degree: {2 * G.number_of_edges() / G.number_of_nodes():.1f}")
    print(f"Density: {nx.density(G):.3f}")

    # --- Edge type breakdown ---
    print("\nEdge breakdown by coupling type:")
    type_counts = {}
    for _, _, data in G.edges(data=True):
        ct = data.get("coupling_type", "unknown")
        type_counts[ct] = type_counts.get(ct, 0) + 1
    for ct, count in sorted(type_counts.items()):
        print(f"  {ct:15s}: {count} edges")

    print("\nEdge breakdown by verification domain:")
    domain_counts = {}
    for _, _, data in G.edges(data=True):
        d = data.get("domain", "unknown")
        domain_counts[d] = domain_counts.get(d, 0) + 1
    for d, count in sorted(domain_counts.items()):
        print(f"  {d:12s}: {count} edges")

    # --- Algebraic connectivity (Theorem 1) ---
    lambda2 = algebraic_connectivity(G)
    print(f"\n--- Theorem 1: k-Compromise Resilience ---")
    print(f"Algebraic connectivity λ₂ = {lambda2:.4f}")

    sigma_values = [0.01, 0.02, 0.05, 0.1, 0.2]
    print(f"\nk-resilience for different noise levels:")
    print(f"  {'σ_noise':>10s}  {'k':>5s}  Interpretation")
    print(f"  {'-'*10}  {'-'*5}  {'-'*40}")
    for sigma in sigma_values:
        k = k_resilience(G, sigma)
        print(f"  {sigma:10.3f}  {k:5d}  Attacker must compromise ≥{k} sensors")

    # --- Vulnerability analysis ---
    print(f"\n--- Vulnerability Analysis ---")
    analysis = vulnerability_analysis(G)

    print(f"\nMost vulnerable sensors (lowest protection score):")
    for name in analysis["most_vulnerable"]:
        deg = analysis["degrees"][name]
        bet = analysis["betweenness"][name]
        domains = analysis["domain_counts"][name]
        active_domains = sum(1 for c in domains.values() if c > 0)
        print(f"  {name:10s}  degree={deg:.1f}  betweenness={bet:.3f}  "
              f"domains={active_domains}/3  {domains}")

    print(f"\nMost protected sensors (highest protection score):")
    for name in analysis["most_protected"]:
        deg = analysis["degrees"][name]
        domains = analysis["domain_counts"][name]
        active_domains = sum(1 for c in domains.values() if c > 0)
        print(f"  {name:10s}  degree={deg:.1f}  domains={active_domains}/3  {domains}")

    # --- Per-stage analysis ---
    print(f"\n--- Per-Stage k-Resilience (σ=0.05) ---")
    stage_k = compute_k_for_subgraphs(G, 0.05)
    for stage, k in stage_k.items():
        status = "GOOD" if k >= 3 else "WEAK" if k >= 1 else "ISOLATED"
        print(f"  {stage}: k={k}  [{status}]")

    # --- Suggested edges for GNN discovery ---
    print(f"\n--- Suggested Edges for GNN Discovery (Member 2) ---")
    suggestions = suggest_edges_to_add(G)
    print("Top edges that would most increase λ₂:")
    for src, tgt, score in suggestions:
        src_stage = G.nodes[src].get("stage", "?")
        tgt_stage = G.nodes[tgt].get("stage", "?")
        print(f"  {src} (S{src_stage}) ↔ {tgt} (S{tgt_stage})  "
              f"Fiedler gap={score:.3f}")

    # --- Visualize ---
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Panel 1: Full coupling graph colored by stage
    ax1 = axes[0]
    stage_colors = {1: "#e74c3c", 2: "#3498db", 3: "#2ecc71",
                    4: "#f39c12", 5: "#9b59b6", 6: "#1abc9c"}
    node_colors = [stage_colors.get(G.nodes[n].get("stage", 0), "#95a5a6")
                   for n in G.nodes()]

    edge_colors_map = {
        "hydraulic": "#2c3e50",
        "inter_stage": "#e74c3c",
        "physics": "#3498db",
        "chemistry": "#2ecc71",
        "discovered": "#f39c12",
    }
    edge_colors = [edge_colors_map.get(G.edges[e].get("coupling_type", ""), "#bdc3c7")
                   for e in G.edges()]
    edge_widths = [G.edges[e].get("weight", 0.5) * 2 for e in G.edges()]

    pos = nx.spring_layout(G, seed=42, k=2.0, iterations=100)
    nx.draw_networkx_nodes(G, pos, ax=ax1, node_color=node_colors,
                           node_size=300, edgecolors="black", linewidths=0.5)
    nx.draw_networkx_labels(G, pos, ax=ax1, font_size=6, font_weight="bold")
    nx.draw_networkx_edges(G, pos, ax=ax1, edge_color=edge_colors,
                           width=edge_widths, alpha=0.6)
    ax1.set_title(f"SWaT Coupling Graph\n{G.number_of_nodes()} nodes, "
                  f"{G.number_of_edges()} edges, λ₂={lambda2:.3f}", fontsize=12)

    # Legend for stages
    for stage, color in stage_colors.items():
        ax1.scatter([], [], c=color, s=100, label=f"Stage {stage}")
    ax1.legend(loc="lower left", fontsize=8)
    ax1.axis("off")

    # Panel 2: Fiedler vector visualization
    ax2 = axes[1]
    nodes = list(G.nodes())
    fv = fiedler_vector(G)
    sorted_indices = np.argsort(fv)
    sorted_nodes = [nodes[i] for i in sorted_indices]
    sorted_fv = fv[sorted_indices]
    sorted_colors = [stage_colors.get(G.nodes[n].get("stage", 0), "#95a5a6")
                     for n in sorted_nodes]

    bars = ax2.barh(range(len(sorted_nodes)), sorted_fv, color=sorted_colors,
                    edgecolor="black", linewidth=0.3)
    ax2.set_yticks(range(len(sorted_nodes)))
    ax2.set_yticklabels(sorted_nodes, fontsize=6)
    ax2.axvline(x=0, color="red", linestyle="--", linewidth=1.5, label="Sign boundary")
    ax2.set_xlabel("Fiedler Vector Value", fontsize=10)
    ax2.set_title("Fiedler Vector (vulnerability boundary at 0)\n"
                  "Sensors near the boundary are most vulnerable", fontsize=11)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("coupling_graph_results.png", dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to coupling_graph_results.png")

    # --- Summary for paper ---
    print(f"\n{'='*70}")
    print("Summary for paper (Section 4):")
    print(f"  Nodes:        {G.number_of_nodes()}")
    print(f"  Edges:        {G.number_of_edges()}")
    print(f"  λ₂:           {lambda2:.4f}")
    print(f"  k (σ=0.05):   {k_resilience(G, 0.05)}")
    print(f"  Physics edges: {domain_counts.get('physics', 0)}")
    print(f"  Chemistry:    {domain_counts.get('chemistry', 0)}")
    print(f"  Math:         {domain_counts.get('math', 0)}")
    print(f"  Multi-domain: {sum(1 for n in G.nodes() if sum(1 for c in analysis['domain_counts'][n].values() if c > 0) >= 2)} "
          f"of {G.number_of_nodes()} sensors verified by ≥2 domains")
    print(f"{'='*70}")


if __name__ == "__main__":
    run_demo()
