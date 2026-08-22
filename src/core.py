#!/usr/bin/env python3
"""
Core Zigscore persistence computation for paper evaluation.

Implements:
1. Zigscore Multi-Modal Dependency Graph construction
2. Coherence scoring (Union, Intersection, Negotiation)
3. Novelty scoring (delta from landmark features)
4. Zigscore persistence barcode computation
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class ModalityNode:
    """A modality node in the Zigscore graph."""
    name: str  # e.g., "Text", "Formulation", "Code", "Figure", "Table"
    index: int  # M1=0, M2=1, ..., M5=4
    layer: str  # R (Research), A (Algorithm), E (Experiment)
    embedding: Optional[np.ndarray] = None  # h_i in R^d


@dataclass
class ZigscoreEdge:
    """A directed edge in the Zigscore graph."""
    source: int  # index of source modality
    target: int  # index of target modality
    weight: float = 1.0  # w_{i->j}
    disagreement: float = 0.0  # delta_{i->j}


@dataclass
class ZigscoreGraph:
    """The full Zigscore multi-modal dependency graph for a paper."""
    paper_id: str
    nodes: List[ModalityNode] = field(default_factory=list)
    edges: List[ZigscoreEdge] = field(default_factory=list)
    coherence: Optional[float] = None
    novelty: Optional[float] = None
    barcode: Optional[List[Tuple[float, float]]] = None  # persistence barcode


# Default modality definitions (Section 2.2 of the paper)
MODALITY_DEFS = [
    ModalityNode("Text", 0, "R"),        # M1
    ModalityNode("Formulation", 1, "R"),  # M2
    ModalityNode("Code", 2, "A"),         # M3
    ModalityNode("Figure", 3, "E"),       # M4
    ModalityNode("Table", 4, "E"),        # M5
]

# Primary Zigscore backbone edges (Eq. 13-17)
BACKBONE_EDGES = [
    (0, 1),  # E1: Text -> Formulation
    (1, 2),  # E2: Formulation -> Code
    (2, 3),  # E3: Code -> Figure
    (3, 4),  # E4: Figure -> Table
    (4, 1),  # E5: Table -> Formulation (feedback)
]

# Zigscore cycle C* (Eq. 18)
ZIGSCORE_CYCLE = [1, 2, 3, 4, 1]  # M2 -> M3 -> M4 -> M5 -> M2


def sigmoid(x: float) -> float:
    """Sigmoid activation."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


class CrossModalAligner:
    """
    Cross-modal alignment map T_{i->j} (Eq. 19).
    Uses learned or fixed linear projection to align embedding spaces.
    """
    
    def __init__(self, dim: int, mode: str = "linear"):
        self.dim = dim
        self.mode = mode
        self.projections: Dict[Tuple[int, int], np.ndarray] = {}
    
    def init_projection(self, src: int, tgt: int, seed: int = 42):
        """Initialize a cross-modal projection matrix."""
        rng = np.random.RandomState(seed + src * 10 + tgt)
        if self.mode == "linear":
            # Orthogonal initialization
            W = rng.randn(self.dim, self.dim)
            U, _, Vt = np.linalg.svd(W, full_matrices=False)
            self.projections[(src, tgt)] = U @ Vt
        elif self.mode == "identity":
            self.projections[(src, tgt)] = np.eye(self.dim)
    
    def transform(self, src: int, tgt: int, h_src: np.ndarray) -> np.ndarray:
        """Apply cross-modal alignment: T_{i->j}(h_i)."""
        key = (src, tgt)
        if key not in self.projections:
            self.init_projection(src, tgt)
        return self.projections[key] @ h_src
    
    def set_projection(self, src: int, tgt: int, W: np.ndarray):
        """Set a learned projection matrix."""
        self.projections[(src, tgt)] = W


def compute_edge_weight(aligner: CrossModalAligner, h_src: np.ndarray, 
                        h_tgt: np.ndarray, src_idx: int, tgt_idx: int) -> Tuple[float, float]:
    """
    Compute edge weight w_{i->j} and disagreement delta_{i->j} (Eq. 19, 22).
    
    Returns: (weight, disagreement)
    """
    projected = aligner.transform(src_idx, tgt_idx, h_src)
    delta = np.linalg.norm(projected - h_tgt)
    weight = np.exp(-delta)
    return float(weight), float(delta)


def compute_intersection(edges: List[ZigscoreEdge], cycle_edges: List[Tuple[int, int]]) -> float:
    """
    Compute Intersection I (Eq. 20).
    Multiplicative consistency on the main Zigscore cycle.
    """
    edge_map = {(e.source, e.target): e for e in edges}
    product = 1.0
    for src, tgt in cycle_edges:
        if (src, tgt) in edge_map:
            product *= edge_map[(src, tgt)].weight
        else:
            product *= 0.0  # Missing edge = no consistency
    return product


def compute_union(edges: List[ZigscoreEdge]) -> float:
    """
    Compute Union U (Eq. 21).
    U = 1 - prod(1 - w_{i->j}) for all edges.
    """
    product = 1.0
    for e in edges:
        product *= (1.0 - e.weight)
    return 1.0 - product


def compute_negotiation(edges: List[ZigscoreEdge]) -> float:
    """
    Compute Negotiation N (Eq. 23).
    N = exp(-sum(delta_{i->j}))
    """
    total_disagreement = sum(e.disagreement for e in edges)
    return np.exp(-total_disagreement)


def compute_coherence(edges: List[ZigscoreEdge], cycle_edges: List[Tuple[int, int]],
                      alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0) -> Tuple[float, Dict]:
    """
    Compute Coherence score Coh(G) (Eq. 24).
    Coh(G) = sigma(alpha * U + beta * I + gamma * N)
    
    Returns: (coherence_score, component_dict)
    """
    U = compute_union(edges)
    I = compute_intersection(edges, cycle_edges)
    N = compute_negotiation(edges)
    
    raw = alpha * U + beta * I + gamma * N
    coh = sigmoid(raw)
    
    return coh, {"union": U, "intersection": I, "negotiation": N, "raw": raw}


def compute_novelty(paper_components: Dict, landmark_components: Dict,
                    alpha_prime: float = 1.0, beta_prime: float = 1.0, 
                    gamma_prime: float = 1.0) -> Tuple[float, Dict]:
    """
    Compute Novelty score N(G) (Eq. 25-26).
    N(G) = sigma(alpha' * delta_U + beta' * delta_I + gamma' * delta_N)
    
    where delta_X = X_paper - X_bar_landmark
    """
    delta_U = paper_components["union"] - landmark_components["union_mean"]
    delta_I = paper_components["intersection"] - landmark_components["intersection_mean"]
    delta_N = paper_components["negotiation"] - landmark_components["negotiation_mean"]
    
    raw = alpha_prime * delta_U + beta_prime * delta_I + gamma_prime * delta_N
    nov = sigmoid(raw)
    
    return nov, {"delta_union": delta_U, "delta_intersection": delta_I, 
                 "delta_negotiation": delta_N, "raw": raw}


def build_zigscore_graph(paper_id: str, embeddings: Dict[str, np.ndarray],
                       aligner: CrossModalAligner,
                       available_modalities: Optional[List[int]] = None) -> ZigscoreGraph:
    """
    Build a complete Zigscore graph for a paper.
    
    Args:
        paper_id: Unique paper identifier
        embeddings: Dict mapping modality name to embedding vector
        aligner: Cross-modal alignment module
        available_modalities: List of available modality indices (for handling missing modalities)
    
    Returns:
        ZigscoreGraph with computed edges and weights
    """
    if available_modalities is None:
        available_modalities = list(range(5))
    
    # Build nodes
    nodes = []
    for i, mdef in enumerate(MODALITY_DEFS):
        node = ModalityNode(
            name=mdef.name,
            index=mdef.index,
            layer=mdef.layer,
            embedding=embeddings.get(mdef.name, None)
        )
        nodes.append(node)
    
    # Build edges (only for available modalities)
    edges = []
    for src, tgt in BACKBONE_EDGES:
        if src in available_modalities and tgt in available_modalities:
            h_src = nodes[src].embedding
            h_tgt = nodes[tgt].embedding
            if h_src is not None and h_tgt is not None:
                w, delta = compute_edge_weight(aligner, h_src, h_tgt, src, tgt)
                edges.append(ZigscoreEdge(source=src, target=tgt, weight=w, disagreement=delta))
    
    graph = ZigscoreGraph(paper_id=paper_id, nodes=nodes, edges=edges)
    
    # Compute coherence
    cycle_edges = [(ZIGSCORE_CYCLE[i], ZIGSCORE_CYCLE[i+1]) for i in range(len(ZIGSCORE_CYCLE)-1)]
    coh, components = compute_coherence(edges, cycle_edges)
    graph.coherence = coh
    
    return graph


def build_landmark(papers: List[ZigscoreGraph], track: str = "all") -> Dict:
    """
    Build landmark statistics from a corpus of papers (for novelty computation).
    
    Args:
        papers: List of ZigscoreGraph objects (historical accepted papers)
        track: Research area / track name
    
    Returns:
        Dict with mean statistics for Union, Intersection, Negotiation
    """
    unions = []
    intersections = []
    negotiations = []
    
    cycle_edges = [(ZIGSCORE_CYCLE[i], ZIGSCORE_CYCLE[i+1]) for i in range(len(ZIGSCORE_CYCLE)-1)]
    
    for g in papers:
        U = compute_union(g.edges)
        I = compute_intersection(g.edges, cycle_edges)
        N = compute_negotiation(g.edges)
        unions.append(U)
        intersections.append(I)
        negotiations.append(N)
    
    landmark = {
        "track": track,
        "n_papers": len(papers),
        "union_mean": np.mean(unions) if unions else 0.0,
        "union_std": np.std(unions) if unions else 0.0,
        "intersection_mean": np.mean(intersections) if intersections else 0.0,
        "intersection_std": np.std(intersections) if intersections else 0.0,
        "negotiation_mean": np.mean(negotiations) if negotiations else 0.0,
        "negotiation_std": np.std(negotiations) if negotiations else 0.0,
        "all_unions": unions,
        "all_intersections": intersections,
        "all_negotiations": negotiations,
    }
    
    return landmark


class ZigscorePersistence:
    """
    Compute Zigscore persistence barcodes for paper evaluation.
    
    Implements the discrete multimodal Zigscore module (Section 6.3):
    K1(ε) ↪ B1(ε) ↩ K2(ε) ↪ ... ↪ B_{T-1}(ε) ↩ K_T(ε)
    """
    
    def __init__(self, epsilon_range: Tuple[float, float] = (0.0, 2.0), 
                 n_epsilon: int = 50, homology_dim: int = 1):
        self.epsilon_range = epsilon_range
        self.n_epsilon = n_epsilon
        self.homology_dim = homology_dim
        self.epsilons = np.linspace(epsilon_range[0], epsilon_range[1], n_epsilon)
    
    def build_simplicial_complex(self, points: np.ndarray, epsilon: float) -> np.ndarray:
        """
        Build a Vietoris-Rips complex from point cloud at threshold epsilon.
        Returns adjacency matrix.
        """
        n = len(points)
        dists = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                dists[i, j] = dists[j, i] = np.linalg.norm(points[i] - points[j])
        adj = (dists <= epsilon).astype(int)
        np.fill_diagonal(adj, 1)
        return adj
    
    def compute_betti_numbers(self, adj: np.ndarray) -> List[int]:
        """
        Compute Betti numbers from adjacency matrix.
        β_0 = number of connected components
        β_1 ≈ (number of cycles) - approximated via rank computation
        """
        n = adj.shape[0]
        
        # β_0: connected components via BFS
        visited = set()
        components = 0
        for start in range(n):
            if start not in visited:
                components += 1
                queue = [start]
                while queue:
                    node = queue.pop(0)
                    if node not in visited:
                        visited.add(node)
                        for neighbor in range(n):
                            if adj[node, neighbor] and neighbor not in visited:
                                queue.append(neighbor)
        
        beta_0 = components
        
        # β_1: number of independent cycles = edges - vertices + components
        n_edges = (np.sum(adj) - n) // 2  # exclude diagonal, divide by 2 for undirected
        beta_1 = max(0, n_edges - n + components)
        
        return [beta_0, beta_1]
    
    def compute_persistence_for_graph(self, graph: ZigscoreGraph) -> List[Tuple[float, float]]:
        """
        Compute persistence barcode for a single paper's Zigscore graph.
        
        Tracks homological features across filtration parameter ε.
        """
        # Collect all embeddings as point cloud
        points = []
        for node in graph.nodes:
            if node.embedding is not None:
                points.append(node.embedding)
        
        if len(points) < 2:
            return []
        
        points = np.array(points)
        
        # Track Betti numbers across filtration
        betti_history = []
        for eps in self.epsilons:
            adj = self.build_simplicial_complex(points, eps)
            betti = self.compute_betti_numbers(adj)
            betti_history.append(betti)
        
        # Extract persistence intervals from Betti number changes
        barcode = self._extract_barcode(betti_history, self.homology_dim)
        return barcode
    
    def _extract_barcode(self, betti_history: List[List[int]], dim: int) -> List[Tuple[float, float]]:
        """
        Extract persistence barcode from Betti number trajectory.
        
        When β_dim increases → birth event at that ε
        When β_dim decreases → death event at that ε
        """
        intervals = []
        births = []  # stack of birth times
        
        prev_betti = 0
        for i, betti_nums in enumerate(betti_history):
            if dim >= len(betti_nums):
                curr_betti = 0
            else:
                curr_betti = betti_nums[dim]
            
            eps = self.epsilons[i]
            
            # New features born
            for _ in range(max(0, curr_betti - prev_betti)):
                births.append(eps)
            
            # Features died
            for _ in range(max(0, prev_betti - curr_betti)):
                if births:
                    birth = births.pop(0)  # FIFO: oldest feature dies first
                    intervals.append((birth, eps))
            
            prev_betti = curr_betti
        
        # Features that never die extend to max epsilon
        for birth in births:
            intervals.append((birth, self.epsilons[-1]))
        
        return intervals
    
    def compute_zigscore_temporal(self, graphs_over_time: List[ZigscoreGraph]) -> List[Tuple[float, float]]:
        """
        Compute Zigscore persistence over a temporal sequence of paper graphs.
        
        This implements the discrete multimodal Zigscore module (Definition 3):
        K1(ε) ↪ B1(ε) ↩ K2(ε) ↪ ... 
        
        Used for temporal analysis (Theorem 2) and landmark construction.
        """
        all_barcodes = []
        
        # For each pair of adjacent time steps, compute bridge complex
        for t in range(len(graphs_over_time)):
            # Per-time barcode
            bc = self.compute_persistence_for_graph(graphs_over_time[t])
            all_barcodes.extend(bc)
            
            # Bridge between t and t+1
            if t < len(graphs_over_time) - 1:
                bridge_bc = self._compute_bridge_persistence(
                    graphs_over_time[t], graphs_over_time[t + 1]
                )
                all_barcodes.extend(bridge_bc)
        
        return all_barcodes
    
    def _compute_bridge_persistence(self, g1: ZigscoreGraph, g2: ZigscoreGraph) -> List[Tuple[float, float]]:
        """
        Compute persistence for bridge complex B_t = K_t ∪ K_{t+1} (union bridging).
        """
        # Union of point clouds
        points1 = [n.embedding for n in g1.nodes if n.embedding is not None]
        points2 = [n.embedding for n in g2.nodes if n.embedding is not None]
        
        if not points1 or not points2:
            return []
        
        combined = np.vstack(points1 + points2)
        
        betti_history = []
        for eps in self.epsilons:
            adj = self.build_simplicial_complex(combined, eps)
            betti = self.compute_betti_numbers(adj)
            betti_history.append(betti)
        
        return self._extract_barcode(betti_history, self.homology_dim)
    
    def bottleneck_distance(self, barcode1: List[Tuple[float, float]], 
                            barcode2: List[Tuple[float, float]]) -> float:
        """
        Compute bottleneck distance between two persistence barcodes.
        d_B(Bar(V), Bar(V~))
        
        Used for stability verification (Theorem 1).
        """
        if not barcode1 and not barcode2:
            return 0.0
        
        # Pad shorter barcode with diagonal points
        bc1 = list(barcode1)
        bc2 = list(barcode2)
        
        # Add diagonal projections for unmatched points
        for (b, d) in bc1:
            mid = (b + d) / 2
            bc2.append((mid, mid))
        for (b, d) in bc2:
            mid = (b + d) / 2
            bc1.append((mid, mid))
        
        # Compute cost matrix
        n = len(bc1)
        cost = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                # L-infinity distance between intervals
                cost[i, j] = max(abs(bc1[i][0] - bc2[j][0]), abs(bc1[i][1] - bc2[j][1]))
        
        # Greedy approximation to bottleneck (exact requires Hungarian)
        # For now use greedy matching
        used_j = set()
        max_cost = 0.0
        for i in range(min(len(barcode1), len(barcode2))):
            best_j = -1
            best_cost = float('inf')
            for j in range(n):
                if j not in used_j and cost[i, j] < best_cost:
                    best_cost = cost[i, j]
                    best_j = j
            if best_j >= 0:
                used_j.add(best_j)
                max_cost = max(max_cost, best_cost)
        
        return max_cost


def novelty_topological_impact(paper_graph: ZigscoreGraph, 
                                landmark_graphs: List[ZigscoreGraph],
                                persistence_engine: ZigscorePersistence) -> Dict:
    """
    Measure novelty as topological impact: how does adding this paper change
    the landmark's persistent homology?
    
    Detects:
    - New long-lived holes (paper opens new research direction)
    - Killed existing holes (paper fills topological gap)
    """
    # Barcode without test paper
    bc_before = persistence_engine.compute_zigscore_temporal(landmark_graphs)
    
    # Barcode with test paper added
    augmented = landmark_graphs + [paper_graph]
    bc_after = persistence_engine.compute_zigscore_temporal(augmented)
    
    # Analyze changes
    bottleneck = persistence_engine.bottleneck_distance(bc_before, bc_after)
    
    # Count new long-lived features (birth-death gap > threshold)
    threshold = 0.5
    long_before = [(b, d) for b, d in bc_before if d - b > threshold]
    long_after = [(b, d) for b, d in bc_after if d - b > threshold]
    
    new_features = len(long_after) - len(long_before)
    
    return {
        "bottleneck_distance": bottleneck,
        "barcode_before": bc_before,
        "barcode_after": bc_after,
        "new_long_lived_features": new_features,
        "n_features_before": len(bc_before),
        "n_features_after": len(bc_after),
    }
