# Gross_Pitaevskii_equation_on_graph
Numerical solver for ground state and time evolution of a Gross-Pitaevskii equation on a given graph geometry:

$i\partial_t \psi_k = \sum_{l} H_{kl}\psi_l + g|\psi_k|^2$, where $H_{kl}$ is adjacency matrix of a graph $G = (V, E)$, where $V$ is a set of nodes, size $|V|=L$, and $E = {(k,l), k\in V, l\in V}$ is a set of edges in a graph.
