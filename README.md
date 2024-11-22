# Gross_Pitaevskii_equation_on_graph
Numerical solver for ground state and time evolution of a Gross-Pitaevskii equation on a given graph geometry, defined via
graph $G = (V, E)$, where $V$ is a set of nodes, size $|V|=L$, and $E = {(k,l), k\in V, l\in V}$ is a set of edges in a graph.

Gross-Pitaevskii equation for a $k$-th node is:
$i\partial_t \psi_k(t) = -\sum_{l} H_{kl}\psi_l(t) + g|\psi_k(t)|^2 \psi_k(t)$, where $H_{kl}$ is adjacency matrix of a graph 
