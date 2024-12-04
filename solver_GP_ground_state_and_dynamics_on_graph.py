#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 13:48:43 2024
@author: Marcin Płodzień
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import networkx as nx
from numpy import linalg as LA
import matplotlib.animation as animation
import matplotlib.colors as mcolors

import pydot
from networkx.drawing.nx_pydot import graphviz_layout


#graph generating functions
def sierpinski_triangle(level, G=None, p1=(0, 0), p2=(1, 0), p3=(0.5, 0.866)):
    if G is None:
        G = nx.Graph()
    if level == 0:
        # Add the triangle edges
        G.add_edge(p1, p2)
        G.add_edge(p2, p3)
        G.add_edge(p3, p1)
    else:
        # Subdivide the triangle into 3 smaller triangles
        mid1 = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        mid2 = ((p2[0] + p3[0]) / 2, (p2[1] + p3[1]) / 2)
        mid3 = ((p3[0] + p1[0]) / 2, (p3[1] + p1[1]) / 2)
        sierpinski_triangle(level - 1, G, p1, mid1, mid3)
        sierpinski_triangle(level - 1, G, mid1, p2, mid2)
        sierpinski_triangle(level - 1, G, mid3, mid2, p3)
    return G, {node: node for node in G.nodes}

def carpet_graph(n):
    #so far it has a unit length between sites
    def carpet(n):
        """Generate a Sierpiński carpet fractal matrix."""
        mat = np.array([[1]])
        for _ in range(n):
            mat = np.block([
                [mat, mat, mat],
                [mat, np.zeros_like(mat), mat],
                [mat, mat, mat],
            ])
        return mat
    """Generate the graph representation of a Sierpiński carpet."""
    fractal = carpet(n)
    rows, cols = fractal.shape
    G = nx.Graph()

    # Get all positions where the value is 1
    positions = np.argwhere(fractal == 1)
    pos_dict = {tuple(p): p for p in positions}

    # Add edges between adjacent positions
    offsets = [(1, 0), (0, 1)]  # Horizontal and vertical adjacency
    for x, y in positions:
        for dx, dy in offsets:
            neighbor = (x + dx, y + dy)
            if neighbor in pos_dict:
                G.add_edge((x, y), neighbor)

    return G, pos_dict


 
custom_cmap = mcolors.LinearSegmentedColormap.from_list(
    "gnuplot_style", ["#000000", "#5e4fa2", "#3288bd", "#66c2a5", "#abdda4", "#e6f598", "#fee08b", "#f46d43", "#9e0142"]
)

def plot_density_on_graph(G, rho, pos=None, node_size = 100, title=" ", path = "./", filename = "fig.png"):
 
    if pos is None:
        pos = nx.spring_layout(G)  # Use spring layout if no position provided

    # Clip and normalize rho to range [0, 1] for consistent color mapping
    rho_clipped = np.clip(rho, 0, 1)

    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Draw the graph with color-coded nodes and black edges with width 2
    nx.draw(
        G, pos, node_size=node_size, 
        node_color=rho_clipped, cmap=custom_cmap, 
        edge_color="black", width=2, with_labels=False, ax=ax
    )
    
    # Add color bar fixed from 0 to 1
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label(r"Density $|\psi_{i,j}|^2$", rotation=270, labelpad=20)
    
    # Set plot title and aspect ratio
    ax.set_title(title)
    ax.set_aspect('equal')
    plt.axis("off")
 
    plt.savefig(path + filename, dpi = 600, format = "png")
    plt.show()
 
 
 
def animate_density_on_graph(G, rho_vs_t, pos=None, node_size=100, title=" ", path="./", filename="animation.gif"):
    """
    Animates the time evolution of the wave function density on the graph.
    
    Parameters:
    - G: NetworkX graph
    - rho_vs_t: List of tuples [(t, rho)], where t is the time and rho is the density at that time
    - pos: Position dictionary for nodes. If None, calculates using spring layout.
    - node_size: Size of the graph nodes
    - title: Title for the plot
    - path: Path to save the animation
    - filename: Filename for the output GIF
    """
    if pos is None:
        pos = nx.spring_layout(G)  # Use spring layout if no position provided

    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Set up fixed color map with range [0, 1] for all frames
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(vmin=0, vmax=1))
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label(r"Density ($|\psi_i|^2$)", rotation=270, labelpad=20)

    # Define the update function for each frame in the animation
    def update(frame):
        ax.clear()  # Clear previous frame        
        t, rho = rho_vs_t[frame]
        
        # Clip and normalize rho to [0, 1] for color mapping
        rho_clipped = np.clip(rho, 0, 1)
        
        # Draw graph with updated node colors and black edges with width 2
        nx.draw(
            G, pos, node_size=node_size, 
            node_color=rho_clipped, cmap=custom_cmap, 
            edge_color="black", width=2, with_labels=False, ax=ax
        )
        
        # Fix color bar scale to [0, 1] for consistency
        sm.set_clim(0, 1)
        cbar.update_normal(sm)
        
        # Set title with time information
        ax.set_title(title + f" | t={t:.2f}")
        ax.set_aspect('equal')
        plt.axis("off")
    
    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=len(rho_vs_t), repeat=True)
    
    # Ensure filename is .gif
    if not filename.endswith(".gif"):
        filename += ".gif"
    
    # Save as GIF using the 'pillow' writer for better compatibility
    save_path = path + filename
    ani.save(save_path, writer='pillow', fps=10)
    print(f"Animation saved as {save_path}")

 




def get_norm(psi):
    return np.sum(np.abs(psi)**2)

#%%



node_size = 50

 

r = 3
h = 4

 
G = nx.balanced_tree(r, h)
# pos = nx.circular_layout(G)

L = 30
G = nx.grid_2d_graph(L, L, periodic = False)
# Set positions for nodes in a 2D plane
pos = {(x, y): (y, -x) for x, y in G.nodes()}


level = 3
G, pos = carpet_graph(level)
 
                  


N = G.number_of_nodes()
print("Number of nodes: {:d}".format(L))
# pos = nx.nx_agraph.graphviz_layout(G, prog="twopi", args="")
fig, ax = plt.subplots(1, 1, figsize=(8, 8))

nx.draw(G, pos, node_size=node_size, alpha=1, node_color="blue", edge_color="black", with_labels=False)

ax.set_aspect('equal')
 
plt.show()

#%%
H_0 = -nx.adjacency_matrix(G).toarray() # Lattice couplings on a graph
evals, P = LA.eigh(H_0)

psi_GS_0 = P[:,0]
rho_GS_0 = np.abs(psi_GS_0)**2 

 
energy_0 = np.conj(psi_GS_0) @ (H_0@psi_GS_0)
energy = energy_0

 


# title_string = r"Ground state density on Graph | interactions $g_i$ = {:2.2f}".format(g_i)
# path = "./"
# filename = "fig_ground_state_g_i#.{:2.2f}".format(g_i) + ".png"
# plot_density_on_graph(G, rho_GS_0, pos=pos, node_size = node_size, title=title_string, path = path, filename = filename)


#%% Define initial state as a Gaussian with graph-distance 

# Get all nodes from the graph
nodes = list(G.nodes)

# Find the approximate center of the graph
# center_node = (300, L // 2)

# Check if the center_node is in the graph
# if center_node not in G:
    # If the expected center is not in the graph, select a node that is approximately central
center_node = nodes[400]  # Choose a node from the middle of the list


# Define distance between nodes:

# 1. Compute the shortest path distances from the chosen center node
distances = nx.single_source_shortest_path_length(G, center_node)
 
# Define the width of the Gaussian
sigma = 4 # You can adjust this value to change the width of the Gaussian
g_i = 0
# Map each node to a unique index for a 1D representation
node_to_index = {node: idx for idx, node in enumerate(G.nodes)}

# Initialize the wavefunction as a dictionary for nodes in G
psi = {node: 0.0 for node in G.nodes}


# Construct the Gaussian wave packet
for node, distance in distances.items():
    psi[node] = np.exp(-distance**2 / (2 * sigma**2))

# Convert the dictionary values to an array and normalize
psi_ini = np.array(list(psi.values()), dtype=np.complex128)
psi_ini /= np.linalg.norm(psi_ini)

rho_ini = np.abs(psi_ini)**2

title_string = r"Initial state for imaginary time evolution | Gauss with graph-distance "
path = "./"
filename = "fig_ground_state_g_i#{:2.2f}".format(g_i) + ".png"
plot_density_on_graph(G, rho_ini, pos=pos, node_size = node_size, title=title_string, path = path, filename = filename)


#%% Define initial state as a Gaussian with euclidean-distance

# Define the width of the Gaussian
sigma = 5  # You can adjust this value to change the width of the Gaussia

# Initialize the wavefunction as a dictionary for nodes in G
psi = {node: 0.0 for node in G.nodes}

# Construct the Gaussian wave packet based on Euclidean distance
for node in G.nodes:
    # Euclidean distance between the center node and the current node
    distance = np.sqrt((node[0] - center_node[0])**2 + (node[1] - center_node[1])**2)
    # Assign Gaussian amplitude based on the Euclidean distance
    psi[node] = np.exp(-distance**2 / (2 * sigma**2))

# Convert the dictionary values to an array and normalize
psi_ini = np.array(list(psi.values()), dtype=np.complex128)
psi_ini /= np.linalg.norm(psi_ini)
rho_ini = np.abs(psi_ini)**2


g_i = 0
title_string = r"Initial state for imaginary time evolution | Gauss with euclidean-distance "
path = "./"
filename = "fig_ground_state_g_i#{:2.2f}".format(g_i) + ".png"
plot_density_on_graph(G, rho_ini, pos=pos, node_size = node_size, title=title_string, path = path, filename = filename)

#%%
dt = 0.01
normalization_step = 100
epsilon = 1e-4
time_i = 0
energy_change_velocity = 1
g_i = -6
# Imaginary time evolution loop to obtain ground state with interactions
psi = psi_ini.copy()
while energy_change_velocity > epsilon:   
    V = g_i*np.abs(psi)**2
    H = H_0 + np.diag(V)   
    H_psi = H @ psi
    psi = psi - dt * H_psi
    norm = np.sum(np.abs(psi)**2)  
    psi /= np.sqrt(norm)    
    if(time_i % normalization_step == 0):
        energy_new = np.conj(psi) @ H_psi  - 0.5*g_i*np.sum(np.abs(psi)**4)
        energy_change_velocity = abs((energy_new - energy) / (normalization_step * dt))
        print(f' E = {energy_new} | dE/dt = {energy_change_velocity}')
        energy = energy_new 
    time_i += 1
psi_GS = psi.copy()
rho_GS = np.abs(psi_GS)**2  

  
title_string = r"Ground state density on Graph | interactions $g_i = {:2.2f}$".format(g_i)
path = "./"
filename = "fig_ground_state_g0#.{:2.2f}".format(g_i) + ".png"
plot_density_on_graph(G, rho_GS, pos=pos, node_size = node_size, title=title_string, path = path, filename = filename)
#%%

def prepare_RK45_evolution_in_dt(H, psi, dt):
    """
    Evolve the wave function using the 4th-order Runge-Kutta (RK45) method.
    """
    # Compute intermediate steps K1, K2, K3, K4 for RK45
    psi_dt = psi.copy()
    K1 = -1j * dt * H @ psi_dt
    psi_tmp1 = psi_dt + 0.5 * K1
    K2 = -1j * dt * H @ psi_tmp1
    psi_tmp2 = psi_dt + 0.5 * K2
    K3 = -1j * dt * H @ psi_tmp2
    psi_tmp3 = psi_dt + K3
    K4 = -1j * dt * H @ psi_tmp3

    # Combine to get the final evolution step
    psi_t_dt = psi_dt + (K1 + 2.0 * K2 + 2.0 * K3 + K4) / 6.0
    return psi_t_dt 


# Time evolution parameters
t_max = 100
Nt = int(np.floor(t_max / dt + 1))
N_shots = 100
time_shot = int(Nt/N_shots)

g_f = g_i
# g_f = 0.
# Time evolution loop
rho_vs_t = []
psi_ini = psi_GS.copy() # to check if GP ground state does not evolve in time
psi_t = psi_ini.copy()  # Initial state for time-evolution
counter = 0
fidelity_vs_t = []
for n in range(Nt + 1):
    t = n * dt                        
   
    ####################################################################
    ## Runge-Kutta 45 algorithm for solving Schroedinger equation
    V = g_f*np.abs(psi_t)**2
    H = H_0 + np.diag(V)            
    psi_t = prepare_RK45_evolution_in_dt(H, psi_t, dt)        
    ####################################################################            
    
    if(np.mod(n, time_shot) == 0):               
            
        rho = np.abs(psi_t)**2                                    
        rho_vs_t.append((t, np.abs(psi_t)**2))
        fidelity = np.abs(np.dot(psi_ini.conj(), psi_t))**2
        fidelity_vs_t.append([t, fidelity])
        print("Progress: {:d}[%]".format(counter))
        counter += 1
        
         
fidelity_vs_t = np.array(fidelity_vs_t)
#%%        
fig, ax = plt.subplots(1, 3, figsize = (20, 10))

# Draw the graph with color-coded nodes and black edges with width 2
nx.draw(
    G, pos, node_size=node_size, 
    node_color=rho_ini, cmap=custom_cmap, 
    edge_color="black", width=2, with_labels=False, ax = ax[0])

# # Add color bar fixed from 0 to 1
sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(vmin=0, vmax=1))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax[0], fraction=0.03, pad=0.04)
ax[0].set_title("Initial state Density $|\psi_{i,j}|^2$" + r" | $g_i $ = {:2.2f}".format(g_i))
ax[0].set_aspect('equal')

rho_fin = rho_vs_t[-1][1]
nx.draw(
    G, pos, node_size=node_size, 
    node_color=rho_fin, cmap=custom_cmap, 
    edge_color="black", width=2, with_labels=False, ax = ax[1])

# # # Add color bar fixed from 0 to 1
sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=plt.Normalize(vmin=0, vmax=1))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax[1], fraction=0.03, pad=0.04)
ax[1].set_title(r"Final state Density $|\psi_{i,j}|^2$" + r" | $g_f $= {:2.2f}".format(g_f))
ax[1].set_aspect('equal')


ax[2].plot(fidelity_vs_t[:,0], fidelity_vs_t[:,1])
ax[2].set_xlabel("time")
ax[2].set_ylabel("Fidelity $|\langle \psi_{i}|\psi_{f}|^2$")
ax[2].set_aspect('auto')

path = "./"
filename = "fig_g_i.{:2.2f}_g_f.{:2.2f}_t_max.{:2.2f}".format(g_i, g_f, t_max) + ".png"
plt.savefig(path + filename, dpi = 600, format = "png")

#%%
path = "./"
filename = "animation_g_i.{:2.2f}_g_f.{:2.2f}".format(g_i, g_f)  
title_string = r"Wave function density | $g_i =$ {:2.2f} | $g_f$ = .{:2.2f}".format(g_i, g_f)
 
animate_density_on_graph(G, rho_vs_t, pos=pos, node_size = node_size, title=title_string, path = path , filename = filename)
 