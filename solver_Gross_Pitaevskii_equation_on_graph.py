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



 
custom_cmap = mcolors.LinearSegmentedColormap.from_list(
    "gnuplot_style", ["#000000", "#5e4fa2", "#3288bd", "#66c2a5", "#abdda4", "#e6f598", "#fee08b", "#f46d43", "#9e0142"]
)

def plot_density_on_graph(G, rho, pos=None, node_size = 100, title=" ", path = "./", filename = "fig.png"):
    """
    Plots the density of the wave function on the graph, with zero density in white and max density in red.
    
    Parameters:
    - G: NetworkX graph
    - rho: Density array corresponding to each node in G
    - pos: Position dictionary for nodes. If None, calculates using spring layout.
    - title: Title of the plot
    """
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
    cbar.set_label("Density (|ψ|²)", rotation=270, labelpad=20)
    
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

 



#%%

def get_norm(psi):
    return np.sum(np.abs(psi)**2)

node_size = 50

 

r = 3
h = 4

 
G = nx.balanced_tree(r, h)
pos = nx.circular_layout(G)

L = G.number_of_nodes()
print("Number of nodes: {:d}".format(L))
pos = nx.nx_agraph.graphviz_layout(G, prog="twopi", args="")
fig, ax = plt.subplots(1, 1, figsize=(8, 8))

nx.draw(G, pos, node_size=node_size, alpha=1, node_color="blue", edge_color="black", with_labels=True)

ax.set_aspect('equal')
 
plt.show()

#%%
H_0 = -nx.adjacency_matrix(G).toarray() # Lattice couplings on a graph
evals, P = LA.eigh(H_0)

psi_GS_0 = P[:,0]
rho_GS_0 = np.abs(psi_GS_0)**2 

 
energy_0 = np.conj(psi_GS_0) @ (H_0@psi_GS_0)
energy = energy_0

 
pos = graphviz_layout(G, prog="twopi")

g = 0
title_string = r"Ground state density on Graph | interactions g = {:2.2f}".format(g)
path = "./"
filename = "fig_ground_state_g.{:2.2f}".format(g) + ".png"
plot_density_on_graph(G, rho_GS_0, pos=pos, node_size = node_size, title=title_string, path = path, filename = filename)


#%%
# Imaginary time evolution parameters

dt = 0.001
normalization_step = 10
epsilon = 1e-5
time_i = 1
energy_change_velocity = 1

g = 1
on_off_interactions = 1
# Imaginary time evolution loop to obtain ground state with interactions

psi = psi_GS_0.copy()
while energy_change_velocity > epsilon:
    time_i += 1
    rho = np.abs(psi)**2
    H = H_0 +  on_off_interactions*g*rho
    H_psi = H @ psi
    psi -= dt * H_psi
    norm = np.sum(np.abs(psi)**2)  
    psi /= np.sqrt(norm)
    
    if time_i % normalization_step == 0:
        energy_new = np.conj(psi) @ H_psi  
        energy_change_velocity = abs((energy_new - energy) / (normalization_step * dt))
        print(f' E = {energy_new} | dE/dt = {energy_change_velocity}')
        energy = energy_new

 
psi_GS = psi.copy()
rho_GS = np.abs(psi_GS)**2

 
title_string = r"Ground state density on Graph | interactions g = {:2.2f}".format(g)
path = "./"
filename = "fig_ground_state_g.{:2.2f}".format(g) + ".png"
plot_density_on_graph(G, rho_GS_0, pos=pos, node_size = node_size, title=title_string, path = path, filename = filename)


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
t_max = 5
Nt = int(np.floor(t_max / dt + 1))
N_shots = 100
time_shot = int(Nt/N_shots)


g = 1
# Time evolution loop
counter = 0
rho_vs_t = []
# psi_ini = psi_GS.copy()
psi_ini = np.zeros((L,), dtype=np.complex128)
psi_ini[0] = 1
 
psi_t = psi_ini  # Initial state for time-evolution
for n in range(Nt + 1):
    t = n * dt                         
    V = g*np.abs(psi_t)**2
    H = H_0 + np.diag(V)            
    psi_t = prepare_RK45_evolution_in_dt(H, psi_t, dt)            
    if(np.mod(n, time_shot) == 0):               
            
        rho = np.abs(psi_t)**2                                    
        rho_vs_t.append((t, np.abs(psi_t)**2))
        print("Progress: {:d}[%]".format(counter))
        counter += 1  
         
        
 
plot_density_on_graph(G, rho_GS_0, pos=pos, node_size = node_size)

path = "./"
filename = "animation_g.{:2.2f}".format(g) + ".png"
title_string = "Wave function density | g = {:2.2f}".format(g)
 
animate_density_on_graph(G, rho_vs_t, pos=pos, node_size = node_size, title=title_string, path = path , filename = filename)
 
