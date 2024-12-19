import matplotlib.pyplot as plt
import numpy as np

def generate_sierpinski_gasket(L, N):
    """
    Generates a Sierpinski gasket on a grid of size LxL with N discrete points.

    Parameters:
        L (int): Size of the grid (LxL).
        N (int): Number of points to generate.

    Returns:
        fractal (ndarray): LxL array representing the Sierpinski gasket fractal (1 where fractal exists, 0 elsewhere).
    """
    # Initial triangle vertices
    vertices = np.array([[0, 0], [L - 1, 0], [(L - 1) // 2, int((L - 1) * np.sqrt(3) / 2)]])

    # Start at a random point inside the triangle
    current_point = np.array([L // 2, L // 2])

    fractal = np.zeros((L, L))

    for _ in range(N):
        # Choose a random vertex
        chosen_vertex = vertices[np.random.randint(0, 3)]

        # Move halfway towards the chosen vertex
        current_point = (current_point + chosen_vertex) // 2

        # Mark the current point as part of the fractal
        fractal[current_point[0], current_point[1]] = 1
    potential = np.where(fractal == 1, 1, 0)
    return fractal, potential

def generate_sierpinski_carpet(L, depth):
    """
    Generates a Sierpinski carpet on a grid of size LxL.

    Parameters:
        L (int): Size of the grid (LxL).
        depth (int): Depth of recursion for the Sierpinski carpet.

    Returns:
        fractal (ndarray): LxL array representing the Sierpinski carpet fractal (1 where fractal exists, 0 elsewhere).
    """
    fractal = np.ones((L, L))

    def carpet(x, y, size, depth):
        if depth == 0:
            return
        step = size // 3
        fractal[x + step:x + 2 * step, y + step:y + 2 * step] = 0  # Clear the middle square
        for i in range(3):
            for j in range(3):
                if i == 1 and j == 1:
                    continue  # Skip the middle square
                carpet(x + i * step, y + j * step, step, depth - 1)

    carpet(0, 0, L, depth)
    potential = np.where(fractal == 0, 1, 0)
    return fractal, potential

# def generate_inverse_potential(fractal, V_0):
#     """
#     Generates a potential with amplitude V_0 where points do NOT belong to the fractal.

#     Parameters:
#         fractal (ndarray): LxL array representing the fractal (1 where fractal exists, 0 elsewhere).
#         V_0 (float): Amplitude of the potential where points do not belong to the fractal.

#     Returns:
#         potential (ndarray): LxL array representing the inverse potential.
#     """
#     potential = np.where(fractal == 1, V_0, 0)
#     return potential

def plot_potential(potential):
    """
    Plots the potential V(x, y) on an LxL grid.

    Parameters:
        potential (ndarray): LxL array representing the potential.
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(potential, cmap="hot_r", origin="lower")
    plt.colorbar(label="Potential V(x, y)")
    plt.title("Potential", fontsize=16)
    plt.axis("off")
    plt.show()

# Parameters for Sierpinski gasket
L_gasket = 512  # Size of the grid for the gasket
N_gasket = 100000  # Number of points for the gasket
V_0_gasket = 100.0  # Amplitude of the potential

# Generate and plot inverse potential for Sierpinski gasket
fractal_gasket, potential_gasket = generate_sierpinski_gasket(L_gasket, N_gasket)
# inverse_potential_gasket = generate_inverse_potential(fractal_gasket, V_0_gasket)
plot_potential(V_0_gasket*potential_gasket)

# Parameters for Sierpinski carpet
L_carpet = 512  # Size of the grid for the carpet (must be a power of 3)
depth_carpet = 6  # Depth of recursion for the carpet
V_0_carpet = 100.0  # Amplitude of the potential

# Generate and plot inverse potential for Sierpinski carpet
fractal_carpet, potential_carpet = generate_sierpinski_carpet(L_carpet, depth_carpet)
# inverse_potential_carpet = generate_inverse_potential(fractal_carpet, V_0_carpet)
plot_potential(V_0_carpet*potential_carpet)
