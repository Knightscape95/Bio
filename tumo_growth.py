import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from pysces import model
from scipy.integrate import odeint
from scipy.optimize import minimize

# TCGA data loading and processing
def load_process_tcga_data():
    tumor_path = 'tumor.csv'
    gene_expression = 'expression.csv'
    treatment_history = 'history.csv'

    # Load synthetic data (replace with actual loading code)
    tumor_sizes_df = pd.read_csv(tumor_path)
    gene_expression_df = pd.read_csv(gene_expression)
    treatment_history_df = pd.read_csv(treatment_history)

    # Extract relevant information (replace column names with your actual column names)
    tumor_sizes = tumor_sizes_df['TumorSize'].values
    gene_expression = gene_expression_df.iloc[:, 1:].values  # Assuming the first column is patient IDs
    treatment_history = treatment_history_df['Treatment'].values

    # Data processing steps  - Normalize gene expression data
    gene_expression_normalized = (gene_expression - np.mean(gene_expression, axis=0)) / np.std(gene_expression, axis=0)

    # - Categorize treatment history where
    treatment_categories = {'Drug A': 0, 'Drug B': 1, 'No Treatment': 2}
    treatment_labels = np.array([treatment_categories.get(treatment, -1) for treatment in treatment_history])

    # Example: Handling missing values in treatment_labels
    treatment_labels[treatment_labels == -1] = 2  # Assign 'No Treatment' for missing or unknown treatments

    # Extract relevant information
    data_info = {
        'tumor_sizes': tumor_sizes,
        'gene_expression_normalized': gene_expression_normalized,
        'treatment_labels': treatment_labels,
    }

    return data_info

# Model parameters extracted from literature source
grid_size = 50
initial_tumor_density = 0.05
initial_immune_density = 0.02
oxygen_diffusion_rate = 0.01

# Acquire and process TCGA data
tcga_data = load_process_tcga_data()

# initialize model parameters
initial_tumor_sizes = tcga_data['tumor_sizes'][:grid_size**2]
initial_gene_expression = tcga_data['gene_expression_normalized'][:grid_size**2, :]
initial_treatment_labels = tcga_data['treatment_labels'][:grid_size**2]

# Assuming gene expression data is used to determine initial immune cell density
initial_immune_density = np.mean(initial_gene_expression, axis=1)

# a graph using NetworkX to represent spatial structure
G = nx.grid_2d_graph(grid_size, grid_size)

# Initialize tumor and immune cell agents on the graph
tumor_cells = {(i, j): size for (i, j), size in zip(G.nodes(), initial_tumor_sizes)}
immune_cells = {(i, j): density for (i, j), density in zip(G.nodes(), initial_immune_density)}

# Set initial conditions
tumor_nodes = list(np.random.choice(list(G.nodes), size=int(initial_tumor_density * grid_size**2), replace=False))
immune_nodes = list(np.random.choice(list(G.nodes), size=int(initial_immune_density * grid_size**2), replace=False))

for node in tumor_nodes:
    tumor_cells[node] = 1

for node in immune_nodes:
    immune_cells[node] = 1

# Setting initial microenvironment conditions 
oxygen_levels = np.ones((grid_size, grid_size))

# Define the logistic growth and Michaelis-Menten kinetics equations
def logistic_growth(t, y, r, K):
    dydt = r * y * (1 - y / K)
    return dydt

def michaelis_menten(t, y, Vmax, Km, drug_concentration):
    dydt = (Vmax * y) / (Km + y) * (1 - y / drug_concentration)
    return dydt

# Define functions for cellular processes
def cell_growth(tumor_cells, params):
    growth_rate = params['k_tumor_growth']
    capacity = params['k_tumor_capacity']
    for node in tumor_cells:
        tumor_cells[node] += logistic_growth(0, tumor_cells[node], growth_rate, capacity)

def cell_death(tumor_cells, params):
    death_rate = params['k_tumor_death']
    for node in tumor_cells:
        tumor_cells[node] -= death_rate * tumor_cells[node]

def immune_proliferation(immune_cells, tumor_cells, params):
    proliferation_rate = params['k_immune_proliferation']
    stimulation_rate = params['k_immune_stimulation']
    for node in immune_cells:
        immune_cells[node] += proliferation_rate * immune_cells[node] * (1 + stimulation_rate * tumor_cells[node])

def immune_death(immune_cells, params):
    death_rate = params['k_immune_death']
    for node in immune_cells:
        immune_cells[node] -= death_rate * immune_cells[node]

def cell_migration(tumor_cells, immune_cells, params):
    migration_rate = params['k_cell_migration']
    for node in tumor_cells:
        neighbors = list(G.neighbors(node))
        tumor_cells[node] += migration_rate * sum(tumor_cells[neighbor] for neighbor in neighbors)

    for node in immune_cells:
        neighbors = list(G.neighbors(node))
        immune_cells[node] += migration_rate * sum(immune_cells[neighbor] for neighbor in neighbors)

def oxygen_diffusion(oxygen_levels, params):
    diffusion_rate = params['k_oxygen_diffusion']
    oxygen_levels += diffusion_rate * (
        np.roll(oxygen_levels, 1, axis=0) + np.roll(oxygen_levels, -1, axis=0) +
        np.roll(oxygen_levels, 1, axis=1) + np.roll(oxygen_levels, -1, axis=1) - 4 * oxygen_levels
    )

# Define functions for microenvironment dynamics
def update_oxygen_levels(oxygen_levels, tumor_cells, params):
    # Example: Adjust oxygen levels based on tumor presence and consumption
    tumor_consumption = params['k_tumor_oxygen_consumption']
    for node in tumor_cells:
        oxygen_levels[node] -= tumor_consumption * tumor_cells[node]

def update_immune_dynamics(immune_cells, tumor_cells, params):
    # Example: Modulate immune cell dynamics based on tumor presence and immune stimulation
    proliferation_rate = params['k_immune_proliferation']
    stimulation_rate = params['k_immune_stimulation']
    for node in immune_cells:
        immune_cells[node] += proliferation_rate * immune_cells[node] * (1 + stimulation_rate * tumor_cells[node])

def visualize_model_state(tumor_cells, immune_cells, oxygen_levels, title=''):
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    pos = {(i, j): (i, -j) for i, j in G.nodes()}
    nx.draw(G, pos, node_color=[tumor_cells[node] for node in G.nodes], cmap='viridis', with_labels=False,
            node_size=20, alpha=0.7, linewidths=0.5, edgecolors='black')
    plt.title('Cell Density')

    plt.subplot(2, 2, 2)
    nx.draw(G, pos, node_color=[tumor_cells[node] for node in G.nodes], cmap='Reds', with_labels=False,
            node_size=20, alpha=0.7, linewidths=0.5, edgecolors='black')
    plt.title('Tumor Cell Density')

    plt.subplot(2, 2, 3)
    nx.draw(G, pos, node_color=[immune_cells[node] for node in G.nodes], cmap='Blues', with_labels=False,
            node_size=20, alpha=0.7, linewidths=0.5, edgecolors='black')
    plt.title('Immune Cell Density')

    plt.subplot(2, 2, 4)
    plt.imshow(oxygen_levels, cmap='Greens', interpolation='nearest', origin='lower')
    plt.title('Oxygen Levels')
    plt.colorbar(label='Oxygen Levels')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# Setting up PySCeS parameters
params = {
    'k_tumor_capacity': 1.0,
    'k_oxygen_consumption': 0.01,
    'k_tumor_immunity': 0.1,
    'k_immune_proliferation': 0.005,
    'k_immune_stimulation': 0.1,
    'k_immune_death': 0.002,
    'k_tumor_growth': 0.02,
    'k_tumor_death': 0.001,
    'k_cell_migration': 0.005,
    'k_oxygen_diffusion': 0.01,
    'k_tumor_oxygen_consumption': 0.005,
}

# Define the ODE system using PySCeS with logistic growth and Michaelis-Menten kinetics
ode_system = """
tumor_cells := logistic_growth(t, tumor_cells, k_tumor_growth, k_tumor_capacity) - k_tumor_death * tumor_cells - k_oxygen_consumption * tumor_cells * (1 + k_tumor_immunity * immune_cells)
immune_cells := k_immune_proliferation * immune_cells * (1 + k_immune_stimulation * tumor_cells) - k_immune_death * immune_cells
"""

# Time space
t = np.linspace(0, 100, 100)

# Particle Swarm Optimization (PSO) for parameter fitting
def objective_function(params, *args):
    ode_model.set_optimizables(params)
    results = odeint(ode_model.merlin, [initial_tumor_density, initial_immune_density], t)
    tumor_sizes, _ = results.T
    target_data = args[0]
    return np.sum((tumor_sizes - target_data)**2)

# Example target data (replace with actual experimental data)
target_data = tcga_data['tumor_sizes']

#  Particle Swarm Optimizatio optimization
initial_params = {
    'k_tumor_growth': 0.02,
    'k_tumor_capacity': 1.0,
    'k_tumor_death': 0.001,
    'k_oxygen_consumption': 0.01,
    'k_tumor_immunity': 0.1,
    'k_immune_proliferation': 0.005,
    'k_immune_stimulation': 0.1,
    'k_immune_death': 0.002,
    'k_cell_migration': 0.005,
    'k_oxygen_diffusion': 0.01,
    'k_tumor_oxygen_consumption': 0.005,
}

result = minimize(objective_function, initial_params, args=(target_data,), method='PSO')

# Updating the parameters with optimized values
optimal_params = result.x
params.update(dict(zip(initial_params.keys(), optimal_params)))
ode_model.set_optimizables(params)

# Main Simulation Loop with cellular and microenvironment processes
for timestep in range(len(t)):
    cell_growth(tumor_cells, params)
    cell_death(tumor_cells, params)
    immune_proliferation(immune_cells, tumor_cells, params)
    immune_death(immune_cells, params)
    cell_migration(tumor_cells, immune_cells, params)
    update_oxygen_levels(oxygen_levels, tumor_cells, params)
    update_immune_dynamics(immune_cells, tumor_cells, params)
    # Visualizing the model state at specific time points
    if timestep % 10 == 0:
        visualize_model_state(tumor_cells, immune_cells, oxygen_levels, title=f'Time Step: {timestep}')

# Visualize the final state
visualize_model_state(tumor_cells, immune_cells, oxygen_levels, title='Final State of the Spatial Model')
