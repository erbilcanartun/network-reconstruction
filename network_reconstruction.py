import numpy as np
import networkx as nx
import scipy as sp
from scipy import sparse
from sklearn.linear_model import Lasso
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt

def generate_scale_free_network(n, eta=0.2):
    """
    Generate a directed scale-free network with n nodes.
    
    Parameters:
    -----------
    n : int
        Number of nodes
    eta : float
        Range for random weights [1-eta, 1+eta]
    
    Returns:
    --------
    G : networkx.DiGraph
        The generated network
    A : numpy.ndarray
        Adjacency matrix
    k_in : numpy.ndarray
        In-degree of each node
    L : numpy.ndarray
        Laplacian matrix
    delta : float
        Maximum in-degree
    """
    # Generate a directed scale-free graph
    G = nx.scale_free_graph(n, alpha=0.2, beta=0.3, gamma=0.5)
    
    # Remove self-loops
    G.remove_edges_from(nx.selfloop_edges(G))
    
    # Remove multi-edges (edges with attribute index > 0)
    removed_list = []
    for edge in G.edges:
        if edge[2] != 0:
            removed_list.append(edge)
    G.remove_edges_from(removed_list)
    
    # Assign random weights to edges
    for edge in range(G.number_of_edges()):
        list(G.edges(data=True))[edge][2]["weight"] = np.random.uniform(1.-eta, 1.+eta)
    
    # Get adjacency matrix (transposed as per the paper)
    # Updated to use the current NetworkX API
    A = nx.to_numpy_array(G).T
    
    # Calculate in-degree for each node
    k_in = np.zeros(G.number_of_nodes())
    for node in range(G.number_of_nodes()):
        if G.in_degree(node) != 0:
            k_in[node] = sum(list(G.in_edges(node, data=True))[i][2]["weight"] for i in range(G.in_degree(node)))
    
    # Create weighted Laplacian matrix
    L = np.diag(k_in) - A
    
    # Maximum in-degree
    delta = np.max(k_in)
    
    return G, A, k_in, L, delta

def undirected_network(n):
    """
    Generate an undirected Barabási-Albert scale-free network.
    
    Parameters:
    -----------
    n : int
        Number of nodes
    
    Returns:
    --------
    G : networkx.Graph
        The generated network
    A : numpy.ndarray
        Adjacency matrix
    L : numpy.ndarray
        Laplacian matrix
    degrees : numpy.ndarray
        Degree of each node
    delta : float
        Maximum degree
    """
    G = nx.barabasi_albert_graph(n, 2)
    
    # Updated to use the current NetworkX API
    A = nx.to_numpy_array(G)
    
    # Calculate degrees
    degrees = np.array([d for n, d in G.degree()])
    
    # Create Laplacian matrix
    L = np.diag(degrees) - A
    
    # Maximum degree
    delta = np.max(degrees)
    
    return G, A, L, degrees, delta

def rulkov_map(x, beta, mu, sigma):
    """
    Rulkov map dynamics for a single iteration.
    
    Parameters:
    -----------
    x : numpy.ndarray
        Current state [u, v]
    beta, mu, sigma : float
        Model parameters
    
    Returns:
    --------
    numpy.ndarray
        Next state [u', v']
    """
    u, v = x
    u_next = beta / (1 + u**2) + v
    v_next = v - mu * u - sigma
    return np.array([u_next, v_next])

def generate_data(n, m, transient, time_steps, beta, mu, sigma, C, L, delta, h, gamma, x_noise):
    """
    Generate time series data for a network of coupled chaotic maps.
    
    Parameters:
    -----------
    n : int
        Number of nodes
    m : int
        Dimension of node state (e.g., 2 for Rulkov map)
    transient : int
        Number of steps to discard as transient
    time_steps : int
        Number of time steps to simulate
    beta, mu, sigma : float
        Parameters for Rulkov map
    C : float
        Coupling strength
    L : numpy.ndarray
        Laplacian matrix
    delta : float
        Maximum in-degree
    h : numpy.ndarray
        Coupling function (matrix)
    gamma : float
        Noise intensity
    x_noise : numpy.ndarray
        Noise mask
    
    Returns:
    --------
    numpy.ndarray
        Time series data of shape (n, m, time_steps)
    """
    def rulkov_network_dynamics(x):
        """Network dynamics for one step."""
        x_reshaped = x.reshape(n, m).T
        # Local dynamics (Rulkov map)
        f_x = np.zeros_like(x_reshaped)
        f_x[0] = beta / (1 + x_reshaped[0]**2) + x_reshaped[1]
        f_x[1] = x_reshaped[1] - mu * x_reshaped[0] - sigma
        
        # Reshape back
        f_x = f_x.T.flatten()
        
        # Add coupling and noise
        return f_x - (C/delta) * sparse.kron(L, h).dot(x) + x_noise * gamma * np.random.uniform(-1, 1, n*m)
    
    # Initialize states
    x = np.zeros((n*m, time_steps))
    x_init = np.random.uniform(0.0, 1.0, n*m)
    
    # Run transient for isolated dynamics first
    x0 = x_init.copy()
    for _ in range(transient // 2):
        # Apply local dynamics only
        x0_reshaped = x0.reshape(n, m).T
        f_x = np.zeros_like(x0_reshaped)
        f_x[0] = beta / (1 + x0_reshaped[0]**2) + x0_reshaped[1]
        f_x[1] = x0_reshaped[1] - mu * x0_reshaped[0] - sigma
        x0 = f_x.T.flatten()
    
    # Run transient for coupled dynamics
    for _ in range(transient // 2):
        x0 = rulkov_network_dynamics(x0)
    
    # Now generate the actual time series
    x[:, 0] = x0
    for t in range(time_steps - 1):
        x[:, t+1] = rulkov_network_dynamics(x[:, t])
    
    # Reshape to (n, m, time_steps) for easier handling
    return x.reshape(n, m, time_steps)

def split_data(n, x):
    """
    Split data into input-output pairs for regression.
    
    Parameters:
    -----------
    n : int
        Number of nodes
    x : numpy.ndarray
        Time series data of shape (n, m, time_steps)
    
    Returns:
    --------
    X : numpy.ndarray
        Input data (states at time t)
    dx : numpy.ndarray
        Output data (states at time t+1)
    """
    X = []
    dx = []
    
    for i in range(n):
        X.append(x[i, :, :-1])
        dx.append(x[i, :, 1:])
    
    return np.array(X), np.array(dx)

def create_library_functions():
    """
    Create library of basis functions for sparse regression.
    
    Returns:
    --------
    functions : list
        List of functions
    function_names : list
        List of function names
    """
    # Define library functions
    library_functions = [
        lambda x: np.sin(x),
        lambda x: np.cos(x),
        lambda x: 1/(1-x),
        lambda x: 1/(1-x**2),
        lambda x: 1/((1-x)**2),
        lambda x: 1/(1+x),
        lambda x: 1/(1+x**2),
        lambda x: 1/((1+x)**2),
        lambda x: 1/x
    ]
    
    # Define corresponding function names for display
    library_function_names = [
        lambda x: 'sin(' + x + ')',
        lambda x: 'cos(' + x + ')',
        lambda x: '1/1-' + x,
        lambda x: '1/1-' + x + '^2',
        lambda x: '1/(1-' + x + ')^2',
        lambda x: '1/1+' + x,
        lambda x: '1/1+' + x + '^2',
        lambda x: '1/(1+' + x + ')^2',
        lambda x: '1/' + x
    ]
    
    return library_functions, library_function_names

def build_feature_matrix(X, library_functions, library_function_names):
    """
    Build feature matrix using the library of basis functions.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Input data
    library_functions : list
        List of basis functions
    
    Returns:
    --------
    Theta : numpy.ndarray
        Feature matrix
    """
    n_samples = X.shape[0]
    # Start with constant term and original features
    features = [np.ones(n_samples), X[:, 0], X[:, 1]]
    feature_names = ['1', 'u', 'v']
    
    # Add polynomial terms
    features.append(X[:, 0]**2)
    feature_names.append('u^2')
    
    features.append(X[:, 0] * X[:, 1])
    feature_names.append('u*v')
    
    features.append(X[:, 1]**2)
    feature_names.append('v^2')
    
    # Add library functions
    for func, func_name in zip(library_functions, 
                             [f_name('u') for f_name in library_function_names]):
        features.append(func(X[:, 0]))
        feature_names.append(func_name)
    
    for func, func_name in zip(library_functions, 
                             [f_name('v') for f_name in library_function_names]):
        features.append(func(X[:, 1]))
        feature_names.append(func_name)
    
    Theta = np.column_stack(features)
    return Theta, feature_names

def sparse_regression(X, y, alpha=0.001):
    """
    Perform sparse regression using LASSO.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Feature matrix
    y : numpy.ndarray
        Output data
    alpha : float
        Regularization parameter
    
    Returns:
    --------
    numpy.ndarray
        Coefficient vector
    """
    model = Lasso(alpha=alpha, fit_intercept=False, max_iter=10000)
    model.fit(X, y)
    return model.coef_

def predict_models(n, X, dx, library_functions, library_function_names):
    """
    Perform sparse regression to identify models for each node.
    
    Parameters:
    -----------
    n : int
        Number of nodes
    X : numpy.ndarray
        Input data
    dx : numpy.ndarray
        Output data
    library_functions : list
        List of basis functions
    library_function_names : list
        List of function names
    
    Returns:
    --------
    coeff : numpy.ndarray
        Coefficients for each node's model
    """
    coeff = []
    
    for i in range(n):
        # For each component of the state vector
        node_coeff = []
        
        for j in range(X.shape[1]):  # Usually 2 for Rulkov (u and v)
            # Prepare data
            input_data = X[i].T  # Time series for this node
            output_data = dx[i, j, :]  # j-th component of output
            
            # Build feature matrix
            Theta, _ = build_feature_matrix(input_data, library_functions, library_function_names)
            
            # Perform sparse regression
            coefficients = sparse_regression(Theta, output_data)
            node_coeff.append(coefficients)
        
        coeff.append(node_coeff)
    
    return np.array(coeff)

def analyze_similarity(n, x, coeff, k_in):
    """
    Analyze similarity between predicted models to classify nodes.
    
    Parameters:
    -----------
    n : int
        Number of nodes
    x : numpy.ndarray
        Time series data
    coeff : numpy.ndarray
        Model coefficients
    k_in : numpy.ndarray
        In-degree of each node
    
    Returns:
    --------
    corr_matrix_gt : numpy.ndarray
        Correlation matrix of ground truth time series
    distance_matrix : numpy.ndarray
        Distance matrix between predicted models
    s : numpy.ndarray
        Sum of distances for each node
    s_gt : numpy.ndarray
        Sum of correlations for each node
    hub_id : int
        Predicted hub node
    ld_id : int
        Predicted low-degree node
    """
    # Compute correlation between ground-truth time series (u-component)
    corr_matrix_gt = np.corrcoef(x[:, 0, :], x[:, 0, :])[:n, :n]
    
    # Compute distance between predicted models
    # Use normalized Euclidean distance weighted by variance
    variances = np.var(coeff[:, 0, :], axis=0)
    variances[variances == 0] = 1  # Avoid division by zero
    
    # Compute pairwise distances with variance normalization
    distance_matrix = pairwise_distances(coeff[:, 0, :], metric='seuclidean', V=variances)
    
    # Sum distances and correlations for each node
    s = np.sum(distance_matrix, axis=1)
    s_gt = np.sum(np.abs(corr_matrix_gt), axis=1)
    
    # Identify hub and low-degree node
    hub_id = np.argmax(s)
    ld_id = np.argmin(s)
    
    print(f'Predicted low-degree node: {ld_id} has {k_in[ld_id]:.2f} in connection(s).')
    print(f'Predicted hub: {hub_id} has {k_in[hub_id]:.2f} in connection(s).')
    print(f'The real hub is: {np.argmax(k_in)} with {np.max(k_in):.2f} in connection(s).')
    
    return corr_matrix_gt, distance_matrix, s, s_gt, hub_id, ld_id

def extract_local_dynamics(n, m, time_steps, x, beta, mu, sigma):
    """
    Extract the isolated dynamics of each node.
    
    Parameters:
    -----------
    n : int
        Number of nodes
    m : int
        Dimension of node state
    time_steps : int
        Number of time steps
    x : numpy.ndarray
        Time series data
    beta, mu, sigma : float
        Parameters for Rulkov map
    
    Returns:
    --------
    Fx : numpy.ndarray
        Isolated dynamics
    """
    # Reshape for easier processing
    y = x.reshape(n*m, time_steps)
    
    # Compute isolated dynamics for each time step
    F_x = np.zeros_like(y)
    
    for i in range(time_steps):
        # Reshape current state
        current = y[:, i].reshape(n, m).T
        
        # Apply Rulkov map dynamics
        next_u = beta / (1 + current[0]**2) + current[1]
        next_v = current[1] - mu * current[0] - sigma
        
        # Store result
        F_x[::2, i] = next_u  # u components
        F_x[1::2, i] = next_v  # v components
    
    # Reshape back to original format
    return F_x.reshape(n, m, time_steps)

def extract_coupling_effect(dx, hub_id, Fx):
    """
    Extract coupling effect by subtracting isolated dynamics.
    
    Parameters:
    -----------
    dx : numpy.ndarray
        Output data
    hub_id : int
        Predicted hub node ID
    Fx : numpy.ndarray
        Isolated dynamics
    
    Returns:
    --------
    Y_hub : numpy.ndarray
        Coupling effect on hub
    Y : numpy.ndarray
        Coupling effect on all nodes
    """
    # Coupling effect on hub
    Y_hub = dx[hub_id, :, :] - Fx[hub_id, :, :-1]
    
    # Coupling effect on all nodes
    Y = dx - Fx[:, :, :-1]
    
    return Y_hub, Y

def reconstruct_network(n, m, time_steps, X, Y, alpha=0.001):
    """
    Reconstruct network connectivity using sparse regression.
    
    Parameters:
    -----------
    n : int
        Number of nodes
    m : int
        Dimension of node state
    time_steps : int
        Number of time steps
    X : numpy.ndarray
        Input data
    Y : numpy.ndarray
        Coupling effect
    alpha : float
        Regularization parameter for LASSO
    
    Returns:
    --------
    L_predicted : numpy.ndarray
        Predicted Laplacian matrix
    """
    # Reshape data for regression
    X_flat = X.reshape(n*m, time_steps-1).T
    Y_flat = Y.reshape(n*m, time_steps-1).T
    
    # Perform sparse regression using LASSO
    model = Lasso(alpha=alpha, fit_intercept=False, max_iter=10000)
    model.fit(X_flat, Y_flat)
    
    # Extract coefficients
    L_predicted = model.coef_
    
    # Reshape to get Laplacian matrix (account for Kronecker product structure)
    # We need to extract one value per m×m block
    step = m
    L_reconstructed = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            # Calculate the average of the block
            block = L_predicted[i*step:(i+1)*step, j*step:(j+1)*step]
            L_reconstructed[i, j] = block[0, 0]  # Just take (0,0) entry as per paper
    
    return L_reconstructed

def evaluate_reconstruction(L_true, L_predicted, tolerance=1e-4):
    """
    Evaluate reconstruction performance using FNR and FPR.
    
    Parameters:
    -----------
    L_true : numpy.ndarray
        True Laplacian matrix
    L_predicted : numpy.ndarray
        Predicted Laplacian matrix
    tolerance : float
        Threshold for considering an entry as non-zero
    
    Returns:
    --------
    FNR : float
        False Negative Rate
    FPR : float
        False Positive Rate
    """
    # Identify positives and negatives in true matrix
    positives = (np.abs(L_true) > tolerance) & ~np.eye(L_true.shape[0], dtype=bool)
    negatives = (np.abs(L_true) <= tolerance) | np.eye(L_true.shape[0], dtype=bool)
    
    # Identify false negatives and false positives
    false_negatives = positives & (np.abs(L_predicted) <= tolerance)
    false_positives = negatives & (np.abs(L_predicted) > tolerance)
    
    # Calculate FNR and FPR
    FNR = np.sum(false_negatives) / np.sum(positives) if np.sum(positives) > 0 else 0
    FPR = np.sum(false_positives) / np.sum(negatives) if np.sum(negatives) > 0 else 0
    
    return FNR, FPR

def plot_similarity_histogram(s, s_gt, hub_id, ld_id):
    """
    Plot histogram of model similarities to visualize node classification.
    
    Parameters:
    -----------
    s : numpy.ndarray
        Sum of distances for each node
    s_gt : numpy.ndarray
        Sum of correlations for each node
    hub_id : int
        Predicted hub node ID
    ld_id : int
        Predicted low-degree node ID
    """
    plt.figure(figsize=(12, 5))
    
    # Plot distance histogram
    plt.subplot(1, 2, 1)
    hist, bins = np.histogram(s, bins=20)
    plt.bar(bins[:-1], hist, width=np.diff(bins), alpha=0.7)
    plt.axvline(s[hub_id], color='red', linestyle='--', label=f'Hub (Node {hub_id})')
    plt.axvline(s[ld_id], color='green', linestyle='--', label=f'Low-degree (Node {ld_id})')
    plt.xlabel('Sum of distances')
    plt.ylabel('Frequency')
    plt.title('Model Similarity Histogram')
    plt.legend()
    
    # Plot correlation histogram
    plt.subplot(1, 2, 2)
    hist, bins = np.histogram(s_gt, bins=20)
    plt.bar(bins[:-1], hist, width=np.diff(bins), alpha=0.7)
    plt.axvline(s_gt[hub_id], color='red', linestyle='--', label=f'Hub (Node {hub_id})')
    plt.axvline(s_gt[ld_id], color='green', linestyle='--', label=f'Low-degree (Node {ld_id})')
    plt.xlabel('Sum of correlations')
    plt.ylabel('Frequency')
    plt.title('Time Series Correlation Histogram')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_hub_low_degree_data(x, hub_id, ld_id):
    """
    Plot time series and return maps for hub and low-degree nodes.
    
    Parameters:
    -----------
    x : numpy.ndarray
        Time series data
    hub_id : int
        Hub node ID
    ld_id : int
        Low-degree node ID
    """
    plt.figure(figsize=(15, 10))
    
    # Time series
    plt.subplot(2, 2, 1)
    plt.plot(x[hub_id, 0, :100], label=f'Hub (Node {hub_id})')
    plt.plot(x[ld_id, 0, :100], label=f'Low-degree (Node {ld_id})', alpha=0.7)
    plt.xlabel('Time step')
    plt.ylabel('u')
    plt.title('Time Series (u component)')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(x[hub_id, 1, :100], label=f'Hub (Node {hub_id})')
    plt.plot(x[ld_id, 1, :100], label=f'Low-degree (Node {ld_id})', alpha=0.7)
    plt.xlabel('Time step')
    plt.ylabel('v')
    plt.title('Time Series (v component)')
    plt.legend()
    
    # Return maps
    plt.subplot(2, 2, 3)
    plt.scatter(x[hub_id, 0, :-1], x[hub_id, 0, 1:], alpha=0.5, label=f'Hub (Node {hub_id})')
    plt.xlabel('u(t)')
    plt.ylabel('u(t+1)')
    plt.title('Return Map for Hub Node')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.scatter(x[ld_id, 0, :-1], x[ld_id, 0, 1:], alpha=0.5, label=f'Low-degree (Node {ld_id})')
    plt.xlabel('u(t)')
    plt.ylabel('u(t+1)')
    plt.title('Return Map for Low-degree Node')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_coupling_function(Y_hub, x, hub_id):
    """
    Plot the coupling function based on hub data.
    
    Parameters:
    -----------
    Y_hub : numpy.ndarray
        Coupling effect on hub
    x : numpy.ndarray
        Time series data
    hub_id : int
        Hub node ID
    """
    plt.figure(figsize=(8, 6))
    
    # Get u values and corresponding coupling effect
    u_values = x[hub_id, 0, :-1]
    coupling_effect = Y_hub[0, :]  # First component (u)
    
    # Sort for better visualization
    sort_idx = np.argsort(u_values)
    u_sorted = u_values[sort_idx]
    effect_sorted = coupling_effect[sort_idx]
    
    # Plot raw data
    plt.scatter(u_values, coupling_effect, alpha=0.3, color='blue', label='Raw data')
    
    # Fit a line to identify the coupling function
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    u_shaped = u_sorted.reshape(-1, 1)
    model.fit(u_shaped, effect_sorted)
    
    # Plot fitted line
    u_range = np.linspace(min(u_values), max(u_values), 100).reshape(-1, 1)
    predicted = model.predict(u_range)
    plt.plot(u_range, predicted, 'r-', linewidth=2, 
             label=f'Fitted: {model.coef_[0]:.2f}*u + {model.intercept_:.2f}')
    
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    plt.xlabel('u_hub(t)')
    plt.ylabel('Coupling effect on u_hub')
    plt.title('Effective Coupling Function V(u)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
    
    print(f"Estimated coupling function: V(u) ≈ {model.coef_[0]:.4f}*u + {model.intercept_:.4f}")
    print(f"According to the reduction theorem, this suggests H(u) ≈ {model.coef_[0]:.4f}*u")
    print(f"with a linear shift (integration constant) C ≈ {model.intercept_:.4f}")

def plot_reconstruction_results(L_true, L_predicted):
    """
    Visualize the original vs. reconstructed Laplacian matrices.
    
    Parameters:
    -----------
    L_true : numpy.ndarray
        True Laplacian matrix
    L_predicted : numpy.ndarray
        Predicted Laplacian matrix
    """
    plt.figure(figsize=(15, 5))
    
    # Original Laplacian
    plt.subplot(1, 3, 1)
    plt.imshow(L_true, cmap='viridis')
    plt.colorbar()
    plt.title('Original Laplacian Matrix')
    
    # Reconstructed Laplacian
    plt.subplot(1, 3, 2)
    plt.imshow(L_predicted, cmap='viridis')
    plt.colorbar()
    plt.title('Reconstructed Laplacian Matrix')
    
    # Difference
    plt.subplot(1, 3, 3)
    diff = L_true - L_predicted
    plt.imshow(diff, cmap='RdBu', vmin=-np.max(np.abs(diff)), vmax=np.max(np.abs(diff)))
    plt.colorbar()
    plt.title('Difference (Error)')
    
    plt.tight_layout()
    plt.show()