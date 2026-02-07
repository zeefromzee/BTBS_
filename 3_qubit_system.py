import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix
from qutip import *
from qiskit.visualization import plot_bloch_multivector
import copy
from qiskit.quantum_info import partial_trace


class BlochTreeNode:
    """Represents a single node in the BTBS tree"""
    def __init__(self, coefficients, level=0, binary_path=""):
        self.coefficients = np.array(coefficients)
        self.level = level
        self.binary_path = binary_path
        self.theta = None
        self.phi = None
        self.left_child = None
        self.right_child = None
        self.is_leaf = len(coefficients) == 2

def coefficients_to_bloch_angles(c0, c1):
    """
    Convert two complex coefficients to Bloch angles (θ, φ)
    
    Input: c0, c1 (complex numbers)
    Output: (theta, phi) where theta ∈ [0, π], phi ∈ [0, 2π]
    """
    # Extract magnitudes and phases
    r0 = np.abs(c0)
    r1 = np.abs(c1)
    
    # Normalize
    norm = np.sqrt(r0**2 + r1**2)
    if norm < 1e-10:
        return 0, 0  # Handle zero case
    
    r0_norm = r0 / norm
    r1_norm = r1 / norm
    
    # Calculate theta: θ = 2 * arctan(r1/r0)
    theta = 2 * np.arctan2(r1_norm, r0_norm)
    
    # Calculate phi from phases
    phase0 = np.angle(c0)
    phase1 = np.angle(c1)
    phi = phase1 - phase0
    phi = phi % (2 * np.pi)  # Normalize to [0, 2π]
    
    return theta, phi

def process_state_recursive(coefficients, level=0, binary_path=""):
    """
    Recursively process quantum state coefficients into BTBS tree
    
    Signature: process_state(coefficients, level, binary_path)
    Base case: coefficients has length 2 → compute Bloch angles
    Recursive case: split in half, process children
    """
    node = BlochTreeNode(coefficients, level, binary_path)
    
    # BASE CASE: leaf node (2 coefficients)
    if len(coefficients) == 2:
        c0, c1 = coefficients[0], coefficients[1]
        node.theta, node.phi = coefficients_to_bloch_angles(c0, c1)
        node.is_leaf = True
        return node
    
    # RECURSIVE CASE: split in half
    mid = len(coefficients) // 2
    left_coeffs = coefficients[:mid]
    right_coeffs = coefficients[mid:]
    
    # Compute combined amplitudes for current level
    left_amplitude = np.sqrt(np.sum(np.abs(left_coeffs)**2))
    right_amplitude = np.sqrt(np.sum(np.abs(right_coeffs)**2))
    
    # Apply helper function to get current node's angles
    node.theta, node.phi = coefficients_to_bloch_angles(left_amplitude, right_amplitude)
    
    # Recursively process children
    node.left_child = process_state_recursive(left_coeffs, level+1, binary_path+"0")
    node.right_child = process_state_recursive(right_coeffs, level+1, binary_path+"1")
    
    return node

def tree_to_flat_list(node, nodes_list=None):
    """Convert tree structure to flat list for easier visualization"""
    if nodes_list is None:
        nodes_list = []
    
    if node is None:
        return nodes_list
    
    nodes_list.append({
        'path': node.binary_path,
        'level': node.level,
        'theta': node.theta,
        'phi': node.phi,
        'is_leaf': node.is_leaf
    })
    
    if node.left_child:
        tree_to_flat_list(node.left_child, nodes_list)
    if node.right_child:
        tree_to_flat_list(node.right_child, nodes_list)
    
    return nodes_list

def calculate_entanglement_entropy(statevector):
    """Calculate von Neumann entropy to measure entanglement"""
    
    # Get density matrix for the full system
    rho = DensityMatrix(statevector)
    # Partial trace over qubits 1 and 2 to keep qubit 0
    rho_reduced = partial_trace(rho, [1, 2])
    # partial_trace may return a DensityMatrix-like object; access its data
    data = getattr(rho_reduced, 'data', rho_reduced)
    # Calculate eigenvalues
    eigenvalues = np.linalg.eigvalsh(data)
    eigenvalues = eigenvalues[eigenvalues > 1e-12]  # Remove near-zero values
    # Von Neumann entropy: S = -Σ λ_i log2(λ_i)
    entropy = -np.sum(eigenvalues * np.log2(eigenvalues + 1e-12))
    return entropy

# ============================================================================
# STATE GENERATORS
# ============================================================================

def generate_product_state_3qubit():
    """Generate a random 3-qubit product state |abc⟩"""
    # Create 3 independent single-qubit states
    qc = QuantumCircuit(3)
    
    for qubit in range(3):
        theta = np.random.uniform(0, np.pi)
        phi = np.random.uniform(0, 2*np.pi)
        qc.ry(theta, qubit)
        qc.rz(phi, qubit)
    
    sv = Statevector(qc)
    return sv.data

def generate_entangled_state_3qubit(entanglement_type="random"):
    """Generate various 3-qubit entangled states"""
    qc = QuantumCircuit(3)
    
    if entanglement_type == "ghz":
        # GHZ state: (|000⟩ + |111⟩) / √2
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        
    elif entanglement_type == "w":
        # W state: (|001⟩ + |010⟩ + |100⟩) / √3
        qc.h(0)
        qc.ry(np.arccos(1/np.sqrt(3)), 1)
        qc.cx(1, 2)
        qc.cx(0, 1)
        
    elif entanglement_type == "partial":
        # Partially entangled: qubits 0-1 entangled, 2 separable
        qc.h(0)
        qc.cx(0, 1)
        theta = np.random.uniform(0, np.pi)
        phi = np.random.uniform(0, 2*np.pi)
        qc.ry(theta, 2)
        qc.rz(phi, 2)
        
    else:  # random
        # Random entangling circuit            
        for qubit in range(3):
            qc.ry(np.random.uniform(0, np.pi), qubit)
        
        # Apply random CNOTs
        for _ in range(np.random.randint(1, 4)):
            control = np.random.randint(0, 3)
            target = np.random.randint(0, 3)
            if control != target:
                qc.cx(control, target)
        
        for qubit in range(3):
            qc.rz(np.random.uniform(0, 2*np.pi), qubit)
    
    sv = Statevector(qc)
    return sv.data

def generate_labeled_dataset_3qubit(num_product=100, num_entangled=100):
    """Generate labeled dataset of 3-qubit states"""
    dataset = []
    
    print(f"Generating {num_product} product states...")
    for i in range(num_product):
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{num_product}")
        
        coeffs = generate_product_state_3qubit()
        sv = Statevector(coeffs)
        entropy = calculate_entanglement_entropy(sv)
        
        dataset.append({
            'statevector': coeffs,
            'label': 0,  # 0 = separable/product
            'entropy': entropy,
            'type': 'product'
        })
    
    print(f"Generating {num_entangled} entangled states...")
    entanglement_types = ["ghz", "w", "partial", "random"]
    
    for i in range(num_entangled):
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{num_entangled}")
        
        ent_type = entanglement_types[i % len(entanglement_types)]
        coeffs = generate_entangled_state_3qubit(ent_type)
        sv = Statevector(coeffs)
        entropy = calculate_entanglement_entropy(sv)
        
        dataset.append({
            'statevector': coeffs,
            'label': 1,  # 1 = entangled
            'entropy': entropy,
            'type': ent_type
        })
    
    return dataset

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_btbs_tree(statevector_coeffs, title="3-Qubit BTBS Tree"):
    """
    Visualize the Binary Tree Bloch Sphere (BTBS) for 3-qubit state
    
    Creates a hierarchical visualization with:
    - Root sphere at top (qubit 0)
    - Left/Right children (qubit 1 conditional states)
    - 4 Leaf spheres (qubit 2 conditional states)
    """
    # Build tree
    root = process_state_recursive(statevector_coeffs)
    nodes = tree_to_flat_list(root)
    
    # Create figure with subplots for tree layout
    fig = plt.figure(figsize=(16, 10))
    
    # Define positions for 7 spheres in tree layout
    positions = {
        '': (0.5, 0.85),      # Root
        '0': (0.25, 0.50),    # Left
        '1': (0.75, 0.50),    # Right
        '00': (0.1, 0.15),    # Left-Left
        '01': (0.35, 0.15),   # Left-Right
        '10': (0.65, 0.15),   # Right-Left
        '11': (0.9, 0.15),    # Right-Right
    }
    
    # Color by depth
    depth_colors = {0: 'red', 1: 'blue', 2: 'green'}
    
    for node_info in nodes:
        path = node_info['path']
        level = node_info['level']
        theta = node_info['theta']
        phi = node_info['phi']
        
        if path not in positions:
            continue
        
        x, y = positions[path]
        
        # Create subplot for this sphere
        ax = fig.add_axes([x-0.08, y-0.12, 0.16, 0.24], projection='3d')
        
        # Draw sphere surface
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 20)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
        
        ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='gray')
        
        # Draw point on sphere
        x_point = np.sin(theta) * np.cos(phi)
        y_point = np.sin(theta) * np.sin(phi)
        z_point = np.cos(theta)
        
        color = depth_colors.get(level, 'purple')
        ax.scatter([x_point], [y_point], [z_point], c=color, s=100, marker='o')
        
        # Labels
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        ax.set_zlim([-1.5, 1.5])
        ax.set_xlabel('X', fontsize=8)
        ax.set_ylabel('Y', fontsize=8)
        ax.set_zlabel('Z', fontsize=8)
        
        # Title for this sphere
        label = f"Level {level}\nθ={theta:.2f}\nφ={phi:.2f}"
        ax.set_title(label, fontsize=8)
        
        # Hide tick labels for clarity
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    # tight_layout is not compatible with 3D axes; adjust manually
    fig.subplots_adjust(top=0.90, left=0.05, right=0.95)
    return fig

def main():
    # Test 1: GHZ state
    print("\n[TEST 1] GHZ State: (|000⟩ + |111⟩) / √2")
    print("-" * 70)
    qc_ghz = QuantumCircuit(3)
    qc_ghz.h(0)
    qc_ghz.cx(0, 1)
    qc_ghz.cx(1, 2)
    sv_ghz = Statevector(qc_ghz)
    entropy_ghz = calculate_entanglement_entropy(sv_ghz)
    
    print(f"Statevector: {sv_ghz.data}")
    print(f"Entanglement Entropy: {entropy_ghz:.4f}")
    print("Visualizing GHZ state BTBS tree...")
    plot_btbs_tree(sv_ghz.data, "GHZ State: (|000⟩ + |111⟩) / √2")
    plt.show()
    
    # Test 2: Product state
    print("\n[TEST 2] Random Product State")
    sv_product = Statevector(generate_product_state_3qubit())
    entropy_product = calculate_entanglement_entropy(sv_product)
    
    print(f"Entanglement Entropy: {entropy_product:.4f}")
    print("Visualizing product state BTBS tree...")
    plot_btbs_tree(sv_product.data, "Random Product State")
    plt.show()
    
    # Test 3: Small dataset
    print("\n[TEST 3] Generating Small Dataset (10 product + 10 entangled)")
    
    dataset = generate_labeled_dataset_3qubit(num_product=10, num_entangled=10)
    
    # Statistics
    product_entropies = [d['entropy'] for d in dataset if d['label'] == 0]
    entangled_entropies = [d['entropy'] for d in dataset if d['label'] == 1]
    
    print(f"\nProduct States:")
    print(f"  Count: {len(product_entropies)}")
    print(f"  Entropy: mean={np.mean(product_entropies):.4f}, std={np.std(product_entropies):.4f}")
    print(f"  Range: [{np.min(product_entropies):.4f}, {np.max(product_entropies):.4f}]")
    
    print(f"\nEntangled States:")
    print(f"  Count: {len(entangled_entropies)}")
    print(f"  Entropy: mean={np.mean(entangled_entropies):.4f}, std={np.std(entangled_entropies):.4f}")
    print(f"  Range: [{np.min(entangled_entropies):.4f}, {np.max(entangled_entropies):.4f}]")
    
    # Entropy histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(product_entropies, bins=15, alpha=0.6, label='Product', color='blue')
    ax.hist(entangled_entropies, bins=15, alpha=0.6, label='Entangled', color='red')
    ax.set_xlabel('Entanglement Entropy')
    ax.set_ylabel('Frequency')
    ax.set_title('Entropy Distribution: Product vs Entangled States')
    ax.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
