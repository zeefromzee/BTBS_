from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_multivector
from qiskit_ibm_runtime import Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.visualization import plot_histogram
import cmath
import math
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt

%matplotlib inline

# Function to generate normalized random amplitudes
def generate_random_amplitudes():
    """Generate normalized random amplitudes for quantum coefficients"""
    # Generate random magnitudes in [0, 1]
    r0 = np.random.uniform(0, 1)
    r1 = np.random.uniform(0, 1)
    
    # Normalize so r0² + r1² = 1
    norm_val = np.sqrt(r0**2 + r1**2)
    r0 = r0 / norm_val
    r1 = r1 / norm_val
    
    return r0, r1

# Creating a dataclass for producing random phase and angles inside a well defined range
@dataclass
class BlochAngles:
    theta: float  # The required range for theta is: [0, π]
    phi: float    # The required range for phi is : [0, 2π]
    
    def __post_init__(self): 
        # the required ranges should be validated hence:
        if not (0 <= self.theta <= np.pi):
            raise ValueError(f"Theta must be in [0, π], got {self.theta}")
        if not (0 <= self.phi <= 2*np.pi):
            raise ValueError(f"Phi must be in [0, 2π], got {self.phi}")
    
    # using static method to create random angles across the establsihed range
    @staticmethod
    def random_theta():
        # Generate random Bloch angles
        theta = np.random.uniform(0, np.pi)
        return theta
    
    @staticmethod
    def random_phi():
        # Generate random Bloch angles
        phi = np.random.uniform(0, 2*np.pi)
        return phi

# Convert Bloch angles (θ, φ) to Cartesian coordinates
def cartesian(theta_val, phi_val):
    """Convert Bloch angles to Cartesian coordinates on unit sphere"""
    x = np.sin(theta_val) * np.cos(phi_val)
    y = np.sin(theta_val) * np.sin(phi_val)
    z = np.cos(theta_val)
    return x, y, z

# creating complex coefficients with random magnitudes and phases
def complex_coeff(r0, r1, phi0, phi1):
    """Create complex coefficients from magnitudes and phases"""
    c1 = r0 * cmath.exp(1j * phi0)
    c2 = r1 * cmath.exp(1j * phi1)
    return c1, c2

# Calculate the combined norm
def calculate_norm(r0, r1):
    """Calculate and return normalized amplitudes"""
    n = math.sqrt((r0**2) + (r1**2))
    r0_norm = r0 / n
    r1_norm = r1 / n
    return n, r0_norm, r1_norm

# Calculate phases for the state vectors
def phase(phi1, phi0):
    """Calculate global and local phases"""
    phi_local = phi1 - phi0  # Local phase
    phi_global = phi0  # Global phase
    return phi_global, phi_local

def verify_basis_states():
    """Verify that |0⟩ and |1⟩ map to correct Bloch sphere positions"""
    print("=== Verifying Basis States ===\n")
    
    # |0⟩ state - should be at north pole (0, 0, 1)
    qc_0 = QuantumCircuit(1)
    sv_0 = Statevector(qc_0)
    print(f"|0⟩ state: {sv_0}")
    x0, y0, z0 = cartesian(0, 0)
    print(f"|0⟩ Cartesian coords: x={x0:.4f}, y={y0:.4f}, z={z0:.4f}")
    print(f"Expected: (0, 0, 1)\n")
    print("Bloch Sphere for |0⟩ (North Pole):")
    plot_bloch_multivector(sv_0)
    plt.show()
    
    # |1⟩ state - should be at south pole (0, 0, -1)
    qc_1 = QuantumCircuit(1)
    qc_1.x(0)
    sv_1 = Statevector(qc_1)
    print(f"\n|1⟩ state: {sv_1}")
    x1, y1, z1 = cartesian(np.pi, 0)
    print(f"|1⟩ Cartesian coords: x={x1:.4f}, y={y1:.4f}, z={z1:.4f}")
    print(f"Expected: (0, 0, -1)\n")
    print("Bloch Sphere for |1⟩ (South Pole):")
    plot_bloch_multivector(sv_1)
    plt.show()

def visualize_random_states(num_states=3):
    """Generate and visualize random single-qubit states"""
    print(f"\n=== Visualizing {num_states} Random Single-Qubit States ===\n")
    
    for i in range(num_states):
        # Generate random Bloch angles
        theta_rand = np.random.uniform(0, np.pi)
        phi_rand = np.random.uniform(0, 2*np.pi)
        
        # Create circuit with rotation gates
        qc = QuantumCircuit(1)
        qc.ry(theta_rand, 0)
        qc.rz(phi_rand, 0)
        
        # Get statevector
        sv = Statevector(qc)
        
        # Convert to Cartesian
        x, y, z = cartesian(theta_rand, phi_rand)
        
        print(f"State {i+1}: θ={theta_rand:.4f}, φ={phi_rand:.4f}")
        print(f"Cartesian: x={x:.4f}, y={y:.4f}, z={z:.4f}")
        print(f"Magnitude: {np.sqrt(x**2 + y**2 + z**2):.4f} (should be 1.0)\n")
        
        # Plot
        print(f"Bloch Sphere for State {i+1}:")
        plot_bloch_multivector(sv)
        plt.show()
        print()

def main():
    # Generate random amplitudes
    r0, r1 = generate_random_amplitudes()
    print(f"Generated random amplitudes: r0={r0:.4f}, r1={r1:.4f}")
    print(f"Check: r0² + r1² = {r0**2 + r1**2:.6f} (should be 1.0)\n")
    
    # Generate random phases
    phi0 = BlochAngles.random_phi()
    phi1 = BlochAngles.random_phi()
    # Create complex coefficients
    c1, c2 = complex_coeff(r0, r1, phi0, phi1)
    print(f"Complex coefficients: c1={c1}, c2={c2}\n")
    
    # Calculate phases
    phi_g, phi_l = phase(phi1, phi0)
    print(f"Global phase: {phi_g:.4f}, Local phase: {phi_l:.4f}\n")
    
    # Verify basis states
    verify_basis_states()
    
    # Visualize random states
    visualize_random_states(3)

if __name__ == "__main__":
    main()
