import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hashlib
import re

# Set page configuration
st.set_page_config(
    page_title="MolOptiMVP | AI Molecular Design Demo",
    page_icon="🧪",
    layout="wide"
)

# Title and description
st.title("🧪 MolOptiMVP: Molecular Optimization Demo")
st.markdown("""
This is a **simplified prototype** demonstrating an AI-agentic workflow for molecular design.
It generates molecular variants and uses a predictive model to simulate optimization.

**⚠️ Important Note:** This is for demonstration purposes only and does not predict real-world properties.
The predictions are simulated for educational purposes.
""")

# Sidebar for controls and information
with st.sidebar:
    st.header("⚙️ Configuration")
    num_cycles = st.slider("Optimization Cycles", 1, 5, 2)
    num_variants = st.slider("Variants per Cycle", 5, 20, 10)
    seed_smiles = st.text_input("Seed SMILES", "CCO")
    
    st.divider()
    st.header("ℹ️ Info")
    st.info("""
    **Example SMILES:**
    - Ethanol: `CCO`
    - Benzene: `c1ccccc1`
    - Aspirin: `CC(=O)Oc1ccccc1C(=O)O`
    - Caffeine: `CN1C=NC2=C1C(=O)N(C(=O)N2C)C`
    """)

    st.warning("""
    ⚠️ **Note:** This demo uses synthetic data for prediction.
    It is not a real drug discovery tool.
    This version uses simplified calculations instead of RDKit.
    """)

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = []
if 'best_molecule' not in st.session_state:
    st.session_state.best_molecule = None
if 'best_score' not in st.session_state:
    st.session_state.best_score = -np.inf
if 'cycle_data' not in st.session_state:
    st.session_state.cycle_data = []

# Simplified molecular property calculator (no RDKit needed)
def calculate_mock_properties(smiles):
    """Calculate mock properties based on SMILES string characteristics."""
    # Score based on string length and complexity
    length_score = min(len(smiles) / 20, 1.0)  # Normalize to 0-1
    
    # Score based on presence of complex patterns
    complexity_score = 0
    complexity_score += 0.2 if '(' in smiles else 0
    complexity_score += 0.2 if '=' in smiles else 0
    complexity_score += 0.2 if 'c' in smiles else 0  # aromatic
    complexity_score += 0.2 if 'N' in smiles else 0
    complexity_score += 0.2 if 'O' in smiles else 0
    
    # Add some randomness
    random_factor = np.random.normal(0, 0.1)
    
    # Combine scores
    total_score = length_score + complexity_score + random_factor
    return max(0.1, total_score)  # Ensure positive score

# Molecular variant generator (no RDKit needed)
def generate_molecular_variants(seed_smiles, num_variants=10):
    """Generates simple molecular variants for demonstration."""
    variants = set()
    
    # Simple modifications to SMILES string
    modifications = [
        lambda s: s + "C",  # Add carbon
        lambda s: s + "O",  # Add oxygen
        lambda s: s + "N",  # Add nitrogen
        lambda s: s.replace("C", "CC", 1) if "C" in s else s,  # Double carbon
        lambda s: s.replace("O", "CO", 1) if "O" in s else s,  # Add carbon to oxygen
        lambda s: s + "(=O)" if len(s) > 3 else s,  # Add carbonyl
        lambda s: s[:-1] if len(s) > 2 else s,  # Remove last character
    ]
    
    for _ in range(num_variants * 3):  # Generate extra to account for duplicates
        # Choose a random modification
        mod_fn = np.random.choice(modifications)
        new_smiles = mod_fn(seed_smiles)
        
        # Basic validation
        if (3 <= len(new_smiles) <= 50 and 
            re.match(r'^[A-Za-z0-9\(\)\=\#\[\]cno]+$', new_smiles)):
            variants.add(new_smiles)
        
        if len(variants) >= num_variants:
            break
    
    return list(variants)[:num_variants]

# Display molecule as text (since we can't render without RDKit)
def display_molecule_info(smiles, caption="Molecule"):
    """Displays information about a molecule."""
    st.write(f"**{caption}**")
    st.code(f"SMILES: {smiles}")
    st.write(f"Length: {len(smiles)} characters")
    st.write(f"Complexity score: {calculate_mock_properties(smiles):.3f}")
    st.divider()

# Main optimization function
def run_optimization():
    """Runs the complete optimization workflow."""
    current_best = seed_smiles
    st.session_state.cycle_data = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for cycle in range(num_cycles):
        status_text.text(f"Running Cycle {cycle + 1}/{num_cycles}...")
        progress_bar.progress((cycle) / num_cycles)
        
        # Generate variants
        variants = generate_molecular_variants(current_best, num_variants)
        
        if not variants:
            st.error("No valid variants generated!")
            return
        
        # Predict properties
        predictions = [calculate_mock_properties(smi) for smi in variants]
        
        # Store results
        cycle_results = pd.DataFrame({
            'SMILES': variants,
            'Predicted_Score': predictions,
            'Cycle': cycle + 1
        }).sort_values('Predicted_Score', ascending=False)
        
        best_in_cycle = cycle_results.iloc[0]
        st.session_state.cycle_data.append(cycle_results)
        
        # Update global best
        if best_in_cycle['Predicted_Score'] > st.session_state.best_score:
            st.session_state.best_molecule = best_in_cycle['SMILES']
            st.session_state.best_score = best_in_cycle['Predicted_Score']
        
        current_best = best_in_cycle['SMILES']
    
    progress_bar.progress(1.0)
    status_text.text("Optimization complete!")

# Main app logic
if st.sidebar.button("🚀 Start Optimization", type="primary"):
    st.session_state.results = []
    st.session_state.best_molecule = None
    st.session_state.best_score = -np.inf
    st.session_state.cycle_data = []
    
    with st.spinner("Initializing optimization..."):
        run_optimization()

# Display results if available
if st.session_state.cycle_data:
    st.success("✅ Optimization completed!")
    
    # Show best result
    st.subheader("🏆 Best Overall Molecule")
    display_molecule_info(st.session_state.best_molecule, "Best Overall Molecule")
    st.metric("Predicted Score", f"{st.session_state.best_score:.3f}")
    
    # Show all cycles results
    st.subheader("📊 Optimization Progress")
    
    all_results = pd.concat(st.session_state.cycle_data)
    
    # Plot results
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = plt.cm.viridis(np.linspace(0, 1, num_cycles))
    
    for i, cycle in enumerate(range(1, num_cycles + 1)):
        cycle_results = all_results[all_results['Cycle'] == cycle]
        ax.scatter([cycle] * len(cycle_results), 
                  cycle_results['Predicted_Score'], 
                  alpha=0.6, color=colors[i], label=f'Cycle {cycle}')
    
    ax.set_xlabel('Optimization Cycle')
    ax.set_ylabel('Simulated Prediction Score')
    ax.set_title('Molecular Scores Across Optimization Cycles')
    ax.legend()
    st.pyplot(fig)
    
    # Show data table
    st.dataframe(all_results.nlargest(10, 'Predicted_Score'), 
                use_container_width=True)

# Footer
st.divider()
st.caption("""
**MolOptiMVP Demo** | This is a conceptual prototype for demonstration purposes only. 
Not for actual drug discovery or molecular design. Built with Streamlit.
""")

# Add information about the simplified approach
with st.expander("ℹ️ About This Simplified Version"):
    st.markdown("""
    This version uses a simplified approach without RDKit to avoid dependency issues on Streamlit Cloud.
    
    **How it works:**
    - Molecular "complexity" is estimated based on SMILES string characteristics
    - Variants are generated by simple string manipulations
    - All predictions are simulated for demonstration purposes
    
    For a more accurate simulation with actual molecular properties, you would need to:
    1. Install RDKit locally
    2. Use a more powerful hosting service that supports RDKit's dependencies
    3. Add a `packages.txt` file with the required system libraries
    """)