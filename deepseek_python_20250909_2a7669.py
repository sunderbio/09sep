import streamlit as st
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors
from rdkit.DataStructs import ConvertToNumpyArray
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Set the page configuration
st.set_page_config(
    page_title="MolOptiMVP | AI-Driven Molecular Design",
    page_icon="🧪",
    layout="wide"
)

# Title and description
st.title("🧪 MolOptiMVP: Molecular Optimization Demo")
st.markdown("""
This is a **simplified prototype** demonstrating an AI-agentic workflow for molecular design.
It generates molecular variants and uses a mock predictive model to simulate optimization.
**This is for demonstration purposes only and does not predict real-world properties.**
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
    """)

# Initialize session state to store results
if 'results' not in st.session_state:
    st.session_state.results = []
if 'best_molecule' not in st.session_state:
    st.session_state.best_molecule = None
if 'best_score' not in st.session_state:
    st.session_state.best_score = -np.inf
if 'cycle_data' not in st.session_state:
    st.session_state.cycle_data = []

# Mock predictive function - SIMULATED FOR DEMONSTRATION
def mock_predict_property(smiles_list):
    """Simulates a predictive model. This does not use real data."""
    predictions = []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                # SIMULATED property based on molecular features
                # In a real app, this would be a trained model
                prop_score = (
                    Descriptors.MolWt(mol) / 1000 +
                    Descriptors.NumRotatableBonds(mol) * 0.1 +
                    np.random.normal(0, 0.1)
                )
                predictions.append(prop_score)
            else:
                predictions.append(0)
        except:
            predictions.append(0)
    return np.array(predictions)

# Molecular variant generator
def generate_molecular_variants(seed_smiles, num_variants=10):
    """Generates simple molecular variants for demonstration."""
    variants = set()
    mol = Chem.MolFromSmiles(seed_smiles)
    
    if not mol:
        return []
    
    # Simple molecular modifications for demonstration
    for _ in range(num_variants * 2):
        new_mol = Chem.RWMol(mol)
        
        # Random modification
        if np.random.random() > 0.5 and new_mol.GetNumAtoms() > 3:
            # Remove a random atom
            atom_idx = np.random.randint(0, new_mol.GetNumAtoms())
            new_mol.RemoveAtom(atom_idx)
        else:
            # Add a random atom
            new_mol.AddAtom(Chem.Atom(6))
            if new_mol.GetNumAtoms() > 1:
                existing_atom = np.random.randint(0, new_mol.GetNumAtoms()-1)
                new_mol.AddBond(existing_atom, new_mol.GetNumAtoms()-1, Chem.BondType.SINGLE)
        
        try:
            new_smi = Chem.MolToSmiles(new_mol.GetMol())
            if new_smi and 3 <= len(new_smi) <= 100:
                variants.add(new_smi)
        except:
            continue
        
        if len(variants) >= num_variants:
            break
    
    return list(variants)[:num_variants]

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
        
        # Predict properties (using mock model)
        predictions = mock_predict_property(variants)
        
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

# Display molecule function
def display_molecule(smiles, caption="Molecule"):
    """Displays a molecule from SMILES string."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            img = Draw.MolToImage(mol, size=(300, 200))
            st.image(img, caption=caption)
        else:
            st.write("Invalid molecule structure")
    except:
        st.write("Could not render molecule")

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
    col1, col2 = st.columns(2)
    
    with col1:
        st.code(f"SMILES: {st.session_state.best_molecule}")
        st.metric("Predicted Score", f"{st.session_state.best_score:.3f}")
    
    with col2:
        display_molecule(st.session_state.best_molecule, "Best Overall Molecule")
    
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
Not for actual drug discovery or molecular design.
""")