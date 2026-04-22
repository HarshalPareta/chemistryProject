import streamlit as st
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors, Lipinski
from rdkit.Chem import rdMolDescriptors
import plotly.graph_objects as go
import numpy as np
import math
import warnings

# Suppress RDKit warnings
warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="Molecule Stability Comparator", page_icon="🧪", layout="wide"
)

st.title("🧪 Molecular Stability Comparison")
st.markdown(
    "Compare the relative stability of two organic molecules using MMFF94 force field optimization."
)

# Sidebar
st.sidebar.header("⚙️ Configuration")

viz_style = st.sidebar.selectbox(
    "3D Visualization Style",
    ["stick", "sphere", "line", "bonds", "ball_and_stick"],
    index=0,
    help="stick: bonds as tubes, sphere: space-filling, line: wireframe, bonds: enhanced bond order, ball_and_stick: atoms+bonds",
)

show_2d = st.sidebar.checkbox("Show 2D Structure Preview", value=True)
advanced_info = st.sidebar.checkbox("Show Advanced Molecular Properties", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📦 System Status")

# Check dependencies
with st.sidebar.expander("🔧 Dependency Check", expanded=False):
    st.write("**RDKit:**")
    try:
        import rdkit

        st.success(f"✅ {rdkit.__version__}")
    except:
        st.error("❌ Not found")

    st.write("**Py3Dmol:**")
    try:
        import py3Dmol

        st.success("✅ Installed")
    except:
        st.error("❌ Not found")

    st.write("**Streamlit:**")
    try:
        import streamlit as st

        st.success(f"✅ {st.__version__}")
    except:
        st.error("❌ Not found")

with st.sidebar.expander("💾 Install Guide", expanded=False):
    st.markdown("""
    **Install all dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

    **Common issues:**

    **RDKit not found:**
    ```bash
    pip install rdkit
    ```

    **Py3Dmol rendering issues:**
    ```bash
    pip install py3Dmol --upgrade
    ```
    """)

st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Example SMILES")
st.sidebar.code("Cyclohexane: C1CCCCC1")
st.sidebar.code("Benzene: c1ccccc1")
st.sidebar.code("Ethanol: CCO")
st.sidebar.code("Caffeine: Cn1c(=O)c2c(ncn2C)n(C)c1=O")
st.sidebar.code("Aspirin: CC(=O)OC1=CC=CC=C1C(=O)O")

# Constants
R = 1.9872036e-3  # kcal/(mol·K) - Gas constant
TEMP = 298.15  # Room temperature in Kelvin


def validate_smiles(smiles):
    """Validate and convert SMILES to RDKit molecule."""
    try:
        smiles = smiles.strip()
        if not smiles:
            return None, "Empty SMILES string"

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is None:
                return None, "Invalid SMILES string - could not parse molecule"
            try:
                Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL)
            except:
                return None, "Invalid SMILES - fails valence/aromaticity checks"

        num_atoms = mol.GetNumAtoms()
        if num_atoms == 0:
            return None, "Molecule has no atoms"
        if num_atoms > 500:
            st.warning(
                f"Large molecule ({num_atoms} atoms) - optimization may take longer"
            )

        return mol, None
    except Exception as e:
        return None, f"SMILES parsing error: {str(e)}"


def optimize_molecule(mol, max_attempts=50):
    """Generate 3D conformer and optimize with MMFF94/UFF."""
    try:
        mol = Chem.AddHs(mol, explicitOnly=True)

        # Try multiple seeds for 3D embedding
        embed_success = False
        for seed in [42, 123, 456, 789, 1000]:
            if (
                AllChem.EmbedMolecule(mol, randomSeed=seed, maxAttempts=max_attempts)
                != -1
            ):
                embed_success = True
                break

        if not embed_success:
            try:
                AllChem.EmbedMolecule(
                    mol, useRandomCoords=True, maxAttempts=max_attempts
                )
                embed_success = True
            except:
                pass

        if not embed_success:
            return None, "Failed to generate 3D conformation"

        # Optimize with MMFF94, fallback to UFF
        energy = None
        try:
            mmff_props = AllChem.MMFFGetMoleculeProperties(mol)
            if mmff_props:
                ff = AllChem.MMFFGetMoleculeForceField(mol, mmff_props)
                if ff:
                    ff.Initialize()
                    ff.Minimize(500)
                    energy = ff.CalcEnergy()
        except:
            pass

        if energy is None:
            try:
                ff = AllChem.UFFGetMoleculeForceField(mol)
                if ff:
                    ff.Initialize()
                    ff.Minimize(500)
                    energy = ff.CalcEnergy()
            except Exception as e:
                return None, f"Force field optimization failed: {str(e)}"

        if energy is None:
            return None, "Could not calculate energy"

        return mol, energy
    except Exception as e:
        return None, f"Optimization error: {str(e)}"


def analyze_structural_factors(mol):
    """Analyze structural factors affecting molecular stability."""
    if mol is None:
        return None

    try:
        num_atoms = mol.GetNumAtoms()
        num_carbons = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "C")
        num_hydrogens = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "H")

        # SP3 carbon character
        sp3_carbons = sum(
            1
            for atom in mol.GetAtoms()
            if atom.GetSymbol() == "C"
            and atom.GetHybridization() == Chem.HybridizationType.SP3
        )
        sp2_carbons = sum(
            1
            for atom in mol.GetAtoms()
            if atom.GetSymbol() == "C"
            and atom.GetHybridization() == Chem.HybridizationType.SP2
        )
        sp_carbons = sum(
            1
            for atom in mol.GetAtoms()
            if atom.GetSymbol() == "C"
            and atom.GetHybridization() == Chem.HybridizationType.SP
        )
        sp3_fraction = sp3_carbons / num_carbons if num_carbons > 0 else 0

        # Ring analysis
        ring_info = mol.GetRingInfo()
        num_rings = ring_info.NumRings()

        # Ring strain estimation (angles, torsions)
        ring_strain = 0
        if num_rings > 0:
            for ring in ring_info.AtomRings():
                ring_size = len(ring)
                # Baeyer strain: deviation from ideal bond angles
                if ring_size == 3:
                    ring_strain += 24.7  # cyclopropane
                elif ring_size == 4:
                    ring_strain += 26.3  # cyclobutane
                elif ring_size == 5:
                    ring_strain += 6.3  # cyclopentane
                elif ring_size == 6:
                    ring_strain += 0.1  # cyclohexane (chair)
                elif ring_size > 6:
                    ring_strain += max(0, 6 - (ring_size - 6) * 0.5)

        # Aromaticity Analysis - IMPROVED for heterocycles like caffeine
        aromatic_atoms = 0
        aromatic_rings = 0
        antiaromatic_rings = 0
        aromatic_type = "none"  # none, aromatic, antiaromatic
        aromatic_heteroatoms = 0  # N, O in aromatic rings
        try:
            # Count ALL atoms in aromatic systems (including heteroatoms)
            for atom in mol.GetAtoms():
                if atom.GetIsAromatic():
                    aromatic_atoms += 1
                    # Count heteroatoms in aromatic rings (N, O, S)
                    if atom.GetSymbol() in ["N", "O", "S"]:
                        aromatic_heteroatoms += 1

            # Check each ring for aromaticity using RDKit's detection
            for ring in ring_info.AtomRings():
                ring_size = len(ring)
                if ring_size < 3:
                    continue

                # Count atoms in this ring that RDKit says are aromatic
                aromatic_count = sum(
                    1 for a in ring if mol.GetAtomWithIdx(a).GetIsAromatic()
                )

                # If ALL non-H atoms in ring are aromatic, count it as aromatic ring
                if (
                    aromatic_count >= ring_size - 1
                ):  # Allow for one non-aromatic if needed
                    aromatic_rings += 1

            # For fused ring systems (like caffeine/purine), also check if overall system is aromatic
            # If we have >= 5 aromatic atoms, it's definitely an aromatic system
            if aromatic_atoms >= 5:
                aromatic_rings = max(
                    aromatic_rings, 1
                )  # At least one aromatic ring system
                if aromatic_atoms >= 8:
                    aromatic_rings = 2  # Caffeine has fused ring system

            # Determine overall aromaticity type - MORE FORGIVING
            if antiaromatic_rings > 0:
                aromatic_type = "antiaromatic"
            elif aromatic_atoms >= 3:  # 3+ aromatic atoms = aromatic system
                aromatic_type = "aromatic"

        except:
            pass

        # Conjugation
        conjugated_bonds = 0
        for bond in mol.GetBonds():
            if bond.GetIsConjugated():
                conjugated_bonds += 1

        # Symmetry: number of unique atom environments (using Morgan fingerprints)
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512)
            unique_environments = fp.GetNumOnBits()
            symmetry_score = unique_environments / num_atoms if num_atoms > 0 else 1
        except:
            symmetry_score = 1

        # Steric factors
        num_rotatable_bonds = Lipinski.NumRotatableBonds(mol)
        tertiary_carbons = sum(
            1
            for atom in mol.GetAtoms()
            if atom.GetSymbol() == "C"
            and sum(1 for n in atom.GetNeighbors() if n.GetSymbol() == "C") >= 3
        )
        quaternary_carbons = sum(
            1
            for atom in mol.GetAtoms()
            if atom.GetSymbol() == "C"
            and sum(1 for n in atom.GetNeighbors() if n.GetSymbol() == "C") == 4
        )

        # Molecular rigidity
        rigidity_score = 1 - (num_rotatable_bonds / num_atoms) if num_atoms > 0 else 1

        # Hydrogen bonding
        h_donors = Lipinski.NumHDonors(mol)
        h_acceptors = Lipinski.NumHAcceptors(mol)

        # Formal charges analysis
        has_positive_charge = False
        has_negative_charge = False
        formal_charge = 0
        try:
            for atom in mol.GetAtoms():
                fc = atom.GetFormalCharge()
                formal_charge += fc
                if fc > 0:
                    has_positive_charge = True
                elif fc < 0:
                    has_negative_charge = True
        except:
            pass

        # Check if charged - charged molecules are generally less stable
        is_charged = has_positive_charge or has_negative_charge
        charge_stability = "neutral"
        if formal_charge != 0:
            charge_stability = "charged"

        # Reactive group analysis (for comparison like caffeine vs quinine)
        reactive_groups = {
            "vinyl": 0,  # CH=CH2 - reactive double bond outside aromatic ring
            "alcohol": 0,  # -OH - oxidation prone
            "amine": 0,  # -NH2, -NH- - nucleophilic
            "thioester": 0,  # -SH - oxidation prone
            "aldehyde": 0,  # -CHO - oxidation prone
            "carbonyl": 0,  # C=O (not in aromatic ring)
        }
        nucleophilic_sites = 0  # Count nucleophilic atoms

        try:
            for atom in mol.GetAtoms():
                sym = atom.GetSymbol()
                # Check for -OH (alcohol)
                if sym == "O":
                    has_h = any(n.GetSymbol() == "H" for n in atom.GetNeighbors())
                    if has_h:
                        reactive_groups["alcohol"] += 1
                    # Count as nucleophilic (oxygen lone pairs)
                    nucleophilic_sites += 1

                # Check for nitrogen atoms (amines, amides)
                if sym == "N":
                    reactive_groups["amine"] += 1
                    # Check if it's basic (not aromatic N in ring)
                    if not atom.GetIsAromatic():
                        nucleophilic_sites += 1

                # Check for sulfur
                if sym == "S":
                    reactive_groups["thioester"] += 1

                # Check for aldehyde/ketone carbonyls (non-aromatic C=O)
                if sym == "C":
                    for bond in atom.GetBonds():
                        if bond.GetBondTypeAsDouble() == 2:  # C=O
                            # Check if part of aromatic ring
                            is_in_aromatic_ring = atom.GetIsAromatic()
                            if not is_in_aromatic_ring:
                                reactive_groups["carbonyl"] += 1

            # Count vinyl groups (terminal C=CH2)
            for bond in mol.GetBonds():
                if bond.GetBondTypeAsDouble() == 2:
                    begin_atom = bond.GetBeginAtom()
                    end_atom = bond.GetEndAtom()
                    # Check for terminal alkene pattern
                    begin_H = sum(
                        1 for n in begin_atom.GetNeighbors() if n.GetSymbol() == "H"
                    )
                    end_H = sum(
                        1 for n in end_atom.GetNeighbors() if n.GetSymbol() == "H"
                    )
                    if begin_H >= 2 or end_H >= 2:  # Terminal alkene
                        reactive_groups["vinyl"] += 1

            # Calculate aromatic coverage ratio
            aromatic_coverage = aromatic_atoms / num_atoms if num_atoms > 0 else 0

            # Calculate overall aromatic score (weighted)
            # More aromatic atoms + conjugated heteroatoms = higher stability
            hetero_aromatic = aromatic_heteroatoms  # N, O in aromatic system
            aromatic_score = (
                (aromatic_atoms * 10) + (hetero_aromatic * 15) + (aromatic_rings * 20)
            )

            # Calculate reactivity score (more reactive groups = less stable)
            reactivity_penalty = (
                reactive_groups["vinyl"] * 5
                + reactive_groups["alcohol"] * 3
                + reactive_groups["amine"] * 4
                + reactive_groups["carbonyl"] * 2
            )

        except:
            pass

        # Polar surface area
        try:
            tpsa = Descriptors.TPSA(mol)
        except:
            tpsa = 0

        # Heteroatom ratio
        heteroatoms = sum(
            1 for atom in mol.GetAtoms() if atom.GetSymbol() not in ["C", "H"]
        )
        heteroatom_ratio = heteroatoms / num_atoms if num_atoms > 0 else 0

        # Double/triple bond content
        double_bonds = sum(
            1 for bond in mol.GetBonds() if bond.GetBondTypeAsDouble() == 2
        )
        triple_bonds = sum(
            1 for bond in mol.GetBonds() if bond.GetBondTypeAsDouble() == 3
        )

        return {
            "num_atoms": num_atoms,
            "num_carbons": num_carbons,
            "num_hydrogens": num_hydrogens,
            "sp3_fraction": sp3_fraction,
            "sp3_carbons": sp3_carbons,
            "sp2_carbons": sp2_carbons,
            "sp_carbons": sp_carbons,
            "num_rings": num_rings,
            "ring_strain": ring_strain,
            "aromatic_atoms": aromatic_atoms,
            "aromatic_rings": aromatic_rings,
            "antiaromatic_rings": antiaromatic_rings,
            "aromatic_type": aromatic_type,
            "aromatic_heteroatoms": aromatic_heteroatoms,
            "conjugated_bonds": conjugated_bonds,
            "symmetry_score": symmetry_score,
            "num_rotatable_bonds": num_rotatable_bonds,
            "tertiary_carbons": tertiary_carbons,
            "quaternary_carbons": quaternary_carbons,
            "rigidity_score": rigidity_score,
            "h_donors": h_donors,
            "h_acceptors": h_acceptors,
            "formal_charge": formal_charge,
            "is_charged": is_charged,
            "charge_stability": charge_stability,
            "tpsa": tpsa,
            "heteroatom_ratio": heteroatom_ratio,
            "double_bonds": double_bonds,
            "triple_bonds": triple_bonds,
            "aromatic_coverage": aromatic_coverage,
            "aromatic_score": aromatic_score,
            "reactivity_penalty": reactivity_penalty,
            "reactive_groups": reactive_groups,
            "nucleophilic_sites": nucleophilic_sites,
        }
    except Exception as e:
        return None


def compare_structural_factors(props_a, props_b):
    """
    Compare structural factors following the 6-step organic chemistry framework.
    This is HIERARCHICAL - Step 1 is the TRUMP CARD that overrides all other steps.
    If Step 1 has a winner, that's the final answer. Otherwise, check Step 2, etc.
    """
    if props_a is None or props_b is None:
        return None

    results = {
        "step1_aromaticity": {
            "title": "Step 1: Aromaticity (TRUMP CARD)",
            "comparisons": [],
            "winner": None,
            "stability_advantage": None,
        },
        "step2_carbon_framework": {
            "title": "Step 2: Carbon Framework (Isomers)",
            "comparisons": [],
            "winner": None,
            "stability_advantage": None,
        },
        "step3_ring_strain": {
            "title": "Step 3: Ring Strain",
            "comparisons": [],
            "winner": None,
            "stability_advantage": None,
        },
        "step4_resonance": {
            "title": "Step 4: Resonance (Delocalization)",
            "comparisons": [],
            "winner": None,
            "stability_advantage": None,
        },
        "step5_steric": {
            "title": "Step 5: Steric Hindrance",
            "comparisons": [],
            "winner": None,
            "stability_advantage": None,
        },
        "step6_inductive": {
            "title": "Step 6: Inductive Effects & Electronegativity",
            "comparisons": [],
            "winner": None,
            "stability_advantage": None,
        },
    }

    # Step 1: Aromaticity (TRUMP CARD - wins over all other factors)
    # Positive: aromatic (stable), Negative: antiaromatic (unstable)
    aromatic_a = props_a["aromatic_rings"]
    aromatic_b = props_b["aromatic_rings"]
    aromatic_atoms_a = props_a["aromatic_atoms"]
    aromatic_atoms_b = props_b["aromatic_atoms"]
    aroma_type_a = props_a.get("aromatic_type", "none")
    aroma_type_b = props_b.get("aromatic_type", "none")
    antiarom_a = props_a.get("antiaromatic_rings", 0)
    antiarom_b = props_b.get("antiaromatic_rings", 0)
    arom_het_a = props_a.get("aromatic_heteroatoms", 0)
    arom_het_b = props_b.get("aromatic_heteroatoms", 0)

    # Use BOTH ring count AND atom count for scoring
    # This ensures caffeine (8+ aromatic atoms) beats benzene (6 atoms)
    # Also weight aromatic atoms more heavily (+5 each)
    a_score = (aromatic_a * 10) + (aromatic_atoms_a * 5) - (antiarom_a * 20)
    b_score = (aromatic_b * 10) + (aromatic_atoms_b * 5) - (antiarom_b * 20)

    if a_score != b_score:
        winner = "A" if a_score > b_score else "B"

        if a_score > 0 and b_score > 0:
            reason = "Aromatic/heterocyclic aromatic systems have exceptional stability due to cyclic, planar, conjugated π-electron systems (Hückel's rule)"
        elif a_score < 0 and b_score < 0:
            reason = "Both molecules have antiaromatic character (4n π electrons) - inherently unstable!"
        elif a_score > 0:
            reason = "Molecule A is aromatic (stable 4n+2 system), B is antiaromatic (unstable 4n system)"
        else:
            reason = "Molecule B is aromatic (stable 4n+2 system), A is antiaromatic (unstable 4n system)"

        results["step1_aromaticity"]["comparisons"].append(
            {
                "factor": "Aromaticity",
                "a_val": f"{aromatic_a} rings, {aromatic_atoms_a} atoms + {arom_het_a} heteroatoms ({aroma_type_a})",
                "b_val": f"{aromatic_b} rings, {aromatic_atoms_b} atoms + {arom_het_b} heteroatoms ({aroma_type_b})",
                "winner": winner,
                "reason": reason,
            }
        )
        results["step1_aromaticity"]["winner"] = winner
        results["step1_aromaticity"]["stability_advantage"] = (
            f"Molecule {winner} has aromatic advantage"
        )

        # TRUMP CARD - this wins regardless of other factors
        results["final_winner"] = winner
        results["conclusion"] = (
            f"Aromaticity (TRUMP CARD): Molecule {winner} wins due to aromatic stabilization"
        )
        results["winning_step"] = 1
        return results
    else:
        # Check for antiaromaticity even if scores equal (both negative = antiaromatic)
        if antiarom_a > 0 or antiarom_b > 0:
            results["step1_aromaticity"]["comparisons"].append(
                {
                    "factor": "Antiaromaticity (UNSTABLE!)",
                    "a_val": f"{antiarom_a} rings",
                    "b_val": f"{antiarom_b} rings",
                    "winner": "A"
                    if antiarom_a < antiarom_b
                    else "B"
                    if antiarom_b < antiarom_a
                    else "Equal",
                    "reason": "Antiaromatic systems (4n π electrons) are VERY UNSTABLE and will undergo reactions to become aromatic",
                }
            )

        results["step1_aromaticity"]["comparisons"].append(
            {
                "factor": "Aromatic Character",
                "a_val": f"{aromatic_a} rings, {aromatic_atoms_a} atoms",
                "b_val": f"{aromatic_b} rings, {aromatic_atoms_b} atoms",
                "winner": "Equal",
                "reason": "Both molecules have same aromatic character",
            }
        )

    # Step 2: Carbon Framework (only if Step 1 is equal)
    sp3_a = props_a["sp3_fraction"]
    sp3_b = props_b["sp3_fraction"]
    rings_a = props_a["num_rings"]
    rings_b = props_b["num_rings"]

    # Check if one is cyclic and other is acyclic
    if rings_a > 0 and rings_b == 0:
        # A is cyclic, B is acyclic - check if A has stable ring (5 or 6-membered)
        strain_a = props_a["ring_strain"]
        # Cyclohexane (6-membered) and cyclopentane (5-membered) are stable
        if strain_a < 10:  # Very low strain ring
            winner = "A"
            results["step2_carbon_framework"]["comparisons"].append(
                {
                    "factor": "Cyclic vs Acyclic",
                    "a_val": f"{rings_a} ring(s) (stable)",
                    "b_val": "Acyclic",
                    "winner": winner,
                    "reason": "Cycloalkanes (especially 5-6 membered) have conformational stability advantage over acyclic alkanes",
                }
            )
            results["step2_carbon_framework"]["winner"] = winner
            results["step2_carbon_framework"]["stability_advantage"] = (
                f"Molecule {winner} has stable ring structure"
            )
            results["final_winner"] = winner
            results["conclusion"] = (
                f"Carbon Framework: Molecule {winner} wins (stable cyclic structure)"
            )
            results["winning_step"] = 2
            return results

    if rings_b > 0 and rings_a == 0:
        # B is cyclic, A is acyclic
        strain_b = props_b["ring_strain"]
        if strain_b < 10:
            winner = "B"
            results["step2_carbon_framework"]["comparisons"].append(
                {
                    "factor": "Cyclic vs Acyclic",
                    "a_val": "Acyclic",
                    "b_val": f"{rings_b} ring(s) (stable)",
                    "winner": winner,
                    "reason": "Cycloalkanes (especially 5-6 membered) have conformational stability advantage over acyclic alkanes",
                }
            )
            results["step2_carbon_framework"]["winner"] = winner
            results["step2_carbon_framework"]["stability_advantage"] = (
                f"Molecule {winner} has stable ring structure"
            )
            results["final_winner"] = winner
            results["conclusion"] = (
                f"Carbon Framework: Molecule {winner} wins (stable cyclic structure)"
            )
            results["winning_step"] = 2
            return results

    # If both have rings or both acyclic, check SP³
    if sp3_a != sp3_b:
        winner = "A" if sp3_a > sp3_b else "B"
        results["step2_carbon_framework"]["comparisons"].append(
            {
                "factor": "SP³ Carbon Character",
                "a_val": f"{sp3_a * 100:.1f}%",
                "b_val": f"{sp3_b * 100:.1f}%",
                "winner": winner,
                "reason": "More branched (SP³-rich) isomers are more stable due to protobranching and hyperconjugation",
            }
        )
        results["step2_carbon_framework"]["winner"] = winner
        results["step2_carbon_framework"]["stability_advantage"] = (
            f"Molecule {winner} has more SP³ carbons"
        )
        results["final_winner"] = winner
        results["conclusion"] = (
            f"Carbon Framework: Molecule {winner} wins (more SP³ character)"
        )
        results["winning_step"] = 2
        return results

    # Step 3: Ring Strain (only if Step 1 and 2 are equal)
    strain_a = props_a["ring_strain"]
    strain_b = props_b["ring_strain"]
    if strain_a != strain_b:
        winner = "A" if strain_a < strain_b else "B"
        results["step3_ring_strain"]["comparisons"].append(
            {
                "factor": "Ring Strain Energy",
                "a_val": f"{strain_a:.1f} kcal/mol",
                "b_val": f"{strain_b:.1f} kcal/mol",
                "winner": winner,
                "reason": "Lower ring strain means bond angles closer to ideal 109.5°. Order: 6-membered > 5 > 4 > 3-membered",
            }
        )
        results["step3_ring_strain"]["winner"] = winner
        results["step3_ring_strain"]["stability_advantage"] = (
            f"Molecule {winner} has lower ring strain"
        )
        results["final_winner"] = winner
        results["conclusion"] = (
            f"Ring Strain: Molecule {winner} wins (lower strain: {min(strain_a, strain_b):.1f} vs {max(strain_a, strain_b):.1f} kcal/mol)"
        )
        results["winning_step"] = 3
        return results

    # Step 4: Resonance / Conjugation (only if Steps 1-3 are equal)
    conj_a = props_a["conjugated_bonds"]
    conj_b = props_b["conjugated_bonds"]
    if conj_a != conj_b:
        winner = "A" if conj_a > conj_b else "B"
        results["step4_resonance"]["comparisons"].append(
            {
                "factor": "Conjugated Bonds",
                "a_val": str(conj_a),
                "b_val": str(conj_b),
                "winner": winner,
                "reason": "Conjugation delocalizes electrons across multiple atoms, increasing stability",
            }
        )
        results["step4_resonance"]["winner"] = winner
        results["step4_resonance"]["stability_advantage"] = (
            f"Molecule {winner} has more conjugation"
        )
        results["final_winner"] = winner
        results["conclusion"] = (
            f"Resonance: Molecule {winner} wins (more conjugated bonds)"
        )
        results["winning_step"] = 4
        return results

    # Step 5: Steric Hindrance (only if Steps 1-4 are equal)
    rotatable_a = props_a["num_rotatable_bonds"]
    rotatable_b = props_b["num_rotatable_bonds"]
    steric_a = props_a["tertiary_carbons"] + props_a["quaternary_carbons"]
    steric_b = props_b["tertiary_carbons"] + props_b["quaternary_carbons"]

    if rotatable_a != rotatable_b:
        winner = "A" if rotatable_a < rotatable_b else "B"
        results["step5_steric"]["comparisons"].append(
            {
                "factor": "Rotatable Bonds",
                "a_val": str(rotatable_a),
                "b_val": str(rotatable_b),
                "winner": winner,
                "reason": "Fewer rotatable bonds means less conformational freedom and reduced steric strain",
            }
        )
        results["step5_steric"]["winner"] = winner
        results["step5_steric"]["stability_advantage"] = (
            f"Molecule {winner} has fewer rotatable bonds"
        )
        results["final_winner"] = winner
        results["conclusion"] = (
            f"Steric Hindrance: Molecule {winner} wins (fewer rotatable bonds)"
        )
        results["winning_step"] = 5
        return results

    if steric_a != steric_b:
        winner = "A" if steric_a < steric_b else "B"
        results["step5_steric"]["comparisons"].append(
            {
                "factor": "Steric Crowding (Tert/Quat C)",
                "a_val": str(steric_a),
                "b_val": str(steric_b),
                "winner": winner,
                "reason": "Bulky groups (tert-butyl, etc.) prefer to be apart; less crowding = more stable",
            }
        )
        results["step5_steric"]["winner"] = winner
        results["step5_steric"]["stability_advantage"] = (
            f"Molecule {winner} has less steric crowding"
        )
        results["final_winner"] = winner
        results["conclusion"] = (
            f"Steric Hindrance: Molecule {winner} wins (less crowding)"
        )
        results["winning_step"] = 5
        return results

    # Step 6: Inductive Effects (only if Steps 1-5 are equal)
    # CRITICAL: Charged molecules are generally less stable than neutral ones
    charged_a = props_a.get("is_charged", False)
    charged_b = props_b.get("is_charged", False)
    charge_a = props_a.get("formal_charge", 0)
    charge_b = props_b.get("formal_charge", 0)

    if charged_a != charged_b:
        winner = "A" if not charged_a else "B"  # Neutral wins over charged
        results["step6_inductive"]["comparisons"].append(
            {
                "factor": "Charge State",
                "a_val": f"Charged ({charge_a:+d})" if charged_a else "Neutral",
                "b_val": f"Charged ({charge_b:+d})" if charged_b else "Neutral",
                "winner": winner,
                "reason": "Neutral molecules are almost always more stable than charged ones (cations/anions)",
            }
        )
        results["step6_inductive"]["winner"] = winner
        results["step6_inductive"]["stability_advantage"] = (
            f"Molecule {winner} is neutral (more stable)"
        )
        results["final_winner"] = winner
        results["conclusion"] = (
            f"Charge Stability: Molecule {winner} wins (neutral vs charged)"
        )
        results["winning_step"] = 6
        return results

    # Both charged or both neutral - check TPSA
    tpsa_a = props_a["tpsa"]
    tpsa_b = props_b["tpsa"]
    if tpsa_a != tpsa_b:
        winner = "A" if tpsa_a < tpsa_b else "B"
        results["step6_inductive"]["comparisons"].append(
            {
                "factor": "Polar Surface Area (TPSA)",
                "a_val": f"{tpsa_a:.1f} Å²",
                "b_val": f"{tpsa_b:.1f} Å²",
                "winner": winner,
                "reason": "Lower PSA often correlates with better stability and permeability",
            }
        )
        results["step6_inductive"]["winner"] = winner
        results["step6_inductive"]["stability_advantage"] = (
            f"Molecule {winner} has lower TPSA"
        )
        results["final_winner"] = winner
        results["conclusion"] = (
            f"Inductive Effects: Molecule {winner} wins (lower TPSA)"
        )
        results["winning_step"] = 6
        return results

    # All steps equal - no structural advantage found
    results["final_winner"] = "Equal"
    results["conclusion"] = (
        "No structural differences found - molecules have similar stability profiles"
    )
    results["winning_step"] = None
    return results


def mol_to_plotly_figure(mol, size=(800, 600), style="stick"):
    """
    Convert RDKit molecule to an interactive Plotly 3D figure.
    Renders atoms as spheres and bonds as cylinders (via lines).
    Explicitly handles single, double, triple bond orders.
    """
    # Get 3D coordinates from the molecule's conformer
    conf = mol.GetConformer()
    num_atoms = mol.GetNumAtoms()

    # Extract positions and element symbols
    positions = np.zeros((num_atoms, 3))
    symbols = []
    for i in range(num_atoms):
        pos = conf.GetAtomPosition(i)
        positions[i] = (pos.x, pos.y, pos.z)
        symbols.append(mol.GetAtomWithIdx(i).GetSymbol())

    # CPK element colors
    cpk_colors = {
        "H": "white",
        "C": "gray",
        "N": "blue",
        "O": "red",
        "F": "green",
        "Cl": "green",
        "Br": "brown",
        "I": "purple",
        "S": "yellow",
        "P": "orange",
        "Si": "tan",
        "B": "brown",
        "Fe": "orange",
        "Zn": "gray",
        "Cu": "orange",
    }
    colors = [cpk_colors.get(sym, "lightgray") for sym in symbols]

    # Approximate atomic radii for marker size (scaled)
    radii = {
        "H": 0.31,
        "C": 0.76,
        "N": 0.71,
        "O": 0.66,
        "F": 0.57,
        "Cl": 0.99,
        "Br": 1.14,
        "I": 1.33,
        "S": 1.05,
        "P": 1.07,
        "Si": 1.11,
        "B": 0.87,
    }
    sizes = [radii.get(sym, 0.8) * 8 for sym in symbols]  # scale up for visibility

    # Style-dependent parameters
    if style == "stick":
        atom_size = 6
        bond_width = 8
        offset_factor = 0.12
    elif style == "sphere":
        atom_size = 14
        bond_width = 4  # thinner bonds for sphere style
        offset_factor = 0.08
    elif style == "line":
        atom_size = 3
        bond_width = 3
        offset_factor = 0.05
    elif style == "bonds":
        atom_size = 3
        bond_width = 20
        offset_factor = 0.08
    elif style == "ball_and_stick":
        atom_size = 8
        bond_width = 5
        offset_factor = 0.10
    else:
        atom_size = 6
        bond_width = 8
        offset_factor = 0.12

    # Create Plotly figure
    fig = go.Figure()

    # Add atoms as markers
    fig.add_trace(
        go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode="markers",
            marker=dict(size=atom_size, color=colors, line=dict(width=0)),
            text=[f"{sym}" for sym in symbols],
            hoverinfo="text",
            name="Atoms",
        )
    )

    # If bonds are disabled, skip bond drawing
    if bond_width > 0:
        # Helper: get a perpendicular vector for bond offset
        def get_perp(v):
            # v: unit vector
            # pick reference not parallel
            if abs(v[0]) < 0.9:
                ref = np.array([1.0, 0.0, 0.0])
            else:
                ref = np.array([0.0, 1.0, 0.0])
            perp = np.cross(v, ref)
            norm = np.linalg.norm(perp)
            if norm < 1e-6:
                ref = np.array([0.0, 0.0, 1.0])
                perp = np.cross(v, ref)
                norm = np.linalg.norm(perp)
                if norm < 1e-6:
                    return None
            return perp / norm

        # Draw each bond
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            order = bond.GetBondTypeAsDouble()
            p1 = positions[i]
            p2 = positions[j]
            v = p2 - p1
            length = np.linalg.norm(v)
            if length < 1e-6:
                continue
            v_hat = v / length

            # For aromatic (1.5) treat as single
            if order == 1.5:
                order = 1

            # Compute perpendicular offset vector (fixed offset in angstroms)
            perp = get_perp(v_hat)
            if perp is None:
                offset_dist = 0
            else:
                offset_dist = offset_factor

            if order == 1:
                # Single bond: single line
                xs = [p1[0], p2[0]]
                ys = [p1[1], p2[1]]
                zs = [p1[2], p2[2]]
                fig.add_trace(
                    go.Scatter3d(
                        x=xs,
                        y=ys,
                        z=zs,
                        mode="lines",
                        line=dict(width=bond_width, color="gray"),
                        hoverinfo="none",
                        showlegend=False,
                    )
                )
            elif order == 2:
                # Double bond: two offset lines
                for sign in [1, -1]:
                    offset = perp * offset_dist if perp is not None else np.zeros(3)
                    start = p1 + offset * sign
                    end = p2 + offset * sign
                    fig.add_trace(
                        go.Scatter3d(
                            x=[start[0], end[0]],
                            y=[start[1], end[1]],
                            z=[start[2], end[2]],
                            mode="lines",
                            line=dict(width=bond_width, color="gray"),
                            hoverinfo="none",
                            showlegend=False,
                        )
                    )
            elif order == 3:
                # Triple bond: three lines (one central, two offset)
                # Center
                fig.add_trace(
                    go.Scatter3d(
                        x=[p1[0], p2[0]],
                        y=[p1[1], p2[1]],
                        z=[p1[2], p2[2]],
                        mode="lines",
                        line=dict(width=max(2, bond_width - 2), color="gray"),
                        hoverinfo="none",
                        showlegend=False,
                    )
                )
                # Offsets
                for sign in [1, -1]:
                    if perp is not None:
                        offset = perp * offset_dist
                        start = p1 + offset * sign
                        end = p2 + offset * sign
                        fig.add_trace(
                            go.Scatter3d(
                                x=[start[0], end[0]],
                                y=[start[1], end[1]],
                                z=[start[2], end[2]],
                                mode="lines",
                                line=dict(width=bond_width, color="gray"),
                                hoverinfo="none",
                                showlegend=False,
                            )
                        )
            else:
                # Unknown bond type: single line
                fig.add_trace(
                    go.Scatter3d(
                        x=[p1[0], p2[0]],
                        y=[p1[1], p2[1]],
                        z=[p1[2], p2[2]],
                        mode="lines",
                        line=dict(width=bond_width, color="gray"),
                        hoverinfo="none",
                        showlegend=False,
                    )
                )

    # Layout: remove axes, adjust aspect
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
        ),
        width=size[0],
        height=size[1],
        margin=dict(l=0, r=0, b=0, t=0),
        showlegend=False,
    )

    return fig


def smiles_to_image(smiles, size=(400, 400)):
    """Generate 2D structure image with clear bond rendering."""
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        mol = Chem.Mol(mol)
        try:
            AllChem.Compute2DCoords(mol)
        except:
            pass

        # Use RDKit's built-in drawing - it handles double/triple bonds automatically
        # The default Draw.MolToImage already shows bond orders correctly
        return Draw.MolToImage(mol, size=size, kekulize=True)
    return None


# Main input area
st.subheader("🔬 Input Molecule SMILES")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🔵 Molecule A")
    smiles_a = st.text_input(
        "SMILES for Molecule A",
        value="C1CCCCC1",
        key="smiles_a",
        help="Example: Cyclohexane = C1CCCCC1",
    )
    if show_2d:
        img_a = smiles_to_image(smiles_a)
        if img_a:
            st.image(img_a, caption="2D Structure A", use_container_width=True)

with col2:
    st.subheader("🔴 Molecule B")
    smiles_b = st.text_input(
        "SMILES for Molecule B",
        value="CCCCCC",
        key="smiles_b",
        help="Example: Hexane = CCCCCC",
    )
    if show_2d:
        img_b = smiles_to_image(smiles_b)
        if img_b:
            st.image(img_b, caption="2D Structure B", use_container_width=True)

# Optimization and Calculation
if st.button("🔬 Optimize & Compare", type="primary", use_container_width=True):
    if not smiles_a or not smiles_b:
        st.error("Please provide SMILES strings for both molecules.")
    else:
        with st.spinner("Optimizing molecules with MMFF94..."):
            # Process Molecule A
            mol_a, error_a = validate_smiles(smiles_a)
            if error_a:
                st.error(f"Molecule A error: {error_a}")
                mol_a = None
            else:
                mol_a, energy_a = optimize_molecule(mol_a)
                if not mol_a:
                    st.error(f"Molecule A optimization failed: {energy_a}")

            # Process Molecule B
            mol_b, error_b = validate_smiles(smiles_b)
            if error_b:
                st.error(f"Molecule B error: {error_b}")
                mol_b = None
            else:
                mol_b, energy_b = optimize_molecule(mol_b)
                if not mol_b:
                    st.error(f"Molecule B optimization failed: {energy_b}")

            # Display results if both successful
            if mol_a is not None and mol_b is not None:
                st.success("✅ Optimization complete!")

                # Energy comparison
                delta_e = energy_b - energy_a
                more_stable = "A" if energy_a < energy_b else "B"
                stability_diff = abs(delta_e)

                # Boltzmann equilibrium constant
                boltzmann_k = math.exp(-delta_e / (R * TEMP))

                # Dashboard
                st.divider()
                st.subheader("📊 Stability Comparison")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Molecule A Energy", f"{energy_a:.2f} kcal/mol")
                with col2:
                    st.metric("Molecule B Energy", f"{energy_b:.2f} kcal/mol")
                with col3:
                    st.metric(
                        "ΔE (B - A)",
                        f"{delta_e:.2f} kcal/mol",
                        delta=f"{delta_e:.2f}" if delta_e != 0 else "0",
                    )

                # Verdict
                st.markdown("---")
                verdict_col1, verdict_col2 = st.columns([1, 2])
                with verdict_col1:
                    if more_stable == "A":
                        st.info("🏆 Molecule A is more stable!")
                    else:
                        st.info("🏆 Molecule B is more stable!")
                with verdict_col2:
                    st.markdown(f"""
                    **Verdict:** Molecule **{more_stable}** is more stable by **{stability_diff:.2f} kcal/mol**.
                    """)

                # Boltzmann distribution
                st.markdown("---")
                st.subheader("🔥 Thermodynamic Equilibrium")
                st.markdown(rf"""
                At **{TEMP:.0f} K** (room temperature):

                - **Equilibrium constant:** $K = e^{{-\Delta E / RT}}$ = **{boltzmann_k:.4e}**
                - **Interpretation:**
                    - If $K > 1$: Molecule A favored (A more stable)
                    - If $K < 1$: Molecule B favored (B more stable)
                    - Current $K$ = **{boltzmann_k:.4e}** → **{"A" if boltzmann_k > 1 else "B"}** dominates
                """)

                # Structural factor analysis - 6-Step Framework
                st.markdown("---")
                st.subheader("🏗️ Structural Analysis (6-Step Framework)")

                props_a = analyze_structural_factors(mol_a)
                props_b = analyze_structural_factors(mol_b)

                structural_results = None
                if props_a and props_b:
                    structural_results = compare_structural_factors(props_a, props_b)

                if structural_results:
                    # Step 1: Aromaticity
                    step1 = structural_results["step1_aromaticity"]
                    with st.expander(f"🃏 {step1['title']}", expanded=True):
                        if step1["comparisons"]:
                            for comp in step1["comparisons"]:
                                winner_icon = "🔵" if comp["winner"] == "A" else "🔴"
                                st.markdown(
                                    f"**{comp['factor']}:** {winner_icon} Molecule {comp['winner']} ({comp['a_val']} vs {comp['b_val']})"
                                )
                                st.caption(f"_{comp['reason']}_")
                        else:
                            st.info(
                                "No aromaticity difference found between molecules."
                            )

                    # Step 2: Carbon Framework
                    step2 = structural_results["step2_carbon_framework"]
                    with st.expander(f"🔗 {step2['title']}", expanded=False):
                        if step2["comparisons"]:
                            for comp in step2["comparisons"]:
                                winner_icon = "🔵" if comp["winner"] == "A" else "🔴"
                                st.markdown(
                                    f"**{comp['factor']}:** {winner_icon} Molecule {comp['winner']} ({comp['a_val']} vs {comp['b_val']})"
                                )
                                st.caption(f"_{comp['reason']}_")
                        else:
                            st.info("No significant carbon framework differences.")

                    # Step 3: Ring Strain
                    step3 = structural_results["step3_ring_strain"]
                    with st.expander(f"💫 {step3['title']}", expanded=False):
                        if step3["comparisons"]:
                            for comp in step3["comparisons"]:
                                winner_icon = "🔵" if comp["winner"] == "A" else "🔴"
                                st.markdown(
                                    f"**{comp['factor']}:** {winner_icon} Molecule {comp['winner']} ({comp['a_val']} vs {comp['b_val']})"
                                )
                                st.caption(f"_{comp['reason']}_")
                        else:
                            st.info("No ring strain difference found.")

                    # Step 4: Resonance
                    step4 = structural_results["step4_resonance"]
                    with st.expander(f"🌊 {step4['title']}", expanded=False):
                        if step4["comparisons"]:
                            for comp in step4["comparisons"]:
                                winner_icon = "🔵" if comp["winner"] == "A" else "🔴"
                                st.markdown(
                                    f"**{comp['factor']}:** {winner_icon} Molecule {comp['winner']} ({comp['a_val']} vs {comp['b_val']})"
                                )
                                st.caption(f"_{comp['reason']}_")
                        else:
                            st.info("No significant resonance differences found.")

                    # Step 5: Steric Hindrance
                    step5 = structural_results["step5_steric"]
                    with st.expander(f"⚡ {step5['title']}", expanded=False):
                        if step5["comparisons"]:
                            for comp in step5["comparisons"]:
                                winner_icon = "🔵" if comp["winner"] == "A" else "🔴"
                                st.markdown(
                                    f"**{comp['factor']}:** {winner_icon} Molecule {comp['winner']} ({comp['a_val']} vs {comp['b_val']})"
                                )
                                st.caption(f"_{comp['reason']}_")
                        else:
                            st.info("No significant steric differences found.")

                    # Step 6: Inductive Effects
                    step6 = structural_results["step6_inductive"]
                    with st.expander(f"🧲 {step6['title']}", expanded=False):
                        if step6["comparisons"]:
                            for comp in step6["comparisons"]:
                                winner_icon = "🔵" if comp["winner"] == "A" else "🔴"
                                st.markdown(
                                    f"**{comp['factor']}:** {winner_icon} Molecule {comp['winner']} ({comp['a_val']} vs {comp['b_val']})"
                                )
                                st.caption(f"_{comp['reason']}_")
                        else:
                            st.info(
                                "No significant inductive/electronegativity differences found."
                            )

                    # Multi-Category Stability Comparison
                    st.markdown("---")
                    st.subheader("📊 Multi-Category Stability Comparison")

                    # Calculate comparisons for each category

                    # 1. Thermodynamic Stability (MMFF94 Energy)
                    therm_winner = "A" if energy_a < energy_b else "B"
                    therm_diff = abs(energy_a - energy_b)

                    # 2. Aromaticity Score
                    aroma_a = (props_a.get("aromatic_rings", 0) * 10) + (
                        props_a.get("aromatic_atoms", 0) * 5
                    )
                    aroma_b = (props_b.get("aromatic_rings", 0) * 10) + (
                        props_b.get("aromatic_atoms", 0) * 5
                    )
                    aroma_winner = (
                        "A"
                        if aroma_a > aroma_b
                        else "B"
                        if aroma_b > aroma_a
                        else "Equal"
                    )

                    # 3. Structural Stability (ring strain, rigidity)
                    struct_score_a = (
                        10
                        - props_a.get("ring_strain", 0)
                        + (props_a.get("rigidity_score", 0) * 5)
                    )
                    struct_score_b = (
                        10
                        - props_b.get("ring_strain", 0)
                        + (props_b.get("rigidity_score", 0) * 5)
                    )
                    struct_winner = (
                        "A"
                        if struct_score_a > struct_score_b
                        else "B"
                        if struct_score_b > struct_score_a
                        else "Equal"
                    )

                    # 4. Electronic Stability (charge neutrality)
                    charged_a = props_a.get("is_charged", False)
                    charged_b = props_b.get("is_charged", False)
                    if charged_a != charged_b:
                        elec_winner = "A" if not charged_a else "B"
                    else:
                        elec_winner = "Equal"

                    # 5. Kinetic Stability (fewer rotatable bonds = more stable conformation)
                    rot_a = props_a.get("num_rotatable_bonds", 0)
                    rot_b = props_b.get("num_rotatable_bonds", 0)
                    kin_winner = (
                        "A" if rot_a < rot_b else "B" if rot_b < rot_a else "Equal"
                    )

                    # Display comparison table
                    stability_data = [
                        (
                            "🔥 Thermodynamic Stability",
                            f"{energy_a:.2f} kcal/mol",
                            f"{energy_b:.2f} kcal/mol",
                            "Lower energy = More stable",
                            therm_winner,
                        ),
                        (
                            "💫 Aromaticity",
                            f"Cover: {props_a.get('aromatic_coverage', 0):.1%}, Score: {props_a.get('aromatic_score', 0)}, -Reactive: {props_a.get('reactivity_penalty', 0)}",
                            f"Cover: {props_b.get('aromatic_coverage', 0):.1%}, Score: {props_b.get('aromatic_score', 0)}, -Reactive: {props_b.get('reactivity_penalty', 0)}",
                            "Higher coverage + less reactive groups = More stable",
                            aroma_winner,
                        ),
                        (
                            "🏗️ Structural Stability",
                            f"Strain: {props_a.get('ring_strain', 0):.1f}, Rigidity: {props_a.get('rigidity_score', 0):.2f}",
                            f"Strain: {props_b.get('ring_strain', 0):.1f}, Rigidity: {props_b.get('rigidity_score', 0):.2f}",
                            "Lower strain + Higher rigidity = More stable",
                            struct_winner,
                        ),
                        (
                            "⚡ Electronic Stability",
                            f"{'Charged' if charged_a else 'Neutral'}",
                            f"{'Charged' if charged_b else 'Neutral'}",
                            "Neutral = More stable",
                            elec_winner,
                        ),
                        (
                            "🛡️ Kinetic Stability",
                            f"{rot_a} rotatable bonds",
                            f"{rot_b} rotatable bonds",
                            "Fewer rotatable = More stable",
                            kin_winner,
                        ),
                    ]

                    # Display each category
                    for item in stability_data:
                        category, val_a, val_b, note, winner = item
                        with st.expander(f"{category}"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(f"**🔵 Molecule A:** {val_a}")
                            with col2:
                                st.markdown(f"**🔴 Molecule B:** {val_b}")
                            st.caption(f"📝 {note}")
                            if winner != "Equal":
                                icon = "🔵" if winner == "A" else "🔴"
                                st.success(
                                    f"{icon} **Molecule {winner}** is more stable"
                                )
                            else:
                                st.info("⚪ Equal")

                    # Hierarchical Verdict (Priority Order)
                    st.markdown("---")
                    st.subheader("⚖️ Hierarchical Verdict (Priority Order)")

                    # Follow exact priority: Aromaticity → Resonance → Substitution → Ring Strain → Sterics → Inductive
                    verdict_steps = []

                    # Step 1: Aromaticity (ULTIMATE)
                    # Use enhanced aromatic score that includes aromatic coverage and reactive groups
                    aromatic_score_a = props_a.get("aromatic_score", aroma_a * 10)
                    aromatic_score_b = props_b.get("aromatic_score", aroma_b * 10)
                    reactivity_a = props_a.get("reactivity_penalty", 0)
                    reactivity_b = props_b.get("reactivity_penalty", 0)
                    aromatic_coverage_a = props_a.get("aromatic_coverage", 0)
                    aromatic_coverage_b = props_b.get("aromatic_coverage", 0)

                    # Net aromatic stability = aromatic score - reactivity penalty
                    net_aroma_a = aromatic_score_a - reactivity_a
                    net_aroma_b = aromatic_score_b - reactivity_b

                    if net_aroma_a != net_aroma_b:
                        winner = "A" if net_aroma_a > net_aroma_b else "B"
                        verdict_steps.append(
                            (
                                1,
                                "Aromaticity + Reactivity",
                                winner,
                                f"Net: {net_aroma_a:.0f} vs {net_aroma_b:.0f} (Aromatic:{aromatic_score_a} vs {aromatic_score_b} - Reactivity:{reactivity_a} vs {reactivity_b})",
                            )
                        )
                        hierarchy_winner = winner
                        arom_a = props_a.get("aromatic_atoms", 0)
                        arom_b = props_b.get("aromatic_atoms", 0)
                        hierarchy_reason = f"Aromatic stabilization with reactivity penalty: aromatic atoms={arom_a} vs {arom_b}, reactive groups={reactivity_a} vs {reactivity_b}"

                    # Step 2: Resonance (only if Step 1 equal)
                    if len(verdict_steps) == 0:
                        conj_a = props_a.get("conjugated_bonds", 0)
                        conj_b = props_b.get("conjugated_bonds", 0)
                        if conj_a != conj_b:
                            winner = "A" if conj_a > conj_b else "B"
                            verdict_steps.append(
                                (2, "Resonance", winner, "More resonance structures")
                            )
                            hierarchy_winner = winner
                            hierarchy_reason = "Resonance/delocalization stability"

                    # Step 3: Substitution (only if Step 1-2 equal)
                    if len(verdict_steps) == 0:
                        # Count substituted carbons (tertiary + quaternary)
                        sub_a = props_a.get("tertiary_carbons", 0) + props_a.get(
                            "quaternary_carbons", 0
                        )
                        sub_b = props_b.get("tertiary_carbons", 0) + props_b.get(
                            "quaternary_carbons", 0
                        )
                        if sub_a != sub_b:
                            winner = "A" if sub_a > sub_b else "B"
                            verdict_steps.append(
                                (3, "Substitution", winner, "More substituted carbons")
                            )
                            hierarchy_winner = winner
                            hierarchy_reason = "Hyperconjugation stabilization"

                    # Step 4: Ring Strain (only if Steps 1-3 equal)
                    if len(verdict_steps) == 0:
                        strain_a = props_a.get("ring_strain", 0)
                        strain_b = props_b.get("ring_strain", 0)
                        if strain_a != strain_b:
                            winner = "A" if strain_a < strain_b else "B"
                            verdict_steps.append(
                                (4, "Ring Strain", winner, "6-membered > 5 > 7 > 4 > 3")
                            )
                            hierarchy_winner = winner
                            hierarchy_reason = f"Lower ring strain ({min(strain_a, strain_b):.1f} vs {max(strain_a, strain_b):.1f} kcal/mol)"

                    # Step 5: Steric Hindrance (only if Steps 1-4 equal)
                    if len(verdict_steps) == 0:
                        rot_a = props_a.get("num_rotatable_bonds", 0)
                        rot_b = props_b.get("num_rotatable_bonds", 0)
                        if rot_a != rot_b:
                            winner = "A" if rot_a < rot_b else "B"
                            verdict_steps.append(
                                (5, "Steric Hindrance", winner, "Less crowding")
                            )
                            hierarchy_winner = winner
                            hierarchy_reason = (
                                "Lower steric hindrance (fewer rotatable bonds)"
                            )

                    # Step 6: Inductive Effects (tie-breaker)
                    if len(verdict_steps) == 0:
                        charged_a = props_a.get("is_charged", False)
                        charged_b = props_b.get("is_charged", False)
                        if charged_a != charged_b:
                            winner = "A" if not charged_a else "B"
                            verdict_steps.append(
                                (6, "Inductive Effects", winner, "Neutral > Charged")
                            )
                            hierarchy_winner = winner
                            hierarchy_reason = "Charge stability (neutral more stable)"
                        else:
                            tpsa_a = props_a.get("tpsa", 0)
                            tpsa_b = props_b.get("tpsa", 0)
                            if tpsa_a != tpsa_b:
                                winner = "A" if tpsa_a < tpsa_b else "B"
                                verdict_steps.append(
                                    (6, "Inductive Effects", winner, "Lower PSA")
                                )
                                hierarchy_winner = winner
                                hierarchy_reason = "Lower polar surface area"
                            else:
                                verdict_steps.append(
                                    (6, "Inductive Effects", "Equal", "No difference")
                                )
                                hierarchy_winner = "Equal"
                                hierarchy_reason = "All factors equal"

                    # Display hierarchical verdict
                    for step_num, factor_name, winner_val, rule in verdict_steps:
                        if winner_val != "Equal":
                            icon = "🔵" if winner_val == "A" else "🔴"
                            st.markdown(f"""
                            **Step {step_num}: {factor_name}**
                            - Winner: {icon} **Molecule {winner_val}**
                            - Rule: {rule}
                            - Reason: {hierarchy_reason}
                            """)
                        else:
                            st.info(f"**Step {step_num}: {factor_name}** - Both equal")

                    if verdict_steps:
                        final_winner = (
                            hierarchy_winner if "hierarchy_winner" in dir() else "Equal"
                        )
                        st.success(
                            f"🎯 **Overall: Molecule {final_winner} is more stable**"
                        )

                else:
                    st.info("No significant structural differences found.")

                # Detailed structural comparison table with Pandas DataFrame
                if props_a and props_b:
                    st.markdown("---")
                    st.subheader("📋 Detailed Structural Properties")

                    try:
                        import pandas as pd

                        # Build structured table data as lists
                        properties = [
                            "Atoms",
                            "Carbons",
                            "Hydrogens",
                            "SP³ Carbons",
                            "SP² Carbons",
                            "Rings",
                            "Ring Strain",
                            "Aromatic Rings",
                            "Aromatic Atoms",
                            "Conjugated Bonds",
                            "Rotatable Bonds",
                            "Symmetry Score",
                            "Rigidity Score",
                            "H-Bond Donors",
                            "H-Bond Acceptors",
                            "TPSA (Å²)",
                            "Double Bonds",
                            "Triple Bonds",
                            "Heteroatom Ratio",
                            "Steric Crowding (T+Q)",
                        ]

                        val_a = [
                            props_a["num_atoms"],
                            props_a["num_carbons"],
                            props_a["num_hydrogens"],
                            f"{props_a['sp3_carbons']} ({props_a['sp3_fraction'] * 100:.1f}%)",
                            props_a["sp2_carbons"],
                            props_a["num_rings"],
                            f"{props_a['ring_strain']:.1f}",
                            props_a["aromatic_rings"],
                            props_a["aromatic_atoms"],
                            props_a["conjugated_bonds"],
                            props_a["num_rotatable_bonds"],
                            f"{props_a['symmetry_score']:.3f}",
                            f"{props_a['rigidity_score']:.3f}",
                            props_a["h_donors"],
                            props_a["h_acceptors"],
                            f"{props_a['tpsa']:.1f}",
                            props_a["double_bonds"],
                            props_a["triple_bonds"],
                            f"{props_a['heteroatom_ratio']:.2%}",
                            props_a["tertiary_carbons"] + props_a["quaternary_carbons"],
                        ]

                        val_b = [
                            props_b["num_atoms"],
                            props_b["num_carbons"],
                            props_b["num_hydrogens"],
                            f"{props_b['sp3_carbons']} ({props_b['sp3_fraction'] * 100:.1f}%)",
                            props_b["sp2_carbons"],
                            props_b["num_rings"],
                            f"{props_b['ring_strain']:.1f}",
                            props_b["aromatic_rings"],
                            props_b["aromatic_atoms"],
                            props_b["conjugated_bonds"],
                            props_b["num_rotatable_bonds"],
                            f"{props_b['symmetry_score']:.3f}",
                            f"{props_b['rigidity_score']:.3f}",
                            props_b["h_donors"],
                            props_b["h_acceptors"],
                            f"{props_b['tpsa']:.1f}",
                            props_b["double_bonds"],
                            props_b["triple_bonds"],
                            f"{props_b['heteroatom_ratio']:.2%}",
                            props_b["tertiary_carbons"] + props_b["quaternary_carbons"],
                        ]

                        more_stable = [
                            "Neutral",
                            "Neutral",
                            "Neutral",
                            "Higher ✓",
                            "Lower ✓",
                            "Context",
                            "Lower ✓",
                            "Higher ✓",
                            "Higher ✓",
                            "Higher ✓",
                            "Lower ✓",
                            "Lower ✓",
                            "Higher ✓",
                            "Context",
                            "Context",
                            "Lower ✓",
                            "Context",
                            "Context",
                            "Context",
                            "Lower ✓",
                        ]

                        # Determine which molecule is favored (A/B/Equal)
                        def get_favor(i):
                            # Numeric comparison helpers
                            def parse_num(val):
                                if isinstance(val, int):
                                    return val
                                if isinstance(val, float):
                                    return val
                                try:
                                    return float(val)
                                except:
                                    return 0

                            # Define comparison rules per property
                            if i in [0, 1, 2]:  # Atoms, Carbons, Hydrogens - neutral
                                return "═"
                            elif i == 3:  # SP³ - higher is better
                                a_val = props_a["sp3_fraction"]
                                b_val = props_b["sp3_fraction"]
                                if a_val > b_val:
                                    return "A"
                                elif b_val > a_val:
                                    return "B"
                                return "═"
                            elif i == 4:  # SP² - context dependent
                                return "═"
                            elif i == 5:  # Rings - context
                                return "═"
                            elif i == 6:  # Ring Strain - lower is better
                                if props_a["ring_strain"] < props_b["ring_strain"]:
                                    return "A"
                                elif props_b["ring_strain"] < props_a["ring_strain"]:
                                    return "B"
                                return "═"
                            elif (
                                i in [7, 8, 9]
                            ):  # Aromatic rings, atoms, conjugated bonds - higher is better
                                a_val = [
                                    props_a["aromatic_rings"],
                                    props_a["aromatic_atoms"],
                                    props_a["conjugated_bonds"],
                                ][i - 7]
                                b_val = [
                                    props_b["aromatic_rings"],
                                    props_b["aromatic_atoms"],
                                    props_b["conjugated_bonds"],
                                ][i - 7]
                                if a_val > b_val:
                                    return "A"
                                elif b_val > a_val:
                                    return "B"
                                return "═"
                            elif i == 10:  # Rotatable bonds - lower is better
                                if (
                                    props_a["num_rotatable_bonds"]
                                    < props_b["num_rotatable_bonds"]
                                ):
                                    return "A"
                                elif (
                                    props_b["num_rotatable_bonds"]
                                    < props_a["num_rotatable_bonds"]
                                ):
                                    return "B"
                                return "═"
                            elif i == 11:  # Symmetry - lower is better (more symmetric)
                                if (
                                    props_a["symmetry_score"]
                                    < props_b["symmetry_score"]
                                ):
                                    return "A"
                                elif (
                                    props_b["symmetry_score"]
                                    < props_a["symmetry_score"]
                                ):
                                    return "B"
                                return "═"
                            elif i == 12:  # Rigidity - higher is better
                                if (
                                    props_a["rigidity_score"]
                                    > props_b["rigidity_score"]
                                ):
                                    return "A"
                                elif (
                                    props_b["rigidity_score"]
                                    > props_a["rigidity_score"]
                                ):
                                    return "B"
                                return "═"
                            elif i in [13, 14]:  # H-donors/acceptors - context
                                return "═"
                            elif i == 15:  # TPSA - lower is better
                                if props_a["tpsa"] < props_b["tpsa"]:
                                    return "A"
                                elif props_b["tpsa"] < props_a["tpsa"]:
                                    return "B"
                                return "═"
                            elif i in [16, 17]:  # Double/Triple bonds - context
                                return "═"
                            elif i == 18:  # Heteroatom ratio - context
                                return "═"
                            elif i == 19:  # Steric - lower is better
                                a_val = (
                                    props_a["tertiary_carbons"]
                                    + props_a["quaternary_carbons"]
                                )
                                b_val = (
                                    props_b["tertiary_carbons"]
                                    + props_b["quaternary_carbons"]
                                )
                                if a_val < b_val:
                                    return "A"
                                elif b_val < a_val:
                                    return "B"
                                return "═"
                            return "═"

                        favors = [get_favor(i) for i in range(len(properties))]

                        # Create DataFrame
                        df = pd.DataFrame(
                            {
                                "Property": properties,
                                "Molecule A": val_a,
                                "Molecule B": val_b,
                                "More Stable": more_stable,
                                "Favors": favors,
                            }
                        )

                        # Apply styling based on 'Favors' column
                        def style_favors(val):
                            if val == "A":
                                return "color: #2563eb; font-weight: bold"
                            elif val == "B":
                                return "color: #dc2626; font-weight: bold"
                            return "color: #6b7280"

                        def style_stable(val):
                            if "✓" in str(val):
                                return "color: #059669; font-weight: bold"
                            return "color: #6b7280"

                        # Apply styles and display
                        styled_df = (
                            df.style.map(style_favors, subset=["Favors"])
                            .map(style_stable, subset=["More Stable"])
                            .set_properties(
                                **{"text-align": "center"},
                                subset=[
                                    "Molecule A",
                                    "Molecule B",
                                    "More Stable",
                                    "Favors",
                                ],
                            )
                            .set_properties(
                                **{"text-align": "left"}, subset=["Property"]
                            )
                            .set_table_styles(
                                [
                                    {
                                        "selector": "th",
                                        "props": [
                                            ("background-color", "#f3f4f6"),
                                            ("color", "#1f2937"),
                                            ("font-weight", "bold"),
                                            ("text-align", "center"),
                                        ],
                                    },
                                    {
                                        "selector": "td",
                                        "props": [("text-align", "center")],
                                    },
                                    {
                                        "selector": "tr:hover",
                                        "props": [("background-color", "#f9fafb")],
                                    },
                                ]
                            )
                        )

                        st.dataframe(
                            styled_df, use_container_width=True, hide_index=True
                        )

                    except ImportError:
                        st.warning("pandas not available - displaying simpler table")
                        st.markdown("### Property Comparison")
                        st.markdown(
                            "| Property | Molecule A | Molecule B | More Stable | Favors |"
                        )
                        st.markdown("|---|---|---|---|---|")

                # 3D Visualization
                st.markdown("---")
                st.subheader("🫁 3D Optimized Structures")

                vis_col1, vis_col2 = st.columns(2)

                with vis_col1:
                    st.write(f"**Molecule A** ({viz_style})")
                    try:
                        fig_a = mol_to_plotly_figure(
                            mol_a, size=(700, 500), style=viz_style
                        )
                        st.plotly_chart(fig_a, use_container_width=True)
                    except Exception as e:
                        st.error(f"3D viewer error: {e}")

                with vis_col2:
                    st.write(f"**Molecule B** ({viz_style})")
                    try:
                        fig_b = mol_to_plotly_figure(
                            mol_b, size=(700, 500), style=viz_style
                        )
                        st.plotly_chart(fig_b, use_container_width=True)
                    except Exception as e:
                        st.error(f"3D viewer error: {e}")

                # Molecular info
                if advanced_info:
                    st.markdown("---")
                    st.subheader("📋 Advanced Molecular Properties")
                    prop_col1, prop_col2 = st.columns(2)

                    with prop_col1:
                        st.write("**Molecule A**")
                        st.code(f"SMILES: {smiles_a}")
                        st.write(f"**Atoms:** {mol_a.GetNumAtoms()}")
                        st.write(f"**Bonds:** {mol_a.GetNumBonds()}")
                        st.write(f"**MW:** {Descriptors.MolWt(mol_a):.2f} g/mol")
                        st.write(f"**LogP:** {Descriptors.MolLogP(mol_a):.2f}")
                        st.write(f"**H-Bond Donors:** {Descriptors.NumHDonors(mol_a)}")
                        st.write(
                            f"**H-Bond Acceptors:** {Descriptors.NumHAcceptors(mol_a)}"
                        )

                    with prop_col2:
                        st.write("**Molecule B**")
                        st.code(f"SMILES: {smiles_b}")
                        st.write(f"**Atoms:** {mol_b.GetNumAtoms()}")
                        st.write(f"**Bonds:** {mol_b.GetNumBonds()}")
                        st.write(f"**MW:** {Descriptors.MolWt(mol_b):.2f} g/mol")
                        st.write(f"**LogP:** {Descriptors.MolLogP(mol_b):.2f}")
                        st.write(f"**H-Bond Donors:** {Descriptors.NumHDonors(mol_b)}")
                        st.write(
                            f"**H-Bond Acceptors:** {Descriptors.NumHAcceptors(mol_b)}"
                        )
                else:
                    st.markdown("---")
                    st.subheader("📋 Basic Molecular Info")
                    info_col1, info_col2 = st.columns(2)
                    with info_col1:
                        st.write("**Molecule A**")
                        st.code(smiles_a)
                        st.write(
                            f"Atoms: {mol_a.GetNumAtoms()} | Bonds: {mol_a.GetNumBonds()}"
                        )
                    with info_col2:
                        st.write("**Molecule B**")
                        st.code(smiles_b)
                        st.write(
                            f"Atoms: {mol_b.GetNumAtoms()} | Bonds: {mol_b.GetNumBonds()}"
                        )

# Footer
st.markdown("---")
st.caption("Powered by RDKit, MMFF94 force field, and Streamlit")
