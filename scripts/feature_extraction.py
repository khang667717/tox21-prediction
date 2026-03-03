#!/usr/bin/env python3
"""
Feature extraction utilities for Tox21 models.
FIXED: Better SMILES parsing with sanitization
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from typing import Dict, List, Any, Optional
import torch
import warnings

def get_canonical_smiles(smiles: str) -> Optional[str]:
    """Get canonical SMILES string with sanitization."""
    if not smiles or not isinstance(smiles, str):
        return None
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # Try with sanitization=False
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is not None:
                try:
                    Chem.SanitizeMol(mol)
                except:
                    pass
        
        if mol is None:
            return None
            
        return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    except Exception as e:
        print(f"Warning: SMILES parsing error '{smiles}': {e}")
        return None

def extract_morgan_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Extract Morgan fingerprint as bit vector."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(n_bits)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp)
    except:
        return np.zeros(n_bits)

def extract_morgan_fingerprint_counts(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Extract Morgan fingerprint as count vector."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(n_bits)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp)
    except:
        return np.zeros(n_bits)

def smiles_to_graph(smiles: str) -> Optional[Dict[str, Any]]:
    """
    Convert SMILES to graph dictionary for GNN.
    FIXED: Better error handling, works with complex molecules like Aflatoxin.
    """
    from rdkit import Chem
    
    if not smiles:
        return None
    
    try:
        # Try with sanitization first
        mol = Chem.MolFromSmiles(smiles)
        
        # If fails, try without sanitization
        if mol is None:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is not None:
                try:
                    Chem.SanitizeMol(mol)
                except:
                    # Still usable even if sanitization fails
                    pass
        
        if mol is None:
            return None
        
        # Atom features - CẦN 43 FEATURES NHƯ TRONG gnn_model.py
        atom_features = []
        for atom in mol.GetAtoms():
            features = []
            
            # Atomic number (one-hot for common elements) - 10 features
            atomic_num = atom.GetAtomicNum()
            common_atoms = [6, 7, 8, 9, 15, 16, 17, 35, 53]  # C, N, O, F, P, S, Cl, Br, I
            for elem in common_atoms:
                features.append(1.0 if atomic_num == elem else 0.0)
            features.append(1.0 if atomic_num not in common_atoms else 0.0)  # Other
            
            # Degree - 7 features
            try:
                degree = atom.GetDegree()
            except:
                degree = 0
            for d in range(6):  # 0-5
                features.append(1.0 if degree == d else 0.0)
            features.append(1.0 if degree > 5 else 0.0)  # >5
            
            # Formal charge - 1 feature
            try:
                charge = atom.GetFormalCharge()
            except:
                charge = 0
            features.append(float(charge))
            
            # Radical electrons - 1 feature
            try:
                radical = atom.GetNumRadicalElectrons()
            except:
                radical = 0
            features.append(float(radical))
            
            # Hybridization - 6 features
            try:
                hybridization = atom.GetHybridization()
                hybrid_types = [
                    Chem.rdchem.HybridizationType.SP,
                    Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3,
                    Chem.rdchem.HybridizationType.SP3D,
                    Chem.rdchem.HybridizationType.SP3D2
                ]
                for ht in hybrid_types:
                    features.append(1.0 if hybridization == ht else 0.0)
                features.append(1.0 if hybridization not in hybrid_types else 0.0)
            except:
                # Default if hybridization can't be determined
                features.extend([0.0] * 6)
            
            # Aromatic - 1 feature
            try:
                aromatic = atom.GetIsAromatic()
            except:
                aromatic = False
            features.append(float(aromatic))
            
            # Ring information - 7 features
            try:
                features.append(float(atom.IsInRing()))
                features.append(float(atom.IsInRingSize(3)))
                features.append(float(atom.IsInRingSize(4)))
                features.append(float(atom.IsInRingSize(5)))
                features.append(float(atom.IsInRingSize(6)))
                features.append(float(atom.IsInRingSize(7)))
                features.append(float(atom.IsInRingSize(8)))
            except:
                features.extend([0.0] * 7)
            
            # Hydrogen count - 6 features
            try:
                h_count = atom.GetTotalNumHs()
            except:
                h_count = 0
            for h in range(5):  # 0-4
                features.append(1.0 if h_count == h else 0.0)
            features.append(1.0 if h_count > 4 else 0.0)
            
            # Chirality - 1 feature
            try:
                chiral = atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED
            except:
                chiral = False
            features.append(float(chiral))
            
            # Additional features - 3 features
            try:
                mass = atom.GetMass() / 100.0
            except:
                mass = 0.0
            features.append(float(mass))
            
            try:
                implicit_h = atom.GetNumImplicitHs()
            except:
                implicit_h = 0
            features.append(float(implicit_h))
            
            try:
                valence = atom.GetTotalValence()
            except:
                valence = 0
            features.append(float(valence))
            
            # TOTAL: 43 features
            atom_features.append(features)
        
        # Bond features
        edge_indices = []
        edge_features = []
        
        try:
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                
                # Both directions
                edge_indices.append([i, j])
                edge_indices.append([j, i])
                
                # Bond features - 4 features
                bond_type = bond.GetBondType()
                bond_features = [
                    1.0 if bond_type == Chem.rdchem.BondType.SINGLE else 0.0,
                    1.0 if bond_type == Chem.rdchem.BondType.DOUBLE else 0.0,
                    1.0 if bond_type == Chem.rdchem.BondType.TRIPLE else 0.0,
                    1.0 if bond_type == Chem.rdchem.BondType.AROMATIC else 0.0,
                ]
                
                edge_features.append(bond_features)
                edge_features.append(bond_features)
        except:
            pass
        
        if len(edge_indices) == 0:
            # Single atom molecule or error
            edge_indices = [[0, 0]]
            edge_features = [[0.0, 0.0, 0.0, 1.0]]
        
        return {
            'x': np.array(atom_features, dtype=np.float32),
            'edge_index': np.array(edge_indices, dtype=np.int64).T,
            'edge_attr': np.array(edge_features, dtype=np.float32),
        }
    except Exception as e:
        print(f"Warning: Failed to convert SMILES to graph '{smiles}': {e}")
        return None