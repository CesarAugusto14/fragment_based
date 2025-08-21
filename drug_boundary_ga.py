#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Boundary-Aware Chemical Space Exploration Genetic Algorithm

This implementation combines stratified sampling, ECFP4 convex hull boundary detection,
and evolutionary algorithms to systematically explore beyond known chemical space
while maintaining drug-likeness constraints.

Key Features:
- Stratified sampling from ZINC database based on molecular properties
- ECFP4 convex hull boundary detection in high-dimensional chemical space
- Boundary-aware genetic operators for systematic extrapolation
- Multi-objective optimization (QED + Lipinski + Boundary Distance)
- Real-time boundary violation tracking and visualization
"""

import os
import random
import pickle
from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from scipy.spatial import ConvexHull, distance_matrix
from scipy.spatial.distance import cdist

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, QED, Crippen, AllChem, BRICS
from rdkit.Chem.Draw import rdMolDraw2D

from fragmentation import BRICSFragmentLibrary

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

class ChemicalSpaceBoundaryDetector:
    """
    Detects chemical space boundaries using ECFP4 fingerprints and convex hull analysis
    """
    
    def __init__(self, fingerprint_bits: int = 2048, pca_components: int = 50):
        self.fingerprint_bits = fingerprint_bits
        self.pca_components = pca_components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=pca_components, random_state=42)
        
        # Boundary data
        self.reference_fingerprints = None
        self.reference_pca = None
        self.convex_hull = None
        self.boundary_equations = None
        self.centroid = None
        
        # Statistics
        self.boundary_stats = {}
        
    def calculate_ecfp4_fingerprints(self, smiles_list: List[str]) -> np.ndarray:
        """Calculate ECFP4 fingerprints for a list of SMILES"""
        fingerprints = []
        valid_indices = []
        
        print(f"Calculating ECFP4 fingerprints for {len(smiles_list)} molecules...")
        
        for i, smiles in enumerate(smiles_list):
            if i % 10000 == 0:
                print(f"  Processed {i}/{len(smiles_list)} molecules")
                
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    # Generate ECFP4 fingerprint (radius=2, specified bits)
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=self.fingerprint_bits)
                    fingerprints.append(np.array(fp))
                    valid_indices.append(i)
            except:
                continue
        
        print(f"Generated {len(fingerprints)} valid fingerprints")
        return np.array(fingerprints), valid_indices
    
    def fit_boundary(self, smiles_list: List[str]) -> Dict:
        """
        Fit convex hull boundary using ECFP4 fingerprints
        
        Returns boundary statistics and sets internal boundary model
        """
        print("=== CHEMICAL SPACE BOUNDARY DETECTION ===")
        
        # Calculate fingerprints
        fingerprints, valid_indices = self.calculate_ecfp4_fingerprints(smiles_list)
        
        if len(fingerprints) < 10:
            raise ValueError("Need at least 10 valid molecules for boundary detection")
        
        # Store reference fingerprints
        self.reference_fingerprints = fingerprints
        
        # Dimensionality reduction with PCA
        print(f"Applying PCA dimensionality reduction ({self.fingerprint_bits} -> {self.pca_components} dims)")
        fingerprints_scaled = self.scaler.fit_transform(fingerprints)
        fingerprints_pca = self.pca.fit_transform(fingerprints_scaled)
        self.reference_pca = fingerprints_pca
        
        # Calculate centroid
        self.centroid = np.mean(fingerprints_pca, axis=0)
        
        # Convex hull in PCA space
        print("Computing convex hull in PCA space...")
        try:
            hull = ConvexHull(fingerprints_pca)
            self.convex_hull = hull
            
            # Store boundary equations (hyperplanes)
            self.boundary_equations = hull.equations
            
            # Calculate boundary statistics
            hull_volume = hull.volume if hasattr(hull, 'volume') else 0
            hull_vertices = len(hull.vertices)
            hull_simplices = len(hull.simplices)
            
            # Calculate distances from centroid to boundary
            boundary_distances = []
            for vertex_idx in hull.vertices:
                vertex = fingerprints_pca[vertex_idx]
                dist = np.linalg.norm(vertex - self.centroid)
                boundary_distances.append(dist)
            
            self.boundary_stats = {
                'total_molecules': len(smiles_list),
                'valid_molecules': len(fingerprints),
                'hull_volume': hull_volume,
                'hull_vertices': hull_vertices,
                'hull_simplices': hull_simplices,
                'avg_boundary_distance': np.mean(boundary_distances),
                'max_boundary_distance': np.max(boundary_distances),
                'min_boundary_distance': np.min(boundary_distances),
                'pca_explained_variance': self.pca.explained_variance_ratio_.sum()
            }
            
            print(f"Boundary detection completed:")
            print(f"  Valid molecules: {len(fingerprints)}")
            print(f"  Hull vertices: {hull_vertices}")
            print(f"  Hull simplices: {hull_simplices}")
            print(f"  PCA explained variance: {self.boundary_stats['pca_explained_variance']:.3f}")
            print(f"  Avg boundary distance: {self.boundary_stats['avg_boundary_distance']:.3f}")
            
            return self.boundary_stats
            
        except Exception as e:
            print(f"Convex hull computation failed: {e}")
            # Fallback: use distance-based boundary
            return self._fit_distance_based_boundary(fingerprints_pca)
    
    def _fit_distance_based_boundary(self, fingerprints_pca: np.ndarray) -> Dict:
        """Fallback boundary detection using distance from centroid"""
        print("Using distance-based boundary detection (fallback)")
        
        distances = np.linalg.norm(fingerprints_pca - self.centroid, axis=1)
        boundary_threshold = np.percentile(distances, 95)  # 95th percentile as boundary
        
        self.boundary_stats = {
            'total_molecules': len(fingerprints_pca),
            'valid_molecules': len(fingerprints_pca),
            'boundary_type': 'distance_based',
            'boundary_threshold': boundary_threshold,
            'avg_distance': np.mean(distances),
            'max_distance': np.max(distances),
            'pca_explained_variance': self.pca.explained_variance_ratio_.sum()
        }
        
        return self.boundary_stats
    
    def calculate_boundary_distance(self, smiles: str) -> float:
        """
        Calculate distance from molecule to chemical space boundary
        
        Positive values = outside boundary (exploration)
        Negative values = inside boundary (known space)
        Zero = on boundary
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                return -999  # Invalid molecule penalty
            
            # Calculate ECFP4 fingerprint
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=self.fingerprint_bits)
            fp_array = np.array(fp).reshape(1, -1)
            
            # Transform to PCA space
            fp_scaled = self.scaler.transform(fp_array)
            fp_pca = self.pca.transform(fp_scaled)
            
            if self.convex_hull is not None:
                # Use convex hull boundary
                point = fp_pca[0]
                
                # Calculate signed distance to each hyperplane
                distances = []
                for equation in self.boundary_equations:
                    # equation format: [a1, a2, ..., an, b] where ax + b = 0
                    coeffs = equation[:-1]
                    const = equation[-1]
                    
                    # Distance = (ax + b) / ||a||
                    dist = (np.dot(coeffs, point) + const) / np.linalg.norm(coeffs)
                    distances.append(dist)
                
                # Return minimum distance (closest to boundary)
                # Positive = outside hull, Negative = inside hull
                return min(distances)
                
            else:
                # Use distance-based boundary (fallback)
                dist_to_centroid = np.linalg.norm(fp_pca[0] - self.centroid)
                boundary_threshold = self.boundary_stats.get('boundary_threshold', 1.0)
                
                # Return signed distance
                return dist_to_centroid - boundary_threshold
                
        except Exception as e:
            return -999  # Error penalty
    
    def is_outside_boundary(self, smiles: str, tolerance: float = 0.0) -> bool:
        """Check if molecule is outside the known chemical space boundary"""
        boundary_dist = self.calculate_boundary_distance(smiles)
        return boundary_dist > tolerance
    
    def visualize_boundary_2d(self, sample_molecules: List[str] = None, save_path: str = "boundary_visualization.png"):
        """Create 2D visualization of chemical space boundary"""
        if self.reference_pca is None:
            print("No boundary data available for visualization")
            return
        
        # Use first 2 PCA components for visualization
        pca_2d = self.reference_pca[:, :2]
        
        plt.figure(figsize=(12, 10))
        
        # Plot reference molecules
        plt.scatter(pca_2d[:, 0], pca_2d[:, 1], c='lightblue', alpha=0.6, s=20, label='Reference Molecules')
        
        # Plot centroid
        centroid_2d = self.centroid[:2]
        plt.scatter(centroid_2d[0], centroid_2d[1], c='red', s=100, marker='*', label='Centroid', zorder=5)
        
        # Plot convex hull if available
        if self.convex_hull is not None:
            # Get 2D hull vertices
            vertices_2d = pca_2d[self.convex_hull.vertices]
            hull_2d = ConvexHull(vertices_2d)
            
            # Plot hull boundary
            for simplex in hull_2d.simplices:
                plt.plot(vertices_2d[simplex, 0], vertices_2d[simplex, 1], 'k-', alpha=0.7, linewidth=2)
            
            plt.scatter(vertices_2d[:, 0], vertices_2d[:, 1], c='orange', s=50, 
                       label='Boundary Vertices', zorder=4)
        
        # Plot sample molecules if provided
        if sample_molecules:
            sample_fps = []
            sample_labels = []
            
            for smiles in sample_molecules[:50]:  # Limit to 50 for clarity
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=self.fingerprint_bits)
                        fp_array = np.array(fp).reshape(1, -1)
                        fp_scaled = self.scaler.transform(fp_array)
                        fp_pca = self.pca.transform(fp_scaled)
                        
                        sample_fps.append(fp_pca[0, :2])
                        
                        # Determine if outside boundary
                        boundary_dist = self.calculate_boundary_distance(smiles)
                        sample_labels.append('Outside' if boundary_dist > 0 else 'Inside')
                except:
                    continue
            
            if sample_fps:
                sample_fps = np.array(sample_fps)
                
                # Color by boundary status
                colors = ['green' if label == 'Outside' else 'purple' for label in sample_labels]
                plt.scatter(sample_fps[:, 0], sample_fps[:, 1], c=colors, s=100, alpha=0.8, 
                           marker='^', label='Sample Molecules', zorder=6)
        
        plt.xlabel(f'PCA Component 1 ({self.pca.explained_variance_ratio_[0]:.3f} variance)', fontsize=12)
        plt.ylabel(f'PCA Component 2 ({self.pca.explained_variance_ratio_[1]:.3f} variance)', fontsize=12)
        plt.title('Chemical Space Boundary Visualization\n(ECFP4 + PCA + Convex Hull)', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Boundary visualization saved to: {save_path}")

class StratifiedZINCLoader:
    """
    Loads ZINC dataset with stratified sampling based on molecular properties
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.property_bins = {}
        self.stratification_features = ['mw_bin', 'logp_bin', 'qed_bin']
        
    def calculate_molecular_properties(self, smiles_series: pd.Series) -> pd.DataFrame:
        """Calculate molecular properties for stratification"""
        print("Calculating molecular properties for stratification...")
        
        properties = []
        valid_indices = []
        
        for i, smiles in enumerate(smiles_series):
            if i % 10000 == 0:
                print(f"  Processed {i}/{len(smiles_series)} molecules")
                
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    props = {
                        'smiles': smiles,
                        'mw': Descriptors.MolWt(mol),
                        'logp': Crippen.MolLogP(mol),
                        'qed': QED.qed(mol),
                        'hba': Descriptors.NumHAcceptors(mol),
                        'hbd': Descriptors.NumHDonors(mol),
                        'valid': True
                    }
                    properties.append(props)
                    valid_indices.append(i)
            except:
                continue
        
        df = pd.DataFrame(properties)
        print(f"Calculated properties for {len(df)} valid molecules")
        return df, valid_indices
    
    def create_stratification_bins(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create stratification bins based on molecular properties"""
        print("Creating stratification bins...")
        
        # Molecular weight bins
        df['mw_bin'] = pd.cut(df['mw'], bins=[0, 200, 300, 400, 500, 1000], 
                             labels=['<200', '200-300', '300-400', '400-500', '>500'])
        
        # LogP bins
        df['logp_bin'] = pd.cut(df['logp'], bins=[-10, -1, 1, 3, 5, 10], 
                               labels=['<-1', '-1-1', '1-3', '3-5', '>5'])
        
        # QED bins
        df['qed_bin'] = pd.cut(df['qed'], bins=[0, 0.3, 0.5, 0.7, 0.9, 1.0], 
                              labels=['<0.3', '0.3-0.5', '0.5-0.7', '0.7-0.9', '>0.9'])
        
        # Create combined stratification key
        df['strat_key'] = df['mw_bin'].astype(str) + '_' + \
                         df['logp_bin'].astype(str) + '_' + \
                         df['qed_bin'].astype(str)
        
        # Remove any rows with NaN bins
        df = df.dropna(subset=['mw_bin', 'logp_bin', 'qed_bin'])
        
        print(f"Created {df['strat_key'].nunique()} unique stratification groups")
        print("Distribution of stratification groups:")
        print(df['strat_key'].value_counts().head(10))
        
        return df
    
    def load_stratified_sample(self, parquet_path: str, sample_size: int = 50000, 
                              cache_path: str = "./stratified.parquet") -> pd.DataFrame:
        """Load stratified sample from ZINC parquet file with caching"""
        
        # Check if cached stratified sample exists
        if os.path.exists(cache_path):
            print(f"Loading cached stratified sample from {cache_path}")
            try:
                cached_df = pd.read_parquet(cache_path)
                print(f"Loaded {len(cached_df)} molecules from cache")
                
                # Check if cache has enough molecules
                if len(cached_df) >= sample_size:
                    # If cache has more molecules than needed, sample from it
                    if len(cached_df) > sample_size:
                        print(f"Sampling {sample_size} molecules from cached {len(cached_df)} molecules")
                        sampled_df = cached_df.sample(n=sample_size, random_state=self.random_state)
                        return sampled_df
                    else:
                        return cached_df
                else:
                    print(f"Warning: Cache only has {len(cached_df)} molecules, need {sample_size}")
                    print("Will regenerate stratified sample...")
            except Exception as e:
                print(f"Error loading cached file: {e}")
                print("Will regenerate stratified sample...")
        else:
            print(f"No cached stratified sample found at {cache_path}")
            print("Will create new stratified sample...")
        
        # Generate new stratified sample
        print(f"Loading ZINC dataset from {parquet_path}")
        
        # Load full dataset
        df_full = pd.read_parquet(parquet_path)
        print(f"Loaded {len(df_full)} molecules from ZINC")
        
        # Get SMILES column
        smiles_col = 'smiles' if 'smiles' in df_full.columns else df_full.columns[0]
        print(f"Using column '{smiles_col}' for SMILES")
        
        # Calculate properties for stratification
        props_df, valid_indices = self.calculate_molecular_properties(df_full[smiles_col])
        
        if len(props_df) < sample_size:
            print(f"Warning: Only {len(props_df)} valid molecules available, using all")
            stratified_sample = props_df
        else:
            # Create stratification bins
            props_df = self.create_stratification_bins(props_df)
            
            # Filter groups with sufficient data
            group_counts = props_df['strat_key'].value_counts()
            valid_groups = group_counts[group_counts >= 5].index  # At least 5 molecules per group
            props_df = props_df[props_df['strat_key'].isin(valid_groups)]
            
            print(f"Using {len(valid_groups)} groups with sufficient data")
            
            # Stratified sampling
            try:
                splitter = StratifiedShuffleSplit(n_splits=1, test_size=sample_size, 
                                                random_state=self.random_state)
                
                # Use stratification key for splitting
                train_idx, sample_idx = next(splitter.split(props_df, props_df['strat_key']))
                
                stratified_sample = props_df.iloc[sample_idx].copy()
                
                print(f"Stratified sampling completed: {len(stratified_sample)} molecules")
                print("Sample distribution:")
                print(stratified_sample['strat_key'].value_counts().head(10))
                
            except Exception as e:
                print(f"Stratified sampling failed: {e}")
                print("Using random sampling as fallback")
                stratified_sample = props_df.sample(n=min(sample_size, len(props_df)), random_state=self.random_state)
        
        # Save to cache
        try:
            print(f"Saving stratified sample to cache: {cache_path}")
            stratified_sample.to_parquet(cache_path, index=False)
            print(f"Successfully cached {len(stratified_sample)} molecules")
        except Exception as e:
            print(f"Warning: Could not save cache file: {e}")
        
        return stratified_sample

class BoundaryExplorationGA:
    """
    Genetic Algorithm for systematic exploration beyond chemical space boundaries
    """
    
    def __init__(self,
                 fragment_library: BRICSFragmentLibrary,
                 boundary_detector: ChemicalSpaceBoundaryDetector,
                 population_size: int = 50,
                 generations: int = 50,
                 mutation_rate: float = 0.3,
                 crossover_rate: float = 0.7,
                 boundary_weight: float = 0.3,  # Weight for boundary exploration
                 diversity_weight: float = 0.2,
                 max_identical: int = 5,
                 output_dir: str = "boundary_exploration_output"
                 ):
        
        self.frag_lib = fragment_library
        self.boundary_detector = boundary_detector
        
        # Get GA-ready fragments
        ga_frags = self.frag_lib.get_genetic_algorithm_fragments(min_frequency=3)
        self.terminals = [f['clean_smiles'] for f in ga_frags['terminals']]
        self.linkers = [f['clean_smiles'] for f in ga_frags['linkers']]
        self.branches = [f['clean_smiles'] for f in ga_frags['branches']]
        
        # GA parameters
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.boundary_weight = boundary_weight
        self.diversity_weight = diversity_weight
        self.max_identical = max_identical
        
        # Tracking
        self.generation_stats = []
        self.exploration_history = []
        
        # Create output directory
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"Boundary Exploration GA initialized:")
        print(f"  Fragments - Terminals: {len(self.terminals)}, Linkers: {len(self.linkers)}")
        print(f"  Boundary weight: {boundary_weight}, Diversity weight: {diversity_weight}")
    
    def is_valid_smiles(self, smiles: str) -> bool:
        """Check if SMILES represents a valid, drug-like molecule"""
        if not smiles:
            return False
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
            
            Chem.SanitizeMol(mol)
            
            # Basic drug-likeness filters
            mw = Descriptors.MolWt(mol)
            if mw > 600 or mw < 50:
                return False
                
            if mol.GetNumAtoms() > 50 or mol.GetNumAtoms() < 3:
                return False
                
            return True
        except:
            return False
    
    def calculate_molecular_properties(self, smiles: str) -> Dict:
        """Calculate comprehensive molecular properties"""
        if not self.is_valid_smiles(smiles):
            return None
            
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                return None
                
            properties = {
                'smiles': smiles,
                'valid': True,
                'qed': QED.qed(mol),
                'logp': Crippen.MolLogP(mol),
                'mw': Descriptors.MolWt(mol),
                'hba': Descriptors.NumHAcceptors(mol),
                'hbd': Descriptors.NumHDonors(mol),
                'boundary_distance': self.boundary_detector.calculate_boundary_distance(smiles),
                'outside_boundary': self.boundary_detector.is_outside_boundary(smiles)
            }
            return properties
        except:
            return None
    
    def boundary_aware_fitness(self, smiles: str) -> float:
        """
        Multi-objective fitness function:
        - QED score (drug-likeness)
        - Lipinski compliance  
        - Boundary exploration bonus
        """
        if not smiles:
            return 0
            
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                return 0
                
            # QED score (0-1, higher is better)
            qed_score = QED.qed(mol)
            
            # Lipinski Rule of 5 compliance
            mw = Descriptors.MolWt(mol)
            logp = Crippen.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            
            # Count violations
            violations = 0
            if mw > 500: violations += 1
            if logp > 5: violations += 1
            if hbd > 5: violations += 1
            if hba > 10: violations += 1
            
            # Lipinski score (higher = fewer violations)
            lipinski_score = (4 - violations) / 4.0
            
            # Boundary exploration component
            boundary_distance = self.boundary_detector.calculate_boundary_distance(smiles)
            
            # Normalize boundary distance (sigmoid transformation)
            # Positive distances (outside boundary) get bonus
            if boundary_distance > 0:
                boundary_bonus = 1.0 / (1.0 + np.exp(-boundary_distance))  # Sigmoid, max ~1.0
            else:
                boundary_bonus = 0.5 / (1.0 + np.exp(boundary_distance))   # Sigmoid, max ~0.5
            
            # Combined fitness
            base_fitness = 0.5 * qed_score + 0.2 * lipinski_score
            exploration_bonus = self.boundary_weight * boundary_bonus
            
            total_fitness = base_fitness + exploration_bonus
            
            return total_fitness
            
        except:
            return 0
    
    def boundary_guided_mutation(self, smiles: str) -> str:
        """Mutation strategy that encourages boundary exploration"""
        if not smiles or random.random() > self.mutation_rate:
            return smiles
        
        # Check current boundary status
        boundary_dist = self.boundary_detector.calculate_boundary_distance(smiles)
        
        # Strategy selection based on boundary position
        if boundary_dist < 0:  # Inside boundary - encourage exploration
            exploration_strategies = [
                self._add_novel_fragment,
                self._extend_molecular_structure,
                self._introduce_rare_functionality
            ]
            strategy = random.choice(exploration_strategies)
        else:  # Outside boundary - maintain novelty but ensure validity
            refinement_strategies = [
                self._refine_structure,
                self._swap_similar_fragments,
                self._optimize_druglikeness
            ]
            strategy = random.choice(refinement_strategies)
        
        try:
            result = strategy(smiles)
            if result != smiles and self.is_valid_smiles(result):
                return result
        except:
            pass
        
        # Fallback to standard mutation
        return self._standard_mutation(smiles)
    
    def _add_novel_fragment(self, smiles: str) -> str:
        """Add fragments to push molecule toward boundary"""
        if not self.terminals and not self.linkers:
            return smiles
            
        # Prefer less common fragments for novelty
        fragment_pool = self.terminals + self.linkers
        chosen_fragment = random.choice(fragment_pool)
        
        candidates = [
            f"{smiles}{chosen_fragment}",
            f"{chosen_fragment}{smiles}",
            f"C{smiles}{chosen_fragment}",
            f"{chosen_fragment}C{smiles}",
            f"N{smiles}{chosen_fragment}",
            f"{smiles}O{chosen_fragment}"
        ]
        
        for candidate in candidates:
            if self.is_valid_smiles(candidate):
                return candidate
        
        return smiles
    
    def _extend_molecular_structure(self, smiles: str) -> str:
        """Extend molecular structure with rings or chains"""
        extensions = [
            "c1ccccc1",      # benzene
            "C1CCCCC1",      # cyclohexane
            "c1ccncc1",      # pyridine
            "CCCC",          # butyl chain
            "c1ccoc1",       # furan
            "C1CCNCC1"       # piperidine
        ]
        
        extension = random.choice(extensions)
        
        candidates = [
            f"{smiles}{extension}",
            f"{extension}{smiles}",
            f"C{smiles}C{extension}",
            f"N{smiles}N{extension}"
        ]
        
        for candidate in candidates:
            if self.is_valid_smiles(candidate):
                return candidate
        
        return smiles
    
    def _introduce_rare_functionality(self, smiles: str) -> str:
        """Introduce less common functional groups"""
        rare_groups = ["S", "F", "Cl", "Br", "I", "[NH2]", "C#C", "C#N"]
        group = random.choice(rare_groups)
        
        candidates = [
            f"{smiles}{group}",
            f"{group}{smiles}",
            smiles.replace("C", group, 1) if "C" in smiles else f"{smiles}{group}"
        ]
        
        for candidate in candidates:
            if self.is_valid_smiles(candidate):
                return candidate
        
        return smiles
    
    def _refine_structure(self, smiles: str) -> str:
        """Refine structure while maintaining novelty"""
        # Small modifications to maintain boundary position
        modifications = [
            lambda s: s.replace("C", "N", 1) if "C" in s else s,
            lambda s: s.replace("N", "O", 1) if "N" in s else s,
            lambda s: f"C{s}",
            lambda s: f"{s}C"
        ]
        
        mod = random.choice(modifications)
        result = mod(smiles)
        
        if self.is_valid_smiles(result):
            return result
        
        return smiles
    
    def _swap_similar_fragments(self, smiles: str) -> str:
        """Swap fragments with chemically similar ones"""
        # This is a simplified version - could be enhanced with fragment analysis
        if self.terminals:
            replacement = random.choice(self.terminals)
            candidates = [
                f"{replacement}{smiles[5:]}",  # Replace beginning
                f"{smiles[:-5]}{replacement}", # Replace end
                replacement  # Complete replacement
            ]
            
            for candidate in candidates:
                if self.is_valid_smiles(candidate):
                    return candidate
        
        return smiles
    
    def _optimize_druglikeness(self, smiles: str) -> str:
        """Small modifications to improve drug-likeness"""
        # Remove potentially problematic groups
        optimizations = [
            ("Br", "F"),
            ("I", "Cl"),
            ("CCCC", "CC"),  # Shorten chains
            ("CCC", "CO"),   # Add polarity
        ]
        
        for old, new in optimizations:
            if old in smiles:
                candidate = smiles.replace(old, new, 1)
                if self.is_valid_smiles(candidate):
                    return candidate
        
        return smiles
    
    def _standard_mutation(self, smiles: str) -> str:
        """Standard mutation fallback"""
        if self.terminals and random.random() < 0.5:
            replacement = random.choice(self.terminals)
            if self.is_valid_smiles(replacement):
                return replacement
        
        modifications = [f"C{smiles}", f"{smiles}C", f"N{smiles}", f"{smiles}O"]
        
        for mod in modifications:
            if self.is_valid_smiles(mod):
                return mod
        
        return smiles
    
    def boundary_aware_crossover(self, parent1: str, parent2: str) -> Tuple[str, str]:
        """Crossover that considers boundary exploration potential"""
        if random.random() > self.crossover_rate:
            return parent1, parent2
        
        # Analyze boundary positions of parents
        boundary1 = self.boundary_detector.calculate_boundary_distance(parent1)
        boundary2 = self.boundary_detector.calculate_boundary_distance(parent2)
        
        # If both parents are outside boundary, preserve exploration
        if boundary1 > 0 and boundary2 > 0:
            return self._exploration_preserving_crossover(parent1, parent2)
        
        # If one parent is exploring, bias toward exploration
        elif boundary1 > 0 or boundary2 > 0:
            return self._exploration_biased_crossover(parent1, parent2)
        
        # Standard crossover for parents inside boundary
        else:
            return self._standard_crossover(parent1, parent2)
    
    def _exploration_preserving_crossover(self, parent1: str, parent2: str) -> Tuple[str, str]:
        """Crossover that preserves exploration characteristics"""
        try:
            # Fragment-based recombination
            if self.linkers:
                linker = random.choice(self.linkers)
                
                # Create hybrid molecules
                candidates = [
                    f"{parent1}{linker}{parent2}",
                    f"{parent2}{linker}{parent1}",
                    f"C{parent1}{linker}C{parent2}",
                    f"N{parent1}{linker}O{parent2}"
                ]
                
                valid_children = []
                for candidate in candidates:
                    if self.is_valid_smiles(candidate):
                        valid_children.append(candidate)
                
                if len(valid_children) >= 2:
                    return valid_children[0], valid_children[1]
                elif len(valid_children) == 1:
                    return valid_children[0], parent2
                    
        except:
            pass
        
        return parent1, parent2
    
    def _exploration_biased_crossover(self, parent1: str, parent2: str) -> Tuple[str, str]:
        """Crossover biased toward the exploring parent"""
        # Identify which parent is exploring
        boundary1 = self.boundary_detector.calculate_boundary_distance(parent1)
        boundary2 = self.boundary_detector.calculate_boundary_distance(parent2)
        
        explorer = parent1 if boundary1 > boundary2 else parent2
        conservative = parent2 if boundary1 > boundary2 else parent1
        
        try:
            # Bias toward explorer characteristics
            candidates = [
                f"{explorer}{conservative}",
                f"{conservative}{explorer}",
                f"C{explorer}",
                f"{explorer}C"
            ]
            
            valid_children = []
            for candidate in candidates:
                if self.is_valid_smiles(candidate):
                    valid_children.append(candidate)
            
            if len(valid_children) >= 2:
                return valid_children[0], valid_children[1]
            elif len(valid_children) == 1:
                return valid_children[0], conservative
                
        except:
            pass
        
        return parent1, parent2
    
    def _standard_crossover(self, parent1: str, parent2: str) -> Tuple[str, str]:
        """Standard crossover operation"""
        try:
            mid1 = len(parent1) // 2
            mid2 = len(parent2) // 2
            
            child1_candidates = [
                parent1[:mid1] + parent2[mid2:],
                parent2[:mid2] + parent1[mid1:],
                f"C{parent1}{parent2}",
                parent1
            ]
            
            child2_candidates = [
                parent2[:mid2] + parent1[mid1:],
                parent1[:mid1] + parent2[mid2:],
                f"C{parent2}{parent1}",
                parent2
            ]
            
            child1 = parent1
            child2 = parent2
            
            for candidate in child1_candidates:
                if self.is_valid_smiles(candidate):
                    child1 = candidate
                    break
                    
            for candidate in child2_candidates:
                if self.is_valid_smiles(candidate):
                    child2 = candidate
                    break
            
            return child1, child2
            
        except:
            return parent1, parent2
    
    def create_boundary_exploring_molecule(self) -> str:
        """Create random molecule with bias toward boundary exploration"""
        max_attempts = 50
        
        for _ in range(max_attempts):
            try:
                # Strategy 1: Combine multiple fragments for complexity
                if random.random() < 0.4 and len(self.terminals) > 1 and len(self.linkers) > 0:
                    terminal1 = random.choice(self.terminals)
                    terminal2 = random.choice(self.terminals)
                    linker = random.choice(self.linkers)
                    
                    candidates = [
                        f"{terminal1}{linker}{terminal2}",
                        f"{terminal2}{linker}{terminal1}",
                        f"C{terminal1}{linker}C{terminal2}",
                        f"N{terminal1}{linker}O{terminal2}"
                    ]
                    
                    for candidate in candidates:
                        if self.is_valid_smiles(candidate):
                            return candidate
                
                # Strategy 2: Use single fragment
                elif random.random() < 0.3 and self.terminals:
                    candidate = random.choice(self.terminals)
                    if self.is_valid_smiles(candidate):
                        return candidate
                
                # Strategy 3: Modified linker
                elif self.linkers:
                    linker = random.choice(self.linkers)
                    modifications = [linker, f"C{linker}", f"{linker}C", f"N{linker}O"]
                    
                    for mod in modifications:
                        if self.is_valid_smiles(mod):
                            return mod
                            
            except:
                continue
        
        return "CCO"  # Fallback
    
    def calculate_population_diversity(self, population: List[str]) -> float:
        """Calculate population diversity using ECFP4 fingerprints"""
        valid_mols = [smiles for smiles in population if self.is_valid_smiles(smiles)]
        
        if len(valid_mols) < 2:
            return 0.0
        
        try:
            fps = []
            for smiles in valid_mols:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                    fps.append(fp)
            
            if len(fps) < 2:
                return 0.0
            
            similarities = []
            for i in range(len(fps)):
                for j in range(i+1, len(fps)):
                    sim = AllChem.DataStructs.TanimotoSimilarity(fps[i], fps[j])
                    similarities.append(sim)
            
            return 1.0 - np.mean(similarities)
            
        except:
            return 0.0
    
    def diversity_selection(self, population: List[str], fitness_scores: List[Tuple[str, float]]) -> List[str]:
        """Selection with diversity preservation and boundary awareness"""
        # Count identical molecules
        mol_counts = Counter(population)
        
        # Filter excess duplicates
        filtered_survivors = []
        for mol, score in fitness_scores:
            if mol_counts[mol] <= self.max_identical:
                filtered_survivors.append((mol, score))
            elif random.random() < 0.1:
                filtered_survivors.append((mol, score))
        
        # Ensure minimum population
        if len(filtered_survivors) < max(5, self.population_size // 4):
            additional = [item for item in fitness_scores if item not in filtered_survivors]
            filtered_survivors.extend(additional[:10])
        
        # Separate by boundary status
        inside_boundary = []
        outside_boundary = []
        
        for mol, score in filtered_survivors:
            if self.boundary_detector.is_outside_boundary(mol):
                outside_boundary.append((mol, score))
            else:
                inside_boundary.append((mol, score))
        
        # Preserve exploration molecules (outside boundary)
        survivors = []
        
        # Always keep top explorers
        outside_boundary.sort(key=lambda x: x[1], reverse=True)
        num_explorers = min(len(outside_boundary), max(3, self.population_size // 4))
        survivors.extend([mol for mol, _ in outside_boundary[:num_explorers]])
        
        # Add best inside-boundary molecules
        inside_boundary.sort(key=lambda x: x[1], reverse=True)
        remaining_slots = max(5, len(filtered_survivors)//2) - len(survivors)
        survivors.extend([mol for mol, _ in inside_boundary[:remaining_slots]])
        
        return survivors
    
    def save_generation_data(self, generation: int, population: List[str]):
        """Save generation data with boundary information"""
        filename = os.path.join(self.output_dir, f"generation_{generation:03d}.txt")
        
        with open(filename, 'w') as f:
            f.write("SMILES\tValid\tQED\tLogP\tMW\tHBA\tHBD\tBoundaryDist\tOutsideBoundary\n")
            
            for smiles in population:
                props = self.calculate_molecular_properties(smiles)
                if props:
                    f.write(f"{props['smiles']}\t{props['valid']}\t{props['qed']:.4f}\t"
                           f"{props['logp']:.4f}\t{props['mw']:.2f}\t{props['hba']}\t{props['hbd']}\t"
                           f"{props['boundary_distance']:.4f}\t{props['outside_boundary']}\n")
                else:
                    f.write(f"{smiles}\tFalse\t0.0000\t0.0000\t0.00\t0\t0\t-999.0000\tFalse\n")
    
    def collect_generation_data(self, generation: int, population: List[str]) -> List[Dict]:
        """Collect generation data with boundary information"""
        valid_data = []
        for smiles in population:
            if self.is_valid_smiles(smiles):
                props = self.calculate_molecular_properties(smiles)
                if props and props['valid']:
                    props['generation'] = generation
                    valid_data.append(props)
        return valid_data
    
    def run_boundary_exploration(self) -> List[Dict]:
        """Run the boundary exploration genetic algorithm"""
        print("=== BOUNDARY EXPLORATION GENETIC ALGORITHM ===")
        print(f"Population size: {self.population_size}")
        print(f"Generations: {self.generations}")
        print(f"Boundary weight: {self.boundary_weight}")
        print(f"Output directory: {self.output_dir}")
        
        # Initialize population
        population = []
        attempts = 0
        max_init_attempts = self.population_size * 10
        
        print("Initializing population...")
        while len(population) < self.population_size and attempts < max_init_attempts:
            mol = self.create_boundary_exploring_molecule()
            if mol and self.is_valid_smiles(mol):
                population.append(mol)
            attempts += 1
        
        # Fill remaining slots
        simple_mols = ["CCO", "CC", "CO", "CCC", "CCN", "CN", "C=O"]
        while len(population) < self.population_size:
            population.append(random.choice(simple_mols))
        
        print(f"Initial population: {len(population)} molecules")
        
        # Track all generation data
        all_generation_data = []
        
        # Evolution loop
        for gen in range(self.generations):
            print(f"\n=== Generation {gen + 1}/{self.generations} ===")
            
            # Save generation data
            self.save_generation_data(gen + 1, population)
            
            # Collect data for analysis
            gen_data = self.collect_generation_data(gen + 1, population)
            all_generation_data.append(gen_data)
            
            # Calculate statistics
            valid_count = sum(1 for mol in population if self.is_valid_smiles(mol))
            unique_count = len(set(population))
            diversity = self.calculate_population_diversity(population)
            
            # Boundary exploration statistics
            outside_boundary_count = 0
            boundary_distances = []
            
            for mol in population:
                if self.is_valid_smiles(mol):
                    boundary_dist = self.boundary_detector.calculate_boundary_distance(mol)
                    boundary_distances.append(boundary_dist)
                    if boundary_dist > 0:
                        outside_boundary_count += 1
            
            exploration_rate = outside_boundary_count / valid_count if valid_count > 0 else 0
            avg_boundary_dist = np.mean(boundary_distances) if boundary_distances else 0
            
            # Evaluate fitness
            fitness_scores = []
            for mol in population:
                if self.is_valid_smiles(mol):
                    fitness = self.boundary_aware_fitness(mol)
                    fitness_scores.append((mol, fitness))
                else:
                    fitness_scores.append((mol, 0.0))
            
            fitness_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Track statistics
            fitnesses = [score for _, score in fitness_scores]
            best_mol, best_fitness = fitness_scores[0]
            avg_fitness = np.mean(fitnesses)
            
            # Calculate drug-likeness statistics
            qed_scores = []
            lipinski_compliant = 0
            
            for mol in population:
                if self.is_valid_smiles(mol):
                    mol_obj = Chem.MolFromSmiles(mol)
                    if mol_obj:
                        qed = QED.qed(mol_obj)
                        qed_scores.append(qed)
                        
                        # Check Lipinski compliance
                        mw = Descriptors.MolWt(mol_obj)
                        logp = Crippen.MolLogP(mol_obj)
                        hbd = Descriptors.NumHDonors(mol_obj)
                        hba = Descriptors.NumHAcceptors(mol_obj)
                        violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
                        
                        if violations == 0:
                            lipinski_compliant += 1
            
            avg_qed = np.mean(qed_scores) if qed_scores else 0
            lipinski_rate = lipinski_compliant / valid_count if valid_count > 0 else 0
            
            self.generation_stats.append({
                'generation': gen + 1,
                'best_fitness': best_fitness,
                'avg_fitness': avg_fitness,
                'best_molecule': best_mol,
                'valid_molecules': valid_count,
                'unique_molecules': unique_count,
                'diversity': diversity,
                'exploration_rate': exploration_rate,
                'avg_boundary_distance': avg_boundary_dist,
                'avg_qed': avg_qed,
                'lipinski_compliance_rate': lipinski_rate,
                'validity_rate': valid_count / len(population)
            })
            
            # Print generation statistics
            print(f"Best fitness: {best_fitness:.3f}")
            print(f"Avg fitness: {avg_fitness:.3f}")
            print(f"Valid molecules: {valid_count}/{len(population)} ({100*valid_count/len(population):.1f}%)")
            print(f"Unique molecules: {unique_count}")
            print(f"Exploration rate: {100*exploration_rate:.1f}% outside boundary")
            print(f"Avg boundary distance: {avg_boundary_dist:.3f}")
            print(f"Avg QED: {avg_qed:.3f}")
            print(f"Lipinski compliance: {100*lipinski_rate:.1f}%")
            print(f"Diversity: {diversity:.3f}")
            print(f"Best molecule: {best_mol}")
            
            # Selection and reproduction
            valid_survivors = [(mol, score) for mol, score in fitness_scores 
                             if self.is_valid_smiles(mol) and score > 0]
            
            if len(valid_survivors) < 5:
                simple_mols = ["CCO", "CC", "CO", "CCC", "CCN", "CN"]
                for simple_mol in simple_mols:
                    valid_survivors.append((simple_mol, self.boundary_aware_fitness(simple_mol)))
                    if len(valid_survivors) >= 10:
                        break
            
            # Use boundary-aware selection
            survivors = self.diversity_selection(population, valid_survivors)
            
            # Create next generation
            new_population = survivors.copy()
            
            while len(new_population) < self.population_size:
                # Tournament selection
                parent1 = random.choice(survivors)
                parent2 = random.choice(survivors)
                
                # Avoid identical parents
                attempts = 0
                while parent1 == parent2 and attempts < 5:
                    parent2 = random.choice(survivors)
                    attempts += 1
                
                # Boundary-aware crossover
                child1, child2 = self.boundary_aware_crossover(parent1, parent2)
                
                # Boundary-guided mutation
                child1 = self.boundary_guided_mutation(child1)
                child2 = self.boundary_guided_mutation(child2)
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
        
        return self.generation_stats
    
    def plot_boundary_exploration_results(self):
        """Plot comprehensive results of boundary exploration"""
        if not self.generation_stats:
            print("No evolution data to plot")
            return
        
        generations = [s['generation'] for s in self.generation_stats]
        exploration_rates = [s['exploration_rate'] for s in self.generation_stats]
        boundary_distances = [s['avg_boundary_distance'] for s in self.generation_stats]
        qed_scores = [s['avg_qed'] for s in self.generation_stats]
        lipinski_rates = [s['lipinski_compliance_rate'] for s in self.generation_stats]
        best_fitnesses = [s['best_fitness'] for s in self.generation_stats]
        
        # Create comprehensive plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Top left: Exploration rate
        axes[0, 0].plot(generations, [100*r for r in exploration_rates], 'g-', linewidth=3)
        axes[0, 0].set_xlabel('Generation', fontsize=12)
        axes[0, 0].set_ylabel('Molecules Outside Boundary (%)', fontsize=12)
        axes[0, 0].set_title('Boundary Exploration Rate', fontsize=14, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Top center: Boundary distance
        axes[0, 1].plot(generations, boundary_distances, 'purple', linewidth=3)
        axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Boundary')
        axes[0, 1].set_xlabel('Generation', fontsize=12)
        axes[0, 1].set_ylabel('Avg Boundary Distance', fontsize=12)
        axes[0, 1].set_title('Distance from Known Chemical Space', fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Top right: Fitness evolution
        axes[0, 2].plot(generations, best_fitnesses, 'b-', linewidth=3)
        axes[0, 2].set_xlabel('Generation', fontsize=12)
        axes[0, 2].set_ylabel('Best Fitness', fontsize=12)
        axes[0, 2].set_title('Fitness Evolution', fontsize=14, fontweight='bold')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Bottom left: QED scores
        axes[1, 0].plot(generations, qed_scores, 'orange', linewidth=3)
        axes[1, 0].axhline(y=0.5, color='gray', linestyle=':', alpha=0.7)
        axes[1, 0].axhline(y=0.8, color='gray', linestyle=':', alpha=0.7)
        axes[1, 0].set_xlabel('Generation', fontsize=12)
        axes[1, 0].set_ylabel('Average QED Score', fontsize=12)
        axes[1, 0].set_title('Drug-likeness Evolution', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Bottom center: Lipinski compliance
        axes[1, 1].plot(generations, [100*r for r in lipinski_rates], 'red', linewidth=3)
        axes[1, 1].set_xlabel('Generation', fontsize=12)
        axes[1, 1].set_ylabel('Lipinski Compliance (%)', fontsize=12)
        axes[1, 1].set_title('Rule of 5 Compliance', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylim(0, 100)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Bottom right: Combined metrics
        # Normalize metrics for comparison
        norm_exploration = np.array(exploration_rates)
        norm_qed = np.array(qed_scores)
        norm_lipinski = np.array(lipinski_rates)
        
        axes[1, 2].plot(generations, norm_exploration, 'g-', label='Exploration Rate', linewidth=2)
        axes[1, 2].plot(generations, norm_qed, 'orange', label='QED Score', linewidth=2)
        axes[1, 2].plot(generations, norm_lipinski, 'red', label='Lipinski Compliance', linewidth=2)
        axes[1, 2].set_xlabel('Generation', fontsize=12)
        axes[1, 2].set_ylabel('Normalized Score', fontsize=12)
        axes[1, 2].set_title('Multi-Objective Progress', fontsize=14, fontweight='bold')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'boundary_exploration_results.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print final summary
        print("\n" + "="*70)
        print("BOUNDARY EXPLORATION GA - FINAL SUMMARY")
        print("="*70)
        
        final_stats = self.generation_stats[-1]
        
        print(f"Best molecule: {final_stats['best_molecule']}")
        print(f"Final fitness: {final_stats['best_fitness']:.3f}")
        print(f"Molecules exploring beyond boundary: {100*final_stats['exploration_rate']:.1f}%")
        print(f"Average boundary distance: {final_stats['avg_boundary_distance']:.3f}")
        print(f"Average QED score: {final_stats['avg_qed']:.3f}")
        print(f"Lipinski compliance rate: {100*final_stats['lipinski_compliance_rate']:.1f}%")
        print(f"Population diversity: {final_stats['diversity']:.3f}")
        
        # Analyze best molecule
        best_mol = final_stats['best_molecule']
        if self.is_valid_smiles(best_mol):
            mol = Chem.MolFromSmiles(best_mol)
            if mol:
                boundary_dist = self.boundary_detector.calculate_boundary_distance(best_mol)
                qed = QED.qed(mol)
                
                print(f"\nBest molecule analysis:")
                print(f"  Boundary distance: {boundary_dist:.3f} ({'EXPLORING' if boundary_dist > 0 else 'KNOWN SPACE'})")
                print(f"  QED score: {qed:.3f}")
                
                # Lipinski analysis
                mw = Descriptors.MolWt(mol)
                logp = Crippen.MolLogP(mol)
                hbd = Descriptors.NumHDonors(mol)
                hba = Descriptors.NumHAcceptors(mol)
                
                print(f"  Molecular Weight: {mw:.1f} ({'✓' if mw <= 500 else '✗'} ≤ 500)")
                print(f"  LogP: {logp:.2f} ({'✓' if logp <= 5 else '✗'} ≤ 5)")
                print(f"  H-bond donors: {hbd} ({'✓' if hbd <= 5 else '✗'} ≤ 5)")
                print(f"  H-bond acceptors: {hba} ({'✓' if hba <= 10 else '✗'} ≤ 10)")
        
        print(f"\nAll results saved to: {self.output_dir}/")

# Main execution function
def run_boundary_exploration_pipeline(parquet_path: str = "../mol_generative/zinc_test/zinc_drug_like_100_million_all.parquet",
                                    sample_size: int = 50000,
                                    population_size: int = 50,
                                    generations: int = 30,
                                    output_dir: str = "boundary_exploration_output",
                                    cache_path: str = "./stratified.parquet"):
    """
    Complete pipeline for boundary-aware chemical space exploration with caching
    """
    
    print("="*80)
    print("BOUNDARY-AWARE CHEMICAL SPACE EXPLORATION PIPELINE")
    print("="*80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Stratified sampling from ZINC (with caching)
    print("\n1. STRATIFIED SAMPLING FROM ZINC DATABASE")
    print("-" * 50)
    
    loader = StratifiedZINCLoader(random_state=42)
    try:
        zinc_sample = loader.load_stratified_sample(parquet_path, sample_size, cache_path)
        print(f"Successfully loaded {len(zinc_sample)} molecules")
        
        # Print sample statistics
        print(f"\nSample statistics:")
        print(f"  MW range: {zinc_sample['mw'].min():.1f} - {zinc_sample['mw'].max():.1f}")
        print(f"  LogP range: {zinc_sample['logp'].min():.2f} - {zinc_sample['logp'].max():.2f}")
        print(f"  QED range: {zinc_sample['qed'].min():.3f} - {zinc_sample['qed'].max():.3f}")
        print(f"  Average QED: {zinc_sample['qed'].mean():.3f}")
        
    except Exception as e:
        print(f"Error loading ZINC dataset: {e}")
        return None
    
    # Step 2: Boundary detection using ECFP4 + Convex Hull
    print("\n2. CHEMICAL SPACE BOUNDARY DETECTION")
    print("-" * 50)
    
    boundary_detector = ChemicalSpaceBoundaryDetector(fingerprint_bits=2048, pca_components=50)
    
    try:
        boundary_stats = boundary_detector.fit_boundary(zinc_sample['smiles'].tolist())
        print("Boundary detection completed successfully")
        
        # Print boundary statistics
        print(f"\nBoundary statistics:")
        for key, value in boundary_stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
        
        # Visualize boundary
        boundary_detector.visualize_boundary_2d(
            sample_molecules=zinc_sample['smiles'].sample(min(100, len(zinc_sample))).tolist(),
            save_path=os.path.join(output_dir, "zinc_boundary_visualization.png")
        )
        
    except Exception as e:
        print(f"Error in boundary detection: {e}")
        return None
    
    # Step 3: Load fragment library
    print("\n3. LOADING FRAGMENT LIBRARY")
    print("-" * 50)
    
    try:
        frag_lib = BRICSFragmentLibrary()
        frag_lib.load_library("zinc_brics_fragment_library.pkl")
        print("Fragment library loaded successfully")
    except Exception as e:
        print(f"Error loading fragment library: {e}")
        print("Attempting to create fragment library from sample...")
        try:
            frag_lib = BRICSFragmentLibrary()
            # Use a subset of the sample to build fragments if library doesn't exist
            sample_smiles = zinc_sample['smiles'].sample(min(10000, len(zinc_sample))).tolist()
            frag_lib.build_fragment_library(sample_smiles)
            frag_lib.save_library("zinc_brics_fragment_library.pkl")
            print("Fragment library created and saved successfully")
        except Exception as e2:
            print(f"Error creating fragment library: {e2}")
            return None
    
    # Step 4: Run boundary exploration GA
    print("\n4. BOUNDARY EXPLORATION GENETIC ALGORITHM")
    print("-" * 50)
    
    ga = BoundaryExplorationGA(
        fragment_library=frag_lib,
        boundary_detector=boundary_detector,
        population_size=population_size,
        generations=generations,
        boundary_weight=0.3,
        diversity_weight=0.2,
        output_dir=output_dir
    )
    
    try:
        stats = ga.run_boundary_exploration()
        
        # Visualize results
        ga.plot_boundary_exploration_results()
        
        # Create final boundary visualization with GA results
        final_generation_file = os.path.join(output_dir, f"generation_{generations:03d}.txt")
        if os.path.exists(final_generation_file):
            try:
                final_df = pd.read_csv(final_generation_file, sep='\t')
                final_molecules = final_df[final_df['Valid'] == True]['SMILES'].tolist()
                
                if final_molecules:
                    boundary_detector.visualize_boundary_2d(
                        sample_molecules=final_molecules,
                        save_path=os.path.join(output_dir, "final_boundary_exploration.png")
                    )
                else:
                    print("No valid molecules found in final generation for visualization")
            except Exception as e:
                print(f"Error creating final visualization: {e}")
        
        print("\nPipeline completed successfully!")
        print(f"Results saved to: {output_dir}/")
        print(f"Cached stratified sample: {cache_path}")
        
        return ga, boundary_detector, stats
        
    except Exception as e:
        print(f"Error in genetic algorithm: {e}")
        return None

# Example usage and testing
if __name__ == "__main__":
    # Test with smaller parameters for development
    print("Testing Boundary Exploration Pipeline...")
    
    # Test parameters
    test_params = {
        'sample_size': 5000,  # Smaller for testing
        'population_size': 20,
        'generations': 10,
        'output_dir': 'test_boundary_exploration',
        'cache_path': './test_stratified.parquet'  # Separate cache for testing
    }
    
    try:
        results = run_boundary_exploration_pipeline(**test_params)
        
        if results:
            ga, boundary_detector, stats = results
            print("\n" + "="*50)
            print("TEST COMPLETED SUCCESSFULLY")
            print("="*50)
            print(f"Final exploration rate: {stats[-1]['exploration_rate']:.2%}")
            print(f"Final QED score: {stats[-1]['avg_qed']:.3f}")
            print(f"Final Lipinski compliance: {stats[-1]['lipinski_compliance_rate']:.2%}")
            print(f"Cache file created: {test_params['cache_path']}")
        else:
            print("Test failed - check error messages above")
            
    except Exception as e:
        print(f"Test execution failed: {e}")
        import traceback
        traceback.print_exc()

# Full production run function
def run_production_boundary_exploration():
    """
    Production run with full parameters and caching
    """
    print("Running PRODUCTION Boundary Exploration...")
    
    production_params = {
        'parquet_path': "../mol_generative/zinc_test/zinc_drug_like_100_million_all.parquet",
        'sample_size': 50000,
        'population_size': 50, 
        'generations': 50,
        'output_dir': 'production_boundary_exploration',
        'cache_path': './stratified.parquet'  # Production cache
    }
    
    return run_boundary_exploration_pipeline(**production_params)

# Quick run function for development
def run_quick_boundary_exploration(sample_size: int = 10000, generations: int = 20):
    """
    Quick run with moderate parameters for faster development/testing
    """
    print(f"Running QUICK Boundary Exploration ({sample_size} molecules, {generations} generations)...")
    
    quick_params = {
        'parquet_path': "../mol_generative/zinc_test/zinc_drug_like_100_million_all.parquet",
        'sample_size': sample_size,
        'population_size': 30, 
        'generations': generations,
        'output_dir': f'quick_boundary_exploration_{sample_size}',
        'cache_path': f'./quick_stratified_{sample_size}.parquet'
    }
    
    return run_boundary_exploration_pipeline(**quick_params)

# Cache management utilities
def check_cache_status():
    """Check status of all cache files"""
    cache_files = [
        './stratified.parquet',
        './test_stratified.parquet',
        './quick_stratified_10000.parquet'
    ]
    
    print("CACHE STATUS:")
    print("-" * 40)
    
    for cache_file in cache_files:
        if os.path.exists(cache_file):
            try:
                df = pd.read_parquet(cache_file)
                file_size = os.path.getsize(cache_file) / (1024 * 1024)  # MB
                print(f"✓ {cache_file}: {len(df)} molecules ({file_size:.1f} MB)")
            except Exception as e:
                print(f"✗ {cache_file}: Error reading - {e}")
        else:
            print(f"✗ {cache_file}: Not found")

def clear_cache(cache_path: str = None):
    """Clear specific cache file or all cache files"""
    if cache_path:
        # Clear specific cache
        if os.path.exists(cache_path):
            os.remove(cache_path)
            print(f"Cleared cache: {cache_path}")
        else:
            print(f"Cache not found: {cache_path}")
    else:
        # Clear all cache files
        cache_files = [
            './stratified.parquet',
            './test_stratified.parquet',
            './quick_stratified_10000.parquet'
        ]
        
        cleared = 0
        for cache_file in cache_files:
            if os.path.exists(cache_file):
                os.remove(cache_file)
                print(f"Cleared: {cache_file}")
                cleared += 1
        
        print(f"Cleared {cleared} cache files")

def load_cached_sample(cache_path: str = './stratified.parquet'):
    """Load and inspect cached stratified sample"""
    if not os.path.exists(cache_path):
        print(f"Cache file not found: {cache_path}")
        return None
    
    try:
        df = pd.read_parquet(cache_path)
        print(f"Loaded cached sample: {len(df)} molecules")
        
        # Display summary statistics
        print(f"\nMolecular Properties Summary:")
        print(f"  MW: {df['mw'].min():.1f} - {df['mw'].max():.1f} (avg: {df['mw'].mean():.1f})")
        print(f"  LogP: {df['logp'].min():.2f} - {df['logp'].max():.2f} (avg: {df['logp'].mean():.2f})")
        print(f"  QED: {df['qed'].min():.3f} - {df['qed'].max():.3f} (avg: {df['qed'].mean():.3f})")
        print(f"  HBA: {df['hba'].min()} - {df['hba'].max()} (avg: {df['hba'].mean():.1f})")
        print(f"  HBD: {df['hbd'].min()} - {df['hbd'].max()} (avg: {df['hbd'].mean():.1f})")
        
        if 'strat_key' in df.columns:
            print(f"\nStratification groups: {df['strat_key'].nunique()}")
            print("Top 5 groups:")
            print(df['strat_key'].value_counts().head())
        
        return df
        
    except Exception as e:
        print(f"Error loading cache: {e}")
        return None

# Utility functions for analysis
def analyze_boundary_exploration_results(output_dir: str):
    """
    Analyze results from boundary exploration GA
    """
    print(f"Analyzing results from {output_dir}...")
    
    # Find all generation files
    generation_files = []
    for f in os.listdir(output_dir):
        if f.startswith('generation_') and f.endswith('.txt'):
            generation_files.append(f)
    
    generation_files.sort()
    
    if not generation_files:
        print("No generation files found")
        return None
    
    # Analyze exploration progress
    exploration_data = []
    
    for gen_file in generation_files:
        gen_num = int(gen_file.split('_')[1].split('.')[0])
        file_path = os.path.join(output_dir, gen_file)
        
        try:
            df = pd.read_csv(file_path, sep='\t')
            valid_df = df[df['Valid'] == True]
            
            if len(valid_df) > 0:
                exploration_count = len(valid_df[valid_df['OutsideBoundary'] == True])
                avg_boundary_dist = valid_df['BoundaryDist'].mean()
                avg_qed = valid_df['QED'].mean()
                
                # Calculate Lipinski compliance
                lipinski_compliant = 0
                for _, row in valid_df.iterrows():
                    violations = sum([
                        row['MW'] > 500,
                        row['LogP'] > 5,
                        row['HBD'] > 5,
                        row['HBA'] > 10
                    ])
                    if violations == 0:
                        lipinski_compliant += 1
                
                lipinski_rate = lipinski_compliant / len(valid_df)
                
                exploration_data.append({
                    'generation': gen_num,
                    'total_valid': len(valid_df),
                    'exploring_molecules': exploration_count,
                    'exploration_rate': exploration_count / len(valid_df),
                    'avg_boundary_distance': avg_boundary_dist,
                    'avg_qed': avg_qed,
                    'lipinski_compliance_rate': lipinski_rate
                })
        except Exception as e:
            print(f"Error processing {gen_file}: {e}")
            continue
    
    if exploration_data:
        results_df = pd.DataFrame(exploration_data)
        
        # Save analysis
        analysis_file = os.path.join(output_dir, 'exploration_analysis.csv')
        results_df.to_csv(analysis_file, index=False)
        print(f"Analysis saved to {analysis_file}")
        
        # Print summary
        print("\nEXPLORATION ANALYSIS SUMMARY")
        print("-" * 40)
        print(f"Generations analyzed: {len(results_df)}")
        print(f"Final exploration rate: {results_df.iloc[-1]['exploration_rate']:.2%}")
        print(f"Max exploration rate: {results_df['exploration_rate'].max():.2%}")
        print(f"Final avg QED: {results_df.iloc[-1]['avg_qed']:.3f}")
        print(f"Final Lipinski compliance: {results_df.iloc[-1]['lipinski_compliance_rate']:.2%}")
        
        # Identify best exploring molecules
        if len(generation_files) > 0:
            final_gen_file = os.path.join(output_dir, generation_files[-1])
            final_df = pd.read_csv(final_gen_file, sep='\t')
            explorers = final_df[
                (final_df['Valid'] == True) & 
                (final_df['OutsideBoundary'] == True)
            ].sort_values('QED', ascending=False)
            
            print(f"\nTOP EXPLORING MOLECULES (Outside Boundary + High QED):")
            print("-" * 60)
            for i, (_, row) in enumerate(explorers.head(5).iterrows()):
                print(f"{i+1}. {row['SMILES']}")
                print(f"   QED: {row['QED']:.3f}, Boundary Dist: {row['BoundaryDist']:.3f}")
                print(f"   MW: {row['MW']:.1f}, LogP: {row['LogP']:.2f}")
        
        return results_df
    
    else:
        print("No valid data found for analysis")
        return None

def find_novel_druglike_molecules(output_dir: str, min_qed: float = 0.6, min_boundary_dist: float = 0.1):
    """
    Extract novel drug-like molecules that are outside the boundary
    """
    print(f"Finding novel drug-like molecules...")
    print(f"Criteria: QED >= {min_qed}, Boundary distance >= {min_boundary_dist}")
    
    # Get final generation
    generation_files = [f for f in os.listdir(output_dir) if f.startswith('generation_') and f.endswith('.txt')]
    if not generation_files:
        print("No generation files found")
        return None
    
    final_gen_file = sorted(generation_files)[-1]
    file_path = os.path.join(output_dir, final_gen_file)
    
    try:
        df = pd.read_csv(file_path, sep='\t')
        
        # Filter for novel drug-like molecules
        novel_molecules = df[
            (df['Valid'] == True) &
            (df['QED'] >= min_qed) &
            (df['BoundaryDist'] >= min_boundary_dist) &
            (df['OutsideBoundary'] == True)
        ].copy()
        
        # Add Lipinski compliance
        novel_molecules['Lipinski_Violations'] = (
            (novel_molecules['MW'] > 500).astype(int) +
            (novel_molecules['LogP'] > 5).astype(int) +
            (novel_molecules['HBD'] > 5).astype(int) +
            (novel_molecules['HBA'] > 10).astype(int)
        )
        
        novel_molecules['Lipinski_Compliant'] = novel_molecules['Lipinski_Violations'] == 0
        
        # Sort by QED score
        novel_molecules = novel_molecules.sort_values('QED', ascending=False)
        
        print(f"\nFound {len(novel_molecules)} novel drug-like molecules")
        
        if len(novel_molecules) > 0:
            # Save results
            output_file = os.path.join(output_dir, 'novel_druglike_molecules.csv')
            novel_molecules.to_csv(output_file, index=False)
            print(f"Results saved to {output_file}")
            
            # Print top candidates
            print(f"\nTOP 10 NOVEL DRUG-LIKE CANDIDATES:")
            print("-" * 80)
            for i, (_, row) in enumerate(novel_molecules.head(10).iterrows()):
                lipinski_status = "✓" if row['Lipinski_Compliant'] else f"✗({row['Lipinski_Violations']})"
                print(f"{i+1:2d}. {row['SMILES']}")
                print(f"    QED: {row['QED']:.3f} | Boundary: {row['BoundaryDist']:.3f} | Lipinski: {lipinski_status}")
                print(f"    MW: {row['MW']:.1f} | LogP: {row['LogP']:.2f} | HBD: {row['HBD']} | HBA: {row['HBA']}")
                print()
        
        return novel_molecules
        
    except Exception as e:
        print(f"Error analyzing molecules: {e}")
        return None