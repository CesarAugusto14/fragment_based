#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart Chemical Genetic Algorithm for Drug Discovery

This class extends the SimpleDrugGA with chemically-aware genetic operators
that respect molecular structure and chemistry rules. It provides improved
diversity preservation and chemical space exploration through intelligent
fragment-based mutations and crossover operations.

Author: cesarasa
Date: 08-18-2024
"""
import os
import random
import re
from collections import Counter, defaultdict
from typing import List, Tuple, Optional, Dict, Set

import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, QED, Crippen, AllChem, BRICS

from fragmentation import BRICSFragmentLibrary

# Suppress RDKit warnings and errors
RDLogger.DisableLog('rdApp.*')

class SmartChemicalGA:
    """
    Enhanced Genetic Algorithm with Smart Chemical Operators
    
    Integrates chemically-aware mutation and crossover strategies to improve
    molecular diversity and chemical space exploration while maintaining
    drug-likeness optimization.
    """
    
    def __init__(self,
                 fragment_library: BRICSFragmentLibrary,
                 population_size: int = 50,
                 generations: int = 50,
                 mutation_rate: float = 0.3,  # Lower rate for smart mutations
                 crossover_rate: float = 0.7,
                 diversity_weight: float = 0.2,  # Weight for diversity in fitness
                 max_identical: int = 5,  # Max identical molecules allowed
                 output_dir: str = "ga_output_smart"
                 ):
        self.frag_lib = fragment_library
        
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
        self.diversity_weight = diversity_weight
        self.max_identical = max_identical
        
        # Tracking
        self.generation_stats = []
        self.best_molecules = []
        
        # Create output directory
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Smart operators data
        self.functional_group_transforms = {
            'O': ['N', 'S'],
            'N': ['O', 'S', '[NH2]'],
            'S': ['O', 'N'],
            'C=O': ['C=N', 'C=S'],
            'C': ['N', 'O'],
            'F': ['Cl', 'Br'],
            'Cl': ['F', 'Br'],
            'Br': ['F', 'Cl']
        }
        
        self.common_rings = [
            'c1ccccc1',      # benzene
            'C1CCCCC1',      # cyclohexane
            'c1ccncc1',      # pyridine
            'c1cccnc1',      # pyrimidine
            'c1ccoc1',       # furan
            'c1ccsc1',       # thiophene
            'C1CCNCC1',      # piperidine
            'C1CCOCC1',      # tetrahydropyran
            'c1cc2ccccc2cc1', # naphthalene
            'c1cnccn1'       # pyrazine
        ]
    
    def is_valid_smiles(self, smiles: str) -> bool:
        """Check if SMILES string represents a valid, drug-like molecule"""
        if not smiles:
            return False
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
            
            # Sanitize to catch chemical issues
            Chem.SanitizeMol(mol)
            
            # Basic drug-likeness filters
            mw = Descriptors.MolWt(mol)
            if mw > 600 or mw < 50:
                return False
                
            # Check for reasonable atom count
            if mol.GetNumAtoms() > 50 or mol.GetNumAtoms() < 3:
                return False
                
            return True
        except:
            return False
    
    def calculate_molecular_properties(self, smiles: str) -> dict:
        """Calculate molecular properties for a SMILES string"""
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
                'hbd': Descriptors.NumHDonors(mol)
            }
            return properties
        except:
            return {
                'smiles': smiles,
                'valid': False,
                'qed': 0.0,
                'logp': 0.0,
                'mw': 0.0,
                'hba': 0,
                'hbd': 0
            }
    
    def calculate_population_diversity(self, population: List[str]) -> float:
        """Calculate population diversity using Tanimoto similarity"""
        valid_mols = [smiles for smiles in population if self.is_valid_smiles(smiles)]
        
        if len(valid_mols) < 2:
            return 0.0
        
        try:
            # Calculate ECFP4 fingerprints
            fps = []
            for smiles in valid_mols:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                    fps.append(fp)
            
            if len(fps) < 2:
                return 0.0
            
            # Calculate pairwise Tanimoto similarities
            similarities = []
            for i in range(len(fps)):
                for j in range(i+1, len(fps)):
                    sim = AllChem.DataStructs.TanimotoSimilarity(fps[i], fps[j])
                    similarities.append(sim)
            
            # Diversity = 1 - average similarity
            avg_similarity = np.mean(similarities)
            return 1.0 - avg_similarity
            
        except Exception as e:
            return 0.0
    
    def extract_molecular_fragments(self, smiles: str) -> Dict:
        """Extract molecular components for smart operations"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                return {}
            
            # Get ring information
            ring_info = mol.GetRingInfo()
            ring_atoms = set()
            rings = []
            
            for ring in ring_info.AtomRings():
                ring_atoms.update(ring)
                # Extract ring SMILES (simplified)
                ring_size = len(ring)
                rings.append({
                    'atoms': ring,
                    'size': ring_size
                })
            
            # Get functional groups
            functional_groups = []
            for atom in mol.GetAtoms():
                if atom.GetSymbol() in ['O', 'N', 'S', 'F', 'Cl', 'Br']:
                    functional_groups.append({
                        'symbol': atom.GetSymbol(),
                        'idx': atom.GetIdx(),
                        'neighbors': [n.GetSymbol() for n in atom.GetNeighbors()]
                    })
            
            return {
                'rings': rings,
                'functional_groups': functional_groups,
                'total_atoms': mol.GetNumAtoms(),
                'smiles': smiles
            }
        except:
            return {}
    
    def smart_fragment_swap(self, smiles: str) -> str:
        """Smart Mutation: Replace molecular fragments with similar ones"""
        if not self.is_valid_smiles(smiles):
            return smiles
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                return smiles
            
            brics_fragments = list(BRICS.BRICSDecompose(mol))
            if len(brics_fragments) < 2:
                return smiles
            
            # Choose a random fragment to replace
            fragment_to_replace = random.choice(brics_fragments)
            
            # Clean the fragment and find similar ones
            clean_frag, metadata = self.frag_lib.clean_brics_fragment(fragment_to_replace)
            if not clean_frag or not metadata:
                return smiles
            
            # Find replacement fragments with same connection pattern
            connection_pattern = f"{metadata['num_connections']}_connections_" + "_".join(sorted(metadata['connection_types']))
            similar_fragments = self.frag_lib.fragment_types.get(connection_pattern, [])
            
            if not similar_fragments:
                return smiles
            
            # Choose a different fragment as replacement
            available_replacements = [f for f in similar_fragments if f['clean_smiles'] != clean_frag]
            if not available_replacements:
                return smiles
            
            replacement_fragment = random.choice(available_replacements)
            replacement_smiles = replacement_fragment['clean_smiles']
            
            # Simple reconstruction attempt
            new_fragments = []
            for frag in brics_fragments:
                if frag == fragment_to_replace:
                    new_fragments.append(replacement_smiles)
                else:
                    clean_f, _ = self.frag_lib.clean_brics_fragment(frag)
                    if clean_f:
                        new_fragments.append(clean_f)
            
            # Try to combine fragments
            if len(new_fragments) >= 2:
                combined = random.choice(new_fragments)
                for frag in new_fragments[1:]:
                    candidates = [
                        f"{combined}{frag}",
                        f"{frag}{combined}",
                        f"C{combined}{frag}",
                        f"{combined}C{frag}"
                    ]
                    
                    for candidate in candidates:
                        if self.is_valid_smiles(candidate):
                            combined = candidate
                            break
                
                if self.is_valid_smiles(combined):
                    return combined
            
        except Exception as e:
            pass
        
        return smiles
    
    def smart_functional_group_change(self, smiles: str) -> str:
        """Smart Mutation: Replace functional groups with similar ones"""
        if not self.is_valid_smiles(smiles):
            return smiles
        
        try:
            fragments = self.extract_molecular_fragments(smiles)
            if not fragments or not fragments.get('functional_groups'):
                return smiles
            
            # Choose a functional group to modify
            fg = random.choice(fragments['functional_groups'])
            original_symbol = fg['symbol']
            
            # Get possible replacements
            replacements = self.functional_group_transforms.get(original_symbol, [])
            if not replacements:
                return smiles
            
            replacement = random.choice(replacements)
            
            # Simple replacement in SMILES string
            new_smiles = smiles.replace(original_symbol, replacement, 1)
            
            if self.is_valid_smiles(new_smiles):
                return new_smiles
                
        except Exception as e:
            pass
        
        return smiles
    
    def smart_ring_modification(self, smiles: str) -> str:
        """Smart Mutation: Add or modify ring systems"""
        if not self.is_valid_smiles(smiles):
            return smiles
        
        try:
            fragments = self.extract_molecular_fragments(smiles)
            if not fragments:
                return smiles
            
            # Strategy 1: Add a ring if molecule has none or few
            if len(fragments.get('rings', [])) == 0:
                ring = random.choice(self.common_rings)
                candidates = [
                    f"{smiles}{ring}",
                    f"{ring}{smiles}",
                    f"C{smiles}{ring}",
                    f"{ring}C{smiles}"
                ]
                
                for candidate in candidates:
                    if self.is_valid_smiles(candidate):
                        return candidate
            
            # Strategy 2: Replace existing ring (simplified)
            elif len(fragments.get('rings', [])) > 0:
                replacement_ring = random.choice(self.common_rings)
                # Try adding ring to existing structure
                candidates = [
                    f"{smiles}{replacement_ring}",
                    f"{replacement_ring}{smiles}",
                    f"C{smiles}{replacement_ring}",
                    f"{replacement_ring}C{smiles}"
                ]
                
                for candidate in candidates:
                    if self.is_valid_smiles(candidate):
                        return candidate
                    
        except Exception as e:
            pass
        
        return smiles
    
    def smart_fragment_exchange_crossover(self, parent1: str, parent2: str) -> Tuple[str, str]:
        """Smart Crossover: Exchange compatible fragments between parents"""
        if not (self.is_valid_smiles(parent1) and self.is_valid_smiles(parent2)):
            return parent1, parent2
        
        try:
            mol1 = Chem.MolFromSmiles(parent1)
            mol2 = Chem.MolFromSmiles(parent2)
            
            if not (mol1 and mol2):
                return parent1, parent2
            
            frags1 = list(BRICS.BRICSDecompose(mol1))
            frags2 = list(BRICS.BRICSDecompose(mol2))
            
            if len(frags1) < 2 or len(frags2) < 2:
                return parent1, parent2
            
            # Find compatible fragments (same connection pattern)
            compatible_pairs = []
            
            for i, frag1 in enumerate(frags1):
                clean1, meta1 = self.frag_lib.clean_brics_fragment(frag1)
                if not clean1 or not meta1:
                    continue
                    
                for j, frag2 in enumerate(frags2):
                    clean2, meta2 = self.frag_lib.clean_brics_fragment(frag2)
                    if not clean2 or not meta2:
                        continue
                    
                    # Check if fragments have compatible connection patterns
                    if (meta1['num_connections'] == meta2['num_connections'] and
                        set(meta1['connection_types']) == set(meta2['connection_types'])):
                        compatible_pairs.append((i, j, clean1, clean2))
            
            if not compatible_pairs:
                return parent1, parent2
            
            # Perform exchange (simplified)
            chosen_pair = random.choice(compatible_pairs)
            i, j, clean1, clean2 = chosen_pair
            
            # Simple combination attempts
            child1_candidates = [
                f"{clean2}{parent1}",
                f"{parent1}{clean2}",
                f"C{clean2}{parent1}",
                f"{parent1}C{clean2}"
            ]
            
            child2_candidates = [
                f"{clean1}{parent2}",
                f"{parent2}{clean1}",
                f"C{clean1}{parent2}",
                f"{parent2}C{clean1}"
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
                
        except Exception as e:
            pass
        
        return parent1, parent2
    
    def smart_scaffold_hopping_crossover(self, parent1: str, parent2: str) -> Tuple[str, str]:
        """Smart Crossover: Keep core structure, swap decorations"""
        if not (self.is_valid_smiles(parent1) and self.is_valid_smiles(parent2)):
            return parent1, parent2
        
        try:
            frags1 = self.extract_molecular_fragments(parent1)
            frags2 = self.extract_molecular_fragments(parent2)
            
            if not (frags1 and frags2):
                return parent1, parent2
            
            # Strategy: Combine parents with decorations from the other
            if frags1.get('functional_groups') or frags2.get('functional_groups'):
                # Take decorations from each parent
                if frags2.get('functional_groups'):
                    decorations = random.sample(
                        frags2['functional_groups'], 
                        min(2, len(frags2['functional_groups']))
                    )
                    
                    child1 = parent1
                    for decoration in decorations:
                        symbol = decoration['symbol']
                        candidates = [
                            f"{child1}{symbol}",
                            f"{symbol}{child1}",
                            f"C{child1}{symbol}",
                            f"{child1}C{symbol}"
                        ]
                        
                        for candidate in candidates:
                            if self.is_valid_smiles(candidate):
                                child1 = candidate
                                break
                else:
                    child1 = parent1
                
                # Create child2 with opposite strategy
                if frags1.get('functional_groups'):
                    decorations2 = random.sample(
                        frags1['functional_groups'],
                        min(2, len(frags1['functional_groups']))
                    )
                    
                    child2 = parent2
                    for decoration in decorations2:
                        symbol = decoration['symbol']
                        candidates = [
                            f"{child2}{symbol}",
                            f"{symbol}{child2}",
                            f"C{child2}{symbol}",
                            f"{child2}C{symbol}"
                        ]
                        
                        for candidate in candidates:
                            if self.is_valid_smiles(candidate):
                                child2 = candidate
                                break
                else:
                    child2 = parent2
                
                return child1, child2
                    
        except Exception as e:
            pass
        
        return parent1, parent2
    
    def smart_chemical_grafting_crossover(self, parent1: str, parent2: str) -> Tuple[str, str]:
        """Smart Crossover: Connect parents through valid linkers"""
        if not (self.is_valid_smiles(parent1) and self.is_valid_smiles(parent2)):
            return parent1, parent2
        
        try:
            if not self.linkers:
                return parent1, parent2
            
            linker = random.choice(self.linkers)
            
            # Try to create children by connecting parents through linker
            grafting_patterns = [
                f"{parent1}{linker}{parent2}",
                f"{parent2}{linker}{parent1}",
                f"C{parent1}{linker}C{parent2}",
                f"C{parent2}{linker}C{parent1}",
                f"{parent1}C{linker}C{parent2}",
                f"{parent2}C{linker}C{parent1}"
            ]
            
            valid_children = []
            for pattern in grafting_patterns:
                if self.is_valid_smiles(pattern):
                    valid_children.append(pattern)
            
            if len(valid_children) >= 2:
                return valid_children[0], valid_children[1]
            elif len(valid_children) == 1:
                return valid_children[0], parent2
                
        except Exception as e:
            pass
        
        return parent1, parent2
    
    def mutate(self, smiles: str) -> str:
        """Enhanced mutation using smart chemical strategies"""
        if not smiles or random.random() > self.mutation_rate:
            return smiles
        
        # Smart mutation strategies
        strategies = [
            self.smart_fragment_swap,
            self.smart_functional_group_change,
            self.smart_ring_modification
        ]
        
        # Try strategies in random order
        random.shuffle(strategies)
        
        for strategy in strategies:
            try:
                result = strategy(smiles)
                if result != smiles and self.is_valid_smiles(result):
                    return result
            except:
                continue
        
        # Fallback to simple mutation
        max_attempts = 10
        for _ in range(max_attempts):
            try:
                if self.terminals and random.random() < 0.5:
                    new_mol = random.choice(self.terminals)
                    if self.is_valid_smiles(new_mol):
                        return new_mol
                
                if self.linkers and random.random() < 0.5:
                    new_mol = random.choice(self.linkers)
                    if self.is_valid_smiles(new_mol):
                        return new_mol
                
                modifications = [
                    f"C{smiles}",
                    f"{smiles}C",
                    f"N{smiles}",
                    f"{smiles}O",
                ]
                
                for mod in modifications:
                    if self.is_valid_smiles(mod):
                        return mod
                        
            except:
                continue
        
        return smiles
    
    def crossover(self, parent1: str, parent2: str) -> Tuple[str, str]:
        """Enhanced crossover using smart chemical strategies"""
        if random.random() > self.crossover_rate:
            return parent1, parent2
        
        # Smart crossover strategies
        strategies = [
            self.smart_fragment_exchange_crossover,
            self.smart_scaffold_hopping_crossover,
            self.smart_chemical_grafting_crossover
        ]
        
        # Try strategies in random order
        random.shuffle(strategies)
        
        for strategy in strategies:
            try:
                child1, child2 = strategy(parent1, parent2)
                if ((child1 != parent1 or child2 != parent2) and 
                    self.is_valid_smiles(child1) and self.is_valid_smiles(child2)):
                    return child1, child2
            except:
                continue
        
        # Fallback to simple crossover
        max_attempts = 5
        for _ in range(max_attempts):
            try:
                if random.random() < 0.5:
                    child1_candidates = [parent1, parent2, f"C{parent1}", f"{parent2}C"]
                    child2_candidates = [parent2, parent1, f"C{parent2}", f"{parent1}C"]
                else:
                    mid1 = len(parent1) // 2
                    mid2 = len(parent2) // 2
                    
                    child1_candidates = [
                        parent1[:mid1] + parent2[mid2:],
                        parent2[:mid2] + parent1[mid1:],
                        parent1,
                        parent2
                    ]
                    child2_candidates = [
                        parent2[:mid2] + parent1[mid1:],
                        parent1[:mid1] + parent2[mid2:],
                        parent2,
                        parent1
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
                continue
        
        return parent1, parent2
    
    def fitness(self, smiles: str) -> float:
        """Fitness function for chemical-based GA"""
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
            
            # Lipinski bonus
            lipinski_score = (4 - violations) / 4.0
            
            # Base fitness
            base_fitness = 0.7 * qed_score + 0.3 * lipinski_score
            
            return base_fitness
            
        except:
            return 0
    
    def create_random_molecule(self) -> str:
        """Create a random valid molecule using fragments"""
        max_attempts = 50
        
        for _ in range(max_attempts):
            try:
                # Strategy 1: Use just a terminal fragment
                if random.random() < 0.3 and self.terminals:
                    candidate = random.choice(self.terminals)
                    if self.is_valid_smiles(candidate):
                        return candidate
                
                # Strategy 2: Use just a linker fragment
                if random.random() < 0.3 and self.linkers:
                    candidate = random.choice(self.linkers)
                    if self.is_valid_smiles(candidate):
                        return candidate
                
                # Strategy 3: Simple combination
                if self.terminals and self.linkers:
                    terminal = random.choice(self.terminals)
                    linker = random.choice(self.linkers)
                    
                    combinations = [
                        f"{terminal}{linker}",
                        f"{linker}{terminal}",
                        f"C{terminal}",
                        f"C{linker}",
                        terminal,
                        linker
                    ]
                    
                    for combo in combinations:
                        if self.is_valid_smiles(combo):
                            return combo
                            
            except:
                continue
        
        return "CCO"  # Fallback
    
    def diversity_selection(self, population: List[str], fitness_scores: List[Tuple[str, float]]) -> List[str]:
        """Enhanced selection with diversity preservation"""
        # Count identical molecules
        mol_counts = Counter(population)
        
        # Filter out excess identical molecules
        filtered_survivors = []
        for mol, score in fitness_scores:
            if mol_counts[mol] <= self.max_identical:
                filtered_survivors.append((mol, score))
            elif random.random() < 0.1:  # Small chance to keep some duplicates
                filtered_survivors.append((mol, score))
        
        # If we filtered too many, add some back
        if len(filtered_survivors) < max(5, self.population_size // 4):
            additional = [item for item in fitness_scores if item not in filtered_survivors]
            filtered_survivors.extend(additional[:10])
        
        # Select top performers
        survivors = [mol for mol, _ in filtered_survivors[:max(5, len(filtered_survivors)//2)]]
        
        return survivors
    
    def save_generation_data(self, generation: int, population: List[str]):
        """Save generation data to txt file"""
        filename = os.path.join(self.output_dir, f"generation_{generation:03d}.txt")
        
        with open(filename, 'w') as f:
            f.write("SMILES\tValid\tQED\tLogP\tMW\tHBA\tHBD\n")
            
            for smiles in population:
                props = self.calculate_molecular_properties(smiles)
                if props:
                    f.write(f"{props['smiles']}\t{props['valid']}\t{props['qed']:.4f}\t"
                           f"{props['logp']:.4f}\t{props['mw']:.2f}\t{props['hba']}\t{props['hbd']}\n")
                else:
                    f.write(f"{smiles}\tFalse\t0.0000\t0.0000\t0.00\t0\t0\n")
    
    def collect_generation_data(self, generation: int, population: List[str]) -> List[Dict]:
        """Collect valid SMILES and their properties for a generation"""
        valid_data = []
        for smiles in population:
            if self.is_valid_smiles(smiles):
                props = self.calculate_molecular_properties(smiles)
                if props and props['valid']:
                    valid_data.append({
                        'generation': generation,
                        'smiles': smiles,
                        'qed': props['qed'],
                        'logp': props['logp'],
                        'mw': props['mw'],
                        'hba': props['hba'],
                        'hbd': props['hbd']
                    })
        return valid_data
    
    def run_ga(self) -> List[Dict]:
        """Run the enhanced genetic algorithm with smart operators"""
        print("Starting Smart Chemical GA...")
        print(f"Terminals: {len(self.terminals)}, Linkers: {len(self.linkers)}, Branches: {len(self.branches)}")
        print(f"Output directory: {self.output_dir}")
        print(f"Diversity weight: {self.diversity_weight}, Max identical: {self.max_identical}")
        
        # Initialize population with valid molecules only
        population = []
        attempts = 0
        max_init_attempts = self.population_size * 10
        
        while len(population) < self.population_size and attempts < max_init_attempts:
            mol = self.create_random_molecule()
            if mol and self.is_valid_smiles(mol):
                population.append(mol)
            attempts += 1
        
        # Fill remaining slots with simple valid molecules if needed
        simple_mols = ["CCO", "CC", "CO", "C", "CCC", "CCN", "CN", "C=O"]
        while len(population) < self.population_size:
            population.append(random.choice(simple_mols))
        
        print(f"Initial population: {len(population)} valid molecules")
        
        # Store all generation data
        all_generation_data = []
        
        # Evolution loop
        for gen in range(self.generations):
            print(f"\nGeneration {gen + 1}/{self.generations}")
            
            # Save generation data to file
            self.save_generation_data(gen + 1, population)
            
            # Collect valid molecules for tracking
            gen_data = self.collect_generation_data(gen + 1, population)
            all_generation_data.append(gen_data)
            
            # Calculate population diversity
            diversity = self.calculate_population_diversity(population)
            
            # Track validity and uniqueness statistics
            valid_count = sum(1 for mol in population if self.is_valid_smiles(mol))
            unique_count = len(set(population))
            mol_counts = Counter(population)
            max_duplicates = max(mol_counts.values()) if mol_counts else 0
            
            # Evaluate fitness (only for valid molecules)
            fitness_scores = []
            for mol in population:
                if self.is_valid_smiles(mol):
                    base_fitness = self.fitness(mol)
                    
                    # Apply diversity bonus/penalty
                    count_penalty = min(mol_counts[mol] - 1, 5) * 0.1  # Penalty for duplicates
                    diversity_bonus = diversity * self.diversity_weight
                    
                    final_fitness = base_fitness + diversity_bonus - count_penalty
                    fitness_scores.append((mol, final_fitness))
                else:
                    fitness_scores.append((mol, 0.0))
            
            fitness_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Track stats
            fitnesses = [score for _, score in fitness_scores]
            best_mol, best_fitness = fitness_scores[0]
            avg_fitness = np.mean(fitnesses)
            
            self.generation_stats.append({
                'generation': gen + 1,
                'best_fitness': best_fitness,
                'avg_fitness': avg_fitness,
                'best_molecule': best_mol,
                'valid_molecules': valid_count,
                'unique_molecules': unique_count,
                'diversity': diversity,
                'max_duplicates': max_duplicates,
                'validity_rate': valid_count / len(population)
            })
            
            print(f"Best fitness: {best_fitness:.3f}")
            print(f"Avg fitness: {avg_fitness:.3f}")
            print(f"Valid molecules: {valid_count}/{len(population)} ({100*valid_count/len(population):.1f}%)")
            print(f"Unique molecules: {unique_count}/{len(population)} ({100*unique_count/len(population):.1f}%)")
            print(f"Diversity score: {diversity:.3f}")
            print(f"Max duplicates: {max_duplicates}")
            print(f"Best molecule: {best_mol}")
            
            # Enhanced selection with diversity preservation
            valid_survivors = [(mol, score) for mol, score in fitness_scores 
                             if self.is_valid_smiles(mol) and score > 0]
            
            if len(valid_survivors) < 5:
                # Add some simple valid molecules to maintain population
                simple_mols = ["CCO", "CC", "CO", "CCC", "CCN", "CN"]
                for simple_mol in simple_mols:
                    valid_survivors.append((simple_mol, self.fitness(simple_mol)))
                    if len(valid_survivors) >= 10:
                        break
            
            # Use diversity-aware selection
            survivors = self.diversity_selection(population, valid_survivors)
            
            # Create next generation
            new_population = survivors.copy()
            
            # Track genetic operator usage
            crossover_successes = 0
            mutation_successes = 0
            
            while len(new_population) < self.population_size:
                # Tournament selection with diversity consideration
                parent1 = random.choice(survivors)
                parent2 = random.choice(survivors)
                
                # Encourage diversity by avoiding identical parents
                attempts = 0
                while parent1 == parent2 and attempts < 5:
                    parent2 = random.choice(survivors)
                    attempts += 1
                
                # Smart crossover
                child1, child2 = self.crossover(parent1, parent2)
                if child1 != parent1 or child2 != parent2:
                    crossover_successes += 1
                
                # Smart mutation
                original_child1 = child1
                original_child2 = child2
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                if child1 != original_child1:
                    mutation_successes += 1
                if child2 != original_child2:
                    mutation_successes += 1
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
            
            # Print operator success rates
            total_operations = (self.population_size - len(survivors)) // 2
            if total_operations > 0:
                print(f"Crossover success rate: {100*crossover_successes/total_operations:.1f}%")
                print(f"Mutation success rate: {100*mutation_successes/(total_operations*2):.1f}%")
        
        return self.generation_stats
    
    def plot_evolution(self):
        """Plot the enhanced evolution progress with separate QED and Lipinski plots"""
        if not self.generation_stats:
            print("No evolution data to plot")
            return
            
        generations = [s['generation'] for s in self.generation_stats]
        best_fitnesses = [s['best_fitness'] for s in self.generation_stats]
        avg_fitnesses = [s['avg_fitness'] for s in self.generation_stats]
        max_duplicates = [s['max_duplicates'] for s in self.generation_stats]
        
        # Create figure with 2x2 layout
        fig = plt.figure(figsize=(16, 12))
        
        # Top left: Fitness Evolution
        ax1 = plt.subplot(2, 2, 1)
        plt.plot(generations, best_fitnesses, 'b-', label='Best Fitness', linewidth=3)
        plt.plot(generations, avg_fitnesses, 'r--', label='Average Fitness', linewidth=3)
        plt.xlabel('Generation', fontsize=14)
        plt.ylabel('Fitness Score', fontsize=14)
        plt.title('Fitness Evolution', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tick_params(axis='both', which='major', labelsize=12)
        
        # Top right: Duplication Control
        ax2 = plt.subplot(2, 2, 2)
        plt.plot(generations, max_duplicates, 'orange', label='Max Duplicates', linewidth=3)
        plt.axhline(y=self.max_identical, color='r', linestyle='--', alpha=0.7, 
                   label=f'Limit ({self.max_identical})', linewidth=2)
        plt.xlabel('Generation', fontsize=14)
        plt.ylabel('Max Duplicate Count', fontsize=14)
        plt.title('Duplication Control', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tick_params(axis='both', which='major', labelsize=12)
        
        # Calculate population statistics for each generation
        qed_means = []
        qed_stds = []
        qed_maxs = []
        qed_mins = []
        lipinski_means = []
        lipinski_stds = []
        lipinski_maxs = []
        lipinski_mins = []
        all_generations = []
        
        # Read all generation files to get population statistics
        for gen in range(1, len(self.generation_stats) + 1):
            filename = os.path.join(self.output_dir, f"generation_{gen:03d}.txt")
            all_generations.append(gen)
            
            try:
                # Read generation file
                qed_scores = []
                lipinski_scores = []
                
                with open(filename, 'r') as f:
                    lines = f.readlines()[1:]  # Skip header
                    for line in lines:
                        parts = line.strip().split('\t')
                        if len(parts) >= 7 and parts[1] == 'True':  # Valid molecule
                            qed = float(parts[2])
                            # Calculate Lipinski compliance from MW, LogP, HBA, HBD
                            mw = float(parts[4])
                            logp = float(parts[3])
                            hba = int(parts[5])
                            hbd = int(parts[6])
                            
                            violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
                            compliance = (4 - violations) / 4.0
                            
                            qed_scores.append(qed)
                            lipinski_scores.append(compliance)
                
                if qed_scores:  # If we have valid data
                    qed_means.append(np.mean(qed_scores))
                    qed_stds.append(np.std(qed_scores))
                    qed_maxs.append(np.max(qed_scores))
                    qed_mins.append(np.min(qed_scores))
                    
                    lipinski_means.append(np.mean(lipinski_scores))
                    lipinski_stds.append(np.std(lipinski_scores))
                    lipinski_maxs.append(np.max(lipinski_scores))
                    lipinski_mins.append(np.min(lipinski_scores))
                else:
                    # Fallback if no valid data
                    qed_means.append(0)
                    qed_stds.append(0)
                    qed_maxs.append(0)
                    qed_mins.append(0)
                    lipinski_means.append(0)
                    lipinski_stds.append(0)
                    lipinski_maxs.append(0)
                    lipinski_mins.append(0)
                    
            except (FileNotFoundError, ValueError, IndexError):
                # Fallback for missing or corrupted files
                qed_means.append(0)
                qed_stds.append(0)
                qed_maxs.append(0)
                qed_mins.append(0)
                lipinski_means.append(0)
                lipinski_stds.append(0)
                lipinski_maxs.append(0)
                lipinski_mins.append(0)
        
        # Convert to numpy arrays for easier manipulation
        qed_means = np.array(qed_means)
        qed_stds = np.array(qed_stds)
        qed_maxs = np.array(qed_maxs)
        qed_mins = np.array(qed_mins)
        lipinski_means = np.array(lipinski_means)
        lipinski_stds = np.array(lipinski_stds)
        lipinski_maxs = np.array(lipinski_maxs)
        lipinski_mins = np.array(lipinski_mins)
        
        # Bottom left: QED Statistics
        ax3 = plt.subplot(2, 2, 3)
        # Shaded region for mean ± std
        plt.fill_between(all_generations, qed_means - qed_stds, qed_means + qed_stds, 
                        alpha=0.3, color='blue', label='Mean ± Std Dev')
        # Mean line
        plt.plot(all_generations, qed_means, 'b-', label='Mean', linewidth=3)
        # Max/Min lines
        plt.plot(all_generations, qed_maxs, 'b--', label='Maximum', linewidth=2, alpha=0.8)
        plt.plot(all_generations, qed_mins, 'b:', label='Minimum', linewidth=2, alpha=0.8)
        
        # Reference lines
        plt.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, linewidth=2)
        plt.axhline(y=0.8, color='gray', linestyle=':', alpha=0.7, linewidth=2)
        
        plt.xlabel('Generation', fontsize=14)
        plt.ylabel('QED Score', fontsize=14)
        plt.title('QED Population Statistics', fontsize=16, fontweight='bold')
        plt.ylim(-0.05, 1.05)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tick_params(axis='both', which='major', labelsize=12)
        
        # Bottom right: Lipinski Statistics
        ax4 = plt.subplot(2, 2, 4)
        # Shaded region for mean ± std
        plt.fill_between(all_generations, lipinski_means - lipinski_stds, lipinski_means + lipinski_stds, 
                        alpha=0.3, color='green', label='Mean ± Std Dev')
        # Mean line
        plt.plot(all_generations, lipinski_means, 'g-', label='Mean', linewidth=3)
        # Max/Min lines
        plt.plot(all_generations, lipinski_maxs, 'g--', label='Maximum', linewidth=2, alpha=0.8)
        plt.plot(all_generations, lipinski_mins, 'g:', label='Minimum', linewidth=2, alpha=0.8)
        
        # Reference lines
        plt.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, linewidth=2)
        plt.axhline(y=0.8, color='gray', linestyle=':', alpha=0.7, linewidth=2)
        
        plt.xlabel('Generation', fontsize=14)
        plt.ylabel('Lipinski Compliance', fontsize=14)
        plt.title('Lipinski Population Statistics', fontsize=16, fontweight='bold')
        plt.ylim(-0.05, 1.05)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tick_params(axis='both', which='major', labelsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'smart_ga_evolution.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print final results
        print("\n" + "="*60)
        print("SMART CHEMICAL GA - FINAL RESULTS")
        print("="*60)
        best_final = self.generation_stats[-1]
        mol = Chem.MolFromSmiles(best_final['best_molecule'])
        
        if mol:
            print(f"Best molecule: {best_final['best_molecule']}")
            print(f"Final fitness: {best_final['best_fitness']:.3f}")
            print(f"QED score: {QED.qed(mol):.3f}")
            
            # Lipinski analysis
            mw = Descriptors.MolWt(mol)
            logp = Crippen.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            
            print(f"\nLipinski Rule of 5:")
            print(f"  Molecular Weight: {mw:.1f} ({'✓' if mw <= 500 else '✗'} ≤ 500)")
            print(f"  LogP: {logp:.2f} ({'✓' if logp <= 5 else '✗'} ≤ 5)")
            print(f"  H-bond donors: {hbd} ({'✓' if hbd <= 5 else '✗'} ≤ 5)")
            print(f"  H-bond acceptors: {hba} ({'✓' if hba <= 10 else '✗'} ≤ 10)")
        
        print(f"\nPopulation Statistics:")
        print(f"  Final diversity: {best_final['diversity']:.3f}")
        print(f"  Unique molecules: {best_final['unique_molecules']}/{self.population_size}")
        print(f"  Validity rate: {100*best_final['validity_rate']:.1f}%")
        print(f"  Max duplicates: {best_final['max_duplicates']}")
        
        print(f"\nAll outputs saved to: {self.output_dir}/")

# Usage example
if __name__ == "__main__":
    # Load fragment library
    print("Loading fragment library...")
    frag_lib = BRICSFragmentLibrary()
    frag_lib.load_library("zinc_brics_fragment_library.pkl")
    
    # Run Smart Chemical GA
    smart_ga = SmartChemicalGA(
        fragment_library=frag_lib,
        population_size=100,
        generations=50,
        mutation_rate=0.5,
        crossover_rate=0.5,
        diversity_weight=0.2,
        max_identical=3,
        output_dir="ga_output_smart"
    )
    
    stats = smart_ga.run_ga()
    
    # Visualize results
    smart_ga.plot_evolution()