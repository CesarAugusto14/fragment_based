# usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Genetic Algorithm for Drug Discovery using Fragment-Based Design

This script implements a basic genetic algorithm (GA) to evolve small drug-like molecules
using a fragment library. The GA optimizes for drug-likeness based on QED score and Lipinski's
Rule of 5 compliance. It includes mutation and crossover operations, and tracks the evolution
of the population over generations.

Author: cesarasa
Date: 08-18-2024
"""
import os
import random

import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, QED, Crippen, AllChem
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_distances

from fragmentation import BRICSFragmentLibrary

# Suppress RDKit warnings and errors
RDLogger.DisableLog('rdApp.*')

class SimpleDrugGA:
    def __init__(self,
                 fragment_library : BRICSFragmentLibrary,
                 population_size  : int = 50,
                 generations      : int = 50,
                 mutation_rate    : float = 0.7,
                 crossover_rate   : float = 0.7,
                 output_dir       : str = "ga_output_simple"
                 ):
        self.frag_lib = fragment_library
        
        # Get GA-ready fragments
        ga_frags = self.frag_lib.get_genetic_algorithm_fragments(min_frequency=5)
        self.terminals = [f['clean_smiles'] for f in ga_frags['terminals']]
        self.linkers = [f['clean_smiles'] for f in ga_frags['linkers']]
        
        # GA parameters
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        # Tracking
        self.generation_stats = []
        self.best_molecules = []
        
        # Create output directory
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def is_valid_smiles(self,
                        smiles : str
                        ) -> bool:
        """Check if SMILES string represents a valid molecule"""
        if not smiles:
            return False
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
            # Additional checks
            Chem.SanitizeMol(mol)
            return True
        except:
            return False
    
    def calculate_molecular_properties(self,
                                       smiles : str
                                       ) -> dict:
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
    
    def save_generation_data(self,
                             generation : int,
                             population : list):
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
    
    def calculate_ecfp4_fingerprints(self, smiles_list):
        """Calculate ECFP4 fingerprints for a list of SMILES"""
        fingerprints = []
        valid_smiles = []
        
        for smiles in smiles_list:
            if self.is_valid_smiles(smiles):
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        # Generate ECFP4 fingerprint (radius=2, 2048 bits)
                        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                        fingerprints.append(np.array(fp))
                        valid_smiles.append(smiles)
                except:
                    continue
        
        return np.array(fingerprints), valid_smiles
    
    def collect_generation_data(self, generation, population):
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
    
    def plot_mds_evolution(self, all_generation_data):
        """Plot single MDS with all valid SMILES colored by generation"""
        if not all_generation_data:
            print("No valid molecules found across generations")
            return
        
        print("Creating combined MDS plot...")
        
        # Collect all unique valid SMILES across all generations
        all_smiles = []
        generation_labels = []
        qed_scores = []
        
        for gen_data in all_generation_data:
            for mol_data in gen_data:
                all_smiles.append(mol_data['smiles'])
                generation_labels.append(mol_data['generation'])
                qed_scores.append(mol_data['qed'])
        
        if len(all_smiles) < 2:
            print("Not enough valid molecules for MDS analysis")
            return
        
        # Check for duplicates
        unique_smiles = list(set(all_smiles))
        if len(unique_smiles) < 2:
            print("Not enough unique valid molecules for MDS analysis")
            return
        
        print(f"Computing ECFP4 fingerprints for {len(all_smiles)} molecules...")
        
        # Calculate ECFP4 fingerprints for all molecules
        fingerprints = []
        valid_indices = []
        
        for i, smiles in enumerate(all_smiles):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                    fingerprints.append(np.array(fp))
                    valid_indices.append(i)
            except:
                continue
        
        if len(fingerprints) < 2:
            print("Not enough valid fingerprints for MDS")
            return
        
        fingerprints = np.array(fingerprints)
        
        # Filter corresponding data for valid fingerprints
        filtered_generations = [generation_labels[i] for i in valid_indices]
        filtered_qed = [qed_scores[i] for i in valid_indices]
        filtered_smiles = [all_smiles[i] for i in valid_indices]
        
        print(f"Computing MDS for {len(fingerprints)} valid molecules...")
        
        # Calculate cosine distances
        distances = cosine_distances(fingerprints)
        
        # Perform MDS
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, verbose=1)
        coords = mds.fit_transform(distances)
        
        # Create the plot
        plt.figure(figsize=(12, 10))
        
        # Get unique generations and create a color map
        unique_gens = sorted(set(filtered_generations))
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_gens)))
        
        # Plot each generation with different colors
        for i, gen in enumerate(unique_gens):
            gen_mask = np.array(filtered_generations) == gen
            gen_coords = coords[gen_mask]
            gen_qed = np.array(filtered_qed)[gen_mask]
            
            scatter = plt.scatter(gen_coords[:, 0], gen_coords[:, 1], 
                                c=colors[i], label=f'Generation {gen}', 
                                alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
        
        plt.xlabel('MDS Dimension 1', fontsize=12)
        plt.ylabel('MDS Dimension 2', fontsize=12)
        plt.title('Chemical Space Evolution - All Valid Molecules\n(MDS of ECFP4 Fingerprints, Colored by Generation)', 
                  fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'mds_evolution_combined.png'), 
                    dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create a second plot colored by QED score for comparison
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(coords[:, 0], coords[:, 1], 
                            c=filtered_qed, cmap='viridis', 
                            alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
        plt.colorbar(scatter, label='QED Score')
        plt.xlabel('MDS Dimension 1', fontsize=12)
        plt.ylabel('MDS Dimension 2', fontsize=12)
        plt.title('Chemical Space - All Valid Molecules\n(Colored by QED Score)', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'mds_evolution_qed.png'), 
                    dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"MDS plots saved to {self.output_dir}/")
        print(f"Total molecules plotted: {len(coords)}")
        print(f"Generations: {min(unique_gens)} to {max(unique_gens)}")
        print(f"QED range: {min(filtered_qed):.3f} to {max(filtered_qed):.3f}")
    
    def create_random_molecule(self):
        """Create a random valid molecule by connecting fragments"""
        max_attempts = 50
        
        for _ in range(max_attempts):
            try:
                # Strategy 1: Use just a terminal fragment (always valid)
                if random.random() < 0.3 and self.terminals:
                    candidate = random.choice(self.terminals)
                    if self.is_valid_smiles(candidate):
                        return candidate
                
                # Strategy 2: Use just a linker fragment
                if random.random() < 0.3 and self.linkers:
                    candidate = random.choice(self.linkers)
                    if self.is_valid_smiles(candidate):
                        return candidate
                
                # Strategy 3: Simple combination with validation
                if self.terminals and self.linkers:
                    terminal = random.choice(self.terminals)
                    linker = random.choice(self.linkers)
                    
                    # Try different combination strategies
                    combinations = [
                        f"{terminal}{linker}",  # Direct connection
                        f"{linker}{terminal}",  # Reverse
                        f"C{terminal}",         # With carbon bridge
                        f"C{linker}",
                        terminal,               # Just terminal
                        linker                  # Just linker
                    ]
                    
                    for combo in combinations:
                        if self.is_valid_smiles(combo):
                            return combo
                            
            except:
                continue
        
        # Fallback: return a simple valid molecule
        return "CCO"  # Ethanol - always valid
    
    def fitness(self, smiles):
        """Calculate fitness: QED score + Lipinski compliance"""
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
            
            # Lipinski bonus (fewer violations = higher score)
            lipinski_score = (4 - violations) / 4.0
            
            # Combined fitness
            fitness = 0.7 * qed_score + 0.3 * lipinski_score
            return fitness
            
        except:
            return 0
    
    def mutate(self, smiles):
        """Simple mutation: replace with a valid fragment"""
        if not smiles or random.random() > self.mutation_rate:
            return smiles
            
        max_attempts = 20
        
        for _ in range(max_attempts):
            try:
                # Strategy 1: Replace with a random terminal (most likely to be valid)
                if self.terminals and random.random() < 0.5:
                    new_mol = random.choice(self.terminals)
                    if self.is_valid_smiles(new_mol):
                        return new_mol
                
                # Strategy 2: Replace with a random linker
                if self.linkers and random.random() < 0.5:
                    new_mol = random.choice(self.linkers)
                    if self.is_valid_smiles(new_mol):
                        return new_mol
                
                # Strategy 3: Try simple modifications
                modifications = [
                    f"C{smiles}",      # Add carbon
                    f"{smiles}C",      # Add carbon at end
                    f"N{smiles}",      # Add nitrogen
                    f"{smiles}O",      # Add oxygen
                ]
                
                for mod in modifications:
                    if self.is_valid_smiles(mod):
                        return mod
                        
            except:
                continue
        
        # If all mutations fail, return original
        return smiles
    
    def crossover(self, parent1, parent2):
        """Simple crossover: try to combine parts validly"""
        if random.random() > self.crossover_rate:
            return parent1, parent2
            
        max_attempts = 10
        
        for _ in range(max_attempts):
            try:
                # Strategy 1: Simple replacement crossover
                if random.random() < 0.5:
                    # Child 1 gets parent1's "core" with parent2's influence
                    child1_candidates = [parent1, parent2, f"C{parent1}", f"{parent2}C"]
                    # Child 2 gets parent2's "core" with parent1's influence  
                    child2_candidates = [parent2, parent1, f"C{parent2}", f"{parent1}C"]
                else:
                    # Try simple substring swaps
                    mid1 = len(parent1) // 2
                    mid2 = len(parent2) // 2
                    
                    child1_candidates = [
                        parent1[:mid1] + parent2[mid2:],
                        parent2[:mid2] + parent1[mid1:],
                        parent1,  # Fallback
                        parent2   # Fallback
                    ]
                    child2_candidates = [
                        parent2[:mid2] + parent1[mid1:],
                        parent1[:mid1] + parent2[mid2:],
                        parent2,  # Fallback
                        parent1   # Fallback
                    ]
                
                # Find first valid child for each
                child1 = parent1  # Default fallback
                child2 = parent2  # Default fallback
                
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
        
        # If all crossovers fail, return parents
        return parent1, parent2
    
    def run_ga(self):
        """Run the genetic algorithm"""
        print("Starting Simple Drug Discovery GA...")
        print(f"Terminals: {len(self.terminals)}, Linkers: {len(self.linkers)}")
        print(f"Output directory: {self.output_dir}")
        
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
        
        # Store all generation data for combined MDS
        all_generation_data = []
        
        # Evolution loop
        for gen in range(self.generations):
            print(f"\nGeneration {gen + 1}/{self.generations}")
            
            # Save generation data to file
            self.save_generation_data(gen + 1, population)
            
            # Collect valid molecules for MDS
            gen_data = self.collect_generation_data(gen + 1, population)
            all_generation_data.append(gen_data)
            
            # Track validity statistics
            valid_count = sum(1 for mol in population if self.is_valid_smiles(mol))
            
            # Evaluate fitness (only for valid molecules)
            fitness_scores = []
            for mol in population:
                if self.is_valid_smiles(mol):
                    fitness_scores.append((mol, self.fitness(mol)))
                else:
                    fitness_scores.append((mol, 0.0))  # Invalid molecules get 0 fitness
            
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
                'validity_rate': valid_count / len(population)
            })
            
            print(f"Best fitness: {best_fitness:.3f}")
            print(f"Avg fitness: {avg_fitness:.3f}")
            print(f"Valid molecules: {valid_count}/{len(population)} ({100*valid_count/len(population):.1f}%)")
            print(f"Best molecule: {best_mol}")
            
            # Only let valid molecules survive and reproduce
            valid_survivors = [(mol, score) for mol, score in fitness_scores 
                             if self.is_valid_smiles(mol) and score > 0]
            
            if len(valid_survivors) < 5:  # Ensure minimum diversity
                # Add some simple valid molecules to maintain population
                simple_mols = ["CCO", "CC", "CO", "CCC", "CCN", "CN"]
                for simple_mol in simple_mols:
                    valid_survivors.append((simple_mol, self.fitness(simple_mol)))
                    if len(valid_survivors) >= 10:
                        break
            
            # Selection (top valid molecules)
            survivors = [mol for mol, _ in valid_survivors[:max(5, len(valid_survivors)//2)]]
            
            # Create next generation
            new_population = survivors.copy()
            
            while len(new_population) < self.population_size:
                # Tournament selection
                parent1 = random.choice(survivors)
                parent2 = random.choice(survivors)
                
                # Crossover
                child1, child2 = self.crossover(parent1, parent2)
                
                # Mutation
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
        
        # Create combined MDS plot after GA completes
        print("\nCreating combined MDS evolution plot...")
        self.plot_mds_evolution(all_generation_data)
        
        return self.generation_stats
    
    def plot_evolution(self):
        """Plot the evolution progress"""
        if not self.generation_stats:
            print("No evolution data to plot")
            return
            
        generations = [s['generation'] for s in self.generation_stats]
        best_fitnesses = [s['best_fitness'] for s in self.generation_stats]
        avg_fitnesses = [s['avg_fitness'] for s in self.generation_stats]
        
        plt.figure(figsize=(12, 5))
        
        # Fitness evolution
        plt.subplot(1, 2, 1)
        plt.plot(generations, best_fitnesses, 'b-', label='Best Fitness', linewidth=2)
        plt.plot(generations, avg_fitnesses, 'r--', label='Average Fitness', linewidth=2)
        plt.xlabel('Generation')
        plt.ylabel('Fitness Score')
        plt.title('Fitness Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Best molecules over time
        plt.subplot(1, 2, 2)
        final_stats = self.generation_stats[-5:]  # Last 5 generations
        
        # Analyze best molecules
        qed_scores = []
        lipinski_violations = []
        
        for stat in final_stats:
            mol_smiles = stat['best_molecule']
            try:
                mol = Chem.MolFromSmiles(mol_smiles)
                if mol:
                    qed = QED.qed(mol)
                    
                    # Calculate Lipinski violations
                    mw = Descriptors.MolWt(mol)
                    logp = Crippen.MolLogP(mol)
                    hbd = Descriptors.NumHDonors(mol)
                    hba = Descriptors.NumHAcceptors(mol)
                    
                    violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
                    
                    qed_scores.append(qed)
                    lipinski_violations.append(violations)
            except:
                qed_scores.append(0)
                lipinski_violations.append(4)
        
        gen_labels = [f"Gen {s['generation']}" for s in final_stats]
        
        x = range(len(gen_labels))
        plt.bar([i - 0.2 for i in x], qed_scores, 0.4, label='QED Score', alpha=0.7)
        plt.bar([i + 0.2 for i in x], [1 - v/4 for v in lipinski_violations], 0.4, 
                label='Lipinski Compliance', alpha=0.7)
        
        plt.xlabel('Generation')
        plt.ylabel('Score')
        plt.title('Drug-likeness Metrics')
        plt.xticks(x, gen_labels, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'ga_evolution.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print final results
        print("\n" + "="*50)
        print("FINAL RESULTS")
        print("="*50)
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
            
        print(f"\nAll outputs saved to: {self.output_dir}/")

# Usage example
if __name__ == "__main__":
    # Load your fragment library
    print("Loading fragment library...")
    frag_lib = BRICSFragmentLibrary()
    frag_lib.load_library("zinc_brics_fragment_library.pkl")
    
    # Run GA
    ga = SimpleDrugGA(frag_lib)
    stats = ga.run_ga()
    
    # Visualize results
    ga.plot_evolution()