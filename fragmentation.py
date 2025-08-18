import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import BRICS, Draw, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random
from collections import Counter, defaultdict
import pickle
import re

class BRICSFragmentLibrary:
    def __init__(self):
        # Raw BRICS fragments with dummy atoms
        self.raw_fragments = set()
        self.raw_fragment_counts = Counter()
        
        # Cleaned fragments (dummy atoms replaced with H)
        self.clean_fragments = set()
        self.clean_fragment_counts = Counter()
        
        # Fragment metadata for genetic algorithm
        self.fragment_metadata = {}  # Maps clean_smiles -> metadata dict
        self.fragment_types = defaultdict(list)  # Organized by connection pattern
        
        # BRICS bond type reference
        self.brics_bond_types = {
            1: "C-C (aliphatic)",
            3: "C-O", 
            4: "C-C (aromatic/special)",
            5: "C-N",
            6: "C-S",
            7: "C-C (carbonyl)", 
            8: "C-C (special contexts)",
            9: "N-N",
            10: "N-O",
            11: "N-S",
            16: "C-aromatic"
        }
    
    def clean_brics_fragment(self, brics_smiles):
        """
        Convert BRICS fragment with dummy atoms to clean SMILES
        and extract connection point information
        """
        # Extract dummy atom types and their counts
        dummy_pattern = r'\[(\d+)\*\]'
        dummy_atoms = re.findall(dummy_pattern, brics_smiles)
        
        # Replace dummy atoms with hydrogen for clean structure
        clean_smiles = re.sub(dummy_pattern, '[H]', brics_smiles)
        
        # Try to create molecule to validate
        try:
            mol = Chem.MolFromSmiles(clean_smiles)
            if mol is not None:
                # Canonicalize the SMILES
                canonical_smiles = Chem.MolToSmiles(mol)
                
                # Create metadata
                metadata = {
                    'original_brics': brics_smiles,
                    'clean_smiles': canonical_smiles,
                    'connection_points': dummy_atoms,
                    'num_connections': len(dummy_atoms),
                    'connection_types': list(set(dummy_atoms)),
                    'bond_descriptions': [self.brics_bond_types.get(int(d), f"Unknown-{d}") for d in set(dummy_atoms)]
                }
                
                return canonical_smiles, metadata
        except Exception as e:
            print(f"Error cleaning fragment {brics_smiles}: {e}")
        
        return None, None
    
    def add_fragment(self, brics_fragment):
        """
        Add a BRICS fragment to the library (both raw and cleaned)
        """
        # Add raw fragment
        self.raw_fragments.add(brics_fragment)
        self.raw_fragment_counts[brics_fragment] += 1
        
        # Clean and add cleaned fragment
        clean_smiles, metadata = self.clean_brics_fragment(brics_fragment)
        if clean_smiles and metadata:
            self.clean_fragments.add(clean_smiles)
            self.clean_fragment_counts[clean_smiles] += 1
            
            # Store metadata (update counts if already exists)
            if clean_smiles in self.fragment_metadata:
                self.fragment_metadata[clean_smiles]['total_count'] += 1
                self.fragment_metadata[clean_smiles]['brics_variants'].add(brics_fragment)
            else:
                metadata['total_count'] = 1
                metadata['brics_variants'] = {brics_fragment}
                self.fragment_metadata[clean_smiles] = metadata
            
            # Categorize by connection pattern
            connection_key = f"{metadata['num_connections']}_connections_" + "_".join(sorted(metadata['connection_types']))
            
            # Check if this clean fragment is already in this category
            existing = next((f for f in self.fragment_types[connection_key] if f['clean_smiles'] == clean_smiles), None)
            if existing:
                existing['count'] += 1
                existing['brics_variants'].add(brics_fragment)
            else:
                self.fragment_types[connection_key].append({
                    'clean_smiles': clean_smiles,
                    'count': 1,
                    'brics_variants': {brics_fragment},
                    'metadata': metadata
                })

    def load_zinc_dataset(self, parquet_path):
        """Load ZINC dataset from parquet file"""
        print(f"Loading ZINC dataset from {parquet_path}")
        df = pd.read_parquet(parquet_path)
        print(f"Loaded {len(df)} molecules")
        return df
    
    def extract_fragments_from_molecule(self, smiles):
        """Extract BRICS fragments from a single molecule"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return []
            
            # Perform BRICS decomposition
            fragments = BRICS.BRICSDecompose(mol)
            return list(fragments)
        except Exception as e:
            print(f"Error processing {smiles}: {e}")
            return []
    
    def build_fragment_library(self, smiles_list, max_molecules=None):
        """Build fragment library from list of SMILES"""
        if max_molecules:
            smiles_list = smiles_list[:max_molecules]
            
        print(f"Processing {len(smiles_list)} molecules for fragmentation...")
        
        for i, smiles in enumerate(smiles_list):
            if i % 10000 == 0:
                print(f"Processed {i}/{len(smiles_list)} molecules")
                print(f"  Raw fragments: {len(self.raw_fragments)}")
                print(f"  Clean fragments: {len(self.clean_fragments)}")
            
            brics_fragments = self.extract_fragments_from_molecule(smiles)
            for frag in brics_fragments:
                self.add_fragment(frag)
        
        print(f"\nFragment library built!")
        print(f"  Raw BRICS fragments: {len(self.raw_fragments)}")
        print(f"  Unique clean fragments: {len(self.clean_fragments)}")
        return self.clean_fragments

    def visualize_molecule_fragmentation(self, smiles, title="Molecule Fragmentation", save_path=None):
        """Visualize a molecule and its BRICS fragments with improved layout"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Invalid SMILES: {smiles}")
            return None
        
        # Get fragments
        brics_fragments = self.extract_fragments_from_molecule(smiles)
        
        print(f"Molecule: {smiles}")
        print(f"Found {len(brics_fragments)} BRICS fragments:")
        
        clean_fragments = []
        for i, brics_frag in enumerate(brics_fragments):
            clean_smiles, metadata = self.clean_brics_fragment(brics_frag)
            if clean_smiles and metadata:
                print(f"  Fragment {i+1}: {brics_frag} -> {clean_smiles}")
                print(f"    Connections: {metadata['num_connections']} ({', '.join(metadata['bond_descriptions'])})")
                clean_fragments.append((clean_smiles, metadata))
            else:
                print(f"  Fragment {i+1}: {brics_frag} (failed to clean)")
        
        # Create molecule objects from clean fragment SMILES
        fragment_mols = []
        for clean_smiles, metadata in clean_fragments:
            frag_mol = Chem.MolFromSmiles(clean_smiles)
            if frag_mol is not None:
                fragment_mols.append((frag_mol, clean_smiles, metadata))
        
        # Calculate grid layout for better organization
        n_fragments = len(fragment_mols)
        if n_fragments == 0:
            print("No valid fragments to display")
            return brics_fragments, clean_fragments
        
        # Create a grid layout: original molecule on top, fragments below
        total_items = n_fragments + 1  # +1 for original molecule
        cols = min(4, total_items)  # Max 4 columns
        rows = (total_items + cols - 1) // cols  # Ceiling division
        
        # Create larger figure with better spacing
        fig = plt.figure(figsize=(cols * 5, rows * 6))
        
        # Draw original molecule (span multiple columns if needed)
        ax_orig = plt.subplot2grid((rows, cols), (0, 0), colspan=min(cols, 2))
        
        # Generate molecule image with larger size for clarity
        img = Draw.MolToImage(mol, size=(400, 400))
        ax_orig.imshow(img)
        
        # Clean up title - truncate long SMILES
        short_smiles = smiles[:50] + "..." if len(smiles) > 50 else smiles
        ax_orig.set_title(f"Original Molecule\n{short_smiles}", fontsize=14, fontweight='bold', pad=20)
        ax_orig.axis('off')
        
        # Draw fragments in organized grid
        for i, (frag_mol, clean_smiles, metadata) in enumerate(fragment_mols):
            # Calculate position in grid
            if cols <= 2:  # If original takes 2 columns, start fragments from next row
                row = (i // cols) + 1
                col = i % cols
            else:  # If original takes 1 column, start fragments from position 2
                if i == 0 and cols > 2:
                    row = 0
                    col = 2
                else:
                    adj_i = i + 1 if cols > 2 else i
                    row = adj_i // cols
                    col = adj_i % cols
                    if cols > 2 and row == 0 and col >= 2:
                        col += 1
            
            if row < rows and col < cols:
                ax = plt.subplot2grid((rows, cols), (row, col))
                
                # Generate fragment image
                frag_img = Draw.MolToImage(frag_mol, size=(300, 300))
                ax.imshow(frag_img)
                
                # Create informative title
                conn_type = "Terminal" if metadata['num_connections'] == 1 else f"{metadata['num_connections']}-way"
                bond_info = metadata['bond_descriptions'][0] if len(metadata['bond_descriptions']) == 1 else "Mixed"
                
                # Truncate long fragment names
                display_smiles = clean_smiles[:20] + "..." if len(clean_smiles) > 20 else clean_smiles
                
                ax.set_title(f"Fragment {i+1}\n{display_smiles}\n({conn_type} - {bond_info})", 
                            fontsize=12, pad=15)
                ax.axis('off')
        
        # Add overall title
        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.95)
        
        # Adjust layout to prevent overlap
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        
        # Save the figure
        save_filename = save_path or "molecule_fragmentation_improved_2.png"
        plt.savefig(save_filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Visualization saved as: {save_filename}")
        
        # Also save individual fragment images for detailed inspection
        if len(fragment_mols) > 0:
            fig_fragments, axes_frag = plt.subplots(1, min(len(fragment_mols), 5), figsize=(20, 4))
            if len(fragment_mols) == 1:
                axes_frag = [axes_frag]
            
            for i, (frag_mol, clean_smiles, metadata) in enumerate(fragment_mols[:5]):
                img = Draw.MolToImage(frag_mol, size=(400, 400))
                axes_frag[i].imshow(img)
                axes_frag[i].set_title(f"{clean_smiles}\n({metadata['num_connections']} connections)", fontsize=14)
                axes_frag[i].axis('off')
            
            plt.suptitle("Fragment Details", fontsize=16)
            plt.tight_layout()
            detail_filename = save_path.replace('.png', '_details.png') if save_path else "fragment_details.png"
            plt.savefig(detail_filename, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Fragment details saved as: {detail_filename}")
        
        plt.show()
        return brics_fragments, clean_fragments

    def visualize_common_fragments(self, n_fragments=10, save_path="common_fragments.png"):
        """Visualize the most common fragments in the library"""
        common_frags = self.get_most_common_fragments(n_fragments, 'clean')
        
        if not common_frags:
            print("No fragments to visualize")
            return
        
        # Create grid layout
        cols = min(5, len(common_frags))
        rows = (len(common_frags) + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
        if rows == 1:
            axes = axes.reshape(1, -1) if len(common_frags) > 1 else [axes]
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i, (frag_smiles, count) in enumerate(common_frags):
            row = i // cols
            col = i % cols
            
            try:
                mol = Chem.MolFromSmiles(frag_smiles)
                if mol:
                    img = Draw.MolToImage(mol, size=(300, 300))
                    axes[row, col].imshow(img)
                    
                    # Get metadata for this fragment
                    metadata = self.fragment_metadata.get(frag_smiles, {})
                    conn_info = f"{metadata.get('num_connections', '?')} conn"
                    
                    axes[row, col].set_title(f"{frag_smiles}\n{count}x ({conn_info})", fontsize=10)
                else:
                    axes[row, col].text(0.5, 0.5, f"Invalid:\n{frag_smiles}", ha='center', va='center')
            except:
                axes[row, col].text(0.5, 0.5, f"Error:\n{frag_smiles}", ha='center', va='center')
            
            axes[row, col].axis('off')
        
        # Hide empty subplots
        for i in range(len(common_frags), rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.suptitle(f"Top {len(common_frags)} Most Common Fragments", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Common fragments visualization saved as: {save_path}")
        plt.show()
    def get_most_common_fragments(self, n=20, fragment_type='clean'):
        """Get the most common fragments (raw BRICS or clean)"""
        if fragment_type == 'clean':
            return self.clean_fragment_counts.most_common(n)
        else:
            return self.raw_fragment_counts.most_common(n)
    
    def get_fragments_by_connection_type(self, num_connections=None, connection_type=None, min_frequency=1):
        """Get fragments filtered by connection characteristics"""
        results = []
        
        for key, fragments in self.fragment_types.items():
            include = True
            
            if num_connections is not None:
                if not key.startswith(f"{num_connections}_connections"):
                    include = False
            
            if connection_type is not None:
                if connection_type not in key:
                    include = False
            
            if include:
                # Filter by frequency and sort
                filtered = [f for f in fragments if f['count'] >= min_frequency]
                filtered.sort(key=lambda x: x['count'], reverse=True)
                results.extend(filtered)
        
        return results
    
    def get_genetic_algorithm_fragments(self, min_frequency=5):
        """Get fragments organized for genetic algorithm use"""
        return {
            'terminals': self.get_fragments_by_connection_type(num_connections=1, min_frequency=min_frequency),
            'linkers': self.get_fragments_by_connection_type(num_connections=2, min_frequency=min_frequency),
            'branches': self.get_fragments_by_connection_type(num_connections=3, min_frequency=min_frequency),
            'all_clean': [(smiles, count) for smiles, count in self.clean_fragment_counts.most_common() if count >= min_frequency]
        }
    
    def print_library_analysis(self):
        """Print comprehensive analysis of the fragment library"""
        print(f"\n{'='*60}")
        print(f"BRICS Fragment Library Analysis")
        print(f"{'='*60}")
        
        print(f"Raw BRICS fragments: {len(self.raw_fragments)}")
        print(f"Unique clean fragments: {len(self.clean_fragments)}")
        
        print(f"\n--- Fragment Types by Connection Pattern ---")
        for conn_type, fragments in self.fragment_types.items():
            total_count = sum(f['count'] for f in fragments)
            print(f"{conn_type}: {len(fragments)} unique ({total_count} total)")
        
        print(f"\n--- Most Common Clean Fragments ---")
        for i, (frag, count) in enumerate(self.clean_fragment_counts.most_common(15)):
            metadata = self.fragment_metadata[frag]
            conn_info = f"{metadata['num_connections']} conn"
            print(f"{i+1:2d}. {frag:<40} {count:>4}x ({conn_info})")
        
        print(f"\n--- Terminal Groups (1 connection) ---")
        terminals = self.get_fragments_by_connection_type(num_connections=1)[:10]
        for i, frag in enumerate(terminals):
            bonds = ", ".join(frag['metadata']['bond_descriptions'])
            print(f"{i+1:2d}. {frag['clean_smiles']:<30} {frag['count']:>3}x ({bonds})")
        
        print(f"\n--- Linker Groups (2 connections) ---")
        linkers = self.get_fragments_by_connection_type(num_connections=2)[:10]
        for i, frag in enumerate(linkers):
            bonds = ", ".join(frag['metadata']['bond_descriptions'])
            print(f"{i+1:2d}. {frag['clean_smiles']:<30} {frag['count']:>3}x ({bonds})")
        
        # GA readiness summary
        ga_frags = self.get_genetic_algorithm_fragments()
        print(f"\n--- Genetic Algorithm Ready Fragments (freq ≥ 5) ---")
        print(f"Terminal groups: {len(ga_frags['terminals'])}")
        print(f"Linker groups: {len(ga_frags['linkers'])}")
        print(f"Branching points: {len(ga_frags['branches'])}")
        print(f"Total usable: {len(ga_frags['all_clean'])}")
    
    def save_library(self, filepath):
        """Save fragment library to file"""
        library_data = {
            'raw_fragments': list(self.raw_fragments),
            'raw_fragment_counts': dict(self.raw_fragment_counts),
            'clean_fragments': list(self.clean_fragments),
            'clean_fragment_counts': dict(self.clean_fragment_counts),
            'fragment_metadata': {k: {**v, 'brics_variants': list(v['brics_variants'])} for k, v in self.fragment_metadata.items()},
            'fragment_types': {k: v for k, v in self.fragment_types.items()}
        }
        with open(filepath, 'wb') as f:
            pickle.dump(library_data, f)
        print(f"Fragment library saved to {filepath}")
    
    def load_library(self, filepath):
        """Load fragment library from file"""
        with open(filepath, 'rb') as f:
            library_data = pickle.load(f)
        
        self.raw_fragments = set(library_data['raw_fragments'])
        self.raw_fragment_counts = Counter(library_data['raw_fragment_counts'])
        self.clean_fragments = set(library_data['clean_fragments'])
        self.clean_fragment_counts = Counter(library_data['clean_fragment_counts'])
        
        # Restore metadata with brics_variants as sets
        self.fragment_metadata = {}
        for k, v in library_data['fragment_metadata'].items():
            metadata = v.copy()
            metadata['brics_variants'] = set(metadata['brics_variants'])
            self.fragment_metadata[k] = metadata
            
        self.fragment_types = defaultdict(list, library_data['fragment_types'])
        
        print(f"Fragment library loaded from {filepath}")
        print(f"  Raw fragments: {len(self.raw_fragments)}")
        print(f"  Clean fragments: {len(self.clean_fragments)}")

# Example usage and demonstration
if __name__ == "__main__":
    # Initialize fragment library
    frag_lib = BRICSFragmentLibrary()
    from time import time
    t1 = time()
    # Example with your ZINC dataset
    try:
        # Load your dataset
        print('Loading ZINC dataset...')
        df = frag_lib.load_zinc_dataset("../mol_generative/zinc_test/zinc_drug_like_100_million_all.parquet")
        # df = df.sample(1000, random_state=42)  # Sample 1000
        # Get SMILES column (adjust column name as needed)
        smiles_col = 'smiles'  # or whatever your column is named
        if smiles_col not in df.columns:
            print(f"Available columns: {df.columns.tolist()}")
            smiles_col = df.columns[0]  # Use first column if 'smiles' not found
        
        smiles_list = df[smiles_col].tolist()
        
        # Demonstrate with a random molecule first
        random_smiles = random.choice(smiles_list)
        print(f"Demonstrating with random molecule: {random_smiles}")
        brics_frags, clean_frags = frag_lib.visualize_molecule_fragmentation(random_smiles, "Random ZINC Molecule Fragmentation")
        
        # Build fragment library (start with subset for testing)
        print("\nBuilding fragment library...")
        fragments = frag_lib.build_fragment_library(smiles_list, max_molecules=100000)  # Start with 10k molecules
        
        # Print comprehensive analysis
        frag_lib.print_library_analysis()
        
        # Save the library
        frag_lib.save_library("zinc_brics_fragment_library.pkl")
        
        # Show how to get GA-ready fragments
        print(f"\n--- Ready for Genetic Algorithm ---")
        ga_fragments = frag_lib.get_genetic_algorithm_fragments(min_frequency=10)
        
        print("Top terminal groups for mutations:")
        for i, frag in enumerate(ga_fragments['terminals'][:5]):
            print(f"  {frag['clean_smiles']} (freq: {frag['count']})")
        
        print("\nTop linkers for crossovers:")
        for i, frag in enumerate(ga_fragments['linkers'][:5]):
            print(f"  {frag['clean_smiles']} (freq: {frag['count']})")
        
        t2 = time()
        print(f"\n Total time: {t2 - t1:.2f} seconds")
    except FileNotFoundError:
        print("ZINC dataset not found. Using example molecules for demonstration...")
        
        # Example