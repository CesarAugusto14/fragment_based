# Drug Discovery Genetic Algorithm

A Python-based genetic algorithm for automated drug discovery using fragment-based molecular generation and multi-objective optimization.

## Quick Start

```bash
# Install dependencies
pip install rdkit matplotlib numpy scikit-learn

# Run the genetic algorithm
python drug_discovery_ga.py
```

## Overview

This project implements a genetic algorithm to evolve drug-like molecules using:
- **Fragment-based generation** with BRICS decomposition
- **Multi-objective fitness** combining QED score and Lipinski compliance
- **Chemical space visualization** using ECFP4 fingerprints and MDS
- **Comprehensive output tracking** with generation-by-generation analysis

## Project Structure

```
drug-discovery-ga/
├── drug_discovery_ga.py          # Main GA implementation
├── fragmentation.py               # Fragment library management
├── zinc_brics_fragment_library.pkl  # Pre-computed fragment database
├── ga_output/                     # Generated results
│   ├── generation_001.txt         # Molecular data per generation
│   ├── generation_002.txt
│   ├── ...
│   ├── mds_evolution_combined.png  # Chemical space evolution
│   ├── mds_evolution_qed.png      # QED distribution plot
│   └── ga_evolution.png           # Fitness evolution graph
├── README.md                      # This file
└── CLAUDE.md                      # Detailed analysis and recommendations
```

## Current Features

### Core Algorithm
- **Population Size**: 50 molecules per generation
- **Evolution**: 20 generations with selection, crossover, and mutation
- **Fitness Function**: 70% QED score + 30% Lipinski compliance
- **Fragment Library**: BRICS-decomposed molecular fragments from ZINC database

### Molecular Validation
- **SMILES Validation**: RDKit-based molecular structure validation
- **Property Calculation**: QED, LogP, Molecular Weight, HBA/HBD
- **Error Handling**: Graceful handling of invalid molecular structures

### Visualization & Output
- **Generation Files**: Tab-separated molecular data for each generation
- **MDS Plots**: Chemical space evolution using ECFP4 fingerprints
- **Fitness Tracking**: Real-time optimization progress monitoring
- **Lipinski Analysis**: Drug-likeness assessment for final molecules

## Current Status

### Working
- Basic genetic algorithm framework
- Molecular generation and validation
- Fitness evaluation and selection
- Data output and visualization
- Chemical space analysis

### Known Issues
- **Premature Convergence**: Population dominated by single high-fitness molecule
- **Limited Diversity**: Poor exploration of chemical space
- **Simple Genetic Operators**: Basic crossover/mutation insufficient for complexity

### Target Metrics
- **Diversity**: >80% unique molecules per generation
- **Quality**: Mean QED >0.6, >90% Lipinski compliance
- **Coverage**: Broad chemical space exploration

## Dependencies

```python
# Core dependencies
rdkit-pypi>=2022.9.5    # Molecular informatics
numpy>=1.21.0           # Numerical computing
matplotlib>=3.5.0       # Plotting and visualization
scikit-learn>=1.0.0     # Machine learning (MDS, clustering)

# Fragment library (custom)
fragmentation.py        # BRICS fragment management
```

## Usage Examples

### Basic Evolution
```python
from fragmentation import BRICSFragmentLibrary

# Load fragment library
frag_lib = BRICSFragmentLibrary()
frag_lib.load_library("zinc_brics_fragment_library.pkl")

# Initialize and run GA
ga = SimpleDrugGA(frag_lib)
stats = ga.run_ga()

# Visualize results
ga.plot_evolution()
```

### Analysis of Results
```python
# Access generation statistics
for stat in ga.generation_stats:
    print(f"Gen {stat['generation']}: Best QED = {stat['best_fitness']:.3f}")

# Analyze final population
best_molecule = ga.generation_stats[-1]['best_molecule']
print(f"Best molecule: {best_molecule}")
```

## Output Files

### Generation Data (`generation_XXX.txt`)
```
SMILES	Valid	QED	LogP	MW	HBA	HBD
O=C1CCCCN1c1cc2n(n1)CCNC2CCCC	True	0.9172	2.2345	276.38	4	1
C[C@H]1CNCCOC1	True	0.4881	0.2423	115.18	2	1
...
```

## Development Roadmap

### Phase 1: Diversity Enhancement (Priority)
- [ ] Implement maximum identical molecule limits
- [ ] Add tournament selection mechanism
- [ ] Introduce diversity-based fitness penalties
- [ ] Periodic injection of random molecules

### Phase 2: Advanced Genetic Operators
- [ ] Fragment-aware crossover operations
- [ ] Chemical knowledge-guided mutations
- [ ] Ring system modifications
- [ ] Functional group transformations

### Phase 3: Multi-Objective Optimization
- [ ] Pareto front optimization
- [ ] Dynamic fitness landscapes
- [ ] Chemical space clustering
- [ ] Novelty-based selection

### Phase 4: Production Features
- [ ] Parallel evolution (island model)
- [ ] Database integration
- [ ] Synthetic accessibility scoring
- [ ] Target-specific optimization


## References

- **RDKit**: Open-source cheminformatics toolkit
- **BRICS**: Breaking of Retrosynthetically Interesting Chemical Substructures
- **QED**: Quantitative Estimate of Drug-likeness
- **Lipinski Rule of 5**: Drug-likeness criteria
- **ECFP**: Extended Connectivity Fingerprints

## Support

For questions, issues, or contributions:
- **Issues**: [GitHub Issues](https://github.com/fragment_based/issues)
- **Discussions**: [GitHub Discussions](https://github.com/fragment_based/discussions)
- **Email**: cesarasa@ttu.edu

---
