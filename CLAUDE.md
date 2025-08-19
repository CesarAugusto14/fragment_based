# PROJECT CONTEXT & CORE DIRECTIVES

## Project Overview
[Fragment-Based Genetic Algorithm Drug Discovery] - [The current project implements a genetic algorithm using BRICS molecular fragments to evolve drug-like molecules through population-based optimization, targeting QED scores and Lipinski compliance for de novo drug design.]

## Domain Context
- **Fragment-Based Design**: BRICS decomposition of ZINC molecules into terminal/linker fragments for recombination
- **Genetic Algorithm**: Population-based optimization with selection, crossover, and mutation operators
- **Drug-likeness**: QED (Quantitative Estimate of Drug-likeness) combined with Lipinski Rule of 5 compliance
- **Chemical Space**: ECFP4 fingerprint-based MDS visualization for population diversity analysis
- **Target**: Generate novel drug-like molecules with high QED scores and chemical diversity

## Technical Stack:
- Language: Python 3.10+
- Cheminformatics: RDKit for molecular manipulation, validation, and property calculation
- Machine Learning: scikit-learn for MDS, cosine distance calculations
- Fragment Library: BRICS decomposition from ZINC database
- Data Processing: numpy for numerical operations, pandas-compatible output
- Visualization: matplotlib for evolution tracking, chemical space plots

## Data Requirements:
- **Fragment Library**: BRICS-decomposed molecules stored in `zinc_brics_fragment_library.pkl`
- **Population Data**: Generation-wise SMILES with validity, QED, LogP, MW, HBA, HBD
- **Chemical Space**: ECFP4 fingerprints for diversity analysis and MDS projection
- **Output Format**: Tab-separated files per generation in `ga_output/` directory

## Algorithm Architecture Considerations:
- **Validity**: % of generated SMILES that pass RDKit sanitization
- **Diversity**: Population uniqueness measured by Tanimoto distance
- **Convergence**: Balance between exploitation (high fitness) and exploration (diversity)
- **Drug-likeness**: Combined QED score (70%) + Lipinski compliance (30%)
- **Chemical Space Coverage**: MDS dispersion analysis using ECFP4 fingerprints
- **Fragment Utilization**: Terminal/linker fragment combination strategies

## Code Standards:
- **Molecular Validation**: RDKit SMILES validation with graceful error handling
- **Population Management**: Diversity preservation through selection pressure control
- **Genetic Operators**: Chemical-aware crossover and mutation maintaining validity
- **Performance Tracking**: Real-time statistics and comprehensive output logging
- **Reproducibility**: Random seed control, parameter documentation

## File Structure
```
ga_drug_discovery/
├── drug_discovery_ga.py          # Main SimpleDrugGA implementation
├── fragmentation.py               # BRICSFragmentLibrary management
├── zinc_brics_fragment_library.pkl # Pre-computed fragment database
├── ga_output/                     # Algorithm outputs
│   ├── generation_001.txt         # Per-generation molecular data
│   ├── generation_002.txt         # SMILES, validity, properties
│   ├── ...
│   ├── mds_evolution_combined.png  # Chemical space by generation
│   ├── mds_evolution_qed.png      # Chemical space by QED score
│   └── ga_evolution.png           # Fitness evolution tracking
├── CLAUDE.md                      # Project directives (this file)
└── README.md                      # User documentation
```

## Implemented Components:
- **SimpleDrugGA**: Core genetic algorithm with population evolution, fitness evaluation
- **Fragment Management**: Terminal/linker fragment selection from BRICS library
- **Molecular Generation**: Random molecule creation with validity validation
- **Genetic Operators**: Selection, crossover, mutation with chemical constraints
- **Property Calculation**: QED, LogP, MW, HBA, HBD using RDKit descriptors
- **Visualization**: MDS chemical space analysis, fitness evolution plots
- **Data Output**: Generation-wise molecular data export with comprehensive metrics

## Current Capabilities:
- BRICS fragment-based molecule generation from ZINC-derived library
- Population-based evolution with configurable parameters (size=50, generations=20)
- Combined fitness function balancing drug-likeness and Lipinski compliance
- Real-time diversity and validity tracking across generations
- Chemical space visualization using ECFP4 fingerprints and MDS projection
- Comprehensive output logging with molecular properties and statistics

## Critical Issues Identified:

### 🚨 Premature Convergence (Priority 1)
**Manifestation**: Single molecule `O=C1CCCCN1c1cc2n(n1)CCNC2CCCC` dominates population
- **Root Cause**: Excessive selection pressure favoring single high-fitness solution
- **Impact**: Loss of genetic diversity, poor chemical space exploration
- **Evidence**: 10+ identical copies in final generations

### 🚨 Population Diversity Collapse (Priority 1)  
**Manifestation**: Limited unique molecules per generation
- **Root Cause**: No diversity preservation mechanisms in selection/reproduction
- **Impact**: Algorithm converges to local optima, ignores novel chemical regions
- **Evidence**: MDS plots show clustering around single points

### 🚨 Inadequate Genetic Operators (Priority 2)
**Manifestation**: Poor molecular innovation despite adequate fitness function
- **Root Cause**: Simple string-based crossover/mutation insufficient for chemical complexity
- **Impact**: Limited ability to generate meaningful molecular variations
- **Evidence**: Fallback to simple molecules when fragment combination fails

## Existing Code Patterns:
- Fragment-based molecular construction using BRICS decomposition
- RDKit-centric molecular validation and property calculation
- Population-based evolution with elitist selection strategies
- Comprehensive output logging with generation-wise tracking
- ECFP4 fingerprint-based chemical space analysis
- Combined fitness optimization (QED + Lipinski compliance)

## Response Guidelines:
- Build upon existing `SimpleDrugGA` class architecture rather than complete rewrites
- Reference actual methods (`create_random_molecule`, `fitness`, `mutate`, `crossover`) when suggesting improvements
- Maintain consistency with BRICS fragment library and RDKit-based validation
- Leverage existing output structure (`ga_output/` directory, tab-separated format)
- Consider computational efficiency for population size 50+ over 20+ generations
- Always include molecular validity checks in genetic operator modifications
- Provide chemical rationale for algorithmic decisions
- Include diversity metrics in performance recommendations
- Reference established genetic algorithm principles for population management

## Common Pitfalls to Avoid:
- Ignoring chemical validity in genetic operator design
- Over-emphasizing fitness without diversity preservation
- Implementing selection pressure without population management
- Neglecting fragment combination chemistry in crossover operations
- Overlooking computational complexity in large population scenarios

## Preferred Implementation Style:
- Start with chemical/algorithmic context for proposed modifications
- Provide concrete code modifications to existing methods
- Include performance impact analysis and diversity considerations
- Reference genetic algorithm literature for population management strategies

## ANTI-PATTERN ELIMINATION

### Prohibited Implementation Patterns:
- Generic genetic algorithm advice without chemical context
- Population management without molecular validity considerations
- Fitness function modifications without diversity impact analysis
- Selection strategies ignoring chemical space exploration requirements
- Crossover/mutation operators breaking molecular structure integrity

### Prohibited Communication Patterns:
- Generic optimization advice not specific to molecular generation
- Multiple solution options without clear chemical reasoning
- Theoretical discussions without concrete implementation guidance
- Performance claims without computational complexity consideration

### Solution Space Optimization:
1. **Diversity-First Approach**: Prioritize population diversity preservation in all modifications
2. **Chemical Validity**: Ensure all genetic operators maintain molecular structure integrity  
3. **Performance Balance**: Optimize for both drug-likeness and chemical space exploration
4. **Computational Efficiency**: Consider scalability for larger populations/generations
5. **Measurable Outcomes**: Include specific metrics for diversity, validity, and fitness

## METACOGNITIVE PROCESSING

### Algorithm Optimization Loop:
1. **Convergence Pattern Recognition**: Identify premature convergence indicators
2. **Diversity Metrics**: Monitor population uniqueness and chemical space coverage
3. **Operator Effectiveness**: Evaluate genetic operator contribution to population evolution
4. **Performance Bottlenecks**: Identify computational constraints in molecular generation
5. **Chemical Validity**: Ensure molecular structure integrity throughout evolution

### Implementation Coherence Maintenance:
- Track population statistics and diversity metrics across generations
- Maintain consistency with BRICS fragment-based generation approach
- Reference existing fitness function components for modification proposals
- Build upon established output format and visualization framework

**ACTIVATION PROTOCOL**: This configuration optimizes genetic algorithm implementation for molecular generation, emphasizing diversity preservation, chemical validity, and drug-likeness optimization within the established BRICS fragment-based framework.