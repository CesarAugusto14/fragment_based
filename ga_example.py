import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import random

class GeneticAlgorithm:
    """
    A comprehensive Genetic Algorithm implementation for function optimization.
    
    This class demonstrates the core GA components: population management,
    selection, crossover, mutation, and fitness evaluation.
    """
    
    def __init__(self, fitness_function, bounds, population_size=50, 
                 mutation_rate=0.1, crossover_rate=0.8, elitism_rate=0.1):
        """
        Initialize the Genetic Algorithm.
        
        Args:
            fitness_function: Function to minimize (we'll handle the maximization internally)
            bounds: List of (min, max) tuples for each variable
            population_size: Number of individuals in each generation
            mutation_rate: Probability of mutation for each gene
            crossover_rate: Probability of crossover between selected parents
            elitism_rate: Fraction of best individuals to preserve each generation
        """
        self.fitness_function = fitness_function
        self.bounds = bounds
        self.dimensions = len(bounds)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_count = max(1, int(elitism_rate * population_size))
        
        # Track evolution progress
        self.best_fitness_history = []
        self.average_fitness_history = []
    
    def initialize_population(self):
        """
        Create initial population with random individuals within bounds.
        Each individual is a real-valued vector representing a potential solution.
        """
        population = []
        for _ in range(self.population_size):
            individual = []
            for min_val, max_val in self.bounds:
                # Generate random value within bounds for each dimension
                individual.append(random.uniform(min_val, max_val))
            population.append(individual)
        return population
    
    def evaluate_fitness(self, population):
        """
        Evaluate fitness for entire population.
        Since we're minimizing, we convert to maximization by negating.
        """
        fitness_scores = []
        for individual in population:
            # Convert minimization to maximization problem
            raw_fitness = self.fitness_function(individual)
            # Add offset to ensure positive fitness values for selection
            fitness_scores.append(-raw_fitness + 1000)  
        return fitness_scores
    
    def tournament_selection(self, population, fitness_scores, tournament_size=3):
        """
        Tournament selection: randomly select a few individuals and pick the best.
        This maintains selection pressure while preserving diversity.
        """
        selected_parents = []
        
        for _ in range(self.population_size - self.elitism_count):
            # Randomly select tournament_size individuals
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            
            # Select the individual with highest fitness (best performance)
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected_parents.append(population[winner_idx].copy())
        
        return selected_parents
    
    def blend_crossover(self, parent1, parent2, alpha=0.5):
        """
        Blend crossover (BLX-α): creates offspring by blending parent genes.
        This works well for real-valued optimization problems.
        """
        if random.random() > self.crossover_rate:
            # No crossover, return parents as-is
            return parent1.copy(), parent2.copy()
        
        offspring1, offspring2 = [], []
        
        for gene1, gene2 in zip(parent1, parent2):
            # Calculate blending range
            min_gene, max_gene = min(gene1, gene2), max(gene1, gene2)
            range_size = max_gene - min_gene
            
            # Extend range by alpha factor on both sides
            lower_bound = min_gene - alpha * range_size
            upper_bound = max_gene + alpha * range_size
            
            # Generate offspring genes within blended range
            offspring1.append(random.uniform(lower_bound, upper_bound))
            offspring2.append(random.uniform(lower_bound, upper_bound))
        
        return offspring1, offspring2
    
    def gaussian_mutation(self, individual, mutation_strength=0.1):
        """
        Apply Gaussian mutation to an individual.
        Each gene has a chance to be perturbed by a random Gaussian value.
        """
        mutated = individual.copy()
        
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                # Apply Gaussian perturbation scaled by the variable's range
                min_val, max_val = self.bounds[i]
                noise_scale = (max_val - min_val) * mutation_strength
                perturbation = random.gauss(0, noise_scale)
                
                # Add perturbation and ensure bounds compliance
                mutated[i] += perturbation
                mutated[i] = max(min_val, min(max_val, mutated[i]))
        
        return mutated
    
    def evolve_generation(self, population, fitness_scores):
        """
        Create the next generation through selection, crossover, and mutation.
        """
        # Elitism: preserve the best individuals
        elite_indices = np.argsort(fitness_scores)[-self.elitism_count:]
        new_population = [population[i].copy() for i in elite_indices]
        
        # Selection: choose parents for reproduction
        parents = self.tournament_selection(population, fitness_scores)
        
        # Crossover and mutation: create offspring
        offspring = []
        for i in range(0, len(parents) - 1, 2):
            parent1, parent2 = parents[i], parents[i + 1]
            
            # Create two offspring through crossover
            child1, child2 = self.blend_crossover(parent1, parent2)
            
            # Apply mutation to offspring
            child1 = self.gaussian_mutation(child1)
            child2 = self.gaussian_mutation(child2)
            
            offspring.extend([child1, child2])
        
        # Combine elites with offspring to form new generation
        new_population.extend(offspring[:self.population_size - len(new_population)])
        
        return new_population
    
    def optimize(self, max_generations=100, target_fitness=None, verbose=True):
        """
        Run the genetic algorithm optimization process.
        """
        # Initialize population and tracking variables
        population = self.initialize_population()
        best_individual = None
        best_fitness_value = float('inf')
        
        for generation in range(max_generations):
            # Evaluate current population
            fitness_scores = self.evaluate_fitness(population)
            
            # Find best individual in current generation
            best_idx = np.argmax(fitness_scores)
            current_best_fitness = -fitness_scores[best_idx] + 1000  # Convert back to original scale
            
            # Update global best if we found a better solution
            if current_best_fitness < best_fitness_value:
                best_fitness_value = current_best_fitness
                best_individual = population[best_idx].copy()
            
            # Track evolution progress
            self.best_fitness_history.append(best_fitness_value)
            avg_fitness = np.mean([-f + 1000 for f in fitness_scores])
            self.average_fitness_history.append(avg_fitness)
            
            # Print progress information
            if verbose and (generation % 10 == 0 or generation == max_generations - 1):
                print(f"Generation {generation:3d}: Best = {best_fitness_value:.6f}, "
                      f"Average = {avg_fitness:.6f}")
            
            # Check if we've reached our target
            if target_fitness and best_fitness_value <= target_fitness:
                print(f"Target fitness reached in generation {generation}")
                break
            
            # Create next generation (skip on last iteration)
            if generation < max_generations - 1:
                population = self.evolve_generation(population, fitness_scores)
        
        return best_individual, best_fitness_value

# Define the Rastrigin function - a classic multimodal optimization test function
def rastrigin_function(x):
    """
    Rastrigin function: a challenging test function with many local minima.
    Global minimum is at x = [0, 0, ...] with f(x) = 0.
    
    This function is particularly difficult because it has many local optima
    that can trap gradient-based methods, making it ideal for testing GAs.
    """
    A = 100
    n = len(x)
    return A * n + sum([xi**2 - A * np.cos(2 * np.pi * xi) for xi in x])

# Comparison function using gradient-based optimization (L-BFGS-B)
def gradient_based_optimization(func, bounds, num_trials=10):
    """
    Optimize using scipy's L-BFGS-B algorithm with multiple random starts.
    This represents a state-of-the-art gradient-based approach.
    """
    best_result = None
    best_fitness = float('inf')
    
    print("Running gradient-based optimization (L-BFGS-B)...")
    
    for trial in range(num_trials):
        # Random starting point
        x0 = [random.uniform(bounds[i][0], bounds[i][1]) for i in range(len(bounds))]
        
        # Optimize using L-BFGS-B
        result = minimize(func, x0, method='L-BFGS-B', bounds=bounds)
        
        if result.fun < best_fitness:
            best_fitness = result.fun
            best_result = result
            
        print(f"Trial {trial+1:2d}: f = {result.fun:.6f}")
    
    return best_result.x, best_fitness

if __name__ == "__main__":
    # Problem setup: optimize 2D Rastrigin function
    dimensions = 2
    bounds = [(-5.12, 5.12) for _ in range(dimensions)]  # Standard Rastrigin bounds
    
    print("=== Genetic Algorithm vs Gradient-Based Optimization ===")
    print(f"Optimizing {dimensions}D Rastrigin function")
    print(f"Search space: {bounds}")
    print(f"Global optimum: x = [0, 0], f(x) = 0")
    print()
    
    # Run Genetic Algorithm
    print("=== GENETIC ALGORITHM ===")
    ga = GeneticAlgorithm(
        fitness_function=rastrigin_function,
        bounds=bounds,
        population_size=100,
        mutation_rate=0.1,
        crossover_rate=0.8
    )
    
    ga_solution, ga_fitness = ga.optimize(max_generations=200, verbose=True)
    
    print(f"\nGA Results:")
    print(f"Best solution: {ga_solution}")
    print(f"Best fitness: {ga_fitness:.6f}")
    print()
    
    # Run gradient-based optimization
    print("=== GRADIENT-BASED OPTIMIZATION ===")
    gb_solution, gb_fitness = gradient_based_optimization(
        rastrigin_function, bounds, num_trials=10
    )
    
    print(f"\nGradient-based Results:")
    print(f"Best solution: {gb_solution}")
    print(f"Best fitness: {gb_fitness:.6f}")
    print()
    
    # Compare results
    print("=== COMPARISON ===")
    print(f"Genetic Algorithm:     {ga_fitness:.6f}")
    print(f"Gradient-based (L-BFGS-B): {gb_fitness:.6f}")
    
    if ga_fitness < gb_fitness:
        print("🏆 Genetic Algorithm found better solution!")
    elif gb_fitness < ga_fitness:
        print("🏆 Gradient-based method found better solution!")
    else:
        print("🤝 Both methods found equivalent solutions!")
    
    # Plot evolution progress
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(ga.best_fitness_history, 'b-', label='Best Fitness', linewidth=2)
    plt.plot(ga.average_fitness_history, 'r--', label='Average Fitness', alpha=0.7)
    plt.xlabel('Generation')
    plt.ylabel('Fitness Value')
    plt.title('Genetic Algorithm Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Visualize the Rastrigin function (for 2D case)
    if dimensions == 2:
        plt.subplot(1, 2, 2)
        x = np.linspace(-5.12, 5.12, 100)
        y = np.linspace(-5.12, 5.12, 100)
        X, Y = np.meshgrid(x, y)
        Z = rastrigin_function([X, Y])
        
        plt.contour(X, Y, Z, levels=20, alpha=0.6)
        plt.scatter(*ga_solution, color='red', s=100, marker='*', 
                   label=f'GA Solution ({ga_fitness:.3f})', zorder=5)
        plt.scatter(*gb_solution, color='blue', s=100, marker='o', 
                   label=f'Gradient Solution ({gb_fitness:.3f})', zorder=5)
        plt.scatter(0, 0, color='green', s=100, marker='x', 
                   label='Global Optimum (0.000)', zorder=5)
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('Solution Comparison on Rastrigin Function')
        plt.legend()
        plt.colorbar(label='Function Value')
    
    plt.tight_layout()
    plt.savefig('ga_vs_gradient_optimization.png')