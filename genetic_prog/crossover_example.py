import random
from pgtree import PGTree
from prog_gen import PGIndividual

def main():
    # Seed for reproducibility
    random.seed(244)

    # Generate two random parent trees
    parent1 = PGTree()
    parent2 = PGTree()

    # Create individuals from the parent trees
    individual1 = PGIndividual(parent1)
    individual2 = PGIndividual(parent2)

    # Print and save parent1
    print("Parent 1 Tree Structure:")
    parent1.pretty_print()
    parent1.to_dot("parent1.dot")

    # Print and save parent2
    print("\nParent 2 Tree Structure:")
    parent2.pretty_print()
    parent2.to_dot("parent2.dot")

    # Perform crossover between two individuals
    child = individual1.crossover(individual2)

    # Print and save child
    print("\nChild Tree Structure:")
    child.genome.pretty_print()
    child.genome.to_dot("child.dot")

    print("DOT files for Parent1, Parent2, and Child have been saved.")

if __name__ == "__main__":
    main()