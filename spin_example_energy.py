import itertools

def ising_energy(spins, J=1.0):
    """
    Calculate Ising energy for 4-spin square lattice
    spins: tuple/list of 4 spins (s1, s2, s3, s4)
    J: coupling constant
    """
    s1, s2, s3, s4 = spins
    
    # nearest neighbor interactions (square)
    energy = -J * (
        s1*s2 +  # top
        s2*s4 +  # right
        s4*s3 +  # bottom
        s3*s1    # left
    )
    
    return energy


# Generate configurations satisfying diagonal constraint
# s1 = s4 and s2 = s3
possible_spins = [-1, 1]

configs = []
for s1, s2 in itertools.product(possible_spins, repeat=2):
    s3 = s2  # diagonal constraint
    s4 = s1  # diagonal constraint
    configs.append((s1, s2, s3, s4))


print("Allowed configurations and energies:\n")

for config in configs:
    energy = ising_energy(config, J=1)
    print(f"Spins: {config}, Energy: {energy}")