import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import matplotlib as mpl
mpl.rcParams['legend.fontsize'] = 15

# Path to the CSV files
csv_path = r"C:\Users\cavus\IdeaProjects\Energies_ADIB\venv"

# Function to generate a time series for load
def TimeSeries():
    try:
        series = np.loadtxt(f"{csv_path}\\TimeSeries.csv", delimiter=",")
    except IOError:
        load_range = pd.read_csv(f"{csv_path}\\MaxMin.csv").values
        x, y = 100, 24
        inc = 100
        ts = np.zeros((x, y))
        for i in range(x):
            for j in range(y):
                ts[i, j] = np.random.randint(load_range[0, j], load_range[1, j])

        series = np.zeros((x // inc, y))
        for i in range(0, x, inc):
            get = ts[i:i+inc, :]
            series[i // inc, :] = np.round(np.mean(get, axis=0))

        np.savetxt(f"{csv_path}\\TimeSeries.csv", series, delimiter=",")
    return series

# Function to store the best chromosome
def store_best(vector, fitness, chromosomes, time_idx):
    best_i = np.nanargmin(fitness)  # Use np.nanargmin to handle NaN values
    vector[time_idx, :] = chromosomes[best_i, :]
    return vector

# Function to create combined plots
def combined_plots(econ_mpc, econ_no_mpc, enviro_mpc, enviro_no_mpc, excess_mpc, excess_no_mpc, fitness_mpc, fitness_no_mpc):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # Economic Impact Comparison
    axs[0, 0].stairs(econ_mpc, label='GPC', linewidth=2.0)
    axs[0, 0].stairs(econ_no_mpc, label='GA', linewidth=2.0)
    axs[0, 0].set_xlabel('Time', fontsize=15)
    axs[0, 0].set_ylabel('Cost ($) Per Watt', fontsize=15)
    axs[0, 0].legend(fontsize=15)
    axs[0, 0].grid(True)
    axs[0, 0].set_title('(a)', fontsize=18)

    # Environmental Impact Comparison
    axs[0, 1].stairs(enviro_mpc, label='GPC', linewidth=2.0)
    axs[0, 1].stairs(enviro_no_mpc, label='GA', linewidth=2.0)
    axs[0, 1].set_xlabel('Time', fontsize=15)
    axs[0, 1].set_ylabel('Emissions (tons)', fontsize=15)
    axs[0, 1].legend(fontsize=15)
    axs[0, 1].grid(True)
    axs[0, 1].set_title('(b)', fontsize=18)

    # Excess Power Comparison
    axs[1, 0].stairs(excess_mpc, label='GPC', linewidth=2.0)
    axs[1, 0].stairs(excess_no_mpc, label='GA', linewidth=2.0)
    axs[1, 0].set_xlabel('Time', fontsize=15)
    axs[1, 0].set_ylabel('Excess Power (W)', fontsize=15)
    axs[1, 0].legend(fontsize=15)
    axs[1, 0].grid(True)
    axs[1, 0].set_title('(c)', fontsize=18)

    # Fitness Evolution Comparison
    axs[1, 1].stairs(fitness_mpc, label='GPC', linewidth=2.0)
    axs[1, 1].stairs(fitness_no_mpc, label='GA', linewidth=2.0)
    axs[1, 1].set_xlabel('Iteration', fontsize=15)
    axs[1, 1].set_ylabel('Fitness', fontsize=15)
    axs[1, 1].legend(fontsize=15)
    axs[1, 1].grid(True)
    axs[1, 1].set_title('(d)', fontsize=18)

    plt.tight_layout()
    plt.savefig("combined_comparison.png", dpi=1200)
    plt.show()

# Function to find fitness
def find_fitness(min_val, max_val, gene):
    m1 = gene - min_val
    if m1 < 0:
        return 0
    m2 = max_val - min_val
    return m1 / m2

# Function to generate chromosomes
def chromosome_gen(pop, sources):
    num_sources, num_genes = sources.shape
    chromosomes = np.zeros((pop, num_sources))
    for i in range(pop):
        for j in range(num_sources):
            chromosomes[i, j] = np.random.randint(0, sources[j, 0])
    return chromosomes

# Function to calculate economic fitness
def econ_fitness(chromosome, sources, boundaries, weights):
    cost_idx = 1
    fit = np.sum(chromosome * sources[:, cost_idx])
    return weights[cost_idx] * find_fitness(boundaries[cost_idx, 0], boundaries[cost_idx, 1], fit)

# Function to calculate environmental fitness
def enviro_fitness(chromosome, sources, boundaries, weights):
    emission_idx = 2
    fit = np.sum(chromosome * sources[:, emission_idx])
    return weights[emission_idx] * find_fitness(boundaries[emission_idx, 0], boundaries[emission_idx, 1], fit)

# Function to check if chromosome is within bounds
def check_bounds(chromosome, sources, bounds):
    power = sum(chromosome)
    cost = np.sum(chromosome * sources[:, 1])
    emission = np.sum(chromosome * sources[:, 2])

    if power < bounds[0, 0] or power > bounds[0, 1]:
        return False
    if cost < bounds[1, 0] or cost > bounds[1, 1]:
        return False
    if emission < bounds[2, 0] or emission > bounds[2, 1]:
        return False
    return True

# Function to perform elitism
def elitism(pop, chromosomes, fitness):
    sorted_indices = np.argsort(fitness)
    return chromosomes[sorted_indices[:pop], :]

# Function to perform crossover
def crossover(chromosomes, pop, sources, boundary, rate, cross_point):
    num_children = int(pop * rate)
    num_genes = chromosomes.shape[1]
    children = np.zeros((num_children, num_genes))

    for i in range(0, num_children, 2):
        parent1, parent2 = chromosomes[np.random.choice(len(chromosomes), 2, replace=False)]
        child1 = np.concatenate((parent1[:cross_point], parent2[cross_point:]))
        child2 = np.concatenate((parent2[:cross_point], parent1[:cross_point:]))
        children[i], children[i+1] = child1, child2

    valid_children = [child for child in children if check_bounds(child, sources, boundary)]
    if len(valid_children) == 0:
        return chromosomes
    return np.vstack((chromosomes, valid_children))

# Function to perform mutation
def mutate(chromosome, sources, boundary):
    new_chromosome = np.array([np.random.randint(0, source[0]) for source in sources])
    return new_chromosome if check_bounds(new_chromosome, sources, boundary) else chromosome

# Function to perform mass mutation
def mass_mutate(chromosomes, pop, sources, boundaries, m_rate):
    mutation_indices = np.random.choice(len(chromosomes), int(len(chromosomes) * m_rate), replace=False)
    for idx in mutation_indices:
        chromosomes[idx, :] = mutate(chromosomes[idx, :], sources, boundaries)
    return chromosomes

# Function to randomly select chromosomes
def random_selection(pop, chromosomes):
    return chromosomes[np.random.choice(len(chromosomes), pop, replace=False)]

# Function to determine overall fitness
def fitness(chromosomes, sources, boundaries, weights, goal_power):
    fitness_vector = np.zeros(len(chromosomes))
    for i, chromosome in enumerate(chromosomes):
        power_fitness = sum(chromosome) * weights[0]
        if power_fitness < goal_power:
            power_fitness = np.nan

        econ_fit = econ_fitness(chromosome, sources, boundaries, weights)
        enviro_fit = enviro_fitness(chromosome, sources, boundaries, weights)

        fitness_vector[i] = (((econ_fit + enviro_fit) / 2) * (power_fitness / goal_power)) / goal_power * 10
    return fitness_vector

# Function to check stop condition
def stop_condition(fit, stop):
    return np.nanmin(fit) < stop  # Use np.nanmin to handle NaN values

# Function to get and print data
def get_data(loads, chromosomes, sources):
    excess, cost, emission = 0, 0, 0
    for i in range(len(chromosomes)):
        excess += sum(chromosomes[i, :]) - loads[i]
        cost += np.dot(chromosomes[i, :], sources[:, 1])
        emission += np.dot(chromosomes[i, :], sources[:, 2])
    return excess, cost, emission

# Function to perform MPC with Genetic Algorithm
def mpc_ga(pop, sources, boundaries, weights, load, horizon, m_rate, c_pt, good_enough):
    vector = np.zeros((len(load), sources.shape[0]))
    econ = np.zeros(len(load))
    enviro = np.zeros(len(load))
    time_elapsed = np.zeros(len(load))
    excess_power = np.zeros(len(load))

    for t in range(0, len(load), horizon):
        start_time = time.time()

        future_load = load[t:t+horizon] if t + horizon <= len(load) else load[t:]
        chromosomes = chromosome_gen(pop, sources)
        fit = fitness(chromosomes, sources, boundaries, weights, future_load[0])

        while not stop_condition(fit, good_enough):
            chromosomes = reproduction(chromosomes, pop, sources, boundaries, m_rate, c_pt)
            fit = fitness(chromosomes, sources, boundaries, weights, future_load[0])
            chromosomes = survival(pop, chromosomes, fit)

        for i in range(len(future_load)):
            best_i = np.nanargmin(fit)
            vector[t + i, :] = chromosomes[best_i, :]
            econ[t + i] = econ_fitness(chromosomes[best_i], sources, boundaries, weights)
            enviro[t + i] = enviro_fitness(chromosomes[best_i], sources, boundaries, weights)
            excess_power[t + i] = sum(chromosomes[best_i, :]) - future_load[i]
            if i < len(future_load) - 1:
                fit = fitness(chromosomes, sources, boundaries, weights, future_load[i + 1])

        time_elapsed[t:t+len(future_load)] = time.time() - start_time

    return vector, econ, enviro, time_elapsed, excess_power

# Function to perform Genetic Algorithm without MPC
def ga_no_mpc(pop, sources, boundaries, weights, load, m_rate, c_pt, good_enough):
    vector = np.zeros((len(load), sources.shape[0]))
    econ = np.zeros(len(load))
    enviro = np.zeros(len(load))
    time_elapsed = np.zeros(len(load))
    excess_power = np.zeros(len(load))

    chromosomes = chromosome_gen(pop, sources)

    for t in range(len(load)):
        start_time = time.time()

        fit = fitness(chromosomes, sources, boundaries, weights, load[t])
        while not stop_condition(fit, good_enough):
            chromosomes = reproduction(chromosomes, pop, sources, boundaries, m_rate, c_pt)
            fit = fitness(chromosomes, sources, boundaries, weights, load[t])
            chromosomes = survival(pop, chromosomes, fit)

        best_i = np.nanargmin(fit)
        vector[t, :] = chromosomes[best_i, :]
        econ[t] = econ_fitness(chromosomes[best_i], sources, boundaries, weights)
        enviro[t] = enviro_fitness(chromosomes[best_i], sources, boundaries, weights)
        excess_power[t] = sum(chromosomes[best_i, :]) - load[t]
        time_elapsed[t] = time.time() - start_time

    return vector, econ, enviro, time_elapsed, excess_power

# Main Script
def main():
    pop = 100            # population
    m_rate = .1          # mutation rate
    c_pt = 2             # crossover point
    good_enough = .08    # when to stop
    horizon = 5          # MPC horizon
    weights = [1, .6, .4]

    sources = np.array([
        [20, .02, .01],
        [120, .20, .05],
        [15, .01, .02],
        [50, .02, .04]
    ])

    boundaries = np.array([
        [15, 5000],        # requested power range
        [.15, 6.55],      # minimum cost to get max power
        [.15, 3.75]       # minimum emissions to get max power
    ])

    # User inputs
    reproduction_method = input("Choose a Reproduction Method (1 - Mutation, 2 - Crossover): ")
    if reproduction_method == '1':
        def reproduction(chromosomes, pop, sources, boundaries, m_rate, cross_point=None):
            return mass_mutate(chromosomes, pop, sources, boundaries, m_rate)
    else:
        def reproduction(chromosomes, pop, sources, boundaries, m_rate, cross_point):
            return crossover(chromosomes, pop, sources, boundaries, m_rate, cross_point)

    survival_method = input("Choose a Survival Method (1 - Random Selection, 2 - Elitism): ")
    survival = random_selection if survival_method == '1' else elitism

    # Generate the load
    load = TimeSeries()

    # Perform MPC with Genetic Algorithm
    vector_mpc, econ_mpc, enviro_mpc, time_elapsed_mpc, excess_mpc = mpc_ga(pop, sources, boundaries, weights, load, horizon, m_rate, c_pt, good_enough)

    # Perform Genetic Algorithm without MPC
    vector_no_mpc, econ_no_mpc, enviro_no_mpc, time_elapsed_no_mpc, excess_no_mpc = ga_no_mpc(pop, sources, boundaries, weights, load, m_rate, c_pt, good_enough)

    # Plot combined comparisons
    combined_plots(econ_mpc, econ_no_mpc, enviro_mpc, enviro_no_mpc, excess_mpc, excess_no_mpc, time_elapsed_mpc, time_elapsed_no_mpc)

    # Output data
    print("Proposed method Results:")
    excess_mpc_total, cost_mpc_total, emission_mpc_total = get_data(load, vector_mpc, sources)
    print(f"Total excess power produced: {excess_mpc_total}W")
    print(f"Total cost for the day: ${cost_mpc_total:.2f}")
    print(f"Total emissions for the day: {emission_mpc_total:.2f} tons")

    print("\nGA Results:")
    excess_no_mpc_total, cost_no_mpc_total, emission_no_mpc_total = get_data(load, vector_no_mpc, sources)
    print(f"Total excess power produced: {excess_no_mpc_total}W")
    print(f"Total cost for the day: ${cost_no_mpc_total:.2f}")
    print(f"Total emissions for the day: {emission_no_mpc_total:.2f} tons")

if __name__ == "__main__":
    main()
