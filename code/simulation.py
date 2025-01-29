import copy

from ca_lecture_hall_model import Grid, Status
import numpy as np
from tqdm import tqdm


def run_simulation(grid, steps=1000, flag_neighbors=0):
    """
    Runs the simulation for a specified number of steps, updating the grid at each iteration.
    Stops simulation if no cell status changes anymore.

    Parameters:
        grid (Grid): The initial grid to run the simulation on
        steps (int): The number of steps to simulate.
        flag_neighborhood (int): The flag to determine the neighborhood type.If 1, the Moore neighborhood is used. If 0, the Von Neumann neighborhood is used.

    Returns:
        (list): A list of grids, one grid for each time step.
    """
    assert isinstance(steps, int), f"steps must be an integer, got {type(steps)}"
    assert steps > 0, f"steps must be greater than 0, got {steps}"

    all_grids = [copy.deepcopy(grid)]

    prev_grid = copy.deepcopy(grid.lecture_hall)

    for step in range(steps):
        # update the grid
        grid.update_grid(flag_neighbors)

        current_grid = copy.deepcopy(grid.lecture_hall)

        # check if any changes happened or not
        if prev_grid == current_grid:
            break

        prev_grid = current_grid
        all_grids.append(copy.deepcopy(grid))

    return all_grids


def count_statuses(grids):
    """
    Iterates through all the grids and for each grid calculates the number of cells of each status.
    Iterates through all the grids and for each grid calculates the number of cells of each status.

    Parameters:
        grids (list of Grid): List of grids.
        grids (list of Grid): List of grids.

    Returns:
        (dict of list): A dictionary containing lists with status counts, one list for each status and one entry for each grid.
        (dict of list): A dictionary containing lists with status counts, one list for each status and one entry for each grid.
    """
    status_counts = {
        "UNOCCUPIED": [],
        "CLUELESS": [],
        "GOSSIP_SPREADER": [],
    }

    for grid in grids:
        new_counts = {
            "UNOCCUPIED": 0,
            "CLUELESS": 0,
            "GOSSIP_SPREADER": 0,
        }

        # count statuses
        for i in range(grid.size):
            for j in range(grid.size):
                cell_status = grid.lecture_hall[i][j].get_status()
                if cell_status == Status.UNOCCUPIED:
                    new_counts["UNOCCUPIED"] += 1
                elif cell_status == Status.CLUELESS:
                    new_counts["CLUELESS"] += 1
                elif cell_status == Status.GOSSIP_SPREADER:
                    new_counts["GOSSIP_SPREADER"] += 1

        status_counts["UNOCCUPIED"].append(new_counts["UNOCCUPIED"])
        status_counts["CLUELESS"].append(new_counts["CLUELESS"])
        status_counts["GOSSIP_SPREADER"].append(new_counts["GOSSIP_SPREADER"])

    return status_counts


def calculate_cluster_size_distribution(grids):
    """
    Calculates the size distribution of connected clusters of cells with the specified status
    across multiple simulation grids (Grid objects).

    Parameters:
        grids (list of Grid): A list of Grid objects from different simulations.
        target_status (Status): The status of the cells to include in the cluster size calculation.

    Returns:
        dict: A dictionary where keys are cluster sizes and values are the total number of clusters
              of that size across all grids.
    """

    cluster_sizes = []

    for grid in grids:
        size = grid.size
        cluster_size = 0

        for i in range(size):
            for j in range(size):
                if grid.lecture_hall[i][j].status == Status.GOSSIP_SPREADER:
                    cluster_size += 1

        cluster_sizes.append(cluster_size)

    # Calculate the size distribution
    cluster_distribution = {}
    for size in cluster_sizes:
        if size in cluster_distribution:
            cluster_distribution[size] += 1
        else:
            cluster_distribution[size] = 1

    return cluster_distribution


def run_multiple_simulations_for_percolation(
    grid_size,
    density,
    bond_probability,
    steps,
    num_simulations,
    flag_center=1,
    flag_neighbors=0,
):
    """
    Runs multiple simulations for a given density and spread threshold.
    """
    results = {
        "grid_size": grid_size,
        "density": density,
        "bond_probability": bond_probability,
        "steps": steps,
        "simulation_outcomes": [],
    }

    for _ in range(num_simulations):
        g = Grid(grid_size, density, bond_probability)
        g.initialize_board(flag_center, flag_neighbors)
        run_simulation(g, steps, flag_neighbors)
        results["simulation_outcomes"].append(g.check_percolation())

    return results


def run_multiple_simulations_same_initial_conditions(
    num_simulations, grid_size, density, bond_probability, steps=100, flag_center=1
):
    """
    Runs multiple simulations with the same initial conditions (same grid size, density, and spreading threshold),
    calculates the cluster size distribution for each simulation and records the number of cells in each status over time.

    Parameters:
        num_simulations (int): The number of simulations to run.
        grid_size (int): The size of the grid (e.g., number of rows and columns).
        density (float): The fraction of cells initially occupied.
        bond_probability (float): The threshold probability for a cell to become a gossip spreader.
        steps(int): The number of steps for each simulation.
        flag_center (int): The flag to determine the initial spreader placement.

    Returns:
        tuple: A tuple containing four dictionaries, each representing the outcomes for a specific status:
            results_gossip (dictionary)
            results_clueless (dictionary)
            results_unoccupied (dictionary)
            list of dict: A list of cluster size distributions (one for each simulation).
    """
    cluster_distributions = []
    count_percolation = 0

    results_gossip = create_results_dict(grid_size, density, bond_probability, steps)
    results_clueless = create_results_dict(grid_size, density, bond_probability, steps)
    results_unoccupied = create_results_dict(
        grid_size, density, bond_probability, steps
    )

    # run multiple simulations
    for _ in range(num_simulations):
        # create and initialize the grid
        grid = Grid(grid_size, density, bond_probability)
        grid.initialize_board(flag_center)

        # run the simulation
        grids = run_simulation(grid, steps=steps)

        status_counts = count_statuses(grids)
        gossip_spreaders = status_counts["GOSSIP_SPREADER"]
        results_gossip["simulation_outcomes"].append(gossip_spreaders)
        clueless = status_counts["CLUELESS"]
        results_clueless["simulation_outcomes"].append(clueless)
        unoccupied = status_counts["UNOCCUPIED"]
        results_unoccupied["simulation_outcomes"].append(unoccupied)

        if grid.check_percolation():
            count_percolation += 1
        # calculate the cluster size distribution for the current grid
        cluster_distribution = calculate_cluster_size_distribution([grid])

        cluster_distributions.append(cluster_distribution)

    print(
        f"Percolation occured in {count_percolation} out of {num_simulations} simulations for density={density}, bond_probability={bond_probability}"
    )
    # return the list of cluster distributions from all simulations
    return cluster_distributions, results_gossip, results_clueless, results_unoccupied


def aggregate_cluster_distributions(cluster_distributions):
    """
    Aggregates multiple cluster size distributions into a single distribution.

    Parameters:
        cluster_distributions (list of dict): A list of list of dictionaries where each dictionary
                                              represents a cluster size distribution.
                                              Keys are cluster sizes (int), and values are
                                              their corresponding frequencies (int).

    Returns:
        dict: A single aggregated cluster size distribution. Keys are cluster sizes (int),
              and values are the total frequencies (int) across all input distributions.
    """
    aggregated_distribution = {}
    # turn the list of list of dictionaries into a flat list of dictionaries
    flat_distributions = [
        distribution
        for sublist in cluster_distributions
        if isinstance(sublist, list)
        for distribution in sublist
    ]
    for distribution in flat_distributions:
        for size, count in distribution.items():
            if size in aggregated_distribution:
                aggregated_distribution[size] += count
            else:
                aggregated_distribution[size] = count
    return aggregated_distribution


def simulate_density(
    grid_size,
    density,
    bond_probability,
    steps,
    num_simulations,
    flag_center=1,
    flag_neighbors=0,
):
    """
    Simulates a single density and spread threshold and returns the fraction of simulations with percolation.
    """
    results = run_multiple_simulations_for_percolation(
        grid_size,
        density,
        bond_probability,
        steps,
        num_simulations,
        flag_center,
        flag_neighbors,
    )
    return sum(results["simulation_outcomes"]) / num_simulations


def simulate_density_vs_bond_probability(
    grid_size, density, steps, num_simulations, flag_center=1
):
    """
    Simulates different bond probabilities for a fixed density and returns percolation probabilities.
    """
    bond_probabilities = np.linspace(0, 1, 20)
    percolations = []

    for probability in tqdm(bond_probabilities, desc="Simulating bond probabilities"):
        percolations.append(
            simulate_density(
                grid_size, density, probability, steps, num_simulations, flag_center
            )
        )

    return bond_probabilities, percolations


def simulate_and_collect_percolations(
    grid_size,
    densities,
    bond_probability,
    steps,
    num_simulations,
    flag_center=1,
    flag_neighbors=0,
):
    """
    Simulates percolation probabilities across a range of densities for a fixed spread threshold.
    """
    percolations = []
    for d in densities:
        if 0.4 <= d <= 0.7:
            percolations.append(
                simulate_density(
                    grid_size,
                    d,
                    bond_probability,
                    steps,
                    num_simulations * 2,
                    flag_center,
                    flag_neighbors,
                )
            )
        else:
            percolations.append(
                simulate_density(
                    grid_size,
                    d,
                    bond_probability,
                    steps,
                    num_simulations,
                    flag_center,
                    flag_neighbors,
                )
            )

    return percolations


def run_multiple_simulations_for_phase_diagram(
    grid_size,
    density,
    bond_probability,
    steps,
    num_simulations,
    flag_center=1,
    flag_neighbors=0,
):
    """
    Runs multiple simulations for a given density and spread threshold and for each simulation returns
    the number of gossip spreaders at the end of the simulation.

    Parameters:
        grid_size (int): The size of the grid for the simulations.
        density (float): A density value to use for the simulations.
        bond_probability (float): A spread threshold value to use for the simulations.
        steps (int, optional): The max number of time steps for each simulation.
        num_simulations (int, optional): The number of simulations to run for each density.

    Returns:
        dict: A dictionary with the following structure:
            grid_size (int): The size of the grid.
            density (float): The initial density of occupied cells in the grid.
            bond_probability (float): The threshold probability for a cell to spread gossip.
            steps (int): The number of time steps (iterations) for each simulation.
            simulation_outcomes (list): A list containing the total number of gossip spreaders at the end of each simulation.
    """
    results = {
        "grid_size": grid_size,
        "density": density,
        "bond_probability": bond_probability,
        "steps": steps,
        "simulation_outcomes": [],
    }

    for i in range(num_simulations):
        g = Grid(grid_size, density, bond_probability)
        g.initialize_board(flag_center, flag_neighbors)

        grids = run_simulation(g, steps, flag_neighbors)

        status_counts = count_statuses(grids)

        # only the final number of GOSSIP_SPREADERS from status_counts for phase diagram
        gossip_spreaders = status_counts["GOSSIP_SPREADER"][-1]
        results["simulation_outcomes"].append(gossip_spreaders)

    return results


def create_results_dict(grid_size, density, bond_probability, steps):
    """
    Creates and returns a dictionary.

    Parameters:
        size (int): The size of the grid.
        grid_size (float): The initial density of occupied cells in the grid.
        bond_probability (float): The threshold probability for a cell to spread gossip.
        steps (int): The number of time steps (iterations) for each simulation.

    Returns:
        dict: A dictionary with the following structure:
            grid_size (int): The size of the grid.
            density (float): The initial density of occupied cells in the grid.
            bond_probability (float): The threshold probability for a cell to spread gossip.
            steps (int): The number of time steps (iterations) for each simulation.
            simulation_outcomes (list): An empty list that will hold simulation outcomes over time.
    """
    return {
        "grid_size": grid_size,
        "density": density,
        "bond_probability": bond_probability,
        "steps": steps,
        "simulation_outcomes": [],
    }


def run_multiple_simulations_for_timeplot_status(
    grid_size,
    density,
    bond_probability,
    steps,
    num_simulations,
    flag_center=1,
    flag_neighbors=0,
):
    """
    Runs multiple simulations of the grid model and records the number of cells in each status over time.

    Parameters:
        grid_size (int): The size of the grid.
        density (float): The initial density of occupied cells in the grid.
        bond_probability (float): The threshold probability for a cell to spread gossip.
        steps (int): The number of time steps (iterations) for each simulation.
        num_simulations (int): The number of simulations to run.
        flag_center (int): The flag to determine the initial spreader placement.
        flag_neighbors (int): The flag to determine the neighborhood type. If 1, the Moore neighborhood is used. If 0, the Von Neumann neighborhood is used.

    Returns:
        tuple: A tuple containing four dictionaries, each representing the outcomes for a specific status:
            results_gossip (dictionary)
            results_clueless (dictionary)
            results_unoccupied (dictionary)
    """

    results_gossip = create_results_dict(grid_size, density, bond_probability, steps)
    results_clueless = create_results_dict(grid_size, density, bond_probability, steps)
    results_unoccupied = create_results_dict(
        grid_size, density, bond_probability, steps
    )

    for i in range(num_simulations):
        g = Grid(grid_size, density, bond_probability)
        g.initialize_board(flag_center, flag_neighbors)

        grids = run_simulation(g, steps, flag_neighbors)

        status_counts = count_statuses(grids)
        actual_steps = len(status_counts["GOSSIP_SPREADER"])

        gossip_spreaders = status_counts["GOSSIP_SPREADER"]
        clueless = status_counts["CLUELESS"]
        unoccupied = status_counts["UNOCCUPIED"]

        if actual_steps < steps:
            last_gossip = gossip_spreaders[-1]
            last_clueless = clueless[-1]
            last_unoccupied = unoccupied[-1]

            gossip_spreaders.extend([last_gossip] * (steps - actual_steps))
            clueless.extend([last_clueless] * (steps - actual_steps))
            unoccupied.extend([last_unoccupied] * (steps - actual_steps))

        results_gossip["simulation_outcomes"].append(gossip_spreaders)
        results_clueless["simulation_outcomes"].append(clueless)
        results_unoccupied["simulation_outcomes"].append(unoccupied)

    return results_gossip, results_clueless, results_unoccupied
