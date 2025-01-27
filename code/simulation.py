import copy

from ca_lecture_hall_model import Grid, Status
import numpy as np
from tqdm import tqdm


def run_simulation(grid, steps=1000):
    """
    Runs the simulation for a specified number of steps, updating the grid at each iteration.
    Stops iteration if no cell status changes for 3 consecutive steps.

    Parameters:
        grid (Grid): The initial grid to run the simulation on
        steps (int): The number of steps to simulate.

    Returns:
        (list): A list of grids, one grid for each time step.
    """
    assert isinstance(steps, int), f"steps must be an integer, got {type(steps)}"
    assert steps > 0, f"steps must be greater than 0, got {steps}"

    all_grids = [copy.deepcopy(grid)]

    same = 0
    prev_grid = None

    for step in range(steps):
        current_grid = copy.deepcopy(grid.lecture_hall)

        # check if grid is the same as the previous grid
        if prev_grid is not None and prev_grid == current_grid:
            same += 1
        else:
            same = 0

        # stop simulation if no cell status changed for 3 consecutive steps
        if same == 3:
            break

        # update the grid
        grid.update_grid()

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
        "SECRET_KEEPER": [],
        "GOSSIP_SPREADER": [],
    }

    for grid in grids:
        new_counts = {
            "UNOCCUPIED": 0,
            "CLUELESS": 0,
            "SECRET_KEEPER": 0,
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
                elif cell_status == Status.SECRET_KEEPER:
                    new_counts["SECRET_KEEPER"] += 1
                elif cell_status == Status.GOSSIP_SPREADER:
                    new_counts["GOSSIP_SPREADER"] += 1

        status_counts["UNOCCUPIED"].append(new_counts["UNOCCUPIED"])
        status_counts["CLUELESS"].append(new_counts["CLUELESS"])
        status_counts["SECRET_KEEPER"].append(new_counts["SECRET_KEEPER"])
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

    def dfs_size(grid, visited, i, j):
        """
        Performs a Depth-First Search (DFS) to explore a cluster of connected cells in the grid.

        This function recursively visits all connected cells, checking all four neighbors (up, down, left, and right)
        for connectivity. It will stop when all possible cells in the cluster have been visited.

        Parameters:
            i (int): The row index of the current cell to start the DFS from.
            j (int): The column index of the current cell to start the DFS from.
            target_row (int, optional): The row index we are trying to reach (for vertical percolation).
            target_col (int, optional): The column index we are trying to reach (for horizontal percolation).

        Returns:
            int: The size of the cluster of connected cells.
        """
        # Check if the current cell is within bounds, is not visited, and is occupied
        if (
            i < 0
            or i >= grid.size
            or j < 0
            or j >= grid.size
            or visited[i][j]
            or grid.lecture_hall[i][j].get_status() != Status.GOSSIP_SPREADER
        ):
            return 0  # Return 0 if the current cell is invalid or already visited

        # Mark this cell as visited
        visited[i][j] = True

        # Initialize the size of the cluster with this cell
        cluster_size = 1

        # Explore all four possible neighbors: down, up, right, left
        cluster_size += dfs_size(grid, visited, i + 1, j)  # Down
        cluster_size += dfs_size(grid, visited, i - 1, j)  # Up
        cluster_size += dfs_size(grid, visited, i, j + 1)  # Right
        cluster_size += dfs_size(grid, visited, i, j - 1)  # Left
        cluster_size += dfs_size(grid, visited, i + 1, j + 1)  # Down-Right
        cluster_size += dfs_size(grid, visited, i + 1, j - 1) # Down-Left
        cluster_size += dfs_size(grid, visited, i - 1, j + 1) # Up-Right
        cluster_size += dfs_size(grid, visited, i - 1, j - 1) # Up-Left

        return cluster_size

    cluster_sizes = []

    for grid in grids:
        size = grid.size
        visited = [[False for _ in range(size)] for _ in range(size)]

        # Iterate through the grid to find clusters
        for i in range(size):
            for j in range(size):
                if not visited[i][j] and (
                    grid.lecture_hall[i][j].get_status() == Status.GOSSIP_SPREADER
                    or grid.lecture_hall[i][j].get_status() == Status.SECRET_KEEPER
                ):
                    cluster_size = dfs_size(grid, visited, i, j)
                    if cluster_size > 0:
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
    grid_size, density, spread_threshold, steps, num_simulations, flag_center=1
):
    """
    Runs multiple simulations for a given density and spread threshold.
    """
    results = {
        "grid_size": grid_size,
        "density": density,
        "spread_threshold": spread_threshold,
        "steps": steps,
        "simulation_outcomes": [],
    }

    for _ in range(num_simulations):
        g = Grid(grid_size, density, spread_threshold)
        g.initialize_board(flag_center)
        run_simulation(g, steps)
        results["simulation_outcomes"].append(g.check_percolation())

    return results


def run_multiple_simulations_same_initial_conditions(
    num_simulations, grid_size, density, spread_threshold, steps=100, flag_center=1
):
    """
    Runs multiple simulations with the same initial conditions (same grid size, density, and spreading threshold)
    and calculates the cluster size distribution for each simulation.

    Parameters:
        num_simulations (int): The number of simulations to run.
        grid_size (int): The size of the grid (e.g., number of rows and columns).
        density (float): The fraction of cells initially occupied.
        spread_threshold (float): The threshold probability for a cell to become a gossip spreader.
        steps(int): The number of steps for each simulation.
        flag_center (int): The flag to determine the initial spreader placement.

    Returns:
        list of dict: A list of cluster size distributions (one for each simulation).
    """
    cluster_distributions = []
    count_percolation = 0

    # run multiple simulations
    for _ in range(num_simulations):
        # create and initialize the grid
        grid = Grid(grid_size, density, spread_threshold)
        grid.initialize_board(flag_center)

        # run the simulation
        run_simulation(grid, steps=steps)
        if grid.check_percolation():
            count_percolation += 1
        # calculate the cluster size distribution for the current grid
        cluster_distribution = calculate_cluster_size_distribution([grid])
        cluster_distributions.append(cluster_distribution)

    print(
        f"Percolation occured in {count_percolation} out of {num_simulations} simulations for density={density}, spread_threshold={spread_threshold}"
    )
    # return the list of cluster distributions from all simulations
    return cluster_distributions


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


def simulate_density(grid_size, density, spread_threshold, steps, num_simulations):
    """
    Simulates a single density and spread threshold and returns the fraction of simulations with percolation.
    """
    results = run_multiple_simulations_for_percolation(
        grid_size, density, spread_threshold, steps, num_simulations
    )
    return sum(results["simulation_outcomes"]) / num_simulations


def simulate_density_vs_threshold(grid_size, density, steps, num_simulations):
    """
    Simulates different spread thresholds for a fixed density and returns percolation probabilities.
    """
    thresholds = np.linspace(0, 1, 20)
    percolations = []

    for threshold in tqdm(thresholds, desc="Simulating thresholds"):
        percolations.append(
            simulate_density(grid_size, density, threshold, steps, num_simulations)
        )

    return thresholds, percolations


def simulate_and_collect_percolations(
    grid_size, densities, spread_threshold, steps, num_simulations
):
    """
    Simulates percolation probabilities across a range of densities for a fixed spread threshold.
    """
    percolations = []
    for d in densities:
        percolations.append(
            simulate_density(grid_size, d, spread_threshold, steps, num_simulations)
        )
    return percolations


def run_multiple_simulations_for_phase_diagram(
    grid_size, density, spread_threshold, steps, num_simulations, flag_center=1
):
    """
    Runs multiple simulations for a given density and spread threshold and for each simulation returns
    the number of gossip spreaders at the end of the simulation.

    Parameters:
        grid_size (int): The size of the grid for the simulations.
        density (float): A density value to use for the simulations.
        spread_threshold (float): A spread threshold value to use for the simulations.
        steps (int, optional): The max number of time steps for each simulation.
        num_simulations (int, optional): The number of simulations to run for each density.

    Returns:
        dict: A dictionary with the following structure:
            grid_size (int): The size of the grid.
            density (float): The initial density of occupied cells in the grid.
            spread_threshold (float): The threshold probability for a cell to spread gossip.
            steps (int): The number of time steps (iterations) for each simulation.
            simulation_outcomes (list): A list containing the total number of gossip spreaders at the end of each simulation.
    """
    results = {
        "grid_size": grid_size,
        "density": density,
        "spread_threshold": spread_threshold,
        "steps": steps,
        "simulation_outcomes": [],
    }

    for i in range(num_simulations):
        g = Grid(grid_size, density, spread_threshold)
    for i in range(num_simulations):
        g = Grid(grid_size, density, spread_threshold)
        g.initialize_board(flag_center)

        grids = run_simulation(g, steps)

        status_counts = count_statuses(grids)

        # only the final number of GOSSIP_SPREADERS from status_counts for phase diagram
        gossip_spreaders = status_counts["GOSSIP_SPREADER"][-1]
        results["simulation_outcomes"].append(gossip_spreaders)

    return results


def create_results_dict(grid_size, density, spread_threshold, steps):
    """
    Creates and returns a dictionary.

    Parameters:
        size (int): The size of the grid.
        grid_size (float): The initial density of occupied cells in the grid.
        spread_threshold (float): The threshold probability for a cell to spread gossip.
        steps (int): The number of time steps (iterations) for each simulation.

    Returns:
        dict: A dictionary with the following structure:
            grid_size (int): The size of the grid.
            density (float): The initial density of occupied cells in the grid.
            spread_threshold (float): The threshold probability for a cell to spread gossip.
            steps (int): The number of time steps (iterations) for each simulation.
            simulation_outcomes (list): An empty list that will hold simulation outcomes over time.
    """
    return {
        "grid_size": grid_size,
        "density": density,
        "spread_threshold": spread_threshold,
        "steps": steps,
        "simulation_outcomes": [],
    }


def run_multiple_simulations_for_timeplot_status(
    grid_size, density, spread_threshold, steps, num_simulations, flag_center=1
    grid_size, density, spread_threshold, steps, num_simulations, flag_center=1
):
    """
    Runs multiple simulations of the grid model and records the number of cells in each status over time.

    Parameters:
        grid_size (int): The size of the grid.
        density (float): The initial density of occupied cells in the grid.
        spread_threshold (float): The threshold probability for a cell to spread gossip.
        steps (int): The number of time steps (iterations) for each simulation.
        num_simulations (int): The number of simulations to run.

    Returns:
        tuple: A tuple containing four dictionaries, each representing the outcomes for a specific status:
            results_gossip (dictionary)
            results_secret (dictionary)
            results_clueless (dictionary)
            results_unoccupied (dictionary)
    """

    results_gossip = create_results_dict(grid_size, density, spread_threshold, steps)
    results_secret = create_results_dict(grid_size, density, spread_threshold, steps)
    results_clueless = create_results_dict(grid_size, density, spread_threshold, steps)
    results_unoccupied = create_results_dict(grid_size, density, spread_threshold, steps)

    for i in range(num_simulations):
        g = Grid(grid_size, density, spread_threshold)
    for i in range(num_simulations):
        g = Grid(grid_size, density, spread_threshold)
        g.initialize_board(flag_center)

        grids = run_simulation(g, steps)

        status_counts = count_statuses(grids)

        gossip_spreaders = status_counts["GOSSIP_SPREADER"]
        results_gossip["simulation_outcomes"].append(gossip_spreaders)
        secret_keepers = status_counts["SECRET_KEEPER"]
        results_secret["simulation_outcomes"].append(secret_keepers)
        clueless = status_counts["CLUELESS"]
        results_clueless["simulation_outcomes"].append(clueless)
        unoccupied = status_counts["UNOCCUPIED"]
        results_unoccupied["simulation_outcomes"].append(unoccupied)

    return results_gossip, results_secret, results_clueless, results_unoccupied
