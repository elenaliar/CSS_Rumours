import copy

from ca_lecture_hall_model import Grid, Status


def run_simulation(grid, steps=1000, flag_neighbors=0):
    """
    Runs the simulation for a specified number of steps, updating the grid at each iteration.
    Stops simulation if no cell status changes anymore.

    Parameters:
        grid (Grid): The initial grid to run the simulation on
        steps (int): The number of steps to simulate.
        flag_neighbors (int): The flag to determine the neighborhood type (1 for Moore, 0 for Von Neumann).

    Returns:
        (list): A list of grids, one grid for each time step.
    """
    assert isinstance(steps, int), f"steps must be an integer, got {type(steps)}"
    assert steps > 0, f"steps must be greater than 0, got {steps}"
    assert flag_neighbors in [
        0,
        1,
    ], f"flag_neighbors must be either 0 or 1, got {flag_neighbors}"

    all_grids = [copy.deepcopy(grid)]

    prev_grid = copy.deepcopy(grid.lecture_hall)

    for _ in range(steps):
        # Update the grid
        grid.update_grid(flag_neighbors)

        current_grid = copy.deepcopy(grid.lecture_hall)

        # Check if any changes happened or not
        if prev_grid == current_grid:
            break

        prev_grid = current_grid
        all_grids.append(copy.deepcopy(grid))

    return all_grids


def count_statuses(grids):
    """
    Iterates through all the grids and for each grid calculates the number of cells of each status.

    Parameters:
        grids (list of Grid): List of grids.

    Returns:
        (dict of list): A dictionary containing lists with status counts, one list for each status and one entry for each grid.
    """
    assert isinstance(grids, list), f"grids must be a list, got {type(grids)}"
    assert isinstance(
        grids[0], Grid
    ), f"Each element in grids must be a Grid, got {type(grids[0])}"

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

        # Count statuses
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
    Runs multiple simulations for a given density and bond probability and determines whether percolation occurs in each run or not.

    Parameters:
        grid_size (int): The size of the grid for the simulations.
        density (float): The initial density of occupied cells in the grid.
        bond_probability (float): The probability that a bond is open between neighboring cells.
        steps (int): The number of time steps (iterations) for each simulation.
        num_simulations (int): The number of simulations to run.
        flag_center (int, optional): Determines the initial spreader placement (1 for center, 0 for random).
        flag_neighbors (int, optional): Determines the neighborhood type (1 for Moore, 0 for Von Neumann).

    Returns:
        dict: A dictionary containing simulation results with the following keys:
            - "grid_size" (int): The grid size.
            - "density" (float): The initial density of occupied cells.
            - "bond_probability" (float): The probability of a bond being open.
            - "steps" (int): The number of time steps.
            - "simulation_outcomes" (list): A list of booleans indicating percolation results for each simulation.
    """
    assert isinstance(
        grid_size, int
    ), f"grid_size must be an integer, got {type(grid_size)}"
    assert grid_size > 0, f"grid_size must be greater than 0, got {grid_size}"
    assert isinstance(
        density, (int, float)
    ), f"density must be a number, got {type(density)}"
    assert (
        0 <= density <= 1
    ), f"density must be between 0 and 1 (inclusive), got {density}"
    assert isinstance(
        bond_probability, (int, float)
    ), f"bond_probability must be a number, got {type(bond_probability)}"
    assert (
        0 <= bond_probability <= 1
    ), f"bond_probability must be between 0 and 1 (inclusive), got {bond_probability}"
    assert isinstance(steps, int), f"steps must be an integer, got {type(steps)}"
    assert steps > 0, f"steps must be greater than 0, got {steps}"
    assert isinstance(
        num_simulations, int
    ), f"num_simulations must be an integer, got {type(num_simulations)}"
    assert (
        num_simulations > 0
    ), f"num_simulations must be greater than 0, got {num_simulations}"
    assert flag_center in [
        0,
        1,
    ], f"flag_center must be either 0 or 1, got {flag_center}"
    assert flag_neighbors in [
        0,
        1,
    ], f"flag_neighbors must be either 0 or 1, got {flag_neighbors}"

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
    Runs multiple simulations for a given density and bond probability and returns the fraction of simulations with percolation.

    Parameters:
        grid_size (int): The size of the grid for the simulations.
        density (float): The density of occupied cells in the grid.
        bond_probability (float): The probability of a bond being open between neighboring cells.
        steps (int): The maximum number of simulation steps.
        num_simulations (int): The number of simulation runs to perform.
        flag_center (int, optional): Determines the initial spreader placement (1 for center, 0 for random).
        flag_neighbors (int, optional): Determines the neighborhood type (1 for Moore, 0 for Von Neumann).

    Returns:
        float: The fraction of simulations that resulted in percolation.
    """
    assert isinstance(
        grid_size, int
    ), f"grid_size must be an integer, got {type(grid_size)}"
    assert grid_size > 0, f"grid_size must be greater than 0, got {grid_size}"
    assert isinstance(
        density, (int, float)
    ), f"density must be a number, got {type(density)}"
    assert (
        0 <= density <= 1
    ), f"density must be between 0 and 1 (inclusive), got {density}"
    assert isinstance(
        bond_probability, (int, float)
    ), f"bond_probability must be a number, got {type(bond_probability)}"
    assert (
        0 <= bond_probability <= 1
    ), f"bond_probability must be between 0 and 1 (inclusive), got {bond_probability}"
    assert isinstance(steps, int), f"steps must be an integer, got {type(steps)}"
    assert steps > 0, f"steps must be greater than 0, got {steps}"
    assert isinstance(
        num_simulations, int
    ), f"num_simulations must be an integer, got {type(num_simulations)}"
    assert (
        num_simulations > 0
    ), f"num_simulations must be greater than 0, got {num_simulations}"
    assert flag_center in [
        0,
        1,
    ], f"flag_center must be either 0 or 1, got {flag_center}"
    assert flag_neighbors in [
        0,
        1,
    ], f"flag_neighbors must be either 0 or 1, got {flag_neighbors}"

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
    Simulates percolation probabilities across a range of densities for a fixed bond probability.

    Parameters:
        grid_size (int): The size of the grid for the simulations.
        densities (list of float): A list of density values to simulate.
        bond_probability (float): The probability of a bond being open.
        steps (int): The maximum number of simulation steps.
        num_simulations (int): The number of simulations per density value.
        flag_center (int, optional): Determines the initial spreader placement (1 for center, 0 for random).
        flag_neighbors (int, optional): Determines the neighborhood type (1 for Moore, 0 for Von Neumann).

    Returns:
        list of float: A list of percolation probabilities, one for each density value in `densities`.
    """
    assert isinstance(
        grid_size, int
    ), f"grid_size must be an integer, got {type(grid_size)}"
    assert grid_size > 0, f"grid_size must be greater than 0, got {grid_size}"
    assert isinstance(
        densities, (list)
    ), f"densities must be a list, got {type(densities)}"
    assert isinstance(
        densities[0], (int, float)
    ), f"The list of densities must contain only numbers, got {type(densities[0])}"
    assert isinstance(
        bond_probability, (int, float)
    ), f"bond_probability must be a number, got {type(bond_probability)}"
    assert (
        0 <= bond_probability <= 1
    ), f"bond_probability must be between 0 and 1 (inclusive), got {bond_probability}"
    assert isinstance(steps, int), f"steps must be an integer, got {type(steps)}"
    assert steps > 0, f"steps must be greater than 0, got {steps}"
    assert isinstance(
        num_simulations, int
    ), f"num_simulations must be an integer, got {type(num_simulations)}"
    assert (
        num_simulations > 0
    ), f"num_simulations must be greater than 0, got {num_simulations}"
    assert flag_center in [
        0,
        1,
    ], f"flag_center must be either 0 or 1, got {flag_center}"
    assert flag_neighbors in [
        0,
        1,
    ], f"flag_neighbors must be either 0 or 1, got {flag_neighbors}"

    percolations = []
    for d in densities:
        # Run twice as many simulations for densities near critical point
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
    Runs multiple simulations for a given density and bond probability and for each simulation returns
    Runs multiple simulations for a given density and bond probability and for each simulation returns
    the number of gossip spreaders at the end of the simulation.

    Parameters:
        grid_size (int): The size of the grid for the simulations.
        density (float): A density value to use for the simulations.
        bond_probability (float): A bond probability value to use for the simulations.
        steps (int, optional): The max number of time steps for each simulation.
        num_simulations (int, optional): The number of simulations to run for each density.

    Returns:
        dict: A dictionary with the following structure:
            grid_size (int): The size of the grid.
            density (float): The initial density of occupied cells in the grid.
            bond_probability (float): The bond probability for a bond to be open.
            steps (int): The number of time steps (iterations) for each simulation.
            simulation_outcomes (list): A list containing the total number of gossip spreaders at the end of each simulation.
    """
    assert isinstance(
        grid_size, int
    ), f"grid_size must be an integer, got {type(grid_size)}"
    assert grid_size > 0, f"grid_size must be greater than 0, got {grid_size}"
    assert isinstance(
        density, (int, float)
    ), f"density must be a number, got {type(density)}"
    assert (
        0 <= density <= 1
    ), f"density must be between 0 and 1 (inclusive), got {density}"
    assert isinstance(
        bond_probability, (int, float)
    ), f"bond_probability must be a number, got {type(bond_probability)}"
    assert (
        0 <= bond_probability <= 1
    ), f"bond_probability must be between 0 and 1 (inclusive), got {bond_probability}"
    assert isinstance(steps, int), f"steps must be an integer, got {type(steps)}"
    assert steps > 0, f"steps must be greater than 0, got {steps}"
    assert isinstance(
        num_simulations, int
    ), f"num_simulations must be an integer, got {type(num_simulations)}"
    assert (
        num_simulations > 0
    ), f"num_simulations must be greater than 0, got {num_simulations}"
    assert flag_center in [
        0,
        1,
    ], f"flag_center must be either 0 or 1, got {flag_center}"
    assert flag_neighbors in [
        0,
        1,
    ], f"flag_neighbors must be either 0 or 1, got {flag_neighbors}"

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

        grids = run_simulation(g, steps, flag_neighbors)

        status_counts = count_statuses(grids)

        # Only the final number of GOSSIP_SPREADERS from status_counts for phase diagram
        gossip_spreaders = status_counts["GOSSIP_SPREADER"][-1]
        results["simulation_outcomes"].append(gossip_spreaders)

    return results


def create_results_dict(grid_size, density, bond_probability, steps):
    """
    Creates and returns a dictionary.

    Parameters:
        size (int): The size of the grid.
        grid_size (float): The initial density of occupied cells in the grid.
        bond_probability (float): The bond probability for a bond to be open.
        steps (int): The number of time steps (iterations) for each simulation.

    Returns:
        dict: A dictionary with the following structure:
            grid_size (int): The size of the grid.
            density (float): The initial density of occupied cells in the grid.
            bond_probability (float): The bond probability for a bond to be open.
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
        bond_probability (float): The bond probability for a bond to be open.
        steps (int): The number of time steps (iterations) for each simulation.
        num_simulations (int): The number of simulations to run.
        flag_center (int, optional): The flag to determine the initial spreader placement (1 for center, 0 for outside).
        flag_neighbors (int, optional): The flag to determine the neighborhood type (1 for Moore, 0 for Von Neumann).

    Returns:
        tuple of dict: A tuple containing three dictionaries, each representing the outcomes for a specific status:
            results_gossip (dictionary)
            results_clueless (dictionary)
            results_unoccupied (dictionary)
    """
    assert isinstance(
        grid_size, int
    ), f"grid_size must be an integer, got {type(grid_size)}"
    assert grid_size > 0, f"grid_size must be greater than 0, got {grid_size}"
    assert isinstance(
        density, (int, float)
    ), f"density must be a number, got {type(density)}"
    assert (
        0 <= density <= 1
    ), f"density must be between 0 and 1 (inclusive), got {density}"
    assert isinstance(
        bond_probability, (int, float)
    ), f"bond_probability must be a number, got {type(bond_probability)}"
    assert (
        0 <= bond_probability <= 1
    ), f"bond_probability must be between 0 and 1 (inclusive), got {bond_probability}"
    assert isinstance(steps, int), f"steps must be an integer, got {type(steps)}"
    assert steps > 0, f"steps must be greater than 0, got {steps}"
    assert isinstance(
        num_simulations, int
    ), f"num_simulations must be an integer, got {type(num_simulations)}"
    assert (
        num_simulations > 0
    ), f"num_simulations must be greater than 0, got {num_simulations}"
    assert flag_center in [
        0,
        1,
    ], f"flag_center must be either 0 or 1, got {flag_center}"
    assert flag_neighbors in [
        0,
        1,
    ], f"flag_neighbors must be either 0 or 1, got {flag_neighbors}"

    results_gossip = create_results_dict(grid_size, density, bond_probability, steps)
    results_clueless = create_results_dict(grid_size, density, bond_probability, steps)
    results_unoccupied = create_results_dict(
        grid_size, density, bond_probability, steps
    )

    for _ in range(num_simulations):
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


def simulate_initial_spreader_with_bond_probability(
    size, density, bond_probability, steps, initial_positions
):
    """
    Simulates the gossip spread model with different initial spreader positions
    while considering bond percolation probabilities.

    This function initializes the grid, places an initial spreader at different positions,
    and runs the simulation until percolation occurs or the maximum number of steps is reached.
    It also ensures that all possible bonds are initialized before updating the grid to prevent errors.

    Parameters:
        size (int): The size of the grid (number of rows and columns).
        density (float): The fraction of initially occupied cells.
        bond_probability (float): The probability of a bond allowing gossip spread.
        steps (int): The maximum number of simulation steps.
        initial_positions (list of tuple): A list of (row, col) positions to test as the initial spreader.

    Returns:
        dict: A dictionary where keys are initial spreader positions (row, col),
              and values are the number of steps taken for percolation (-1 if no percolation occurred).
    """

    results = {}

    for position in tqdm(
        initial_positions, desc="Simulating initial spreader positions"
    ):
        grid = Grid(size, density, bond_probability)
        grid.initialize_board()
        row, col = position
        grid.lecture_hall[row][col].set_status(Status.GOSSIP_SPREADER)

        # ✅ FIX: Ensure all bonds exist before running the simulation
        for i in range(size):
            for j in range(size):
                neighbors = grid.get_neighbours(i, j)
                for m, n in neighbors:
                    if ((i, j), (m, n)) not in grid.bonds:
                        grid.bonds[((i, j), (m, n))] = 0  # Default to closed bond
                    if ((m, n), (i, j)) not in grid.bonds:
                        grid.bonds[((m, n), (i, j))] = 0  # Ensure bidirectionality

        # Run simulation
        for step in range(steps):
            grid.update_grid()
            if grid.check_percolation():
                results[position] = step + 1
                break
        else:
            results[position] = -1  # No percolation

    return results
