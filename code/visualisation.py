import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
from matplotlib import animation
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from tqdm import tqdm
from simulation import (
    run_multiple_simulations_for_timeplot_status,
    run_multiple_simulations_for_phase_diagram,
    simulate_and_collect_percolations,
    simulate_density,
    aggregate_cluster_distributions,
    run_multiple_simulations_same_initial_conditions,
    run_simulation,
)
from ca_lecture_hall_model import Colors, Grid


def get_pink_colormap(N):
    pink_colormap = LinearSegmentedColormap.from_list(
        "pink_shades",
        [Colors.DARK_PINK.value, Colors.LIGHT_PINK.value],
        N=N,  # Dark pink to light pink
    )

    return pink_colormap


def show_lecture_hall_over_time(
    cell_grids, save_animation=False, animation_name="gossip_spread_simulation.mp4"
):
    """
    Visualise the spread of a rumour over time using an animated grid and outlining the central region

    Parameters:
        cell_grids (list of list of list of Cell): List of lecture hall grids of cells, one grid for each time step.
        save_animation (bool, optional): Whether to save the animation as an MP4 file. Default is False.
        animation_name (str, optional): The name of the file to save the animation if `save_animation` is True.

    Returns:
            HTML: An HTML object containing the animation for rendering in Jupyter Notebook.
    """
    # Set up the figure and color map
    init_grid = [
        [cell.get_status().value for cell in row] for row in cell_grids[0].lecture_hall
    ]

    fig = plt.figure()
    colors = [
        Colors.UNOCCUPIED.value,
        Colors.CLUELESS.value,
        Colors.GOSSIP_SPREADER.value,
    ]
    cmap = ListedColormap(colors)

    im = plt.imshow(init_grid, cmap=cmap, interpolation="none", animated=True)

    # Add a square specifying the central area, an outer border and legend
    cell_grids[0].add_central_square()
    cell_grids[0].add_outer_box()
    cell_grids[0].add_legend(colors)

    # Get the state number for each cell
    grids = [
        [[cell.get_status().value for cell in row] for row in cell_grid.lecture_hall]
        for cell_grid in cell_grids
    ]

    # Animation function, called sequentally
    def animate(i):
        im.set_array(grids[i])
        return (im,)

    # Call the animator
    anim = animation.FuncAnimation(
        fig, animate, frames=len(grids), interval=200, blit=True
    )

    # Save the animation as an mp4
    if save_animation:
        anim.save(animation_name, fps=30, extra_args=["-vcodec", "libx264"])

    return HTML(anim.to_html5_video())


def simulate_and_create_video(
    grid_size,
    density,
    bond_probability,
    steps=1000,
    flag_center=1,
    save_animation=False,
    animation_name="gossip_spread_simulation.mp4",
):
    """
    Runs one simulation for given grid size, density and spread threshold and
    visualises the spread of a gossip over time as an animated grid outlining the central region.

    Parameters:
        grid_size (int): The size of the grid for the simulations.
        density (float): A density value to use for the simulations.
        bond_probability (float): A spread threshold value to use for the simulations.
        steps (int, optional): The max number of time steps for each simulation. Defaults to 1000.
        flag_center (int, optional): Flag to determine the position of initial spreader. 1 for central area, 0 for edges. Defaults to 1.
        save_animation (bool, optional): Whether to save the animation as an MP4 file. Default is False.
        animation_name (str, optional): The name of the file to save the animation if `save_animation` is True.

    Returns:
            HTML: An HTML object containing the animation for rendering in Jupyter Notebook.
    """
    g = Grid(grid_size, density, bond_probability)
    g.initialize_board(flag_center)
    grids = run_simulation(g, steps)

    return show_lecture_hall_over_time(grids, save_animation, animation_name)


def plot_log_log_distribution(cluster_distribution, density, bond_probability, color):
    """
    Plots the log-log graph of cluster size distribution.

    Parameters:
        cluster_distribution (dict): A dictionary where keys are cluster sizes and values
                                     are the frequencies of those sizes.
        density (float): The density used in the simulation.
        bond_probability (float): The spread threshold used in the simulation.
        color (str): The color for plotting the curve.
    """
    # get the data for the log-log plot
    cluster_sizes = list(cluster_distribution.keys())
    frequencies = list(cluster_distribution.values())

    # apply logarithmic transformation
    log_sizes = np.log10(cluster_sizes)
    log_frequencies = np.log10(frequencies)

    # plot the log-log graph
    plt.plot(
        log_sizes,
        log_frequencies,
        color=color,
        label=f"Density={density}, Spread ={bond_probability}",
        marker=".",
        linestyle="none",
    )


def aggregate_and_plot_cluster_distributions(
    aggregated_distributions, grid_size, labels
):
    """
    Plots the log-log graph of multiple aggregated cluster size distributions.

    Parameters:
        aggregated_distributions (list of dict): A list of aggregated cluster size distributions.
                                                 Each dictionary represents a cluster size distribution
                                                 with keys as cluster sizes (int) and values as their
                                                 corresponding frequencies (int).
        grid_size (int): The size of the grid used in the simulations.
        labels (list of str): A list of labels corresponding to each aggregated cluster distribution,
                              describing the conditions under which the data was generated (e.g.,
                              "Density=0.3, Threshold=0.2").

    Returns:
        None: Displays the log-log plot of the cluster size distributions.
    """
    plt.figure(figsize=(10, 8))

    # plot each aggregated distribution with its label
    for distribution, label in zip(aggregated_distributions, labels):
        cluster_sizes = list(distribution.keys())
        frequencies = list(distribution.values())

        # log transformation
        log_sizes = np.log10(cluster_sizes)
        log_frequencies = np.log10(frequencies)

        plt.plot(log_sizes, log_frequencies, marker=".", linestyle="none", label=label)

    # plot settings
    plt.title(
        f"Log-Log Plot of Cluster Size Distributions (Grid={grid_size}x{grid_size})",
        fontsize=14,
    )
    plt.xlabel("Log10(Cluster Size)", fontsize=12)
    plt.ylabel("Log10(Frequency)", fontsize=12)
    plt.grid(True)
    # plt.legend()
    plt.show()


def simulate_and_plot_gossip_model_all_combinations(
    grid_size, densities, bond_probabilities, num_simulations=100, flag_center=1
):
    """
    Runs multiple simulations for all combinations of densities and spread thresholds,
    aggregates the results, plots the log-log distributions and the counts of each status over time (iterations).

    Parameters:
        simulation_function (function): A function that runs the gossip model simulation and returns a cluster size distribution.
        grid_size (int): The size of the grid for the simulations.
        densities (list): A list of densities to simulate.
        bond_probabilities (list): A list of spread thresholds to simulate.
        num_simulations (int, optional): The number of simulations to run for each set of initial conditions. Defaults to 100.
        flag_center (int, optional): The flag to determine the initial spreader placement. Defaults to 1.
    """
    # Store aggregated cluster distributions and labels
    all_aggregated_distributions = []
    labels = []

    # Iterate over all combinations of densities and spread thresholds
    for density in densities:
        for bond_probability in bond_probabilities:
            print(
                f"Running simulations for Density={density}, Bond Probability={bond_probability}, Grid={grid_size}x{grid_size}"
            )

            # Run multiple simulations for this parameter set
            cluster_distributions = []
            (
                cluster_distribution,
                results_gossip,
                results_clueless,
                results_unoccupied,
            ) = run_multiple_simulations_same_initial_conditions(
                num_simulations,
                grid_size,
                density,
                bond_probability,
                flag_center=flag_center,
            )

            cluster_distributions.append(cluster_distribution)

            # plot_time_status(
            #     results_gossip,
            #     results_clueless,
            #     results_unoccupied,
            #     num_simulations,
            # )

            # Aggregate the cluster size distributions
            aggregated_distribution = aggregate_cluster_distributions(
                cluster_distributions
            )
            all_aggregated_distributions.append(aggregated_distribution)

            # Create a label for this parameter combination
            labels.append(f"Density={density}, Bond Probability={bond_probability}")

    # Plot all aggregated distributions on the same log-log graph
    aggregate_and_plot_cluster_distributions(
        all_aggregated_distributions, grid_size, labels
    )


def plot_percolation_results(
    densities,
    percolations,
    bond_probability=None,
    label=None,
    color=Colors.DARK_PINK.value,
):
    """
    Plots percolation results for different densities or spread thresholds.
    """
    label = label or (
        f"bond_probability = {bond_probability}"
        if bond_probability is not None
        else "Percolation"
    )
    plt.plot(
        densities, percolations, marker="o", linestyle="-", label=label, color=color
    )


def plot_percolation_vs_density(
    grid_size, bond_probability, steps=1000, num_simulations=100, flag_center=1, flag_neighbors=0
):
    """
    Runs multiple simulations for 20 different densities and a given spread thresholds,
    calculates the probability of a percolation occuring, and plots it.

    Parameters:
        grid_size (int): The size of the grid for the simulations.
        bond_probability (float): A spread threshold value to use for the simulations.
        steps (int, optional): The max number of time steps for each simulation. Defaults to 1000.
        num_simulations (int, optional): The number of simulations to run for each density. Defaults to 100.
    """
    densities = np.linspace(0.1, 1, 20)
    percolations = []

    print("Starting simulation for different densities...")

    for density in tqdm(densities, desc="Simulating densities"):
        if 0.4 <= density <= 0.7:
            percolations.append(
                simulate_density(
                    grid_size, density, bond_probability, steps, num_simulations * 2, flag_center, flag_neighbors
                )
            )
        else:
            percolations.append(
                simulate_density(
                    grid_size, density, bond_probability, steps, num_simulations, flag_center, flag_neighbors
                )
            )

    print("Simulations completed.")

    plt.figure(figsize=(8, 6))
    plot_percolation_results(densities, percolations, bond_probability)
    plt.xlabel("Density")
    plt.ylabel("Fraction of simulations with percolation")
    plt.title("Plot of percolation occurence for different density")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_percolation_vs_bond_probability(
    grid_size, density, steps=1000, num_simulations=100, flag_center=1, flag_neighbors=0
):
    """
    Runs multiple simulations for 20 different Bond Probabilitys and a given density,
    calculates the probability of a percolation occuring, and plots it.

    Parameters:
        grid_size (int): The size of the grid for the simulations.
        density (float): A density value to use for the simulations.
        steps (int, optional): The max number of time steps for each simulation. Defaults to 1000.
        num_simulations (int, optional): The number of simulations to run for each density. Defaults to 100.
    """
    bond_probabilities = np.linspace(0.1, 1, 20)
    percolations = []

    print("Starting simulation for different bond probabilities...")

    for bond_probability in tqdm(
        bond_probabilities, desc="Simulating bond probabilities"
    ):
        if 0.3 <= bond_probability <= 0.8:
            percolations.append(
                simulate_density(
                    grid_size, density, bond_probability, steps, num_simulations * 2, flag_center, flag_neighbors
                )
            )
        else:
            percolations.append(
                simulate_density(
                    grid_size, density, bond_probability, steps, num_simulations, flag_center, flag_neighbors
                )
            )

    print("Simulations completed.")

    plt.figure(figsize=(8, 6))
    plt.plot(
        bond_probabilities,
        percolations,
        marker="o",
        linestyle="-",
        color=Colors.DARK_PINK.value,
        label=f"density = {density:.2f}",
    )
    plt.xlabel("Bond Probability")
    plt.ylabel("Fraction of Simulations with Percolation")
    plt.title("Plot of Percolation vs Bond Probability")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_percolation_vs_density_vs_bond_probability(
    grid_size, steps=1000, num_simulations=100, flag_center=1, flag_neighbors=0
):
    """
    Runs multiple simulations for 20 different densities and 10 different Bond Probabilitys,
    calculates the probability of a percolation occuring, and plots it.

    Parameters:
        grid_size (int): The size of the grid for the simulations.
        steps (int, optional): The max number of time steps for each simulation. Defaults to 1000.
        num_simulations (int, optional): The number of simulations to run for each density. Defaults to 100.
        flag_center (int, optional): The flag to determine the initial spreader placement. Defaults to 1.
        flag_neighbors (int, optional): The flag to determine the neighborhood type. If 1, use the Moore neighborhood. If 0, use the Von Neumann neighborhood.
    """
    bond_probabilities = np.linspace(0.1, 1, 10)
    densities = np.linspace(0.1, 1, 20)

    plt.figure(figsize=(10, 8))

    # Get the custom colormap
    pink_colormap = get_pink_colormap(len(bond_probabilities))

    colors = [pink_colormap(i) for i in range(len(bond_probabilities))]

    print("Starting simulation for different spread thresholds and densities...")

    for i, bond_probability in tqdm(
        enumerate(bond_probabilities),
        desc="Simulating bond probabilities",
        total=len(bond_probabilities),
    ):
        if 0.3 <= bond_probability <= 0.8:
            percolations = simulate_and_collect_percolations(
                grid_size,
                densities,
                bond_probability,
                steps,
                num_simulations * 2,
                flag_center,
                flag_neighbors,
            )
            plot_percolation_results(
                densities, percolations, bond_probability, color=colors[i]
            )
        else:
            percolations = simulate_and_collect_percolations(
                grid_size,
                densities,
                bond_probability,
                steps,
                num_simulations,
                flag_center,
                flag_neighbors,
            )
            plot_percolation_results(
                densities, percolations, bond_probability, color=colors[i]
            )

    print("Simulations completed.")

    plt.xlabel("Density")
    plt.ylabel("Fraction of simulations with percolation")
    plt.title("Percolation vs Density and Bond Probability")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_3d_percolation_vs_density_and_bond_probability(
    grid_size, steps=1000, num_simulations=100, flag_center=1
):
    """
    Plots the percolation probability against density and Bond Probability as a 3D plot.

    Parameters:
        grid_size (int): The size of the grid for the simulations.
        steps (int, optional): The max number of time steps for each simulation. Defaults to 1000.
        num_simulations (int, optional): The number of simulations to run for each density. Defaults to 100.
        flag_center (int, optional): The flag to determine the initial spreader placement. Defaults to 1.
    """
    densities = np.linspace(0.1, 1, 10)
    bond_probabilities = np.linspace(0.1, 1, 10)

    percolation_data = []

    for probability in tqdm(bond_probabilities, desc="Simulating bond probabilities"):
        # Use the simulate_and_collect_percolations function for densities
        percolations = simulate_and_collect_percolations(
            grid_size, densities, probability, steps, num_simulations, flag_center
        )
        percolation_data.append(percolations)

    # Convert data to arrays for plotting
    bond_probabilities, densities = np.meshgrid(bond_probabilities, densities)
    percolations = np.transpose(np.array(percolation_data))

    pink_colormap = get_pink_colormap(len(bond_probabilities))
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        bond_probabilities,
        densities,
        percolations,
        cmap=pink_colormap,
        edgecolor="none",
    )

    ax.set_ylabel("Density")
    ax.set_xlabel("Bond Probability")
    ax.set_zlabel("Percolation Probability")
    ax.set_title("3D Plot of Percolation vs Density and Bond Probability")

    fig.colorbar(surf, shrink=0.5, aspect=10, label="Percolation Probability")
    plt.show()


def plot_3d_gossip_spreader_counts(
    grid_size, steps=1000, num_simulations=100, flag_center=1
):
    """
    Plots the count of the GOSSIP_SPREADERS against density and Bond Probability as a 3D plot.

    Parameters:
        grid_size (int): The size of the grid for the simulations.
        steps (int, optional): The max number of time steps for each simulation. Defaults to 1000.
        num_simulations (int, optional): The number of simulations to run for each density. Defaults to 100.
    """
    # range of densities and Bond Probabilitys
    densities = np.linspace(0.1, 1, 10)
    bond_probabilities = np.linspace(0.1, 1, 10)

    spreader_counts = np.zeros((len(densities), len(bond_probabilities)))

    for i, bond_probability in tqdm(
        enumerate(bond_probabilities),
        desc="Bond Probability",
        total=len(bond_probabilities),
    ):
        for j, density in enumerate(densities):
            results = run_multiple_simulations_for_phase_diagram(
                grid_size,
                density,
                bond_probability,
                steps,
                num_simulations,
                flag_center,
            )

            # average number of GOSSIP_SPREADERS across simulations
            average_spreaders = np.mean(results["simulation_outcomes"])
            # print(f"Average GOSSIP_SPREADERS for density={density:.2f}, bond_probability={bond_probability:.2f}: {average_spreaders}\n") # print statements for checking
            spreader_counts[j, i] = average_spreaders  # used for Z

    # 3D plot
    pink_colormap = get_pink_colormap(len(bond_probabilities))
    X, Y = np.meshgrid(bond_probabilities, densities)
    Z = spreader_counts
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection="3d")
    surface = ax.plot_surface(X, Y, Z, cmap=pink_colormap)
    ax.set_xlabel("Bond Probability")
    ax.set_ylabel("Density")
    ax.set_zlabel("Average Amount of Gossip Spreaders")
    ax.set_title(
        "Phase Diagram of Gossip Spreaders against Density and Bond Probability"
    )
    fig.colorbar(surface, ax=ax, shrink=0.6, aspect=10)
    plt.show()


def plot_time_status(
    ax,
    grid_size,
    density,
    bond_probability,
    steps,
    num_simulations,
    flag_center,
    x_limits,
    y_limits,
    flag_neighbors=0,
):
    """
    Plots the counts of each status over time (iterations)..

    Parameters:
        grid_size (int): The size of the grid.
        steps (int): The number of time steps (iterations) for each simulation.
        num_simulations (int): The number of simulations to run.
        density (float): The density of the grid.
        bond_probability (float): The Bond Probability for the gossip model.
        flag_center (int): Flag to determine the position of initial spreader
        x_limits (tuple): Limits for the x-axis (time).
        y_limits (tuple): Limits for the y-axis (number of cells).
        flag_neighbors (int): Flag to determine the neighborhood type. If 1, use the Moore neighborhood. If 0, use the Von Neumann neighborhood.
    """
    results_gossip, results_clueless, results_unoccupied = (
        run_multiple_simulations_for_timeplot_status(
            grid_size,
            density,
            bond_probability,
            steps,
            num_simulations,
            flag_center,
            flag_neighbors,
        )
    )

    # average results over simulations
    average_gossip = [
        sum(x) / num_simulations for x in zip(*results_gossip["simulation_outcomes"])
    ]
    average_clueless = [
        sum(x) / num_simulations for x in zip(*results_clueless["simulation_outcomes"])
    ]
    average_unoccupied = [
        sum(x) / num_simulations
        for x in zip(*results_unoccupied["simulation_outcomes"])
    ]

    std_gossip = [
        np.std(x) for x in zip(*results_gossip["simulation_outcomes"])
    ]

    std_clueless = [
        np.std(x) for x in zip(*results_clueless["simulation_outcomes"])
    ]

    iterations = range(len(average_unoccupied))

    ax.plot(
        iterations,
        average_unoccupied,
        label="UNOCCUPIED",
        color=Colors.UNOCCUPIED.value,
    )
    ax.plot(
        iterations, average_clueless, label="CLUELESS", color=Colors.CLUELESS_DARK.value
    )
    ax.plot(
        iterations,
        average_gossip,
        label="GOSSIP_SPREADER",
        color=Colors.GOSSIP_SPREADER.value,
    )

    ax.fill_between(iterations, np.array(average_gossip) - np.array(std_gossip), np.array(average_gossip) + np.array(std_gossip), color=Colors.GOSSIP_SPREADER.value, alpha=0.3, label="Standard Deviation Gossip Spreader")
    ax.fill_between(iterations, np.array(average_clueless) - np.array(std_clueless), np.array(average_clueless) + np.array(std_clueless), color=Colors.CLUELESS_DARK.value, alpha=0.3, label="Standard Deviation Clueless")


    ax.set_title(f"Density: {density}, Bond prob: {bond_probability}")
    ax.set_xlabel("Time Steps", fontsize=14)
    ax.set_ylabel("Number of Cells", fontsize=14)
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)
    ax.grid(True)
