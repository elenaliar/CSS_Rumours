import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
from matplotlib import animation
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle
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
    # colors = ["lightgray", "white", "gold", "goldenrod"]
    colors = [
        Colors.UNOCCUPIED.value,
        Colors.CLUELESS.value,
        Colors.SECRET_KEEPER.value,
        Colors.GOSSIP_SPREADER.value,
    ]
    cmap = ListedColormap(colors)

    im = plt.imshow(init_grid, cmap=cmap, interpolation="none", animated=True)

    # Add a square specifying the central area
    ax = plt.gca()
    left_corner_coordinate = (len(init_grid[0]) // 4) - 0.5
    square = Rectangle(
        (left_corner_coordinate, left_corner_coordinate),
        len(init_grid[0]) // 2 + 1,
        len(init_grid[0]) // 2 + 1,
        edgecolor="dimgray",
        facecolor="none",
        linewidth=2,
    )
    ax.add_patch(square)

    # Remove axes and add black square as a border
    square = Rectangle(
        (-0.5, -0.5),
        len(init_grid[0]),
        len(init_grid[0]),
        edgecolor="black",
        facecolor="none",
        linewidth=3,
    )
    ax.add_patch(square)
    plt.axis("off")

    # Add legend to the plot
    legend_patches = [
        Patch(
            facecolor=color, edgecolor="black", label=f"{state}"
        )  # TODO: use the names of states instead of numbers
        for state, color in zip(
            ["Unoccupied", "Clueless", "Secret keeper", "Gossip spreader"], colors
        )
    ]

    plt.legend(
        handles=legend_patches,
        title="State",
        loc="upper right",
        bbox_to_anchor=(1.3, 1),
    )

    # Get the state number for each cell
    grids = [
        [[cell.get_status().value for cell in row] for row in cell_grid.lecture_hall]
        for cell_grid in cell_grids
    ]

    # animation function. This is called sequentially
    def animate(i):
        im.set_array(grids[i])
        return (im,)

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(
        fig, animate, frames=len(grids), interval=200, blit=True
    )

    # save the animation as an mp4.  This requires ffmpeg or mencoder to be
    # installed.  The extra_args ensure that the x264 codec is used, so that
    # the video can be embedded in html5.  You may need to adjust this for
    # your system: for more information, see
    # http://matplotlib.sourceforge.net/api/animation_api.html
    if save_animation:
        anim.save(animation_name, fps=30, extra_args=["-vcodec", "libx264"])

    return HTML(anim.to_html5_video())


def simulate_and_create_video(
    grid_size,
    density,
    spread_threshold,
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
        spread_threshold (float): A spread threshold value to use for the simulations.
        steps (int, optional): The max number of time steps for each simulation. Defaults to 1000.
        flag_center (int, optional): Flag to determine the position of initial spreader. 1 for central area, 0 for edges. Defaults to 1.
        save_animation (bool, optional): Whether to save the animation as an MP4 file. Default is False.
        animation_name (str, optional): The name of the file to save the animation if `save_animation` is True.

    Returns:
            HTML: An HTML object containing the animation for rendering in Jupyter Notebook.
    """
    g = Grid(grid_size, density, spread_threshold)
    g.initialize_board(flag_center)
    grids = run_simulation(g, steps)

    return show_lecture_hall_over_time(grids, save_animation, animation_name)


def plot_log_log_distribution(cluster_distribution, density, spread_threshold, color):
    """
    Plots the log-log graph of cluster size distribution.

    Parameters:
        cluster_distribution (dict): A dictionary where keys are cluster sizes and values
                                     are the frequencies of those sizes.
        density (float): The density used in the simulation.
        spread_threshold (float): The spread threshold used in the simulation.
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
        label=f"Density={density}, Spread ={spread_threshold}",
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
    plt.legend()
    plt.show()


def simulate_and_plot_gossip_model_all_combinations(
    grid_size, densities, spread_thresholds, num_simulations=100, flag_center=1
):
    """
    Runs multiple simulations for all combinations of densities and spread thresholds,
    aggregates the results, and plots the log-log distributions.

    Parameters:
        simulation_function (function): A function that runs the gossip model simulation and returns a cluster size distribution.
        grid_size (int): The size of the grid for the simulations.
        densities (list): A list of densities to simulate.
        spread_thresholds (list): A list of spread thresholds to simulate.
        num_simulations (int, optional): The number of simulations to run for each set of initial conditions. Defaults to 100.
        flag_center (int, optional): The flag to determine the initial spreader placement. Defaults to 1.
    """
    # Store aggregated cluster distributions and labels
    all_aggregated_distributions = []
    labels = []

    # Iterate over all combinations of densities and spread thresholds
    for density in densities:
        for spread_threshold in spread_thresholds:
            print(
                f"Running simulations for Density={density}, Spread Threshold={spread_threshold}..."
            )

            # Run multiple simulations for this parameter set
            cluster_distributions = []
            cluster_distribution = run_multiple_simulations_same_initial_conditions(
                num_simulations,
                grid_size,
                density,
                spread_threshold,
                flag_center=flag_center,
            )
            cluster_distributions.append(cluster_distribution)

            # Aggregate the cluster size distributions
            aggregated_distribution = aggregate_cluster_distributions(
                cluster_distributions
            )
            all_aggregated_distributions.append(aggregated_distribution)

            # Create a label for this parameter combination
            labels.append(f"Density={density}, Threshold={spread_threshold}")

    # Plot all aggregated distributions on the same log-log graph
    aggregate_and_plot_cluster_distributions(
        all_aggregated_distributions, grid_size, labels
    )


def plot_percolation_results(
    densities, percolations, spread_threshold=None, label=None
):
    """
    Plots percolation results for different densities or spread thresholds.
    """
    label = label or (
        f"spread_threshold = {spread_threshold}"
        if spread_threshold is not None
        else "Percolation"
    )
    plt.plot(
        densities,
        percolations,
        marker="o",
        linestyle="-",
        label=label,
    )


def plot_percolation_vs_density(
    grid_size, spread_threshold, steps=1000, num_simulations=100
):
    """
    Runs multiple simulations for 20 different densities and a given spread thresholds,
    calculates the probability of a percolation occuring, and plots it.

    Parameters:
        grid_size (int): The size of the grid for the simulations.
        spread_threshold (float): A spread threshold value to use for the simulations.
        steps (int, optional): The max number of time steps for each simulation. Defaults to 1000.
        num_simulations (int, optional): The number of simulations to run for each density. Defaults to 100.
    """
    densities = np.linspace(0, 1, 20)
    percolations = []

    print("Starting simulation for different densities...")

    for density in tqdm(densities, desc="Simulating densities"):
        percolations.append(
            simulate_density(
                grid_size, density, spread_threshold, steps, num_simulations
            )
        )

    print("Simulations completed.")

    plt.figure(figsize=(8, 6))
    plot_percolation_results(densities, percolations, spread_threshold)
    plt.xlabel("Density")
    plt.ylabel("Fraction of simulations with percolation")
    plt.title("Plot of percolation occurence for different density")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_percolation_vs_spread_threshold(
    grid_size, density, steps=1000, num_simulations=100
):
    """
    Runs multiple simulations for 20 different spreading thresholds and a given density,
    calculates the probability of a percolation occuring, and plots it.

    Parameters:
        grid_size (int): The size of the grid for the simulations.
        density (float): A density value to use for the simulations.
        steps (int, optional): The max number of time steps for each simulation. Defaults to 1000.
        num_simulations (int, optional): The number of simulations to run for each density. Defaults to 100.
    """
    spread_thresholds = np.linspace(0, 1, 20)
    percolations = []

    print("Starting simulation for different spread thresholds...")

    for spread_threshold in tqdm(spread_thresholds, desc="Simulating thresholds"):
        percolations.append(
            simulate_density(
                grid_size, density, spread_threshold, steps, num_simulations
            )
        )

    print("Simulations completed.")

    plt.figure(figsize=(8, 6))
    plt.plot(
        spread_thresholds,
        percolations,
        marker="o",
        linestyle="-",
        color="blue",
        label=f"density = {density:.2f}",
    )
    plt.xlabel("Spread Threshold")
    plt.ylabel("Fraction of Simulations with Percolation")
    plt.title("Plot of Percolation vs Spread Threshold")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_percolation_vs_density_vs_spread_threshold(
    grid_size, steps=1000, num_simulations=100
):
    """
    Runs multiple simulations for 20 different densities and 10 different spreading thresholds,
    calculates the probability of a percolation occuring, and plots it.

    Parameters:
        grid_size (int): The size of the grid for the simulations.
        steps (int, optional): The max number of time steps for each simulation. Defaults to 1000.
        num_simulations (int, optional): The number of simulations to run for each density. Defaults to 100.
    """
    spread_thresholds = np.linspace(0, 1, 10)
    densities = np.linspace(0, 1, 20)

    plt.figure(figsize=(10, 8))

    print("Starting simulation for different spread thresholds and densities...")

    for spread_threshold in tqdm(spread_thresholds, desc="Simulating thresholds"):
        percolations = simulate_and_collect_percolations(
            grid_size, densities, spread_threshold, steps, num_simulations
        )
        plot_percolation_results(densities, percolations, spread_threshold)

    print("Simulations completed.")

    plt.xlabel("Density")
    plt.ylabel("Fraction of simulations with percolation")
    plt.title("Percolation vs Density and Spread Threshold")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_3d_percolation_vs_density_and_threshold(
    grid_size, steps=1000, num_simulations=100
):
    """
    Plots the percolation probability against density and spreading threshold as a 3D plot.

    Parameters:
        grid_size (int): The size of the grid for the simulations.
        steps (int, optional): The max number of time steps for each simulation. Defaults to 1000.
        num_simulations (int, optional): The number of simulations to run for each density. Defaults to 100.
    """
    densities = np.linspace(0, 1, 10)
    thresholds = np.linspace(0, 1, 10)

    percolation_data = []

    for threshold in tqdm(thresholds, desc="Simulating thresholds"):
        # Use the simulate_and_collect_percolations function for densities
        percolations = simulate_and_collect_percolations(
            grid_size, densities, threshold, steps, num_simulations
        )
        percolation_data.append(percolations)

    # Convert data to arrays for plotting
    densities, thresholds = np.meshgrid(densities, thresholds)
    percolations = np.array(percolation_data)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        densities, thresholds, percolations, cmap="viridis", edgecolor="none"
    )

    ax.set_xlabel("Density")
    ax.set_ylabel("Spread Threshold")
    ax.set_zlabel("Percolation Probability")
    ax.set_title("3D Plot of Percolation vs Density and Spread Threshold")

    fig.colorbar(surf, shrink=0.5, aspect=10, label="Percolation Probability")
    plt.show()


def plot_3d_gossip_spreader_counts(grid_size, steps=1000, num_simulations=100):
    """
    Plots the count of the GOSSIP_SPREADERS against density and spreading threshold as a 3D plot.

    Parameters:
        grid_size (int): The size of the grid for the simulations.
        steps (int, optional): The max number of time steps for each simulation. Defaults to 1000.
        num_simulations (int, optional): The number of simulations to run for each density. Defaults to 100.
    """
    # range of densities and spreading thresholds
    densities = np.linspace(0, 1, 10)
    spread_thresholds = np.linspace(0, 1, 10)

    spreader_counts = np.zeros((len(densities), len(spread_thresholds)))

    for i, spread_threshold in tqdm(
        enumerate(spread_thresholds),
        desc="Spread Thresholds",
        total=len(spread_thresholds),
    ):
        for j, density in tqdm(
            enumerate(densities),
            desc=f"Densities (Threshold {spread_threshold:.2f})",
            total=len(densities),
            leave=False,
        ):
            results = run_multiple_simulations_for_phase_diagram(
                grid_size, density, spread_threshold, steps, num_simulations
            )

            # average number of GOSSIP_SPREADERS across simulations
            average_spreaders = np.mean(results["simulation_outcomes"])
            # print(f"Average GOSSIP_SPREADERS for density={density:.2f}, spread_threshold={spread_threshold:.2f}: {average_spreaders}\n") # print statements for checking
            spreader_counts[j, i] = average_spreaders  # used for Z

    # 3D plot
    X, Y = np.meshgrid(spread_thresholds, densities)
    Z = spreader_counts
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection="3d")
    surface = ax.plot_surface(X, Y, Z, cmap="coolwarm")
    ax.set_xlabel("Spreading Threshold")
    ax.set_ylabel("Density")
    ax.set_zlabel("Average Amount of Gossip Spreaders")
    ax.set_title(
        "Phase Diagram of Gossip Spreaders against Density and Spreading Threshold"
    )
    fig.colorbar(surface, ax=ax, shrink=0.6, aspect=10)
    plt.show()


def plot_time_status(
    grid_size, density, spread_threshold, steps, num_simulations, flag_center=1
):
    """
    Plots the counts of each status over time (iterations)..

    Parameters:
        grid_size (int): The size of the grid.
        steps (int): The number of time steps (iterations) for each simulation.
        num_simulations (int): The number of simulations to run.
        density (float): The density of the grid.
        spread_threshold (float): The spreading threshold for the gossip model.
        flag_center (int): Flag to determine the position of initial spreader
    """
    results_gossip, results_secret, results_clueless, results_unoccupied = (
        run_multiple_simulations_for_timeplot_status(
            grid_size, density, spread_threshold, steps, num_simulations, flag_center
        )
    )

    # average results over simulations
    average_gossip = [
        sum(x) / num_simulations for x in zip(*results_gossip["simulation_outcomes"])
    ]
    average_secret = [
        sum(x) / num_simulations for x in zip(*results_secret["simulation_outcomes"])
    ]
    average_clueless = [
        sum(x) / num_simulations for x in zip(*results_clueless["simulation_outcomes"])
    ]
    average_unoccupied = [
        sum(x) / num_simulations
        for x in zip(*results_unoccupied["simulation_outcomes"])
    ]

    iterations = range(len(average_unoccupied))

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, average_unoccupied, label="UNOCCUPIED")
    plt.plot(iterations, average_clueless, label="CLUELESS")
    plt.plot(iterations, average_secret, label="SECRET_KEEPER")
    plt.plot(iterations, average_gossip, label="GOSSIP_SPREADER")
    plt.xlabel("Time Steps")
    plt.ylabel("Number of Cells")
    plt.title("Status Counts Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
