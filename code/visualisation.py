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
    run_simulation,
)
from ca_lecture_hall_model import Colors, Grid


def get_pink_colormap(N):
    """
    Returns a custom colormap from dark to light pink with N shades in between.
    """
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
    flag_neighbors=0,
    save_animation=False,
    animation_name="gossip_spread_simulation.mp4",
):
    """
    Runs one simulation for given grid size, density and bond probability and
    visualises the spread of a gossip over time as an animated grid outlining the central region.

    Parameters:
        grid_size (int): The size of the grid for the simulations.
        density (float): A density value to use for the simulations.
        bond_probability (float): A bond probability value to use for the simulations.
        steps (int, optional): The max number of time steps for each simulation. Defaults to 1000.
        flag_center (int, optional): Determines the initial spreader placement (1 for center, 0 for random). Defaults to 1.
        flag_neighbors (int, optional): Determines the neighborhood type (1 for Moore, 0 for Von Neumann). Defaults to 0.
        save_animation (bool, optional): Whether to save the animation as an MP4 file. Default is False.
        animation_name (str, optional): The name of the file to save the animation if `save_animation` is True.

    Returns:
            HTML: An HTML object containing the animation for rendering in Jupyter Notebook.
    """
    g = Grid(grid_size, density, bond_probability)
    g.initialize_board(flag_center, flag_neighbors)
    grids = run_simulation(g, steps, flag_neighbors)

    return show_lecture_hall_over_time(grids, save_animation, animation_name)


def plot_percolation_results(
    densities,
    percolations,
    bond_probability=None,
    label=None,
    color=Colors.DARK_PINK.value,
):
    """
    Plots percolation results for different densities or bond probabilities.
    """
    label = label or (
        f"bond_probability = {bond_probability:.2f}"
        if bond_probability is not None
        else "Percolation"
    )
    plt.plot(
        densities, percolations, marker="o", linestyle="-", label=label, color=color
    )


def plot_percolation_vs_density(
    grid_size,
    bond_probability,
    steps=1000,
    num_simulations=100,
    flag_center=1,
    flag_neighbors=0,
    save=False,
    filename="percolation_density.png",
):
    """
    Runs multiple simulations for 20 different densities and a given bond probability,
    calculates the probability of a percolation occuring, and plots it.

    Parameters:
        grid_size (int): The size of the grid for the simulations.
        bond_probability (float): A bond probability value to use for the simulations.
        steps (int, optional): The max number of time steps for each simulation. Defaults to 1000.
        num_simulations (int, optional): The number of simulations to run for each density. Defaults to 100.
        flag_center (int, optional): Determines the initial spreader placement (1 for center, 0 for random). Defaults to 1.
        flag_neighbors (int, optional): Determines the neighborhood type (1 for Moore, 0 for Von Neumann). Defaults to 0.
        save (bool, optional):  Whether to save the plot as an png file. Default is False.
        filename (str, optional): The name of the file to save the plot if `save` is True.
    """
    densities = np.linspace(0.1, 1, 20)
    percolations = []

    print("Starting simulation for different densities...")

    for density in tqdm(densities, desc="Simulating densities"):
        if 0.4 <= density <= 0.7:
            percolations.append(
                simulate_density(
                    grid_size,
                    density,
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
                    density,
                    bond_probability,
                    steps,
                    num_simulations,
                    flag_center,
                    flag_neighbors,
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

    if save:
        plt.savefig(filename, dpi=300)

    plt.show()


def plot_percolation_vs_bond_probability(
    grid_size,
    density,
    steps=1000,
    num_simulations=100,
    flag_center=1,
    flag_neighbors=0,
    save=False,
    filename="percolation_bond_probability.png",
):
    """
    Runs multiple simulations for 20 different Bond Probabilitys and a given density,
    calculates the probability of a percolation occuring, and plots it.

    Parameters:
        grid_size (int): The size of the grid for the simulations.
        density (float): A density value to use for the simulations.
        steps (int, optional): The max number of time steps for each simulation. Defaults to 1000.
        num_simulations (int, optional): The number of simulations to run for each density. Defaults to 100.
        flag_center (int, optional): Determines the initial spreader placement (1 for center, 0 for random). Defaults to 1.
        flag_neighbors (int, optional): Determines the neighborhood type (1 for Moore, 0 for Von Neumann). Defaults to 0.
        save (bool, optional):  Whether to save the plot as an png file. Default is False.
        filename (str, optional): The name of the file to save the plot if `save` is True.
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
                    grid_size,
                    density,
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
                    density,
                    bond_probability,
                    steps,
                    num_simulations,
                    flag_center,
                    flag_neighbors,
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

    if save:
        plt.savefig(filename, dpi=300)

    plt.show()


def plot_percolation_vs_density_vs_bond_probability(
    grid_size,
    steps=1000,
    num_simulations=100,
    flag_center=1,
    flag_neighbors=0,
    save=False,
    filename="percolation_density_bond_probability.png",
):
    """
    Runs multiple simulations for 20 different densities and 10 different Bond Probabilitys,
    calculates the probability of a percolation occuring, and plots it.

    Parameters:
        grid_size (int): The size of the grid for the simulations.
        steps (int, optional): The max number of time steps for each simulation. Defaults to 1000.
        num_simulations (int, optional): The number of simulations to run for each density. Defaults to 100.
        flag_center (int, optional): Determines the initial spreader placement (1 for center, 0 for random). Defaults to 1.
        flag_neighbors (int, optional): Determines the neighborhood type (1 for Moore, 0 for Von Neumann). Defaults to 0.
        save (bool, optional):  Whether to save the plot as an png file. Default is False.
        filename (str, optional): The name of the file to save the plot if `save` is True.
    """
    bond_probabilities = np.linspace(0.1, 1, 10)
    densities = np.linspace(0.1, 1, 20)

    plt.figure(figsize=(10, 8))

    # Get the custom colormap
    pink_colormap = get_pink_colormap(len(bond_probabilities))

    colors = [pink_colormap(i) for i in range(len(bond_probabilities))]

    print("Starting simulation for different bond probabilities and densities...")

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

    if save:
        plt.savefig(filename, dpi=300)

    plt.show()


def plot_3d_percolation_vs_density_and_bond_probability(
    grid_size,
    steps=1000,
    num_simulations=100,
    flag_center=1,
    save=False,
    filename="3d_percolation.png",
):
    """
    Plots the percolation probability against density and Bond Probability as a 3D plot.

    Parameters:
        grid_size (int): The size of the grid for the simulations.
        steps (int, optional): The max number of time steps for each simulation. Defaults to 1000.
        num_simulations (int, optional): The number of simulations to run for each density. Defaults to 100.
        flag_center (int, optional): Determines the initial spreader placement (1 for center, 0 for random). Defaults to 1.
        save (bool, optional):  Whether to save the plot as an png file. Default is False.
        filename (str, optional): The name of the file to save the plot if `save` is True.
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

    if save:
        plt.savefig(filename, dpi=300)

    plt.show()


def plot_3d_gossip_spreader_counts(
    grid_size,
    steps=1000,
    num_simulations=100,
    flag_center=1,
    save=False,
    filename="3d_gossipers.png",
):
    """
    Plots the count of the GOSSIP_SPREADERS against density and Bond Probability as a 3D plot.

    Parameters:
        grid_size (int): The size of the grid for the simulations.
        steps (int, optional): The max number of time steps for each simulation. Defaults to 1000.
        num_simulations (int, optional): The number of simulations to run for each density. Defaults to 100.
        flag_center (int, optional): Determines the initial spreader placement (1 for center, 0 for random). Defaults to 1.
        save (bool, optional):  Whether to save the plot as an png file. Default is False.
        filename (str, optional): The name of the file to save the plot if `save` is True.
    """
    # Range of densities and Bond Probabilitys
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

            # Average number of GOSSIP_SPREADERS across simulations
            average_spreaders = np.mean(results["simulation_outcomes"])
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

    if save:
        plt.savefig(filename, dpi=300)

    plt.show()


def plot_time_status(
    ax,
    grid_size,
    density,
    bond_probability,
    steps,
    num_simulations,
    x_limits,
    y_limits,
    flag_center=1,
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
        x_limits (tuple): Limits for the x-axis (time).
        y_limits (tuple): Limits for the y-axis (number of cells).
        flag_center (int, optional): Determines the initial spreader placement (1 for center, 0 for random). Defaults to 1.
        flag_neighbors (int, optional): Determines the neighborhood type (1 for Moore, 0 for Von Neumann). Defaults to 0.
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

    std_gossip = [np.std(x) for x in zip(*results_gossip["simulation_outcomes"])]

    std_clueless = [np.std(x) for x in zip(*results_clueless["simulation_outcomes"])]

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

    ax.fill_between(
        iterations,
        np.array(average_gossip) - np.array(std_gossip),
        np.array(average_gossip) + np.array(std_gossip),
        color=Colors.GOSSIP_SPREADER.value,
        alpha=0.3,
        label="Standard Deviation Gossip Spreader",
    )
    ax.fill_between(
        iterations,
        np.array(average_clueless) - np.array(std_clueless),
        np.array(average_clueless) + np.array(std_clueless),
        color=Colors.CLUELESS_DARK.value,
        alpha=0.3,
        label="Standard Deviation Clueless",
    )

    ax.set_title(f"Density: {density}, Bond prob: {bond_probability}")
    ax.set_xlabel("Time Steps", fontsize=14)
    ax.set_ylabel("Number of Cells", fontsize=14)
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)
    ax.grid(True)


def plot_spreader_effect(results):
    """
    Plots the effect of the initial spreader position on percolation speed.

    Parameters:
        results (dict): Dictionary with keys as initial spreader positions (row, col) and values as steps to percolation.

    Returns:
        None: Displays a scatter plot.
    """
    positions = list(results.keys())
    steps = list(results.values())

    x = [pos[1] for pos in positions]
    y = [pos[0] for pos in positions]

    plt.figure(figsize=(10, 8))

    # Use the same colormap as the 3D plot
    pink_colormap = get_pink_colormap(len(steps))

    scatter = plt.scatter(x, y, c=steps, cmap=pink_colormap, edgecolor="black", s=100)

    plt.colorbar(scatter, label="Steps to Percolation (-1 means no percolation)")
    plt.xlabel("Column", fontsize=14)
    plt.ylabel("Row", fontsize=14)
    plt.title("Effect of Initial Spreader Position on Percolation Speed", fontsize=16)
    plt.grid(True, linestyle="--", linewidth=0.5, color=Colors.UNOCCUPIED.value)

    plt.show()


def plot_distance_vs_steps(results, size):
    """
    Plots the relationship between distance from the center and the number of steps to percolation.

    Parameters:
        results (dict): Dictionary with keys as initial spreader positions (row, col) and values as steps to percolation.
        size (int): Size of the grid.

    Returns:
        None: Displays a line plot.
    """
    center = (size // 2, size // 2)
    distances = []
    steps = []

    for (row, col), step_count in results.items():
        distance_to_center = np.sqrt((row - center[0]) ** 2 + (col - center[1]) ** 2)
        distances.append(distance_to_center)
        steps.append(step_count)

    plt.figure(figsize=(10, 6))
    plt.plot(
        distances,
        steps,
        "o-",
        label="Steps to Percolation",
        color=Colors.GOSSIP_SPREADER.value,  # Fixed color reference
        markersize=6,
        markerfacecolor=Colors.CLUELESS_DARK.value,  # Fixed color reference
    )

    plt.xlabel("Distance from Center", fontsize=14)
    plt.ylabel("Steps to Percolation", fontsize=14)
    plt.title("Effect of Distance from Center on Percolation Steps", fontsize=16)
    plt.legend()

    # Grid color matches the 3D plot theme
    plt.grid(True, linestyle="--", linewidth=0.5, color=Colors.UNOCCUPIED.value)

    plt.show()


def plot_distance_vs_percolation_heatmap(results, size):
    """
    Plots the heatmap of average percolation steps against distance from the center.

    Parameters:
        results (dict): Dictionary with keys as initial spreader positions (row, col) and values as steps to percolation.
        size (int): Size of the grid.

    Returns:
        None: Displays a heatmap plot.
    """
    center = (size // 2, size // 2)
    distance_map = {}

    for (row, col), steps in results.items():
        distance = np.sqrt((row - center[0]) ** 2 + (col - center[1]) ** 2)
        if steps != -1:  # Ignore cases where percolation didn't happen
            distance_map.setdefault(distance, []).append(steps)

    avg_steps = {d: np.mean(steps) for d, steps in distance_map.items()}

    distances = list(avg_steps.keys())
    percolation_steps = list(avg_steps.values())

    plt.figure(figsize=(10, 8))

    # Use the pink colormap to align with the 3D plot
    pink_colormap = get_pink_colormap(len(percolation_steps))

    plt.scatter(
        distances,
        percolation_steps,
        c=percolation_steps,
        cmap=pink_colormap,
        edgecolor="black",
        s=80,
    )

    plt.colorbar(label="Average Steps to Percolation")
    plt.xlabel("Distance from Center", fontsize=14)
    plt.ylabel("Average Steps to Percolation", fontsize=14)
    plt.title("Distance from Center vs Average Percolation Steps", fontsize=16)
    plt.grid(True, linestyle="--", linewidth=0.5, color=Colors.UNOCCUPIED.value)

    plt.show()
