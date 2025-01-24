import copy
import os
import random
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
from matplotlib import animation
from matplotlib.animation import PillowWriter
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle


class Status(Enum):
    """
    Enum representing possible statuses for a cell
    """

    UNOCCUPIED = 0
    CLUELESS = 1
    SECRET_KEEPER = 2
    GOSSIP_SPREADER = 3


class Cell:
    """
    Represents a cell in a grid, with a status and spreading probability.

    Attributes:
        status (Status): The current status of the cell (e.g. UNOCCUPIED, CLUELESS)
        spreading_prob (float): The probability of becoming a gossip spreader, between 0 and 1

    Methods:
        get_status(): Returns the current status of the cell
        set_status(status): Sets the status of the cell to the provided value
        is_gossip_spreader(): Returns True if the cell's status is 'GOSSIP_SPREADER', False otherwise
        is_clueless(): Returns True if the cell's status is 'CLUELESS', False otherwise
        get_spreading_prob(): Returns the current spreading probability of the cell
        set_spreading_prob(spreading_prob): Sets the spreading probability of the cell
    """

    def __init__(self, status, spreading_prob=0):
        """
        Initializes a new Cell instance with a status and spreading probability.

        Parameters:
            status (Status): The initial status of the cell. It must be a valid status from the Status enum.
            spreading_prob (float, optional): The probability of the cell becoming a gossip spreader.
                                              Must be between 0 and 1 (inclusive). Defaults to 0.
        """
        assert isinstance(
            status, Status
        ), f"status must be an instance of Status enum, got {type(status)}"
        assert isinstance(
            spreading_prob, (int, float)
        ), f"spreading_prob must be a number, got {type(spreading_prob)}"
        assert (
            0 <= spreading_prob <= 1
        ), f"spreading_prob must be between 0 and 1 (inclusive), got {spreading_prob}"

        self.status = status
        self.spreading_prob = spreading_prob

    def get_status(self):
        """
        Returns the current status of the cell.

        Returns:
            Status: The current status of the cell.
        """
        return self.status

    def set_status(self, status):
        """
        Sets the status of the cell.

        Parameters:
            status (Status): The new status to set for the cell. It must be a valid status from the Status enum.
        """
        assert isinstance(
            status, Status
        ), f"status must be an instance of Status enum, got {type(status)}"

        self.status = status

    def is_gossip_spreader(self):
        """
        Checks if the cell is a gossip spreader.

        Returns:
            bool: True if the cell's status is 'GOSSIP_SPREADER', False otherwise.
        """
        return self.status == Status.GOSSIP_SPREADER

    def is_clueless(self):
        """
        Checks if the cell is clueless.

        Returns:
            bool: True if the cell's status is 'CLUELESS', False otherwise.
        """
        return self.status == Status.CLUELESS

    def get_spreading_prob(self):
        """
        Returns the current spreading probability of the cell.

        Returns:
            float: The current spreading probability.
        """
        return self.spreading_prob

    def set_spreading_prob(self, spreading_prob):
        """
        Sets the spreading probability of the cell, ensuring it is between 0 and 1.

        Parameters:
            spreading_prob (float): The new spreading probability to set for the cell. Must be between 0 and 1 (inclusive).
        """
        assert isinstance(
            spreading_prob, (int, float)
        ), f"spreading_prob must be a number, got {type(spreading_prob)}"
        assert (
            0 <= spreading_prob <= 1
        ), f"spreading_prob must be between 0 and 1 (inclusive), got {spreading_prob}"

        self.spreading_prob = spreading_prob

    def __eq__(self, other):
        """
        Check if two cells are equal based on their status and spreading probability.

        Parameters:
            other (Cell): Another Cell instance to compare with.

        Returns:
            bool: True if the two cells are equal, False otherwise.
        """
        if not isinstance(other, Cell):
            return NotImplemented

        return (
            self.status == other.status and self.spreading_prob == other.spreading_prob
        )


class Grid:
    """
    Represents a square grid of cells of given size, occupation density and spreading threshold.

    Attributes:
        size (int): The size of the grid (number of rows and columns). The grid will have size*size cells in total.
        lecture_hall (list of lists of Cell): The square grid of cells.
        density (float): The fraction of cells that are initially occupied (range between 0 and 1).
        spread_threshold (float): The threshold probability for a clueless cell to become a gossip spreader.

    Methods:
        initialize_board(): Initializes the grid with unoccupied cells and randomly fills some cells as clueless based on the density.
        set_spreader(i, j): Sets the status of a cell at position (i, j) to 'GOSSIP_SPREADER'.
        get_neighbours(i, j): Returns the list of neighboring cells (top, bottom, left, right) of the cell at (i, j).
        update_grid(): Updates the grid by iterating through each cell, changing statuses based on neighboring gossip spreaders and the cell's spreading probability.
        show_grid(): Displays the current state of the grid using `matplotlib`.
        run_simulation(steps): Runs the simulation for a given number of steps.
        save_grid(iteration, save_path): Saves the current grid into a given filepath including the iteration number in the title.
        generate_gif(steps, gif_name): Runs the simulation for given number of steps and saves it as a gif.
        check_percolation(): Checks whether percolation happens in both directions and returns either True or False.
    """

    def __init__(self, size, density, spread_threshold):
        """
        Initializes the Grid with the specified size, density, and spreading threshold.

        Parameters:
            size (int): The size of the grid (number of rows and columns).
            density (float): The fraction of cells to be initially filled as clueless. Must be between 0 and 1 (inclusive).
            spread_threshold (float): The probability threshold for a clueless cell to become a gossip spreader. Must be between 0 and 1 (inclusive).
        """
        assert isinstance(size, int), f"size must be an integer, got {type(size)}"
        assert size > 0, f"size must be greater than 0, got {size}"
        assert isinstance(
            density, (int, float)
        ), f"density must be a number, got {type(density)}"
        assert (
            0 <= density <= 1
        ), f"density must be between 0 and 1 (inclusive), got {density}"
        assert isinstance(
            spread_threshold, (int, float)
        ), f"spread_threshold must be a number, got {type(spread_threshold)}"
        assert (
            0 <= spread_threshold <= 1
        ), f"spread_threshold must be between 0 and 1 (inclusive), got {spread_threshold}"

        self.size = size
        self.lecture_hall = []
        self.density = density
        self.spread_threshold = spread_threshold

    def set_initial_spreader(self, flag_center):
        """Sets the initial spreader in the lecture hall grid.

        Parameters:
            flag_center (int):
            - If 1, the initial spreader is placed in the central subgrid of the lecture hall
              (approximately a square of size self.size/2 x self.size/2).
            - If 0, the initial spreader is placed near the edges of the lecture hall, outside the central region.
        """
        assert flag_center in [
            0,
            1,
        ], f"flag_center must be either 0 or 1, got {flag_center}"

        if flag_center == 1:
            # Set the spreader in the central 10x10 subgrid
            start = self.size // 4  # Start index for the 10x10 subgrid
            end = 3 * (self.size // 4)  # End index for the 10x10 subgrid
            initial_spreader_i = random.randint(start, end - 1)
            initial_spreader_j = random.randint(start, end - 1)
        else:
            # Set the spreader outside the central subgrid
            # Randomly choose a row/column outside of the central block
            initial_spreader_i = random.choice([0, self.size - 1])
            initial_spreader_j = random.choice([0, self.size - 1])

            # Ensure the spreader is outside the 10x10 region
            while (self.size // 4 <= initial_spreader_i < 3 * (self.size // 4)) and (
                self.size // 4 <= initial_spreader_j < 3 * (self.size // 4)
            ):
                initial_spreader_i = random.choice([0, self.size - 1])
                initial_spreader_j = random.choice([0, self.size - 1])

        self.lecture_hall[initial_spreader_i][initial_spreader_j].set_status(
            Status.GOSSIP_SPREADER
        )

    def initialize_board(self, flag_center=1):
        """
        Initializes the grid as fully unoccupied, and randomly selects some cells to be filled as clueless.

        The number of clueless cells is determined by the `density` attribute, which indicates the fraction of cells to be filled.
        The spreading probability for each clueless cell is also randomly assigned between 0 and 1.

        Parameters:
            flag_center (int):
            - If 1, the initial spreader is placed in the central subgrid of the lecture hall
              (approximately a square of size self.size/2 x self.size/2).
            - If 0, the initial spreader is placed near the edges of the lecture hall, outside the central region.
        """
        assert flag_center in [
            0,
            1,
        ], f"flag_center must be either 0 or 1, got {flag_center}"

        self.lecture_hall = [
            [Cell(Status.UNOCCUPIED) for _ in range(self.size)]
            for _ in range(self.size)
        ]

        # Pick randomly the cells to be initially occupied given occupation density
        total_cells = self.size * self.size
        cells_to_fill = int(total_cells * self.density)
        all_cells = [(i, j) for i in range(self.size) for j in range(self.size)]

        selected_cells = random.sample(all_cells, cells_to_fill)

        # Set the status of selected cells to clueless and assign spreading probability
        for i, j in selected_cells:
            self.lecture_hall[i][j].set_status(Status.CLUELESS)
            self.lecture_hall[i][j].set_spreading_prob(np.random.uniform(0, 1))

        # set initial spot, flag=1 for center, 0 for outside
        self.set_initial_spreader(flag_center)

    def set_spreader(self, i, j):
        """
        Sets the status of the cell at position (i, j) to 'GOSSIP_SPREADER'.

        Parameters:
            i (int): The row index of the cell.
            j (int): The column index of the cell.
        """
        assert isinstance(i, int), f"i must be an integer, got {type(i)}"
        assert 0 <= i < self.size, f"i must be between 0 and {self.size - 1}, got {i}"
        assert isinstance(j, int), f"j must be an integer, got {type(j)}"
        assert 0 <= j < self.size, f"j must be between 0 and {self.size - 1}, got {j}"

        self.lecture_hall[i][j].set_status(Status.GOSSIP_SPREADER)

    def get_neighbours(self, i, j, flag_neighbors=1):
        """
        Returns the list of neighbors for the cell at position (i, j).

        The neighbors include the top, bottom, left, and right cells, as long as the cell is not on the edge of the grid.
        If a neighboring cell is out of bounds, it will not be included.

        Parameters:
            i (int): The row index of the cell.
            j (int): The column index of the cell.
            flag_neighbors (int): The flag to determine the type of neighbors to include. If 1, implement Moore neighborhood, if 0, implement Von Neumann neighborhood.

        Returns:
            list of Cell: The list of neighboring cells (top, bottom, left, right).
        """
        assert isinstance(i, int), f"i must be an integer, got {type(i)}"
        assert 0 <= i < self.size, f"i must be between 0 and {self.size - 1}, got {i}"
        assert isinstance(j, int), f"j must be an integer, got {type(j)}"
        assert 0 <= j < self.size, f"j must be between 0 and {self.size - 1}, got {j}"

        neighbours = []

        # Top neighbour
        if i > 0:
            neighbours.append(self.lecture_hall[i - 1][j])
            #upper diagonal neighbors
            if flag_neighbors == 1:
                if j > 0:
                    neighbours.append(self.lecture_hall[i - 1][j - 1])
                if j < self.size - 1:
                    neighbours.append(self.lecture_hall[i - 1][j + 1])

        # Bottom neighbour
        if i < self.size - 1:
            neighbours.append(self.lecture_hall[i + 1][j])
            #bottom diagonal neighbors
            if flag_neighbors == 1:
                if j > 0:
                    neighbours.append(self.lecture_hall[i + 1][j - 1])
                if j < self.size - 1:
                    neighbours.append(self.lecture_hall[i + 1][j + 1])

        # Left neighbour
        if j > 0:
            neighbours.append(self.lecture_hall[i][j - 1])

        # Right neighbour
        if j < self.size - 1:
            neighbours.append(self.lecture_hall[i][j + 1])

        return neighbours

    def update_grid(self):
        """
        Updates the grid by iterating through each cell and changing their status based on neighboring gossip spreaders.

        For each clueless cell, the method checks if any of its neighbors are gossip spreaders. If a neighboring cell is a gossip spreader,
        the current cell will update its status. If its spreading probability is greater than the defined `spread_threshold`,
        it will become a gossip spreader; otherwise, it will become a secret keeper.

        The grid is updated by making a deep copy of the current state, ensuring that changes do not affect the current iteration and all
        cells are changed at the same time.
        """
        new_lecture_hall = copy.deepcopy(self.lecture_hall)

        # Iterate through all the cells and update their status if applicable
        for i in range(self.size):
            for j in range(self.size):
                current_cell = self.lecture_hall[i][j]

                # We only update the status if the cell is clueless
                if not current_cell.is_clueless():
                    continue

                # Get neighbours and check if at least one of them is a gossip_spreader
                neighbours = self.get_neighbours(i, j)

                neighbour_statuses = [
                    neighbour.is_gossip_spreader() for neighbour in neighbours
                ]

                if True in neighbour_statuses:
                    # Update the status of the cell depending on the spreading probability of the cell and threshold
                    s = current_cell.get_spreading_prob()

                    if s > self.spread_threshold:
                        new_lecture_hall[i][j].set_status(Status.GOSSIP_SPREADER)

                    else:
                        new_lecture_hall[i][j].set_status(Status.SECRET_KEEPER)

        self.lecture_hall = new_lecture_hall

    def show_grid(self, iteration=None):
        """
        Displays the current state of the grid using `matplotlib`.

        Each cell's status is represented by an integer value corresponding to its status in the `Status` enum.
        A color map is used to visualize the different statuses of the cells.

        Parameters:
            iteration (int, optional): The current iteration number, used for titles. Defaults to None.
        """
        grid = [[cell.get_status().value for cell in row] for row in self.lecture_hall]

        center_start, center_end = self.size // 4, 3 * self.size // 4
        for i in range(center_start, center_end):
            for j in range(center_start, center_end):
                if grid[i][j] != Status.UNOCCUPIED.value:
                    grid[i][j] += 0.5

        plt.imshow(grid, cmap="coolwarm", interpolation="none")
        plt.colorbar(label="Status")
        if iteration is not None:
            plt.title(f"Iteration {iteration}")
        plt.show()

    def run_simulation(self, steps=1000):
        """
        Runs the simulation for a specified number of steps, updating the grid at each iteration.
        Stops iteration if no cell status changes for 3 consecutive steps.

        Parameters:
            steps (int): The number of steps to simulate.

        Returns:
            (list, dict): A tuple containing a list of 2D grids, where each grid represents the state of the lecture hall
                  at a given time step, including the initial state and a dictionary containing the counts of each status over iterations.
        """
        assert isinstance(steps, int), f"steps must be an integer, got {type(steps)}"
        assert steps > 0, f"steps must be greater than 0, got {steps}"

        all_grids = [self.lecture_hall]

        same = 0
        prev_grid = None

        status_counts = {
            "UNOCCUPIED": [],
            "CLUELESS": [],
            "SECRET_KEEPER": [],
            "GOSSIP_SPREADER": [],
        }

        for step in range(steps):
            new_counts = {
                "UNOCCUPIED": 0,
                "CLUELESS": 0,
                "SECRET_KEEPER": 0,
                "GOSSIP_SPREADER": 0,
            }
            current_grid = copy.deepcopy(self.lecture_hall)

            # check if grid is the same as the previous grid
            if prev_grid is not None and prev_grid == current_grid:
                same += 1
            else:
                same = 0

            # stop simulation if no cell status changed for 3 consecutive steps
            if same == 3:
                break

            # count statuses
            for i in range(self.size):
                for j in range(self.size):
                    cell_status = self.lecture_hall[i][j].get_status()
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

            # update the grid
            self.update_grid()

            prev_grid = current_grid
            all_grids.append(self.lecture_hall)

        return all_grids, status_counts

    def save_grid(self, iteration=None, save_path=None):
        """
        Saves the current state of the grid as an image file.

        Each cell's status is represented by an integer value corresponding to its status in the `Status` enum.
        A color map is used to visualize the different statuses of the cells.

        Parameters:
            iteration (int, optional): The current iteration number, used for file names or titles. Defaults to None.
            save_path (str): Path to save the current grid visualization. Defaults to None.
        """
        grid = [[cell.get_status().value for cell in row] for row in self.lecture_hall]

        center_start, center_end = self.size // 4, 3 * self.size // 4
        for i in range(center_start, center_end):
            for j in range(center_start, center_end):
                if grid[i][j] != Status.UNOCCUPIED.value:
                    grid[i][j] += 0.5

        plt.imshow(grid, cmap="coolwarm", interpolation="none")
        plt.colorbar(label="Status")
        if iteration is not None:
            plt.title(f"Iteration {iteration}")
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.close()

    def generate_gif(self, steps=10, gif_name="spread_simulation.gif"):
        """
        Generates a GIF animation of the grid's spread simulation, saving all frames and the GIF to an external 'images' folder.

        Parameters:
            steps (int): Number of steps to simulate.
            gif_name (str): Name of the output GIF file.
        """
        current_dir = os.getcwd()
        parent_dir = os.path.dirname(current_dir)
        images_dir = os.path.join(parent_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        frames = []

        for step in range(steps):
            fig_path = os.path.join(images_dir, f"frame_{step}.png")
            self.save_grid(iteration=step, save_path=fig_path)
            frames.append(fig_path)
            self.update_grid()

        gif_path = os.path.join(images_dir, gif_name)
        writer = PillowWriter(fps=2)
        fig = plt.figure()

        writer.setup(fig, gif_path, dpi=100)
        for frame in frames:
            img = plt.imread(frame)
            plt.imshow(img)
            plt.axis("off")
            writer.grab_frame()

        writer.finish()

        print(f"GIF saved to: {gif_path}")
        print(f"All frames saved to: {images_dir}")

    def dfs(self, visited, i, j, target_row=None, target_col=None):
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
            bool: True if we have reached the target row or column (i.e., percolation has occurred), False otherwise.
        """
        # check if the current cell is within bounds, is not visited, and is occupied
        if (
            i < 0
            or i >= self.size
            or j < 0
            or j >= self.size
            or visited[i][j]
            or self.lecture_hall[i][j].get_status() != Status.GOSSIP_SPREADER
        ):
            return False

        # mark this cell as visited
        visited[i][j] = True

        # if we've reached the target row or column, return True (percolation occurred)
        if (target_row is not None and i == target_row) or (
            target_col is not None and j == target_col
        ):
            return True

        # explore all four possible neighbors: down, up, right, left
        return (
            self.dfs(visited, i + 1, j, target_row, target_col)
            or self.dfs(visited, i - 1, j, target_row, target_col)
            or self.dfs(visited, i, j + 1, target_row, target_col)
            or self.dfs(visited, i, j - 1, target_row, target_col)
        )

    def check_percolation_direction(self, direction):
        """
        Checks if percolation occurs in the grid in a given direction (vertical or horizontal).

        Parameters:
            direction (str): The direction to check for percolation. Can be either 'vertical' or 'horizontal'.

        Returns:
            bool: True if percolation occurs in the specified direction, False otherwise.

        Raises:
            ValueError: If the direction is not 'vertical' or 'horizontal'.
        """
        if direction == "vertical":
            visited = [[False for _ in range(self.size)] for _ in range(self.size)]

            # start DFS from any occupied cell in the top row (row 0)
            for j in range(self.size):
                if (
                    (self.lecture_hall[0][j].get_status() == Status.GOSSIP_SPREADER or self.lecture_hall[0][j].get_status() == Status.SECRET_KEEPER)
                    and not visited[0][j]
                ):
                    if self.dfs(
                        visited, 0, j, target_row=self.size - 1
                    ):  # Target row is the last row
                        return True
            return False
        elif direction == "horizontal":
            visited = [[False for _ in range(self.size)] for _ in range(self.size)]

            # start DFS from any occupied cell in the leftmost column (column 0)
            for i in range(self.size):
                if (
                    (self.lecture_hall[i][0].get_status() == Status.GOSSIP_SPREADER or self.lecture_hall[i][0].get_status() == Status.SECRET_KEEPER)
                    and not visited[i][0]
                ):
                    if self.dfs(
                        visited, i, 0, target_col=self.size - 1
                    ):  # target column is the last column
                        return True
            return False
        else:
            raise ValueError("direction must be either 'vertical' or 'horizontal'")

    def check_percolation(self):
        """
        Checks if percolation occurs in the grid by checking for vertical and horizontal percolation.

        Returns:
            bool: True if both vertical and horizontal percolation occurs, False otherwise.
        """
        return self.check_percolation_direction(
            "vertical"
        ) and self.check_percolation_direction("horizontal")


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
    init_grid = [[cell.get_status().value for cell in row] for row in cell_grids[0]]

    fig = plt.figure()
    colors = ["lightgray", "white", "gold", "goldenrod"]
    cmap = ListedColormap(colors)

    im = plt.imshow(init_grid, cmap=cmap, interpolation="none", animated=True)

    # Add a square specifying the central 10x10 area
    ax = plt.gca()
    left_corner_coordinate = (len(init_grid[0]) // 4) - 0.5
    square = Rectangle(
        (left_corner_coordinate, left_corner_coordinate),
        11,
        11,
        edgecolor="dimgray",
        facecolor="none",
        linewidth=2,
    )
    ax.add_patch(square)

    # Add legend to the plot
    legend_patches = [
        Patch(
            facecolor=color, edgecolor="black", label=f"State {i}"
        )  # TODO: use the names of states instead of numbers
        for i, color in enumerate(colors)
    ]

    plt.legend(
        handles=legend_patches,
        title="States",
        loc="upper right",
        bbox_to_anchor=(1.2, 1),
    )

    # Get the state number for each cell
    grids = [
        [[cell.get_status().value for cell in row] for row in cell_grid]
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

        return cluster_size

    cluster_sizes = []

    for grid in grids:
        size = grid.size
        visited = [[False for _ in range(size)] for _ in range(size)]

        # Iterate through the grid to find clusters
        for i in range(size):
            for j in range(size):
                if not visited[i][j] and (grid.lecture_hall[i][j].get_status() == Status.GOSSIP_SPREADER or grid.lecture_hall[i][j].get_status() == Status.SECRET_KEEPER):
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
    
    #run multiple simulations
    for _ in range(num_simulations):
        # create and initialize the grid
        grid = Grid(size=grid_size, density=density, spread_threshold=spread_threshold)
        grid.initialize_board(flag_center)

        # run the simulation
        grid.run_simulation(steps=steps)
        if grid.check_percolation():
            count_percolation += 1
        #calculate the cluster size distribution for the current grid
        cluster_distribution = calculate_cluster_size_distribution([grid])
        cluster_distributions.append(cluster_distribution)


    print(f"Percolation occured in {count_percolation} out of {num_simulations} simulations for density={density}, spread_threshold={spread_threshold}")
    #return the list of cluster distributions from all simulations
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
    grid_size, densities, spread_thresholds, num_simulations, flag_center=1
):
    """
    Runs multiple simulations for all combinations of densities and spread thresholds,
    aggregates the results, and plots the log-log distributions.

    Parameters:
        simulation_function (function): A function that runs the gossip model simulation and returns a cluster size distribution.
        grid_size (int): The size of the grid for the simulations.
        densities (list): A list of densities to simulate.
        spread_thresholds (list): A list of spread thresholds to simulate.
        num_simulations (int): The number of simulations to run for each set of initial conditions.
        flag_center (int): The flag to determine the initial spreader placement.
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
