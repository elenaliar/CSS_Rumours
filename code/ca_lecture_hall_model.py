from enum import Enum
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import random
import os


class Status(Enum):
    """
    Enum representing possible statuses for a cell
    """

    UNOCCUPIED = 0
    CLUELESS = 1
    GOSSIP_SPREADER = 2
    SECRET_KEEPER = 3


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
            spreading_prob (float, optional): The probability of the cell becoming a gossip spreader. Defaults to 0.

        Raises:
            ValueError: If the status is not an instance of the Status enum.
        """
        if not isinstance(status, Status):
            raise ValueError(
                f"status must be an instance of Status enum, got {type(status)}"
            )

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

        Raises:
            ValueError: If the provided status is not an instance of the Status enum.
        """
        if not isinstance(status, Status):
            raise ValueError(
                f"status must be an instance of Status enum, got {type(status)}"
            )

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
            spreading_prob (float): The new spreading probability to set for the cell.

        Raises:
            ValueError: If spreading_prob is not between 0 and 1 (inclusive).
        """
        if not (0 <= spreading_prob <= 1):
            raise ValueError(
                f"spreading_prob must be between 0 and 1, got {spreading_prob}"
            )

        self.spreading_prob = spreading_prob


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
    """

    def __init__(self, size, density, spread_threshold):
        """
        Initializes the Grid with the specified size, density, and spreading threshold.

        Parameters:
            size (int): The size of the grid (number of rows and columns).
            density (float): The fraction of cells to be initially filled as clueless (range between 0 and 1).
            spread_threshold (float): The probability threshold for a clueless cell to become a gossip spreader.

        Raises:
            ValueError: If density or spread_threshold is not between 0 and 1 (inclusive).
        """
        if not (0 <= density <= 1):
            raise ValueError(f"density must be between 0 and 1, got {density}")
        if not (0 <= spread_threshold <= 1):
            raise ValueError(
                f"spread_threshold must be between 0 and 1, got {spread_threshold}"
            )

        self.size = size
        self.lecture_hall = []
        self.density = density
        self.spread_threshold = spread_threshold

    def initialize_board(self):
        """
        Initializes the grid as fully unoccupied, and randomly selects some cells to be filled as clueless.

        The number of clueless cells is determined by the `density` attribute, which indicates the fraction of cells to be filled.
        The spreading probability for each clueless cell is also randomly assigned between 0 and 1.
        """
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

        # randomly set the initial spot
        spreader_position = random.choice(selected_cells)
        self.set_spreader(*spreader_position)

    def set_spreader(self, i, j):
        """
        Sets the status of the cell at position (i, j) to 'GOSSIP_SPREADER'.

        Parameters:
            i (int): The row index of the cell.
            j (int): The column index of the cell.
        """
        self.lecture_hall[i][j].set_status(Status.GOSSIP_SPREADER)

    def get_neighbours(self, i, j):
        """
        Returns the list of neighbors for the cell at position (i, j).

        The neighbors include the top, bottom, left, and right cells, as long as the cell is not on the edge of the grid.
        If a neighboring cell is out of bounds, it will not be included.

        Parameters:
            i (int): The row index of the cell.
            j (int): The column index of the cell.

        Returns:
            list of Cell: The list of neighboring cells (top, bottom, left, right).
        """
        neighbours = []

        # Top neighbour
        if i > 0:
            neighbours.append(self.lecture_hall[i - 1][j])

        # Bottom neighbour
        if i < self.size - 1:
            neighbours.append(self.lecture_hall[i + 1][j])

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

    def show_grid(self, iteration=None, save_path=None):
        """
        Displays the current state of the grid using `matplotlib`.

        Each cell's status is represented by an integer value corresponding to its status in the `Status` enum.
        A color map is used to visualize the different statuses of the cells.

        Parameters:
            iteration (int, optional): The current iteration number, used for titles. Defaults to None.
            save_path (str, optional): Path to save the current grid visualization. Defaults to None.
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


def generate_gif(grid, steps=10, gif_name="spread_simulation.gif"):
    """
    Generates a GIF animation of the grid's spread simulation, saving all frames and the GIF to an external 'images' folder.

    Parameters:
        grid (Grid): An instance of the Grid class.
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
        grid.show_grid(iteration=step, save_path=fig_path)
        frames.append(fig_path)
        grid.update_grid()

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
