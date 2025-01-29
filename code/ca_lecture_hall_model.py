import copy
import os
import random
from enum import Enum

import matplotlib.pyplot as plt
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


class Colors(Enum):
    """
    Enum representing the colors used for visualisations
    """

    UNOCCUPIED = LIGHTGRAY = "#D3D3D3"
    CLUELESS = WHITESMOKE = "#F5F5F5"
    CLUELESS_DARK = DIMGRAY = "#696969"
    SECRET_KEEPER = LIGHT_PINK = "#FFB6C1"
    GOSSIP_SPREADER = DARK_PINK = "#B03060"


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

    def __init__(self, status):
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

        self.status = status

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

        return self.status == other.status


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
        save_grid(iteration, save_path): Saves the current grid into a given filepath including the iteration number in the title.
        generate_gif(steps, gif_name): Runs the simulation for given number of steps and saves it as a gif.
        check_percolation(): Checks whether percolation happens in both directions and returns either True or False.
    """

    def __init__(self, size, density, bond_probability):
        """
        Initializes the Grid with the specified size, density, and bond probability.

        Parameters:
            size (int): The size of the grid (number of rows and columns).
            density (float): The fraction of cells to be initially filled as clueless. Must be between 0 and 1 (inclusive).
            bond_probability (float): TODO
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
            bond_probability, (int, float)
        ), f"bond_probability must be a number, got {type(bond_probability)}"
        assert (
            0 <= bond_probability <= 1
        ), f"bond_probability must be between 0 and 1 (inclusive), got {bond_probability}"

        self.size = size
        self.lecture_hall = []
        self.bonds = {}
        self.density = density
        self.bond_probability = bond_probability

    def set_initial_spreader(self, flag_center=1, flag_neighbors=0):
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
            initial_spreader_i = random.randint(0, self.size - 1)
            initial_spreader_j = random.randint(0, self.size - 1)

            # Ensure the spreader is outside the 10x10 region
            while (self.size // 4 <= initial_spreader_i < 3 * (self.size // 4)) and (
                self.size // 4 <= initial_spreader_j < 3 * (self.size // 4)
            ):
                initial_spreader_i = random.randint(0, self.size - 1)
                initial_spreader_j = random.randint(0, self.size - 1)

        self.lecture_hall[initial_spreader_i][initial_spreader_j].set_status(
            Status.GOSSIP_SPREADER
        )

        # Add the bonds
        if (
            self.lecture_hall[initial_spreader_i][initial_spreader_j].status
            == Status.UNOCCUPIED
        ):
            neighbours = self.get_neighbours(
                initial_spreader_i, initial_spreader_j, flag_neighbors
            )

            for neighbour in neighbours:
                m, n = neighbour[0], neighbour[1]
                if self.lecture_hall[m][n].status == Status.CLUELESS:
                    if (
                        (m, n),
                        (initial_spreader_i, initial_spreader_j),
                    ) in self.bonds.keys():
                        self.bonds[
                            ((initial_spreader_i, initial_spreader_j), (m, n))
                        ] = self.bonds[
                            ((m, n), (initial_spreader_i, initial_spreader_j))
                        ]
                    else:
                        self.bonds[
                            ((initial_spreader_i, initial_spreader_j), (m, n))
                        ] = random.choices(
                            [0, 1],
                            weights=[1 - self.bond_probability, self.bond_probability],
                        )[0]

    def initialize_board(self, flag_center=1, flag_neighbors=0):
        """
        Initializes the grid as fully unoccupied, and randomly selects some cells to be filled as clueless.

        The number of clueless cells is determined by the `density` attribute, which indicates the fraction of cells to be filled.
        The spreading probability for each clueless cell is also randomly assigned between 0 and 1.

        Parameters:
            flag_center (int):
            - If 1, the initial spreader is placed in the central subgrid of the lecture hall
              (approximately a square of size self.size/2 x self.size/2).
            - If 0, the initial spreader is placed near the edges of the lecture hall, outside the central region.
            TODO
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

        for i, j in selected_cells:
            # get all the neighbours
            neighbours = self.get_neighbours(i, j, flag_neighbors)

            for neighbour in neighbours:
                m, n = neighbour[0], neighbour[1]
                if self.lecture_hall[m][n].status == Status.CLUELESS:
                    if ((m, n), (i, j)) in self.bonds.keys():
                        self.bonds[((i, j), (m, n))] = self.bonds[((m, n), (i, j))]
                    else:
                        self.bonds[((i, j), (m, n))] = random.choices(
                            [0, 1],
                            weights=[1 - self.bond_probability, self.bond_probability],
                        )[0]

        # set initial spot, flag=1 for center, 0 for outside
        self.set_initial_spreader(flag_center, flag_neighbors)

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

    def get_neighbours(self, i, j, flag_neighbors=0):
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
            neighbours.append((i - 1, j))
            # upper diagonal neighbors
            if flag_neighbors == 1:
                if j > 0:
                    neighbours.append((i - 1, j - 1))
                if j < self.size - 1:
                    neighbours.append((i - 1, j + 1))

        # Bottom neighbour
        if i < self.size - 1:
            neighbours.append((i + 1, j))
            # bottom diagonal neighbors
            if flag_neighbors == 1:
                if j > 0:
                    neighbours.append((i + 1, j - 1))
                if j < self.size - 1:
                    neighbours.append((i + 1, j + 1))

        # Left neighbour
        if j > 0:
            neighbours.append((i, j - 1))

        # Right neighbour
        if j < self.size - 1:
            neighbours.append((i, j + 1))

        return neighbours

    def update_grid(self, flag_neighbors=0):
        """
        Updates the grid by iterating through each cell and changing their status based on neighboring gossip spreaders.

        For each clueless cell, the method checks if any of its neighbors are gossip spreaders. If a neighboring cell is a gossip spreader,
        the current cell will update its status. If its spreading probability is greater than the defined `spread_threshold`,
        it will become a gossip spreader; otherwise, it will become a secret keeper.

        The grid is updated by making a deep copy of the current state, ensuring that changes do not affect the current iteration and all
        cells are changed at the same time.
        paremeters:
            flag_neighbors (int): The flag to determine the type of neighbors to include. If 1, implement Moore neighborhood, if 0, implement Von Neumann neighborhood
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
                neighbours = self.get_neighbours(i, j, flag_neighbors)

                for neighbour in neighbours:
                    m, n = neighbour[0], neighbour[1]

                    if self.lecture_hall[m][n].is_gossip_spreader():
                        if self.bonds[((i, j), (m, n))]:
                            new_lecture_hall[i][j].set_status(Status.GOSSIP_SPREADER)
                            break

        self.lecture_hall = new_lecture_hall

    def add_central_square(self):
        ax = plt.gca()
        left_corner_coordinate = (self.size // 4) - 0.5
        square = Rectangle(
            (left_corner_coordinate, left_corner_coordinate),
            self.size // 2,
            self.size // 2,
            edgecolor="dimgray",
            facecolor="none",
            linewidth=2,
        )
        ax.add_patch(square)

    def add_outer_box(self):
        ax = plt.gca()
        square = Rectangle(
            (-0.5, -0.5),
            self.size,
            self.size,
            edgecolor="black",
            facecolor="none",
            linewidth=3,
        )
        ax.add_patch(square)
        plt.axis("off")

    def add_legend(self, colors):
        legend_patches = [
            Patch(
                facecolor=color, edgecolor="black", label=f"{state}"
            )  # TODO: use the names of states instead of numbers
            for state, color in zip(
                ["UNOCCUPIED", "CLUELESS", "GOSSIP SPREADER"], colors
            )
        ]

        plt.legend(
            handles=legend_patches,
            loc="upper right",
            bbox_to_anchor=(1.3, 1),
        )

    def show_grid(self, iteration=None):
        """
        Displays the current state of the grid using `matplotlib`.

        Each cell's status is represented by an integer value corresponding to its status in the `Status` enum.
        A color map is used to visualize the different statuses of the cells.

        Parameters:
            iteration (int, optional): The current iteration number, used for titles. Defaults to None.
        """
        grid = [[cell.get_status().value for cell in row] for row in self.lecture_hall]

        # Define colormap
        colors = [
            Colors.UNOCCUPIED.value,
            Colors.CLUELESS.value,
            Colors.GOSSIP_SPREADER.value,
        ]
        cmap = ListedColormap(colors)

        plt.imshow(grid, cmap=cmap, interpolation="none")

        # Add a square specifying the central area, an outer border and legend
        self.add_central_square()
        self.add_outer_box()
        self.add_legend(colors)

        if iteration is not None:
            plt.title(f"Iteration {iteration}")

        plt.show()

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

        # Define colormap
        colors = [
            Colors.UNOCCUPIED.value,
            Colors.CLUELESS.value,
            Colors.GOSSIP_SPREADER.value,
        ]
        cmap = ListedColormap(colors)

        plt.imshow(grid, cmap=cmap, interpolation="none")

        # Add a square specifying the central area, an outer border and legend
        self.add_central_square()
        self.add_outer_box()
        self.add_legend(colors)

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

        for step in range(steps + 1):
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

    def check_percolation(self):
        percolation_vertical = 0
        percolation_horizonatl = 0

        # First row
        first_row = [cell.status for cell in self.lecture_hall[0]]
        if Status.GOSSIP_SPREADER in first_row:
            percolation_vertical += 1

        # Last row
        last_row = [cell.status for cell in self.lecture_hall[-1]]
        if Status.GOSSIP_SPREADER in last_row:
            percolation_vertical += 1

        # First column
        first_col = [self.lecture_hall[row][0].status for row in range(self.size)]
        if Status.GOSSIP_SPREADER in first_col:
            percolation_horizonatl += 1

        # Last column
        last_col = [self.lecture_hall[row][-1].status for row in range(self.size)]
        if Status.GOSSIP_SPREADER in last_col:
            percolation_horizonatl += 1

        return percolation_vertical == 2 or percolation_horizonatl == 2
