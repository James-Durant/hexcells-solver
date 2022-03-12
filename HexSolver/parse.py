import os
import cv2
import time
import json
import pickle
import numpy as np

from grid import Grid, Cell, Constraint

# The resolutions for which screen hashes are pre-computed for.
RESOLUTIONS = [(2560, 1920), (2560, 1600), (2048, 1152), (1920, 1440),
               (1920, 1200), (1920, 1080), (1680, 1050), (1600, 1200)]

# Use the absolute path of this file to define the path to the resources directory.
RESOURCES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resources')

# Path to the Steam executable.
STEAM_PATH = 'C:\\Program Files (x86)\\Steam'


def average_hash(image):
    """Computes a perceptual hash of an image using the average hashing algorithm.

    Args:
        image (numpy.ndarray): image to hash.

    Returns:
        numpy.ndarray: a perceptual hash of the image.
    """
    # For each pixel, compare it to its mean. Flatten to get a 1D array of binary values.
    return (image > image.mean()).flatten()


class Parser:
    """Contains the code common to the MenuParser and LevelParser classes."""

    @staticmethod
    def _load_hashes(dataset_type):
        """Load a dataset of pre-computed perceptual hashes.

        Args:
            dataset_type (str): identifier of the dataset to load.

        Returns:
            tuple: hashes and associated labels loaded from the file.
        """
        # Try to open the file for the dataset.
        file_path = os.path.join(RESOURCES_PATH, dataset_type, 'hashes.pickle')
        try:
            with open(file_path, 'rb') as file:
                return pickle.load(file)

        # Raise an error if the file cannot be found.
        except FileNotFoundError:
            raise RuntimeError(f'The {dataset_type} dataset is missing.')

    @staticmethod
    def _merge_boxes(boxes, x_dist, y_dist):
        """Merge bounding boxes in proximity.

        Args:
            boxes (list): bounding boxes to merge.
            x_dist (float): distance between boxes in the x-axis to merge.
            y_dist (float): distance between boxes in the y-axis to merge.

        Returns:
            list: merged bounding boxes.
        """
        # Sort the bounding boxes by x coordinate in ascending order.
        boxes.sort(key=lambda x: x[0])

        # Keep iterating until all boxes have been considered.
        merged = []
        while boxes:
            # Get a bounding box.
            x1, y1, w1, h1 = boxes.pop()
            x_min, y_min = x1, y1
            x_max, y_max = x1 + w1, y1 + h1

            # Iterate over the other boxes.
            i = 0
            while i < len(boxes):
                x2, y2, w2, h2 = boxes[i]

                # Check if the box is in proximity.
                if abs(x1 + w1 - x2) < x_dist and abs(y1 - y2) < y_dist:
                    # If so, merge the two bounding boxes.
                    x_min = min(x_min, x2)
                    y_min = min(y_min, y2)
                    x_max = max(x_max, x2 + w2)
                    y_max = max(y_max, y2 + h2)
                    del boxes[i]  # Delete the box after merging.

                else:
                    # Otherwise, the box is too far away, so it is skipped.
                    i += 1

            # Record the merged bounding box after considering all others.
            merged.append((x_min, y_min, x_max - x_min, y_max - y_min))

        return merged

    @staticmethod
    def _find_match(hashes, labels, hashed):
        """Find the perceptual hash with minimum hamming distance in a given list of hashes.

        Args:
            hashes (list): hashes to compute the distances over.
            labels (list): label corresponding to each perceptual hash.
            hashed (numpy.ndarray): perceptual hash to compute the distance from.

        Returns:
            str: label of the perceptual hash with minimum Hamming distance.
        """
        # Compute the Hamming distance between each perceptual hash in the list of hashes and the given hash.
        similarities = [np.sum(hashed != h) for h in hashes]
        # Return the label of the perceptual hash with minimum distance.
        return labels[np.argmin(similarities)]


class MenuParser(Parser):
    """Contains the code for automatic parsing of menu screens."""

    # Dimensions to resize screenshots to before applying hashing.
    SCREEN_HASH_DIMS = (480, 270)

    def __init__(self, window, use_level_hashes=True, use_screen_hashes=True):
        """Initialise the menu parser by loading the required screen hashes.

        Args:
            window (window.Window): active game window to parse.
            use_level_hashes (bool, optional): whether to load level selection hashes or not.
            use_screen_hashes (bool, optional): whether to load screen hashes or not.
        """
        self.__window = window

        # Do not load level selection data if it is being created by data.py
        if use_level_hashes:
            self.__level_data = Parser._load_hashes('level_select')

        # Do not load screen data if it is being created by data.py
        if use_screen_hashes:
            # Load the options path for the game currently open.
            options_path = f'steamapps\\common\\{self.__window.title}\\saves\\options.txt'
            options_path = os.path.join(STEAM_PATH, options_path)
            with open(options_path, 'r') as file:
                data = json.load(file)

            # Get the resolution of the game.
            # If a non-standard resolution is being used, default to 1920x1080.
            resolution = data['screenWidth'], data['screenHeight']
            if resolution not in RESOLUTIONS:
                resolution = (1920, 1080)

            # Load the screen hashes specific to the resolution.
            dataset_type = os.path.join('screen', '{0}x{1}'.format(*resolution))
            hashes, labels = Parser._load_hashes(dataset_type)

            # Take out the level exit data as it is handled slightly differently.
            self.__level_exit = hashes.pop(-1)
            labels.pop(-1)
            self.__screen_data = hashes, labels

    def get_screen(self):
        """Identify the menu screen that is currently being displayed.

        Returns:
            str: menu screen being displayed.
        """
        # Take a screenshot of the game window, resize it, and calculate its perceptual hash.
        image = cv2.resize(self.__window.screenshot(), MenuParser.SCREEN_HASH_DIMS, interpolation=cv2.INTER_AREA)
        hashed = average_hash(image)

        # Compute the Hamming distance between each reference hash and the screenshot's hash.
        hashes, labels = self.__screen_data
        similarities = [np.sum(hashed != h) for h in hashes]

        # If the minimum distance is too low, the screenshot is most likely of the level exit screen.
        if min(similarities) > 22000:
            # Threshold the image.
            image = cv2.inRange(image, (180, 180, 180), (255, 255, 255))

            # Compute the number of pixels where the level exit screen and the image differ.
            if np.sum(self.__level_exit != image) > 25000:
                # If large, the screenshot is likely to be that of a level.
                return 'in_level'
            else:
                # Otherwise, the screenshot is a match to the level exit screen.
                return 'level_exit'

        # Return the screen with minimum Hamming distance.
        return labels[np.argmin(similarities)]

    def wait_until_loaded(self):
        """Forces the program to wait until a menu screen has fully loaded."""
        # Add a small fixed delay initially.
        time.sleep(0.75)
        for _ in range(50):
            image = self.__window.screenshot()  # Take a screenshot.

            # Define a mask for orange and blue cells.
            # The third part is a slightly different shade of blue that only appears on the level generator screen.
            mask = cv2.inRange(image, Cell.ORANGE, Cell.ORANGE)
            mask += cv2.inRange(image, Cell.BLUE, Cell.BLUE)
            mask += cv2.inRange(image, (234, 164, 6), (234, 164, 6))

            # Apply the mask to identify contours.
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            time.sleep(0.5)
            if contours:
                # If there are any contours, then the screen has loaded.
                return

        # If this is reached, the screen never loaded for some reason.
        raise RuntimeError('Wait until loaded failed.')

    def parse_main_menu(self):
        """Identify and parse the buttons on the main menu screen.

        Returns:
            dict: coordinates of the centre of each button.
        """
        image = self.__window.screenshot()

        # Set how much of the image should be ignored (from the top).
        cropped = round(0.24 * image.shape[1])

        # Threshold the image on blue and identify contours.
        mask = cv2.inRange(image, (240, 240, 240), (255, 255, 255))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Calculate bounding boxes from the contours and merge boxes in proximity.
        boxes = [cv2.boundingRect(contour) for contour in contours]
        boxes = Parser._merge_boxes(boxes, 100, 100)

        # Filter out any bounding boxes in the top 24% of the image.
        # Use the centre of each button as its coordinates.
        buttons = [(x + w // 2, y + h // 2) for x, y, w, h in boxes if y > cropped]

        # If four buttons were found, separate the level generator button from the save slots.
        if len(buttons) == 4:
            buttons.sort(key=lambda x: tuple(reversed(x)))
            level_generator = buttons.pop()
        else:
            level_generator = None

        # Sort the save slot buttons by x coordinate (i.e., the order is 1, 2, 3).
        slots = sorted(buttons, key=lambda x: x[0])

        # Now thresholded the original image on grey and identify contours.
        mask = cv2.inRange(image, (180, 180, 180), (180, 180, 180))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Calculate bounding boxes from the contours and merge boxes in proximity.
        boxes = [cv2.boundingRect(contour) for contour in contours]
        boxes = Parser._merge_boxes(boxes, image.shape[1], 50)

        # Filter out any bounding boxes in the top 24% of the image.
        # Use the centre of each button as its coordinates.
        buttons = [(x + w // 2, y + h // 2) for x, y, w, h in boxes if y > cropped]

        # If three buttons were found, the game must be Hexcells Infinite which has the user levels button.
        if len(buttons) == 3:
            user_levels, options, menu_exit = buttons

        # Otherwise, the game must be Hexcells or Hexcells Plus where there is no user levels button.
        elif len(buttons) == 2:
            user_levels = None
            options, menu_exit = buttons

        else:
            # Otherwise, a button was missed.
            raise RuntimeError('Main menu parsing failed.')

        # Return the coordinates of the buttons as a dictionary.
        return {'save_slots': slots, 'level_generator': level_generator,
                'user_levels': user_levels, 'options': options, 'exit': menu_exit}

    def parse_generator(self):
        """Identify and parse the buttons on the level generator screen.

        Returns:
            dict: coordinates of the centre of each button.
        """
        image = self.__window.screenshot()

        # Threshold the image on white and identify contours.
        mask = cv2.inRange(image, (240, 240, 240), (255, 255, 255))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Calculate bounding boxes from the contours and sort them by y coordinate in descending order.
        boxes = [cv2.boundingRect(contour) for contour in contours]
        boxes.sort(key=lambda box: box[1], reverse=True)

        # Get the play button (closest to the bottom of the screenshot).
        x, y, w, h = boxes.pop(0)
        play_button = (x + w // 2, y + h // 2)

        # Now sort the boxes by x coordinate to get the random seed button.
        # The random seed button is the furthest to the left in the screenshot.
        boxes.sort(key=lambda box: box[0])
        x, y, w, h = boxes.pop(0)
        random_button = (x + w // 2, y + h // 2)

        # Threshold the original image on blue (the shade unique to this screen).
        mask = cv2.inRange(image, (234, 164, 6), (234, 164, 6))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Calculate bounding boxes from the contours and merge boxes in proximity.
        boxes = [cv2.boundingRect(contour) for contour in contours]
        boxes = Parser._merge_boxes(boxes, image.shape[1], 100)

        # Get the coordinates of the centre of the seed selection button.
        x, y, w, h = boxes[0]
        seed_button = (x + w // 2, y + h // 2)

        # Return the coordinates of the buttons as a dictionary.
        return {'play': play_button, 'random': random_button, 'seed': seed_button}

    def parse_level_selection(self, use_hashes=True):
        """Identify and parse the buttons on the level generator screen.

        Args:
            use_hashes (bool, optional): whether to use pre-computed hashes or not.

        Returns:
            dict: coordinates of the centre of each button.
        """
        image = self.__window.screenshot()

        # Apply a white mask and identify contours.
        mask = cv2.inRange(image, (244, 244, 244), (255, 255, 255))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Calculate the median contour area.
        areas = [cv2.contourArea(contour) for contour in contours]
        median_area = np.median(areas)

        # Filter out boxes that are not within 5% of the median area.
        boxes = [cv2.boundingRect(contour) for contour, area in list(zip(contours, areas))
                 if 0.95 * median_area < area < 1.05 * median_area]
        boxes.sort(key=lambda box: box[:2], reverse=True)

        # Iterate over each bounding box.
        levels = {}
        hashes = []
        for x, y, w, h in boxes:
            # Crop the image to the region within the bounding box.
            x_crop, y_crop = round(w * 0.25), round(h * 0.3)
            cropped = image[y + y_crop:y + h - y_crop, x + x_crop:x + w - x_crop]

            # Threshold the cropped region using a white mask.
            # Invert the result so that the text is black and the background is white.
            thresh = 255 - cv2.inRange(cropped, (240, 240, 240), (255, 255, 255))

            # If there are fewer than 20 black pixels, this is probably not a valid box.
            if np.count_nonzero(thresh == 0) < 20:
                continue

            # Otherwise, crop the image further to filter out unnecessary bordering whitespace.
            coords = np.argwhere(thresh == 0)
            x0, y0 = coords.min(axis=0)
            x1, y1 = coords.max(axis=0) + 1

            # Resize the cropped image and compute its perceptual hash.
            thresh = cv2.resize(thresh[x0:x1, y0:y1], (52, 27), interpolation=cv2.INTER_AREA)
            hashed = average_hash(thresh)

            # Get the label of the reference hash with minimum Hamming distance.
            if use_hashes:
                hashes, labels = self.__level_data
                match = Parser._find_match(hashes, labels, hashed)
                # Record the centre of the button.
                levels[match] = (x + w // 2, y + h // 2)
            else:
                # Otherwise, this is being run by data.py, so record the hash. Do not try to parse it.
                hashes.append(hashed)

        # Return the level button coordinates if parsing.
        # Return the level button hashes if making the reference dataset.
        return levels if use_hashes else hashes

    def parse_level_exit(self):
        """Identify and parse the buttons on the level exit screen.

        Returns:
            list: coordinates of the centre of each button.
        """
        # Take a screenshot, apply a mask to extract the buttons' text, and identify contours.
        image = self.__window.screenshot()
        mask = cv2.inRange(image, (180, 180, 180), (255, 255, 255))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Calculate bounding boxes and merge boxes in proximity.
        boxes = [cv2.boundingRect(contour) for contour in contours]
        boxes = Parser._merge_boxes(boxes, image.shape[1], 100)

        # Calculate the centre of the merged boxes.
        return [(x + w // 2, y + h // 2) for x, y, w, h in boxes]

    def parse_level_completion(self):
        """Identify and parse the buttons on the level completion screen.

        Returns:
            list: coordinates of the centre of each button.
        """
        # Take a screenshot, apply a white mask and identify contours.
        image = self.__window.screenshot()
        mask = cv2.inRange(image, (255, 255, 255), (255, 255, 255))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Calculating bounding boxes and sort them by (y, x) in descending order.
        boxes = [cv2.boundingRect(contour) for contour in contours]
        boxes.sort(key=lambda x: (x[1], x[0]), reverse=True)

        # If six boxes were identified, the "next level" button is present.
        if len(boxes) == 6:
            next_button = (boxes[0][0] + boxes[0][2] // 2, boxes[0][1] + boxes[0][3] // 2)
            menu_button = (boxes[1][0] + boxes[1][2] // 2, boxes[1][1] + boxes[1][3] // 2)
            return next_button, menu_button

        else:
            # Otherwise, this must be the last level in a set, or it is a randomly generated level.
            # In either case, there is no "next level" button.
            menu_button = (boxes[0][0] + boxes[0][2] // 2, boxes[0][1] + boxes[0][3] // 2)
            return None, menu_button


class LevelParser(Parser):
    """Contains the code for automatic parsing of levels."""

    def __init__(self, window, use_cell_counter_hashes=True, use_grid_hashes=True):
        """Load the hashes and reference images used in parsing.

        Args:
            window (window.Window): the active game window.
            use_cell_counter_hashes (bool, optional): whether to load cell and counter hashes or not.
            use_grid_hashes (bool, optional): whether to load grid constraint hashes or not.
        """
        self.__window = window

        # Load the cell and counter datasets if requested.
        if use_cell_counter_hashes:
            self.__black_data = Parser._load_hashes('black')
            self.__blue_data = Parser._load_hashes('blue')
            self.__counter_data = Parser._load_hashes('counter')

        # Load the grid constraint datasets if requested.
        if use_grid_hashes:
            self.__column_data = Parser._load_hashes('column')
            self.__diagonal_data = Parser._load_hashes('diagonal')

        # Load the reference "cell.png" file, threshold it and extract its shape.
        cell_image = cv2.imread(os.path.join(RESOURCES_PATH, 'shapes', 'cell.png'))
        cell_mask = cv2.inRange(cell_image, Cell.ORANGE, Cell.ORANGE)
        cell_contour, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.__cell_shape = cell_contour[0]

        # Load the reference "counter.png" file, threshold it and extract its shape.
        counter_image = cv2.imread(os.path.join(RESOURCES_PATH, 'shapes', 'counter.png'))
        counter_mask = cv2.inRange(counter_image, Cell.BLUE, Cell.BLUE)
        counter_contour, _ = cv2.findContours(counter_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.__counter_shape = counter_contour[0]

    def parse_grid(self, use_hashes=True):
        """Parse the level currently shown by parsing its cells, counter values and grid constraints.

        Args:
            use_hashes (bool, optional): whether to load hashes or not.

        Returns:
            grid.Grid: the parsed representation of the level.
        """
        # Parse the blue, black and orange cells.
        image = self.__window.screenshot()
        blue_cells = self.parse_cells(image, Cell.BLUE)
        black_cells = self.parse_cells(image, Cell.BLACK)
        orange_cells = self.parse_cells(image, Cell.ORANGE)

        # Combine the parsed cells into a single list.
        cells = blue_cells + black_cells + orange_cells

        # Calculate the median cell width and height in the screenshot.
        self.__cell_width = int(np.median([cell.width for cell in cells]))
        self.__cell_height = int(np.median([cell.height for cell in cells]))

        # Get the minimum and maximum x and y image coordinates.
        self.__x_min, self.__x_max = float('inf'), -float('inf')
        self.__y_min, self.__y_max = float('inf'), -float('inf')
        for cell in cells:
            x, y = cell.image_coords
            self.__x_min = min(x, self.__x_min)
            self.__x_max = max(x, self.__x_max)
            self.__y_min = min(y, self.__y_min)
            self.__y_max = max(y, self.__y_max)

        # Get the spacing between cells in the x-axis based on the median cell width.
        if self.__cell_width > 70:
            x_spacing = self.__cell_width * 1.085
        elif 54 <= self.__cell_width <= 70:
            x_spacing = self.__cell_width * 1.098
        else:
            x_spacing = self.__cell_width * 1.105

        # Get the spacing between cells in the y-axis based on the median cell height.
        if self.__cell_height < 65:
            y_spacing = self.__cell_height * 0.72
        else:
            y_spacing = self.__cell_height * 0.69

        # Calculate the number of rows and columns in the grid.
        cols = int(round((self.__x_max - self.__x_min) / x_spacing) + 1)
        rows = int(round((self.__y_max - self.__y_min) / y_spacing) + 1)

        # Define the grid layout using the row and column of each cell.
        layout = [[None] * cols for _ in range(rows)]
        for cell in cells:
            # Map the cell's image coordinates to grid coordinates.
            x, y = cell.image_coords
            col = int(round((x - self.__x_min) / x_spacing))
            row = int(round((y - self.__y_min) / y_spacing))

            # Added the cell to the grid at the calculated index.
            cell.grid_coords = (row, col)
            layout[row][col] = cell

        # Parse the remaining and mistake counters' values.
        _, remaining = self.parse_counters(image)

        # Define a grid using the parsed layout and the number of remaining blue cells.
        grid = Grid(layout, cells, remaining)

        # Parse the level's grid constraints..
        parsed = self.__parse_grid_constraints(image, grid, use_hashes=use_hashes)

        # If this is being run from data.py, return the hashes from column parsing.
        # Otherwise, return the parsed grid.
        return grid if use_hashes else parsed

    def parse_cells(self, image, cell_colour, use_hashes=True):
        """Identify and parse cells of a given colour.

        Args:
            image (numpy.ndarray): screenshot to parse.
            cell_colour (tuple): colour of the cells to identify.
            use_hashes (bool, optional): whether to use hashes or not.

        Returns:
            list: parsed cells of a given colour.
        """
        # Threshold the screenshot on the given cell colour and identify contours.
        mask = cv2.inRange(image, cell_colour, cell_colour)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Iterate over each contour.
        cells = []
        for contour in contours:
            # Check that the contour bounds a large enough area and that its shape matches reference.
            if cv2.contourArea(contour) > 550 and cv2.matchShapes(contour, self.__cell_shape, 1, 0) < 0.08:
                # Calculate a bounding box for the contour.
                x, y, w, h = cv2.boundingRect(contour)

                # Crop the image to the region within the box.
                x_crop, y_crop = round(w * 0.18), round(h * 0.18)
                cropped = image[y + y_crop:y + h - y_crop, x + x_crop:x + w - x_crop]

                # Parse the cropped region.
                parsed = self.__parse_cell_contents(cropped, cell_colour, use_hashes)

                # Create a cell object with the parsed number and hint type.
                if use_hashes:
                    centre = (x + w // 2, y + h // 2)
                    cell = Cell(centre, w, h, cell_colour, parsed)
                    cells.append(cell)
                else:
                    # If this is being run from data.py, and the cell has a number,
                    # record the perceptual hash of the cropped region.
                    if parsed is not None:
                        cells.append(parsed)

        return cells

    def __parse_cell_contents(self, image, cell_colour, use_hashes=True):
        """Parse the contents of a cell.

        Args:
            image (numpy.ndarray): screenshot to parse.
            cell_colour (tuple): colour of the cell to be parsed.
            use_hashes (bool, optional): whether to use hashes or not.

        Returns:
            str: number and hint type of the cell (if present).
        """
        # Orange cells do not have a cell number.
        if cell_colour == Cell.ORANGE:
            return None

        # Convert the image to greyscale and threshold on white.
        thresh = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = np.where(thresh > 220, 0, 255).astype(np.uint8)

        # If the number of black pixels is less than 20, there is no cell number.
        if np.count_nonzero(thresh == 0) < 20:
            return None

        # Pre-process the cell's contents and calculate its perceptual hash.
        thresh = LevelParser.__process_image(thresh)
        hashed = average_hash(thresh)

        # Return the hash if this is being run from data.py
        if not use_hashes:
            return hashed

        # Otherwise, use the appropriate dataset for matching the hash.
        if cell_colour == Cell.BLACK:
            hashes, labels = self.__black_data

        elif cell_colour == Cell.BLUE:
            hashes, labels = self.__blue_data

        else:
            # This should never be raised.
            raise RuntimeError('Invalid cell colour.')

        # Get the label of the reference hash with minimum Hamming distance.
        match = Parser._find_match(hashes, labels, hashed)

        # Extra step for {n} black cells as they can be tricky to parse.
        if cell_colour == Cell.BLACK and match[0] == '{' and match[-1] == '}':
            # Crop the region further to just the number (i.e., {n} -> n)
            temp = thresh.copy()
            temp[:, :15] = 255
            temp[:, -15:] = 255

            # Pre-process the cropped region and calculate its perceptual hash.
            temp = LevelParser.__process_image(temp)
            hashed = average_hash(temp)

            # Parse the number only (ignore the hint type).
            number = Parser._find_match(hashes, labels, hashed)
            match = f'{{{number}}}'  # Add the hint type back.

        return match

    @staticmethod
    def __process_image(image):
        """Pre-process a given image to prepare for perceptual hashing.

        Args:
            image (numpy.ndarray): image to pre-process.

        Returns:
            numpy.ndarray: the pre-processed image.
        """
        # Get the coordinates of the black pixels in the image.
        coords = np.argwhere(image == 0)

        # Crop the image such that the surrounding whitespace is minimised.
        x1, x2 = coords[:, 0].min(), coords[:, 0].max()
        y1, y2 = coords[:, 1].min(), coords[:, 1].max()
        image = image[x1:x2 + 1, y1:y2 + 1]

        # Resize the cropped image and apply a slight blur to smooth edges.
        image = cv2.resize(image, (53, 45), interpolation=cv2.INTER_AREA)
        return cv2.medianBlur(image, 3)

    def parse_counters(self, image=None, use_hashes=True):
        """Identify and parse the values of the remaining and mistakes counters.

        Args:
            image (numpy.ndarray, optional): screenshot to parse the counters from.
            use_hashes (bool, optional): whether to use hashes or not.

        Returns:
            list: parsed counter values.
        """
        # If a screenshot was not given, take one.
        if image is None:
            image = self.__window.screenshot()

        # Threshold the image on blue and identify contours.
        mask = cv2.inRange(image, Cell.BLUE, Cell.BLUE)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Iterate over each contour.
        parsed = []
        for contour in contours:
            # Check that the contour bounds a large enough area and that its shape matches the reference.
            if cv2.contourArea(contour) > 550 and cv2.matchShapes(contour, self.__counter_shape, 1, 0) < 0.1:
                # Calculate a bounding box for the contour.
                x, y, w, h = cv2.boundingRect(contour)

                # Crop the bounding box and get the corresponding region of the screenshot.
                y = round(y + h * 0.35)
                h = round(h * 0.65)
                cropped = image[y:y + h, x:x + w]

                # Threshold the screenshot on blue, convert to greyscale and resize.
                thresh = np.where(cropped == Cell.BLUE, 255, 0).astype(np.uint8)
                thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
                thresh = cv2.resize(thresh, (200, 50), interpolation=cv2.INTER_AREA)

                # Calculate a perceptual hash of the pre-processed region.
                hashed = average_hash(thresh)

                # If using hashes, find the label of the reference hash with minimum Hamming distance.
                if use_hashes:
                    hashes, labels = self.__counter_data
                    match = Parser._find_match(hashes, labels, hashed)
                    parsed.append(match)
                else:
                    # Otherwise, if this is being run from data.py, record the hash and do not parse.
                    parsed.append(hashed)

        if len(parsed) != 2:
            # Check that only two values are parsed (there are only two counters).
            raise RuntimeError(f'Counters parsed incorrectly: {len(parsed)}/2')

        return parsed

    def __parse_grid_constraints(self, image, grid, use_hashes=True):
        """Identify and parse grid constraints.

        Args:
            image (numpy.ndarray): screenshot to parse.
            grid (grid.Grid): parsed grid to add the constraints to.
            use_hashes (bool, optional): whether to use hashes or not.

        Returns:
            list: perceptual hashes of grid constraints (only used by data.py).
        """
        # Threshold the image and identify contours.
        _, thresh = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 120, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Iterate over each contour.
        boxes = []
        for contour in contours:
            # Check that the contour bounds a small enough region.
            if cv2.contourArea(contour) < 1000:
                # Calculate a bounding box for the contour.
                x, y, w, h = cv2.boundingRect(contour)
                coords = np.array([x + w // 2, y + h // 2])

                # Check that the contour is within the playable area (with slight padding) and that
                # it is not too far away from a cell.
                if (self.__x_min - 2 * self.__cell_width < x < self.__x_max + 2 * self.__cell_width
                        and self.__y_min - 2 * self.__cell_height < y < self.__y_max + self.__cell_height
                        and np.linalg.norm(coords - grid.nearest_cell(coords).image_coords) < 130):
                    # If the contour was not filtered out, record it.
                    boxes.append((x, y, w, h))

        # Merge bounding boxes in proximity as contours may separate hint types from numbers.
        boxes = Parser._merge_boxes(boxes, self.__cell_width * 0.76, self.__cell_height * 0.7)

        # Iterate over each merged bounding box.
        parsed = []
        for x, y, w, h in boxes:
            # Get the cell nearest to the centre of the bounding box.
            box_coords = np.array([x + w // 2, y + h // 2])
            nearest_cell = grid.nearest_cell(box_coords)
            nearest_coords = nearest_cell.image_coords

            # Calculate the difference in position between the box centre and the nearest cell.
            delta_x = box_coords[0] - nearest_coords[0]
            delta_y = nearest_coords[1] - box_coords[1]  # The direction is reversed here.

            # Calculate the angle between the box centre and the nearest cell.
            theta = (90 - np.degrees(np.arctan2(delta_y, delta_x))) % 360
            # Get the "true" angle that is closest to the computed angle.
            angle = Constraint.ANGLES[np.argmin(np.abs(Constraint.ANGLES - theta))]

            # Get the region of the image corresponding to the cropped bounding box.
            # Invert the mask from above so that the text is black and the background is white.
            cropped = 255 - thresh[y:y + h, x:x + w]

            # Parse the contents of the cropped region.
            # Number is the parsed constraint number/hint when use_hashes=True.
            # Number is a perceptual hash when use_hashes=False.
            number = self.__parse_grid_constraint_contents(cropped, angle, use_hashes)

            # If the box did not contain anything, exit the method.
            if number is None:
                continue

            # If using hashes, add the parsed constraint to the grid.
            if use_hashes:
                row, col = nearest_cell.grid_coords
                grid.add_constraint(row, col, number, angle)

            # Record the perceptual hash of the grid constraint for use in data.py
            parsed.append(number)

        return parsed

    def __parse_grid_constraint_contents(self, thresh, angle, use_hashes=True):
        """Parse the contents of a grid constraint.

        Args:
            thresh (image): thresholded screenshot containing the contents to parse.
            angle (float): angle of orientation of the constraint.
            use_hashes (bool, optional): whether to use hashes or not.

        Returns:
            str: parsed number and hint type for the constraint.
        """
        # If there are fewer than 20 black pixels, this is not a valid region.
        if np.count_nonzero(thresh == 0) < 20:
            return None

        # Pre-process the given region of the screenshot.
        thresh = LevelParser.__process_image(thresh)

        # For these angles, flip the image twice (in the y-axis then the x-axis) as the diagonal
        #  dataset (i.e., the data for 60 and 300) can be used.
        if angle in [120, 240]:
            thresh = cv2.flip(cv2.flip(thresh, 1), 0)

        # Rotate to 0 so that the column dataset can be used.
        elif angle in [90, 270]:
            thresh = LevelParser.__rotate(thresh, angle)

        # Calculate the perceptual hash of the pre-processed region.
        hashed = average_hash(thresh)

        # Return the perceptual hash if this is being run from data.py
        if not use_hashes:
            return hashed

        # Select the appropriate dataset of reference hashes.
        if angle in [0, 90, 270, 360]:
            hashes, labels = self.__column_data
        else:
            hashes, labels = self.__diagonal_data

        # Find the label of the reference hash with minimum Hamming distance.
        match = Parser._find_match(hashes, labels, hashed)

        # {3}, {5} and {8} constraints are tricky to parse as they look very similar.
        # If one of these was identified, ignore the hint type and parse the number again.
        if match[0] == '{' and match[-1] == '}' and match[1] in ['3', '5', '8']:
            temp = thresh.copy()

            # If the constraint is at an angle, rotate it to be upright.
            if angle not in [0, 360]:
                temp = LevelParser.__rotate(temp, angle)

            # Crop the image to just the constraint number.
            temp[:, :15] = 255
            temp[:, -15:] = 255

            # Pre-process the cropped image.
            temp = LevelParser.__process_image(temp)

            # Use the column dataset (since the constraint is upright) to find the matching number.
            hashes, labels = self.__column_data
            match = Parser._find_match(hashes, labels, average_hash(temp))
            match = f'{{{match}}}'  # Add the hint type back.

        return match

    @staticmethod
    def __rotate(image, angle):
        """Rotate an image by a given angle, using the image's centre as the point of rotation.

        Args:
            image (numpy.ndarray): image to rotate.
            angle (float): angle to rotate by.

        Returns:
            numpy.ndarray: the rotated image.
        """
        # Get the centre of the image.
        centre = tuple(np.array(image.shape[1::-1]) / 2)

        # Get the rotation matrix for the given angle.
        rotation_matrix = cv2.getRotationMatrix2D(centre, angle, 1.0)

        # Rotate the image by the angle. If the rotated image is not of the same size as before,
        # pad the blank areas with whitespace.
        return cv2.warpAffine(image, rotation_matrix, image.shape[1::-1],
                              flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))

    def parse_clicked(self, grid, left_click_cells, right_click_cells, delay=False):
        """Left and right click cells and parse their uncovered contents.

        Args:
            grid (grid.Grid): grid containing the cells to parse.
            left_click_cells (list): cells to left click and parse.
            right_click_cells (list): cells to right click and parse.
            delay (bool, optional): whether to use a delay after clicking cells.

        Returns:
            list: new values of the remaining and mistakes counters.
        """
        image = None
        # If there are any cells to left click, click them.
        if left_click_cells:
            self.click_cells(left_click_cells, 'left')

            # Take a screenshot of the resulting level state and parse it.
            time.sleep(0.1)
            image = self.__window.screenshot()
            self.__parse_clicked_cells(image, left_click_cells)

        # If there are any cells to right click, click them.
        if right_click_cells:
            self.click_cells(right_click_cells, 'right')

            # If the particle effect has not been disabled, add a delay based on the number of rows.
            if delay:
                # Get the cell to right click that is highest in the level.
                min_row = min(cell.grid_coords[0] for cell in right_click_cells)
                # Wait based on how far the particles have to fall.
                time.sleep(max(1.5, 0.15 * (grid.rows - min_row)))
            else:
                time.sleep(0.15)

            # Take a screenshot of the resulting level state and parse it.
            image = self.__window.screenshot()
            self.__parse_clicked_cells(image, right_click_cells)

        # If a screenshot was not taken, take one.
        if image is None:
            image = self.__window.screenshot()

        # Parse the new counter values.
        return self.parse_counters(image)

    def click_cells(self, cells, button):
        """Click, but do not parse, each cell in a given list.

        Args:
            cells (list): cells to click.
            button (str): either left or right.
        """
        # Sort the cells by (row, col) grid coordinates in descending order.
        cells.sort(key=lambda c: c.grid_coords, reverse=True)

        # Click each cell with the given mouse button action.
        for cell in cells:
            self.__window.click_cell(cell, button)

    def __parse_clicked_cells(self, image, cells):
        """Parse the contents of each cell in a given list, given that they have been correctly uncovered.

        Args:
            image (numpy.ndarray): screenshot to parse.
            cells (list): cells to parse the new contents of.
        """
        # Iterate over each cell to parse.
        for cell in cells:
            # Get the centre of the cell in terms of screenshot coordinates.
            cx, cy = cell.image_coords
            w, h = self.__cell_width, self.__cell_height

            # Calculate a bounding box for the cell.
            x1, x2 = cx - w // 2, cx + w // 2
            y1, y2 = cy - h // 2, cy + h // 2

            # Get the region of the image corresponding to the cropped bounding box.
            x_crop, y_crop = round(w * 0.18), round(h * 0.18)
            cropped = image[y1 + y_crop:y2 - y_crop, x1 + x_crop:x2 - x_crop]

            # Determine if the cell is blue or black after being uncovered.
            if np.count_nonzero(cropped == Cell.BLACK) > 10:
                cell.colour = Cell.BLACK
            elif np.count_nonzero(cropped == Cell.BLUE) > 10:
                cell.colour = Cell.BLUE
            else:
                # This should never be reached as a cell can only be revealed as blue or black.
                raise RuntimeError('Cell must be blue or black after being revealed')

            # Parse the revealed cell number and hint type.
            cell.number = self.__parse_cell_contents(cropped, cell.colour)
