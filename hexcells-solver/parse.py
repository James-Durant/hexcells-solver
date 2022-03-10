import os
import cv2
import time
import json
import pickle
import numpy as np

from grid import Grid, Cell

# The resolutions for which there are screen hashes for.
RESOLUTIONS = [(2560, 1920), (2560, 1600), (2048, 1152), (1920, 1440),
               (1920, 1200), (1920, 1080), (1680, 1050), (1600, 1200)]

# Use the absolute path of this file.
RESOURCES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'resources')

# Path to the Steam executable.
STEAM_PATH = 'C:\Program Files (x86)\Steam'


def average_hash(image):
    """Computes a perceptual hash of a given image using the average hashing algorithm.

    Args:
        image (numpy.ndarray): the image to hash.

    Returns:
        numpy.ndarray: a perceptual hash of the image.
    """
    # For each pixel, compare it to its mean.
    # Flatten to get a 1D array of binary values.
    return (image > image.mean()).flatten()


class Parser:
    """The parent class for the MenuParser and LevelParser classes."""

    @staticmethod
    def _load_hashes(dataset_type):
        """Loads a dataset of pre-computed hashes.

        Args:
            dataset_type (str): the dataset type to load.

        Returns:
            tuple: the hashes and associated labels loaded from the file.
        """
        # Try to open the file for the dataset.
        file_path = os.path.join(RESOURCES_PATH, dataset_type, 'hashes.pickle')
        try:
            with open(file_path, 'rb') as file:
                return pickle.load(file)

        # Raise an error if the file cannot be found.
        except FileNotFoundError:
            raise RuntimeError(f'Hashes missing for {dataset_type}')

    @staticmethod
    def _merge_boxes(boxes, x_dist, y_dist):
        """Merge bounding boxes in close proximity.

        Args:
            boxes (list): bounding boxes to be merged.
            x_dist (float): distance between boxes in the x-axis to be merge.
            y_dist (float): distance between boxes in the y-axis to be merge.

        Returns:
            list: the merged bounding boxes.
        """
        # Sort the bounding boxes by x coordinate in ascending order.
        boxes.sort(key=lambda x: x[0])

        # Keep iterating until all boxes have been merged.
        merged = []
        while boxes:
            # Get one of the boxes.
            x1, y1, w1, h1 = boxes.pop()
            x_min, y_min = x1, y1
            x_max, y_max = x1 + w1, y1 + h1

            # Iterate over the other boxes.
            i = 0
            while i < len(boxes):
                x2, y2, w2, h2 = boxes[i]

                # Check if the box is in close proximity.
                if (abs(x1 + w1 - x2) < x_dist and abs(y1 - y2) < y_dist):
                    # If so, merge the bounding boxes.
                    x_min = min(x_min, x2)
                    y_min = min(y_min, y2)
                    x_max = max(x_max, x2 + w2)
                    y_max = max(y_max, y2 + h2)
                    del boxes[i] # Delete the box after merging.

                else:
                    # Otherwise, the box is too far away so it is skipped.
                    i += 1

            # Record the merged bounding box after considering all others.
            merged.append((x_min, y_min, x_max - x_min, y_max - y_min))

        return merged

    @staticmethod
    def _find_match(hashes, labels, hashed):
        """Find the hash with minimum hamming distance in a given list of hashes.

        Args:
            hashes (list): hashes to compute the distances over.
            labels (list): label corresponding to each hash.
            hashed (numpy.ndarray): hash to compute the distance with.

        Returns:
            str: label of the hash with minimum Hamming distance.
        """
        # Compute the Hamming distance between each hash in the list of hashes and the given hash.
        similarities = [np.sum(hashed != h) for h in hashes]
        # Return the label of the hash with minimum distance.
        return labels[np.argmin(similarities)]


class MenuParser(Parser):
    """Contains the code for automatic parsing of menu screens."""

    # Dimensions to resize screenshots to before applying hashing.
    SCREEN_HASH_DIMS = (480, 270)

    def __init__(self, window, use_hashes=True):
        """Initialises the menu parser by loading the required screen hashes.

        Args:
            window (_type_): _description_
            use_hashes (bool, optional): _description_. Defaults to True.
        """
        self.__window = window

        # Level selection data is created first in data.py
        self.__level_data = Parser._load_hashes('level_select')

        # If generating the level selection hashes, do not try to load the hashes for the other screens.
        if use_hashes:
            self.__load_hashes()

    def __load_hashes(self):
        """Load the screen hashes based on the game window's resolution."""
        # Load the options path for the game currently open.
        options_path = f'steamapps\\common\\{self.__window.title}\\saves\\options.txt'
        options_path = os.path.join(STEAM_PATH, options_path)
        with open(options_path, 'r') as file:
            data = json.load(file)

        # Get the resolution of the game.
        # If a non-standard resolution is being used, default to 1920x1080
        resolution = data['screenWidth'], data['screenHeight']
        if resolution not in RESOLUTIONS:
            resolution = (1920, 1080)

        # Load the screen hashes specific to the game window.
        dataset_type = os.path.join('screen', '{0}x{1}'.format(*resolution))
        hashes, labels = Parser._load_hashes(dataset_type)
        
        # Take out the level exit data as it is handled slightly differently.
        self.__level_exit = hashes.pop(-1)
        labels.pop(-1)
        self.__screen_data = hashes, labels

    def get_screen(self):
        """Identify which menu screen is currently being displayed.

        Returns:
            str: the menu screen being displayed.
        """
        # Take a screenshot of the game window, resize it, and calculate its hash.
        image = cv2.resize(self.__window.screenshot(), MenuParser.SCREEN_HASH_DIMS, interpolation=cv2.INTER_AREA)
        hashed = average_hash(image)

        # Compute the Hamming distance between each reference hash and the screenshot's hash.
        hashes, labels = self.__screen_data
        similarities = [np.sum(hashed != h) for h in hashes]

        # If the minimum similarity is too low, the screen is most likely the level exit screen.
        if min(similarities) > 22000:
            # Threshold the image.
            image = cv2.inRange(image, (180, 180, 180), (255, 255, 255))

            # Compute the number of pixels where the level exit screen and image differ.
            if np.sum(self.__level_exit != image) > 25000:
                # If large, the screen is likely to be in a level.
                return 'in_level'
            else:
                # Otherwise, the image is a match and it is the level exit screen.
                return 'level_exit'

        # Otherwise, return the screen with minimum Hamming distance.
        return labels[np.argmin(similarities)]

    def wait_until_loaded(self):
        """Forces the program to wait until a menu screen has fully loaded."""
        # Add a small fixed delay initially.
        time.sleep(0.75)
        for _ in range(50):
            image = self.__window.screenshot() # Take a screenshot.

            # Define a mask for orange and blue cells. 
            # The third part is a slighly different shade of blue that appears once  on the level generator screen.
            mask = cv2.inRange(image, Cell.ORANGE, Cell.ORANGE)
            mask += cv2.inRange(image, Cell.BLUE, Cell.BLUE)
            mask += cv2.inRange(image, (234, 164, 6), (234, 164, 6))

            # Apply the mask to identify contours.
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # If there are any contours, then the level had almost loaded.
            time.sleep(0.5)
            if contours:
                return

        # If this is reached, the screen never loaded for some reason.
        raise RuntimeError('Wait until loaded failed.')

    def parse_main_menu(self):
        """Detect and parse the buttons on the main menu screen.

        Returns:
            dict: the coordinates of the centre of each button.
        """
        image = self.__window.screenshot()

        # Set how much of the image should be ignored (from the top).
        cropped = round(0.24 * image.shape[1])

        # Threshold the image on blue and identify contours.
        mask = cv2.inRange(image, (240, 240, 240), (255, 255, 255))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Calculate bounding boxes from the contours and merge boxes in close proximity.
        boxes = [cv2.boundingRect(contour) for contour in contours]
        boxes = Parser._merge_boxes(boxes, 100, 100)

        # Filter out any bounding boxes in the top 24% of the image.
        # Use the centre of each button as its coordinates.
        buttons = [(x + w // 2, y + h // 2) for x, y, w, h in boxes if y > cropped]

        # If 4 buttons were found, extract the level generator button from the save slots.
        if len(buttons) == 4:
            buttons.sort(key=lambda x: tuple(reversed(x)))
            level_generator = buttons.pop()
        else:
            level_generator = None

        # Sort the save slot buttons by x coordinate (i.e., order is 1, 2 and 3).
        slots = sorted(buttons, key=lambda x: x[0])

        # Now thresholded the original image on grey and identify contours.
        mask = cv2.inRange(image, (180, 180, 180), (180, 180, 180))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Calculate bounding boxes from the contours and merge boxes in close proximity.
        boxes = [cv2.boundingRect(contour) for contour in contours]
        boxes = Parser._merge_boxes(boxes, image.shape[1], 50)

        # Filter out any bounding boxes in the top 24% of the image.
        # Use the centre of each button as its coordinates.
        buttons = [(x + w // 2, y + h // 2) for x, y, w, h in boxes if y > cropped]

        # If 3 buttons were found, the game must be Hexcells Infinite which has the user levels button.
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
        return {'save_slots': slots,
                'level_generator': level_generator,
                'user_levels': user_levels,
                'options': options,
                'exit': menu_exit}

    def parse_generator(self):
        """Detect and parse the buttons on the level generator screen.

        Returns:
            dict: the coordinates of the centre of each button.
        """
        image = self.__window.screenshot()
        
        # Threshold the image on white and identify contours.
        mask = cv2.inRange(image, (240, 240, 240), (255, 255, 255))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Calculate bounding boxes from the contours and sort them by y coordinate (descending).
        boxes = [cv2.boundingRect(contour) for contour in contours]
        boxes.sort(key=lambda box: box[1], reverse=True)

        # Get the play button which is the closest to the bottom in the screenshot.
        x, y, w, h = boxes.pop(0)
        play_button = (x + w // 2, y + h // 2)

        # Now sort the boxes by x coordinate to get get the random seed button.
        # The random seed button is the furthest to the left in the screenshot.
        boxes.sort(key=lambda box: box[0])
        x, y, w, h = boxes.pop(0)
        random_button = (x + w // 2, y + h // 2)

        # Threshold the original image on blue (the shade unique to this screen).
        mask = cv2.inRange(image, (234, 164, 6), (234, 164, 6))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Calculate bounding boxes from the contours and merge boxes in close proximity.
        boxes = [cv2.boundingRect(contour) for contour in contours]
        boxes = Parser._merge_boxes(boxes, image.shape[1], 100)

        # Get the coordinates of the seed selection button.
        x, y, w, h = boxes[0]
        seed_button = (x + w // 2, y + h // 2)

        # Return the coordinates of the buttons as a dictionary.
        return {'play': play_button,
                'random': random_button,
                'seed': seed_button}

    def parse_level_selection(self, use_hashes=True):
        """Detect and parse the buttons on the level generator screen.

        Args:
            use_hashes (bool, optional): whether use pre-computed hashes or not.

        Returns:
            dict: the coordinates of the centre of each level's button.
        """
        image = self.__window.screenshot()

        # Apply a white mask and identify contours.
        mask = cv2.inRange(image, (244, 244, 244), (255, 255, 255))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Calculate median contour area.
        areas = [cv2.contourArea(contour) for contour in contours]
        median_area = np.median(areas)

        # Filter out boxes that are not within 5% of the median contour area.
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

            # Threshold the cropped region using a white mask to extract the text.
            # Then invert the result so that the text is black and the background is white.
            thresh = 255 - cv2.inRange(cropped, (240, 240, 240), (255, 255, 255))
            
            # If there are fewer than 20 black pixels, this is probably not a valid box.
            if np.count_nonzero(thresh == 0) < 20:
                continue
            
            # Otherwise, crop the image further to filter out any unecessary bordering whitespace.
            coords = np.argwhere(thresh == 0)
            x0, y0 = coords.min(axis=0)
            x1, y1 = coords.max(axis=0) + 1

            # Resize the cropped image and compute its hash.
            thresh = cv2.resize(thresh[x0:x1, y0:y1], (52, 27), interpolation=cv2.INTER_AREA)
            hashed = average_hash(thresh)

            # If this is being run by data.py, record the hash. Do not try to parse it.
            if not use_hashes:
                hashes.append(hashed)
            else:
                # Otherwise, get the label of the reference hash with minimum Hamming distance.
                hashes, labels = self.__level_data
                match = Parser._find_match(hashes, labels, hashed)
                # Record the centre of the button for the parsed level.
                levels[match] = (x + w // 2, y + h // 2)

        # Return the level button coordinates if parsing.
        # Return the level button hashes if obtaining the reference dataset.
        return levels if use_hashes else hashes

    def parse_level_exit(self):
        """Detect and parse the buttons on the level exit screen.

        Returns:
            list: the coordinates of the centre of each button.
        """
        # Take a screenshot, apply a mask to extract the buttons' text and identify contours.
        image = self.__window.screenshot()
        mask = cv2.inRange(image, (180, 180, 180), (255, 255, 255))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Calculate bounding boxes and merge boxes in close proximity.
        boxes = [cv2.boundingRect(contour) for contour in contours]
        boxes = Parser._merge_boxes(boxes, image.shape[1], 100)

        # Calculate the centre of the merged boxes.
        return [(x + w // 2, y + h // 2) for x, y, w, h in boxes]

    def parse_level_completion(self):
        """Detect and parse the buttons on the level completion screen.

        Returns:
            list: the coordinates of the centre of each button.
        """
        # Take a screenshot, apply a white mask and identify contours.
        image = self.__window.screenshot()
        mask = cv2.inRange(image, (255, 255, 255), (255, 255, 255))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Calculating bounding boxes and sort them by (y, x) in descending order.
        boxes = [cv2.boundingRect(contour) for contour in contours]
        boxes.sort(key=lambda x: (x[1], x[0]), reverse=True)

        # If 6 boxes were identified, the "next level" button is present.
        if len(boxes) == 6:
            next_button = (boxes[0][0] + boxes[0][2] // 2, boxes[0][1] + boxes[0][3] // 2)
            menu_button = (boxes[1][0] + boxes[1][2] // 2, boxes[1][1] + boxes[1][3] // 2)
            return next_button, menu_button

        else:
            # Otherwise, this must be the last level in the set or a randomly generated level
            # in which case there is no "next level" button.
            menu_button = (boxes[0][0] + boxes[0][2] // 2, boxes[0][1] + boxes[0][3] // 2)
            return None, menu_button


class LevelParser(Parser):
    """Contains the code for automatic parsing of levels."""

    __hex_match_threshold = 0.08
    __counter_match_threshold = 0.1
    __area_threshold = 550
    __angles = np.asarray([0, 60, 90, 120, 240, 270, 300, 360])
    __number_dims = (45, 30)
    __counter_dims = (200, 50)

    def __init__(self, window):
        self.__window = window

        self.__black_data = Parser._load_hashes('black')
        self.__blue_data = Parser._load_hashes('blue')
        self.__counter_data = Parser._load_hashes('counter')
        self.__column_data = Parser._load_hashes('column')
        self.__diagonal_data = Parser._load_hashes('diagonal')

        hex_image = cv2.imread(os.path.join(RESOURCES_PATH, 'cell.png'))
        hex_mask = cv2.inRange(hex_image, Cell.ORANGE, Cell.ORANGE)
        hex_contour, _ = cv2.findContours(hex_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        counter_image = cv2.imread(os.path.join(RESOURCES_PATH, 'counter.png'))
        counter_mask = cv2.inRange(counter_image, Cell.BLUE, Cell.BLUE)
        counter_contour, _ = cv2.findContours(counter_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.__hex_contour = hex_contour[0]
        self.__counter_contour = counter_contour[0]

        self.__x_min, self.__x_max = float('inf'), -float('inf')
        self.__y_min, self.__y_max = float('inf'), -float('inf')
        self.__hex_width = float('inf')
        self.__hex_height = float('inf')

    def parse_grid(self, use_hashes=True):
        image = self.__window.screenshot()

        blue_cells = self.parse_cells(image, Cell.BLUE)
        black_cells = self.parse_cells(image, Cell.BLACK)
        orange_cells = self.parse_cells(image, Cell.ORANGE)

        cells = blue_cells + black_cells + orange_cells

        widths = [cell.width for cell in cells]
        heights = [cell.height for cell in cells]
        xs = [cell.image_coords[0] for cell in cells]
        ys = [cell.image_coords[1] for cell in cells]

        self.__x_min, self.__x_max = np.min(xs), np.max(xs)
        self.__y_min, self.__y_max = np.min(ys), np.max(ys)
        self.__hex_width = int(np.median(widths))
        self.__hex_height = int(np.median(heights))

        if self.__hex_width > 70:
            x_spacing = self.__hex_width * 1.085
        elif 54 <= self.__hex_width <= 70:
            x_spacing = self.__hex_width * 1.098
        else:
            x_spacing = self.__hex_width * 1.105

        # 67 - 0.69
        # 43 - 0.72
        if self.__hex_height < 65:
            y_spacing = self.__hex_height * 0.72
        else:
            y_spacing = self.__hex_height * 0.69

        cols = int(round((self.__x_max - self.__x_min) / x_spacing) + 1)
        rows = int(round((self.__y_max - self.__y_min) / y_spacing) + 1)

        # print(self.__hex_width, (self.__x_max - self.__x_min) / x_spacing)
        # print(self.__hex_height, (self.__y_max - self.__y_min) / y_spacing)

        layout = [[None] * cols for _ in range(rows)]
        for cell in cells:
            x, y = cell.image_coords

            col = int(round((x - self.__x_min) / x_spacing))
            row = int(round((y - self.__y_min) / y_spacing))
            cell.grid_coords = (row, col)
            layout[row][col] = cell

        _, remaining = self.parse_counters(image)

        grid = Grid(layout, cells, remaining)
        parsed = self.__parse_columns(image, grid, use_hashes=use_hashes)

        return grid if use_hashes else parsed

    def parse_cells(self, image, cell_colour, use_hashes=True):
        mask = cv2.inRange(image, cell_colour, cell_colour)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cells = []
        for contour in contours:
            if (cv2.contourArea(contour) > LevelParser.__area_threshold and
                    cv2.matchShapes(contour, self.__hex_contour, 1, 0) < LevelParser.__hex_match_threshold):
                x, y, w, h = cv2.boundingRect(contour)

                x_crop, y_crop = round(w * 0.18), round(h * 0.18)
                cropped = image[y + y_crop:y + h - y_crop, x + x_crop:x + w - x_crop]

                parsed = self.__parse_cell_number(cropped, cell_colour, use_hashes)

                if use_hashes:
                    centre = (x + w // 2, y + h // 2)
                    cell = Cell(centre, w, h, cell_colour, parsed)
                    cells.append(cell)
                else:
                    if parsed is not None:
                        cells.append(parsed)

        return cells

    def __parse_cell_number(self, image, cell_colour, use_hashes=True):
        if cell_colour == Cell.ORANGE:
            return None

        thresh = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = np.where(thresh > 220, 0, 255).astype(np.uint8)

        if np.count_nonzero(thresh == 0) < 20:
            return None

        thresh = LevelParser.__process_image(thresh)
        hashed = average_hash(thresh)

        if not use_hashes:
            return hashed

        if cell_colour == Cell.BLACK:
            hashes, labels = self.__black_data

        elif cell_colour == Cell.BLUE:
            hashes, labels = self.__blue_data

        else:
            raise RuntimeError('invalid cell colour')

        match = Parser._find_match(hashes, labels, hashed)

        if cell_colour == Cell.BLACK and match[0] == '{' and match[-1] == '}':
            temp = thresh.copy()
            temp[:, :15] = 255
            temp[:, -15:] = 255

            temp = LevelParser.__process_image(temp)
            number = Parser._find_match(hashes, labels, average_hash(temp))
            match = '{' + number + '}'

        return match

    def parse_counters(self, image=None, use_hashes=True):
        if image is None:
            image = self.__window.screenshot()

        mask = cv2.inRange(image, Cell.BLUE, Cell.BLUE)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        parsed = []
        for contour in contours:
            if (cv2.contourArea(contour) > LevelParser.__area_threshold and
                    cv2.matchShapes(contour, self.__counter_contour, 1, 0) < LevelParser.__counter_match_threshold):

                x, y, w, h = cv2.boundingRect(contour)
                y = round(y + h * 0.35)
                h = round(h * 0.65)

                cropped = image[y:y + h, x:x + w]
                thresh = cv2.cvtColor(np.where(cropped == Cell.BLUE, 255, 0).astype(np.uint8), cv2.COLOR_BGR2GRAY)
                thresh = cv2.resize(thresh, LevelParser.__counter_dims, interpolation=cv2.INTER_AREA)

                hashed = average_hash(thresh)

                if not use_hashes:
                    parsed.append(hashed)

                else:
                    hashes, labels = self.__counter_data
                    similarities = [np.sum(hashed != h) for h in hashes]
                    match = labels[np.argmin(similarities)]
                    parsed.append(match)

        if len(parsed) != 2:
            raise RuntimeError('counters parsed incorrectly: {}/2'.format(len(parsed)))

        return parsed

    def __parse_columns(self, image, grid, use_hashes=True):
        _, thresh = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 120, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for contour in contours:
            if cv2.contourArea(contour) < 1000:
                x, y, w, h = cv2.boundingRect(contour)
                coords = np.asarray([x + w // 2, y + h // 2])
                if (self.__x_min - 2 * self.__hex_width < x < self.__x_max + 2 * self.__hex_width and
                        self.__y_min - 2 * self.__hex_height < y < self.__y_max + self.__hex_height and
                        np.linalg.norm(coords - grid.nearest_cell(coords).image_coords) < 130):

                    boxes.append((x, y, w, h))

        bounding_boxes = Parser._merge_boxes(boxes, self.__hex_width * 0.76, self.__hex_height * 0.7)

        parsed = []
        for x, y, w, h in bounding_boxes:
            box_coords = np.asarray([x + w // 2, y + h // 2])
            nearest_cell = grid.nearest_cell(box_coords)
            nearest_coords = nearest_cell.image_coords

            delta_x = box_coords[0] - nearest_coords[0]
            delta_y = nearest_coords[1] - box_coords[1]
            theta = (90 - np.degrees(np.arctan2(delta_y, delta_x))) % 360
            angle = LevelParser.__angles[np.argmin(np.abs(LevelParser.__angles - theta))]

            cropped = 255 - thresh[y:y + h, x:x + w]
            number = self.__parse_grid_number(cropped, angle, use_hashes)

            if use_hashes:
                row, col = nearest_cell.grid_coords
                grid.add_constraint(row, col, number, angle)

            parsed.append(number)

        return parsed

    @staticmethod
    def __rotate(image, angle):
        centre = tuple(np.array(image.shape[1::-1]) / 2)
        rotation_matrix = cv2.getRotationMatrix2D(centre, angle, 1.0)
        return cv2.warpAffine(image, rotation_matrix, image.shape[1::-1], flags=cv2.INTER_LINEAR,
                              borderValue=(255, 255, 255))

    def __parse_grid_number(self, thresh, angle, use_hashes=True):
        if np.count_nonzero(thresh == 0) < 20:
            return None

        thresh = LevelParser.__process_image(thresh)

        if angle in [120, 240]:
            thresh = cv2.flip(cv2.flip(thresh, 1), 0)

        elif angle in [90, 270]:
            thresh = LevelParser.__rotate(thresh, angle)

        hashed = average_hash(thresh)

        if not use_hashes:
            return hashed

        if angle in [0, 90, 270, 360]:
            hashes, labels = self.__column_data
        else:
            hashes, labels = self.__diagonal_data

        match = Parser._find_match(hashes, labels, hashed)

        if match[0] == '{' and match[-1] == '}' and match[1] in ['3', '5', '8']:
            temp = thresh.copy()

            if angle in [0, 360]:
                temp[:, :15] = 255
                temp[:, -15:] = 255
            else:
                temp = LevelParser.__rotate(temp, angle)

            temp = LevelParser.__process_image(temp)

            hashes, labels = self.__column_data
            match = Parser._find_match(hashes, labels, average_hash(temp))

            if angle in [0, 360]:
                match = '{' + match + '}'

        return match

    def parse_clicked(self, grid, left_click_cells, right_click_cells, delay=False):
        image = None
        if left_click_cells:
            self.click_cells(left_click_cells, 'left')

            time.sleep(0.1)
            image = self.__window.screenshot()
            self.__parse_clicked_cells(image, left_click_cells)

        if right_click_cells:
            self.click_cells(right_click_cells, 'right')

            if delay:
                min_row = min(right_click_cells, key=lambda cell: cell.grid_coords[0]).grid_coords[0]
                time.sleep(max(1.5, 0.15*(grid.rows-min_row)))
            else:
                time.sleep(0.1)

            image = self.__window.screenshot()
            self.__parse_clicked_cells(image, right_click_cells)

        if image is None:
            image = self.__window.screenshot()

        return self.parse_counters(image)

    def click_cells(self, cells, button):
        cells.sort(key=lambda c: c.grid_coords, reverse=True)
        for cell in cells:
            self.__window.click_cell(cell, button)

    def __parse_clicked_cells(self, image, cells):
        for cell in cells:
            cx, cy = cell.image_coords
            w, h = self.__hex_width, self.__hex_height

            x1, x2 = cx - w // 2, cx + w // 2
            y1, y2 = cy - h // 2, cy + h // 2

            x_crop, y_crop = round(w * 0.18), round(h * 0.18)
            cropped = image[y1 + y_crop:y2 - y_crop, x1 + x_crop:x2 - x_crop]

            if np.count_nonzero(cropped == Cell.BLACK) > 10:
                cell.colour = Cell.BLACK
            elif np.count_nonzero(cropped == Cell.BLUE) > 10:
                cell.colour = Cell.BLUE
            else:
                return

            cell.number = self.__parse_cell_number(cropped, cell.colour)

    @staticmethod
    def __process_image(image):
        coords = np.argwhere(image == 0)
        x0, x1 = coords[:, 0].min(), coords[:, 0].max()
        y0, y1 = coords[:, 1].min(), coords[:, 1].max()

        image = cv2.resize(image[x0:x1 + 1, y0:y1 + 1], (53, 45), interpolation=cv2.INTER_AREA)
        image = cv2.medianBlur(image, 3)

        return image
