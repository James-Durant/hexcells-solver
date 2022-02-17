import os
import cv2
import time
import json
import pickle

import numpy as np
from grid import Grid, Cell

RESOLUTIONS = [(2560, 1920), (2560, 1600), (2048, 1152),
               (1920, 1440), (1920, 1200), (1920, 1080),
               (1680, 1050), (1600, 1200)]


def average_hash(image):
    return (image > image.mean()).flatten()


class Parser:
    @staticmethod
    def _load_hashes(dataset_type):
        path = os.path.join('resources', dataset_type, 'hashes.pickle')
        try:
            with open(path, 'rb') as file:
                return pickle.load(file)
        except FileNotFoundError:
            raise RuntimeError(f'Reference hashes missing for {dataset_type}')

    @staticmethod
    def _merge_rects(rects, xdist, ydist):
        rects.sort(key=lambda x: x[0])

        merged = []
        while rects:
            x1, y1, w1, h1 = rects.pop()
            x_min, y_min = x1, y1
            x_max, y_max = x1 + w1, y1 + h1

            i = 0
            while i < len(rects):
                x2, y2, w2, h2 = rects[i]
                if (abs(x1 + w1 - x2) < xdist and
                        abs(y1 - y2) < ydist):

                    x_min = min(x_min, x2)
                    y_min = min(y_min, y2)
                    x_max = max(x_max, x2 + w2)
                    y_max = max(y_max, y2 + h2)
                    del rects[i]

                else:
                    i += 1

            merged.append((x_min, y_min, x_max - x_min, y_max - y_min))

        return merged

    @staticmethod
    def _find_match(hashes, labels, hashed):
        similarities = [np.sum(hashed != h) for h in hashes]
        return labels[np.argmin(similarities)]


class MenuParser(Parser):
    def __init__(self, window, use_hashes=True, steam_path=r'C:\Program Files (x86)\Steam'):
        self.__window = window
        self.__steam_path = steam_path
        self.__level_data = Parser._load_hashes('level_select')
        if use_hashes:
            self.__load_hashes()

    def __load_hashes(self):
        options_path = f'steamapps\\common\\{self.__window.title}\\saves\\options.txt'
        options_path = os.path.join(self.__steam_path, options_path)
        with open(options_path, 'r') as file:
            data = json.load(file)

        resolution = data['screenWidth'], data['screenHeight']
        if resolution not in RESOLUTIONS:
            resolution = (1920, 1080)
        
        dataset_type = os.path.join('screen', '{0}x{1}'.format(*resolution))
        hashes, labels = Parser._load_hashes(dataset_type)
        self.__level_exit = hashes.pop(-1)
        labels.pop(-1)
        self.__screen_data = hashes, labels

    def get_screen(self):
        image = cv2.resize(self.__window.screenshot(), (480, 270), interpolation=cv2.INTER_AREA)
        hashed = average_hash(image)
        
        hashes, labels = self.__screen_data
        similarities = [np.sum(hashed != h) for h in hashes]
            
        for label, sim in zip(labels, similarities):
            print(label, sim)
        
        if min(similarities) > 22000:
            # Check for level_exit
            image = cv2.inRange(image, (180, 180, 180), (255, 255, 255))

            sim = np.sum(self.__level_exit != image) 
            print(sim)
            if sim > 25000:
                print('in_level')
                print()
                return 'in_level'
            else:
                print('level_exit')
                print()
                return 'level_exit'
        
        print(labels[np.argmin(similarities)])
        print()
        return labels[np.argmin(similarities)]

    def wait_until_loaded(self):
        time.sleep(0.75)
        for _ in range(50):
            image = self.__window.screenshot()
            mask = cv2.inRange(image, Cell.ORANGE, Cell.ORANGE)
            mask += cv2.inRange(image, Cell.BLUE, Cell.BLUE)
            mask += cv2.inRange(image, (234, 164, 6), (234, 164, 6))

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                time.sleep(0.5)
                return

            time.sleep(0.05)

        raise RuntimeError('wait until loaded failed')

    def parse_main_menu(self):
        image = self.__window.screenshot()

        threshold = round(0.24 * image.shape[1])

        mask = cv2.inRange(image, (240, 240, 240), (255, 255, 255))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rects = [cv2.boundingRect(contour) for contour in contours]
        bounding_boxes = Parser._merge_rects(rects, 100, 100)

        buttons = [(x + w // 2, y + h // 2) for x, y, w, h in bounding_boxes if y > threshold]

        if len(buttons) == 4:
            buttons.sort(key=lambda x: tuple(reversed(x)))
            level_generator = buttons.pop()
        else:
            level_generator = None

        slots = sorted(buttons, key=lambda x: x[0])

        mask = cv2.inRange(image, (180, 180, 180), (180, 180, 180))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rects = [cv2.boundingRect(contour) for contour in contours]

        bounding_boxes = Parser._merge_rects(rects, image.shape[1], 50)
        buttons = [(x + w // 2, y + h // 2) for x, y, w, h in bounding_boxes if y > threshold]

        if len(buttons) == 3:
            user_levels, options, menu_exit = buttons
        elif len(buttons) == 2:
            user_levels = None
            options, menu_exit = buttons
        else:
            raise RuntimeError('main menu parsing failed')

        return {'save_slots': slots,
                'level_generator': level_generator,
                'user_levels': user_levels,
                'options': options,
                'exit': menu_exit}

    def parse_generator(self):
        image = self.__window.screenshot()

        mask = cv2.inRange(image, (240, 240, 240), (255, 255, 255))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
        bounding_boxes.sort(key=lambda box: box[1], reverse=True)
        x, y, w, h = bounding_boxes.pop(0)
        play_button = (x + w // 2, y + h // 2)

        bounding_boxes.sort(key=lambda box: box[0])
        x, y, w, h = bounding_boxes.pop(0)
        random_button = (x + w // 2, y + h // 2)

        mask = cv2.inRange(image, (234, 164, 6), (234, 164, 6))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rects = [cv2.boundingRect(contour) for contour in contours]

        bounding_boxes = Parser._merge_rects(rects, image.shape[1], 100)
        x, y, w, h = bounding_boxes[0]
        seed_button = (x + w // 2, y + h // 2)

        return {'play': play_button,
                'random': random_button,
                'seed': seed_button}

    def parse_levels(self, use_hashes=True):
        image = self.__window.screenshot()

        mask = cv2.inRange(image, (244, 244, 244), (255, 255, 255))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        areas = [cv2.contourArea(contour) for contour in contours]
        median_area = np.median(areas)

        boxes = [cv2.boundingRect(contour) for contour, area in list(zip(contours, areas))
                 if 0.95 * median_area < area < 1.05 * median_area]

        boxes.sort(key=lambda box: box[:2], reverse=True)

        levels = {}
        hashes = []
        for x, y, w, h in boxes:
            x_crop, y_crop = round(w * 0.25), round(h * 0.3)
            cropped = image[y + y_crop:y + h - y_crop, x + x_crop:x + w - x_crop]
            thresh = 255 - cv2.inRange(cropped, (240, 240, 240), (255, 255, 255))

            if np.count_nonzero(thresh == 0) < 20:
                continue

            coords = np.argwhere(thresh == 0)
            x0, y0 = coords.min(axis=0)
            x1, y1 = coords.max(axis=0) + 1

            thresh = cv2.resize(thresh[x0:x1, y0:y1], (52, 27), interpolation=cv2.INTER_AREA)

            hashed = average_hash(thresh)

            if not use_hashes:
                hashes.append(hashed)

            else:
                hashes, labels = self.__level_data
                match = Parser._find_match(hashes, labels, hashed)
                    
                levels[match] = (x + w // 2, y + h // 2)

        return levels if use_hashes else hashes

    def parse_level_exit(self):
        image = self.__window.screenshot()
        mask = cv2.inRange(image, (180, 180, 180), (255, 255, 255))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rects = [cv2.boundingRect(contour) for contour in contours]

        bounding_boxes = Parser._merge_rects(rects, image.shape[1], 100)
        return [(x + w // 2, y + h // 2) for x, y, w, h in bounding_boxes]

    def parse_level_end(self):
        # end -> completion
        image = self.__window.screenshot()

        mask = cv2.inRange(image, (255, 255, 255), (255, 255, 255))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
        bounding_boxes.sort(key=lambda x: (x[1], x[0]), reverse=True)

        if len(contours) == 6:
            next_button = (bounding_boxes[0][0] + bounding_boxes[0][2] // 2,
                           bounding_boxes[0][1] + bounding_boxes[0][3] // 2)

            menu_button = (bounding_boxes[1][0] + bounding_boxes[1][2] // 2,
                           bounding_boxes[1][1] + bounding_boxes[1][3] // 2)

            return next_button, menu_button

        else:
            menu_button = (bounding_boxes[0][0] + bounding_boxes[0][2] // 2,
                           bounding_boxes[0][1] + bounding_boxes[0][3] // 2)

        return None, menu_button


class LevelParser(Parser):
    __hex_mask_path = 'resources/hex_mask.png'
    __counter_mask_path = 'resources/counter_mask.png'
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

        hex_image = cv2.imread(LevelParser.__hex_mask_path)
        hex_mask = cv2.inRange(hex_image, Cell.ORANGE, Cell.ORANGE)
        hex_contour, _ = cv2.findContours(hex_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        counter_image = cv2.imread(LevelParser.__counter_mask_path)
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

        rects = []
        for contour in contours:
            if cv2.contourArea(contour) < 1000:
                x, y, w, h = cv2.boundingRect(contour)
                coords = np.asarray([x + w // 2, y + h // 2])
                if (self.__x_min - 2 * self.__hex_width < x < self.__x_max + 2 * self.__hex_width and
                        self.__y_min - 2 * self.__hex_height < y < self.__y_max + self.__hex_height and
                        np.linalg.norm(coords - grid.nearest_cell(coords).image_coords) < 130):

                    rects.append((x, y, w, h))

        bounding_boxes = Parser._merge_rects(rects, self.__hex_width * 0.76, self.__hex_height * 0.7)

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
