import os
import cv2
import time
import json
import pickle

import numpy as np
from grid import Grid, Cell

IMAGE_PATH = r'..\..\Figures\Implementation\Parsing'

RESOLUTIONS = [(2560, 1920), (2560, 1600), (2048, 1152),
               (1920, 1440), (1920, 1200), (1920, 1080),
               (1680, 1050), (1600, 1200)]


def average_hash(image):
    diff = image > image.mean()
    return diff.flatten()


class Parser:
    @staticmethod
    def _load_hashes(digit_type, resolution=None):
        path = os.path.join('resources', digit_type)
        if resolution:
            path = os.path.join(path, '{0}x{1}'.format(*resolution))

        path = os.path.join(path, 'hashes.pickle')
        try:
            with open(path, 'rb') as file:
                return pickle.load(file)
        except FileNotFoundError:
            return None

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


class MenuParser(Parser):
    def __init__(self, window, steam_path=r'C:\Program Files (x86)\Steam'):
        self.__window = window
        self.__steam_path = steam_path

        self.__load_data()
        self.__level_data = Parser._load_hashes('level_select')

    def __load_data(self):
        options_path = os.path.join(self.__steam_path,
                                    r'steamapps\common\{}\saves\options.txt'.format(self.__window.title))
        with open(options_path, 'r') as file:
            data = json.load(file)

        self.__resolution = data['screenWidth'], data['screenHeight']
        if self.__resolution not in RESOLUTIONS:
            self.__resolution = (1920, 1080)

        self.__screen_data = Parser._load_hashes('screen', self.__resolution)

    def get_screen(self):
        image = cv2.inRange(self.__window.screenshot(), (230, 230, 230), (255, 255, 255))
        if image.shape[0] != self.__resolution[1] or image.shape[1] != self.__resolution[0]:
            self.__load_data()
            if image.shape[0] != self.__resolution[1] or image.shape[1] != self.__resolution[0]:
                image = cv2.resize(image, tuple(reversed(self.__resolution)), interpolation=cv2.INTER_AREA)

        # Using image hashing I think
        images, labels = self.__screen_data
        similarities = [np.square(image - cv2.inRange(x, (230, 230, 230), (255, 255, 255))).mean() for x in images]

        # for label, sim in zip(labels, similarities):
        #    print(label, sim)
        # print(labels[np.argmin(similarities)])

        if min(similarities) > 0.1:
            return 'in_level'

        return labels[np.argmin(similarities)]

    def wait_until_loaded(self):
        time.sleep(0.5)
        for _ in range(100):
            image = self.__window.screenshot()
            mask = cv2.inRange(image, Cell.ORANGE, Cell.ORANGE) + cv2.inRange(image, Cell.BLUE, Cell.BLUE)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                time.sleep(0.5)
                return

            time.sleep(0.1)

        raise RuntimeError('wait until loaded failed')

    def parse_main_menu(self):
        image = self.__window.screenshot()

        threshold = round(0.24 * image.shape[1])

        mask = cv2.inRange(image, (240, 240, 240), (255, 255, 255))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rects = [cv2.boundingRect(contour) for contour in contours]
        bounding_boxes = Parser._merge_rects(rects, 100, 100)

        # cv2.imwrite(IMAGE_PATH+'\menus\implementation_parsing_main_mask_1.png', mask)
        # temp1 = image.copy()
        # temp2 = image.copy()
        # cv2.drawContours(temp1, contours, -1, (0,255,0), 2)
        # for x, y, w, h in bounding_boxes:
        #    cv2.rectangle(temp1, (x, y), (x+w, y+h), (0,0,255), 2)
        #    cv2.rectangle(temp2, (x, y), (x+w, y+h), (0,0,255), 2)

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

        # cv2.imwrite(IMAGE_PATH+'\menus\implementation_parsing_main_mask_2.png', mask)
        # cv2.drawContours(temp1, contours, -1, (0,255,0), 2)
        # for x, y, w, h in rects:
        #    if y > threshold:
        #        cv2.rectangle(temp1, (x, y), (x+w, y+h), (0,0,255), 2)
        # cv2.imwrite(IMAGE_PATH+'\menus\implementation_parsing_main_contours.png', temp1)

        bounding_boxes = Parser._merge_rects(rects, image.shape[1], 20)
        buttons = [(x + w // 2, y + h // 2) for x, y, w, h in bounding_boxes if y > threshold]

        # for x, y, w, h in bounding_boxes:
        #    if y > threshold:
        #        cv2.rectangle(temp2, (x,y), (x+w,y+h), (0,0,255), 2)
        # cv2.imwrite(IMAGE_PATH+'\menus\implementation_parsing_main_boxes.png', temp2)

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

        # cv2.imwrite(IMAGE_PATH+'\menus\implementation_parsing_level_generation_mask_1.png', mask)
        # temp1 = image.copy()
        # temp2 = image.copy()
        # cv2.drawContours(temp1, contours, -1, (0,255,0), 2)
        # for x, y, w, h in bounding_boxes:
        #    cv2.rectangle(temp1, (x, y), (x+w, y+h), (0,0,255), 2)

        bounding_boxes.sort(key=lambda box: box[1], reverse=True)
        x, y, w, h = bounding_boxes.pop(0)
        # cv2.rectangle(temp2, (x, y), (x+w, y+h), (0,0,255), 2)
        play_button = (x + w // 2, y + h // 2)

        bounding_boxes.sort(key=lambda box: box[0])
        x, y, w, h = bounding_boxes.pop(0)
        # cv2.rectangle(temp2, (x, y), (x+w, y+h), (0,0,255), 2)
        random_button = (x + w // 2, y + h // 2)

        mask = cv2.inRange(image, (234, 164, 6), (234, 164, 6))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.imwrite(IMAGE_PATH+'\menus\implementation_parsing_level_generation_mask_2.png', mask)

        rects = [cv2.boundingRect(contour) for contour in contours]

        # cv2.drawContours(temp1, contours, -1, (0,255,0), 2)
        # for x, y, w, h in rects:
        #    cv2.rectangle(temp1, (x, y), (x+w, y+h), (0,0,255), 2)

        bounding_boxes = Parser._merge_rects(rects, image.shape[1], 100)
        x, y, w, h = bounding_boxes[0]
        # cv2.rectangle(temp2, (x, y), (x+w, y+h), (0,0,255), 2)
        seed_button = (x + w // 2, y + h // 2)

        # cv2.imwrite(IMAGE_PATH+'\menus\implementation_parsing_level_generation_contours.png', temp1)
        # cv2.imwrite(IMAGE_PATH+'\menus\implementation_parsing_level_generation_boxes.png', temp2)

        return {'play': play_button,
                'random': random_button,
                'seed': seed_button}

    def parse_levels(self, training=False):
        image = self.__window.screenshot()

        mask = cv2.inRange(image, (245, 245, 245), (255, 255, 255))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # cv2.imwrite(IMAGE_PATH+'\menus\implementation_parsing_level_selection_mask.png', mask)
        # temp = image.copy()
        # cv2.drawContours(temp, contours, -1, (0,255,0), 2)
        # cv2.imwrite(IMAGE_PATH+'\menus\implementation_parsing_level_selection_contours.png', temp)
        # temp = image.copy()

        areas = [cv2.contourArea(contour) for contour in contours]
        median_area = np.median(areas)

        boxes = [cv2.boundingRect(contour) for contour, area in list(zip(contours, areas))
                 if 0.95 * median_area < area < 1.05 * median_area]

        boxes.sort(key=lambda box: box[:2], reverse=True)

        # for x, y, w, h in boxes:
        #    cv2.rectangle(temp, (x,y), (x+w,y+h), (0,0,255), 2)
        # cv2.imwrite(IMAGE_PATH+'\menus\implementation_parsing_level_selection_boxes.png', temp)
        # temp = image.copy()

        levels = {}
        training_hashes = []
        for x, y, w, h in boxes:
            x_crop, y_crop = round(w * 0.25), round(h * 0.3)
            cropped = image[y + y_crop:y + h - y_crop, x + x_crop:x + w - x_crop]
            thresh = 255 - cv2.inRange(cropped, (240, 240, 240), (255, 255, 255))

            if np.count_nonzero(thresh == 0) < 20:
                continue

            # temp[y+y_crop:y+h-y_crop, x+x_crop:x+w-x_crop] = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            # cv2.rectangle(temp, (x+x_crop,y+y_crop), (x+w-x_crop,y+h-y_crop), (0,0,255), 2)

            coords = np.argwhere(thresh == 0)
            x0, y0 = coords.min(axis=0)
            x1, y1 = coords.max(axis=0) + 1

            thresh = cv2.resize(thresh[x0:x1, y0:y1], (52, 27), interpolation=cv2.INTER_AREA)

            hashed = average_hash(thresh)

            if training:
                training_hashes.append(hashed)

            else:
                hashes, labels = self.__level_data

                similarities = [np.sum(hashed != h) for h in hashes]
                best_matches = np.array(labels)[np.argsort(similarities)[:3]].tolist()
                match = max(set(best_matches), key=best_matches.count)

                levels[match] = (x + w // 2, y + h // 2)

                # print(match, best_matches)
                # cv2.imshow('test', thresh)
                # cv2.waitKey(0)

        # cv2.imwrite(IMAGE_PATH+'\menus\implementation_parsing_level_selection_parsed.png', temp)

        return training_hashes if training else levels

    def parse_level_exit(self):
        image = self.__window.screenshot()
        mask = cv2.inRange(image, (180, 180, 180), (255, 255, 255))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rects = [cv2.boundingRect(contour) for contour in contours]

        # cv2.imwrite(IMAGE_PATH+'\menus\implementation_parsing_level_exit_image.png', image)
        # cv2.imwrite(IMAGE_PATH+'\menus\implementation_parsing_level_exit_mask.png', mask)
        # temp = image.copy()
        # for x, y, w, h in rects:
        #    cv2.rectangle(temp, (x, y), (x+w, y+h), (0,0,255), 2)

        bounding_boxes = Parser._merge_rects(rects, image.shape[1], 100)
        buttons = [(x + w // 2, y + h // 2) for x, y, w, h in bounding_boxes]

        # cv2.drawContours(temp, contours, -1, (0,255,0), 2)
        # for x, y, w, h in rects:
        #    cv2.rectangle(temp, (x-3, y-3), (x+w+3, y+h+3), (0,0,255), 2)
        # cv2.imwrite(IMAGE_PATH+'\menus\implementation_parsing_level_exit_contours.png', temp)
        # temp = image.copy()
        # for x, y, w, h in bounding_boxes:
        #    cv2.rectangle(temp, (x, y), (x+w, y+h), (0,0,255), 2)
        # cv2.imwrite(IMAGE_PATH+'\menus\implementation_parsing_level_exit_boxes.png', temp)

        return buttons

    def parse_level_end(self):
        # end -> completion
        image = self.__window.screenshot()

        mask = cv2.inRange(image, (255, 255, 255), (255, 255, 255))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
        bounding_boxes.sort(key=lambda x: (x[1], x[0]), reverse=True)

        # temp = image.copy()
        # cv2.imwrite(IMAGE_PATH+'\menus\implementation_parsing_level_completion_mask.png', mask)
        # for i, (x, y, w, h) in enumerate(bounding_boxes):
        #    if i <= 1:
        #        cv2.rectangle(temp, (x, y), (x+w-1, y+h-1), (0,0,255), 2)
        #    else:
        #        cv2.rectangle(temp, (x, y), (x+w-1, y+h-1), (0,255,0), 2)
        # cv2.imwrite(IMAGE_PATH+'\menus\implementation_parsing_level_completion_contours.png', temp)

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


class GameParser(Parser):
    __hex_mask_path = 'resources/hex_mask.png'
    __counter_mask_path = 'resources/counter_mask.png'
    __hex_match_threshold = 0.08
    __counter_match_threshold = 0.1
    __area_threshold = 550
    __angles = np.asarray([0, 60, 90, 120, 240, 270, 300, 360])
    __digit_dims = (45, 30)
    __counter_dims = (200, 50)

    def __init__(self, window):
        self.__window = window

        self.__black_data = Parser._load_hashes('black')
        self.__blue_data = Parser._load_hashes('blue')
        self.__counter_data = Parser._load_hashes('counter')
        self.__column_data = Parser._load_hashes('column')
        self.__diagonal_data = Parser._load_hashes('diagonal')

        hex_image = cv2.imread(GameParser.__hex_mask_path)
        hex_mask = cv2.inRange(hex_image, Cell.ORANGE, Cell.ORANGE)
        hex_contour, _ = cv2.findContours(hex_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        counter_image = cv2.imread(GameParser.__counter_mask_path)
        counter_mask = cv2.inRange(counter_image, Cell.BLUE, Cell.BLUE)
        counter_contour, _ = cv2.findContours(counter_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.__hex_contour = hex_contour[0]
        self.__counter_contour = counter_contour[0]

        self.__x_min, self.__x_max = float('inf'), -float('inf')
        self.__y_min, self.__y_max = float('inf'), -float('inf')
        self.__hex_width = float('inf')
        self.__hex_height = float('inf')

    def parse_grid(self, training=False):
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

        if self.__hex_height > 70:
            y_spacing = self.__hex_height * 0.70
        else:
            y_spacing = self.__hex_height * 0.72

        cols = int(round((self.__x_max - self.__x_min) / x_spacing) + 1)
        rows = int(round((self.__y_max - self.__y_min) / y_spacing) + 1)

        # print(self.__hex_width, (self.__x_max - self.__x_min) / x_spacing)
        # print(self.__hex_height, (self.__y_max - self.__y_min) / y_spacing)

        grid = [[None] * cols for _ in range(rows)]
        for cell in cells:
            x, y = cell.image_coords

            col = int(round((x - self.__x_min) / x_spacing))
            row = int(round((y - self.__y_min) / y_spacing))
            cell.grid_coords = (row, col)
            grid[row][col] = cell

        # temp = image.copy()
        # img_height, img_width, _ = temp.shape

        # diff =  grid[1][1].image_coords[1] - grid[0][0].image_coords[1]
        # for row in range(rows):
        #    y = int(round(self.__y_min+row*diff))
        #    cv2.line(temp, (0, y), (img_width, y), (0,255,0), 2)

        # diff =  grid[1][1].image_coords[0] - grid[0][0].image_coords[0]
        # for col in range(cols):
        #    x = int(round(self.__x_min+col*diff))
        #    cv2.line(temp, (x, 0), (x, img_height), (0,255,0), 2)

        # for cell in cells:
        #    x, y = cell.image_coords
        #    row, col = cell.grid_coords
        #    cv2.circle(temp, (x,y), radius=4, color=(0,0,255), thickness=-1)
        #    cv2.putText(temp, str((col,row)), (x+5,y-10), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0,0,255), thickness=1)

        # cv2.imwrite(IMAGE_PATH+'\implementation_parsing_grid.png', temp)

        _, remaining = self.parse_counters(image)

        scene = Grid(grid, cells, remaining)
        parsed = self.__parse_columns(image, scene, training=training)

        return parsed if training else scene

    def parse_cells(self, image, cell_colour, training=False):
        mask = cv2.inRange(image, cell_colour, cell_colour)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # temp = image.copy()
        # cv2.imwrite(IMAGE_PATH+'\Cells\implementation_parsing_cells_mask_{}.png'.format(cell_colour), mask)

        cells = []
        for contour in contours:
            if (cv2.contourArea(contour) > GameParser.__area_threshold and
                    cv2.matchShapes(contour, self.__hex_contour, 1, 0) < GameParser.__hex_match_threshold):
                x, y, w, h = cv2.boundingRect(contour)

                x_crop, y_crop = round(w * 0.18), round(h * 0.18)
                cropped = image[y + y_crop:y + h - y_crop, x + x_crop:x + w - x_crop]

                parsed = self.__parse_cell_digit(cropped, cell_colour, training)

                # cv2.drawContours(temp, [contour], -1, (0,255,0), 2)
                # if parsed is not None:
                #    thresh = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                #    thresh = np.where(thresh > 220, 0, 255).astype(np.uint8)
                #    temp[y+y_crop:y+h-y_crop, x+x_crop:x+w-x_crop] = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
                #    cv2.rectangle(temp, (x+x_crop+1, y+y_crop+1), (x+w-x_crop-1, y+h-y_crop-1), (0,0,255), 2)

                if training:
                    if parsed is not None:
                        cells.append(parsed)
                else:
                    centre = (x + w // 2, y + h // 2)
                    cell = Cell(centre, w, h, cell_colour, parsed)
                    cells.append(cell)

        # cv2.imwrite(IMAGE_PATH+'\Cells\implementation_parsing_cells_boxes_{}.png'.format(cell_colour), temp)

        return cells

    def __parse_cell_digit(self, image, cell_colour, training=False):
        if cell_colour == Cell.ORANGE:
            return None

        thresh = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = np.where(thresh > 220, 0, 255).astype(np.uint8)

        if np.count_nonzero(thresh == 0) < 20:
            return None

        thresh = GameParser.__process_image(thresh)
        hashed = average_hash(thresh)

        if training:
            return hashed

        if cell_colour == Cell.BLACK:
            hashes, labels = self.__black_data

        elif cell_colour == Cell.BLUE:
            hashes, labels = self.__blue_data

        else:
            raise RuntimeError('invalid cell colour')

        match = GameParser.__find_match(hashes, labels, hashed)

        if match[0] == '{' and match[-1] == '}':
            temp = thresh.copy()
            temp[:, :15] = 255
            temp[:, -15:] = 255

            temp = GameParser.__process_image(temp)

            # cv2.imshow('test', temp)
            # cv2.waitKey(0)

            digit = GameParser.__find_match(hashes, labels, average_hash(temp))
            match = '{' + digit + '}'

        # print(match)
        # cv2.imshow('test', thresh)
        # cv2.waitKey(0)

        return match

    def parse_counters(self, image=None, training=False):
        if image is None:
            image = self.__window.screenshot()

        mask = cv2.inRange(image, Cell.BLUE, Cell.BLUE)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # temp = image.copy()
        # cv2.drawContours(temp, contours, -1, (0,255,0), 2)
        # cv2.imwrite(IMAGE_PATH+'\Counters\implementation_parsing_counters_image.png', image)
        # cv2.imwrite(IMAGE_PATH+'\Counters\implementation_parsing_counters_mask.png', mask)
        # cv2.imwrite(IMAGE_PATH+'\Counters\implementation_parsing_counters_contours.png', temp)
        # temp = image.copy()

        parsed = []
        for contour in contours:
            if (cv2.contourArea(contour) > GameParser.__area_threshold and
                    cv2.matchShapes(contour, self.__counter_contour, 1, 0) < GameParser.__counter_match_threshold):

                x, y, w, h = cv2.boundingRect(contour)
                y = round(y + h * 0.35)
                h = round(h * 0.65)

                # cv2.rectangle(temp, (x, y), (x+w-1, y+h-1), (0,0,255), 2)

                cropped = image[y:y + h, x:x + w]
                thresh = cv2.cvtColor(np.where(cropped == Cell.BLUE, 255, 0).astype(np.uint8), cv2.COLOR_BGR2GRAY)

                # cv2.imwrite(IMAGE_PATH+'\Counters\implementation_parsing_counters_thresh_{}.png'.format(len(parsed)), thresh)

                thresh = cv2.resize(thresh, GameParser.__counter_dims, interpolation=cv2.INTER_AREA)
                hashed = average_hash(thresh)

                if training:
                    parsed.append(hashed)

                else:
                    hashes, labels = self.__counter_data
                    similarities = [np.sum(hashed != h) for h in hashes]
                    match = labels[np.argmin(similarities)]

                    # print(match)
                    # cv2.imshow('test', thresh)
                    # cv2.waitKey(0)

                    parsed.append(match)

        if len(parsed) != 2:
            raise RuntimeError('counters parsed incorrectly: {}/2'.format(len(parsed)))

        # cv2.imwrite(IMAGE_PATH+'\Counters\implementation_parsing_counters_filtered.png', temp)

        return parsed

    def __parse_columns(self, image, grid, training=False):
        _, thresh = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 120, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # temp1 = image.copy()
        # cv2.imwrite(IMAGE_PATH+'\Columns\implementation_parsing_columns_mask.png', thresh)

        rects = []
        for contour in contours:
            if cv2.contourArea(contour) < 1000:
                x, y, w, h = cv2.boundingRect(contour)
                coords = np.asarray([x + w // 2, y + h // 2])
                if (self.__x_min - 2 * self.__hex_width < x < self.__x_max + 2 * self.__hex_width and
                        self.__y_min - 2 * self.__hex_height < y < self.__y_max + self.__hex_height and
                        np.linalg.norm(coords - grid.nearest_cell(coords).image_coords) < 144):
                    rects.append((x, y, w, h))
                    # cv2.drawContours(temp1, [contour], -1, (0,255,0), 1)
                    # cv2.rectangle(temp1, (x-1, y-1), (x+w, y+h), (0,0,255), 1)

        bounding_boxes = Parser._merge_rects(rects, self.__hex_width * 0.76, self.__hex_height * 0.7)

        # cv2.imwrite(IMAGE_PATH+'\Columns\implementation_parsing_columns_boxes_1.png', temp1)
        # temp2 = image.copy()

        parsed = []
        for x, y, w, h in bounding_boxes:
            box_coords = np.asarray([x + w // 2, y + h // 2])
            nearest_cell = grid.nearest_cell(box_coords)
            nearest_coords = nearest_cell.image_coords

            delta_x = box_coords[0] - nearest_coords[0]
            delta_y = nearest_coords[1] - box_coords[1]
            theta = (90 - np.degrees(np.arctan2(delta_y, delta_x))) % 360
            angle = GameParser.__angles[np.argmin(np.abs(GameParser.__angles - theta))]

            cropped = 255 - thresh[y:y + h, x:x + w]
            digit = self.__parse_grid_digit(cropped, angle, training)

            # cv2.rectangle(temp2, (x-1, y-1), (x+w, y+h), (0,0,255), 2)
            # cv2.arrowedLine(temp2, (x+w//2, y+h//2), nearest_coords, (0,0,255), 2)
            # if angle == 360:
            #    angle = 0
            # cv2.putText(temp2, str(angle), (nearest_coords[0]+5,nearest_coords[1]+15), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0,0,255), 1)
            # temp2[y:y+h, x:x+w] = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)

            if training:
                parsed.append(digit)
            else:
                row, col = nearest_cell.grid_coords
                grid.add_constraint(row, col, digit, angle)

        # cv2.imwrite(IMAGE_PATH+'\Columns\implementation_parsing_columns_boxes_2.png', temp2)

        return parsed

    def __parse_grid_digit(self, thresh, angle, training=False):
        if np.count_nonzero(thresh == 0) < 20:
            return None

        if angle in [120, 240]:
            thresh = cv2.flip(cv2.flip(thresh, 1), 0)

        elif angle in [90, 270]:
            centre = tuple(np.array(thresh.shape[1::-1]) / 2)
            rotation_matrix = cv2.getRotationMatrix2D(centre, angle, 1.0)
            thresh = cv2.warpAffine(thresh, rotation_matrix, thresh.shape[1::-1],
                                    flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))

        thresh = GameParser.__process_image(thresh)
        hashed = average_hash(thresh)

        if training:
            return hashed

        if angle in [0, 90, 270, 360]:
            hashes, labels = self.__column_data
        else:
            hashes, labels = self.__diagonal_data

        match = GameParser.__find_match(hashes, labels, hashed)

        if match[0] == '{' and match[-1] == '}' and match[1] in ['3', '5', '8']:
            temp = thresh.copy()

            if angle in [0, 360]:
                temp[:, :15] = 255
                temp[:, -15:] = 255
            else:
                centre = tuple(np.array(temp.shape[1::-1]) / 2)
                rotation_matrix = cv2.getRotationMatrix2D(centre, angle, 1.0)
                temp = cv2.warpAffine(temp, rotation_matrix, temp.shape[1::-1],
                                      flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))

            temp = GameParser.__process_image(temp)

            hashes, labels = self.__column_data
            match = GameParser.__find_match(hashes, labels, average_hash(temp))

            if angle in [0, 360]:
                match = '{' + match + '}'

        # print(match)
        # cv2.imshow('test', thresh)
        # cv2.waitKey(0)

        return match

    def parse_clicked(self, left_click_cells, right_click_cells):
        image = None
        if left_click_cells:
            self.click_cells(left_click_cells, 'left')

            time.sleep(0.1)
            image = self.__window.screenshot()
            self.__parse_clicked_cells(image, left_click_cells)

        if right_click_cells:
            self.click_cells(right_click_cells, 'right')

            # min_row = min(right_click_cells, key=lambda cell: cell.grid_coords[0]).grid_coords[0]
            time.sleep(0.1)
            # time.sleep(max(1.5, 0.12*(grid.rows-min_row)))
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
                # raise RuntimeError('cell must be blue or black after click')

            cell.digit = self.__parse_cell_digit(cropped, cell.colour)

    @staticmethod
    def __process_image(image):
        coords = np.argwhere(image == 0)
        x0, x1 = coords[:, 0].min(), coords[:, 0].max()
        y0, y1 = coords[:, 1].min(), coords[:, 1].max()

        image = cv2.resize(image[x0:x1 + 1, y0:y1 + 1], (53, 45), interpolation=cv2.INTER_AREA)
        image = cv2.medianBlur(image, 3)

        return image

    @staticmethod
    def __find_match(hashes, labels, hashed):
        similarities = [np.sum(hashed != h) for h in hashes]
        return labels[np.argmin(similarities)]
