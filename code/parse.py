import numpy as np
import cv2, pytesseract, os
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

from grid import Grid, Cell

from PIL import Image
import imagehash

class Parser:
    __hex_mask_path = '../resources/hex_mask.png'
    __counter_mask_path = '../resources/counter_mask.png'
    __hex_match_threshold = 0.05
    __counter_match_threshold = 0.1
    __area_threshold = 550
    __angles = np.asarray([-60, 0, 60])
    __DIMS = (65, 60)

    def __init__(self, window):
        self.__window = window
        self.__hex_contour, self.__counter_contour = Parser.__load_masks()
        self.__black_digits = self.__load_digits('black')
        self.__blue_digits = self.__load_digits('blue')

    @property
    def window(self):
        return self.__window

    @staticmethod
    def __load_masks():
        hex_image = cv2.imread(Parser.__hex_mask_path)
        hex_mask  = cv2.inRange(hex_image, Cell.ORANGE , Cell.ORANGE)
        hex_contour, _ = cv2.findContours(hex_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        counter_image = cv2.imread(Parser.__counter_mask_path)
        counter_mask  = cv2.inRange(counter_image, Cell.BLUE, Cell.BLUE)
        counter_contour, _ = cv2.findContours(counter_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return hex_contour[0], counter_contour[0]

    @staticmethod
    def __load_digits(digit_type):
        path = '../resources/'+digit_type+'_digits/'
        images, labels = [], []
        for filename in os.listdir(path):
            img = cv2.imread(path+filename, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                labels.append(filename.split('_')[0])
                
        return images, labels

    def parse_grid(self):
        image = self.__window.screenshot()

        blue_cells = self.parse_cells(image, Cell.BLUE)
        black_cells = self.parse_cells(image, Cell.BLACK)
        orange_cells = self.parse_cells(image, Cell.ORANGE)

        cells = blue_cells + black_cells + orange_cells

        widths  = [cell.width  for cell in cells]
        heights = [cell.height for cell in cells]
        xs = [cell.image_coords[0] for cell in cells]
        ys = [cell.image_coords[1] for cell in cells]

        x_min, x_max = np.min(xs), np.max(xs)
        y_min, y_max = np.min(ys), np.max(ys)
        self.__hex_width  = int(np.median(widths))
        self.__hex_height = int(np.median(heights))

        x_spacing = self.__hex_width*1.085
        y_spacing = self.__hex_height*0.70

        cols = int(round((x_max - x_min) / x_spacing) + 1)
        rows = int(round((y_max - y_min) / y_spacing) + 1)

        grid = [[None]*cols for _ in range(rows)]

        for cell in cells:
            x, y = cell.image_coords
            col = int(round((x - x_min) / x_spacing))
            row = int(round((y - y_min) / y_spacing))
            cell.grid_coords = (row, col)
            grid[row][col] = cell

        _, remaining = self.__parse_counters(image)

        scene = Grid(grid, cells, remaining)
        self.__parse_columns(image, scene)

        return scene

    def parse_cells(self, image, cell_colour, training=False):
        mask = cv2.inRange(image, cell_colour, cell_colour)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cells = []
        for contour in contours:
            if (cv2.contourArea(contour) > Parser.__area_threshold and
                cv2.matchShapes(contour, self.__hex_contour, 1, 0) < Parser.__hex_match_threshold):
                x, y, w, h = cv2.boundingRect(contour)

                x_crop, y_crop = round(w*0.2), round(h*0.2)

                cropped = image[y+y_crop:y+h-y_crop, x+x_crop:x+w-x_crop]

                parsed = self.__parse_cell_digit(cropped, cell_colour, training)
                if training:
                    cells.append(parsed)
                else:
                    centre = (x + w//2, y + h//2)
                    cell = Cell(centre, w, h, cell_colour, parsed)
                    cells.append(cell)

        return cells

    def __parse_cell_digit(self, image, cell_colour, training=False):
        if cell_colour == Cell.ORANGE or cell_colour == Cell.ORANGE_OLD:
            return None

        thresh = cv2.cvtColor(np.where(image==cell_colour, 255, 0).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        thresh = cv2.resize(thresh, Parser.__DIMS, interpolation=cv2.INTER_AREA)

        if training:
            return thresh

        if 0 not in thresh:
            return None

        if cell_colour == Cell.BLACK:
            images, labels = self.__black_digits
        elif cell_colour == Cell.BLUE:
            images, labels = self.__blue_digits
        
        similarities = [imagehash.average_hash(Image.fromarray(thresh), hash_size=16)-imagehash.average_hash(Image.fromarray(img), hash_size=16) for img in images]
        match = labels[np.argmin(similarities)]
        print(match)
        cv2.imshow('test', thresh)
        cv2.waitKey(0)
        
        return match

    def parse_counters(self, image, training=False):
        mask = cv2.inRange(image, Cell.BLUE, Cell.BLUE)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        values = []
        for contour in contours:
            if (cv2.contourArea(contour) > Parser.__area_threshold and
                cv2.matchShapes(contour, self.__counter_contour, 1, 0) < Parser.__counter_match_threshold):
                x, y, w, h = cv2.boundingRect(contour)
                y = round(y+h*0.35)
                h = round(h*0.65)

                cropped = image[y: y+h, x: x+w]
                thresh = cv2.cvtColor(np.where(cropped==Cell.BLUE, 255, 0).astype(np.uint8), cv2.COLOR_BGR2GRAY)

                if training:
                    values.append(thresh)
                else:
                    values.append(1)

        return values

    def __parse_columns(self, image, grid):
        _, thresh = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 100, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rects = []
        for contour in contours:
            if cv2.matchShapes(contour, self.__hex_contour, 1, 0) > Parser.__hex_match_threshold:
                x, y, w, h = cv2.boundingRect(contour)
                rects += [(x,y,w,h), (x,y,w,h)]

        bounding_boxes, _ = cv2.groupRectangles(rects, 1, 1.5)
        for x, y, w, h in bounding_boxes:
            box_coords = np.asarray([x + w//2, y + h//2])
            nearest_cell = grid.nearest_cell(box_coords)
            nearest_coords = nearest_cell.image_coords

            delta_x = box_coords[0] - nearest_coords[0]
            delta_y = nearest_coords[1] - box_coords[1]
            theta = 90 - (180 / np.pi * np.arctan2(delta_y, delta_x))
            angle = Parser.__angles[np.argmin(np.abs(Parser.__angles-theta))]

            cropped = image[y-10: y+h+10, x-15: x+w+15]

            cv2.imshow('test', cropped)
            cv2.waitKey(0)

            centre  = (w//2 + 15, h//2 + 10)
            rot_mat = cv2.getRotationMatrix2D(centre, angle, 1.0)
            rotated = cv2.warpAffine(cropped, rot_mat, cropped.shape[1::-1],
                                     flags=cv2.INTER_LINEAR, borderValue=(255,255,255))

            _, thresh = cv2.threshold(cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY), 200, 255, cv2.THRESH_BINARY)

            lang = 'eng'
            config = '--psm 6 -c tessedit_char_whitelist=0123456789{}-'

            digit_str = pytesseract.image_to_string(thresh, lang=lang, config=config)
            digit = digit_str.split('\n')[0]

            row, col = nearest_cell.grid_coords
            grid.add_constraint(row, col, digit, angle)

    def parse_clicked_cells(self, clicked_cells):
        image = self.__window.screenshot()
        for cell in clicked_cells:
            self.__parse_clicked_cell(image, cell)

        _, remaining = self.parse_counters(image)
        return remaining

    def __parse_clicked_cell(self, image, cell):
        cx, cy, w, h = *cell.image_coords, self.__hex_width, self.__hex_height

        x1, x2  = cx-w//2, cx+w//2
        y1, y2  = cy-h//2, cy+h//2
        cropped = image[y1+10: y2-10, x1+10: x2-10]

        if np.count_nonzero(cropped == Cell.BLACK) > 10:
            cell.colour = Cell.BLACK
        elif np.count_nonzero(cropped == Cell.BLUE) > 10:
            cell.colour = Cell.BLUE
        else:
            raise RuntimeError('cell must be blue or black after click')

        cell.digit = Parser.__parse_cell_digit(cropped, cell.colour)
