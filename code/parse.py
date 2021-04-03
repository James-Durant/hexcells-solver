import numpy as np
import cv2, os, pickle, time

from PIL import Image

from grid import Grid, Cell

def average_hash(image, hash_size=16):
    image = Image.fromarray(image).convert("L").resize((hash_size, hash_size), Image.ANTIALIAS)
    pixels = np.array(image.getdata()).reshape((hash_size, hash_size))
    diff = pixels > pixels.mean()
    return diff.flatten()

class Parser:
    __hex_mask_path = '../resources/hex_mask.png'
    __counter_mask_path = '../resources/counter_mask.png'
    __hex_match_threshold = 0.05
    __counter_match_threshold = 0.1
    __area_threshold = 550
    __angles = np.asarray([-60, 0, 60])
    __cell_dims = (65, 60)
    __counter_dims = (200, 50)
    __grid_dims = (60, 55)

    def __init__(self, window, load_counter_hex_digits=True, load_grid_digits=True):
        self.__window = window
        self.__hex_contour, self.__counter_contour = Parser.__load_masks()
        
        if load_counter_hex_digits:
            self.__black_data = Parser.__load_hashes('black')
            self.__blue_data = Parser.__load_hashes('blue')
            self.__counter_data = Parser.__load_hashes('counter')
          
        if load_grid_digits:
            grid_hashes, grid_labels = [], []
            for digit_type in ['column']: #'diag_normal', 'diag_consecutive', 'diag_non-consecutive']:
                hashes, labels = Parser.__load_hashes(digit_type)
                grid_hashes += hashes
                grid_labels += labels
            
            self.__grid_data = (grid_hashes, grid_labels)

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
    def __load_hashes(digit_type):
        path = os.path.join('../resources', digit_type, 'hashes.pickle')
        with open(path, 'rb') as file:
            return pickle.load(file)

    def parse_grid(self, training=False):
        image = self.__window.screenshot()

        blue_cells = self.parse_cells(image, Cell.BLUE)
        black_cells = self.parse_cells(image, Cell.BLACK)
        orange_cells = self.parse_cells(image, Cell.ORANGE_OLD if training else Cell.ORANGE)

        cells = blue_cells + black_cells + orange_cells

        widths  = [cell.width  for cell in cells]
        heights = [cell.height for cell in cells]
        xs = [cell.image_coords[0] for cell in cells]
        ys = [cell.image_coords[1] for cell in cells]

        self.__x_min, self.__x_max = np.min(xs), np.max(xs)
        self.__y_min, self.__y_max = np.min(ys), np.max(ys)
        self.__hex_width  = int(np.median(widths))
        self.__hex_height = int(np.median(heights))
        
        x_spacing = self.__hex_width*1.085
        
        if self.__hex_height > 70:
            y_spacing = self.__hex_height*0.70
        else:
            y_spacing = self.__hex_height*0.72
            
        cols = int(round((self.__x_max - self.__x_min) / x_spacing) + 1)
        rows = int(round((self.__y_max - self.__y_min) / y_spacing) + 1)

        grid = [[None]*cols for _ in range(rows)]

        for cell in cells:
            x, y = cell.image_coords
            
            col = int(round((x - self.__x_min) / x_spacing))
            row = int(round((y - self.__y_min) / y_spacing))
            cell.grid_coords = (row, col)
            grid[row][col] = cell

        _, remaining = self.parse_counters(image)

        scene = Grid(grid, cells, remaining)
        parsed = self.__parse_columns(image, scene, training=training)

        return parsed if training else scene

    def parse_cells(self, image, cell_colour, training=False):
        mask = cv2.inRange(image, cell_colour, cell_colour)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cells = []
        for contour in contours:
            if (cv2.contourArea(contour) > Parser.__area_threshold and
                cv2.matchShapes(contour, self.__hex_contour, 1, 0) < Parser.__hex_match_threshold):
                x, y, w, h = cv2.boundingRect(contour)
                
                x_crop, y_crop = round(w*0.18), round(h*0.18)
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
        thresh = cv2.resize(thresh, Parser.__cell_dims, interpolation=cv2.INTER_AREA)

        if np.count_nonzero(thresh==0) < 20:
            return None
        
        hashed = average_hash(thresh, hash_size=32)
                                        
        if training:
            return hashed

        if cell_colour == Cell.BLACK:
            hashes, labels = self.__black_data
        elif cell_colour == Cell.BLUE:
            hashes, labels = self.__blue_data
        
        similarities = [np.sum(hashed != h) for h in hashes]
        match = labels[np.argmin(similarities)]
        #print(match)
        #cv2.imshow('test', thresh)
        #cv2.waitKey(0)
        
        return match

    def parse_counters(self, image, training=False):
        mask = cv2.inRange(image, Cell.BLUE, Cell.BLUE)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        i = 0
        parsed = [0]
        for contour in contours:
            if (cv2.contourArea(contour) > Parser.__area_threshold and
                cv2.matchShapes(contour, self.__counter_contour, 1, 0) < Parser.__counter_match_threshold):
                
                i += 1
                if i > 2:
                    raise RuntimeError('counters parsed incorrectly')
                
                if i == 2:
                    x, y, w, h = cv2.boundingRect(contour)
                    y = round(y+h*0.35)
                    h = round(h*0.65)

                    cropped = image[y: y+h, x: x+w]
                    thresh = cv2.cvtColor(np.where(cropped==Cell.BLUE, 255, 0).astype(np.uint8), cv2.COLOR_BGR2GRAY)
                    thresh = cv2.resize(thresh, Parser.__counter_dims, interpolation=cv2.INTER_AREA)
    
                    hashed = average_hash(thresh, hash_size=64)
    
                    if training:
                        parsed.append(hashed)
                        
                    else:
                        hashes, labels = self.__counter_data
                        similarities = [np.sum(hashed != h) for h in hashes]
                        match = labels[np.argmin(similarities)]
                        
                        #print(match)
                        #cv2.imshow('test', thresh)
                        #cv2.waitKey(0)
                        
                        parsed.append(match)

        return parsed

    def parse_clicked_cells(self, left_click_cells, right_click_cells):
        for cell in left_click_cells:
            self.__window.click_cell(cell, 'left')
          
        time.sleep(0.1)
            
        image = self.__window.screenshot()
        for cell in left_click_cells:   
            self.__parse_clicked_cell(image, cell)

        for cell in right_click_cells:
            self.__window.click_cell(cell, 'right')
            
        right_click_cells.sort(key=lambda cell: tuple(reversed(cell.grid_coords)))
        
        time.sleep(1.5)
        
        image = self.__window.screenshot()
        for cell in right_click_cells:   
            self.__parse_clicked_cell(image, cell)

        _, remaining = self.parse_counters(image)
        return remaining

    def __parse_clicked_cell(self, image, cell):
        cx, cy, w, h = *cell.image_coords, self.__hex_width, self.__hex_height

        x1, x2  = cx-w//2, cx+w//2
        y1, y2  = cy-h//2, cy+h//2
        
        x_crop, y_crop = round(w*0.18), round(h*0.18)
        cropped = image[y1+y_crop:y2-y_crop, x1+x_crop:x2-x_crop]

        if np.count_nonzero(cropped == Cell.BLACK) > 10:
            cell.colour = Cell.BLACK
        elif np.count_nonzero(cropped == Cell.BLUE) > 10:
            cell.colour = Cell.BLUE
        else:
            raise RuntimeError('cell must be blue or black after click')

        cell.digit = self.__parse_cell_digit(cropped, cell.colour)
        
    def __merge_rects(self, rects):
        rects.sort(key=lambda x: x[0])
        
        bounding_boxes = []
        while rects:
            rect1 = rects.pop()
            to_merge = [rect1]
            i = 0
            while i < len(rects):
                rect2 = rects[i]
                if (abs(rect1[0]-rect2[0]) < self.__hex_width*0.7 and
                    abs(rect1[1]-rect2[1]) < self.__hex_height*0.7):
                    to_merge.append(rect2)
                    del rects[i]
                else:
                    i += 1
            
            to_merge = np.asarray(to_merge)
            x = to_merge[:,0].min()
            y = to_merge[:,1].min()
            w = np.max(to_merge[:,0] + to_merge[:,2]) - x
            h = np.max(to_merge[:,1] + to_merge[:,3]) - y
            
            bounding_boxes.append((x, y, w, h))
            
        return bounding_boxes
        
    def __parse_columns(self, image, grid, training=False):
        _, thresh = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 100, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rects = []
        for contour in contours:
            if cv2.contourArea(contour) < 1000:
                rect = cv2.boundingRect(contour)
                if (self.__x_min-2*self.__hex_width < rect[0] <  self.__x_max+2*self.__hex_width and
                    self.__y_min-2*self.__hex_height < rect[1] < self.__y_max+self.__hex_height):
                    rects.append(rect)
                    
        bounding_boxes = self.__merge_rects(rects)

        #for x, y, w, h in bounding_boxes:
        #    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255))
        
        #cv2.imshow('test', image)
        #cv2.waitKey(0)
        
        parsed = []
        for x, y, w, h in bounding_boxes:
            box_coords = np.asarray([x + w//2, y + h//2])
            nearest_cell = grid.nearest_cell(box_coords)
            nearest_coords = nearest_cell.image_coords

            delta_x = box_coords[0] - nearest_coords[0]
            delta_y = nearest_coords[1] - box_coords[1]
            theta = 90 - (180 / np.pi * np.arctan2(delta_y, delta_x))
            angle = Parser.__angles[np.argmin(np.abs(Parser.__angles-theta))]
            
            x_pad = round(self.__hex_width*0.18)
            y_pad = round(self.__hex_height*0.1)
            
            cropped = thresh[y-y_pad: y+h+y_pad, x-x_pad: x+w+x_pad]
            centre = (w//2 + x_pad, h//2 + y_pad)
            
            rot_mat = cv2.getRotationMatrix2D(centre, angle, 1.0)
            rotated = cv2.warpAffine(cropped, rot_mat, cropped.shape[1::-1],
                                     flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))

            digit = self.__parse_grid_digit(255-rotated, training)
            if training:
                parsed.append(digit)
            else:
                row, col = nearest_cell.grid_coords
                grid.add_constraint(row, col, digit, angle)
        
        return parsed
            
    def __parse_grid_digit(self, image, training=False):
        thresh = cv2.resize(image, Parser.__grid_dims, interpolation=cv2.INTER_AREA)

        if np.count_nonzero(thresh==0) < 20:
            return None
        
        #cv2.imshow('test', thresh)
        #cv2.waitKey(0)
        
        hashed = average_hash(thresh, hash_size=64)
                                        
        if training:
            return hashed

        hashes, labels = self.__grid_data
        
        similarities = [np.sum(hashed != h) for h in hashes]
        match = labels[np.argmin(similarities)]
        
        print(match)
        cv2.imshow('test', thresh)
        cv2.waitKey(0)
        
        return match