import numpy as np
import cv2, os, pickle, time

from grid import Grid, Cell 

def average_hash(image):
    diff = image > image.mean()
    return diff.flatten()

class Parser:
    @staticmethod 
    def _load_hashes(digit_type):
        path = os.path.join('..', 'resources', digit_type, 'hashes.pickle')
        try:
            with open(path, 'rb') as file:
                return pickle.load(file)
        except FileNotFoundError:
            return None
    
class MenuParser(Parser):
    def __init__(self, window):
        self.__window = window
        self.__level_data = Parser._load_hashes('level')
        self.__screen_data = Parser._load_hashes('screen')
    
    def get_screen(self):
        image = cv2.resize(self.__window.screenshot(), (1920, 1080), interpolation=cv2.INTER_AREA)
        
        images, labels = self.__screen_data
        similarities = [np.square(image-x).mean() for x in images]

        if min(similarities) > 10:
            return 'in_level'
        
        return labels[np.argmin(similarities)]
      
    def parse_slots(self, training=False):
        image = self.__window.screenshot()
        
        height = image.shape[1]
        if training:
            image = image[round(height*0.2):round(height*0.5), :]
            mask = cv2.inRange(image, (254,254,254), (255,255,255))
        else:
            image = image[round(height*0.2):round(height*0.4), :]
            mask = cv2.inRange(image, (240,240,240), (255,255,255))
        
        #cv2.imshow('test', mask)
        #cv2.waitKey(0)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        slots = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            y += round(height*0.2)
            
            slots.append((x+w//2, y+h//2))
        
        if len(slots) == 4:
            slots.sort(key=lambda x: tuple(reversed(x)))
            generator = slots.pop()
        else:
            generator = None
            
        slots.sort(key=lambda x: x[0])
        return slots, generator        
   
    def parse_generator(self):
        image = self.__window.screenshot()
        
        mask = cv2.inRange(image, (240,240,240), (255,255,255))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes = [cv2.boundingRect(contour) for contour in contours]
        boxes.sort(key=lambda x: x[1], reverse=True)
        x, y, w, h = boxes.pop(0)
        play = (x+w//2, y+h//2)
        
        boxes.sort(key=lambda x: x[0])
        x, y, w, h = boxes.pop(0)
        random = (x+w//2, y+h//2)
        
        return play, random
   
    def parse_levels(self, training=False):
        image = self.__window.screenshot()
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if training:
            mask = 255-cv2.inRange(image, (220,220,220), (255,255,255))
        else:
            mask = cv2.inRange(image, (245,245,245), (255,255,255))
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        #cv2.imshow('test', mask)
        #cv2.waitKey(0)
        
        #image = cv2.drawContours(image, contours, -1, (0,255,0), 3)
        #cv2.imshow('test', image)
        #cv2.waitKey(0)
        
        areas = [cv2.contourArea(contour) for contour in contours]
        median_area = np.median(areas)
    
        boxes = [cv2.boundingRect(contour) for contour, area in list(zip(contours, areas))
                 if 0.95*median_area < area < 1.05*median_area]
        
        boxes.sort(key=lambda x: x[:2], reverse=True)
        
        #for x, y, w, h in boxes:
        #    image = cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 1)
        #cv2.imshow('test', image)
        #cv2.waitKey(0)
        
        levels, training_hashes = {}, []
        for x, y, w, h in boxes:
            x_crop, y_crop = round(w*0.25), round(h*0.3)
            cropped = grey[y+y_crop:y+h-y_crop, x+x_crop:x+w-x_crop]
            cropped = np.where(cropped > 240, 0, 255).astype(np.uint8)
    
            if np.count_nonzero(cropped==0) < 20:
                continue
    
            coords = np.argwhere(cropped==0)
            x0, y0 = coords.min(axis=0)
            x1, y1 = coords.max(axis=0) + 1

            cropped = cv2.resize(cropped[x0:x1, y0:y1], (52, 27), interpolation=cv2.INTER_AREA)
    
            hashed = average_hash(cropped)
                           
            if training:
                training_hashes.append(hashed)

            hashes, labels = self.__level_data
    
            similarities = [np.sum(hashed != h) for h in hashes]
            best_matches = np.array(labels)[np.argsort(similarities)[:3]].tolist()
            match = max(set(best_matches), key=best_matches.count)
            
            levels[match] = (x+w//2, y+h//2)
            
            #print(match, best_matches)
            #cv2.imshow('test', cropped)
            #cv2.waitKey(0)

        return training_hashes if training else levels
            
    def parse_level_end(self):
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
    
    def wait_until_loaded(self):
        while True:
            image = self.__window.screenshot()
            mask = cv2.inRange(image, Cell.ORANGE, Cell.ORANGE) + cv2.inRange(image, Cell.BLUE, Cell.BLUE)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                break
            time.sleep(0.1)
            
        time.sleep(0.5)
    
class GameParser(Parser):
    __hex_mask_path = '../resources/hex_mask.png'
    __counter_mask_path = '../resources/counter_mask.png'
    __hex_match_threshold = 0.08
    __counter_match_threshold = 0.1
    __area_threshold = 550
    __angles = np.asarray([0, 60, 90, 120, 240, 270, 300, 360])
    __digit_dims = (45, 30)
    __counter_dims = (200, 50)

    def __init__(self, window, load_counter_hashes=True, load_grid_hashes=True):
        self.__window = window
        self.__hex_contour, self.__counter_contour = GameParser.__load_masks()
        
        self.__black_data = Parser._load_hashes('black')
        self.__blue_data = Parser._load_hashes('blue')
        self.__counter_data = Parser._load_hashes('counter')
        self.__column_data = Parser._load_hashes('column')
        self.__diagonal_data = Parser._load_hashes('diagonal')

    @staticmethod
    def __load_masks():
        hex_image = cv2.imread(GameParser.__hex_mask_path)
        hex_mask  = cv2.inRange(hex_image, Cell.ORANGE , Cell.ORANGE)
        hex_contour, _ = cv2.findContours(hex_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        counter_image = cv2.imread(GameParser.__counter_mask_path)
        counter_mask  = cv2.inRange(counter_image, Cell.BLUE, Cell.BLUE)
        counter_contour, _ = cv2.findContours(counter_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return hex_contour[0], counter_contour[0]

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
        
        if self.__hex_width > 70:
            x_spacing = self.__hex_width*1.085
        elif 54 <= self.__hex_width <= 70:
            x_spacing = self.__hex_width*1.098
        else:
            x_spacing = self.__hex_width*1.105
            
        if self.__hex_height > 70:
            y_spacing = self.__hex_height*0.70
        else:
            y_spacing = self.__hex_height*0.72
        
        cols = int(round((self.__x_max - self.__x_min) / x_spacing) + 1)
        rows = int(round((self.__y_max - self.__y_min) / y_spacing) + 1)
        
        #print(self.__hex_width, (self.__x_max - self.__x_min) / x_spacing)
        #print(self.__hex_height, (self.__y_max - self.__y_min) / y_spacing)

        grid = [[None]*cols for _ in range(rows)]

        for cell in cells:
            x, y = cell.image_coords
            
            col = int(round((x - self.__x_min) / x_spacing))
            row = int(round((y - self.__y_min) / y_spacing))
            cell.grid_coords = (row, col)
            grid[row][col] = cell

        remaining = self.parse_counters(image)

        scene = Grid(grid, cells, remaining)
        parsed = self.__parse_columns(image, scene, training=training)

        return parsed if training else scene

    def parse_cells(self, image, cell_colour, training=False):
        mask = cv2.inRange(image, cell_colour, cell_colour)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cells = []
        for contour in contours:
            if (cv2.contourArea(contour) > GameParser.__area_threshold and
                cv2.matchShapes(contour, self.__hex_contour, 1, 0) < GameParser.__hex_match_threshold):
                x, y, w, h = cv2.boundingRect(contour)
                
                x_crop, y_crop = round(w*0.18), round(h*0.18)
                cropped = image[y+y_crop:y+h-y_crop, x+x_crop:x+w-x_crop]

                parsed = self.__parse_cell_digit(cropped, cell_colour, training)
                if training:
                    if parsed is not None:
                        cells.append(parsed)
                else:
                    centre = (x + w//2, y + h//2)
                    cell = Cell(centre, w, h, cell_colour, parsed)
                    cells.append(cell)

        return cells

    def __parse_cell_digit(self, image, cell_colour, training=False):
        if cell_colour == Cell.ORANGE or cell_colour == Cell.ORANGE_OLD:
            return None
        
        thresh = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)   
        thresh = np.where(thresh > 220, 0, 255).astype(np.uint8)
        
        if np.count_nonzero(thresh==0) < 20:
            return None
        
        thresh = GameParser.__process_image(thresh)
        hashed = average_hash(thresh)
                               
        if training:
            return hashed

        if cell_colour == Cell.BLACK:
            hashes, labels = self.__black_data
            
        elif cell_colour == Cell.BLUE:
            hashes, labels = self.__blue_data
        
        match = GameParser.__find_match(hashes, labels, hashed)
    
        if match[0] == '{' and match[-1] == '}':
            temp = thresh.copy()
            temp[:, :15] = 255
            temp[:, -15:] = 255
            
            temp = GameParser.__process_image(temp)
            
            #cv2.imshow('test', temp)
            #cv2.waitKey(0)
            
            digit = GameParser.__find_match(hashes, labels, average_hash(temp))
            match = '{' + digit +'}'
    
        #print(match)
        #cv2.imshow('test', thresh)
        #cv2.waitKey(0)
    
        return match

    def parse_counters(self, image, training=False):
        mask = cv2.inRange(image, Cell.BLUE, Cell.BLUE)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        i = 0
        remaining = None
        for contour in contours:
            if (cv2.contourArea(contour) > GameParser.__area_threshold and
                cv2.matchShapes(contour, self.__counter_contour, 1, 0) < GameParser.__counter_match_threshold):
                
                i += 1
                if i > 2:
                    raise RuntimeError('counters parsed incorrectly')
                
                if i == 2:
                    x, y, w, h = cv2.boundingRect(contour)
                    y = round(y+h*0.35)
                    h = round(h*0.65)
    
                    cropped = image[y: y+h, x: x+w]
                    thresh = cv2.cvtColor(np.where(cropped==Cell.BLUE, 255, 0).astype(np.uint8), cv2.COLOR_BGR2GRAY)
                    thresh = cv2.resize(thresh, GameParser.__counter_dims, interpolation=cv2.INTER_AREA)
    
                    hashed = average_hash(thresh)
    
                    if training:
                        remaining = hashed
                        
                    else:
                        hashes, labels = self.__counter_data
                        similarities = [np.sum(hashed != h) for h in hashes]
                        match = labels[np.argmin(similarities)]
                        
                        #print(match)
                        #cv2.imshow('test', thresh)
                        #cv2.waitKey(0)
                        
                        remaining = match

        return remaining

    def parse_clicked(self, grid, left_click_cells, right_click_cells):
        left_click_cells.sort(key=lambda cell: cell.grid_coords, reverse=True)
        image = None
        if left_click_cells:
            self.click_cells(left_click_cells, 'left')
          
            time.sleep(0.1) 
            image = self.__window.screenshot()
            self.__parse_clicked_cells(image, left_click_cells)

        if right_click_cells:
            right_click_cells.sort(key=lambda cell: cell.grid_coords, reverse=True)
            self.click_cells(right_click_cells, 'right')
            
            #min_row = min(right_click_cells, key=lambda cell: cell.grid_coords[0]).grid_coords[0]
            time.sleep(0.1) 
            #time.sleep(max(1.5, 0.12*(grid.rows-min_row)))
            image = self.__window.screenshot()
            self.__parse_clicked_cells(image, right_click_cells)
            
        if image is None:
            image = self.__window.screenshot()
            
        return self.parse_counters(image)

    def click_cells(self, cells, button):
        cells.sort(key=lambda cell: cell.grid_coords, reverse=True)
        for cell in cells:
            self.__window.click_cell(cell, button)

    def __parse_clicked_cells(self, image, cells):
        for cell in cells:
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
                return
                #raise RuntimeError('cell must be blue or black after click')
    
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
                if (abs(rect1[0]+rect1[2]-rect2[0]) < self.__hex_width*0.75 and
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
        _, thresh = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 120, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        rects = []
        for contour in contours:
            if cv2.contourArea(contour) < 1000:
                x, y, w, h = cv2.boundingRect(contour)
                coords = np.asarray([x+w//2, y+h//2])
                if (self.__x_min-2*self.__hex_width < x <  self.__x_max+2*self.__hex_width and
                    self.__y_min-2*self.__hex_height < y < self.__y_max+self.__hex_height and 
                    np.linalg.norm(coords-grid.nearest_cell(coords).image_coords) < 140):
                    rects.append((x, y, w, h))
                    
        bounding_boxes = self.__merge_rects(rects)

        #for x, y, w, h in bounding_boxes:
        #    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255))
        
        #cv2.imshow('test', image)
        #while cv2.waitKey(0) != 27:
        #    pass
        
        parsed = []
        for x, y, w, h in bounding_boxes:
            box_coords = np.asarray([x + w//2, y + h//2])
            nearest_cell = grid.nearest_cell(box_coords)
            nearest_coords = nearest_cell.image_coords

            delta_x = box_coords[0] - nearest_coords[0]
            delta_y = nearest_coords[1] - box_coords[1]
            theta = (90 - np.degrees(np.arctan2(delta_y, delta_x))) % 360
            angle = GameParser.__angles[np.argmin(np.abs(GameParser.__angles-theta))]

            cropped = thresh[y: y+h, x: x+w]
            digit = self.__parse_grid_digit(255-cropped, angle, training)
            
            if training:
                parsed.append(digit)
            else:
                row, col = nearest_cell.grid_coords
                grid.add_constraint(row, col, digit, angle)
        
        return parsed
    
    def __parse_grid_digit(self, thresh, angle, training=False):  
        if np.count_nonzero(thresh==0) < 20:
            return None

        thresh = GameParser.__process_image(thresh)
        
        if angle in [120, 240]:
            thresh = cv2.flip(cv2.flip(thresh, 1), 0)
            
        elif angle in [90, 270]:
            centre = tuple(np.array(thresh.shape[1::-1]) / 2)
            rotation_matrix = cv2.getRotationMatrix2D(centre, angle, 1.0)
            thresh = cv2.warpAffine(thresh, rotation_matrix, thresh.shape[1::-1],
                                    flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
        
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
        
        #print(match)
        #cv2.imshow('test', thresh)
        #cv2.waitKey(0)
        
        return match

    @staticmethod
    def __process_image(image):
        coords = np.argwhere(image==0)
        x0, x1 = coords[:,0].min(), coords[:,0].max()
        y0, y1 = coords[:,1].min(), coords[:,1].max()

        image = cv2.resize(image[x0:x1+1, y0:y1+1], (53, 45), interpolation=cv2.INTER_AREA)
        image = cv2.medianBlur(image, 3)
        
        return image
    
    @staticmethod
    def __find_match(hashes, labels, hashed):
        similarities = [np.sum(hashed != h) for h in hashes]
        return labels[np.argmin(similarities)]
