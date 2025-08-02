
import sys
from core import Colour
from core import is_texture
import copy

def main():
    # This is here for you to optionally use for your own testing / running. 
    # This function will NOT be tested. Feel free to experment here.
    print("Enter a string representation for a image, followed by EOF or \"end\".")
    input_bmp1 = read_bmp_from_stream(sys.stdin)
    input_bmp2 = read_bmp_from_stream(sys.stdin)
    # print(input_bmp)
    # print(input_bmp.get_size())
    # input_bmp.blacken_north_west_quadrant()
    # print(input_bmp.to_tree_string())
    # print(input_bmp.count_pixels(Colour.BLACK))
    # input_bmp.invert_colours()
    # input_bmp.set_pixel(0,0,Colour.BLACK)
    print(QuadtreeImage.compute_overlay(input_bmp1,input_bmp2).to_tree_string())
    print("----------------------")
    print(input_bmp1.to_tree_string())

# convenience function for constructing quadtrees from string representations of 
# images from stdin
def read_bmp_from_stream(stream):
    lines = []
    for line in stream:
        line = line.rstrip("\n")
        if line == "end":
            break
        else:
            lines.append(line)
    return quadtree_image_from_string("\n".join(lines))

class QuadtreeImage:
    
    def __init__(self, size, colour = Colour.WHITE):
        """Constructs a quadtree image with height and width equal to the specified size, 
        and every pixel initialized to the given colour. The specified size must be a power 
        of 2, and must be greater than zero. If no colour is specified, default colour is 
        white."""
        # only supporting power-of-2 dimensions
        if not _power_of_two(size):
            raise ValueError("Size not power of 2.")
        self._x = 0
        self._y = 0
        self._size = size
        self._leaf = True
        self._colour = colour
        self._north_west = None
        self._north_east = None
        self._south_west = None
        self._south_east = None
    
    # specifying location only supported internally
    @classmethod
    def _construct_at_location(cls, x, y, size, colour):
        bmp = QuadtreeImage(size, colour)
        bmp._x = x
        bmp._y = y
        return bmp
    
    # combining quads to form tree only supported internally, assumes well-positioned
    @classmethod
    def _construct_from_quadrants(cls, x, y, size, quads):
        bmp = QuadtreeImage._construct_at_location(x, y, size, Colour.WHITE)
        bmp._north_west = quads[0]
        bmp._north_east = quads[1]
        bmp._south_west = quads[2]
        bmp._south_east = quads[3]
        bmp._leaf = False
        return bmp
    
    # for any basic task which needs to be repeated all four quadrants
    def _quadrants(self):
        return [ self._north_west, self._north_east, self._south_west, self._south_east ]
    
    # retrieves the quadrant within which the specified location lies
    def _quadrant_of(self, x, y):
        for quad in self._quadrants():
            if quad.contains_point(x, y):
                return quad
        return None
    
    def contains_point(self, x, y):
        """Returns True of this QuadtreeImage contains the location specified by the input 
        coordinates."""
        return self._x <= x \
            and self._y <= y \
            and x < self._x + self._size \
            and y < self._y + self._size
    
    def get_size(self):
        """Returns the height and width of this quadtree image."""
        return self._size
    
    #########################################################################
    ## Assignment questions from here on in
    #########################################################################
    
    # helper function
    def recustructure(self, node):
        if node._leaf:
            return
        quads = node._quadrants() 
        black_colours = 0
        if all(q and q._leaf for q in quads):
            for q in quads:
                if q._colour == Colour.BLACK:
                    black_colours += 1
            if black_colours % 4 == 0:
                node._leaf = True
                node._colour = Colour.BLACK if black_colours == 4 else Colour.WHITE
                node._north_west = node._north_east = node._south_west = node._south_east = None

    # helper function 2
    def divide(self):
        half = self._size // 2
        self._north_west = QuadtreeImage._construct_at_location(self._x, self._y, half, self._colour)
        self._north_east = QuadtreeImage._construct_at_location(self._x + half, self._y, half, self._colour)
        self._south_west = QuadtreeImage._construct_at_location(self._x, self._y + half, half, self._colour)
        self._south_east = QuadtreeImage._construct_at_location(self._x + half, self._y + half, half, self._colour)
        self._leaf = False

    def blacken_north_west_quadrant(self):
        """Sets the colour of every pixel in the north-west quadrant of this quadtree 
        image to black."""
        # TODO: implement this
        def blacken(node: QuadtreeImage):
            # if leaf, just black it 
            if node._leaf:
                node._colour = Colour.BLACK
            # if not leaf, go into it until finding the leaf
            else:
                for i in node._quadrants():
                    if i:
                        blacken(i) 
                self.recustructure(node)

        # if it is a whole leaf, just split it into 4 piece
        if self._leaf:
            self.divide()
        if self._north_west:
            blacken(self._north_west)
            self.recustructure(self)

    



    def count_pixels(self, colour):
        """Counts the number of pixels of the given colour in the image represented 
        by this quadtree."""
        # TODO: implement this
        def count_part_pixels(node: QuadtreeImage, colour):
            if node._leaf: 
                return node._size * node._size if node._colour == colour else 0
            res = 0 
            for i in node._quadrants():
                res += count_part_pixels(i,colour)
            return res
        return count_part_pixels(self,colour)
    
    def invert_colours(self):
        """Inverts the colours in the image represented by this quadtree, 
        i.e. turns every black pixel white and every white pixel black."""
        # TODO: implement this
        if self._leaf :
            self._colour = Colour.BLACK if self._colour == Colour.WHITE else Colour.WHITE
            return 
        for i in self._quadrants():
            i.invert_colours()
        return 
    
    def set_pixel(self, x, y, colour):
        """Sets the colour of a single pixel at the specified location to the 
        given colour."""
        # TODO: implement this
        if not self.contains_point(x,y):
            return 
        if self._leaf:
            if self._colour == colour:
                return 
            if self._size == 1:
                self._colour = colour
            else:
                self.divide()
        for i in self._quadrants():
            if i:
                i.set_pixel(x, y, colour)
        self.recustructure(self)
        return 
    
    
    @classmethod
    def compute_overlay(cls, bmp1, bmp2):
        """Constructs and returns the overlay of the two given quadtree images. The overlay of 
        two images is the image which has a black pixel at every location at which either 
        input image has a black pixel, and a white pixel at every location at which both 
        input images have a white pixel. Can be thought of as the bitwise OR of two images."""
        # TODO: implement this
        total_size = bmp1._size
        if bmp1._leaf and bmp2._leaf:
            colour = Colour.BLACK if bmp1._colour == Colour.BLACK or bmp2._colour == Colour.BLACK else Colour.WHITE
            return cls._construct_at_location(x = bmp1._x, y = bmp1._y, size = bmp1._size, colour = colour)
        if bmp1._leaf:
            bmp1.divide()
        if bmp2._leaf:
            bmp2.divide()
        res = []
        for i in range(0, 4, 1):
            res.append(cls.compute_overlay(bmp1._quadrants()[i], bmp2._quadrants()[i]))
        res = cls._construct_from_quadrants(bmp1._x, bmp1._y, total_size, res)
        res.recustructure(res)
        return res

        '''
        def area_colour(bmp1, bmp2):
            if bmp1._leaf and bmp2._leaf:
                if bmp1._colour == Colour.BLACK or bmp2._colour == Colour.BLACK:
                    return Colour.BLACK
                return Colour.WHITE
        res = copy.deepcopy(bmp1)
        if bmp1._leaf and bmp2._leaf: 
            res._colour = area_colour(bmp1, bmp2)
            return res
        
        if bmp1._leaf:
            bmp1.divide()
        if bmp2._leaf:
            bmp2.divide()

        for i in range(0,4,1):
            res = cls.compute_overlay(bmp1._quadrants()[i], bmp2._quadrants()[i])
        res.recustructure(res)
        return res
        
        '''
        
            
        
                
    
    ###########################################################
    ## End of assignment questions
    ###########################################################
    
    ###########################################################
    ## You do not need to concern yourself with the code beyond this point
    ###########################################################
    
    
    def to_tree_string(self):
        """Returns a string representation of the tree structure of this quadtree image. 
        The string representation is similar to the representation returned by __repr__,
        but with boxing interspersed to indicate the boundaries of the regions represented by 
        leaf nodes in the quadtree."""
        canvas = [[" "]*(2*self._size + 1) for _ in range(2*self._size + 1)]
        self._print_tree_to_canvas(canvas, 2*self._x, 2*self._y)
        rows = []
        for row in canvas:
            rows.append("".join(row))
        return "\n".join(rows)

    def _print_tree_to_canvas(self, canvas, x_offset, y_offset):
        CORNER = '+' ; V_WALL = '|' ;H_WALL = '-' ; FILLER = ' '
        if self._leaf:
            # top left is 2x - x_offset, 2y - y_offset
            top_y = 2*self._y - y_offset
            bottom_y = 2*self._y + 2*self._size - y_offset
            left_x = 2*self._x - x_offset
            right_x = 2*self._x + 2*self._size - x_offset
            # corners
            canvas[top_y][left_x] = CORNER
            canvas[top_y][right_x] = CORNER
            canvas[bottom_y][left_x] = CORNER
            canvas[bottom_y][right_x] = CORNER
            # top
            for i in range(left_x + 1, right_x):
                if canvas[top_y][i] != CORNER:
                    canvas[top_y][i] = H_WALL
            # bottom
            for i in range(left_x + 1, right_x):
                if canvas[bottom_y][i] != CORNER:
                    canvas[bottom_y][i] = H_WALL
            # left
            for i in range(top_y + 1, bottom_y):
                if canvas[i][left_x] != CORNER:
                    canvas[i][left_x] = V_WALL
            # right
            for i in range(top_y + 1, bottom_y):
                if canvas[i][right_x] != CORNER:
                    canvas[i][right_x] = V_WALL
            # fill every odd coordinate in interior
            for i in range(top_y + 1, bottom_y, 2):
                for j in range(left_x + 1, right_x, 2):
                    canvas[i][j] = self._colour.get_texture()
        else:
            for quad in self._quadrants():
                quad._print_tree_to_canvas(canvas, x_offset, y_offset)
    
    
    def __repr__(self):
        """Returns a string representation of this image. The string representation consists
        of a newline-separated sequence of rows, where each row consists of a sequence of 
        characters which each encode the colour of a pixel.
        For a string representation which depicts the quadtree structure of this image,
        see to_tree_string."""
        canvas = [[Colour.WHITE.get_texture()]*self._size for _ in range(self._size)]
        self._print_to_canvas(canvas, self._x, self._y)
        rows = []
        for row in canvas:
            rows.append("".join(row))
        return "\n".join(rows)

    def _print_to_canvas(self, canvas, x_offset, y_offset):
        if self._leaf:
            for i in range(self._y, self._y + self._size):
                for j in range(self._x, self._x + self._size):
                    canvas[i - y_offset][j - x_offset] = self._colour.get_texture()
        else:
            for quad in self._quadrants():
                quad._print_to_canvas(canvas, x_offset, y_offset)


def quadtree_image_from_string(bmp_string):
    """Constructs a quadtree from the image represented by the input string. 
    Fails with a ValueError if the input string does not properly encode a valid 
    image."""
    _validate_bmp_string(bmp_string)
    return _from_row_strings(0, 0, bmp_string.splitlines())

# recursive helper method for fromString
def _from_row_strings(x, y, rows):
    size = len(rows)
    if not any(Colour.BLACK.get_texture() in row for row in rows):
        # all white
        return QuadtreeImage._construct_at_location(x, y, size, Colour.WHITE)
    elif not any(Colour.WHITE.get_texture() in row for row in rows):
        # all black
        return QuadtreeImage._construct_at_location(x, y, size, Colour.BLACK)
    else:
        x_mid = x + size//2
        y_mid = y + size//2
        # split rows into quadrants
        nw_rows = _quad_row_strings(0, 0, rows)
        ne_rows = _quad_row_strings(size//2, 0, rows)
        sw_rows = _quad_row_strings(0, size//2, rows)
        se_rows = _quad_row_strings(size//2, size//2, rows)
        # build each subtree
        north_west = _from_row_strings(x, y, nw_rows) 
        north_east = _from_row_strings(x_mid, y, ne_rows)
        south_west = _from_row_strings(x, y_mid, sw_rows)
        south_east = _from_row_strings(x_mid, y_mid, se_rows)
        # combine
        quads = [ north_west, north_east, south_west, south_east ]
        return QuadtreeImage._construct_from_quadrants(x, y, size, quads)

# extracts row strings for quadrant from row strings for image
def _quad_row_strings(x_rel, y_rel, rows):
    size = len(rows)
    # sublist selects rows, substring selects columns
    return list(map(lambda row: row[x_rel:x_rel + size//2], rows[y_rel:y_rel + size//2]))

# does nothing if input valid, communicates invalidity via errors
def _validate_bmp_string(bmp_string):
    rows = bmp_string.splitlines()
    if len(rows) == 0:
        raise ValueError("Empty image string.")
    elif not _power_of_two(len(rows)):
        raise ValueError("Number of rows not a power of 2.")
    elif not _power_of_two(len(rows[0])):
        raise ValueError("Row width not a power of 2.")
    else:
        # using first row to determine row width
        width = len(rows[0])
        for i in range(1, len(rows)):
            if len(rows[i]) != width:
                raise ValueError("Row " + str(i) + " not same width as other rows.")
        for row in rows:
            if not all(is_texture(c) for c in list(row)):
                ic = next(filter(lambda c : not is_texture(c), list(row)))
                raise ValueError("Illegal character detected: '" + ic + "'")
        if len(rows) != width:
            raise ValueError("Number of rows not equal to row width.")


def _power_of_two(n):
    x = 1
    while x < n:
        x *= 2
    if x == n:
        return True
    else:
        return False

if __name__ == "__main__":
    main()

