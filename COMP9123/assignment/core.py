
from enum import Enum

class Colour(Enum):
    WHITE = "."
    BLACK = "*"
    
    def get_texture(self):
        """Returns a character representation of this colour, to be used for rendering as a string."""
        return self.value
    
    def __repr__(self):
        """Returns a string representation of this colour."""
        return self.value
    

def is_texture(c):
    for name, colour in Colour.__members__.items():
        if colour.value == c:
            return True
    return False

