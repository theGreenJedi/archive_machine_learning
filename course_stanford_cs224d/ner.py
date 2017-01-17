##
# Utility functions for NER assignment
# Assigment 2, part 1 for CS224D
##

from numpy import *




def save_predictions(y, filename):
    """Save predictions, one per line."""
    with open(filename, 'w') as fd:
        fd.write("\n".join(map(str, y)))
        fd.write("\n")