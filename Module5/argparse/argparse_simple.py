import math
import argparse

parser = argparse.ArgumentParser(description = "Calculate volume of a Cylinder")
parser.add_argument('-r', '--radius', type = int, metavar = '', required = True, help = 'Radius of Cylinder')
parser.add_argument('-H', '--height', type = int, metavar = '', required = True, help = 'Height of Cylinder')
args = parser.parse_args()

def cylinder_volume(radius, height):
    vol = math.pi * radius ** 2 * height
    return vol

# Call the function
volume  = cylinder_volume(args.radius, args.height)
print(volume)