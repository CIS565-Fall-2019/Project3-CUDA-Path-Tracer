# Simple script for generating random scenes based on input.
# Mostly for stress testing the render

import sys, getopt, random

RES = [2000, 1200]
FOVY = 45
ITERATIONS = 5000
DEPTH = 8
FILE = 'cornell'
EYE  =  [0, 5, 10.5]
LOOKAT = [0, 5, 0]
UP = [0, 0, 0]

def rng():
    return random.random()

def rngi(low, high):
    return random.randint(low, high)

def rngr(low, high):
    return random.random() + 0.001 + random.randrange(low, high)

def genCamera():
    cam = """
CAMERA
RES         {} {}
FOVY        {}
ITERATIONS  {}
DEPTH       {}
FILE        {}
EYE         {} {} {}
LOOKAT      {} {} {}
UP          {} {} {}
""".format(
    RES[0], RES[1],
    FOVY,
    ITERATIONS,
    DEPTH,
    FILE,
    EYE[0], EYE[1], EYE[2],
    LOOKAT[0], LOOKAT[1], LOOKAT[2],
    UP[0], UP[1], UP[2]
)
    return cam

def genMaterials(num_mats):
    mats = ''

    # Put in wall and light mats for now
    mats +="""
// Emissive material (light)
MATERIAL 0
RGB         1 1 1
SPECEX      0
SPECRGB     0 0 0
REFL        0
REFR        0
REFRIOR     0
EMITTANCE   5

// White diffuse
MATERIAL 1
RGB         .95 .95 .95
SPECEX      0
SPECRGB     .95 .95 .95
REFL        0
REFR        0
REFRIOR     0
EMITTANCE   0
"""

    for i in range(num_mats):
        mats += """
MATERIAL    {}
RGB         {} {} {}
SPECEX      {}
SPECRGB     {} {} {}
REFL        {}
REFR        {}
REFRIOR     {}
EMITTANCE   {}
""".format(
    i + 2,
    rng(), rng(), rng(),
    rng(),
    rng(), rng(), rng(),
    0,
    0,
    rng(),
    0)
    
    return mats

def genObjects(num_objs, num_mats):
    objs = ''

    # Put in the walls and lights for now
    objs += """
// Ceiling light 1
OBJECT 0
cube
material 0
TRANS       8 10 0
ROTAT       0 0 0
SCALE       1.5 .3 1.5

// Ceiling light 2
OBJECT 1
cube
material 0
TRANS       -8 10 0
ROTAT       0 0 0
SCALE       1.5 .3 1.5

// Ceiling light 3
OBJECT 2
cube
material 0
TRANS       0 10 0
ROTAT       0 0 0
SCALE       9 .3 0.5

// Floor
OBJECT 3
cube
material 1
TRANS       0 0 0
ROTAT       0 0 0
SCALE       20 .01 20

// Ceiling
OBJECT 4
cube
material 1
TRANS       0 10 0
ROTAT       0 0 90
SCALE       .01 20 20

// Back wall
OBJECT 5
cube
material 1
TRANS       0 5 -10
ROTAT       0 90 0
SCALE       .01 20 20

// Left wall
OBJECT 6
cube
material 1
TRANS       -10 5 0
ROTAT       0 0 0
SCALE       .01 20 20

// Right wall
OBJECT 7
cube
material 1
TRANS       10 5 0
ROTAT       0 0 0
SCALE       .01 20 20
"""

    for i in range(num_objs):
        objs +="""
OBJECT   {}
sphere
material {}
TRANS    {} {} {}
ROTAT    {} {} {}
SCALE    {} {} {}
""".format(
    i + 8,
    rngi(2, num_mats + 2 - 1),
    rngi(-9, 9), rngi(1, 9), rngi(-9, 9),
    rngi(-180, 180), rngi(-180, 180), rngi(-180, 180),
    rngr(0, 2), rngr(0, 2), rngr(0, 2),
)
    return objs


def main(argv):
   num_objects = 0
   output = ''

   try:
      opts, args = getopt.getopt(argv,"n:o:",["num_objects=", "output="])
   except getopt.GetoptError:
      print('test.py -i <inputfile> -o <outputfile>')
      sys.exit(2)

   for opt, arg in opts:
      if opt in ("-n", "--num_objects"):
         num_objects = int(arg)
      elif opt in ("-o", "--output"):
         outputfile = arg

   NUM_OBJECTS = 20
   NUM_MATS = 5

   print(genMaterials(NUM_MATS))
   print(genCamera())
   print(genObjects(NUM_OBJECTS, NUM_MATS))

if __name__ == "__main__":
   main(sys.argv[1:])