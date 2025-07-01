"""
Extract bed level position from simulation and save it in file.
It should be done only if a modification in the code is expected
to modify the bed level position.
"""

from os import popen
import os.path
import numpy as np
from fluidfoam import readof as rdf


prec = 8

filePath = "./dataZb01.txt"


def r(x):
    return str(round(x, prec))


Xb, Yb, Zb = rdf.readmesh("./", time_name="0.1", boundary="bed")

if os.path.isfile(filePath):
    print(f"\nfile {filePath} already exists, "
          + "remove it before saving data")
else:
    with open(filePath, "w") as data:
        data.write("# x [m]; zb[m]\n")
        for xb, zb in zip(Xb, Zb):
            line = r(xb) + ";" + r(zb) + "\n"
            data.write(line)
