import numpy as np
import argparse

parser = argparse.ArgumentParser("create_file")
parser.add_argument('filename')
args = parser.parse_args()

array = np.random.rand(10**8, 12)

np.save(args.filename, array)
