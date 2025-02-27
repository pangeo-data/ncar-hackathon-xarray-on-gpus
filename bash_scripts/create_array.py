import numpy as np

array = np.random.rand(10**8, 12)
np.save('/glade/derecho/scratch/maxjones/tmp/large_array.npy', array)
