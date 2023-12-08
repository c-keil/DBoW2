import numpy as np
import time

n_samples = 10_000_000
desc1 = np.ones((256), dtype = np.float32)
desc2 = np.ones((256), dtype = np.float32)

print("timing sp for {} distance calcualtions".format(n_samples))
start_time = time.time()
for _ in range(n_samples):
    dist = ((desc1 - desc2)**2).sum()
end_time = time.time()
print("total time = {}".format(end_time - start_time))

split = 10
n_samp = int(n_samples/split)
descs = np.random.rand(n_samp, 256).astype(np.float32)
descs2 = np.random.rand(n_samp, 256).astype(np.float32)
# print("descriptor subset has {} bytes".format(descs.nbytes))
print("uses {} Gb memory".format(split * descs.nbytes/(1024**3)))
# print("each descriptor element has {} bytes".format(descs.itemsize))
# print(descs.shape)
print()
print("timing optimal version")
start_time = time.time()
for _ in range(split):
    dist = ((descs - descs2)**2).sum()
end_time = time.time()
print("total time = {}".format(end_time - start_time))