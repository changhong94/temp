import torch
import numpy
import os
torch.manual_seed(0)

from transformers import pipeline

torch.cuda.is_available()
generator = pipeline('fill-mask',model = 'prajjwal1/bert-tiny', device=0)
print("------------------")
out = generator("Today is a [MASK] day.")

#out_cuda = out.to("cpu")
print(out)
#numpy_arr = out_cuda.detach().numpy()
#numpy.save("numpy_arr.npy", numpy_arr)
#local = numpy.load("numpy_arr.npy")

#print(numpy_arr - local)
