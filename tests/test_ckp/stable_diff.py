from unittest import result
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
import numpy
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context
# print(torch.cuda.memory_summary(device=None, abbreviated=False))

model_id = "CompVis/stable-diffusion-v1-2"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompt = "a photo of a robot that plays the piano"
with autocast("cuda"):
    image_cuda = pipe(prompt, guidance_scale=7.5).images[0]

data_cuda = numpy.asarray(image_cuda)
#data_reshaped = data_cuda.reshape(data_cuda.shape[0], -1)
#numpy.save("expected.npy",data_reshaped)
#b = numpy.load("expected.npy")
print(image_cuda)
#print("diff:", data_reshaped - b)
#print(ret.images[0])
#image_cuda.save("m_result.png")
