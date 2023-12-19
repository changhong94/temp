import torchvision
import torch
import numpy
torch.manual_seed(0)

model = torchvision.models.resnet18(pretrained=True)
torch.backends.cudnn.enabled = False
model = model.eval()
input_cpu = torch.rand(1, 3, 224, 224)
model_cuda = model.to("cuda")
input_cuda = input_cpu.to("cuda")
raw_output = model(input_cuda)
out_cuda = raw_output.to("cpu")

numpy_arr = out_cuda.detach().numpy()
#numpy.save("numpy_arr.npy", numpy_arr)
local = numpy.load("numpy_arr.npy")

print(numpy_arr - local)
