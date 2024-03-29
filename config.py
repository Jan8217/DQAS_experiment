import torch

# selecting device for pytorch (either gpu or cpu)
is_cuda = True
gpuid = 0

# selecting float32 for data type (can also be float64)
dtype = torch.float64
print(torch.__version__)
print(torch.cuda.is_available())
# default grad_enabled
grad_enabled_bool = False

if is_cuda:
	device = torch.device("cpu")
else:
	device = torch.device("cuda:"+ str(gpuid))

#print(torch.__version__)
#print(torch.cuda.is_available())
#print(torch.version.cuda)