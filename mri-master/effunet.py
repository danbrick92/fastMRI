# Modifications - Dan Brickner - modified in_channels = 1 and changed init_hook.torch.rand([1,3,576,576]) -> torch.rand([1,1,576,576])
# Original file adapted from https://github.com/pranshu97/effunet/blob/main/effunet/effunet.py; cited in research paper

import torch
import torch.nn as nn
import torchvision.transforms as T
from efficientnet_pytorch import EfficientNet


# Utility Functions for the model

# Hook functions to get values of intermediate layers for cross connection
encoder_out = []
def hook(_, input, output):
	global encoder_out
	encoder_out.append(output) # stores values of each layers in encoder_out
def inhook(_,inp,output):
	global encoder_out
	encoder_out.append(inp[0])

# Initialize encoder to get shapes of selected layers
shapes = []
def init_hook(model,device,concat_input):
	global shapes, encoder_out
	shapes = []
	encoder_out = []

	if concat_input:
		model._conv_stem.register_forward_hook(inhook)

	hooks = []
	for i in range(len(model._blocks)):
		hooks.append(model._blocks[i].register_forward_hook(hook)) #register hooks
	
	image = torch.rand([1,1,576,576]) # doesn't matter if input size is different from this during training.
	image = image.to(device)
	out = model(image) # generate hook values to get shapes

	shapes = [encoder_out[i].shape for i in range(len(encoder_out)-1) if encoder_out[i].shape[2] != encoder_out[i+1].shape[2]]

	if concat_input:
		indices = [i-1 for i in range(1,len(encoder_out)-1) if encoder_out[i].shape[2] != encoder_out[i+1].shape[2]]
		shapes.append(encoder_out[-1].shape)
		indices.append(len(encoder_out[1:])-1)
	else:
		indices = [i for i in range(0,len(encoder_out)-1) if encoder_out[i].shape[2] != encoder_out[i+1].shape[2]]
		shapes.append(encoder_out[-1].shape)
		indices.append(len(encoder_out)-1)

	for i,_hook in enumerate(hooks):
		if i not in indices:
			_hook.remove()

	shapes = shapes[::-1] 
	encoder_out=[]

def double_conv(in_,out_,drop): # Double convolution layer for decoder 
	conv = nn.Sequential(
		nn.Conv2d(in_,out_,kernel_size=3,padding=(1,1)),
		nn.BatchNorm2d(out_),
		nn.ReLU(inplace=True),
		nn.Conv2d(out_,out_,kernel_size=3,padding=(1,1)),
		nn.BatchNorm2d(out_),
		nn.ReLU(inplace=True),
		nn.Dropout(drop)
		)
	return conv

class EffUNet(nn.Module):

	def __init__(self,model='b0',out_channels=1,dropout=0.1,freeze_backbone=True,pretrained=True,concat_input=True,device='cuda'):
		super(EffUNet,self).__init__()
		global layers, shapes

		if model not in set(['b0','b1','b2','b3','b4','b5','b6','b7']):
			raise Exception(f'{model} unavailable.')
		if pretrained:
			self.encoder = EfficientNet.from_pretrained(f'efficientnet-{model}')
		else:
			self.encoder = EfficientNet.from_name(f'efficientnet-{model}', in_channels=1)

		# Disable non required layers by replacing them with identity to save time and memory
		self.encoder._conv_head=torch.nn.Identity()
		self.encoder._bn1=torch.nn.Identity()
		self.encoder._avg_pooling=torch.nn.Identity()
		self.encoder._dropout=torch.nn.Identity()
		self.encoder._fc=torch.nn.Identity()
		self.encoder._swish=torch.nn.Identity()

		if not concat_input:
			self.encoder._conv_stem.stride=(1,1)

		if isinstance(device, str):
			self.device = torch.device(device)
		else:
			self.device = device
		self.encoder.to(self.device)

		# freeze encoder
		if freeze_backbone:
			for param in self.encoder.parameters():
				param.requires_grad = False

		# register hooks & get shapes
		init_hook(self.encoder,self.device,concat_input)

		# Building decoder
		self.decoder = torch.nn.modules.container.ModuleList()
		for i in range(len(shapes)-1):
			self.decoder.append(torch.nn.modules.container.ModuleList())
			self.decoder[i].append(nn.ConvTranspose2d(shapes[i][1],shapes[i][1]-shapes[i+1][1],kernel_size=2,stride=2).to(self.device))
			self.decoder[i].append(double_conv(shapes[i][1],shapes[i+1][1],dropout).to(self.device))

		#output layer
		self.out = nn.Conv2d(shapes[-1][1],out_channels,kernel_size=1).to(self.device)

	def forward(self, image):
		global layers,encoder_out

		# Encoder
		encoder_out=[]
		self.encoder(image) # required outputs accumulate in "encoder_out"
		#Decoder
		x = encoder_out.pop()
		for i in range(len(self.decoder)):
			x = self.decoder[i][0](x) # conv transpose
			prev = encoder_out.pop()
			prev = torch.cat([x,prev],axis=1) # concatenating 
			x = self.decoder[i][1](prev) # double conv
		
		#out
		x = self.out(x)
		return x

# img = torch.rand([1,3,576,576]).cuda()
# from time import time
# start=time()
# model = EffUNet()
# print(time()-start)
# # print(model)
# out = model(img)
# print(out.shape)
