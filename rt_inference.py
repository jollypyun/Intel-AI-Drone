import cv2
import numpy as np
import os, sys
from time import strftime, localtime
import random
from openvino.inference_engine import IENetwork, IEPlugin

plugin = IEPlugin("CPU", "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64")
plugin.add_cpu_extension("/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so")

model_xml = '/home/intel/my_model/cifar.xml'
model_bin = '/home/intel/my_model/cifar.bin'
print('Loading network files:\n\t{}\n\t{}'.format(model_xml, model_bin))

net = IENetwork(model=model_xml, weights=model_bin)

supported_layers = plugin.get_supported_layers(net)
not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
if len(not_supported_layers) != 0:
	print("Following layers are not supported by the plugin for specified device {}:\n {}".formaT(plugin.device, ', '.join(not_supported_layers)))
	print("Please try to specify cpu extensions library path in sample's command line parameters using –l or --cpu_extension command line argument")
	sys.exit(1)

net.batch_size = 1

input_blob = next(iter(net.inputs))
out_blob = next(iter(net.outputs))

exec_net = plugin.load(network=net)

cap = cv2.VideoCapture(0)
if not cap.isOpened(): sys.exit('camera error')

name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck', 'rock', 'paper', 'scissors', 'LG', 'KISTI', 'Intel']
# name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']	
while True:
	ret, frame = cap.read()
	if not ret: continue 

	rows, cols, channels = frame.shape
	width = cols
	height = rows
	length = min(width, height)
	pt = [60,60]
	if width < height: pt[1] += int((height - length) / 2)
	else: pt[0] += int((width - length) / 2)
	green = (0, 255, 0)  #BGR
	length -= 120
	cv2.rectangle(frame, (pt[0], pt[1]), (pt[0]+length, pt[1]+length), green, 4)

	ch = cv2.waitKey(1) & 0xFF
	if ch == 27: break

	mid_frame = frame[pt[1]:pt[1]+length, pt[0]:pt[0]+length]
	cut_frame = cv2.resize(mid_frame, (32, 32))
	img = cut_frame
	height, width, _ = img.shape
	n, c, h, w = net.inputs[input_blob].shape
	img2 = img
	if height != h or width != w:
		img2 = cv2.resize(img, (w, h))

	img2 = img2.transpose((2, 0, 1))  
	images = np.ndarray(shape=(n, c, h, w))
	images[0] = img2

	res = exec_net.infer(inputs={input_blob: images})

	probs = res[out_blob]
	print('Top 3 results:')
	top_ind = np.argsort(probs)[0][:-4:-1]
	print(probs, top_ind)
	for id in top_ind:
		print('label #{} : {:0.2f}'.format(id, probs[0][id]))

	
	sorted_probs = np.argsort(probs)[0][::-1]
	id = sorted_probs[0]
	#for i in range(3): 
	#	print(name[sorted_probs[i]])
	#	print(probs[0][sorted_probs[i]])
	prob = probs[0][id]
	inf_res = ''
	if prob >= 0.6: inf_res = name[id]

	if inf_res != '':
		cv2.putText(frame, inf_res, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,255,0), lineType=cv2.LINE_AA)
	cv2.imshow('view', frame)
