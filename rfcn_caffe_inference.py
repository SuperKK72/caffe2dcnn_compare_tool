import os
os.environ['GLOG_minloglevel'] = '3'
import warnings
warnings.filterwarnings("ignore")
import caffe
import numpy as np
import sys
import cv2
net_name = "rfcn"
prototxt_path = "./caffemodel/rfcn/deploy.prototxt"
caffemodel_path = "./caffemodel//rfcn/deploy.caffemodel"
net = caffe.Net(prototxt_path,caffemodel_path,caffe.TEST)
print("----------------------------------"+net_name+"----------------------------------")
for layer_name, blobs in net.blobs.items():
    print(layer_name + '\t' + str(blobs.data.shape))

caffe_blobs = net.blobs
caffe_blobs_names = list(net.blobs.keys())
input_node = net.inputs[0]
input_node1 = net.inputs[1]
output_node = net.outputs[0]
print(input_node)
print(net.outputs)

input_shape = net.blobs[input_node].data.shape
n = input_shape[0]
c = input_shape[1]
h = input_shape[2]
w = input_shape[3]

image_path = "./dog.jpg"
image=cv2.imread(image_path, -1)
imgInfo = image.shape
# print(imgInfo)
img_h = image.shape[0]
img_w = image.shape[1]
if(img_h != h or img_w != w):
    image=cv2.resize(image, (h, w))
X=np.array(image).astype(np.float32)
X=X.reshape((n, c, h, w))
net.blobs[input_node].data[...] = X

outputs = net.forward()
output_blob = outputs[output_node]
output = output_blob.flatten()
output_shape = output_blob.shape
output_dim = len(output_shape)
print(output_shape)
print(type(output_blob))
print(output)

