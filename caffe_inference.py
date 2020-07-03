import os
os.environ['GLOG_minloglevel'] = '3'
import warnings
warnings.filterwarnings("ignore")
import caffe
import numpy as np
import sys
import cv2


network_config_file_path = './caffe_inference_config/' + sys.argv[1]
network_config_file = open(network_config_file_path)
network_config_parameters = []
for network_config_parameter in network_config_file:
    if len(network_config_parameter.strip()) != 0:
        # print(network_config_parameter)
        network_config_parameter = network_config_parameter[network_config_parameter.find('=')+1:].strip()
        if network_config_parameter[0] != '#':
            network_config_parameters.append(network_config_parameter)
network_config_file.close()

net_name = network_config_parameters[0]
prototxt_path = network_config_parameters[1]
caffemodel_path = network_config_parameters[2]
mean_file_path = network_config_parameters[3]
image_path = network_config_parameters[4]
val_B = float(network_config_parameters[5])
val_G = float(network_config_parameters[6])
val_R = float(network_config_parameters[7])
std = float(network_config_parameters[8])
output_path = network_config_parameters[9]


net = caffe.Net(prototxt_path,caffemodel_path,caffe.TEST)
print("----------------------------------"+net_name+"----------------------------------")
for layer_name, blobs in net.blobs.items():
    print(layer_name + '\t' + str(blobs.data.shape))
# print("-----------------------------------PARAMS------------------------------------")
# for layer_name, layer_params in net.params.items():
#     print(layer_name + '\t' + str(layer_params[0].data.shape) + str(layer_params[1].data.shape))

caffe_blobs = net.blobs
caffe_blobs_names = list(net.blobs.keys())
input_node = net.inputs[0]
output_node = net.outputs[0]

# output_node1 = net.outputs[1]
# output_node2 = net.outputs[2]
# print(net.outputs)

# transformer = caffe.io.Transformer({input_node: net.blobs[input_node].data.shape})
# transformer.set_transpose(input_node, (2,0,1))
# #transformer.set_channel_swap(input_node, (2,1,0))
# if(mean_file_path == "null"):
#     transformer.set_mean(input_node,np.array([val_B,val_G,val_R]))
# else:
#     transformer.set_mean(input_node, np.load(mean_file_path).mean(1).mean(1))
# #transformer.set_raw_scale(input_node, std)
# #img = caffe.io.load_image(image_path)
# img=cv2.imread(image_path, -1)
# #img = np.array(img).astype(np.float32)

input_shape = net.blobs[input_node].data.shape
n = input_shape[0]
c = input_shape[1]
h = input_shape[2]
w = input_shape[3]

image=cv2.imread(image_path, -1)
imgInfo = image.shape
# print(imgInfo)
img_h = image.shape[0]
img_w = image.shape[1]
if(img_h != h or img_w != w):
    image=cv2.resize(image, (h, w))
X=np.array(image).astype(np.float32)
if(c == 3):
    if(mean_file_path != "null"):
        mean_blob = caffe.proto.caffe_pb2.BlobProto()
        mean_blob.ParseFromString(open(mean_file_path, 'rb').read())
        mean_npy = caffe.io.blobproto_to_array(mean_blob)
        # print(mean_npy.shape)
        # print(X.shape)
        X = X.transpose(2, 0, 1)
        X = X.flatten()
        mean_npy = mean_npy.flatten()
        pixel_num = n * c * h * w
        for i in range(pixel_num):
            X[i] -= mean_npy[i]
        X *= std
    else:
        X[:, :, 0] = (X[:, :, 0] - val_B) * std
        X[:, :, 1] = (X[:, :, 1] - val_G) * std
        X[:, :, 2] = (X[:, :, 2] - val_R) * std
        # print(X.shape)
        X = X.transpose((2, 0, 1))
X=X.reshape((n, c, h, w))
net.blobs[input_node].data[...] = X

# if(c == 3):
#     mean_list = []
#     if(mean_file_path != "null"):
#         mean_blob = caffe.proto.caffe_pb2.BlobProto()
#         mean_blob.ParseFromString(open(mean_file_path, 'rb').read())
#         mean_npy = caffe.io.blobproto_to_array(mean_blob)
#         mean_array = np.mean(mean_npy, -1)
#         mean_array = np.mean(mean_array, -1).flatten()
#         mean_list = list(mean_array)
#         val_B = mean_list[0]
#         val_G = mean_list[1]
#         val_R = mean_list[2]
#     X[:, :, 0] = (X[:, :, 0] - val_B) * std
#     X[:, :, 1] = (X[:, :, 1] - val_G) * std
#     X[:, :, 2] = (X[:, :, 2] - val_R) * std
#     #print(X.shape)
#     X = X.transpose((2, 0, 1))



#net.blobs[input_node].data[...] = transformer.preprocess(input_node,img)
outputs = net.forward()
output_blob = outputs[output_node]
output = output_blob.flatten()
output_shape = output_blob.shape
output_dim = len(output_shape)

# output1 = outputs[output_node1].flatten()
# output_path1 = output_node1 + '.txt'
# output2 = outputs[output_node2].flatten()
# output_path2 = output_node2 + '.txt'
# f = open(output_path1, 'w')
# for value in output1:
#     f.write(str(value))
#     f.write('\n')
# f.close()
# f = open(output_path2, 'w')
# for value in output2:
#     f.write(str(value))
#     f.write('\n')
# f.close()

print("------------------------------------OUTPUT------------------------------------")
print("net name: ", net_name)
print("input name: ", input_node)
print("input size: ", caffe_blobs[input_node].data.shape)
print("output name: ", output_node)
print("output shape: ", outputs[output_node].data.shape)
print("len of result: ", len(output))
print("max of result: ", max(output))
print("index of max result: ", np.argmax(output))

print("------------------------------------SAVE RESULT-------------------------------")
f = open(output_path, 'w')
for value in output:
    f.write(str(value))
    f.write('\n')
f.close()
print("---------------------------------------END------------------------------------")