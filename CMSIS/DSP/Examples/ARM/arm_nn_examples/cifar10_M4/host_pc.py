#!/usr/bin/env python

import serial
import array
import sys
import os
import threading
from time import sleep

import numpy as np
from matplotlib import pyplot as PLT

import scipy.misc
from cnn_data_scaler import *

# Make sure that caffe is on the python path:
caffe_root = "$CAFFE_ROOT" # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

class cifar10_minicom(object):
  """\
  A communication tool between host PC and guest mbed
  running cifar10 classification
  """

  def __init__(self, serial_instance, input_file, label_file):
    self.serial = serial_instance
    self.raw = False
    self.alive = None
    self._reader_alive = None
    self.receiver_thread = None
    self.rx_decoder = None
    self.tx_decoder = None
    self.read_buffer = ""
    self.mbed_ready = False
    self.mbed_results_ready = False
    self.result_list = None

  def _start_reader(self):
    """Start reader thread"""
    self._reader_alive = True
    # start serial->console thread
    self.receiver_thread = threading.Thread(target=self.reader, name='rx')
    self.receiver_thread.daemon = True
    self.receiver_thread.start()

  def _stop_reader(self):
    """Stop reader thread only, wait for clean exit of thread"""
    self._reader_alive = False
    if hasattr(self.serial, 'cancel_read'):
      self.serial.cancel_read()
    self.receiver_thread.join()

  def _start_test(self):
    if (self.mbed_ready):
      self.host_print("Sending signal n")
      self.serial.write(b'n')
      # add some wait time for the mbed side to process the command
      sleep(0.2) 
      for i in range(0, 32*32*3):
        values = bytearray([i%256])
        self.serial.write(values)
      self.host_print("Data sent")
    else:
      self.host_print("Mbed not ready")

  def _task_loop(self):
    while (1):
      try:
        self.host_print("Host ready for sending commmand")
        line = sys.stdin.readline()
        self.host_print("Received command " + line)
        instr = line[0]
        if (instr == 'n'):
          self._start_test()
        elif (instr == 's'):
          self.serial.write(b's')
        else:
          self.serial.write(b'k')
      except KeyboardInterrupt:
        break;

  def start(self):
    """start worker threads"""
    self.host_print("Start com")
    self.alive = True
    self._start_reader()
    # enter console->serial loop

  def send_image(self, image_data, output):
    """This sends an eimage and receives the classification back"""
    if len(image_data) != 32*32*3 :
      raise ValueError('Input image dimension incorrect')

    if (self.alive == False):
      print "Fault detected, trying to resume here"
      #we have trouble here, try reset things
      self.serial.write(b'n')
      for i in range(len(image_data)):
        self.serial.write(b'k')
      self.alive = True

    while (not self.mbed_ready):
      sleep(0.01)

    self.serial.write(b'n')
    sleep(0.2)
    values = bytearray(image_data)
    try:
      self.serial.write(values) 
    except serial.SerialException:
      self.alive = False
      raise       # XXX handle instead of re-raise?
    while(not self.mbed_results_ready):
      sleep(0.01)
    for i in range(len(output)):
      output[i] = (int)(self.result_list[i])
    self.mbed_results_ready = False

  def stop(self):
    """set flag to stop worker threads"""
    self.alive = False

  def join(self, transmit_only=False):
    """wait for worker threads to terminate"""
    self.transmitter_thread.join()
    if not transmit_only:
      if hasattr(self.serial, 'cancel_read'):
        self.serial.cancel_read()
      self.receiver_thread.join()

  def close(self):
    self.serial.close()

  def reader(self):
    """loop and copy serial->console"""
    try:
      while self.alive and self._reader_alive:
        # read all that is there or wait for one byte
        data = self.serial.read(self.serial.in_waiting or 1)
        if data:
          self.mbed_print(data)
    except serial.SerialException:
      self.alive = False
      raise       # XXX handle instead of re-raise?

  def host_print(self, string):
    print "HOST: "+string

  def mbed_print(self, data):
    self.read_buffer = self.read_buffer + data
    ind1 = self.read_buffer.find('\n')
    if (ind1 >= 0):
      print "MBED: " + self.read_buffer[0:ind1]
      if (self.read_buffer[0:13] ==   "cifar10 ready"):
        self.mbed_ready = True
      elif (self.read_buffer[0:12] == "Final output"):
        self.result_list = self.read_buffer[13:].split()
        self.mbed_results_ready = True
      else:
        self.mbed_ready = False
      self.read_buffer = self.read_buffer[ind1+1:]

#serial_instance = serial.Serial('COM4', 9600)
try:
  serial_instance = serial.Serial('/dev/ttyACM0', 9600)
except serial.SerialException:
  serial_instance = serial.Serial('/dev/ttyACM1', 9600)

my_com = cifar10_minicom(serial_instance, "input.h", "label.h")

caffe.set_mode_gpu()
net = caffe.Net(os.getcwd()+'/models/cifar10_full_train_test.prototxt',
                os.getcwd()+'/models/quantized_cifar10_full_iter_70000.caffemodel',
                caffe.TEST)

cnn_model = Caffe_Quantizer(os.getcwd()+"/models/cifar10_full_train_test.prototxt",os.getcwd()+"/models/cifar10_full_iter_70000.caffemodel.h5")
cnn_model.load_quant_params(os.getcwd()+"/models/cifar10.p")


# N C H W

batch_size = net.blobs['data'].data.shape[0]
image_channel = net.blobs['data'].data.shape[1]
image_size_y = net.blobs['data'].data.shape[2]
image_size_x = net.blobs['data'].data.shape[3]

total_count = 0
correct_count = 0
include_count = 0

base_count = 0
base_total = 0

match_matrix = np.zeros([10, 10], dtype=np.int)

mean_f = open("$CAFFE_ROOT/examples/cifar10/mean.binaryproto", "rb")

mean_array = array.array('B')

mean_array.fromfile(mean_f, 32*32*3)

acc = np.zeros(100)
for i in range(100):
    out = net.forward()
    acc[i] = out[cnn_model.accuracy_layer]*100
print("Accuracy with quantized weights/biases: %.2f%%" %(acc.mean()))

my_com.start()

for i in range(100):
    for layer_no in range(0,len(cnn_model.start_layer)):
        if layer_no==0:
            net.forward(end=str(cnn_model.end_layer[layer_no]))
        else:
            net.forward(start=str(cnn_model.start_layer[layer_no]),end=str(cnn_model.end_layer[layer_no]))
        if layer_no < len(cnn_model.start_layer)-1: # not quantizing accuracy layer
            net.blobs[cnn_model.end_layer[layer_no]].data[:]=np.floor(net.blobs[cnn_model.end_layer[layer_no]].data*\
                (2**cnn_model.act_dec_bits[cnn_model.end_layer[layer_no]]))
            net.blobs[cnn_model.end_layer[layer_no]].data[net.blobs[cnn_model.end_layer[layer_no]].data>126]=127
            net.blobs[cnn_model.end_layer[layer_no]].data[net.blobs[cnn_model.end_layer[layer_no]].data<-127]=-128
            net.blobs[cnn_model.end_layer[layer_no]].data[:]=net.blobs[cnn_model.end_layer[layer_no]].data/\
                (2**cnn_model.act_dec_bits[cnn_model.end_layer[layer_no]])
    for n_img in range(batch_size):
        image = np.zeros(image_channel*image_size_y*image_size_x, dtype=np.uint8)
        display_image = np.zeros((32,32,3)).astype(np.float64)
        for i_ch in range(image_channel):
            for i_y in range(image_size_y):
                for i_x in range(image_size_x):
                    image[(i_y*image_size_x+i_x)*image_channel+i_ch] = np.floor(net.blobs['data'].data[n_img][i_ch][i_y][i_x]*\
                        2**cnn_model.act_dec_bits[cnn_model.data_layer]) + 128
                    if (np.floor(net.blobs['data'].data[n_img][i_ch][i_y][i_x]*2**cnn_model.act_dec_bits[cnn_model.data_layer]) + 128 > 255):
                        image[(i_y*image_size_x+i_x)*image_channel+i_ch] = 255
                    if (np.floor(net.blobs['data'].data[n_img][i_ch][i_y][i_x]*2**cnn_model.act_dec_bits[cnn_model.data_layer]) + 128 < 0):
                        image[(i_y*image_size_x+i_x)*image_channel+i_ch] = 0
                    display_image[i_y][i_x][i_ch] =  (image[(i_y*image_size_x+i_x)*image_channel+i_ch]) / 256.0;
                    #display_image[i_y][i_x][i_ch] = net.blobs['data'].data[n_img][i_ch][i_y][i_x] + mean_array[(i_y*image_size_x+i_x)+i_ch*image_size_x*image_size_y]
    
        #display_image.reshape(3,32,32).transpose(1,2,0)      
    
        #scipy.misc.imsave('output.png', display_image)
        #PLT.imshow(display_image)
        #PLT.savefig('output.png')
        #PLT.show()
        class_output = [None] * 10
        my_com.send_image(image, class_output)
        print "output " + str(class_output)
        print "Expected output:"+str(np.floor(net.blobs['ip1'].data[n_img]*2**cnn_model.act_dec_bits['ip1']))
        max_value = -128
        max_index = -1
        for j in range(10):
          if (class_output[j] > max_value):
            max_value = class_output[j]
            max_index = j
        max_list = []
        for j in range(10):
          if (class_output[j] == max_value):
            max_list.append(j)
        print "output "+ str(max_index) + " label " + str(net.blobs['label'].data[n_img])
        total_count += 1
        if (max_index == net.blobs['label'].data[n_img]):
          correct_count += 1
        if (len(max_list) < 4):
          if net.blobs['label'].data[n_img] in max_list :
            include_count += 1
        print "Results " + str(correct_count) + "/" + str(total_count) + ", Accuracy: {:.2f}".format(correct_count*100.0/total_count)
        print "Include " + str(include_count) + "/" + str(total_count) + ", Accuracy: {:.2f}".format(include_count*100.0/total_count)
        match_matrix[max_index][int(net.blobs['label'].data[n_img])] += 1;
    base_count += net.blobs[cnn_model.accuracy_layer].data*100
    base_total += 100
    print "baseline is " + str(base_count) + "/" + str(base_total) + ", Accuracy: {:.2f}".format(base_count*100.0/base_total)

    print match_matrix
