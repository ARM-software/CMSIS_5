#!/usr/bin/env python
# DECAF : Caffe model to M-class binary

import sys
# Include <Caffe installation path>/python in PYTHONPATH environment variable 
# or uncomment the following 2 lines
caffe_root = "$CAFFE_ROOT/python"
sys.path.insert(0, caffe_root)
import os
import caffe
from caffe.proto import caffe_pb2
import numpy as np
import argparse
from google.protobuf import text_format
import pickle

def convert_to_x4_weights(weights):
    """This function convert the fullly-connected layer weights
       to the format that accepted by X4 implementation"""
    [r, h, w, c] = weights.shape
    weights = np.reshape(weights, (r, h*w*c))
    num_of_rows = r
    num_of_cols = h*w*c
    new_weights = np.copy(weights)
    new_weights = np.reshape(new_weights, (r*h*w*c))
    counter = 0
    for i in range(int(num_of_rows)/4):
      # we only need to do the re-ordering for every 4 rows
      row_base = 4*i
      for j in range (int(num_of_cols)/4):
        # for each 4 entries
        column_base = 4*j
        new_weights[counter]   =  weights[row_base  ][column_base  ]
        new_weights[counter+1] =  weights[row_base+1][column_base  ]
        new_weights[counter+2] =  weights[row_base  ][column_base+2]
        new_weights[counter+3] =  weights[row_base+1][column_base+2]
        new_weights[counter+4] =  weights[row_base+2][column_base  ]
        new_weights[counter+5] =  weights[row_base+3][column_base  ]
        new_weights[counter+6] =  weights[row_base+2][column_base+2]
        new_weights[counter+7] =  weights[row_base+3][column_base+2]

        new_weights[counter+8] =  weights[row_base  ][column_base+1]
        new_weights[counter+9] =  weights[row_base+1][column_base+1]
        new_weights[counter+10] = weights[row_base  ][column_base+3]
        new_weights[counter+11] = weights[row_base+1][column_base+3]
        new_weights[counter+12] = weights[row_base+2][column_base+1]
        new_weights[counter+13] = weights[row_base+3][column_base+1]
        new_weights[counter+14] = weights[row_base+2][column_base+3]
        new_weights[counter+15] = weights[row_base+3][column_base+3]
        counter = counter + 16
      # the remaining ones are in order
      for j in range((int)(num_of_cols-num_of_cols%4), int(num_of_cols)):
        new_weights[counter] = weights[row_base][j]
        new_weights[counter+1] = weights[row_base+1][j]
        new_weights[counter+2] = weights[row_base+2][j]
        new_weights[counter+3] = weights[row_base+3][j]
        counter = counter + 4
    return new_weights

class Caffe_Quantizer(object):
    """\
    Quantize a trained caffe model to 8-bits 
    """
    def __init__(self,model_file,weight_file,iterations=100,
            accuracy_layer='accuracy',gpu=False):
        self.model_file=model_file
        self.weight_file=weight_file
        self.quant_weight_file=""
        self.conv_layer=[]
        self.ip_layer=[]
        self.start_layer=[]
        self.end_layer=[]
        self.layer=[]
        self.layer_shape={}
        self.top_blob={}
        self.bottom_blob={}
        self.layer_type={}
        self.kernel_size={}
        self.stride={}
        self.pad={}
        self.group={}
        self.pool_type={}
        self.lrn_type={}
        self.lrn_size={}
        self.lrn_alpha={}
        self.lrn_beta={}
        self.num_ops={}
        self.num_wts={}
        self.wt_int_bits={}
        self.wt_dec_bits={}
        self.bias_int_bits={}
        self.bias_dec_bits={}
        self.act_int_bits={}
        self.act_dec_bits={}
        self.bias_lshift={}
        self.act_rshift={}
        self.data_layer=None
        self.accuracy_layer=accuracy_layer
        self.iterations=iterations
        self.gpu=gpu

    def save_quant_params(self,model_info_file):
        pickle.dump(self,open(model_info_file,'wb'))

    def load_quant_params(self,model_info_file):
        model_par=pickle.load(open(model_info_file,'rb'))
        self.model_file=model_par.model_file
        self.weight_file=model_par.weight_file
        self.quant_weight_file=model_par.quant_weight_file
        self.conv_layer=model_par.conv_layer
        self.ip_layer=model_par.ip_layer
        self.start_layer=model_par.start_layer
        self.end_layer=model_par.end_layer
        self.layer=model_par.layer
        self.layer_shape=model_par.layer_shape
        self.top_blob=model_par.top_blob
        self.bottom_blob=model_par.bottom_blob
        self.layer_type=model_par.layer_type
        self.kernel_size=model_par.kernel_size
        self.stride=model_par.stride
        self.pad=model_par.pad
        self.group=model_par.group
        self.pool_type=model_par.pool_type
        self.lrn_type=model_par.lrn_type
        self.lrn_size=model_par.lrn_size
        self.lrn_alpha=model_par.lrn_alpha
        self.lrn_beta=model_par.lrn_beta
        self.num_ops=model_par.num_ops
        self.num_wts=model_par.num_wts
        self.wt_int_bits=model_par.wt_int_bits
        self.wt_dec_bits=model_par.wt_dec_bits
        self.bias_int_bits=model_par.bias_int_bits
        self.bias_dec_bits=model_par.bias_dec_bits
        self.act_int_bits=model_par.act_int_bits
        self.act_dec_bits=model_par.act_dec_bits
        self.bias_lshift=model_par.bias_lshift
        self.act_rshift=model_par.act_rshift
        self.data_layer=model_par.data_layer
        self.accuracy_layer=model_par.accuracy_layer
        self.iterations=model_par.iterations
        self.gpu=model_par.gpu

    def run_full_network(self):
        if self.gpu==True:
            caffe.set_mode_gpu()
        net = caffe.Net(self.model_file,self.weight_file,caffe.TEST)
        acc = np.zeros(self.iterations)
        for i in range(0,self.iterations):
            out = net.forward()
            acc[i] = out[self.accuracy_layer]*100
        print("Full precision accuracy: %.2f%%" %(acc.mean()))
        return acc.mean()

    def run_quantized_network(self):
        if self.gpu==True:
            caffe.set_mode_gpu()
        net = caffe.Net(self.model_file,self.quant_weight_file,caffe.TEST)
        acc = np.zeros(self.iterations)
        for i in range(0,self.iterations):
            out = net.forward()
            acc[i] = out[self.accuracy_layer]*100
        print("Accuracy with quantized weights/biases: %.2f%%" %(acc.mean()))
        for i in range(0,self.iterations):
            for layer_no in range(0,len(self.start_layer)):
                if layer_no==0:
                    net.forward(end=str(self.end_layer[layer_no]))
                else:
                    net.forward(start=str(self.start_layer[layer_no]),end=str(self.end_layer[layer_no]))
                if layer_no < len(self.start_layer)-1: # not quantizing accuracy layer
                    net.blobs[self.end_layer[layer_no]].data[:]=np.floor(net.blobs[self.end_layer[layer_no]].data*\
                        (2**self.act_dec_bits[self.end_layer[layer_no]]))
                    net.blobs[self.end_layer[layer_no]].data[net.blobs[self.end_layer[layer_no]].data>126]=127
                    net.blobs[self.end_layer[layer_no]].data[net.blobs[self.end_layer[layer_no]].data<-127]=-128
                    net.blobs[self.end_layer[layer_no]].data[:]=net.blobs[self.end_layer[layer_no]].data/\
                        (2**self.act_dec_bits[self.end_layer[layer_no]])
            acc[i] = net.blobs[self.accuracy_layer].data*100
        accuracy = acc.mean()
        print("Accuracy with quantized weights/biases and activations: %.2f%%" %(accuracy))
        return accuracy

    def get_layer_info(self):
        #TODO: Return layer dimensions to derive OPs and number of weights
        net=caffe_pb2.NetParameter()
        text_format.Merge(open(self.model_file,'r').read(),net)
        cnn = caffe.Net(self.model_file,self.weight_file,caffe.TEST)

        if len(net.layer)==0:	#some prototxts use "layer", some use "layers"
        	layers = net.layers
        else:
        	layers = net.layer

        for layer in layers:
            layer_name=[]
            for val in layer.top:
                layer_name+=[str(val)]
            self.top_blob[str(layer.name)]=layer_name
            self.layer_shape[str(layer.name)]=cnn.blobs[self.top_blob[str(layer.name)][0]].data.shape
            layer_name=[]
            for val in layer.bottom:
                layer_name+=[str(val)]
            self.bottom_blob[str(layer.name)]=layer_name
            self.layer_type[str(layer.name)] = str(layer.type).lower()

            if str(layer.type).lower() == 'convolution' or str(layer.type)=='4':
                self.conv_layer.append(str(layer.name))
                self.kernel_size[str(layer.name)] = layer.convolution_param.kernel_size[0]
                self.stride[str(layer.name)] = 1;
                self.pad[str(layer.name)] = 0;
                if(len(layer.convolution_param.stride)!=0):
                    self.stride[str(layer.name)] = layer.convolution_param.stride[0]
                if(len(layer.convolution_param.pad)!=0):
                    self.pad[str(layer.name)] = layer.convolution_param.pad[0]
                self.group[str(layer.name)] = layer.convolution_param.group
            elif str(layer.type).lower() == 'pooling' or str(layer.type)=='17':
                self.pool_type[str(layer.name)] = layer.pooling_param.pool
                self.kernel_size[str(layer.name)] = layer.pooling_param.kernel_size
                self.stride[str(layer.name)] = layer.pooling_param.stride
                self.pad[str(layer.name)] = layer.pooling_param.pad
            elif str(layer.type).lower() == 'lrn' or str(layer.type)=='15':
                self.lrn_type[str(layer.name)] = layer.lrn_param.norm_region
                self.lrn_size[str(layer.name)] = layer.lrn_param.local_size
                self.lrn_alpha[str(layer.name)] = layer.lrn_param.alpha
                self.lrn_beta[str(layer.name)] = layer.lrn_param.beta
            elif str(layer.type).lower() == 'innerproduct' or str(layer.type)=='14':
                self.ip_layer.append(str(layer.name))
            elif str(layer.type).lower() == 'data' or str(layer.type)=='5':
                included = False
                for layer_phase in layer.include:
                    included = included or layer_phase.phase == caffe.TEST
                if(included == True):
                    batch_size = layer.data_param.batch_size
                    self.data_layer = str(layer.top[0])

    def get_graph_connectivity(self):

    # Extract network connectivity for running CNN functions in the correct order
    # Traversing back from output layer (accuracy) to input layer (data) especially because 
    # googLeNet has many accuracy labels, which branch out and end at a different accuracy 
    # label with forward traversal

        net=caffe_pb2.NetParameter()
        text_format.Merge(open(self.model_file,'r').read(),net)
        allowed_layer_types = ['data','convolution','innerproduct','pooling','lrn','relu',\
            'accuracy','concat','5','4','14','17','15','18','1','3']
        current_layer = self.accuracy_layer
        traversed=[]
        while current_layer != str(self.data_layer):
            traversed += [current_layer]
            num_branch = len(self.bottom_blob[current_layer])
            current_blob = self.bottom_blob[current_layer][0]
            has_unused_relu = 0
            for key, value in self.top_blob.iteritems():
                if (current_blob in value) and (key not in traversed) and \
                        (self.layer_type[key] == 'relu' or self.layer_type[key]=='18'):
                    has_unused_relu = 1
                    break
            for key, value in self.top_blob.iteritems():
                if(has_unused_relu == 1):
                    if (current_blob in value) and (key not in traversed) and \
                            (self.layer_type[key]=='relu' or self.layer_type[key]=='18'):
                        has_unused_relu = 0
                        current_layer = key
                        break
                else:
                     if (current_blob in value) and (key not in traversed) and \
                             (self.layer_type[key] in allowed_layer_types):
                         current_layer = key
                         break
        traversed += [current_layer]
        traversed.reverse()
        self.layer=traversed[:]

        self.start_layer+=['']
        for layer_no in range(0,len(self.layer)):
            layer = self.layer[layer_no]
            if layer == self.data_layer or layer in self.conv_layer or \
                    layer in self.ip_layer or layer in self.accuracy_layer:
                self.end_layer+=[layer]
                if layer_no < len(self.layer)-1:
                    self.start_layer+=[self.layer[layer_no+1]]
        print(self.start_layer)
        print(self.end_layer)

    #   Quantize weights to 8 bits
    #       Using min and max of weights as nearest power of 2, quantize to 8bits (QM.N) and check accuracy
    #       If accuracy is lost, try QM-1:N+1, QM-2,N+2,... with saturation to find out the best combination
    #           with least accuracy loss (Trading-off weights that occur infrequently for more precision)
    #
    #                     -2^(M+N)		0	   2^(M+N)
    #			 |		^             |
    #			 |	      *|||*	      |
    #		<--------|	     *|||||*          |------->
    #	        Saturated|	    *|||||||*         |Saturated
    #			 |	  *|||||||||||*       |
    #			 |     *|||||||||||||||||*    |
    #			*|			      |*
    #	*		 |<-------------------------->|			*
    #		             Weight quantization and  
    #                            truncation with minimal
    #			        loss of accuracy
    #
    
    def quantize_wts_8bit(self,tolerance=0.001,search_range=3):
        if self.gpu==True:
            caffe.set_mode_gpu()
        net = caffe.Net(self.model_file,self.weight_file,caffe.TEST)
        acc = np.zeros(self.iterations)
        for i in range(0,self.iterations):
            out = net.forward()
            acc[i] = out[self.accuracy_layer]*100
        target_accuracy = acc.mean()
        print("Full precision accuracy: %.2f%%" %(target_accuracy))
        self.quant_weight_file = self.weight_file
        wfile = os.path.basename(self.weight_file)
        qwfile = 'quantized_'+wfile
        self.quant_weight_file = self.weight_file.replace(wfile,qwfile)
        self.quant_weight_file = self.quant_weight_file.replace('.h5','')
        net.save(self.quant_weight_file)
        for layer_name in self.conv_layer+self.ip_layer:
            #Start with min/max of weights to the rounded up to nearest power of 2.
            wt_max = net.params[layer_name][0].data.max()
            wt_min = net.params[layer_name][0].data.min()
            self.wt_int_bits[layer_name] = int(np.ceil(np.log2(max(abs(wt_min),abs(wt_max)))))
            self.wt_dec_bits[layer_name] = 7-self.wt_int_bits[layer_name]
            max_int_bits = self.wt_int_bits[layer_name]-search_range
            print('Layer: '+ layer_name + ' weights max: '+str(wt_max)+' min: '+str(wt_min)+\
                ' Format: Q'+str(self.wt_int_bits[layer_name])+'.'+str(self.wt_dec_bits[layer_name]))
            net.params[layer_name][0].data[:]=np.round(net.params[layer_name][0].data*\
                (2**self.wt_dec_bits[layer_name]))/(2**self.wt_dec_bits[layer_name])
            for i in range(0,self.iterations):
                out = net.forward()
                acc[i] = out[self.accuracy_layer]*100
            accuracy = acc.mean()
            print("Accuracy: %.2f%%" %(accuracy))
            best_int_bits = self.wt_int_bits[layer_name]
            best_dec_bits = self.wt_dec_bits[layer_name]
            best_accuracy = accuracy
            while target_accuracy-accuracy>tolerance and self.wt_int_bits[layer_name]>max_int_bits:
                self.wt_int_bits[layer_name] = self.wt_int_bits[layer_name]-1
                self.wt_dec_bits[layer_name] = self.wt_dec_bits[layer_name]+1
                net.copy_from(self.quant_weight_file)
                net.params[layer_name][0].data[:]=np.round(net.params[layer_name][0].data*\
                    (2**self.wt_dec_bits[layer_name]))
                net.params[layer_name][0].data[net.params[layer_name][0].data>126]=127
                net.params[layer_name][0].data[net.params[layer_name][0].data<-127]=-128
                net.params[layer_name][0].data[:]=net.params[layer_name][0].data/\
                    (2**self.wt_dec_bits[layer_name])
                for i in range(0,self.iterations):
                    out = net.forward()
                    acc[i] = out[self.accuracy_layer]*100
                accuracy = acc.mean()
                print('Format Q'+str(self.wt_int_bits[layer_name])+'.'+\
                    str(self.wt_dec_bits[layer_name])+' Accuracy: %.2f%%' %(accuracy))
                if accuracy>best_accuracy:
                    best_int_bits = self.wt_int_bits[layer_name]
                    best_dec_bits = self.wt_dec_bits[layer_name]
                    best_accuracy = accuracy
            self.wt_int_bits[layer_name] = best_int_bits
            self.wt_dec_bits[layer_name] = best_dec_bits
            net.copy_from(self.quant_weight_file)
            net.params[layer_name][0].data[:]=np.round(net.params[layer_name][0].data*\
                (2**self.wt_dec_bits[layer_name]))
            net.params[layer_name][0].data[net.params[layer_name][0].data>126]=127
            net.params[layer_name][0].data[net.params[layer_name][0].data<-127]=-128
            net.params[layer_name][0].data[:]=net.params[layer_name][0].data/\
                (2**self.wt_dec_bits[layer_name])
            print('Final '+layer_name+ ' weights format Q'+str(best_int_bits)+'.'+\
                str(best_dec_bits)+' Accuracy: %.2f%%' %(best_accuracy))
            net.save(self.quant_weight_file)

    #   Quantize activations (inter-layer data) to 8 bits
    #       Using min and max of activations as nearest power of 2, quantize to 8bits (QM.N) and check accuracy
    #       If accuracy is lost, try QM-1:N+1, QM-2,N+2,... with saturation to find out the best combination
    #           with least accuracy loss (Trading-off activations that occur infrequently for more precision)
    
    def quantize_activations_8bit(self,tolerance=0.001,search_range=3):
        if self.gpu==True:
            caffe.set_mode_gpu()
        net = caffe.Net(self.model_file,self.quant_weight_file,caffe.TEST)
        acc = np.zeros(self.iterations)
        for i in range(0,self.iterations):
            out = net.forward()
            acc[i] = out[self.accuracy_layer]*100
        target_accuracy = acc.mean()
        print("Accuracy with quantized weights: %.2f%%" %(target_accuracy))
        max_val={}
        min_val={}
        quant_layer_flag={}
        for layer in self.end_layer:
            max_val[layer]=float('-inf')
            min_val[layer]=float('inf')
            quant_layer_flag[layer]=0
        #Finding min max for output of all layers
        for i in range(0,self.iterations):
            for layer_no in range(0,len(self.start_layer)):
                if layer_no==0:
                    net.forward(end=str(self.end_layer[layer_no]))
                else:
                    net.forward(start=str(self.start_layer[layer_no]),end=str(self.end_layer[layer_no]))
                layer_max = net.blobs[self.end_layer[layer_no]].data.max()
                layer_min = net.blobs[self.end_layer[layer_no]].data.min()
                if(layer_max>max_val[self.end_layer[layer_no]]):
                    max_val[self.end_layer[layer_no]]=layer_max
                if(layer_min<min_val[self.end_layer[layer_no]]):
                    min_val[self.end_layer[layer_no]]=layer_min
                #print("Running %s layer, max,min : %.2f,%.2f" %(self.end_layer[layer_no],layer_max,layer_min)) 
        max_int_bits={}
        for layer in self.end_layer:
            self.act_int_bits[layer] = int(np.ceil(np.log2(max(abs(max_val[layer]),abs(min_val[layer])))))
            self.act_dec_bits[layer] = 7-self.act_int_bits[layer]
            max_int_bits[layer]=self.act_int_bits[layer]-search_range
            print('Layer: '+layer+' max: '+ str(max_val[layer]) + ' min: '+str(min_val[layer])+ \
                ' Format: Q'+str(self.act_int_bits[layer])+'.'+str(self.act_dec_bits[layer]))
        quant_max_val={}
        quant_min_val={}
        for layer in self.end_layer:
            quant_max_val[layer]=float('-inf')
            quant_min_val[layer]=float('inf')
        for quant_layer in range(0,len(self.start_layer)-1): #No need to quantize accuracy layer
            quant_layer_flag[self.end_layer[quant_layer]]=1
            # quantize layer by layer
            for i in range(0,self.iterations):
                for layer_no in range(0,len(self.start_layer)):
                    if layer_no==0:
                        net.forward(end=str(self.end_layer[layer_no]))
                    else:
                        net.forward(start=str(self.start_layer[layer_no]),end=str(self.end_layer[layer_no]))
                    if quant_layer_flag[self.end_layer[layer_no]]==1:	# quantize incrementally layer by layer
                        net.blobs[self.end_layer[layer_no]].data[:]=np.floor(net.blobs[self.end_layer[layer_no]].data*\
                            (2**self.act_dec_bits[self.end_layer[layer_no]]))/(2**self.act_dec_bits[self.end_layer[layer_no]])
                    layer_max = net.blobs[self.end_layer[layer_no]].data.max()
                    layer_min = net.blobs[self.end_layer[layer_no]].data.min()
                    if(layer_max>quant_max_val[self.end_layer[layer_no]]):
                        quant_max_val[self.end_layer[layer_no]]=layer_max
                    if(layer_min<quant_min_val[self.end_layer[layer_no]]):
                        quant_min_val[self.end_layer[layer_no]]=layer_min
                acc[i] = net.blobs[self.accuracy_layer].data*100
            accuracy=acc.mean()
            print('Layer-'+self.end_layer[quant_layer]+' max: '+str(quant_max_val[self.end_layer[quant_layer]])+\
                ' min: '+str(quant_min_val[self.end_layer[quant_layer]])+' format: Q'+\
                str(self.act_int_bits[self.end_layer[quant_layer]])+'.'+str(self.act_dec_bits[self.end_layer[quant_layer]])+\
                ' accuracy: %.2f%%' %(acc.mean()))
            best_accuracy = accuracy
            best_int_bits = self.act_int_bits[self.end_layer[quant_layer]]
            best_dec_bits = self.act_dec_bits[self.end_layer[quant_layer]]
            while target_accuracy-accuracy>tolerance and self.act_int_bits[self.end_layer[quant_layer]]>\
                max_int_bits[self.end_layer[quant_layer]]:
                for layer in self.end_layer:
                    quant_max_val[layer]=float('-inf')
                    quant_min_val[layer]=float('inf')
                self.act_int_bits[self.end_layer[quant_layer]] = self.act_int_bits[self.end_layer[quant_layer]]-1
                self.act_dec_bits[self.end_layer[quant_layer]] = self.act_dec_bits[self.end_layer[quant_layer]]+1
                for i in range(0,self.iterations):
                    for layer_no in range(0,len(self.start_layer)):
                        if layer_no==0:
                            net.forward(end=str(self.end_layer[layer_no]))
                        else:
                            net.forward(start=str(self.start_layer[layer_no]),end=str(self.end_layer[layer_no]))
                        if quant_layer_flag[self.end_layer[layer_no]]==1:	
                            net.blobs[self.end_layer[layer_no]].data[:]=np.floor(net.blobs[self.end_layer[layer_no]].data*\
                                (2**self.act_dec_bits[self.end_layer[layer_no]]))
                            net.blobs[self.end_layer[layer_no]].data[net.blobs[self.end_layer[layer_no]].data>126]=127
                            net.blobs[self.end_layer[layer_no]].data[net.blobs[self.end_layer[layer_no]].data<-127]=-128
                            net.blobs[self.end_layer[layer_no]].data[:]=net.blobs[self.end_layer[layer_no]].data/\
                                (2**self.act_dec_bits[self.end_layer[layer_no]])
                        layer_max = net.blobs[self.end_layer[layer_no]].data.max()
                        layer_min = net.blobs[self.end_layer[layer_no]].data.min()
                        if(layer_max>quant_max_val[self.end_layer[layer_no]]):
                            quant_max_val[self.end_layer[layer_no]]=layer_max
                        if(layer_min<quant_min_val[self.end_layer[layer_no]]):
                            quant_min_val[self.end_layer[layer_no]]=layer_min
                    acc[i] = net.blobs[self.accuracy_layer].data*100
                accuracy=acc.mean()
                if accuracy>best_accuracy:
                    best_int_bits = self.act_int_bits[self.end_layer[quant_layer]]
                    best_dec_bits = self.act_dec_bits[self.end_layer[quant_layer]]
                    best_accuracy = accuracy
                print('Layer-'+self.end_layer[quant_layer]+' max: '+str(quant_max_val[self.end_layer[quant_layer]])+\
                    'min: '+str(quant_min_val[self.end_layer[quant_layer]])+' format: Q'+\
                    str(self.act_int_bits[self.end_layer[quant_layer]])+'.'+str(self.act_dec_bits[self.end_layer[quant_layer]])+\
                    ' accuracy: %.2f%%' %(acc.mean()))
            self.act_int_bits[self.end_layer[quant_layer]] = best_int_bits 
            self.act_dec_bits[self.end_layer[quant_layer]] = best_dec_bits 
            print('Layer-'+self.end_layer[quant_layer]+' final format: Q'+str(self.act_int_bits[self.end_layer[quant_layer]])+\
                '.'+str(self.act_dec_bits[self.end_layer[quant_layer]])+ ' accuracy: %.2f%%' %(best_accuracy))
    
    def quantize_bias_8bit(self,tolerance=0.001,search_range=3):
        if self.gpu==True:
            caffe.set_mode_gpu()
        net = caffe.Net(self.model_file,self.quant_weight_file,caffe.TEST)
        acc = np.zeros(self.iterations)
        for i in range(0,self.iterations):
            net.forward()
            acc[i] = net.blobs[self.accuracy_layer].data*100
        target_accuracy = acc.mean()
        print("Accuracy with quantized weights: %.2f%%" %(target_accuracy))
        for i in range(0,self.iterations):
            for layer_no in range(0,len(self.start_layer)):
                if layer_no==0:
                    net.forward(end=str(self.end_layer[layer_no]))
                else:
                    net.forward(start=str(self.start_layer[layer_no]),end=str(self.end_layer[layer_no]))
                if layer_no < len(self.start_layer)-1: # not quantizing accuracy layer
                    net.blobs[self.end_layer[layer_no]].data[:]=np.floor(net.blobs[self.end_layer[layer_no]].data*\
                        (2**self.act_dec_bits[self.end_layer[layer_no]]))
                    net.blobs[self.end_layer[layer_no]].data[net.blobs[self.end_layer[layer_no]].data>126]=127
                    net.blobs[self.end_layer[layer_no]].data[net.blobs[self.end_layer[layer_no]].data<-127]=-128
                    net.blobs[self.end_layer[layer_no]].data[:]=net.blobs[self.end_layer[layer_no]].data/\
                        (2**self.act_dec_bits[self.end_layer[layer_no]])
            acc[i] = net.blobs[self.accuracy_layer].data*100
        target_accuracy = acc.mean()
        print("Accuracy with quantized weights and activations: %.2f%%" %(target_accuracy))
        input_of={}
        for i in range (1,len(self.end_layer)):
            input_of[self.end_layer[i]]=self.end_layer[i-1]
        for layer_name in self.conv_layer+self.ip_layer:
            mac_dec_bits = self.wt_dec_bits[layer_name]+self.act_dec_bits[input_of[layer_name]]
            bias_max = net.params[layer_name][1].data.max()
            bias_min = net.params[layer_name][1].data.min()
            int_bits = int(np.ceil(np.log2(max(abs(bias_min),abs(bias_max)))))
            dec_bits = 7-int_bits
            max_int_bits = int_bits-search_range
            if(dec_bits>mac_dec_bits):
                dec_bits=mac_dec_bits
                int_bits=7-dec_bits
                max_int_bits=int_bits #can't increase dec_bits any more as they will be shifted right anyway
            print('Layer: '+ layer_name + ' biases max: '+str(bias_max)+' min: '+str(bias_min)+\
                ' Format: Q'+str(int_bits)+'.'+str(dec_bits))
            net.params[layer_name][1].data[:]=np.round(net.params[layer_name][1].data*(2**dec_bits))/(2**dec_bits)
            for i in range(0,self.iterations):
                for layer_no in range(0,len(self.start_layer)):
                    if layer_no==0:
                        net.forward(end=str(self.end_layer[layer_no]))
                    else:
                        net.forward(start=str(self.start_layer[layer_no]),end=str(self.end_layer[layer_no]))
                    if layer_no < len(self.start_layer)-1: # not quantizing accuracy layer
                        net.blobs[self.end_layer[layer_no]].data[:]=np.floor(net.blobs[self.end_layer[layer_no]].data*\
                            (2**self.act_dec_bits[self.end_layer[layer_no]]))
                        net.blobs[self.end_layer[layer_no]].data[net.blobs[self.end_layer[layer_no]].data>126]=127
                        net.blobs[self.end_layer[layer_no]].data[net.blobs[self.end_layer[layer_no]].data<-127]=-128
                        net.blobs[self.end_layer[layer_no]].data[:]=net.blobs[self.end_layer[layer_no]].data/\
                            (2**self.act_dec_bits[self.end_layer[layer_no]])
                acc[i] = net.blobs[self.accuracy_layer].data*100
            accuracy = acc.mean()
            print("Accuracy: %.2f%%" %(accuracy))
            best_int_bits = int_bits
            best_dec_bits = dec_bits
            best_accuracy = accuracy
            while target_accuracy-accuracy>tolerance and int_bits>max_int_bits:
                int_bits = int_bits-1
                dec_bits = dec_bits+1
                net.copy_from(self.quant_weight_file)
                net.params[layer_name][1].data[:]=np.round(net.params[layer_name][1].data*(2**dec_bits))
                net.params[layer_name][1].data[net.params[layer_name][1].data>126]=127
                net.params[layer_name][1].data[net.params[layer_name][1].data<-127]=-128
                net.params[layer_name][1].data[:]=net.params[layer_name][1].data/(2**dec_bits)
                for i in range(0,self.iterations):
                    for layer_no in range(0,len(self.start_layer)):
                        if layer_no==0:
                            net.forward(end=str(self.end_layer[layer_no]))
                        else:
                            net.forward(start=str(self.start_layer[layer_no]),end=str(self.end_layer[layer_no]))
                        if layer_no < len(self.start_layer)-1: # not quantizing accuracy layer
                            net.blobs[self.end_layer[layer_no]].data[:]=np.floor(net.blobs[self.end_layer[layer_no]].data*\
                                (2**self.act_dec_bits[self.end_layer[layer_no]]))
                            net.blobs[self.end_layer[layer_no]].data[net.blobs[self.end_layer[layer_no]].data>126]=127
                            net.blobs[self.end_layer[layer_no]].data[net.blobs[self.end_layer[layer_no]].data<-127]=-128
                            net.blobs[self.end_layer[layer_no]].data[:]=net.blobs[self.end_layer[layer_no]].data/\
                                (2**self.act_dec_bits[self.end_layer[layer_no]])
                    acc[i] = net.blobs[self.accuracy_layer].data*100
                accuracy = acc.mean()
                print('Format Q'+str(int_bits)+'.'+str(dec_bits)+' Accuracy: %.2f%%' %(accuracy))
                if accuracy>best_accuracy:
                    best_int_bits = int_bits
                    best_dec_bits = dec_bits
                    best_accuracy = accuracy
            self.bias_int_bits[layer_name] = best_int_bits
            self.bias_dec_bits[layer_name] = best_dec_bits
            self.bias_lshift[layer_name]=mac_dec_bits-best_dec_bits
            self.act_rshift[layer_name]=mac_dec_bits-self.act_dec_bits[layer_name]
            net.copy_from(self.quant_weight_file)
            net.params[layer_name][1].data[:]=np.round(net.params[layer_name][1].data*(2**best_dec_bits))
            net.params[layer_name][1].data[net.params[layer_name][1].data>126]=127
            net.params[layer_name][1].data[net.params[layer_name][1].data<-127]=-128
            net.params[layer_name][1].data[:]=net.params[layer_name][1].data/(2**best_dec_bits)
            print('Final '+layer_name+ ' biases format Q'+str(best_int_bits)+'.'+str(best_dec_bits)+\
                ' Accuracy: %.2f%%' %(best_accuracy))
            net.save(self.quant_weight_file)
    
    def generate_weight_file(self,file_name,ip_4x):
        print('Generating weights file: '+file_name)
        f=open(file_name,'w')
        net = caffe.Net(self.model_file,self.quant_weight_file,caffe.TEST)
        for layer_name in self.conv_layer:
            net.params[layer_name][0].data[:]=np.round(net.params[layer_name][0].data*\
                (2**self.wt_dec_bits[layer_name])) 
            net.params[layer_name][1].data[:]=np.round(net.params[layer_name][1].data*\
                (2**self.bias_dec_bits[layer_name]))
            f.write('#define '+layer_name.upper()+'_WT {')
            #CHW to HWC layout conversion
            reordered_wts = np.swapaxes(np.swapaxes(net.params[layer_name][0].data, 1, 2), 2, 3)
            reordered_wts.tofile(f, sep=",",format="%d")
            f.write('}\n\n')
            f.write('#define '+layer_name.upper()+'_BIAS {')
            net.params[layer_name][1].data.tofile(f,sep=",",format="%d")
            f.write('}\n\n')
        for layer_name in self.ip_layer:
            net.params[layer_name][0].data[:]=np.round(net.params[layer_name][0].data*\
                (2**self.wt_dec_bits[layer_name]))
            net.params[layer_name][1].data[:]=np.round(net.params[layer_name][1].data*\
                (2**self.bias_dec_bits[layer_name]))
            f.write('#define '+layer_name.upper()+'_WT {')
            layer_no=self.layer.index(layer_name)
            prev_layer_name=self.layer[layer_no-1]  #needed to find input shape of 'ip' layer
            if(len(self.layer_shape[prev_layer_name])>2): #assuming HWC input format
                reshaped_shape=(self.layer_shape[layer_name][1],self.layer_shape[prev_layer_name][1],\
                    self.layer_shape[prev_layer_name][2],self.layer_shape[prev_layer_name][3])
                reordered_wts = np.reshape(net.params[layer_name][0].data,reshaped_shape)  
                #Reorder the weights to use fullyconnected_x4 kernel 
                reordered_wts = np.swapaxes(np.swapaxes(reordered_wts, 1, 2), 2, 3)   
                reordered_wts = convert_to_x4_weights(reordered_wts)
            else:
                reordered_wts = net.params[layer_name][0].data
            reordered_wts.tofile(f,sep=",",format="%d")
            f.write('}\n\n')
            f.write('#define '+layer_name.upper()+'_BIAS {')
            net.params[layer_name][1].data.tofile(f,sep=",",format="%d")
            f.write('}\n\n')
        f.close()
    
    def generate_sample_input(self,file_name,sample_no):
        print('Generating sample input file: '+file_name)
        f=open(file_name,'w')
        net = caffe.Net(self.model_file,self.quant_weight_file,caffe.TEST)
        net.forward(end=self.data_layer)
        net.blobs[self.data_layer].data[:] = np.floor(net.blobs[self.data_layer].data*\
            (2**self.act_dec_bits[self.data_layer]))
        net.blobs[self.data_layer].data[net.blobs[self.data_layer].data>126]=127
        net.blobs[self.data_layer].data[net.blobs[self.data_layer].data<-127]=-128
        # write 'sample_no' input sample (Assumption : sample_no is < batch_size)
        f.write('#define '+self.data_layer.upper()+' {')
        #Reorder data_sample from CHW to HWC format 
        data_sample = np.swapaxes(np.swapaxes(net.blobs[self.data_layer].data[sample_no], 1, 2), 2, 3)
        data_sample.tofile(f,sep=",",format="%d")
        f.write('}\n\n')
        f.close()
    
    def generate_sample_output(self,file_name,sample_no):
        print('Generating sample output file: '+file_name)
        f=open(file_name,'w')
        net = caffe.Net(self.model_file,self.quant_weight_file,caffe.TEST)
        net.forward(end=self.end_layer[-2]) #ip layer before accuracy
        net.blobs[self.end_layer[-2]].data[:] = np.floor(net.blobs[self.end_layer[-2]].data*\
            (2**self.act_dec_bits[self.end_layer[-2]]))
        # write 'sample_no' input sample (Assumption : sample_no is < batch_size)
        data_sample = net.blobs[self.end_layer[-2]].data[sample_no]
        f.write('#define '+self.end_layer[-2].upper()+' {')
        data_sample.tofile(f,sep=",",format="%d")
        f.write('}\n\n')
        f.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_flag', type=int, default=1,
            help='run on gpu? (yes-1, no-0)')
    parser.add_argument('--accuracy', type=str, default="accuracy",
            help='target accuracy')
    parser.add_argument('--iterations', type=int, default=100,
            help='number of iterations: data_size/batch_size')
    parser.add_argument('--tolerance', type=float, default=0.001,
            help='accuracy tolerance')
    parser.add_argument('--model', type=str, default=\
        os.getcwd()+"/models/cifar10_full_train_test.prototxt",
            help='caffe model definition (.prototxt)')
    parser.add_argument('--weights', type=str, default=\
        os.getcwd()+"/models/cifar10_full_iter_70000.caffemodel.h5",
            help='caffe model weights (.caffemodel)')

    cmd_args, _ = parser.parse_known_args()

    if(cmd_args.gpu_flag==1):
        gpu_flag = True
    else:
        gpu_flag = False
    model_file=cmd_args.model
    weight_file=cmd_args.weights
    iterations=cmd_args.iterations
    tolerance=cmd_args.tolerance
    target_accuracy_layer=cmd_args.accuracy
   
    my_model=Caffe_Quantizer(model_file,weight_file,iterations,accuracy_layer=target_accuracy_layer,gpu=gpu_flag)
    my_model.get_layer_info()
    my_model.get_graph_connectivity()
    my_model.run_full_network()
    #First quantize weights to 8 bits
    my_model.quantize_wts_8bit()
    #Then quantize activations to 8 bits
    my_model.quantize_activations_8bit()
    #Quantize biases to 8 bits based on the quantization outputs of weights and activations
    my_model.quantize_bias_8bit()
    my_model.run_quantized_network()

    my_model.generate_weight_file('weights.h',True)
    #my_model.generate_sample_input('inputs.txt',0)
    #my_model.generate_sample_output('outputs.txt',0)

    my_model.save_quant_params('models/cifar10.p')
    #To load the parameters use the following:
    #my_model.load_quant_params('mymodel.p')

    #Print dataformats
    print('Input: '+my_model.data_layer+' Q'+str(my_model.act_int_bits[my_model.data_layer])+'.'+\
        str(my_model.act_dec_bits[my_model.data_layer])+'(scaling factor:'+\
        str(2**(my_model.act_dec_bits[my_model.data_layer]))+')')
    for layer in my_model.conv_layer+my_model.ip_layer:
        print('Layer: '+layer+' Q'+str(my_model.act_int_bits[layer])+'.'+str(my_model.act_dec_bits[layer])+\
            ' (scaling factor:'+str(2**(my_model.act_dec_bits[layer]))+') Wts: Q'+\
            str(my_model.wt_int_bits[layer])+'.'+str(my_model.wt_dec_bits[layer])+\
            ' (scaling factor:'+str(2**(my_model.wt_dec_bits[layer]))+') Biases: Q'+\
            str(my_model.bias_int_bits[layer])+'.'+str(my_model.bias_dec_bits[layer])+\
            '(scaling factor:'+str(2**(my_model.bias_dec_bits[layer]))+')')

    #Print data shifts to be used by ML kernels
    shift_file = open("./shift.h", "w")
    for layer in my_model.conv_layer+my_model.ip_layer:
        print('Layer: '+layer+' bias left shift: '+str(my_model.bias_lshift[layer])+\
            ' act_rshift: '+str(my_model.act_rshift[layer]))
        shift_file.write("#define "+layer.upper()+"_BIAS_LSHIFT " + str(my_model.bias_lshift[layer])+"\n")
        shift_file.write("#define "+layer.upper()+"_OUT_RSHIFT " + str(my_model.act_rshift[layer])+"\n")
    shift_file.close()

    #TODO: extract full connectivity and stitch up ML kernels - 
    #   conv, relu, pool, full connected layers with the appropriate shifts
    #   generate_host_code('main.cpp',start_layer,end_layer,bias_lshift,act_rshift)

