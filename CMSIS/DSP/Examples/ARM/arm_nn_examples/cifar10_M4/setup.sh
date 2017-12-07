#!/bin/bash

#get CMSIS5
if [ ! -e "DSP" ]; then
  echo "Creating soft link to DSP"
  ln -s ../../../../../DSP ./
else
  echo "DSP soft link exists"
fi

read -p "Enter caffe installation path: " caffe_root
caffe_python_root=$caffe_root'/python'
if [ ! -e $caffe_python_root ]; then
  echo "$caffe_root does not exist"
  echo "Please install caffe and run setup again"
  exit
fi

#get cifar10 data
if [ ! -e $caffe_root'/examples/cifar10/cifar10_test_lmdb' ]; then
  echo "CIFAR10 data doesn't exist. Downloading it"
  cwd=$PWD
  cd $caffe_root
  ./data/cifar10/get_cifar10.sh
  ./examples/cifar10/create_cifar10.sh
  cd $cwd
fi

for f in models/*.prototxt
do
  sed -i -e 's|$CAFFE_ROOT|'"${caffe_root}"'|g' $f
done

sed -i -e 's|$CAFFE_ROOT|'"${caffe_root}"'|g' host_pc.py
sed -i -e 's|$CAFFE_ROOT|'"${caffe_root}"'|g' cnn_data_scaler.py
