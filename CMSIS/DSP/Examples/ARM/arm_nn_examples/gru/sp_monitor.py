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

my_com.start()

while (1):
  sleep(1)
