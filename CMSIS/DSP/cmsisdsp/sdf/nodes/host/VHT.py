from multiprocessing import Process,Semaphore
import multiprocessing as mp
import socket
import cmsisdsp.sdf.nodes.host.message as msg

HOST = '127.0.0.1'    # The remote host
PORT = 50007 

class ModelicaConnectionLost(Exception):
    pass

def connectToServer(inputMode,theid):
   s=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
   s.connect((HOST, PORT))
   # Identify as vht input
   if inputMode:
      print("Connecting as INPUT")
      theBytes=msg.list_to_bytes(msg.clientID(msg.VSIINPUT,theid))
   else:
      print("Connecting as OUTPUT")
      theBytes=msg.list_to_bytes(msg.clientID(msg.VSIOUTPUT,theid))
   #print("vs0: %d %d" % (int(theBytes[0]),int(theBytes[1])))
   msg.sendBytes(s,theBytes)
   return(s)

def source(theid,size,queue,started):
   s=connectToServer(True,theid)
   started.release()
   try:
      while True:
         received=msg.receiveBytes(s,size)
         queue.put(received)
   except Exception as inst:
      print(inst)
   finally:
      queue.close()


def sink(theid,size,queue,started):
   s=connectToServer(False,theid)
   data= bytes(size)
   msg.sendBytes(s,data)
   started.release()
   try:
      while True:
         tosend=queue.get(True,2)
         msg.sendBytes(s,tosend)
   except Exception as inst:
      print(inst)
   finally:
      queue.close()

class Source:
   def __init__(self,theid,bufferSize):
      self._bufferSize_ = bufferSize 
      self._srcQueue_ = mp.Queue()
      self._started_ = Semaphore()
      # Q15 data is sent so a *2 factor for bufferSize since
      # source function is working with bytes
      self._src_ = Process(target=source, args=(theid,2*bufferSize,self._srcQueue_,self._started_))
      self._src_.start()

   @property
   def queue(self):
      return(self._srcQueue_)

   def get(self):
      if self._src_.exitcode is None:  
         return(msg.bytes_to_list(self.queue.get(True,2)))
      else:
         raise ModelicaConnectionLost

   def end(self):
      self._src_.terminate()

   def wait(self):
      self._started_.acquire()



class Sink:
   def __init__(self,theid,bufferSize):
      self._bufferSize_ = bufferSize 
      self._sinkQueue_ = mp.Queue()
      self._started_ = Semaphore()
      # Q15 data is sent so a *2 factor for bufferSize since
      # sink function is working with bytes
      self._sink_ = Process(target=sink, args=(theid,2*bufferSize,self._sinkQueue_,self._started_))
      self._sink_.start()

   @property
   def queue(self):
      return(self._sinkQueue_)

   def put(self,data):
      if self._sink_.exitcode is None: 
         q15list=[int(x) for x in data]
         self.queue.put(msg.list_to_bytes(q15list),True,1)
      else:
         raise ModelicaConnectionLost

   def end(self):
      self._sink_.terminate()

   def wait(self):
      self._started_.acquire()

