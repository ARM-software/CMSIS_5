# --------------------------------------------------------------------------
# Copyright (c) 2020-2022 Arm Limited (or its affiliates). All rights reserved.
# 
# SPDX-License-Identifier: Apache-2.0
# 
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
import sys 

# VSI KIND
VSIOUTPUT = 0
VSIINPUT = 1

# MESSAGE IDs 
CLIENTREADBUF=1 
CLIENTWRITEBUF=2 
CLIENTSTOP=3 

# PACKETSIZE : default number of bytes read on a socket
PACKETSIZE = 1024
# Conersion between size expressed in bytes or in Q15
INTSIZE = 2 

# Error raised when trying to read / write to sockets
class ErrorTooMuchDataReceived(Exception):
    pass

class CantReceiveData(Exception):
    pass


def clientID(inputMode,theID):
       return([(theID << 1) | inputMode])
       
# Receive a given number of bytes
# Socket is read by block of PACKETSIZE
def receiveBytes(conn,nb):
    data = b""
    while nb > 0:
       if nb < PACKETSIZE:
         newData = conn.recv(nb)
         if not newData: raise CantReceiveData
       else:
         newData= conn.recv(PACKETSIZE)
         if not newData: raise CantReceiveData
       nb = nb - len(newData)
       if nb < 0:
          raise ErrorTooMuchDataReceived

       data += newData
    return(data)

# Send bytes
def sendBytes(conn,data):
    conn.sendall(data)

# Convert a list of Q15 to a bytestream
def list_to_bytes(l):
    return(b"".join([x.to_bytes(INTSIZE,byteorder=sys.byteorder,signed=True) for x in l]))

# Convert a bytestream to a list of Q15
def bytes_to_list(l):
    res=[] 
    i = 0
    while(i<len(l)):
        res.append(int.from_bytes(l[i:i+INTSIZE],byteorder=sys.byteorder,signed=True))
        i = i+INTSIZE
    return(res)

# Send a list of Q15
def sendIntList(conn,l):
    data = list_to_bytes(l)
    sendBytes(conn,data)

# Receive a list of Q15
def getIntList(conn,length):
    data = receiveBytes(conn,INTSIZE*length)
    return(bytes_to_list(data))


# Low level bytes management
# Return the message ID and the number of bytes expected in the message
def getMsgAndNbOfBytes(data):
    msgID = int(data[0])
    length= int.from_bytes(data[1:5],byteorder=sys.byteorder,signed=False)
    return(msgID,length)

# A client is requesting data from the server. It is the input of VHT
# Client -> Server
def getBufferMsg(conn,nbBytes):
    # Ask buffer from server
    a=(CLIENTREADBUF).to_bytes(1,byteorder=sys.byteorder)
    b=(nbBytes).to_bytes(4,byteorder=sys.byteorder)
    msg=a+b 
    sendBytes(conn,msg)
    # Receive buffer from server
    data = receiveBytes(conn,nbBytes)
    return(data)

# Stop the server when the end of the SDF scheduling has been reached.
# It is to make it easier to end the demo.
# Only the VHT client has to be killed.
# Client -> Server
def stopMsg(conn):
    # Send a stop message to server
    a=(CLIENTSTOP).to_bytes(1,byteorder=sys.byteorder)
    b=(0).to_bytes(4,byteorder=sys.byteorder)
    msg=a+b 
    sendBytes(conn,msg)

# Data in bytes
# A client is sending that some bytes be sent to the server
# It is the output of VHT
# Client -> Server
def writeBufferMsg(conn,theBytes):
    # Tell server a buffer is coming
    a=(CLIENTWRITEBUF).to_bytes(1,byteorder=sys.byteorder)
    nbBytes = len(theBytes)
    b=(nbBytes).to_bytes(4,byteorder=sys.byteorder)
    msg = a+b+theBytes
    # Send message and buffer to server
    sendBytes(conn,msg)




