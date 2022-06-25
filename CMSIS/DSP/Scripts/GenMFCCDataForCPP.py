###########################################
# Project:      CMSIS DSP Library
# Title:        GenMFCCDataForCPP.py
# Description:  Generation of MFCC arrays for the MFCC function
# 
# $Date:        07 September 2021
# $Revision:    V1.10.0
# 
# Target Processor: Cortex-M and Cortex-A cores
# -------------------------------------------------------------------- */
# 
# Copyright (C) 2010-2021 ARM Limited or its affiliates. All rights reserved.
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
############################################

#########################
#
# This script is generating arrays required by the MFCC implementation:
# DCT coefficients and Mel filters
# Those arrays must be used with the arm_mfcc_init functions
# The configuration is done through a yaml file describing the values for the
# MFCC and the type
#
import argparse
import yaml
import mfccdata 
import os.path

parser = argparse.ArgumentParser(description='Generate MFCC Data for CPP')


parser.add_argument('-n', nargs='?',type = str, default="mfccdata", help="mfcc file name")
parser.add_argument('-d', nargs='?',type = str, default="Testing/Source/Tests", help="mfcc c file directory")
parser.add_argument('-i', nargs='?',type = str, default="Testing/Include/Tests", help="mfcc h file directory")
parser.add_argument('others', help="yaml configuration file", nargs=argparse.REMAINDER)

args = parser.parse_args()



if args.n and args.d and args.others:
   cpath=os.path.join(args.d,args.n + ".c")
   hpath=os.path.join(args.i,args.n + ".h")

   with open(args.others[0],"r") as f:
     configs=yaml.safe_load(f) 
     mfccdata.checkF16(configs)
     mfccdata.prepareDctconfig(configs["dct"])
     mfccdata.prepareMelconfig(configs["melfilter"])
     mfccdata.prepareWindowConfig(configs["window"])
     with open(hpath,"w") as h:
        mfccdata.genMfccHeader(h,configs,args.n)
     with open(cpath,"w") as c:
        mfccdata.genMfccInit(c,configs,args.n)
           