###########################################
# Project:      CMSIS DSP Library
# Title:        config.py
# Description:  Configuration of the code generator
# 
# $Date:        29 July 2021
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
"""Configuration of C code generation"""
class Configuration:

    def __init__(self):
        # Number of iterations of the schedule. By default it is infinite
        # represented by the value 0
        self.debugLimit = 0 

        # If FIFO content must be dumped during execution of the schedule.
        self.dumpFIFO = False

        # If FIFOs size must be dumped during computation of the schedule
        self.displayFIFOSizes=False

        # Name of scheduling function in the generated code
        # (Must be valid C and Python name)
        self.schedName = "scheduler"

        # Additional arguments for the scheduler API
        # must be valid C
        self.cOptionalArgs=""

        # Additional arguments for the scheduler API
        # must be valid C
        self.pyOptionalArgs=""

        # Prefix to add before the global FIFO buffer names
        self.prefix = ""

        # Experimental so disbaled by default
        self.memoryOptimization = False

        # Path to SDF module for Python simu 
        self.pathToSDFModule="../.."

        # When codeArray  is true, instead of using
        # function calls we parse un array giving
        # the index of functions to call in another array
        self.codeArray = False

        # True for an horizontal graphviz layout
        self.horizontal = True

        # Display FIFO buffers in graph instead of datatype
        self.displayFIFOBuf = False

    @property
    def debug(self):
        return (self.debugLimit > 0)
    

