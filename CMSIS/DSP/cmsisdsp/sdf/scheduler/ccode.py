###########################################
# Project:      CMSIS DSP Library
# Title:        ccode.py
# Description:  C++ code generator for SDF scheduler
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
from jinja2 import Environment, PackageLoader, select_autoescape
import os.path
import pathlib
from .config import *

def gencode(sched,directory,config):


    env = Environment(
       loader=PackageLoader("cmsisdsp.sdf.scheduler"),
       autoescape=select_autoescape(),
       trim_blocks=True
    )
    
    schedDescription=""

    if config.codeArray:
       ctemplate = env.get_template("codeArray.cpp")
       nb = 0
       for s in sched.schedule:
         schedDescription = schedDescription + ("%d," % sched.nodes[s].codeID)
         nb = nb + 1
         if nb == 40:
            nb=0 
            schedDescription = schedDescription + "\n"
    else:
       ctemplate = env.get_template("code.cpp")
    htemplate = env.get_template("code.h")


    cfile=os.path.join(directory,"scheduler.cpp")
    hfile=os.path.join(directory,"scheduler.h")

    nbFifos = len(sched._graph._allFIFOs)
    
    with open(cfile,"w") as f:
         print(ctemplate.render(fifos=sched._graph._allFIFOs,
            nbFifos=nbFifos,
            nbNodes=len(sched.nodes),
            nodes=sched.nodes,
            pureNodes=sched.pureNodes,
            schedule=sched.schedule,
            schedLen=len(sched.schedule),
            config=config,
            sched=sched,
            schedDescription=schedDescription
            ),file=f)

    with open(hfile,"w") as f:
         print(htemplate.render(fifos=sched._graph._allFIFOs,
            nbFifos=nbFifos,
            nodes=sched.nodes,
            schedule=sched.schedule,
            config=config,
            sched=sched
            ),file=f)