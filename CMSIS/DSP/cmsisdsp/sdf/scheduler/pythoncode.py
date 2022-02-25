###########################################
# Project:      CMSIS DSP Library
# Title:        pythoncode.py
# Description:  Generation of Python code for the static scheduler
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
    
    template = env.get_template("code.py")

    cfile=os.path.join(directory,"sched.py")


    with open(cfile,"w") as f:
         nbFifos = len(sched._graph._allFIFOs)
         print(template.render(fifos=sched._graph._allFIFOs,
            nbFifos=nbFifos,
            nodes=sched.nodes,
            schedule=sched.schedule,
            config=config,
            sched=sched
            ),file=f)