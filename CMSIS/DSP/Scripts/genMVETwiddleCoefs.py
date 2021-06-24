import numpy as np
import math
import argparse
import sys

sys.path.append("PatternGeneration")

import Tools

parser = argparse.ArgumentParser(description='Generate C arrays')
parser.add_argument('-f', nargs='?',type = str, default="../Source/CommonTables/arm_mve_tables.c", help="C File path")
parser.add_argument('-f16', nargs='?',type = str, default="../Source/CommonTables/arm_mve_tables_f16.c", help="C File path")
parser.add_argument('-he', nargs='?',type = str, default="../Include/arm_mve_tables.h", help="H File path")
parser.add_argument('-he16', nargs='?',type = str, default="../Include/arm_mve_tables_f16.h", help="H File path")

args = parser.parse_args()

COLLIM = 80 

condition="""#if !defined(ARM_DSP_CONFIG_TABLES) || defined(ARM_ALL_FFT_TABLES) || defined(ARM_TABLE_TWIDDLECOEF_%s_%d) || defined(ARM_TABLE_TWIDDLECOEF_%s_%d)
"""

F32 = 1
F16 = 2
Q31 = 3
Q15 = 4
Q7  = 5

def printCUInt32Array(f,name,arr):
    nb = 0
    print("uint32_t %s[%d]={" % (name,len(arr)),file=f)

    for d in arr:
        val = "%d," % d
        nb = nb + len(val)
        if nb > COLLIM:
            print("",file=f)
            nb = len(val)
        print(val,end="",file=f)

    print("};\n",file=f)

def printCFloat32Array(f,name,arr):
    nb = 0
    print("float32_t %s[%d]={" % (name,len(arr)),file=f)

    for d in arr:
        val = "%.20ff," % d
        nb = nb + len(val)
        if nb > COLLIM:
            print("",file=f)
            nb = len(val)
        print(val,end="",file=f)

    print("};\n",file=f)

def printCFloat16Array(f,name,arr):
    nb = 0
    print("float16_t %s[%d]={" % (name,len(arr)),file=f)

    for d in arr:
        val = "(float16_t)%.20ff," % d
        nb = nb + len(val)
        if nb > COLLIM:
            print("",file=f)
            nb = len(val)
        print(val,end="",file=f)

    print("};\n",file=f)

def printCQ31Array(f,name,arr):
    nb = 0
    print("q31_t %s[%d]={" % (name,len(arr)),file=f)

    for d in arr:
        val = "%s," % Tools.to_q31(d)
        nb = nb + len(val)
        if nb > COLLIM:
            print("",file=f)
            nb = len(val)
        print(val,end="",file=f)

    print("};\n",file=f)

def printCQ15Array(f,name,arr):
    nb = 0
    print("q15_t %s[%d]={" % (name,len(arr)),file=f)

    for d in arr:
        val = "%s," % Tools.to_q15(d)
        nb = nb + len(val)
        if nb > COLLIM:
            print("",file=f)
            nb = len(val)
        print(val,end="",file=f)

    print("};\n",file=f)

def printCQ7Array(f,name,arr):
    nb = 0
    print("q7_t %s[%d]={" % (name,len(arr)),file=f)

    for d in arr:
        val = "%s," % Tools.to_q7(d)
        nb = nb + len(val)
        if nb > COLLIM:
            print("",file=f)
            nb = len(val)
        print(val,end="",file=f)

    print("};\n",file=f)

def printHUInt32Array(f,name,arr):
 print("extern uint32_t %s[%d];" % (name,len(arr)),file=f)

def printHFloat32Array(f,name,arr):
 print("extern float32_t %s[%d];" % (name,len(arr)),file=f)

def printHFloat16Array(f,name,arr):
 print("extern float16_t %s[%d];" % (name,len(arr)),file=f)

def printHQ31Array(f,name,arr):
 print("extern q31_t %s[%d];" % (name,len(arr)),file=f)

def printHQ15Array(f,name,arr):
 print("extern q15_t %s[%d];" % (name,len(arr)),file=f)

def printHQ7Array(f,name,arr):
 print("extern q7_t %s[%d];" % (name,len(arr)),file=f)

def twiddle(n):
    a=2.0*math.pi*np.linspace(0,n,num=n,endpoint=False)/n
    c=np.cos(a)
    s=np.sin(a)

    r = np.empty((c.size + s.size,), dtype=c.dtype)
    r[0::2] = c
    r[1::2] = s
    return(r)

def reorderTwiddle(theType,conjugate,f,h,n):
    numStages = 6
    coefs= twiddle(n)

    
    if n == 4096:                                                                                
       numStages = 6  
       arraySize = 1364                                                
                                                                                                   
    if n == 1024:                                                                                
       numStages = 5  
       arraySize = 340                                                                     
                                                                                                   
    if n == 256:                                                                                 
       numStages = 4
       arraySize = 84                                                                        

    if n == 64:                                                                                  
       numStages = 3 
       arraySize = 20                                                                       
                                                                                                   
    if n == 16:                                                                                  
       numStages = 2 
       arraySize = 4

    incr = 1
    nbOfElt = n

    maxNb = 0

    tab1 = np.zeros(2*arraySize)
    tab2 = np.zeros(2*arraySize)
    tab3 = np.zeros(2*arraySize)

    tab1Index=0
    tab2Index=0
    tab3Index=0

    tab1Offset = np.zeros(numStages)
    tab2Offset = np.zeros(numStages)
    tab3Offset = np.zeros(numStages)



    for stage in range(0,numStages-1):  
        nbOfElt = nbOfElt >> 2                                                               
        pVectCoef1 = 0
        pVectCoef2 = 0
        pVectCoef3 = 0

        tab1Offset[stage] = tab1Index
        tab2Offset[stage] = tab2Index
        tab3Offset[stage] = tab3Index
        
        for i in range(0,nbOfElt):
            tab1[tab1Index] = coefs[pVectCoef1]  
            if not conjugate:                             
               tab1[tab1Index + 1] = coefs[pVectCoef1 + 1]
            else:
               tab1[tab1Index + 1] = -coefs[pVectCoef1 + 1]
            tab1Index = tab1Index + 2
            pVectCoef1 = pVectCoef1 + (incr * 1 * 2)

            tab2[tab2Index] = coefs[pVectCoef2]      
            if not conjugate:                         
               tab2[tab2Index + 1] = coefs[pVectCoef2 + 1]
            else:
               tab2[tab2Index + 1] = -coefs[pVectCoef2 + 1]
            tab2Index = tab2Index + 2
            pVectCoef2 = pVectCoef2 + (incr * 2 * 2)

            tab3[tab3Index] = coefs[pVectCoef3] 
            if not conjugate:                              
               tab3[tab3Index + 1] = coefs[pVectCoef3 + 1]
            else:
              tab3[tab3Index + 1] = -coefs[pVectCoef3 + 1]
            tab3Index = tab3Index + 2
            pVectCoef3 = pVectCoef3 + (incr * 3 * 2)

            maxNb = maxNb + 1

        incr = 4 * incr

    # F32 SECTION FOR THIS FFT LENGTH
    if theType == F32:
       print(condition % ("F32",n, "F32",n << 1),file=f)
       print(condition % ("F32",n, "F32",n << 1),file=h)
       printCUInt32Array(f,"rearranged_twiddle_tab_stride1_arr_%d_f32" % n,list(tab1Offset))
       printHUInt32Array(h,"rearranged_twiddle_tab_stride1_arr_%d_f32" % n,list(tab1Offset)) 
   
       printCUInt32Array(f,"rearranged_twiddle_tab_stride2_arr_%d_f32" % n,list(tab2Offset))
       printHUInt32Array(h,"rearranged_twiddle_tab_stride2_arr_%d_f32" % n,list(tab2Offset))
      
       printCUInt32Array(f,"rearranged_twiddle_tab_stride3_arr_%d_f32" % n,list(tab3Offset))
       printHUInt32Array(h,"rearranged_twiddle_tab_stride3_arr_%d_f32" % n,list(tab3Offset))
   
       printCFloat32Array(f,"rearranged_twiddle_stride1_%d_f32" % n,list(tab1))
       printHFloat32Array(h,"rearranged_twiddle_stride1_%d_f32" % n,list(tab1))
   
       printCFloat32Array(f,"rearranged_twiddle_stride2_%d_f32" % n,list(tab2))
       printHFloat32Array(h,"rearranged_twiddle_stride2_%d_f32" % n,list(tab2))
   
       printCFloat32Array(f,"rearranged_twiddle_stride3_%d_f32" % n,list(tab3))
       printHFloat32Array(h,"rearranged_twiddle_stride3_%d_f32" % n,list(tab3))
       print("#endif\n",file=f)
       print("#endif\n",file=h)

    # F16 SECTION FOR THIS FFT LENGTH
    if theType == F16:
       print(condition % ("F16",n, "F16",n << 1),file=f)
       print(condition % ("F16",n, "F16",n << 1),file=h)
       printCUInt32Array(f,"rearranged_twiddle_tab_stride1_arr_%d_f16" % n,list(tab1Offset))
       printHUInt32Array(h,"rearranged_twiddle_tab_stride1_arr_%d_f16" % n,list(tab1Offset)) 
   
       printCUInt32Array(f,"rearranged_twiddle_tab_stride2_arr_%d_f16" % n,list(tab2Offset))
       printHUInt32Array(h,"rearranged_twiddle_tab_stride2_arr_%d_f16" % n,list(tab2Offset))
      
       printCUInt32Array(f,"rearranged_twiddle_tab_stride3_arr_%d_f16" % n,list(tab3Offset))
       printHUInt32Array(h,"rearranged_twiddle_tab_stride3_arr_%d_f16" % n,list(tab3Offset))
   
       printCFloat16Array(f,"rearranged_twiddle_stride1_%d_f16" % n,list(tab1))
       printHFloat16Array(h,"rearranged_twiddle_stride1_%d_f16" % n,list(tab1))
   
       printCFloat16Array(f,"rearranged_twiddle_stride2_%d_f16" % n,list(tab2))
       printHFloat16Array(h,"rearranged_twiddle_stride2_%d_f16" % n,list(tab2))
   
       printCFloat16Array(f,"rearranged_twiddle_stride3_%d_f16" % n,list(tab3))
       printHFloat16Array(h,"rearranged_twiddle_stride3_%d_f16" % n,list(tab3))
       print("#endif\n",file=f)
       print("#endif\n",file=h)

    # Q31 SECTION FOR THIS FFT LENGTH
    if theType == Q31:
       print(condition % ("Q31",n, "Q31",n << 1),file=f)
       print(condition % ("Q31",n, "Q31",n << 1),file=h)
       printCUInt32Array(f,"rearranged_twiddle_tab_stride1_arr_%d_q31" % n,list(tab1Offset))
       printHUInt32Array(h,"rearranged_twiddle_tab_stride1_arr_%d_q31" % n,list(tab1Offset)) 
   
       printCUInt32Array(f,"rearranged_twiddle_tab_stride2_arr_%d_q31" % n,list(tab2Offset))
       printHUInt32Array(h,"rearranged_twiddle_tab_stride2_arr_%d_q31" % n,list(tab2Offset))
      
       printCUInt32Array(f,"rearranged_twiddle_tab_stride3_arr_%d_q31" % n,list(tab3Offset))
       printHUInt32Array(h,"rearranged_twiddle_tab_stride3_arr_%d_q31" % n,list(tab3Offset))
   
       printCQ31Array(f,"rearranged_twiddle_stride1_%d_q31" % n,list(tab1))
       printHQ31Array(h,"rearranged_twiddle_stride1_%d_q31" % n,list(tab1))
   
       printCQ31Array(f,"rearranged_twiddle_stride2_%d_q31" % n,list(tab2))
       printHQ31Array(h,"rearranged_twiddle_stride2_%d_q31" % n,list(tab2))
   
       printCQ31Array(f,"rearranged_twiddle_stride3_%d_q31" % n,list(tab3))
       printHQ31Array(h,"rearranged_twiddle_stride3_%d_q31" % n,list(tab3))
       print("#endif\n",file=f)
       print("#endif\n",file=h)

    # Q15 SECTION FOR THIS FFT LENGTH
    if theType == Q15:
       print(condition % ("Q15",n, "Q15",n << 1),file=f)
       print(condition % ("Q15",n, "Q15",n << 1),file=h)
       printCUInt32Array(f,"rearranged_twiddle_tab_stride1_arr_%d_q15" % n,list(tab1Offset))
       printHUInt32Array(h,"rearranged_twiddle_tab_stride1_arr_%d_q15" % n,list(tab1Offset)) 
   
       printCUInt32Array(f,"rearranged_twiddle_tab_stride2_arr_%d_q15" % n,list(tab2Offset))
       printHUInt32Array(h,"rearranged_twiddle_tab_stride2_arr_%d_q15" % n,list(tab2Offset))
      
       printCUInt32Array(f,"rearranged_twiddle_tab_stride3_arr_%d_q15" % n,list(tab3Offset))
       printHUInt32Array(h,"rearranged_twiddle_tab_stride3_arr_%d_q15" % n,list(tab3Offset))
   
       printCQ15Array(f,"rearranged_twiddle_stride1_%d_q15" % n,list(tab1))
       printHQ15Array(h,"rearranged_twiddle_stride1_%d_q15" % n,list(tab1))
   
       printCQ15Array(f,"rearranged_twiddle_stride2_%d_q15" % n,list(tab2))
       printHQ15Array(h,"rearranged_twiddle_stride2_%d_q15" % n,list(tab2))
   
       printCQ15Array(f,"rearranged_twiddle_stride3_%d_q15" % n,list(tab3))
       printHQ15Array(h,"rearranged_twiddle_stride3_%d_q15" % n,list(tab3))
       print("#endif\n",file=f)
       print("#endif\n",file=h)




#test = twiddle(16)
#printCFloat32Array("Test",list(test))

cheader="""/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        arm_mve_tables%s.c
 * Description:  common tables like fft twiddle factors, Bitreverse, reciprocal etc
 *               used for MVE implementation only
 *
 * $Date:        14. April 2020
 *
 * Target Processor: Cortex-M cores
 * -------------------------------------------------------------------- */
/*
 * Copyright (C) 2010-2020 ARM Limited or its affiliates. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

 """ 

cifdeMVEF="""#include "arm_math%s.h"

#if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)

#if !defined(ARM_DSP_CONFIG_TABLES) || defined(ARM_FFT_ALLOW_TABLES)
"""

cfooterMVEF="""

#endif /* !defined(ARM_DSP_CONFIG_TABLES) || defined(ARM_FFT_ALLOW_TABLES) */
#endif /* defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE) */
"""

cifdeMVEI="""#include "arm_math.h"

#if defined(ARM_MATH_MVEI) 

#if !defined(ARM_DSP_CONFIG_TABLES) || defined(ARM_FFT_ALLOW_TABLES)
"""

cfooterMVEI="""

#endif /* !defined(ARM_DSP_CONFIG_TABLES) || defined(ARM_FFT_ALLOW_TABLES) */
#endif /* defined(ARM_MATH_MVEI)  */
"""

hheader="""/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        arm_mve_tables%s.h
 * Description:  common tables like fft twiddle factors, Bitreverse, reciprocal etc
 *               used for MVE implementation only
 *
 * $Date:        14. April 2020
 *
 * Target Processor: Cortex-M cores
 * -------------------------------------------------------------------- */
/*
 * Copyright (C) 2010-2020 ARM Limited or its affiliates. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

 #ifndef _ARM_MVE_TABLES_%sH
 #define _ARM_MVE_TABLES_%sH

 #include "arm_math%s.h"

#ifdef   __cplusplus
extern "C"
{
#endif


 """ 

hifdefMVEF="""
#if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)

#if !defined(ARM_DSP_CONFIG_TABLES) || defined(ARM_FFT_ALLOW_TABLES)
"""

hfooterMVEF="""
#endif /* !defined(ARM_DSP_CONFIG_TABLES) || defined(ARM_FFT_ALLOW_TABLES) */

#endif /* defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE) */

"""

hifdefMVEI="""
#if defined(ARM_MATH_MVEI) 

#if !defined(ARM_DSP_CONFIG_TABLES) || defined(ARM_FFT_ALLOW_TABLES)
"""

hfooterMVEI="""
#endif /* !defined(ARM_DSP_CONFIG_TABLES) || defined(ARM_FFT_ALLOW_TABLES) */

#endif /* defined(ARM_MATH_MVEI) */

"""

hfooter="""
#ifdef   __cplusplus
}
#endif

#endif /*_ARM_MVE_TABLES_%sH*/
"""

with open(args.f16,'w') as f:
  with open(args.he16,'w') as h:
     print(cheader % "_f16",file=f)
     print(hheader % ("_f16","F16_","F16_","_f16"),file=h)

     print("#if defined(ARM_FLOAT16_SUPPORTED)",file=f)

     print(cifdeMVEF % "_f16",file=f)
     print(hifdefMVEF,file=h)
     reorderTwiddle(F16,False,f,h,16)
     reorderTwiddle(F16,False,f,h,64)
     reorderTwiddle(F16,False,f,h,256)
     reorderTwiddle(F16,False,f,h,1024)
     reorderTwiddle(F16,False,f,h,4096)
     print(cfooterMVEF,file=f)
     print(hfooterMVEF,file=h)

     print("#endif /* if defined(ARM_FLOAT16_SUPPORTED) */",file=f)

     print(hfooter % "F16_",file=h)

with open(args.f,'w') as f:
  with open(args.he,'w') as h:
     print(cheader % "",file=f)
     print(hheader % ("","","",""),file=h)

    
     print(cifdeMVEF % "",file=f)
     print(hifdefMVEF,file=h)
     reorderTwiddle(F32,False,f,h,16)
     reorderTwiddle(F32,False,f,h,64)
     reorderTwiddle(F32,False,f,h,256)
     reorderTwiddle(F32,False,f,h,1024)
     reorderTwiddle(F32,False,f,h,4096)
     print(cfooterMVEF,file=f)
     print(hfooterMVEF,file=h)

     print(cifdeMVEI,file=f)
     print(hifdefMVEI,file=h)
     reorderTwiddle(Q31,True,f,h,16)
     reorderTwiddle(Q31,True,f,h,64)
     reorderTwiddle(Q31,True,f,h,256)
     reorderTwiddle(Q31,True,f,h,1024)
     reorderTwiddle(Q31,True,f,h,4096)
     print(cfooterMVEI,file=f)
     print(hfooterMVEI,file=h)

     print(cifdeMVEI,file=f)
     print(hifdefMVEI,file=h)
     reorderTwiddle(Q15,True,f,h,16)
     reorderTwiddle(Q15,True,f,h,64)
     reorderTwiddle(Q15,True,f,h,256)
     reorderTwiddle(Q15,True,f,h,1024)
     reorderTwiddle(Q15,True,f,h,4096)
     print(cfooterMVEI,file=f)
     print(hfooterMVEI,file=h)

     #print(cifdeMVEI,file=f)
     #print(hifdefMVEI,file=h)
     #reorderTwiddle(Q7,True,f,h,16)
     #reorderTwiddle(Q7,True,f,h,64)
     #reorderTwiddle(Q7,True,f,h,256)
     #reorderTwiddle(Q7,True,f,h,1024)
     #reorderTwiddle(Q7,True,f,h,4096)
     #print(cfooterMVEI,file=f)
     #print(hfooterMVEI,file=h)

     print(hfooter % "",file=h)
