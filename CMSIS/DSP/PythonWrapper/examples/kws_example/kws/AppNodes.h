/* ----------------------------------------------------------------------
 * Project:      CMSIS DSP Library
 * Title:        AppNodes.h
 * Description:  Application nodes for the C compute graph
 *
 * $Date:        16 March 2022
 *
 * Target Processor: Cortex-M and Cortex-A cores
 * -------------------------------------------------------------------- */
/*
 * Copyright (C) 2010-2022 ARM Limited or its affiliates. All rights reserved.
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


#ifndef _APPNODES_H_
#define _APPNODES_H_

#include <hal/nrf_pdm.h>
#include "coef.h"

#include <Arduino.h>
#include <HardwareSerial.h>

// When enabled, lots of additional trace is generated
//#define DEBUG

// Buffer to read samples into, each sample is 16-bits
// This is written by the PDM driver
extern short sampleBuffer[AUDIOBUFFER_LENGTH];

// Number of audio samples available in the audio buffer
extern volatile int samplesRead;

// Sink node. It is just printing the recognized word
template<typename IN, int inputSize> class Sink;

template<int inputSize>
class Sink<q15_t, inputSize>: public GenericSink<q15_t, inputSize>
{
public:
    Sink(FIFOBase<q15_t> &src):GenericSink<q15_t,inputSize>(src){};

    int run()
    {
        #if defined(DEBUG)
        Serial.println("==== Sink");
        #endif

        q15_t *b=this->getReadBuffer();

        if (b[0]==-1)
        {
            Serial.println("Unknown");
        };
        
        if (b[0]==0)
        {
            Serial.println("Yes");
        };

        return(0);
    };

};


// Source node. It is getting audio data from the PDM driver
template<typename OUT, int outputSize> class Source;

template<int outputSize>
class Source<q15_t,outputSize>: public GenericSource<q15_t,outputSize>
{
public:
    Source(FIFOBase<q15_t> &dst):GenericSource<q15_t,outputSize>(dst)
    {

    };

    int run(){
       
        #if defined(DEBUG)
        Serial.println("==== Source"); 
        #endif 
        q15_t *b=this->getWriteBuffer();

        // We wait until enough samples are available.
        // In a future version we may experiment with sleeping the board
        while(samplesRead<outputSize)
        {
            #if defined(DEBUG)
              Serial.print("Sample reads ");  
              Serial.println(samplesRead);  
            #endif
        };
        
        #if defined(DEBUG)
        Serial.println("Received");  
        #endif
       
        // We get the samples and update the 
        // sampleBuffer.
        // Since this buffer is also accessed by the IRQ, we need to disable it
        NVIC_DisableIRQ(PDM_IRQn);
        memcpy(b,sampleBuffer,sizeof(q15_t)*outputSize);
        if ((samplesRead-outputSize) > 0)
        {
            memmove(sampleBuffer,sampleBuffer+outputSize,sizeof(q15_t)*(samplesRead-outputSize));
        }
        samplesRead = samplesRead - outputSize;
        NVIC_EnableIRQ(PDM_IRQn);

        #if defined(DEBUG)
        Serial.print("After read : Sample reads ");  
        Serial.println(samplesRead);
        #endif
        

        return(0);
    };


};

template<typename IN, int inputSize,typename OUT,int outputSize> class FIR;


// FIR node
template<int inputSize>
class FIR<q15_t,inputSize,q15_t,inputSize>: public GenericNode<q15_t,inputSize,q15_t,inputSize>
{
public:
    FIR(FIFOBase<q15_t> &src,FIFOBase<q15_t> &dst):GenericNode<q15_t,inputSize,q15_t,inputSize>(src,dst){
        int blockSize=inputSize;
        int numTaps=10;
        int stateLength = numTaps + blockSize - 1;

        state=(q15_t*)malloc(stateLength * sizeof(q15_t*));
    };

    int run(){
        #if defined(DEBUG)
        Serial.println("==== FIR");
        #endif
        q15_t *a=this->getReadBuffer();
        q15_t *b=this->getWriteBuffer();
        int blockSize=inputSize;
        int stateLength = NUMTAPS + blockSize - 1;

        arm_status status=arm_fir_init_q15(&(this->firq15),NUMTAPS,fir_coefs,state,blockSize);

        arm_fir_q15(&(this->firq15),a,b,blockSize);
        return(0);
    };

arm_fir_instance_q15 firq15;
q15_t *state;

};

/* Not available in the older CMSIS-DSP version provided with Arduino.
So we copy the definition here */

arm_status arm_divide_q15(q15_t numerator,
  q15_t denominator,
  q15_t *quotient,
  int16_t *shift)
{
  int16_t sign=0;
  q31_t temp;
  int16_t shiftForNormalizing;

  *shift = 0;

  sign = (numerator>>15) ^ (denominator>>15);

  if (denominator == 0)
  {
     if (sign)
     {
        *quotient = 0x8000;
     }
     else
     {
        *quotient = 0x7FFF;
     }
     return(ARM_MATH_NANINF);
  }

  numerator = abs(numerator);
  denominator = abs(denominator);
  
  temp = ((q31_t)numerator << 15) / ((q31_t)denominator);

  shiftForNormalizing= 17 - __CLZ(temp);
  if (shiftForNormalizing > 0)
  {
     *shift = shiftForNormalizing;
     temp = temp >> shiftForNormalizing;
  }

  if (sign)
  {
    temp = -temp;
  }

  *quotient=temp;

  return(ARM_MATH_SUCCESS);
}


// We similar to the Python implementation
q15_t dsp_zcr_q15(q15_t *w,int blockSize)
{
    q15_t m;
    arm_mean_q15(w,blockSize,&m);

    // Negate can saturate so we use CMSIS-DSP function which is working on array (and we have a scalar)
    arm_negate_q15(&m,&m,1);
    arm_offset_q15(w,m,w,blockSize);
    
    int k=0;
    for(int i=0;i<blockSize-1;i++)
    {
         int f = w[i];
         int g = w[i+1];
         if ((((f>0) && (g<0)) || ((f<0) && (g>0))) && g>f)
         {
            k++;
         }
    }

    
    // k < len(f) so shift should be 0 except when k == len(f)
    // When k==len(f) normally quotient is 0x4000 and shift 1 and we convert
    // this to 0x7FFF

    q15_t quotient;
    int16_t shift;

    arm_status status=arm_divide_q15(k,blockSize-1,&quotient,&shift);

    if (shift==1)
    {
        arm_shift_q15(&quotient,shift,&quotient,1);
        return(quotient);
    }
    else
    {
        return(quotient);
    }
};

template<typename IN, int inputSize,typename OUT,int outputSize> class Feature;

template<int inputSize>
class Feature<q15_t,inputSize,q15_t,1>: public GenericNode<q15_t,inputSize,q15_t,1>
{
public:
    Feature(FIFOBase<q15_t> &src,FIFOBase<q15_t> &dst,const q15_t *window):
       GenericNode<q15_t,inputSize,q15_t,1>(src,dst),mWindow(window){
    };

    int run(){
        #if defined(DEBUG)
        Serial.println("==== Feature");
        #endif
        q15_t *a=this->getReadBuffer();
        q15_t *b=this->getWriteBuffer();

        arm_mult_q15(a,this->mWindow,a,inputSize);

        b[0] = dsp_zcr_q15(a,inputSize);

        return(0);
    };

const q15_t* mWindow;

};

template<typename IN, int inputSize,typename OUT,int outputSize> class KWS;

template<int inputSize>
class KWS<q15_t,inputSize,q15_t,1>: public GenericNode<q15_t,inputSize,q15_t,1>
{
public:
    KWS(FIFOBase<q15_t> &src,FIFOBase<q15_t> &dst,
    const q15_t* coef_q15,
    const int coef_shift,
    const q15_t intercept_q15,
    const int intercept_shift):GenericNode<q15_t,inputSize,q15_t,1>(src,dst),
    mCoef_q15(coef_q15),
    mCoef_shift(coef_shift),
    mIntercept_q15(intercept_q15),
    mIntercept_shift(intercept_shift)
    {
         
    };

    int run(){
        #if defined(DEBUG)
        Serial.println("==== KWS");
        #endif
        q15_t *a=this->getReadBuffer();
        q15_t *b=this->getWriteBuffer();

        q63_t res;
        arm_dot_prod_q15(this->mCoef_q15,a,inputSize,&res);
    
        q15_t scaled;
        arm_shift_q15(&(this->mIntercept_q15),this->mIntercept_shift-this->mCoef_shift,&scaled,1);
        // Because dot prod output is in Q34.30
        // and ret is on 64 bits
        q63_t scaled_Q30 = (q63_t)(scaled) << 15; 
    
        res = res + scaled_Q30;
    
        if (res<0)
        {
            b[0]=-1;
        }
        else
        {
            b[0]=0;
        }
        

        return(0);
    };

const q15_t* mCoef_q15;
const int mCoef_shift;
const q15_t mIntercept_q15;
const int mIntercept_shift;

};
#endif
