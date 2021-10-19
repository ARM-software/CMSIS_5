/*

Generated with CMSIS-DSP SDF Scripts.
The generated code is not covered by CMSIS-DSP license.

The support classes and code is covered by CMSIS-DSP license.

*/


#include "arm_math.h"
#include "custom.h"
#include "GenericNodes.h"
#include "AppNodes.h"
#include "scheduler.h"

/***********

FIFO buffers

************/
#define FIFOSIZE0 330
#define FIFOSIZE1 160
#define FIFOSIZE2 160
#define FIFOSIZE3 160
#define FIFOSIZE4 160
#define FIFOSIZE5 320
#define FIFOSIZE6 640
#define FIFOSIZE7 250
#define FIFOSIZE8 500

#define BUFFERSIZE0 330
float32_t buf0[BUFFERSIZE0]={0};

#define BUFFERSIZE1 160
float32_t buf1[BUFFERSIZE1]={0};

#define BUFFERSIZE2 160
float32_t buf2[BUFFERSIZE2]={0};

#define BUFFERSIZE3 160
float32_t buf3[BUFFERSIZE3]={0};

#define BUFFERSIZE4 160
float32_t buf4[BUFFERSIZE4]={0};

#define BUFFERSIZE5 320
float32_t buf5[BUFFERSIZE5]={0};

#define BUFFERSIZE6 640
float32_t buf6[BUFFERSIZE6]={0};

#define BUFFERSIZE7 250
float32_t buf7[BUFFERSIZE7]={0};

#define BUFFERSIZE8 500
float32_t buf8[BUFFERSIZE8]={0};


uint32_t scheduler(int *error,int opt1,int opt2)
{
    int sdfError=0;
    uint32_t nbSchedule=0;
    int32_t debugCounter=1;

    /*
    Create FIFOs objects
    */
    FIFO<float32_t,FIFOSIZE0,0> fifo0(buf0,10);
    FIFO<float32_t,FIFOSIZE1,1> fifo1(buf1);
    FIFO<float32_t,FIFOSIZE2,1> fifo2(buf2);
    FIFO<float32_t,FIFOSIZE3,1> fifo3(buf3);
    FIFO<float32_t,FIFOSIZE4,1> fifo4(buf4);
    FIFO<float32_t,FIFOSIZE5,0> fifo5(buf5);
    FIFO<float32_t,FIFOSIZE6,1> fifo6(buf6);
    FIFO<float32_t,FIFOSIZE7,0> fifo7(buf7);
    FIFO<float32_t,FIFOSIZE8,1> fifo8(buf8);

    /* 
    Create node objects
    */
    TFLite<float32_t,500> TFLite(fifo8);
    SlidingBuffer<float32_t,640,320> audioWin(fifo5,fifo6);
    MFCC<float32_t,640,float32_t,10> mfcc(fifo6,fifo7);
    SlidingBuffer<float32_t,500,250> mfccWind(fifo7,fifo8);
    StereoSource<float32_t,320> src(fifo0);
    Unzip<float32_t,320,float32_t,160,float32_t,160> toMono(fifo0,fifo1,fifo2);

    /* Run several schedule iterations */
    while((sdfError==0) && (debugCounter > 0))
    {
       /* Run a schedule iteration */
       sdfError = src.run();
       CHECKERROR;
       sdfError = toMono.run();
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo2.getReadBuffer(160);
         o2=fifo4.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo1.getReadBuffer(160);
         o2=fifo3.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* i1;
         float32_t* o2;
         i0=fifo3.getReadBuffer(160);
         i1=fifo4.getReadBuffer(160);
         o2=fifo5.getWriteBuffer(160);
         arm_add_f32(i0,i1,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       sdfError = src.run();
       CHECKERROR;
       sdfError = toMono.run();
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo2.getReadBuffer(160);
         o2=fifo4.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo1.getReadBuffer(160);
         o2=fifo3.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* i1;
         float32_t* o2;
         i0=fifo3.getReadBuffer(160);
         i1=fifo4.getReadBuffer(160);
         o2=fifo5.getWriteBuffer(160);
         arm_add_f32(i0,i1,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       sdfError = audioWin.run();
       CHECKERROR;
       sdfError = mfcc.run();
       CHECKERROR;
       sdfError = src.run();
       CHECKERROR;
       sdfError = toMono.run();
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo2.getReadBuffer(160);
         o2=fifo4.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo1.getReadBuffer(160);
         o2=fifo3.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* i1;
         float32_t* o2;
         i0=fifo3.getReadBuffer(160);
         i1=fifo4.getReadBuffer(160);
         o2=fifo5.getWriteBuffer(160);
         arm_add_f32(i0,i1,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       sdfError = src.run();
       CHECKERROR;
       sdfError = toMono.run();
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo2.getReadBuffer(160);
         o2=fifo4.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo1.getReadBuffer(160);
         o2=fifo3.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* i1;
         float32_t* o2;
         i0=fifo3.getReadBuffer(160);
         i1=fifo4.getReadBuffer(160);
         o2=fifo5.getWriteBuffer(160);
         arm_add_f32(i0,i1,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       sdfError = audioWin.run();
       CHECKERROR;
       sdfError = mfcc.run();
       CHECKERROR;
       sdfError = src.run();
       CHECKERROR;
       sdfError = toMono.run();
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo2.getReadBuffer(160);
         o2=fifo4.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo1.getReadBuffer(160);
         o2=fifo3.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* i1;
         float32_t* o2;
         i0=fifo3.getReadBuffer(160);
         i1=fifo4.getReadBuffer(160);
         o2=fifo5.getWriteBuffer(160);
         arm_add_f32(i0,i1,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       sdfError = src.run();
       CHECKERROR;
       sdfError = toMono.run();
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo2.getReadBuffer(160);
         o2=fifo4.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo1.getReadBuffer(160);
         o2=fifo3.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* i1;
         float32_t* o2;
         i0=fifo3.getReadBuffer(160);
         i1=fifo4.getReadBuffer(160);
         o2=fifo5.getWriteBuffer(160);
         arm_add_f32(i0,i1,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       sdfError = audioWin.run();
       CHECKERROR;
       sdfError = mfcc.run();
       CHECKERROR;
       sdfError = src.run();
       CHECKERROR;
       sdfError = toMono.run();
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo2.getReadBuffer(160);
         o2=fifo4.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo1.getReadBuffer(160);
         o2=fifo3.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* i1;
         float32_t* o2;
         i0=fifo3.getReadBuffer(160);
         i1=fifo4.getReadBuffer(160);
         o2=fifo5.getWriteBuffer(160);
         arm_add_f32(i0,i1,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       sdfError = src.run();
       CHECKERROR;
       sdfError = toMono.run();
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo2.getReadBuffer(160);
         o2=fifo4.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo1.getReadBuffer(160);
         o2=fifo3.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* i1;
         float32_t* o2;
         i0=fifo3.getReadBuffer(160);
         i1=fifo4.getReadBuffer(160);
         o2=fifo5.getWriteBuffer(160);
         arm_add_f32(i0,i1,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       sdfError = audioWin.run();
       CHECKERROR;
       sdfError = mfcc.run();
       CHECKERROR;
       sdfError = src.run();
       CHECKERROR;
       sdfError = toMono.run();
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo2.getReadBuffer(160);
         o2=fifo4.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo1.getReadBuffer(160);
         o2=fifo3.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* i1;
         float32_t* o2;
         i0=fifo3.getReadBuffer(160);
         i1=fifo4.getReadBuffer(160);
         o2=fifo5.getWriteBuffer(160);
         arm_add_f32(i0,i1,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       sdfError = src.run();
       CHECKERROR;
       sdfError = toMono.run();
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo2.getReadBuffer(160);
         o2=fifo4.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo1.getReadBuffer(160);
         o2=fifo3.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* i1;
         float32_t* o2;
         i0=fifo3.getReadBuffer(160);
         i1=fifo4.getReadBuffer(160);
         o2=fifo5.getWriteBuffer(160);
         arm_add_f32(i0,i1,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       sdfError = audioWin.run();
       CHECKERROR;
       sdfError = mfcc.run();
       CHECKERROR;
       sdfError = src.run();
       CHECKERROR;
       sdfError = toMono.run();
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo2.getReadBuffer(160);
         o2=fifo4.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo1.getReadBuffer(160);
         o2=fifo3.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* i1;
         float32_t* o2;
         i0=fifo3.getReadBuffer(160);
         i1=fifo4.getReadBuffer(160);
         o2=fifo5.getWriteBuffer(160);
         arm_add_f32(i0,i1,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       sdfError = src.run();
       CHECKERROR;
       sdfError = toMono.run();
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo2.getReadBuffer(160);
         o2=fifo4.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo1.getReadBuffer(160);
         o2=fifo3.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* i1;
         float32_t* o2;
         i0=fifo3.getReadBuffer(160);
         i1=fifo4.getReadBuffer(160);
         o2=fifo5.getWriteBuffer(160);
         arm_add_f32(i0,i1,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       sdfError = audioWin.run();
       CHECKERROR;
       sdfError = mfcc.run();
       CHECKERROR;
       sdfError = src.run();
       CHECKERROR;
       sdfError = toMono.run();
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo2.getReadBuffer(160);
         o2=fifo4.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo1.getReadBuffer(160);
         o2=fifo3.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* i1;
         float32_t* o2;
         i0=fifo3.getReadBuffer(160);
         i1=fifo4.getReadBuffer(160);
         o2=fifo5.getWriteBuffer(160);
         arm_add_f32(i0,i1,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       sdfError = src.run();
       CHECKERROR;
       sdfError = toMono.run();
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo2.getReadBuffer(160);
         o2=fifo4.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo1.getReadBuffer(160);
         o2=fifo3.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* i1;
         float32_t* o2;
         i0=fifo3.getReadBuffer(160);
         i1=fifo4.getReadBuffer(160);
         o2=fifo5.getWriteBuffer(160);
         arm_add_f32(i0,i1,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       sdfError = audioWin.run();
       CHECKERROR;
       sdfError = mfcc.run();
       CHECKERROR;
       sdfError = src.run();
       CHECKERROR;
       sdfError = toMono.run();
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo2.getReadBuffer(160);
         o2=fifo4.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo1.getReadBuffer(160);
         o2=fifo3.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* i1;
         float32_t* o2;
         i0=fifo3.getReadBuffer(160);
         i1=fifo4.getReadBuffer(160);
         o2=fifo5.getWriteBuffer(160);
         arm_add_f32(i0,i1,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       sdfError = src.run();
       CHECKERROR;
       sdfError = toMono.run();
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo2.getReadBuffer(160);
         o2=fifo4.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo1.getReadBuffer(160);
         o2=fifo3.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* i1;
         float32_t* o2;
         i0=fifo3.getReadBuffer(160);
         i1=fifo4.getReadBuffer(160);
         o2=fifo5.getWriteBuffer(160);
         arm_add_f32(i0,i1,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       sdfError = audioWin.run();
       CHECKERROR;
       sdfError = mfcc.run();
       CHECKERROR;
       sdfError = src.run();
       CHECKERROR;
       sdfError = toMono.run();
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo2.getReadBuffer(160);
         o2=fifo4.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo1.getReadBuffer(160);
         o2=fifo3.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* i1;
         float32_t* o2;
         i0=fifo3.getReadBuffer(160);
         i1=fifo4.getReadBuffer(160);
         o2=fifo5.getWriteBuffer(160);
         arm_add_f32(i0,i1,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       sdfError = src.run();
       CHECKERROR;
       sdfError = toMono.run();
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo2.getReadBuffer(160);
         o2=fifo4.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo1.getReadBuffer(160);
         o2=fifo3.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* i1;
         float32_t* o2;
         i0=fifo3.getReadBuffer(160);
         i1=fifo4.getReadBuffer(160);
         o2=fifo5.getWriteBuffer(160);
         arm_add_f32(i0,i1,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       sdfError = audioWin.run();
       CHECKERROR;
       sdfError = mfcc.run();
       CHECKERROR;
       sdfError = src.run();
       CHECKERROR;
       sdfError = toMono.run();
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo2.getReadBuffer(160);
         o2=fifo4.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo1.getReadBuffer(160);
         o2=fifo3.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* i1;
         float32_t* o2;
         i0=fifo3.getReadBuffer(160);
         i1=fifo4.getReadBuffer(160);
         o2=fifo5.getWriteBuffer(160);
         arm_add_f32(i0,i1,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       sdfError = src.run();
       CHECKERROR;
       sdfError = toMono.run();
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo2.getReadBuffer(160);
         o2=fifo4.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo1.getReadBuffer(160);
         o2=fifo3.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* i1;
         float32_t* o2;
         i0=fifo3.getReadBuffer(160);
         i1=fifo4.getReadBuffer(160);
         o2=fifo5.getWriteBuffer(160);
         arm_add_f32(i0,i1,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       sdfError = audioWin.run();
       CHECKERROR;
       sdfError = mfcc.run();
       CHECKERROR;
       sdfError = src.run();
       CHECKERROR;
       sdfError = toMono.run();
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo2.getReadBuffer(160);
         o2=fifo4.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo1.getReadBuffer(160);
         o2=fifo3.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* i1;
         float32_t* o2;
         i0=fifo3.getReadBuffer(160);
         i1=fifo4.getReadBuffer(160);
         o2=fifo5.getWriteBuffer(160);
         arm_add_f32(i0,i1,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       sdfError = src.run();
       CHECKERROR;
       sdfError = toMono.run();
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo2.getReadBuffer(160);
         o2=fifo4.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo1.getReadBuffer(160);
         o2=fifo3.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* i1;
         float32_t* o2;
         i0=fifo3.getReadBuffer(160);
         i1=fifo4.getReadBuffer(160);
         o2=fifo5.getWriteBuffer(160);
         arm_add_f32(i0,i1,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       sdfError = audioWin.run();
       CHECKERROR;
       sdfError = mfcc.run();
       CHECKERROR;
       sdfError = src.run();
       CHECKERROR;
       sdfError = toMono.run();
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo2.getReadBuffer(160);
         o2=fifo4.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo1.getReadBuffer(160);
         o2=fifo3.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* i1;
         float32_t* o2;
         i0=fifo3.getReadBuffer(160);
         i1=fifo4.getReadBuffer(160);
         o2=fifo5.getWriteBuffer(160);
         arm_add_f32(i0,i1,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       sdfError = src.run();
       CHECKERROR;
       sdfError = toMono.run();
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo2.getReadBuffer(160);
         o2=fifo4.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo1.getReadBuffer(160);
         o2=fifo3.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* i1;
         float32_t* o2;
         i0=fifo3.getReadBuffer(160);
         i1=fifo4.getReadBuffer(160);
         o2=fifo5.getWriteBuffer(160);
         arm_add_f32(i0,i1,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       sdfError = audioWin.run();
       CHECKERROR;
       sdfError = mfcc.run();
       CHECKERROR;
       sdfError = src.run();
       CHECKERROR;
       sdfError = toMono.run();
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo2.getReadBuffer(160);
         o2=fifo4.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo1.getReadBuffer(160);
         o2=fifo3.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* i1;
         float32_t* o2;
         i0=fifo3.getReadBuffer(160);
         i1=fifo4.getReadBuffer(160);
         o2=fifo5.getWriteBuffer(160);
         arm_add_f32(i0,i1,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       sdfError = src.run();
       CHECKERROR;
       sdfError = toMono.run();
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo2.getReadBuffer(160);
         o2=fifo4.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo1.getReadBuffer(160);
         o2=fifo3.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* i1;
         float32_t* o2;
         i0=fifo3.getReadBuffer(160);
         i1=fifo4.getReadBuffer(160);
         o2=fifo5.getWriteBuffer(160);
         arm_add_f32(i0,i1,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       sdfError = audioWin.run();
       CHECKERROR;
       sdfError = mfcc.run();
       CHECKERROR;
       sdfError = src.run();
       CHECKERROR;
       sdfError = toMono.run();
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo2.getReadBuffer(160);
         o2=fifo4.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo1.getReadBuffer(160);
         o2=fifo3.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* i1;
         float32_t* o2;
         i0=fifo3.getReadBuffer(160);
         i1=fifo4.getReadBuffer(160);
         o2=fifo5.getWriteBuffer(160);
         arm_add_f32(i0,i1,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       sdfError = src.run();
       CHECKERROR;
       sdfError = toMono.run();
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo2.getReadBuffer(160);
         o2=fifo4.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo1.getReadBuffer(160);
         o2=fifo3.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* i1;
         float32_t* o2;
         i0=fifo3.getReadBuffer(160);
         i1=fifo4.getReadBuffer(160);
         o2=fifo5.getWriteBuffer(160);
         arm_add_f32(i0,i1,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       sdfError = audioWin.run();
       CHECKERROR;
       sdfError = mfcc.run();
       CHECKERROR;
       sdfError = src.run();
       CHECKERROR;
       sdfError = toMono.run();
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo2.getReadBuffer(160);
         o2=fifo4.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo1.getReadBuffer(160);
         o2=fifo3.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* i1;
         float32_t* o2;
         i0=fifo3.getReadBuffer(160);
         i1=fifo4.getReadBuffer(160);
         o2=fifo5.getWriteBuffer(160);
         arm_add_f32(i0,i1,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       sdfError = src.run();
       CHECKERROR;
       sdfError = toMono.run();
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo2.getReadBuffer(160);
         o2=fifo4.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo1.getReadBuffer(160);
         o2=fifo3.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* i1;
         float32_t* o2;
         i0=fifo3.getReadBuffer(160);
         i1=fifo4.getReadBuffer(160);
         o2=fifo5.getWriteBuffer(160);
         arm_add_f32(i0,i1,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       sdfError = audioWin.run();
       CHECKERROR;
       sdfError = mfcc.run();
       CHECKERROR;
       sdfError = src.run();
       CHECKERROR;
       sdfError = toMono.run();
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo2.getReadBuffer(160);
         o2=fifo4.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo1.getReadBuffer(160);
         o2=fifo3.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* i1;
         float32_t* o2;
         i0=fifo3.getReadBuffer(160);
         i1=fifo4.getReadBuffer(160);
         o2=fifo5.getWriteBuffer(160);
         arm_add_f32(i0,i1,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       sdfError = src.run();
       CHECKERROR;
       sdfError = toMono.run();
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo2.getReadBuffer(160);
         o2=fifo4.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo1.getReadBuffer(160);
         o2=fifo3.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* i1;
         float32_t* o2;
         i0=fifo3.getReadBuffer(160);
         i1=fifo4.getReadBuffer(160);
         o2=fifo5.getWriteBuffer(160);
         arm_add_f32(i0,i1,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       sdfError = audioWin.run();
       CHECKERROR;
       sdfError = mfcc.run();
       CHECKERROR;
       sdfError = src.run();
       CHECKERROR;
       sdfError = toMono.run();
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo2.getReadBuffer(160);
         o2=fifo4.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo1.getReadBuffer(160);
         o2=fifo3.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* i1;
         float32_t* o2;
         i0=fifo3.getReadBuffer(160);
         i1=fifo4.getReadBuffer(160);
         o2=fifo5.getWriteBuffer(160);
         arm_add_f32(i0,i1,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       sdfError = src.run();
       CHECKERROR;
       sdfError = toMono.run();
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo2.getReadBuffer(160);
         o2=fifo4.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo1.getReadBuffer(160);
         o2=fifo3.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* i1;
         float32_t* o2;
         i0=fifo3.getReadBuffer(160);
         i1=fifo4.getReadBuffer(160);
         o2=fifo5.getWriteBuffer(160);
         arm_add_f32(i0,i1,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       sdfError = audioWin.run();
       CHECKERROR;
       sdfError = mfcc.run();
       CHECKERROR;
       sdfError = src.run();
       CHECKERROR;
       sdfError = toMono.run();
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo2.getReadBuffer(160);
         o2=fifo4.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo1.getReadBuffer(160);
         o2=fifo3.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* i1;
         float32_t* o2;
         i0=fifo3.getReadBuffer(160);
         i1=fifo4.getReadBuffer(160);
         o2=fifo5.getWriteBuffer(160);
         arm_add_f32(i0,i1,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       sdfError = src.run();
       CHECKERROR;
       sdfError = toMono.run();
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo2.getReadBuffer(160);
         o2=fifo4.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo1.getReadBuffer(160);
         o2=fifo3.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* i1;
         float32_t* o2;
         i0=fifo3.getReadBuffer(160);
         i1=fifo4.getReadBuffer(160);
         o2=fifo5.getWriteBuffer(160);
         arm_add_f32(i0,i1,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       sdfError = audioWin.run();
       CHECKERROR;
       sdfError = mfcc.run();
       CHECKERROR;
       sdfError = src.run();
       CHECKERROR;
       sdfError = toMono.run();
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo2.getReadBuffer(160);
         o2=fifo4.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo1.getReadBuffer(160);
         o2=fifo3.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* i1;
         float32_t* o2;
         i0=fifo3.getReadBuffer(160);
         i1=fifo4.getReadBuffer(160);
         o2=fifo5.getWriteBuffer(160);
         arm_add_f32(i0,i1,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       sdfError = src.run();
       CHECKERROR;
       sdfError = toMono.run();
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo2.getReadBuffer(160);
         o2=fifo4.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo1.getReadBuffer(160);
         o2=fifo3.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* i1;
         float32_t* o2;
         i0=fifo3.getReadBuffer(160);
         i1=fifo4.getReadBuffer(160);
         o2=fifo5.getWriteBuffer(160);
         arm_add_f32(i0,i1,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       sdfError = audioWin.run();
       CHECKERROR;
       sdfError = mfcc.run();
       CHECKERROR;
       sdfError = src.run();
       CHECKERROR;
       sdfError = toMono.run();
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo2.getReadBuffer(160);
         o2=fifo4.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo1.getReadBuffer(160);
         o2=fifo3.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* i1;
         float32_t* o2;
         i0=fifo3.getReadBuffer(160);
         i1=fifo4.getReadBuffer(160);
         o2=fifo5.getWriteBuffer(160);
         arm_add_f32(i0,i1,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       sdfError = src.run();
       CHECKERROR;
       sdfError = toMono.run();
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo2.getReadBuffer(160);
         o2=fifo4.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo1.getReadBuffer(160);
         o2=fifo3.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* i1;
         float32_t* o2;
         i0=fifo3.getReadBuffer(160);
         i1=fifo4.getReadBuffer(160);
         o2=fifo5.getWriteBuffer(160);
         arm_add_f32(i0,i1,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       sdfError = audioWin.run();
       CHECKERROR;
       sdfError = mfcc.run();
       CHECKERROR;
       sdfError = src.run();
       CHECKERROR;
       sdfError = toMono.run();
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo2.getReadBuffer(160);
         o2=fifo4.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo1.getReadBuffer(160);
         o2=fifo3.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* i1;
         float32_t* o2;
         i0=fifo3.getReadBuffer(160);
         i1=fifo4.getReadBuffer(160);
         o2=fifo5.getWriteBuffer(160);
         arm_add_f32(i0,i1,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       sdfError = src.run();
       CHECKERROR;
       sdfError = toMono.run();
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo2.getReadBuffer(160);
         o2=fifo4.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo1.getReadBuffer(160);
         o2=fifo3.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* i1;
         float32_t* o2;
         i0=fifo3.getReadBuffer(160);
         i1=fifo4.getReadBuffer(160);
         o2=fifo5.getWriteBuffer(160);
         arm_add_f32(i0,i1,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       sdfError = audioWin.run();
       CHECKERROR;
       sdfError = mfcc.run();
       CHECKERROR;
       sdfError = src.run();
       CHECKERROR;
       sdfError = toMono.run();
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo2.getReadBuffer(160);
         o2=fifo4.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo1.getReadBuffer(160);
         o2=fifo3.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* i1;
         float32_t* o2;
         i0=fifo3.getReadBuffer(160);
         i1=fifo4.getReadBuffer(160);
         o2=fifo5.getWriteBuffer(160);
         arm_add_f32(i0,i1,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       sdfError = src.run();
       CHECKERROR;
       sdfError = toMono.run();
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo2.getReadBuffer(160);
         o2=fifo4.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo1.getReadBuffer(160);
         o2=fifo3.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* i1;
         float32_t* o2;
         i0=fifo3.getReadBuffer(160);
         i1=fifo4.getReadBuffer(160);
         o2=fifo5.getWriteBuffer(160);
         arm_add_f32(i0,i1,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       sdfError = audioWin.run();
       CHECKERROR;
       sdfError = mfcc.run();
       CHECKERROR;
       sdfError = src.run();
       CHECKERROR;
       sdfError = toMono.run();
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo2.getReadBuffer(160);
         o2=fifo4.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo1.getReadBuffer(160);
         o2=fifo3.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* i1;
         float32_t* o2;
         i0=fifo3.getReadBuffer(160);
         i1=fifo4.getReadBuffer(160);
         o2=fifo5.getWriteBuffer(160);
         arm_add_f32(i0,i1,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       sdfError = src.run();
       CHECKERROR;
       sdfError = toMono.run();
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo2.getReadBuffer(160);
         o2=fifo4.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo1.getReadBuffer(160);
         o2=fifo3.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* i1;
         float32_t* o2;
         i0=fifo3.getReadBuffer(160);
         i1=fifo4.getReadBuffer(160);
         o2=fifo5.getWriteBuffer(160);
         arm_add_f32(i0,i1,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       sdfError = audioWin.run();
       CHECKERROR;
       sdfError = mfcc.run();
       CHECKERROR;
       sdfError = src.run();
       CHECKERROR;
       sdfError = toMono.run();
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo2.getReadBuffer(160);
         o2=fifo4.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo1.getReadBuffer(160);
         o2=fifo3.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* i1;
         float32_t* o2;
         i0=fifo3.getReadBuffer(160);
         i1=fifo4.getReadBuffer(160);
         o2=fifo5.getWriteBuffer(160);
         arm_add_f32(i0,i1,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       sdfError = src.run();
       CHECKERROR;
       sdfError = toMono.run();
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo2.getReadBuffer(160);
         o2=fifo4.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo1.getReadBuffer(160);
         o2=fifo3.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* i1;
         float32_t* o2;
         i0=fifo3.getReadBuffer(160);
         i1=fifo4.getReadBuffer(160);
         o2=fifo5.getWriteBuffer(160);
         arm_add_f32(i0,i1,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       sdfError = audioWin.run();
       CHECKERROR;
       sdfError = mfcc.run();
       CHECKERROR;
       sdfError = src.run();
       CHECKERROR;
       sdfError = toMono.run();
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo2.getReadBuffer(160);
         o2=fifo4.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo1.getReadBuffer(160);
         o2=fifo3.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* i1;
         float32_t* o2;
         i0=fifo3.getReadBuffer(160);
         i1=fifo4.getReadBuffer(160);
         o2=fifo5.getWriteBuffer(160);
         arm_add_f32(i0,i1,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       sdfError = src.run();
       CHECKERROR;
       sdfError = toMono.run();
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo2.getReadBuffer(160);
         o2=fifo4.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* o2;
         i0=fifo1.getReadBuffer(160);
         o2=fifo3.getWriteBuffer(160);
         arm_scale_f32(i0,HALF,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       {
         float32_t* i0;
         float32_t* i1;
         float32_t* o2;
         i0=fifo3.getReadBuffer(160);
         i1=fifo4.getReadBuffer(160);
         o2=fifo5.getWriteBuffer(160);
         arm_add_f32(i0,i1,o2,160);
         sdfError = 0;
       }
       CHECKERROR;
       sdfError = audioWin.run();
       CHECKERROR;
       sdfError = mfcc.run();
       CHECKERROR;
       sdfError = mfccWind.run();
       CHECKERROR;
       sdfError = TFLite.run();
       CHECKERROR;

       debugCounter--;
       nbSchedule++;
    }
    *error=sdfError;
    return(nbSchedule);
}
