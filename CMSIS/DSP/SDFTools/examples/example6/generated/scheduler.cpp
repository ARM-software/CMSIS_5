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
#define FIFOSIZE0 256
#define FIFOSIZE1 256
#define FIFOSIZE2 13
#define FIFOSIZE3 26

#define BUFFERSIZE0 256
float32_t buf0[BUFFERSIZE0]={0};

#define BUFFERSIZE1 256
float32_t buf1[BUFFERSIZE1]={0};

#define BUFFERSIZE2 13
float32_t buf2[BUFFERSIZE2]={0};

#define BUFFERSIZE3 26
float32_t buf3[BUFFERSIZE3]={0};


uint32_t scheduler(int *error,arm_mfcc_instance_f32 *mfccConfig)
{
    int sdfError=0;
    uint32_t nbSchedule=0;
    int32_t debugCounter=1;

    /*
    Create FIFOs objects
    */
    FIFO<float32_t,FIFOSIZE0,0> fifo0(buf0);
    FIFO<float32_t,FIFOSIZE1,1> fifo1(buf1);
    FIFO<float32_t,FIFOSIZE2,1> fifo2(buf2);
    FIFO<float32_t,FIFOSIZE3,0> fifo3(buf3);

    /* 
    Create node objects
    */
    SlidingBuffer<float32_t,256,128> audioWin(fifo0,fifo1);
    MFCC<float32_t,256,float32_t,13> mfcc(fifo1,fifo2,mfccConfig);
    SlidingBuffer<float32_t,26,13> mfccWin(fifo2,fifo3);
    FileSink<float32_t,13> sink(fifo3,"output_example6.txt");
    FileSource<float32_t,192> src(fifo0,"input_example6.txt");

    /* Run several schedule iterations */
    while((sdfError==0) && (debugCounter > 0))
    {
       /* Run a schedule iteration */
       sdfError = src.run();
       CHECKERROR;
       sdfError = audioWin.run();
       CHECKERROR;
       sdfError = mfcc.run();
       CHECKERROR;
       sdfError = mfccWin.run();
       CHECKERROR;
       sdfError = sink.run();
       CHECKERROR;
       sdfError = sink.run();
       CHECKERROR;
       sdfError = src.run();
       CHECKERROR;
       sdfError = audioWin.run();
       CHECKERROR;
       sdfError = mfcc.run();
       CHECKERROR;
       sdfError = mfccWin.run();
       CHECKERROR;
       sdfError = sink.run();
       CHECKERROR;
       sdfError = sink.run();
       CHECKERROR;
       sdfError = audioWin.run();
       CHECKERROR;
       sdfError = mfcc.run();
       CHECKERROR;
       sdfError = mfccWin.run();
       CHECKERROR;
       sdfError = sink.run();
       CHECKERROR;
       sdfError = sink.run();
       CHECKERROR;

       debugCounter--;
       nbSchedule++;
    }
    *error=sdfError;
    return(nbSchedule);
}
