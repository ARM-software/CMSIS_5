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
#define FIFOSIZE0 11
#define FIFOSIZE1 5

#define BUFFERSIZE0 11
float32_t buf0[BUFFERSIZE0]={0};

#define BUFFERSIZE1 5
float32_t buf1[BUFFERSIZE1]={0};


uint32_t scheduler(int *error,int someVariable)
{
    int sdfError=0;
    uint32_t nbSchedule=0;
    int32_t debugCounter=1;

    /*
    Create FIFOs objects
    */
    FIFO<float32_t,FIFOSIZE0,0> fifo0(buf0);
    FIFO<float32_t,FIFOSIZE1,1> fifo1(buf1);

    /* 
    Create node objects
    */
    ProcessingNode<float32_t,7,float32_t,5> filter(fifo0,fifo1,4,"Test",someVariable);
    Sink<float32_t,5> sink(fifo1);
    Source<float32_t,5> source(fifo0);

    /* Run several schedule iterations */
    while((sdfError==0) && (debugCounter > 0))
    {
       /* Run a schedule iteration */
       sdfError = source.run();
       CHECKERROR;
       sdfError = source.run();
       CHECKERROR;
       sdfError = filter.run();
       CHECKERROR;
       sdfError = sink.run();
       CHECKERROR;
       sdfError = source.run();
       CHECKERROR;
       sdfError = filter.run();
       CHECKERROR;
       sdfError = sink.run();
       CHECKERROR;
       sdfError = source.run();
       CHECKERROR;
       sdfError = source.run();
       CHECKERROR;
       sdfError = filter.run();
       CHECKERROR;
       sdfError = sink.run();
       CHECKERROR;
       sdfError = source.run();
       CHECKERROR;
       sdfError = filter.run();
       CHECKERROR;
       sdfError = sink.run();
       CHECKERROR;
       sdfError = source.run();
       CHECKERROR;
       sdfError = filter.run();
       CHECKERROR;
       sdfError = sink.run();
       CHECKERROR;

       debugCounter--;
       nbSchedule++;
    }
    *error=sdfError;
    return(nbSchedule);
}
