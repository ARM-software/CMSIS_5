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

/* List of nodes */
static NodeBase *nodeArray[7]={0};

/*

Description of the scheduling. It is a list of nodes to call.
The values are indexes in the previous array.

*/
static unsigned int schedule[151]=
{ 
6,0,1,6,0,1,6,0,1,6,0,1,6,0,1,6,0,1,6,0,1,6,0,1,6,0,1,6,0,1,6,0,1,6,0,1,6,0,1,6,
0,1,6,0,1,6,0,1,6,0,1,6,0,1,6,0,1,6,0,1,6,0,1,6,0,1,6,0,1,6,0,1,6,0,1,6,0,1,6,0,
1,6,0,1,6,0,1,6,0,1,6,0,1,6,0,1,6,0,1,6,0,1,6,0,1,6,0,1,6,0,1,6,0,1,6,0,1,6,0,1,
6,0,1,6,0,1,6,0,1,6,0,1,6,0,1,6,0,1,6,0,1,6,0,1,6,0,1,2,3,4,5,
};

/***********

FIFO buffers

************/
#define FIFOSIZE0 160
#define FIFOSIZE1 400
#define FIFOSIZE2 49
#define FIFOSIZE3 98
#define FIFOSIZE4 98
#define FIFOSIZE5 1

#define BUFFERSIZE0 160
q15_t buf0[BUFFERSIZE0]={0};

#define BUFFERSIZE1 400
q15_t buf1[BUFFERSIZE1]={0};

#define BUFFERSIZE2 49
q15_t buf2[BUFFERSIZE2]={0};

#define BUFFERSIZE3 98
q15_t buf3[BUFFERSIZE3]={0};

#define BUFFERSIZE4 98
q15_t buf4[BUFFERSIZE4]={0};

#define BUFFERSIZE5 1
q15_t buf5[BUFFERSIZE5]={0};


/**************
 
 Classes created for pure function calls (like some CMSIS-DSP functions)

***************/


uint32_t scheduler(int *error,const q15_t *window,
        const q15_t *coef_q15,
        const int coef_shift,
        const q15_t intercept_q15,
        const int intercept_shift)
{
    int sdfError=0;
    uint32_t nbSchedule=0;

    /*
    Create FIFOs objects
    */
    FIFO<q15_t,FIFOSIZE0,1> fifo0(buf0);
    FIFO<q15_t,FIFOSIZE1,1> fifo1(buf1);
    FIFO<q15_t,FIFOSIZE2,0> fifo2(buf2);
    FIFO<q15_t,FIFOSIZE3,1> fifo3(buf3);
    FIFO<q15_t,FIFOSIZE4,1> fifo4(buf4);
    FIFO<q15_t,FIFOSIZE5,1> fifo5(buf5);

    /* 
    Create node objects 
    */

    SlidingBuffer<q15_t,400,240> audioWin(fifo0,fifo1);
    nodeArray[0]=(NodeBase*)&audioWin;

    Feature<q15_t,400,q15_t,1> feature(fifo1,fifo2,window);
    nodeArray[1]=(NodeBase*)&feature;

    SlidingBuffer<q15_t,98,49> featureWin(fifo2,fifo3);
    nodeArray[2]=(NodeBase*)&featureWin;

    FIR<q15_t,98,q15_t,98> fir(fifo3,fifo4);
    nodeArray[3]=(NodeBase*)&fir;

    KWS<q15_t,98,q15_t,1> kws(fifo4,fifo5,coef_q15,coef_shift,intercept_q15,intercept_shift);
    nodeArray[4]=(NodeBase*)&kws;

    Sink<q15_t,1> sink(fifo5);
    nodeArray[5]=(NodeBase*)&sink;

    Source<q15_t,160> src(fifo0);
    nodeArray[6]=(NodeBase*)&src;

    /* Run several schedule iterations */
    while(sdfError==0)
    {
        /* Run a schedule iteration */
        for(unsigned long id=0 ; id < 151; id++)
        {
            unsigned int nodeId = schedule[id];
            sdfError = nodeArray[nodeId]->run();
            CHECKERROR;
        }
       nbSchedule++;
    }

    *error=sdfError;
    return(nbSchedule);
}
