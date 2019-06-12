#ifndef _TIMING_H_
#define _TIMING_H_

#include "Test.h"
#include "arm_math.h"
void initCycleMeasurement();
void cycleMeasurementStart();
void cycleMeasurementStop();

Testing::cycles_t getCycles();

#endif