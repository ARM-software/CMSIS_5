#include <stdio.h>
#include "arm_math.h"
extern int testmain(const char *);

extern "C" const char *patternData;

int main()
{
    return(testmain(patternData));
}
