#include <cstdio>
#include <cstdint>
#include "scheduler.h"


int main(int argc, char const *argv[])
{
    int error;
    printf("Start\n");
    uint32_t nb = scheduler(&error);
    printf("Nb = %d\n",nb);
    return 0;
}