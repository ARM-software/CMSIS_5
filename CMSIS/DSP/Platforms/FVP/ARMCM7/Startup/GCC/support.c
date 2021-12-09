
#ifdef   __cplusplus
extern "C"
{
#endif
    
char * _sbrk(int incr);

void __malloc_lock() ;
void __malloc_unlock();

char __HeapBase, __HeapLimit;  // make sure to define these symbols in linker command file
#ifdef   __cplusplus
}
#endif

static int totalBytesProvidedBySBRK = 0;
/*
//! sbrk/_sbrk version supporting reentrant newlib (depends upon above symbols defined by linker control file).
char * sbrk(int incr) {
    static char *currentHeapEnd = &__HeapBase;
    char *previousHeapEnd = currentHeapEnd;
    if (currentHeapEnd + incr > &__HeapLimit) {
        return (char *)-1; // the malloc-family routine that called sbrk will return 0
    }
    currentHeapEnd += incr;
    
    totalBytesProvidedBySBRK += incr;
    
    return (char *) previousHeapEnd;
}
//! Synonym for sbrk.
char * _sbrk(int incr) { return sbrk(incr); };
*/

void __malloc_lock()     {       };
void __malloc_unlock()   {  };