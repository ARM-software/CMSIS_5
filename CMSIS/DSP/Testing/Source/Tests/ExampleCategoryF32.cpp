#include "ExampleCategoryF32.h"
#include <stdio.h>
#include "Error.h"

/*

Tests to write and test criteria depend on the algorithm.
When SNR is meaningful, SNR threshold depends on the type.
CMSIS-DSP tests are using similar SNR values for different type (f32, q31, q15, q7)

*/
#define SNR_THRESHOLD 120

/*

With thie threshold, the test will fail

#define REL_ERROR (2.0e-6) 

*/

#define REL_ERROR (5.0e-6)

    void ExampleCategoryF32::test_op_f32()
    {
        /* Get a pointer to the input data.
           For benchmark, getting pointers should be done in the setUp function
           since there is an overhead. Lot of checks are done before returning a pointer.
        */
        const float32_t *inp1=input1.ptr();
        const float32_t *inp2=input2.ptr();

        /* Get a pointer to the output buffer */
        float32_t *outp=output.ptr();

        /* Run the test */
        arm_add_f32(inp1,inp2,outp,input1.nbSamples());

        /* Check there is no buffer overflow on the output */
        ASSERT_EMPTY_TAIL(output);

        /* Check SNR error */
        ASSERT_SNR(output,ref,(float32_t)SNR_THRESHOLD);

        /* Check relative error */
        ASSERT_REL_ERROR(output,ref,REL_ERROR);

    } 


 
    /*

     setUp function is used to load the patterns and create required buffers

    */
    void ExampleCategoryF32::setUp(Testing::testID_t id,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr)
    {
      
       Testing::nbSamples_t nb=MAX_NB_SAMPLES; 
       (void)params;
       
       /*
         
         All IDs can be found in GeneratedInclude/ExampleCategoryF32_decl.h

       */

       /*
           Different allocations depending on the test.
       */
       switch(id)
       {
          /* In both tests, the same function is tested as defined in desc.txt.
             But different configurations are used.
          */
          case ExampleCategoryF32::TEST_OP_F32_1:
             /* Load  patterns with all samples */
             input1.reload(ExampleCategoryF32::INPUT1_F32_ID,mgr);
             input2.reload(ExampleCategoryF32::INPUT2_F32_ID,mgr);
             ref.reload(ExampleCategoryF32::REF_OUT_F32_ID,mgr);
          break;

          case ExampleCategoryF32::TEST_OP_F32_2:
             nb = 9;
             /* Load  patterns with 9 samples */
             input1.reload(ExampleCategoryF32::INPUT1_F32_ID,mgr,nb);
             input2.reload(ExampleCategoryF32::INPUT2_F32_ID,mgr,nb);
             ref.reload(ExampleCategoryF32::REF_OUT_F32_ID,mgr,nb);
          break;

       }
    
       /* Create output buffer with same size as reference pattern */
       output.create(ref.nbSamples(),ExampleCategoryF32::OUT_F32_ID,mgr);
    }

    void ExampleCategoryF32::tearDown(Testing::testID_t id,Client::PatternMgr *mgr)
    {
       (void)id;
        /* 
           Dump output buffer into a file.

           Location of the file is defined by Folder directives in desc.txt test
           description file and relative to the Output folder.

        */
        output.dump(mgr);
    }
