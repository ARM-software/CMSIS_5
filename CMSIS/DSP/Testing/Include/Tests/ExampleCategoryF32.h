#include "Test.h"
#include "Pattern.h"
 /* 

  Code below is generic. Only the name of the class must be customized and
  correspond to what is used for the test suite.

*/

#include "dsp/basic_math_functions.h"

class ExampleCategoryF32:public Client::Suite
    {
        public:
            ExampleCategoryF32(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "ExampleCategoryF32_decl.h"
            
            /*

            Code below must be customized and depends on the tests.
            There are always some input patterns, some reference patterns
            and some output or temporary storage.

            If any dynamical memory is required it should be done only by using
            LocalPattern -> A temporary area where to save output of a test.


            */

            /* Input patterns */
            Client::Pattern<float32_t> input1;
            Client::Pattern<float32_t> input2;


            /* Output and temporary storage */
            Client::LocalPattern<float32_t> output;

            /* Reference patterns */
            Client::RefPattern<float32_t> ref;
    };
