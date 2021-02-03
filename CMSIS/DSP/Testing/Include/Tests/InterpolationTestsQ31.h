#include "Test.h"
#include "Pattern.h"

#include "dsp/interpolation_functions.h"

class InterpolationTestsQ31:public Client::Suite
    {
        public:
            InterpolationTestsQ31(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "InterpolationTestsQ31_decl.h"
            
            Client::Pattern<q31_t> input;
            Client::Pattern<q31_t> y;
            Client::Pattern<int16_t> config;
            Client::LocalPattern<q31_t> output;
            // Reference patterns are not loaded when we are in dump mode
            Client::RefPattern<q31_t> ref;

            arm_bilinear_interp_instance_q31 SBI;
    };
