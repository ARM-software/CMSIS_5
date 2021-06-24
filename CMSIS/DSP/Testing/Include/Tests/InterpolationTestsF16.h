#include "Test.h"
#include "Pattern.h"

#include "dsp/interpolation_functions_f16.h"

class InterpolationTestsF16:public Client::Suite
    {
        public:
            InterpolationTestsF16(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "InterpolationTestsF16_decl.h"
            
            Client::Pattern<float16_t> input;
            Client::Pattern<float16_t> y;
            Client::Pattern<int16_t> config;
            Client::LocalPattern<float16_t> output;
            // Reference patterns are not loaded when we are in dump mode
            Client::RefPattern<float16_t> ref;

            arm_linear_interp_instance_f16 S;
            arm_bilinear_interp_instance_f16 SBI;


            Client::Pattern<float16_t> inputX;
            Client::Pattern<float16_t> inputY;
            Client::Pattern<float16_t> outputX;

            Client::LocalPattern<float16_t> buffer;
            Client::LocalPattern<float16_t> splineCoefs;

    };
