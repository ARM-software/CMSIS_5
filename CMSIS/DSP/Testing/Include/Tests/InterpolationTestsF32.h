#include "Test.h"
#include "Pattern.h"
class InterpolationTestsF32:public Client::Suite
    {
        public:
            InterpolationTestsF32(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "InterpolationTestsF32_decl.h"
            
            Client::Pattern<float32_t> input;
            Client::Pattern<float32_t> y;
            Client::Pattern<int16_t> config;
            Client::LocalPattern<float32_t> output;
            // Reference patterns are not loaded when we are in dump mode
            Client::RefPattern<float32_t> ref;

            arm_linear_interp_instance_f32 S;
            arm_bilinear_interp_instance_f32 SBI;

    };
