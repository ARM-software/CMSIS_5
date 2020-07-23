#include "Test.h"
#include "Pattern.h"

#include "dsp/filtering_functions.h"

class DECIMQ15:public Client::Suite
    {
        public:
            DECIMQ15(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& params,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "DECIMQ15_decl.h"
            
            Client::Pattern<q15_t> input;
            Client::Pattern<q15_t> coefs;
            Client::Pattern<uint32_t> config;

            Client::LocalPattern<q15_t> output;
            Client::LocalPattern<q15_t> state;
            // Reference patterns are not loaded when we are in dump mode
            Client::RefPattern<q15_t> ref;


            arm_fir_decimate_instance_q15 S;
            arm_fir_interpolate_instance_q15 SI;

            int q;
            int numTaps;
            int blocksize;
            int refsize;

            arm_status status;
    };
