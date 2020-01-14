#include "Test.h"
#include "Pattern.h"
class TransformCQ15:public Client::Suite
    {
        public:
            TransformCQ15(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "TransformCQ15_decl.h"
            
            Client::Pattern<q15_t> input;
            Client::LocalPattern<q15_t> outputfft;
            Client::LocalPattern<q15_t> outputifft;

            Client::RefPattern<q15_t> ref;

            arm_cfft_instance_q15 instCfftQ15;

            int ifft;

            /*  

            ifft pattern is using the output of the fft and the input of the fft.
            Since output of the fft is scaled, the input is not recovered without an additional scaling.


            */
            int scaling;

            arm_status status;
            
    };
