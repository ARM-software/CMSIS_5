#include "Test.h"
#include "Pattern.h"
class Softmax:public Client::Suite
    {
        public:
            Softmax(Testing::testID_t id);
            virtual void setUp(Testing::testID_t,std::vector<Testing::param_t>& paramsArgs,Client::PatternMgr *mgr);
            virtual void tearDown(Testing::testID_t,Client::PatternMgr *mgr);
        private:
            #include "Softmax_decl.h"
            
            Client::Pattern<int16_t> dims;
            Client::Pattern<q7_t> input;

            Client::RefPattern<int16_t> ref;
            Client::RefPattern<q7_t> samples;

            Client::LocalPattern<int16_t> output;
            Client::LocalPattern<q7_t> temp;

            int nbSamples;
            int vecDim;
           

    };
