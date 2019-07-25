#include "Test.h"

#include "SupportTestsF32.h"
    SupportTestsF32::SupportTestsF32(Testing::testID_t id):Client::Suite(id)
    {
        this->addTest(1,(Client::test)&SupportTestsF32::test_barycenter_f32);
this->addTest(2,(Client::test)&SupportTestsF32::test_weighted_sum_f32);

    }
