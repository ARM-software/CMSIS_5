#include "Test.h"

#include "FullyConnected.h"
    FullyConnected::FullyConnected(Testing::testID_t id):Client::Suite(id)
    {
        this->addTest(1,(Client::test)&FullyConnected::test_fully_connected_tflite_s8);
this->addTest(2,(Client::test)&FullyConnected::test_fully_connected_tflite_s8);
this->addTest(3,(Client::test)&FullyConnected::test_fully_connected_tflite_s8);
this->addTest(4,(Client::test)&FullyConnected::test_fully_connected_tflite_s8);
this->addTest(5,(Client::test)&FullyConnected::test_fully_connected_tflite_s8);
this->addTest(6,(Client::test)&FullyConnected::test_fully_connected_tflite_s8);
this->addTest(7,(Client::test)&FullyConnected::test_fully_connected_tflite_s8);
this->addTest(8,(Client::test)&FullyConnected::test_fully_connected_tflite_s8);
this->addTest(9,(Client::test)&FullyConnected::test_fully_connected_tflite_s8);
this->addTest(10,(Client::test)&FullyConnected::test_fully_connected_tflite_s8);
this->addTest(11,(Client::test)&FullyConnected::test_fully_connected_tflite_s8);
this->addTest(12,(Client::test)&FullyConnected::test_fully_connected_tflite_s8);
this->addTest(13,(Client::test)&FullyConnected::test_fully_connected_tflite_s8);
this->addTest(14,(Client::test)&FullyConnected::test_fully_connected_tflite_s8);
this->addTest(15,(Client::test)&FullyConnected::test_fully_connected_tflite_s8);

    }
