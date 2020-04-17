# Sine Detection

This example is showing how to detect a sine of 1 kHz with Support Vector Machine.

It is not the simplest nor best way to detect a sine. It is just an example of the use of a SVM classifier which may be extended to signal a bit more complex than a sine by using the same method.

The performance of the app is highly dependent on the training data which was used.
On my tests, it is working well. But if your environment is quite different from mine (more noisy ...), then the training data I have used may not give good results.

The difficulty with machine learning is to find the right training set which will give a good generalization and a good behavior of unseen data.

For detection of more complex signals, smart features may be required. In this example, we work on the raw data. There is a bit of pre-processing:

1 - The data is rescale because SVM are not scale indepdendent ;
2 - Energy is used to rescale. We don't use the amplitude to avoid being impacted too much by sample outliers ;
3 - An Hanning window is applied. This step may not be needed but we have not experimented without it.

The training is done with this pre-processing applied to the signals.

If you want to know how to use a SVM with CMSIS-DSP, you can refer to this tutorial:
https://developer.arm.com/solutions/machine-learning-on-arm/developer-material/how-to-guides/implement-classical-ml-with-arm-cmsis-dsp-libraries

and the DSP/Examples/ARM/arm_svm_example folder.

## Sine Detection App

It is an Arduino app. It was tested on an Arduino Nano 33 BLE Sense.
It is using the PDM driver coming with this board.

If you want to use BLE, you'll need to install the ArduinoBLE and define BLEOUTPUT in the codee.
Then you'll need to install a BLE scanner on your phone.

If you don't enable BLE, you can see the detection status in the serial console.

You can also use the DetectionDisplay app.

## DetectionDisplay 

This app is using https://processing.org/

You will have to change the serial port name in the app before building it.

This app is connecting to the serial port and listening to message from the Arduino.
When a sine is detected, it is displaying a green word with some fading.




