arduino-cli board list

arduino-cli config init

arduino-cli lib install Arduino_CMSIS-DSP

arduino-cli compile -b arduino:mbed_nano:nano33ble -v

# Bootloader COM port
arduino-cli upload -b arduino:mbed_nano:nano33ble -p COM5

pip install pyserial

python getData.py