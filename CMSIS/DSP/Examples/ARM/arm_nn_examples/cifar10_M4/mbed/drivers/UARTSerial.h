/* mbed Microcontroller Library
 * Copyright (c) 2006-2017 ARM Limited
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MBED_UARTSERIAL_H
#define MBED_UARTSERIAL_H

#include "platform/platform.h"

#if DEVICE_SERIAL

#include "FileHandle.h"
#include "SerialBase.h"
#include "InterruptIn.h"
#include "PlatformMutex.h"
#include "serial_api.h"
#include "CircularBuffer.h"

#ifndef MBED_CONF_DRIVERS_UART_SERIAL_RXBUF_SIZE
#define MBED_CONF_DRIVERS_UART_SERIAL_RXBUF_SIZE  256
#endif

#ifndef MBED_CONF_DRIVERS_UART_SERIAL_TXBUF_SIZE
#define MBED_CONF_DRIVERS_UART_SERIAL_TXBUF_SIZE  256
#endif

namespace mbed {

class UARTSerial : private SerialBase, public FileHandle {

public:

    /** Create a UARTSerial port, connected to the specified transmit and receive pins, with a particular baud rate.
     *  @param tx Transmit pin
     *  @param rx Receive pin
     *  @param baud The baud rate of the serial port (optional, defaults to MBED_CONF_PLATFORM_DEFAULT_SERIAL_BAUD_RATE)
     */
    UARTSerial(PinName tx, PinName rx, int baud = MBED_CONF_PLATFORM_DEFAULT_SERIAL_BAUD_RATE);
    virtual ~UARTSerial();

    /** Equivalent to POSIX poll(). Derived from FileHandle.
     *  Provides a mechanism to multiplex input/output over a set of file handles.
     */
    virtual short poll(short events) const;

    /** Write the contents of a buffer to a file
     *
     *  @param buffer   The buffer to write from
     *  @param size     The number of bytes to write
     *  @return         The number of bytes written, negative error on failure
     */
    virtual ssize_t write(const void* buffer, size_t length);

    /** Read the contents of a file into a buffer
     *
     *  Follows POSIX semantics:
     *
     *  * if no data is available, and non-blocking set return -EAGAIN
     *  * if no data is available, and blocking set, wait until data is available
     *  * If any data is available, call returns immediately
     *
     *  @param buffer   The buffer to read in to
     *  @param size     The number of bytes to read
     *  @return         The number of bytes read, 0 at end of file, negative error on failure
     */
    virtual ssize_t read(void* buffer, size_t length);

    /** Acquire mutex */
    virtual void lock(void);

    /** Release mutex */
    virtual void unlock(void);

    /** Close a file
     *
     *  @return         0 on success, negative error code on failure
     */
    virtual int close();

    /** Check if the file in an interactive terminal device
     *
     *  @return         True if the file is a terminal
     *  @return         False if the file is not a terminal
     *  @return         Negative error code on failure
     */
    virtual int isatty();

    /** Move the file position to a given offset from from a given location
     *
     * Not valid for a device type FileHandle like UARTSerial.
     * In case of UARTSerial, returns ESPIPE
     *
     *  @param offset   The offset from whence to move to
     *  @param whence   The start of where to seek
     *      SEEK_SET to start from beginning of file,
     *      SEEK_CUR to start from current position in file,
     *      SEEK_END to start from end of file
     *  @return         The new offset of the file, negative error code on failure
     */
    virtual off_t seek(off_t offset, int whence);

    /** Flush any buffers associated with the file
     *
     *  @return         0 on success, negative error code on failure
     */
    virtual int sync();

    /** Set blocking or non-blocking mode
     *  The default is blocking.
     *
     *  @param blocking true for blocking mode, false for non-blocking mode.
     */
    virtual int set_blocking(bool blocking)
    {
        _blocking = blocking;
        return 0;
    }

    /** Register a callback on state change of the file.
     *
     *  The specified callback will be called on state changes such as when
     *  the file can be written to or read from.
     *
     *  The callback may be called in an interrupt context and should not
     *  perform expensive operations.
     *
     *  Note! This is not intended as an attach-like asynchronous api, but rather
     *  as a building block for constructing  such functionality.
     *
     *  The exact timing of when the registered function
     *  is called is not guaranteed and susceptible to change. It should be used
     *  as a cue to make read/write/poll calls to find the current state.
     *
     *  @param func     Function to call on state change
     */
    virtual void sigio(Callback<void()> func);

    /** Setup interrupt handler for DCD line
     *
     *  If DCD line is connected, an IRQ handler will be setup.
     *  Does nothing if DCD is NC, i.e., not connected.
     *
     *  @param dcd_pin         Pin-name for DCD
     *  @param active_high     a boolean set to true if DCD polarity is active low
     */
    void set_data_carrier_detect(PinName dcd_pin, bool active_high = false);

private:

    /** Software serial buffers
     *  By default buffer size is 256 for TX and 256 for RX. Configurable through mbed_app.json
     */
    CircularBuffer<char, MBED_CONF_DRIVERS_UART_SERIAL_RXBUF_SIZE> _rxbuf;
    CircularBuffer<char, MBED_CONF_DRIVERS_UART_SERIAL_TXBUF_SIZE> _txbuf;

    PlatformMutex _mutex;

    Callback<void()> _sigio_cb;

    bool _blocking;
    bool _tx_irq_enabled;
    InterruptIn *_dcd_irq;

    /** Device Hanged up
     *  Determines if the device hanged up on us.
     *
     *  @return True, if hanged up
     */
    bool hup() const;

    /** ISRs for serial
     *  Routines to handle interrupts on serial pins.
     *  Copies data into Circular Buffer.
     *  Reports the state change to File handle.
     */
    void tx_irq(void);
    void rx_irq(void);

    void wake(void);

    void dcd_irq(void);
};
} //namespace mbed

#endif //DEVICE_SERIAL
#endif //MBED_UARTSERIAL_H
