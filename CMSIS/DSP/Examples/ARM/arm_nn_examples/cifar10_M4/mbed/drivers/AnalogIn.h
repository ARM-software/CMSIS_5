/* mbed Microcontroller Library
 * Copyright (c) 2006-2013 ARM Limited
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
#ifndef MBED_ANALOGIN_H
#define MBED_ANALOGIN_H

#include "platform/platform.h"

#if defined (DEVICE_ANALOGIN) || defined(DOXYGEN_ONLY)

#include "hal/analogin_api.h"
#include "platform/SingletonPtr.h"
#include "platform/PlatformMutex.h"

namespace mbed {
/** \addtogroup drivers */

/** An analog input, used for reading the voltage on a pin
 *
 * @note Synchronization level: Thread safe
 *
 * Example:
 * @code
 * // Print messages when the AnalogIn is greater than 50%
 *
 * #include "mbed.h"
 *
 * AnalogIn temperature(p20);
 *
 * int main() {
 *     while(1) {
 *         if(temperature > 0.5) {
 *             printf("Too hot! (%f)", temperature.read());
 *         }
 *     }
 * }
 * @endcode
 * @ingroup drivers
 */
class AnalogIn {

public:

    /** Create an AnalogIn, connected to the specified pin
     *
     * @param pin AnalogIn pin to connect to
     */
    AnalogIn(PinName pin) {
        lock();
        analogin_init(&_adc, pin);
        unlock();
    }

    /** Read the input voltage, represented as a float in the range [0.0, 1.0]
     *
     * @returns A floating-point value representing the current input voltage, measured as a percentage
     */
    float read() {
        lock();
        float ret = analogin_read(&_adc);
        unlock();
        return ret;
    }

    /** Read the input voltage, represented as an unsigned short in the range [0x0, 0xFFFF]
     *
     * @returns
     *   16-bit unsigned short representing the current input voltage, normalised to a 16-bit value
     */
    unsigned short read_u16() {
        lock();
        unsigned short ret = analogin_read_u16(&_adc);
        unlock();
        return ret;
    }

    /** An operator shorthand for read()
     *
     * The float() operator can be used as a shorthand for read() to simplify common code sequences
     *
     * Example:
     * @code
     * float x = volume.read();
     * float x = volume;
     *
     * if(volume.read() > 0.25) { ... }
     * if(volume > 0.25) { ... }
     * @endcode
     */
    operator float() {
        // Underlying call is thread safe
        return read();
    }

    virtual ~AnalogIn() {
        // Do nothing
    }

protected:

    virtual void lock() {
        _mutex->lock();
    }

    virtual void unlock() {
        _mutex->unlock();
    }

    analogin_t _adc;
    static SingletonPtr<PlatformMutex> _mutex;
};

} // namespace mbed

#endif

#endif

