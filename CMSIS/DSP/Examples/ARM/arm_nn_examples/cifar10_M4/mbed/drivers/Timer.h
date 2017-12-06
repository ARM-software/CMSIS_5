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
#ifndef MBED_TIMER_H
#define MBED_TIMER_H

#include "platform/platform.h"
#include "hal/ticker_api.h"

namespace mbed {
/** \addtogroup drivers */

/** A general purpose timer
 *
 * @note Synchronization level: Interrupt safe
 *
 * Example:
 * @code
 * // Count the time to toggle a LED
 *
 * #include "mbed.h"
 *
 * Timer timer;
 * DigitalOut led(LED1);
 * int begin, end;
 *
 * int main() {
 *     timer.start();
 *     begin = timer.read_us();
 *     led = !led;
 *     end = timer.read_us();
 *     printf("Toggle the led takes %d us", end - begin);
 * }
 * @endcode
 * @ingroup drivers
 */
class Timer {

public:
    Timer();
    Timer(const ticker_data_t *data);

    /** Start the timer
     */
    void start();

    /** Stop the timer
     */
    void stop();

    /** Reset the timer to 0.
     *
     * If it was already counting, it will continue
     */
    void reset();

    /** Get the time passed in seconds
     *
     *  @returns    Time passed in seconds
     */
    float read();

    /** Get the time passed in milli-seconds
     *
     *  @returns    Time passed in milli seconds
     */
    int read_ms();

    /** Get the time passed in micro-seconds
     *
     *  @returns    Time passed in micro seconds
     */
    int read_us();

    /** An operator shorthand for read()
     */
    operator float();

    /** Get in a high resolution type the time passed in micro-seconds.
     */
    us_timestamp_t read_high_resolution_us();

protected:
    us_timestamp_t slicetime();
    int _running;            // whether the timer is running
    us_timestamp_t _start;   // the start time of the latest slice
    us_timestamp_t _time;    // any accumulated time from previous slices
    const ticker_data_t *_ticker_data;
};

} // namespace mbed

#endif
