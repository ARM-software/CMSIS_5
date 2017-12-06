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
#ifndef MBED_TIMEREVENT_H
#define MBED_TIMEREVENT_H

#include "hal/ticker_api.h"
#include "hal/us_ticker_api.h"

namespace mbed {
/** \addtogroup drivers */

/** Base abstraction for timer interrupts
 *
 * @note Synchronization level: Interrupt safe
 * @ingroup drivers
 */
class TimerEvent {
public:
    TimerEvent();
    TimerEvent(const ticker_data_t *data);

    /** The handler registered with the underlying timer interrupt
     *
     *  @param id       Timer Event ID
     */
    static void irq(uint32_t id);

    /** Destruction removes it...
     */
    virtual ~TimerEvent();

protected:
    // The handler called to service the timer event of the derived class
    virtual void handler() = 0;

    // insert relative timestamp in to linked list
    void insert(timestamp_t timestamp);

    // insert absolute timestamp into linked list
    void insert_absolute(us_timestamp_t timestamp);

    // remove from linked list, if in it
    void remove();

    ticker_event_t event;

    const ticker_data_t *_ticker_data;
};

} // namespace mbed

#endif
