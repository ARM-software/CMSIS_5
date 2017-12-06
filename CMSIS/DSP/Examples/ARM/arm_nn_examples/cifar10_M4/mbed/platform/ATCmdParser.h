/* Copyright (c) 2017 ARM Limited
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
 *
 * @section DESCRIPTION
 *
 * Parser for the AT command syntax
 *
 */
#ifndef MBED_ATCMDPARSER_H
#define MBED_ATCMDPARSER_H

#include "mbed.h"
#include <cstdarg>
#include "Callback.h"

/**
 * Parser class for parsing AT commands
 *
 * Here are some examples:
 * @code
 * ATCmdParser at = ATCmdParser(serial, "\r\n");
 * int value;
 * char buffer[100];
 *
 * at.send("AT") && at.recv("OK");
 * at.send("AT+CWMODE=%d", 3) && at.recv("OK");
 * at.send("AT+CWMODE?") && at.recv("+CWMODE:%d\r\nOK", &value);
 * at.recv("+IPD,%d:", &value);
 * at.read(buffer, value);
 * at.recv("OK");
 * @endcode
 */

namespace mbed {

class ATCmdParser
{
private:
    // File handle
    // Not owned by ATCmdParser
    FileHandle *_fh;

    int _buffer_size;
    char *_buffer;
    int _timeout;

    // Parsing information
    const char *_output_delimiter;
    int _output_delim_size;
    char _in_prev;
    bool _dbg_on;
    bool _aborted;

    struct oob {
        unsigned len;
        const char *prefix;
        mbed::Callback<void()> cb;
        oob *next;
    };
    oob *_oobs;

    // Prohibiting use of of copy constructor
    ATCmdParser(const ATCmdParser &);
    // Prohibiting copy assignment Operator
    ATCmdParser &operator=(const ATCmdParser &);

public:

    /**
     * Constructor
     *
     * @param fh A FileHandle to a digital interface to use for AT commands
     * @param output_delimiter end of command line termination
     * @param buffer_size size of internal buffer for transaction
     * @param timeout timeout of the connection
     * @param debug turns on/off debug output for AT commands
     */
    ATCmdParser(FileHandle *fh, const char *output_delimiter = "\r",
             int buffer_size = 256, int timeout = 8000, bool debug = false)
            : _fh(fh), _buffer_size(buffer_size), _in_prev(0), _oobs(NULL)
    {
        _buffer = new char[buffer_size];
        set_timeout(timeout);
        set_delimiter(output_delimiter);
        debug_on(debug);
    }

    /**
     * Destructor
     */
    ~ATCmdParser()
    {
        while (_oobs) {
            struct oob *oob = _oobs;
            _oobs = oob->next;
            delete oob;
        }
        delete[] _buffer;
    }

    /**
     * Allows timeout to be changed between commands
     *
     * @param timeout timeout of the connection
     */
    void set_timeout(int timeout)
    {
        _timeout = timeout;
    }

    /**
     * For backwards compatibility.
     *
     * Please use set_timeout(int) API only from now on.
     * Allows timeout to be changed between commands
     *
     * @param timeout timeout of the connection
     */
    MBED_DEPRECATED_SINCE("mbed-os-5.5.0", "Replaced with set_timeout for consistency")
    void setTimeout(int timeout)
    {
        set_timeout(timeout);
    }

    /**
     * Sets string of characters to use as line delimiters
     *
     * @param output_delimiter string of characters to use as line delimiters
     */
    void set_delimiter(const char *output_delimiter)
    {
        _output_delimiter = output_delimiter;
        _output_delim_size = strlen(output_delimiter);
    }

    /**
     * For backwards compatibility.
     *
     * Please use set_delimiter(const char *) API only from now on.
     * Sets string of characters to use as line delimiters
     *
     * @param output_delimiter string of characters to use as line delimiters
     */
    MBED_DEPRECATED_SINCE("mbed-os-5.5.0", "Replaced with set_delimiter for consistency")
    void setDelimiter(const char *output_delimiter)
    {
        set_delimiter(output_delimiter);
    }

    /**
     * Allows traces from modem to be turned on or off
     *
     * @param on set as 1 to turn on traces and vice versa.
     */
    void debug_on(uint8_t on)
    {
        _dbg_on = (on) ? 1 : 0;
    }

    /**
     * For backwards compatibility.
     *
     * Allows traces from modem to be turned on or off
     *
     * @param on set as 1 to turn on traces and vice versa.
     */
    MBED_DEPRECATED_SINCE("mbed-os-5.5.0", "Replaced with debug_on for consistency")
    void debugOn(uint8_t on)
    {
        debug_on(on);
    }

    /**
     * Sends an AT command
     *
     * Sends a formatted command using printf style formatting
     * @see printf
     *
     * @param command printf-like format string of command to send which
     *                is appended with a newline
     * @param ... all printf-like arguments to insert into command
     * @return true only if command is successfully sent
     */
    bool send(const char *command, ...) MBED_PRINTF_METHOD(1,2);

    bool vsend(const char *command, va_list args);

    /**
     * Receive an AT response
     *
     * Receives a formatted response using scanf style formatting
     * @see scanf
     *
     * Responses are parsed line at a time.
     * Any received data that does not match the response is ignored until
     * a timeout occurs.
     *
     * @param response scanf-like format string of response to expect
     * @param ... all scanf-like arguments to extract from response
     * @return true only if response is successfully matched
     */
    bool recv(const char *response, ...) MBED_SCANF_METHOD(1,2);

    bool vrecv(const char *response, va_list args);

    /**
     * Write a single byte to the underlying stream
     *
     * @param c The byte to write
     * @return The byte that was written or -1 during a timeout
     */
    int putc(char c);

    /**
     * Get a single byte from the underlying stream
     *
     * @return The byte that was read or -1 during a timeout
     */
    int getc();

    /**
     * Write an array of bytes to the underlying stream
     *
     * @param data the array of bytes to write
     * @param size number of bytes to write
     * @return number of bytes written or -1 on failure
     */
    int write(const char *data, int size);

    /**
     * Read an array of bytes from the underlying stream
     *
     * @param data the destination for the read bytes
     * @param size number of bytes to read
     * @return number of bytes read or -1 on failure
     */
    int read(char *data, int size);

    /**
     * Direct printf to underlying stream
     * @see printf
     *
     * @param format format string to pass to printf
     * @param ... arguments to printf
     * @return number of bytes written or -1 on failure
     */
    int printf(const char *format, ...) MBED_PRINTF_METHOD(1,2);

    int vprintf(const char *format, va_list args);

    /**
     * Direct scanf on underlying stream
     * @see scanf
     *
     * @param format format string to pass to scanf
     * @param ... arguments to scanf
     * @return number of bytes read or -1 on failure
     */
    int scanf(const char *format, ...) MBED_SCANF_METHOD(1,2);

    int vscanf(const char *format, va_list args);

    /**
     * Attach a callback for out-of-band data
     *
     * @param prefix string on when to initiate callback
     * @param func callback to call when string is read
     * @note out-of-band data is only processed during a scanf call
     */
    void oob(const char *prefix, mbed::Callback<void()> func);

    /**
     * Flushes the underlying stream
     */
    void flush();

    /**
     * Abort current recv
     *
     * Can be called from oob handler to interrupt the current
     * recv operation.
     */
    void abort();
};
} //namespace mbed

#endif //MBED_ATCMDPARSER_H
