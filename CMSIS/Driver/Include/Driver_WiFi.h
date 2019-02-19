/*
 * Copyright (c) 2013-2019 Arm Limited. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * $Date:        14. February 2019
 * $Revision:    V1.0 (beta)
 *
 * Project:      WiFi (Wireless Fidelity Interface) Driver definitions
 */

/* History:
 *  Version 1.0 (beta)
 *    Initial beta version
 */

#ifndef DRIVER_WIFI_H_
#define DRIVER_WIFI_H_

#ifdef  __cplusplus
extern "C"
{
#endif

#include "Driver_Common.h"

#define ARM_WIFI_API_VERSION ARM_DRIVER_VERSION_MAJOR_MINOR(1,0)  /* API version */

#define _ARM_Driver_WiFi_(n)      Driver_WiFi##n
#define  ARM_Driver_WiFi_(n) _ARM_Driver_WiFi_(n)


/****** WiFi SetOption/GetOption Function Option Codes *****/
#define ARM_WIFI_SSID                       1U          ///< Station     Get SSID of connected AP;            data = &ssid,     len<= 33, ssid     (char[32+1]), null-terminated string
#define ARM_WIFI_BSSID                      2U          ///< Station     Get BSSID of connected AP;           data = &bssid,    len =  6, bssid    (uint8_t[6])
#define ARM_WIFI_PASS                       3U          ///< Station     Get Password of connected AP;        data = &pass,     len<= 65, pass     (char[64+1]), null-terminated string
#define ARM_WIFI_SECURITY                   4U          ///< Station     Get Security Type of connected AP;   data = &security, len =  4, security (uint32_t): ARM_WIFI_SECURITY_xxx
#define ARM_WIFI_CHANNEL                    5U          ///< Station     Get Channel of connected AP;         data = &ch,       len =  4, ch       (uint32_t)
#define ARM_WIFI_RSSI                       6U          ///< Station     Get RSSI of connected AP;            data = &rssi,     len =  4, rssi     (uint32_t)
#define ARM_WIFI_TX_POWER                   7U          ///< Station Set/Get transmit power;                  data = &dBm,      len =  4, dBm      (uint32_t): 0 .. 20 [dBm]
#define ARM_WIFI_MAC                        8U          ///< Station Set/Get MAC;                             data = &mac,      len =  6, mac      (uint8_t[6])
#define ARM_WIFI_IP                         9U          ///< Station Set/Get IPv4 static/assigned address;    data = &ip,       len =  4, ip       (uint8_t[4])
#define ARM_WIFI_IP_SUBNET_MASK             10U         ///< Station Set/Get IPv4 subnet mask;                data = &msk,      len =  4, msk      (uint8_t[4])
#define ARM_WIFI_IP_GATEWAY                 11U         ///< Station Set/Get IPv4 gateway address;            data = &ip,       len =  4, ip       (uint8_t[4])
#define ARM_WIFI_IP_DNS1                    12U         ///< Station Set/Get IPv4 primary   DNS address;      data = &ip,       len =  4, ip       (uint8_t[4])
#define ARM_WIFI_IP_DNS2                    13U         ///< Station Set/Get IPv4 secondary DNS address;      data = &ip,       len =  4, ip       (uint8_t[4])
#define ARM_WIFI_IP_DHCP                    14U         ///< Station Set/Get IPv4 DHCP client enable/disable; data = &en,       len =  4, en       (uint32_t): 0 = disable, non-zero = enable (default)
#define ARM_WIFI_IP6_GLOBAL                 15U         ///< Station Set/Get IPv6 global address;             data = &ip6,      len = 16, ip6      (uint8_t[16])
#define ARM_WIFI_IP6_LINK_LOCAL             16U         ///< Station Set/Get IPv6 link local address;         data = &ip6,      len = 16, ip6      (uint8_t[16])
#define ARM_WIFI_IP6_SUBNET_PREFIX_LEN      17U         ///< Station Set/Get IPv6 subnet prefix length;       data = &len,      len =  4, len      (uint32_t): 1 .. 127
#define ARM_WIFI_IP6_GATEWAY                18U         ///< Station Set/Get IPv6 gateway address;            data = &ip6,      len = 16, ip6      (uint8_t[16])
#define ARM_WIFI_IP6_DNS1                   19U         ///< Station Set/Get IPv6 primary   DNS address;      data = &ip6,      len = 16, ip6      (uint8_t[16])
#define ARM_WIFI_IP6_DNS2                   20U         ///< Station Set/Get IPv6 secondary DNS address;      data = &ip6,      len = 16, ip6      (uint8_t[16])
#define ARM_WIFI_IP6_DHCP_MODE              21U         ///< Station Set/Get IPv6 DHCPv6 client mode;         data = &mode,     len =  4, mode     (uint32_t): ARM_WIFI_IP6_DHCP_xxx (default Off)
#define ARM_WIFI_AP_SSID_HIDE               22U         ///< AP      Set/Get SSID hide option;                data = &en,       len =  4, en       (uint32_t): 0 = disable (default), non-zero = enable
#define ARM_WIFI_AP_TX_POWER                23U         ///< AP      Set/Get transmit power;                  data = &dBm,      len =  4, dBm      (uint32_t): 0 .. 20 [dBm]
#define ARM_WIFI_AP_MAC                     24U         ///< AP      Set/Get MAC;                             data = &mac,      len =  6, mac      (uint8_t[6])
#define ARM_WIFI_AP_IP                      25U         ///< AP      Set/Get IPv4 static/assigned address;    data = &ip,       len =  4, ip       (uint8_t[4])
#define ARM_WIFI_AP_IP_SUBNET_MASK          26U         ///< AP      Set/Get IPv4 subnet mask;                data = &msk,      len =  4, msk      (uint8_t[4])
#define ARM_WIFI_AP_IP_GATEWAY              27U         ///< AP      Set/Get IPv4 gateway address;            data = &ip,       len =  4, ip       (uint8_t[4])
#define ARM_WIFI_AP_IP_DNS1                 28U         ///< AP      Set/Get IPv4 primary   DNS address;      data = &ip,       len =  4, ip       (uint8_t[4])
#define ARM_WIFI_AP_IP_DNS2                 29U         ///< AP      Set/Get IPv4 secondary DNS address;      data = &ip,       len =  4, ip       (uint8_t[4])
#define ARM_WIFI_AP_IP_DHCP                 30U         ///< AP      Set/Get IPv4 DHCP server enable/disable; data = &en,       len =  4, en       (uint32_t): 0 = disable, non-zero = enable (default)
#define ARM_WIFI_AP_IP_DHCP_POOL_BEGIN      31U         ///< AP      Set/Get IPv4 DHCP pool begin address;    data = &ip,       len =  4, ip       (uint8_t[4])
#define ARM_WIFI_AP_IP_DHCP_POOL_END        32U         ///< AP      Set/Get IPv4 DHCP pool end address;      data = &ip,       len =  4, ip       (uint8_t[4])
#define ARM_WIFI_AP_IP_DHCP_LEASE_TIME      33U         ///< AP      Set/Get IPv4 DHCP lease time;            data = &sec,      len =  4, sec      (uint32_t) [seconds]
#define ARM_WIFI_AP_IP_DHCP_TABLE           34U         ///< AP          Get IPv4 DHCP table;                 data = &mac_ip4[],len = sizeof(mac_ip4[]), mac_ip4 (array of ARM_WIFI_MAC_IP4_t structures)
#define ARM_WIFI_AP_IP6_GLOBAL              35U         ///< AP      Set/Get IPv6 global address;             data = &ip6,      len = 16, ip6      (uint8_t[16])
#define ARM_WIFI_AP_IP6_LINK_LOCAL          36U         ///< AP      Set/Get IPv6 link local address;         data = &ip6,      len = 16, ip6      (uint8_t[16])
#define ARM_WIFI_AP_IP6_SUBNET_PREFIX_LEN   37U         ///< AP      Set/Get IPv6 subnet prefix length;       data = &len,      len =  4, len      (uint32_t): 1 .. 127
#define ARM_WIFI_AP_IP6_GATEWAY             38U         ///< AP      Set/Get IPv6 gateway address;            data = &ip6,      len = 16, ip6      (uint8_t[16])
#define ARM_WIFI_AP_IP6_DNS1                39U         ///< AP      Set/Get IPv6 primary   DNS address;      data = &ip6,      len = 16, ip6      (uint8_t[16])
#define ARM_WIFI_AP_IP6_DNS2                40U         ///< AP      Set/Get IPv6 secondary DNS address;      data = &ip6,      len = 16, ip6      (uint8_t[16])

/****** WiFi Security Type *****/
#define ARM_WIFI_SECURITY_OPEN              0U          ///< Unsecured
#define ARM_WIFI_SECURITY_WEP               1U          ///< Wired Equivalent Privacy (WEP)
#define ARM_WIFI_SECURITY_WPA               2U          ///< WiFi Protected Access (WPA)
#define ARM_WIFI_SECURITY_WPA2              3U          ///< WiFi Protected Access II (WPA2)
#define ARM_WIFI_SECURITY_UNKNOWN           7U          ///< Unknown

/****** WiFi IPv6 Dynamic Host Configuration Protocol (DHCP) Mode *****/
#define ARM_WIFI_IP6_DHCP_OFF               0U          ///< Static Host Configuration
#define ARM_WIFI_IP6_DHCP_STATELESS         1U          ///< Dynamic Host Configuration stateless DHCPv6
#define ARM_WIFI_IP6_DHCP_STATEFULL         2U          ///< Dynamic Host Configuration statefull DHCPv6

/****** WiFi Event *****/
#define ARM_WIFI_EVENT_AP_CONNECT          (1UL << 0)   ///< Access Point: Station has connected;           arg = &mac, mac (uint8_t[6])
#define ARM_WIFI_EVENT_AP_DISCONNECT       (1UL << 1)   ///< Access Point: Station has disconnected;        arg = &mac, mac (uint8_t[6])
#define ARM_WIFI_EVENT_ETH_RX_FRAME        (1UL << 4)   ///< Ethernet Frame Received (in bypass mode only); arg = NULL


/**
\brief WiFi Media Access Control / Internet Protocol (MAC/IP4) Information
*/
typedef struct {
  uint8_t mac[6];                                       ///< Media Access Control Information
  uint8_t ip4[4];                                       ///< Internet Protocol v4 address
} ARM_WIFI_MAC_IP4_t;

/**
\brief WiFi Access Point (AP) Information
*/
typedef struct {
  char    ssid[32+1];                                   ///< Service Set Identifier (SSID) null-terminated string
  uint8_t bssid[6];                                     ///< Basic Service Set Identifier (BSSID)
  uint8_t security;                                     ///< Security type (ARM_WIFI_SECURITY_xxx)
  uint8_t ch;                                           ///< WiFi Channel
  uint8_t rssi;                                         ///< Received Signal Strength Indicator
} ARM_WIFI_AP_INFO_t;


/****** Socket Address Family definitions *****/
#define ARM_SOCKET_AF_INET                  1           ///< IPv4
#define ARM_SOCKET_AF_INET6                 2           ///< IPv6

/****** Socket Type definitions *****/
#define ARM_SOCKET_SOCK_STREAM              1           ///< Stream socket
#define ARM_SOCKET_SOCK_DGRAM               2           ///< Datagram socket

/****** Socket Protocol definitions *****/
#define ARM_SOCKET_IPPROTO_TCP              1           ///< TCP
#define ARM_SOCKET_IPPROTO_UDP              2           ///< UDP

/****** Socket Option definitions *****/
#define ARM_SOCKET_IO_FIONBIO               1           ///< Non-blocking I/O (Set only, default = 0); opt_val = &nbio, opt_len = sizeof(nbio), nbio (integer): 0=blocking, non-blocking otherwise
#define ARM_SOCKET_SO_RCVTIMEO              2           ///< Receive timeout in ms (default = 0); opt_val = &timeout, opt_len = sizeof(timeout)
#define ARM_SOCKET_SO_SNDTIMEO              3           ///< Send timeout in ms (default = 0); opt_val = &timeout, opt_len = sizeof(timeout)
#define ARM_SOCKET_SO_KEEPALIVE             4           ///< Keep-alive messages (default = 0); opt_val = &keepalive, opt_len = sizeof(keepalive), keepalive (integer): 0=disabled, enabled otherwise
#define ARM_SOCKET_SO_TYPE                  5           ///< Socket Type (Get only); opt_val = &socket_type, opt_len = sizeof(socket_type), socket_type (integer): ARM_SOCKET_SOCK_xxx

/****** Socket Function return codes *****/
#define ARM_SOCKET_ERROR                   (-1)         ///< Unspecified error
#define ARM_SOCKET_ESOCK                   (-2)         ///< Invalid socket
#define ARM_SOCKET_EINVAL                  (-3)         ///< Invalid argument
#define ARM_SOCKET_ENOTSUP                 (-4)         ///< Operation not supported
#define ARM_SOCKET_ENOMEM                  (-5)         ///< Not enough memory
#define ARM_SOCKET_EAGAIN                  (-6)         ///< Operation would block or timed out
#define ARM_SOCKET_EINPROGRESS             (-7)         ///< Operation in progress
#define ARM_SOCKET_ETIMEDOUT               (-8)         ///< Operation timed out
#define ARM_SOCKET_EISCONN                 (-9)         ///< Socket is connected
#define ARM_SOCKET_ENOTCONN                (-10)        ///< Socket is not connected
#define ARM_SOCKET_ECONNREFUSED            (-11)        ///< Connection rejected by the peer
#define ARM_SOCKET_ECONNRESET              (-12)        ///< Connection reset by the peer
#define ARM_SOCKET_ECONNABORTED            (-13)        ///< Connection aborted locally
#define ARM_SOCKET_EALREADY                (-14)        ///< Connection already in progress
#define ARM_SOCKET_EADDRINUSE              (-15)        ///< Address in use
#define ARM_SOCKET_EHOSTNOTFOUND           (-16)        ///< Host not found


// Function documentation
/**
  \fn            ARM_DRIVER_VERSION ARM_WIFI_GetVersion (void)
  \brief         Get driver version.
  \return        \ref ARM_DRIVER_VERSION
*/
/**
  \fn            ARM_WIFI_CAPABILITIES ARM_WIFI_GetCapabilities (void)
  \brief         Get driver capabilities.
  \return        \ref ARM_WIFI_CAPABILITIES
*/
/**
  \fn            int32_t ARM_WIFI_Initialize (ARM_WIFI_SignalEvent_t cb_event)
  \brief         Initialize WiFi Interface.
  \param[in]     cb_event Pointer to \ref ARM_WIFI_SignalEvent_t
  \return        execution status
                   - \ref ARM_DRIVER_OK                : Operation successful
                   - \ref ARM_DRIVER_ERROR             : Operation failed
*/
/**
  \fn            int32_t ARM_WIFI_Uninitialize (void)
  \brief         De-initialize WiFi Interface.
  \return        execution status
                   - \ref ARM_DRIVER_OK                : Operation successful
                   - \ref ARM_DRIVER_ERROR             : Operation failed
*/
/**
  \fn            int32_t ARM_WIFI_PowerControl (ARM_POWER_STATE state)
  \brief         Control WiFi Interface Power.
  \param[in]     state    Power state
                   - \ref ARM_POWER_OFF                : Power off: no operation possible
                   - \ref ARM_POWER_LOW                : Low power mode: retain state, detect and signal wake-up events
                   - \ref ARM_POWER_FULL               : Power on: full operation at maximum performance
  \return        execution status
                   - \ref ARM_DRIVER_OK                : Operation successful
                   - \ref ARM_DRIVER_ERROR             : Operation failed
                   - \ref ARM_DRIVER_ERROR_UNSUPPORTED : Operation not supported
*/
/**
  \fn            int32_t ARM_WIFI_SetOption (uint32_t option, const void *data, uint32_t len)
  \brief         Set WiFi Interface Options.
  \param[in]     option   Option to set
  \param[in]     data     Pointer to data relevant to selected option
  \param[in]     len      Length of data (in bytes)
  \return        execution status
                   - \ref ARM_DRIVER_OK                : Operation successful
                   - \ref ARM_DRIVER_ERROR             : Operation failed
                   - \ref ARM_DRIVER_ERROR_UNSUPPORTED : Operation not supported
                   - \ref ARM_DRIVER_ERROR_PARAMETER   : Parameter error (NULL data pointer or len less than option specifies)
*/
/**
  \fn            int32_t ARM_WIFI_GetOption (uint32_t option, void *data, uint32_t *len)
  \brief         Get WiFi Interface Options.
  \param[in]     option   Option to get
  \param[out]    data     Pointer to memory where data for selected option will be returned
  \param[in,out] len      Pointer to length of data (input/output)
                   - input: maximum length of data that can be returned (in bytes)
                   - output: length of returned data (in bytes)
  \return        execution status
                   - \ref ARM_DRIVER_OK                : Operation successful
                   - \ref ARM_DRIVER_ERROR             : Operation failed
                   - \ref ARM_DRIVER_ERROR_UNSUPPORTED : Operation not supported
                   - \ref ARM_DRIVER_ERROR_PARAMETER   : Parameter error (NULL data or len pointer, or *len less than option specifies)
*/
/**
  \fn            int32_t ARM_WIFI_Scan (ARM_WIFI_AP_INFO_t ap_info[], uint32_t max_num)
  \brief         Scan for Access Points in range.
  \param[out]    ap_info  Pointer to array of ARM_WIFI_AP_INFO_t structures where Access Point Information will be returned
  \param[in]     max_num  Maximum number of Access Point information structures to return
  \return        number of ARM_WIFI_AP_INFO_t structures returned or error code
                   - value >= 0                        : Number of ARM_WIFI_AP_INFO_t structures returned
                   - \ref ARM_DRIVER_ERROR             : Operation failed
                   - \ref ARM_DRIVER_ERROR_PARAMETER   : Parameter error (NULL ap_info pointer or max_num equal to 0)
*/
/**
  \fn            int32_t ARM_WIFI_Connect (const char *ssid, const char *pass, uint8_t security, uint8_t ch)
  \brief         Connect Station to Access Point (join the AP).
  \param[in]     ssid     Pointer to Service Set Identifier (SSID) null-terminated string
  \param[in]     pass     Pointer to password null-terminated string
  \param[in]     security Security standard used
                   - \ref ARM_WIFI_SECURITY_OPEN       : Unsecured
                   - \ref ARM_WIFI_SECURITY_WEP        : Wired Equivalent Privacy (WEP)
                   - \ref ARM_WIFI_SECURITY_WPA        : WiFi Protected Access (WPA)
                   - \ref ARM_WIFI_SECURITY_WPA2       : WiFi Protected Access II (WPA2)
  \param[in]     ch       Channel
                   - value = 0: autodetect
                   - value > 0: exact channel to connect on
  \return        execution status
                   - \ref ARM_DRIVER_OK                : Operation successful
                   - \ref ARM_DRIVER_ERROR             : Operation failed
                   - \ref ARM_DRIVER_ERROR_TIMEOUT     : Timeout occurred
                   - \ref ARM_DRIVER_ERROR_UNSUPPORTED : Operation not supported (security type or channel autodetect not supported)
                   - \ref ARM_DRIVER_ERROR_PARAMETER   : Parameter error (NULL ssid pointer, or NULL pass pointer if security different then ARM_WIFI_SECURITY_OPEN or invalid security parameter)
*/
/**
  \fn            int32_t ARM_WIFI_ConnectWPS (const char *pin)
  \brief         Connect Station to Access Point via WiFi Protected Setup (WPS). Access Point information can be retrieved through 
                 GetOption function with ARM_WIFI_INFO_AP option.
  \param[in]     pin      Pointer to pin null-terminated string or push-button connection trigger
                   - value != NULL: pointer to pin null-terminated string
                   - value == NULL: push-button connection trigger
  \return        execution status
                   - \ref ARM_DRIVER_OK                : Operation successful
                   - \ref ARM_DRIVER_ERROR             : Operation failed
                   - \ref ARM_DRIVER_ERROR_TIMEOUT     : Timeout occurred
                   - \ref ARM_DRIVER_ERROR_UNSUPPORTED : Operation not supported
*/
/**
  \fn            int32_t ARM_WIFI_Disconnect (void)
  \brief         Disconnect Station from currently connected Access Point.
  \return        execution status
                   - \ref ARM_DRIVER_OK                : Operation successful
                   - \ref ARM_DRIVER_ERROR             : Operation failed
*/
/**
  \fn            int32_t ARM_WIFI_IsConnected (void)
  \brief         Check Station connection status.
  \return        connection status
                   - value != 0: connected
                   - value = 0: not connected
*/
/**
  \fn            int32_t ARM_WIFI_AP_Start (const char *ssid, const char *pass, uint8_t security, uint8_t ch)
  \brief         Start Access Point.
  \param[in]     ssid     Pointer to Service Set Identifier (SSID) null-terminated string
  \param[in]     pass     Pointer to password null-terminated string
  \param[in]     security Security standard used
                   - \ref ARM_WIFI_SECURITY_OPEN       : Unsecured
                   - \ref ARM_WIFI_SECURITY_WEP        : Wired Equivalent Privacy (WEP)
                   - \ref ARM_WIFI_SECURITY_WPA        : WiFi Protected Access (WPA)
                   - \ref ARM_WIFI_SECURITY_WPA2       : WiFi Protected Access II (WPA2)
  \param[in]     ch       Channel
  \return        execution status
                   - \ref ARM_DRIVER_OK                : Operation successful
                   - \ref ARM_DRIVER_ERROR             : Operation failed
                   - \ref ARM_DRIVER_ERROR_UNSUPPORTED : Operation not supported (security type or channel autodetect not supported)
                   - \ref ARM_DRIVER_ERROR_PARAMETER   : Parameter error (NULL ssid pointer, or NULL pass pointer if security different then ARM_WIFI_SECURITY_OPEN or invalid security parameter)
*/
/**
  \fn            int32_t ARM_WIFI_AP_Stop (void)
  \brief         Stop Access Point.
  \return        execution status
                   - \ref ARM_DRIVER_OK                : Operation successful
                   - \ref ARM_DRIVER_ERROR             : Operation failed
                   - \ref ARM_DRIVER_ERROR_UNSUPPORTED : Operation not supported
*/
/**
  \fn            int32_t ARM_WIFI_AP_IsRunning (void)
  \brief         Check Access Point running status.
  \return        running status
                   - value != 0: running
                   - value = 0: not running
*/
/**
  \fn            int32_t ARM_WIFI_BypassControl (uint32_t enable)
  \brief         Enable or disable bypass (pass-through) mode. Transmit and receive Ethernet frames (IP layer bypassed and WiFi/Ethernet translation).
  \param[in]     enable
                   - value != 0: enable
                   - value = 0: disable
  \return        execution status
                   - \ref ARM_DRIVER_OK                : Operation successful
                   - \ref ARM_DRIVER_ERROR             : Operation failed
                   - \ref ARM_DRIVER_ERROR_UNSUPPORTED : Operation not supported
*/
/**
  \fn            int32_t ARM_WIFI_EthSendFrame (const uint8_t *frame, uint32_t len)
  \brief         Send Ethernet frame (in bypass mode only).
  \param[in]     frame    Pointer to frame buffer with data to send
  \param[in]     len      Frame buffer length in bytes
  \return        execution status
                   - \ref ARM_DRIVER_OK                : Operation successful
                   - \ref ARM_DRIVER_ERROR             : Operation failed
                   - \ref ARM_DRIVER_ERROR_BUSY        : Driver is busy
                   - \ref ARM_DRIVER_ERROR_UNSUPPORTED : Operation not supported
                   - \ref ARM_DRIVER_ERROR_PARAMETER   : Parameter error (NULL frame pointer)
*/
/**
  \fn            int32_t ARM_WIFI_EthReadFrame (uint8_t *frame, uint32_t len)
  \brief         Read data of received Ethernet frame (in bypass mode only).
  \param[in]     frame    Pointer to frame buffer for data to read into
  \param[in]     len      Frame buffer length in bytes
  \return        number of data bytes read or error code
                   - value >= 0                        : Number of data bytes read
                   - \ref ARM_DRIVER_ERROR             : Operation failed
                   - \ref ARM_DRIVER_ERROR_UNSUPPORTED : Operation not supported
                   - \ref ARM_DRIVER_ERROR_PARAMETER   : Parameter error (NULL frame pointer)
*/
/**
  \fn            uint32_t ARM_WIFI_EthGetRxFrameSize (void)
  \brief         Get size of received Ethernet frame (in bypass mode only).
  \return        number of bytes in received frame
*/
/**
  \fn            int32_t ARM_WIFI_SocketCreate (int32_t af, int32_t type, int32_t protocol)
  \brief         Create a communication socket.
  \param[in]     af       Address family
  \param[in]     type     Socket type
  \param[in]     protocol Socket protocol
  \return        status information
                   - Socket identification number (>=0)
                   - \ref ARM_SOCKET_EINVAL            : Invalid argument
                   - \ref ARM_SOCKET_ENOTSUP           : Operation not supported
                   - \ref ARM_SOCKET_ENOMEM            : Not enough memory
                   - \ref ARM_SOCKET_ERROR             : Unspecified error
*/
/**
  \fn            int32_t ARM_WIFI_SocketBind (int32_t socket, const uint8_t *ip, uint32_t ip_len, uint16_t port)
  \brief         Assign a local address to a socket.
  \param[in]     socket   Socket identification number
  \param[in]     ip       Pointer to local IP address
  \param[in]     ip_len   Length of 'ip' address in bytes
  \param[in]     port     Local port number
  \return        status information
                   - 0                                 : Operation successful
                   - \ref ARM_SOCKET_ESOCK             : Invalid socket
                   - \ref ARM_SOCKET_EINVAL            : Invalid argument (address or socket already bound)
                   - \ref ARM_SOCKET_EADDRINUSE        : Address already in use
                   - \ref ARM_SOCKET_ERROR             : Unspecified error
*/
/**
  \fn            int32_t ARM_WIFI_SocketListen (int32_t socket, int32_t backlog)
  \brief         Listen for socket connections.
  \param[in]     socket   Socket identification number
  \param[in]     backlog  Number of connection requests that can be queued
  \return        status information
                   - 0                                 : Operation successful
                   - \ref ARM_SOCKET_ESOCK             : Invalid socket
                   - \ref ARM_SOCKET_EINVAL            : Invalid argument (socket not bound)
                   - \ref ARM_SOCKET_ENOTSUP           : Operation not supported
                   - \ref ARM_SOCKET_EISCONN           : Socket is already connected
                   - \ref ARM_SOCKET_ERROR             : Unspecified error
*/
/**
  \fn            int32_t ARM_WIFI_SocketAccept (int32_t socket, uint8_t *ip, uint32_t *ip_len, uint16_t *port)
  \brief         Accept a new connection on a socket.
  \param[in]     socket   Socket identification number
  \param[out]    ip       Pointer to buffer where address of connecting socket shall be returned (NULL for none)
  \param[in,out] ip_len   Pointer to length of 'ip' (or NULL if 'ip' is NULL)
                   - length of supplied 'ip' on input
                   - length of stored 'ip' on output
  \param[out]    port     Pointer to buffer where port of connecting socket shall be returned (NULL for none)
  \return        status information
                   - socket identification number of accepted socket (>=0)
                   - \ref ARM_SOCKET_ESOCK             : Invalid socket
                   - \ref ARM_SOCKET_EINVAL            : Invalid argument (socket not in listen mode)
                   - \ref ARM_SOCKET_ENOTSUP           : Operation not supported (socket type does not support accepting connections)
                   - \ref ARM_SOCKET_ECONNRESET        : Connection reset by the peer
                   - \ref ARM_SOCKET_ECONNABORTED      : Connection aborted locally
                   - \ref ARM_SOCKET_EAGAIN            : Operation would block or timed out (may be called again)
                   - \ref ARM_SOCKET_ERROR             : Unspecified error
*/
/**
  \fn            int32_t ARM_WIFI_SocketConnect (int32_t socket, const uint8_t *ip, uint32_t ip_len, uint16_t port)
  \brief         Connect a socket to a remote host.
  \param[in]     socket   socket identification number
  \param[in]     ip       Pointer to remote IP address
  \param[in]     ip_len   Length of 'ip' address in bytes
  \param[in]     port     Remote port number
  \return        status information
                   - 0                                 : Operation successful
                   - \ref ARM_SOCKET_ESOCK             : Invalid socket
                   - \ref ARM_SOCKET_EINVAL            : Invalid argument
                   - \ref ARM_SOCKET_EALREADY          : Connection already in progress
                   - \ref ARM_SOCKET_EINPROGRESS       : Operation in progress
                   - \ref ARM_SOCKET_EISCONN           : Socket is connected
                   - \ref ARM_SOCKET_ECONNREFUSED      : Connection rejected by the peer
                   - \ref ARM_SOCKET_ECONNABORTED      : Connection aborted locally
                   - \ref ARM_SOCKET_EADDRINUSE        : Address already in use
                   - \ref ARM_SOCKET_ETIMEDOUT         : Operation timed out
                   - \ref ARM_SOCKET_ERROR             : Unspecified error
*/
/**
  \fn            int32_t ARM_WIFI_SocketRecv (int32_t socket, void *buf, uint32_t len)
  \brief         Receive data from a connected socket.
  \param[in]     socket   Socket identification number
  \param[out]    buf      Pointer to buffer where data should be stored
  \param[in]     len      Length of buffer (in bytes)
  \return        status information
                   - number of bytes received (>0)
                   - \ref ARM_SOCKET_ESOCK             : Invalid socket
                   - \ref ARM_SOCKET_EINVAL            : Invalid argument (pointer to buffer or length)
                   - \ref ARM_SOCKET_ENOTCONN          : Socket is not connected
                   - \ref ARM_SOCKET_ECONNRESET        : Connection reset by the peer
                   - \ref ARM_SOCKET_ECONNABORTED      : Connection aborted locally
                   - \ref ARM_SOCKET_EAGAIN            : Operation would block or timed out (may be called again)
                   - \ref ARM_SOCKET_ERROR             : Unspecified error
*/
/**
  \fn            int32_t ARM_WIFI_SocketRecvFrom (int32_t socket, void *buf, uint32_t len, uint8_t *ip, uint32_t *ip_len, uint16_t *port)
  \brief         Receive data from a socket.
  \param[in]     socket   Socket identification number
  \param[out]    buf      Pointer to buffer where data should be stored
  \param[in]     len      Length of buffer (in bytes)
  \param[out]    ip       Pointer to buffer where remote source address shall be returned (NULL for none)
  \param[in,out] ip_len   Pointer to length of 'ip' (or NULL if 'ip' is NULL)
                   - length of supplied 'ip' on input
                   - length of stored 'ip' on output
  \param[out]    port     Pointer to buffer where remote source port shall be returned (NULL for none)
  \return        status information
                   - number of bytes received (>0)
                   - \ref ARM_SOCKET_ESOCK             : Invalid socket
                   - \ref ARM_SOCKET_EINVAL            : Invalid argument (pointer to buffer or length)
                   - \ref ARM_SOCKET_ENOTCONN          : Socket is not connected
                   - \ref ARM_SOCKET_ECONNRESET        : Connection reset by the peer
                   - \ref ARM_SOCKET_ECONNABORTED      : Connection aborted locally
                   - \ref ARM_SOCKET_EAGAIN            : Operation would block or timed out (may be called again)
                   - \ref ARM_SOCKET_ERROR             : Unspecified error
*/
/**
  \fn            int32_t ARM_WIFI_SocketSend (int32_t socket, const void *buf, uint32_t len)
  \brief         Send data to a connected socket.
  \param[in]     socket   Socket identification number
  \param[in]     buf      Pointer to buffer containing data to send
  \param[in]     len      Length of data (in bytes)
  \return        status information
                   - number of bytes sent (>0)
                   - \ref ARM_SOCKET_ESOCK             : Invalid socket
                   - \ref ARM_SOCKET_EINVAL            : Invalid argument (pointer to buffer or length)
                   - \ref ARM_SOCKET_ENOTCONN          : Socket is not connected
                   - \ref ARM_SOCKET_ECONNRESET        : Connection reset by the peer
                   - \ref ARM_SOCKET_ECONNABORTED      : Connection aborted locally
                   - \ref ARM_SOCKET_EAGAIN            : Operation would block or timed out (may be called again)
                   - \ref ARM_SOCKET_ERROR             : Unspecified error
*/
/**
  \fn            int32_t ARM_WIFI_SocketSendTo (int32_t socket, const void *buf, uint32_t len, const uint8_t *ip, uint32_t ip_len, uint16_t port)
  \brief         Send data to a socket.
  \param[in]     socket   Socket identification number
  \param[in]     buf      Pointer to buffer containing data to send
  \param[in]     len      Length of data (in bytes)
  \param[in]     ip       Pointer to remote destination IP address
  \param[in]     ip_len   Length of 'ip' address in bytes
  \param[in]     port     Remote destination port number
  \return        status information
                   - number of bytes sent (>0)
                   - \ref ARM_SOCKET_ESOCK             : Invalid socket
                   - \ref ARM_SOCKET_EINVAL            : Invalid argument (pointer to buffer or length)
                   - \ref ARM_SOCKET_ENOTCONN          : Socket is not connected
                   - \ref ARM_SOCKET_ECONNRESET        : Connection reset by the peer
                   - \ref ARM_SOCKET_ECONNABORTED      : Connection aborted locally
                   - \ref ARM_SOCKET_EAGAIN            : Operation would block or timed out (may be called again)
                   - \ref ARM_SOCKET_ERROR             : Unspecified error
*/
/**
  \fn            int32_t ARM_WIFI_SocketGetSockName (int32_t socket, uint8_t *ip, uint32_t *ip_len, uint16_t *port)
  \brief         Retrieve local IP address and port of a socket.
  \param[in]     socket   Socket identification number
  \param[out]    ip       Pointer to buffer where local address shall be returned (NULL for none)
  \param[in,out] ip_len   Pointer to length of 'ip' (or NULL if 'ip' is NULL)
                   - length of supplied 'ip' on input
                   - length of stored 'ip' on output
  \param[out]    port     Pointer to buffer where local port shall be returned (NULL for none)
  \return        status information
                   - 0                                 : Operation successful
                   - \ref ARM_SOCKET_ESOCK             : Invalid socket
                   - \ref ARM_SOCKET_EINVAL            : Invalid argument (pointer to buffer or length)
                   - \ref ARM_SOCKET_ERROR             : Unspecified error
*/
/**
  \fn            int32_t ARM_WIFI_SocketGetPeerName (int32_t socket, uint8_t *ip, uint32_t *ip_len, uint16_t *port)
  \brief         Retrieve remote IP address and port of a socket
  \param[in]     socket   Socket identification number
  \param[out]    ip       Pointer to buffer where remote address shall be returned (NULL for none)
  \param[in,out] ip_len   Pointer to length of 'ip' (or NULL if 'ip' is NULL)
                   - length of supplied 'ip' on input
                   - length of stored 'ip' on output
  \param[out]    port     Pointer to buffer where remote port shall be returned (NULL for none)
  \return        status information
                   - 0                                 : Operation successful
                   - \ref ARM_SOCKET_ESOCK             : Invalid socket
                   - \ref ARM_SOCKET_EINVAL            : Invalid argument (pointer to buffer or length)
                   - \ref ARM_SOCKET_ENOTCONN          : Socket is not connected
                   - \ref ARM_SOCKET_ERROR             : Unspecified error
*/
/**
  \fn            int32_t ARM_WIFI_SocketGetOpt (int32_t socket, int32_t opt_id, void *opt_val, uint32_t *opt_len)
  \brief         Get socket option.
  \param[in]     socket   Socket identification number
  \param[in]     opt_id   Option identifier
  \param[out]    opt_val  Pointer to the buffer that will receive the option value
  \param[in,out] opt_len  Pointer to length of the option value
                   - length of buffer on input
                   - length of data on output
  \return        status information
                   - 0                                 : Operation successful
                   - \ref ARM_SOCKET_ESOCK             : Invalid socket
                   - \ref ARM_SOCKET_EINVAL            : Invalid argument
                   - \ref ARM_SOCKET_ENOTSUP           : Operation not supported
                   - \ref ARM_SOCKET_ERROR             : Unspecified error
*/
/**
  \fn            int32_t ARM_WIFI_SocketSetOpt (int32_t socket, int32_t opt_id, const void *opt_val, uint32_t opt_len)
  \brief         Set socket option.
  \param[in]     socket   Socket identification number
  \param[in]     opt_id   Option identifier
  \param[in]     opt_val  Pointer to the option value
  \param[in]     opt_len  Length of the option value in bytes
  \return        status information
                   - 0                                 : Operation successful
                   - \ref ARM_SOCKET_ESOCK             : Invalid socket
                   - \ref ARM_SOCKET_EINVAL            : Invalid argument
                   - \ref ARM_SOCKET_ENOTSUP           : Operation not supported
                   - \ref ARM_SOCKET_ERROR             : Unspecified error
*/
/**
  \fn            int32_t ARM_WIFI_SocketClose (int32_t socket)
  \brief         Close and release a socket.
  \param[in]     socket   Socket identification number
  \return        status information
                   - 0                                 : Operation successful
                   - \ref ARM_SOCKET_ESOCK             : Invalid socket
                   - \ref ARM_SOCKET_EAGAIN            : Operation would block (may be called again)
                   - \ref ARM_SOCKET_ERROR             : Unspecified error
*/
/**
  \fn            int32_t ARM_WIFI_SocketGetHostByName (const char *name, int32_t af, uint8_t *ip, uint32_t *ip_len)
  \brief         Retrieve host IP address from host name.
  \param[in]     name     Host name
  \param[in]     af       Address family
  \param[out]    ip       Pointer to buffer where resolved IP address shall be returned
  \param[in,out] ip_len   Pointer to length of 'ip'
                   - length of supplied 'ip' on input
                   - length of stored 'ip' on output
  \return        status information
                   - 0                                 : Operation successful
                   - \ref ARM_SOCKET_EINVAL            : Invalid argument
                   - \ref ARM_SOCKET_ENOTSUP           : Operation not supported
                   - \ref ARM_SOCKET_ETIMEDOUT         : Operation timed out
                   - \ref ARM_SOCKET_EHOSTNOTFOUND     : Host not found
                   - \ref ARM_SOCKET_ERROR             : Unspecified error
*/
/**
  \fn            int32_t ARM_WIFI_Ping (const uint8_t *ip, uint32_t ip_len)
  \brief         Probe remote host with Ping command.
  \param[in]     ip       Pointer to remote host IP address
  \param[in]     ip_len   Length of 'ip' address in bytes
  \return        execution status
                   - \ref ARM_DRIVER_OK                : Operation successful
                   - \ref ARM_DRIVER_ERROR             : Operation failed
                   - \ref ARM_DRIVER_ERROR_TIMEOUT     : Timeout occurred
                   - \ref ARM_DRIVER_ERROR_UNSUPPORTED : Operation not supported
                   - \ref ARM_DRIVER_ERROR_PARAMETER   : Parameter error (NULL ip pointer or ip_len different than 4 or 16)
*/
/**
  \fn            void ARM_WIFI_SignalEvent (uint32_t event, void *arg)
  \brief         Signal WiFi Events.
  \param[in]     event    \ref wifi_event notification mask
  \param[in]     arg      Pointer to argument of signaled event
  \return        none
*/

typedef void (*ARM_WIFI_SignalEvent_t) (uint32_t event, void *arg); ///< Pointer to \ref ARM_WIFI_SignalEvent : Signal WiFi Event.


/**
\brief WiFi Driver Capabilities.
*/
typedef struct {
  uint32_t wps                   : 1;   ///< Station: WiFi Protected Setup (WPS)
  uint32_t ap                    : 1;   ///< Access Point
  uint32_t ap_connect_event      : 1;   ///< Access Point: event generated on Station connect
  uint32_t ap_disconnect_event   : 1;   ///< Access Point: event generated on Station disconnect
  uint32_t bypass_mode           : 1;   ///< Bypass or pass-through mode (Eth interface)
  uint32_t eth_rx_frame_event    : 1;   ///< Event generated on Ethernet frame reception in bypass mode
  uint32_t ip                    : 1;   ///< IP (UDP/TCP) (Socket interface)
  uint32_t ip6                   : 1;   ///< IPv6 (Socket interface)
  uint32_t ping                  : 1;   ///< Ping (ICMP)
  uint32_t reserved              : 23;  ///< Reserved (must be zero)
} ARM_WIFI_CAPABILITIES;

/**
\brief Access structure of the WiFi Driver.
*/
typedef struct {
  ARM_DRIVER_VERSION    (*GetVersion)                  (void);
  ARM_WIFI_CAPABILITIES (*GetCapabilities)             (void);
  int32_t               (*Initialize)                  (ARM_WIFI_SignalEvent_t cb_event);
  int32_t               (*Uninitialize)                (void);
  int32_t               (*PowerControl)                (ARM_POWER_STATE state);
  int32_t               (*SetOption)                   (uint32_t option, const void *data, uint32_t  len);
  int32_t               (*GetOption)                   (uint32_t option,       void *data, uint32_t *len);
  int32_t               (*Scan)                        (ARM_WIFI_AP_INFO_t ap_info[], uint32_t max_num);
  int32_t               (*Connect)                     (const char *ssid, const char *pass, uint8_t security, uint8_t ch);
  int32_t               (*ConnectWPS)                  (const char *pin);
  int32_t               (*Disconnect)                  (void);
  int32_t               (*IsConnected)                 (void);
  int32_t               (*AP_Start)                    (const char *ssid, const char *pass, uint8_t security, uint8_t ch);
  int32_t               (*AP_Stop)                     (void);
  int32_t               (*AP_IsRunning)                (void);
  int32_t               (*BypassControl)               (uint32_t enable);
  int32_t               (*EthSendFrame)                (const uint8_t *frame, uint32_t len);
  int32_t               (*EthReadFrame)                (      uint8_t *frame, uint32_t len);
  uint32_t              (*EthGetRxFrameSize)           (void);
  int32_t               (*SocketCreate)                (int32_t af, int32_t type, int32_t protocol);
  int32_t               (*SocketBind)                  (int32_t socket, const uint8_t *ip, uint32_t  ip_len, uint16_t  port);
  int32_t               (*SocketListen)                (int32_t socket, int32_t backlog);
  int32_t               (*SocketAccept)                (int32_t socket,       uint8_t *ip, uint32_t *ip_len, uint16_t *port);
  int32_t               (*SocketConnect)               (int32_t socket, const uint8_t *ip, uint32_t  ip_len, uint16_t  port);
  int32_t               (*SocketRecv)                  (int32_t socket, void *buf, uint32_t len);
  int32_t               (*SocketRecvFrom)              (int32_t socket, void *buf, uint32_t len, uint8_t *ip, uint32_t *ip_len, uint16_t *port);
  int32_t               (*SocketSend)                  (int32_t socket, const void *buf, uint32_t len);
  int32_t               (*SocketSendTo)                (int32_t socket, const void *buf, uint32_t len, const uint8_t *ip, uint32_t ip_len, uint16_t port);
  int32_t               (*SocketGetSockName)           (int32_t socket, uint8_t *ip, uint32_t *ip_len, uint16_t *port);
  int32_t               (*SocketGetPeerName)           (int32_t socket, uint8_t *ip, uint32_t *ip_len, uint16_t *port);
  int32_t               (*SocketGetOpt)                (int32_t socket, int32_t opt_id,       void *opt_val, uint32_t *opt_len);
  int32_t               (*SocketSetOpt)                (int32_t socket, int32_t opt_id, const void *opt_val, uint32_t  opt_len);
  int32_t               (*SocketClose)                 (int32_t socket);
  int32_t               (*SocketGetHostByName)         (const char *name, int32_t af, uint8_t *ip, uint32_t *ip_len);
  int32_t               (*Ping)                        (const uint8_t *ip, uint32_t ip_len);
} const ARM_DRIVER_WIFI;

#ifdef  __cplusplus
}
#endif

#endif /* DRIVER_WIFI_H_ */
