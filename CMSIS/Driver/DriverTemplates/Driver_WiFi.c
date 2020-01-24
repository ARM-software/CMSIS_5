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
 */

#include "Driver_WiFi.h"

#define ARM_WIFI_DRV_VERSION ARM_DRIVER_VERSION_MAJOR_MINOR(1, 0)        // Driver version

// Driver Version
static const ARM_DRIVER_VERSION driver_version = {
  ARM_WIFI_API_VERSION,
  ARM_WIFI_DRV_VERSION 
};

// Driver Capabilities
static const ARM_WIFI_CAPABILITIES driver_capabilities = { 
  0U,                                   // Station supported
  0U,                                   // Access Point supported
  0U,                                   // Concurrent Station and Access Point not supported
  0U,                                   // WiFi Protected Setup (WPS) for Station supported
  0U,                                   // WiFi Protected Setup (WPS) for Access Point not supported
  0U,                                   // Access Point: event generated on Station connect
  0U,                                   // Access Point: event not generated on Station disconnect
  0U,                                   // Event not generated on Ethernet frame reception in bypass mode
  0U,                                   // Bypass or pass-through mode (Ethernet interface) not supported
  0U,                                   // IP (UDP/TCP) (Socket interface) supported
  0U,                                   // IPv6 (Socket interface) not supported
  0U,                                   // Ping (ICMP) supported
  0U                                    // Reserved (must be zero)
};
static ARM_DRIVER_VERSION WiFi_GetVersion (void) {
  return driver_version; 
}

static ARM_WIFI_CAPABILITIES WiFi_GetCapabilities (void) { 
  return driver_capabilities; 
}

static int32_t WiFi_Initialize (ARM_WIFI_SignalEvent_t cb_event) {
  (void)cb_event;
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}

static int32_t WiFi_Uninitialize (void) {
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}

static int32_t WiFi_PowerControl (ARM_POWER_STATE state) {
  (void)state;
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}

static int32_t WiFi_GetModuleInfo (char *module_info, uint32_t max_len) {
  (void)module_info; (void) max_len;
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}

static int32_t WiFi_SetOption (uint32_t interface, uint32_t option, const void *data, uint32_t len) {
  (void)interface; (void) option; (void)data; (void)len;
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}

static int32_t WiFi_GetOption (uint32_t interface, uint32_t option, void *data, uint32_t *len) {
  (void)interface; (void) option; (void)data; (void)len;
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}
static int32_t WiFi_Scan (ARM_WIFI_SCAN_INFO_t scan_info[], uint32_t max_num) {
  (void)scan_info; (void)max_num;
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}

static int32_t WiFi_Activate (uint32_t interface, const ARM_WIFI_CONFIG_t *config) {
  (void)interface; (void)config;
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}

static int32_t WiFi_Deactivate (uint32_t interface) {
  (void)interface;
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}

static uint32_t WiFi_IsConnected (void) {
  return 0U;
}

static int32_t WiFi_GetNetInfo (ARM_WIFI_NET_INFO_t *net_info) {
  (void)net_info;
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}

static int32_t WiFi_BypassControl (uint32_t interface, uint32_t mode) {
  (void)interface; (void)mode;
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}

static int32_t WiFi_EthSendFrame (uint32_t interface, const uint8_t *frame, uint32_t len){
  (void)interface; (void)frame; (void)len;
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}

static int32_t WiFi_EthReadFrame (uint32_t interface, uint8_t *frame, uint32_t len){
  (void)interface; (void)frame; (void)len;
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}

static int32_t WiFi_EthGetRxFrameSize (uint32_t interface){
  (void)interface;
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}

static int32_t WiFi_SocketCreate (int32_t af, int32_t type, int32_t protocol) {
  (void)af; (void)type; (void)protocol;
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}

static int32_t WiFi_SocketBind (int32_t socket, const uint8_t *ip, uint32_t ip_len, uint16_t port) {
  (void)socket; (void)ip; (void)ip_len; (void)port;
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}


static int32_t WiFi_SocketListen (int32_t socket, int32_t backlog) {
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}

static int32_t WiFi_SocketAccept (int32_t socket, uint8_t *ip, uint32_t *ip_len, uint16_t *port) {
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}

static int32_t WiFi_SocketConnect (int32_t socket, const uint8_t *ip, uint32_t ip_len, uint16_t port) {
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}

static int32_t WiFi_SocketRecv (int32_t socket, void *buf, uint32_t len) {
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}

static int32_t WiFi_SocketRecvFrom (int32_t socket, void *buf, uint32_t len, uint8_t *ip, uint32_t *ip_len, uint16_t *port) {
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}

static int32_t WiFi_SocketSend (int32_t socket, const void *buf, uint32_t len) {
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}

static int32_t WiFi_SocketSendTo (int32_t socket, const void *buf, uint32_t len, const uint8_t *ip, uint32_t ip_len, uint16_t port) {
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}

static int32_t WiFi_SocketGetSockName (int32_t socket, uint8_t *ip, uint32_t *ip_len, uint16_t *port) {
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}

static int32_t WiFi_SocketGetPeerName (int32_t socket, uint8_t *ip, uint32_t *ip_len, uint16_t *port) {
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}

static int32_t WiFi_SocketGetOpt (int32_t socket, int32_t opt_id, void *opt_val, uint32_t *opt_len) {
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}

static int32_t WiFi_SocketSetOpt (int32_t socket, int32_t opt_id, const void *opt_val, uint32_t opt_len) {
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}

static int32_t WiFi_SocketClose (int32_t socket) {
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}

static int32_t WiFi_SocketGetHostByName (const char *name, int32_t af, uint8_t *ip, uint32_t *ip_len) {
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}

static int32_t WiFi_Ping (const uint8_t *ip, uint32_t ip_len) {
  return ARM_DRIVER_ERROR_UNSUPPORTED;
}

/* WiFi Driver Control Block */
extern
ARM_DRIVER_WIFI Driver_WiFi_(WIFI_DRIVER);
ARM_DRIVER_WIFI Driver_WiFi_(WIFI_DRIVER) = { 
  WiFi_GetVersion,
  WiFi_GetCapabilities,
  WiFi_Initialize,
  WiFi_Uninitialize,
  WiFi_PowerControl,
  WiFi_GetModuleInfo,
  WiFi_SetOption,
  WiFi_GetOption,
  WiFi_Scan,
  WiFi_Activate,
  WiFi_Deactivate,
  WiFi_IsConnected,
  WiFi_GetNetInfo,
  WiFi_BypassControl,
  WiFi_EthSendFrame,
  WiFi_EthGetRxFrameSize,
  WiFi_EthGetRxFrameSize,
  WiFi_SocketCreate,
  WiFi_SocketBind,
  WiFi_SocketListen,
  WiFi_SocketAccept,
  WiFi_SocketConnect,
  WiFi_SocketRecv,
  WiFi_SocketRecvFrom,
  WiFi_SocketSend,
  WiFi_SocketSendTo,
  WiFi_SocketGetSockName,
  WiFi_SocketGetPeerName,
  WiFi_SocketGetOpt,
  WiFi_SocketSetOpt,
  WiFi_SocketClose,
  WiFi_SocketGetHostByName,
  WiFi_Ping
};
