/**
\defgroup wifi_interface_gr WiFi Interface
\brief Driver API for WiFi (%Driver_WiFi.h)
\details 

Wi-Fi is technology for radio wireless local area networking of devices. Wi-Fi compatible devices typically connect to 
the Internet via a WLAN and a wireless access point (AP) also called hotspot.

Wikipedia offers more information about 
the <a href="http://en.wikipedia.org/wiki/Ethernet" target="_blank"><b>WiFi</b></a>.

<b>Driver Block Diagram</b>

\image html WiFi.png  "Block Diagram of the WiFi interface"

<b>WiFi API</b>

The following header files define the Application Programming Interface (API) for the WiFi interface:
  - \b %Driver_WiFi.h : Driver API for WiFi

The CMSIS-Driver WiFi provides access to the following interfaces:

 - \ref wifi_control_gr "Control interface": setup and control the WiFi API functions.
 - \ref wifi_management_gr "Management interface": allows to configure the connection to a WiFi access point (AP).
 - \ref wifi_socket_gr "Socket interface": provides the interface to an IP stack that is running on WiFi module. This IP stack handles data communication.
 - \ref wifi_bypass_gr "Bypass interface": is an optional interface an allows to transfer Ethernet frames by WiFi module. Using this interface requires that the IP stack is running on microcontroller.

The WiFi interface typically requires CMSIS-RTOS features (i.e. mutex) and is frequently implemented with a peripheral that connected to a system using a SPI or UART interface. 
However there are also some microcontroller devices with on-chip WiFi interface available.  

The WiFi CMSIS-Driver implementation is therefore mostly provided as separate software pack.  It is frequently implemented as wrapper to the SDK (Software Development Kit) of the
WiFi chipset.


<b>Driver Functions</b>

The driver functions are published in the access struct as explained in \ref DriverFunctions
  - \ref ARM_DRIVER_WIFI : access struct for WiFi driver functions

  
<b>Example Code</b>

@{
*/


/**
\struct     ARM_DRIVER_WIFI
\details 
The functions of the WiFi driver are accessed by function pointers exposed by this structure.
Refer to \ref DriverFunctions for overview information.

Each instance of a WiFi interface provides such an access structure. 
The instance is identified by a postfix number in the symbol name of the access structure, for example:
 - \b Driver_WiFi0 is the name of the access struct of the first instance (no. 0).
 - \b Driver_WiFi1 is the name of the access struct of the second instance (no. 1).

A middleware configuration setting allows connecting the middleware to a specific driver instance \b %Driver_WiFi<i>n</i>.
The default is \token{0}, which connects a middleware to the first instance of a driver.
**************************************************************************************************************************/


//
//  Functions
//

/**
\defgroup wifi_control_gr WiFi Control
\ingroup wifi_interface_gr
\brief Control functions for the WiFi API interface
\details  
The \ref wifi_control_gr functions setup and control the WiFi API interface.
@{
*/

/** 
\struct     ARM_WIFI_CAPABILITIES
\details
A WiFi driver can be implemented with different capabilities.
The data fields of this structure encode the capabilities implemented by this driver.

<b>Returned by:</b>
  - \ref ARM_WIFI_GetCapabilities
**************************************************************************************************************************/

/**
\typedef    ARM_WIFI_SignalEvent_t
\details

<b>Parameter for:</b>
  - \ref ARM_WIFI_Initialize
*******************************************************************************************************************/

/**
\defgroup wifi_event WiFi Event
\ingroup wifi_control_gr
\brief Specifies WiFi events notified via \ref ARM_WIFI_SignalEvent.
\details
@{
\def ARM_WIFI_EVENT_AP_CONNECT   
\def ARM_WIFI_EVENT_AP_DISCONNECT    
\def ARM_WIFI_EVENT_ETH_RX_FRAME    
@}
*/

ARM_DRIVER_VERSION ARM_WIFI_GetVersion (void) {
  return { 0, 0 };	
}
/**
  \fn            ARM_DRIVER_VERSION ARM_WIFI_GetVersion (void)
\details
The function \ref ARM_WIFI_GetVersion returns version information of the driver implementation in \ref ARM_DRIVER_VERSION.

API version is the version of the CMSIS-Driver specification used to implement this driver.
Driver version is source code version of the actual driver implementation.
 
Example:
\code
extern ARM_DRIVER_WIFI Driver_WIFI0;
ARM_DRIVER_WIFI *drv_info;
 
void setup_wifi (void)  {
  ARM_DRIVER_VERSION  version;
 
  drv_info = &Driver_WIFI0;  
  version = drv_info->GetVersion ();
  if (version.api < 0x10A)   {      // requires at minimum API version 1.10 or higher
    // error handling
    return;
  }
}
\endcode
*/

ARM_WIFI_CAPABILITIES ARM_WIFI_GetCapabilities (void) {
  return { 0 };
}
/**
  \fn            ARM_WIFI_CAPABILITIES ARM_WIFI_GetCapabilities (void)
\details
 
Example:
\code
\endcode
*/

int32_t ARM_WIFI_Initialize (ARM_WIFI_SignalEvent_t cb_event) {
  return ARM_DRIVER_OK;	
}
/**
  \fn            int32_t ARM_WIFI_Initialize (ARM_WIFI_SignalEvent_t cb_event)
\details
 
Example:
\code
\endcode
*/

int32_t ARM_WIFI_Uninitialize (void) {
  return ARM_DRIVER_OK;	
}
/**
  \fn            int32_t ARM_WIFI_Uninitialize (void)
\details
 
Example:
\code
\endcode
*/

int32_t ARM_WIFI_PowerControl (ARM_POWER_STATE state) {
  return ARM_DRIVER_OK;	
}
/**
  \fn            int32_t ARM_WIFI_PowerControl (ARM_POWER_STATE state)
\details
 
Example:
\code
\endcode
*/

void ARM_WIFI_SignalEvent (uint32_t event, void *arg) {
}
/**
\fn            void ARM_WIFI_SignalEvent (uint32_t event, void *arg)
\details
The function ARM_WIFI_SignalEvent is a callback function registered by the function ARM_WIFI_Initialize. It is called by the WiFi driver 
to notify the application about WiFi Events occurred during operation.

The parameter \em event indicates the event that occurred during driver operation.

The parameter \arg is a pointer to additional information about the event.

The following events can be generated:

Parameter \em event                  | Description
:------------------------------------|:------------------------------------------
\ref ARM_WIFI_EVENT_AP_CONNECT       | \copybrief ARM_WIFI_EVENT_AP_CONNECT
\ref ARM_WIFI_EVENT_AP_DISCONNECT    | \copybrief ARM_WIFI_EVENT_AP_DISCONNECT
\ref ARM_WIFI_EVENT_ETH_RX_FRAME     | \copybrief ARM_WIFI_EVENT_ETH_RX_FRAME

 
Example:
\code
\endcode
*/

/**
@}  //wifi_control_gr
*/

/**
\defgroup wifi_management_gr WiFi Management
\ingroup wifi_interface_gr
\brief Configure the connection to a WiFi access point (AP)
\details The \ref wifi_management_gr functions allows to configure the connection to a WiFi access point (AP) also called hotspot.
@{
*/

/**
\defgroup WiFi_option WiFi Option Codes
\ingroup wifi_management_gr
\brief  WiFi Option Codes for \ref ARM_WIFI_SetOption or \ref ARM_WIFI_GetOption function.
\details 
Many parameters of the WiFi driver are configured using the \ref ARM_WIFI_SetOption or \ref ARM_WIFI_GetOption function.
@{
\def ARM_WIFI_SSID 
\sa WiFi_option                   
\def ARM_WIFI_BSSID                   
\sa WiFi_option                   
\def ARM_WIFI_PASS                    
\sa WiFi_option                   
\def ARM_WIFI_SECURITY                
\sa WiFi_option                   
\def ARM_WIFI_CHANNEL                 
\sa WiFi_option                   
\def ARM_WIFI_RSSI                    
\sa WiFi_option                   
\def ARM_WIFI_TX_POWER                
\sa WiFi_option                   
\def ARM_WIFI_MAC                     
\sa WiFi_option                   
\def ARM_WIFI_IP                      
\sa WiFi_option                   
\def ARM_WIFI_IP_SUBNET_MASK          
\sa WiFi_option                   
\def ARM_WIFI_IP_GATEWAY              
\sa WiFi_option                   
\def ARM_WIFI_IP_DNS1                 
\sa WiFi_option                   
\def ARM_WIFI_IP_DNS2                 
\sa WiFi_option                   
\def ARM_WIFI_IP_DHCP                 
\sa WiFi_option                   
\def ARM_WIFI_IP6_GLOBAL              
\sa WiFi_option                   
\def ARM_WIFI_IP6_LINK_LOCAL          
\sa WiFi_option                   
\def ARM_WIFI_IP6_SUBNET_PREFIX_LEN   
\sa WiFi_option                   
\def ARM_WIFI_IP6_GATEWAY             
\sa WiFi_option                   
\def ARM_WIFI_IP6_DNS1                
\sa WiFi_option                   
\def ARM_WIFI_IP6_DNS2                
\sa WiFi_option                   
\def ARM_WIFI_IP6_DHCP_MODE           
\sa WiFi_option                   
\def ARM_WIFI_AP_SSID_HIDE            
\sa WiFi_option                   
\sa WiFi_option                   
\def ARM_WIFI_AP_TX_POWER             
\sa WiFi_option                   
\def ARM_WIFI_AP_MAC                  
\sa WiFi_option                   
\def ARM_WIFI_AP_IP                   
\sa WiFi_option                   
\def ARM_WIFI_AP_IP_SUBNET_MASK       
\sa WiFi_option                   
\def ARM_WIFI_AP_IP_GATEWAY           
\sa WiFi_option                   
\def ARM_WIFI_AP_IP_DNS1              
\sa WiFi_option                   
\def ARM_WIFI_AP_IP_DNS2              
\sa WiFi_option                   
\def ARM_WIFI_AP_IP_DHCP              
\sa WiFi_option                   
\sa WiFi_option                   
\def ARM_WIFI_AP_IP_DHCP_POOL_BEGIN   
\sa WiFi_option                   
\def ARM_WIFI_AP_IP_DHCP_POOL_END     
\sa WiFi_option                   
\def ARM_WIFI_AP_IP_DHCP_LEASE_TIME   
\sa WiFi_option                   
\def ARM_WIFI_AP_IP_DHCP_TABLE        
\sa WiFi_option                   
\def ARM_WIFI_AP_IP6_GLOBAL           
\sa WiFi_option                   
\def ARM_WIFI_AP_IP6_LINK_LOCAL       
\sa WiFi_option                   
\def ARM_WIFI_AP_IP6_SUBNET_PREFIX_LEN
\sa WiFi_option                   
\def ARM_WIFI_AP_IP6_GATEWAY          
\sa WiFi_option                   
\def ARM_WIFI_AP_IP6_DNS1             
\sa WiFi_option                   
\def ARM_WIFI_AP_IP6_DNS2
\sa WiFi_option                   
@}
*/

/**
\defgroup wifi_sec_type WiFi Security Type
\ingroup wifi_management_gr
\brief Specifies WiFi security type for \ref ARM_WIFI_Connect.
\details
The WiFi security type defines that standard used to protect the wireless network from unauthorized access.
@{
\def ARM_WIFI_SECURITY_OPEN   
\sa wifi_sec_type
\def ARM_WIFI_SECURITY_WEP    
\sa wifi_sec_type
\def ARM_WIFI_SECURITY_WPA    
\sa wifi_sec_type
\def ARM_WIFI_SECURITY_WPA2   
\sa wifi_sec_type
\def ARM_WIFI_SECURITY_UNKNOWN
\sa wifi_sec_type
@}
*/

/**
\defgroup wifi_dhcp_mode WiFi DHCP Mode
\ingroup wifi_management_gr
\brief Specifies IPv6 Dynamic Host Configuration Protocol (DHCP) Mode.
\details
@{
\def ARM_WIFI_IP6_DHCP_OFF
\sa wifi_dhcp_mode
\def ARM_WIFI_IP6_DHCP_STATELESS
\sa wifi_dhcp_mode
\def ARM_WIFI_IP6_DHCP_STATEFULL
\sa wifi_dhcp_mode
@}
*/

int32_t ARM_WIFI_SetOption (uint32_t option, const void *data, uint32_t len) {
  return ARM_DRIVER_OK;	
}
/**
  \fn            int32_t ARM_WIFI_SetOption (uint32_t option, const void *data, uint32_t len)
\details
 
Example:
\code
\endcode
*/

int32_t ARM_WIFI_GetOption (uint32_t option, void *data, uint32_t *len) {
  return ARM_DRIVER_OK;	
}
/**
  \fn            int32_t ARM_WIFI_GetOption (uint32_t option, void *data, uint32_t *len)
\details
 
Example:
\code
\endcode
*/

int32_t ARM_WIFI_Scan (ARM_WIFI_AP_INFO_t ap_info[], uint32_t max_num) {
  return ARM_DRIVER_OK;	
}
/**
  \fn            int32_t ARM_WIFI_Scan (ARM_WIFI_AP_INFO_t ap_info[], uint32_t max_num)
\details
 
Example:
\code
\endcode
*/

int32_t ARM_WIFI_Connect (const char *ssid, const char *pass, uint8_t security, uint8_t ch) {
  return ARM_DRIVER_OK;	
}
/**
  \fn            int32_t ARM_WIFI_Connect (const char *ssid, const char *pass, uint8_t security, uint8_t ch)
\details
 
Example:
\code
\endcode
*/

int32_t ARM_WIFI_ConnectWPS (const char *pin) {
  return ARM_DRIVER_OK;	
}
/**
  \fn            int32_t ARM_WIFI_ConnectWPS (const char *pin)
\details
 
Example:
\code
\endcode
*/

int32_t ARM_WIFI_Disconnect (void) {
  return ARM_DRIVER_OK;	
}
/**
  \fn            int32_t ARM_WIFI_Disconnect (void)
\details
 
Example:
\code
\endcode
*/

int32_t ARM_WIFI_IsConnected (void) {
  return 0;	
}
/**
  \fn            int32_t ARM_WIFI_IsConnected (void)
\details
 
Example:
\code
\endcode
*/

int32_t ARM_WIFI_AP_Start (const char *ssid, const char *pass, uint8_t security, uint8_t ch) {
  return ARM_DRIVER_OK;	
}
/**
  \fn            int32_t ARM_WIFI_AP_Start (const char *ssid, const char *pass, uint8_t security, uint8_t ch)
\details
 
Example:
\code
\endcode
*/

int32_t ARM_WIFI_AP_Stop (void) {
  return ARM_DRIVER_OK;	
}
/**
  \fn            int32_t ARM_WIFI_AP_Stop (void)
\details
 
Example:
\code
\endcode
*/

int32_t ARM_WIFI_AP_IsRunning (void) {
  return 0;	
}
/**
  \fn            int32_t ARM_WIFI_AP_IsRunning (void)
\details
 
Example:
\code
\endcode
*/

/**
@}
*/

/**
\defgroup wifi_bypass_gr WiFi Bypass
\ingroup wifi_interface_gr
\brief Transfer Ethernet frames by WiFi module.
\details The \ref wifi_bypass_gr functions are and optional interface and allow to transfer Ethernet frames by WiFi module. Using this interface requires that the IP stack is running on microcontroller.
@{
*/

int32_t ARM_WIFI_BypassControl (uint32_t enable) {
  return ARM_DRIVER_OK;	
}
/**
  \fn            int32_t ARM_WIFI_BypassControl (uint32_t enable)
\details
 
Example:
\code
\endcode
*/

int32_t ARM_WIFI_EthSendFrame (const uint8_t *frame, uint32_t len) {
  return ARM_DRIVER_OK;	
}
/**
  \fn            int32_t ARM_WIFI_EthSendFrame (const uint8_t *frame, uint32_t len)
\details
 
Example:
\code
\endcode
*/

int32_t ARM_WIFI_EthReadFrame (uint8_t *frame, uint32_t len) {
  return ARM_DRIVER_OK;	
}
/**
  \fn            int32_t ARM_WIFI_EthReadFrame (uint8_t *frame, uint32_t len)
\details
 
Example:
\code
\endcode
*/

uint32_t ARM_WIFI_EthGetRxFrameSize (void) {
  return 0;	
}
/**
  \fn            uint32_t ARM_WIFI_EthGetRxFrameSize (void)
\details
 
Example:
\code
\endcode
*/
/**
@}  // wifi_control_gr
*/

/**
\defgroup wifi_socket_gr WiFi Socket
\ingroup wifi_interface_gr
\brief Socket interface to IP stack running on WiFi module
\details The \ref wifi_socket_gr functions provide the interface to an IP stack that is running on WiFi module. This IP stack handles data communication with the network.
@{
*/


/**
\defgroup wifi_addr_family WiFi Socket Address Family definitions
\ingroup wifi_socket_gr
\brief WiFi Socket Address Family definitions.
\details
@{
\def ARM_SOCKET_AF_INET
\def ARM_SOCKET_AF_INET6
@}
*/

/**
\defgroup wifi_socket_type WiFi Socket Type definitions
\ingroup wifi_socket_gr
\brief WiFi Socket Type definitions.
\details
@{
\def ARM_SOCKET_SOCK_STREAM
\def ARM_SOCKET_SOCK_DGRAM
@}
*/

/**
\defgroup wifi_protocol WiFi Socket Protocol definitions
\ingroup WiFi_socket_gr
\brief WiFi Socket Protocol definitions.
\details
@{
\def ARM_SOCKET_IPPROTO_TCP
\def ARM_SOCKET_IPPROTO_UDP
@}
*/

/**
\defgroup wifi_soc_opt WiFi Socket Option definitions
\ingroup WiFi_socket_gr
\brief WiFi Socket Option definitions.
\details
@{
\def ARM_SOCKET_IO_FIONBIO
\sa wifi_soc_opt
\def ARM_SOCKET_SO_RCVTIMEO
\sa wifi_soc_opt
\def ARM_SOCKET_SO_SNDTIMEO
\sa wifi_soc_opt
\def ARM_SOCKET_SO_KEEPALIVE
\sa wifi_soc_opt
\def ARM_SOCKET_SO_TYPE
\sa wifi_soc_opt
@}
*/

/**
\defgroup wifi_soc_func WiFi Socket Function return codes
\ingroup WiFi_socket_gr
\brief WiFi Socket Function return codes.
\details
@{
\def ARM_SOCKET_ERROR        
\sa wifi_soc_func
\def ARM_SOCKET_ESOCK        
\sa wifi_soc_func
\def ARM_SOCKET_EINVAL       
\sa wifi_soc_func
\def ARM_SOCKET_ENOTSUP      
\sa wifi_soc_func
\def ARM_SOCKET_ENOMEM       
\sa wifi_soc_func
\def ARM_SOCKET_EAGAIN       
\sa wifi_soc_func
\def ARM_SOCKET_EINPROGRESS  
\sa wifi_soc_func
\def ARM_SOCKET_ETIMEDOUT    
\sa wifi_soc_func
\def ARM_SOCKET_EISCONN      
\sa wifi_soc_func
\def ARM_SOCKET_ENOTCONN     
\sa WiFi_soc_func
\def ARM_SOCKET_ECONNREFUSED 
\sa WiFi_soc_func
\def ARM_SOCKET_ECONNRESET   
\sa WiFi_soc_func
\def ARM_SOCKET_ECONNABORTED 
\sa WiFi_soc_func
\def ARM_SOCKET_EALREADY     
\sa WiFi_soc_func
\def ARM_SOCKET_EADDRINUSE   
\sa WiFi_soc_func
\def ARM_SOCKET_EHOSTNOTFOUND
\sa WiFi_soc_func
@}
*/


int32_t ARM_WIFI_SocketCreate (int32_t af, int32_t type, int32_t protocol) {
  return 0;	
}
/**
  \fn            int32_t ARM_WIFI_SocketCreate (int32_t af, int32_t type, int32_t protocol)
\details
 
Example:
\code
\endcode
*/

int32_t ARM_WIFI_SocketBind (int32_t socket, const uint8_t *ip, uint32_t ip_len, uint16_t port) {
  return 0;	
}
/**
  \fn            int32_t ARM_WIFI_SocketBind (int32_t socket, const uint8_t *ip, uint32_t ip_len, uint16_t port)
\details
 
Example:
\code
\endcode
*/

int32_t ARM_WIFI_SocketListen (int32_t socket, int32_t backlog) {
  return 0;	
}
/**
  \fn            int32_t ARM_WIFI_SocketListen (int32_t socket, int32_t backlog)
\details
 
Example:
\code
\endcode
*/

int32_t ARM_WIFI_SocketAccept (int32_t socket, uint8_t *ip, uint32_t *ip_len, uint16_t *port) {
  return 0;	
}
/**
  \fn            int32_t ARM_WIFI_SocketAccept (int32_t socket, uint8_t *ip, uint32_t *ip_len, uint16_t *port)
\details
 
Example:
\code
\endcode
*/

int32_t ARM_WIFI_SocketConnect (int32_t socket, const uint8_t *ip, uint32_t ip_len, uint16_t port) {
  return 0;	
}
/**
  \fn            int32_t ARM_WIFI_SocketConnect (int32_t socket, const uint8_t *ip, uint32_t ip_len, uint16_t port)
\details
 
Example:
\code
\endcode
*/

int32_t ARM_WIFI_SocketRecv (int32_t socket, void *buf, uint32_t len) {
  return 1;	
}
/**
  \fn            int32_t ARM_WIFI_SocketRecv (int32_t socket, void *buf, uint32_t len)
\details
 
Example:
\code
\endcode
*/

int32_t ARM_WIFI_SocketRecvFrom (int32_t socket, void *buf, uint32_t len, uint8_t *ip, uint32_t *ip_len, uint16_t *port) {
  return 1;	
}
/**
  \fn            int32_t ARM_WIFI_SocketRecvFrom (int32_t socket, void *buf, uint32_t len, uint8_t *ip, uint32_t *ip_len, uint16_t *port)
\details
 
Example:
\code
\endcode
*/

int32_t ARM_WIFI_SocketSend (int32_t socket, const void *buf, uint32_t len) {
  return 1;	
}
/**
  \fn            int32_t ARM_WIFI_SocketSend (int32_t socket, const void *buf, uint32_t len)
\details
 
Example:
\code
\endcode
*/

int32_t ARM_WIFI_SocketSendTo (int32_t socket, const void *buf, uint32_t len, const uint8_t *ip, uint32_t ip_len, uint16_t port) {
  return 1;	
}
/**
  \fn            int32_t ARM_WIFI_SocketSendTo (int32_t socket, const void *buf, uint32_t len, const uint8_t *ip, uint32_t ip_len, uint16_t port)
\details
 
Example:
\code
\endcode
*/

int32_t ARM_WIFI_SocketGetSockName (int32_t socket, uint8_t *ip, uint32_t *ip_len, uint16_t *port) {
  return 0;	
}
/**
  \fn            int32_t ARM_WIFI_SocketGetSockName (int32_t socket, uint8_t *ip, uint32_t *ip_len, uint16_t *port)
\details
 
Example:
\code
\endcode
*/

int32_t ARM_WIFI_SocketGetPeerName (int32_t socket, uint8_t *ip, uint32_t *ip_len, uint16_t *port) {
  return 0;	
}
/**
  \fn            int32_t ARM_WIFI_SocketGetPeerName (int32_t socket, uint8_t *ip, uint32_t *ip_len, uint16_t *port)
\details
 
Example:
\code
\endcode
*/

int32_t ARM_WIFI_SocketGetOpt (int32_t socket, int32_t opt_id, void *opt_val, uint32_t *opt_len) {
  return 0;	
}
/**
  \fn            int32_t ARM_WIFI_SocketGetOpt (int32_t socket, int32_t opt_id, void *opt_val, uint32_t *opt_len)
\details
 
Example:
\code
\endcode
*/

int32_t ARM_WIFI_SocketSetOpt (int32_t socket, int32_t opt_id, const void *opt_val, uint32_t opt_len) {
  return 0;	
}
/**
\fn int32_t ARM_WIFI_SocketSetOpt (int32_t socket, int32_t opt_id, const void *opt_val, uint32_t opt_len)
\details
 
Example:
\code
\endcode
*/

int32_t ARM_WIFI_SocketClose (int32_t socket) {
  return 0;	
}
/**
\fn int32_t ARM_WIFI_SocketClose (int32_t socket)
\details
 
Example:
\code
\endcode
*/

int32_t ARM_WIFI_SocketGetHostByName (const char *name, int32_t af, uint8_t *ip, uint32_t *ip_len) {
  return 0;	
}
/**
\fn int32_t ARM_WIFI_SocketGetHostByName (const char *name, int32_t af, uint8_t *ip, uint32_t *ip_len)
\details
 
Example:
\code
\endcode
*/

int32_t ARM_WIFI_Ping (const uint8_t *ip, uint32_t ip_len) {
  return ARM_DRIVER_OK;	
}
/**
\fn int32_t ARM_WIFI_Ping (const uint8_t *ip, uint32_t ip_len)
\details
 
Example:
\code
\endcode
*/
/**
@} //wifi_socket_gr
*/



/**
@}
*/
// End WiFi Interface
