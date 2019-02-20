/**
\defgroup wifi_interface_gr WiFi Interface
\brief Driver API for WiFi (%Driver_WiFi.h)
\details 

<b>Block Diagram</b>


<b>WiFi API</b>

The following header files define the Application Programming Interface (API) for the WiFi interface:
  - \b %Driver_WiFi.h : Driver API for WiFi

The driver implementation is a typical part of the Device Family Pack (DFP) that supports the 
peripherals of the microcontroller family.


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

//
//  Functions
//

ARM_DRIVER_VERSION ARM_WIFI_GetVersion (void) {
  return { 0, 0 };	
}
/**
  \fn            ARM_DRIVER_VERSION ARM_WIFI_GetVersion (void)
\details
 
Example:
\code
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
  \fn            int32_t ARM_WIFI_SocketSetOpt (int32_t socket, int32_t opt_id, const void *opt_val, uint32_t opt_len)
\details
 
Example:
\code
\endcode
*/

int32_t ARM_WIFI_SocketClose (int32_t socket) {
  return 0;	
}
/**
  \fn            int32_t ARM_WIFI_SocketClose (int32_t socket)
\details
 
Example:
\code
\endcode
*/

int32_t ARM_WIFI_SocketGetHostByName (const char *name, int32_t af, uint8_t *ip, uint32_t *ip_len) {
  return 0;	
}
/**
  \fn            int32_t ARM_WIFI_SocketGetHostByName (const char *name, int32_t af, uint8_t *ip, uint32_t *ip_len)
\details
 
Example:
\code
\endcode
*/

int32_t ARM_WIFI_Ping (const uint8_t *ip, uint32_t ip_len) {
  return ARM_DRIVER_OK;	
}
/**
  \fn            int32_t ARM_WIFI_Ping (const uint8_t *ip, uint32_t ip_len)
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
 
Example:
\code
\endcode
*/


/**
\defgroup WiFi_option WiFi SetOption/GetOption Function Option Codes
\ingroup wifi_interface_gr
\brief Many parameters of the WiFi driver are configured using the SetOption/GetOption functions.
\details 
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
*/

/**
\defgroup wifi_sec_type WiFi Security Type
\ingroup WiFi_option
\brief Specifies WiFi security types.
\details
@{
\def ARM_WIFI_SECURITY_OPEN   
\sa WiFi_option                   
\def ARM_WIFI_SECURITY_WEP    
\sa WiFi_option                   
\def ARM_WIFI_SECURITY_WPA    
\sa WiFi_option                   
\def ARM_WIFI_SECURITY_WPA2   
\sa WiFi_option                   
\def ARM_WIFI_SECURITY_UNKNOWN
\sa WiFi_option                   
@}
*/

/**
\defgroup wifi_dhcp_mode WiFi DHCP Mode
\ingroup WiFi_option
\brief Specifies IPv6 Dynamic Host Configuration Protocol (DHCP) Mode.
\details
@{
\def ARM_WIFI_IP6_DHCP_OFF
\sa WiFi_option
\def ARM_WIFI_IP6_DHCP_STATELESS
\sa WiFi_option
\def ARM_WIFI_IP6_DHCP_STATEFULL
\sa WiFi_option
@}
*/

/**
\defgroup wifi_event WiFi Event
\ingroup WiFi_option
\brief Specifies WiFi events.
\details
@{
\def ARM_WIFI_EVENT_AP_CONNECT   
\sa WiFi_option                   
\def ARM_WIFI_EVENT_AP_DISCONNECT    
\sa WiFi_option                   
\def ARM_WIFI_EVENT_ETH_RX_FRAME    
\sa WiFi_option                   
@}
*/

/**
@}
*/
// end group WiFi_option 

/**
\defgroup WiFi_sockets WiFi Socket codes
\ingroup wifi_interface_gr
\brief Parameters of the WiFi Sockets.
\details 
@{
*/

/**
\defgroup wifi_addr_family WiFi Socket Address Family definitions
\ingroup WiFi_sockets
\brief WiFi Socket Address Family definitions.
\details
@{
\def ARM_SOCKET_AF_INET
\sa WiFi_sockets
\def ARM_SOCKET_AF_INET6
\sa WiFi_sockets
@}
*/

/**
\defgroup wifi_type WiFi Socket Type definitions
\ingroup WiFi_sockets
\brief WiFi Socket Type definitions.
\details
@{
\def ARM_SOCKET_SOCK_STREAM
\sa WiFi_sockets
\def ARM_SOCKET_SOCK_DGRAM
\sa WiFi_sockets
@}
*/

/**
\defgroup wifi_protocol WiFi Socket Protocol definitions
\ingroup WiFi_sockets
\brief WiFi Socket Protocol definitions.
\details
@{
\def ARM_SOCKET_IPPROTO_TCP
\sa WiFi_sockets
\def ARM_SOCKET_IPPROTO_UDP
\sa WiFi_sockets
@}
*/

/**
\defgroup wifi_soc_opt WiFi Socket Option definitions
\ingroup WiFi_sockets
\brief WiFi Socket Option definitions.
\details
@{
\def ARM_SOCKET_IO_FIONBIO
\sa WiFi_sockets
\def ARM_SOCKET_SO_RCVTIMEO
\sa WiFi_sockets
\def ARM_SOCKET_SO_SNDTIMEO
\sa WiFi_sockets
\def ARM_SOCKET_SO_KEEPALIVE
\sa WiFi_sockets
\def ARM_SOCKET_SO_TYPE
\sa WiFi_sockets
@}
*/

/**
\defgroup wifi_soc_func WiFi Socket Function return codes
\ingroup WiFi_sockets
\brief WiFi Socket Function return codes.
\details
@{
\def ARM_SOCKET_ERROR        
\sa WiFi_sockets
\def ARM_SOCKET_ESOCK        
\sa WiFi_sockets
\def ARM_SOCKET_EINVAL       
\sa WiFi_sockets
\def ARM_SOCKET_ENOTSUP      
\sa WiFi_sockets
\def ARM_SOCKET_ENOMEM       
\sa WiFi_sockets
\def ARM_SOCKET_EAGAIN       
\sa WiFi_sockets
\def ARM_SOCKET_EINPROGRESS  
\sa WiFi_sockets
\def ARM_SOCKET_ETIMEDOUT    
\sa WiFi_sockets
\def ARM_SOCKET_EISCONN      
\sa WiFi_sockets
\def ARM_SOCKET_ENOTCONN     
\sa WiFi_sockets
\sa WiFi_sockets
\sa WiFi_sockets
\def ARM_SOCKET_ECONNREFUSED 
\sa WiFi_sockets
\def ARM_SOCKET_ECONNRESET   
\sa WiFi_sockets
\def ARM_SOCKET_ECONNABORTED 
\sa WiFi_sockets
\def ARM_SOCKET_EALREADY     
\sa WiFi_sockets
\sa WiFi_sockets
\def ARM_SOCKET_EADDRINUSE   
\sa WiFi_sockets
\def ARM_SOCKET_EHOSTNOTFOUND
\sa WiFi_sockets
@}
*/

/**
@}
*/
// end group WiFi_sockets 

/**
@}
*/
// End WiFi Interface
