/* -----------------------------------------------------------------------------
 * Copyright (c) 2013-2016 ARM Limited. All rights reserved.
 *
 * $Date:        7. March 2016
 * $Revision:    V1.00
 *
 * Project:      Storage Driver API
 * -------------------------------------------------------------------------- */


/**
\defgroup storage_interface_gr Storage Interface
\brief    Driver API for Storage Device Interface (%Driver_Storage.h)
\details
<a href="http://en.wikipedia.org/wiki/Flash_memory" target="_blank">Flash devices</a> based on NOR memory cells are the
preferred technology for embedded applications requiring a discrete non-volatile memory device. The low read latency
characteristic of these Flash devices allow a direct code execution
(<a href="http://en.wikipedia.org/wiki/Execute_in_place" target="_blank">XIP</a>) and data storage in a single memory
product.

**Storage API**

The \b Storage \b API provides a generic API suitable for Storage devices with NOR memory cells independent from the actual interface
to the MCU (memory bus, SPI, ...). <a href="http://en.wikipedia.org/wiki/Flash_memory#Serial_flash" target="_blank">SPI</a>
flashes are typically not named NOR flashes but have usually same flash cell properties.

The following header files define the Application Programming Interface (API) for the Flash interface:
  - \b %Driver_Storage.h : Driver API for Storage Device Interface


**Driver Functions**

The driver functions are published in the access struct as explained in \ref DriverFunctions
  - \ref ARM_DRIVER_STOR : access struct for Storage driver functions

@{
*/
/*
\todo provide more text for the driver implementation above

A typical setup sequence for the driver is shown below:

<b>Example Code:</b>

\todo example
*******************************************************************************************************************/


/**
\defgroup Storage_events Storage Events
\ingroup storage_interface_gr
\brief The Storage driver generates call back events that are notified via the function \ref ARM_Storage_SignalEvent.
\details
This section provides the event values for the \ref ARM_Storage_SignalEvent callback function.

The following call back notification events are generated:
@{
\def ARM_Storage_EVENT_READY
\def ARM_Storage_EVENT_ERROR
@}
*/


/**
\struct     ARM_Storage_SECTOR
\details
Specifies sector start and end address.

<b>Element of</b>:
  - \ref ARM_Storage_INFO structure
*******************************************************************************************************************/

/**
\struct     ARM_Storage_INFO
\details
Stores the characteristics of a Flash device. This includes sector layout, programming size and a default value for erased memory.
This information can be obtained from the Flash device datasheet and is used by the middleware in order to properly interact with the Flash device.

Sector layout is described by specifying the \em sector_info which points to an array of sector information (start and end address) and by specifying the \em sector_count which defines the number of sectors.
The element \em sector_size is not used in this case and needs to be \em 0.
Flash sectors need not to be aligned continuously. Gaps are allowed in the device memory space in order to reserve sectors for other usage (for example application code).

When the device has uniform sector size than the sector layout can be described by specifying the \em sector_size which defines the size of a single sector and by specifying the \em sector_count which defines the number of sectors.
The element \em sector_info is not used in this case and needs to be \em NULL.

The smallest programmable unit within a sector is specified by the \em program_unit. It defines the granularity for programming data.

Optimal programming page size is specified by the \em page_size and defines the amount of data that should be programmed in one step to achieve maximum programming speed.

Contents of erased memory is specified by the \em erased_value and is typically \em 0xFF. This value can be used before erasing a sector to check if the sector is blank and erase can be skipped.

*******************************************************************************************************************/

/**
\struct     ARM_DRIVER_STOR
\details
The functions of the Flash driver are accessed by function pointers exposed by this structure. Refer to \ref DriverFunctions for overview information.

Each instance of a Flash interface provides such an access structure.
The instance is identified by a postfix number in the symbol name of the access structure, for example:
 - \b Driver_Flash0 is the name of the access struct of the first instance (no. 0).
 - \b Driver_Flash1 is the name of the access struct of the second instance (no. 1).

A middleware configuration setting allows connecting the middleware to a specific driver instance \b %Driver_Flash<i>n</i>.
The default is \token{0}, which connects a middleware to the first instance of a driver.
*******************************************************************************************************************/

/**
\struct     ARM_Storage_CAPABILITIES
\details
A Flash driver can be implemented with different capabilities. The data fields of this struct encode
the capabilities implemented by this driver.

The element \em event_ready indicates that the driver is able to generate the \ref ARM_Storage_EVENT_READY event. In case that this event is not available it is possible to poll the driver status by calling the \ref ARM_Storage_GetStatus and check the \em busy flag.

The element \em data_width specifies the data access size and also defines the data type (uint8_t, uint16_t or uint32_t) for the \em data parameter in \ref ARM_Storage_ReadData and \ref ARM_Storage_ProgramData functions.

The element \em erase_chip specifies that the \ref ARM_Storage_EraseChip function is supported. Typically full chip erase is much faster than erasing the whole device sector per sector.

<b>Returned by:</b>
  - \ref ARM_Storage_GetCapabilities
*******************************************************************************************************************/

/**
\struct     ARM_Storage_STATUS
\details
Structure with information about the status of the Flash.

The flag \em busy indicates that the driver is busy executing read/program/erase operation.

The flag \em error flag is cleared on start of read/program/erase operation and is set at the end of the current operation in case of error.

<b>Returned by:</b>
  - \ref ARM_Storage_GetStatus
*****************************************************************************************************************/

/**
\typedef    ARM_Storage_SignalEvent_t
\details
Provides the typedef for the callback function \ref ARM_Storage_SignalEvent.

<b>Parameter for:</b>
  - \ref ARM_Storage_Initialize
*******************************************************************************************************************/


//
// Functions
//

ARM_DRIVER_VERSION ARM_Storage_GetVersion (void)  {
  return { 0, 0 };
}
/**
 * \fn ARM_DRIVER_VERSION ARM_Storage_GetVersion (void)
 * \brief Get driver version.
 * \details
 * The function \b ARM_Storage_GetVersion returns version information of the driver implementation in \ref ARM_DRIVER_VERSION
 *  - API version is the version of the CMSIS-Driver specification used to implement this driver.
 *  - Driver version is source code version of the actual driver implementation.
 *
 * Example:
 * \code
 *     extern ARM_DRIVER_STOR *drv_info;
 *
 *     void read_version (void)  {
 *       ARM_DRIVER_VERSION  version;
 *
 *       version = drv_info->GetVersion ();
 *       if (version.api < 0x10A)   {      // requires at minimum API version 1.10 or higher
 *         // error handling
 *         return;
 *       }
 *     }
 * \endcode
 *
 * \note This API returns synchronously--it does not result in an invocation
 *     of a completion callback.
 *
 * \note The functions GetVersion() can be called any time to obtain the
 *     required information from the driver (even before initialization). It
 *     always returns the same information.
 *
 *******************************************************************************************************************/

ARM_STOR_CAPABILITIES ARM_Storage_GetCapabilities (void)  {
  return { 0 };
}
/**
 * \fn ARM_STOR_CAPABILITIES ARM_Storage_GetCapabilities (void)
 * \brief Get device capabilities.
 *
 * \details The function GetCapabilities() returns information about
 * capabilities in this driver implementation. The data fields of the struct
 * ARM_STOR_CAPABILITIES encode various capabilities, for example if the device
 * is able to execute operations asynchronously.
 *
 * Example:
 * \code
 *     extern ARM_DRIVER_STOR *drv_info;
 *
 *     void read_capabilities (void)  {
 *       ARM_STOR_CAPABILITIES drv_capabilities;
 *
 *       drv_capabilities = drv_info->GetCapabilities ();
 *       // interrogate capabilities
 *
 *     }
 * \endcode
 *
 * @return \ref ARM_STOR_CAPABILITIES.
 *
 * \note This API returns synchronously--it does not result in an invocation
 *     of a completion callback.
 *
 * \note The functions GetCapabilities() can be called any time to obtain the
 *     required information from the driver (even before initialization). It
 *     always returns the same information.
 *
 *******************************************************************************************************************/

int32_t ARM_Storage_Initialize (ARM_Storage_Callback_t callback)  {
  return 0;
}
/**
 * \fn int32_t ARM_Storage_Initialize (ARM_Storage_Callback_t callback)
 * \details
 * The function \b ARM_Storage_Initialize initializes the Storage interface.
 * It is called when the middleware component starts operation.
 *
 * Initialize() needs to be called explicitly before
 * powering the peripheral using PowerControl(), and before initiating other
 * accesses to the storage controller.
 *
 * The function performs the following operations:
 *   - Initializes the resources needed for the Storage interface.
 *   - Registers the \ref ARM_Storage_Callback_t callback function.
 *
 * The parameter \em callback is a pointer to the \ref ARM_Storage_Callback_t
 * callback function to be invoked upon command completion for asynchronous APIs
 * (including the completion of initialization); use a NULL pointer when no
 * callback signals are required.
 *
 * \b Example:
 *  - see \ref storage_interface_gr - Driver Functions
 *
 * \return
 *   The function executes in the following ways:
 *   - When the operation is non-blocking (asynchronous)--i.e. when
 *     ARM_STOR_CAPABILITIES::asynchronous_ops is set,--the function
 *     only starts the operation and returns with ARM_DRIVER_OK (or an
 *     appropriate error code in case of failure). When the operation is
 *     completed the command callback is invoked. In case of errors, the
 *     completion callback is invoked with an error status. Progress of the
 *     operation can also be monitored by calling GetStatus() and checking
 *     the busy or error flags.
 *   - When the operation is blocking (synchronous)--i.e. when
 *     ARM_STOR_CAPABILITIES::asynchronous_ops is not set,--the function
 *     returns after initialization completes with either ARM_DRIVER_OK
 *     (successful completion) or an error code.
 *   - When the operation can be finished synchronously in spite of
 *     ARM_STOR_CAPABILITIES::asynchronous_ops being set, a positive value is
 *     returned to indicate successful completion--in this case no further
 *     invocation of completion callback should be expected at a later time.
 *
 * \note This operation can execute asynchronously if
 *     ARM_STOR_CAPABILITIES::asynchronous_ops is set; in which case the
 *     invocation returns quickly and results in a completion callback later.
 *
 *******************************************************************************************************************/

int32_t ARM_Storage_Uninitialize (void)  {
  return 0;
}
/**
 * \fn int32_t ARM_Storage_Uninitialize (void)
 * \brief De-initialize the Storage Interface.
 *
 * \details The function Uninitialize() de-initializes the resources of Storage interface.
 *
 * It is called when the middleware component stops operation, and wishes to
 * release the software resources used by the interface.
 *
 * \return
 *   The function executes in the following ways:
 *   - When the operation is non-blocking (asynchronous)--i.e. when
 *     ARM_STOR_CAPABILITIES::asynchronous_ops is set,--the function only
 *     starts the operation and returns with ARM_DRIVER_OK (or an appropriate
 *     error code in case of failure). When the operation is completed, the
 *     command callback is invoked. In case of errors, the completion callback
 *     is invoked with an error status. Progress of the operation can also be
 *     monitored by calling GetStatus() and checking the busy or error flags.
 *   - When the operation is blocking (synchronous)--i.e. when
 *     ARM_STOR_CAPABILITIES::asynchronous_ops is not set,--the function
 *     returns after un-initialization completes with either ARM_DRIVER_OK
 *     (successful completion) or an error code.
 *   - When the operation can be finished synchronously in spite of
 *     ARM_STOR_CAPABILITIES::asynchronous_ops being set, a positive value is
 *     returned to indicate successful completion--in this case no further
 *     invocation of completion callback should be expected at a later time.
 *
 * \note This operation can execute asynchronously if
 *     ARM_STOR_CAPABILITIES::asynchronous_ops is set; in which case the
 *     invocation returns quickly and results in a completion callback later.
 *
 *******************************************************************************************************************/

int32_t ARM_Storage_PowerControl (ARM_POWER_STATE state)  {
  return 0;
}
/**
 * \fn int32_t ARM_Storage_PowerControl (ARM_POWER_STATE state)
 * \brief Control the Storage interface power.
 *
 * \details The function ARM_STOR_PowerControl operates the power modes of the
 * Storage interface.
 *
 * The parameter \em state can have the following values:
 *   - \ref ARM_POWER_FULL : set-up peripheral for data transfers, enable interrupts (NVIC) and optionally DMA. Can be called multiple times.
 *                           If the peripheral is already in this mode, then the function performs no operation and returns with \ref ARM_DRIVER_OK.
 *   - \ref ARM_POWER_LOW : may use power saving. Returns \ref ARM_DRIVER_ERROR_UNSUPPORTED when not implemented.
 *   - \ref ARM_POWER_OFF : terminates any pending data transfers, disables peripheral, disables related interrupts and DMA.
 *
 * @return
 *   The function executes in the following ways:
 *   - When the operation is non-blocking (asynchronous)--i.e. when
 *     ARM_STOR_CAPABILITIES::asynchronous_ops is set,--the function only
 *     starts the operation and returns with ARM_DRIVER_OK (or an appropriate
 *     error code in case of failure). When the operation is completed, the
 *     command callback is invoked. In case of errors, the appropriate
 *     callback is invoked with an error status. Progress of the operation can
 *     also be monitored by calling GetStatus() and checking the busy or error
 *     flags.
 *   - When the operation is blocking (synchronous)--i.e. when
 *     ARM_STOR_CAPABILITIES::asynchronous_ops is not set,--the function
 *     returns with either ARM_DRIVER_OK (successful completion) or an error
 *     code.
 *   - When the operation can be finished synchronously in spite of
 *     ARM_STOR_CAPABILITIES::asynchronous_ops being set, a positive value is
 *     returned to indicate successful completion--in this case no further
 *     invocation of completion callback should be expected at a later time.
 *
 * \note This operation can execute asynchronously if
 *     ARM_STOR_CAPABILITIES::asynchronous_ops is set; in which case the
 *     invocation returns quickly and results in a completion callback later.
 *
 * Refer to \ref CallSequence for more information.
 *******************************************************************************************************************/

int32_t ARM_Storage_ReadData (uint32_t addr, void *data, uint32_t cnt)  {
  return 0;
}
/**
 * \fn int32_t ARM_Storage_ReadData (uint32_t addr, void *data, uint32_t cnt)
 * \brief read the contents of a given address range from the storage device.
 *
 * \details Read the contents of a range of storage memory into a buffer
 *   supplied by the caller. The buffer is owned by the caller and should
 *   remain accessible for the lifetime of this command.
 *
 * \param  [in] addr
 *                This specifies the address from where to read data. It needs
 *                to be aligned to data type size--i.e. the offset of this
 *                start-address within the containing storage block must be a
 *                multiple of the granularity determined by the 'data_width'
 *                member of ARM_STOR_CAPABILITIES (0=8-bit, 1=16-bit, 2=32-bit).
 *
 * \param [out] data
 *                The destination of the read operation. The data type is uint8_t,
 *                uint16_t or uint32_t, and is specified by the data_width in
 *                ARM_STOR_CAPABILITIES: (0=8-bit, 1=16-bit, 2=32-bit). The buffer
 *                is owned by the caller and should remain accessible for the
 *                lifetime of this command.
 *
 * \param  [in] cnt
 *                The number of data items requested to read. The units for this
 *                count are determined by the 'data_width' member of
 *                ARM_STOR_CAPABILITIES: (0=8-bit, 1=16-bit, 2=32-bit). The
 *                data buffer should be at least as large as this size.
 *
 * @return
 *    The function executes in the following ways:
 *   - When the operation is non-blocking (asynchronous)--i.e. when
 *     ARM_STOR_CAPABILITIES::asynchronous_ops is set,--the function only
 *     starts the operation and returns with ARM_DRIVER_OK--i.e. zero number
 *     of data items read--or an appropriate error code in case of failure.
 *     When the operation is completed, the command callback is invoked with
 *     the number of successfully transferred data items passed in as
 *     'status'. In case of errors the completion callback is invoked with an
 *     error status. Progress of the operation can also be monitored by
 *     calling GetStatus() and checking the busy or error flags.
 *   - When the operation is blocking (typical for memory mapped storage)--i.e.
 *     when ARM_STOR_CAPABILITIES::asynchronous_ops is not set,--the
 *     function returns after the data is read, and returns the number of data
 *     items read or an appropriate error code.
 *   - When the operation can be finished synchronously in spite of
 *     ARM_STOR_CAPABILITIES::asynchronous_ops being set, the invocation
 *     returns the number of data items read to indicate successful
 *     completion, or an appropriate error code--in this case, no further
 *     invocation of completion callback should be expected at a later time.
 *
 * \note This operation can execute asynchronously if
 *     ARM_STOR_CAPABILITIES::asynchronous_ops is set; in which case the
 *     invocation returns quickly and results in a completion callback later.
 *
 *******************************************************************************************************************/

int32_t ARM_Storage_ProgramData (uint32_t addr, const void *data, uint32_t cnt)  {
  return 0;
}
/**
 * \fn int32_t ARM_Storage_ProgramData (uint32_t addr, const void *data, uint32_t cnt)
 * \brief program (write into) the contents of a given address range of the storage device.
 *
 * \details Write the contents of a given memory buffer into a range of
 *   storage memory. In the case of flash memory, the destination range in
 *   storage memory typically has its contents in an erased state from a
 *   preceding erase operation. The source memory buffer is owned by the
 *   caller and should remain accessible for the lifetime of this command.
 *
 * \param [in] addr
 *               This is the start address of the range to be written into. It
 *               needs to be aligned to \em program_unit specified in the
 *               attributes of the underlying storage-block--i.e. in \ref
 *               ARM_STOR_BLOCK_ATTRIBUTES of the \ref ARM_STOR_BLOCK
 *               encompassing 'addr'.
 *
 * \param [in] data
 *               The source of the write operation. The data type is uint8_t,
 *               uint16_t or uint32_t, as specified by the 'data_width' in
 *               ARM_STOR_CAPABILITIES: (0=8-bit, 1=16-bit, 2=32-bit). The buffer
 *               is owned by the caller and should remain accessible for the
 *               lifetime of this command.
 *
 * \param [in] cnt
 *               The number of data items requested to be written. The units
 *               for this count are determined by the 'data_width' member of
 *               ARM_STOR_CAPABILITIES: (0=8-bit, 1=16-bit, 2=32-bit). The
 *               buffer should be at least as large as this size. \note It is
 *               best for the middleware to write in units of
 *               'optimal_program_unit' (\ref ARM_STOR_BLOCK_ATTRIBUTES) of
 *               the underlying block (\ref ARM_STOR_BLOCK). Writing in
 *               amounts larger than 'optimal_program_unit' may not be
 *               supported.
 *
 * \note It may not be safe or permissible for a write to straddle an erase
 * boundary. Refer to 'erase_unit' within \ref ARM_STOR_BLOCK_ATTRIBUTES.
 *
 * \return
 *    The function executes in the following ways:
 *   - When the operation is non-blocking (asynchronous)--i.e. when
 *     ARM_STOR_CAPABILITIES::asynchronous_ops is set,--the function only
 *     starts the operation and returns with ARM_DRIVER_OK--i.e. zero number
 *     of data items written--or an appropriate error code in case of failure.
 *     When the operation is completed, the command callback is invoked with
 *     the number of successfully transferred data items passed in as
 *     'status'. In case of errors the completion callback is invoked with an
 *     error status. Progress of the operation can also be monitored by calling
 *     GetStatus() and checking the busy or error flags.
 *   - When the operation is blocking (typical for memory mapped storage)--i.e.
 *     when ARM_STOR_CAPABILITIES::asynchronous_ops is not set,--the
 *     function returns after the data is programmed, and returns the number of
 *     data items programmed or an appropriate error code.
 *   - When the operation can be finished synchronously in spite of
 *     ARM_STOR_CAPABILITIES::asynchronous_ops being set, the invocation
 *     returns the number of data items programmed to indicate successful
 *     completion, or an appropriate error code--in this case no further
 *     invocation of completion callback should be expected at a later time.
 *
 * \note This operation can execute asynchronously if
 *     ARM_STOR_CAPABILITIES::asynchronous_ops is set; in which case the
 *     invocation returns quickly and results in a completion callback later.
 *
 *******************************************************************************************************************/

int32_t ARM_Storage_EraseSector (uint32_t addr)  {
  return 0;
}
/**
 * \fn int32_t ARM_Storage_EraseSector (uint32_t addr)
 * \brief Erase Storage Sector.
 *
 * \details This function erases a storage sector specified by the
 * parameter addr (points to start of the sector). A sector corresponds to the
 * 'erase_unit' of the owning storage block (\ref ARM_STOR_BLOCK). The range
 * to be erased will have its contents returned to the un-programmed state--
 * i.e. to 'erased_value' within \ref ARM_STOR_BLOCK_ATTRIBUTES, which is
 * usually all 1s.
 *
 * \param [in] addr
 *               This is the start-address of the sector to be erased. The
 *               offset of this start-address within the containing storage
 *               block (\ref ARM_STOR_BLOCK) must be a multiple of the
 *               'erase_unit'.
 *
 * \return
 *    The function executes in the following ways:
 *   - When the operation is non-blocking (asynchronous)--i.e. when
 *     ARM_STOR_CAPABILITIES::asynchronous_ops is set,--the function only
 *     starts the operation and returns with ARM_DRIVER_OK, or an appropriate
 *     error code in case of failure. When the operation is completed, the
 *     command callback is invoked. In case of errors the completion callback is
 *     invoked with an error status. Progress of the operation can also be
 *     monitored by calling GetStatus() and checking the busy or error flags.
 *   - When the operation is blocking (typical for memory mapped storage)--i.e.
 *     when ARM_STOR_CAPABILITIES::asynchronous_ops is not set,--the
 *     function returns either ARM_DRIVER_OK or an error code.
 *   - When the operation can be finished synchronously in spite of
 *     ARM_STOR_CAPABILITIES::asynchronous_ops being set, a positive value is
 *     returned to indicate successful completion--in this case no further
 *     invocation of completion callback should be expected at a later time.
 *
 * \note This operation can execute asynchronously if
 *     ARM_STOR_CAPABILITIES::asynchronous_ops is set; in which case the
 *     invocation returns quickly and results in a completion callback later.
 *
 *******************************************************************************************************************/

int32_t ARM_Storage_EraseChip (void)  {
  return 0;
}
/**
 * \fn int32_t ARM_Storage_EraseChip (void)
 * \brief Erase complete storage. Optional function for faster erase of the complete device.
 *
 * This optional function erases the complete device. If the device does not
 *    support global erase then the function returns the error value \ref
 *    ARM_DRIVER_ERROR_UNSUPPORTED. The data field \em 'erase_chip' =
 *    \token{1} of the structure \ref ARM_STOR_CAPABILITIES encodes that
 *    \ref ARM_STOR_EraseChip is supported.
 *
 * \return
 *    The function executes in the following ways:
 *   - When the operation is non-blocking (asynchronous)--i.e. when
 *     ARM_STOR_CAPABILITIES::asynchronous_ops is set,--the function only
 *     starts the operation and returns with ARM_DRIVER_OK or an appropriate
 *     error code in case of failure. When the operation is completed, the
 *     command callback is invoked. In case of errors the completion callback is
 *     invoked with an error status. Progress of the operation can also be
 *     monitored by calling GetStatus() and checking the busy or error flags.
 *   - When the operation is blocking (typical for memory mapped storage)--i.e.
 *     when ARM_STOR_CAPABILITIES::asynchronous_ops is not set,--the
 *     function returns either ARM_DRIVER_OK or an error code.
 *   - When the operation can be finished synchronously in spite of
 *     ARM_STOR_CAPABILITIES::asynchronous_ops being set, a positive value is
 *     returned to indicate successful completion--in this case no further
 *     invocation of completion callback should be expected at a later time.
 *
 * \note This operation can execute asynchronously if
 *     ARM_STOR_CAPABILITIES::asynchronous_ops is set; in which case the
 *     invocation returns quickly and results in a completion callback later.
 *
 * <b>See also:</b>
 *  - ARM_Storage_Callback_t
 *
 *******************************************************************************************************************/

ARM_Storage_STATUS ARM_Storage_GetStatus (void)  {
  return 0;
}
/**
 * \fn ARM_Storage_STATUS ARM_Storage_GetStatus (void)
 * \brief Get the status of the current (or previous) command executed by the
 *     storage controller; stored in the structure \ref ARM_STOR_STATUS.
 *
 * \return
 *          The status of the underlying controller.
 *
 * \note This API returns synchronously--it does not result in an invocation
 *     of a completion callback.
 *
 *******************************************************************************************************************/

void ARM_Storage_GetInfo (ARM_Storage_INFO *info)  {
  return;
}
/**
 * \fn void ARM_Storage_GetInfo (ARM_Storage_INFO *info)
 * \brief Get information about the Storage device; stored in the structure \ref ARM_STOR_INFO.
 *
 * @param [out] info
 *                A caller-supplied buffer capable of being filled in with an
 *                \ref ARM_STOR_INFO.
 *
 * @return ARM_DRIVER_OK if a ARM_STOR_INFO structure containing top level
 *         metadata about the storage controller is filled into the supplied
 *         buffer, else an appropriate error value.
 *
 * @note It is the caller's responsibility to ensure that the buffer passed in
 *         is able to be initialized with a \ref ARM_STOR_INFO.
 *
 * \note This API returns synchronously--it does not result in an invocation
 *     of a completion callback.
 *
 *******************************************************************************************************************/

uint32_t ARM_Storage_ResolveAddress(uint64_t addr) {
  return 0;
}
/**
 * \fn uint32_t ARM_Storage_ResolveAddress (uint64_t addr)
 *
 * \brief For memory-mapped storage, this function resolves an address managed
 *     by the storage controller into a memory address within the processor's
 *     address space.
 *
 * \param [in] addr
 *               This is the address for which we want a resolution to the
 *               processor's physical address space.
 *
 * @return
 *          The resolved address in the processor's address space; else if no
 *          resolution is possible return \ref ARM_STOR_INVALID_RESOLVED_ADDRESS.
 *
 * \note This API returns synchronously--it does not result in an invocation
 *     of a completion callback.
 *
 *******************************************************************************************************************/

int32_t ARM_Storage_FirstBlock(ARM_STOR_BLOCK *firstBlockP) {
  return 0;
}
/**
 * \fn int32_t ARM_Storage_FirstBlock (ARM_STOR_BLOCK *firstBlockP)
 * \brief Fetch an iterator to the first storage-block.
 *
 * \details This helper function fetches the first (of the potentially
 *     multiple) block(s) making up the storage space managed by the storage
 *     controller. In combination with \ref ARM_Storage_NextBlock() and \ref
 *     ARM_STOR_VALID_BLOCK(), it can be used to iterate over the sequence of
 *     blocks within the storage map:
 *
 * \code
 *   ARM_STOR_BLOCK block;
 *   for (drv->FirstBlock(&block); ARM_STOR_VALID_BLOCK(&block); drv->NextBlock(&block, &block)) {
 *       // make use of block
 *   }
 * \endcode
 *
 * \param[out] firstBlockP
 *               A caller-owned buffer large enough to be filled in with the
 *               first ARM_STOR_BLOCK. This value can also be passed in as
 *               NULL if the caller isn't interested in populating a buffer
 *               with the block--i.e. if the caller only wishes to establish
 *               the presence of a first block.
 *
 * \note: It is the caller's responsibility for supplying the memory which
 * gets filled.
 *
 * \return ARM_DRIVER_OK if a valid first block is found; in this case, the
 *     contents of the first block are filled into the supplied buffer, and
 *     ARM_STOR_VALID_BLOCK(firstBlockP) would return true following
 *     this call. In the very unusual case where a storage controller has no
 *     blocks, or in case the driver is unable to fetch information about
 *     the first block, an error (negative) value is returned and an invalid
 *     StorageBlock is populated into the supplied buffer.
 *
 * \note This API returns synchronously--it does not result in an invocation
 *     of a completion callback.
 *
 *******************************************************************************************************************/

int32_t ARM_Storage_NextBlock(const ARM_STOR_BLOCK* prevP, ARM_STOR_BLOCK *nextP) {
  return 0;
}
/**
 * \fn int32_t ARM_Storage_NextBlock (const ARM_STOR_BLOCK* prevP, ARM_STOR_BLOCK *nextP)
 * \brief Advance to the successor of the current block (iterator).
 *
 * \details This helper function fetches (an iterator to) the next block,
 *     or else returns a terminating, invalid block iterator. In combination
 *     with \ref ARM_Storage_FirstBlock() and \ref ARM_STOR_VALID_BLOCK(), it
 *     can be used to iterate over the sequence of blocks within the storage
 *     map:
 *
 * \code
 *   ARM_STOR_BLOCK block;
 *   for (drv->FirstBlock(&block); ARM_STOR_VALID_BLOCK(&block); drv->NextBlock(&block, &block)) {
 *       // make use of block
 *   }
 * \endcode
 *
 * \param[in]  prevP
 *               An existing block (iterator) within the same storage
 *               controller. The memory buffer holding this block is owned
 *               by the caller. This buffer must not be NULL; if so, the call
 *               is invalid and ARM_DRIVER_ERROR_PARAMETER will be returned.
 *
 * \param[out] nextP
 *               A caller-owned buffer large enough to be filled in with the
 *               the following ARM_STOR_BLOCK. It is legal to provide the
 *               same buffer using 'nextP' as was passed in with 'prevP'. It
 *               is also legal to pass a NULL into this parameter if the
 *               caller isn't interested in populating a buffer with the next
 *               block--i.e. if the caller only wishes to establish the
 *               presence of a next block.
 *
 * \return ARM_DRIVER_OK if a valid next block is found; in this case the
 *     contents of the next block are filled into the buffer pointed to by
 *     nextP; ARM_STOR_VALID_BLOCK(nextP) would return true following this
 *     call. Upon reaching the end of the sequence of blocks (iterators), or
 *     in case the driver is unable to fetch information about the next block,
 *     an error (negative) value is returned and an invalid StorageBlock is
 *     populated into the supplied buffer. If prevP is NULL,
 *     ARM_DRIVER_ERROR_PARAMETER will be returned.
 *
 * \note This API returns synchronously--it does not result in an invocation
 *     of a completion callback.
 *
 *******************************************************************************************************************/

int32_t ARM_Storage_GetBlockIterator(uint64_t offset, ARM_STOR_BLOCK *blockP) {
  return 0;
}
/**
 * \fn int32_t ARM_Storage_GetBlockIterator (uint64_t offset, ARM_STOR_BLOCK *blockP)
 *
 * \brief Find the storage block (iterator) encompassing a given storage address.
 *
 * \param[in]  offset
 *               Storage address in units of octets.
 *
 * \param[out] blockP
 *               A caller-owned buffer large enough to be filled in with the
 *               ARM_STOR_BLOCK encapsulating the given address. This value
 *               can also be passed in as NULL if the caller isn't interested
 *               in populating a buffer with the block--if the caller only
 *               wishes to establish the presence of a containing storage
 *               block.
 *
 * \return ARM_DRIVER_OK if a containing storage-block is found. In this case,
 *     if blockP is non-NULL, the buffer pointed to by it is populated with
 *     the contents of the storage block--i.e. if blockP is valid and a block is
 *     found, ARM_STOR_VALID_BLOCK(blockP) would return true following this
 *     call. If there is no storage block containing the given offset, or in
 *     case the driver is unable to resolve an address to a storage-block, an
 *     error (negative) value is returned and an invalid StorageBlock is
 *     populated into the supplied buffer.
 *
 * \note This API returns synchronously--it does not result in an invocation
 *     of a completion callback.
 *
 *******************************************************************************************************************/

void ARM_Storage_SignalEvent (uint32_t event)  {
  return 0;
}
/**
\fn void ARM_Storage_SignalEvent (uint32_t event)
\details

The function \b ARM_Storage_SignalEvent is a callback function registered by the function \ref ARM_Storage_Initialize.
The function is called automatically after read/program/erase operation completes.

The parameter \em event indicates one or more events that occurred during driver operation. Each event is coded in a separate bit and
therefore it is possible to signal multiple events in the event call back function.

Not every event is necessarily generated by the driver. This depends on the implemented capabilities stored in the
data fields of the structure \ref ARM_Storage_CAPABILITIES, which can be retrieved with the function \ref ARM_Storage_GetCapabilities.

The following events can be generated:

Parameter \em event                 | Bit | Description
:-----------------------------------|:---:|:-----------
\ref ARM_Storage_EVENT_READY          |  0  | Occurs after read/program/erase operation completes.
\ref ARM_Storage_EVENT_ERROR          |  1  | Occurs together with \ref ARM_Storage_EVENT_READY when operation completes with errors.

<b>See also:</b>
 - \ref ARM_Storage_EraseChip
*******************************************************************************************************************/

/**
@}
*/
// End Flash Interface
