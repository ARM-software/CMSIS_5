#include "arm_math.h"
#include "arm_error.h"

/**
 * @brief  Error handler function used for post-mortem analysis and debug
 * @param[in]       error_code   Error code
 * @param[in, out]  error_desc   Error description
 * @return          none
 *
 */
void arm_error_handler(arm_error error_code, const char *error_desc)
{
  /* TODO */
  const char* desc;

  switch(error_code)
  {
    case ARM_ERROR_MATH:
    desc=error_desc;
    break;

    case ARM_ERROR_ALIGNMENT:
    desc=error_desc;
    break;

    default:
    desc=error_desc;
    break;
  }
}
