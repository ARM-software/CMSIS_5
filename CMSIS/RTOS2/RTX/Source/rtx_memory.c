/*
 * Copyright (c) 2013-2016 ARM Limited. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * -----------------------------------------------------------------------------
 *
 * Project:     CMSIS-RTOS RTX
 * Title:       Memory functions
 *
 * -----------------------------------------------------------------------------
 */

#include "rtx_lib.h"


//  Memory Pool Header structure
typedef struct mem_head_s {
  uint32_t size;                // Memory Pool size
  uint32_t used;                // Used Memory
} mem_head_t;

//  Memory Block Header structure
typedef struct mem_block_s {
  struct mem_block_s *next;     // Next Memory Block in list
  uint32_t             len;     // Memory Block length
} mem_block_t;


//  ==== Library functions ====

/// Initialize Memory Pool with variable block size.
/// \param[in]  mem             pointer to memory pool.
/// \param[in]  size            size of a memory pool in bytes.
/// \return 1 - success, 0 - failure.
uint32_t os_MemoryInit (void *mem, uint32_t size) {
  mem_head_t  *head;
  mem_block_t *ptr;

  if ((mem == NULL) || ((uint32_t)mem & 3U) || (size & 3U) ||
      (size < (sizeof(mem_head_t) + sizeof(mem_block_t) + sizeof(mem_block_t *)))) {
    return 0U;
  }

  head = (mem_head_t *)mem;
  head->size = size;
  head->used = sizeof(mem_head_t) + sizeof(mem_block_t *);

  ptr = (mem_block_t *)((uint32_t)mem + sizeof(mem_head_t));
  ptr->next = (mem_block_t *)((uint32_t)mem + size - sizeof(mem_block_t *));
  ptr->next->next = NULL;
  ptr->len = 0U;

  return 1U;
}

/// Allocate a memory block from a Memory Pool.
/// \param[in]  mem             pointer to memory pool.
/// \param[in]  size            size of a memory block in bytes.
/// \return allocated memory block or NULL in case of no memory is available.
void *os_MemoryAlloc (void *mem, uint32_t size) {
  mem_block_t *p, *p_new, *ptr;
  uint32_t     hole_size;

  if ((mem == NULL) || (size == 0U)) {
    return NULL;
  }

  // Add header to size
  size += sizeof(mem_block_t);
  // Make sure that block is 4-byte aligned
  size = (size + 3U) & ~((uint32_t)3U);

  // Search for hole big enough
  p = (mem_block_t *)((uint32_t)mem + sizeof(mem_head_t));
  for (;;) {
    hole_size  = (uint32_t)p->next - (uint32_t)p;
    hole_size -= p->len;
    if (hole_size >= size) {
      // Hole found
      break;
    }
    p = p->next;
    if (p->next == NULL) {
      // Failed (end of list)
      return NULL;
    }
  }

  ((mem_head_t *)mem)->used += size;

  if (p->len == 0U) {
    // No block allocated, set length of first element
    p->len = size;
    ptr = (mem_block_t *)((uint32_t)p + sizeof(mem_block_t));
  } else {
    // Insert new element into the list
    p_new = (mem_block_t *)((uint32_t)p + p->len);
    p_new->next = p->next;
    p_new->len  = size;
    p->next = p_new;
    ptr = (mem_block_t *)((uint32_t)p_new + sizeof(mem_block_t));
  }

  return ptr;
}

/// Return an allocated memory block back to a Memory Pool.
/// \param[in]  mem             pointer to memory pool.
/// \param[in]  block           memory block to be returned to the memory pool.
/// \return 1 - success, 0 - failure.
uint32_t os_MemoryFree (void *mem, void *block) {
  mem_block_t *p, *p_prev, *ptr;

  if ((mem == NULL) || (block == NULL)) {
    return 0U;
  }

  ptr = (mem_block_t *)((uint32_t)block - sizeof(mem_block_t));

  // Search for header
  p_prev = NULL;
  p = (mem_block_t *)((uint32_t)mem + sizeof(mem_head_t));
  while (p != ptr) {
    p_prev = p;
    p = p->next;
    if (p == NULL) {
      // Not found
      return 0U;
    }
  }

  ((mem_head_t *)mem)->used -= p->len;

  if (p_prev == NULL) {
    // Release first block, only set len to 0
    p->len = 0U;
  } else {
    // Discard block from chained list
    p_prev->next = p->next;
  }

  return 0U;
}
