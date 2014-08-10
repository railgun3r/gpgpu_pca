/*
 *   -- clMAGMA (version 1.0.0) --
 *      Univ. of Tennessee, Knoxville
 *      Univ. of California, Berkeley
 *      Univ. of Colorado, Denver
 *      April 2012
 *
 * @author Mark Gates
 */

// ACD: include for _aligned_malloc(,) and _aligned_free()
#include <malloc.h>
 
#ifndef MAGMA_H
#define MAGMA_H

/* ------------------------------------------------------------
 * MAGMA Blas Functions 
 * --------------------------------------------------------- */ 
//#include "magmablas.h"
//#include "auxiliary.h"

/* ------------------------------------------------------------
 * MAGMA functions
 * --------------------------------------------------------- */
//#include "magma_z.h"
//#include "magma_c.h"
//#include "magma_d.h"
//#include "magma_s.h"

#include "magma_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// ========================================
// initialization
magma_err_t
magma_init( void );

magma_err_t
magma_finalize( void );


// ========================================
// memory allocation
magma_err_t
magma_malloc( magma_ptr* ptrPtr, size_t size );

magma_err_t
magma_free( magma_ptr ptr );

magma_err_t
magma_malloc_host( void** ptrPtr, size_t size );

magma_err_t
magma_free_host( void* ptr );

static inline magma_int_t magma_smalloc(magmaFloat_ptr         *ptrPtr, size_t n) { return magma_malloc((magma_ptr*)ptrPtr, n*sizeof(float)); }


// ========================================
// device & queue support
magma_err_t
magma_get_devices(
    magma_device_t* devices,
    magma_int_t     size,
    magma_int_t*    numPtr );

magma_err_t
magma_queue_create( magma_device_t device, magma_queue_t* queuePtr );

magma_err_t
magma_queue_destroy( magma_queue_t  queue );

magma_err_t
magma_queue_sync( magma_queue_t queue );


// ========================================
// event support
magma_err_t
magma_event_create( magma_event_t* eventPtr );

magma_err_t
magma_event_destroy( magma_event_t event );

magma_err_t
magma_event_record( magma_event_t event, magma_queue_t queue );

magma_err_t
magma_event_query( magma_event_t event );

magma_err_t
magma_event_sync( magma_event_t event );

static inline magma_int_t magma_dmalloc(magmaDouble_ptr        *ptrPtr, size_t n) { return magma_malloc((magma_ptr*)ptrPtr, n*sizeof(double)); }

// ========================================
// error handler
void magma_xerbla( const char *name, magma_int_t info );

const char* magma_strerror( int err );

#ifdef __cplusplus
}
#endif

#endif // MAGMA_H
