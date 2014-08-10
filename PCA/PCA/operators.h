/**
 *
 *  @file operators.h
 *
 *  clMAGMA (version 1.0.0) --
 *  Univ. of Tennessee, Knoxville
 *  Univ. of California, Berkeley
 *  Univ. of Colorado, Denver
 *  April 2012
 *
 **/

// ACD:  See ATI_Stream/include/CL/cl_platform.h for cl_double2 union members.
// all ".x" replaced with ".s[0]", and all ".y" replaced with ".s[1]"
 
#ifndef MAGMA_OPERATORS_H
#define MAGMA_OPERATORS_H

// todo define these correctly for CUDA
#define __host__
#define __device__
#define __inline__ inline

/*************************************************************
 *              magmaDoubleComplex
 */

__host__ __device__ static __inline__ magmaDoubleComplex 
operator-(const magmaDoubleComplex &a)
{
    return MAGMA_Z_MAKE(-a.s[0], -a.s[1]);
}

__host__ __device__ static __inline__ magmaDoubleComplex 
operator+(const magmaDoubleComplex a, const magmaDoubleComplex b)
{
    return MAGMA_Z_MAKE(a.s[0] + b.s[0], a.s[1] + b.s[1]);
}

__host__ __device__ static __inline__ void
operator+=(magmaDoubleComplex &a, const magmaDoubleComplex b)
{
    a.s[0] += b.s[0];
    a.s[1] += b.s[1];
}

__host__ __device__ static __inline__ magmaDoubleComplex 
operator-(const magmaDoubleComplex a, const magmaDoubleComplex b)
{
    return MAGMA_Z_MAKE(a.s[0] - b.s[0], a.s[1] - b.s[1]);
}

__host__ __device__ static __inline__ void
operator-=(magmaDoubleComplex &a, const magmaDoubleComplex b)
{
    a.s[0] -= b.s[0];
    a.s[1] -= b.s[1];
}

__host__ __device__ static __inline__ magmaDoubleComplex 
operator*(const magmaDoubleComplex a, const magmaDoubleComplex b)
{
    return MAGMA_Z_MAKE(a.s[0] * b.s[0] - a.s[1] * b.s[1], a.s[1] * b.s[0] + a.s[0] * b.s[1]);
}

__host__ __device__ static __inline__ magmaDoubleComplex 
operator*(const magmaDoubleComplex a, const double s)
{
    return MAGMA_Z_MAKE(a.s[0] * s, a.s[1] * s);
}

__host__ __device__ static __inline__ magmaDoubleComplex 
operator*(const double s, const magmaDoubleComplex a)
{
    return MAGMA_Z_MAKE(a.s[0] * s, a.s[1] * s);
}

__host__ __device__ static __inline__ void 
operator*=(magmaDoubleComplex &a, const magmaDoubleComplex b)
{
    double tmp = a.s[1] * b.s[0] + a.s[0] * b.s[1];
    a.s[0] = a.s[0] * b.s[0] - a.s[1] * b.s[1];
    a.s[1] = tmp;
}

__host__ __device__ static __inline__ void 
operator*=(magmaDoubleComplex &a, const double s)
{
    a.s[0] *= s;
    a.s[1] *= s;
}

/*************************************************************
 *              magmaFloatComplex
 */

__host__ __device__ static __inline__ magmaFloatComplex 
operator-(const magmaFloatComplex &a)
{
    return MAGMA_C_MAKE(-a.s[0], -a.s[1]);
}

__host__ __device__ static __inline__ magmaFloatComplex 
operator+(const magmaFloatComplex a, const magmaFloatComplex b)
{
    return MAGMA_C_MAKE(a.s[0] + b.s[0], a.s[1] + b.s[1]);
}

__host__ __device__ static __inline__ void
operator+=(magmaFloatComplex &a, const magmaFloatComplex b)
{
    a.s[0] += b.s[0];
    a.s[1] += b.s[1];
}

__host__ __device__ static __inline__ magmaFloatComplex 
operator-(const magmaFloatComplex a, const magmaFloatComplex b)
{
    return MAGMA_C_MAKE(a.s[0] - b.s[0], a.s[1] - b.s[1]);
}

__host__ __device__ static __inline__ void
operator-=(magmaFloatComplex &a, const magmaFloatComplex b)
{
    a.s[0] -= b.s[0];
    a.s[1] -= b.s[1];
}

__host__ __device__ static __inline__ magmaFloatComplex 
operator*(const magmaFloatComplex a, const magmaFloatComplex b)
{
    return MAGMA_C_MAKE(a.s[0] * b.s[0] - a.s[1] * b.s[1], a.s[1] * b.s[0] + a.s[0] * b.s[1]);
}

__host__ __device__ static __inline__ magmaFloatComplex 
operator*(const magmaFloatComplex a, const float s)
{
    return MAGMA_C_MAKE(a.s[0] * s, a.s[1] * s);
}

__host__ __device__ static __inline__ magmaFloatComplex 
operator*(const float s, const magmaFloatComplex a)
{
    return MAGMA_C_MAKE(a.s[0] * s, a.s[1] * s);
}

__host__ __device__ static __inline__ void 
operator*=(magmaFloatComplex &a, const magmaFloatComplex b)
{
    float tmp = a.s[1] * b.s[0] + a.s[0] * b.s[1];
    a.s[0] = a.s[0] * b.s[0] - a.s[1] * b.s[1];
    a.s[1] = tmp;
}

__host__ __device__ static __inline__ void 
operator*=(magmaFloatComplex &a, const float s)
{
    a.s[0] *= s;
    a.s[1] *= s;
}

#endif  // MAGMA_OPERATORS_H
