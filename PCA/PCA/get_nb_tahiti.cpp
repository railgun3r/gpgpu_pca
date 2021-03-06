#include "stdafx.h"
/*
    -- clMAGMA (version 1.0.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       April 2012
*/
#include <stdio.h>
// ==== Definition of blocking sizes for AMD Tahiti cards

/* ////////////////////////////////////////////////////////////////////////////
   -- Return nb for potrf based on n
*/ 
extern "C"
int magma_get_spotrf_nb(int n) {
    if (n <= 1024)
        return 128;
    else
        return 320;
}

extern "C"
int magma_get_dpotrf_nb(int n) {
    if (n <= 4256)
        return 128;
    else 
        return 256;
}

extern "C"
int magma_get_cpotrf_nb(int n) {
    return 128;
}

extern "C"
int magma_get_zpotrf_nb(int n) {
    return 64;
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Return nb for geqrf based on m
*/
extern "C"
int magma_get_sgeqrf_nb(int m) {
    if (m < 2000)
        return 128; // 256 seems to make sgeqrf_gpu faster but causes problem with sgeqrs_gpu
    else
        return 128;
}

extern "C"
int magma_get_dgeqrf_nb(int m) {
    if (m <= 2048)
        return 64;
    else 
        return 128;
}

extern "C"
int magma_get_cgeqrf_nb(int m) {
    if (m <= 2048)
        return 32;
    else if (m <= 4032)
        return 64;
    else
        return 128;
}

extern "C"
int magma_get_zgeqrf_nb(int m) {
    if (m <= 2048)
        return 32;
    else if (m <= 4032)
        return 64;
    else
        return 128;
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Return nb for geqlf based on m
*/
extern "C"
int magma_get_sgeqlf_nb(int m) {
    return magma_get_sgeqrf_nb(m);
}

extern "C"
int magma_get_dgeqlf_nb(int m) {
    return magma_get_dgeqrf_nb(m);
}

extern "C"
int magma_get_cgeqlf_nb(int m) {
    if (m <= 2048)
        return 32;
    else if (m <= 4032)
        return 64;
    else
        return 128;
}

extern "C"
int magma_get_zgeqlf_nb(int m) {
    if (m <= 1024)
        return 64;
    else
        return 128;
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Return nb for gelqf based on m
*/
extern "C"
int magma_get_sgelqf_nb(int m) {
    return magma_get_sgeqrf_nb(m);
}

extern "C"
int magma_get_dgelqf_nb(int m) {
    return magma_get_dgeqrf_nb(m);
}

extern "C"
int magma_get_cgelqf_nb(int m) {
    if (m <= 2048)
        return 32;
    else if (m <= 4032)
        return 64;
    else
        return 128;
}

extern "C"
int magma_get_zgelqf_nb(int m) {
    if (m <= 1024)
        return 64;
    else
        return 128;
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Return nb for getrf based on m;
      the return value should be multiple of 64
*/
extern "C"
int magma_get_sgetrf_nb(int m) {
     if (m <= 6016)
        return 32; 
    else 
        return 64;
/* ACD 2013-09-18 below replaced with above as test
    if (m <= 2048)
        return 64;
    else
        return 128;

/* ACD 2013-09-18 below replaced with above as a test
    if (m <= 6016){
        return 32; 
    }else if (m <= 9024){
        return 64;
    }else{
        return 128;
	}
/* ACD 2013-09-17 the above numbers seem to let my Cypress run faster than with the Tahiti numbers below
    if (m <= 3200)
        return 128;
    else if (m < 9000)
        return 256;
    else
        return 320;
*/
}
extern "C"
int magma_get_dgetrf_nb(int m) {
     if (m <= 6016)
        return 32; 
    else 
        return 64;
	
/* ACD 2013-09-18 below replaced with above as test
   if (m <= 2048)
        return 64;
    else if (m < 7200)
        return 192;
    else
        return 256;
*/
}

extern "C"
int magma_get_cgetrf_nb(int m) {
     if (m <= 6016)
        return 32; 
    else 
        return 64;
/* ACD 2013-09-18 below replaced with above as test
    if (m <= 2048)
        return 64;
    else
        return 128;
*/
}

extern "C"
int magma_get_zgetrf_nb(int m) {
     if (m <= 6016)
        return 32; 
    else 
        return 64;

/* ACD 2013-09-18 below replaced with above as a test
    if (m <= 3072)
        return 32; 
    else if (m <= 9024)
        return 64;
    else
        return 128;
*/
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Return nb for gehrd based on m;
      the return value should be a multiple of 32
*/
extern "C"
int magma_get_sgehrd_nb(int m) {
    if (m <= 1024)
        return 32;
    else
        return 96;
}

extern "C"
int magma_get_dgehrd_nb(int m) {
    if (m <= 2048)
        return 32;
    else
        return 64;
}

extern "C"
int magma_get_cgehrd_nb(int m) {
    if (m <= 1024)
        return 32;
    else
        return 64;
}

extern "C"
int magma_get_zgehrd_nb(int m) {
    if (m <= 2048)
        return 32;
    else
        return 64;
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Return nb for sytrd based on m;
      the return value should be a multiple of 32
*/
extern "C"
int magma_get_ssytrd_nb(int m) {
    return 32;
    //if (m <= 1024)
    //    return 64;
    //else
    //    return 64;
}

extern "C"
int magma_get_dsytrd_nb(int m) {
    return 32;
    //return 64;
    //if (m <= 1024)
    //    return 64;
    //else
    //    return 64;
}

extern "C"
int magma_get_chetrd_nb(int m) {
    return 32;
    //if (m <= 1024)
    //    return 64;
    //else
    //    return 64;
}

extern "C"
int magma_get_zhetrd_nb(int m) {
    return 32;
    //return 64;
    //if (m <= 1024)
    //    return 64;
    //else
    //    return 64;
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Return nb for gebrd based on m;
      the return value should be a multiple of 32
*/
extern "C"
int magma_get_sgebrd_nb(int m) {
    return 64;//32;
    //if (m <= 1024)
    //    return 64;
    //else
    //    return 64;
}

extern "C"
int magma_get_dgebrd_nb(int m) {
    return 64;//32;
    //if (m <= 1024)
    //    return 64;
    //else
    //    return 64;
}

extern "C"
int magma_get_cgebrd_nb(int m) {
    return 64;//32;
    //if (m <= 1024)
    //    return 64;
    //else
    //    return 64;
}

extern "C"
int magma_get_zgebrd_nb(int m) {
    return 64;//32;
    //if (m <= 1024)
    //    return 64;
    //else
    //    return 64;
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Return nb for zhegst based on m;
      the return value should be a multiple of 32
*/
extern "C"
int magma_get_ssygst_nb(int m) {
    return 64;
}

extern "C"
int magma_get_dsygst_nb(int m) {
    return 64;
}

extern "C"
int magma_get_chegst_nb(int m) {
    return 64;
}

extern "C"
int magma_get_zhegst_nb(int m) {
    return 64;
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Return nb for getri based on m
*/
extern "C"
int magma_get_sgetri_nb(int m) {
    return 64;
}

extern "C"
int magma_get_dgetri_nb(int m) {
    return 64;
}

extern "C"
int magma_get_cgetri_nb(int m) {
    return 64;
}

extern "C"
int magma_get_zgetri_nb(int m) {
    return 64;
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Return nb for gesvd based on m
*/
extern "C" 
int magma_get_sgesvd_nb(int m){
    return magma_get_sgebrd_nb(m);
}

extern "C"
int magma_get_dgesvd_nb(int m){
	return magma_get_dgebrd_nb(m);
}

extern "C"
int magma_get_cgesvd_nb(int m){
	return magma_get_cgebrd_nb(m);
}

extern "C" 
int magma_get_zgesvd_nb(int m){
	return magma_get_zgebrd_nb(m);
}
