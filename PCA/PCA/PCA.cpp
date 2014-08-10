#include "stdafx.h"

// includes, system
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <ctime>

#include <fstream>
#include <vector>
#include <map>
#include <string>
#include <algorithm>


#include "magma.h"
#include "magma_malloc.h"
#include "magmablas_d.h"
#include "magma_d.h"
#include "read_matrix.h"
#include "mkl.h"

using namespace std;

#define SQR(x) ((x)*(x))
#define absv(v1) ((v1)>0? (v1): -(v1))

#define INP_MATRIX_SIZE1 10
#define INP_MATRIX_SIZE2 2000
#define FILENAME "D:\WriteText2000.txt"
#define MATRIX_LINE_SIZE_MAX 100

[module(name="magma_sfortran")];

class CLParser
{
public:

	CLParser(int argc_, char * argv_[], bool switches_on_ = false);
	~CLParser(){}

	string get_arg(int i);
	string get_arg(string s);

private:

	int argc;
	vector<string> argv;

	bool switches_on;
	map<string, string> switch_map;
};

CLParser::CLParser(int argc_, char * argv_[], bool switches_on_)
{
	argc = argc_;
	argv.resize(argc);
	copy(argv_, argv_ + argc, argv.begin());
	switches_on = switches_on_;

	//map the switches to the actual
	//arguments if necessary
	if (switches_on)
	{
		vector<string>::iterator it1, it2;
		it1 = argv.begin();
		it2 = it1 + 1;

		while (true)
		{
			if (it1 == argv.end()) break;
			if (it2 == argv.end()) break;

			if ((*it1)[0] == '-')
				switch_map[*it1] = *(it2);

			it1++;
			it2++;
		}
	}
}

string CLParser::get_arg(int i)
{
	if (i >= 0 && i<argc)
		return argv[i];

	return "";
}

string CLParser::get_arg(string s)
{
	if (!switches_on) return "";

	if (switch_map.find(s) != switch_map.end())
		return switch_map[s];

	return "";
}

void getTime() {
	time_t t = time(0);   // get time now
	struct tm * now = localtime(&t);
	cout << (now->tm_year + 1900) << '-'
		<< (now->tm_mon + 1) << '-'
		<< now->tm_mday << ' '
		<< (now->tm_hour) << ':'
		<< (now->tm_min) << ':'
		<< (now->tm_sec)
		<< endl;
}

int main(int argc, char * argv[])
{
	//CLParser cmd_line(argc, argv, true);
	//std::string temp;
	std::string filename;
	magma_int_t n2, nm, m = INP_MATRIX_SIZE1, n = INP_MATRIX_SIZE2, one = 1;
	filename = FILENAME;
	//temp = cmd_line.get_arg("-m");
	//m = atoi(temp.c_str());
	//cout << m;
	//temp = cmd_line.get_arg("-n");
	//n = atoi(temp.c_str());
	//cout << n;
	//filename = cmd_line.get_arg("-f");
	//cout << filename;
	n2 = n*n; nm = n*m;
    double *A, *EIGENVALUES, *V1, *C, *CORR_MATRIX, *H_WORK, *RESULT_MATRIX;
	magmaDouble_ptr d_A, d_Corr, d_res, d_EV;
	double alpha; double beta;
    magma_uplo_t uplo = MagmaLower;
    magma_vec_t jobz = MagmaVec;
	magma_int_t i, j, info;
    magma_int_t *iwork;
	magma_int_t lwork, liwork;

	/* Magma Initialization (start) */
    magma_queue_t  queue;
    magma_device_t device;
    int num = 0;
    magma_err_t err;
    magma_init();
    err = magma_get_devices( &device, 2, &num ); //TODO: 2 devices
    if ( err != 0 || num < 1 ) {
      fprintf( stderr, "magma_get_devices failed: %d\n", err );
      exit(-1);
    }
    err = magma_queue_create( device, &queue );
    if ( err != 0 ) {
      fprintf( stderr, "magma_queue_create failed: %d\n", err );
      exit(-1);
	}
	/* Magma Initialization (finish) */

	/* Initial matrix reading (start) */
    MAGMA_MALLOC(A, double, nm );
	if (read_matrix(n, m, A, filename.c_str()) != 0) {
		printf("input matrix contains errors");
		exit(-1);
	};
	/* Initial matrix reading (finish) */

	getTime();

	/* Initial matrix centering (srart) */
	MAGMA_MALLOC( V1, double, m );
	for(j=0; j<m; j++){V1[j]=1;} // V1 is vector [1,1,1,...,1]
	MAGMA_MALLOC(C, double, n);
	for(j=0; j<n; j++){C[j]=0;}
	alpha = 1/double(m);
	beta = 1;
	cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, one, n, m, alpha, V1, one, A, m, beta, C, one); //means array calculation: C = (alpha) x V1 x A
	alpha = -1;
	cblas_dger(CblasColMajor, m, n, alpha, V1, one, C, one, A, m); //matrix centering: A = (alpha) x V1 x C' + A
    MAGMA_FREE(     V1    );
	/* Initial matrix centering (finish) */

	/* Initial matrix normalization (start)*/
	alpha = sqrt(1/double(m));
	for(j=0; j<n; j++){
		C[j] = cblas_dnrm2(m, &A[j*m], 1) * alpha; //Euclidean norm calculation
		cblas_dscal(m, 1 / C[j], &A[j*m], 1); //normalization
	}
    MAGMA_FREE(     C    );
	/* Initial matrix normalization (finish)*/

	/* Correlation matrix calculation (start) */	
	getTime();
	MAGMA_MALLOC(CORR_MATRIX, double, n2 );
	alpha = 1 / double(m);
	beta = 1;
	//GPU
	MAGMA_MALLOC_DEV(d_Corr, double, n2);
	MAGMA_MALLOC_DEV(d_A, double, n2);
	magma_dsetmatrix( m, n, A, 0, m, d_A, 0, m, queue);
	magma_queue_sync( queue);
	magma_dsyrk(uplo,MagmaTrans,n,m,alpha,d_A,0,m,beta,d_Corr,0,n,queue); // CORR_MATRIX = (alpha) x A' x A
	magma_queue_sync( queue);
	MAGMA_FREE_DEV(d_A);
	magma_dgetmatrix( n, n, d_Corr, 0, n, CORR_MATRIX, 0, n, queue );
	MAGMA_FREE_DEV(d_Corr);
			//CPU
			//for(i=0; i<n; i++){for(j=0; j<n; j++){CORR_MATRIX[j*n+i] = 0;}}
			//cblas_dsyrk(CblasColMajor, CblasLower, CblasTrans, n, m, alpha, A, m, beta, CORR_MATRIX, n);
	getTime();
	/* Correlation matrix calculation (finish) */

    /* Eigenvalue decomposition: Query for optimal workspace sizes (start)*/
	double      aux_work[1];
	magma_int_t aux_iwork[1];
	MAGMA_MALLOC(EIGENVALUES, double, n);
	magma_queue_sync(queue);
	magma_dsyevd(jobz, uplo,
		n, CORR_MATRIX, n, EIGENVALUES,
		aux_work, -1,
		aux_iwork, -1,
		&info, queue);
	magma_queue_sync(queue);
	lwork = (magma_int_t)aux_work[0];
	liwork = aux_iwork[0];
   /* Eigenvalue decomposition: Query for optimal workspace sizes (finish)*/

	/* Eigenvalue decomposition: Divide and Conquer Algorithm (start)*/
    MAGMA_MALLOC( H_WORK, double,      lwork  );
    MAGMA_MALLOC(    iwork,  magma_int_t, liwork );
	//GPU
	magma_queue_sync( queue);
	magma_dsyevd(jobz, uplo,
					 n, CORR_MATRIX, n, EIGENVALUES,
					 H_WORK, lwork, 
					 iwork, liwork, 
					 &info, queue);	//Corr_matrix now cointains eigenvectors
	magma_queue_sync( queue);
	//CPU
	    //dsyevd(lapack_const(jobz), lapack_const(uplo),
	    //                 &n, CORR_MATRIX, &n, EIGENVALUES,
	    //                 H_WORK, &lwork,
	    //                 iwork, &liwork,
	    //                 &info);
	MAGMA_FREE(     iwork  );
	MAGMA_FREE(H_WORK);
	/* Eigenvalue decomposition: Divide and Conquer Algorithm (finish)*/

	/* PCA results calculation (start) */
	alpha = 1;	beta = 1;
	//GPU
	MAGMA_MALLOC_DEV(d_res, double, nm);
	MAGMA_MALLOC_DEV(d_A, double, nm);
	MAGMA_MALLOC_DEV(d_EV, double, n2);
	magma_dsetmatrix( m, n, A, 0, m, d_A, 0, m, queue);
	magma_dsetmatrix( n, n, CORR_MATRIX, 0, n, d_EV, 0, n, queue);
	magma_queue_sync( queue);
	magma_dgemm(MagmaNoTrans,MagmaNoTrans,m,n,n,alpha,d_A,0,m,d_EV,0,n,beta,d_res,0,m,queue);
	magma_queue_sync( queue);
	MAGMA_FREE_DEV(d_A);
	MAGMA_FREE_DEV(d_EV);
	MAGMA_FREE(A);
	MAGMA_FREE(CORR_MATRIX);
	MAGMA_MALLOC(RESULT_MATRIX, double, nm);
	magma_dgetmatrix( m, n, d_res, 0, m, RESULT_MATRIX, 0, m, queue );
	MAGMA_FREE_DEV(d_res);
	//CPU
	//MAGMA_MALLOC(RESULT_MATRIX, double, nm);
	//cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, n, alpha, A, m, CORR_MATRIX, n, beta, RESULT_MATRIX, m);
	//MAGMA_FREE(A);
	//MAGMA_FREE(CORR_MATRIX);
	/* PCA results calculation (finish) */

	getTime();
 //   /* Shutdown */
    magma_queue_destroy( queue );
    magma_finalize();
	printf("Please press <ENTER> to continue.");
	while(getchar() != '\n');
	printf("done");
}