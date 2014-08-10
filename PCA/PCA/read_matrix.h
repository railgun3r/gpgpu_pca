#include "stdafx.h"

#include <stdio.h>
#include <string.h>

#include "magma.h"

#define MATRIX_LINE_SIZE_MAX 100
int read_matrix (const magma_int_t n, const magma_int_t m, double *a, const char *FileName) {
    char line[MATRIX_LINE_SIZE_MAX];
    int i,j,iter=0,max_iter=n*m;
	double val;
    FILE *fIn;
    fIn = fopen (FileName, "r");
    if (fIn == NULL) {
        printf ("Cannot open file\n");
        return 1;
    }
    while (fgets (line, sizeof(line), fIn) != NULL && iter<max_iter) {
        line[strlen (line)-1] = '\0';

        if (sscanf (line, "%d %d %lf",
            &i, &j, &val) != 3)
        {
            printf ("Line didn't scan properly\n");
            return 1;
        }
		if (i>(m-1) || j>(n-1) || i<0 || j<0)
		{
            printf ("An error occurs while reading input matrix\n");
            return 1;
		}
		a[j*m+i] = val;
		iter++;
    }

    fclose (fIn);

    return 0;
}