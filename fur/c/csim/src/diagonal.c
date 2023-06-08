#include <stdlib.h>
#include <math.h>
#include <omp.h>

#include <diagonal.h>


void apply_diagonal(double* sv_real, double* sv_imag, double theta, double* const diag, size_t n)
{    
    #pragma omp parallel for
    for (size_t i=0; i<n; i++)
    {
        double exp_real = cos(theta * diag[i]);
        double exp_imag = sin(theta * diag[i]);
        double res_real = sv_real[i]*exp_real - sv_imag[i]*exp_imag;
        double res_imag = sv_real[i]*exp_imag + sv_imag[i]*exp_real;
        sv_real[i] = res_real;
        sv_imag[i] = res_imag;
    }
}
