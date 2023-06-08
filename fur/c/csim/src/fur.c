#include <fur.h>
#include <omp.h>
#include <stdio.h>
#include <math.h>


void furxy(double* a_real, double* a_imag, double theta, unsigned int q1, unsigned int q2, size_t n_states)
{
    // make sure q1 < q2
    if (q1 > q2)
    {
        q1 ^= q2;
        q2 ^= q1;
        q1 ^= q2;
    }

    // number of groups of states on which the operation is applied locally
    size_t num_groups = n_states / 4;

    // helper digit masks for constructing the locality indices
    // the mask is applied on the index through all groups of local operations
    size_t mask1 = ((size_t)1<<q1) - 1;  // digits lower than q1
    size_t mask2 = ((size_t)1<<(q2-1)) - 1;  // digits lower than q2
    size_t maskm = mask1^mask2;  // digits in between
    mask2 ^= (n_states-1) >> 2;  // digits higher than q2
    
    // pre-compute coefficients in transformation
    double cx = cos(theta), sx = sin(theta);

    #pragma omp parallel for
    for (size_t i = 0; i < num_groups; i++)
    {
        size_t i0 = (i&mask1) | ((i&maskm)<<1) | ((i&mask2)<<2);
        size_t i1 = i0 | ((size_t)1<<q1);
        size_t i2 = i0 | ((size_t)1<<q2);

        double a1r = a_real[i1];
        double a2r = a_real[i2];
        double a1i = a_imag[i1];
        double a2i = a_imag[i2];

        a_real[i1] = cx * a1r + sx * a2i;
        a_real[i2] = sx * a1i + cx * a2r;
        
        a_imag[i1] = cx * a1i - sx * a2r;
        a_imag[i2] = -sx * a1r + cx * a2i;
    }
}


void furxy_ring(double* a_real, double* a_imag, double theta, unsigned int n_qubits, size_t n_states)
{
    
    for (int i=0; i<2; i++)
    {
        for (unsigned int j=i; j<n_qubits-1; j+=2)
        {
            furxy(a_real, a_imag, theta, j, j+1, n_states);
        }
    }
    furxy(a_real, a_imag, theta, 0, n_qubits-1, n_states);
}


void furxy_complete(double* a_real, double* a_imag, double theta, unsigned int n_qubits, size_t n_states)
{
    
    for (unsigned int i=0; i<n_qubits-1; i++)
    {
        for (unsigned int j=i+1; j<n_qubits; j++)
        {
            furxy(a_real, a_imag, theta, i, j, n_states);
        }
    }
}
