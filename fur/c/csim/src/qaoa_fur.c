#include <omp.h>

#include <fur.h>
#include <diagonal.h>

#include <qaoa_fur.h>


void apply_qaoa_furxy_ring(double* sv_real, double* sv_imag, double* const gammas, double* const betas, double* const hc_diag, unsigned int n_qubits, size_t n_states, size_t n_layers, size_t n_trotters)
{
    for (size_t i=0; i<n_layers; i++)
    {
        apply_diagonal(sv_real, sv_imag, -0.5*gammas[i], hc_diag, n_states);
        for (size_t j=0; j<n_trotters; j++)
        {
            furxy_ring(sv_real, sv_imag, 0.5*betas[i]/n_trotters, n_qubits, n_states);
        }
    }
}


void apply_qaoa_furxy_complete(double* sv_real, double* sv_imag, double* const gammas, double* const betas, double* const hc_diag, unsigned int n_qubits, size_t n_states, size_t n_layers, size_t n_trotters)
{
    for (size_t i=0; i<n_layers; i++)
    {
        apply_diagonal(sv_real, sv_imag, -0.5*gammas[i], hc_diag, n_states);
        for (size_t j=0; j<n_trotters; j++)
        {
            furxy_complete(sv_real, sv_imag, 0.5*betas[i]/n_trotters, n_qubits, n_states);
        }
    }
}
