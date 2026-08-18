#ifndef CLOSEDLOOP_DENSELINALG_H
#define CLOSEDLOOP_DENSELINALG_H

#include <vector>
#include <stdexcept>
#include <cmath>
#include <string>

// Minimal dense linear algebra for LcmvFit's small (dim = templateLength *
// n_channels, e.g. 305x305) symmetric-positive-definite solves.
//
// No third-party dependency: this machine has no Eigen installation and no
// reliable way to vendor one in this environment, so this is a small,
// self-contained Cholesky (LLT) solver instead. This is a deliberate,
// documented deviation from the original plan (which called for vendoring
// Eigen) -- the risk this session has repeatedly flagged is re-deriving
// DOMAIN-SPECIFIC signal-processing math (filtering phase, NMS windowing),
// not implementing a textbook Cholesky decomposition, which is mechanical
// and validated here against numpy's np.linalg.solve
// (see test_dense_linalg_equivalence.py).
//
// Flat row-major storage throughout: A[i*n+j].
class CholeskySolver {
public:
    // Factorizes A (must be symmetric positive-definite, e.g. R + ridge
    // regularization) in place into its lower-triangular Cholesky factor L
    // (A == L * L^T). Throws std::runtime_error if a diagonal pivot is
    // non-positive (A isn't actually SPD -- e.g. ridge regularization was
    // too weak for how ill-conditioned the input noise covariance is).
    explicit CholeskySolver( const std::vector<double> &A, int n )
        : n_( n ), L_( static_cast<size_t>( n ) * n, 0.0 )
    {
        for( int i = 0; i < n; ++i ) {
            for( int j = 0; j <= i; ++j ) {

                double sum = A[static_cast<size_t>( i ) * n + j];
                for( int k = 0; k < j; ++k )
                    sum -= L_[static_cast<size_t>( i ) * n + k] * L_[static_cast<size_t>( j ) * n + k];

                if( i == j ) {
                    if( sum <= 0.0 )
                        throw std::runtime_error(
                            "CholeskySolver: matrix is not positive-definite "
                            "(non-positive pivot at diagonal " + std::to_string( i ) +
                            ") -- increase ridge regularization" );
                    L_[static_cast<size_t>( i ) * n + j] = std::sqrt( sum );
                }
                else {
                    L_[static_cast<size_t>( i ) * n + j] = sum / L_[static_cast<size_t>( j ) * n + j];
                }
            }
        }
    }

    // Solves A * x = b via forward/back substitution against the stored
    // Cholesky factor. b.size() must equal n.
    std::vector<double> solve( const std::vector<double> &b ) const
    {
        if( static_cast<int>( b.size() ) != n_ )
            throw std::runtime_error( "CholeskySolver::solve: size mismatch" );

        // Forward: L * y = b
        std::vector<double> y( n_ );
        for( int i = 0; i < n_; ++i ) {
            double sum = b[i];
            for( int k = 0; k < i; ++k )
                sum -= L_[static_cast<size_t>( i ) * n_ + k] * y[k];
            y[i] = sum / L_[static_cast<size_t>( i ) * n_ + i];
        }

        // Back: L^T * x = y
        std::vector<double> x( n_ );
        for( int i = n_ - 1; i >= 0; --i ) {
            double sum = y[i];
            for( int k = i + 1; k < n_; ++k )
                sum -= L_[static_cast<size_t>( k ) * n_ + i] * x[k];
            x[i] = sum / L_[static_cast<size_t>( i ) * n_ + i];
        }

        return x;
    }

    // Solves A * X = B for a matrix right-hand side B (n x m, row-major),
    // one column at a time -- used for R^-1 * C where C has multiple
    // columns (target + interferer templates).
    std::vector<double> solveMatrix( const std::vector<double> &B, int m ) const
    {
        std::vector<double> X( static_cast<size_t>( n_ ) * m );
        std::vector<double> col( n_ );
        for( int c = 0; c < m; ++c ) {
            for( int i = 0; i < n_; ++i )
                col[i] = B[static_cast<size_t>( i ) * m + c];
            std::vector<double> xc = solve( col );
            for( int i = 0; i < n_; ++i )
                X[static_cast<size_t>( i ) * m + c] = xc[i];
        }
        return X;
    }

private:
    int n_;
    std::vector<double> L_;
};

#endif // CLOSEDLOOP_DENSELINALG_H
