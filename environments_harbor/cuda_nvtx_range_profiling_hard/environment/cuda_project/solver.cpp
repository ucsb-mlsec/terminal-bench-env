#include <iostream>
#include <vector>
#include <cmath>
#include <nvToolsExt.h>

class ConjugateGradientSolver {
private:
    int maxIterations;
    double tolerance;
    
public:
    ConjugateGradientSolver(int maxIter = 1000, double tol = 1e-6) 
        : maxIterations(maxIter), tolerance(tol) {}
    
    double dotProduct(const std::vector<double>& a, const std::vector<double>& b) {
        double result = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            result += a[i] * b[i];
        }
        return result;
    }
    
    double vectorNorm(const std::vector<double>& v) {
        return std::sqrt(dotProduct(v, v));
    }
    
    void matrixVectorMultiply(const std::vector<std::vector<double>>& A,
                              const std::vector<double>& x,
                              std::vector<double>& result) {
        size_t n = A.size();
        for (size_t i = 0; i < n; ++i) {
            result[i] = 0.0;
            for (size_t j = 0; j < n; ++j) {
                result[i] += A[i][j] * x[j];
            }
        }
    }
    
    void vectorScale(std::vector<double>& v, double scalar) {
        for (size_t i = 0; i < v.size(); ++i) {
            v[i] *= scalar;
        }
    }
    
    void vectorAdd(std::vector<double>& result, 
                   const std::vector<double>& a,
                   const std::vector<double>& b,
                   double scalarA = 1.0,
                   double scalarB = 1.0) {
        for (size_t i = 0; i < result.size(); ++i) {
            result[i] = scalarA * a[i] + scalarB * b[i];
        }
    }
    
    bool checkConvergence(const std::vector<double>& residual, double& residualNorm) {
        nvtxRangePush("Convergence Check");
        
        residualNorm = vectorNorm(residual);
        bool converged = (residualNorm < tolerance);
        
        if (converged) {
            std::cout << "Convergence achieved with residual norm: " << residualNorm << std::endl;
        }
        
        nvtxRangePop();
        return converged;
    }
    
    int solve(const std::vector<std::vector<double>>& A,
              const std::vector<double>& b,
              std::vector<double>& x) {
        
        nvtxRangePush("Solver Iteration");
        
        size_t n = A.size();
        std::vector<double> r(n), p(n), Ap(n);
        
        matrixVectorMultiply(A, x, Ap);
        vectorAdd(r, b, Ap, 1.0, -1.0);
        p = r;
        
        double rsold = dotProduct(r, r);
        int iteration = 0;
        
        for (iteration = 0; iteration < maxIterations; ++iteration) {
            matrixVectorMultiply(A, p, Ap);
            
            double pAp = dotProduct(p, Ap);
            double alpha = rsold / pAp;
            
            vectorAdd(x, x, p, 1.0, alpha);
            vectorAdd(r, r, Ap, 1.0, -alpha);
            
            double residualNorm;
            if (checkConvergence(r, residualNorm)) {
                break;
            }
            
            double rsnew = dotProduct(r, r);
            double beta = rsnew / rsold;
            
            vectorAdd(p, r, p, 1.0, beta);
            rsold = rsnew;
            
            if (iteration % 100 == 0) {
                std::cout << "Iteration " << iteration << ", residual: " << std::sqrt(rsnew) << std::endl;
            }
        }
        
        nvtxRangePop();
        
        return iteration;
    }
};