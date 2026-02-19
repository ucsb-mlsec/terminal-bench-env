#ifndef __GSL_MATH_H__
#define __GSL_MATH_H__

#ifdef __cplusplus
extern "C" {
#endif

double gsl_pow_int(double x, int n);
double gsl_log1p(const double x);
double gsl_hypot(const double x, const double y);

#ifdef __cplusplus
}
#endif

#endif /* __GSL_MATH_H__ */