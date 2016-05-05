/*
 * Copyright (c) 1999
 * Silicon Graphics Computer Systems, Inc.
 *
 * Copyright (c) 1999
 * Boris Fomitchev
 *
 * Copyright (c) 2012
 * Andreas Kloeckner
 *
 * This material is provided "as is", with absolutely no warranty expressed
 * or implied. Any use is at your own risk.
 *
 * Permission to use or copy this software for any purpose is hereby granted
 * without fee, provided the above notices are retained on all copies.
 * Permission to modify the code and to distribute modified code is granted,
 * provided the above notices are retained, and a notice that the code was
 * modified is included with the above copyright notice.
 *
 */

// This file is available for inclusion in pyopencl kernels and provides
// complex types 'cfloat_t' and 'cdouble_t', along with a number of special
// functions as visible below, e.g. cdouble_log(z).
//
// Under the hood, the complex types are simply float2 and double2.
// Note that native (operator-based) addition (float + float2) and
// multiplication (float2*float1) is defined for these types,
// but do not match the rules of complex arithmetic.

#define PYOPENCL_DECLARE_COMPLEX_TYPE_INT(REAL_TP, REAL_3LTR, TPROOT, TP) \
  \
  REAL_TP TPROOT##_real(TP a) { return a.x; } \
  REAL_TP TPROOT##_imag(TP a) { return a.y; } \
  REAL_TP TPROOT##_abs(TP a) { return hypot(a.x, a.y); } \
  \
  TP TPROOT##_fromreal(REAL_TP a) { return (TP)(a, 0); } \
  TP TPROOT##_new(REAL_TP a, REAL_TP b) { return (TP)(a, b); } \
  TP TPROOT##_conj(TP a) { return (TP)(a.x, -a.y); } \
  \
  TP TPROOT##_add(TP a, TP b) \
  { \
    return a+b; \
  } \
  TP TPROOT##_addr(TP a, REAL_TP b) \
  { \
    return (TP)(b+a.x, a.y); \
  } \
  TP TPROOT##_radd(REAL_TP a, TP b) \
  { \
    return (TP)(a+b.x, b.y); \
  } \
  \
  TP TPROOT##_mul(TP a, TP b) \
  { \
    return (TP)( \
        a.x*b.x - a.y*b.y, \
        a.x*b.y + a.y*b.x); \
  } \
  \
  TP TPROOT##_mulr(TP a, REAL_TP b) \
  { \
    return a*b; \
  } \
  \
  TP TPROOT##_rmul(REAL_TP a, TP b) \
  { \
    return a*b; \
  } \
  \
  TP TPROOT##_rdivide(REAL_TP z1, TP z2) \
  { \
    if (fabs(z2.x) <= fabs(z2.y)) { \
      REAL_TP ratio = z2.x / z2.y; \
      REAL_TP denom = z2.y * (1 + ratio * ratio); \
      return (TP)((z1 * ratio) / denom, - z1 / denom); \
    } \
    else { \
      REAL_TP ratio = z2.y / z2.x; \
      REAL_TP denom = z2.x * (1 + ratio * ratio); \
      return (TP)(z1 / denom, - (z1 * ratio) / denom); \
    } \
  } \
  \
  TP TPROOT##_divide(TP z1, TP z2) \
  { \
    REAL_TP ratio, denom, a, b, c, d; \
    \
    if (fabs(z2.x) <= fabs(z2.y)) { \
      ratio = z2.x / z2.y; \
      denom = z2.y; \
      a = z1.y; \
      b = z1.x; \
      c = -z1.x; \
      d = z1.y; \
    } \
    else { \
      ratio = z2.y / z2.x; \
      denom = z2.x; \
      a = z1.x; \
      b = z1.y; \
      c = z1.y; \
      d = -z1.x; \
    } \
    denom *= (1 + ratio * ratio); \
    return (TP)( \
       (a + b * ratio) / denom, \
       (c + d * ratio) / denom); \
  } \
  \
  TP TPROOT##_divider(TP a, REAL_TP b) \
  { \
    return a/b; \
  } \
  \
  TP TPROOT##_pow(TP a, TP b) \
  { \
    REAL_TP logr = log(hypot(a.x, a.y)); \
    REAL_TP logi = atan2(a.y, a.x); \
    REAL_TP x = exp(logr * b.x - logi * b.y); \
    REAL_TP y = logr * b.y + logi * b.x; \
    \
    REAL_TP cosy; \
    REAL_TP siny = sincos(y, &cosy); \
    return (TP) (x*cosy, x*siny); \
  } \
  \
  TP TPROOT##_powr(TP a, REAL_TP b) \
  { \
    REAL_TP logr = log(hypot(a.x, a.y)); \
    REAL_TP logi = atan2(a.y, a.x); \
    REAL_TP x = exp(logr * b); \
    REAL_TP y = logi * b; \
    \
    REAL_TP cosy; \
    REAL_TP siny = sincos(y, &cosy); \
    \
    return (TP)(x * cosy, x*siny); \
  } \
  \
  TP TPROOT##_rpow(REAL_TP a, TP b) \
  { \
    REAL_TP logr = log(a); \
    REAL_TP x = exp(logr * b.x); \
    REAL_TP y = logr * b.y; \
    \
    REAL_TP cosy; \
    REAL_TP siny = sincos(y, &cosy); \
    return (TP) (x * cosy, x * siny); \
  } \
  \
  TP TPROOT##_sqrt(TP a) \
  { \
    REAL_TP re = a.x; \
    REAL_TP im = a.y; \
    REAL_TP mag = hypot(re, im); \
    TP result; \
    \
    if (mag == 0.f) { \
      result.x = result.y = 0.f; \
    } else if (re > 0.f) { \
      result.x = sqrt(0.5f * (mag + re)); \
      result.y = im/result.x/2.f; \
    } else { \
      result.y = sqrt(0.5f * (mag - re)); \
      if (im < 0.f) \
        result.y = - result.y; \
      result.x = im/result.y/2.f; \
    } \
    return result; \
  } \
  \
  TP TPROOT##_exp(TP a) \
  { \
    REAL_TP expr = exp(a.x); \
    REAL_TP cosi; \
    REAL_TP sini = sincos(a.y, &cosi); \
    return (TP)(expr * cosi, expr * sini); \
  } \
  \
  TP TPROOT##_log(TP a) \
  { return (TP)(log(hypot(a.x, a.y)), atan2(a.y, a.x)); } \
  \
  TP TPROOT##_sin(TP a) \
  { \
    REAL_TP cosr; \
    REAL_TP sinr = sincos(a.x, &cosr); \
    return (TP)(sinr*cosh(a.y), cosr*sinh(a.y)); \
  } \
  \
  TP TPROOT##_cos(TP a) \
  { \
    REAL_TP cosr; \
    REAL_TP sinr = sincos(a.x, &cosr); \
    return (TP)(cosr*cosh(a.y), -sinr*sinh(a.y)); \
  } \
  \
  TP TPROOT##_tan(TP a) \
  { \
    REAL_TP re2 = 2.f * a.x; \
    REAL_TP im2 = 2.f * a.y; \
    \
    const REAL_TP limit = log(REAL_3LTR##_MAX); \
    \
    if (fabs(im2) > limit) \
      return (TP)(0.f, (im2 > 0 ? 1.f : -1.f)); \
    else \
    { \
      REAL_TP den = cos(re2) + cosh(im2); \
      return (TP) (sin(re2) / den, sinh(im2) / den); \
    } \
  } \
  \
  TP TPROOT##_sinh(TP a) \
  { \
    REAL_TP cosi; \
    REAL_TP sini = sincos(a.y, &cosi); \
    return (TP)(sinh(a.x)*cosi, cosh(a.x)*sini); \
  } \
  \
  TP TPROOT##_cosh(TP a) \
  { \
    REAL_TP cosi; \
    REAL_TP sini = sincos(a.y, &cosi); \
    return (TP)(cosh(a.x)*cosi, sinh(a.x)*sini); \
  } \
  \
  TP TPROOT##_tanh(TP a) \
  { \
    REAL_TP re2 = 2.f * a.x; \
    REAL_TP im2 = 2.f * a.y; \
    \
    const REAL_TP limit = log(REAL_3LTR##_MAX); \
    \
    if (fabs(re2) > limit) \
      return (TP)((re2 > 0 ? 1.f : -1.f), 0.f); \
    else \
    { \
      REAL_TP den = cosh(re2) + cos(im2); \
      return (TP) (sinh(re2) / den, sin(im2) / den); \
    } \
  } \

#define PYOPENCL_DECLARE_COMPLEX_TYPE(BASE, BASE_3LTR) \
  typedef BASE##2 c##BASE##_t; \
  \
  PYOPENCL_DECLARE_COMPLEX_TYPE_INT(BASE, BASE_3LTR, c##BASE, c##BASE##_t)

PYOPENCL_DECLARE_COMPLEX_TYPE(float, FLT);
#define cfloat_cast(a) ((cfloat_t) ((a).x, (a).y))

#ifdef PYOPENCL_DEFINE_CDOUBLE
PYOPENCL_DECLARE_COMPLEX_TYPE(double, DBL);
#define cdouble_cast(a) ((cdouble_t) ((a).x, (a).y))
#endif


/*
Bessel functions 

taken from
http://www.atnf.csiro.au/computing/software/gipsy/sub/bessel.c
 */

#define ACC 40.0
#define BIGNO 1.0e10
#define BIGNI 1.0e-10
#define M_PI 3.141592653589793


float bessel_j0( float x )
{
   float ax,z;
   float xx,y,ans,ans1,ans2;

   if ((ax=fabs(x)) < 8.0) {
      y=x*x;
      ans1=57568490574.0+y*(-13362590354.0+y*(651619640.7
         +y*(-11214424.18+y*(77392.33017+y*(-184.9052456)))));
      ans2=57568490411.0+y*(1029532985.0+y*(9494680.718
         +y*(59272.64853+y*(267.8532712+y*1.0))));
      ans=ans1/ans2;
   } else {
      z=8.0/ax;
      y=z*z;
      xx=ax-0.785398164;
      ans1=1.0+y*(-0.1098628627e-2+y*(0.2734510407e-4
         +y*(-0.2073370639e-5+y*0.2093887211e-6)));
      ans2 = -0.1562499995e-1+y*(0.1430488765e-3
         +y*(-0.6911147651e-5+y*(0.7621095161e-6
         -y*0.934935152e-7)));
      ans=sqrt(0.636619772/ax)*(cos(xx)*ans1-z*sin(xx)*ans2);
   }
   return ans;
}

float bessel_j1( float x )
{
   float ax,z;
   float xx,y,ans,ans1,ans2;

   if ((ax=fabs(x)) < 8.0) {
      y=x*x;
      ans1=x*(72362614232.0+y*(-7895059235.0+y*(242396853.1
         +y*(-2972611.439+y*(15704.48260+y*(-30.16036606))))));
      ans2=144725228442.0+y*(2300535178.0+y*(18583304.74
         +y*(99447.43394+y*(376.9991397+y*1.0))));
      ans=ans1/ans2;
   } else {
      z=8.0/ax;
      y=z*z;
      xx=ax-2.356194491;
      ans1=1.0+y*(0.183105e-2+y*(-0.3516396496e-4
         +y*(0.2457520174e-5+y*(-0.240337019e-6))));
      ans2=0.04687499995+y*(-0.2002690873e-3
         +y*(0.8449199096e-5+y*(-0.88228987e-6
         +y*0.105787412e-6)));
      ans=sqrt(0.636619772/ax)*(cos(xx)*ans1-z*sin(xx)*ans2);
      if (x < 0.0) ans = -ans;
   }
   return ans;
}


float bessel_jn( int n, float x )
{
   int    j, jsum, m;
   float ax, bj, bjm, bjp, sum, tox, ans;


   ax=fabs(x);
   if (n == 0)
      return( bessel_j0(ax) );
   if (n == 1)
      return( bessel_j1(ax) );
   if (ax == 0.0)
      return 0.0;
   else if (ax > (float) n) {
      tox=2.0/ax;
      bjm=bessel_j0(ax);
      bj=bessel_j1(ax);
      for (j=1;j<n;j++) {
         bjp=j*tox*bj-bjm;
         bjm=bj;
         bj=bjp;
      }
      ans=bj;
   } else {
      tox=2.0/ax;
      m=2*((n+(int) sqrt(ACC*n))/2);
      jsum=0;
      bjp=ans=sum=0.0;
      bj=1.0;
      for (j=m;j>0;j--) {
         bjm=j*tox*bj-bjp;
         bjp=bj;
         bj=bjm;
         if (fabs(bj) > BIGNO) {
            bj *= BIGNI;
            bjp *= BIGNI;
            ans *= BIGNI;
            sum *= BIGNI;
         }
         if (jsum) sum += bj;
         jsum=!jsum;
         if (j == n) ans=bjp;
      }
      sum=2.0*sum-bj;
      ans /= sum;
   }
   return  x < 0.0 && n%2 == 1 ? -ans : ans;
}




__kernel void bessel_fill( __global float * x,
						   __global float * out,
						 const int n){

  int i = get_global_id(0);

  out[i] = bessel_jn(n,x[i]);
  
  
}





#ifndef INT_STEPS
#define INT_STEPS 100
#endif

__kernel void debye_wolf(__global cfloat_t * Ex,
						 __global cfloat_t * Ey,
						 __global cfloat_t * Ez,
						 __global float * I,
						 const float Ex0,
						 const float Ey0,
						 const float x1,const float x2,
						 const float y1,const float y2,
						 const float z1,const float z2,
						 const float lam,
						 __constant float* alphas, const int Nalphas){

  int i = get_global_id(0);
  int j = get_global_id(1);
  int k = get_global_id(2);

  int Nx = get_global_size(0);
  int Ny = get_global_size(1);
  int Nz = get_global_size(2);

  float x = x1+i*(x2-x1)/(Nx-1.f);
  float y = y1+j*(y2-y1)/(Ny-1.f);
  float z = z1+k*(z2-z1)/(Nz-1.f);

  float kr = 2.f*M_PI/lam*sqrt(x*x+y*y);
  float kz = 2.f*M_PI/lam*z;
  
  float phi = atan2(y,x); 
  
  cfloat_t I0 = (cfloat_t)(0.f,0.f);
  cfloat_t I1 = (cfloat_t)(0.f,0.f);
  cfloat_t I2 = (cfloat_t)(0.f,0.f);


  // simple traziodal rule

  for (int i_n = 0; i_n < Nalphas/2; i_n++) {
    float alpha1 = alphas[2*i_n];
	float alpha2 = alphas[2*i_n+1];
	
	float dt = (alpha2-alpha1)/(INT_STEPS-1.f);

	for (int i_t = 0; i_t <= INT_STEPS; i_t++) {

	  float t = alpha1 + i_t *dt;
	  float co = cos(t);
	  float si = sin(t);
	  cfloat_t phase = (cfloat_t)(cos(kz*co),sin(kz*co));

	  float prefac = ((t==alpha1)||(t==alpha2))?.5f:1.f;

	  prefac *= dt*sqrt(co)*si;
	
	  I0 += prefac*(co+1.f)*bessel_jn(0,kr*si)*phase;
	  I1 += prefac*si*bessel_jn(1,kr*si)*phase;
	  I2 += prefac*(co-1.f)*bessel_jn(2,kr*si)*phase;

	}
  }

  cfloat_t ex = Ex0*(I0+I2*cos(2.f*phi))+Ey0*I2*sin(2.f*phi);
  cfloat_t ey = Ey0*(I0-I2*cos(2.f*phi))+Ex0*I2*sin(2.f*phi);
  cfloat_t ez = cfloat_mul((cfloat_t)(0.f,-2.f),I1)*(Ex0*cos(phi)+Ey0*sin(phi));

  float vx = cfloat_abs(ex);
  float vy = cfloat_abs(ey);
  float vz = cfloat_abs(ez);

  Ex[i+j*Nx+k*Nx*Ny] = ex;
  Ey[i+j*Nx+k*Nx*Ny] = ey;
  Ez[i+j*Nx+k*Nx*Ny] = ez;

  I[i+j*Nx+k*Nx*Ny] = vx*vx+vy*vy+vz*vz;
}
 



__kernel void debye_wolf_slit(__global cfloat_t * Ex,
						 __global cfloat_t * Ey,
						 __global cfloat_t * Ez,
						 __global float * I,
						 const float Ex0,
						 const float Ey0,
						 const float x1,const float x2,
						 const float y1,const float y2,
						 const float z1,const float z2,
						 const float lam,
							  __constant float* alphas, const int Nalphas,
							  __constant float* slit_x, __constant float* slit_sigma,
							  const int Nslit_x
							  ){

  int i = get_global_id(0);
  int j = get_global_id(1);
  int k = get_global_id(2);

  int Nx = get_global_size(0);
  int Ny = get_global_size(1);
  int Nz = get_global_size(2);

  float x = x1+i*(x2-x1)/(Nx-1.f);
  float y = y1+j*(y2-y1)/(Ny-1.f);
  float z = z1+k*(z2-z1)/(Nz-1.f);

  float kr = 2.f*M_PI/lam*sqrt(x*x+y*y);
  float kz = 2.f*M_PI/lam*z;
  
  float phi = atan2(y,x); 
  
  cfloat_t I0 = (cfloat_t)(0.f,0.f);
  cfloat_t I1 = (cfloat_t)(0.f,0.f);
  cfloat_t I2 = (cfloat_t)(0.f,0.f);


  // simple traziodal rule

  for (int i_n = 0; i_n < Nalphas/2; i_n++) {
    float alpha1 = alphas[2*i_n];
	float alpha2 = alphas[2*i_n+1];
	
	float dt = (alpha2-alpha1)/(INT_STEPS-1.f);

	for (int i_t = 0; i_t <= INT_STEPS; i_t++) {

	  float t = alpha1 + i_t *dt;
	  float co = cos(t);
	  float si = sin(t);
	  cfloat_t phase = (cfloat_t)(cos(kz*co),sin(kz*co));

	  float prefac = ((t==alpha1)||(t==alpha2))?.5f:1.f;

	  prefac *= dt*sqrt(co)*si;
	
	  I0 += prefac*(co+1.f)*bessel_jn(0,kr*si)*phase;
	  I1 += prefac*si*bessel_jn(1,kr*si)*phase;
	  I2 += prefac*(co-1.f)*bessel_jn(2,kr*si)*phase;

	}
  }

  cfloat_t ex = Ex0*(I0+I2*cos(2.f*phi))+Ey0*I2*sin(2.f*phi);
  cfloat_t ey = Ey0*(I0-I2*cos(2.f*phi))+Ex0*I2*sin(2.f*phi);
  cfloat_t ez = cfloat_mul((cfloat_t)(0.f,-2.f),I1)*(Ex0*cos(phi)+Ey0*sin(phi));

  float vx = cfloat_abs(ex);
  float vy = cfloat_abs(ey);
  float vz = cfloat_abs(ez);

  Ex[i+j*Nx+k*Nx*Ny] = ex;
  Ey[i+j*Nx+k*Nx*Ny] = ey;
  Ez[i+j*Nx+k*Nx*Ny] = ez;

  I[i+j*Nx+k*Nx*Ny] = vx*vx+vy*vy+vz*vz;
}
 
