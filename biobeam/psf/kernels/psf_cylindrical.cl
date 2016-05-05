#include <pyopencl-complex.h>


#define M_PI 3.141592653589793f

#ifndef INT_STEPS
#define INT_STEPS 100
#endif

__kernel void psf_cylindrical(
							  __global cfloat_t * Ex,
							  __global float * I,
							  const float y1,const float y2,
							  const float z1,const float z2,
							  const float lam,
							  const float n0,
							  const float alpha){

  int i = get_global_id(0);
  int j = get_global_id(1);

  int Ny = get_global_size(0);
  int Nz = get_global_size(1);

  float y = y1+i*(y2-y1)/(Ny-1.f);
  float z = z1+j*(z2-z1)/(Nz-1.f);

  float kr = 2.f*M_PI/lam*sqrt(z*z+y*y);  
  float phi = atan2(y,z);

  float dt = 2.f*alpha/(INT_STEPS-1.f);

  float E_re = 0.f, E_im = 0.f;
  
  for (int i_t = 0; i_t <= INT_STEPS; i_t++) {

	float t = - alpha + i_t *dt;

	float prefac = dt*sqrt(cos(t));
	E_re += prefac*cos(kr*cos(t-phi));
	E_im += prefac*sin(kr*cos(t-phi));

  }

  Ex[i+j*Ny] = (cfloat_t)(E_re,E_im);
  
  I[i+j*Ny] = E_re*E_re+E_im*E_im;
  
}
