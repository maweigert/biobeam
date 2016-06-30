/*

Electric field at focal region for a cylindrical lens

see:

Colin J. R. Sheppard,
Cylindrical lenses—focusing and imaging: a review
Appl. Opt. 52, 538-545 (2013)


*/

#include <pyopencl-complex.h>


#define M_PI 3.141592653589793f

#ifndef INT_STEPS
#define INT_STEPS 200
#endif

__kernel void psf_cylindrical(
							  __global cfloat_t * Ex,
							  __global cfloat_t * Ey,
							  __global cfloat_t * Ez,
							  __global float * I,
							  const float y1,const float y2,
							  const float z1,const float z2,
							  const float lam,

							  const float alpha){


  int i = get_global_id(0);
  int j = get_global_id(1);


  int Ny = get_global_size(0);
  int Nz = get_global_size(1);

  float y = y1+i*(y2-y1)/(Ny-1.f);
  float z = z1+j*(z2-z1)/(Nz-1.f);

  float ky = 2.f*M_PI/lam*y;
  float kz = 2.f*M_PI/lam*z;


  float dt = 2.f*alpha/(INT_STEPS-1.f);

  cfloat_t ex = cfloat_new(0.f,0.f);
  cfloat_t ey = cfloat_new(0.f,0.f);
  cfloat_t ez = cfloat_new(0.f,0.f);

  
  for (int i_t = 0; i_t <= INT_STEPS; i_t++) {

	float t = - alpha + i_t *dt;
	float co = cos(t);
	float si = sin(t);

    float prefac = ((i_t==0)||(i_t==INT_STEPS))?.5f:1.f;

	prefac *= dt*sqrt(co);

    cfloat_t phase = cfloat_new(cos(ky*si+kz*co),sin((ky*si+kz*co)));


	ex = cfloat_add(ex,cfloat_rmul(prefac,phase));
    ey = cfloat_add(ey,cfloat_rmul(prefac*co,phase));
    ez = cfloat_add(ez,cfloat_rmul(prefac*si,phase));
  }

  Ex[i+j*Ny] = ex;
  Ey[i+j*Ny] = ey;
  Ez[i+j*Ny] = ez;


  float vx = cfloat_abs(ex);
  float vy = cfloat_abs(ey);
  float vz = cfloat_abs(ez);

  I[i+j*Ny] = vx*vx+vy*vy+vz*vz;

}



__kernel void psf_cylindrical_plane(
							  __global cfloat_t * Ex,
							  const float y1,const float y2,
							  const float z,
							  const float lam,

							  const float alpha){


  int i = get_global_id(0);
  int j = get_global_id(1);


  int Nx = get_global_size(0);
  int Ny = get_global_size(1);

  float y = y1+j*(y2-y1)/(Ny-1.f);

  float ky = 2.f*M_PI/lam*y;
  float kz = 2.f*M_PI/lam*z;


  float dt = 2.f*alpha/(INT_STEPS-1.f);

  cfloat_t ex = cfloat_new(0.f,0.f);


  for (int i_t = 0; i_t <= INT_STEPS; i_t++) {

	float t = - alpha + i_t *dt;
	float co = cos(t);
	float si = sin(t);

    float prefac = ((t==-alpha)||(t==alpha))?.5f:1.f;

	prefac *= dt*sqrt(co);

    cfloat_t phase = cfloat_new(cos(ky*si+kz*co),sin((ky*si+kz*co)));


	ex = cfloat_add(ex,cfloat_rmul(prefac,phase));
  }

  Ex[i+j*Nx] = ex;

}
