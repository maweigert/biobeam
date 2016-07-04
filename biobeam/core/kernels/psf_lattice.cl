/*

calculates the focus field of a bessel lattice

see
Matthew R. Foreman, Peter Toeroek,
Computational methods in vectorial imaging,
Journal of Modern Optics, 2011, 58, 5-6, 339



*/



#include <pyopencl-complex.h>

#ifndef M_PI
#define M_PI 3.141592653589793f
#endif

#ifndef INT_STEPS
#define INT_STEPS 100
#endif

__kernel void debye_wolf_lattice(__global cfloat_t * Ex,
								 __global cfloat_t * Ey,
								 __global cfloat_t * Ez,
								 __global float * I,
								 const float Ex0,
								 const float Ey0,
								 const float x1,const float x2,
								 const float y1,const float y2,
								 const float z1,const float z2,
								 const float lam,
								 const float NA1,
								 const float NA2,
								 __constant float* kxs,
								 __constant float* kys,
								 const int Nks,
								 const float sigma){

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


  float varphi = atan2(y,x);


  cfloat_t ex = cfloat_new(0.f,0.f);
  cfloat_t ey = cfloat_new(0.f,0.f);
  cfloat_t ez = cfloat_new(0.f,0.f);

  // simple traziodal rule

  for (int i_n = 0; i_n < Nks; i_n++) {
    float x_pos0 = kxs[i_n];
    float y_pos0 = kys[i_n];


	float y1 = y_pos0 - 4.f*sigma;
	float y2 = y_pos0 + 4.f*sigma;
	float dy = (y2-y1)/(INT_STEPS-1.f);

  	for (int i_y = 0; i_y <= INT_STEPS; i_y++) {

  	  float y_pos = y1 + i_y *dy;

	  float r_pos = sqrt(x_pos0*x_pos0+y_pos*y_pos);
	  if ((r_pos<NA1)||(r_pos>NA2))
	  	continue;
	  
	  float theta = sqrt(x_pos0*x_pos0+y_pos*y_pos);
	  float phi = atan2(y_pos,x_pos0);


  	  float co_p = cos(phi);
  	  float si_p = sin(phi);
  	  float co2_p = cos(2*phi);
  	  float si2_p = sin(2*phi);   
  	  float co_t = cos(theta);
  	  float si_t = sin(theta);

	  float ang = kr*si_t*cos(phi-varphi)+kz*co_t;
	  
  	  cfloat_t phase = cfloat_new(cos(ang),sin(ang));

	  float exfac = 1.f/sigma*exp(-(y_pos-y_pos0)*(y_pos-y_pos0)/2.f/sigma/sigma);

  	  float prefac = ((y_pos==y1)||(y_pos==y2))?.5f:1.f;

  	  prefac *= 1.f/theta*dy*exfac*sqrt(co_t)*si_t;
	
	  ex = cfloat_add(ex,cfloat_rmul(prefac*((co_t+1.f)+(co_t-1)*co2_p),phase));

      ey = cfloat_add(ey,cfloat_rmul(prefac*(co_t-1.f)*si2_p,phase));

      ez = cfloat_add(ez,cfloat_rmul(-2.*prefac*si_t,phase));
  	}
	  
  }
  float vx = cfloat_abs(ex);
  float vy = cfloat_abs(ey);
  float vz = cfloat_abs(ez);



  Ex[i+j*Nx+k*Nx*Ny] = ex;
  Ey[i+j*Nx+k*Nx*Ny] = ey;
  Ez[i+j*Nx+k*Nx*Ny] = ez;

  I[i+j*Nx+k*Nx*Ny] = vx*vx+vy*vy+vz*vz;


}


__kernel void debye_wolf_lattice_plane(__global cfloat_t * Ex,
								 const float Ex0,
								 const float Ey0,
								 const float x1,const float x2,
								 const float y1,const float y2,
								 const float z,
								 const float lam,
								 const float NA1,
								 const float NA2,
								 __constant float* kxs,
								 __constant float* kys,
								 const int Nks,
									   const float sigma,
									   const int apo_bound){

  int i = get_global_id(0);
  int j = get_global_id(1);

  int Nx = get_global_size(0);
  int Ny = get_global_size(1);

  float x = x1+i*(x2-x1)/(Nx-1.f);
  float y = y1+j*(y2-y1)/(Ny-1.f);


  float kr = 2.f*M_PI/lam*sqrt(x*x+y*y);
  float kz = 2.f*M_PI/lam*z;


  float varphi = atan2(y,x);


  cfloat_t ex = cfloat_new(0.f,0.f);

  // simple traziodal rule

  for (int i_n = 0; i_n < Nks; i_n++) {
    float x_pos0 = kxs[i_n];
    float y_pos0 = kys[i_n];


	float y1 = y_pos0 - 4.f*sigma;
	float y2 = y_pos0 + 4.f*sigma;
	float dy = (y2-y1)/(INT_STEPS-1.f);

  	for (int i_y = 0; i_y <= INT_STEPS; i_y++) {

  	  float y_pos = y1 + i_y *dy;

	  float r_pos = sqrt(x_pos0*x_pos0+y_pos*y_pos);
	  if ((r_pos<NA1)||(r_pos>NA2))
	  	continue;
	  
	  float theta = sqrt(x_pos0*x_pos0+y_pos*y_pos);
	  float phi = atan2(y_pos,x_pos0);



	  //float co_p = cos(phi);
  	  //float si_p = sin(phi);
  	  float co2_p = cos(2*phi);
  	  ///float si2_p = sin(2*phi);
  	  float co_t = cos(theta);
  	  float si_t = sin(theta);

	  float ang = kr*si_t*cos(phi-varphi)+kz*co_t;
	  
  	  cfloat_t phase = cfloat_new(cos(ang),sin(ang));

	  float exfac = 1.f/sigma*exp(-(y_pos-y_pos0)*(y_pos-y_pos0)/2.f/sigma/sigma);

  	  float prefac = ((y_pos==y1)||(y_pos==y2))?.5f:1.f;

  	  prefac *= 1.f/theta*dy*exfac*sqrt(co_t)*si_t;
	
	  ex = cfloat_add(ex,cfloat_rmul(prefac*((co_t+1.f)+(co_t-1)*co2_p),phase));

  	}
	  
  }

  //apodize with a hanning window
  float dist_bound = fmin(1.f*i, 1.f*(Nx-i-1));

  float dx_apo = (dist_bound<apo_bound)?(1.f*dist_bound/apo_bound):1.f;

  float window_fac = .5f*(1.f-cos(M_PI*dx_apo));


  // if (i+j==0)
  // 	printf("%i %.5f \n",i,dx_apo);

  ex = cfloat_rmul(window_fac,ex);
  
  Ex[i+j*Nx] = ex;


  // Ex[i+j*Nx] = cfloat_new(window_fac,0.);
  


}


