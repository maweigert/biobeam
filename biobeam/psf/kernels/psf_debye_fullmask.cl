#include <pyopencl-complex.h>
#include <bessel.cl>


#ifndef INT_STEPS
#define INT_STEPS 100
#endif


float transmission_a_from_phi(float phi){

  return .3*(1.+.7*phi/2./M_PI);
  
}

__kernel void precalculate_I(__write_only image2d_t output_re,
							 __write_only image2d_t output_im,
							 const float lam,
							 const float Rmax,
							 const float Zmax){

  int i = get_global_id(0);
  int j = get_global_id(1);

  int Nz = get_global_size(0);
  int Nrad = get_global_size(1);

  
  float kr = 2.f*M_PI/lam*j*Rmax/(Nrad-1);
  float kz = 2.f*M_PI/lam*i*Zmax/(Nz-1);

  cfloat_t I0 = (cfloat_t)(0.f,0.f);
  cfloat_t I1 = (cfloat_t)(0.f,0.f);
  cfloat_t I2 = (cfloat_t)(0.f,0.f);


  // simple traziodal rule

  float dt = 2*M_PI/(INT_STEPS-1.f);

  for (int i_t = 0; i_t <= INT_STEPS; i_t++) {

	float t = i_t *dt;
	float a = transmission_a_from_phi(t);
	
	float co = cos(a);
	float si = sin(a);
	cfloat_t phase = (cfloat_t)(cos(kz*co),sin(kz*co));

	float prefac = dt*sqrt(co)*si;
	
	I0 += prefac*(co+1.f)*bessel_jn(0,kr*si)*phase;
	I1 += prefac*si*bessel_jn(1,kr*si)*phase;
	I2 += prefac*(co-1.f)*bessel_jn(2,kr*si)*phase;

  }

  write_imagef(output_re,(int2)(i,j),(float4)(I0.x,I1.x,I2.x,0));
  write_imagef(output_im,(int2)(i,j),(float4)(I0.y,I1.y,I2.y,0));

  

}


	
__kernel void assemble_I(__read_only image2d_t input_re,
						 __read_only image2d_t input_im,
						 __global cfloat_t * Ex,
						 __global cfloat_t * Ey,
						 __global cfloat_t * Ez,
						 __global float * I,
						 const float Ex0,
						 const float Ey0,
						 const float Rmax,
						 const float Zmax,
						 const float x1,const float x2,
						 const float y1,const float y2,
						 const float z1,const float z2){

  int i = get_global_id(0);
  int j = get_global_id(1);
  int k = get_global_id(2);

  int Nx = get_global_size(0);
  int Ny = get_global_size(1);
  int Nz = get_global_size(2);

  float x = x1+i*(x2-x1)/(Nx-1.f);
  float y = y1+j*(y2-y1)/(Ny-1.f);
  float z = z1+k*(z2-z1)/(Nz-1.f);
  
  float phi = atan2(y,x); 

  float r = sqrt(x*x+y*y);

		   
  const sampler_t sampler =  CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

  float4 I_re = read_imagef(input_re,sampler,(float2)(fabs(z)/Zmax,r/Rmax));
  float4 I_im = read_imagef(input_im,sampler,(float2)(fabs(z)/Zmax,r/Rmax));
  
  cfloat_t I0 = (cfloat_t)(I_re.x,I_im.x);
  cfloat_t I1 = (cfloat_t)(I_re.y,I_im.y);
  cfloat_t I2 = (cfloat_t)(I_re.z,I_im.z);

  // conjugate if z is negative 
  if (z<0){
	I0 = cfloat_conj(I0);
	I1 = cfloat_conj(I1);
	I2 = cfloat_conj(I2);
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
