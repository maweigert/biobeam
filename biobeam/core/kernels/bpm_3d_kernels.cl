#include <pyopencl-complex.h>

#define M_PI 3.14159265358979f


#define FFTFREQ(i,N,dx) ((i<((N-1)/2+1))?1.f*i/N/dx:1.f*((i-((N-1)/2+1))-N/2)/N/dx)




__kernel void mult(__global cfloat_t* a,
				   __global cfloat_t* b){

  uint i = get_global_id(0);
    
  a[i] = cfloat_mul(a[i], b[i]);

}

__kernel void mult_dn(__global cfloat_t* input,
					  __global float* dn,
					  const float unit_k,
					  const float dn0,
					  const int offset){

  int i = get_global_id(0);
  int j = get_global_id(1);
  int Nx = get_global_size(0);
  int Ny = get_global_size(1);

  float dn_val = dn[i+Nx*j+offset];
  float dnDiff = unit_k*(dn_val-dn0);

  // int distx = min(Nx-i-1,i);
  // int disty = min(Ny-j-1,j);
  // int dist = min(distx,disty);

  // float absorb_val = (dist<absorb)?0.5f*(1-cos(M_PI*dist/absorb)):1.f;

  cfloat_t dPhase = cfloat_new(cos(dnDiff),sin(dnDiff));

  cfloat_t res = cfloat_mul(input[i+Nx*j],dPhase);

  input[i+Nx*j] = res;


}


__kernel void mult_dn_complex(__global cfloat_t* input,
					  __global cfloat_t* dn,
					  const float unit_k,
					  const float dn0,
					  const int offset){

  int i = get_global_id(0);
  int j = get_global_id(1);
  int Nx = get_global_size(0);
  int Ny = get_global_size(1);

  
  // int distx = min(Nx-i-1,i);
  // int disty = min(Ny-j-1,j);
  // int dist = min(distx,disty);

  // float absorb_val = (dist<absorb)?0.5f*(1-cos(M_PI*dist/absorb)):1.f;

  cfloat_t dn_val = dn[i+Nx*j+offset];

  cfloat_t dnDiff = cfloat_mul(cfloat_new(0.f,unit_k),cfloat_addr(dn_val,-dn0));

  cfloat_t dPhase = cfloat_exp(dnDiff);


  cfloat_t res = cfloat_mul(input[i+Nx*j],dPhase);

  input[i+Nx*j] = res;

}




__kernel void mult_dn_image(__global cfloat_t* plane,
							__read_only image3d_t dn,
							const float unit_k,
							const float dn0,
							const float zpos){

  // const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE| CLK_ADDRESS_CLAMP_TO_EDGE| CLK_FILTER_NEAREST;
  const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

  uint i = get_global_id(0);
  uint j = get_global_id(1);
  uint Nx = get_global_size(0);
  uint Ny = get_global_size(1);
  

  float dn_val = read_imagef(dn, sampler, (float4)(1.f*i/(Nx-1.f),1.f*j/(Ny-1.f),zpos,0)).x;


  float dnDiff = unit_k*(dn_val-dn0);

  cfloat_t dPhase = cfloat_new(cos(dnDiff),sin(dnDiff));

  plane[i+Nx*j] = cfloat_mul(plane[i+Nx*j],dPhase);

}

__kernel void mult_dn_image_complex(__global cfloat_t* plane,
							__read_only image3d_t dn,
							const float unit_k,
							const float dn0,
							const float zpos){

  //const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE| CLK_ADDRESS_CLAMP_TO_EDGE| CLK_FILTER_NEAREST;
  const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

  uint i = get_global_id(0);
  uint j = get_global_id(1);
  uint Nx = get_global_size(0);
  uint Ny = get_global_size(1);
  
  float2 dn_val = read_imagef(dn, sampler, (float4)(1.f*i/(Nx-1.f),1.f*j/(Ny-1.f),zpos,0)).xy;
  
  cfloat_t dnDiff = cfloat_mul(cfloat_new(0.f,unit_k),cfloat_new(dn_val.x-dn0,dn_val.y));

  cfloat_t dPhase = cfloat_exp(dnDiff);


  cfloat_t res = cfloat_mul(plane[i+Nx*j],dPhase);

  plane[i+Nx*j] = res;

}




__kernel void mult_dn_local(__global cfloat_t* input,
					  __global float* dn,
					  const float unit_k,
					  __constant float * buf_g_sum,
					  __constant  float * buf_dng_sum,
					  const int offset,
					  __global float * buf_sum1,
					  __global float * buf_sum2){

  int i = get_global_id(0);
  int j = get_global_id(1);
  int Nx = get_global_size(0);
  int Ny = get_global_size(1);

  float dn0 = buf_dng_sum[i+Nx*j]/(buf_g_sum[i+Nx*j]+1.e-17f);



  float dn_val = dn[i+Nx*j+offset];
  float dnDiff = unit_k*(dn_val-dn0);


  //if ((i+j)==0)
    //printf("dn0: %.6f %d %d\n",dnDiff,i,j);

  // int distx = min(Nx-i-1,i);
  // int disty = min(Ny-j-1,j);
  // int dist = min(distx,disty);

  // float absorb_val = (dist<absorb)?0.5f*(1-cos(M_PI*dist/absorb)):1.f;

  cfloat_t dPhase = cfloat_new(cos(dnDiff),sin(dnDiff));

  cfloat_t res = cfloat_mul(input[i+Nx*j],dPhase);

  input[i+Nx*j] = res;

  float sum1 = cfloat_abs(res);
  buf_sum1[i+Nx*j] = sum1*sum1;
  buf_sum2[i+Nx*j] = sum1*sum1*dn_val;

}


__kernel void mult_dn_complex_local(__global cfloat_t* input,
					  __global cfloat_t* dn,
					  const float unit_k,
					  __global float * buf_g_sum,
					  __global float * buf_dng_sum,
					  const int offset,
					  __global float * buf_sum1,
					  __global float * buf_sum2){

  int i = get_global_id(0);
  int j = get_global_id(1);
  int Nx = get_global_size(0);
  int Ny = get_global_size(1);

   float dn0 = buf_dng_sum[i+Nx*j]/buf_g_sum[i+Nx*j];

  // int distx = min(Nx-i-1,i);
  // int disty = min(Ny-j-1,j);
  // int dist = min(distx,disty);

  // float absorb_val = (dist<absorb)?0.5f*(1-cos(M_PI*dist/absorb)):1.f;

  cfloat_t dn_val = dn[i+Nx*j+offset];

  cfloat_t dnDiff = cfloat_mul(cfloat_new(0.f,unit_k),cfloat_addr(dn_val,-dn0));

  cfloat_t dPhase = cfloat_exp(dnDiff);


  cfloat_t res = cfloat_mul(input[i+Nx*j],dPhase);

  input[i+Nx*j] = res;

  float sum1 = cfloat_abs(res);
  buf_sum1[i+Nx*j] = sum1*sum1;
  buf_sum2[i+Nx*j] = sum1*sum1*dn_val.real;

}




__kernel void mult_dn_image_local(__global cfloat_t* plane,
							__read_only image3d_t dn,
							const float unit_k,
							 __global float * buf_g_sum,
					  __global float * buf_dng_sum,
							const float zpos,
					  __global float * buf_sum1,
					  __global float * buf_sum2){

  // const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE| CLK_ADDRESS_CLAMP_TO_EDGE| CLK_FILTER_NEAREST;
  const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

  uint i = get_global_id(0);
  uint j = get_global_id(1);
  uint Nx = get_global_size(0);
  uint Ny = get_global_size(1);

float dn0 = buf_dng_sum[i+Nx*j]/buf_g_sum[i+Nx*j];
  float dn_val = read_imagef(dn, sampler, (float4)(1.f*i/(Nx-1.f),1.f*j/(Ny-1.f),zpos,0)).x;


  float dnDiff = unit_k*(dn_val-dn0);

  cfloat_t dPhase = cfloat_new(cos(dnDiff),sin(dnDiff));

  plane[i+Nx*j] = cfloat_mul(plane[i+Nx*j],dPhase);

  float sum1 = cfloat_abs(plane[i+Nx*j]);
  buf_sum1[i+Nx*j] = sum1;
  buf_sum2[i+Nx*j] = sum1*dn_val;


}

__kernel void mult_dn_image_complex_local(__global cfloat_t* plane,
							__read_only image3d_t dn,
							const float unit_k,
							 __global float * buf_g_sum,
					  __global float * buf_dng_sum,
							const float zpos,
					  __global float * buf_sum1,
					  __global float * buf_sum2){

  //const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE| CLK_ADDRESS_CLAMP_TO_EDGE| CLK_FILTER_NEAREST;
  const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

  uint i = get_global_id(0);
  uint j = get_global_id(1);
  uint Nx = get_global_size(0);
  uint Ny = get_global_size(1);
float dn0 = buf_dng_sum[i+Nx*j]/buf_g_sum[i+Nx*j];
  float2 dn_val = read_imagef(dn, sampler, (float4)(1.f*i/(Nx-1.f),1.f*j/(Ny-1.f),zpos,0)).xy;

  cfloat_t dnDiff = cfloat_mul(cfloat_new(0.f,unit_k),cfloat_new(dn_val.x-dn0,dn_val.y));

  cfloat_t dPhase = cfloat_exp(dnDiff);


  cfloat_t res = cfloat_mul(plane[i+Nx*j],dPhase);

  plane[i+Nx*j] = res;

  float sum1 = cfloat_abs(res);
  buf_sum1[i+Nx*j] = sum1*sum1;
  buf_sum2[i+Nx*j] = sum1*sum1*dn_val.x;

}



__kernel void compute_propagator(__global cfloat_t* H,
								 const float n0,
								 const float k0,
								 const float dx,
								 const float dy,
								 const float dz
								 ){

  int i = get_global_id(0);
  int j = get_global_id(1);
  int Nx = get_global_size(0);
  int Ny = get_global_size(1);

  float kx = 2.f*M_PI*FFTFREQ(i,Nx,dx);
  float ky = 2.f*M_PI*FFTFREQ(j,Ny,dy);


  float tmp = n0*n0*k0*k0-kx*kx-ky*ky;

  float root = (tmp>=0.f)?sqrt(tmp):sqrt(-tmp);
  //float h0 = (tmp<0.f)?0.f:sqrt(tmp);

  //cfloat_t h = (tmp>=0.f)?cfloat_new(cos(-dz*root),sin(-dz*root)):cfloat_new(exp(-dz*root),0.f);

  cfloat_t h = (tmp>=0.f)?cfloat_new(cos(dz*root),sin(dz*root)):cfloat_new(exp(-dz*root),0.f);


  H[i+Nx*j] = h;
}

__kernel void compute_propagator_buf(__global cfloat_t* H,
								 const float n0,
								 const float k0,
								 const float dx,
								 const float dy,
								 const float dz,
								 __constant float *sum_dn_intens,
								 __constant float *sum_intens
								 ){

  int i = get_global_id(0);
  int j = get_global_id(1);
  int Nx = get_global_size(0);
  int Ny = get_global_size(1);

  float kx = 2.f*M_PI*FFTFREQ(i,Nx,dx);
  float ky = 2.f*M_PI*FFTFREQ(j,Ny,dy);


  float n00 = n0+sum_dn_intens[0]/sum_intens[0];


  float tmp = n00*n00*k0*k0-kx*kx-ky*ky;

  float root = (tmp>=0.f)?sqrt(tmp):sqrt(-tmp);
  //float h0 = (tmp<0.f)?0.f:sqrt(tmp);

  //cfloat_t h = (tmp>=0.f)?cfloat_new(cos(-dz*root),sin(-dz*root)):cfloat_new(exp(-dz*root),0.f);

  cfloat_t h = (tmp>=0.f)?cfloat_new(cos(dz*root),sin(dz*root)):cfloat_new(exp(-dz*root),0.f);


  H[i+Nx*j] = h;
}


__kernel void img_to_buf_field(__read_only image2d_t src,
								   __global cfloat_t *dest, const int offset){

  const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE| CLK_ADDRESS_CLAMP_TO_EDGE|
  //CLK_FILTER_NEAREST;
  CLK_FILTER_LINEAR;

  uint i = get_global_id(0);
  uint j = get_global_id(1);
  uint Nx = get_global_size(0);
  uint Ny = get_global_size(1);

  float4 val = read_imagef(src,
  						   sampler,
						   (float2)(1.f*i/(Nx-1.f),1.f*j/(Ny-1.f)));
  dest[i+Nx*j+offset] = cfloat_new(val.x,val.y);

 
}



__kernel void img_to_buf_intensity(__read_only image2d_t src,
								   __global float *dest, const int offset){

  const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE| CLK_ADDRESS_CLAMP_TO_EDGE|
  //CLK_FILTER_NEAREST;
  CLK_FILTER_LINEAR;

  uint i = get_global_id(0);
  uint j = get_global_id(1);
  uint Nx = get_global_size(0);
  uint Ny = get_global_size(1);

  float4 val = read_imagef(src,
  						   sampler,
						   (float2)(1.f*i/(Nx-1.f),1.f*j/(Ny-1.f)));


  dest[i+Nx*j+offset] = val.x*val.x+val.y*val.y;
}


__kernel void img_to_img_intensity(__read_only image2d_t src,
								   __write_only image3d_t dest, const int zPos){

  const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE| CLK_ADDRESS_CLAMP_TO_EDGE|
  //CLK_FILTER_NEAREST;
  CLK_FILTER_LINEAR;

  uint i = get_global_id(0);
  uint j = get_global_id(1);
  uint Nx = get_global_size(0);
  uint Ny = get_global_size(1);

  float4 val = read_imagef(src,
  						   sampler,
						   (float2)(1.f*i/(Nx-1.f),1.f*j/(Ny-1.f)));


  write_imagef(dest, (int4)(i,j,zPos,0),(float4)(val.x*val.x+val.y*val.y,0.f,0.f,0.f));
}



__kernel void buf_to_buf_field(__global cfloat_t *src,
							   __global cfloat_t *dest, const int offset){

  int i = get_global_id(0);

  dest[i+offset] = src[i];

}

__kernel void buf_to_buf_intensity(__global cfloat_t *src,
							   __global float *dest, const int offset){

  int i = get_global_id(0);
  float v = cfloat_abs(src[i]);
  
  dest[i+offset] = v*v;


}

