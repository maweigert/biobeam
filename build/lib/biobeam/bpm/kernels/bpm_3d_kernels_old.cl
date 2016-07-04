#include <pyopencl-complex.h>

#define M_PI 3.14159265358979f


#define FFTFREQ(i,N,dx) ((i<((N-1)/2+1))?1.f*i/N/dx:1.f*((i-((N-1)/2+1))-N/2)/N/dx)




__kernel void mult(__global cfloat_t* a,
				   __global cfloat_t* b){

  uint i = get_global_id(0);
    
  a[i] = cfloat_mul(a[i], b[i]);

}

__kernel void mult_dn(__global cfloat_t* input,
					  __global float* dn,const float unit_k,
					  const int stride,
					  const int absorb){

  int i = get_global_id(0);
  int j = get_global_id(1);
  int Nx = get_global_size(0);
  int Ny = get_global_size(1);



  float dnDiff = -unit_k*dn[i+Nx*j+stride];

  int distx = min(Nx-i-1,i);
  int disty = min(Ny-j-1,j);
  int dist = min(distx,disty);

  float absorb_val = (dist<absorb)?0.5f*(1-cos(M_PI*dist/absorb)):1.f;

  cfloat_t dPhase = cfloat_new(cos(dnDiff),sin(dnDiff));

  cfloat_t res = cfloat_mul(input[i+Nx*j],dPhase);

  res = cfloat_mul(res,cfloat_new(absorb_val,0.f));


  input[i+Nx*j] = res;

}


__kernel void mult_dn_half(__global cfloat_t* input,
					  __global half * dn,const float unit_k,
					  const int stride,
					  const int absorb){

  int i = get_global_id(0);
  int j = get_global_id(1);
  int Nx = get_global_size(0);
  int Ny = get_global_size(1);


  float dn_val = vload_half(i+Nx*j+stride,dn);
  float dnDiff = -unit_k*dn_val;

  int distx = min(Nx-i-1,i);
  int disty = min(Ny-j-1,j);
  int dist = min(distx,disty);

  float absorb_val = (dist<absorb)?0.5f*(1.f-cos(M_PI*dist/absorb)):1.f;

  cfloat_t dPhase = cfloat_new(cos(dnDiff),sin(dnDiff));

  cfloat_t res = cfloat_mul(input[i+Nx*j],dPhase);

  res = cfloat_mul(res,cfloat_new(absorb_val,0.f));


  input[i+Nx*j] = res;

}


__kernel void mult_dn_complex(__global cfloat_t* input,
					  __global cfloat_t* dn,
					  const float unit_k,
					  const int stride,
					  const int absorb){

  int i = get_global_id(0);
  int j = get_global_id(1);
  int Nx = get_global_size(0);

  cfloat_t dnDiff = cfloat_mul(cfloat_new(0.f,-unit_k),dn[i+Nx*j+stride]);

  cfloat_t dPhase = cfloat_exp(dnDiff);


  input[i+Nx*j] = cfloat_mul(input[i+Nx*j],dPhase);

}


__kernel void mult_dn_mean_buff(__global cfloat_t* input,
						   __global float* dn,
                           __global float* dn_sum,
                           __global float* u_sum,
						   const float unit_k,
						   const int stride,
						   const int absorb){

  int i = get_global_id(0);
  int j = get_global_id(1);
  int Nx = get_global_size(0);
  int Ny = get_global_size(1);

  float dn0 = dn_sum[0]/u_sum[0];

  float dnDiff = -unit_k*(dn[i+Nx*j+stride]-dn0);

  int distx = min(Nx-i-1,i);
  int disty = min(Ny-j-1,j);
  int dist = min(distx,disty);

  float absorb_val = (dist<absorb)?0.5f*(1-cos(M_PI*dist/absorb)):1.f;

  cfloat_t dPhase = cfloat_new(cos(dnDiff),sin(dnDiff));

  cfloat_t res = cfloat_mul(input[i+Nx*j],dPhase);

  res = cfloat_mul(res,cfloat_new(absorb_val,0.f));


  input[i+Nx*j] = res;

}


__kernel void mult_dn_mean_half_buff(__global cfloat_t* input,
					  __global half * dn,
                      __global float* dn_sum,
                      __global float* u_sum,
					  const float unit_k,
					  const int stride,
					  const int absorb){

  int i = get_global_id(0);
  int j = get_global_id(1);
  int Nx = get_global_size(0);
  int Ny = get_global_size(1);

  float dn0 = dn_sum[0]/u_sum[0];
  float dn_val = vload_half(i+Nx*j+stride,dn);
  float dnDiff = -unit_k*(dn_val-dn0);

  int distx = min(Nx-i-1,i);
  int disty = min(Ny-j-1,j);
  int dist = min(distx,disty);

  float absorb_val = (dist<absorb)?0.5f*(1.f-cos(M_PI*dist/absorb)):1.f;

  cfloat_t dPhase = cfloat_new(cos(dnDiff),sin(dnDiff));

  cfloat_t res = cfloat_mul(input[i+Nx*j],dPhase);

  res = cfloat_mul(res,cfloat_new(absorb_val,0.f));


  input[i+Nx*j] = res;

}


__kernel void mult_dn_mean_complex_buff(__global cfloat_t* input,
					  __global cfloat_t* dn,
                      __global cfloat_t* dn_sum,
                      __global float* u_sum,

					  const float unit_k,
					  const int stride,
					  const int absorb){

  int i = get_global_id(0);
  int j = get_global_id(1);
  int Nx = get_global_size(0);

  cfloat_t dn0 = cfloat_divider(dn_sum[0],u_sum[0]);
  cfloat_t dnDiff = cfloat_mul(cfloat_new(0.f,-unit_k),cfloat_sub(dn[i+Nx*j+stride],dn0));

  cfloat_t dPhase = cfloat_exp(dnDiff);


  input[i+Nx*j] = cfloat_mul(input[i+Nx*j],dPhase);

}


__kernel void mult_dn_mean(__global cfloat_t* input,
						   __global float* dn,const float unit_k,
						   const float dn0,
						   const int stride,
						   const int absorb){

  int i = get_global_id(0);
  int j = get_global_id(1);
  int Nx = get_global_size(0);
  int Ny = get_global_size(1);

  // if(i+j>0)
  // 	printf("haiuhu\n");

  float dnDiff = -unit_k*(dn[i+Nx*j+stride]-dn0);

  int distx = min(Nx-i-1,i);
  int disty = min(Ny-j-1,j);
  int dist = min(distx,disty);

  float absorb_val = (dist<absorb)?0.5f*(1-cos(M_PI*dist/absorb)):1.f;

  cfloat_t dPhase = cfloat_new(cos(dnDiff),sin(dnDiff));

  cfloat_t res = cfloat_mul(input[i+Nx*j],dPhase);

  res = cfloat_mul(res,cfloat_new(absorb_val,0.f));


  input[i+Nx*j] = res;

}


__kernel void mult_dn_mean_half(__global cfloat_t* input,
					  __global half * dn,const float unit_k,
								const float dn0,
					  const int stride,
					  const int absorb){

  int i = get_global_id(0);
  int j = get_global_id(1);
  int Nx = get_global_size(0);
  int Ny = get_global_size(1);


  float dn_val = vload_half(i+Nx*j+stride,dn);
  float dnDiff = -unit_k*(dn_val-dn0);

  int distx = min(Nx-i-1,i);
  int disty = min(Ny-j-1,j);
  int dist = min(distx,disty);

  float absorb_val = (dist<absorb)?0.5f*(1.f-cos(M_PI*dist/absorb)):1.f;

  cfloat_t dPhase = cfloat_new(cos(dnDiff),sin(dnDiff));

  cfloat_t res = cfloat_mul(input[i+Nx*j],dPhase);

  res = cfloat_mul(res,cfloat_new(absorb_val,0.f));


  input[i+Nx*j] = res;

}


__kernel void mult_dn_mean_complex(__global cfloat_t* input,
					  __global cfloat_t* dn,
					  const float unit_k,
								   const cfloat_t dn0,
					  const int stride,
					  const int absorb){

  int i = get_global_id(0);
  int j = get_global_id(1);
  int Nx = get_global_size(0);

  cfloat_t dnDiff = cfloat_mul(cfloat_new(0.f,-unit_k),cfloat_sub(dn[i+Nx*j+stride],dn0));

  cfloat_t dPhase = cfloat_exp(dnDiff);


  input[i+Nx*j] = cfloat_mul(input[i+Nx*j],dPhase);

}



__kernel void mult_dn_image(__global cfloat_t* input,
							__read_only image3d_t dn,
							const float unit_k,
							const float n0,
							const int zpos,
							const int subsample){

  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE| CLK_ADDRESS_CLAMP_TO_EDGE| CLK_FILTER_NEAREST;

  uint i = get_global_id(0);
  uint j = get_global_id(1);
  uint Nx2 = get_global_size(0);

  float dn_val = read_imagef(dn, sampler, (float4)(1.f*i/subsample,1.f*j/subsample,1.f*zpos/subsample,0)).x;

  float dnDiff = -unit_k*dn_val;

  // dnDiff = -unit_k*dn_val*(1.f+.5f*dn_val/n0);
  
  cfloat_t dPhase = cfloat_new(cos(dnDiff),sin(dnDiff));

  input[i+Nx2*j] = cfloat_mul(input[i+Nx2*j],dPhase);



  
}




__kernel void mult_dn_complex_image(__global cfloat_t* input,
									__read_only image3d_t dn,
									const float unit_k,
									const float n0,

									const int zpos,
									const int subsample){

  const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE| CLK_ADDRESS_CLAMP_TO_EDGE| CLK_FILTER_LINEAR;

  uint i = get_global_id(0);
  uint j = get_global_id(1);
  uint Nx = get_global_size(0);

  float2 dn_val = read_imagef(dn, sampler, (float4)(1.f*i/subsample,1.f*j/subsample,1.f*zpos/subsample,0)).xy;
  
  cfloat_t dnDiff = cfloat_mul(cfloat_new(0,-unit_k),cfloat_new(dn_val.x,dn_val.y));
  cfloat_t dPhase = cfloat_exp(dnDiff);
  
  input[i+Nx*j] = cfloat_mul(input[i+Nx*j],dPhase);

  //if ((i==64) &&(j==64))
  //  printf("kernel %.10f \n",dn_val.y);

  //input[i+Nx*j] = cfloat_new(1.f,0.f);

}




__kernel void divide_dn_complex(__global cfloat_t* plane0,__global cfloat_t* plane1,
					  __global cfloat_t* dn,const float unit_k, const int stride){

  uint i = get_global_id(0);
  
  cfloat_t phase;

  float dn_val;
  // res = cfloat_divide(plane2[i],plane1[i]);

  phase = cfloat_divide(plane0[i],plane1[i]);

  dn_val = atan2(phase.y,phase.x);

  dn_val *= 1./unit_k;

  // dn_val = clamp(dn_val,0.f,4.f);
  cfloat_t res = cfloat_new(dn_val,0.);
  
  dn[i+stride] = res;

  
}


__kernel void copy_subsampled_buffer(__global cfloat_t* buffer,__global cfloat_t* plane,
									 const int subsample,
									 const int stride){

  uint i = get_global_id(0);
  uint j = get_global_id(1);

  uint Nx = get_global_size(0);

  buffer[i+Nx*j+stride] = plane[i*subsample+subsample*subsample*Nx*j];  
}



__kernel void copy_complex(__global cfloat_t* input,__global cfloat_t* plane,
					  const int stride){

  uint i = get_global_id(0);
  plane[i] = input[i+stride];  
}


__kernel void copy_intens(__global cfloat_t* plane,__global float* output,
					  const int stride){

  uint i = get_global_id(0);

  float res = cfloat_abs(plane[i]);

  output[i+stride] = res*res;
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

  float h0 = (tmp<0.f)?0.f:sqrt(tmp);

  
  H[i+Nx*j] = cfloat_new((float)(tmp>=0.f)*cos(-dz*h0),(float)(tmp>=0.f)*sin(-dz*h0));  

}




__kernel void mult_propagator(__global cfloat_t* plane,
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

  float h0 = (tmp<0.f)?0.f:sqrt(tmp);

  
  cfloat_t h = cfloat_new((float)(tmp>=0.f)*cos(-dz*h0),(float)(tmp>=0.f)*sin(-dz*h0));  

  plane[i+Nx*j] = cfloat_mul(plane[i+Nx*j],h);
}


__kernel void mult_propagator_buff(__global cfloat_t* plane,
                                    __global float* dn_sum,
                                    __global float* u_sum,
								 const float n00,
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

  float n0 = n00+dn_sum[0]/u_sum[0];

  float tmp = n0*n0*k0*k0-kx*kx-ky*ky;

  float h0 = (tmp<0.f)?0.f:sqrt(tmp);


  cfloat_t h = cfloat_new((float)(tmp>=0.f)*cos(-dz*h0),(float)(tmp>=0.f)*sin(-dz*h0));

  plane[i+Nx*j] = cfloat_mul(plane[i+Nx*j],h);
}


__kernel void img2d_to_buf_complex(__read_only image2d_t src,
								   __global cfloat_t *dest, const int offset){

  const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE| CLK_ADDRESS_CLAMP_TO_EDGE| CLK_FILTER_NEAREST;

  uint i = get_global_id(0);
  uint j = get_global_id(1);
  uint Nx = get_global_size(0);
  uint Ny = get_global_size(1);

  float4 val = read_imagef(src,
  						   sampler,
						   (float2)(1.f*i/(Nx-1.f),1.f*j/(Ny-1.f)));
  dest[i+Nx*j+offset] = cfloat_new(val.x,val.y);
}


