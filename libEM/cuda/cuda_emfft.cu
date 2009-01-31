

#include <cufft.h>
#include <cuda_runtime_api.h>
#include "cuda_defs.h"
#include "cuda_util.h"

#define FFTW_PLAN_CACHING 1

#ifdef FFTW_PLAN_CACHING

const int EMCUDA_FFT_CACHE_SIZE = 10;
int EMCUDA_FFT_CACHE_NUM_PLANS = 0;

struct cufft_plan_cache {
	int rank; // Store the rank of the plan
	int plan_dims[3]; // Store the dimensions of the plan (always in 3D, if dimensions are "unused" they are taken to be 1)
	int r2c; // store whether the plan was real to complex or vice versa
	cufftHandle handle;  // Store the plans themselves
	int ip; // Store whether or not the plan was inplace
	cufftReal* real; // Device real pointer
	cufftComplex* complex; // Device complex pointer 
	
};

cufft_plan_cache CudaFftPlanCache[EMCUDA_FFT_CACHE_SIZE];

void reset_cuda_fft_cache(cufft_plan_cache* c) {
	c->rank = 0;
	c->plan_dims[0] = 0; c->plan_dims[1] = 0; c->plan_dims[2] = 0;
	c->r2c = -1;
	c->ip = -1;
	c->handle = 0;
	if (c->real != 0) {
		CUDA_SAFE_CALL(cudaFree(c->real));
		c->real = 0;
	}
	if (c->complex != 0) {
		CUDA_SAFE_CALL(cudaFree(c->complex));
		c->complex = 0;
	}
	
	
}

void copy_cuda_fft_cache(cufft_plan_cache* to,  cufft_plan_cache* from) {
	to->rank = from->rank;
	to->plan_dims[0] = from->plan_dims[0];
	to->plan_dims[1] = from->plan_dims[1];
	to->plan_dims[2] = from->plan_dims[2];
	to->r2c = from->r2c;
	to->ip = from->ip;
	to->handle = from->handle;
	to->real = from->real;
	to->complex = from->complex;
}

void init_cuda_emfft_cache() {
	for(int i = 0; i < EMCUDA_FFT_CACHE_SIZE; ++i)
	{
		reset_cuda_fft_cache(&(CudaFftPlanCache[i]));
	}
}



void cleanup_cuda_emfft_cache() {
	
	for(int i = 0; i < EMCUDA_FFT_CACHE_SIZE; ++i)
	{
		if (CudaFftPlanCache[i].handle != 0) {
			CUDA_SAFE_CALL(cufftDestroy(CudaFftPlanCache[i].handle));
			CudaFftPlanCache[i].handle = 0;
		}
	}
}

int real_2_complex = 1;
int complex_2_real = 2;

cufft_plan_cache* get_cuda_emfft_plan(const int rank_in, const int x, const int y, const int z, const int r2c_flag, const int ip_flag) {

	if ( rank_in > 3 || rank_in < 1 ) throw; //InvalidValueException(rank_in, "Error, can not get an FFTW plan using rank out of the range [1,3]")
	if ( r2c_flag != real_2_complex && r2c_flag != complex_2_real ) throw; //InvalidValueException(r2c_flag, "The real two complex flag is not supported");
	
	// First check to see if we already have the plan
	int i;
	for (i=0; i<EMCUDA_FFT_CACHE_NUM_PLANS; i++) {
		cufft_plan_cache* c = &CudaFftPlanCache[i];
		if (c->plan_dims[0]==x && c->plan_dims[1]==y && c->plan_dims[2]==z 
				  && c->rank==rank_in && c->r2c==r2c_flag && c->ip==ip_flag) {
			//printf("Already have that cache\n");
			return c;
		}
	}
	//printf("Returning a new cache\n");
	cufftHandle plan;
	cufftReal* real; // Device real pointer
	cufftComplex* complex; // Device complex pointer 
	
	if (r2c_flag == complex_2_real) {
		int x2 = x + 2;
		int complex_mem_size = sizeof(cufftComplex) * x2 * y * z/2;
		int real_mem_size = sizeof(cufftReal) * x2 * y * z;
		CUDA_SAFE_CALL(cudaMalloc((void**)&complex, complex_mem_size));
		CUDA_SAFE_CALL(cudaMalloc((void**)&real, real_mem_size));
	} else {
		int offset = 2 - x%2;
		int x2 = x + offset;
		int complex_mem_size = sizeof(cufftComplex) * x2 * y * z/2;
		int real_mem_size = sizeof(cufftReal) * x * y * z;
		CUDA_SAFE_CALL(cudaMalloc((void**)&complex, complex_mem_size));
		CUDA_SAFE_CALL(cudaMalloc((void**)&real, real_mem_size));
	}
	// Create the plan
	if ( y == 1 && z == 1 )
	{
		if ( r2c_flag == real_2_complex ) {
			cufftPlan1d(&plan,x,CUFFT_R2C,1);
		}
		else { // r2c_flag == complex_2_real, this is guaranteed by the error checking at the beginning of the function
			cufftPlan1d(&plan, x, CUFFT_C2R,1);
		}
	}
	else if ( z == 1 )
	{
		if ( r2c_flag == real_2_complex ) {
			cufftPlan2d(&plan,x,y,CUFFT_R2C);
		}
		else // r2c_flag == complex_2_real, this is guaranteed by the error checking at the beginning of the function
			cufftPlan2d(&plan,x,y,CUFFT_C2R);
	}
	else /* 3D */ {
		if ( r2c_flag == real_2_complex ) {
			cufftPlan3d(&plan,x,y,z,CUFFT_R2C);
		}
		else // r2c_flag == complex_2_real, this is guaranteed by the error checking at the beginning of the function
			cufftPlan3d(&plan,x,y,z,CUFFT_C2R);
	}

	if (CudaFftPlanCache[EMCUDA_FFT_CACHE_SIZE-1].handle != 0 )
	{
		cufftDestroy(CudaFftPlanCache[EMCUDA_FFT_CACHE_SIZE-1].handle);
		reset_cuda_fft_cache(&(CudaFftPlanCache[EMCUDA_FFT_CACHE_SIZE-1]));
	}
				
	int upper_limit = EMCUDA_FFT_CACHE_NUM_PLANS;
	if ( upper_limit == EMCUDA_FFT_CACHE_SIZE ) upper_limit -= 1;
	
	for (int i=upper_limit-1; i>0; i--)
	{
		copy_cuda_fft_cache(&(CudaFftPlanCache[i]),&(CudaFftPlanCache[i-1]));
		
	}
		//dimplan[0]=-1;
	
	cufft_plan_cache* c = &CudaFftPlanCache[0];
	
	c->plan_dims[0]=x;
	c->plan_dims[1]=y;
	c->plan_dims[2]=z;
	c->r2c=r2c_flag;
	c->ip=ip_flag;
	c->handle = plan;
	c->rank =rank_in;
	c->complex = complex;
	c->real = real;
	if (EMCUDA_FFT_CACHE_NUM_PLANS<EMCUDA_FFT_CACHE_SIZE) EMCUDA_FFT_CACHE_NUM_PLANS++;

	return c;
}
#endif // FFTW_PLAN_CACHING

int get_rank(int ny, int nz)
{
	int rank = 3;
	if (ny == 1) {
		rank = 1;
	}
	else if (nz == 1) {
		rank = 2;
	}
	return rank;
}

int cuda_fft_real_to_complex_1d(float *real_data, float *complex_data, int n)
{
	device_init();
#ifdef FFTW_PLAN_CACHING
	bool ip = false;
	int offset = 2 - n%2;
	int n2 = n + offset;
	int complex_mem_size = sizeof(cufftComplex) * n2/2;
	int real_mem_size = sizeof(cufftReal) * n;
	cufft_plan_cache* cache =  get_cuda_emfft_plan(1,n,1,1,real_2_complex,ip);
	CUDA_SAFE_CALL(cudaMemcpy(cache->real,real_data, real_mem_size, cudaMemcpyHostToDevice));
	cufftExecR2C(cache->handle, cache->real, cache->complex );
	CUDA_SAFE_CALL(cudaMemcpy(complex_data,cache->complex, complex_mem_size, cudaMemcpyDeviceToHost));
	//cufftExecR2C(plan, (cufftReal*)real_data,(cufftComplex *) complex_data);
#else
	cufftHandle plan;
	cufftPlan1d(&plan,n,CUFFT_R2C,1);
	cufftExecR2C(plan,(cufftReal*)real_data,(cufftComplex *) complex_data);
	cufftDestroy(plan);
#endif // FFTW_PLAN_CACHING
	return 0;
};

int cuda_fft_complex_to_real_1d(float *complex_data, float *real_data, int n)
{
	device_init();
#ifdef FFTW_PLAN_CACHING
	//bool ip = ( complex_data == real_data );
	bool ip = false;
	int offset = 2 - n%2;
	int n2 = n + offset;
	int complex_mem_size = sizeof(cufftComplex) * n2/2;
	int real_mem_size = sizeof(cufftReal) * n2;
	cufft_plan_cache* cache = get_cuda_emfft_plan(1,n,1,1,complex_2_real,ip);
	CUDA_SAFE_CALL(cudaMemcpy(cache->complex,complex_data, complex_mem_size, cudaMemcpyHostToDevice));
	cufftExecC2R(cache->handle, cache->complex, cache->real);
	CUDA_SAFE_CALL(cudaMemcpy(real_data,cache->real, real_mem_size, cudaMemcpyDeviceToHost));
#else
	cufftHandle plan;
	cufftPlan1d(&plan,n,CUFFT_C2C,1);
	cufftExecC2R(plan,(cufftComplex*)complex_data,(cufftReal *) real_data);
	cufftDestroy(plan);
#endif // FFTW_PLAN_CACHING
	
	return 0;
}

int cuda_fft_real_to_complex_nd(float *real_data, float *complex_data, int nx, int ny, int nz)
{
	device_init();
	const int rank = get_rank(ny, nz);
#ifdef FFTW_PLAN_CACHING
	bool ip;
	int offset = 2 - nx%2;
	int nx2 = nx + offset;
	int complex_mem_size = sizeof(cufftComplex) * nx2 * ny * nz/2;
	int real_mem_size = sizeof(cufftReal) * nx * ny * nz;
	cufft_plan_cache* cache = 0;
#endif //FFTW_PLAN_CACHING
	//cufftHandle plan;
	
	switch(rank) {
		case 1:
			cuda_fft_real_to_complex_1d(real_data, complex_data, nx);
			break;
		
		
#ifdef FFTW_PLAN_CACHING
		case 2:
		case 3:
			ip = ( complex_data == real_data );
			//ip = false;
			
			if ( !ip ) {
				cache = get_cuda_emfft_plan(rank,nx,ny,nz,real_2_complex,ip);
				CUDA_SAFE_CALL(cudaMemcpy(cache->real,real_data, real_mem_size, cudaMemcpyHostToDevice));
				cufftExecR2C(cache->handle, cache->real, cache->complex );
				CUDA_SAFE_CALL(cudaMemcpy(complex_data,cache->complex, complex_mem_size, cudaMemcpyDeviceToHost));
			}
			else {
				cache = get_cuda_emfft_plan(rank,nx,ny,nz,real_2_complex,ip);
				CUDA_SAFE_CALL(cudaMemcpy(cache->complex,real_data, complex_mem_size, cudaMemcpyHostToDevice));
				cufftExecR2C(cache->handle, (cufftReal*)cache->complex, cache->complex );
				CUDA_SAFE_CALL(cudaMemcpy(complex_data,cache->complex, complex_mem_size, cudaMemcpyDeviceToHost));
			}
		break;
#else
		case 2:
			cufftPlan2d(&plan,nx,ny,CUFFT_R2C);
			cufftExecR2C(plan, (cufftReal*)real_data,(cufftComplex *) complex_data);
			cufftDestroy(plan);
		break;
		case 3:
			cufftPlan3d(&plan,nx,ny,nz,CUFFT_R2C);
			cufftExecR2C(plan, (cufftReal*)real_data,(cufftComplex *) complex_data);
			cufftDestroy(plan);
		break;
#endif // FFTW_PLAN_CACHING
		
		default:throw;
	}
	
	return 0;
}

int cuda_fft_complex_to_real_nd(float *complex_data, float *real_data, int nx, int ny, int nz)
{
	device_init();
	const int rank = get_rank(ny, nz);
	
#ifdef FFTW_PLAN_CACHING
	bool ip;
	int offset = 2 - nx%2;
	int nx2 = nx + offset;
	//printf("using nx2 %d\n",nx2);
	int complex_mem_size = sizeof(cufftComplex) * nx2 * ny * nz/2;
	int real_mem_size = sizeof(cufftReal) * nx2 * ny * nz;
	cufft_plan_cache* cache = 0;
#endif
	//cufftHandle plan;
	switch(rank) {
		case 1:
			cuda_fft_complex_to_real_1d(complex_data, real_data, nx);
			break;
		
#ifdef FFTW_PLAN_CACHING
		case 2:
		case 3:
			ip = ( complex_data == real_data );
			cache = get_cuda_emfft_plan(rank,nx,ny,nz,complex_2_real,ip);
			CUDA_SAFE_CALL(cudaMemcpy(cache->complex,complex_data, complex_mem_size, cudaMemcpyHostToDevice));
			cufftExecC2R(cache->handle, cache->complex, cache->real);
			CUDA_SAFE_CALL(cudaMemcpy(real_data,cache->real, real_mem_size, cudaMemcpyDeviceToHost));
			break;
#else
		case 2:
			cufftPlan2d(&plan,nx,ny,CUFFT_C2C);
			cufftExecC2R(plan, (cufftComplex*)complex_data,(cufftReal *) real_data);
			cufftDestroy(plan);
			break;
		case 3:
			cufftPlan3d(&plan,nx,ny,nz,CUFFT_C2C);
			cufftExecC2R(plan, (cufftComplex*)complex_data, (cufftReal *) real_data);
			cufftDestroy(plan);
			break;
#endif // FFTW_PLAN_CACHING
			
		default:throw;
	}
	
	return 0;
}

