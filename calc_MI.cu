#include <stdio.h>
#include "mex.h"
#include <algorithm>
#include "stdint.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_functions.h"

cudaError_t calcCuda(float *output, const uint8_t *input, const size_t *size);

//********************************************************************************************\\ 
static cudaDeviceProp deviceProperties_;
/*__device__ void computejoint_Kernel(float *mis, uint8_t *inputA, uint8_t *inputB, int colCount, int countNZA, int countNZB)
{
  int i = threadIdx.x;
  int a = i % 2,
    b = i / 2, j = 0;
  for (int k = 0; k < colCount; k++, inputA += colCount, inputB += colCount)
  {
    if (*inputA == a && *inputB == b)
      j++;
  }
  if (j == 0) return;
  j /= colCount;
  if (a) a = countNZA;
  else a = colCount - countNZA;
  if (b) b = countNZB;
  else b = colCount - countNZB;

  mis[i] = j * log2(j / (a / colCount) / (b / colCount));
}
*/
__device__ float computejoint_Kernel(uint8_t *inputA, uint8_t *inputB, int colCount, int countNZA, int countNZB)
{
  float res = 0;
  int joints[2][2] = { 0 };
  for (int k = 0; k < colCount; k++, inputA += colCount, inputB += colCount)
  {
    joints[*inputA][*inputB]++;
  }

  for (size_t i = 0; i < 4; i++)
  {
    int a = i % 2,
      b = i / 2;
    float j = joints[a][b];
    if (j == 0)
      continue;
    j /= colCount;
    if (a) a = countNZA;
    else a = colCount - countNZA;
    if (b) b = countNZB;
    else b = colCount - countNZB;
    
    res += j * log2f(j / ((float)a / colCount) / ((float)b / colCount));
  }
  return res;
}

//********************************************************************************************\\ 
/*__global__ void computeMI1_Kernel(float *MIs, uint8_t *input, int varA, int rowCount, int colCount, int *countNZ)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i > varA) return;
  float mis[4] = { 0 };
  computejoint_Kernel << <1, 4 >> > (MIs, input + varA, input + i, colCount, 
    countNZ[varA], countNZ[i]);
  cudaDeviceSynchronize();

  MIs[+i * colCount + varA] = mis[0];

}
*/
__device__ void computeMI1_Kernel(float *MIs, uint8_t *input, int varA, int rowCount, int colCount, int *countNZ)
{
  MIs += varA;
  for (size_t i = 0; i < varA; i++, MIs += colCount)
  {
    *MIs = computejoint_Kernel(input + varA, input + i, colCount,
      countNZ[varA], countNZ[i]);
  }
}

//********************************************************************************************\\ 
__global__ void computeMI_Kernel(float *MIs, uint8_t *input, int rowCount, int colCount, int *countNZ, int offset)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x + offset;
  if (i > rowCount*(rowCount-1)/2) return;
  int joints[2][2] = { 0 };
  int countNZA , countNZB, a, b, j, k;
  float joint;
  uint8_t *inputA = 0,
    *inputB = 0;

  MIs += i;
  a = 1;
  b = 0;
  for (j = rowCount - 1; j > 1; j--)
  {
    if (i < j) break;

    a++;
    b++;
    i -= j;
  }
  j = b;
  i += a;

//  MIs += i + j*rowCount;
  *MIs = 0;
 // *MIs = i * 1000 + j;
  
//  for (j = 0; j < i; j++, MIs += colCount)
  {
    
    inputA = input + i;
    inputB = input + j;
    countNZA = countNZ[i];
    countNZB = countNZ[j];
    for (k = 0; k < colCount; k++, inputA += rowCount, inputB += rowCount)
    {
      joints[*inputA][*inputB]++;
    }

    for (k = 0; k < 4; k++)
    {
      a = k % 2;
      b = k / 2;

      joint = joints[a][b];
      if (joint == 0)
        continue;
      joint /= colCount;
      if (a) a = countNZA;
      else a = colCount - countNZA;
      if (b) b = countNZB;
      else b = colCount - countNZB;

      *MIs += joint * log2f(joint / ((float)a / colCount) / ((float)b / colCount));
    }
  }
/*  size_t i, t_count, b_count;
  t_count = rowCount > deviceProperties_.maxThreadsPerBlock ? deviceProperties_.maxThreadsPerBlock : rowCount;
  b_count = rowCount / deviceProperties_.maxThreadsPerBlock + 1;
  computeMI_Kernel << <b_count, t_count >> > (MIs, input, i, rowCount, colCount, countNZ)
*/
}//********************************************************************************************\\ 
__global__ void countNZ(int *countNZ, uint8_t *input, int rowCount, int colCount)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i > rowCount) return;
  int nz = 0;
  uint8_t *p = input + i;
  for (int j = 0; j < colCount; j++, p+=rowCount)
  {
    if (*p)
      nz++;
  }
  countNZ[i] = nz;
}

//********************************************************************************************\\ 
//********************************************************************************************\\ 
//int main()

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
/*  uint8_t *dev_a = 0;
  float *dev_c = 0;
  cudaError_t cudaStatus1;

  // Choose which GPU to run on, change this on a multi-GPU system.
  cudaStatus1 = cudaSetDevice(0);
  if (cudaStatus1 != cudaSuccess) 
  {
    printf("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    return ;
  }

  // Allocate GPU buffers for three vectors (two input, one output)    .
  cudaStatus1 = cudaMalloc((void**)&dev_c, 10 * sizeof(float));
  if (cudaStatus1 != cudaSuccess)
  {
    printf("cudaMalloc failed, size %d, Error %d!", 10 * sizeof(float), cudaStatus1);
    return ;
  }
  printf("AAA");
  return ;

  */
  cudaGetDeviceProperties(&deviceProperties_, 0);
	  
  printf("Starting over %s\n", deviceProperties_.name);
  
  uint8_t *dMatrix;               
	float *outMatrixf;               

  if (!mxIsUint8(prhs[0])) {
    mexErrMsgTxt("Input argument is not uint8 class.");
  }
  dMatrix = (uint8_t *)mxGetData(prhs[0]);
  const size_t * pSize = mxGetDimensions(prhs[0]);
  int outSize = pSize[0] * (pSize[0] - 1) / 2;
  
  outMatrixf = new float[outSize];

	// Add vectors in parallel.
	cudaError_t cudaStatus;
  cudaStatus = calcCuda(outMatrixf, dMatrix, pSize);
  if (cudaStatus != cudaSuccess) {
      printf( "addWithCuda failed!");
      return ;
  }
  
  cudaStatus = cudaDeviceReset();
  if (cudaStatus != cudaSuccess) {
      printf( "cudaDeviceReset failed!");
      return ;
  }

  plhs[0] = mxCreateDoubleMatrix(1, outSize, mxREAL);
  double *tempOut = mxGetPr(plhs[0]);
  std::copy(outMatrixf, outMatrixf + outSize, tempOut);
  printf("Done\n");
}
//********************************************************************************************\\ 
//********************************************************************************************\\ 
//********************************************************************************************\\ 
//********************************************************************************************\\ 

cudaError_t calcCuda(float *output, const uint8_t *input, const size_t *size)
{
	
	uint8_t *dev_a = 0;
  int * dev_nz = 0;
  float *dev_o = 0;
  cudaError_t cudaStatus;
  size_t totalSize = size[0] * size[1];
  size_t outSize = size[0] * (size[0] - 1) / 2;
  cudaEvent_t start, stop;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // Choose which GPU to run on, change this on a multi-GPU system.
  cudaStatus = cudaSetDevice(0);
  if (cudaStatus != cudaSuccess) {
      printf( "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
      goto Error;
  }

    // Allocate GPU buffers for three vectors (two input, one output)    .
  cudaStatus = cudaMalloc((void**)&dev_nz, size[0] * sizeof(int));
  if (cudaStatus != cudaSuccess)
  {
    printf("cudaMalloc failed, size %d, Error %d!", size[0] * sizeof(int), cudaStatus);
    goto Error;
  }
  cudaStatus = cudaMalloc((void**)&dev_o, outSize * sizeof(float));
  if (cudaStatus != cudaSuccess)
  {
    printf("cudaMalloc failed, size %d, Error %d!", outSize * sizeof(float), cudaStatus);
    goto Error;
  }

  cudaStatus = cudaMalloc((void**)&dev_a, totalSize * sizeof(uint8_t));
  if (cudaStatus != cudaSuccess) 
  {
    printf("cudaMalloc failed, size %d, Error %d!", totalSize * sizeof(uint8_t), cudaStatus);
    goto Error;
  }

  // Copy input vectors from host memory to GPU buffers.
  cudaStatus = cudaMemcpy(dev_a, input, totalSize * sizeof(uint8_t), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
      printf( "cudaMemcpy failed!");
      goto Error;
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  printf("Time for the MemCopy: %f s\n", time/1000);

  // Launch a kernel on the GPU with one thread for each element.
  cudaEventRecord(start, 0);
  
  size_t i, t_count, b_count;
	t_count = size[0] > deviceProperties_.maxThreadsPerBlock ? deviceProperties_.maxThreadsPerBlock : size[0];
	b_count = size[0] / deviceProperties_.maxThreadsPerBlock + 1;

  countNZ << <b_count, t_count >> > (dev_nz, dev_a, size[0], size[1]);
  // Check for any errors launching the kernel
  cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    printf("addKernel1 launch failed: %s\n", cudaGetErrorString(cudaStatus));
    goto Error;
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  printf("Time for the Count NZ: %f s\n", time/1000);
  
  // cudaDeviceSynchronize waits for the kernel to finish, and returns
  // any errors encountered during the launch.
  cudaStatus = cudaDeviceSynchronize();
  if (cudaStatus != cudaSuccess) {
    printf("cudaDeviceSynchronize1 returned error code %d after launching addKernel!\n", cudaStatus);
    goto Error;
  }

  int k = (size[0])*(size[0] - 1) / 2;
  int j = 0;
#define CYCLE_COUNT 1000000
  while (k > j)
  {
    cudaEventRecord(start, 0);
    b_count = min(k - j, CYCLE_COUNT);
    t_count = b_count > deviceProperties_.maxThreadsPerBlock ? deviceProperties_.maxThreadsPerBlock : b_count;
    b_count = b_count / deviceProperties_.maxThreadsPerBlock + 1;
    computeMI_Kernel << <b_count, t_count >> > (dev_o, dev_a, size[0], size[1], dev_nz, j);
    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
      printf("addKernel2 launch failed: %s\n", cudaGetErrorString(cudaStatus));
      goto Error;
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Time for the Calc MI: %f s\n", time / 1000);
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
      printf("cudaDeviceSynchronize2 returned error code %d after launching addKernel!\n", cudaStatus);
      goto Error;
    }
    j += CYCLE_COUNT;
    
  }
    // Copy output vector from GPU buffer to host memory.
  cudaStatus = cudaMemcpy(output, dev_o, outSize * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        printf( "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_nz);
    cudaFree(dev_o);
    cudaFree(dev_a);
  return cudaStatus;
}
