
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cuda_runtime.h>

#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <time.h>
//#include <cmath>

#define uint unsigned int

#define GpuErrorCheck(ans) { GpuAssert((ans), __FILE__, __LINE__); }
inline void GpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

int DivRound(int n, int m) { return (n + m - 1) / m; }

#if 0

__global__ void Histogram(int* input, int* histogram, int* offset)
{
	__shared__ int lds[16];

	if (threadIdx.x < 16)
	{
		lds[threadIdx.x] = 0;
	}

	__syncthreads();

	int old = atomicAdd(lds + input[threadIdx.x], 1);
	offset[threadIdx.x] = old;

	__syncthreads();

	if (threadIdx.x < 16)
	{
		histogram[threadIdx.x] = lds[threadIdx.x];
	}
}

__global__ void PrefixScan16(int* num)
{
	int laneId = threadIdx.x;
	int v = num[laneId];

	int v1 = __shfl_sync(0xffffffff, v, laneId - 1);
	v = v1;

	if (laneId == 0) { v = 0; }

	v1 = __shfl_sync(0xffffffff, v, laneId - 1);
	if (laneId > 0) { v += v1; }

	v1 = __shfl_sync(0xffffffff, v, laneId - 2);
	if (laneId > 1) { v += v1; }

	v1 = __shfl_sync(0xffffffff, v, laneId - 4);
	if (laneId > 3) { v += v1; }

	v1 = __shfl_sync(0xffffffff, v, laneId - 8);
	if (laneId > 7) { v += v1; }

	num[laneId] = v;
}

__global__ void Reorder(int* input, int* output, int* orderBuffer, int* numOffset)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int v = input[x];
	int idx = orderBuffer[v] + numOffset[x];
	output[idx] = v;
}

#endif

__global__ void RadixSort(uint * inout)
{
	__shared__ uint lds[16];
	__shared__ uint temp[256];

	uint num = inout[blockIdx.x * blockDim.x + threadIdx.x];
	uint temp[threadIdx.x] = num;

	for (uint bitOffset = 0; bitOffset < 32; bitOffset += )
	{

	}
	uint bits = (num & (15u << bitOffset)) >> bitOffset;

	if (threadIdx.x < 16) { lds[threadIdx.x] = 0; }

	__syncthreads();

	// count
	uint offset = atomicAdd(lds + bits, 1u);

	__syncthreads();

	// prefix scan
	if (threadIdx.x < 32)
	{
		uint laneId = threadIdx.x;
		uint v = (laneId < 16) ? lds[laneId] : 0;

		uint v1 = __shfl_sync(0xffffffff, v, laneId - 1); v = v1;
		if (laneId == 0) { v = 0; }
		v1 = __shfl_sync(0xffffffff, v, laneId - 1); if (laneId > 0) { v += v1; }
		v1 = __shfl_sync(0xffffffff, v, laneId - 2); if (laneId > 1) { v += v1; }
		v1 = __shfl_sync(0xffffffff, v, laneId - 4); if (laneId > 3) { v += v1; }
		v1 = __shfl_sync(0xffffffff, v, laneId - 8); if (laneId > 7) { v += v1; }

		if (laneId < 16) { lds[laneId] = v; }
	}

	__syncthreads();

	// reorder
	uint idx = lds[bits] + offset;
	inout[blockIdx.x * blockDim.x + idx] = num;
}

__global__ void MergeSortedArray(uint * num)
{
	__shared__ uint lds[512];
	uint i = threadIdx.x;

	lds[i] = num[i];
	lds[511 - i] = num[256 + i];
	__syncthreads();

	uint v1, v2, v3, v4, idx1, idx2;

#pragma unroll
	for (uint n = 256; n >= 1; n /= 2)
	{
		idx1 = (i / n) * (2 * n) + (i % n);
		idx2 = idx1 + n;

		v1 = lds[idx1];
		v2 = lds[idx2];

		v3 = min(v1, v2);
		v4 = max(v1, v2);

		__syncthreads();

		lds[idx1] = v3;
		lds[idx2] = v4;
		__syncthreads();
	}

	num[i] = lds[i];
	num[i + 256] = lds[i + 256];
}

int main()
{
	srand(time(NULL));

	// create cpu buffer
	uint numCount = 256;
	uint* h_num = new uint[numCount];
	for (uint i = 0; i < numCount; ++i) { h_num[i] = rand() % (uint)pow(2, 8); }

	std::cout << "input:\n";
	for (uint i = 0; i < numCount; ++i)
	{
		std::cout << h_num[i] << ",";
	}
	std::cout << "\n\n";

	// create gpu buffer
	uint* d_num;
	GpuErrorCheck(cudaMalloc((void**)& d_num, numCount * sizeof(uint)));

	// copy from cpu to gpu
	GpuErrorCheck(cudaMemcpy(d_num, h_num, numCount * sizeof(uint), cudaMemcpyHostToDevice));

	// dispatch kernel
	RadixSort << <DivRound(numCount, 256), 256 >> > (d_num);

	GpuErrorCheck(cudaDeviceSynchronize());
	GpuErrorCheck(cudaPeekAtLastError());

	//MergeSortedArray <<<DivRound(numCount, 512), 256>>> (d_num);
	//GpuErrorCheck(cudaDeviceSynchronize());
	//GpuErrorCheck(cudaPeekAtLastError());

	// copy from gpu to cpu
	GpuErrorCheck(cudaMemcpy(h_num, d_num, numCount * sizeof(uint), cudaMemcpyDeviceToHost));

	std::cout << "output:\n";
	for (uint i = 0; i < numCount; ++i)
	{
		std::cout << h_num[i] << ",";
	}
	std::cout << "\n\n";

	// delete
	delete[] h_num;
	cudaFree(d_num);

	return 0;
}