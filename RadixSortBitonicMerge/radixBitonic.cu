
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cuda_runtime.h>

#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <random>
//#include <cmath>

#define uint unsigned int
#define ushort unsigned short

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

__global__ void RadixSort(uint * inout)
{
	struct LDS
	{
		uint temp[256];
		ushort histo[16 * 8];
		ushort histoScan[16];
	};

	__shared__ LDS lds;

	//------------------------------------ Read data in ----------------------------------------
    lds.temp[threadIdx.x] = inout[blockIdx.x * blockDim.x + threadIdx.x];
	__syncthreads();

	// lane id and warp id
	uint laneId = threadIdx.x & 0x1f;
	uint warpId = (threadIdx.x & 0xffffffe0) >> 5u;

	// loop for each 4 bits
	#pragma unroll
	for (uint bitOffset = 0; bitOffset < 4 * 8; bitOffset += 4)
	{
		// load number
		uint num = lds.temp[threadIdx.x];

		// extract 4 bits
		uint num4bit = (num & (0xf << bitOffset)) >> bitOffset;

		//------------------------------------ Init LDS ----------------------------------------
		if (laneId < 16) { lds.histo[warpId * 16 + laneId] = 0; }
		if (warpId == 0 && laneId < 16) { lds.histoScan[laneId] = 0; }
		__syncthreads();

		//------------------------------------ Warp count and offset ----------------------------------------

		// mask indicates threads having equal value number with current thread
		uint mask = 0xffffffff;
		#pragma unroll
		for (int i = 0; i < 4; ++i)
		{
			uint bitPred = num4bit & (0x1 << i);
			uint maskOne = __ballot_sync(0xffffffff, bitPred);
			mask = mask & (bitPred ? maskOne : ~maskOne);
		}

		// offset of current value number
		uint pos = __popc(mask & (0xffffffff >> (31u - laneId)));

		// count of current value number
		uint count = __popc(mask);

		//------------------------------------ Block count and offset ----------------------------------------

		// Re-arrange data for warp level scan
		if (pos == 1) { lds.histo[warpId * 16 + num4bit] = count; }
		__syncthreads();

		uint v, v1;
		if (laneId < 8) { v = lds.histo[laneId * 16 + warpId]; }
		else if (laneId < 16) { v = lds.histo[(laneId - 8) * 16 + warpId + 8]; }
		__syncthreads();

		// Warp inclusive scan of 0-7, 8-15
		v1 = __shfl_sync(0xffffffff, v, laneId - 1); if ((laneId > 0 && laneId < 8) || laneId > 8) { v += v1; }
		v1 = __shfl_sync(0xffffffff, v, laneId - 2); if ((laneId > 1 && laneId < 8) || laneId > 9) { v += v1; }
		v1 = __shfl_sync(0xffffffff, v, laneId - 4); if ((laneId > 3 && laneId < 8) || laneId > 11) { v += v1; }

		// Write back
		if (laneId < 8) { lds.histo[laneId * 16 + warpId] = v; }
		else if (laneId < 16) { lds.histo[(laneId - 8) * 16 + warpId + 8] = v; }
		__syncthreads();

		//------------------------------------ Warp prefix scan for histogram ----------------------------------------
		if (warpId == 7)
		{
			v = (laneId < 16) ? lds.histo[warpId * 16 + laneId] : 0;

			v1 = __shfl_sync(0xffffffff, v, laneId - 1); v = v1;
			if (laneId == 0) { v = 0; }

			v1 = __shfl_sync(0xffffffff, v, laneId - 1); if (laneId > 0) { v += v1; }
			v1 = __shfl_sync(0xffffffff, v, laneId - 2); if (laneId > 1) { v += v1; }
			v1 = __shfl_sync(0xffffffff, v, laneId - 4); if (laneId > 3) { v += v1; }
			v1 = __shfl_sync(0xffffffff, v, laneId - 8); if (laneId > 7) { v += v1; }

			if (laneId < 16) { lds.histoScan[laneId] = v; }
		}
		__syncthreads();

		//------------------------------------ Reorder ----------------------------------------
		uint idxAllNum          = lds.histoScan[num4bit];
		uint idxCurrentNumBlock = (warpId > 0) ? lds.histo[(warpId - 1) * 16 + num4bit] : 0;
		uint idxCurrentNumWarp  = pos - 1;

		lds.temp[idxAllNum + idxCurrentNumBlock + idxCurrentNumWarp] = num;

		__syncthreads();
	}

	//------------------------------------ Write out ----------------------------------------
	inout[blockIdx.x * blockDim.x + threadIdx.x] = lds.temp[threadIdx.x];
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
	std::random_device device;
    std::mt19937 generator(device());
    std::uniform_int_distribution<uint> distribution(1, UINT_MAX);

	// create cpu buffer
	uint numCount = 512;
	uint* h_num = new uint[numCount];
	for (uint i = 0; i < numCount; ++i) { h_num[i] = distribution(generator); }

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

	MergeSortedArray <<<DivRound(numCount, 512), 256>>> (d_num);
	GpuErrorCheck(cudaDeviceSynchronize());
	GpuErrorCheck(cudaPeekAtLastError());

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