
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cuda_runtime.h>

#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <time.h>

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

void CpuSequentialScan(int* h_num, int numCount)
{
	// print scan array
	int sum = 0;
	for (int i = 0; i < numCount; ++i)
	{
		sum += h_num[i];
		if ((i < (numCount - 1)) &&
			(i > (numCount - 5)))
		{
			std::cout << sum << ",";
		}
	}
	sum -= h_num[numCount - 1];
	std::cout << "\n\n";
}

void CpuCount(int* num, int numCount)
{
	int countTable[16] = { 0 };

	for (int i = 0; i < numCount; ++i)
	{
		++countTable[num[i]];
	}

	for (int i = 0; i < 16; ++i)
	{
		std::cout << countTable[i] << ",";
	}

	std::cout << "\n\n";
}

__inline__ __device__ void Scan(int* lds, int i)
{
	int step;

	// bottom up
#pragma unroll
	for (step = 1; step < 2048; step *= 2)
	{
		if (i < 1024 / step)
		{
			int rightIdx = 2047 - 2 * i * step;
			int leftIdx = rightIdx - step;

			lds[rightIdx] += lds[leftIdx];
		}
		__syncthreads();
	}

	if (i == 1023)
	{
		lds[2047] = 0;
	}
	__syncthreads();

	// top down
#pragma unroll
	for (step = 1024; step >= 1; step /= 2)
	{
		if (i < 1024 / step)
		{
			int rightIdx = 2047 - 2 * i * step;
			int leftIdx = rightIdx - step;

			int left = lds[leftIdx];
			int right = lds[rightIdx];

			lds[leftIdx] = right;
			lds[rightIdx] = left + right;
		}
		__syncthreads();
	}
}

__global__ void PrefixScanMultiBlock(int* num, int* blockSum)
{
	__shared__ int lds[2048];

	int i = threadIdx.x;
	int j = blockIdx.x;

	int idx1 = i * 2;
	int idx2 = i * 2 + 1;

	int idx3 = j * 2048 + i * 2;
	int idx4 = j * 2048 + i * 2 + 1;

	lds[idx1] = num[idx3];
	lds[idx2] = num[idx4];

	__syncthreads();

	Scan(lds, i);

	if (i == 1023)
	{
		blockSum[j] = lds[2047] + num[2048 * j + 2047];
	}

	num[idx3] = lds[idx1];
	num[idx4] = lds[idx2];
}

__global__ void PrefixScanSingleBlock(int* num)
{
	__shared__ int lds[2048];

	int i = threadIdx.x;
	int j = blockIdx.x;

	int idx1 = i * 2;
	int idx2 = i * 2 + 1;

	int idx3 = j * 2048 + i * 2;
	int idx4 = j * 2048 + i * 2 + 1;

	lds[idx1] = num[idx3];
	lds[idx2] = num[idx4];

	__syncthreads();

	Scan(lds, i);

	num[idx3] = lds[idx1];
	num[idx4] = lds[idx2];
}

__global__ void PrefixScanAdd(int* num, int* blockSum)
{
	int i = threadIdx.x;
	int j = blockIdx.x;

	int idx3 = j * 2048 + i * 2;
	int idx4 = j * 2048 + i * 2 + 1;

	int blocksum = blockSum[j];

	num[idx3] += blocksum;
	num[idx4] += blocksum;
}

void GpuScan(int* h_num, int numCount)
{
	int* d_num;
	int* d_blockSum;
	int* h_blockSum;

	// malloc and copy
	h_blockSum = new int[2048];

	GpuErrorCheck(cudaMalloc((void**)& d_num, numCount * sizeof(int)));
	GpuErrorCheck(cudaMemcpy(d_num, h_num, numCount * sizeof(int), cudaMemcpyHostToDevice));

	GpuErrorCheck(cudaMalloc((void**)& d_blockSum, 2048 * sizeof(int)));
	GpuErrorCheck(cudaMemset(d_blockSum, 0, 2048 * sizeof(int)));

	// grid dim, block dim
	dim3 gridDim(numCount / 2048, 1, 1);
	dim3 blockDim(1024, 1, 1);

	// dispatch
	PrefixScanMultiBlock << <gridDim, blockDim >> > (d_num, d_blockSum);
	PrefixScanSingleBlock << <1, blockDim >> > (d_blockSum);
	PrefixScanAdd << <gridDim, blockDim >> > (d_num, d_blockSum);

	GpuErrorCheck(cudaDeviceSynchronize());

	// copy to cpu
	GpuErrorCheck(cudaMemcpy(h_num, d_num, numCount * sizeof(int), cudaMemcpyDeviceToHost));
	GpuErrorCheck(cudaMemcpy(h_blockSum, d_blockSum, 2048 * sizeof(int), cudaMemcpyDeviceToHost));

	GpuErrorCheck(cudaPeekAtLastError());

	// print
	for (int i = numCount - 3; i < numCount; ++i)
	{
		std::cout << h_num[i] << ",";
	}
	std::cout << "\n\n";

	// free
	delete h_blockSum;
	cudaFree(d_num);
	cudaFree(d_blockSum);
}

__global__ void CountMultiBlock(int* num, int* blockCount, int* numOffset)
{
	__shared__ int lds[16];

	if (threadIdx.x < 16)
	{
		lds[threadIdx.x] = 0;
	}

	__syncthreads();

	int old1 = atomicAdd(lds + num[blockIdx.x * 2048 + threadIdx.x * 2], 1);
	int old2 = atomicAdd(lds + num[blockIdx.x * 2048 + threadIdx.x * 2 + 1], 1);

	numOffset[blockIdx.x * 2048 + threadIdx.x * 2] = old1;
	numOffset[blockIdx.x * 2048 + threadIdx.x * 2 + 1] = old2;

	__syncthreads();

	if (threadIdx.x < 16)
	{
		blockCount[blockIdx.x * 16 + threadIdx.x] = lds[threadIdx.x];
	}
}

__global__ void CountSum(int* blockCount, int* resultCount, int numBlock)
{
	const int n = 128;
	const int m = 16;

	__shared__ int lds[n][m];

	int i = threadIdx.x;
	int j = threadIdx.y;
	int k = blockIdx.x;

	lds[i * 2][j] = 0;
	lds[i * 2 + 1][j] = 0;

	__syncthreads();

	if (i * 2 < numBlock)
	{
		lds[i * 2][j] = blockCount[k * 2048 + (i * 2) * m + j];
		lds[i * 2 + 1][j] = blockCount[k * 2048 + (i * 2 + 1) * m + j];
	}
	__syncthreads();

#pragma unroll
	for (int step = 1; step < n; step *= 2)
	{
		if (i < (n / 2) / step)
		{
			int leftIdx = 2 * i * step;
			int rightIdx = leftIdx + step;

			lds[leftIdx][j] += lds[rightIdx][j];
		}
		__syncthreads();
	}

	if (i == 0)
	{
		resultCount[k * 16 + j] = lds[0][j];
	}
}

void GpuCount(int* h_num, int* d_num, int* d_orderBuffer, int* d_numOffset, int numCount)
{
	int* d_blockCount;
	// int* h_blockCount;

	int* d_resultCount;
	int* h_resultCount;

	// malloc and copy
	int blockCountSize = DivRound(numCount, 2048) * 16;
	//h_blockCount = new int[blockCountSize];
	int* h_numOffset = new int[numCount];

	h_resultCount = new int[16];

	GpuErrorCheck(cudaMalloc((void**)& d_blockCount, blockCountSize * sizeof(int)));
	GpuErrorCheck(cudaMemset(d_blockCount, 0, blockCountSize * sizeof(int)));

	GpuErrorCheck(cudaMalloc((void**)& d_resultCount, DivRound(DivRound(numCount, 2048), 128) * 16 * sizeof(int)));
	GpuErrorCheck(cudaMemset(d_resultCount, 0, DivRound(DivRound(numCount, 2048), 128) * 16 * sizeof(int)));

	// dispatch
	CountMultiBlock << <dim3(DivRound(numCount, 2048), 1, 1), dim3(1024, 1, 1) >> > (d_num, d_blockCount, d_numOffset);

	GpuErrorCheck(cudaDeviceSynchronize());
	GpuErrorCheck(cudaPeekAtLastError());

	if (DivRound(DivRound(numCount, 2048), 128) > 1)
	{
		CountSum << <dim3(DivRound(DivRound(numCount, 2048), 128), 1, 1), dim3(64, 16, 1) >> > (d_blockCount, d_resultCount, DivRound(numCount, 2048));

		GpuErrorCheck(cudaDeviceSynchronize());
		GpuErrorCheck(cudaPeekAtLastError());

		CountSum << <dim3(DivRound(DivRound(DivRound(numCount, 2048), 128), 128), 1, 1), dim3(64, 16, 1) >> > (d_resultCount, d_orderBuffer, DivRound(DivRound(numCount, 2048), 128));

		GpuErrorCheck(cudaDeviceSynchronize());
		GpuErrorCheck(cudaPeekAtLastError());

		GpuErrorCheck(cudaMemcpy(h_resultCount, d_orderBuffer, 16 * sizeof(int), cudaMemcpyDeviceToHost));
	}
	else
	{
		CountSum << <dim3(DivRound(DivRound(numCount, 2048), 128), 1, 1), dim3(64, 16, 1) >> > (d_blockCount, d_orderBuffer, DivRound(numCount, 2048));

		GpuErrorCheck(cudaDeviceSynchronize());
		GpuErrorCheck(cudaPeekAtLastError());

		GpuErrorCheck(cudaMemcpy(h_resultCount, d_orderBuffer, 16 * sizeof(int), cudaMemcpyDeviceToHost));
	}

	// copy to cpu
	// GpuErrorCheck(cudaMemcpy(h_blockCount, d_blockCount, blockCountSize * sizeof(int), cudaMemcpyDeviceToHost));
	GpuErrorCheck(cudaMemcpy(h_numOffset, d_numOffset, numCount * sizeof(int), cudaMemcpyDeviceToHost));

	// print
	// for (int i = 0; i < blockCountSize; ++i)
	// {
	// 	std::cout << h_blockCount[i] << ",";
	// }
	// std::cout << "\n\n";

	// std::cout << "numOffset:\n";
	// for (int i = 0; i < numCount; ++i)
	// {
	// 	std::cout << h_numOffset[i] << ",";
	// }
	// std::cout << "\n\n";

	std::cout << "resultCount:\n";
	for (int i = 0; i < 16; ++i)
	{
		std::cout << h_resultCount[i] << ",";
	}
	std::cout << "\n\n";

	// free
	delete h_resultCount;
	delete h_numOffset;
	//delete h_blockCount;

	cudaFree(d_blockCount);
	cudaFree(d_resultCount);

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

void RadixSort(int* h_num, int numCount)
{
	int* d_orderBuffer;
	int h_orderBuffer[16];

	int* d_numOffset;

	int* d_output;
	int* h_output = new int[numCount];

	int* d_num;

	GpuErrorCheck(cudaMalloc((void**)& d_num, numCount * sizeof(int)));
	GpuErrorCheck(cudaMemcpy(d_num, h_num, numCount * sizeof(int), cudaMemcpyHostToDevice));

	GpuErrorCheck(cudaMalloc((void**)& d_orderBuffer, 16 * sizeof(int)));
	GpuErrorCheck(cudaMemset(d_orderBuffer, 0, 16 * sizeof(int)));

	GpuErrorCheck(cudaMalloc((void**)& d_numOffset, numCount * sizeof(int)));

	GpuErrorCheck(cudaMalloc((void**)& d_output, numCount * sizeof(int)));

	GpuCount(h_num, d_num, d_orderBuffer, d_numOffset, numCount);

	PrefixScan16 << <1, 16 >> > (d_orderBuffer);
	GpuErrorCheck(cudaDeviceSynchronize());
	GpuErrorCheck(cudaPeekAtLastError());

	GpuErrorCheck(cudaMemcpy(h_orderBuffer, d_orderBuffer, 16 * sizeof(int), cudaMemcpyDeviceToHost));

	std::cout << "orderBuffer:\n";
	for (int i = 0; i < 16; ++i)
	{
		std::cout << h_orderBuffer[i] << ",";
	}
	std::cout << "\n\n";

	Reorder << <dim3(DivRound(numCount, 1024), 1, 1), dim3(1024, 1, 1) >> > (d_num, d_output, d_orderBuffer, d_numOffset);
	GpuErrorCheck(cudaDeviceSynchronize());
	GpuErrorCheck(cudaPeekAtLastError());

	GpuErrorCheck(cudaMemcpy(h_output, d_output, numCount * sizeof(int), cudaMemcpyDeviceToHost));

	std::cout << "sorted:\n";
	for (int i = 0; i < numCount; ++i)
	{
		std::cout << h_output[i] << ",";
	}
	std::cout << "\n\n";

	delete h_output;

	cudaFree(d_num);
	cudaFree(d_orderBuffer);
	cudaFree(d_numOffset);
}

int main()
{
	srand(time(NULL));

	// create cpu buffer
	int numCount = 2048 * 2;
	int* h_num = new int[numCount];
	for (int i = 0; i < numCount; ++i) { h_num[i] = rand() % 16; }

	// radix sort
	RadixSort(h_num, numCount);

	// cpu count
	//CpuCount(h_num, numCount);

	// gpu count
	//GpuCount(h_num, numCount);

	// cpu sequential scan
	//CpuSequentialScan(h_num, numCount);

	// gpu scan
	//GpuScan(h_num, numCount);

	// delete
	delete h_num;
	return 0;
}