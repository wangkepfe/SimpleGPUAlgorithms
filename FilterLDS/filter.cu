
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <iomanip>

#include "linear_math.h"

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

__device__ int max3(int a, int b, int c) { return max(a, max(b, c)); }

__device__ int LoadBuffer(int* data, int w, int h, int x, int y)
{
	x = min(max(x, 0), w - 1);
	y = min(max(y, 0), h - 1);
	return data[x + y * w];
}

__global__ void MaxFilterNaive(int* data, int w, int h)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	int v[9];
	for (int i = 0; i <= 2; ++i)
	{
		for (int j = 0; j <= 2; ++j)
		{
			v[j + 3 * i] = LoadBuffer(data, w, h, x + i - 1, y + j - 1);
		}
	}

	int result = 0;
	for (int i = 0; i <= 2; ++i)
	{
		for (int j = 0; j <= 2; ++j)
		{
			result = max(result, v[j + 3 * i]);
		}
	}

	data[x + y * w] = result;
}

__global__ void MaxFilterLds(int* data, int w, int h)
{
	__shared__ int lds[100];

	int x = threadIdx.x + blockIdx.x * 8;
	int y = threadIdx.y + blockIdx.y * 8;

	int id = (threadIdx.x + threadIdx.y * 8);

	int x1 = blockIdx.x * 8 - 1 + id % 10;
	int y1 = blockIdx.y * 8 - 1 + id / 10;

	int x2 = blockIdx.x * 8 - 1 + (id + 64) % 10;
	int y2 = blockIdx.y * 8 - 1 + (id + 64) / 10;

	int v1 = LoadBuffer(data, w, h, x1, y1);
	int v2 = LoadBuffer(data, w, h, x2, y2);

	lds[id] = v1;
	lds[id + 64] = v2;

	__syncthreads();

	int result = 0;
	for (int i = 0; i <= 2; ++i)
	{
		for (int j = 0; j <= 2; ++j)
		{
			result = max(result, lds[threadIdx.x + j + (threadIdx.y + i) * 10]);
		}
	}

	data[x + y * w] = result;
}

__global__ void MaxFilterDLP(int* data, int w, int h)
{
	int laneId = threadIdx.x + threadIdx.y * blockDim.x;

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	int v_1, v_2;

	int v1 = LoadBuffer(data, w, h, x - 1, y - 1); // row [0:3] = [-1:2] left
	int v2 = LoadBuffer(data, w, h, x + 1, y - 1); // row [0:3] = [-1:2] right
	int v3 = LoadBuffer(data, w, h, (laneId < 16) ? (x - 1) : (x + 1), (laneId < 16) ? (y + 3) : (y + 1)); // row [0:1] = [3:4] left, row [2:3] = [3:4] right

	v_1 = __shfl_sync(0xffffffff, v1, laneId + 1);
	v_2 = __shfl_sync(0xffffffff, v2, laneId - 1);
	int v4 = (threadIdx.x < 7) ? v_1 : v_2; // row [0:3] = [-1:2] middle

	v_1 = __shfl_sync(0xffffffff, v3, laneId + 1);
	v_2 = __shfl_sync(0xffffffff, v3, (laneId ^ 0x10) - 1);
	int v5 = (threadIdx.x < 7) ? v_1 : v_2; // row [0:1] = [3:4] middle
	int v6 = __shfl_sync(0xffffffff, v3, laneId ^ 0x10); // row [0:1] = [3:4] right

	v1 = max3(v1, v2, v4); // row [0:3] = [-1:2] horizontal max ;  = [0:3] top
	v2 = max3(v3, v5, v6); // row [0:1] = [3:4] horizontal max ;

	v_1 = __shfl_sync(0xffffffff, v1, laneId + 8);
	v_2 = __shfl_sync(0xffffffff, v2, laneId & 0x7);
	v3 = (threadIdx.y < 3) ? v_1 : v_2; // row [0:3] = [0:3] middle

	v_1 = __shfl_sync(0xffffffff, v1, laneId ^ 0x10);
	v_2 = __shfl_sync(0xffffffff, v2, laneId ^ 0x10);
	v4 = (threadIdx.y < 2) ? v_1 : v_2; // row [0:3] = [0:3] bottom

	v1 = max3(v1, v3, v4);

	data[x + y * w] = v1;
}

__device__ void min_max_idxmax_3x1(int& vmin, int& vmax, int& idx_x, int v1, int v2, int v3)
{
	vmin = min(v1, v2);
	vmax = max(v1, v2);
	idx_x = (v1 > v2) ? -1 : 0;

	vmin = min(vmin, v3);
	vmax = max(vmax, v3);
	idx_x = (vmax > v3) ? idx_x : 1;
}

__device__ void min_1x3(int& vmin, int v1, int v2, int v3)
{
	vmin = min(v1, v2);
	vmin = min(vmin, v3);
}

__device__ void max_idxmax_1x3(int& vmax, int& idx_x, int& idx_y, int v1, int v2, int v3, int idx_x1, int idx_x2, int idx_x3)
{
	vmax = max(v1, v2);
	idx_x = (v1 > v2) ? idx_x1 : idx_x2;
	idx_y = (v1 > v2) ? -1 : 0;

	vmax = max(vmax, v3);
	idx_x = (vmax > v3) ? idx_x : idx_x3;
	idx_y = (vmax > v3) ? idx_y : 1;
}

__global__ void MaxMinIdxmaxDLP(int* data, int* outMax, int* outMin, Int2* outIdxMax, int w, int h)
{
	int laneId = threadIdx.x + threadIdx.y * blockDim.x;

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	// ----------------------- read buffer -------------------------

	int leftup = LoadBuffer(data, w, h, x - 1, y - 1); // row [0:3] = [-1:2] left
	int rightup = LoadBuffer(data, w, h, x + 1, y - 1); // row [0:3] = [-1:2] right
	int leftrightdown = LoadBuffer(data, w, h, (laneId < 16) ? (x - 1) : (x + 1), (laneId < 16) ? (y + 3) : (y + 1)); // row [0:1] = [3:4] left, row [2:3] = [3:4] right

	// ----------------------- horizontal pass -------------------------

	int v_1, v_2;

	v_1 = __shfl_sync(0xffffffff, leftup, laneId + 1);
	v_2 = __shfl_sync(0xffffffff, rightup, laneId - 1);
	int middleup = (threadIdx.x < 7) ? v_1 : v_2; // row [0:3] = [-1:2] middle

	v_1 = __shfl_sync(0xffffffff, leftrightdown, laneId + 1);
	v_2 = __shfl_sync(0xffffffff, leftrightdown, (laneId ^ 0x10) - 1);
	int middledown = (threadIdx.x < 7) ? v_1 : v_2; // row [0:1] = [3:4] middle

	int rightdown = __shfl_sync(0xffffffff, leftrightdown, laneId ^ 0x10); // row [0:1] = [3:4] right

	int upmin, upmax, up_idx_x;
	min_max_idxmax_3x1(upmin, upmax, up_idx_x, leftup, middleup, rightup); // row [0:3] = [-1:2] horizontal max ;  = [0:3] top

	int downmin, downmax, down_idx_x;
	min_max_idxmax_3x1(downmin, downmax, down_idx_x, leftrightdown, middledown, rightdown); // row [0:1] = [3:4] horizontal max ;

	// ----------------------- vertical pass -------------------------

	// min
	v_1 = __shfl_sync(0xffffffff, upmin, laneId + 8);
	v_2 = __shfl_sync(0xffffffff, downmin, laneId & 0x7);
	int middlemin = (threadIdx.y < 3) ? v_1 : v_2; // row [0:3] = [0:3] middle

	v_1 = __shfl_sync(0xffffffff, upmin, laneId ^ 0x10);
	v_2 = __shfl_sync(0xffffffff, downmin, laneId ^ 0x10);
	downmin = (threadIdx.y < 2) ? v_1 : v_2; // row [0:3] = [0:3] bottom

	int omin;
	min_1x3(omin, upmin, middlemin, downmin);

	// max
	v_1 = __shfl_sync(0xffffffff, upmax, laneId + 8);
	v_2 = __shfl_sync(0xffffffff, downmax, laneId & 0x7);
	int middlemax = (threadIdx.y < 3) ? v_1 : v_2; // row [0:3] = [0:3] middle

	v_1 = __shfl_sync(0xffffffff, upmax, laneId ^ 0x10);
	v_2 = __shfl_sync(0xffffffff, downmax, laneId ^ 0x10);
	downmax = (threadIdx.y < 2) ? v_1 : v_2; // row [0:3] = [0:3] bottom

	v_1 = __shfl_sync(0xffffffff, up_idx_x, laneId + 8);
	v_2 = __shfl_sync(0xffffffff, down_idx_x, laneId & 0x7);
	int middle_idx_x = (threadIdx.y < 3) ? v_1 : v_2; // row [0:3] = [0:3] middle

	v_1 = __shfl_sync(0xffffffff, up_idx_x, laneId ^ 0x10);
	v_2 = __shfl_sync(0xffffffff, down_idx_x, laneId ^ 0x10);
	down_idx_x = (threadIdx.y < 2) ? v_1 : v_2; // row [0:3] = [0:3] bottom

	int omax, oidx_x, oidx_y;
	max_idxmax_1x3(omax, oidx_x, oidx_y, upmax, middlemax, downmax, up_idx_x, middle_idx_x, down_idx_x);

	// data
	v_1 = __shfl_sync(0xffffffff, middleup, laneId + 8);
	v_2 = __shfl_sync(0xffffffff, middledown, laneId & 0x7);
	int middle = (threadIdx.y < 3) ? v_1 : v_2; // row [0:3] = [0:3] middle

	// ----------------------- out -------------------------

	data[x + y * w] = middle;
	outMax[x + y * w] = omax;
	outMin[x + y * w] = omin;
	outIdxMax[x + y * w] = Int2(oidx_x, oidx_y);
}

int main()
{
	srand(time(NULL));

	// create cpu buffer
	int w = 8;
	int h = 8;
	int numCount = h * w;
	int* h_data = new int[numCount];
	for (int i = 0; i < numCount; ++i) { h_data[i] = rand() % 16; }

	// print original
	for (int i = 0; i < h; ++i)
	{
		for (int j = 0; j < w; ++j)
		{
			std::cout << std::setw(3) << std::left << h_data[i * w + j] << ",";
			if (j % 8 == 7) std::cout << "   ,";
		}
		std::cout << "\n";
		if (i % 8 == 7) std::cout << "\n";
	}
	std::cout << "\n\n";

	// create gpu buffer and copy to gpu
	int* d_data;
	int* d_outMin;
	int* d_outMax;
	Int2* d_outIdxMax;

	GpuErrorCheck(cudaMalloc((void**)& d_data, numCount * sizeof(int)));
	GpuErrorCheck(cudaMemcpy(d_data, h_data, numCount * sizeof(int), cudaMemcpyHostToDevice));

	GpuErrorCheck(cudaMalloc((void**)& d_outMin, numCount * sizeof(int)));
	GpuErrorCheck(cudaMalloc((void**)& d_outMax, numCount * sizeof(int)));
	GpuErrorCheck(cudaMalloc((void**)& d_outIdxMax, numCount * sizeof(Int2)));

	// dispatch
	//MaxFilterNaive << < dim3(DivRound(w, 8), DivRound(h, 8), 1), dim3(8, 8, 1) >> > (d_data1, w, h);
	//MaxFilterLds << < dim3(DivRound(w, 8), DivRound(h, 8), 1), dim3(8, 8, 1) >> > (d_data2, w, h);
	//MaxFilterDLP << < dim3(DivRound(w, 8), DivRound(h, 4), 1), dim3(8, 4, 1) >> > (d_data3, w, h);
	MaxMinIdxmaxDLP <<< dim3(DivRound(w, 8), DivRound(h, 4), 1), dim3(8, 4, 1) >>> (d_data, d_outMax, d_outMin, d_outIdxMax, w, h);

	GpuErrorCheck(cudaDeviceSynchronize());
	GpuErrorCheck(cudaPeekAtLastError());

	// copy to cpu
	int* h_datamin = new int[numCount];
	int* h_datamax = new int[numCount];
	Int2* h_dataidx = new Int2[numCount];
	GpuErrorCheck(cudaMemcpy(h_data,    d_data,      numCount * sizeof(int), cudaMemcpyDeviceToHost));
	GpuErrorCheck(cudaMemcpy(h_datamin, d_outMin,    numCount * sizeof(int), cudaMemcpyDeviceToHost));
	GpuErrorCheck(cudaMemcpy(h_datamax, d_outMax,    numCount * sizeof(int), cudaMemcpyDeviceToHost));
	GpuErrorCheck(cudaMemcpy(h_dataidx, d_outIdxMax, numCount * sizeof(Int2), cudaMemcpyDeviceToHost));

	// print processed
	std::cout << "data:\n";
	for (int i = 0; i < h; ++i)
	{
		for (int j = 0; j < w; ++j)
		{
			std::cout << std::setw(3) << std::left << h_data[i * w + j] << ",";
			if (j % 8 == 7) std::cout << "   ,";
		}
		std::cout << "\n";
		if (i % 8 == 7) std::cout << "\n";
	}
	std::cout << "\n\n";

	std::cout << "min:\n";
	for (int i = 0; i < h; ++i)
	{
		for (int j = 0; j < w; ++j)
		{
			std::cout << std::setw(3) << std::left << h_datamin[i * w + j] << ",";
			if (j % 8 == 7) std::cout << "   ,";
		}
		std::cout << "\n";
		if (i % 8 == 7) std::cout << "\n";
	}
	std::cout << "\n\n";

	std::cout << "max:\n";
	for (int i = 0; i < h; ++i)
	{
		for (int j = 0; j < w; ++j)
		{
			std::cout << std::setw(3) << std::left << h_datamax[i * w + j] << ",";
			if (j % 8 == 7) std::cout << "   ,";
		}
		std::cout << "\n";
		if (i % 8 == 7) std::cout << "\n";
	}
	std::cout << "\n\n";

	std::cout << "idx:\n";
	for (int i = 0; i < h; ++i)
	{
		for (int j = 0; j < w; ++j)
		{
			std::cout << "(" << std::setw(3) << std::left << h_dataidx[i * w + j].x << "," << std::setw(3) << std::left << h_dataidx[i * w + j].y << "),";
			if (j % 8 == 7) std::cout << "   ,";
		}
		std::cout << "\n";
		if (i % 8 == 7) std::cout << "\n";
	}
	std::cout << "\n\n";

	delete h_data;
	delete h_datamax;
	delete h_dataidx;
	delete h_datamin;
	cudaFree(d_data);
	cudaFree(d_outMin);
	cudaFree(d_outMax);
	cudaFree(d_outIdxMax);
	return 0;
}