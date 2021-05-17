
//----------------------------------------------------------------------------------------------
//------------------------------------- Bilateral Filter ---------------------------------------
//----------------------------------------------------------------------------------------------

__constant__ float cGaussian[64];
#define BilateralFilterGuassianDelta 1.0f
#define BilateralFilterEuclideanFactor 1.0f
#define BilateralFilterRadius 2

__device__ __inline__ float euclideanLen(Float4 a, Float4 b, float d)
{

    float mod = (b.x - a.x) * (b.x - a.x) +
                (b.y - a.y) * (b.y - a.y) +
                (b.z - a.z) * (b.z - a.z);

    return __expf(-mod / (2.f * d * d));
}

void updateGaussian()
{
	float delta = BilateralFilterGuassianDelta;
	int radius = BilateralFilterRadius;

    float  fGaussian[64];

    for (int i = 0; i < 2*radius + 1; ++i)
    {
        float x = i-radius;
        fGaussian[i] = expf(-(x*x) / (2*delta*delta));
    }

    checkCudaErrors(cudaMemcpyToSymbol(cGaussian, fGaussian, sizeof(float)*(2*radius+1)));
}

__global__ void BilateralFilter(
	SurfObj   colorBuffer,
	SurfObj   accumulateBuffer,
	SurfObj   normalDepthBuffer,
	SurfObj   normalDepthHistoryBuffer,
	Int2      size)
{
	// float e_d = BilateralFilterEuclideanFactor;
	// int r = BilateralFilterRadius;

	// Int2 idx2 (blockIdx.x * 28 + threadIdx.x - 2, blockIdx.y * 28 + threadIdx.y - 2); // index for pixel 28 x 28
	// Int2 idx3 (threadIdx.x, threadIdx.y); // index for shared memory buffer

	// // read global memory buffer. One-to-one mapping
	// Float3Ushort1 colorAndMask = Load2DHalf3Ushort1(colorBuffer, idx2);
	// Float3 colorValue = colorAndMask.xyz;
	// ushort maskValue = colorAndMask.w;

	// Float2 normalAndDepth  = Load2DFloat2(normalDepthBuffer, idx2);
	// float normalEncoded = normalAndDepth.x;
	// Float3 normalValue = DecodeNormal_R11_G10_B11(normalEncoded);
	// float depthValue = normalAndDepth.y;

    // if (idx3.x < 2 || idx3.y < 2 || idx3.x > 29 || idx3.y > 29) { return; }

    // float sum = 0.0f;
    // float factor;
    // Float4 t = {0.f, 0.f, 0.f, 0.f};
	// Float4 center = Load2D_float4 (colorBuffer, Int2(x, y));

    // for (int i = -r; i <= r; i++)
    // {
    //     for (int j = -r; j <= r; j++)
    //     {
	// 		Float4 curPix = Load2D_float4 (colorBuffer, Int2(x + j, y + i));
    //         factor = cGaussian[i + r] * cGaussian[j + r] * euclideanLen(curPix, center, e_d);

    //         t += factor * curPix;
    //         sum += factor;
    //     }
    // }

    // Store2D_float4 (Float4(t / sum), colorBuffer, Int2(x, y));
}