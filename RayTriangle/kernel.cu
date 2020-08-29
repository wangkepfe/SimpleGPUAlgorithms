
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "linear_math.h"
#include "samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d_1spp.h"
#include <string>

#define RAY_MAX 1e10f

#define RAY_TRIANGLE_MOLLER_TRUMBORE 0
#define RAY_TRIANGLE_COORDINATE_TRANSFORM 1

#define RAY_TRIANGLE_CULLING 0
#define PRE_CALU_TRIANGLE_COORD_TRANS_OPT 0

#define GpuErrorCheck(ans) { GpuAssert((ans), __FILE__, __LINE__); }
inline void GpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__device__ void Print(const char* name) { printf("%\n", name); }
__device__ void Print(const char* name, const int& n) { printf("%s = %d\n", name, n); }
__device__ void Print(const char* name, const bool& n) { printf("%s = %s\n", name, n ? "true" : "false"); }
__device__ void Print(const char* name, const uint& n) { printf("%s = %d\n", name, n); }
__device__ void Print(const char* name, const Int2& n) { printf("%s = (%d, %d)\n", name, n.x, n.y); }
__device__ void Print(const char* name, const uint3& n) { printf("%s = (%d, %d, %d)\n", name, n.x, n.y, n.z); }
__device__ void Print(const char* name, const float& n) { printf("%s = %f\n", name, n); }
__device__ void Print(const char* name, const Float2& f3) { printf("%s = (%f, %f)\n", name, f3[0], f3[1]); }
__device__ void Print(const char* name, const Float3& f3) { printf("%s = (%f, %f, %f)\n", name, f3[0], f3[1], f3[2]); }
__device__ void Print(const char* name, const Float4& f4) { printf("%s = (%f, %f, %f, %f)\n", name, f4[0], f4[1], f4[2], f4[3]); }

struct __align__(16) Ray
{
	Float3 orig; float unused1;
	Float3 dir; float unused2;
};

struct __align__(16) Triangle
{
	Float3 v1; float w1;
	Float3 v2; float w2;
	Float3 v3; float w3;
};

struct __align__(16) Camera
{
	Float3 pos;
	float  unused1;
	Float3 dir;
	float  focal;
	Float3 left;
	float  aperture;
	Float3 up;
	float  unused2;
	Float2 resolution;
	Float2 inversedResolution;
	Float2 fov;
	Float2 tanHalfFov;
	Float3 adjustedLeft;
	float  unused3;
	Float3 adjustedUp;
	float  unused4;
	Float3 adjustedFront;
	float  unused5;
	Float3 apertureLeft;
	float  unused6;
	Float3 apertureUp;
	float  unused7;

	void update()
	{
		inversedResolution = 1.0f / resolution;
		fov.y = fov.x / resolution.x * resolution.y;
		tanHalfFov = Float2(tanf(fov.x / 2), tanf(fov.y / 2));

		left = normalize(cross(up, dir));
		up = normalize(cross(dir, left));

		adjustedFront = dir * focal;
		adjustedLeft = left * tanHalfFov.x * focal;
		adjustedUp = up * tanHalfFov.y * focal;

		apertureLeft = left * aperture;
		apertureUp = up * aperture;
	}
};

void CameraSetup(Camera& camera)
{
	camera.pos = { 0.1f, 0.2f, 0.3f };
	camera.dir = { 0.0f, 0.0f, 1.0f };
	camera.up =  { 0.0f, 1.0f, 0.0f };

	camera.focal = 5.0f;
	camera.aperture = 0.0001f;

	camera.resolution = { 2560.0f, 1440.0f };
	camera.fov.x = 60.0f * Pi_over_180;

	camera.update();
}

// ------------------------------ Machine Epsilon -----------------------------------------------
// The smallest number that is larger than one minus one. ULP (unit in the last place) of 1
// ----------------------------------------------------------------------------------------------
__device__ __inline__ constexpr float MachineEpsilon()
{
	typedef union {
		float f32;
		int i32;
	} flt_32;

	flt_32 s{ 1.0f };

	s.i32++;
	return (s.f32 - 1.0f);
}

// ------------------------------ Error Gamma -------------------------------------------------------
// return 32bit floating point arithmatic calculation error upper bound, n is number of calculation
// --------------------------------------------------------------------------------------------------
__device__ __inline__ constexpr float ErrGamma(int n)
{
	return (n * MachineEpsilon()) / (1.0f - n * MachineEpsilon());
}

__device__ __forceinline__ bool MollerTrumbore(const Float3 &orig, const Float3 &dir, const Float3 &v0, const Float3 &v1, const Float3 &v2, float &t, float &u, float &v)
{
	Float3 v0v1 = v1 - v0;
	Float3 v0v2 = v2 - v0;
	Float3 pvec = cross(dir, v0v2);
	float det = dot(v0v1, pvec);

#if RAY_TRIANGLE_CULLING
	// if the determinant is negative the triangle is backfacing
	// if the determinant is close to 0, the ray misses the triangle
	if (det < ErrGamma(8)) return false;
#else
	// ray and triangle are parallel if det is close to 0
	if (abs(det) < ErrGamma(8)) return false;
#endif

	float invDet = 1 / det;

	Float3 tvec = orig - v0;
	u = dot(tvec, pvec) * invDet;
	if (u < 0 || u > 1) return false;

	Float3 qvec = cross(tvec, v0v1);
	v = dot(dir, qvec) * invDet;
	if (v < 0 || u + v > 1) return false;

	t = dot(v0v2, qvec) * invDet;

	return true;
}

__device__ __forceinline__ bool RayTriangleCoordinateTransform(Ray ray, Triangle triangle, float tCurrentHit, float& t, float& u, float& v, float& e)
{
	float origZ = triangle.w1 - dot(ray.orig, triangle.v1);
	float inverseDirZ = 1.0f / dot(ray.dir, triangle.v1);
	t = origZ * inverseDirZ;
	e = abs(t) * ErrGamma(7);

	if (t > ErrGamma(7) && t < tCurrentHit)
	{
		float origX = triangle.w2 + dot(ray.orig, triangle.v2);
		float dirX = dot(ray.dir, triangle.v2);
		u = origX + t * dirX;

		if (u >= 0.0f && u <= 1.0f)
		{
			float origY = triangle.w3 + dot(ray.orig, triangle.v3);
			float dirY = dot(ray.dir, triangle.v3);
			v = origY + t * dirY;

			if (v >= 0.0f && u + v <= 1.0f)
			{
				//normal = cross(triangle.v[1].xyz, triangle.v[2].xyz);
				return true;
			}
		}
	}
	return false;
}

//-------------------------------------------------------
// Inverse the transform matrix
// [ E1x E2x a v1x ]
// [ E1y E2y b v1y ]
// [ E1z E2z c v1z ]
// [ 0   0   0  1  ]
//   Select f = (a,b,c) = (1,0,0) or (0,1,0) or (0,0,1)
// based on largest of (E1 x E2).xyz
// so that the matrix has a stable inverse
//-------------------------------------------------------
void PreCalcTriangleCoordTrans(Triangle& triangle)
{
	Float3 v1 = triangle.v1;
	Float3 v2 = triangle.v2;
	Float3 v3 = triangle.v3;

	Float3 e1 = v2 - v1;
	Float3 e2 = v3 - v1;

	Float3 n = cross(e1, e2);

#if PRE_CALU_TRIANGLE_COORD_TRANS_OPT
	Float3 n_abs = abs(n);
	if (n_abs.x > n_abs.y && n_abs.x > n_abs.z)
	{
		// free vector (1, 0, 0)
		triangle.v1 = { 1           , n.y / n.x   , n.z / n.x };     triangle.w1 = dot(n, v1) / n.x;     // row3
		triangle.v2 = { 0           , e2.z / n.x  , -e2.y / n.x };  triangle.w2 = cross(v3, v1).x / n.x; // row1
		triangle.v3 = { 0           , -e1.z / n.x, e1.y / n.x };    triangle.w3 = -cross(v2, v1).x / n.x; // row2
	}
	else if (n_abs.y > n_abs.x && n_abs.y > n_abs.z)
	{
		// free vector (0, 1, 0)
		triangle.v1 = { n.x / n.y   , 1           , n.z / n.y };     triangle.w1 = dot(n, v1) / n.y;     // row3
		triangle.v2 = { -e2.z / n.y, 0           , e2.x / n.y };    triangle.w2 = cross(v3, v1).y / n.y; // row1
		triangle.v3 = { e1.z / n.y  , 0           , -e1.x / n.y };  triangle.w3 = -cross(v2, v1).y / n.y; // row2
	}
	else
	{
		// free vector (0, 0, 1)
		triangle.v1 = { n.x / n.z   , n.y / n.z   , 1 };             triangle.w1 = dot(n, v1) / n.z;     // row3
		triangle.v2 = { e2.y / n.z  , -e2.x / n.z, 0 };             triangle.w2 = cross(v3, v1).z / n.z; // row1
		triangle.v3 = { -e1.y / n.z, e1.x / n.z  , 0 };             triangle.w3 = -cross(v2, v1).z / n.z; // row2
	}
#else
	Mat4 mtx;

	mtx.setCol(0, { e1, 0 });
	mtx.setCol(1, { e2, 0 });
	mtx.setCol(2, { n, 0 });
	mtx.setCol(3, { v1, 1 });

	mtx = invert(mtx);

	triangle.v1 = mtx.getRow(2).xyz; triangle.w1 = - mtx.getRow(2).w;
	triangle.v2 = mtx.getRow(0).xyz; triangle.w2 = mtx.getRow(0).w;
	triangle.v3 = mtx.getRow(1).xyz; triangle.w3 = mtx.getRow(1).w;
#endif
}

__device__ bool RayTriangleIntersect(Ray ray, Triangle triangle, float tCurr, float& t, float& u, float& v, float& e)
{
#if RAY_TRIANGLE_MOLLER_TRUMBORE
	bool hit = MollerTrumbore(ray.orig, ray.dir, triangle.v1, triangle.v2, triangle.v3, t, u, v);
	if (t > tCurr) { return false; }
	e = ErrGamma(15) * abs(t);
	return hit;
#endif

#if RAY_TRIANGLE_COORDINATE_TRANSFORM
	return RayTriangleCoordinateTransform(ray, triangle, tCurr, t, u, v, e);
#endif
}

__device__  Float2 ConcentricSampleDisk(Float2 u)
{
	// Map uniform random numbers to [-1, 1]
	Float2 uOffset = 2.0 * u - 1.0;

	// Handle degeneracy at the origin
	if (abs(uOffset.x) < 1e-10 && abs(uOffset.y) < 1e-10)
	{
		return Float2(0, 0);
	}

	// Apply concentric mapping to point
	float theta;
	float r;

	if (abs(uOffset.x) > abs(uOffset.y))
	{
		r = uOffset.x;
		theta = PI_OVER_4 * (uOffset.y / uOffset.x);
	}
	else
	{
		r = uOffset.y;
		theta = PI_OVER_2 - PI_OVER_4 * (uOffset.x / uOffset.y);
	}

	return r * Float2(cosf(theta), sinf(theta));
}

__device__ void RayGen(Ray& ray, Int2 idx, Camera camera, Float2 randPixelOffset, Float2 randAperture)
{
	// [0, 1] coordinates
	Float2 uv = (Float2(idx.x, idx.y) + randPixelOffset) * camera.inversedResolution;

	// [0, 1] -> [1, -1], since left/up vector should be 1 when uv is 0
	uv = uv * -2.0 + 1.0;

	// Point on the image plane
	Float3 pointOnImagePlane = camera.adjustedFront + camera.adjustedLeft * uv.x + camera.adjustedUp * uv.y;

	// Point on the aperture
	Float2 diskSample = ConcentricSampleDisk(randAperture);
	Float3 pointOnAperture = diskSample.x * camera.apertureLeft + diskSample.y * camera.apertureUp;

	// ray
	ray.orig = camera.pos + pointOnAperture;
	ray.dir = normalize(pointOnImagePlane - pointOnAperture);
}

__global__ void RenderTriangle(Float3* frameBuffer, Triangle* triangles, Camera camera, BlueNoiseRandGenerator randGen)
{
	Int2 idx = Int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	int pixelIdx = idx.y * gridDim.x * blockDim.x + idx.x;

	Ray ray;
	RayGen(ray, idx, camera, randGen.Rand2(idx, 0, 0), randGen.Rand2(idx, 0, 2));

	float t, u, v, e;
	float tCurr, uCurr, vCurr, eCurr;
	bool hit;
	int iHit = -1;

	tCurr = RAY_MAX;

	for (int i = 0; i < 64; ++i)
	{
		hit = RayTriangleIntersect(ray, triangles[i], tCurr, t, u, v, e);
		if (hit) { iHit = i; tCurr = t; uCurr = u; vCurr = v; eCurr = e; }
	}

	// if (idx.x == gridDim.x * blockDim.x * 0.5 && idx.y == gridDim.y * blockDim.y * 0.5)
	// {
	// 	Print("error", e);
	// }

	frameBuffer[pixelIdx] = (iHit != -1) ? Float3(uCurr, vCurr, 0) : Float3(0);
}

inline float clamp(float x) { return x < 0.0f ? 0.0f : x > 1.0f ? 1.0f : x; }
inline int pixelToInt(float x) { return int(pow(clamp(x), 1 / 2.2) * 255 + .5); }
void writeToPPM(const char* fname, int width, int height, Float3* accuBuffer, unsigned int frameNum)
{
	FILE* f = fopen(fname, "w");
	fprintf(f, "P3\n%d %d\n%d\n", width, height, 255);
	for (int i = 0; i < width * height; i++) {
		accuBuffer[i] /= static_cast<float>(frameNum);
		fprintf(f, "%d %d %d ", pixelToInt(accuBuffer[i].x), pixelToInt(accuBuffer[i].y), pixelToInt(accuBuffer[i].z));
	}
	fclose(f);
	printf("Successfully wrote result image to %s\n", fname);
}

int main()
{
	Camera camera;
	CameraSetup(camera);

	BlueNoiseRandGeneratorHost randGen_h;
	randGen_h.init();

	BlueNoiseRandGenerator randGen(randGen_h);

	Float3* frameBuffer;
	GpuErrorCheck(cudaMalloc((void**) &frameBuffer, 2560 * 1440 * sizeof(Float3)));
	GpuErrorCheck(cudaMemset(frameBuffer, 0, 2560 * 1440 * sizeof(Float3)));

	const unsigned int triCount = 64;
	Triangle* trianglesHost = new Triangle[triCount];

	Float3 lineStart = {-2, -2, 2 };
	Float3 lineEnd = {2, 2, 30};
	Float3 lineStep = (lineEnd - lineStart) / (float)triCount;
	Float3 v2Offset = {-0.5, 0, 0};
	Float3 v3Offset = {0, -0.5, 0};
	for (int i = 0; i < triCount; ++i)
	{
		trianglesHost[i].v1 = lineStart + lineStep * i;
		trianglesHost[i].v2 = trianglesHost[i].v1 + v2Offset;
		trianglesHost[i].v3 = trianglesHost[i].v1 + v3Offset;

#if RAY_TRIANGLE_COORDINATE_TRANSFORM
		PreCalcTriangleCoordTrans(trianglesHost[i]);
#endif
	}

	Triangle* triangles;
	GpuErrorCheck(cudaMalloc((void**) &triangles, triCount * sizeof(Triangle)));
	GpuErrorCheck(cudaMemcpy(triangles, trianglesHost, triCount * sizeof(Triangle), cudaMemcpyHostToDevice));

	dim3 blockDim(8, 8, 1);
	dim3 gridDim(2560/8, 1440/8, 1);
	RenderTriangle << <gridDim, blockDim >> > (frameBuffer, triangles, camera, randGen);

	GpuErrorCheck(cudaDeviceSynchronize());
	GpuErrorCheck(cudaPeekAtLastError());

	Float3* frameBufferHost = new Float3[2560 * 1440 * sizeof(Float3)];
	GpuErrorCheck(cudaMemcpy(frameBufferHost, frameBuffer, 2560 * 1440 * sizeof(Float3), cudaMemcpyDeviceToHost));

	writeToPPM("output.ppm", 2560, 1440, frameBufferHost, 1);

	cudaFree(frameBuffer);
	cudaFree(triangles);

	delete frameBufferHost;
	delete trianglesHost;

	randGen_h.clear();

	return 0;
}