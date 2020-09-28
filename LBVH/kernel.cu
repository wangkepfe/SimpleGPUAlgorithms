
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "linear_math.h"
#include "samplerBlueNoiseErrorDistribution_128x128_OptimizedFor_2d2d2d2d_1spp.h"
#include <string>
#include <iostream>

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

#define ushort unsigned short

//#define IS_DEBUG_PIXEL idx.x == gridDim.x * blockDim.x * 0.5 && idx.y == gridDim.y * blockDim.y * 0.5
//#define IS_DEBUG_PIXEL idx.x == 1152 && idx.y == 277
#define IS_DEBUG_PIXEL 0

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
	__host__ __device__ Triangle() {}
	__host__ __device__ Triangle(const Float3& v1, const Float3& v2, const Float3& v3) : v1(v1), v2(v2), v3(v3) {}

	Float3 v1; float w1;
	Float3 v2; float w2;
	Float3 v3; float w3;
};

struct AABB
{
	__host__ __device__ AABB() : max(), min() {}
	__host__ __device__ AABB(const Float3& max, const Float3& min) : max(max), min(min) {}

	Float3 max;
	Float3 min;
};

struct __align__(16) AABBCompact
{
	__host__ __device__ AABBCompact() {}
	__host__ __device__ AABBCompact(const AABB& aabb1, const AABB& aabb2)
	: box1maxX (aabb1.max.x), box1maxY (aabb1.max.y), box1minX (aabb1.min.x), box1minY (aabb1.min.y),
	  box2maxX (aabb2.max.x), box2maxY (aabb2.max.y), box2minX (aabb2.min.x), box2minY (aabb2.min.y),
	  box1maxZ (aabb1.max.z), box1minZ (aabb1.min.z), box2maxZ (aabb2.max.z), box2minZ (aabb2.min.z) {}

	__host__ __device__ AABB GetMerged() const {
		return AABB(Float3(max(box1maxX, box2maxX), max(box1maxY, box2maxY), max(box1maxZ, box2maxZ)),
		            Float3(min(box1minX, box2minX), min(box1minY, box2minY), min(box1minZ, box2minZ)));
	}

	__host__ __device__ AABB GetLeftAABB() const {
		return AABB(Float3(box1maxX, box1maxY, box1maxZ), Float3(box1minX, box1minY, box1minZ));
	}

	__host__ __device__ AABB GetRightAABB() const  {
		return AABB(Float3(box2maxX, box2maxY, box2maxZ), Float3(box2minX, box2minY, box2minZ));
	}

	// float box1maxX;
	// float box1maxY;
	// float box1minX;
	// float box1minY;

	// float box2maxX;
	// float box2maxY;
	// float box2minX;
	// float box2minY;

	// float box1maxZ;
	// float box1minZ;
	// float box2maxZ;
	// float box2minZ;

	float box1maxX;
	float box1maxY;
	float box1maxZ;
	float pad0;

	float box1minX;
	float box1minY;
	float box1minZ;
	float pad1;

	float box2maxX;
	float box2maxY;
	float box2maxZ;
	float pad2;

	float box2minX;
	float box2minY;
	float box2minZ;
	float pad3;
};

struct __align__(16) BVHNode
{
	AABBCompact aabb;

	uint idxLeft;
	uint idxRight;
	uint isLeftLeaf;
	uint isRightLeaf;
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
__device__ __forceinline__ constexpr float MachineEpsilon()
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
__device__ __forceinline__ constexpr float ErrGamma(int n)
{
	return (n * MachineEpsilon()) / (1.0f - n * MachineEpsilon());
}

// ------------------------------ Morton Code 3D ----------------------------------------------------
// 32bit 3D morton code encode
// --------------------------------------------------------------------------------------------------
__device__ __forceinline__ unsigned int MortonCode3D(unsigned int x, unsigned int y, unsigned int z) {
	x = (x | (x << 16)) & 0x030000FF;
	x = (x | (x << 8)) & 0x0300F00F;
	x = (x | (x << 4)) & 0x030C30C3;
	x = (x | (x << 2)) & 0x09249249;
	y = (y | (y << 16)) & 0x030000FF;
	y = (y | (y << 8)) & 0x0300F00F;
	y = (y | (y << 4)) & 0x030C30C3;
	y = (y | (y << 2)) & 0x09249249;
	z = (z | (z << 16)) & 0x030000FF;
	z = (z | (z << 8)) & 0x0300F00F;
	z = (z | (z << 4)) & 0x030C30C3;
	z = (z | (z << 2)) & 0x09249249;
	return x | (y << 1) | (z << 2);
}

// ------------------------------ Moller Trumbore ----------------------------------------------------
// Ray Triangle intersection without pre-transformation
// --------------------------------------------------------------------------------------------------
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

// ------------------------------ Ray Triangle Coordinate Transform ----------------------------------------------------
// Ray Triangle intersection with pre-transformation
// --------------------------------------------------------------------------------------------------
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

//-------------------------- Pre Calc Triangle Coord Trans -----------------------------
// Inverse the transform matrix
// [ E1x E2x a v1x ]
// [ E1y E2y b v1y ]
// [ E1z E2z c v1z ]
// [ 0   0   0  1  ]
//   Select f = (a,b,c) = (1,0,0) or (0,1,0) or (0,0,1)
// based on largest of (E1 x E2).xyz
// so that the matrix has a stable inverse
//-------------------------------------------------------
__device__ __forceinline__ void PreCalcTriangleCoordTrans(Triangle& triangle)
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

__device__ __forceinline__ bool RayTriangleIntersect(Ray ray, Triangle triangle, float tCurr, float& t, float& u, float& v, float& e)
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

__device__ __forceinline__ bool RayAABBIntersect(const Float3& invRayDir, const Float3& rayOrig, const AABB& aabb, float& tmin) {

	Float3 t0s = (aabb.min - rayOrig) * invRayDir;
  	Float3 t1s = (aabb.max - rayOrig) * invRayDir;

  	Float3 tsmaller = min3f(t0s, t1s);
    Float3 tbigger  = max3f(t0s, t1s);

    tmin = max(tsmaller[0], max(tsmaller[1], tsmaller[2]));
	float tmax = min(tbigger[0], min(tbigger[1], tbigger[2]));

	Int2 idx = Int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	if (IS_DEBUG_PIXEL)
	{
		printf("RayAABBIntersect: t0s=(%f, %f, %f), t1s=(%f, %f, %f)\n", t0s.x, t0s.y, t0s.z, t1s.x, t1s.y, t1s.z);
		printf("RayAABBIntersect: tsmaller=(%f, %f, %f), tbigger=(%f, %f, %f)\n", tsmaller.x, tsmaller.y, tsmaller.z, tbigger.x, tbigger.y, tbigger.z);
		printf("RayAABBIntersect: tmin=%f, tmax=%f\n", tmin, tmax);
	}

	return (tmin < tmax);
}

__device__ __forceinline__ void RayAabbPairIntersect(const Float3& invRayDir, const Float3& rayOrig, const AABBCompact& aabbpair, bool& intersect1, bool& intersect2, bool& isClosestIntersect1)
{
	AABB aabbLeft = aabbpair.GetLeftAABB();
	AABB aabbRight = aabbpair.GetRightAABB();

	float tmin1, tmin2;

	intersect1 = RayAABBIntersect(invRayDir, rayOrig, aabbLeft, tmin1);
	intersect2 = RayAABBIntersect(invRayDir, rayOrig, aabbRight, tmin2);

	isClosestIntersect1 = (tmin1 < tmin2);

	Int2 idx = Int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	if (IS_DEBUG_PIXEL)
	{
		printf("RayAabbPairIntersect: aabbLeft.min=(%f, %f, %f), aabbLeft.max=(%f, %f, %f)\n", aabbLeft.min.x, aabbLeft.min.y, aabbLeft.min.z, aabbLeft.max.x, aabbLeft.max.y, aabbLeft.max.z);
		printf("RayAabbPairIntersect: aabbRight.min=(%f, %f, %f), aabbRight.max=(%f, %f, %f)\n", aabbRight.min.x, aabbRight.min.y, aabbRight.min.z, aabbRight.max.x, aabbRight.max.y, aabbRight.max.z);
		printf("RayAabbPairIntersect: intersect1=%d, intersect2=%d, isClosestIntersect1=%d\n", intersect1, intersect2, isClosestIntersect1);
	}
}

__device__ __forceinline__ Float2 ConcentricSampleDisk(Float2 u)
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

__device__ __forceinline__ void RayGen(Ray& ray, Int2 idx, Camera camera, Float2 randPixelOffset, Float2 randAperture)
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

__global__ void RenderTriangle(Float3* frameBuffer, Triangle* triangles, BVHNode* bvhNodes, Camera camera, BlueNoiseRandGenerator randGen)
{
	Int2 idx = Int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	int pixelIdx = idx.y * gridDim.x * blockDim.x + idx.x;

	Ray ray;
	RayGen(ray, idx, camera, randGen.Rand2(idx, 0, 0), randGen.Rand2(idx, 0, 2));

	Float3 invRayDir = Float3(1) / ray.dir;

	if (IS_DEBUG_PIXEL)
	{
		Print("RenderTriangle:");
		Print("ray.orig", ray.orig);
		Print("ray.dir", ray.dir);
		Print("invRayDir", invRayDir);
	}

	float t, u, v, e;
	float tCurr, uCurr, vCurr, eCurr;
	bool hit;
	int iHit = -1;
	tCurr = RAY_MAX;

#if 0
	// linear traversal
	for (int i = 0; i < 64; ++i)
	{
		hit = RayTriangleIntersect(ray, triangles[i], tCurr, t, u, v, e);
		if (hit) {
			iHit = i; tCurr = t; uCurr = u; vCurr = v; eCurr = e;

			if (IS_DEBUG_PIXEL)
			{
				printf("we got a hit!\n");
				Print("iHit", iHit);
			}
		}
	}
#else
	// BVH traversal
	bool intersect1, intersect2, isClosestIntersect1;
	// init stack
	volatile int bvhNodeStack[8];
	volatile int isLeaf[8];
	for (int i = 0; i < 8; ++i)
	{
		bvhNodeStack[i] = -1;
		isLeaf[i] = -1;
	}
	int stackTop = -1;

	int currIdx = 0;
	int isCurrLeaf = 0;

	while(1)
	{
		if (IS_DEBUG_PIXEL)
		{
			Print("i", i);
			Print("stackTop", stackTop);
			Print("isCurrLeaf", isCurrLeaf);
			Print("currIdx", currIdx);
			printf("bvhNodeStack[] = {%d, %d, %d, %d, ...}\n", bvhNodeStack[0], bvhNodeStack[1],bvhNodeStack[2],bvhNodeStack[3]);
		}

		if (isCurrLeaf)
		{
			// triangle test
			if (IS_DEBUG_PIXEL)
			{
				Print("triangles[currIdx].v1", triangles[currIdx].v1);
				Print("triangles[currIdx].v2", triangles[currIdx].v2);
				Print("triangles[currIdx].v3", triangles[currIdx].v3);

				Print("triangles[63].v1", triangles[63].v1);
				Print("triangles[63].v2", triangles[63].v2);
				Print("triangles[63].v3", triangles[63].v3);
			}

			hit = RayTriangleIntersect(ray, triangles[currIdx], tCurr, t, u, v, e);
			if (hit)
			{
				iHit = currIdx; tCurr = t; uCurr = u; vCurr = v; eCurr = e;
				if (IS_DEBUG_PIXEL)
				{
					Print("iHit", iHit);
				}
				break;
			}

			// pop
			if (stackTop < 0)
			{
				if (IS_DEBUG_PIXEL)
				{
					Print("iHit", iHit);
				}
				break;
			}

			currIdx = bvhNodeStack[stackTop];
			isCurrLeaf = isLeaf[stackTop];

			--stackTop;
		}
		else
		{
			BVHNode currNode = bvhNodes[currIdx];

			if (IS_DEBUG_PIXEL)
			{
				Print("currNode.idxLeft", currNode.idxLeft);
				Print("currNode.idxRight", currNode.idxRight);
				Print("currNode.isLeftLeaf", currNode.isLeftLeaf);
				Print("currNode.isRightLeaf", currNode.isRightLeaf);
			}

			// test two aabb
			RayAabbPairIntersect(invRayDir, ray.orig, currNode.aabb, intersect1, intersect2, isClosestIntersect1);

			if (!intersect1 && !intersect2)
			{
				// no hit, pop
				if (stackTop < 0) { break; }

				currIdx = bvhNodeStack[stackTop];
				isCurrLeaf = isLeaf[stackTop];

				--stackTop;
			}
			else if (intersect1 && !intersect2)
			{
				// left hit
				currIdx = currNode.idxLeft;
				isCurrLeaf = currNode.isLeftLeaf;
			}
			else if (!intersect1 && intersect2)
			{
				// right hit
				currIdx = currNode.idxRight;
				isCurrLeaf = currNode.isRightLeaf;
			}
			else
			{
				// both hit
				int idx1, idx2;
				bool isLeaf1, isLeaf2;

				if (isClosestIntersect1)
				{
					// left closer. push right
					idx2    = currNode.idxRight;
					isLeaf2 = currNode.isRightLeaf;
					idx1    = currNode.idxLeft;
					isLeaf1 = currNode.isLeftLeaf;
				}
				else
				{
					// right closer. push left
					idx2    = currNode.idxLeft;
					isLeaf2 = currNode.isLeftLeaf;
					idx1    = currNode.idxRight;
					isLeaf1 = currNode.isRightLeaf;
				}

				// push
				++stackTop;
				bvhNodeStack[stackTop] = idx2;
				isLeaf[stackTop]       = isLeaf2;

				currIdx                = idx1;
				isCurrLeaf             = isLeaf1;

				if (IS_DEBUG_PIXEL)
				{
					Print("both hit: stackTop", stackTop);
					Print("both hit: idx2", idx2);
					Print("both hit: idx1", idx1);
				}
			}
		}
	}
#endif
	frameBuffer[pixelIdx] = (iHit != -1) ? Float3(uCurr, vCurr, 0) : Float3(0, 0, 1);
}

__device__ __forceinline__ int RingIncrease(int& idx)
{
	int size = 64;
	if (idx < size - 1) { ++idx; }
	else { idx = 0; }
	return idx;
}

__device__ __forceinline__ bool IsQueueNotEmpty(int head, int tail)
{
	int size = 64;
	return (head != size - 1 && head != head - 1) || (head == size - 1 && tail != 0);
}

__global__ void RenderBVH(Float3* frameBuffer, Triangle* triangles, AABB* aabbs, BVHNode* bvhNodes, uint* reorderIdx, Camera camera, BlueNoiseRandGenerator randGen)
{
	Int2 idx = Int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
	int zslice = blockIdx.z;
	int pixelIdx = idx.y * gridDim.x * blockDim.x + idx.x;

	Ray ray;
	RayGen(ray, idx, camera, randGen.Rand2(idx, 0, 0), randGen.Rand2(idx, 0, 2));

	Float3 inverseRayDir = Float3(1.0) / ray.dir;

	float tCurr = RAY_MAX;
	float t, u, v, e;
	float uCurr, vCurr, eCurr;
	bool hit;
	int iHit = -1;
	bool triangleHit = 0;

	int bvhDepth = 0;

	// queue init
	int queue[64];
	int queueDepth[64];
	int queueHead = 0;
	int queueTail = 1;

	// push
	queueDepth[RingIncrease(queueHead)] = 0;
	queue[queueHead] = 0;

	while(IsQueueNotEmpty(queueHead, queueTail))
	{
		// every loop is for a depth

		if (zslice == bvhDepth) // the final depth we want
		{
			// render aabb and tri in a certain depth
			while(IsQueueNotEmpty(queueHead, queueTail))
			{
				// peek tail
				int currDepth = queueDepth[queueTail];
				BVHNode curr = bvhNodes[queue[queueTail]];

				if (currDepth > bvhDepth)
					break;

				if (curr.isLeftLeaf)
				{
					// render leaf
					hit = RayTriangleIntersect(ray, triangles[curr.idxLeft], tCurr, t, u, v, e);
					if (hit) { iHit = curr.idxLeft; tCurr = t; uCurr = u; vCurr = v; eCurr = e; triangleHit = 1; }
				}
				else
				{
					// push
					queueDepth[RingIncrease(queueHead)] = currDepth + 1;
					queue[queueHead] = curr.idxLeft;

					// render aabb
					hit = RayAABBIntersect(inverseRayDir, ray.orig, curr.aabb.GetLeftAABB(), t);
					if (t < tCurr && hit) { tCurr = t; iHit = reorderIdx[curr.idxLeft]; triangleHit = 0; }
				}


				if (curr.isRightLeaf)
				{
					hit = RayTriangleIntersect(ray, triangles[curr.idxRight], tCurr, t, u, v, e);
					if (hit) { iHit = curr.idxRight; tCurr = t; uCurr = u; vCurr = v; eCurr = e; triangleHit = 1; }
				}
				else
				{
					queueDepth[RingIncrease(queueHead)] = currDepth + 1;
					queue[queueHead] = curr.idxRight;

					hit = RayAABBIntersect(inverseRayDir, ray.orig, curr.aabb.GetRightAABB(), t);
					if (t < tCurr && hit) { tCurr = t; iHit = reorderIdx[curr.idxRight]; triangleHit = 0; }
				}

				// pop
				RingIncrease(queueTail);
			}

			// done
			break;
		}
		else
		{
			// go next depth
			while(IsQueueNotEmpty(queueHead, queueTail))
			{
				// peek tail
				int currDepth = queueDepth[queueTail];
				BVHNode curr = bvhNodes[queue[queueTail]];

				if (currDepth > bvhDepth)
					break;

				if (curr.isLeftLeaf)
				{
					// render leaf
					hit = RayTriangleIntersect(ray, triangles[curr.idxLeft], tCurr, t, u, v, e);
					if (hit) { iHit = curr.idxLeft; tCurr = t; uCurr = u; vCurr = v; eCurr = e; triangleHit = 0; }
				}
				else
				{
					// push
					queueDepth[RingIncrease(queueHead)] = currDepth + 1;
					queue[queueHead] = curr.idxLeft;
				}


				if (curr.isRightLeaf)
				{
					hit = RayTriangleIntersect(ray, triangles[curr.idxRight], tCurr, t, u, v, e);
					if (hit) { iHit = curr.idxRight; tCurr = t; uCurr = u; vCurr = v; eCurr = e; triangleHit = 0; }
				}
				else
				{
					queueDepth[RingIncrease(queueHead)] = currDepth + 1;
					queue[queueHead] = curr.idxRight;
				}

				// pop
				RingIncrease(queueTail);
			}
		}

		++bvhDepth;
	}

	// for (int i = 0; i < 64; ++i)
	// {
	// 	hit = RayAABBIntersect(inverseRayDir, ray.orig, aabbs[i], t);
	// 	if (t < tCurr && hit) { tCurr = t; iHit = i; }
	// }

	frameBuffer[zslice * 2560 * 1440 + pixelIdx] = (iHit != -1) ? (triangleHit ? Float3(uCurr, vCurr, 0) : Float3(randGen.Rand(iHit, 0, 0), randGen.Rand(iHit, 1, 0), randGen.Rand(iHit, 2, 0))) : Float3(0);
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

__inline__ __device__ void WarpReduceMaxMin3f(Float3& vmax, Float3& vmin) {
	const int warpSize = 32;
	#pragma unroll
	for (unsigned int offset = warpSize / 2; offset > 0; offset /= 2)
	{
		vmin.x = min(__shfl_down_sync(0xffffffff, vmin.x, offset), vmin.x);
		vmin.y = min(__shfl_down_sync(0xffffffff, vmin.y, offset), vmin.y);
		vmin.z = min(__shfl_down_sync(0xffffffff, vmin.z, offset), vmin.z);

		vmax.x = max(__shfl_down_sync(0xffffffff, vmax.x, offset), vmax.x);
		vmax.y = max(__shfl_down_sync(0xffffffff, vmax.y, offset), vmax.y);
		vmax.z = max(__shfl_down_sync(0xffffffff, vmax.z, offset), vmax.z);
	}
}

template<int ldsSize>
__global__ void UpdateSceneGeometry(Triangle* triangles, AABB* aabbs, AABB* sceneBoundingBox, uint* morton, unsigned int triCount)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	uint laneId = threadIdx.x & 0x1f;
	uint warpId = (threadIdx.x & 0xffffffe0) >> 5u;

	// ------------------------------------ update triangle position ------------------------------------
	// a line
	// Float3 lineStart = {-2, -1, 2 };
	// Float3 lineEnd = {18, 10, 30};
	// Float3 lineStep = (lineEnd - lineStart) / (float)triCount;
	// Float3 v1 = lineStart + lineStep * idx;

	// a plane
	uint idxx = idx / 8;
	uint idxy = idx % 8;
	Float3 startPos = {-3, 0, 8 };
	Float3 endPos1 = {-3, 8, 38 };
	Float3 endPos2 = {3, 0, 8 };
	Float3 lineStep1 = (endPos1 - startPos) / (float)triCount * 8;
	Float3 lineStep2 = (endPos2 - startPos) / (float)triCount * 8;
	Float3 v1 = startPos + lineStep1 * idxx + lineStep2 * idxy;

	Float3 v2Offset = {-0.5, -0.5, 0};
	Float3 v3Offset = {0, -0.5, 0};
	Float3 v2 = v1 + v2Offset;
	Float3 v3 = v1 + v3Offset;

	Triangle mytriangle(v1, v2, v3);

#if RAY_TRIANGLE_COORDINATE_TRANSFORM
	PreCalcTriangleCoordTrans(mytriangle);
#endif

	// write out
	triangles[idx] = mytriangle;

	// ------------------------------------ update aabb ------------------------------------
	Float3 aabbmin = min3f(v1, min3f(v2, v3));
	Float3 aabbmax = max3f(v1, max3f(v2, v3));

	Float3 diff = aabbmax - aabbmin;
	diff = max3f(Float3(0.001f), diff);
	aabbmax = aabbmin + diff;

	aabbs[idx].min = aabbmin;
	aabbs[idx].max = aabbmax;

	Float3 aabbcenter = (aabbmax + aabbmin) / 2.0f;

	// ------------------------------------ reduce for scene bounding box ------------------------------------
	__shared__ AABB lds[ldsSize];

	if (idx < ldsSize)
	{
		lds[idx] = AABB(Float3(FLT_MIN), Float3(FLT_MAX));
	}
	__syncthreads();

	Float3 aabbmintemp = aabbmin;
	Float3 aabbmaxtemp = aabbmax;

	// warp reduce
	WarpReduceMaxMin3f(aabbmaxtemp, aabbmintemp);

	// compact re-group
	if (laneId == 0)
	{
		lds[warpId] = AABB(aabbmaxtemp, aabbmintemp);
	}
	__syncthreads();

	//#pragma unroll
	//for (int activeWarpCount = triCount / 32; activeWarpCount > 1; activeWarpCount = (activeWarpCount + 31) / 32)
	//{
	if (warpId < ((triCount / 32) + 31) / 32)
	{
		// read
		aabbmintemp = lds[idx].min;
		aabbmaxtemp = lds[idx].max;

		// warp reduce
		WarpReduceMaxMin3f(aabbmaxtemp, aabbmintemp);

		// compact re-group
		if (laneId == 0)
		{
			lds[warpId] = AABB(aabbmaxtemp, aabbmintemp);
		}
	}
	__syncthreads();
	//}

	AABB sceneAabb = lds[0];

	if (idx == 0)
	{
		sceneBoundingBox[0] = sceneAabb;
	}
	__syncthreads();

	// ------------------------------------ assign morton code to aabb ------------------------------------
	Float3 unitBox = (aabbcenter - sceneAabb.min) / (sceneAabb.max - sceneAabb.min);

	uint mortonCode = MortonCode3D((uint)(unitBox.x * 1023.0f),
	                               (uint)(unitBox.y * 1023.0f),
								   (uint)(unitBox.z * 1023.0f));

	morton[idx] = mortonCode;

	// if (idx == 63)
	// {
	// 	Print("aabbmin", aabbmin);
	// 	Print("aabbmax", aabbmax);
	// 	Print("sceneAabb.min", sceneAabb.min);
	// 	Print("sceneAabb.max", sceneAabb.max);
	// 	Print("aabbcenter", aabbcenter);
	// 	Print("unitBox", unitBox);
	// 	Print("(uint)(unitBox.x * 1023.0f)", (uint)(unitBox.x * 1023.0f));
	// 	Print("(uint)(unitBox.y * 1023.0f)", (uint)(unitBox.y * 1023.0f));
	// 	Print("(uint)(unitBox.z * 1023.0f)", (uint)(unitBox.z * 1023.0f));
	// 	Print("mortonCode", mortonCode);
	// }
}

__global__ void RadixSort(uint* inout, uint* reorderIdx)
{
	struct LDS
	{
		uint tempIdx[256];
		uint temp[256];
		ushort histo[16 * 8];
		ushort histoScan[16];
	};

	__shared__ LDS lds;
	lds.tempIdx[threadIdx.x] = threadIdx.x;

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
		uint finalIdx = idxAllNum + idxCurrentNumBlock + idxCurrentNumWarp;

		lds.temp[finalIdx] = num;

		uint currentReorderIdx = lds.tempIdx[threadIdx.x];
		__syncthreads();

		lds.tempIdx[finalIdx] = currentReorderIdx;
		__syncthreads();
	}

	//------------------------------------ Write out ----------------------------------------
	inout[blockIdx.x * blockDim.x + threadIdx.x] = lds.temp[threadIdx.x];
	reorderIdx[blockIdx.x * blockDim.x + threadIdx.x] = lds.tempIdx[threadIdx.x];
}

__device__ __inline__ int LCP(uint* morton, uint triCount, int m0, int j)
{
	int res;
	if (j < 0 || j >= triCount) { res = 0; }
	else { res = __clz(m0 ^ morton[j]); }
	return res;
}

__global__ void BuildLBVH (BVHNode* bvhNodes, AABB* aabbs, uint* morton, uint* reorderIdx, uint* bvhNodeParent, uint* isAabbDone, uint triCount)
{
	uint i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= triCount - 1) { return; }

	//-------------------------------------------------------------------------------------------------------------------------
	// https://developer.nvidia.com/blog/wp-content/uploads/2012/11/karras2012hpg_paper.pdf
	// Determine direction of the range (+1 or -1)
	int m0 = morton[i];
	int deltaLeft = LCP(morton, triCount, m0, i - 1);
	int deltaRight = LCP(morton, triCount, m0, i + 1);
	int d = (deltaRight - deltaLeft) >= 0 ? 1 : -1;

	// Compute upper bound for the length of the range
	int deltaMin = LCP(morton, triCount, m0, i - d);
	int lmax = 2;
	while (LCP(morton, triCount, m0, i + lmax * d) > deltaMin)
	{
		lmax *= 2;
	}

	// Find the other end using binary search
	int l = 0;
	for (int t = lmax / 2; t >= 1; t /= 2)
	{
		if (LCP(morton, triCount, m0, i + (l + t) * d) > deltaMin)
		{
			l += t;
		}
	}
	int j = i + l * d;

	// Find the split position using binary search
	int deltaNode = LCP(morton, triCount, m0, j);
	int s = 0;
	for (int div = 2, int t = (l + div - 1) / div; t >= 1; div *= 2, t = (l + div - 1) / div)
	{
		if (LCP(morton, triCount, m0, i + (s + t) * d) > deltaNode)
		{
			s += t;
		}
	}
	int gamma = i + s * d + min(d, 0);

	// Output child pointers. the children of Ii cover the ranges [min(i, j), γ] and [γ + 1,max(i, j)]
	if (min(i, j) == gamma)
	{
		bvhNodes[i].isLeftLeaf = 1;
		bvhNodes[i].idxLeft = reorderIdx[gamma];
	}
	else
	{
		bvhNodes[i].isLeftLeaf = 0;
		bvhNodes[i].idxLeft = gamma;
		bvhNodeParent[gamma] = i;
	}

	if (max(i, j) == gamma + 1)
	{
		bvhNodes[i].isRightLeaf = 1;
		bvhNodes[i].idxRight = reorderIdx[gamma + 1];
	}
	else
	{
		bvhNodes[i].isRightLeaf = 0;
		bvhNodes[i].idxRight = gamma + 1;
		bvhNodeParent[gamma + 1] = i;
	}
	//-------------------------------------------------------------------------------------------------------------------------

	while(atomicCAS(&isAabbDone[0], 0, 0) == 0)
	{
		if (((bvhNodes[i].isLeftLeaf == 1)  || (bvhNodes[i].isLeftLeaf == 0  && atomicCAS(&isAabbDone[bvhNodes[i].idxLeft], 0, 0) == 1)) &&
		    ((bvhNodes[i].isRightLeaf == 1) || (bvhNodes[i].isRightLeaf == 0 && atomicCAS(&isAabbDone[bvhNodes[i].idxRight], 0, 0) == 1)))
		{
			AABB leftAabb  = (bvhNodes[i].isLeftLeaf  == 1) ? aabbs[bvhNodes[i].idxLeft]  : bvhNodes[bvhNodes[i].idxLeft ].aabb.GetMerged();
			AABB rightAabb = (bvhNodes[i].isRightLeaf == 1) ? aabbs[bvhNodes[i].idxRight] : bvhNodes[bvhNodes[i].idxRight].aabb.GetMerged();

			bvhNodes[i].aabb = AABBCompact(leftAabb, rightAabb);

			atomicExch(&isAabbDone[i], 1);
		}
	}
}

int DivRound(int n, int m) { return (n + m - 1) / m; }

int main()
{
	Camera camera;
	CameraSetup(camera);

	BlueNoiseRandGeneratorHost randGen_h;
	randGen_h.init();

	BlueNoiseRandGenerator randGen(randGen_h);

	Float3* frameBuffer;
	GpuErrorCheck(cudaMalloc((void**) &frameBuffer, 2560 * 1440 * 1 * sizeof(Float3)));
	GpuErrorCheck(cudaMemset(frameBuffer, 0, 2560 * 1440 * 1 * sizeof(Float3)));

	const unsigned int triCount = 64;
	Triangle* triangles;
	GpuErrorCheck(cudaMalloc((void**) &triangles, triCount * sizeof(Triangle)));

	AABB* aabbs;
	GpuErrorCheck(cudaMalloc((void**) &aabbs, triCount * sizeof(AABB)));

	AABB* sceneBoundingBox;
	GpuErrorCheck(cudaMalloc((void**) &sceneBoundingBox, sizeof(AABB)));

	uint* morton;
	GpuErrorCheck(cudaMalloc((void**)& morton, 256 * sizeof(uint)));
	GpuErrorCheck(cudaMemset(morton, UINT_MAX, 256 * sizeof(uint)));

	uint* reorderIdx;
	GpuErrorCheck(cudaMalloc((void**)& reorderIdx, 256 * sizeof(uint)));

	BVHNode* bvhNodes;
	uint* bvhNodeParent;
	uint* isAabbDone;
	GpuErrorCheck(cudaMalloc((void**)& bvhNodes, (triCount - 1) * sizeof(BVHNode)));
	GpuErrorCheck(cudaMalloc((void**)& bvhNodeParent, (triCount - 1) * sizeof(uint)));
	GpuErrorCheck(cudaMalloc((void**)& isAabbDone, (triCount - 1) * sizeof(uint)));
	GpuErrorCheck(cudaMemset(isAabbDone, 0, (triCount - 1) * sizeof(uint)));

// ------------------------------- Update Geometry -----------------------------------
// out: triangles
//      aabbs
//      morton codes
#if CREATE_TRIANGLE_HOST
	Triangle* trianglesHost = new Triangle[triCount];

	for (int i = 0; i < triCount; ++i)
	{
		trianglesHost[i].v1 = lineStart + lineStep * i;
		trianglesHost[i].v2 = trianglesHost[i].v1 + v2Offset;
		trianglesHost[i].v3 = trianglesHost[i].v1 + v3Offset;

#if RAY_TRIANGLE_COORDINATE_TRANSFORM
		PreCalcTriangleCoordTrans(trianglesHost[i]);
#endif
	}
	GpuErrorCheck(cudaMemcpy(triangles, trianglesHost, triCount * sizeof(Triangle), cudaMemcpyHostToDevice));
#else
	int warpCount = triCount / 32;
	int ldsSize = ((warpCount + 31) / 32) * 32;
	switch (ldsSize)
	{
		case 32: UpdateSceneGeometry <32> <<< 1, 64, 32 * sizeof(AABB) >>> (triangles, aabbs, sceneBoundingBox, morton, triCount);
	}
#endif

	GpuErrorCheck(cudaDeviceSynchronize());
	GpuErrorCheck(cudaPeekAtLastError());

	AABB sceneBoundingBoxHost;
	GpuErrorCheck(cudaMemcpy(&sceneBoundingBoxHost, sceneBoundingBox, sizeof(AABB), cudaMemcpyDeviceToHost));
	std::cout << "max = (" << sceneBoundingBoxHost.max.x << "," << sceneBoundingBoxHost.max.y << "," << sceneBoundingBoxHost.max.z << ")\n"
			  << "min = (" << sceneBoundingBoxHost.min.x << "," << sceneBoundingBoxHost.min.y << "," << sceneBoundingBoxHost.min.z << ")\n";

	uint mortonHost[64];
	GpuErrorCheck(cudaMemcpy(&mortonHost, morton, 64 * sizeof(uint), cudaMemcpyDeviceToHost));
	for (int i = 0; i < 64; ++i) { std::cout << mortonHost[i] << " "; }
	std::cout << "\n\n\n";

	// ------------------------------- Radix Sort -----------------------------------
	// in: morton code
	// out: reorder idx
	RadixSort <<< 1, 256 >>> (morton, reorderIdx);

	GpuErrorCheck(cudaDeviceSynchronize());
	GpuErrorCheck(cudaPeekAtLastError());

	GpuErrorCheck(cudaMemcpy(&mortonHost, morton, 64 * sizeof(uint), cudaMemcpyDeviceToHost));
	for (int i = 0; i < 64; ++i) { std::cout << mortonHost[i] << " "; }
	std::cout << "\n\n\n";

	uint reorderIdxHost[64];
	GpuErrorCheck(cudaMemcpy(&reorderIdxHost, reorderIdx, 64 * sizeof(uint), cudaMemcpyDeviceToHost));
	for (int i = 0; i < 64; ++i) { std::cout << reorderIdxHost[i] << " "; }
	std::cout << "\n\n\n";

	// ------------------------------- Build LBVH -----------------------------------
	// in: aabbs
	//     morton code
	//     reorder idx
	// out: lbvh
	BuildLBVH <<< 1 , 64 >>> (bvhNodes, aabbs, morton, reorderIdx, bvhNodeParent, isAabbDone, triCount);

	GpuErrorCheck(cudaDeviceSynchronize());
	GpuErrorCheck(cudaPeekAtLastError());

	AABBCompact sceneBoundingBoxHost2compact;
	GpuErrorCheck(cudaMemcpy(&sceneBoundingBoxHost2compact, bvhNodes, sizeof(AABBCompact), cudaMemcpyDeviceToHost));
	AABB sceneBoundingBoxHost2 = sceneBoundingBoxHost2compact.GetMerged();
	std::cout << "max = (" << sceneBoundingBoxHost2.max.x << "," << sceneBoundingBoxHost2.max.y << "," << sceneBoundingBoxHost2.max.z << ")\n"
			  << "min = (" << sceneBoundingBoxHost2.min.x << "," << sceneBoundingBoxHost2.min.y << "," << sceneBoundingBoxHost2.min.z << ")\n";

	// ------------------------------- Rendering -----------------------------------
	RenderTriangle <<< dim3(2560/8, 1440/8, 1), dim3(8, 8, 1) >>> (frameBuffer, triangles, bvhNodes, camera, randGen);
	// RenderBVH <<< dim3(2560/8, 1440/8, 16), dim3(8, 8, 1) >>> (frameBuffer, triangles, aabbs, bvhNodes, reorderIdx, camera, randGen);

	// GpuErrorCheck(cudaDeviceSynchronize());
	// GpuErrorCheck(cudaPeekAtLastError());

	Float3* frameBufferHost = new Float3[2560 * 1440 * 1 * sizeof(Float3)];
	GpuErrorCheck(cudaMemcpy(frameBufferHost, frameBuffer, 2560 * 1440 * 1 * sizeof(Float3), cudaMemcpyDeviceToHost));

	for (int i = 0; i < 1; ++i)
	{
		std::string name = "output" + std::to_string(i) + ".ppm";
		writeToPPM(name.c_str(), 2560, 1440, frameBufferHost + i * 2560 * 1440, 1);
	}

	cudaFree(frameBuffer);
	cudaFree(triangles);
	cudaFree(aabbs);
	cudaFree(sceneBoundingBox);
	cudaFree(morton);
	cudaFree(reorderIdx);
	cudaFree(bvhNodes);
	cudaFree(bvhNodeParent);
	cudaFree(isAabbDone);

	delete frameBufferHost;

	randGen_h.clear();

	return 0;
}