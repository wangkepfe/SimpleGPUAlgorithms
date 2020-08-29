
#include <iostream>

struct Float3
{
	union {
		struct { float x, y, z; };
		float _v[3];
	};

	 Float3() : x(0), y(0), z(0) {}
	 Float3(float _x) : x(_x), y(_x), z(_x) {}
	 Float3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) {}

	inline  Float3  operator+(const Float3& v) const { return Float3(x + v.x, y + v.y, z + v.z); }
	inline  Float3  operator-(const Float3 & v) const { return Float3(x - v.x, y - v.y, z - v.z); }
	inline  Float3  operator*(const Float3 & v) const { return Float3(x * v.x, y * v.y, z * v.z); }
	inline  Float3  operator/(const Float3 & v) const { return Float3(x / v.x, y / v.y, z / v.z); }

	inline  Float3  operator+(float a) const { return Float3(x + a, y + a, z + a); }
	inline  Float3  operator-(float a) const { return Float3(x - a, y - a, z - a); }
	inline  Float3  operator*(float a) const { return Float3(x * a, y * a, z * a); }
	inline  Float3  operator/(float a) const { return Float3(x / a, y / a, z / a); }

	inline  Float3& operator+=(const Float3 & v) { x += v.x; y += v.y; z += v.z; return *this; }
	inline  Float3& operator-=(const Float3 & v) { x -= v.x; y -= v.y; z -= v.z; return *this; }
	inline  Float3& operator*=(const Float3 & v) { x *= v.x; y *= v.y; z *= v.z; return *this; }
	inline  Float3& operator/=(const Float3 & v) { x /= v.x; y /= v.y; z /= v.z; return *this; }

	inline  Float3& operator+=(const float& a) { x += a; y += a; z += a; return *this; }
	inline  Float3& operator-=(const float& a) { x -= a; y -= a; z -= a; return *this; }
	inline  Float3& operator*=(const float& a) { x *= a; y *= a; z *= a; return *this; }
	inline  Float3& operator/=(const float& a) { x /= a; y /= a; z /= a; return *this; }

	inline  Float3 operator-() const { return Float3(-x, -y, -z); }

	inline  bool operator!=(const Float3 & v) const { return x != v.x || y != v.y || z != v.z; }
	inline  bool operator==(const Float3 & v) const { return x == v.x && y == v.y && z == v.z; }

	inline  float& operator[](int i) { return _v[i]; }
	inline  float  operator[](int i) const { return _v[i]; }

	inline  float   length() const { return sqrtf(x * x + y * y + z * z); }
	inline  float   length2() const { return x * x + y * y + z * z; }
	//inline  float   max() const { return max1f(max1f(x, y), z); }
	//inline  float   min() const { return min1f(min1f(x, y), z); }
	inline  Float3& normalize() { float norm = sqrtf(x * x + y * y + z * z); x /= norm; y /= norm; z /= norm; return *this; }
	inline  Float3  normalized() const { float norm = sqrtf(x * x + y * y + z * z); return Float3(x / norm, y / norm, z / norm); }
};

inline  Float3 operator+(float a, const Float3& v) { return Float3(v.x + a, v.y + a, v.z + a); }
inline  Float3 operator-(float a, const Float3 & v) { return Float3(a - v.x, a - v.y, a - v.z); }
inline  Float3 operator*(float a, const Float3 & v) { return Float3(v.x * a, v.y * a, v.z * a); }
inline  Float3 operator/(float a, const Float3 & v) { return Float3(a / v.x, a / v.y, a / v.z); }

inline  Float3 normalize(const Float3& v) { float norm = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z); return Float3(v.x / norm, v.y / norm, v.z / norm); }
inline  float  dot(const Float3& v1, const Float3& v2) { return v1.x* v2.x + v1.y * v2.y + v1.z * v2.z; }
inline  Float3 cross(const Float3& v1, const Float3& v2) { return Float3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x); }

struct Quat
{
	union {
		Float3 v;
		struct {
			float x, y, z;
		};
	};
	float w;

	Quat() : v(), w(0) {}
	Quat(const Float3& v) : v(v), w(0) {}
	Quat(const Float3& v, float w) : v(v), w(w) {}
	Quat(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}

	// construct with rotation (sin_phi/2 * axis, cos_phi/2) 
	static Quat Create(const Float3& axis, float angle) { return Quat(axis.normalized() * sin(angle / 2), cos(angle / 2)); }

	// conjugate
	Quat conj() const { return Quat(-v, w); }

	// normal
	float norm2() const { return x*x + y*y + z*z + w*w; }
	float norm() const { return sqrt(norm2()); }
	Quat normalize() const { float n = norm(); return Quat(v / n, w / n); }

	// inverse
	Quat inv() const { return conj() / norm2(); }

	// get rotation
	float getAngle() const { return acos(w) * 2; }
	Float3 getAxis() const { return v / sqrtf(1.0f - w * w);  }

	// operator with constant
	Quat operator/(float a) const { return Quat(v / a, w / a); }

	// operator with quaternion
	Quat operator+(const Quat& q) const { const Quat& p = *this; return Quat(p.v + q.v, p.w + q.w); }
	Quat operator*(const Quat & q) const { const Quat& p = *this; return Quat(p.w * q.v + q.w * p.v + cross(p.v, q.v), p.w * q.w - dot(p.v, q.v)); }
	Quat& operator+=(const Quat& q) { Quat ret = *this + q; return (*this = ret); }
	Quat& operator*=(const Quat & q) { Quat ret = *this * q; return (*this = ret); }

	// power
	Quat pow(float a) { return Quat::Create(v, acos(w) * 2 * a);  }
};

// rotate q by p: q * p * q-1
inline Quat rotate(const Quat& q, const Quat& v) { return q * v * q.conj(); }

// slerp: spherical linear interpolation (r * q-1)^t * q
inline Quat slerp(const Quat& q, const Quat& r, float t) { return (r * q.conj()).pow(t) * q; }

// rotation between two vectors
inline Quat rotationBetween(const Quat& p, const Quat& q) 
{ 
	return Quat(cross(p.v, q.v), sqrt(p.v.length2() * q.v.length2()) + dot(p.v, q.v)).normalize(); 
}

// rotation between two pre-normalized vectors
inline Quat rotationBetween2(Float3 p, Float3 q) 
{ 
	//p = normalize(p); 
	//q = normalize(q); 
	Float3 u = cross(p, q); 
	float e = dot(p, q); 
	e = sqrtf(2.0f * (1.0f + e));
	return Quat(u / e, e / 2.0f); 
}


#define PI                      3.1415926535897932384626422832795028841971f
#define TWO_PI                  6.2831853071795864769252867665590057683943f

using namespace std;

int main()
{
	//Quat quat = Quat::Create(Float3(1, 1, 1), TWO_PI / 3.0f);
	//cout << "quat " << " is " << quat.x << ", " << quat.y << ", " << quat.z << ", " << quat.w << "\n\n";
	//Float3 vec(1, 0, 0);

	//Float3 result = rotate(quat, vec).v;

	//cout << "result is " << result.x << ", " << result.y << ", " << result.z << "\n\n";

	//for (int i = 0; i <= 10; ++i)
	//{
	//	Float3 arc = slerp(vec, result, i / 10.0f).v;
	//	cout << "arc " << i << " is " << arc.x << ", " << arc.y << ", " << arc.z << "\n";
	//}

	//Quat rotQuat = rotationBetween(vec, result);

	//cout << "\nrotQuat " << " is " << rotQuat.x << ", " << rotQuat.y << ", " << rotQuat.z << ", " << rotQuat.w << "\n\n";

	//result = rotate(rotQuat, vec).v;

	//cout << "result is " << result.x << ", " << result.y << ", " << result.z << "\n\n";

	Float3 test = Float3(0, 0, 1);

	Quat p = Quat::Create(Float3(1, 0, 0), 80.0f / 180.0f * PI);
	Quat q = Quat::Create(Float3(0, 1, 0), 80.0f / 180.0f * PI);

	Quat r = rotate(q, rotate(p, test));
	Quat rotQuat = rotationBetween(test, r.v);
	Quat rotQuat2 = rotationBetween2(test, r.v);

	Quat pq = q * p;

	cout << r.x << " " << r.y << " " << r.z << " " << r.w << endl;
	cout << rotQuat.x << " " << rotQuat.y << " " << rotQuat.z << " " << rotQuat.w << " " << rotQuat.getAngle() / PI * 180.0f << endl;
	cout << pq.x << " " << pq.y << " " << pq.z << " " << pq.w << " " << pq.getAngle() / PI * 180.0f << endl;
	cout << rotQuat2.x << " " << rotQuat2.y << " " << rotQuat2.z << " " << rotQuat2.w << " " << rotQuat2.getAngle() / PI * 180.0f << endl;
	
	return 0;
}