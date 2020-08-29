#include <iostream>
#include <iomanip>
#include <string>
#include <map>
#include <random>
#include <limits>

constexpr float MachineEpsilon() {
	typedef union {
		float f32;
		int i32;
	} flt_32;

	flt_32 s{ 1.0 };

	s.i32++;
	return (s.f32 - 1.0);
}

constexpr float ErrGamma(int n)
{
	return (n * MachineEpsilon()) / (1.0 - n * MachineEpsilon());
}

void test1()
{
	std::random_device rd;
	std::mt19937 e2(rd());
	std::uniform_real_distribution<> dist(0, 5000);

	double real_error_max = 0;

	double error_add_max_fp1 = 0;
	double error_add_max_fp2 = 0;
	double error_add_max_fpres = 0;

	double error_add_max_dp1 = 0;
	double error_add_max_dp2 = 0;
	double error_add_max_dpres = 0;

	float error1 = 0;
	float error2 = 0;
	float error3 = 0;

	for (int i = 0; i < 10000000; ++i)
	{
		double dp1 = (double)dist(e2);
		double dp11 = (double)dist(e2);
		double dp2 = (double)dist(e2);
		double dp22 = (double)dist(e2);

		float fp1 = (float)dp1;
		float fp2 = (float)dp2;
		float fp11 = (float)dp11;
		float fp22 = (float)dp22;

		//std::cout << fp1 << ' ' << fp2 << ' ';

		dp1 -= dp11;
		fp1 -= fp11;

		dp2 -= dp22;
		fp2 -= fp22;

		float esti_error_fp1 = (std::abs(fp1) + std::abs(fp11)) * ErrGamma(1);
		float esti_error_fp2 = (std::abs(fp2) + std::abs(fp22)) * ErrGamma(1);
		
		double dpRes = dp1 * dp2;
		float fpRes = fp1 * fp2;

		double real_error = std::abs(dpRes - (double)fpRes);

		if (real_error > real_error_max)
		{
			error_add_max_fp1 = fp1;
			error_add_max_fp2 = fp2;
			error_add_max_fpres = fpRes;

			error_add_max_dp1 = dp1;
			error_add_max_dp2 = dp2;
			error_add_max_dpres = dpRes;

			error1 = (std::abs(fp1) * std::abs(fp2)) * ErrGamma(1);
			error2 = (std::abs(fp1) * std::abs(fp2)) * ErrGamma(2);
			error3 = esti_error_fp1 + esti_error_fp2 + (std::abs(fp1) * std::abs(fp2)) * ErrGamma(1);

			std::swap(real_error, real_error_max);
		}
	}

	std::cout << std::endl;
	std::cout << "real_error_max = " << std::setprecision(10) << real_error_max << std::endl << std::endl;

	std::cout << "error_add_max_fp1 = " << std::setprecision(10) << error_add_max_fp1 << std::endl;
	std::cout << "error_add_max_fp2 = " << std::setprecision(10) << error_add_max_fp2 << std::endl << std::endl;

	std::cout << "error_add_max_dp1 = " << std::setprecision(10) << error_add_max_dp1 << std::endl;
	std::cout << "error_add_max_dp2 = " << std::setprecision(10) << error_add_max_dp2 << std::endl << std::endl;

	std::cout << "error_add_max_fpres = " << std::setprecision(10) << error_add_max_fpres << std::endl;
	std::cout << "error_add_max_dpres = " << std::setprecision(10) << error_add_max_dpres << std::endl << std::endl;

	std::cout << "error1 = " << std::setprecision(10) << error1 << std::endl;
	std::cout << "error2 = " << std::setprecision(10) << error2 << std::endl;
	std::cout << "error3 = " << std::setprecision(10) << error3 << std::endl;
}
            
void test2()
{
	using namespace std;

	std::random_device rd;
	std::mt19937 e2(rd());
	std::uniform_real_distribution<> dist(0.000001, 5.0);

	double max_real_error = 0;
	float esti_error = 0;

	for (int i = 0; i < 100000; ++i)
	{
		float fp = 1.0;
		double dp = 1.0;

		double real_error = 0;

		for (int j = 0; j < 10; ++j)
		{
			double r = dist(e2);

			dp *= r;
			fp *= (float)r;
		}

		real_error = abs(dp - fp);

		if (real_error > max_real_error)
		{
			swap(real_error, max_real_error);

			esti_error = abs(fp) * ErrGamma(10);
		}
	}

	std::cout << "max_real_error = " << std::setprecision(10) << max_real_error << std::endl << std::endl;
	std::cout << "esti_error = " << std::setprecision(10) << esti_error << std::endl << std::endl;
}

int main()
{
	test2();
	return 0;
}