#include <random>
#include <chrono>

#include <iostream>
#include <stdexcept>

#if defined(__arm__) || defined(__aarch64__)
#include <arm_neon.h>
#elif defined(__SSE__) || defined(_M_X64) || defined(_M_IX86_FP) || defined(__AVX__)
#include <immintrin.h>
#endif

float l2_distance(const float *x, const float *y, int d) {
	if (d) {
		float sum = 0;
		for (size_t i = 0; i < d; i++) {
			auto substract = *x - *y;
			sum += substract * substract;
			++x;
			++y;
		}
		return sum;
	}
	else {
		throw std::runtime_error("length of input vecotrs must be the same");
	}
	return 0;
}

#if defined(__AVX__)
float l2_distance_avx(const float *x, const float *y, int d) {
	__m256 msum1 = _mm256_setzero_ps();

	while (d >= 8) {
		const __m256 mx = _mm256_loadu_ps(x); x += 8;
		const __m256 my = _mm256_loadu_ps(y); y += 8;
		const  __m256 a_m_b1 = _mm256_sub_ps(mx, my);
		const __m256 a_minus_b_sq = _mm256_mul_ps(a_m_b1, a_m_b1);
		msum1 = _mm256_add_ps(msum1, a_minus_b_sq);
		d -= 8;
	}

	const __m128 msum2_1 = _mm256_extractf128_ps(msum1, 1);
	__m128 msum2 = _mm256_extractf128_ps(msum1, 0);
	msum2 = _mm_add_ps(msum2_1, msum2);

	if (d >= 4) {
		const __m128 mx = _mm_loadu_ps(x); x += 4;
		const __m128 my = _mm_loadu_ps(y); y += 4;
		const  __m128 a_m_b1 = _mm_sub_ps(mx, my);
		const __m128 a_minus_b_sq = _mm_mul_ps(a_m_b1, a_m_b1);
		msum2 = _mm_add_ps(msum2, a_minus_b_sq);
		d -= 4;
	}

	msum2 = _mm_hadd_ps(msum2, msum2);
	msum2 = _mm_hadd_ps(msum2, msum2);
	float sum = _mm_cvtss_f32(msum2);

	if (d > 0) {
		for (size_t i = 0; i < d; i++) {
			auto sub = x[i] - y[i];
			sum += sub * sub;
		}
	}
	return sum;
}

#elif defined(__SSE__) || defined(_M_X64) || defined(_M_IX86_FP)
float l2_distance_sse(const float* x, const float* y, int n) {
	float result = 0;
	__m128 euclidean = _mm_setzero_ps();
	for (; n > 3; n -= 4) {
		const __m128 a = _mm_loadu_ps(x);
		const __m128 b = _mm_loadu_ps(y);
		const __m128 a_minus_b = _mm_sub_ps(a, b);
		const __m128 a_minus_b_sq = _mm_mul_ps(a_minus_b, a_minus_b);
		euclidean = _mm_add_ps(euclidean, a_minus_b_sq);
		x += 4;
		y += 4;
	}
	const __m128 shuffle1 = _mm_shuffle_ps(euclidean, euclidean, _MM_SHUFFLE(1, 0, 3, 2));
	const __m128 sum1 = _mm_add_ps(euclidean, shuffle1);
	const __m128 shuffle2 = _mm_shuffle_ps(sum1, sum1, _MM_SHUFFLE(2, 3, 0, 1));
	const __m128 sum2 = _mm_add_ps(sum1, shuffle2);

	_mm_store_ss(&result, sum2);
	if (n)
		result += l2_distance(x, y, n);
	return result;
}

#elif defined(__arm__) || defined(__aarch64__)
float l2_distance_neon(const float *v1, const float *v2, int n_elems) {
	int n_handled_elems = 0;
	float32x4_t sum = vdupq_n_f32(0);
	while (n_handled_elems <= n_elems - 4) {
		float32x4_t x = vld1q_f32(v1 + n_handled_elems);
		float32x4_t y = vld1q_f32(v2 + n_handled_elems);
		float32x4_t sub = vsubq_f32(x, y);
		float32x4_t sub_sqr = vmulq_f32(sub, sub);
		sum = vaddq_f32(sum, sub_sqr);
		n_handled_elems += 4;
	}

	float32x2_t r = vadd_f32(vget_high_f32(sum), vget_low_f32(sum));
	float32_t result = vget_lane_f32(r, 0) + vget_lane_f32(r, 1);

	// handle remaining elements (< 4);
	while (n_handled_elems < n_elems) {
		result += v1[n_handled_elems] * v2[n_handled_elems];
		++n_handled_elems;
	}
	return result;
}
#endif

float SIMD_l2_distance(const float *x, const float *y, int d) {
#if defined(__AVX__)
	return l2_distance_avx(x, y, d);
#elif defined(__SSE__) || defined(_M_X64) || defined(_M_IX86_FP)
	return l2_distance_sse(x, y, d);
#elif defined(__arm__) || defined(__aarch64__)
	return l2_distance_neon(x, y, d);
#endif
}

int main() {
	int vec_length = 16;
	int repeat_count = 10000;

	double time_SIMD = 0.0;
	double time_NO_SIMD = 0.0;

	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_real_distribution<float> dist(-100., 100.);
	
	for (int r = 0; r < repeat_count; r++) {
		// generate two random vectors with float values
		std::vector<float> a_vec;
		std::vector<float> b_vec;
		a_vec.reserve(vec_length);
		b_vec.reserve(vec_length);
		for (int i = 0; i < vec_length; i++) {
			a_vec.push_back(dist(mt));
			b_vec.push_back(dist(mt));
		}

		float *a = a_vec.data();
		float *b = b_vec.data();
		
		// get execution time without SIMD
		auto start_NO_SIMD = std::chrono::high_resolution_clock::now();
		float res_NO_SIMD = l2_distance(a, b, vec_length);
		auto finish_NO_SIMD = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> elapsed_NO_SIMD = finish_NO_SIMD - start_NO_SIMD;
		time_NO_SIMD += elapsed_NO_SIMD.count();

		// get execution time for SIMD
		auto start_SIMD = std::chrono::high_resolution_clock::now();
		float res_SIMD = SIMD_l2_distance(a, b, vec_length);
		auto finish_SIMD = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> elapsed_SIMD = finish_SIMD - start_SIMD;
		time_SIMD += elapsed_SIMD.count();

		// check if l2 distance have the same value with and without SIMD
		if (res_SIMD - res_NO_SIMD > 0.05) {
			std::cout << "mis-matched values: " << res_SIMD << " | " << res_NO_SIMD << " | " << res_SIMD-res_NO_SIMD << std::endl;
		}
	}

	std::cout << "overall time for SIMD is: " << time_SIMD << std::endl;
	std::cout << "overall time without SIMD is: "  << time_NO_SIMD << std::endl;
}