#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <numeric>

#include "match_pict.h"

using namespace cv;

const int pict_size = 200;

Matcher::Matcher()
{
	cv::Size sz(fft_size_, fft_size_);
	Point2f center(sz.width / 2, sz.height / 2);
	double d = hypot(center.x, center.y);
	log_base_ = pow(10.0, log10(d) / sz.width);
	map_x_.create(sz);
	map_y_.create(sz);
	for (int i = 0; i < sz.height; i++)
	{
		double theta = CV_PI * (0.5 + (double)i / sz.height);
		double sin_theta = sin(theta);
		double cos_theta = cos(theta);
		for (int j = 0; j < sz.width; j++)
		{
			double radius = pow(log_base_, (double)j); //d * j / sz.width
			double x = radius * sin_theta + center.x;
			double y = radius * cos_theta + center.y;
			map_x_(i, j) = (float)x;
			map_y_(i, j) = (float)y;
		}
	}
	highpass_f_ = create_gauss_FFT(40, 1).mul(create_highpass_FFT());
//	highpass_f_ = create_highpass_FFT();

#if 1
	int D = (fft_size_ / 4) - 1;
	int R = D / 2;
	mask_.create(fft_size_, fft_size_);
	mask_.setTo(0);
	cv::circle(mask_, cv::Point(mask_.cols / 2, mask_.rows / 2), mask_.cols / 2 - R, cv::Scalar(1), -1);
	cv::GaussianBlur(mask_, mask_, cv::Size(D, D), R * .5, R * .5, BORDER_CONSTANT);
#else
	cv::createHanningWindow(mask_, cv::Size(fft_size_, fft_size_), CV_32F);
#endif
}

void Matcher::fft_shift(cv::Mat1f& a)
{
	cv::Mat1f tmp(a.size());
	int w = a.cols / 2, h = a.rows / 2;
	a(Rect(0, 0, w, h)).copyTo(tmp(Rect(w, h, w, h)));
	a(Rect(w, h, w, h)).copyTo(tmp(Rect(0, 0, w, h)));
	a(Rect(w, 0, w, h)).copyTo(tmp(Rect(0, h, w, h)));
	a(Rect(0, h, w, h)).copyTo(tmp(Rect(w, 0, w, h)));
	a = tmp;
}

cv::Mat1f Matcher::create_gauss_FFT(int radius, double sigma)
{
	cv::Size imsize(fft_size_, fft_size_);
	Mat kernelX = getGaussianKernel(2 * radius + 1, sigma, CV_32F);
	Mat kernelY = getGaussianKernel(2 * radius + 1, sigma, CV_32F);
	// create 2d gauss
	Mat1f kernel = kernelX * kernelY.t();

	int w = imsize.width - kernel.cols;
	int h = imsize.height - kernel.rows;

	int r = w / 2;
	int l = imsize.width - kernel.cols - r;

	int b = h / 2;
	int t = imsize.height - kernel.rows - b;

	copyMakeBorder(kernel, kernel, t, b, l, r, BORDER_CONSTANT, Scalar::all(0));
	fft_shift(kernel);

	cv::Mat1f F0[2];
	Mat complexImg;
	dft(kernel, complexImg, DFT_COMPLEX_OUTPUT);
	split(complexImg, F0);
	fft_shift(F0[0]);

	return F0[0];
}

cv::Mat1f Matcher::create_highpass_FFT()
{
	Size sz(fft_size_, fft_size_);
	Mat a = Mat(sz.height, 1, CV_32FC1);
	Mat b = Mat(1, sz.width, CV_32FC1);

	float step_y = CV_PI / sz.height;
	float val = -CV_PI * 0.5;

	for (int i = 0; i < sz.height; ++i)
	{
		a.at<float>(i) = cos(val);
		val += step_y;
	}

	val = -CV_PI * 0.5;
	float step_x = CV_PI / sz.width;
	for (int i = 0; i < sz.width; ++i)
	{
		b.at<float>(i) = cos(val);
		val += step_x;
	}

	Mat tmp = a * b;
	tmp = (1.0 - tmp).mul(2.0 - tmp);

	return tmp;
}

#pragma pack(push, 1)
struct complex_t
{
	float Re;
	float Im;
};
#pragma pack(pop)

static void inline norm_v(float* v, int step)
{
	float len = hypotf(v[0], v[step]);
	if (len > FLT_EPSILON)
	{
		v[0] /= len;
		v[step] /= len;
	}
	else
	{
		v[0] = 0.f;
		v[step] = 0.f;
	}
}

static void inline norm_v(float* v)
{
	float len = fabs(*v);
	if (len > FLT_EPSILON)
	{
		*v /= len;
	}
	else
	{
		*v = 0.f;
	}
}

void perf_norm(cv::Mat1f& FFT)
{
	// CCS packed format (complex-conjugate-symmetrical)
	int M = FFT.rows;
	int N = FFT.cols;
	int K = N / 2 - 1;

	float* f_a = FFT.ptr<float>(0, 0);

	{
		float* p_a = f_a + 1;
		for (int i = 0; i < M; i++)
		{
			for (int j = 0; j < K; j++)
			{
				norm_v(p_a, 1);
				p_a += 2;
			}
			p_a += 2;
		}
	}

	int ofs = N;
	for (int i = 1; i < M / 2; i++)
	{
		norm_v(f_a + ofs, N);
		ofs += N - 1;
		norm_v(f_a + ofs, N);
		ofs += N + 1;
	}
	norm_v(FFT.ptr<float>(0, 0));
	norm_v(FFT.ptr<float>(0, FFT.cols - 1));
	norm_v(FFT.ptr<float>(FFT.rows - 1, 0));
	norm_v(FFT.ptr<float>(FFT.rows - 1, FFT.cols - 1));
}

static void inline phase_corr_eq(const complex_t& a, const complex_t& b, complex_t& dst)
{
	dst.Re = a.Re * b.Re + a.Im * b.Im;
	dst.Im = a.Im * b.Re - a.Re * b.Im;
}

static void inline phase_corr_eq_f(float* a, float* b, float* dst, int step)
{
	complex_t t_c;
	phase_corr_eq(complex_t{ a[0], a[step] }, complex_t{ b[0], b[step] }, t_c);
	dst[0] = t_c.Re;
	dst[step] = t_c.Im;
}

void perf_phase_corr_eq(cv::Mat1f& FFT1, cv::Mat1f& FFT2, cv::Mat1f& C)
{
	// CCS packed format (complex-conjugate-symmetrical)
	int M = FFT1.rows;
	int N = FFT1.cols;
	int K = N / 2 - 1;
	C.create(M, N);

	float* f_a = FFT1.ptr<float>(0, 0);
	float* f_b = FFT2.ptr<float>(0, 0);
	float* f_c = C.ptr<float>(0, 0);

	complex_t* p_a = (complex_t*)(f_a + 1);
	complex_t* p_b = (complex_t*)(f_b + 1);
	complex_t* p_c = (complex_t*)(f_c + 1);
	for (int i = 0; i < M; i++, p_a++, p_b++, p_c++)
	{
		for (int j = 0; j < K; j++)
		{
			phase_corr_eq(*p_a++, *p_b++, *p_c++);
		}
	}

	int ofs = N;
	for (int i = 1; i < M / 2; i++)
	{
		phase_corr_eq_f(f_a + ofs, f_b + ofs, f_c + ofs, N);
		ofs += N - 1;
		phase_corr_eq_f(f_a + ofs, f_b + ofs, f_c + ofs, N);
		ofs += N + 1;
	}
	cv::Point p[] = { cv::Point(0, 0), cv::Point(0, C.cols - 1), cv::Point(C.rows - 1, 0),cv::Point(C.rows - 1, C.cols - 1) };
	for (int i = 0; i < 4; i++)
	{
		C.at<float>(p[i]) = FFT1.at<float>(p[i]) * FFT2.at<float>(p[i]);
	}
}

void Matcher::to_fft_log_polar_v3(const cv::Mat1b& img, cv::Mat1f& img_f, cv::Mat1f& lp_FFT)
{
	Mat1f F0[2];
	cv::Mat1f magn, magn_lp;
	img.convertTo(img_f, CV_32FC1, 1.0 / 255.0);

	img_f = img_f.mul(mask_);

	Mat complexImg;
	dft(img_f, complexImg, DFT_COMPLEX_OUTPUT);
	split(complexImg, F0);
	magnitude(F0[0], F0[1], magn);
	fft_shift(magn);

	magn = magn.mul(highpass_f_);
	cv::normalize(magn, magn, 0, 1, NORM_MINMAX);
	cv::remap(magn, magn_lp, map_x_, map_y_, cv::INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));
	cv::dft(magn_lp, lp_FFT, DFT_REAL_OUTPUT);
//	cv::log(lp_FFT, lp_FFT);
//	cv::normalize(lp_FFT, lp_FFT, 0, 1, NORM_MINMAX);

	perf_norm(lp_FFT);
/*
	double minv, maxv;
	cv::minMaxLoc(lp_FFT, &minv, &maxv);
	cv::imshow("img_f", img_f);
	cv::imshow("magn_lp", magn_lp);
	cv::imshow("lp_FFT", lp_FFT);
	cv::waitKey();
*/
}

static Point2d weightedCentroid(cv::Mat1f src, cv::Point peakLocation, cv::Size weightBoxSize, double* response)
{
	int minr = peakLocation.y - (weightBoxSize.height >> 1);
	int maxr = peakLocation.y + (weightBoxSize.height >> 1);
	int minc = peakLocation.x - (weightBoxSize.width >> 1);
	int maxc = peakLocation.x + (weightBoxSize.width >> 1);

	Point2d centroid;
	double sumIntensity = 0.0;

	// clamp the values to min and max if needed.
	if (minr < 0)
	{
		minr = 0;
	}

	if (minc < 0)
	{
		minc = 0;
	}

	if (maxr > src.rows - 1)
	{
		maxr = src.rows - 1;
	}

	if (maxc > src.cols - 1)
	{
		maxc = src.cols - 1;
	}

	const float* dataIn = src.ptr<float>();
	dataIn += minr * src.cols;
	for (int y = minr; y <= maxr; y++)
	{
		for (int x = minc; x <= maxc; x++)
		{
			centroid.x += (double)x * dataIn[x];
			centroid.y += (double)y * dataIn[x];
			sumIntensity += (double)dataIn[x];
		}

		dataIn += src.cols;
	}

	if (response)
		*response = sumIntensity;

	sumIntensity += DBL_EPSILON; // prevent div0 problems...

	centroid.x /= sumIntensity;
	centroid.y /= sumIntensity;

	return centroid;
}

cv::Point2d Matcher::phase_correlate_custom(cv::Mat1f& FFT1, cv::Mat1f& FFT2, double &response)
{
	cv::Mat1f C;
	perf_phase_corr_eq(FFT1, FFT2, C);

	idft(C, C);
	fft_shift(C); // shift the energy to the center of the frame.

	// locate the highest peak
	Point peakLoc;
	minMaxLoc(C, NULL, NULL, NULL, &peakLoc);

	// get the phase shift with sub-pixel accuracy, 5x5 window seems about right here...
	Point2d t;
	t = weightedCentroid(C, peakLoc, Size(5, 5), &response);

	// max response is M*N (not exactly, might be slightly larger due to rounding errors)

	int M = FFT1.rows;
	int N = FFT1.cols;

	response /= M * N;

	// adjust shift relative to image center...
	Point2d center((double)FFT1.cols / 2.0, (double)FFT1.rows / 2.0);

	return (center - t);
}

void Matcher::match2imgs_v3(cv::Mat1f& lp0_FFT, cv::Mat1f& lp1_FFT, double& response, double& angle, double& scale)
{
	Point2d rotation_and_scale = phase_correlate_custom(lp0_FFT, lp1_FFT, response);
	angle = 180.0 * -rotation_and_scale.y / fft_size_;
	scale = pow(log_base_, -rotation_and_scale.x);
}

void Matcher::aling2imgs(cv::Mat1f& img0, cv::Mat1f& img1, double angle, double scale, cv::Mat& affine)
{
	double max_response = -DBL_MAX;
	for (int j = 0; j < 2; j++)
	{
		Mat rot_mat = cv::getRotationMatrix2D(Point(fft_size_ / 2, fft_size_ / 2), angle, 1.0 / scale);

		// rotate and scale
		Mat1f img1_rs;
		cv::warpAffine(img1, img1_rs, rot_mat, cv::Size(fft_size_, fft_size_), 1, BORDER_CONSTANT, Scalar::all(0));

		// find translation
		cv::Mat1f fft_img1_rs, fft_img0;
		dft(img1_rs, fft_img1_rs, DFT_REAL_OUTPUT);
		dft(img0, fft_img0, DFT_REAL_OUTPUT);
		perf_norm(fft_img1_rs);
		perf_norm(fft_img0);
		double response;
		Point2d tr = phase_correlate_custom(fft_img1_rs, fft_img0, response);
		printf("%f\n", response);
		if (response > max_response)
		{
			max_response = response;
			affine = rot_mat.clone();
			affine.at<double>(0, 2) += tr.x;
			affine.at<double>(1, 2) += tr.y;
		}
		angle += 180;
	}
}

cv::Mat3b draw_pict(const std::vector<cv::Mat3b>& pict_sampl)
{
	cv::Mat3b img2(cv::Size(pict_size * 4, pict_size * 2));
	for (size_t i = 0; i < pict_sampl.size(); i++)
	{
		int x = (i % 4) * pict_size;
		int y = (i / 4) * pict_size;
		pict_sampl[i].copyTo(img2(cv::Rect(x, y, pict_size, pict_size)));
	}
	return img2;
}
