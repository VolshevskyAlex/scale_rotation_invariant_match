#pragma once

#include <opencv2/opencv.hpp>

cv::Mat1b select_foreground(cv::Mat3b& img);
void split_imgs(cv::Mat3b img, std::vector<cv::Mat3b>& pict_sampl, cv::Mat3b& v);

cv::Mat3b draw_pict(const std::vector<cv::Mat3b>& pict_sampl);

struct MatchInfo
{
	int i;
	int j;
	double response;

	cv::Point2d center;

	// from j to i
	cv::Mat affine;
	double angle;
	double scale;
};

class Matcher
{
public:
	Matcher();
	void match_pict_list(const  std::vector<cv::Mat3b>& sampl0, const  std::vector<cv::Mat3b>& sampl1, MatchInfo& match);

//private:
	struct A
	{
		int i;
		int j;
		double response;
		cv::Point2d rotation_and_scale;
	};
	cv::Mat highpass_f_;
	cv::Mat1f map_x_;
	cv::Mat1f map_y_;
	cv::Mat1f mask_;
	double log_base_;
	const int fft_size_ = 256;

	void to_fft_log_polar_v3(const cv::Mat1b& img, cv::Mat1f& img_f, cv::Mat1f& lp_FFT);
	void match2imgs_v3(cv::Mat1f& lp0_FFT, cv::Mat1f& lp1_FFT, double& response, double& angle, double& scale);
	void aling2imgs(cv::Mat1f& img0, cv::Mat1f& img1, double angle, double scale, cv::Mat &affine);
//	void match2imgs(cv::Mat1f& img_f0, cv::Mat1f& magn_lp0, cv::Mat1f& img_f1, cv::Mat1f& magn_lp1, MatchInfo &info);


	cv::Mat1f create_highpass_FFT();
	cv::Mat1f create_gauss_FFT(int radius, double sigma);
	void fft_shift(cv::Mat1f& a);

	cv::Point2d phase_correlate_custom(cv::Mat1f& FFT1, cv::Mat1f& FFT2, double& response);
};
