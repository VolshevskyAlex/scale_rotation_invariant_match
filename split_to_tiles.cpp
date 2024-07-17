#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>

#include "match_pict.h"
#include "visualization.h"

#define SRC_FILE	R"(D:\al_proj\n_pro\templ_match\t\satellite10.png)"
#define TILES_FILE	"tiles.bin"

void gen_cells(cv::Size img_sz, int r0, std::vector<cv::Point>& a)
{
	double r = r0 * .75;
	double dx = 2 * r;
	double dy = dx * sin(CV_PI / 3.);
	double r2 = hypot(r, dy / 3.);
	int b = ceil(r2 - r + r0 - r);
	int N = (img_sz.width - 2 * b) / dx;
	int M = (img_sz.height - 2 * b) / dy;
	for (int i = 0; i < M; i++)
	{
		double ofs_y = b + r + i * dy;
		for (int j = 0; j < N - (i & 1); j++)
		{
			double ofs_x = b + (i & 1 ? 2. * r : r) + j * dx;
			a.push_back(cv::Point(ofs_x, ofs_y));
		}
	}
}

void gen_tiles()
{
	cv::Mat1b img = cv::imread(SRC_FILE, 0);

	std::vector<cv::Point> a;
	int R = 128;
	gen_cells(img.size(), R, a);

	Matcher m;
	std::vector<cv::Mat1f> fft_lp(a.size()), fft_img(a.size());
	for (int i = 0; i < a.size(); i++)
	{
		printf("%d/%d\n", i, a.size());
		m.to_fft_log_polar_v3(img(cv::Rect(a[i].x - R, a[i].y - R, 2 * R, 2 * R)), fft_img[i], fft_lp[i]);
	}

	std::ofstream os(TILES_FILE, std::ios::binary);
	Save(os, fft_lp);

	cv::imwrite("satellite10.jpg", img);
	cv::imshow("img", img);
	cv::waitKey();
}

static cv::Mat get_rot_mat(const cv::Point2f& c, double phi, double s, const cv::Point2f& new_c)
{
	phi *= CV_PI / 180.;
	double a = s * cos(phi);
	double b = s * sin(phi);
	cv::Mat1d M1 = cv::Mat1d::eye(3, 3), M2 = cv::Mat1d::eye(3, 3), M3 = cv::Mat1d::eye(3, 3);
	M1(0, 2) = -c.x;
	M1(1, 2) = -c.y;
	M2(0, 0) = a;
	M2(0, 1) = b;
	M2(1, 0) = -b;
	M2(1, 1) = a;
	M3(0, 2) = new_c.x;
	M3(1, 2) = new_c.y;
	return (M3 * M2 * M1)(cv::Rect(0, 0, 3, 2));
}

void test_tiles()
{
	cv::Mat1b img = cv::imread("satellite10.jpg", 0);

	std::vector<cv::Mat1f> fft_lp;
	std::ifstream is(TILES_FILE, std::ios::binary);
	Load(is, fft_lp);

	std::vector<cv::Point> a;
	int R = 128;
	gen_cells(img.size(), R, a);

	double sc_x = 1920. / img.cols;
	double sc_y = 1080. / img.rows;
	double sc = std::min(sc_x, sc_y);
	cv::Mat1b img_sc;
	cv::resize(img, img_sc, cv::Size(), sc, sc);

	Matcher m;

	cv::Rect roi(4569, 4007, 4000, 4000);
	for (;;)
	{
		cv::Mat1b img4s;

		cv::Point2f loc(roi.x + (double)roi.width * rand() / RAND_MAX, roi.y + (double)roi.height * rand() / RAND_MAX);
		double gen_angle = 180. * rand() / RAND_MAX;
		double gen_scale = 1 + .3 * rand() / RAND_MAX;
		printf("angle=%f, scale=%f\n", gen_angle, gen_scale);

		cv::Size dst_size(256, 256);
		cv::Mat M1 = get_rot_mat(loc, gen_angle, gen_scale, cv::Point2f(dst_size.width / 2, dst_size.height / 2));
		cv::warpAffine(img, img4s, M1, dst_size, cv::INTER_LINEAR);

		cv::Mat1f a_img4s, fft_lp_img4s;
		m.to_fft_log_polar_v3(img4s, a_img4s, fft_lp_img4s);

		size_t cnt = fft_lp.size();
		std::vector<double> resp(cnt), angle(cnt), scale(cnt);
		double max_resp = 0;
		int max_id = 0;
		for (int i = 0; i < cnt; i++)
		{
			double dx = a[i].x - roi.x;
			double dy = a[i].y - roi.y;
			if (dx < -R || dx > roi.width + R || dy < -R || dy > roi.height + R)
				continue;

//			printf("%d/%d\n", i, a.size());
			m.match2imgs_v3(fft_lp[i], fft_lp_img4s, resp[i], angle[i], scale[i]);
			if (resp[i] > max_resp)
			{
				max_resp = resp[i];
				max_id = i;
			}
		}
		cv::Mat1b img_sim = img(cv::Rect(a[max_id].x - R, a[max_id].y - R, 2 * R, 2 * R)), img4s_rt;
		cv::Mat1f img_sim_f(img_sim);
		cv::Mat1f img4s_f(img4s);
		cv::Mat affine;
		m.aling2imgs(img_sim_f, img4s_f, angle[max_id], scale[max_id], affine);
		cv::warpAffine(img4s, img4s_rt, affine, cv::Size(img_sim.size()));

		cv::Mat1b img_sc_c = img_sc.clone();
		cv::circle(img_sc_c, loc * sc, 15, cv::Scalar(255));
		cv::Mat3b s(img_sc.size());
		s.setTo(0);
		for (int i = 0; i < a.size(); i++)
		{
			cv::circle(s, a[i] * sc, R * sc, cv::Scalar(255, 255, 255) * resp[i] / max_resp, -1);
			cv::line(s, a[i] * sc, cv::Point(a[i].x + R * cos(angle[i] * CV_PI / 180), a[i].y + R * sin(angle[i] * CV_PI / 180)) * sc, cv::Scalar(0, 0, 255));
		}
		cv::circle(s, loc * sc, 15, cv::Scalar(255, 0, 0));
		cv::imshow("satellite", img_sc_c);
		cv::imshow("reponse_map", s);
		cv::imshow("img_similar", img_sim);
		cv::imshow("img4search", img4s);
		cv::imshow("img4search_wrap", img4s_rt);
		if (cv::waitKey() == 27)
			break;
	}
}
