#pragma once

#include <fstream>

void text_out(cv::Mat &image, std::string text, cv::Point org, int fontFace, double fontScale, cv::Scalar color);
void show_1f(char *wn, const cv::Mat1f &src, double min_v, double max_v);
void show_1f(char *wn, const cv::Mat1f &src);
void write_1f(char *fn, const cv::Mat1f &src, double min_v, double max_v);
void write_1f(char *fn, const cv::Mat1f &src);
void show_threshold(char *wn, const cv::Mat &image, const cv::Mat1f &a);
void show_color_map(char *wn, cv::Mat1f &a);
void debug_printf(char *fmt, ...);

struct prof_time {
	int64 m_dur;

	prof_time()
	{
		m_dur = 0;
	}
	void start()
	{
		m_dur -= cv::getTickCount();
	}
	void stop()
	{
		m_dur += cv::getTickCount();
	}
	double duration()
	{
		static double freq = cv::getTickFrequency();
		return m_dur / freq;
	}
};

void ReadMat(std::istream &s, cv::Mat &M);
void WriteMat(std::ostream &s, const cv::Mat &M);

template <class T>
static inline void Save(std::ostream &s, T &v)
{
	s.write((char*)&v, sizeof(T));
}

template <class T>
static inline void Save(std::ostream &s, cv::Mat_<T> &v)
{
	s.write((char*)&v.rows, sizeof(v.rows));
	s.write((char*)&v.cols, sizeof(v.cols));
	s.write((char*)v.data, v.total() * v.elemSize());
}

template <class T>
static inline void Save(std::ostream &s, std::vector<T> &v)
{
	int size = v.size();
	s.write((char*)&size, sizeof(size));
	for (int i = 0; i < size; i++)
	{
		Save(s, v[i]);
	}
}

template <class T, class Q>
static inline void Save(std::ostream &s, std::pair<T, Q> &v)
{
	Save(s, v.first);
	Save(s, v.second);
}

static inline void Save(std::ostream &s, std::string &v)
{
	int size = v.size();
	s.write((char*)&size, sizeof(size));
	s.write(v.c_str(), size);
}

///
template <class T>
static inline void Load(std::istream &s, T &v)
{
	s.read((char*)&v, sizeof(T));
}

template <class T>
static inline void Load(std::istream &s, cv::Mat_<T> &v)
{
	int rows, cols;
	s.read((char*)&rows, sizeof(v.rows));
	s.read((char*)&cols, sizeof(v.cols));
	v.create(rows, cols);
	s.read((char*)v.data, v.total() * v.elemSize());
}

template <class T>
static inline void Load(std::istream &s, std::vector<T> &v)
{
	int size;
	s.read((char*)&size, sizeof(size));
	v.resize(size);
	for (int i = 0; i < size; i++)
	{
		Load(s, v[i]);
	}
}

template <class T, class Q>
static inline void Load(std::istream &s, std::pair<T, Q> &v)
{
	Load(s, v.first);
	Load(s, v.second);
}

static inline void Load(std::istream &s, std::string &v)
{
	int size;
	s.read((char*)&size, sizeof(size));
	v.resize(size);
	s.read(&v[0], size);
}
