#include "VideoBackground.h"


VideoBackground::VideoBackground(std::string filename)
{
	m_filename = filename;
	m_capture = cvCaptureFromFile(filename.c_str());
}


VideoBackground::~VideoBackground(void)
{
	cvReleaseCapture(&m_capture);
}


IplImage * VideoBackground::GetBackground()
{
	int current = cvGetCaptureProperty(m_capture, CV_CAP_PROP_POS_FRAMES);
	int last = cvGetCaptureProperty(m_capture, CV_CAP_PROP_FRAME_COUNT);
	if(current == last) 
	{
		cvReleaseCapture(&m_capture);
		m_capture = cvCaptureFromFile(m_filename.c_str());
	}
	return cvQueryFrame(m_capture);
}
