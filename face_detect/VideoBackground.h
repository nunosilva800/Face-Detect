#pragma once
#include "ibackground.h"
#include <string>

class VideoBackground :
	public IBackground
{
public:
	VideoBackground(std::string filename);
	~VideoBackground(void);

	IplImage * GetBackground();

private:
	CvCapture * m_capture;
	std::string m_filename;
};

