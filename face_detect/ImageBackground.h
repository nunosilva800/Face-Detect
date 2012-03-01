#pragma once
#include "ibackground.h"

#include <string>

class ImageBackground :
	public IBackground
{
public:
	ImageBackground(std::string filename) 
	{
		m_background = cvLoadImage(filename.c_str(), -1);
	}

	~ImageBackground(void)
	{
		cvReleaseImage(&m_background);
	}

	IplImage * GetBackground() { return m_background;}

private:
	IplImage * m_background;
};

