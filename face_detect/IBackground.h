#pragma once 

#include "cv.h"
#include "cvaux.h"
#include "highgui.h"

class IBackground
{
public:

	IBackground()
	{
	}

	virtual ~IBackground(void)
	{
	}


	virtual IplImage * GetBackground() = 0;
};

