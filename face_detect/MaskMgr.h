#pragma once

#include <vector>
#include <direct.h> //for mkdir()

#include "ImageBackground.h"
#include "VideoBackground.h"

#include "cv.h"
#include "cvaux.h"
#include "highgui.h"

/*//forward declarations
typedef struct IplImage;
*/


class MaskMgr
{

public:
	static MaskMgr * GetInstance();

	IplImage * GetFace(unsigned i);
	IplImage * GetBackground(unsigned i);		//operator[] (unsigned i);


	int GetNumFaces();
	int GetNumBackgrounds();
	
	int Length();	


	void ApplyMask(IplImage* src, unsigned maskidx, CvRect location);
	void ApplyBackground(IplImage* src, IplImage* bg_wallpaper_mask, int bckidx);

private:
	MaskMgr(void);
	~MaskMgr(void);

	static MaskMgr * m_pInstance;

	
	std::vector<IplImage *> m_faces;
	std::vector<IBackground *> m_backgrounds;
};

