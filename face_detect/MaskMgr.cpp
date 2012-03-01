#include "MaskMgr.h"

#include <sys/types.h>
#include <windows.h>
#include <iostream>

#include "image_utils.h"

using namespace std;

MaskMgr * MaskMgr::m_pInstance = 0;

MaskMgr * MaskMgr::GetInstance()
{	
	if(!m_pInstance)
	{
		m_pInstance = new MaskMgr();
		return m_pInstance;
	}

	return m_pInstance;
}


MaskMgr::MaskMgr(void)
{
	//make sure directorys exist
	mkdir("masks");
	mkdir("masks/backgrounds");
	mkdir("masks/faces");
	mkdir("masks/faces_alpha");
	
	
	//Load all the face masks from the default directory 

	WIN32_FIND_DATA FindFileData;
	HANDLE hFind;

	hFind = FindFirstFile("./masks/faces/*", &FindFileData);
	
	if(hFind == INVALID_HANDLE_VALUE)
	{
		cout << "Error loading the masks files\n";
	}

	string filename;
	string alpha_filename;
	//Let the first face mask be inexistent
	this->m_faces.push_back(NULL);


	cout << "LOADING FACE MASKS\n";

	do
	{
		
		//Ignore any directories
		if (!(FindFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
		{
			cout << "Loading face mask: " << FindFileData.cFileName << std::endl;


			filename = "./masks/faces/";
			alpha_filename = "./masks/faces_alpha/";

			filename += FindFileData.cFileName;
			alpha_filename += FindFileData.cFileName;


			// We need a little hack here and load the alpha chanel from a separate image since
			// openCV doesn't load the alpha channel
			IplImage * p = cvLoadImage(filename.c_str(), -1);
			IplImage * p_alpha = cvLoadImage(alpha_filename.c_str(), -1);


			IplImage * r = cvCreateImage(cvSize(p->width, p->height), p->depth, 1);
			IplImage * g = cvCreateImage(cvSize(p->width, p->height), p->depth, 1);
			IplImage * b = cvCreateImage(cvSize(p->width, p->height), p->depth, 1);
			
			IplImage * p_with_alpha = cvCreateImage(cvSize(p->width, p->height), p->depth, 4);
			cvSplit(p, r, g, b, NULL);

			cvMerge(r, g, b, p_alpha, p_with_alpha);

			if(p) 
			{				

				this->m_faces.push_back(p_with_alpha);
				cout << "Loaded mask: " << FindFileData.cFileName << std::endl;
			}
			else
			{
				cout << "Error loading mask: " << FindFileData.cFileName << std::endl;
			}
		}
		
	}while(FindNextFile(hFind, &FindFileData) != 0);


	 FindClose(hFind);

	 hFind = FindFirstFile("./masks/backgrounds/images/*", &FindFileData);
	
	if(hFind == INVALID_HANDLE_VALUE)
	{
		cout << "Error loading the background masks files\n";
	}


	//Let the first background mask be inexistent
	this->m_backgrounds.push_back(NULL);
	

	cout << "LOADING BACKGROUND MASKS:\n";

	do
	{
		
		//Ignore any directories
		if (!(FindFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
		{
			cout << "loading background MASK: " << FindFileData.cFileName << std::endl;


			filename = "./masks/backgrounds/images/";
			filename += FindFileData.cFileName;

			
			this->m_backgrounds.push_back(new ImageBackground(filename));
			cout << "Loaded background mask: " << FindFileData.cFileName << std::endl;
			
		}
		
	}while(FindNextFile(hFind, &FindFileData) != 0);

	 FindClose(hFind);






	 hFind = FindFirstFile("./masks/backgrounds/videos/*", &FindFileData);
	
	if(hFind == INVALID_HANDLE_VALUE)
	{
		cout << "Error loading the background masks files\n";
	}



	cout << "LOADING BACKGROUND MASKS:\n";

	do
	{
		
		//Ignore any directories
		if (!(FindFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
		{
			cout << "loading background MASK: " << FindFileData.cFileName << std::endl;


			filename = "./masks/backgrounds/videos/";
			filename += FindFileData.cFileName;

			
			this->m_backgrounds.push_back(new VideoBackground(filename));
			cout << "Loaded background mask: " << FindFileData.cFileName << std::endl;
			
		}
		
	}while(FindNextFile(hFind, &FindFileData) != 0);

	 FindClose(hFind);

}


void MaskMgr::ApplyMask(IplImage * dst, unsigned maskidx, CvRect location)
{
	//if(maskidx > MaskMgr::GetInstance()->GetNumFaces()){
	//	printf("Bad maskidx: %d", maskidx);
	//	return;
	//}
	IplImage *overlay = ImageUtils::resizeImage(GetFace(maskidx), location.width, location.height);

	int xwidth = overlay->width;
	int yheight = overlay->height;

	for(int x=0;x < xwidth;x++)
	{
		if(x+location.x>= dst->width) continue;

		for(int y=0; y < yheight;y++)
		{
			if(y+location.y>=dst->height) continue;  
				
			CvScalar source = cvGet2D(dst, y+location.y, x+location.x);
			CvScalar over = cvGet2D(overlay, y, x);   
			CvScalar merged;

			if(over.val[3] != 0)
			{
				merged.val[0] = (over.val[0]);        
				merged.val[1] = (over.val[1]);        
				merged.val[2] = (over.val[2]);    
				cvSet2D(dst, y+location.y, x+location.x, merged);
			}
		}
	}
	cvReleaseImage(&overlay);
}


void MaskMgr::ApplyBackground(IplImage * dst, IplImage * bg_wallpaper_mask, int bckidx)
{
	//step the movies forward :( simply horrible way to do this
	IplImage * bg_wallpaper = MaskMgr::GetInstance()->GetBackground(bckidx);
	
	if(bg_wallpaper==0)
	{
		printf("Problem reading video!\n");

		return;
	}

	bg_wallpaper = ImageUtils::resizeImage(bg_wallpaper, dst->width, dst->height);


	cvCopy(dst, bg_wallpaper, bg_wallpaper_mask);
	cvCopy(bg_wallpaper, dst);
	
	cvReleaseImage(&bg_wallpaper);
}

	
IplImage * MaskMgr::GetFace(unsigned i)
{
	if(i==0) return m_faces[1];
	else return m_faces[i];
}

IplImage * MaskMgr::GetBackground(unsigned i)
{
	if(m_backgrounds[i])
		return m_backgrounds[i]->GetBackground();

	return NULL;
}

int MaskMgr::GetNumFaces()
{
	return m_faces.size();
}

int MaskMgr::GetNumBackgrounds()
{
	return m_backgrounds.size();
}

MaskMgr::~MaskMgr(void)
{
	m_faces.clear();
	m_backgrounds.clear();
}
