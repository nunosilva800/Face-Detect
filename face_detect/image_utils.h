#pragma once;

#include <omp.h>
#include "cv.h"
#include "cvaux.h"
#include "highgui.h"


namespace ImageUtils
{


	

// Creates a new image copy that is of a desired size.
// Remember to free the new image later.
IplImage* resizeImage(const IplImage *origImg, int newWidth, int newHeight);

// Return a new image that is always greyscale, whether the input image was RGB or Greyscale.
// Remember to free the returned image using cvReleaseImage() when finished.
IplImage* convertImageToGreyscale(const IplImage *imageSrc);


// Returns a new image that is a cropped version of the original image. 
IplImage* cropImage(const IplImage *img, const CvRect region);


// Get an 8-bit equivalent of the 32-bit Float image.
// Returns a new image, so remember to call 'cvReleaseImage()' on the result.
IplImage* convertFloatImageToUcharImage(const IplImage *srcImg);







}

