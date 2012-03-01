/* 
Face Detection and Recognition with (not yet)per-user "masking", and background changes

Nuno Silva
Arsénio Costa
2010

General algorithm:
	Grab a frame from the camera
	Convert the color frame to greyscale 
	Detect a face within the greyscale camera frame 
	Crop the frame to just show the face region (using cvSetImageROI() and cvCopyImage()). 
	Preprocess the face image (train)
	Recognize the person in the image
	Change Face
	Change Background
*/

/***
TODO :>
	| check if mask has dif. deph bits	
	| multi user support!
	| associate mask with registered user
***/

#include <stdio.h>
#include <conio.h>		// For _kbhit()
#include <direct.h>		// For mkdir()
#include <vector>
#include <string>
#include <omp.h>
#include <iostream>
#include "Timer.h"

#include "image_utils.h"

using namespace std;
using namespace ImageUtils;

#include "cv.h"
#include "cvaux.h"
#include "highgui.h"
#include "MaskMgr.h"


//################################################################
//						Global variables
//################################################################
//indexes for the trackbars
//set maskid to 0 disables masking
int maskidx = 1;
int maskidx_array[5] = {0,0,0,0,0};
int bckg_sub_method = 0; //0 -> cvCreateGaussianBGModel; 1 -> codebooks
int bckidx = 0;
int overlay = 1;
int face_detect_method = 1; // single or multi face
int n_previous_faces = 0;
int frames_without_faces = 0;

// Haar	 Cascade file, used for Face Detection.
const char *faceCascadeFilename = "haarcascades/haarcascade_frontalface_alt2.xml";
CvHaarClassifierCascade* faceCascade = NULL;
int SAVE_EIGENFACE_IMAGES = 1;		// Set to 0 if you dont want images of the Eigenvectors saved to files (for debugging).
// How detailed should the face recognition should the search be.
int iSearch_scale_factor = 1;

IplImage ** faceImgArr        = 0; // array of face images
CvMat    *  personNumTruthMat = 0; // array of person numbers

IplImage* bg_wallpaper;				// raw background 
IplImage* bg_wallpaper_mask;		// background binary mask

vector<string> personNames;			// array of person names (indexed by the person number). Added by Shervin.

CvSeq* rects = 0;
CvMemStorage* storage = cvCreateMemStorage(0);
CvMemStorage* multiFaceStorage = cvCreateMemStorage(0); //storage for multifacedetect
int pyrDownScale = 2;	//down-scale factor for the image onto detect faces
int faceWidth = 120;	// Default dimensions for faces in the face recognition database. Added by Shervin.
int faceHeight = 120;	//	"		"		"		"		"		"		"		"
int nPersons                  = 0; // the number of people in the training set. Added by Shervin.
int nTrainFaces               = 0; // the number of training images
int nEigens                   = 0; // the number of eigenvalues
IplImage * pAvgTrainImg       = 0; // the average image
IplImage ** eigenVectArr      = 0; // eigenvectors
CvMat * eigenValMat           = 0; // eigenvalues
CvMat * projectedTrainFaceMat = 0; // projected training faces
CvSeq * faces = 0;


CvCapture* camera = 0;	// The camera device.

/* parameters for segmentation */
bool update_bg_model = true;
CvBGStatModel* bg_model = 0;

// VARIABLES for CODEBOOK METHOD: 
CvBGCodeBookModel* codebook_model = 0;
const int NCHANNELS = 3;
bool ch[NCHANNELS]={true,true,true}; // This sets what channels should be adjusted for background bounds
unsigned int nframes=1;
int nframesToLearnBG = 50;

CvMat * trainPersonNumMat = NULL;  // the person numbers during training
float * projectedTestFace = NULL;
char cstr[256];
BOOL saveNextFaces = FALSE;
char newPersonName[256];
int newPersonFaces;

//----------------------------------------------------------------


void print_dumb()
{
	cout << "Dumb\n";
}

//################################################################
//						function signatures
//################################################################
IplImage* codebookAuxImage = 0; //codebookAuxImage is for codebook method

void doPCA();
int loadFaceImgArray(char * filename);
CvSeq* detectFaceInImage(IplImage *inputImg, CvHaarClassifierCascade* cascade);
CvSeq* detectMultiFaceInImage(IplImage *imageProcessed, CvHaarClassifierCascade* cascade);
IplImage* getCameraFrame(void);
CvMat* retrainOnline(void);
int findNearestNeighbor(float * projectedTestFace, float *pConfidence);
void storeEigenfaceImages();
void storeTrainingData();
int  loadTrainingData(CvMat ** pTrainPersonNumMat);
void doSegm(IplImage * imgCam);
void doCodeBook(IplImage * imgCam);
void OnKeyPressed(char key);
void CleanUp();
void on_change_background(int pos);
void on_change_background_method(int pos);

//----------------------------------------------------------------


void on_change_background(int pos)
{
	bg_wallpaper = MaskMgr::GetInstance()->GetBackground(pos);
}

void on_change_background_method(int pos)
{
	cvReleaseBGStatModel(&bg_model);
	cvReleaseBGCodeBookModel(&codebook_model);
	nframes = 1;
}

void background_change_callback()
{
	bckidx = (int) rand() % ( MaskMgr::GetInstance()->GetNumBackgrounds() );
}

void on_mouse( int event, int x, int y, int flags, void* param )
{
	
	switch( event )
	{
		case CV_EVENT_LBUTTONDOWN:
			bckidx = (int) rand() % ( MaskMgr::GetInstance()->GetNumBackgrounds() );
			break;
	}
}

int main( int argc, const char** argv )
{
	trainPersonNumMat = 0;  // the person numbers during training
	projectedTestFace = 0;
	saveNextFaces = FALSE;
	newPersonFaces = 0;

	omp_set_num_threads(omp_get_num_procs());

	MaskMgr *mask_mgr = MaskMgr::GetInstance();

	// Load the previously saved training data
	if( loadTrainingData( &trainPersonNumMat ) ) {
		faceWidth = pAvgTrainImg->width;
		faceHeight = pAvgTrainImg->height;
	}
	
	// Project the test images onto the PCA subspace
	projectedTestFace = (float *)cvAlloc( nEigens*sizeof(float) );

	// Make sure there is a "data" folder, for storing the new person.
	mkdir("data");

	// Load the HaarCascade classifier for face detection.
	faceCascade = (CvHaarClassifierCascade*)cvLoad(faceCascadeFilename, 0, 0, 0);
	if( !faceCascade ) {
		printf("Couldnt load Face detector '%s'\n", faceCascadeFilename);
		exit(-1);
	}

	//open the trackbar windows
#ifdef _DEBUG
	cvNamedWindow("Options", 0);
	if(MaskMgr::GetInstance()->GetNumFaces() > 1) 
		cvCreateTrackbar("Face Mask", "Options", &maskidx, MaskMgr::GetInstance()->GetNumFaces() - 1, NULL);
	if(MaskMgr::GetInstance()->GetNumBackgrounds() > 1)
		cvCreateTrackbar("Background Mask", "Options", &bckidx, MaskMgr::GetInstance()->GetNumBackgrounds() - 1, on_change_background);
	cvCreateTrackbar("Recogniton precision", "Options", &iSearch_scale_factor, 10, NULL);
	cvCreateTrackbar("Bckg method", "Options", &bckg_sub_method, 1, on_change_background_method);
	cvCreateTrackbar("Faces", "Options", &face_detect_method, 1, NULL);
#endif

	IplImage* bg_img = NULL;
	cvNamedWindow("Output", 0);

	CvFont font;
	cvInitFont(&font, CV_FONT_HERSHEY_PLAIN, 1.0, 1.0, 0,1,CV_AA);
	CvScalar textColor = CV_RGB(0,255,255);	// light blue text

	IplImage *greyImg;
	IplImage *faceImg;
	IplImage *sizedImg;
	IplImage *equalizedImg;
	IplImage *camImg;

	CvRect faceRect;
	int keyPressed = 0;
	
	float confidence = 0;
	int iNearest, nearest, truth;
	char text1[256], text2[256];

	CTimer timer;
	timer.SetTrigger(background_change_callback, 60);

	cvSetMouseCallback( "Output", on_mouse, 0 );

	while (cvWaitKey(15) != VK_ESCAPE)
	{

		timer.Update();
		// Handle keyboard input in the console.
		if (_kbhit())
			OnKeyPressed(getch());

		// Get the camera frame
		camImg = getCameraFrame();
		if (!camImg) {
			printf("ERROR in recognizeFromCam(): Bad input image!\n");
			exit(1);
		}

		// flip the image
		cvFlip(camImg, camImg, 1);

		// train the model for bg sub
		if(bckg_sub_method == 0)
		{
			doSegm(camImg);
		}
		else
		{
			doCodeBook(camImg);
		}
		
		// Make sure the image is greyscale, since the Eigenfaces is only done on greyscale image.
		greyImg = convertImageToGreyscale(camImg);
		//greyImg = cvCreateImage( cvGetSize(camImg), IPL_DEPTH_8U, 1 );
		//cvCvtColor( camImg, greyImg, CV_BGR2GRAY );

		// Perform face detection on the input image, using the given Haar cascade classifier.
		if(face_detect_method == 0)
		{
			pyrDownScale = 1;
			faces = detectFaceInImage(greyImg, faceCascade );
		}
		else
		{
			pyrDownScale = 2;
			faces = detectMultiFaceInImage(greyImg, faceCascade );
		}

		// Get the first detected face (the biggest).
		if (faces->total > 0)
		{
			faceRect = *(CvRect*)cvGetSeqElem( faces, 0 );
			faceRect.height *= pyrDownScale;
			faceRect.width *= pyrDownScale;
			faceRect.x *= pyrDownScale;
			faceRect.y *= pyrDownScale;
		}
		else
		{ // no face found 
			faceRect.height = -1;
			faceRect.width =-1;
			faceRect.x = -1;
			faceRect.y = -1;
		}

		// Make sure a valid face was detected.
		if (faceRect.width > 0) 
		{
			faceImg = cropImage(greyImg, faceRect);	// Get the detected face image.
			
			// Make sure the image is the same dimensions as the training images.
			sizedImg = resizeImage(faceImg, faceWidth, faceHeight);
			
			// Give the image a standard brightness and contrast, in case it was too dark or low contrast.
			equalizedImg = cvCreateImage(cvGetSize(sizedImg), 8, 1);	// Create an empty greyscale image
			cvEqualizeHist(sizedImg, equalizedImg);

#ifdef _DEBUG
			cvShowImage("heigenface",  equalizedImg );
#endif

			// If the face rec database has been loaded, then try to recognize the person currently detected.
			if (nEigens > 0)
			{
				// project the test image onto the PCA subspace
				cvEigenDecomposite(
					equalizedImg,
					nEigens,
					eigenVectArr,
					0, 0,
					pAvgTrainImg,
					projectedTestFace);

				// Check which person it is most likely to be.
				iNearest = findNearestNeighbor(projectedTestFace, &confidence);
				nearest  = trainPersonNumMat->data.i[iNearest];
				//printf("Most likely person in camera: '%s' (confidence=%f.\n", personNames[nearest-1].c_str(), confidence);
			}


			// Possibly save the processed face to the training set.
			if (saveNextFaces) 
			{
				// MAYBE GET IT TO ONLY TRAIN SOME IMAGES ?
				// Use a different filename each time.
				sprintf(cstr, "data/%d_%s%d.pgm", nPersons+1, newPersonName, newPersonFaces+1);
				printf("Storing the current face of '%s' into image '%s'.\n", newPersonName, cstr);
				cvSaveImage(cstr, equalizedImg, NULL);
				newPersonFaces++;
			}

			// Free the resources used for this frame.
			
			cvReleaseImage( &faceImg );
			cvReleaseImage( &sizedImg );
			cvReleaseImage( &equalizedImg );
		}

		cvReleaseImage( &greyImg );	
		
		//apply bg_wallpaper
		if(bckidx)
		{
			MaskMgr::GetInstance()->ApplyBackground(camImg, bg_wallpaper_mask, bckidx);
		}

		// get a random face when no face is detected
		if(faces->total != n_previous_faces)
		{
			maskidx_array[0] = (int) rand() % ( MaskMgr::GetInstance()->GetNumFaces() );
			maskidx_array[1] = (int) rand() % ( MaskMgr::GetInstance()->GetNumFaces() );
			maskidx_array[2] = (int) rand() % ( MaskMgr::GetInstance()->GetNumFaces() );
			maskidx_array[3] = (int) rand() % ( MaskMgr::GetInstance()->GetNumFaces() );
			maskidx_array[4] = (int) rand() % ( MaskMgr::GetInstance()->GetNumFaces() );
		}
		n_previous_faces = faces->total;

		//apply mask
		for(int i = 0; (i < min(faces->total, 5) && maskidx_array[i]); i++ )
		{
			/* extract the rectanlges only */
			CvRect faceRect = *(CvRect*)cvGetSeqElem( faces, i );
			
			faceRect.height *= pyrDownScale;
			faceRect.width *= pyrDownScale;
			faceRect.x *= pyrDownScale;
			faceRect.y *= pyrDownScale;

			//cvRectangle( camImg, cvPoint(faceRect.x, faceRect.y),
			//			 cvPoint((faceRect.x+faceRect.width),
			//					 (faceRect.y+faceRect.height)),
			//			 CV_RGB(255,0,0));

			MaskMgr::GetInstance()->ApplyMask(camImg, maskidx_array[i], faceRect);
		}

#ifdef _DEBUG
		cvShowImage("FG", bg_wallpaper_mask);
#endif

		//draw name and stuff like that
		if (faces->total > 0) 
		{	
			// Show the detected face region.
			//cvRectangle(camImg, cvPoint(faceRect.x, faceRect.y), cvPoint(faceRect.x + faceRect.width-1, faceRect.y + faceRect.height-1), CV_RGB(0,255,0), 1, 8, 0);

			if (nEigens > 0) 
			{	// Check if the face recognition database is loaded and a person was recognized.
				// Show the name of the recognized person, overlayed on the image below their face.
				sprintf_s(text1, sizeof(text1)-1, "%s", personNames[nearest-1].c_str());
#ifdef _DEBUG
				sprintf_s(text2, sizeof(text2)-1, "Confidence: %f", confidence);
				cvPutText(camImg, text2, cvPoint(faceRect.x, faceRect.y + faceRect.height + 30), &font, textColor);
#endif
				cvPutText(camImg, text1, cvPoint(faceRect.x, faceRect.y + faceRect.height + 15), &font, textColor);										
			}
		}
		else // update the background models so that they adjust to dayligh changes and such
		{
			if(bckg_sub_method==0)
			{
				// implemened onto doSegm() because it seams not to work so well here :\
				//cvUpdateBGStatModel( camImg, bg_model,  -10 );
			}
			else
			{
				cvCvtColor( camImg, codebookAuxImage, CV_BGR2HSV );
				cvBGCodeBookUpdate( codebook_model, codebookAuxImage );
				//cvReleaseImage(&codebookAuxImage);
			}
		}

		// display the final output image depending on if we're using a background mask or not
		cvShowImage("Output", camImg);				

		cvReleaseImage( &bg_img );
	}

	//cvReleaseImage(&camImg);
	CleanUp();

	return 0;
}

void CleanUp()
{
	printf("Cleaning up... ");
	// Free the Face Detector resources when the program is finished
	cvReleaseCapture(&camera);

	cvReleaseMemStorage( &multiFaceStorage );

	cvReleaseImage(&bg_wallpaper_mask);
	cvReleaseImage(&bg_wallpaper);	

	cvReleaseImage(&codebookAuxImage);
	cvReleaseBGStatModel(&bg_model);
	cvReleaseBGCodeBookModel(&codebook_model);

	cvReleaseMat(&personNumTruthMat);
	cvReleaseMat(&eigenValMat);
	cvReleaseMat(&projectedTrainFaceMat);
	cvReleaseMat(&trainPersonNumMat);

	cvReleaseHaarClassifierCascade( &faceCascade );
	cvDestroyAllWindows();
	

	printf("Done!\n");
	exit(0);
}

void OnKeyPressed(char key)
{		
		switch (key) 
		{
		case VK_ESCAPE:
				CleanUp();
			break;
			case 'n':	// Add a new person to the training set.
				// Train from the following images.
				printf("Enter your name: ");
				strcpy(newPersonName, "newPerson");
				gets(newPersonName);
				printf("Collecting all images until you hit 't', to start Training the images as '%s' ...\n", newPersonName);
				newPersonFaces = 0;	// restart training a new person
				saveNextFaces = TRUE;
			break;
			case 't':	// Start training
				saveNextFaces = FALSE;	// stop saving next faces.
				// Store the saved data into the training file.
				printf("Storing the training data for new person '%s'.\n", newPersonName);
				// Append the new person to the end of the training data.
				FILE *trainFile = fopen("train.txt", "a");
				for (int i=0; i<newPersonFaces; i++) {
					sprintf(cstr, "data/%d_%s%d.pgm", nPersons+1, newPersonName, i+1);
					fprintf(trainFile, "%d %s %s\n", nPersons+1, newPersonName, cstr);
				}
				fclose(trainFile);

				// Now there is one more person in the database, ready for retraining.
				//nPersons++;

				//break;
				//case 'r':

				// Re-initialize the local data.
				projectedTestFace = 0;
				saveNextFaces = FALSE;
				newPersonFaces = 0;

				// Retrain from the new database without shutting down.
				// Depending on the number of images in the training set and number of people, it might take 30 seconds or so.
				cvFree( &trainPersonNumMat );	// Free the previous data before getting new data
				trainPersonNumMat = retrainOnline();
				// Project the test images onto the PCA subspace
				cvFree(&projectedTestFace);	// Free the previous data before getting new data
				projectedTestFace = (float *)cvAlloc( nEigens*sizeof(float) );
			
				printf("Recognizing person in the camera ...\n");
				//continue;	// Begin with the next frame.
				break;
		}
	}

// Train from the data in the given text file, and store the trained data into the file 'facedata.xml'.
void learn(char *szFileTrain)
{
	int i, offset;

	// load training data
	printf("Loading the training images in '%s'\n", szFileTrain);
	nTrainFaces = loadFaceImgArray(szFileTrain);
	printf("Got %d training images.\n", nTrainFaces);
	if( nTrainFaces < 2 )
	{
		fprintf(stderr,
			"Need 2 or more training faces\n"
			"Input file contains only %d\n", nTrainFaces);
		return;
	}

	// do PCA on the training faces
	doPCA();

	// project the training images onto the PCA subspace
	projectedTrainFaceMat = cvCreateMat( nTrainFaces, nEigens, CV_32FC1 );
	offset = projectedTrainFaceMat->step / sizeof(float);
	for(i=0; i<nTrainFaces; i++)
	{
		//int offset = i * nEigens;
		cvEigenDecomposite(
			faceImgArr[i],
			nEigens,
			eigenVectArr,
			0, 0,
			pAvgTrainImg,
			//projectedTrainFaceMat->data.fl + i*nEigens);
			projectedTrainFaceMat->data.fl + i*offset);
	}

	// store the recognition data as an xml file
	storeTrainingData();

	// Save all the eigenvectors as images, so that they can be checked.
	if (SAVE_EIGENFACE_IMAGES) {
		storeEigenfaceImages();
	}

}

// Re-train the new face rec database without shutting down.
// Depending on the number of images in the training set and number of people, it might take 30 seconds or so.
CvMat* retrainOnline(void)
{
	CvMat *trainPersonNumMat;
	int i;

	// Free & Re-initialize the global variables.
	if (faceImgArr) {
		for (i=0; i<nTrainFaces; i++) {
			if (faceImgArr[i])
				cvReleaseImage( &faceImgArr[i] );
		}
	}
	cvFree( &faceImgArr ); // array of face images
	cvFree( &personNumTruthMat ); // array of person numbers
	personNames.clear();			// array of person names (indexed by the person number). Added by Shervin.
	nPersons = 0; // the number of people in the training set. Added by Shervin.
	nTrainFaces = 0; // the number of training images
	nEigens = 0; // the number of eigenvalues
	cvReleaseImage( &pAvgTrainImg ); // the average image
	for (i=0; i<nTrainFaces; i++) {
		if (eigenVectArr[i])
			cvReleaseImage( &eigenVectArr[i] );
	}
	cvFree( &eigenVectArr ); // eigenvectors
	cvFree( &eigenValMat ); // eigenvalues
	cvFree( &projectedTrainFaceMat ); // projected training faces

	// Retrain from the data in the files
	printf("Retraining with the new person ...\n");
	learn("train.txt");
	printf("Done retraining.\n");

	// Load the previously saved training data
	if( !loadTrainingData( &trainPersonNumMat ) ) {
		printf("ERROR in recognizeFromCam(): Couldn't load the training data!\n");
		exit(1);
	}

	return trainPersonNumMat;
}

// Find the most likely person based on a detection. Returns the index, and stores the confidence value into pConfidence.
int findNearestNeighbor(float * projectedTestFace, float *pConfidence)
{
	//double leastDistSq = 1e12;
	double leastDistSq = DBL_MAX;
	int i, iTrain, iNearest = 0;

	for(iTrain=0; iTrain<nTrainFaces; iTrain++)
	{
		double distSq=0;

		for(i=0; i<nEigens; i++)
		{
			float d_i = projectedTestFace[i] - projectedTrainFaceMat->data.fl[iTrain*nEigens + i];
#ifdef USE_MAHALANOBIS_DISTANCE
			distSq += d_i*d_i / eigenValMat->data.fl[i];  // Mahalanobis distance (might give better results than Eucalidean distance)
#else
			distSq += d_i*d_i; // Euclidean distance.
#endif
		}

		if(distSq < leastDistSq)
		{
			leastDistSq = distSq;
			iNearest = iTrain;
		}
	}

	// Return the confidence level based on the Euclidean distance,
	// so that similar images should give a confidence between 0.5 to 1.0,
	// and very different images should give a confidence between 0.0 to 0.5.
	*pConfidence = 1.0f - sqrt( leastDistSq / (float)(nTrainFaces * nEigens) ) / 255.0f;

	// Return the found index.
	return iNearest;
}

// Grab the next camera frame. Waits until the next frame is ready,
// and provides direct access to it, so do NOT modify the returned image or free it!
// Will automatically initialize the camera on the first frame.
IplImage* getCameraFrame(void)
{
	IplImage *frame;

	// If the camera hasn't been initialized, then open it.
	if (!camera) {
		printf("Acessing the camera ...\n");
		camera = cvCaptureFromCAM( 0 );
		if (!camera) {
			printf("ERROR in getCameraFrame(): Couldn't access the camera.\n");
			exit(1);
		}

		// Try to set the camera resolution
#ifdef _DEBUG
		cvSetCaptureProperty( camera, CV_CAP_PROP_FRAME_WIDTH, 640 );
		cvSetCaptureProperty( camera, CV_CAP_PROP_FRAME_HEIGHT, 480 );
#endif

		// Wait a little, so that the camera can auto-adjust itself
		Sleep(1000);	// (in milliseconds)
		
		// attempt to disable auto-exposure, white balance, auto-focus, or similar adaptive settings
		cvSetCaptureProperty( camera, CV_CAP_PROP_RECTIFICATION, 0);
		cvSetCaptureProperty( camera, CV_CAP_PROP_EXPOSURE, 0);
		cvSetCaptureProperty( camera, CV_CAP_PROP_WHITE_BALANCE, 0);

		frame = cvQueryFrame( camera );	// get the first frame, to make sure the camera is initialized.
		if (frame) {
			printf("Got a camera using a resolution of %dx%d.\n", (int)cvGetCaptureProperty( camera, CV_CAP_PROP_FRAME_WIDTH), (int)cvGetCaptureProperty( camera, CV_CAP_PROP_FRAME_HEIGHT) );
		//background binary mask
		bg_wallpaper_mask = cvCreateImage(cvSize(frame->width, frame->height), frame->depth, 1);
		}
		else
		{
			printf("Failed to aquire camera! Terminating....\n");
			Sleep(1000);
			exit(0);
		}
		
	}
	
	frame = cvQueryFrame( camera );

	return frame;
}

// Do the Principal Component Analysis, finding the average image
// and the eigenfaces that represent any image in the given dataset.
void doPCA()
{
	int i;
	CvTermCriteria calcLimit;

	CvSize faceImgSize;

	// set the number of eigenvalues to use
	nEigens = nTrainFaces-1;

	// allocate the eigenvector images
	faceImgSize.width  = faceImgArr[0]->width;
	faceImgSize.height = faceImgArr[0]->height;
	eigenVectArr = (IplImage**)cvAlloc(sizeof(IplImage*) * nEigens);
	for(i=0; i<nEigens; i++)
		eigenVectArr[i] = cvCreateImage(faceImgSize, IPL_DEPTH_32F, 1);

	// allocate the eigenvalue array
	eigenValMat = cvCreateMat( 1, nEigens, CV_32FC1 );

	// allocate the averaged image
	pAvgTrainImg = cvCreateImage(faceImgSize, IPL_DEPTH_32F, 1);

	// set the PCA termination criterion
	calcLimit = cvTermCriteria( CV_TERMCRIT_ITER, nEigens, 1);

	// compute average image, eigenvalues, and eigenvectors
	cvCalcEigenObjects(
		nTrainFaces,
		(void*)faceImgArr,
		(void*)eigenVectArr,
		CV_EIGOBJ_NO_CALLBACK,
		0,
		0,
		&calcLimit,
		pAvgTrainImg,
		eigenValMat->data.fl);

	cvNormalize(eigenValMat, eigenValMat, 1, 0, CV_L1, 0);
}

// Read the names & image filenames of people from a text file, and load all those images listed.
int loadFaceImgArray(char * filename)
{
	FILE * imgListFile = 0;
	char imgFilename[512];
	int iFace, nFaces=0;
	int i;

	// open the input file
	if( !(imgListFile = fopen(filename, "r")) )
	{
		fprintf(stderr, "Can\'t open file %s\n", filename);
		return 0;
	}

	// count the number of faces
	while( fgets(imgFilename, 512, imgListFile) ) ++nFaces;
	rewind(imgListFile);

	// allocate the face-image array and person number matrix
	faceImgArr        = (IplImage **)cvAlloc( nFaces*sizeof(IplImage *) );
	personNumTruthMat = cvCreateMat( 1, nFaces, CV_32SC1 );

	personNames.clear();	// Make sure it starts as empty.
	nPersons = 0;

	// store the face images in an array
	for(iFace=0; iFace<nFaces; iFace++)
	{
		char personName[256];
		string sPersonName;
		int personNumber;

		// read person number (beginning with 1), their name and the image filename.
		fscanf(imgListFile, "%d %s %s", &personNumber, personName, imgFilename);
		sPersonName = personName;
		//printf("Got %d: %d, <%s>, <%s>.\n", iFace, personNumber, personName, imgFilename);

		// Check if a new person is being loaded.
		if (personNumber > nPersons) {
			// Allocate memory for the extra person (or possibly multiple), using this new person's name.
			for (i=nPersons; i < personNumber; i++) {
				personNames.push_back( sPersonName );
			}
			nPersons = personNumber;
			//printf("Got new person <%s> -> nPersons = %d [%d]\n", sPersonName.c_str(), nPersons, personNames.size());
		}

		// Keep the data
		personNumTruthMat->data.i[iFace] = personNumber;

		// load the face image
		faceImgArr[iFace] = cvLoadImage(imgFilename, CV_LOAD_IMAGE_GRAYSCALE);

		if( !faceImgArr[iFace] )
		{
			fprintf(stderr, "Can\'t load image from %s\n", imgFilename);
			return 0;
		}
	}

	fclose(imgListFile);

	printf("Data loaded from '%s': (%d images of %d people).\n", filename, nFaces, nPersons);
	printf("People: ");
	if (nPersons > 0)
		printf("<%s>", personNames[0].c_str());
	for (i=1; i<nPersons; i++) {
		printf(", <%s>", personNames[i].c_str());
	}
	printf(".\n");

	return nFaces;
}

// Perform face detection on the input image, using the given Haar Cascade.
// Returns a rectangle for the detected region in the given image.
CvSeq* detectFaceInImage(IplImage *imageProcessed, CvHaarClassifierCascade* cascade)
{
	// Smallest face size.
	const CvSize minFeatureSize = cvSize(20, 20);
	// Only search for 1 face.
	const int flags = CV_HAAR_FIND_BIGGEST_OBJECT | CV_HAAR_DO_ROUGH_SEARCH | CV_HAAR_DO_CANNY_PRUNING;

	float search_scale_factor =  1.1f + (static_cast<float>(iSearch_scale_factor) / 10) ; //1.5f;

	//CvRect rc;
	double t;
	//CvSeq* rects;
	unsigned int ms = 0, nFaces = 0;

	cvClearMemStorage( storage );

	// Detect all the faces in the greyscale image.
	t = (double)cvGetTickCount();
	rects = cvHaarDetectObjects( imageProcessed, cascade, storage,
				search_scale_factor, 2, flags, minFeatureSize);
	t = (double)cvGetTickCount() - t;
	ms = cvRound( t / ((double)cvGetTickFrequency() * 1000.0) );
	nFaces = rects->total;
	printf("Face Detection took %d ms and found %d objects\n", ms, nFaces);

	// Get the first detected face (the biggest).
	/*
	if (nFaces > 0)
		rc = *(CvRect*)cvGetSeqElem( rects, 0 );
	else
		rc = cvRect(-1,-1,-1,-1);	// Couldn't find the face.

	cvReleaseMemStorage( &storage );
	*/

	return rects;	// Return the biggest face found, or (-1,-1,-1,-1).
}

// Perform face detection on the input image, using the given Haar Cascade.
// Returns a sequence of rectangles with the probable face locations.
CvSeq* detectMultiFaceInImage(IplImage *imageProcessed, CvHaarClassifierCascade* cascade)
{
	const CvSize minFeatureSize = cvSize(15/pyrDownScale, 15/pyrDownScale);
	const int flags = CV_HAAR_DO_ROUGH_SEARCH | CV_HAAR_DO_CANNY_PRUNING;
	float search_scale_factor =  1.1f + (static_cast<float>(iSearch_scale_factor) / 10) ;

	double t;
	unsigned int ms = 0;

	cvClearMemStorage( multiFaceStorage );

	// down-scale the image to get performance boost
	IplImage * small_img = cvCreateImage( 
		cvSize(imageProcessed->width/pyrDownScale,imageProcessed->height/pyrDownScale),
		IPL_DEPTH_8U, imageProcessed->nChannels 
		);
	cvPyrDown(imageProcessed, small_img, CV_GAUSSIAN_5x5);
	
	// Detect all the faces in the greyscale image.
	t = (double)cvGetTickCount();
	rects = cvHaarDetectObjects( small_img, cascade, multiFaceStorage,
				search_scale_factor, 2, flags, minFeatureSize);
	t = (double)cvGetTickCount() - t;
	ms = cvRound( t / ((double)cvGetTickFrequency() * 1000.0) );
	printf("Face Detection took %d ms and found %d objects\n", ms, rects->total);

	//cvReleaseMemStorage( &storage );
	cvReleaseImage(&small_img);

	return rects;
}

// Save all the eigenvectors as images, so that they can be checked.
void storeEigenfaceImages()
{
	// Store the average image to a file
	printf("Saving the image of the average face as 'out_averageImage.bmp'.\n");
	cvSaveImage("out_averageImage.bmp", pAvgTrainImg);
	// Create a large image made of many eigenface images.
	// Must also convert each eigenface image to a normal 8-bit UCHAR image instead of a 32-bit float image.
	printf("Saving the %d eigenvector images as 'out_eigenfaces.bmp'\n", nEigens);
	if (nEigens > 0) {
		// Put all the eigenfaces next to each other.
		int COLUMNS = 8;	// Put upto 8 images on a row.
		int nCols = min(nEigens, COLUMNS);
		int nRows = 1 + (nEigens / COLUMNS);	// Put the rest on new rows.
		int w = eigenVectArr[0]->width;
		int h = eigenVectArr[0]->height;
		CvSize size;
		size = cvSize(nCols * w, nRows * h);
		IplImage *bigImg = cvCreateImage(size, IPL_DEPTH_8U, 1);	// 8-bit Greyscale UCHAR image
			
		for (int i=0; i<nEigens; i++) {
			// Get the eigenface image.
			IplImage *byteImg = convertFloatImageToUcharImage(eigenVectArr[i]);
			// Paste it into the correct position.
			int x = w * (i % COLUMNS);
			int y = h * (i / COLUMNS);
			CvRect ROI = cvRect(x, y, w, h);
			cvSetImageROI(bigImg, ROI);
			cvCopyImage(byteImg, bigImg);
			cvResetImageROI(bigImg);
			cvReleaseImage(&byteImg);
		}
		cvSaveImage("out_eigenfaces.bmp", bigImg);
		cvReleaseImage(&bigImg);
	}
}

// Open the training data from the file 'facedata.xml'.
int loadTrainingData(CvMat ** pTrainPersonNumMat)
{
	CvFileStorage * fileStorage;
	int i;

	// create a file-storage interface
	fileStorage = cvOpenFileStorage( "facedata.xml", 0, CV_STORAGE_READ );
	if( !fileStorage ) {
		printf("Can't open training database file 'facedata.xml'.\n");
		return 0;
	}

	// Load the person names. Added by Shervin.
	personNames.clear();	// Make sure it starts as empty.
	nPersons = cvReadIntByName( fileStorage, 0, "nPersons", 0 );
	if (nPersons == 0) {
		printf("No people found in the training database 'facedata.xml'.\n");
		return 0;
	}
	// Load each person's name.
	for (i=0; i<nPersons; i++) {
		string sPersonName;
		char varname[200];
		sprintf( varname, "personName_%d", (i+1) );
		sPersonName = cvReadStringByName(fileStorage, 0, varname );
		personNames.push_back( sPersonName );
	}

	// Load the data
	nEigens = cvReadIntByName(fileStorage, 0, "nEigens", 0);
	nTrainFaces = cvReadIntByName(fileStorage, 0, "nTrainFaces", 0);
	*pTrainPersonNumMat = (CvMat *)cvReadByName(fileStorage, 0, "trainPersonNumMat", 0);
	eigenValMat  = (CvMat *)cvReadByName(fileStorage, 0, "eigenValMat", 0);
	projectedTrainFaceMat = (CvMat *)cvReadByName(fileStorage, 0, "projectedTrainFaceMat", 0);
	pAvgTrainImg = (IplImage *)cvReadByName(fileStorage, 0, "avgTrainImg", 0);
	eigenVectArr = (IplImage **)cvAlloc(nTrainFaces*sizeof(IplImage *));
	for(i=0; i<nEigens; i++)
	{
		char varname[200];
		sprintf( varname, "eigenVect_%d", i );
		eigenVectArr[i] = (IplImage *)cvReadByName(fileStorage, 0, varname, 0);
	}

	// release the file-storage interface
	cvReleaseFileStorage( &fileStorage );

	printf("Training data loaded (%d training images of %d people):\n", nTrainFaces, nPersons);
	printf("People: ");
	if (nPersons > 0)
		printf("<%s>", personNames[0].c_str());
	for (i=1; i<nPersons; i++) {
		printf(", <%s>", personNames[i].c_str());
	}
	printf(".\n");

	return 1;
}

// Save the training data to the file 'facedata.xml'.
void storeTrainingData()
{
	CvFileStorage * fileStorage;
	int i;

	// create a file-storage interface
	fileStorage = cvOpenFileStorage( "facedata.xml", 0, CV_STORAGE_WRITE );

	// Store the person names. Added by Shervin.
	cvWriteInt( fileStorage, "nPersons", nPersons );
	for (i=0; i<nPersons; i++) {
		char varname[200];
		sprintf( varname, "personName_%d", (i+1) );
		cvWriteString(fileStorage, varname, personNames[i].c_str(), 0);
	}

	// store all the data
	cvWriteInt( fileStorage, "nEigens", nEigens );
	cvWriteInt( fileStorage, "nTrainFaces", nTrainFaces );
	cvWrite(fileStorage, "trainPersonNumMat", personNumTruthMat, cvAttrList(0,0));
	cvWrite(fileStorage, "eigenValMat", eigenValMat, cvAttrList(0,0));
	cvWrite(fileStorage, "projectedTrainFaceMat", projectedTrainFaceMat, cvAttrList(0,0));
	cvWrite(fileStorage, "avgTrainImg", pAvgTrainImg, cvAttrList(0,0));
	for(i=0; i<nEigens; i++)
	{
		char varname[200];
		sprintf( varname, "eigenVect_%d", i );
		cvWrite(fileStorage, varname, eigenVectArr[i], cvAttrList(0,0));
	}

	// release the file-storage interface
	cvReleaseFileStorage( &fileStorage );
}


void doSegm(IplImage * imgCam)
{
	if(!bg_model)
	{
		//create BG model
		bg_model = cvCreateGaussianBGModel( imgCam );
		//bg_model = cvCreateFGDStatModel( imgCam );
	}

	// if we have a face dont update bg_model
	if(faces && faces->total)
	{
		frames_without_faces=0;
		cvUpdateBGStatModel( imgCam, bg_model,  0 );
	}
	else // update the background models so that they adjust to dayligh changes and such
	{
		frames_without_faces++;
		if(frames_without_faces > 100)
			cvUpdateBGStatModel( imgCam, bg_model,  -1 );
		else
			cvUpdateBGStatModel( imgCam, bg_model,  0 );
	}

	cvCopy(bg_model->foreground, bg_wallpaper_mask);

	// morphological closing to clean small disturbances and emphatize our foreground

	cvDilate(bg_wallpaper_mask, bg_wallpaper_mask, NULL, 4);
	cvErode(bg_wallpaper_mask, bg_wallpaper_mask, NULL, 4);

	//cvShowImage("BG", bg_model->background);
	//cvShowImage("FG", bg_model->foreground);

}


void doCodeBook(IplImage * imgCam)
{
	//1st frame
	if (!codebook_model)
	{
		codebook_model = cvCreateBGCodeBookModel();

		//Set color thresholds
		codebook_model->modMin[0] = 1;
		codebook_model->modMin[1] = codebook_model->modMin[2] = 1;
		codebook_model->modMax[0] = 10;
		codebook_model->modMax[1] = codebook_model->modMax[2] = 10;
		codebook_model->cbBounds[0] = codebook_model->cbBounds[1] = codebook_model->cbBounds[2] = 10;

		codebookAuxImage = cvCloneImage(imgCam);
		cvSet(bg_wallpaper_mask,cvScalar(255));

	}

	cvCvtColor( imgCam, codebookAuxImage, CV_BGR2HSV );//YUV For codebook method

	//This is where we build our background model
	if( nframes-1 < nframesToLearnBG  )
	{
		cvBGCodeBookUpdate( codebook_model, codebookAuxImage );
		printf("Training the model, please hold %d frames\n", nframesToLearnBG-nframes);
		++nframes;
	}

	if( nframes-1 == nframesToLearnBG  )
	{
		cvBGCodeBookClearStale( codebook_model, codebook_model->t/2 );
		++nframes;
	}
	
	//Find the foreground if any
	if( nframes-1 >= nframesToLearnBG  )
	{
		// Find foreground by codebook method
		cvBGCodeBookDiff( codebook_model, codebookAuxImage, bg_wallpaper_mask );
		cvSegmentFGMask( bg_wallpaper_mask );
	}

}

