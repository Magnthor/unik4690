#include "opencv2/highgui.hpp"
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv/cv.hpp>
#include "opencv2/xfeatures2d.hpp"

using namespace cv;
using namespace std;

std::vector<cv::KeyPoint> MakePoint(cv::Mat frame);
cv::Ptr<cv::Feature2D> detector = cv::xfeatures2d::SURF::create();
cv::Ptr<cv::Feature2D> BruteForceDetector = cv::xfeatures2d::SURF::create();
cv::BFMatcher BruteForceMatcher{BruteForceDetector->defaultNorm()};

int maxFeatures = 100;


int main() {
    cv::VideoCapture input_stream(0);


    if (!input_stream.isOpened()) {
        std::cerr << "could not open camera\n";
        return EXIT_FAILURE;

    }

    cv::Mat frame;
    cv::Mat image;
    input_stream >> frame;
    frame.copyTo(image);

    cv::Mat ReferenceImage;
    std::vector<cv::KeyPoint> ReferencePoints;
    cv::Mat RefrenceFeatures;



    while (true) {
        input_stream >> frame;
        frame.copyTo(image);

        //cv::TermCriteria CriteriaTerm(TermCriteria::COUNT|TermCriteria::EPS, 20, 0.03);
        cv::Mat GrayImage;
        cv::cvtColor(frame, GrayImage, CV_BGR2GRAY);
        std::vector<cv::KeyPoint> Points = MakePoint(GrayImage);

        //if(PrevPoints.empty() == true){
            //To fix the problem with the first iteration
            //std::vector<cv::KeyPoint> TempVector(Points);
            //PrevPoints.swap(TempVector);
            //   GrayImage.copyTo(PrevImage);
        //}
        //std::vector<uchar> StatusVector;
        //std::vector<float> ErrorVector;

        //cv::calcOpticalFlowPyrLK(PrevImage, GrayImage, PrevPoints, Points, StatusVector, ErrorVector);


        cv::Mat OutImage;
        if(!ReferencePoints.empty()){
            Points = ReferencePoints;
        }
        cv::Mat CurrentFeatures;
        std::vector<std::vector<cv::DMatch>> matches;
        BruteForceDetector->compute(image, Points, CurrentFeatures);
        BruteForceMatcher.knnMatch(CurrentFeatures, RefrenceFeatures, matches, 2);





        for(unsigned int i = 0; i < Points.size(); i++){
            cv::drawKeypoints(image, Points, OutImage, cv::Scalar(0,255,0));
        }

        cv::imshow("TheAwsomestAugmentedRealityGame", OutImage);


        //std::vector<cv::KeyPoint> TempVector(Points);
        //GrayImage.copyTo(PrevImage);
        //PrevPoints.swap(TempVector);



        // Trigger detection and saving
        int key = cv::waitKey(30);
        if (key == ' '){
            ReferencePoints = Points;
            ReferenceImage = image;
            RefrenceFeatures = CurrentFeatures;
        }
        else if ( key == 'r'){
            ReferenceImage = cv::Mat();
            ReferencePoints.clear();
            RefrenceFeatures = cv::Mat();
        }
        else if (key >= 0){
            break;
        }

        if (frame.empty()) {
            break;
        }


    }


    return EXIT_SUCCESS;
}

std::vector<cv::KeyPoint> MakePoint(cv::Mat frame){
//Function to set a SetPoint-frame so we can track from that point.


    //double QualityLevel = 0.01;
    //double MinDistance = 10;
    //cv::Size WinSize(5,5);


    std::vector<cv::KeyPoint> Keypoints;

    detector->detect(frame, Keypoints);
    cv::KeyPointsFilter::retainBest(Keypoints, maxFeatures);
    //cv::goodFeaturesToTrack(frame, OutputImage, maxCorners, QualityLevel, MinDistance, Mat(), 3, 3, 0, 0.04);
    //cv::cornerSubPix(frame, OutputImage, WinSize, Size(-1,-1), CriteriaTerm);


    return Keypoints;
}