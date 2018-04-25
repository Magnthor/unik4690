#include "opencv2/highgui.hpp"
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv/cv.hpp>
#include "opencv2/xfeatures2d.hpp"

using namespace cv;
using namespace std;

std::vector<cv::KeyPoint> MakePoint(cv::Mat frame, cv::Ptr<cv::Feature2D> detector, int maxFeatures);
std::vector<cv::DMatch> FindBestMatches(std::vector<std::vector<cv::DMatch>> matches, float ratio);
void FindPointsToDraw(std::vector<cv::DMatch> BestMatches, std::vector<cv::KeyPoint> Points, cv::Mat image, int CoinSetPoint);



int maxFeatures = 100;


int main() {
    cv::VideoCapture input_stream(0);
    cv::Ptr<cv::Feature2D> detector = cv::xfeatures2d::SURF::create();
    cv::Ptr<cv::Feature2D> BruteForceDetector = cv::xfeatures2d::SURF::create();
    cv::BFMatcher BruteForceMatcher{BruteForceDetector->defaultNorm()};



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
    cv::Mat ReferenceFeatures;
    int CoinPoint;



    while (true) {
        input_stream >> frame;
        frame.copyTo(image);

        //cv::TermCriteria CriteriaTerm(TermCriteria::COUNT|TermCriteria::EPS, 20, 0.03);
        cv::Mat GrayImage;
        cv::cvtColor(frame, GrayImage, CV_BGR2GRAY);
        std::vector<cv::KeyPoint> Points = MakePoint(GrayImage, detector, maxFeatures);

        //if(PrevPoints.empty() == true){
            //To fix the problem with the first iteration
            //std::vector<cv::KeyPoint> TempVector(Points);
            //PrevPoints.swap(TempVector);
            //   GrayImage.copyTo(PrevImage);
        //}
        //std::vector<uchar> StatusVector;
        //std::vector<float> ErrorVector;

        //cv::calcOpticalFlowPyrLK(PrevImage, GrayImage, PrevPoints, Points, StatusVector, ErrorVector);



        cv::Mat CurrentFeatures;
        BruteForceDetector->compute(GrayImage, Points, CurrentFeatures);

        if(ReferenceFeatures.empty()){
            cv::Mat OutImage;
            cv::drawKeypoints(image, Points, OutImage, cv::Scalar(0,255,0));
            cv::imshow("TheAwsomestAugmentedRealityGame", OutImage);
        }
        else{
            std::vector<std::vector<cv::DMatch>> matches;
            BruteForceMatcher.knnMatch(CurrentFeatures, ReferenceFeatures, matches, 2);
            float ratio = 0.7;
            std::vector<cv::DMatch> BestMatches = FindBestMatches(matches, ratio);
            FindPointsToDraw(BestMatches, Points, image, CoinPoint);
        }



        //std::vector<cv::KeyPoint> TempVector(Points);
        //GrayImage.copyTo(PrevImage);
        //PrevPoints.swap(TempVector);



        // Trigger detection and saving
        int key = cv::waitKey(30);
        if (key == ' '){
            ReferencePoints = MakePoint(GrayImage, detector, 25);;
            ReferenceImage = image;
            BruteForceDetector->compute(GrayImage, ReferencePoints, ReferenceFeatures);
            CoinPoint = 2;
        }
        else if ( key == 'r'){
            ReferenceImage = cv::Mat();
            ReferencePoints.clear();
            ReferenceFeatures = cv::Mat();
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

std::vector<cv::KeyPoint> MakePoint(cv::Mat frame, cv::Ptr<cv::Feature2D> detector, int maxFeatures){
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

std::vector<cv::DMatch> FindBestMatches(std::vector<std::vector<cv::DMatch>> matches, float ratio){
    std::vector<cv::DMatch> BestMatches;

    for(size_t i = 0;i< matches.size() ;i++){
        if(matches[i][0].distance < (matches[i][1].distance * ratio)){
            BestMatches.push_back(matches[i][0]);

        }


    }
    return BestMatches;

}

void FindPointsToDraw(std::vector<cv::DMatch> BestMatches, std::vector<cv::KeyPoint> Points, cv::Mat image, int CoinSetPoint){

    std::vector<cv::KeyPoint> KeyPointsToDraw;
    std::vector<cv::KeyPoint> CoinPoint;
    cv::Mat OutImage;
    cv::Mat OutImage2;

    for(size_t i = 0; i< BestMatches.size();i++){
        KeyPointsToDraw.push_back(Points[BestMatches[i].trainIdx]);
        if(BestMatches[i].imgIdx == CoinSetPoint){
            CoinPoint[0] = Points[BestMatches[i].trainIdx];
        }
    }
    for(unsigned int i = 0; i < Points.size(); i++){
        cv::drawKeypoints(image, KeyPointsToDraw, OutImage, cv::Scalar(0,255,0));
    }
    if(!CoinPoint.empty()){
        cv::drawKeypoints(OutImage, CoinPoint, OutImage2, cv::Scalar(0,0,255));

        cv::imshow("TheAwsomestAugmentedRealityGame", OutImage2);
    }
    else{

        cv::imshow("TheAwsomestAugmentedRealityGame", OutImage);
    }





}