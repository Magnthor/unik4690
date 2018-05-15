#include "opencv2/highgui.hpp"
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv/cv.hpp>
#include "opencv2/xfeatures2d.hpp"
#include <chrono>
#include <cstdlib>


//TODO LIST:
// Make sure it doesent crash if it had many features, but loses them all
//
//
//
//
//

using namespace cv;
using namespace std;

std::vector<cv::KeyPoint> MakePoint(cv::Mat frame, cv::Ptr<cv::Feature2D> detector, int maxFeatures);
std::vector<cv::DMatch> FindBestMatches(std::vector<std::vector<cv::DMatch>> matches, std::vector<std::vector<cv::DMatch>> matches2, std::vector<cv::KeyPoint> ReferencePoints, float ratio);
void FindPointsToDraw(std::vector<cv::DMatch> BestMatches, std::vector<cv::KeyPoint> Points, cv::Mat image, int CoinSetPoint);
void drawCoin(cv::Mat image, std::vector<cv::KeyPoint> CoinPoint);
void DrawGraphics(cv::Mat OutImage);
bool checkForScore(cv::Mat image, std::vector<cv::KeyPoint> CoinPoint);
bool timingTracker(long duration);


int maxFeatures = 40;
cv::Mat CoinPicture = cv::imread("../coin.jpeg", cv::IMREAD_UNCHANGED);//original pic of coin
cv::Mat CoinPictureScaled;//scaled pic of coin
float scaleFactor = 0.20;
int score = 0;
int flag = 0;

using timer = std::chrono::high_resolution_clock;




int main() {
    cv::VideoCapture input_stream(1);
    cv::Ptr<cv::Feature2D> detector = cv::xfeatures2d::SURF::create();
    cv::Ptr<cv::Feature2D> BruteForceDetector = cv::xfeatures2d::SURF::create();
    cv::BFMatcher BruteForceMatcher{BruteForceDetector->defaultNorm()};

    cv::Mat ReferenceImage;
    std::vector<cv::KeyPoint> ReferencePoints;
    cv::Mat ReferenceFeatures;
    int CoinPoint;

    auto lastEvent = timer::now();


    cv::resize(CoinPicture, CoinPictureScaled, Size(), scaleFactor, scaleFactor, cv::INTER_LANCZOS4);



    if (!input_stream.isOpened()) {
        std::cerr << "could not open camera\n";
        return EXIT_FAILURE;
    }

    cv::Mat frame;
    cv::Mat image;
    input_stream >> frame;
    frame.copyTo(image);


    cv::Mat vis_img;


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
            //std::vector< cv::Point2f > 	point_ind,
            //cv::KeyPoint::convert(Points, point_ind);
            DrawGraphics(OutImage);

        }
        else{
            std::vector<std::vector<cv::DMatch>> matches;
            std::vector<std::vector<cv::DMatch>> matches2;

            int numberOfN = 2;
            // current and reference features have to be the same size!!
            BruteForceMatcher.knnMatch(ReferenceFeatures, CurrentFeatures, matches, numberOfN);
            BruteForceMatcher.knnMatch(CurrentFeatures, ReferenceFeatures, matches2, numberOfN);
            float ratio = 0.7;

            // show the current matches between ref.img and current image frame
            std::vector<cv::DMatch> BestMatches = FindBestMatches(matches, matches2, ReferencePoints, ratio);
            FindPointsToDraw(BestMatches, Points, image, CoinPoint);
            cv::drawMatches(ReferenceImage, ReferencePoints, image, Points, BestMatches, vis_img);
            //cv::imshow("DrawMatchesOutput", vis_img);

            /*
            std::vector<cv::KeyPoint> matching_pts1;
            std::vector<cv::KeyPoint> matching_pts2;
            extractMatchingPoints(Points, ReferencePoints, BestMatches, matching_pts1, matching_pts2);
            */

            //trying to show only the match with index 1 in the reference image

            /*
            if (BestMatches.size() > 1) {
                cv::Point2f pointA = {ReferencePoints[BestMatches[1].queryIdx].pt.x,
                                      ReferencePoints[BestMatches[1].queryIdx].pt.y};
                cv::Mat imgWithPointA;
                imgWithPointA = ReferenceImage.clone();
                cv::circle(imgWithPointA, pointA, 10, cv::Scalar(0, 0, 255));
                cv::imshow("query_idx", imgWithPointA);

                //"coin point"
                cv::Point2f pointB = {Points[BestMatches[1].trainIdx].pt.x,
                                      Points[BestMatches[1].trainIdx].pt.y};
                cv::Mat imgWithPointB;
                imgWithPointB = image.clone();
                cv::circle(imgWithPointB, pointB, 10, cv::Scalar(255, 0, 0));
                cv::imshow("train_idx", imgWithPointB);
                //"coin point"

            }*/

        }



        //std::vector<cv::KeyPoint> TempVector(Points);
        //GrayImage.copyTo(PrevImage);
        //PrevPoints.swap(TempVector);



        // Trigger detection and saving when space is pressed
        int key = cv::waitKey(30);

        auto timing = std::chrono::duration_cast<std::chrono::seconds>(timer::now() - lastEvent);
        bool event = timingTracker(timing.count());

        if (key == ' ' || event){
            lastEvent = timer::now();
            ReferencePoints = MakePoint(GrayImage, detector, 20);;
            ReferenceImage = image.clone();
            BruteForceDetector->compute(GrayImage, ReferencePoints, ReferenceFeatures);
            CoinPoint = 1;
        }
        else if ( key == 'r' || flag == 1){
            lastEvent = timer::now();
            ReferenceImage = cv::Mat();
            ReferencePoints.clear();
            ReferenceFeatures = cv::Mat();
            flag = 0;
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

std::vector<cv::DMatch> FindBestMatches(std::vector<std::vector<cv::DMatch>> matches, std::vector<std::vector<cv::DMatch>> matches2, std::vector<cv::KeyPoint> ReferencePoints, float ratio){
    std::vector<cv::DMatch> BestMatches;
    std::vector<cv::DMatch> BestMatches2;
    std::vector<cv::DMatch> BestMatchesBoth;
    //finding the better match from knn matcher

    for(size_t i = 0;i< matches.size() ;i++) {
        //ReferenceFeatures, CurrentFeatures
        if (matches[i][0].distance < (matches[i][1].distance * ratio)) {
            BestMatches.push_back(matches[i][0]);
        }
        //CurrentFeatures, ReferenceFeatures
        if (matches2[i][0].distance < (matches2[i][1].distance * ratio)) {
            BestMatches2.push_back(matches2[i][0]);
        }
    }
        //new vector with only the points we have duplicates of.
    for(size_t i = 0;i< BestMatches.size() ;i++) {
        for (size_t j = 0; j < BestMatches2.size(); j++) {
            if ((ReferencePoints[BestMatches[i].queryIdx].pt.x == ReferencePoints[BestMatches2[j].trainIdx].pt.x) &&
                (ReferencePoints[BestMatches[i].queryIdx].pt.y == ReferencePoints[BestMatches2[j].trainIdx].pt.y)){
                BestMatchesBoth.push_back(BestMatches[i]);
            }
        }
    }
    return BestMatchesBoth;

}

void FindPointsToDraw(std::vector<cv::DMatch> BestMatches, std::vector<cv::KeyPoint> Points, cv::Mat image, int CoinSetPoint){

    std::vector<cv::KeyPoint> KeyPointsToDraw;
    std::vector<cv::KeyPoint> CoinPoint;
    cv::Mat OutImage;
    cv::Mat OutImage2;

    // imgIdx is not used in this case
    // trainIdx should correspond to the current img
    // queryIdx should correspond to the reference img
    // the indexes of queryIdx and trainIdx are matching, so first match in trainIdx(current img) is also first in queryIdx(reference img)

    for(size_t i = 0; i < BestMatches.size(); i++){
        KeyPointsToDraw.push_back(Points[BestMatches[i].trainIdx]);
        if(BestMatches[i].queryIdx == CoinSetPoint){
            CoinPoint.push_back(Points[BestMatches[i].trainIdx]);
        }
    }
    for(unsigned int i = 0; i < Points.size(); i++){
        cv::drawKeypoints(image, KeyPointsToDraw, OutImage, cv::Scalar(0,255,0));
    }

    drawCoin(image, CoinPoint);


}


void drawCoin(cv::Mat image, std::vector<cv::KeyPoint> CoinPoint){

    cv::Mat OutImage = image.clone();
    float x, y;

    if(!CoinPoint.empty() && !checkForScore(OutImage, CoinPoint)){
        x = CoinPoint[0].pt.x;
        y = CoinPoint[0].pt.y;
        for(unsigned int j = 0; j < CoinPictureScaled.cols; j++){
            for(unsigned int i = 0; i < CoinPictureScaled.rows; i++){
                Vec3b color = CoinPictureScaled.at<Vec3b>(Point(i,j));
                if((color[0] <= 253) && (color[1] <= 253) && (color[2] <= 253)){
                    OutImage.at<Vec3b>(Point(i+x-((CoinPictureScaled.rows/2)-1), j+y-((CoinPictureScaled.cols/2)-1))) = color;

                }
            }
        }
        //CoinPictureScaled.copyTo(OutImage(cv::Rect(x,y,CoinPictureScaled.cols, CoinPictureScaled.rows)));
        //cv::drawKeypoints(image, CoinPoint, OutImage, cv::Scalar(0,0,255));

        DrawGraphics(OutImage);
    }
    else{
        //cv::imshow("TheAwsomestAugmentedRealityGame_noCoinPoint", OutImage);
        DrawGraphics(OutImage);
    }

}

void DrawGraphics(cv::Mat OutImage){
    //Draws the X-sigth at the middle of the OutImage

    int centerX = floor(OutImage.cols/2);
    int centerY = floor(OutImage.rows/2);
    int length = 5;
    std::string scoreText = "Score: " + std::to_string(score);

    //Creates the aim-lines
    cv::line(OutImage, Point(centerX-length, centerY-length), Point(centerX+length, centerY+length), Scalar(255, 0, 0), 2);
    cv::line(OutImage, Point(centerX-length, centerY+length), Point(centerX+length, centerY-length), Scalar(255, 0, 0), 2);
    //Writes the score at the bottom left
    cv::putText(OutImage, scoreText, Point(10, OutImage.rows-2), FONT_HERSHEY_TRIPLEX, 1, Scalar(0, 0, 0), 2, true);

    cv::imshow("WithCoinPoint", OutImage);
}

bool checkForScore(cv::Mat image, std::vector<cv::KeyPoint> CoinPoint){

    cv::Mat OutImage = image.clone();
    float x, y;
    int slack = 10;

    if(!CoinPoint.empty()){
        x = CoinPoint[0].pt.x;
        y = CoinPoint[0].pt.y;
        int centerX = floor(OutImage.cols/2);
        int centerY = floor(OutImage.rows/2);
        if(((centerX > x-slack) && (centerX < x+slack)) && ((centerY > y-slack) && (centerY < y+slack))){
            flag = 1;
            score = score + 1;
            return true;
        }
        else{
            return false;
        }
    }
    return false;

}

bool timingTracker(long duration){
    int randomNumber = rand() % 10 + 5; //random number between 10 and 15

    if(duration >= randomNumber){
        return true;
    }
    else{
        return false;
    }

}

