#include "opencv2/highgui.hpp"
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv/cv.hpp>
#include "opencv2/xfeatures2d.hpp"
#include <chrono>
#include <cstdlib>
#include <string>


using namespace cv;
using namespace std;

std::vector<cv::KeyPoint> MakePoint(cv::Mat image, cv::Ptr<cv::Feature2D> detector, int maxFeatures);
std::vector<cv::DMatch> FindBestMatches(std::vector<std::vector<cv::DMatch>> matches, std::vector<cv::KeyPoint> ReferencePoints, float ratio);
void FindPointsToDraw(std::vector<cv::DMatch> BestMatches, std::vector<cv::KeyPoint> Points, cv::Mat image, bool isRed);
void drawCoin(cv::Mat image, std::vector<cv::KeyPoint> CoinPoint, bool isRed, float scale);
void DrawGraphics(cv::Mat OutImage);
bool checkForScore(cv::Mat image, std::vector<cv::KeyPoint> CoinPoint, bool isRed);
bool timingTracker(long duration);
std::vector<cv::KeyPoint> findPointsFromBestSquare(std::vector<cv::KeyPoint> Points, cv::Mat image);
std::vector<float> MeanVarOfKeypoints(std::vector<cv::KeyPoint> KeyPoints);


int maxFeatures = 400;
cv::Mat CoinPictureGold = cv::imread("../coin.jpeg", cv::IMREAD_UNCHANGED);//original pic of gold coin
cv::Mat CoinPictureRed = cv::imread("../redCoin.jpg", cv::IMREAD_UNCHANGED);//original pic of red coin
cv::Mat CoinPictureScaledGold; //scaled pic of Gold coin
cv::Mat CoinPictureScaledRed; //scaled pic of Red coin
float scaleFactor = 0.20;
int score = 0;
int flag = 0;
std::vector<float> ReferenceMeanVar;
float ReferanceScale;

using timer = std::chrono::high_resolution_clock;




int main() {
    cv::VideoCapture input_stream(1);
    cv::Ptr<cv::Feature2D> detector = cv::xfeatures2d::SURF::create();
        cv::Ptr<cv::Feature2D> BruteForceDetector = cv::xfeatures2d::SURF::create();
    cv::BFMatcher BruteForceMatcher{BruteForceDetector->defaultNorm()};

    cv::Mat ReferenceImage;
    std::vector<cv::KeyPoint> ReferencePoints;
    cv::Mat ReferenceFeatures;
    bool isRed;

    auto lastEvent = timer::now();


    cv::resize(CoinPictureGold, CoinPictureScaledGold, Size(), scaleFactor, scaleFactor, cv::INTER_LANCZOS4);
    cv::resize(CoinPictureRed, CoinPictureScaledRed, Size(), scaleFactor*1.7, scaleFactor*1.7, cv::INTER_LANCZOS4);



    if (!input_stream.isOpened()) {
        std::cerr << "could not open camera\n";
        return EXIT_FAILURE;
    }
    input_stream.set(cv::CAP_PROP_AUTOFOCUS, false);
    input_stream.set(cv::CAP_PROP_FOCUS, 0.375);

    cv::Mat frame;
    cv::Mat image;
    cv::Mat vis_img;
    input_stream >> frame;
    frame.copyTo(image);

    while (true) {
        input_stream >> frame;
        frame.copyTo(image);

        //cv::TermCriteria CriteriaTerm(TermCriteria::COUNT|TermCriteria::EPS, 20, 0.03);
        cv::Mat GrayImage;
        cv::cvtColor(image, GrayImage, CV_BGR2GRAY);
        std::vector<cv::KeyPoint> Points = MakePoint(GrayImage, detector, maxFeatures);


        cv::Mat CurrentFeatures;
        BruteForceDetector->compute(GrayImage, Points, CurrentFeatures);

        if(ReferenceFeatures.empty()){
            //cv::drawKeypoints(image, Points, OutImage, cv::Scalar(0,255,0));
            //cv::imshow("TheAwsomestAugmentedRealityGame", OutImage);
            //std::vector< cv::Point2f > 	point_ind,
            //cv::KeyPoint::convert(Points, point_ind);
            DrawGraphics(image);

        }
        else{
            std::vector<std::vector<cv::DMatch>> matches;
            //std::vector<std::vector<cv::DMatch>> matches2;

            int numberOfN = 2;
            //Current and reference features have to be the same size!!
            BruteForceMatcher.knnMatch(ReferenceFeatures, CurrentFeatures, matches, numberOfN);
            //BruteForceMatcher.knnMatch(CurrentFeatures, ReferenceFeatures, matches2, numberOfN);
            float ratio = 0.5;

            // show the current matches between ref.img and current image frame
            std::vector<cv::DMatch> BestMatches = FindBestMatches(matches, ReferencePoints, ratio);
            FindPointsToDraw(BestMatches, Points, image, isRed);
            cv::drawMatches(ReferenceImage, ReferencePoints, image, Points, BestMatches, vis_img);
            imshow("Feature matching", vis_img);

        }

        // Trigger detection and saving when space is pressed
        int key = cv::waitKey(5);

        auto timing = std::chrono::duration_cast<std::chrono::seconds>(timer::now() - lastEvent);
        bool event = timingTracker(timing.count());

        if (key == ' ' || event){
            cout << "New coinpoint set." << endl;
            lastEvent = timer::now();
            ReferencePoints = findPointsFromBestSquare(MakePoint(GrayImage, detector, maxFeatures), GrayImage);
            image.copyTo(ReferenceImage);
            BruteForceDetector->compute(GrayImage, ReferencePoints, ReferenceFeatures);
            int randomNumber = rand() % 100;
            ReferenceMeanVar = MeanVarOfKeypoints(ReferencePoints);
            ReferanceScale = 1;
            if (randomNumber <  30 ){
                isRed = true;
            }
            else{
                isRed = false;
            }

            cv::Mat OutImage;
            cv::drawKeypoints(image, ReferencePoints, OutImage, cv::Scalar(0,255,0));
            cv::imshow("The Referance Points", OutImage);
        }
        else if ( key == 'r' || flag == 1){
            lastEvent = timer::now();
            ReferenceImage = cv::Mat();
            ReferencePoints.clear();
            ReferenceFeatures = cv::Mat();
            ReferenceMeanVar = std::vector<float>();
            flag = 0;
        }
        else if (key >= 0){
            cout << "Key was pressed. Exiting" << endl;
            break;
        }

        if (frame.empty()) {
            cout << "Frame was empty. Exiting" << endl;
            break;
        }


    }


    return EXIT_SUCCESS;
}

std::vector<cv::KeyPoint> MakePoint(cv::Mat image, cv::Ptr<cv::Feature2D> detector, int maxFeatures){
//Function to set a SetPoint-frame so we can track from that point.

    //double QualityLevel = 0.01;
    //double MinDistance = 10;
    //cv::Size WinSize(5,5);

    std::vector<cv::KeyPoint> Keypoints;

    detector->detect(image, Keypoints);
    if(!Keypoints.empty()){
        cv::KeyPointsFilter::retainBest(Keypoints, maxFeatures);
    }
    //cv::goodFeaturesToTrack(frame, OutputImage, maxCorners, QualityLevel, MinDistance, Mat(), 3, 3, 0, 0.04);
    //cv::cornerSubPix(frame, OutputImage, WinSize, Size(-1,-1), CriteriaTerm);

    return Keypoints;
}

std::vector<cv::DMatch> FindBestMatches(std::vector<std::vector<cv::DMatch>> matches, std::vector<cv::KeyPoint> ReferencePoints, float ratio){
    std::vector<cv::DMatch> BestMatches;
    std::vector<cv::DMatch> BestMatchesBoth;
    //finding the better match from knn matcher

    for(size_t i = 0;i < matches.size() ;i++) {
        //ReferenceFeatures, CurrentFeatures
        if (matches[i][0].distance < (matches[i][1].distance * ratio)) {
            BestMatches.push_back(matches[i][0]);
        }
        //CurrentFeatures, ReferenceFeatures
    }
    /*
    for(size_t j = 0;j < matches2.size() ;j++) {
        if (matches2[j][0].distance < (matches2[j][1].distance * ratio)) {
            BestMatches2.push_back(matches2[j][0]);
        }
    }*/
    //new vector with only the points we have duplicates of.
    /*for(size_t i = 0;i< BestMatches.size() ;i++) {
        for (size_t j = 0; j < BestMatches2.size(); j++) {
            if ((ReferencePoints[BestMatches[i].queryIdx].pt.x == ReferencePoints[BestMatches2[j].trainIdx].pt.x) &&
                (ReferencePoints[BestMatches[i].queryIdx].pt.y == ReferencePoints[BestMatches2[j].trainIdx].pt.y)){
                BestMatchesBoth.push_back(BestMatches[i]);
            }
        }
    }
    return BestMatchesBoth;*/
    return BestMatches;

}

void FindPointsToDraw(std::vector<cv::DMatch> BestMatches, std::vector<cv::KeyPoint> Points, cv::Mat image, bool isRed){
    std::vector<cv::KeyPoint> KeyPointsToDraw;
    std::vector<cv::KeyPoint> CoinPoint;
    cv::Mat OutImage;
    float increment = 0.1;

    // imgIdx is not used in this case
    // trainIdx should correspond to the current img
    // queryIdx should correspond to the reference img
    // the indexes of queryIdx and trainIdx are matching, so first match in trainIdx(current img) is also first in queryIdx(reference img)

    //Average x and y cordinates of the matches we have. To determine where to place the coin

    if(!Points.empty() && !BestMatches.empty()) {
        CoinPoint.push_back(Points[0]);

        for (size_t i = 0; i < BestMatches.size(); i++) {
            KeyPointsToDraw.push_back(Points[BestMatches[i].trainIdx]);
        }
        std::vector<float> MeanVar = MeanVarOfKeypoints(KeyPointsToDraw);

        CoinPoint[0].pt.x = MeanVar[0];
        CoinPoint[0].pt.y = MeanVar[1];
        cv::drawKeypoints(image, KeyPointsToDraw, OutImage, cv::Scalar(0, 255, 0));
        //image.copyTo(OutImage);

        if (BestMatches.size() > 5 && !CoinPoint.empty()) {
            if(MeanVar[2]>1.1*ReferenceMeanVar[2]){
                ReferanceScale += increment;
            }
            else if(1.1*MeanVar[2]<ReferenceMeanVar[2]){
                ReferanceScale -= increment;
            }
            ReferenceMeanVar[2] = MeanVar[2];

            drawCoin(OutImage, CoinPoint, isRed, ReferanceScale);
        }
    }
    else{
        image.copyTo(OutImage);
        DrawGraphics(OutImage);
    }

}


void drawCoin(cv::Mat image, std::vector<cv::KeyPoint> CoinPoint, bool isRed, float scale){


    float x, y;
    cv::Mat CoinPicture;
    cv::Mat CoinPictureScaled;
    if(isRed){
        CoinPicture = CoinPictureScaledRed;
    }
    else{
        CoinPicture = CoinPictureScaledGold;
    }

    if(!CoinPoint.empty() && !checkForScore(image, CoinPoint, isRed)) {
        cv::resize(CoinPicture, CoinPictureScaled, Size(), scale, scale, cv::INTER_LANCZOS4);
        x = CoinPoint[0].pt.x;
        y = CoinPoint[0].pt.y;
        if ((image.rows > (x + (CoinPictureScaled.rows / 2))) && (x - (CoinPictureScaled.rows / 2) > 0) &&
            (image.cols > (y + (CoinPictureScaled.cols / 2))) && (y - (CoinPictureScaled.cols / 2) > 0)) {


            //cv::Mat insertImage(image, Rect((x - (CoinPictureScaled.rows / 2)), (y - (CoinPictureScaled.cols / 2)), CoinPictureScaled.rows, CoinPictureScaled.cols));
            //CoinPictureScaled.copyTo(insertImage);
            for (unsigned int j = 0; j < CoinPictureScaled.cols; j++) {
                for (unsigned int i = 0; i < CoinPictureScaled.rows; i++) {
                    Vec3b color = CoinPictureScaled.at<Vec3b>(Point(i, j));
                    if ((color[0] <= 249) && (color[1] <= 249) && (color[2] <= 249)) {
                        image.at<Vec3b>(Point(i + x - (floor(CoinPictureScaled.rows / 2) - 1),
                                                 j + y - (floor(CoinPictureScaled.cols / 2) - 1))) = color;
                        //TODO make sure this is entirely inside the image
                    }
                }
            }
        }
        DrawGraphics(image);
    }
    else{
        DrawGraphics(image);
    }
    //cv::imshow("TheAwsomestAugmentedRealityGame_noCoinPoint", OutImage);
}

void DrawGraphics(cv::Mat OutImage){
    //Draws the X-sight at the middle of the OutImage

    int centerX = floor(OutImage.cols/2);
    int centerY = floor(OutImage.rows/2);
    int length = 5;
    std::string scoreText = "Score: " + std::to_string(score);

    //Creates the aim-lines
    cv::line(OutImage, Point(centerX-length, centerY-length), Point(centerX+length, centerY+length), Scalar(255, 0, 0), 2);
    cv::line(OutImage, Point(centerX-length, centerY+length), Point(centerX+length, centerY-length), Scalar(255, 0, 0), 2);
    //Writes the score at the bottom left
    cv::putText(OutImage, scoreText, Point(10, OutImage.rows-2), FONT_HERSHEY_TRIPLEX, 1, Scalar(0, 0, 0), 2, true);

    cv::imshow("The Game", OutImage);
}

bool checkForScore(cv::Mat image, std::vector<cv::KeyPoint> CoinPoint, bool isRed){

    float x, y;
    int slack = 10;

    if(!CoinPoint.empty()){
        x = CoinPoint[0].pt.x;
        y = CoinPoint[0].pt.y;
        int centerX = floor(image.cols/2);
        int centerY = floor(image.rows/2);
        if(((centerX > x-slack) && (centerX < x+slack)) && ((centerY > y-slack) && (centerY < y+slack))){
            if(isRed){
                score = score - 5;
            }
            else{
                score = score + 1;
            }
            flag = 1;
            return true;
        }
    }
    return false;

}

bool timingTracker(long duration){

    if(duration >= 15){
        return true;
    }
    else{
        return false;
    }

}


std::vector<cv::KeyPoint> findPointsFromBestSquare(std::vector<cv::KeyPoint> Points, cv::Mat image){
    int gridSize = 5;
    int row = floor(image.rows/gridSize);
    int col = floor(image.cols/gridSize);

    std::vector<std::vector<cv::KeyPoint>> occurences;

    for(size_t k = 0; k < gridSize*gridSize;k++){
        std::vector<cv::KeyPoint> temp;
        occurences.push_back(temp);
    }

    for(size_t i = 0; i < Points.size() ;i++){
        int x = floor((Points[i].pt.x)/col);
        int y = floor((Points[i].pt.y)/row);
        int nr = (x*gridSize) + y; //which of the squares we put find the point
        if(nr < occurences.size()){
            occurences[nr].push_back(Points[i]);
        }
        else{
            cout << "NR was bigger then occurences. Exiting" << endl;
        }
    }

    std::vector<cv::KeyPoint> largest = occurences[0];
    for(size_t j = 0; j < occurences.size();j++){
        if(largest.size() < occurences[j].size()){
            largest = occurences[j];
        }

    }
    return largest;
}

std::vector<float> MeanVarOfKeypoints(std::vector<cv::KeyPoint> KeyPoints){
    std::vector<float> MeanVar(3);

    int avgx = 0;
    int avgy = 0;
    std::vector<float> variance;

    for (size_t i = 0; i < KeyPoints.size(); i++) {
        avgx += KeyPoints[i].pt.x;
        avgy += KeyPoints[i].pt.y;
    }
    MeanVar[0] = avgx / KeyPoints.size();
    MeanVar[1] = avgy / KeyPoints.size();
    for (size_t i = 0; i < KeyPoints.size(); i++) {
        variance.push_back(pow((KeyPoints[i].pt.x - MeanVar[0]), 2) + pow((KeyPoints[i].pt.y - MeanVar[1]), 2));
    }
    std::sort(variance.begin(), variance.end());

    MeanVar[2] = variance[floor(variance.size()/2)];


    return MeanVar;
}
