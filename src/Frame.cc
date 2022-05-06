/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#include "Frame.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include <thread>

namespace ORB_SLAM2
{

long unsigned int Frame::nNextId=0;
bool Frame::mbInitialComputations=true;
float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

Frame::Frame()
{}

//Copy Constructor
Frame::Frame(const Frame &frame)
    :mpORBvocabulary(frame.mpORBvocabulary), mpORBextractorLeft(frame.mpORBextractorLeft), mpORBextractorRight(frame.mpORBextractorRight),
     mTimeStamp(frame.mTimeStamp), mK(frame.mK.clone()), mDistCoef(frame.mDistCoef.clone()),
     mbf(frame.mbf), mb(frame.mb), mThDepth(frame.mThDepth), N(frame.N), mvKeys(frame.mvKeys),
     mvKeysRight(frame.mvKeysRight), mvKeysUn(frame.mvKeysUn),  mvuRight(frame.mvuRight),
     mvDepth(frame.mvDepth), mBowVec(frame.mBowVec), mFeatVec(frame.mFeatVec),
     mDescriptors(frame.mDescriptors.clone()), mDescriptorsRight(frame.mDescriptorsRight.clone()),
     mvpMapPoints(frame.mvpMapPoints), mvbOutlier(frame.mvbOutlier), mnId(frame.mnId),
     mpReferenceKF(frame.mpReferenceKF), mnScaleLevels(frame.mnScaleLevels),
     mfScaleFactor(frame.mfScaleFactor), mfLogScaleFactor(frame.mfLogScaleFactor),
     mvScaleFactors(frame.mvScaleFactors), mvInvScaleFactors(frame.mvInvScaleFactors),
     mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2),
     imgGray(frame.imgGray.clone()), mvdynKeys(frame.mvdynKeys), mdynDescriptors(frame.mdynDescriptors), 
     mvdynKeysUn(frame.mvdynKeysUn), mvdynDepth(frame.mvdynDepth), mvudynRight(frame.mvudynRight), 
     objects(frame.objects), box_idx(frame.box_idx), box_status(frame.box_status), omit(frame.omit), box_depth(frame.box_depth), box_velocity(frame.box_velocity)
{
    for(int i=0;i<FRAME_GRID_COLS;i++)
        for(int j=0; j<FRAME_GRID_ROWS; j++)
            mGrid[i][j]=frame.mGrid[i][j];

    if(!frame.mTcw.empty())
        SetPose(frame.mTcw);
}

// ori stereo
Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc),mpORBextractorLeft(extractorLeft),mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mpReferenceKF(static_cast<KeyFrame*>(NULL))
{
    imgGray = cv::Mat(imLeft.rows, imLeft.cols, CV_8UC3);
    cv::cvtColor(imLeft, imgGray, CV_GRAY2BGR);

    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    thread threadLeft(&Frame::ExtractORB,this,0,imLeft);
    thread threadRight(&Frame::ExtractORB,this,1,imRight);
    threadLeft.join();
    threadRight.join();

    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

    ComputeStereoMatches();

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));    
    mvbOutlier = vector<bool>(N,false);


    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imLeft);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    AssignFeaturesToGrid();
}

// new stereo
Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, vector<cv::Rect2d> &boxes, Frame &last_frame, const double &timeStamp,
            ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc),mpORBextractorLeft(extractorLeft),mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()),
    mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth), mpReferenceKF(static_cast<KeyFrame*>(NULL))
{
    imgGray = cv::Mat(imLeft.rows, imLeft.cols, CV_8UC3);
    cv::cvtColor(imLeft, imgGray, CV_GRAY2BGR);
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    thread threadLeft(&Frame::ExtractORB,this,0,imLeft);
    thread threadRight(&Frame::ExtractORB,this,1,imRight);
    threadLeft.join();
    threadRight.join();

    //如果没有能够成功提取出特征点，那么就直接返回了
    if(mvKeys.empty())
        return;
    N = mvKeys.size();

    // 对上一帧的boxes进行跟踪
    boxTrack(boxes, last_frame);

    vector<vector<int>> index; //当一个特征点出现在多个box中时进行记录，需要将特征点复制到各个目标框
    vector<bool> hasKpts(boxes.size(), false); //记录每个box是否包含特征点
    bool empty_box = true; // firstSeparate(boxes, index, hasKpts);

    UndistortKeyPoints();

    ComputeStereoMatches();

    // 将box信息分离
    if(N_d>0) {
        int N_s = N - N_d;
        
        mvdynKeys.resize(boxes.size());
        mvdynKeysUn.resize(boxes.size());
        mdynDescriptors.resize(boxes.size());
        mvudynRight.resize(boxes.size());
        mvdynDepth.resize(boxes.size());
        // cout<<"vv "<< boxes.size() << " N "<< N << " mvKeys " << mvKeys.size() <<" N_d " << N_d <<endl;

        for(size_t i(0); i < N_d; i++) {    
            int k = N_s + i;
            for(int j(0); j<index[i].size(); j++) {
                if(empty_box)
                    index[i][j] -= count(hasKpts.begin(), hasKpts.begin() + index[i][j] + 1, false); // 減去當前索引前面的空box个数，如不会出现小于0

                mvdynKeys[index[i][j]].push_back(mvKeys[k]);
                mvdynKeysUn[index[i][j]].push_back(mvKeysUn[k]);
                mdynDescriptors[index[i][j]].push_back(mDescriptors.row(k));
                mvudynRight[index[i][j]].push_back(mvuRight[k]);
                mvdynDepth[index[i][j]].push_back(mvDepth[k]);
            }
        }
        mvKeys.erase(mvKeys.begin() + N_s, mvKeys.end());
        mDescriptors = mDescriptors(cv::Range(0, N_s), cv::Range(0, mDescriptors.cols));
        mvKeysUn.erase(mvKeysUn.begin() + N_s, mvKeysUn.end());
        mvuRight.erase(mvuRight.begin() + N_s, mvuRight.end());
        mvDepth.erase(mvDepth.begin() + N_s, mvDepth.end());

        N = N_s;
    }

    objects = boxes;
    box_status = vector<int>(boxes.size(), -1);
    box_depth = vector<float>(boxes.size(), -1);
    for(auto box:boxes) {
        cv::rectangle(imgGray, box, cv::Scalar(0, 0, 255), 2);
    }

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));    
    mvbOutlier = vector<bool>(N,false);


    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imLeft);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    AssignFeaturesToGrid();
}

// RGBD
Frame::Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth)
{
    imgGray = cv::Mat(imGray.rows, imGray.cols, CV_8UC3);
    cv::cvtColor(imGray, imgGray, CV_GRAY2BGR);
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();    
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    ExtractORB(0,imGray);

    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

    ComputeStereoFromRGBD(imDepth);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    AssignFeaturesToGrid();
}

//new RGBD
Frame::Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const cv::Mat &mask,vector<cv::Rect2d> &boxes, Frame &last_frame, const double &timeStamp, 
                ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth)
{
    imgGray = cv::Mat(imGray.rows, imGray.cols, CV_8UC3);
    cv::cvtColor(imGray, imgGray, CV_GRAY2BGR);
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();    
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    ExtractORB(0,imGray);

    if(mvKeys.empty())
        return;
    N = mvKeys.size();

    boxTrack(boxes, last_frame);

    vector<vector<int>> index; //当一个特征点出现在多个box中时进行记录，需要将特征点复制到各个目标框
    vector<bool> hasKpts(boxes.size(), false); //记录每个box是否包含特征点
    bool empty_box = firstSeparate(mask, boxes, index, hasKpts);

    UndistortKeyPoints();

    ComputeStereoFromRGBD(imDepth);

    // 将box信息分离
    if(N_d>0) {
        int N_s = N - N_d;
        
        mvdynKeys.resize(boxes.size());
        mvdynKeysUn.resize(boxes.size());
        mdynDescriptors.resize(boxes.size());
        mvudynRight.resize(boxes.size());
        mvdynDepth.resize(boxes.size());
        // cout<<"vv "<< boxes.size() << " N "<< N << " mvKeys " << mvKeys.size() <<" N_d " << N_d <<endl;

        for(size_t i(0); i < N_d; i++) {    
            int k = N_s + i;
            for(int j(0); j<index[i].size(); j++) {
                if(empty_box)
                    index[i][j] -= count(hasKpts.begin(), hasKpts.begin() + index[i][j] + 1, false); // 減去當前索引前面的空box个数，如不会出现小于0

                mvdynKeys[index[i][j]].push_back(mvKeys[k]);
                mvdynKeysUn[index[i][j]].push_back(mvKeysUn[k]);
                mdynDescriptors[index[i][j]].push_back(mDescriptors.row(k));
                mvudynRight[index[i][j]].push_back(mvuRight[k]);
                mvdynDepth[index[i][j]].push_back(mvDepth[k]);
            }
        }
        mvKeys.erase(mvKeys.begin() + N_s, mvKeys.end());
        mDescriptors = mDescriptors(cv::Range(0, N_s), cv::Range(0, mDescriptors.cols));
        mvKeysUn.erase(mvKeysUn.begin() + N_s, mvKeysUn.end());
        mvuRight.erase(mvuRight.begin() + N_s, mvuRight.end());
        mvDepth.erase(mvDepth.begin() + N_s, mvDepth.end());

        N = N_s;
    }

    objects = boxes;
    box_status = vector<int>(boxes.size(), -1);
    box_depth = vector<float>(boxes.size(), -1);
    for(int i=0; i< boxes.size(); i++) {
        cv::rectangle(imgGray, boxes[i], cv::Scalar(0, 0, 255), 2);
        cv::Point2f center(boxes[i].x + boxes[i].width/2, boxes[i].y + boxes[i].height/2);
        cv::Point2f predict(center.x + box_velocity[i].x, center.y + box_velocity[i].y);
        cv::arrowedLine(imgGray, center, predict, cv::Scalar(255, 0, 255), 3);
    }

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    AssignFeaturesToGrid();
}


// Monocular
Frame::Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth)
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    ExtractORB(0,imGray);

    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

    // Set no stereo information
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    AssignFeaturesToGrid();
}

void Frame::AssignFeaturesToGrid()
{
    int nReserve = 0.5f*N/(FRAME_GRID_COLS*FRAME_GRID_ROWS);
    for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
        for (unsigned int j=0; j<FRAME_GRID_ROWS;j++)
            mGrid[i][j].reserve(nReserve);

    for(int i=0;i<N;i++)
    {
        const cv::KeyPoint &kp = mvKeysUn[i];

        int nGridPosX, nGridPosY;
        if(PosInGrid(kp,nGridPosX,nGridPosY))
            mGrid[nGridPosX][nGridPosY].push_back(i);
    }
}

// 帧间目标跟踪
void Frame::boxTrack(vector<cv::Rect2d> &boxes, Frame &last_frame){
    int n_box = boxes.size();
    box_idx = vector<int>(n_box, -1);
    omit = vector<bool>(n_box, false);
    box_velocity = vector<cv::Point2d>(n_box, cv::Point2d(0,0));
    // cout<< "last_frame.objects.size() " << last_frame.objects.size() << ", n_box" <<n_box<<endl;
    if(!last_frame.objects.empty()) {
        for(int i(0); i<last_frame.objects.size(); i++) {
            double minCost = 1;
            int minPosy = -1;
            for(int j(0); j<n_box; j++) {
                // 匹配过了直接跳过
                if(box_idx[j] != -1) {
                    continue;
                }
                cv::Rect2d inter, uni;
                inter = last_frame.objects[i] & boxes[j];
                uni = last_frame.objects[i] | boxes[j];
                // cout << ", inter" << inter<<endl;
                double cost = 1 - inter.area() / uni.area();

                if(cost < minCost) {
                    minCost = cost;
                    minPosy = j;
                }
            }
            // cout<<"min_y "<<minPosy<<endl;
            if(-1 != minPosy) {
                box_idx[minPosy] = last_frame.box_idx[i];
                box_velocity[minPosy].x = boxes[minPosy].x + boxes[minPosy].width/2 - last_frame.objects[i].x - last_frame.objects[i].width/2;
                box_velocity[minPosy].y = boxes[minPosy].y + boxes[minPosy].height/2 - last_frame.objects[i].y - last_frame.objects[i].height/2;
            }
            // 未匹配上，则将上一帧的框添加给下一帧
            else {
                // if(!last_frame.omit[i]) //只允许添加一次
                cv::Point2f center_cur(last_frame.objects[i].x + last_frame.objects[i].width/2 + last_frame.box_velocity[i].x,
                                        last_frame.objects[i].y + last_frame.objects[i].height/2 + last_frame.box_velocity[i].y);
                cv::Rect2f rect_cur(cv::Point2f(0,0), cv::Point2f(imgGray.cols, imgGray.rows));
                if(rect_cur.contains(center_cur) && !last_frame.omit[i])
                {
                    cv::Rect2d box_cur = last_frame.objects[i] + last_frame.box_velocity[i];
                    boxes.push_back(box_cur);
                    box_idx.push_back(last_frame.box_idx[i]);
                    omit.push_back(true);
                    box_velocity.push_back(last_frame.box_velocity[i]);
                    last_frame.box_status[i] = 0;
                }
            }
        }

        // 当前帧中未匹配的框视为新出现的目标，添加编号
        for(int i(0); i<n_box; i++)
        {
            if(box_idx[i] != -1) continue;
            int max_idx = *max_element(box_idx.begin(), box_idx.end());
            box_idx[i] = max_idx + 1;
        }
    }
    // 如果是重新初始化了，则重新编号
    else {
        for(int i(0); i<n_box; i++) {
            box_idx[i] = i;
        }
    }
}

//直接根据目标框筛选特征点
bool Frame::firstSeparate(const cv::Mat &mask, vector<cv::Rect2d> &boxes, vector<vector<int>> &index, vector<bool> &hasKpts){
        std::vector<cv::KeyPoint> _mvKeys, _mdynKeys;
    cv::Mat _mDescriptors, _mdynDescriptors;
    for (size_t i(0); i < mvKeys.size(); ++i)
    {
        vector<int> idx;
        for(int j=0; j<boxes.size(); j++){
            if(boxes[j].contains(mvKeys[i].pt) && mask.at<char>(mvKeys[i].pt.x, mvKeys[i].pt.y) != 0) {
                hasKpts[j] = true; 
                if(-1 == mvKeys[i].class_id){
                    idx.push_back(j);
                    mvKeys[i].class_id = i;
                }
                else {
                    idx.push_back(j);
                }
            }
        }
        if(idx.size()>0) {
            index.push_back(idx);
            _mdynKeys.push_back(mvKeys[i]);
            _mdynDescriptors.push_back(mDescriptors.row(i));
        }
        else {
            _mvKeys.push_back(mvKeys[i]);
            _mDescriptors.push_back(mDescriptors.row(i));
        }
    }
    bool empty_box = false;
    for(int i(0); i<boxes.size(); i++) {
        if(hasKpts[i] == true) continue;
        boxes.erase(boxes.begin() + i);
        box_idx.erase(box_idx.begin() + i);
        omit.erase(omit.begin() + i);
        box_velocity.erase(box_velocity.begin() + i);
        empty_box = true;
    }

    N_d = index.size();
    // 若存在动态点，则重新排序,将动态区域的点放在后面，便于后续进行erase
    if(N_d>0) {
        mvKeys = _mvKeys;
        mvKeys.insert(mvKeys.end(), _mdynKeys.begin(), _mdynKeys.end());
        if(_mDescriptors.cols>0)
            cv::vconcat(_mDescriptors, _mdynDescriptors, mDescriptors);
        else mDescriptors = _mdynDescriptors;
    }
    return empty_box;
}

//将动态区域的静态特征点更新到当前帧
void Frame::UpdateFrame(const vector<vector<int>> &dynStatus) {
    // cout<<mvdynKeys.size()<<" "<<dynStatus.size()<<endl;
    // 可能多个框有相同的特征点，使用set进行查重避免重复push_back
    unordered_set<int> index;
    for(int i(0); i<dynStatus.size(); i++) {
        // cout << dynStatus[i].size() << " mvdynKeys[i].size(): "<<mvdynKeys[i].size()<< endl;
        if(dynStatus[i].size() == 0) continue;

        for(int j(0); j<dynStatus[i].size(); j++) {
            if(dynStatus[i][j] == -1) continue;
            if(index.find(mvdynKeys[i][dynStatus[i][j]].class_id) != index.end()) continue;
            // cout<<"dynStatus[i][j] "<<dynStatus[i][j]<<endl;

            index.insert(mvdynKeys[i][dynStatus[i][j]].class_id); 

            mvKeys.push_back(mvdynKeys[i][dynStatus[i][j]]);
            mvKeysUn.push_back(mvdynKeysUn[i][dynStatus[i][j]]);
            mDescriptors.push_back(mdynDescriptors[i].row(dynStatus[i][j]));
            mvuRight.push_back(mvudynRight[i][dynStatus[i][j]]);
            mvDepth.push_back(mvdynDepth[i][dynStatus[i][j]]);
        }
    }

    //更新特征点的个数
    N_ori = N;
    N = mvKeys.size();

    // 初始化本帧的地图点
    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
	// 记录地图点是否为外点，初始化均为外点false
    mvbOutlier = vector<bool>(N,false);

    // 将新添加的特征点分配到图像网格中 
    UpdateFeaturesToGrid();
}

void Frame::UpdateFeaturesToGrid()
{
    for(int i=N_ori;i<N;i++)
    {
        const cv::KeyPoint &kp = mvKeysUn[i];

        int nGridPosX, nGridPosY;
        if(PosInGrid(kp,nGridPosX,nGridPosY))
            mGrid[nGridPosX][nGridPosY].push_back(i);
    }
}

void Frame::ExtractORB(int flag, const cv::Mat &im)
{
    if(flag==0)
        (*mpORBextractorLeft)(im,cv::Mat(),mvKeys,mDescriptors);
    else
        (*mpORBextractorRight)(im,cv::Mat(),mvKeysRight,mDescriptorsRight);
}

void Frame::SetPose(cv::Mat Tcw)
{
    mTcw = Tcw.clone();
    UpdatePoseMatrices();
}

void Frame::UpdatePoseMatrices()
{ 
    mRcw = mTcw.rowRange(0,3).colRange(0,3);
    mRwc = mRcw.t();
    mtcw = mTcw.rowRange(0,3).col(3);
    mOw = -mRcw.t()*mtcw;
}

bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit)
{
    pMP->mbTrackInView = false;

    // 3D in absolute coordinates
    cv::Mat P = pMP->GetWorldPos(); 

    // 3D in camera coordinates
    const cv::Mat Pc = mRcw*P+mtcw;
    const float &PcX = Pc.at<float>(0);
    const float &PcY= Pc.at<float>(1);
    const float &PcZ = Pc.at<float>(2);

    // Check positive depth
    if(PcZ<0.0f)
        return false;

    // Project in image and check it is not outside
    const float invz = 1.0f/PcZ;
    const float u=fx*PcX*invz+cx;
    const float v=fy*PcY*invz+cy;

    if(u<mnMinX || u>mnMaxX)
        return false;
    if(v<mnMinY || v>mnMaxY)
        return false;

    // Check distance is in the scale invariance region of the MapPoint
    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    const cv::Mat PO = P-mOw;
    const float dist = cv::norm(PO);

    if(dist<minDistance || dist>maxDistance)
        return false;

   // Check viewing angle
    cv::Mat Pn = pMP->GetNormal();

    const float viewCos = PO.dot(Pn)/dist;

    if(viewCos<viewingCosLimit)
        return false;

    // Predict scale in the image
    const int nPredictedLevel = pMP->PredictScale(dist,this);

    // Data used by the tracking
    pMP->mbTrackInView = true;
    pMP->mTrackProjX = u;
    pMP->mTrackProjXR = u - mbf*invz;
    pMP->mTrackProjY = v;
    pMP->mnTrackScaleLevel= nPredictedLevel;
    pMP->mTrackViewCos = viewCos;

    return true;
}

vector<size_t> Frame::GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N);

    const int nMinCellX = max(0,(int)floor((x-mnMinX-r)*mfGridElementWidthInv));
    if(nMinCellX>=FRAME_GRID_COLS)
        return vIndices;

    const int nMaxCellX = min((int)FRAME_GRID_COLS-1,(int)ceil((x-mnMinX+r)*mfGridElementWidthInv));
    if(nMaxCellX<0)
        return vIndices;

    const int nMinCellY = max(0,(int)floor((y-mnMinY-r)*mfGridElementHeightInv));
    if(nMinCellY>=FRAME_GRID_ROWS)
        return vIndices;

    const int nMaxCellY = min((int)FRAME_GRID_ROWS-1,(int)ceil((y-mnMinY+r)*mfGridElementHeightInv));
    if(nMaxCellY<0)
        return vIndices;

    const bool bCheckLevels = (minLevel>0) || (maxLevel>=0);

    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            const vector<size_t> vCell = mGrid[ix][iy];
            if(vCell.empty())
                continue;

            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
                if(bCheckLevels)
                {
                    if(kpUn.octave<minLevel)
                        continue;
                    if(maxLevel>=0)
                        if(kpUn.octave>maxLevel)
                            continue;
                }

                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

                if(fabs(distx)<r && fabs(disty)<r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
{
    posX = round((kp.pt.x-mnMinX)*mfGridElementWidthInv);
    posY = round((kp.pt.y-mnMinY)*mfGridElementHeightInv);

    //Keypoint's coordinates are undistorted, which could cause to go out of the image
    if(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)
        return false;

    return true;
}


void Frame::ComputeBoW()
{
    if(mBowVec.empty())
    {
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
    }
}

void Frame::UndistortKeyPoints()
{
    if(mDistCoef.at<float>(0)==0.0)
    {
        mvKeysUn=mvKeys;
        return;
    }

    // Fill matrix with points
    cv::Mat mat(N,2,CV_32F);
    for(int i=0; i<N; i++)
    {
        mat.at<float>(i,0)=mvKeys[i].pt.x;
        mat.at<float>(i,1)=mvKeys[i].pt.y;
    }

    // Undistort points
    mat=mat.reshape(2);
    cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
    mat=mat.reshape(1);

    // Fill undistorted keypoint vector
    mvKeysUn.resize(N);
    for(int i=0; i<N; i++)
    {
        cv::KeyPoint kp = mvKeys[i];
        kp.pt.x=mat.at<float>(i,0);
        kp.pt.y=mat.at<float>(i,1);
        mvKeysUn[i]=kp;
    }
}

void Frame::ComputeImageBounds(const cv::Mat &imLeft)
{
    if(mDistCoef.at<float>(0)!=0.0)
    {
        cv::Mat mat(4,2,CV_32F);
        mat.at<float>(0,0)=0.0; mat.at<float>(0,1)=0.0;
        mat.at<float>(1,0)=imLeft.cols; mat.at<float>(1,1)=0.0;
        mat.at<float>(2,0)=0.0; mat.at<float>(2,1)=imLeft.rows;
        mat.at<float>(3,0)=imLeft.cols; mat.at<float>(3,1)=imLeft.rows;

        // Undistort corners
        mat=mat.reshape(2);
        cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
        mat=mat.reshape(1);

        mnMinX = min(mat.at<float>(0,0),mat.at<float>(2,0));
        mnMaxX = max(mat.at<float>(1,0),mat.at<float>(3,0));
        mnMinY = min(mat.at<float>(0,1),mat.at<float>(1,1));
        mnMaxY = max(mat.at<float>(2,1),mat.at<float>(3,1));

    }
    else
    {
        mnMinX = 0.0f;
        mnMaxX = imLeft.cols;
        mnMinY = 0.0f;
        mnMaxY = imLeft.rows;
    }
}

void Frame::ComputeStereoMatches()
{
    mvuRight = vector<float>(N,-1.0f);
    mvDepth = vector<float>(N,-1.0f);

    const int thOrbDist = (ORBmatcher::TH_HIGH+ORBmatcher::TH_LOW)/2;

    const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;

    //Assign keypoints to row table
    vector<vector<size_t> > vRowIndices(nRows,vector<size_t>());

    for(int i=0; i<nRows; i++)
        vRowIndices[i].reserve(200);

    const int Nr = mvKeysRight.size();

    for(int iR=0; iR<Nr; iR++)
    {
        const cv::KeyPoint &kp = mvKeysRight[iR];
        const float &kpY = kp.pt.y;
        const float r = 2.0f*mvScaleFactors[mvKeysRight[iR].octave];
        const int maxr = ceil(kpY+r);
        const int minr = floor(kpY-r);

        for(int yi=minr;yi<=maxr;yi++)
            vRowIndices[yi].push_back(iR);
    }

    // Set limits for search
    const float minZ = mb;
    const float minD = 0;
    const float maxD = mbf/minZ;

    // For each left keypoint search a match in the right image
    vector<pair<int, int> > vDistIdx;
    vDistIdx.reserve(N);

    for(int iL=0; iL<N; iL++)
    {
        const cv::KeyPoint &kpL = mvKeys[iL];
        const int &levelL = kpL.octave;
        const float &vL = kpL.pt.y;
        const float &uL = kpL.pt.x;

        const vector<size_t> &vCandidates = vRowIndices[vL];

        if(vCandidates.empty())
            continue;

        const float minU = uL-maxD;
        const float maxU = uL-minD;

        if(maxU<0)
            continue;

        int bestDist = ORBmatcher::TH_HIGH;
        size_t bestIdxR = 0;

        const cv::Mat &dL = mDescriptors.row(iL);

        // Compare descriptor to right keypoints
        for(size_t iC=0; iC<vCandidates.size(); iC++)
        {
            const size_t iR = vCandidates[iC];
            const cv::KeyPoint &kpR = mvKeysRight[iR];

            if(kpR.octave<levelL-1 || kpR.octave>levelL+1)
                continue;

            const float &uR = kpR.pt.x;

            if(uR>=minU && uR<=maxU)
            {
                const cv::Mat &dR = mDescriptorsRight.row(iR);
                const int dist = ORBmatcher::DescriptorDistance(dL,dR);

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdxR = iR;
                }
            }
        }

        // Subpixel match by correlation
        if(bestDist<thOrbDist)
        {
            // coordinates in image pyramid at keypoint scale
            const float uR0 = mvKeysRight[bestIdxR].pt.x;
            const float scaleFactor = mvInvScaleFactors[kpL.octave];
            const float scaleduL = round(kpL.pt.x*scaleFactor);
            const float scaledvL = round(kpL.pt.y*scaleFactor);
            const float scaleduR0 = round(uR0*scaleFactor);

            // sliding window search
            const int w = 5;
            cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduL-w,scaleduL+w+1);
            IL.convertTo(IL,CV_32F);
            IL = IL - IL.at<float>(w,w) *cv::Mat::ones(IL.rows,IL.cols,CV_32F);

            int bestDist = INT_MAX;
            int bestincR = 0;
            const int L = 5;
            vector<float> vDists;
            vDists.resize(2*L+1);

            const float iniu = scaleduR0+L-w;
            const float endu = scaleduR0+L+w+1;
            if(iniu<0 || endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols)
                continue;

            for(int incR=-L; incR<=+L; incR++)
            {
                cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduR0+incR-w,scaleduR0+incR+w+1);
                IR.convertTo(IR,CV_32F);
                IR = IR - IR.at<float>(w,w) *cv::Mat::ones(IR.rows,IR.cols,CV_32F);

                float dist = cv::norm(IL,IR,cv::NORM_L1);
                if(dist<bestDist)
                {
                    bestDist =  dist;
                    bestincR = incR;
                }

                vDists[L+incR] = dist;
            }

            if(bestincR==-L || bestincR==L)
                continue;

            // Sub-pixel match (Parabola fitting)
            const float dist1 = vDists[L+bestincR-1];
            const float dist2 = vDists[L+bestincR];
            const float dist3 = vDists[L+bestincR+1];

            const float deltaR = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));

            if(deltaR<-1 || deltaR>1)
                continue;

            // Re-scaled coordinate
            float bestuR = mvScaleFactors[kpL.octave]*((float)scaleduR0+(float)bestincR+deltaR);

            float disparity = (uL-bestuR);

            if(disparity>=minD && disparity<maxD)
            {
                if(disparity<=0)
                {
                    disparity=0.01;
                    bestuR = uL-0.01;
                }
                mvDepth[iL]=mbf/disparity;
                mvuRight[iL] = bestuR;
                vDistIdx.push_back(pair<int,int>(bestDist,iL));
            }
        }
    }

    sort(vDistIdx.begin(),vDistIdx.end());
    const float median = vDistIdx[vDistIdx.size()/2].first;
    const float thDist = 1.5f*1.4f*median;

    for(int i=vDistIdx.size()-1;i>=0;i--)
    {
        if(vDistIdx[i].first<thDist)
            break;
        else
        {
            mvuRight[vDistIdx[i].second]=-1;
            mvDepth[vDistIdx[i].second]=-1;
        }
    }
}


void Frame::ComputeStereoFromRGBD(const cv::Mat &imDepth)
{
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    for(int i=0; i<N; i++)
    {
        const cv::KeyPoint &kp = mvKeys[i];
        const cv::KeyPoint &kpU = mvKeysUn[i];

        const float &v = kp.pt.y;
        const float &u = kp.pt.x;

        const float d = imDepth.at<float>(v,u);

        if(d>0)
        {
            mvDepth[i] = d;
            mvuRight[i] = kpU.pt.x-mbf/d;
        }
    }
}

cv::Mat Frame::UnprojectStereo(const int &i)
{
    const float z = mvDepth[i];
    if(z>0)
    {
        const float u = mvKeysUn[i].pt.x;
        const float v = mvKeysUn[i].pt.y;
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);
        return mRwc*x3Dc+mOw;
    }
    else
        return cv::Mat();
}

} //namespace ORB_SLAM
