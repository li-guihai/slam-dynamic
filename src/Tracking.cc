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

#include <unistd.h>

#include "Tracking.h"

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include"ORBmatcher.h"
#include"FrameDrawer.h"
#include"Converter.h"
#include"Map.h"
#include"Initializer.h"

#include"Optimizer.h"
#include"PnPsolver.h"

#include<iostream>

#include<mutex>

#define YOLO_S true

using namespace std;

namespace ORB_SLAM2
{

Tracking::Tracking(System *pSys, ORBVocabulary* pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Map *pMap, shared_ptr<PointCloudMapping> pPointCloud, KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor):
    mState(NO_IMAGES_YET), mSensor(sensor), mbOnlyTracking(false), mbVO(false), mpORBVocabulary(pVoc),
    mpPointCloudMapping( pPointCloud ),
    mpKeyFrameDB(pKFDB), mpInitializer(static_cast<Initializer*>(NULL)), mpSystem(pSys), mpViewer(NULL),
    mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpMap(pMap), mnLastRelocFrameId(0)
{
    // Load camera parameters from settings file

    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    float fps = fSettings["Camera.fps"];
    if(fps==0)
        fps=30;

    // Max/Min Frames to insert keyframes and to check relocalisation
    mMinFrames = 0;
    mMaxFrames = fps;

    cout << endl << "Camera Parameters: " << endl;
    cout << "- fx: " << fx << endl;
    cout << "- fy: " << fy << endl;
    cout << "- cx: " << cx << endl;
    cout << "- cy: " << cy << endl;
    cout << "- k1: " << DistCoef.at<float>(0) << endl;
    cout << "- k2: " << DistCoef.at<float>(1) << endl;
    if(DistCoef.rows==5)
        cout << "- k3: " << DistCoef.at<float>(4) << endl;
    cout << "- p1: " << DistCoef.at<float>(2) << endl;
    cout << "- p2: " << DistCoef.at<float>(3) << endl;
    cout << "- fps: " << fps << endl;


    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    if(mbRGB)
        cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
        cout << "- color order: BGR (ignored if grayscale)" << endl;

    // Load ORB parameters

    int nFeatures = fSettings["ORBextractor.nFeatures"];
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
    int nLevels = fSettings["ORBextractor.nLevels"];
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
    int fMinThFAST = fSettings["ORBextractor.minThFAST"];

    mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(sensor==System::STEREO)
        mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(sensor==System::MONOCULAR)
        mpIniORBextractor = new ORBextractor(2*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    cout << endl  << "ORB Extractor Parameters: " << endl;
    cout << "- Number of Features: " << nFeatures << endl;
    cout << "- Scale Levels: " << nLevels << endl;
    cout << "- Scale Factor: " << fScaleFactor << endl;
    cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
    cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

    if(sensor==System::STEREO || sensor==System::RGBD)
    {
        mThDepth = mbf*(float)fSettings["ThDepth"]/fx;
        cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
    }

    if(sensor==System::RGBD)
    {
        mDepthMapFactor = fSettings["DepthMapFactor"];
        if(fabs(mDepthMapFactor)<1e-5)
            mDepthMapFactor=1;
        else
            mDepthMapFactor = 1.0f/mDepthMapFactor;
    }

}

void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}

void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
{
    mpLoopClosing=pLoopClosing;
}

void Tracking::SetViewer(Viewer *pViewer)
{
    mpViewer=pViewer;
}


cv::Mat Tracking::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp)
{
    mImGray = imRectLeft;
    cv::Mat imGrayRight = imRectRight;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGB2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGR2GRAY);
        }
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGBA2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGRA2GRAY);
        }
    }

    mCurrentFrame = Frame(mImGray,imGrayRight,imRectLeft, timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    Track();

    return mCurrentFrame.mTcw.clone();
}

// new stereo
cv::Mat Tracking::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, vector<cv::Rect2d> &boxes, const double &timestamp)
{
    mImGray = imRectLeft;
    cv::Mat imGrayRight = imRectRight;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGB2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGR2GRAY);
        }
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGBA2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGRA2GRAY);
        }
    }

    mCurrentFrame = Frame(mImGray,imGrayRight,imRectLeft, boxes,mLastFrame, timestamp,
                        mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);
    cout<<"frame done"<<endl<<endl;
    Track_new();

    return mCurrentFrame.mTcw.clone();
}

cv::Mat Tracking::GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp)
{
    mImGray = imRGB;
    mImRGB = imRGB;
    imDepth = imD;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    if((fabs(mDepthMapFactor-1.0f)>1e-5) || imDepth.type()!=CV_32F)
        imDepth.convertTo(imDepth,CV_32F,mDepthMapFactor);

    mCurrentFrame = Frame(mImGray,imDepth,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);
    cout<<"frame "<<mCurrentFrame.mnId<<" done"<<endl;
    Track_new();

    return mCurrentFrame.mTcw.clone();
}

// new RGBD
cv::Mat Tracking::GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const cv::Mat &mask, vector<cv::Rect2d> &boxes, const double &timestamp)
{
    mImGray = imRGB;
    mImRGB = imRGB;
    mImMask = mask;
    imDepth = imD;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    if((fabs(mDepthMapFactor-1.0f)>1e-5) || imDepth.type()!=CV_32F)
        imDepth.convertTo(imDepth,CV_32F,mDepthMapFactor);

    mCurrentFrame = Frame(mImGray, mImRGB, imDepth,mask,boxes,mLastFrame, 
                            timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);
    cout<<"frame done"<<endl;

    Track_new();

    return mCurrentFrame.mTcw.clone();
}

cv::Mat Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp)
{
    mImGray = im;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET)
        mCurrentFrame = Frame(mImGray,timestamp,im,mpIniORBextractor,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);
    else
        mCurrentFrame = Frame(mImGray,timestamp,im,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);
    
    Track_new();

    return mCurrentFrame.mTcw.clone();
}

void Tracking::Track()
{
    if(mState==NO_IMAGES_YET)
    {
        mState = NOT_INITIALIZED;
    }

    mLastProcessedState=mState;

    // Get Map Mutex -> Map cannot be changed
    unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

    if(mState==NOT_INITIALIZED)
    {
        if(mSensor==System::STEREO || mSensor==System::RGBD)
            StereoInitialization();
        else
            MonocularInitialization();

        mpFrameDrawer->Update(this);

        if(mState!=OK)
            return;
    }
    else
    {
        // System is initialized. Track Frame.
        bool bOK;

        // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
        if(!mbOnlyTracking)
        {
            // Local Mapping is activated. This is the normal behaviour, unless
            // you explicitly activate the "only tracking" mode.

            if(mState==OK)
            {
                // Local Mapping might have changed some MapPoints tracked in last frame
                CheckReplacedInLastFrame();

                if(mVelocity.empty() || mCurrentFrame.mnId<mnLastRelocFrameId+2)
                {
                    bOK = TrackReferenceKeyFrame();
                }
                else
                {
                    bOK = TrackWithMotionModel();
                    if(!bOK)
                        bOK = TrackReferenceKeyFrame();
                }
            }
            else
            {
                bOK = Relocalization();
            }
        }
        else
        {
            // Localization Mode: Local Mapping is deactivated

            if(mState==LOST)
            {
                bOK = Relocalization();
            }
            else
            {
                if(!mbVO)
                {
                    // In last frame we tracked enough MapPoints in the map

                    if(!mVelocity.empty())
                    {
                        bOK = TrackWithMotionModel();
                    }
                    else
                    {
                        bOK = TrackReferenceKeyFrame();
                    }
                }
                else
                {
                    // In last frame we tracked mainly "visual odometry" points.

                    // We compute two camera poses, one from motion model and one doing relocalization.
                    // If relocalization is sucessfull we choose that solution, otherwise we retain
                    // the "visual odometry" solution.

                    bool bOKMM = false;
                    bool bOKReloc = false;
                    vector<MapPoint*> vpMPsMM;
                    vector<bool> vbOutMM;
                    cv::Mat TcwMM;
                    if(!mVelocity.empty())
                    {
                        bOKMM = TrackWithMotionModel();
                        vpMPsMM = mCurrentFrame.mvpMapPoints;
                        vbOutMM = mCurrentFrame.mvbOutlier;
                        TcwMM = mCurrentFrame.mTcw.clone();
                    }
                    bOKReloc = Relocalization();

                    if(bOKMM && !bOKReloc)
                    {
                        mCurrentFrame.SetPose(TcwMM);
                        mCurrentFrame.mvpMapPoints = vpMPsMM;
                        mCurrentFrame.mvbOutlier = vbOutMM;

                        if(mbVO)
                        {
                            for(int i =0; i<mCurrentFrame.N; i++)
                            {
                                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                                {
                                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                                }
                            }
                        }
                    }
                    else if(bOKReloc)
                    {
                        mbVO = false;
                    }

                    bOK = bOKReloc || bOKMM;
                }
            }
        }

        mCurrentFrame.mpReferenceKF = mpReferenceKF;

        // If we have an initial estimation of the camera pose and matching. Track the local map.
        if(!mbOnlyTracking)
        {
            if(bOK)
                bOK = TrackLocalMap();
        }
        else
        {
            // mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
            // a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
            // the camera we will use the local map again.
            if(bOK && !mbVO)
                bOK = TrackLocalMap();
        }

        if(bOK)
            mState = OK;
        else
            mState=LOST;

        // Update drawer
        mpFrameDrawer->Update(this);

        // If tracking were good, check if we insert a keyframe
        if(bOK)
        {
            // Update motion model
            if(!mLastFrame.mTcw.empty())
            {
                cv::Mat LastTwc = cv::Mat::eye(4,4,CV_32F);
                mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0,3).colRange(0,3));
                mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0,3).col(3));
                mVelocity = mCurrentFrame.mTcw*LastTwc;
            }
            else
                mVelocity = cv::Mat();

            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

            // Clean VO matches
            for(int i=0; i<mCurrentFrame.N; i++)
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(pMP)
                    if(pMP->Observations()<1)
                    {
                        mCurrentFrame.mvbOutlier[i] = false;
                        mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                    }
            }

            // Delete temporal MapPoints
            for(list<MapPoint*>::iterator lit = mlpTemporalPoints.begin(), lend =  mlpTemporalPoints.end(); lit!=lend; lit++)
            {
                MapPoint* pMP = *lit;
                delete pMP;
            }
            mlpTemporalPoints.clear();

            // Check if we need to insert a new keyframe
            if(NeedNewKeyFrame())
                CreateNewKeyFrame();

            // We allow points with high innovation (considererd outliers by the Huber Function)
            // pass to the new keyframe, so that bundle adjustment will finally decide
            // if they are outliers or not. We don't want next frame to estimate its position
            // with those points so we discard them in the frame.
            for(int i=0; i<mCurrentFrame.N;i++)
            {
                if(mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                    mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
            }
        }

        // Reset if the camera get lost soon after initialization
        if(mState==LOST)
        {
            if(mpMap->KeyFramesInMap()<=5)
            {
                cout << "Track lost soon after initialisation, reseting..." << endl;
                mpSystem->Reset();
                return;
            }
        }

        if(!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

        mLastFrame = Frame(mCurrentFrame);
    }

    // Store frame pose information to retrieve the complete camera trajectory afterwards.
    if(!mCurrentFrame.mTcw.empty())
    {
        cv::Mat Tcr = mCurrentFrame.mTcw*mCurrentFrame.mpReferenceKF->GetPoseInverse();
        mlRelativeFramePoses.push_back(Tcr);
        mlpReferences.push_back(mpReferenceKF);
        mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
        mlbLost.push_back(mState==LOST);
    }
    else
    {
        // This can happen if tracking is lost
        mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
        mlpReferences.push_back(mlpReferences.back());
        mlFrameTimes.push_back(mlFrameTimes.back());
        mlbLost.push_back(mState==LOST);
    }

}

void Tracking::Track_new()
{
    if(mState==NO_IMAGES_YET)
    {
        mState = NOT_INITIALIZED;
    }

    mLastProcessedState=mState;

    // Get Map Mutex -> Map cannot be changed
    unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

    if(mState==NOT_INITIALIZED)
    {
        #if YOLO_S
        // // 初始化时清空队列
        if(!q_frame.empty())
        {
            q_frame = queue<Frame>();
        }
        #endif

        if(mSensor==System::STEREO || mSensor==System::RGBD)
            StereoInitialization();
        else
            MonocularInitialization();

        mpFrameDrawer->Update(this);

        if(mState!=OK)
            return;
    }
    else
    {
        // 只有当前帧动态点占一定比例时进行筛选，从上n帧一直查找到前0.4秒为止
        #if YOLO_S
        if(!mCurrentFrame.objects.empty() && !q_frame.empty()) {
            while(mCurrentFrame.mTimeStamp - q_frame.front().mTimeStamp > 0.2f) {
                // if(mCurrentFrame.N_d > 10) {
                mRefFrame = &q_frame.front();
                if(mRefFrame->objects.empty()) {
                    q_frame.pop();
                    continue;
                }
                // cout<<"q.size " << q_frame.size() << ", mRefFrame->mTimeStamp " << mRefFrame->mTimeStamp <<endl;
                cout<<"track frame "<<mCurrentFrame.mnId<< "-" << mRefFrame->mnId << ", time "<<mCurrentFrame.mTimeStamp - mRefFrame->mTimeStamp<< endl;
                // mRefFrame = &mLastFrame;
                
                cv::Mat HorF;
                int flag = TrackHomo(HorF);
                // cout<<"0mref "<<mRefFrame->mdynDescriptors.rows <<" mvkeys " << mRefFrame->mvKeys.size()<<" mvDepth "<<mRefFrame->mvdynDepth.size()<<endl;

                // cout<<"flag: "<<flag<<endl;
                if(flag != 0) {
                    // Optimize frame pose with all matches, 优化位姿用于后面重投影
                    // if(mSensor == System::RGBD)
                    //     Optimizer::PoseOptimization(&mCurrentFrame);
                    if(flag == 1)
                        cout << "TrackH success" << endl;
                    else 
                        cout << "TrackF success" << endl;

                    vector<vector<int>> dynSatus; //初始化为都是动态点
                    // cout<<"mref "<<mRefFrame->mdynDescriptors.rows <<" mvkeys " << mRefFrame->mvKeys.size()<<" mvDepth "<<mRefFrame->mvdynDepth.size()<<endl;

                    if(Separate(HorF, flag, dynSatus) == 1) {
                        mCurrentFrame.UpdateFrame(dynSatus);
                    }
                    break;
                }
                // 若两帧没有匹配上则切换到下一帧
                else{
                    if(q_frame.size()==1)
                        break;
                    q_frame.pop();
                    // cout<< q_frame.front().mTimeStamp << endl;
                }
            }
        }

        #endif

        // System is initialized. Track Frame.
        bool bOK;

        // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
        if(!mbOnlyTracking)
        {
            // Local Mapping is activated. This is the normal behaviour, unless
            // you explicitly activate the "only tracking" mode.

            if(mState==OK)
            {
                // Local Mapping might have changed some MapPoints tracked in last frame
                CheckReplacedInLastFrame();

                if(mVelocity.empty() || mCurrentFrame.mnId<mnLastRelocFrameId+2)
                {
                    bOK = TrackReferenceKeyFrame();
                }
                else
                {
                    bOK = TrackWithMotionModel();
                    if(!bOK)
                        bOK = TrackReferenceKeyFrame();
                }
            }
            else
            {
                bOK = Relocalization();
            }
        }
        else
        {
            // Localization Mode: Local Mapping is deactivated

            if(mState==LOST)
            {
                bOK = Relocalization();
            }
            else
            {
                if(!mbVO)
                {
                    // In last frame we tracked enough MapPoints in the map

                    if(!mVelocity.empty())
                    {
                        bOK = TrackWithMotionModel();
                    }
                    else
                    {
                        bOK = TrackReferenceKeyFrame();
                    }
                }
                else
                {
                    // In last frame we tracked mainly "visual odometry" points.

                    // We compute two camera poses, one from motion model and one doing relocalization.
                    // If relocalization is sucessfull we choose that solution, otherwise we retain
                    // the "visual odometry" solution.

                    bool bOKMM = false;
                    bool bOKReloc = false;
                    vector<MapPoint*> vpMPsMM;
                    vector<bool> vbOutMM;
                    cv::Mat TcwMM;
                    if(!mVelocity.empty())
                    {
                        bOKMM = TrackWithMotionModel();
                        vpMPsMM = mCurrentFrame.mvpMapPoints;
                        vbOutMM = mCurrentFrame.mvbOutlier;
                        TcwMM = mCurrentFrame.mTcw.clone();
                    }
                    bOKReloc = Relocalization();

                    if(bOKMM && !bOKReloc)
                    {
                        mCurrentFrame.SetPose(TcwMM);
                        mCurrentFrame.mvpMapPoints = vpMPsMM;
                        mCurrentFrame.mvbOutlier = vbOutMM;

                        if(mbVO)
                        {
                            for(int i =0; i<mCurrentFrame.N; i++)
                            {
                                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                                {
                                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                                }
                            }
                        }
                    }
                    else if(bOKReloc)
                    {
                        mbVO = false;
                    }

                    bOK = bOKReloc || bOKMM;
                }
            }
        }

        mCurrentFrame.mpReferenceKF = mpReferenceKF;

        // If we have an initial estimation of the camera pose and matching. Track the local map.
        if(!mbOnlyTracking)
        {
            if(bOK)
                bOK = TrackLocalMap();
        }
        else
        {
            // mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
            // a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
            // the camera we will use the local map again.
            if(bOK && !mbVO)
                bOK = TrackLocalMap();
        }

        if(bOK)
            mState = OK;
        else
            mState=LOST;

        // Update drawer
        mpFrameDrawer->Update(this);

        // If tracking were good, check if we insert a keyframe
        if(bOK)
        {
            // Update motion model
            if(!mLastFrame.mTcw.empty())
            {
                cv::Mat LastTwc = cv::Mat::eye(4,4,CV_32F);
                mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0,3).colRange(0,3));
                mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0,3).col(3));
                mVelocity = mCurrentFrame.mTcw*LastTwc;
            }
            else
                mVelocity = cv::Mat();
            
            // 加入动态信息
            if(!mCurrentFrame.objects.empty() && count(mCurrentFrame.box_status.begin(), mCurrentFrame.box_status.end(), 2)) {
                vector<dynamic> dyns;

                for(int i(0); i<mCurrentFrame.objects.size(); i++) {
                    if(mCurrentFrame.box_status[i] != 2) continue;

                    float center_x = mCurrentFrame.objects[i].x+mCurrentFrame.objects[i].width/2;
                    // cout<<"center_x "<<center_x<<", " <<mCurrentFrame.imgGray.cols*2.f/3.f<<endl;
                    if(center_x < mCurrentFrame.imgGray.cols*0.2 || center_x > mCurrentFrame.imgGray.cols*0.8) continue; //在边缘的深度不准确
                    float center_y = mCurrentFrame.objects[i].y+mCurrentFrame.objects[i].height/2;
                    float d=0;
                    float min_dist = FLT_MAX;

                    for(int j(0); j<mCurrentFrame.mvdynKeys[i].size(); j++) {
                        if(mCurrentFrame.mvdynDepth[i][j] == -1) continue;
                        // cout<<"pow(mCurrentFrame.mvdynKeys[i][j].pt.x - center_x, 2) + pow(mCurrentFrame.mvdynKeys[i][j].pt.x - center_y, 2) "<<pow(mCurrentFrame.mvdynKeys[i][j].pt.x - center_x, 2) + pow(mCurrentFrame.mvdynKeys[i][j].pt.x - center_y, 2)<<endl;
                        if(pow(mCurrentFrame.mvdynKeys[i][j].pt.x - center_x, 2) + pow(mCurrentFrame.mvdynKeys[i][j].pt.x - center_y, 2) < min_dist) {
                            min_dist = pow(mCurrentFrame.mvdynKeys[i][j].pt.x - center_x, 2) + pow(mCurrentFrame.mvdynKeys[i][j].pt.x - center_y, 2);
                            d = mCurrentFrame.mvdynDepth[i][j];
                        }
                    }
                    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ> ());
                    pcl::PointXYZ tempPoint;
                    for(int j=0; j<mCurrentFrame.mvdynKeys[i].size(); j++) {
                        if(mCurrentFrame.mvdynDepth[i][j] == -1) continue; 
                        tempPoint.z = mCurrentFrame.mvdynDepth[i][j];
                        tempPoint.x = (mCurrentFrame.mvdynKeys[i][j].pt.x - mCurrentFrame.cx) * mCurrentFrame.invfx * tempPoint.z;
                        tempPoint.y = (mCurrentFrame.mvdynKeys[i][j].pt.y - mCurrentFrame.cy) * mCurrentFrame.invfy * tempPoint.z;
                        cloud->push_back(tempPoint);
                    }
                    pcl::MomentOfInertiaEstimation<pcl::PointXYZ> feature_extractor;
                    feature_extractor.setInputCloud(cloud);
                    feature_extractor.compute();
                    pcl::PointXYZ max_point_AABB;
                    pcl::PointXYZ min_point_OBB;
                    pcl::PointXYZ max_point_OBB;
                    pcl::PointXYZ position_OBB;
                    Eigen::Matrix3f rotational_matrix_OBB;
                    feature_extractor.getOBB(min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);
                    // d = position_OBB.z;

                    int last_idx = find(mLastFrame.box_idx.begin(), mLastFrame.box_idx.end(), mCurrentFrame.box_idx[i]) - mLastFrame.box_idx.begin();

                    if(d > 20) continue;
                    int weight = 3;
                    if(mLastFrame.box_depth[last_idx] != -1) {
                        if(abs(mLastFrame.box_depth[last_idx] - d) > 5) continue;
                        d = (mLastFrame.box_depth[last_idx]  + d* (weight - 1)) / weight;
                    }

                    mCurrentFrame.box_depth[i] = d;
                    
                    float x = (mCurrentFrame.objects[i].x+mCurrentFrame.objects[i].width/2 - mCurrentFrame.cx) * mCurrentFrame.invfx * d;
                    float y = (mCurrentFrame.objects[i].y+mCurrentFrame.objects[i].height/2 - mCurrentFrame.cy) * mCurrentFrame.invfy * d;
                    // cout<<"mCurrentFrame.objects[i].x+mCurrentFrame.objects[i].width/2 " << mCurrentFrame.objects[i].x+mCurrentFrame.objects[i].width/2<<", x "<<x << ", y"<<y<<endl;

                    cv::Mat tcd = (cv::Mat_<float>(3,1) << x, y, d);
                    cv::Mat Rcd;
                    cv::eigen2cv(rotational_matrix_OBB, Rcd);
                    dynamic dyn;
                    dyn.tcd = tcd;
                    dyn.Rcd = Rcd;
                    dyn.box_id = mCurrentFrame.box_idx[i];
                    dyns.push_back(dyn);
                }
                if(dyns.size())
                    mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw, dyns);
                else
                    mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);
            }
            else
                mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

            // Clean VO matches
            for(int i=0; i<mCurrentFrame.N; i++)
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(pMP)
                    if(pMP->Observations()<1)
                    {
                        mCurrentFrame.mvbOutlier[i] = false;
                        mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                    }
            }

            // Delete temporal MapPoints
            for(list<MapPoint*>::iterator lit = mlpTemporalPoints.begin(), lend =  mlpTemporalPoints.end(); lit!=lend; lit++)
            {
                MapPoint* pMP = *lit;
                delete pMP;
            }
            mlpTemporalPoints.clear();

            // Check if we need to insert a new keyframe
            if(NeedNewKeyFrame())
                CreateNewKeyFrame();

            // We allow points with high innovation (considererd outliers by the Huber Function)
            // pass to the new keyframe, so that bundle adjustment will finally decide
            // if they are outliers or not. We don't want next frame to estimate its position
            // with those points so we discard them in the frame.
            for(int i=0; i<mCurrentFrame.N;i++)
            {
                if(mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                    mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
            }
        }

        // Reset if the camera get lost soon after initialization
        if(mState==LOST)
        {
            if(mpMap->KeyFramesInMap()<=5)
            {
                cout << "Track lost soon after initialisation, reseting..." << endl;
                mpSystem->Reset();
                return;
            }
        }

        if(!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

        mLastFrame = Frame(mCurrentFrame);
    }

    // Store frame pose information to retrieve the complete camera trajectory afterwards.
    if(!mCurrentFrame.mTcw.empty())
    {
        cv::Mat Tcr = mCurrentFrame.mTcw*mCurrentFrame.mpReferenceKF->GetPoseInverse();
        mlRelativeFramePoses.push_back(Tcr);
        mlpReferences.push_back(mpReferenceKF);
        mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
        mlbLost.push_back(mState==LOST);
    }
    else
    {
        // This can happen if tracking is lost
        mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
        mlpReferences.push_back(mlpReferences.back());
        mlFrameTimes.push_back(mlFrameTimes.back());
        mlbLost.push_back(mState==LOST);
    }
    #if YOLO_S
    if(mState == OK) {
        if(q_frame.size() >= mMaxFrames * 0.3) 
            q_frame.pop();
        // if(!mCurrentFrame.objects.empty())
        q_frame.push(mCurrentFrame);
    }
    #endif
}



// 两帧之间得到单应/基础变换矩阵====
// return 0; //track failed
// return 1; //track H;
// return 2; //track F
int Tracking::TrackHomo(cv::Mat& HorF)
{

    if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET)
    {
        cout << "Light Tracking homo not working because Tracking is not initialized..." << endl;
        return 0;
    }	
    if(mState==OK && !mVelocity.empty())//状态ok 未跟丢 且 有速度模型===
    {
        // Frame refFrameBU = *mRefFrame; // 备份上一帧=======
        // list<MapPoint*> lpTemporalPointsBU = mlpTemporalPoints;// 备份临时点====
	    ORBmatcher matcher(0.9,true);// 匹配点匹配器 最小距离 < 0.9*次短距离 匹配成功

	    mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);
	    // 当前帧位姿 mVelocity 为当前帧和上一帧的 位姿变换
        // 初始化空指针
	    fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));

	    // Project points seen in previous frame
	    int th;
	    if(mSensor  != System::STEREO)
		    th=15;// 搜索窗口
	    else
		    th=7;
        vector<cv::Point2f> points_last;
        vector<cv::Point2f> points_current;
	    int nmatches = matcher.SearchByProjection(mCurrentFrame,*mRefFrame,th,mSensor==System::MONOCULAR,
                                                      points_last,points_current);

	    // If few matches, uses a wider window search
	    if(nmatches<20)
	    {
            points_last.clear();
            points_current.clear();
            fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));
            nmatches = matcher.SearchByProjection(mCurrentFrame,*mRefFrame,2*th,mSensor==System::MONOCULAR,
                                                      points_last,points_current);
	    }
        // 回复原来的帧，不改变=====================
        // mRefFrame = &refFrameBU;
        // mlpTemporalPoints = lpTemporalPointsBU;
	    if(nmatches<20)
        {
            cout << "Too few matches, Tracking homo faild..." << endl;
            return 0;
        }
        // if(points_current.size() > 50)
        else
        {
            // cv::Mat cur = mCurrentFrame.imgGray.clone(), last = mRefFrame->imgGray.clone();
            // for(int i(0); i<points_current.size(); i++) {
            //     cv::drawMarker(cur, points_current[i], cv::Scalar(255, 0, 0));
            //     cv::drawMarker(last, points_last[i], cv::Scalar(255, 0, 0));
            // }
            // cv::imshow("cur", cur);
            // cv::imshow("last", last);
            
            cv::Mat H, F;
            vector<uchar> inliers_H(points_last.size()), inliers_F(points_last.size());
            H = findHomography ( points_last, points_current, CV_RANSAC, 3, inliers_H);// points_current = H21 * points_last
            F = findFundamentalMat (points_last, points_current, CV_RANSAC, 3, 0.99, inliers_F);
            // 类型转换成float，与后续函数参数对应
            H.convertTo(H, CV_32F);
            F.convertTo(F, CV_32F);
            
            vector<cv::KeyPoint> cur_ks, last_ks;
            vector<cv::DMatch> matches;
            cv::KeyPoint cur_k, last_k;
            cv::DMatch match;
            for(int i(0); i<inliers_F.size(); i++) {
                if(inliers_F[i] == 1) {
                    cur_k.pt = points_current[i];
                    last_k.pt = points_last[i];
                    cur_ks.push_back(cur_k);
                    last_ks.push_back(last_k);

                    match.queryIdx = cur_ks.size() - 1;
                    match.trainIdx = cur_ks.size() - 1;
                    matches.push_back(match);
                }
            }
            // cv::Mat cur = mCurrentFrame.imgGray.clone(), last = mRefFrame->imgGray.clone(), out;
            // cv::drawMatches(cur, cur_ks, last, last_ks, matches, out);
            // cv::imshow("out", out);
            // cv::waitKey(0);

            // float s_H, s_F;
            // CheckHF(H, F, points_last, points_current, 1.0, s_H, s_F);
            // if(s_H+s_F == 0.f) {
            //     cout << "CheckHF failed" << endl;
            //     return 0;
            // }
            // float RH = s_H/(s_H+s_F);
            // if(RH > 0.5) cout << "checkH";
            // else cout << "checkF ";

            int n_H = count(inliers_H.begin(), inliers_H.end(), 1);
            int n_F = count(inliers_F.begin(), inliers_F.end(), 1);
            // cout<<"n_H: " << n_H <<", n_F "<<n_F << " of background kpts "<<inliers_F.size()<<endl;
            if(n_F > 10 || n_H > 10) {
                if(n_H > n_F) {
                    HorF = H;
                    return 1;
                }
                else {
                    HorF = F;
                    return 2;
                }
            }
            else {
                cout << "CheckHF failed!" << endl;
            }

        }
    }
    cout << "Tracking homo faild..." << endl;
    return 0;
}

// 筛选出假动态点
// dynStatus[i]为1表示光流能跟上
// return 0; //放弃动态点
// return 1; //动态区域中静态点占多数，可以加入
// return 2; //动态点大部分处于相同运动状态（整体移动）
int Tracking::Separate(cv::Mat HorF, int flag, vector<vector<int>> &dynStatus) {
    
    // cv::Ptr<cv::DescriptorMatcher> matcher_  = cv::DescriptorMatcher::create ( "BruteForce-Hamming" );
    cv::BFMatcher matcherBF(cv::NORM_HAMMING, true);
    vector<cv::DMatch> good_matches;
    dynStatus.resize(mCurrentFrame.objects.size());
    // cout<<mRefFrame->mdynDescriptors.rows<<"mdynDescriptors"<<endl;

    // // 当前帧 旋转 平移矩阵	    
    // const cv::Mat Rcw = mCurrentFrame.mTcw.rowRange(0,3).colRange(0,3);
    // const cv::Mat tcw = mCurrentFrame.mTcw.rowRange(0,3).col(3);
    // const cv::Mat twc = -Rcw.t()*tcw;// // twc(w)
    // // 参考帧 旋转 平移矩阵
    // const cv::Mat Rlw = mRefFrame->mTcw.rowRange(0,3).colRange(0,3);
    // const cv::Mat tlw = mRefFrame->mTcw.rowRange(0,3).col(3);
    // const cv::Mat tlc = Rlw*twc + tlw;//当前帧到上一帧的 平移向量
    // const cv::Mat Rlc = Rlw*Rcw.t(); //当前帧到上一帧的 旋转向量

    bool static_exit = false;

    // 对每一个box_match 进行筛选
    for(int n_box(0); n_box<mCurrentFrame.objects.size(); n_box++) {

        if(!count(mRefFrame->box_idx.begin(), mRefFrame->box_idx.end(), mCurrentFrame.box_idx[n_box]))
            continue; 
        int ref_idx = find(mRefFrame->box_idx.begin(), mRefFrame->box_idx.end(), mCurrentFrame.box_idx[n_box]) - mRefFrame->box_idx.begin();
        // cv::drawKeypoints(cur, mCurrentFrame.mvdynKeys[n_box], cur);
        // cv::drawKeypoints(ref, mRefFrame->mvdynKeys[ref_idx], ref);
        if(mCurrentFrame.mdynDescriptors[n_box].cols == 0 || mRefFrame->mdynDescriptors[ref_idx].cols == 0) continue;
        matcherBF.match(mCurrentFrame.mdynDescriptors[n_box], mRefFrame->mdynDescriptors[ref_idx], good_matches);
        cout<< "matched "<<good_matches.size() <<" of " << mCurrentFrame.mvdynKeys[n_box].size() <<" in box " <<mCurrentFrame.box_idx[n_box] <<endl;
        // 匹配数量太少(太远不考虑)则无法进行后续处理
        if(good_matches.size() < 3 || good_matches.size() < 0.2*mCurrentFrame.mvdynKeys[n_box].size()) 
            continue;

        vector<int> &dynStatus_ = dynStatus[n_box];
        dynStatus_ = vector<int>(good_matches.size(), -1);
        if(flag == 1)
            classifyH(HorF, mCurrentFrame.mvdynKeysUn[n_box], mRefFrame->mvdynKeysUn[ref_idx], good_matches, dynStatus_);
        else
            classifyF(HorF, mCurrentFrame.mvdynKeysUn[n_box], mRefFrame->mvdynKeysUn[ref_idx], good_matches, dynStatus_);

        int num0 = dynStatus_.size() - count(dynStatus_.begin(), dynStatus_.end(), -1);
        cout << "after classify, "<<num0<<" matches left."<<endl;
                    // 画图用于调试
        cv::Mat out;
        vector<cv::DMatch> temp_matches;
        for(int i=0; i<good_matches.size(); i++) {
            if(dynStatus_[i] == -1) continue;
            temp_matches.push_back(good_matches[i]);
        }

        cv::drawMatches(mCurrentFrame.imgRGB, mCurrentFrame.mvdynKeys[n_box], mRefFrame->imgRGB, 
                        mRefFrame->mvdynKeys[ref_idx], temp_matches, out);
        // cv::drawMatches(mCurrentFrame.imgGray, mCurrentFrame.mvdynKeys[n_box], mRefFrame->imgGray, mRefFrame->mvdynKeys[ref_idx], good_matches, out);
        // cv::imshow("match" , out);
        cv::imwrite("/home/hai/img_/" + to_string(mCurrentFrame.mnId)+"_"+to_string(n_box) + ".png", out);
        int last_idx = find(mLastFrame.box_idx.begin(), mLastFrame.box_idx.end(), mCurrentFrame.box_idx[n_box]) - mLastFrame.box_idx.begin();
        // 包括完全静止和局部运动（没有整体移动）两种情况
        if(num0 > max((double)1, 0.2*good_matches.size())) {

            // 利用深度信息进行进一步过滤
            /*
            int filtered_num = 0;
            for(int i=0; i<good_matches.size(); i++) {
                if(dynStatus_[i] == -1) continue;
                //只对前一步判断为静态点的进行进一步判断
                const float zl = mRefFrame->mvdynDepth[ref_idx][good_matches[i].trainIdx];
                const float zc = mCurrentFrame.mvdynDepth[n_box][good_matches[i].queryIdx];
                // 深度有效
                // 把当期帧的空间点变换到参考帧，与其真实深度比较
                
                if(zl>0 && zc>0) { 
                    const float u = mCurrentFrame.mvdynKeys[n_box][good_matches[i].queryIdx].pt.x;
                    const float v = mCurrentFrame.mvdynKeys[n_box][good_matches[i].queryIdx].pt.y;
                    const float x = (u-mCurrentFrame.cx)*zc*mCurrentFrame.invfx;
                    const float y = (v-mCurrentFrame.cy)*zc*mCurrentFrame.invfy;
                    cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x,y,zc);

                    cv::Mat x3Dl = Rlc*x3Dc + tlc;
                    const float zl_ = x3Dl.at<float>(2);
                    // cout<<"zl_"<<zl_<<" zl:"<<zl<<endl;
                    if(fabs(zl - zl_) > 0.1) {
                        dynStatus_[i] = -1;
                        filtered_num++;
                    }
                }
                // 深度无效也认为是动态点
                // else {
                //     dynStatus_[i] = -1;
                // }
            }
            int num = dynStatus_.size() - count(dynStatus_.begin(), dynStatus_.end(), -1);
            cout<<"using Depth constrtaint filters "<< filtered_num << " matches."<<endl;
            */
            mCurrentFrame.box_status[n_box] == 1;

            static_exit = true;
            cv::putText(mCurrentFrame.imgRGB, "Static"+to_string(mCurrentFrame.box_idx[n_box]),
                        cv::Point(mCurrentFrame.objects[n_box].x, mCurrentFrame.objects[n_box].y),
                        2, 0.7, cv::Scalar(0,255,255), 2);
            // continue;
            

            // // 画图用于调试
            // cv::Mat out;
            // vector<cv::DMatch> temp_matches;
            // for(int i=0; i<good_matches.size(); i++) {
            //     if(dynStatus_[i] == -1) continue;
            //     temp_matches.push_back(good_matches[i]);
            // }

            // cv::drawMatches(mCurrentFrame.imgGray, mCurrentFrame.mvdynKeys[n_box], mRefFrame->imgGray, mRefFrame->mvdynKeys[ref_idx], temp_matches, out);
            // // cv::drawMatches(mCurrentFrame.imgGray, mCurrentFrame.mvdynKeys[n_box], mRefFrame->imgGray, mRefFrame->mvdynKeys[ref_idx], good_matches, out);
            // cv::imshow("match" , out);
            // cv::imwrite("/home/hai/img_/" + to_string(mCurrentFrame.mnId)+"_"+to_string(n_box) + ".png", out);
        }

        else { 
            // 连续两帧是动态才认为是动态， 避免框直接赋值过来而误当做动态物体的情况, 第一次dyn:0, 第二次：2
            if(mLastFrame.box_status[last_idx] == 0 || mLastFrame.box_status[last_idx] == 2) {
                cout<<"dyn"<<endl;
                mCurrentFrame.box_status[n_box] = 2;
                cv::putText(mCurrentFrame.imgRGB, "Dynamic"+to_string(mCurrentFrame.box_idx[n_box]), 
                    cv::Point(mCurrentFrame.objects[n_box].x, mCurrentFrame.objects[n_box].y), 2, 0.7, cv::Scalar(255,0, 255), 2);
            }
            // else if(mLastFrame.box_status[last_idx] == 0) 
            //     mCurrentFrame.box_status[n_box] = 2;
            else {
                cv::putText(mCurrentFrame.imgRGB, "d" + to_string(mCurrentFrame.box_idx[n_box]),
                            cv::Point(mCurrentFrame.objects[n_box].x, mCurrentFrame.objects[n_box].y),
                            2, 0.7, cv::Scalar(0,255,255), 2);
                mCurrentFrame.box_status[n_box] = 0;
            }
        }
    }
    // cv::imwrite("/home/hai/img/all/" + to_string(mCurrentFrame.mnId) + ".png", mCurrentFrame.imgGray);

    // cv::Mat l_r;
    // cv::vconcat(mCurrentFrame.imgGray, mRefFrame->imgGray, l_r);
    // cv::imshow("cur_ref", l_r);
    // cv::waitKey();

    if(static_exit) return 1;

    return 0;
}

void Tracking::classifyH(const cv::Mat &H21, const vector<cv::KeyPoint> &cur_kpts, const vector<cv::KeyPoint> &ref_kpts, vector<cv::DMatch> &matches, vector<int> &falseDyn) {

    const float sigma = 1.0;

    const float h11 = H21.at<float>(0,0);
    const float h12 = H21.at<float>(0,1);
    const float h13 = H21.at<float>(0,2);
    const float h21 = H21.at<float>(1,0);
    const float h22 = H21.at<float>(1,1);
    const float h23 = H21.at<float>(1,2);
    const float h31 = H21.at<float>(2,0);
    const float h32 = H21.at<float>(2,1);
    const float h33 = H21.at<float>(2,2);

	// 获取从当前帧到参考帧的单应矩阵的各个元素
    cv::Mat H12 = H21.inv();
    const float h11inv = H12.at<float>(0,0);
    const float h12inv = H12.at<float>(0,1);
    const float h13inv = H12.at<float>(0,2);
    const float h21inv = H12.at<float>(1,0);
    const float h22inv = H12.at<float>(1,1);
    const float h23inv = H12.at<float>(1,2);
    const float h31inv = H12.at<float>(2,0);
    const float h32inv = H12.at<float>(2,1);
    const float h33inv = H12.at<float>(2,2);

    const float th = 5.991;

    //信息矩阵，方差平方的倒数
    const float invSigmaSquare = 1.0/(sigma*sigma);

    // Step 2 通过H矩阵，进行参考帧和当前帧之间的双向投影，并计算起加权最小二乘投影误差
    for(int i=0; i<matches.size(); i++)
    {
		// Step 2.1 参考帧和当前帧之间的潜在动态点
        const float u1 = ref_kpts[matches[i].trainIdx].pt.x;
        const float v1 = ref_kpts[matches[i].trainIdx].pt.y;
        const float u2 = cur_kpts[matches[i].queryIdx].pt.x;
        const float v2 = cur_kpts[matches[i].queryIdx].pt.y;

        // Step 2.2 计算 img2 到 img1 的重投影误差
        // x2in1 = H12*x2
        // 将图像2中的特征点单应到图像1中
        // |u1|   |h11inv h12inv h13inv||u2|   |u2in1|
        // |v1| = |h21inv h22inv h23inv||v2| = |v2in1| * w2in1inv
        // |1 |   |h31inv h32inv h33inv||1 |   |  1  |
		// 计算投影归一化坐标
        const float w2in1inv = 1.0/(h31inv*u2+h32inv*v2+h33inv);
        const float u2in1 = (h11inv*u2+h12inv*v2+h13inv)*w2in1inv;
        const float v2in1 = (h21inv*u2+h22inv*v2+h23inv)*w2in1inv;
        // 计算重投影误差 = ||p1(i) - H12 * p2(i)||2
        float squareDist1 = (u1-u2in1)*(u1-u2in1)+(v1-v2in1)*(v1-v2in1);
        float chiSquare1 = squareDist1*invSigmaSquare;

        // 计算从img1 到 img2 的投影变换误差
        // x1in2 = H21*x1
        const float w1in2inv = 1.0/(h31*u1+h32*v1+h33);
        const float u1in2 = (h11*u1+h12*v1+h13)*w1in2inv;
        const float v1in2 = (h21*u1+h22*v1+h23)*w1in2inv;
        // 计算重投影误差 
        float squareDist2 = (u2-u1in2)*(u2-u1in2)+(v2-v1in2)*(v2-v1in2);
        float chiSquare2 = squareDist2*invSigmaSquare;
        // 内点的话表示为假动态点（静态点）
        if(chiSquare2<=th && chiSquare1<=th) {
            // cout << chiSquare1 << "," << chiSquare2 << "  ";
            falseDyn[i] = matches[i].queryIdx;
        }
    }
}

void Tracking::classifyF(const cv::Mat &F21, const vector<cv::KeyPoint> &cur_kpts, const vector<cv::KeyPoint> &ref_kpts, vector<cv::DMatch> &matches, vector<int> &falseDyn) {

    const float sigma = 1.0;

        // 提取基础矩阵中的元素数据
    const float f11 = F21.at<float>(0,0);
    const float f12 = F21.at<float>(0,1);
    const float f13 = F21.at<float>(0,2);
    const float f21 = F21.at<float>(1,0);
    const float f22 = F21.at<float>(1,1);
    const float f23 = F21.at<float>(1,2);
    const float f31 = F21.at<float>(2,0);
    const float f32 = F21.at<float>(2,1);
    const float f33 = F21.at<float>(2,2);

    const float th_F = 5.841;
    //信息矩阵，方差平方的倒数
    const float invSigmaSquare = 1.0/(sigma*sigma);

    // Step 2 通过H矩阵，进行参考帧和当前帧之间的双向投影，并计算起加权最小二乘投影误差
    for(int i=0; i<matches.size(); i++)
    {
        // Step 2.1 参考帧和当前帧之间的潜在动态点
        const float u1 = ref_kpts[matches[i].trainIdx].pt.x;
        const float v1 = ref_kpts[matches[i].trainIdx].pt.y;
        const float u2 = cur_kpts[matches[i].queryIdx].pt.x;
        const float v2 = cur_kpts[matches[i].queryIdx].pt.y;

        const float a2 = f11*u1+f12*v1+f13;
		const float b2 = f21*u1+f22*v1+f23;
		const float c2 = f31*u1+f32*v1+f33;
		
        // x2应该在l这条线上:x2点乘l = 0 
		// 计算x2特征点到 极线 的距离：
		// 极线l：ax + by + c = 0
		// (u,v)到l的距离为：d = |au+bv+c| / sqrt(a^2+b^2) 
		// d^2 = |au+bv+c|^2/(a^2+b^2)
		const float num2 = a2*u2+b2*v2+c2;
		const float squareDist1 = num2*num2/(a2*a2+b2*b2);// 点到线的几何距离 的平方
		// 根据方差归一化误差
		const float chiSquare1 = squareDist1*invSigmaSquare;

		// Reprojection error in second image
		// l1 =x2转置 × F21=(a1,b1,c1)
        //  p2 ------> p1 误差 得分-------------------------
		const float a1 = f11*u2+f21*v2+f31;
		const float b1 = f12*u2+f22*v2+f32;
		const float c1 = f13*u2+f23*v2+f33;
		const float num1 = a1*u1+b1*v1+c1;
		const float squareDist2 = num1*num1/(a1*a1+b1*b1);
		const float chiSquare2 = squareDist2*invSigmaSquare;
		if(chiSquare1 <= th_F && chiSquare2 <= th_F) {
            // cout << chiSquare1 << "," << chiSquare2 << "  ";
            falseDyn[i] = matches[i].queryIdx;
        }
    }
}


void Tracking::StereoInitialization()
{
    if(mCurrentFrame.N>200) //ori:500
    {
        // Set Frame pose to the origin
        mCurrentFrame.SetPose(cv::Mat::eye(4,4,CV_32F));

        // Create KeyFrame
        KeyFrame* pKFini = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

        // Insert KeyFrame in the map
        mpMap->AddKeyFrame(pKFini);

        // Create MapPoints and asscoiate to KeyFrame
        for(int i=0; i<mCurrentFrame.N;i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                MapPoint* pNewMP = new MapPoint(x3D,pKFini,mpMap);
                pNewMP->AddObservation(pKFini,i);
                pKFini->AddMapPoint(pNewMP,i);
                pNewMP->ComputeDistinctiveDescriptors();
                pNewMP->UpdateNormalAndDepth();
                mpMap->AddMapPoint(pNewMP);

                mCurrentFrame.mvpMapPoints[i]=pNewMP;
            }
        }

        cout << "New map created with " << mpMap->MapPointsInMap() << " points" << endl;

        mpLocalMapper->InsertKeyFrame(pKFini);

        mLastFrame = Frame(mCurrentFrame);
        mnLastKeyFrameId=mCurrentFrame.mnId;
        mpLastKeyFrame = pKFini;

        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints=mpMap->GetAllMapPoints();
        mpReferenceKF = pKFini;
        mCurrentFrame.mpReferenceKF = pKFini;

        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

        mpMap->mvpKeyFrameOrigins.push_back(pKFini);

        mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

        mState=OK;
    }
}

void Tracking::MonocularInitialization()
{

    if(!mpInitializer)
    {
        // Set Reference Frame
        if(mCurrentFrame.mvKeys.size()>100)
        {
            mInitialFrame = Frame(mCurrentFrame);
            mLastFrame = Frame(mCurrentFrame);
            mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
            for(size_t i=0; i<mCurrentFrame.mvKeysUn.size(); i++)
                mvbPrevMatched[i]=mCurrentFrame.mvKeysUn[i].pt;

            if(mpInitializer)
                delete mpInitializer;

            mpInitializer =  new Initializer(mCurrentFrame,1.0,200);

            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);

            return;
        }
    }
    else
    {
        // Try to initialize
        if((int)mCurrentFrame.mvKeys.size()<=100)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);
            return;
        }

        // Find correspondences
        ORBmatcher matcher(0.9,true);
        int nmatches = matcher.SearchForInitialization(mInitialFrame,mCurrentFrame,mvbPrevMatched,mvIniMatches,100);

        // Check if there are enough correspondences
        if(nmatches<100)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            return;
        }

        cv::Mat Rcw; // Current Camera Rotation
        cv::Mat tcw; // Current Camera Translation
        vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)

        if(mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated))
        {
            for(size_t i=0, iend=mvIniMatches.size(); i<iend;i++)
            {
                if(mvIniMatches[i]>=0 && !vbTriangulated[i])
                {
                    mvIniMatches[i]=-1;
                    nmatches--;
                }
            }

            // Set Frame Poses
            mInitialFrame.SetPose(cv::Mat::eye(4,4,CV_32F));
            cv::Mat Tcw = cv::Mat::eye(4,4,CV_32F);
            Rcw.copyTo(Tcw.rowRange(0,3).colRange(0,3));
            tcw.copyTo(Tcw.rowRange(0,3).col(3));
            mCurrentFrame.SetPose(Tcw);

            CreateInitialMapMonocular();
        }
    }
}

void Tracking::CreateInitialMapMonocular()
{
    // Create KeyFrames
    KeyFrame* pKFini = new KeyFrame(mInitialFrame,mpMap,mpKeyFrameDB);
    KeyFrame* pKFcur = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);


    pKFini->ComputeBoW();
    pKFcur->ComputeBoW();

    // Insert KFs in the map
    mpMap->AddKeyFrame(pKFini);
    mpMap->AddKeyFrame(pKFcur);

    // Create MapPoints and asscoiate to keyframes
    for(size_t i=0; i<mvIniMatches.size();i++)
    {
        if(mvIniMatches[i]<0)
            continue;

        //Create MapPoint.
        cv::Mat worldPos(mvIniP3D[i]);

        MapPoint* pMP = new MapPoint(worldPos,pKFcur,mpMap);

        pKFini->AddMapPoint(pMP,i);
        pKFcur->AddMapPoint(pMP,mvIniMatches[i]);

        pMP->AddObservation(pKFini,i);
        pMP->AddObservation(pKFcur,mvIniMatches[i]);

        pMP->ComputeDistinctiveDescriptors();
        pMP->UpdateNormalAndDepth();

        //Fill Current Frame structure
        mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
        mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

        //Add to Map
        mpMap->AddMapPoint(pMP);
    }

    // Update Connections
    pKFini->UpdateConnections();
    pKFcur->UpdateConnections();

    // Bundle Adjustment
    cout << "New Map created with " << mpMap->MapPointsInMap() << " points" << endl;

    Optimizer::GlobalBundleAdjustemnt(mpMap,20);

    // Set median depth to 1
    float medianDepth = pKFini->ComputeSceneMedianDepth(2);
    float invMedianDepth = 1.0f/medianDepth;

    if(medianDepth<0 || pKFcur->TrackedMapPoints(1)<100)
    {
        cout << "Wrong initialization, reseting..." << endl;
        Reset();
        return;
    }

    // Scale initial baseline
    cv::Mat Tc2w = pKFcur->GetPose();
    Tc2w.col(3).rowRange(0,3) = Tc2w.col(3).rowRange(0,3)*invMedianDepth;
    pKFcur->SetPose(Tc2w);

    // Scale points
    vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
    for(size_t iMP=0; iMP<vpAllMapPoints.size(); iMP++)
    {
        if(vpAllMapPoints[iMP])
        {
            MapPoint* pMP = vpAllMapPoints[iMP];
            pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth);
        }
    }

    mpLocalMapper->InsertKeyFrame(pKFini);
    mpLocalMapper->InsertKeyFrame(pKFcur);

    mCurrentFrame.SetPose(pKFcur->GetPose());
    mnLastKeyFrameId=mCurrentFrame.mnId;
    mpLastKeyFrame = pKFcur;

    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints=mpMap->GetAllMapPoints();
    mpReferenceKF = pKFcur;
    mCurrentFrame.mpReferenceKF = pKFcur;

    mLastFrame = Frame(mCurrentFrame);

    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

    mpMap->mvpKeyFrameOrigins.push_back(pKFini);

    mState=OK;
}

void Tracking::CheckReplacedInLastFrame()
{
    for(int i =0; i<mLastFrame.N; i++)
    {
        MapPoint* pMP = mLastFrame.mvpMapPoints[i];

        if(pMP)
        {
            MapPoint* pRep = pMP->GetReplaced();
            if(pRep)
            {
                mLastFrame.mvpMapPoints[i] = pRep;
            }
        }
    }
}


bool Tracking::TrackReferenceKeyFrame()
{
    // Compute Bag of Words vector
    mCurrentFrame.ComputeBoW();

    // We perform first an ORB matching with the reference keyframe
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.7,true);
    vector<MapPoint*> vpMapPointMatches;

    int nmatches = matcher.SearchByBoW(mpReferenceKF,mCurrentFrame,vpMapPointMatches);

    if(nmatches<15)
        return false;

    mCurrentFrame.mvpMapPoints = vpMapPointMatches;
    mCurrentFrame.SetPose(mLastFrame.mTcw);

    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }

    return nmatchesMap>=10;
}

void Tracking::UpdateLastFrame()
{
    // Update pose according to reference keyframe
    KeyFrame* pRef = mLastFrame.mpReferenceKF;
    cv::Mat Tlr = mlRelativeFramePoses.back();

    mLastFrame.SetPose(Tlr*pRef->GetPose());

    if(mnLastKeyFrameId==mLastFrame.mnId || mSensor==System::MONOCULAR || !mbOnlyTracking)
        return;

    // Create "visual odometry" MapPoints
    // We sort points according to their measured depth by the stereo/RGB-D sensor
    vector<pair<float,int> > vDepthIdx;
    vDepthIdx.reserve(mLastFrame.N);
    for(int i=0; i<mLastFrame.N;i++)
    {
        float z = mLastFrame.mvDepth[i];
        if(z>0)
        {
            vDepthIdx.push_back(make_pair(z,i));
        }
    }

    if(vDepthIdx.empty())
        return;

    sort(vDepthIdx.begin(),vDepthIdx.end());

    // We insert all close points (depth<mThDepth)
    // If less than 100 close points, we insert the 100 closest ones.
    int nPoints = 0;
    for(size_t j=0; j<vDepthIdx.size();j++)
    {
        int i = vDepthIdx[j].second;

        bool bCreateNew = false;

        MapPoint* pMP = mLastFrame.mvpMapPoints[i];
        if(!pMP)
            bCreateNew = true;
        else if(pMP->Observations()<1)
        {
            bCreateNew = true;
        }

        if(bCreateNew)
        {
            cv::Mat x3D = mLastFrame.UnprojectStereo(i);
            MapPoint* pNewMP = new MapPoint(x3D,mpMap,&mLastFrame,i);

            mLastFrame.mvpMapPoints[i]=pNewMP;

            mlpTemporalPoints.push_back(pNewMP);
            nPoints++;
        }
        else
        {
            nPoints++;
        }

        if(vDepthIdx[j].first>mThDepth && nPoints>100)
            break;
    }
}

bool Tracking::TrackWithMotionModel()
{
    ORBmatcher matcher(0.9,true);

    // Update last frame pose according to its reference keyframe
    // Create "visual odometry" points if in Localization Mode
    UpdateLastFrame();

    mCurrentFrame.SetPose(mVelocity*mLastFrame.mTcw);

    fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));

    // Project points seen in previous frame
    int th;
    if(mSensor!=System::STEREO)
        th=15;
    else
        th=7;
    int nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,th,mSensor==System::MONOCULAR);

    // If few matches, uses a wider window search
    if(nmatches<20)
    {
        fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));
        nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,2*th,mSensor==System::MONOCULAR);
    }

    if(nmatches<20)
        return false;

    // Optimize frame pose with all matches
    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--;
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }    

    if(mbOnlyTracking)
    {
        mbVO = nmatchesMap<10;
        return nmatches>20;
    }

    return nmatchesMap>=10;
}

bool Tracking::TrackLocalMap()
{
    // We have an estimation of the camera pose and some map points tracked in the frame.
    // We retrieve the local map and try to find matches to points in the local map.

    UpdateLocalMap();

    SearchLocalPoints();

    // Optimize Pose
    Optimizer::PoseOptimization(&mCurrentFrame);
    mnMatchesInliers = 0;

    // Update MapPoints Statistics
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(!mCurrentFrame.mvbOutlier[i])
            {
                mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                if(!mbOnlyTracking)
                {
                    if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                        mnMatchesInliers++;
                }
                else
                    mnMatchesInliers++;
            }
            else if(mSensor==System::STEREO)
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);

        }
    }

    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && mnMatchesInliers<50)
        return false;

    if(mnMatchesInliers<30)
        return false;
    else
        return true;
}


bool Tracking::NeedNewKeyFrame()
{
    if(mbOnlyTracking)
        return false;

    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
        return false;

    const int nKFs = mpMap->KeyFramesInMap();

    // Do not insert keyframes if not enough frames have passed from last relocalisation
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && nKFs>mMaxFrames)
        return false;

    // Tracked MapPoints in the reference keyframe
    int nMinObs = 3;
    if(nKFs<=2)
        nMinObs=2;
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

    // Local Mapping accept keyframes?
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

    // Check how many "close" points are being tracked and how many could be potentially created.
    int nNonTrackedClose = 0;
    int nTrackedClose= 0;
    if(mSensor!=System::MONOCULAR)
    {
        for(int i =0; i<mCurrentFrame.N; i++)
        {
            if(mCurrentFrame.mvDepth[i]>0 && mCurrentFrame.mvDepth[i]<mThDepth)
            {
                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                    nTrackedClose++;
                else
                    nNonTrackedClose++;
            }
        }
    }

    bool bNeedToInsertClose = (nTrackedClose<100) && (nNonTrackedClose>70);

    // Thresholds
    float thRefRatio = 0.75f;
    if(nKFs<2)
        thRefRatio = 0.4f;

    if(mSensor==System::MONOCULAR)
        thRefRatio = 0.9f;

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    const bool c1a = mCurrentFrame.mnId>=mnLastKeyFrameId+mMaxFrames;
    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    const bool c1b = (mCurrentFrame.mnId>=mnLastKeyFrameId+mMinFrames && bLocalMappingIdle);
    //Condition 1c: tracking is weak
    const bool c1c =  mSensor!=System::MONOCULAR && (mnMatchesInliers<nRefMatches*0.25 || bNeedToInsertClose) ;
    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    const bool c2 = ((mnMatchesInliers<nRefMatches*thRefRatio|| bNeedToInsertClose) && mnMatchesInliers>15);

    if((c1a||c1b||c1c)&&c2)
    {
        // If the mapping accepts keyframes, insert keyframe.
        // Otherwise send a signal to interrupt BA
        if(bLocalMappingIdle)
        {
            return true;
        }
        else
        {
            mpLocalMapper->InterruptBA();
            if(mSensor!=System::MONOCULAR)
            {
                if(mpLocalMapper->KeyframesInQueue()<3)
                    return true;
                else
                    return false;
            }
            else
                return false;
        }
    }
    else
        return false;
}

void Tracking::CreateNewKeyFrame()
{
    if(!mpLocalMapper->SetNotStop(true))
        return;

    KeyFrame* pKF = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);

    mpReferenceKF = pKF;
    mCurrentFrame.mpReferenceKF = pKF;

    if(mSensor!=System::MONOCULAR)
    {
        mCurrentFrame.UpdatePoseMatrices();

        // We sort points by the measured depth by the stereo/RGBD sensor.
        // We create all those MapPoints whose depth < mThDepth.
        // If there are less than 100 close points we create the 100 closest.
        vector<pair<float,int> > vDepthIdx;
        vDepthIdx.reserve(mCurrentFrame.N);
        for(int i=0; i<mCurrentFrame.N; i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                vDepthIdx.push_back(make_pair(z,i));
            }
        }

        if(!vDepthIdx.empty())
        {
            sort(vDepthIdx.begin(),vDepthIdx.end());

            int nPoints = 0;
            for(size_t j=0; j<vDepthIdx.size();j++)
            {
                int i = vDepthIdx[j].second;

                bool bCreateNew = false;

                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(!pMP)
                    bCreateNew = true;
                else if(pMP->Observations()<1)
                {
                    bCreateNew = true;
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
                }

                if(bCreateNew)
                {
                    cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                    MapPoint* pNewMP = new MapPoint(x3D,pKF,mpMap);
                    pNewMP->AddObservation(pKF,i);
                    pKF->AddMapPoint(pNewMP,i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpMap->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;
                    nPoints++;
                }
                else
                {
                    nPoints++;
                }

                if(vDepthIdx[j].first>mThDepth && nPoints>100)
                    break;
            }
        }
    }

    mpLocalMapper->InsertKeyFrame(pKF);

    mpLocalMapper->SetNotStop(false);

    // filter dynamic objects before cloudmapping
    vector<cv::Rect2d> dyn_boxes;
    for(int i=0; i<mCurrentFrame.objects.size(); i++) {
        if(mCurrentFrame.box_status[i] == 2 || mCurrentFrame.box_status[i] == 0) {
            dyn_boxes.push_back(mCurrentFrame.objects[i]);
        }
    }
    mpPointCloudMapping->insertKeyFrame( pKF, this->mImRGB, this->imDepth, this->mImMask, dyn_boxes);


    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKF;
}

void Tracking::SearchLocalPoints()
{
    // Do not search map points already matched
    for(vector<MapPoint*>::iterator vit=mCurrentFrame.mvpMapPoints.begin(), vend=mCurrentFrame.mvpMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP)
        {
            if(pMP->isBad())
            {
                *vit = static_cast<MapPoint*>(NULL);
            }
            else
            {
                pMP->IncreaseVisible();
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                pMP->mbTrackInView = false;
            }
        }
    }

    int nToMatch=0;

    // Project points in frame and check its visibility
    for(vector<MapPoint*>::iterator vit=mvpLocalMapPoints.begin(), vend=mvpLocalMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP->mnLastFrameSeen == mCurrentFrame.mnId)
            continue;
        if(pMP->isBad())
            continue;
        // Project (this fills MapPoint variables for matching)
        if(mCurrentFrame.isInFrustum(pMP,0.5))
        {
            pMP->IncreaseVisible();
            nToMatch++;
        }
    }

    if(nToMatch>0)
    {
        ORBmatcher matcher(0.8);
        int th = 1;
        if(mSensor==System::RGBD)
            th=3;
        // If the camera has been relocalised recently, perform a coarser search
        if(mCurrentFrame.mnId<mnLastRelocFrameId+2)
            th=5;
        matcher.SearchByProjection(mCurrentFrame,mvpLocalMapPoints,th);
    }
}

void Tracking::UpdateLocalMap()
{
    // This is for visualization
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    // Update
    UpdateLocalKeyFrames();
    UpdateLocalPoints();
}

void Tracking::UpdateLocalPoints()
{
    mvpLocalMapPoints.clear();

    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        KeyFrame* pKF = *itKF;
        const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

        for(vector<MapPoint*>::const_iterator itMP=vpMPs.begin(), itEndMP=vpMPs.end(); itMP!=itEndMP; itMP++)
        {
            MapPoint* pMP = *itMP;
            if(!pMP)
                continue;
            if(pMP->mnTrackReferenceForFrame==mCurrentFrame.mnId)
                continue;
            if(!pMP->isBad())
            {
                mvpLocalMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame=mCurrentFrame.mnId;
            }
        }
    }
}


void Tracking::UpdateLocalKeyFrames()
{
    // Each map point vote for the keyframes in which it has been observed
    map<KeyFrame*,int> keyframeCounter;
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if(!pMP->isBad())
            {
                const map<KeyFrame*,size_t> observations = pMP->GetObservations();
                for(map<KeyFrame*,size_t>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                    keyframeCounter[it->first]++;
            }
            else
            {
                mCurrentFrame.mvpMapPoints[i]=NULL;
            }
        }
    }

    if(keyframeCounter.empty())
        return;

    int max=0;
    KeyFrame* pKFmax= static_cast<KeyFrame*>(NULL);

    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.reserve(3*keyframeCounter.size());

    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    for(map<KeyFrame*,int>::const_iterator it=keyframeCounter.begin(), itEnd=keyframeCounter.end(); it!=itEnd; it++)
    {
        KeyFrame* pKF = it->first;

        if(pKF->isBad())
            continue;

        if(it->second>max)
        {
            max=it->second;
            pKFmax=pKF;
        }

        mvpLocalKeyFrames.push_back(it->first);
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
    }


    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        // Limit the number of keyframes
        if(mvpLocalKeyFrames.size()>80)
            break;

        KeyFrame* pKF = *itKF;

        const vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

        for(vector<KeyFrame*>::const_iterator itNeighKF=vNeighs.begin(), itEndNeighKF=vNeighs.end(); itNeighKF!=itEndNeighKF; itNeighKF++)
        {
            KeyFrame* pNeighKF = *itNeighKF;
            if(!pNeighKF->isBad())
            {
                if(pNeighKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        const set<KeyFrame*> spChilds = pKF->GetChilds();
        for(set<KeyFrame*>::const_iterator sit=spChilds.begin(), send=spChilds.end(); sit!=send; sit++)
        {
            KeyFrame* pChildKF = *sit;
            if(!pChildKF->isBad())
            {
                if(pChildKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        KeyFrame* pParent = pKF->GetParent();
        if(pParent)
        {
            if(pParent->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                break;
            }
        }

    }

    if(pKFmax)
    {
        mpReferenceKF = pKFmax;
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }
}

bool Tracking::Relocalization()
{
    // Compute Bag of Words Vector
    mCurrentFrame.ComputeBoW();

    // Relocalization is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);

    if(vpCandidateKFs.empty())
        return false;

    const int nKFs = vpCandidateKFs.size();

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.75,true);

    vector<PnPsolver*> vpPnPsolvers;
    vpPnPsolvers.resize(nKFs);

    vector<vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    int nCandidates=0;

    for(int i=0; i<nKFs; i++)
    {
        KeyFrame* pKF = vpCandidateKFs[i];
        if(pKF->isBad())
            vbDiscarded[i] = true;
        else
        {
            int nmatches = matcher.SearchByBoW(pKF,mCurrentFrame,vvpMapPointMatches[i]);
            if(nmatches<15)
            {
                vbDiscarded[i] = true;
                continue;
            }
            else
            {
                PnPsolver* pSolver = new PnPsolver(mCurrentFrame,vvpMapPointMatches[i]);
                pSolver->SetRansacParameters(0.99,10,300,4,0.5,5.991);
                vpPnPsolvers[i] = pSolver;
                nCandidates++;
            }
        }
    }

    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    bool bMatch = false;
    ORBmatcher matcher2(0.9,true);

    while(nCandidates>0 && !bMatch)
    {
        for(int i=0; i<nKFs; i++)
        {
            if(vbDiscarded[i])
                continue;

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

            PnPsolver* pSolver = vpPnPsolvers[i];
            cv::Mat Tcw = pSolver->iterate(5,bNoMore,vbInliers,nInliers);

            // If Ransac reachs max. iterations discard keyframe
            if(bNoMore)
            {
                vbDiscarded[i]=true;
                nCandidates--;
            }

            // If a Camera Pose is computed, optimize
            if(!Tcw.empty())
            {
                Tcw.copyTo(mCurrentFrame.mTcw);

                set<MapPoint*> sFound;

                const int np = vbInliers.size();

                for(int j=0; j<np; j++)
                {
                    if(vbInliers[j])
                    {
                        mCurrentFrame.mvpMapPoints[j]=vvpMapPointMatches[i][j];
                        sFound.insert(vvpMapPointMatches[i][j]);
                    }
                    else
                        mCurrentFrame.mvpMapPoints[j]=NULL;
                }

                int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                if(nGood<10)
                    continue;

                for(int io =0; io<mCurrentFrame.N; io++)
                    if(mCurrentFrame.mvbOutlier[io])
                        mCurrentFrame.mvpMapPoints[io]=static_cast<MapPoint*>(NULL);

                // If few inliers, search by projection in a coarse window and optimize again
                if(nGood<50)
                {
                    int nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,10,100);

                    if(nadditional+nGood>=50)
                    {
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        if(nGood>30 && nGood<50)
                        {
                            sFound.clear();
                            for(int ip =0; ip<mCurrentFrame.N; ip++)
                                if(mCurrentFrame.mvpMapPoints[ip])
                                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                            nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,3,64);

                            // Final optimization
                            if(nGood+nadditional>=50)
                            {
                                nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                                for(int io =0; io<mCurrentFrame.N; io++)
                                    if(mCurrentFrame.mvbOutlier[io])
                                        mCurrentFrame.mvpMapPoints[io]=NULL;
                            }
                        }
                    }
                }


                // If the pose is supported by enough inliers stop ransacs and continue
                if(nGood>=50)
                {
                    bMatch = true;
                    break;
                }
            }
        }
    }

    if(!bMatch)
    {
        return false;
    }
    else
    {
        mnLastRelocFrameId = mCurrentFrame.mnId;
        return true;
    }

}

void Tracking::Reset()
{

    cout << "System Reseting" << endl;
    if(mpViewer)
    {
        mpViewer->RequestStop();
        while(!mpViewer->isStopped())
            usleep(3000);
    }

    // Reset Local Mapping
    cout << "Reseting Local Mapper...";
    mpLocalMapper->RequestReset();
    cout << " done" << endl;

    // Reset Loop Closing
    cout << "Reseting Loop Closing...";
    mpLoopClosing->RequestReset();
    cout << " done" << endl;

    // Clear BoW Database
    cout << "Reseting Database...";
    mpKeyFrameDB->clear();
    cout << " done" << endl;

    // Clear Map (this erase MapPoints and KeyFrames)
    mpMap->clear();

    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    mState = NO_IMAGES_YET;

    if(mpInitializer)
    {
        delete mpInitializer;
        mpInitializer = static_cast<Initializer*>(NULL);
    }

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();

    if(mpViewer)
        mpViewer->Release();
}

void Tracking::ChangeCalibration(const string &strSettingPath)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    Frame::mbInitialComputations = true;
}

void Tracking::InformOnlyTracking(const bool &flag)
{
    mbOnlyTracking = flag;
}



} //namespace ORB_SLAM
