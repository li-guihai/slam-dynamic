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

#include "MapDrawer.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include <pangolin/pangolin.h>
#include <mutex>

namespace ORB_SLAM2
{


MapDrawer::MapDrawer(Map* pMap, const string &strSettingPath):mpMap(pMap)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

    mKeyFrameSize = fSettings["Viewer.KeyFrameSize"];
    mKeyFrameLineWidth = fSettings["Viewer.KeyFrameLineWidth"];
    mGraphLineWidth = fSettings["Viewer.GraphLineWidth"];
    mPointSize = fSettings["Viewer.PointSize"];
    mCameraSize = fSettings["Viewer.CameraSize"];
    mCameraLineWidth = fSettings["Viewer.CameraLineWidth"];

    nums = vector<int>(2,0);

}

void MapDrawer::DrawMapPoints()
{
    const vector<MapPoint*> &vpMPs = mpMap->GetAllMapPoints();
    const vector<MapPoint*> &vpRefMPs = mpMap->GetReferenceMapPoints();

    set<MapPoint*> spRefMPs(vpRefMPs.begin(), vpRefMPs.end());

    if(vpMPs.empty())
        return;

    glPointSize(mPointSize);
    glBegin(GL_POINTS);
    glColor3f(0.0,0.0,0.0);

    for(size_t i=0, iend=vpMPs.size(); i<iend;i++)
    {
        if(vpMPs[i]->isBad() || spRefMPs.count(vpMPs[i]))
            continue;
        cv::Mat pos = vpMPs[i]->GetWorldPos();
        glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
    }
    glEnd();

    glPointSize(mPointSize);
    glBegin(GL_POINTS);
    glColor3f(1.0,0.0,0.0);

    for(set<MapPoint*>::iterator sit=spRefMPs.begin(), send=spRefMPs.end(); sit!=send; sit++)
    {
        if((*sit)->isBad())
            continue;
        cv::Mat pos = (*sit)->GetWorldPos();
        glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));

    }

    glEnd();
}

void MapDrawer::DrawKeyFrames(const bool bDrawKF, const bool bDrawGraph)
{
    const float &w = mKeyFrameSize;
    const float h = w*0.75;
    const float z = w*0.6;

    const vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();

    if(bDrawKF)
    {
        for(size_t i=0; i<vpKFs.size(); i++)
        {
            KeyFrame* pKF = vpKFs[i];
            cv::Mat Twc = pKF->GetPoseInverse().t();

            glPushMatrix();

            glMultMatrixf(Twc.ptr<GLfloat>(0));

            glLineWidth(mKeyFrameLineWidth);
            glColor3f(0.0f,0.0f,1.0f);
            glBegin(GL_LINES);
            glVertex3f(0,0,0);
            glVertex3f(w,h,z);
            glVertex3f(0,0,0);
            glVertex3f(w,-h,z);
            glVertex3f(0,0,0);
            glVertex3f(-w,-h,z);
            glVertex3f(0,0,0);
            glVertex3f(-w,h,z);

            glVertex3f(w,h,z);
            glVertex3f(w,-h,z);

            glVertex3f(-w,h,z);
            glVertex3f(-w,-h,z);

            glVertex3f(-w,h,z);
            glVertex3f(w,h,z);

            glVertex3f(-w,-h,z);
            glVertex3f(w,-h,z);
            glEnd();

            glPopMatrix();
        }
    }

    if(bDrawGraph)
    {
        glLineWidth(mGraphLineWidth);
        glColor4f(0.0f,1.0f,0.0f,0.6f);
        glBegin(GL_LINES);

        for(size_t i=0; i<vpKFs.size(); i++)
        {
            // Covisibility Graph
            const vector<KeyFrame*> vCovKFs = vpKFs[i]->GetCovisiblesByWeight(100);
            cv::Mat Ow = vpKFs[i]->GetCameraCenter();
            if(!vCovKFs.empty())
            {
                for(vector<KeyFrame*>::const_iterator vit=vCovKFs.begin(), vend=vCovKFs.end(); vit!=vend; vit++)
                {
                    if((*vit)->mnId<vpKFs[i]->mnId)
                        continue;
                    cv::Mat Ow2 = (*vit)->GetCameraCenter();
                    glVertex3f(Ow.at<float>(0),Ow.at<float>(1),Ow.at<float>(2));
                    glVertex3f(Ow2.at<float>(0),Ow2.at<float>(1),Ow2.at<float>(2));
                }
            }

            // Spanning tree
            KeyFrame* pParent = vpKFs[i]->GetParent();
            if(pParent)
            {
                cv::Mat Owp = pParent->GetCameraCenter();
                glVertex3f(Ow.at<float>(0),Ow.at<float>(1),Ow.at<float>(2));
                glVertex3f(Owp.at<float>(0),Owp.at<float>(1),Owp.at<float>(2));
            }

            // Loops
            set<KeyFrame*> sLoopKFs = vpKFs[i]->GetLoopEdges();
            for(set<KeyFrame*>::iterator sit=sLoopKFs.begin(), send=sLoopKFs.end(); sit!=send; sit++)
            {
                if((*sit)->mnId<vpKFs[i]->mnId)
                    continue;
                cv::Mat Owl = (*sit)->GetCameraCenter();
                glVertex3f(Ow.at<float>(0),Ow.at<float>(1),Ow.at<float>(2));
                glVertex3f(Owl.at<float>(0),Owl.at<float>(1),Owl.at<float>(2));
            }
        }

        glEnd();
    }
}

void MapDrawer::DrawDynamics()
{
    const float &w = mKeyFrameSize;
    const float h = w*0.75;
    const float z = w*0.6;

    // const vector<dynamic> dynamics = mpMap->GetAllDynamics();
    for(size_t i=0; i<mDynamics.size() - nums[0] - nums[1]; i++)
    {
        // if(count(mbox_id.begin(), mbox_id.begin()+i, mbox_id[i])) {
        //     for(size_t j(i-1); j>=0; j--) {
        //         find()
        //     }
        // }

        glPushMatrix();

        glMultMatrixd(mDynamics[i].m);

    // 长方体，12条边
        glLineWidth(mCameraLineWidth);
        float g = (float)(mbox_id[i] % 2) / 2.f, b = (float)(mbox_id[i] % 5) / 5.f;
        glColor3f(1.0f,g,b);
        glBegin(GL_LINES);
        //1
        glVertex3f(-w,-h,-z);
        glVertex3f(w,-h,-z);
    //2
        glVertex3f(w,-h,-z);
        glVertex3f(w,h,-z);
    //3
        glVertex3f(w,h,-z);
        glVertex3f(-w,h,-z);
    //4
        glVertex3f(-w,h,-z);
        glVertex3f(-w,-h,-z);
    //5
        glVertex3f(-w,-h,-z);
        glVertex3f(-w,-h,z);
    //6
        glVertex3f(-w,-h,z);
        glVertex3f(w,-h,z);
    //7
        glVertex3f(w,-h,z);
        glVertex3f(w,h,z);
    //8
        glVertex3f(w,h,z);
        glVertex3f(-w,h,z);
    //9
        glVertex3f(-w,-h,z);
        glVertex3f(-w,h,z);
    //10
        glVertex3f(-w,h,z);
        glVertex3f(-w,h,-z);
    //11
        glVertex3f(w,h,z);
        glVertex3f(w,h,-z);
    //12
        glVertex3f(w,-h,z);
        glVertex3f(w,-h,-z);

        glEnd();

        glPopMatrix();

    }
    for(size_t i=mDynamics.size() - nums[0] - nums[1]; i<mDynamics.size(); i++) {
        glPushMatrix();

        glMultMatrixd(mDynamics[i].m);

    // 长方体，12条边
        glLineWidth(mCameraLineWidth);
        // float r = (float)(mbox_id[i] % 2) / 2.f, b = (float)(mbox_id[i] % 5) / 5.f;
        glColor3f(1.0f,0.0f,1.0f);
        glBegin(GL_LINES);
        //1
        glVertex3f(-w,-h,-z);
        glVertex3f(w,-h,-z);
    //2
        glVertex3f(w,-h,-z);
        glVertex3f(w,h,-z);
    //3
        glVertex3f(w,h,-z);
        glVertex3f(-w,h,-z);
    //4
        glVertex3f(-w,h,-z);
        glVertex3f(-w,-h,-z);
    //5
        glVertex3f(-w,-h,-z);
        glVertex3f(-w,-h,z);
    //6
        glVertex3f(-w,-h,z);
        glVertex3f(w,-h,z);
    //7
        glVertex3f(w,-h,z);
        glVertex3f(w,h,z);
    //8
        glVertex3f(w,h,z);
        glVertex3f(-w,h,z);
    //9
        glVertex3f(-w,-h,z);
        glVertex3f(-w,h,z);
    //10
        glVertex3f(-w,h,z);
        glVertex3f(-w,h,-z);
    //11
        glVertex3f(w,h,z);
        glVertex3f(w,h,-z);
    //12
        glVertex3f(w,-h,z);
        glVertex3f(w,-h,-z);

        glEnd();

        glPopMatrix();
    }
    
}

void MapDrawer::DrawCurrentCamera(pangolin::OpenGlMatrix &Twc)
{
    const float &w = mCameraSize;
    const float h = w*0.75;
    const float z = w*0.6;

    glPushMatrix();

#ifdef HAVE_GLES
        glMultMatrixf(Twc.m);
#else
        glMultMatrixd(Twc.m);
#endif

    glLineWidth(mCameraLineWidth);
    glColor3f(0.0f,1.0f,0.0f);
    glBegin(GL_LINES);
    glVertex3f(0,0,0);
    glVertex3f(w,h,z);
    glVertex3f(0,0,0);
    glVertex3f(w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,h,z);

    glVertex3f(w,h,z);
    glVertex3f(w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(-w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(w,h,z);

    glVertex3f(-w,-h,z);
    glVertex3f(w,-h,z);
    glEnd();

    glPopMatrix();
}

// void MapDrawer::DrawCurrentDynamic(pangolin::OpenGlMatrix &Twd)
// {
//     const float &w = mCameraSize;
//     const float h = w*0.75;
//     const float z = w*0.6;

//     glPushMatrix();

// #ifdef HAVE_GLES
//         glMultMatrixf(Twc.m);
// #else
//         glMultMatrixd(Twd.m);
// #endif
// // 长方体，12条边
//     glLineWidth(mCameraLineWidth);
//     glColor3f(0.0f,1.0f,0.0f);
//     glBegin(GL_LINES);
//     //1
//     glVertex3f(-w,-h,-z);
//     glVertex3f(w,-h,-z);
// //2
//     glVertex3f(w,-h,-z);
//     glVertex3f(w,h,-z);
// //3
//     glVertex3f(w,h,-z);
//     glVertex3f(-w,h,-z);
// //4
//     glVertex3f(-w,h,-z);
//     glVertex3f(-w,-h,-z);
// //5
//     glVertex3f(-w,-h,-z);
//     glVertex3f(-w,-h,z);
// //6
//     glVertex3f(-w,-h,z);
//     glVertex3f(w,-h,z);
// //7
//     glVertex3f(w,-h,z);
//     glVertex3f(w,h,z);
// //8
//     glVertex3f(w,h,z);
//     glVertex3f(-w,h,z);
// //9
//     glVertex3f(-w,-h,z);
//     glVertex3f(-w,h,z);
// //10
//     glVertex3f(-w,h,z);
//     glVertex3f(-w,h,-z);
// //11
//     glVertex3f(w,h,z);
//     glVertex3f(w,h,-z);
// //12
//     glVertex3f(w,-h,z);
//     glVertex3f(w,-h,-z);

//     glEnd();

//     glPopMatrix();
// }


void MapDrawer::SetCurrentCameraPose(const cv::Mat &Tcw)
{
    unique_lock<mutex> lock(mMutexCamera);
    mCameraPose = Tcw.clone();
}

void MapDrawer::SetCurrentCameraPose(const cv::Mat &Tcw, const vector<dynamic> &dyns)
{
    unique_lock<mutex> lock(mMutexCamera);
    mCameraPose = Tcw.clone();

    cv::Mat Rwc = mCameraPose.rowRange(0,3).colRange(0,3).t();
    for(auto dyn:dyns) {
        cv::Mat twd = Rwc*dyn.tcd;
        cv::Mat Rwd = Rwc*dyn.Rcd;
        mtwd_s.push_back(twd) ; // 世界坐标系下当前帧到动态物体的位移
        mRwd_s.push_back(Rwd);
        mbox_id.push_back(dyn.box_id) ; 
    }
    // 存储最近两帧的数量用于显示
    nums[0] = nums[1];
    nums[1] = dyns.size();
}

void MapDrawer::GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M)
{
    if(!mCameraPose.empty())
    {
        cv::Mat Rwc(3,3,CV_32F);
        cv::Mat twc(3,1,CV_32F);
        {
            unique_lock<mutex> lock(mMutexCamera);
            Rwc = mCameraPose.rowRange(0,3).colRange(0,3).t();
            twc = -Rwc*mCameraPose.rowRange(0,3).col(3);
        }

        M.m[0] = Rwc.at<float>(0,0);
        M.m[1] = Rwc.at<float>(1,0);
        M.m[2] = Rwc.at<float>(2,0);
        M.m[3]  = 0.0;

        M.m[4] = Rwc.at<float>(0,1);
        M.m[5] = Rwc.at<float>(1,1);
        M.m[6] = Rwc.at<float>(2,1);
        M.m[7]  = 0.0;

        M.m[8] = Rwc.at<float>(0,2);
        M.m[9] = Rwc.at<float>(1,2);
        M.m[10] = Rwc.at<float>(2,2);
        M.m[11]  = 0.0;

        M.m[12] = twc.at<float>(0);
        M.m[13] = twc.at<float>(1);
        M.m[14] = twc.at<float>(2);
        M.m[15]  = 1.0;
    }
    else
        M.SetIdentity();

    if(mtwd_s.empty()) {
        nums[0] = nums[1];
        nums[1] = 0;
    }
    else {
        for(int i=0; i<mtwd_s.size(); i++) {
            pangolin::OpenGlMatrix M_temp = M;
            M_temp.m[0] = mRwd_s[i].at<float>(0,0);
            M_temp.m[1] = mRwd_s[i].at<float>(1,0);
            M_temp.m[2] = mRwd_s[i].at<float>(2,0);

            M_temp.m[4] = mRwd_s[i].at<float>(0,1);
            M_temp.m[5] = mRwd_s[i].at<float>(1,1);
            M_temp.m[6] = mRwd_s[i].at<float>(2,1);

            M_temp.m[8] = mRwd_s[i].at<float>(0,2);
            M_temp.m[9] = mRwd_s[i].at<float>(1,2);
            M_temp.m[10] = mRwd_s[i].at<float>(2,2);

            M_temp.m[12] += mtwd_s[i].at<float>(0);
            M_temp.m[13] += mtwd_s[i].at<float>(1);
            M_temp.m[14] += mtwd_s[i].at<float>(2);

            mDynamics.push_back(M_temp);
        }
        mtwd_s.clear();
        mRwd_s.clear();
    }

}

// void MapDrawer::GetCurrentOpenGLDynamicMatrix(const pangolin::OpenGlMatrix &M, vector<pangolin::OpenGlMatrix> &M_dyn)
// {

//     for(auto mtwd:mtwd_s) {
//         pangolin::OpenGlMatrix M_temp = M;
//         M_temp.m[12] += mtwd.at<float>(0);
//         M_temp.m[13] += mtwd.at<float>(1);
//         M_temp.m[14] += mtwd.at<float>(2);

//         M_dyn.push_back(M_temp);
//         mDynamics.push_back(M_temp);
//     }
//     mtwd_s.clear();
// }

} //namespace ORB_SLAM
