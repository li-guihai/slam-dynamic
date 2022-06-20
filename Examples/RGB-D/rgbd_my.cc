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


#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>

#include <dirent.h>

#include<opencv2/core/core.hpp>

#include<System.h>

using namespace std;
using namespace cv;

void LoadMYImages(const string &rootDir, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps);

void LoadKITTIImages(const string &strPathToSequence, vector<string> &vstrImageLeft,
                vector<string> &vstrImageDepth, vector<string> &masks,
                vector<vector<Rect2d>> &boxesAll, vector<double> &vTimestamps);

int main(int argc, char **argv)
{
    if(argc != 4)
    {
        cerr << endl << "Usage: ./Examples/RGB-D/rgbd_my Vocabulary/ORBvoc.txt Examples/RGB-D/KITTI03.yaml /PATH/kittiOdem/color/03" << endl;
        return 1;
    }

    // Retrieve paths to images
    vector<string> vstrImageFilenamesRGB;
    vector<string> vstrImageFilenamesD;
    vector<string> masks;
    vector<vector<Rect2d>> boxesAll;
    vector<double> vTimestamps;
    LoadKITTIImages(argv[3], vstrImageFilenamesRGB, vstrImageFilenamesD, masks, boxesAll, vTimestamps);

    // Check consistency in the number of images and depthmaps
    int nImages = vstrImageFilenamesD.size();
    if(vstrImageFilenamesRGB.empty())
    {
        cerr << endl << "No images found in provided path." << endl;
        return 1;
    }
    else if(vstrImageFilenamesD.size()!=vstrImageFilenamesRGB.size())
    {
        cerr << endl << "Different number of images for rgb and depth." << endl;
        return 1;
    }
    cout << "Loaded " << nImages << " imgs from " << argv[3] << endl;

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::RGBD,true);

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    // Main loop
    cv::Mat imRGB, imD, mask;
    for(int ni=0; ni<nImages; ni++)
    {
        // Read image and depthmap from file
        imRGB = cv::imread(vstrImageFilenamesRGB[ni],CV_LOAD_IMAGE_UNCHANGED);
        imD = cv::imread(vstrImageFilenamesD[ni],CV_LOAD_IMAGE_UNCHANGED);
        mask = cv::imread(masks[ni],CV_LOAD_IMAGE_UNCHANGED);
        vector<cv::Rect2d> boxes = boxesAll[ni];
        double tframe = vTimestamps[ni];

        if(imRGB.empty() || imD.empty() || mask.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << string(argv[3]) << "/" << vstrImageFilenamesRGB[ni] << endl;
            return 1;
        }
        mask.convertTo(mask, CV_32F);

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        // Pass the image to the SLAM system
        SLAM.TrackRGBD(imRGB,imD, mask, boxes, tframe);

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        vTimesTrack[ni]=ttrack;

        // Wait to load the next frame
        double T=0;
        if(ni<nImages-1)
            T = vTimestamps[ni+1]-tframe;
        else if(ni>0)
            T = tframe-vTimestamps[ni-1];

        if(ttrack<T)
            usleep((T-ttrack)*1e6);
    }
    cv::waitKey();

    // Stop all threads
    SLAM.Shutdown();

    // Tracking time statistics
    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    cout << "mean tracking time: " << totaltime/nImages << endl;

    // Save camera trajectory
    SLAM.SaveTrajectoryTUM("CameraTrajectory.txt");
    SLAM.SavePCD("kitti03.pcd");

    return 0;
}

void LoadMYImages(const string &rootDir, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps)
{
    struct dirent *ptr;    
    DIR *dir;
    string rgbPath = rootDir+"/left";
    dir=opendir(rgbPath.c_str()); 
    vector<string> files;
    
    while((ptr=readdir(dir))!=NULL)
    {
        //跳过'.'和'..'两个目录
        if(ptr->d_name[0] == '.')
            continue;
        files.push_back(ptr->d_name);
    }

    double t=0.0;
    sort(files.begin(), files.end());
    cout << "图片数: " << files.size() << endl;
    for(auto &f:files) {
        string depth = "depth/" + f;
        string rgb = "left/" + f;
        vstrImageFilenamesRGB.push_back(rgb);
        vstrImageFilenamesD.push_back(depth);
        vTimestamps.push_back(t);
        t+=0.04;
    }
}

void LoadKITTIImages(const string &strPathToSequence, vector<string> &vstrImageLeft,
                vector<string> &vstrImageDepth, vector<string> &masks,
                vector<vector<Rect2d>> &boxesAll, vector<double> &vTimestamps)
{
    ifstream fTimes;
    string strPathTimeFile = strPathToSequence + "/times.txt";
    fTimes.open(strPathTimeFile.c_str());
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            ss >> t;
            vTimestamps.push_back(t);
        }
    }

    // string strPrefixMask = strPathToSequence + "/depth/";

    // struct dirent *ptr;    
    // DIR *dir;
    // dir=opendir(strPrefixMask.c_str()); 
    // vector<string> files;
    
    // while((ptr=readdir(dir))!=NULL)
    // {
    //     //跳过'.'和'..'两个目录
    //     if(ptr->d_name[0] == '.' || ptr->d_name[0] == '..')
    //         continue;
    //     files.push_back(ptr->d_name);
    // }
    // sort(files.begin(), files.end());
    const int nTimes = 340; //mask只有340张
    boxesAll.resize(nTimes);

    for(int i=0; i<nTimes; i++)
    {
        stringstream ss;
        ss << setfill('0') << setw(6) << i;
        vstrImageLeft.push_back(strPathToSequence + "/image_2/" + ss.str() + ".png");
        vstrImageDepth.push_back(strPathToSequence + "/depth/" + ss.str() + ".png");
        masks.push_back(strPathToSequence + "/mask/mask_" + ss.str() + ".png");

        ifstream box_info;
        string name = strPathToSequence + "/yolov5_2Dbbox/" + ss.str() + ".txt";
        box_info.open(name.c_str());
        if(!box_info.good())
            continue;
        vector<Rect2d> boxes;
        while (!box_info.eof())
        {
            string box_;
            getline(box_info, box_);
            if(!box_.empty())
            {
                stringstream ss_box;
                ss_box << box_;
                double id, center_x, center_y, width, height;
                ss_box >> id >> center_x >> center_y >> width >> height;
                cv::Rect2d temp(MAX(center_x - width/2, 0), MAX(center_y - height / 2, 0), width, height);
                boxes.push_back(temp);
            }
        } 
        boxesAll[i] = boxes;
    }
}