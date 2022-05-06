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
#include<iostream>
#include<algorithm>
#include<fstream>
#include<iomanip>
#include<chrono>
#include <opencv2/tracking.hpp>

#include<opencv2/core/core.hpp>

#include<System.h>
#include"yolo.h"
using namespace std;

void LoadImages(const string &strPathToSequence, vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight, vector<double> &vTimestamps);

int main(int argc, char **argv)
{
    if(argc != 4 && argc != 5)
    {
        cerr << endl << "Usage: ./stereo_kitti path_to_vocabulary path_to_settings path_to_sequence (yolo)" << endl;
        return 1;
    }

    // Retrieve paths to images
    vector<string> vstrImageLeft;
    vector<string> vstrImageRight;
    vector<double> vTimestamps;
    LoadImages(string(argv[3]), vstrImageLeft, vstrImageRight, vTimestamps);

    const int nImages = vstrImageLeft.size();

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::STEREO,true);

    //yolo init
    yolov3::yolov3Segment* yolo;
    if (argc==5)
    {
        //cout << "Loading Mask R-CNN. This could take a while..." << endl;
        //MaskNet = new DynaSLAM::SegmentDynObject();
        //cout << "Mask R-CNN loaded!" << endl;
        cout << endl <<"Loading Yolov3 net. This could take a while..." << endl;
        yolo = new yolov3::yolov3Segment();
        cout << "Yolov3 net loaded!" << endl;
    }

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;   

    // Main loop
    cv::Mat imLeft, imRight;
    // bool TragetInitialized = false, initFrame;
    // double ref_tframe = vTimestamps[900];
    for(int ni=0; ni<nImages; ni++)
    {
        // Read left and right images from file
        imLeft = cv::imread(vstrImageLeft[ni],CV_LOAD_IMAGE_UNCHANGED);
        imRight = cv::imread(vstrImageRight[ni],CV_LOAD_IMAGE_UNCHANGED);
        double tframe = vTimestamps[ni];

        if(imLeft.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << string(vstrImageLeft[ni]) << endl;
            return 1;
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif
        cout<<"img_"<<ni<<endl;
        // Pass the image to the SLAM system
        if(argc == 5) {
            std::chrono::steady_clock::time_point t_yolo = std::chrono::steady_clock::now();
            // cv::Mat l_r, temp;
            // cv::vconcat(imLeft, imRight, l_r);
            // cv::Mat mask = cv::Mat::ones(imLeft.size(), CV_8U);
            vector<cv::Rect2d> boxes;
            boxes = yolo->Segmentation_(imLeft);
            // temp = mask(cv::Range(0,mask.rows/2), cv::Range(0, mask.cols));
            // cv::bitwise_and(mask(cv::Range(0,mask.rows/2), cv::Range(0, mask.cols)), 
            // mask(cv::Range(mask.rows/2,mask.rows), cv::Range(0, mask.cols)), mask); // 将左右图的mask 合并

            std::chrono::steady_clock::time_point t_yolo1 = std::chrono::steady_clock::now();
            double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t_yolo1 - t_yolo).count();
            cout<<" detect time: "<<ttrack<<endl;
            // cv::imshow("l_r", l_r);
            // cv::imshow("mask", 255*mask);
            // cv::waitKey();

            if(boxes.empty())
                cout<<"No Target."<<endl;
                
            SLAM.TrackStereo(imLeft,imRight,boxes, tframe);
                // for(auto box:boxes) {
                //     cv::rectangle(imLeft, box, Scalar(255, 0, 0));
                // }
                // cv::imshow("imLeft", imLeft);
                // waitKey();
            
            ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(std::chrono::steady_clock::now() - t_yolo1).count();
            cout<<"track time: "<<ttrack<<endl<<"==============" <<endl;
        }
        else{
            SLAM.TrackStereo(imLeft,imRight,tframe);
        }

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
    SLAM.SaveTrajectoryKITTI("CameraTrajectory.txt");

    return 0;
}

void LoadImages(const string &strPathToSequence, vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight, vector<double> &vTimestamps)
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

    string strPrefixLeft = strPathToSequence + "/image_2/";
    string strPrefixRight = strPathToSequence + "/image_3/";

    const int nTimes = vTimestamps.size();
    vstrImageLeft.resize(nTimes);
    vstrImageRight.resize(nTimes);

    for(int i=0; i<nTimes; i++)
    {
        stringstream ss;
        ss << setfill('0') << setw(6) << i;
        vstrImageLeft[i] = strPrefixLeft + ss.str() + ".png";
        vstrImageRight[i] = strPrefixRight + ss.str() + ".png";
    }
}
