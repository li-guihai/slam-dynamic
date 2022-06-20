/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 */

#include "pointcloudmapping.h"
#include <KeyFrame.h>
#include <opencv2/highgui/highgui.hpp>
#include <pcl/visualization/cloud_viewer.h>
#include "Converter.h"

PointCloudMapping::PointCloudMapping(double resolution_)
{
    this->resolution = resolution_;
    voxel.setLeafSize( resolution, resolution, resolution);
    globalMap = boost::make_shared< PointCloud >( );
    
    viewerThread = make_shared<thread>( bind(&PointCloudMapping::viewer, this ) );
}

void PointCloudMapping::shutdown()
{
    {
        unique_lock<mutex> lck(shutDownMutex);
        shutDownFlag = true;
        keyFrameUpdated.notify_one();
    }
    viewerThread->join();
}

void PointCloudMapping::insertKeyFrame(KeyFrame* kf, cv::Mat& color, cv::Mat& depth, cv::Mat& mask, vector<cv::Rect2d> &dyn_obj)
{
    cout<<"receive a keyframe, id = "<<kf->mnId<<endl;
    unique_lock<mutex> lck(keyframeMutex);
    keyframes.push_back( kf );
    colorImgs.push_back( color.clone() );
    depthImgs.push_back( depth.clone() );
    masks.push_back(mask);
    cout<<"dyn_obj.size() = "<<dyn_obj.size()<<endl;
    dyn_objs.push_back(dyn_obj);
    
    keyFrameUpdated.notify_one();
}

pcl::PointCloud< PointCloudMapping::PointT >::Ptr PointCloudMapping::generatePointCloud(KeyFrame* kf,cv::Mat& color,
                                                    cv::Mat& depth, cv::Mat& mask, vector<cv::Rect2d> &dyn_obj)
{
    PointCloud::Ptr tmp( new PointCloud() );
    // point cloud is null ptr
    int masked_num = 0;
    for ( int m=0; m<depth.rows; m+=3 )
    {
        for ( int n=0; n<depth.cols; n+=3 )
        {
            bool need_continue = false;            
            for (auto &obj:dyn_obj) {
                if(obj.contains(cv::Point2f(n, m)) && mask.at<float>(m, n)!= 0){
                    need_continue = true;
                    masked_num++;
                    break;
                }
            }
            float d = depth.ptr<float>(m)[n];
            if (d < 0.01 || d>5 || need_continue)// 
                continue;


            PointT p;
            p.z = d;
            p.x = ( n - kf->cx) * p.z / kf->fx;
            p.y = ( m - kf->cy) * p.z / kf->fy;
            
            p.b = color.ptr<uchar>(m)[n*3];
            p.g = color.ptr<uchar>(m)[n*3+1];
            p.r = color.ptr<uchar>(m)[n*3+2];
                
            tmp->points.push_back(p);
        }
    }
    cout<<"masked num: "<<masked_num<<endl;
    
    Eigen::Isometry3d T = ORB_SLAM2::Converter::toSE3Quat( kf->GetPose() );
    PointCloud::Ptr cloud(new PointCloud);
    pcl::transformPointCloud( *tmp, *cloud, T.inverse().matrix());
    cloud->is_dense = false;
    
    cout<<"generate point cloud for kf "<<kf->mnId<<", size="<<cloud->points.size()<<endl;
    return cloud;
}

void viewerOneOff(pcl::visualization::PCLVisualizer& viewer)
{
    viewer.setBackgroundColor(0, 0.0, 0.5);//设置背景颜色
}


void PointCloudMapping::viewer()
{
    pcl::visualization::CloudViewer viewer("viewer");
    
    while(1)
    {
        {
            unique_lock<mutex> lck_shutdown( shutDownMutex );
            if (shutDownFlag)
            {
                break;
            }
        }
        {
            unique_lock<mutex> lck_keyframeUpdated( keyFrameUpdateMutex );
            keyFrameUpdated.wait( lck_keyframeUpdated );
        }
        
        // keyframe is updated 
        size_t N=0;
        {
            unique_lock<mutex> lck( keyframeMutex );
            N = keyframes.size();
        }
        
        for ( size_t i=lastKeyframeSize; i<N ; i++ )
        {
            PointCloud::Ptr p = generatePointCloud( keyframes[i], colorImgs[i], depthImgs[i],masks[i], dyn_objs[i]);
            *globalMap += *p;
            // if(i % 3 == 0) {
            // PointCloud::Ptr tmp(new PointCloud());

            // voxel.setInputCloud(p);
            // voxel.filter( *tmp );
            // string save_path = "/home/hai/pcds/" + std::to_string(i) + ".pcd";
            // pcl::io::savePCDFile(save_path, *tmp);
            // }
        }
        PointCloud::Ptr tmp(new PointCloud());
        voxel.setInputCloud( globalMap );
        voxel.filter( *tmp );
        globalMap->swap( *tmp );
        viewer.showCloud( globalMap );
        viewer.runOnVisualizationThreadOnce(viewerOneOff);
        cout<<"show global map, size="<<globalMap->points.size()<<endl;
        lastKeyframeSize = N;
    }
}

//保存全局点云地图点
void PointCloudMapping::savePCD(const string &filename) {
    pcl::PointCloud< PointCloudMapping::PointT >::Ptr global_map(new pcl::PointCloud< PointCloudMapping::PointT >);
    global_map = this->globalMap;
    pcl::io::savePCDFile( filename, *globalMap );
    cout<<"globalMap save finished as "<< filename <<endl<<endl;
}

//获取全局点云地图点，智能指针，return 回来
pcl::PointCloud< PointCloudMapping::PointT >::Ptr PointCloudMapping::getPCD() {
    return this->globalMap;
}