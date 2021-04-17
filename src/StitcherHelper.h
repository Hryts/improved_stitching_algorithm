//
// Created by hryts on 16.04.21.
//

#ifndef IMPROVEDSTITCHING_STITCHERHELPER_H
#define IMPROVEDSTITCHING_STITCHERHELPER_H

// external includes
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <Eigen/Dense>

#include "../include/Stitching/Stitcher.h"



class StitcherHelper {
public:
	static size_t euclidianDistanceSquared(const Eigen::Vector3d& point1, const Eigen::Vector3d& point2);

	static void getMatchingPairs(const cv::Mat& ref_image_descriptors,
	                             const cv::Mat& inp_image_descriptors,
	                             std::vector<std::pair<size_t, size_t>>& result);

	static size_t countInliersNumber(const std::vector<std::pair<size_t, size_t>>& matching_pairs,
	                                 const ImageDescribed& reference_image, const ImageDescribed& new_input_image,
	                                 const Eigen::Matrix3d& transformation, size_t epsilon_squared=4);

	static bool areCollinear(const Eigen::Matrix3d& vectors, const Eigen::Matrix2d& covariance_eigenvectors,
	                         const Eigen::Vector2d& threshold);

	static Eigen::Matrix2d getCovarianceMatrix(const KeypointsVector& keypoints);

	static double covariance(const std::vector<double>& x, const std::vector<double>& y, const Eigen::Vector2d& mean);

	static Eigen::Matrix2d getEigenVectors(const Eigen::Matrix2d& matrix);

	static double mean(const std::vector<double>& values);

	static cv::Mat warp(cv::Mat& image1, cv::Mat& image2, Eigen::Matrix3d& eigen_homography);
};


#endif //IMPROVEDSTITCHING_STITCHERHELPER_H
