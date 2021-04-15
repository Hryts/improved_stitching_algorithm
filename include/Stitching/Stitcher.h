//
// Created by hryts on 19.03.21.
//

#ifndef IMPROVEDSTITCHING_STITCHER_H
#define IMPROVEDSTITCHING_STITCHER_H

// external includes
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <Eigen/Dense>


// internal includes
#include "../../src/KDTree.h"

typedef std::tuple<cv::Mat, std::vector<cv::KeyPoint>, cv::Mat> ImageDescribed;

enum ImageDescribedGetter
{
	IMAGE,
	KEYPOINTS,
	DESCRIPTORS
};

class Stitcher
{
public:
    Stitcher (const std::vector<cv::Mat>& images);
    Stitcher (const std::vector<std::string>& paths);
    void addImage(const cv::Mat& image);
    Eigen::Matrix3d computeTransformationMatrix(const size_t image1, const size_t image2);
    cv::Mat drawKeypoints(size_t ind1, size_t ind2, std::vector<std::pair<size_t, size_t>> matching_pairs);
	cv::Mat warp(cv::Mat& image1, cv::Mat& image2, Eigen::Matrix3d& eigen_homography);
	cv::Mat warp_public(size_t im1, size_t im2, Eigen::Matrix3d& eigen_homography);
	cv::Mat stitchAll();
private:
    cv::Ptr<cv::Feature2D> m_sift;
    std::vector<ImageDescribed> m_described_images;
	std::vector<std::pair<size_t, size_t>> m_matching_pairs;
	std::vector<Eigen::Matrix3d> m_transformations;
private:
	void calculateAdjacentTransformations(std::vector<Eigen::Matrix3d>& result);
	void calculateTransformations();
    void getAffineTransformation(const size_t reference_image_index, const size_t new_input_image_index,
								 Eigen::Matrix3d& result,
                                 size_t number_of_iterations=1000, size_t number_of_matching_pairs=3);
    static void getMatchingPairs(const cv::Mat& ref_image_descriptors,
                                 const cv::Mat& inp_image_descriptors,
                                 std::vector<std::pair<size_t, size_t>>& result);
    static size_t countInliersNumber(const std::vector<std::pair<size_t, size_t>>& matching_pairs,
                                     const ImageDescribed& reference_image, const ImageDescribed& new_input_image,
                                     const Eigen::Matrix3d& transformation, size_t epsilon_squared=4);
    static bool areCollinear(const Eigen::Matrix3d& vectors, const Eigen::Matrix2d& covariance_eigenvectors,
                             const Eigen::Vector2d& threshold);
	static Eigen::Matrix2d getCovarianceMatrix(const std::vector<cv::KeyPoint>& keypoints);
	static double covariance(const std::vector<double>& x, const std::vector<double>& y, const Eigen::Vector2d& mean);
	static Eigen::Matrix2d getEigenVectors(const Eigen::Matrix2d& matrix);
	static double mean(const std::vector<double>& values);
};


#endif //IMPROVEDSTITCHING_STITCHER_H