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
    Eigen::Matrix3f computeTransformationMatrix(const size_t image1, const size_t image2);
    cv::Mat drawKeypoints(size_t imageIndex);
private:
    cv::Ptr<cv::Feature2D> m_sift;
    std::vector<ImageDescribed> m_described_images;
private:
    void getAffineTransformation(const size_t reference_image_index, const size_t new_input_image_index,
								 Eigen::Matrix3f& result,
                                 size_t number_of_iterations=100, size_t number_of_matching_pairs=3);
    static void getMatchingPairs(const cv::Mat& reference_descriptors,
                                 const cv::Mat& inp_image_descriptors,
                                 std::vector<std::pair<size_t, size_t>>& result);
    static size_t countInliersNumber(const std::vector<std::pair<size_t, size_t>>& matching_pairs,
                                     const ImageDescribed& reference_image, const ImageDescribed& new_input_image,
                                     const Eigen::Matrix3f& transformation, size_t epsilon_squared=2500);
	static bool isCollinear(const Eigen::Matrix3f& matrix, float threshold=.1f);
};


#endif //IMPROVEDSTITCHING_STITCHER_H
