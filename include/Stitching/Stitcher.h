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

typedef std::vector<cv::KeyPoint> KeypointsVector;
typedef std::tuple<cv::Mat, KeypointsVector, cv::Mat> ImageDescribed;

enum PartOfImageDescribed
{
	IMAGE,
	KEYPOINTS,
	DESCRIPTORS
};

class Stitcher
{
public:
    Stitcher (const std::vector<std::string>& paths);

    void addImage(const cv::Mat& image);

    Eigen::Matrix3d computeTransformationMatrix(const size_t image1, const size_t image2);

    cv::Mat stitchAll();

private:
    cv::Ptr<cv::Feature2D> m_sift;

    std::vector<ImageDescribed> m_described_images;

	std::vector<std::pair<size_t, size_t>> m_matching_pairs;

	std::vector<Eigen::Matrix3d> m_transformations;

private:
	void calculateAdjacentTransformations(std::vector<Eigen::Matrix3d>& result);

	void calculateTransformations();

    void getAffineTransformation(size_t reference_image_index, size_t new_input_image_index,
								 Eigen::Matrix3d& result,
                                 size_t number_of_iterations=1500, size_t number_of_matching_pairs=3);

	std::vector<std::pair<Eigen::Vector2d , Eigen::Vector2d>> filter_matching_pairs(
								 size_t image1,
								 size_t image2,
					             const Eigen::Matrix3d& transformation,
					             double threshold = 100);

	static cv::Mat get_image(ImageDescribed image);

	static KeypointsVector get_keypoints(ImageDescribed image);

	static cv::Mat get_descriptor(ImageDescribed image);
};

#endif //IMPROVEDSTITCHING_STITCHER_H