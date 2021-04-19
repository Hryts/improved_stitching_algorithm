//
// Created by hryts on 19.03.21.
//

#include "../include/Stitching/Stitcher.h"
#include "StitcherHelper.h"
#include "LMOptimizer.h"

#include <Eigen/LU>
#include <opencv2/core/eigen.hpp>

Stitcher::Stitcher(const std::vector<std::string> &paths) : m_sift(cv::SIFT::create(cv::Stitcher::PANORAMA))
{
    for (const std::string& path : paths)
        addImage(cv::imread(path, cv::IMREAD_UNCHANGED));
}

void
Stitcher::addImage(const cv::Mat &image)
{
	KeypointsVector keypoints;
    cv::Mat descriptors;
    m_sift->detect(image, keypoints);
    m_sift->compute(image, keypoints, descriptors);
    m_described_images.emplace_back(image, keypoints, descriptors);
}

Eigen::Matrix3d
Stitcher::computeTransformationMatrix(const size_t image1, const size_t image2)
{
    Eigen::Matrix3d result;
    getAffineTransformation(image1, image2, result);

    std::cout << "affine matrix det: " << result.determinant() << std::endl ;

    // transform data for LM optimizer
	std::vector<std::pair<Eigen::Vector2d , Eigen::Vector2d >> lm_input_data =
			filter_matching_pairs(image1, image2, result);

	Eigen::VectorXd lm_parameters(Eigen::Map<Eigen::VectorXd>(result.data(), result.cols() * result.rows()));


	LMOptimizer lm_optimizer(lm_input_data, lm_parameters);
	result = lm_optimizer.optimize();

	std::cout << "optimized det: " << result.determinant() << std::endl;

    return result;
}

void
Stitcher::getAffineTransformation(size_t reference_image_index, size_t new_input_image_index,
									   Eigen::Matrix3d& result,
									   size_t number_of_iterations,
									   size_t number_of_matching_pairs)
{
	const Eigen::Vector2d KCollinearityThreshold(.5f, .5f);

	Eigen::Matrix3d left, right, candidate;
	size_t current_number_of_inliers;
	size_t max_number_of_inliers = 0;
	size_t current_iteration;

	ImageDescribed reference_image = m_described_images[reference_image_index];
	ImageDescribed new_input_image = m_described_images[new_input_image_index];

	KeypointsVector ref_image_keypoints = get_keypoints(reference_image);
	KeypointsVector inp_image_keypoints = get_keypoints(new_input_image);

	std::cout << "ref_keypoints: " << ref_image_keypoints.size()
			  << " inp_keypoints: " << inp_image_keypoints.size() << std::endl;

	StitcherHelper::getMatchingPairs(get_descriptor(reference_image),
	                                 get_descriptor(new_input_image),
	                                m_matching_pairs);

	// Eigenvectors of covariance matrix of reference image
	Eigen::Matrix2d ref_covariance_matrix_eigenvectors = StitcherHelper::getEigenVectors(
			StitcherHelper::getCovarianceMatrix(ref_image_keypoints)
			);

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> distribution(0, m_matching_pairs.size() - 1);
	std::set<int> random_indexes;
	int random_index;

	for (size_t i = 0; i < number_of_iterations; ++i)
    {
	    random_indexes.clear();
	    while (random_indexes.size() < number_of_matching_pairs)
        {
	        random_index = distribution(gen);

	        if (!random_indexes.insert(random_index).second)
		        continue; // random_index value is already in random_indexes

	        current_iteration = random_indexes.size();
	        std::pair<size_t, size_t> random_pair = m_matching_pairs[random_index];

	        cv::Point2f ref_coordinates = ref_image_keypoints[random_pair.first].pt;
	        cv::Point2f inp_coordinates = inp_image_keypoints[random_pair.second].pt;

	        left.col(current_iteration-1) << ref_coordinates.x, ref_coordinates.y, 1;
	        right.col(current_iteration-1) << inp_coordinates.x, inp_coordinates.y, 1;

	        if (current_iteration != number_of_matching_pairs)
		        continue;

	        candidate = left * right.inverse();
	        double det = candidate.determinant();

	        if (StitcherHelper::areCollinear(right, ref_covariance_matrix_eigenvectors, KCollinearityThreshold) ||
	            det < .5 || det > 2.0)
              random_indexes.clear();
        }

	    current_number_of_inliers = StitcherHelper::countInliersNumber(
	    		m_matching_pairs, reference_image, new_input_image, candidate
	    		);

	    if(current_number_of_inliers > max_number_of_inliers)
        {
            max_number_of_inliers = current_number_of_inliers;
            result = candidate;
        }
    }
	std::cout << "maximum number of inliers: " << max_number_of_inliers << std::endl;
}

cv::Mat
Stitcher::stitchAll()
{
	calculateTransformations();
	cv::Mat result = get_image(m_described_images[0]);

	for (size_t i = 1; i < m_described_images.size(); ++i) {
		cv::Mat inp = get_image(m_described_images[i]);
		result = StitcherHelper::warp_to_right(result, inp, m_transformations[i - 1]);
	}
	return result;
}

void
Stitcher::calculateAdjacentTransformations(std::vector<Eigen::Matrix3d>& result)
{
	for (size_t image_index = 1; image_index < m_described_images.size(); ++image_index)
	{
		m_matching_pairs.clear();
		result.emplace_back(computeTransformationMatrix(image_index - 1, image_index));
	}
}

void
Stitcher::calculateTransformations()
{
	std::vector<Eigen::Matrix3d> adjacent_transformations;
	calculateAdjacentTransformations(adjacent_transformations);

	m_transformations = adjacent_transformations;

	if (m_transformations.size() == 1)
		return;

	for (size_t i = 1; i < m_transformations.size(); ++i)
	{
		m_transformations[i] = m_transformations[i - 1] * m_transformations[i];
	}
}

cv::Mat
Stitcher::get_image(ImageDescribed image)
{
	return std::get<PartOfImageDescribed::IMAGE>(image);
}

KeypointsVector
Stitcher::get_keypoints(ImageDescribed image)
{
	return std::get<PartOfImageDescribed::KEYPOINTS>(image);
}

cv::Mat
Stitcher::get_descriptor(ImageDescribed image)
{
	return std::get<PartOfImageDescribed::DESCRIPTORS>(image);
}

std::vector<std::pair<Eigen::Vector2d , Eigen::Vector2d>>
Stitcher::filter_matching_pairs(size_t image1,
								size_t image2,
                                const Eigen::Matrix3d& transformation,
                                double threshold)
{
	KeypointsVector keypoints1 = get_keypoints(m_described_images[image1]);
	KeypointsVector keypoints2 = get_keypoints(m_described_images[image2]);

	std::vector<std::pair<Eigen::Vector2d , Eigen::Vector2d >> result;

	for (std::pair<size_t, size_t> matching_pair : m_matching_pairs)
	{
		cv::Point2f ref_coordinates = keypoints1[matching_pair.first].pt;
		cv::Point2f inp_coordinates = keypoints2[matching_pair.second].pt;

		Eigen::Vector3d reference_v(ref_coordinates.x, ref_coordinates.y, 1);
		Eigen::Vector3d new_input_v(inp_coordinates.x, inp_coordinates.y, 1);

		Eigen::Vector3d transformed = transformation * new_input_v;

		if (StitcherHelper::euclidianDistanceSquared(reference_v, transformed) < threshold)
		{
			result.emplace_back(
					std::make_pair(Eigen::Vector2d (ref_coordinates.x, ref_coordinates.y),
					               Eigen::Vector2d (inp_coordinates.x, inp_coordinates.y)));
		}
	}
	return result;
}

cv::Mat
Stitcher::drawKeypoints(size_t image_index)
{
	cv::Mat image = get_image(m_described_images[image_index]);
	KeypointsVector keypoints = get_keypoints(m_described_images[image_index]);
	cv::Mat result;
	cv::drawKeypoints(image, keypoints, result);
	return result;
}