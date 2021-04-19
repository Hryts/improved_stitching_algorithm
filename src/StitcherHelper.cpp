//
// Created by hryts on 16.04.21.
//

#include "StitcherHelper.h"

#include <Eigen/LU>
#include <opencv2/core/eigen.hpp>

size_t
StitcherHelper::euclidianDistanceSquared(const Eigen::Vector3d& point1, const Eigen::Vector3d& point2)
{
	double result = 0;
	for (int i = 0; i < point1.size(); ++i)
		result += pow((point1(i) - point2(i)), 2);
	return result;
}

void
StitcherHelper::getMatchingPairs(const cv::Mat& ref_image_descriptors,
	                                  const cv::Mat& inp_image_descriptors,
	                                  std::vector<std::pair<size_t, size_t>>& result)
{
	// TODO: explore Lowe's ratio test
	// TODO: explore cross check
	KDTree kd_tree(inp_image_descriptors);
	for (int i = 0; i < ref_image_descriptors.rows; ++i)
	{
		KDPoint ref(ref_image_descriptors.row(i).data, i);
		KDPoint nearest = kd_tree.nearest(ref);
		result.emplace_back(std::make_pair(ref.get_descriptor_index(), nearest.get_descriptor_index()));
	}
}

size_t
StitcherHelper::countInliersNumber(const std::vector<std::pair<size_t, size_t>>& matching_pairs,
										  const ImageDescribed& reference_image, const ImageDescribed& new_input_image,
										  const Eigen::Matrix3d& transformation, size_t epsilon_squared)
{
	cv::Point_<double > reference_point, new_input_point;
	Eigen::Vector3d reference_v, new_input_v, transformed;
	size_t result = 0;

	for (const std::pair<size_t, size_t>& matching_pair : matching_pairs)
	{
		reference_point = std::get<PartOfImageDescribed::KEYPOINTS>(reference_image)[matching_pair.first].pt;
		new_input_point = std::get<PartOfImageDescribed::KEYPOINTS>(new_input_image)[matching_pair.second].pt;
		reference_v << reference_point.x, reference_point.y, 1;
		new_input_v << new_input_point.x, new_input_point.y, 1;
		transformed = transformation * new_input_v;
		if (euclidianDistanceSquared(reference_v, transformed) < epsilon_squared)
			++result;
	}
	return result;
}

bool
StitcherHelper::areCollinear(const Eigen::Matrix3d& vectors, const Eigen::Matrix2d& covariance_eigenvectors,
                            const Eigen::Vector2d& threshold)
{
	auto isGreater = [](const Eigen::Vector2d & first, const Eigen::Vector2d & second)
	{
		return first(0) > second(0) && first(1) > second(1);
 	};

	auto covariance_eigenvectors_inverse = covariance_eigenvectors.inverse();

	Eigen::Vector2d first(vectors.col(0)(0), vectors.col(0)(1));
	Eigen::Vector2d second(vectors.col(1)(0), vectors.col(1)(1));
	Eigen::Vector2d third(vectors.col(2)(0), vectors.col(2)(1));

	Eigen::Vector2d side_a = first - second;
	Eigen::Vector2d side_b = first - third;

	if (isGreater(covariance_eigenvectors_inverse * side_a, threshold))
		return true;

	Eigen::Vector2d distance = side_b - side_a.normalized() * (side_a.dot(side_b));

	if (isGreater(covariance_eigenvectors_inverse * distance, threshold))
		return true;

	return false;
}

Eigen::Matrix2d
StitcherHelper::getCovarianceMatrix(const KeypointsVector &keypoints)
{
	Eigen::Matrix2d result;
	std::vector<double > x_data, y_data;
	for (const auto& keypoint : keypoints)
	{
		cv::Point2f coordinates = keypoint.pt;
		x_data.emplace_back(coordinates.x);
		y_data.emplace_back(coordinates.y);
	}

	double mean_x = mean(x_data);
	double mean_y = mean(y_data);

	Eigen::Vector2d mass_center_xy(mean_x, mean_y);
	Eigen::Vector2d mass_center_x(mean_x, mean_x);
	Eigen::Vector2d mass_center_y(mean_y, mean_y);

	double covariance_x_y = covariance(x_data, y_data, mass_center_xy);
	double covariance_xx = covariance(x_data, x_data, mass_center_x);
	double covariance_yy = covariance(y_data, y_data, mass_center_y);

	result << covariance_xx, covariance_x_y,
	          covariance_x_y, covariance_yy;

	return result;
}

double
StitcherHelper::covariance(const std::vector<double > &x,
								  const std::vector<double >& y,
								  const Eigen::Vector2d & mean)
{
	double result = 0;
	size_t n_of_points = x.size();
	for (size_t i = 0; i < n_of_points; ++i)
	{
		result += (x[i] - mean(0)) * (y[i] - mean(1));
	}
	result /= static_cast<double>(n_of_points - 1);
	return result;
}

Eigen::Matrix2d
StitcherHelper::getEigenVectors(const Eigen::Matrix2d& matrix)
{
	Eigen::Matrix2d result;
	Eigen::EigenSolver<Eigen::Matrix2d> eigen_solver(matrix);
	if (eigen_solver.info() != Eigen::Success)
	{
		std::cout << "bad eigen solver: " << eigen_solver.info() << std::endl;
		return result;
	}
	auto eigenvectors = eigen_solver.eigenvectors();
	const auto& eigenvalues = eigen_solver.eigenvalues();
	result << eigenvectors.col(0).real() * eigenvalues(0).real(),
			eigenvectors.col(1).real() * eigenvalues(1).real();
	return result;
}

double
StitcherHelper::mean(const std::vector<double >& values)
{
	double result = 0;
	for (double value : values)
		result += value;
	return result / values.size();
}

cv::Mat
StitcherHelper::warp_to_right(cv::Mat& image1, cv::Mat& image2, Eigen::Matrix3d& eigen_homography)
{
	cv::Mat result;
	cv::Mat cv_homography;
	cv::eigen2cv(eigen_homography, cv_homography);
	double result_height = image1.rows;

	cv::warpPerspective(image2, result, cv_homography,cv::Size(image1.cols + image2.cols, result_height));

	cv::Mat roi_for_image1(result,cv::Rect(0, 0, image1.cols, image1.rows));
	image1.copyTo(roi_for_image1);

	// crop result
	Eigen::Vector3d top_right(image2.cols, 0, 1);
	Eigen::Vector3d low_right(image2.cols, image2.rows,1);
	top_right = eigen_homography * top_right;
	low_right = eigen_homography * low_right;

	double right_bound = std::fminf(top_right(0), low_right(0));
	cv::Rect roi_for_crop(0, 0, right_bound, result_height);
	cv::Mat cropped_result = result(roi_for_crop);

	return cropped_result;
}
