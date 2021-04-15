//
// Created by hryts on 19.03.21.
//

#include "../include/Stitching/Stitcher.h"
#include "LMOptimizer.h"

#include <Eigen/LU>
#include <opencv2/core/eigen.hpp>

Stitcher::Stitcher(const std::vector<cv::Mat>& images) : m_sift(cv::SIFT::create(cv::Stitcher::PANORAMA))
{
    for (const auto& image : images)
        addImage(image);
}

Stitcher::Stitcher(const std::vector<std::string> &paths) : m_sift(cv::SIFT::create(cv::Stitcher::PANORAMA))
{
    for (const auto& path : paths)
        addImage(cv::imread(path, cv::IMREAD_UNCHANGED));
}

void Stitcher::addImage(const cv::Mat &image)
{
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    m_sift->detect(image, keypoints);
    m_sift->compute(image, keypoints, descriptors);
    m_described_images.emplace_back(image, keypoints, descriptors);
}

size_t euclidianDistanceSquared(const Eigen::Vector3d& point1, const Eigen::Vector3d& point2)
{
	double result = 0;
	size_t dim = 3;
	for (size_t i = 0; i < dim; ++i) {
		result += pow((point1(i) - point2(i)), 2);
	}
	return result;
}

Eigen::Matrix3d Stitcher::computeTransformationMatrix(const size_t image1, const size_t image2)
{
	m_matching_pairs.clear();
    Eigen::Matrix3d result;
    getAffineTransformation(image1, image2, result);
//	result << 1.01008, -8.87567e-05, 153.888, 0.0122352, 1.0044, -7.28397, 0, 0, 1;

//    std::cout << "affine matrix: " << std::endl;
//    std::cout << result << std::endl;
    std::cout << "affine det: " << result.determinant();

    std::cout << std::endl << "\t\tn of pairs: " << m_matching_pairs.size() << std::endl;

	std::vector<std::pair<Eigen::Vector2d , Eigen::Vector2d >> measured_values;
	auto ref_image_keypoints = std::get<ImageDescribedGetter::KEYPOINTS>(m_described_images[image1]);
	auto inp_image_keypoints = std::get<ImageDescribedGetter::KEYPOINTS>(m_described_images[image2]);

	for (const auto& matching_pair : m_matching_pairs)
	{
		auto ref_coordinates = ref_image_keypoints[matching_pair.first].pt;
		auto inp_coordinates = inp_image_keypoints[matching_pair.second].pt;
		Eigen::Vector3d reference_v, new_input_v;
		reference_v << ref_coordinates.x, ref_coordinates.y, 1;
		new_input_v << inp_coordinates.x, inp_coordinates.y, 1;
		Eigen::Vector3d transformed = result * new_input_v;
		if (euclidianDistanceSquared(reference_v, transformed) < 100)
		{
			measured_values.emplace_back(
					std::make_pair(
							Eigen::Vector2d (ref_coordinates.x, ref_coordinates.y),
							Eigen::Vector2d (inp_coordinates.x, inp_coordinates.y)
					)
			);
		}
	}

	Eigen::VectorXd parameters(9);
	parameters << result(0,0), result(1, 0), result(2, 0),
			      result(0,1), result(1, 1), result(2, 1),
				  result(0,2), result(1, 2), result(2, 2);


	LMOptimizer lm_optimizer(measured_values, parameters);

	result = lm_optimizer.optimize();

	std::cout << "measured_values.size(): " << measured_values.size() << " \t\tnew number of inliers: "
	<< countInliersNumber(m_matching_pairs,
	                      m_described_images[image1],
	                      m_described_images[image2],
					      result)
	<< "\t det: " << result.determinant() << std::endl;

    return result;
}

void Stitcher::getAffineTransformation(const size_t reference_image_index, const size_t new_input_image_index,
									   Eigen::Matrix3d& result,
									   size_t number_of_iterations, size_t number_of_matching_pairs)
{
	const Eigen::Vector2d KCollinearityThreshold(.5f, .5f);

	std::set<int> random_indexes;
	Eigen::Matrix3d left, right, candidate;
	size_t max_number_of_inliers = 0;
	size_t current_number_of_inliers;
	size_t current_iteration;
	int random_index;

	ImageDescribed reference_image = m_described_images[reference_image_index];
	ImageDescribed new_input_image = m_described_images[new_input_image_index];

	auto ref_image_keypoints = std::get<ImageDescribedGetter::KEYPOINTS>(reference_image);
	auto inp_image_keypoints = std::get<ImageDescribedGetter::KEYPOINTS>(new_input_image);
	std::cout << "ref_keypoints: " << ref_image_keypoints.size() << " inp_keypoints: " << inp_image_keypoints.size() << std::endl;

	getMatchingPairs(std::get<ImageDescribedGetter::DESCRIPTORS>(reference_image),
	                 std::get<ImageDescribedGetter::DESCRIPTORS>(new_input_image),
	                 m_matching_pairs);

	Eigen::Matrix2d ref_covariance_matrix_eigenvectors = getEigenVectors(getCovarianceMatrix(ref_image_keypoints));

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> distribution(0, m_matching_pairs.size() - 1);

//	drawKeypoints(0, 1, m_matching_pairs);

	for (size_t i = 0; i < number_of_iterations; ++i)
    {
	    random_indexes.clear();
	    while (random_indexes.size() < number_of_matching_pairs)
        {
	        random_index = distribution(gen);

	        if (!random_indexes.insert(random_index).second) // random_index value is already in random_indexes
		        continue;

	        current_iteration = random_indexes.size();
	        auto random_pair = m_matching_pairs[random_index];

	        auto ref_coordinates = ref_image_keypoints[random_pair.first].pt;
	        auto inp_coordinates = inp_image_keypoints[random_pair.second].pt;

	        left.col(current_iteration-1) << ref_coordinates.x, ref_coordinates.y, 1;
	        right.col(current_iteration-1) << inp_coordinates.x, inp_coordinates.y, 1;

	        if (current_iteration != number_of_matching_pairs)
		        continue;

	        candidate = left * right.inverse();
	        auto det = candidate.determinant();

	        if (areCollinear(right, ref_covariance_matrix_eigenvectors, KCollinearityThreshold) ||
	            det < .5f || det > 2.0f)
              random_indexes.clear();
        }

	    current_number_of_inliers = countInliersNumber(m_matching_pairs, reference_image, new_input_image, candidate);

	    if(current_number_of_inliers > max_number_of_inliers) // TODO: continue
        {
            max_number_of_inliers = current_number_of_inliers;
            result = candidate;
        }
    }
	std::cout << "maximum number of inliers: " << max_number_of_inliers << std::endl;
}

void Stitcher::getMatchingPairs(const cv::Mat& ref_image_descriptors,
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

size_t Stitcher::countInliersNumber(const std::vector<std::pair<size_t, size_t>>& matching_pairs,
                                    const ImageDescribed& reference_image, const ImageDescribed& new_input_image,
                                    const Eigen::Matrix3d& transformation, size_t epsilon_squared)
{
	cv::Point_<double > reference_point, new_input_point;
	Eigen::Vector3d reference_v, new_input_v, transformed;
	size_t result = 0;

	for (const auto& matching_pair : matching_pairs)
	{
		reference_point = std::get<ImageDescribedGetter::KEYPOINTS>(reference_image)[matching_pair.first].pt;
		new_input_point = std::get<ImageDescribedGetter::KEYPOINTS>(new_input_image)[matching_pair.second].pt;
		reference_v << reference_point.x, reference_point.y, 1;
		new_input_v << new_input_point.x, new_input_point.y, 1;
		transformed = transformation * new_input_v;
		if (euclidianDistanceSquared(reference_v, transformed) < epsilon_squared)
			++result;
	}
    return result;
}


cv::Mat Stitcher::drawKeypoints(size_t ind1, size_t ind2, std::vector<std::pair<size_t, size_t>> matching_pairs)
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> distribution(0, matching_pairs.size() - 1);
	int random_index = distribution(gen);

    cv::Mat res_l, res_r;
    auto l = m_described_images[ind1];
    auto r = m_described_images[ind2];
    auto keys_l = std::get<ImageDescribedGetter::KEYPOINTS>(l);
    auto keys_r = std::get<ImageDescribedGetter::KEYPOINTS>(r);
    std::vector<cv::KeyPoint> left_match, right_match;
    left_match.emplace_back(keys_l[matching_pairs[random_index].first]);
    right_match.emplace_back(keys_r[matching_pairs[random_index].second]);
    cv::drawKeypoints(std::get<ImageDescribedGetter::IMAGE>(l), left_match, res_l, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	cv::drawKeypoints(std::get<ImageDescribedGetter::IMAGE>(r), right_match, res_r, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	std::cout << left_match[0].pt.x << ", " << left_match[0].pt.y << std::endl;
	std::cout << right_match[0].pt.x << ", " << right_match[0].pt.y << std::endl;
	std::cout << random_index << std::endl;
	imwrite("../images/left_match.jpg", res_l);
	imwrite("../images/right_match.jpg", res_r);
	return res_l;
}

Eigen::Matrix2d Stitcher::getCovarianceMatrix(const std::vector<cv::KeyPoint> &keypoints)
{
	Eigen::Matrix2d result;
	std::vector<double > x_data, y_data;
	for (const auto& keypoint : keypoints)
	{
		auto coordinates = keypoint.pt;
		x_data.emplace_back(coordinates.x);
		y_data.emplace_back(coordinates.y);
	}
	double mean_x = mean(x_data);
	double mean_y = mean(y_data);
	Eigen::Vector2d mass_center_both(mean_x, mean_y);
	Eigen::Vector2d mass_center_x(mean_x, mean_x);
	Eigen::Vector2d mass_center_y(mean_y, mean_y);
	double covariance_x_y = Stitcher::covariance(x_data, y_data, mass_center_both);
	result << Stitcher::covariance(x_data, x_data, mass_center_x), covariance_x_y,
			  covariance_x_y, Stitcher::covariance(y_data, y_data, mass_center_y);
	return result;
}

Eigen::Matrix2d Stitcher::getEigenVectors(const Eigen::Matrix2d& matrix)
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

double Stitcher::covariance(const std::vector<double > &x, const std::vector<double >& y, const Eigen::Vector2d & mean)
{
	double result = 0;
	size_t n_of_points = x.size();
	for (size_t i = 0; i < n_of_points; ++i)
	{
		result += (x[i] - mean(0)) * (y[i] - mean(1));
	}
	result /= (n_of_points - 1);
	return result;
}

double Stitcher::mean(const std::vector<double >& values)
{
	double result = 0;
	for (double value : values)
		result += value;
	return result / values.size();
}

bool Stitcher::areCollinear(const Eigen::Matrix3d& vectors, const Eigen::Matrix2d& covariance_eigenvectors,
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

	if (isGreater(covariance_eigenvectors_inverse *side_a, threshold))
		return true;

	Eigen::Vector2d distance = side_b - side_a.normalized() * (side_a.dot(side_b));

	if (isGreater(covariance_eigenvectors_inverse * distance, threshold))
		return true;

	return false;
}

cv::Mat Stitcher::warp(cv::Mat& image1, cv::Mat& image2, Eigen::Matrix3d& eigen_homography)
{
	imwrite("../images/debug.jpg", image1);

	cv::Mat result;
	cv::Mat cv_homography;
	cv::eigen2cv(eigen_homography, cv_homography);
	double result_height = image1.rows;

	cv::warpPerspective(image2, result, cv_homography,cv::Size(image1.cols + image2.cols, result_height));

	cv::Mat roi_for_image1(result,cv::Rect(0, 0, image1.cols, image1.rows));
	image1.copyTo(roi_for_image1);

	imwrite("../images/debug.jpg", result);

	// crop result
	Eigen::Vector3d top_right(image2.cols, 0, 1);
	Eigen::Vector3d low_right(image2.cols,image2.rows, 1);
	top_right = eigen_homography * top_right;
	low_right = eigen_homography * low_right;

	double right_bound = std::fminf(top_right(0), low_right(0));
	cv::Rect roi_for_crop(0, 0, right_bound, result_height);
	cv::Mat cropped_result = result(roi_for_crop);

	imwrite("../images/debug.jpg", cropped_result);


	return cropped_result;
}

cv::Mat Stitcher::stitchAll()
{
	calculateTransformations();
	cv::Mat result = std::get<ImageDescribedGetter::IMAGE>(m_described_images[0]);

	for (size_t i = 1; i < m_described_images.size(); ++i) {
		cv::Mat inp = std::get<ImageDescribedGetter::IMAGE>(m_described_images[i]);
		result = warp(result, inp, m_transformations[i-1]);
	}
	return result;
}

void Stitcher::calculateAdjacentTransformations(std::vector<Eigen::Matrix3d>& result)
{
	for (size_t image_index = 1; image_index < m_described_images.size(); ++image_index)
	{
		result.emplace_back(computeTransformationMatrix(image_index - 1, image_index));
	}
}

void Stitcher::calculateTransformations()
{
	std::vector<Eigen::Matrix3d> adjacent_transformations;
	calculateAdjacentTransformations(adjacent_transformations);

	m_transformations = adjacent_transformations;

	if (m_transformations.size() == 1)
		return;

	for (size_t i = 1; i < m_transformations.size(); ++i) {
		m_transformations[i] = m_transformations[i - 1] * m_transformations[i];
	}
}

cv::Mat Stitcher::warp_public(size_t im1, size_t im2, Eigen::Matrix3d& eigen_homography)
{
	cv::Mat i1 = std::get<ImageDescribedGetter::IMAGE>(m_described_images[im1]);
	cv::Mat i2 = std::get<ImageDescribedGetter::IMAGE>(m_described_images[im2]);
	return warp(i1, i2, eigen_homography);

}