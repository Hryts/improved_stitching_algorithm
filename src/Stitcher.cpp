//
// Created by hryts on 19.03.21.
//

#include "../include/Stitching/Stitcher.h"
#include <Eigen/LU>


Stitcher::Stitcher(const std::vector<cv::Mat>& images) : m_sift(cv::SIFT::create())
{
    for (const auto& image : images)
        addImage(image);
}

Stitcher::Stitcher(const std::vector<std::string> &paths) : m_sift(cv::SIFT::create())
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

Eigen::Matrix3f Stitcher::computeTransformationMatrix(const size_t image1, const size_t image2)
{
    Eigen::Matrix3f result;
    getAffineTransformation(image1, image2, result);

    // TODO: optimization

    return result;
}

void Stitcher::getAffineTransformation(const size_t reference_image_index, const size_t new_input_image_index,
									   Eigen::Matrix3f& result,
									   size_t number_of_iterations, size_t number_of_matching_pairs)
{
    ImageDescribed reference_image = m_described_images[reference_image_index];
    ImageDescribed new_input_image = m_described_images[new_input_image_index];

    auto ref_image_descriptors = std::get<ImageDescribedGetter::DESCRIPTORS>(reference_image);
    auto inp_image_descriptors = std::get<ImageDescribedGetter::DESCRIPTORS>(new_input_image);

    auto ref_image_keypoints = std::get<ImageDescribedGetter::KEYPOINTS>(reference_image);
    auto inp_image_keypoints = std::get<ImageDescribedGetter::KEYPOINTS>(reference_image);

    std::set<int> features_set;
    Eigen::Matrix3f left, right, candidate;
    std::vector<std::pair<size_t, size_t>> matching_pairs;
    getMatchingPairs(ref_image_descriptors, inp_image_descriptors, matching_pairs);
    size_t max_number_of_inliers = 0;
    size_t current_inliers_number;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distribution(0, matching_pairs.size() - 1);
    int random_index;

    for (size_t i = 0; i < number_of_iterations; ++i)
    {
	    features_set.clear();
	    while (features_set.size() < number_of_matching_pairs)
        {
	        random_index = distribution(gen);

	        if (!features_set.insert(random_index).second) // random_index value is already in features_set
		        continue;
	        auto random_pair = matching_pairs[random_index];
	        auto ref_keypoint_coordinates = ref_image_keypoints[random_pair.first].pt;
	        auto inp_keypoint_coordinates = inp_image_keypoints[random_pair.second].pt;

	        auto current_column_index = features_set.size() - 1;
	        left.col(current_column_index) << ref_keypoint_coordinates.x, ref_keypoint_coordinates.y, 1;
	        right.col(current_column_index) << inp_keypoint_coordinates.x, inp_keypoint_coordinates.y, 1;

        }

	    // TODO: determine minimum value of determinant (because of precision with floating points)
	    if (abs(right.determinant()) < 0.001f) // || isCollinear(left) || isCollinear(right)
		    continue;

        candidate = left * right.inverse();

        current_inliers_number = countInliersNumber(matching_pairs, reference_image, new_input_image, candidate);
        if(current_inliers_number > max_number_of_inliers)
        {
            max_number_of_inliers = current_inliers_number;
            result = candidate;
        }
    }
	std::cout << "maximum number of inliers: " << max_number_of_inliers << std::endl;
}

void Stitcher::getMatchingPairs(const cv::Mat& reference_descriptors,
                                const cv::Mat& inp_image_descriptors,
                                std::vector<std::pair<size_t, size_t>>& result)
{
	KDTree kd_tree(inp_image_descriptors);
    for (int i = 0; i < reference_descriptors.rows; ++i)
    {
        KDPoint ref(reference_descriptors.row(i).data, i);
        KDPoint nearest = kd_tree.nearest(ref);
        result.emplace_back(std::make_pair(ref.get_descriptor_index(), nearest.get_descriptor_index()));
    }
}

size_t euclidianDistanceSquared(const Eigen::Vector3f& point1, const Eigen::Vector3f& point2)
{
    return pow((point1(0) - point2(0)), 2) + pow((point1(1) - point2(1)), 2);
}

size_t Stitcher::countInliersNumber(const std::vector<std::pair<size_t, size_t>>& matching_pairs,
                                    const ImageDescribed& reference_image, const ImageDescribed& new_input_image,
                                    const Eigen::Matrix3f& transformation, size_t epsilon_squared)
{
	cv::Point_<float> reference_point, new_input_point;
	Eigen::Vector3f reference_v, new_input_v, transformed;
    size_t result = 0;
    for (const auto& matching_pair : matching_pairs)
    {
        reference_point = std::get<ImageDescribedGetter::KEYPOINTS>(reference_image)[matching_pair.first].pt;
        new_input_point = std::get<ImageDescribedGetter::KEYPOINTS>(new_input_image)[matching_pair.second].pt;
        reference_v << reference_point.x, reference_point.y, 1;
        new_input_v << new_input_point.x, new_input_point.y, 1;
        transformed = transformation * new_input_v;
        result += euclidianDistanceSquared(reference_v, transformed) < epsilon_squared ? 1 : 0;
    }
    return result;
}

bool Stitcher::isCollinear(const Eigen::Matrix3f& matrix, float threshold)
{
	// vectors a, b and c are collinear if and only if (a - b) = k(a - c) for some scalar K
	// it means that dot product of normalized (a - b) and (a - c) is equal to 1
	auto v_01 = matrix.col(0) - matrix.col(1);
	auto v_02 = matrix.col(0) - matrix.col(2);
	return v_01.normalized().dot(v_02.normalized()) > threshold;
}

cv::Mat Stitcher::drawKeypoints(size_t imageIndex)
{
    cv::Mat result;
    auto tmp = m_described_images[imageIndex];
    auto keys = std::get<ImageDescribedGetter::KEYPOINTS>(tmp);
    std::vector<cv::KeyPoint> new_one;
    new_one.emplace_back(keys[100]);
    cv::drawKeypoints(std::get<ImageDescribedGetter::IMAGE>(tmp), keys, result);
    imwrite("../images/tests.jpg", result);
    return result;
}