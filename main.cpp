#include "include/Stitching/Stitcher.h"
#include "src/KDTree.h"

#include <iostream>

int main()
{
    std::vector<std::string> image_paths;
    image_paths.emplace_back("../images/piano1.jpg");
    image_paths.emplace_back("../images/piano2.jpg");
    Stitcher stitcher(image_paths);
    auto transformationMatrix = stitcher.computeTransformationMatrix(0, 1);
    std::cout << transformationMatrix << std::endl;
	std::cout << "resulting matrix det: " << transformationMatrix.determinant() << std::endl;

	auto warped = stitcher.warp(0, 1, transformationMatrix);

	imwrite("../images/final_result1.jpg", warped);


	return 0;
}

// TODO: filter sift features