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
	std::cout << "affine matrix det: " << transformationMatrix.determinant() << std::endl;

    return 0;
}

// TODO: filter sift features