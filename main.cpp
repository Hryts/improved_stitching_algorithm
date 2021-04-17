#include "include/Stitching/Stitcher.h"
#include "src/KDTree.h"

#include <iostream>

int main()
{
    std::vector<std::string> image_paths;
    image_paths.emplace_back("../images/p1.jpg");
    image_paths.emplace_back("../images/p2.jpg");
    image_paths.emplace_back("../images/p3.jpg");
    image_paths.emplace_back("../images/p4.jpg");
    image_paths.emplace_back("../images/p5.jpg");
//    image_paths.emplace_back("../images/buz/buz1.jpg");
//    image_paths.emplace_back("../images/buz/buz2.jpg");
//    image_paths.emplace_back("../images/buz/buz3.jpg");
//    image_paths.emplace_back("../images/buz/buz4.jpg");
//    image_paths.emplace_back("../images/buz/buz5.jpg");

    Stitcher stitcher(image_paths);

	auto warped = stitcher.stitchAll();


	imwrite("../images/final_result.jpg", warped);
	return 0;
}

// TODO: filter sift features