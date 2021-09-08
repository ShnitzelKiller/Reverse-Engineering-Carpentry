//
// Created by James Noeckel on 1/29/20.
//

#include "imgproc/multilabel_graph_cut.h"
#include <opencv2/opencv.hpp>
#include "reconstruction/Image.h"

int main(int argc, char **argv) {
    std::string images_filename = "../data/birdhouse_data/images.bin";
    std::unordered_map<int32_t, Image> images = Image::parse_file(images_filename, "../data/birdhouse_data/images/resized/", "../data/birdhouse_data/depth_maps/");

    for (auto pair : images) {
        std::cout << "================== image " << pair.first << ": "<< pair.second.image_name_ <<" ==================" << std::endl;
        cv::Mat img = pair.second.getImage();
        if (img.empty()) {
            std::cout << "image not found" << std::endl;
            return 1;
        }

        cv::Mat seg = cv::imread("../data/test_data/depth_seg_" + std::to_string(pair.first) + "_outlier.png", cv::IMREAD_GRAYSCALE);
        if (seg.empty()) {
            std::cout << "segmentation mask not found" << std::endl;
            return 1;
        }
        cv::resize(img, img, seg.size());
        cv::cvtColor(img, img, cv::COLOR_BGR2Lab);
        seg.convertTo(seg, CV_32SC1);
        seg -= 1;

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    cv::Mat seg2 = seg.clone();
                    float w_d = 1 + 3 * i;
                    float w_s = 1 + 3 * j;
                    float sigma = 1 + 10 * k;
                    std::cout << "parameters: w_d=" << w_d << ", w_s=" << w_s << "sigma=" << sigma << std::endl;
                    auto start_t = clock();
                    multilabel_graph_cut(img, seg2, w_d, w_s, 100, sigma);
                    auto total_t = clock() - start_t;
                    float time_sec = static_cast<float>(total_t) / CLOCKS_PER_SEC;
                    std::cout << "segmentation completed in " << time_sec << " seconds" << std::endl;
                    cv::imwrite("test_segmentation_result_"+std::to_string(pair.first)+"_w_d=" + std::to_string(w_d) + "w_s=" + std::to_string(w_s) +
                                "sigma=" + std::to_string(sigma) + ".png", (seg2 + 1) * 20);
                }
            }
        }
    }
    return 0;
}