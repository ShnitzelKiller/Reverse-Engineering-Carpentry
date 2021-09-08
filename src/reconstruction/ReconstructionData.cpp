//
// Created by James Noeckel on 3/15/20.
//

#include "ReconstructionData.h"
#include <iomanip>
#include <dirent.h>
#include <utils/sorted_data_structures.hpp>
#include "parsing.hpp"
#include "geometry/primitives3/intersect_planes.h"
#include "geometry/find_affine.h"

void swapBinary(std::uint32_t &value) {
    std::uint32_t tmp = ((value << 8) & 0xFF00FF00) | ((value >> 8) & 0xFF00FF);
    value = (tmp << 16) | (tmp >> 16);
}

bool ReconstructionData::export_image_resolutions(const std::string &filename) const {
    std::ofstream of(filename);
    if (of) {
        std::vector<int32_t> imageIds;
        imageIds.reserve(images.size());
        for (const auto &pair : images) {
            imageIds.push_back(pair.first);
        }
        std::sort(imageIds.begin(), imageIds.end());
        for (auto imageId : imageIds) {
            const auto &camera = cameras.find(images.find(imageId)->second.camera_id_)->second;
            of << imageId << " " << camera.width_ << " " << camera.height_ << std::endl;
        }
    } else {
        return false;
    }
    return true;
}

bool ReconstructionData::load_bundler_file(const std::string &filename, const std::string &depth_path) {
    std::string ending = filename.substr(filename.rfind('.') + 1);
    std::transform(ending.begin(), ending.end(), ending.begin(),
                   [](unsigned char c) -> unsigned char { return std::tolower(c); });
    if (ending != "out") {
        std::cerr << "expected .out file, got ." << ending << std::endl;
        return false;
    }

    //list depth map filenames to associate with images
    std::vector<std::string> depth_filenames;
    DIR *dir;
    struct dirent *ent;
    if (!depth_path.empty()) {
        if ((dir = opendir(depth_path.c_str())) != nullptr) {
            /* print all the files and directories within directory */
            while ((ent = readdir(dir)) != nullptr) {
                std::string d_name_str(ent->d_name);
                if (d_name_str.rfind(".exr") != std::string::npos) {
                    depth_filenames.push_back(std::move(d_name_str));
                }
            }
            closedir(dir);
        } else {
            /* could not open directory */
            std::cerr << "WARNING: could not open depth map directory " << depth_path << std::endl;
        }
    }

    std::string path = std::string(filename);
    size_t delim_pos = path.rfind('/');
    if (delim_pos != std::string::npos) {
        path = path.substr(0, delim_pos + 1);
    } else {
        path = "";
    }

    std::ifstream fs(filename);
    if (!fs) {
        std::cerr << "file not found " << filename << std::endl;
        return false;
    }
    fs.ignore(1000,'\n');
    int num_cameras, num_points;
    if (!(fs >> num_cameras >> num_points)) {
        std::cerr << "invalid header" << std::endl;
        return false;
    }
    if (num_cameras != depth_filenames.size()) {
        std::cerr << "WARNING: number of views and number of depth maps does not match (" << num_cameras << " vs " << depth_filenames.size() << ")" << std::endl;
    }
    std::sort(depth_filenames.begin(), depth_filenames.end());
    //need to change from a -Z forward camera to a Z forward camera
    Eigen::Matrix3d camera_basis_transform;
    camera_basis_transform << 1, 0, 0,
                                0, -1, 0,
                                0, 0, -1;
    Eigen::Matrix3d world_basis_transform;
    world_basis_transform << 1, 0, 0,
                             0, 0, -1,
                             0, 1, 0;
    for (int i=0; i<num_cameras; i++) {
        CameraIntrinsics camera;
        camera.params_.resize(4);
        double fdist, k1, k2;
        Eigen::Matrix3d rot;
        Image imageData;
        if (
        !(fs >> fdist >> k1 >> k2)
        || !(fs >> rot(0, 0) >> rot(0, 1) >> rot(0, 2) >> rot(1, 0) >> rot(1, 1) >> rot(1, 2) >> rot(2, 0) >> rot(2, 1) >> rot(2,2))
        || !(fs >> imageData.trans_(0) >> imageData.trans_(1) >> imageData.trans_(2)))
        {
            std::cerr << "invalid camera " << i << std::endl;
            return false;
        }
        if (k1 != 0 || k2 != 0) {
            std::cerr << "distortion unsupported" << std::endl;
            return false;
        }

        //read image for metadata
        std::ostringstream imname;
        imname << std::setw(5) << std::setfill('0') << i << ".png";
        unsigned int width, height;
        {
            size_t size;
            std::unique_ptr<const char[]> data = read_file(path + imname.str(), size, 0, 24);
            if (size == 0) {
                std::cerr << "missing image " << imname.str() << std::endl;
                return false;
            }
            unsigned char png_header[] = {137, 80, 78, 71, 13, 10, 26, 10};
            if (memcmp(data.get(), png_header, 8) != 0) {
                std::cerr << "not a PNG file!" << std::endl;
                return false;
            }
            unsigned char ihdr_name[] = "IHDR";
            if (memcmp(data.get() + 8 + 4, ihdr_name, 4) != 0) {
                std::cerr << "invalid PNG file" << std::endl;
                return false;
            }

            read_object(data.get()+24-8, width);
            read_object(data.get()+24-4, height);
            swapBinary(width);
            swapBinary(height);
        }

        imageData.image_name_ = imname.str();
        imageData.depth_name_ = i < depth_filenames.size() ? depth_filenames[i] : "";
        imageData.rot_ = Eigen::Quaterniond(camera_basis_transform * rot * world_basis_transform);
        imageData.trans_ = camera_basis_transform * imageData.trans_;
        imageData.image_path_ = path;
        imageData.depth_path_ = depth_path;
        imageData.camera_id_ = i+1;
        std::cout << "Image " << i << " (" << width << 'x' << height << "): " << imageData << std::endl;

        //set camera parameters
        camera.params_[0] = camera.params_[1] = fdist;
        camera.params_[2] = static_cast<double>(width)/2;
        camera.params_[3] = static_cast<double>(height)/2;
        camera.width_ = width;
        camera.height_ = height;
        camera.model_id_ = 1;
        images[i+1] = std::move(imageData);
        cameras[i+1] = std::move(camera);

    }

    for (int i=0; i<num_points; i++) {
        Point3D point;
        size_t view_list_len;
        if (!(fs >> point.xyz_(0) >> point.xyz_(1) >> point.xyz_(2)
        >> ((int&)point.rgb_(0)) >> ((int&)point.rgb_(1)) >> ((int&)point.rgb_(2))
        >> view_list_len)) {
            std::cerr << "failed to parse point " << i << std::endl;
            return false;
        }
            //std::cout << "xyz: " << point.xyz_.transpose() << std::endl;
            //std::cout << "rgb: " << point.rgb_.transpose() << std::endl;
        point.xyz_ = Eigen::Vector3d(point.xyz_.x(), point.xyz_.z(), -point.xyz_.y());
        for (int j=0; j<view_list_len; j++) {
            int cam_index, sift_index;
            double x, y;
            if (!(fs >> cam_index >> sift_index >> x >> y)) {
                std::cerr << "failed to parse track " << j << " of point " << i << std::endl;
                return false;
            }
            cam_index += 1;
            //std::cout << "x: " << x << " y: " << y << " cam_index: " << cam_index << std::endl;
            //transform point into image coordinates
            x += static_cast<double>(cameras[cam_index].width_)/2;
            y += static_cast<double>(cameras[cam_index].height_)/2;
            auto &image = images[cam_index];
            point.image_ids_.push_back(cam_index);
            point.point2D_idxs_.push_back(image.point3D_ids_.size());
            image.xys_.emplace_back(x, cameras[cam_index].height_-y);
            image.point3D_ids_.push_back(i);
        }

        points[i] = std::move(point);
    }

    return true;
}

bool ReconstructionData::load_colmap_reconstruction(const std::string &filename, const std::string &image_path, const std::string &depth_path) {
    std::string path = filename;
    if (!path.empty() && *(path.end()-1) != '/') {
        path.push_back('/');
    }

    std::string cameras_filename = path + "cameras.bin";
    std::string images_filename = path + "images.bin";
    std::string points_filename = path + "points3D.bin";

    cameras = CameraIntrinsics::parse_file(cameras_filename);
    images = Image::parse_file(images_filename, image_path, depth_path);
    points = Point3D::parse_file(points_filename);
    return true;
}

bool ReconstructionData::export_colmap_reconstruction(const std::string &filename) const {
    std::string path = filename;
    if (!path.empty() && *(path.end()-1) != '/') {
        path.push_back('/');
    }
    std::string cameras_filename = path + "cameras.txt";
    std::string images_filename = path + "images.txt";
    std::string points_filename = path + "points3D.txt";

    {
        std::ofstream f(cameras_filename);
        std::vector<uint64_t> keys;
        keys.reserve(cameras.size());
        for (const auto &pair : cameras) {
            keys.push_back(pair.first);
        }
        std::sort(keys.begin(), keys.end());
        for (auto key : keys) {
            auto it = cameras.find(key);
            if (it->second.model_id_ != 1) {
                std::cout << "unsupported camera type" << std::endl;
                return false;
            }
            f << key << " PINHOLE ";
            f << it->second.width_ << " " << it->second.height_ << " ";
            f << it->second.params_[0] << " " << it->second.params_[1] << " " << it->second.params_[2] << " " << it->second.params_[3] << std::endl;
        }
    }
    {
        std::ofstream f(images_filename);
        std::vector<uint64_t> keys;
        keys.reserve(images.size());
        for (const auto &pair : images) {
            keys.push_back(pair.first);
        }
        std::sort(keys.begin(), keys.end());
        for (auto key : keys) {
            auto it = images.find(key);
            f << key << " ";
            f << it->second.rot_.w() << " " << it->second.rot_.x() << " " << it->second.rot_.y() << " " << it->second.rot_.z() << " ";
            f << it->second.trans_.x() << " " << it->second.trans_.y() << " " << it->second.trans_.z() << " ";
            f << it->second.camera_id_ << " ";
            f << it->second.image_path_ + it->second.image_name_ << std::endl;
            for (size_t p=0; p<it->second.xys_.size(); ++p) {
                f << it->second.xys_[p].x() << " " << it->second.xys_[p].y() << " " << it->second.point3D_ids_[p];
                if (p < it->second.xys_.size()-1) {
                    f << " ";
                }
            }
            f << std::endl;
        }
    }
    {
        std::ofstream f(points_filename);
        std::vector<uint64_t> keys;
        keys.reserve(points.size());
        for (const auto &pair : points) {
            keys.push_back(pair.first);
        }
        std::sort(keys.begin(), keys.end());
        for (auto key : keys) {
            auto it = points.find(key);
            f << key << " ";
            f << it->second.xyz_.x() << " " << it->second.xyz_.y() << " " << it->second.xyz_.z() << " ";
            f << (int)it->second.rgb_.x() << " " << (int)it->second.rgb_.y() << " " << (int)it->second.rgb_.z() << " ";
            f << it->second.error_ << " ";
            for (size_t p=0; p<it->second.image_ids_.size(); ++p) {
                f << it->second.image_ids_[p] << " " << it->second.point2D_idxs_[p];
                if (p < it->second.image_ids_.size()-1) {
                    f << " ";
                }
            }
            f << std::endl;
        }
    }
    return true;
}

bool ReconstructionData::export_rhino_camera(const std::string &filename) const {
    std::ofstream f(filename);
    if (f) {
        for (const auto &imagePair : images) {
            f << "view " << imagePair.first << ": (" << imagePair.second.image_name_ << ')' << std::endl;
            const auto &camera = cameras.find(imagePair.second.camera_id_)->second;
            double width = camera.width_;
            double height = camera.height_;
            double lens = camera.params_[0];
            double scale = 500 / width;
            width *= scale;
            height *= scale;
            lens *= scale;
            f << "dims: " << width << " " << height << std::endl;
            f << "lens: " << lens << std::endl;
            Eigen::Vector3d origin = imagePair.second.origin();
            Eigen::Vector3d direction = imagePair.second.direction();
            f << "camera position: " << origin.transpose() << std::endl;
            f << "camera lookAt: " << (origin + direction).transpose() << std::endl;
            Eigen::Vector3d up(0, 0, 1);
            Eigen::Vector3d u = direction.cross(up);
            Eigen::Vector3d uCam = imagePair.second.rot_ * u;
//            f << "uCam: " << uCam.transpose() << std::endl;
            double ang = std::atan2(uCam.y(), uCam.x());
            f << "roll: " << ang/M_PI*180 << std::endl;
        }
    } else {
        return false;
    }
    return true;
}

Eigen::MatrixX2d ReconstructionData::project(const Eigen::Ref<const Eigen::MatrixX3d> &world_points, int32_t image_id) const {
    int N = world_points.rows();
    const auto &image = images.find(image_id)->second;
    const auto &camera = cameras.find(image.camera_id_)->second;
    Eigen::Array2d fdist(camera.params_.data());
    Eigen::Array2d principal_point(camera.params_.data() + 2);
    Eigen::Array3Xd points_proj(3, N);
    for (int i=0; i<N; i++) {
        points_proj.col(i) = image.rot_ * world_points.row(i).transpose() + image.trans_;
    }
    Eigen::Array2Xd points_pix =
            (((points_proj.block(0, 0, 2, N).rowwise() / points_proj.row(2)).colwise() * fdist).colwise() +
             principal_point);
    if (image.scale_ != 1) {
        points_pix *= image.scale_;
    }
    return points_pix.transpose();
}

Eigen::RowVector2d ReconstructionData::directionalDerivative(const Eigen::Ref<const Eigen::Vector3d> &loc, const Eigen::Ref<const Eigen::Vector3d> &dir, int32_t image_id) const {
    const auto &image = images.find(image_id)->second;
    const auto &camera = cameras.find(image.camera_id_)->second;
    Eigen::Array2d fdist(camera.params_.data());
    Eigen::Array2d principal_point(camera.params_.data() + 2);
    Eigen::Vector3d loc_cam = image.rot_ * loc + image.trans_;
    /** jacobian */
    Eigen::Matrix<double, 2, 3> D;
    double z2 = loc_cam.z() * loc_cam.z();
    D << Eigen::RowVector3d(1.0/loc_cam.z(), 0, -loc_cam.x()/z2),
        Eigen::RowVector3d(0, 1.0/loc_cam.z(), -loc_cam.y()/z2);
    Eigen::Quaterniond rotinv = image.rot_.conjugate();
    D.row(0) = (rotinv * D.row(0).transpose()).transpose();
    D.row(1) = (rotinv * D.row(1).transpose()).transpose();
    return ((D * dir).array() * fdist).transpose() * image.scale_;
}

Eigen::Vector3d ReconstructionData::initRay(double i, double j, int32_t image_id) const {
    const auto &image = images.find(image_id)->second;
    const auto &camera = cameras.find(image.camera_id_)->second;
    Eigen::Array2d fdist(camera.params_.data());
    Eigen::Array2d principal_point(camera.params_.data() + 2);
    Eigen::Array3d dir(j, i, 1);
    dir.head(2) /= image.scale_;
    dir.head(2) -= principal_point;
    dir.head(2) /= fdist;
    return (image.rot_.conjugate() * dir).normalized();
}

void ReconstructionData::clearImages() {
    for (auto &pair : images) {
        pair.second.clearImages();
    }
}

void ReconstructionData::setImageScale(double scale) {
    for (auto &pair : images) {
        if (pair.second.scale_ != scale) {
            pair.second.scale_ = scale;
            pair.second.clearImages();
        }
    }
}

Eigen::Vector2d ReconstructionData::epipolar_line(double i, double j, int32_t image1_id, int32_t image2_id) const {
    Eigen::Vector3d origin2 = images.find(image2_id)->second.origin();
    Eigen::Vector2d projected_origin2 = project(origin2.transpose(), image1_id).transpose();
    return (Eigen::Vector2d(j, i) - projected_origin2).normalized();
}

Eigen::Vector2d ReconstructionData::resolution(int32_t image_id) const {
    const auto &image = images.find(image_id)->second;
    const auto &camera = cameras.find(image.camera_id_)->second;
    return {camera.width_ * image.scale_, camera.height_ * image.scale_};
}

void ReconstructionData::findAffineTransform(ReconstructionData &other, double &scale, Eigen::Quaterniond& rot, Eigen::Vector3d &trans) {
    auto matches = correspondingImages(other);
    std::cout << "number of matching views: " << matches.size() << std::endl;

    using namespace Eigen;
    MatrixX3d P(matches.size(), 3);
    MatrixX3d Q(matches.size(), 3);
    size_t ind=0;
    for (const auto &pair : matches) {
        P.row(ind) = images[pair.first].origin().transpose();
        Q.row(ind) = other.images[pair.second].origin().transpose();
        ++ind;
    }
    std::cout << "P: " << std::endl << P << std::endl;
    std::cout << "Q: " << std::endl << Q << std::endl;
    find_affine(P, Q, scale, trans, rot);

    //evaluate*/
    double totalError = 0.0;
    for (const auto &pair : matches) {
        Eigen::Vector3d pt1trans = scale * (rot * images[pair.first].origin()) + trans;
        double errorSquared = (other.images[pair.second].origin() - pt1trans).squaredNorm();
        std::cout << "error for image " << pair.first << ": " << std::sqrt(errorSquared) << std::endl;
        totalError += errorSquared;
    }
    //double totalError = ((scale * (P * W)).rowwise() + trans.transpose() - Q).rowwise().squaredNorm().sum();
    std::cout << "total error squared: " << totalError << std::endl;
}

std::unordered_map<int32_t, int32_t> ReconstructionData::correspondingImages(const ReconstructionData &other) {
    std::vector<std::pair<std::string, int32_t>> otherNames;
    otherNames.reserve(other.images.size());
    for (const auto &pair : other.images) {
        otherNames.emplace_back(pair.second.image_name_, pair.first);
    }
    std::sort(otherNames.begin(), otherNames.end());
    std::unordered_map<int32_t, int32_t> matches;
    for (const auto &pair : images) {
        auto it = sorted_find(otherNames, pair.second.image_name_);
        if (it != otherNames.end()) {
            std::cout << "match " << pair.first << " (" << pair.second.image_name_ << ") with " << it->second << " (" << other.images.find(it->second)->second.image_name_ << ")" << std::endl;
            matches[pair.first] = it->second;
        } /*else {
            matches[pair.first] = -1;
        }*/
    }
    return matches;
}
