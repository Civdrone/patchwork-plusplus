#include <memory>
#include <utility>
#include <vector>

#include <Eigen/Core>

// TF2
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_eigen/tf2_eigen.hpp>

// Patchwork++-ROS
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/qos.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/string.hpp>
#include <civ_interfaces/msg/obstacle_state.hpp>

#include "GroundSegmentationServer.hpp"
#include "Utils.hpp"

namespace patchworkpp_ros {

using utils::EigenToPointCloud2;
using utils::GetTimestamps;
using utils::PointCloud2ToEigen;

GroundSegmentationServer::GroundSegmentationServer(const rclcpp::NodeOptions &options)
    : rclcpp::Node("patchworkpp_node", options) {
  patchwork::Params params;
  base_frame_ = declare_parameter<std::string>("base_frame", base_frame_);

  params.sensor_height = declare_parameter<double>("sensor_height", params.sensor_height);
  params.num_iter      = declare_parameter<int>("num_iter", params.num_iter);
  params.num_lpr       = declare_parameter<int>("num_lpr", params.num_lpr);
  params.num_min_pts   = declare_parameter<int>("num_min_pts", params.num_min_pts);
  params.th_seeds      = declare_parameter<double>("th_seeds", params.th_seeds);

  params.th_dist    = declare_parameter<double>("th_dist", params.th_dist);
  params.th_seeds_v = declare_parameter<double>("th_seeds_v", params.th_seeds_v);
  params.th_dist_v  = declare_parameter<double>("th_dist_v", params.th_dist_v);

  params.max_range       = declare_parameter<double>("max_range", params.max_range);
  params.min_range       = declare_parameter<double>("min_range", params.min_range);
  params.uprightness_thr = declare_parameter<double>("uprightness_thr", params.uprightness_thr);

  params.obstacle_min_height = declare_parameter<double>("obstacle_min_height", params.obstacle_min_height);
  params.obstacle_max_radius = declare_parameter<double>("obstacle_max_radius", params.obstacle_max_radius);

  fov_angle_deg_ = declare_parameter<double>("fov_angle_deg", 360.0);
  fov_angle_rad_ = fov_angle_deg_ * M_PI / 180.0;
  target_frame_ = declare_parameter<std::string>("target_frame", "");

  // Obstacle clustering parameters
  cluster_tolerance_ = declare_parameter<double>("cluster_tolerance", 0.5);
  min_cluster_size_ = declare_parameter<int>("min_cluster_size", 5);
  enable_persistent_tracking_ = declare_parameter<bool>("enable_persistent_tracking", false);
  min_frames_for_obstacle_ = declare_parameter<int>("min_frames_for_obstacle", 3);
  max_cluster_distance_ = declare_parameter<double>("max_cluster_distance", 1.0);
  current_frame_id_ = 0;

  // Initialize TF2
  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  params.verbose = get_parameter<bool>("verbose", params.verbose);

  // ToDo. Support intensity
  params.enable_RNR = false;

  // Construct the main Patchwork++ node
  Patchworkpp_ = std::make_unique<patchwork::PatchWorkpp>(params);

  // Initialize subscribers
  pointcloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      "pointcloud_topic",
      rclcpp::SensorDataQoS(),
      std::bind(&GroundSegmentationServer::EstimateGround, this, std::placeholders::_1));

  /*
   * We use the following QoS setting for reliable ground segmentation.
   * If you want to run Patchwork++ in real-time and real-world operation,
   * please change the QoS setting
   */
  //  rclcpp::QoS qos((rclcpp::SystemDefaultsQoS().keep_last(1).durability_volatile()));
  rclcpp::QoS qos(rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_default));
  qos.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
  qos.durability(RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL);

  cloud_publisher_  = create_publisher<sensor_msgs::msg::PointCloud2>("/patchworkpp/cloud", qos);
  ground_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>("/patchworkpp/ground", qos);
  nonground_publisher_ =
      create_publisher<sensor_msgs::msg::PointCloud2>("/patchworkpp/nonground", qos);
  obstacles_publisher_ =
      create_publisher<sensor_msgs::msg::PointCloud2>("/patchworkpp/obstacles", qos);
  obstacle_state_publisher_ =
      create_publisher<civ_interfaces::msg::ObstacleState>("/civ/obstacle/state", qos);

  RCLCPP_INFO(this->get_logger(), "Patchwork++ ROS 2 node initialized");
}

void GroundSegmentationServer::EstimateGround(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg) {
  const auto &cloud_raw = patchworkpp_ros::utils::PointCloud2ToEigenMat(msg);

  // Apply transformation and FOV filtering
  const auto &cloud = FilterPointCloudByFOV(cloud_raw, msg->header);

  // Estimate ground
  Patchworkpp_->estimateGround(cloud);

  // Create header with correct frame_id for transformed points
  std_msgs::msg::Header output_header = msg->header;
  output_header.frame_id = GetOutputFrameId();
  cloud_publisher_->publish(patchworkpp_ros::utils::EigenMatToPointCloud2(cloud, output_header));
  // Get ground and nonground
  Eigen::MatrixX3f ground    = Patchworkpp_->getGround();
  Eigen::MatrixX3f nonground = Patchworkpp_->getNonground();
  Eigen::MatrixX3f obstacles_raw = Patchworkpp_->getObstacles();

  // Apply clustering to filter small obstacle clusters
  Eigen::MatrixX3f obstacles_filtered = FilterObstaclesByClusterSize(obstacles_raw);

  double time_taken = Patchworkpp_->getTimeTaken();
  PublishClouds(ground, nonground, obstacles_filtered, msg->header);
}

Eigen::MatrixX3f GroundSegmentationServer::FilterPointCloudByFOV(const Eigen::MatrixX3f &cloud, const std_msgs::msg::Header &header) {
  Eigen::MatrixX3f working_cloud;

  // Apply TF2 transformation if target frame is specified
  if (!target_frame_.empty() && target_frame_ != header.frame_id) {
    try {
      // Check if transform is available (wait longer for static transforms)
      if (tf_buffer_->canTransform(target_frame_, header.frame_id,
                                   header.stamp, tf2::durationFromSec(5.0))) {

        // Get the transform
        geometry_msgs::msg::TransformStamped transform = tf_buffer_->lookupTransform(
            target_frame_, header.frame_id, header.stamp);

        // Convert to Eigen transform
        Eigen::Isometry3d eigen_transform = tf2::transformToEigen(transform);

        // Copy and transform points
        working_cloud = cloud;
        for (int i = 0; i < cloud.rows(); ++i) {
          Eigen::Vector3d point(cloud(i, 0), cloud(i, 1), cloud(i, 2));
          Eigen::Vector3d transformed_point = eigen_transform * point;
          working_cloud(i, 0) = transformed_point.x();
          working_cloud(i, 1) = transformed_point.y();
          working_cloud(i, 2) = transformed_point.z();
        }

      } else {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                             "Transform from %s to %s not available, using original points",
                             header.frame_id.c_str(), target_frame_.c_str());
      }
    } catch (tf2::TransformException &ex) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                           "Transform failed: %s, using original points", ex.what());
    }
  } else {
    // No transformation needed, use original cloud
    working_cloud = cloud;
  }

  // Apply FOV filtering on (possibly transformed) points
  if (fov_angle_deg_ >= 360.0) {
    // No filtering needed if FOV is 360 degrees or more
    return working_cloud;
  }

  std::vector<int> valid_indices;
  valid_indices.reserve(working_cloud.rows());

  const double half_fov = fov_angle_rad_ / 2.0;

  for (int i = 0; i < working_cloud.rows(); ++i) {
    const double x = working_cloud(i, 0);
    const double y = working_cloud(i, 1);

    // Calculate angle from positive x-axis in x-y plane
    const double angle = std::atan2(y, x);

    // Check if point is within FOV (symmetric around positive x-axis)
    if (std::abs(angle) <= half_fov) {
      valid_indices.push_back(i);
    }
  }

  // Create filtered cloud
  Eigen::MatrixX3f filtered_cloud(valid_indices.size(), 3);
  for (size_t i = 0; i < valid_indices.size(); ++i) {
    filtered_cloud.row(i) = working_cloud.row(valid_indices[i]);
  }

  return filtered_cloud;
}

std::string GroundSegmentationServer::GetOutputFrameId() const {
  // If we're transforming to a target frame, use that frame for output
  // Otherwise, use the base_frame
  return (!target_frame_.empty()) ? target_frame_ : base_frame_;
}

Eigen::MatrixX3f GroundSegmentationServer::FilterObstaclesByClusterSize(const Eigen::MatrixX3f &obstacles) {
  if (obstacles.rows() == 0) {
    return obstacles;
  }

  // Convert Eigen matrix to PCL point cloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  cloud->points.reserve(obstacles.rows());

  for (int i = 0; i < obstacles.rows(); ++i) {
    pcl::PointXYZ point;
    point.x = obstacles(i, 0);
    point.y = obstacles(i, 1);
    point.z = obstacles(i, 2);
    cloud->points.push_back(point);
  }
  cloud->width = cloud->points.size();
  cloud->height = 1;
  cloud->is_dense = true;

  // Create KD-Tree for efficient nearest neighbor search
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
  tree->setInputCloud(cloud);

  // Perform Euclidean clustering
  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
  ec.setClusterTolerance(cluster_tolerance_);
  ec.setMinClusterSize(min_cluster_size_);
  ec.setMaxClusterSize(obstacles.rows()); // No upper limit
  ec.setSearchMethod(tree);
  ec.setInputCloud(cloud);
  ec.extract(cluster_indices);

  if (enable_persistent_tracking_) {
    // Track persistent clusters across frames
    return TrackPersistentClusters(cluster_indices, cloud);
  } else {
    // Build filtered obstacle cloud from valid clusters (fast path)
    std::vector<int> valid_indices;
    for (const auto& cluster : cluster_indices) {
      valid_indices.insert(valid_indices.end(), cluster.indices.begin(), cluster.indices.end());
    }
    
    if (valid_indices.empty()) {
      return Eigen::MatrixX3f(0, 3);
    }
    
    // Convert back to Eigen matrix
    Eigen::MatrixX3f filtered_obstacles(valid_indices.size(), 3);
    for (size_t i = 0; i < valid_indices.size(); ++i) {
      filtered_obstacles(i, 0) = cloud->points[valid_indices[i]].x;
      filtered_obstacles(i, 1) = cloud->points[valid_indices[i]].y;
      filtered_obstacles(i, 2) = cloud->points[valid_indices[i]].z;
    }
    
    return filtered_obstacles;
  }
}

Eigen::MatrixX3f GroundSegmentationServer::TrackPersistentClusters(
    const std::vector<pcl::PointIndices> &cluster_indices,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {

  if (!cloud || cloud->points.empty()) {
    return Eigen::MatrixX3f(0, 3);
  }

  // Safety fallback: if persistent tracking causes issues, fall back to simple clustering
  try {
    current_frame_id_++;
    
    RCLCPP_DEBUG(this->get_logger(), "TrackPersistentClusters: frame %d, %zu input clusters, %zu persistent clusters", 
                 current_frame_id_, cluster_indices.size(), persistent_clusters_.size());
  } catch (const std::exception& e) {
    RCLCPP_ERROR(this->get_logger(), "Exception in persistent tracking, falling back: %s", e.what());
    // Fall back to simple clustering
    std::vector<int> valid_indices;
    for (const auto& cluster : cluster_indices) {
      valid_indices.insert(valid_indices.end(), cluster.indices.begin(), cluster.indices.end());
    }
    
    if (valid_indices.empty()) {
      return Eigen::MatrixX3f(0, 3);
    }
    
    Eigen::MatrixX3f filtered_obstacles(valid_indices.size(), 3);
    for (size_t i = 0; i < valid_indices.size(); ++i) {
      filtered_obstacles(i, 0) = cloud->points[valid_indices[i]].x;
      filtered_obstacles(i, 1) = cloud->points[valid_indices[i]].y;
      filtered_obstacles(i, 2) = cloud->points[valid_indices[i]].z;
    }
    return filtered_obstacles;
  }
  
  // Extract current frame cluster centers (lightweight)
  std::vector<std::pair<Eigen::Vector3f, std::vector<int>>> current_clusters;
  current_clusters.reserve(cluster_indices.size());

  for (const auto& cluster : cluster_indices) {
    if (cluster.indices.size() < static_cast<size_t>(min_cluster_size_)) {
      continue;
    }

    // Calculate cluster center only
    Eigen::Vector3f center(0.0f, 0.0f, 0.0f);
    std::vector<int> valid_indices;
    
    for (int idx : cluster.indices) {
      if (idx >= 0 && static_cast<size_t>(idx) < cloud->points.size()) {
        const auto& point = cloud->points[idx];
        center.x() += point.x;
        center.y() += point.y;
        center.z() += point.z;
        valid_indices.push_back(idx);
      }
    }
    
    if (!valid_indices.empty()) {
      center /= static_cast<float>(valid_indices.size());
      current_clusters.emplace_back(center, std::move(valid_indices));
    }
  }

  // Efficient matching using squared distances to avoid sqrt
  const float max_distance_sq = max_cluster_distance_ * max_cluster_distance_;
  std::vector<bool> matched(persistent_clusters_.size(), false);

  // Update persistent clusters
  for (const auto& current : current_clusters) {
    int best_match = -1;
    float min_distance_sq = max_distance_sq;

    // Find closest persistent cluster
    for (size_t j = 0; j < persistent_clusters_.size(); ++j) {
      if (matched[j]) continue;
      
      const Eigen::Vector3f diff = current.first - persistent_clusters_[j].center;
      const float distance_sq = diff.squaredNorm();
      
      if (distance_sq < min_distance_sq) {
        min_distance_sq = distance_sq;
        best_match = j;
      }
    }

    if (best_match >= 0) {
      // Update existing cluster
      persistent_clusters_[best_match].center = current.first;
      persistent_clusters_[best_match].point_count = current.second.size();
      persistent_clusters_[best_match].frame_count++;
      persistent_clusters_[best_match].last_seen_frame = current_frame_id_;
      matched[best_match] = true;
    } else {
      // Add new cluster
      persistent_clusters_.emplace_back(current.first, current.second.size(), current_frame_id_);
    }
  }

  // Remove old clusters (not seen for too long)
  if (!persistent_clusters_.empty()) {
    auto it = persistent_clusters_.begin();
    while (it != persistent_clusters_.end()) {
      if (current_frame_id_ - it->last_seen_frame > 2) {  // Remove if not seen for 2+ frames
        it = persistent_clusters_.erase(it);
      } else {
        ++it;
      }
    }
  }

  // Build result from valid persistent clusters
  std::vector<int> valid_cluster_indices;
  for (const auto& current : current_clusters) {
    // Check if this cluster matches a valid persistent cluster
    for (const auto& persistent : persistent_clusters_) {
      if (persistent.frame_count >= min_frames_for_obstacle_) {
        const float distance_sq = (current.first - persistent.center).squaredNorm();
        if (distance_sq < max_distance_sq) {
          valid_cluster_indices.insert(valid_cluster_indices.end(),
                                     current.second.begin(),
                                     current.second.end());
          break;
        }
      }
    }
  }

  if (valid_cluster_indices.empty()) {
    return Eigen::MatrixX3f(0, 3);
  }

  // Convert to Eigen matrix
  Eigen::MatrixX3f result(valid_cluster_indices.size(), 3);
  for (size_t i = 0; i < valid_cluster_indices.size(); ++i) {
    int idx = valid_cluster_indices[i];
    if (idx >= 0 && static_cast<size_t>(idx) < cloud->points.size()) {
      const auto& point = cloud->points[idx];
      result(i, 0) = point.x;
      result(i, 1) = point.y;
      result(i, 2) = point.z;
    } else {
      RCLCPP_ERROR(this->get_logger(), "Invalid point index in result: %d, cloud size: %zu", idx, cloud->points.size());
      result(i, 0) = 0.0f;
      result(i, 1) = 0.0f;
      result(i, 2) = 0.0f;
    }
  }
  
  return result;
}

void GroundSegmentationServer::PublishClouds(const Eigen::MatrixX3f &est_ground,
                                             const Eigen::MatrixX3f &est_nonground,
                                             const Eigen::MatrixX3f &est_obstacles,
                                             const std_msgs::msg::Header header_msg) {
  std_msgs::msg::Header header = header_msg;
  header.frame_id              = GetOutputFrameId();
  ground_publisher_->publish(
      std::move(patchworkpp_ros::utils::EigenMatToPointCloud2(est_ground, header)));
  nonground_publisher_->publish(
      std::move(patchworkpp_ros::utils::EigenMatToPointCloud2(est_nonground, header)));
  obstacles_publisher_->publish(
      std::move(patchworkpp_ros::utils::EigenMatToPointCloud2(est_obstacles, header)));

  civ_interfaces::msg::ObstacleState obstacle_state;
  obstacle_state.state = (est_obstacles.rows() > 0) ? civ_interfaces::msg::ObstacleState::OBSTACLE : civ_interfaces::msg::ObstacleState::FREE;
  obstacle_state_publisher_->publish(obstacle_state);
}
}  // namespace patchworkpp_ros

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(patchworkpp_ros::GroundSegmentationServer)
