#include <memory>
#include <utility>
#include <vector>
#include <limits>
#include <fstream>
#include <sstream>

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
#include <rclcpp/duration.hpp>
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
  sensor_height_ = params.sensor_height;  // Store for simple obstacle detection
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

  // Clustering optimization parameters
  enable_voxel_downsampling_ = declare_parameter<bool>("enable_voxel_downsampling", false);
  voxel_leaf_size_ = declare_parameter<double>("voxel_leaf_size", 0.1);
  current_frame_id_ = 0;
  transform_cached_ = false;

  // Simple obstacle detection parameters
  use_simple_obstacle_detection_ = declare_parameter<bool>("use_simple_obstacle_detection", false);
  simple_obstacle_min_height_ = declare_parameter<double>("simple_obstacle_min_height", 0.2);
  simple_obstacle_max_height_ = declare_parameter<double>("simple_obstacle_max_height", 3.0);
  simple_obstacle_max_distance_ = declare_parameter<double>("simple_obstacle_max_distance", 5.0);
  debug_logging_ = declare_parameter<bool>("debug_logging", false);

  // Floating obstacle filtering parameters
  filter_floating_obstacles_ = declare_parameter<bool>("filter_floating_obstacles", true);
  max_ground_clearance_ = declare_parameter<double>("max_ground_clearance", 0.5);

  // Performance profiling parameters
  enable_profiling_ = declare_parameter<bool>("enable_profiling", false);
  enable_memory_profiling_ = declare_parameter<bool>("enable_memory_profiling", false);
  profiling_window_size_ = declare_parameter<int>("profiling_window_size", 50);
  profiling_output_interval_ = declare_parameter<double>("profiling_output_interval", 10.0);

  // Frame rate decimation parameters
  frame_decimation_ratio_ = declare_parameter<int>("frame_decimation_ratio", 1);

  // Initialize profiling
  if (enable_profiling_ || enable_memory_profiling_) {
    profiling_data_.last_output_time = std::chrono::high_resolution_clock::now();

    if (enable_memory_profiling_) {
      auto baseline_memory = GetCurrentMemoryUsage();
      profiling_data_.baseline_rss_mb = baseline_memory.rss_mb;
      profiling_data_.absolute_peak_rss_mb = baseline_memory.rss_mb;
      RCLCPP_INFO(this->get_logger(), "Memory profiling enabled - baseline RSS: %zu MB", baseline_memory.rss_mb);
    }

    RCLCPP_INFO(this->get_logger(), "Performance profiling enabled - timing: %s, memory: %s, window size: %d frames, output interval: %.1f seconds",
                enable_profiling_ ? "ON" : "OFF",
                enable_memory_profiling_ ? "ON" : "OFF",
                profiling_window_size_, profiling_output_interval_);
  }

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
  bounding_box_publisher_ =
      create_publisher<visualization_msgs::msg::MarkerArray>("/civ/obstacle/bounding_boxes", qos);

  // Log decimation configuration
  if (frame_decimation_ratio_ > 1) {
    RCLCPP_INFO(this->get_logger(), "Frame decimation enabled: processing every %dth frame (%.1fx slower)",
                frame_decimation_ratio_, static_cast<double>(frame_decimation_ratio_));
  } else {
    RCLCPP_INFO(this->get_logger(), "Frame decimation disabled: processing all frames");
  }

  // Log voxel downsampling configuration
  if (enable_voxel_downsampling_) {
    RCLCPP_INFO(this->get_logger(), "Voxel downsampling enabled: leaf size %.2fm (clustering optimization)",
                voxel_leaf_size_);
  } else {
    RCLCPP_INFO(this->get_logger(), "Voxel downsampling disabled: using full point density for clustering");
  }

  RCLCPP_INFO(this->get_logger(), "Patchwork++ ROS 2 node initialized");
}

void GroundSegmentationServer::EstimateGround(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg) {
  // Frame rate decimation - skip processing if not the target frame
  frame_counter_++;
  if (frame_counter_ % frame_decimation_ratio_ != 0) {
    if (debug_logging_) {
      RCLCPP_DEBUG(this->get_logger(), "Skipping frame %d (decimation ratio: %d)",
                   frame_counter_, frame_decimation_ratio_);
    }
    return; // Skip processing this frame
  }

  if (debug_logging_ && frame_decimation_ratio_ > 1) {
    RCLCPP_DEBUG(this->get_logger(), "Processing frame %d (decimation ratio: %d)",
                 frame_counter_, frame_decimation_ratio_);
  }

  // Total frame timing
  ScopedTimer total_timer(profiling_data_.total_frame_times, profiling_window_size_, enable_profiling_);

  // Clear previous frame's cluster details at start of processing
  current_cluster_details_.clear();

  const auto &cloud_raw = patchworkpp_ros::utils::PointCloud2ToEigenMat(msg);

  // Apply transformation and FOV filtering (timing done inside FilterPointCloudByFOV)
  const auto &cloud = FilterPointCloudByFOV(cloud_raw, msg->header);

  // Create header with correct frame_id for transformed points
  std_msgs::msg::Header output_header = msg->header;
  output_header.frame_id = GetOutputFrameId();
  cloud_publisher_->publish(patchworkpp_ros::utils::EigenMatToPointCloud2(cloud, output_header));

  Eigen::MatrixX3f ground;
  Eigen::MatrixX3f nonground;
  Eigen::MatrixX3f obstacles_raw;

  // Ground segmentation phase with timing
  {
    ScopedTimer ground_seg_timer(profiling_data_.ground_segmentation_times, profiling_window_size_, enable_profiling_);

    if (use_simple_obstacle_detection_) {
      // Simple obstacle detection mode: no ground segmentation, direct obstacle detection
      try {
        RCLCPP_DEBUG(this->get_logger(), "Running SimpleObstacleDetection on cloud with %d points", (int)cloud.rows());
        obstacles_raw = SimpleObstacleDetection(cloud);
        RCLCPP_DEBUG(this->get_logger(), "SimpleObstacleDetection completed, found %d obstacles", (int)obstacles_raw.rows());
      } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "SimpleObstacleDetection failed: %s", e.what());
        obstacles_raw = Eigen::MatrixX3f(0, 3);  // Return empty on error
      } catch (...) {
        RCLCPP_ERROR(this->get_logger(), "SimpleObstacleDetection failed with unknown exception");
        obstacles_raw = Eigen::MatrixX3f(0, 3);  // Return empty on error
      }
      // For simple mode, we don't separate ground/nonground - all points are either obstacles or ignored
      ground = Eigen::MatrixX3f(0, 3);      // Empty ground
      nonground = Eigen::MatrixX3f(0, 3);   // Empty nonground
    } else {
      // Patchwork++ mode: full ground segmentation
      Patchworkpp_->estimateGround(cloud);
      ground = Patchworkpp_->getGround();
      nonground = Patchworkpp_->getNonground();
      obstacles_raw = Patchworkpp_->getObstacles();
    }
  }

  // Apply clustering to filter small obstacle clusters (common to both modes)
  Eigen::MatrixX3f obstacles_filtered;
  {
    ScopedTimer clustering_timer(profiling_data_.clustering_times, profiling_window_size_, enable_profiling_);

    try {
      if (debug_logging_) {
        RCLCPP_INFO(this->get_logger(), "Running FilterObstaclesByClusterSize on %d obstacle points", (int)obstacles_raw.rows());
      }

      if (obstacles_raw.rows() == 0) {
        if (debug_logging_) {
          RCLCPP_INFO(this->get_logger(), "No obstacles to cluster, returning empty");
        }
        obstacles_filtered = Eigen::MatrixX3f(0, 3);
      } else {
        obstacles_filtered = FilterObstaclesByClusterSize(obstacles_raw);
        if (debug_logging_) {
          RCLCPP_INFO(this->get_logger(), "Clustering completed, result has %d points", (int)obstacles_filtered.rows());
        }
      }
    } catch (const std::exception& e) {
      RCLCPP_ERROR(this->get_logger(), "FilterObstaclesByClusterSize failed: %s", e.what());
      obstacles_filtered = Eigen::MatrixX3f(0, 3);  // Return empty on error
    } catch (...) {
      RCLCPP_ERROR(this->get_logger(), "FilterObstaclesByClusterSize failed with unknown exception");
      obstacles_filtered = Eigen::MatrixX3f(0, 3);  // Return empty on error
    }
  }

  // Publishing phase with timing
  {
    ScopedTimer publishing_timer(profiling_data_.publishing_times, profiling_window_size_, enable_profiling_);
    PublishClouds(ground, nonground, obstacles_filtered, msg->header);
  }

  // Collect memory usage snapshot
  if (enable_memory_profiling_) {
    auto current_memory = GetCurrentMemoryUsage();
    profiling_data_.AddMemorySnapshot(current_memory, profiling_window_size_);
  }

  // Output profiling statistics periodically
  OutputProfilingStatistics();
}

Eigen::MatrixX3f GroundSegmentationServer::FilterPointCloudByFOV(const Eigen::MatrixX3f &cloud, const std_msgs::msg::Header &header) {
  Eigen::MatrixX3f working_cloud;

  // Apply TF2 transformation if target frame is specified
  {
    ScopedTimer transform_timer(profiling_data_.transform_times, profiling_window_size_, enable_profiling_);

    if (!target_frame_.empty() && target_frame_ != header.frame_id) {
    try {
      // Use cached transform for static transforms
      std::string transform_key = header.frame_id + "->" + target_frame_;

      if (!transform_cached_ || cached_transform_key_ != transform_key) {
        // Check if transform is available (wait longer for static transforms)
        if (tf_buffer_->canTransform(target_frame_, header.frame_id,
                                     header.stamp, tf2::durationFromSec(5.0))) {
          // Get and cache the transform
          geometry_msgs::msg::TransformStamped transform = tf_buffer_->lookupTransform(
              target_frame_, header.frame_id, header.stamp);
          cached_transform_ = tf2::transformToEigen(transform);
          cached_transform_key_ = transform_key;
          transform_cached_ = true;
        } else {
          RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                               "Transform from %s to %s not available, using original points",
                               header.frame_id.c_str(), target_frame_.c_str());
          working_cloud = cloud;
        }
      }

      if (transform_cached_) {
        // Pre-compute matrix multiplication components
        const auto& R = cached_transform_.linear();
        const auto& t = cached_transform_.translation();

        // Apply transformation efficiently
        working_cloud.resize(cloud.rows(), cloud.cols());
        for (int i = 0; i < cloud.rows(); ++i) {
          // Direct matrix operations (faster than Vector3d construction)
          Eigen::Vector3d point = cloud.row(i).cast<double>();
          Eigen::Vector3d transformed_point = R * point + t;
          working_cloud.row(i) = transformed_point.cast<float>();
        }
      }

    } catch (tf2::TransformException &ex) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                           "Transform failed: %s, using original points", ex.what());
      working_cloud = cloud;
    }
    } else {
      // No transformation needed, use original cloud
      working_cloud = cloud;
    }
  }  // End transform timing

  // Apply FOV filtering on (possibly transformed) points
  {
    ScopedTimer fov_timer(profiling_data_.fov_filter_times, profiling_window_size_, enable_profiling_);
  if (fov_angle_deg_ >= 360.0) {
    // No filtering needed if FOV is 360 degrees or more
    return working_cloud;
  }
  else
  {
    if (debug_logging_) {
      RCLCPP_INFO(this->get_logger(), "FOV is set to %f degrees.", fov_angle_deg_);
    }
  }

  // Pre-compute trigonometric values for FOV
  const float half_fov = static_cast<float>(fov_angle_rad_ / 2.0);
  const float cos_half_fov = std::cos(half_fov);

  std::vector<int> valid_indices;
  valid_indices.reserve(working_cloud.rows());

  for (int i = 0; i < working_cloud.rows(); ++i) {
    const float x = working_cloud(i, 0);
    const float y = working_cloud(i, 1);

    // Fast FOV check using dot product instead of atan2
    // Point is within FOV if cos(angle) >= cos(half_fov)
    const float magnitude_sq = x * x + y * y;
    if (magnitude_sq > 0.0f) {
      const float cos_angle = x / std::sqrt(magnitude_sq);
      if (cos_angle >= cos_half_fov) {
        valid_indices.push_back(i);
      }
    }
  }

  // Create filtered cloud efficiently
  if (valid_indices.empty()) {
    return Eigen::MatrixX3f(0, 3);
  }

  Eigen::MatrixX3f filtered_cloud(valid_indices.size(), 3);
  for (size_t i = 0; i < valid_indices.size(); ++i) {
    filtered_cloud.row(i) = working_cloud.row(valid_indices[i]);
  }

    return filtered_cloud;
  }  // End FOV timing
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

  // Apply voxel grid downsampling for clustering optimization if enabled
  pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cloud = cloud;
  if (enable_voxel_downsampling_) {
    ScopedTimer downsample_timer(profiling_data_.clustering_times, profiling_window_size_, enable_profiling_);

    pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
    voxel_filter.setInputCloud(cloud);
    voxel_filter.setLeafSize(voxel_leaf_size_, voxel_leaf_size_, voxel_leaf_size_);

    downsampled_cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
    voxel_filter.filter(*downsampled_cloud);

    if (debug_logging_) {
      RCLCPP_DEBUG(this->get_logger(), "Voxel downsampling: %zu → %zu points (%.1f%% reduction)",
                   cloud->points.size(), downsampled_cloud->points.size(),
                   100.0 * (1.0 - static_cast<double>(downsampled_cloud->points.size()) / cloud->points.size()));
    }
  }

  // Create KD-Tree for efficient nearest neighbor search
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
  tree->setInputCloud(downsampled_cloud);

  // Perform Euclidean clustering
  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
  ec.setClusterTolerance(cluster_tolerance_);
  ec.setMinClusterSize(min_cluster_size_);
  ec.setMaxClusterSize(downsampled_cloud->points.size()); // No upper limit
  ec.setSearchMethod(tree);
  ec.setInputCloud(downsampled_cloud);
  ec.extract(cluster_indices);

  if (enable_persistent_tracking_) {
    // Track persistent clusters across frames
    ScopedTimer persistent_timer(profiling_data_.persistent_tracking_times, profiling_window_size_, enable_profiling_);

    try {
      if (debug_logging_) {
        RCLCPP_INFO(this->get_logger(), "Running persistent tracking on %zu clusters", cluster_indices.size());
      }
      return TrackPersistentClusters(cluster_indices, downsampled_cloud);
    } catch (const std::exception& e) {
      RCLCPP_ERROR(this->get_logger(), "Persistent tracking failed: %s, falling back to simple clustering", e.what());
      // Fall back to simple clustering on error
    } catch (...) {
      RCLCPP_ERROR(this->get_logger(), "Persistent tracking failed with unknown exception, falling back to simple clustering");
      // Fall back to simple clustering on error
    }
  }

  // Build filtered obstacle cloud from valid clusters (fast path) and extract cluster details
  std::vector<int> valid_indices;

  for (const auto& cluster : cluster_indices) {
    if (cluster.indices.size() < static_cast<size_t>(min_cluster_size_)) {
      continue;
    }

    // Calculate cluster centroid for stable ID generation
    Eigen::Vector3f centroid(0.0f, 0.0f, 0.0f);
    for (int idx : cluster.indices) {
      if (idx >= 0 && static_cast<size_t>(idx) < downsampled_cloud->points.size()) {
        const auto& point = downsampled_cloud->points[idx];
        centroid.x() += point.x;
        centroid.y() += point.y;
        centroid.z() += point.z;
      }
    }
    centroid /= static_cast<float>(cluster.indices.size());

    // Generate stable ID based on spatial position (more consistent than sequential)
    // Use a spatial hash of the centroid position for semi-persistent IDs
    uint32_t stable_id = static_cast<uint32_t>(
        std::abs(static_cast<int>(centroid.x() * 100) + static_cast<int>(centroid.y() * 100) * 1000) % 10000);

    // Extract cluster details using helper method with stable ID
    ClusterDetail detail = ExtractClusterDetail(cluster.indices, downsampled_cloud, stable_id, 100); // 100% confidence for non-persistent mode
    current_cluster_details_.push_back(detail);

    // Add cluster points to valid indices
    valid_indices.insert(valid_indices.end(), cluster.indices.begin(), cluster.indices.end());
  }

  if (valid_indices.empty()) {
    return Eigen::MatrixX3f(0, 3);
  }

  // Convert back to Eigen matrix
  Eigen::MatrixX3f filtered_obstacles(valid_indices.size(), 3);
  for (size_t i = 0; i < valid_indices.size(); ++i) {
    filtered_obstacles(i, 0) = downsampled_cloud->points[valid_indices[i]].x;
    filtered_obstacles(i, 1) = downsampled_cloud->points[valid_indices[i]].y;
    filtered_obstacles(i, 2) = downsampled_cloud->points[valid_indices[i]].z;
  }

  return filtered_obstacles;
}

Eigen::MatrixX3f GroundSegmentationServer::TrackPersistentClusters(
    const std::vector<pcl::PointIndices> &cluster_indices,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {

  if (debug_logging_) {
    RCLCPP_INFO(this->get_logger(), "TrackPersistentClusters: starting with %zu clusters, cloud size %zu", 
                cluster_indices.size(), cloud ? cloud->points.size() : 0);
  }

  if (!cloud || cloud->points.empty()) {
    RCLCPP_WARN(this->get_logger(), "TrackPersistentClusters: empty or null cloud");
    return Eigen::MatrixX3f(0, 3);
  }

  if (cluster_indices.empty()) {
    if (debug_logging_) {
      RCLCPP_INFO(this->get_logger(), "TrackPersistentClusters: no clusters to process");
    }
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

  for (size_t cluster_idx = 0; cluster_idx < cluster_indices.size(); ++cluster_idx) {
    const auto& cluster = cluster_indices[cluster_idx];

    RCLCPP_DEBUG(this->get_logger(), "Processing cluster %zu with %zu indices", cluster_idx, cluster.indices.size());

    if (cluster.indices.size() < static_cast<size_t>(min_cluster_size_)) {
      RCLCPP_DEBUG(this->get_logger(), "Cluster %zu too small (%zu < %d), skipping",
                   cluster_idx, cluster.indices.size(), min_cluster_size_);
      continue;
    }

    // Calculate cluster center only
    Eigen::Vector3f center(0.0f, 0.0f, 0.0f);
    std::vector<int> valid_indices;
    
    for (int idx : cluster.indices) {
      if (idx < 0 || static_cast<size_t>(idx) >= cloud->points.size()) {
        RCLCPP_ERROR(this->get_logger(), "Invalid cluster index %d (cloud size: %zu)", idx, cloud->points.size());
        continue;
      }

      try {
        const auto& point = cloud->points[idx];
        // Check for valid finite values
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
          RCLCPP_DEBUG(this->get_logger(), "Non-finite point at index %d: [%f, %f, %f]", idx, point.x, point.y, point.z);
          continue;
        }
        center.x() += point.x;
        center.y() += point.y;
        center.z() += point.z;
        valid_indices.push_back(idx);
      } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Exception accessing point %d: %s", idx, e.what());
        break;
      }
    }
    
    if (!valid_indices.empty()) {
      center /= static_cast<float>(valid_indices.size());
      current_clusters.emplace_back(center, std::move(valid_indices));
      RCLCPP_DEBUG(this->get_logger(), "Added cluster %zu with center [%.2f, %.2f, %.2f] and %zu points",
                   current_clusters.size()-1, center.x(), center.y(), center.z(), valid_indices.size());
    }
  }

  if (debug_logging_) {
    RCLCPP_INFO(this->get_logger(), "Processed clusters: found %zu valid clusters from %zu input clusters",
                current_clusters.size(), cluster_indices.size());
  }

  // Efficient matching using squared distances to avoid sqrt
  const float max_distance_sq = max_cluster_distance_ * max_cluster_distance_;
  const size_t original_persistent_count = persistent_clusters_.size();
  std::vector<bool> matched(original_persistent_count, false);

  if (debug_logging_) {
    RCLCPP_INFO(this->get_logger(), "Starting cluster matching: %zu current clusters, %zu persistent clusters",
                current_clusters.size(), persistent_clusters_.size());
  }

  // Update persistent clusters
  for (size_t current_idx = 0; current_idx < current_clusters.size(); ++current_idx) {
    const auto& current = current_clusters[current_idx];

    if (debug_logging_) {
      RCLCPP_INFO(this->get_logger(), "Processing current cluster %zu with center [%.2f, %.2f, %.2f]",
                  current_idx, current.first.x(), current.first.y(), current.first.z());
    }
    int best_match = -1;
    float min_distance_sq = max_distance_sq;

    // Find closest persistent cluster
    for (size_t j = 0; j < original_persistent_count; ++j) {
      if (matched[j]) continue;

      if (debug_logging_) {
        RCLCPP_INFO(this->get_logger(), "Checking match with persistent cluster %zu, center [%.2f, %.2f, %.2f]",
                    j, persistent_clusters_[j].center.x(), persistent_clusters_[j].center.y(), persistent_clusters_[j].center.z());
      }

      try {
        const Eigen::Vector3f diff = current.first - persistent_clusters_[j].center;
        const float distance_sq = diff.squaredNorm();

        if (debug_logging_) {
          RCLCPP_INFO(this->get_logger(), "Distance squared: %.4f, threshold: %.4f", distance_sq, min_distance_sq);
        }

        if (distance_sq < min_distance_sq) {
          min_distance_sq = distance_sq;
          best_match = j;
        }
      } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Exception in distance calculation for cluster %zu: %s", j, e.what());
        continue;
      }
    }

    if (best_match >= 0) {
      if (debug_logging_) {
        RCLCPP_INFO(this->get_logger(), "Updating persistent cluster %d with current cluster %zu", best_match, current_idx);
      }

      // Update existing cluster
      try {
        persistent_clusters_[best_match].center = current.first;
        persistent_clusters_[best_match].point_count = current.second.size();
        persistent_clusters_[best_match].frame_count++;
        persistent_clusters_[best_match].last_seen_frame = current_frame_id_;
        matched[best_match] = true;
      } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Exception updating persistent cluster %d: %s", best_match, e.what());
      }
    } else {
      if (debug_logging_) {
        RCLCPP_INFO(this->get_logger(), "Adding new persistent cluster from current cluster %zu", current_idx);
      }

      // Add new cluster
      try {
        uint32_t new_id = GetNextClusterID();
        persistent_clusters_.emplace_back(current.first, current.second.size(), current_frame_id_, new_id);

        if (debug_logging_) {
          RCLCPP_INFO(this->get_logger(), "Successfully added cluster, persistent_clusters size: %zu", persistent_clusters_.size());
          // Validate the just-added cluster
          if (!persistent_clusters_.empty()) {
            auto& new_cluster = persistent_clusters_.back();
            RCLCPP_INFO(this->get_logger(), "New cluster validation: center [%.2f, %.2f, %.2f], frame_count: %d",
                        new_cluster.center.x(), new_cluster.center.y(), new_cluster.center.z(), new_cluster.frame_count);
          }
        }
      } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Exception adding new persistent cluster: %s", e.what());
      }
    }
  }

  if (debug_logging_) {
    RCLCPP_INFO(this->get_logger(), "Cluster matching completed. Starting cleanup of old clusters.");
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

  // Build result from valid persistent clusters and extract cluster details
  std::vector<int> valid_cluster_indices;

  for (const auto& current : current_clusters) {
    // Check if this cluster matches a persistent cluster (both validated and unvalidated)
    for (const auto& persistent : persistent_clusters_) {
      const float distance_sq = (current.first - persistent.center).squaredNorm();
      if (distance_sq < max_distance_sq) {
        // Calculate improved confidence level with better scaling
        float confidence_ratio = static_cast<float>(persistent.frame_count) / static_cast<float>(min_frames_for_obstacle_);
        uint8_t confidence_level;

        if (persistent.frame_count < min_frames_for_obstacle_) {
          // Being validated: 0-99% based on progress toward minimum frames
          confidence_level = static_cast<uint8_t>((confidence_ratio * 99.0f));
        } else {
          // Validated: 100-255% using full uint8_t range for long-term obstacles
          float excess_ratio = confidence_ratio - 1.0f; // How much beyond minimum
          confidence_level = static_cast<uint8_t>(100 + std::min(155.0f, excess_ratio * 10.0f)); // Use full range to 255
        }

        ClusterDetail detail = ExtractClusterDetail(current.second, cloud, persistent.persistent_id, confidence_level);
        current_cluster_details_.push_back(detail);

        // Only add to published obstacles if validated (meets minimum frames threshold)
        if (persistent.frame_count >= min_frames_for_obstacle_) {
          valid_cluster_indices.insert(valid_cluster_indices.end(),
                                     current.second.begin(),
                                     current.second.end());
        }
        break;
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

Eigen::MatrixX3f GroundSegmentationServer::SimpleObstacleDetection(const Eigen::MatrixX3f &cloud) {
  // Basic safety checks
  if (cloud.rows() == 0 || cloud.cols() != 3) {
    RCLCPP_WARN(this->get_logger(), "Invalid cloud dimensions: rows=%d, cols=%d", (int)cloud.rows(), (int)cloud.cols());
    return Eigen::MatrixX3f(0, 3);
  }

  if (debug_logging_) {
    RCLCPP_INFO(this->get_logger(), "SimpleObstacleDetection: processing %d points, sensor_height=%.2f",
                (int)cloud.rows(), sensor_height_);
  }

  // Ground level in sensor coordinate frame (negative because sensor is above ground)
  const double ground_z = -sensor_height_;
  const double obstacle_min_z = ground_z + simple_obstacle_min_height_;
  const double obstacle_max_z = ground_z + simple_obstacle_max_height_;
  const double max_distance_sq = simple_obstacle_max_distance_ * simple_obstacle_max_distance_;

  if (debug_logging_) {
    RCLCPP_INFO(this->get_logger(), "Ground level: %.2f, obstacle range: [%.2f, %.2f], max_dist: %.2f",
                ground_z, obstacle_min_z, obstacle_max_z, simple_obstacle_max_distance_);
  }

  std::vector<Eigen::Vector3f> obstacle_points;
  obstacle_points.reserve(cloud.rows() / 10);  // Reserve some space

  // Process each point
  for (int i = 0; i < cloud.rows(); ++i) {
    try {
      const double x = static_cast<double>(cloud(i, 0));
      const double y = static_cast<double>(cloud(i, 1));
      const double z = static_cast<double>(cloud(i, 2));

      // Check for valid finite values
      if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
        continue;
      }

      // Calculate distance squared (avoid sqrt for performance)
      const double distance_sq = x * x + y * y;
      if (distance_sq > max_distance_sq) {
        continue;
      }

      // Check height thresholds
      if (z >= obstacle_min_z && z <= obstacle_max_z) {
        obstacle_points.emplace_back(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z));
      }
    } catch (...) {
      RCLCPP_ERROR(this->get_logger(), "Exception processing point %d", i);
      break;
    }
  }

  if (debug_logging_) {
    RCLCPP_INFO(this->get_logger(), "Found %zu obstacle points", obstacle_points.size());
  }

  if (obstacle_points.empty()) {
    return Eigen::MatrixX3f(0, 3);
  }

  // Convert to Eigen matrix
  Eigen::MatrixX3f obstacles(obstacle_points.size(), 3);
  for (size_t i = 0; i < obstacle_points.size(); ++i) {
    obstacles(i, 0) = obstacle_points[i][0];
    obstacles(i, 1) = obstacle_points[i][1];
    obstacles(i, 2) = obstacle_points[i][2];
  }

  return obstacles;
}

GroundSegmentationServer::ClusterDetail GroundSegmentationServer::ExtractClusterDetail(
    const std::vector<int>& cluster_indices,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    uint32_t cluster_id, uint8_t confidence_level) {

  // Initialize bounding box extremes
  Eigen::Vector3f min_point(std::numeric_limits<float>::max(),
                           std::numeric_limits<float>::max(),
                           std::numeric_limits<float>::max());
  Eigen::Vector3f max_point(std::numeric_limits<float>::lowest(),
                           std::numeric_limits<float>::lowest(),
                           std::numeric_limits<float>::lowest());

  // Calculate cluster centroid and bounding box
  Eigen::Vector3f centroid(0.0f, 0.0f, 0.0f);
  size_t valid_points = 0;

  for (int idx : cluster_indices) {
    if (idx >= 0 && static_cast<size_t>(idx) < cloud->points.size()) {
      const auto& point = cloud->points[idx];

      // Accumulate for centroid
      centroid.x() += point.x;
      centroid.y() += point.y;
      centroid.z() += point.z;

      // Update bounding box extremes
      min_point.x() = std::min(min_point.x(), point.x);
      min_point.y() = std::min(min_point.y(), point.y);
      min_point.z() = std::min(min_point.z(), point.z);

      max_point.x() = std::max(max_point.x(), point.x);
      max_point.y() = std::max(max_point.y(), point.y);
      max_point.z() = std::max(max_point.z(), point.z);

      valid_points++;
    }
  }

  if (valid_points > 0) {
    centroid /= static_cast<float>(valid_points);
  } else {
    // Handle case with no valid points
    min_point = Eigen::Vector3f::Zero();
    max_point = Eigen::Vector3f::Zero();
  }

  // Calculate bounding box dimensions
  Eigen::Vector3f dimensions = max_point - min_point;

  // Convert to polar coordinates
  float range = std::sqrt(centroid.x() * centroid.x() + centroid.y() * centroid.y());
  float bearing = std::atan2(centroid.y(), centroid.x());
  float elevation = std::atan2(centroid.z(), range);

  // Create and return cluster detail
  ClusterDetail detail;
  detail.id = cluster_id;
  detail.centroid_cartesian = centroid;
  detail.centroid_polar = Eigen::Vector3f(range, bearing, elevation);
  detail.point_count = valid_points;
  detail.confidence_level = confidence_level;
  detail.bounding_box_min = min_point;
  detail.bounding_box_max = max_point;
  detail.bounding_box_dimensions = dimensions;

  return detail;
}

bool GroundSegmentationServer::IsObstacleGrounded(const ClusterDetail& cluster_detail, double ground_level) const {
  if (!filter_floating_obstacles_) {
    return true; // Don't filter if disabled
  }

  // Check if the lowest point of the bounding box is within ground clearance threshold
  double lowest_point = static_cast<double>(cluster_detail.bounding_box_min.z());
  double distance_from_ground = lowest_point - ground_level;

  // Obstacle is considered grounded if its lowest point is close to or below ground level
  bool is_grounded = distance_from_ground <= max_ground_clearance_;

  if (debug_logging_) {
    RCLCPP_DEBUG(this->get_logger(),
                 "Cluster %u: lowest_point=%.3f, ground_level=%.3f, distance_from_ground=%.3f, is_grounded=%s",
                 cluster_detail.id, lowest_point, ground_level, distance_from_ground,
                 is_grounded ? "true" : "false");
  }

  return is_grounded;
}

uint32_t GroundSegmentationServer::GetNextClusterID() {
  uint32_t candidate_id = next_cluster_id_;

  // Handle overflow
  if (next_cluster_id_ == std::numeric_limits<uint32_t>::max()) {
    next_cluster_id_ = 1; // Reset to 1, avoiding 0
  } else {
    next_cluster_id_++;
  }

  // Check for ID collision with existing persistent clusters (extremely rare after overflow)
  // This is a safety check for the very unlikely case where IDs wrap around and collide
  bool collision_found = false;
  for (const auto& persistent : persistent_clusters_) {
    if (persistent.persistent_id == candidate_id) {
      collision_found = true;
      break;
    }
  }

  // If collision found (very rare), try a few more IDs
  if (collision_found) {
    for (uint32_t attempt = 0; attempt < 1000; ++attempt) {
      candidate_id = next_cluster_id_;

      // Advance next_cluster_id_ for next attempt
      if (next_cluster_id_ == std::numeric_limits<uint32_t>::max()) {
        next_cluster_id_ = 1;
      } else {
        next_cluster_id_++;
      }

      // Check if this ID is available
      bool collision = false;
      for (const auto& persistent : persistent_clusters_) {
        if (persistent.persistent_id == candidate_id) {
          collision = true;
          break;
        }
      }

      if (!collision) {
        break; // Found available ID
      }
    }

    // If we still have collision after 1000 attempts, just use the candidate anyway
    // This is extremely unlikely (would require 4 billion+ active clusters)
    if (collision_found) {
      RCLCPP_WARN(this->get_logger(), "Cluster ID collision detected but using ID %u anyway", candidate_id);
    }
  }

  return candidate_id;
}

visualization_msgs::msg::MarkerArray GroundSegmentationServer::CreateBoundingBoxMarkers(
    const std::vector<ClusterDetail>& clusters,
    const std_msgs::msg::Header& header) const {

  visualization_msgs::msg::MarkerArray marker_array;

  // Clear all existing markers first
  visualization_msgs::msg::Marker clear_marker;
  clear_marker.header = header;
  clear_marker.ns = "obstacle_bounding_boxes";
  clear_marker.action = visualization_msgs::msg::Marker::DELETEALL;
  marker_array.markers.push_back(clear_marker);

  // Create a marker for each cluster's bounding box
  for (size_t i = 0; i < clusters.size(); ++i) {
    const auto& cluster = clusters[i];

    visualization_msgs::msg::Marker marker;
    marker.header = header;
    marker.ns = "obstacle_bounding_boxes";
    marker.id = static_cast<int>(cluster.id);
    marker.type = visualization_msgs::msg::Marker::CUBE;
    marker.action = visualization_msgs::msg::Marker::ADD;

    // Position at centroid
    marker.pose.position.x = cluster.centroid_cartesian.x();
    marker.pose.position.y = cluster.centroid_cartesian.y();
    marker.pose.position.z = cluster.centroid_cartesian.z();

    // No rotation (axis-aligned boxes)
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;

    // Set dimensions
    marker.scale.x = cluster.bounding_box_dimensions.x(); // depth
    marker.scale.y = cluster.bounding_box_dimensions.y(); // width
    marker.scale.z = cluster.bounding_box_dimensions.z(); // height

    // Color based on confidence level
    float confidence_ratio = static_cast<float>(cluster.confidence_level) / 255.0f;

    // Color scheme: Low confidence = Red, High confidence = Green
    marker.color.r = 1.0f - confidence_ratio; // Red decreases with confidence
    marker.color.g = confidence_ratio;         // Green increases with confidence
    marker.color.b = 0.2f;                     // Small blue component
    marker.color.a = 0.6f;                     // Semi-transparent

    // Lifetime - markers persist until explicitly deleted
    marker.lifetime = rclcpp::Duration::from_seconds(0.5); // 0.5 second lifetime

    marker_array.markers.push_back(marker);

    // Add text label showing cluster ID and confidence
    visualization_msgs::msg::Marker text_marker;
    text_marker.header = header;
    text_marker.ns = "obstacle_labels";
    text_marker.id = static_cast<int>(cluster.id);
    text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    text_marker.action = visualization_msgs::msg::Marker::ADD;

    // Position slightly above the bounding box
    text_marker.pose.position.x = cluster.centroid_cartesian.x();
    text_marker.pose.position.y = cluster.centroid_cartesian.y();
    text_marker.pose.position.z = cluster.centroid_cartesian.z() + cluster.bounding_box_dimensions.z() / 2.0f + 0.2f;

    text_marker.pose.orientation.w = 1.0;

    // Text content
    text_marker.text = "ID:" + std::to_string(cluster.id) + " C:" + std::to_string(cluster.confidence_level) + "%";

    // Text appearance
    text_marker.scale.z = 0.3; // Text height
    text_marker.color.r = 1.0;
    text_marker.color.g = 1.0;
    text_marker.color.b = 1.0;
    text_marker.color.a = 0.9;

    text_marker.lifetime = rclcpp::Duration::from_seconds(0.5);

    marker_array.markers.push_back(text_marker);
  }

  return marker_array;
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

  // Create enhanced obstacle state message with cluster details
  civ_interfaces::msg::ObstacleState obstacle_state;
  obstacle_state.header = header;
  // Note: We'll update the state after filtering, so temporarily set it here
  obstacle_state.state = (est_obstacles.rows() > 0) ? civ_interfaces::msg::ObstacleState::OBSTACLE : civ_interfaces::msg::ObstacleState::FREE;

  // Calculate ground level for floating obstacle filtering
  double ground_level;
  if (use_simple_obstacle_detection_) {
    ground_level = -sensor_height_;
  } else {
    // For Patchwork++ mode, estimate ground level from ground points
    ground_level = -sensor_height_; // Fallback to sensor height
    if (est_ground.rows() > 0) {
      // Use median Z value of ground points as ground level
      std::vector<float> ground_z_values;
      ground_z_values.reserve(est_ground.rows());
      for (int i = 0; i < est_ground.rows(); ++i) {
        ground_z_values.push_back(est_ground(i, 2));
      }
      std::sort(ground_z_values.begin(), ground_z_values.end());
      ground_level = static_cast<double>(ground_z_values[ground_z_values.size() / 2]);
    }
  }

  // Filter out floating obstacles before publishing with timing
  std::vector<ClusterDetail> grounded_clusters;
  {
    ScopedTimer floating_timer(profiling_data_.floating_filter_times, profiling_window_size_, enable_profiling_);

    for (const auto& cluster : current_cluster_details_) {
      if (IsObstacleGrounded(cluster, ground_level)) {
        grounded_clusters.push_back(cluster);
      } else if (debug_logging_) {
        RCLCPP_INFO(this->get_logger(), "Filtered out floating obstacle cluster %u", cluster.id);
      }
    }
  }

  // Populate cluster details with grounded obstacles only
  obstacle_state.cluster_count = grounded_clusters.size();

  for (const auto& cluster : grounded_clusters) {
    obstacle_state.cluster_ids.push_back(cluster.id);

    geometry_msgs::msg::Point cart_point;
    cart_point.x = cluster.centroid_cartesian.x();
    cart_point.y = cluster.centroid_cartesian.y();
    cart_point.z = cluster.centroid_cartesian.z();
    obstacle_state.centroids_cartesian.push_back(cart_point);

    geometry_msgs::msg::Point polar_point;
    polar_point.x = cluster.centroid_polar.x(); // range
    polar_point.y = cluster.centroid_polar.y(); // bearing
    polar_point.z = cluster.centroid_polar.z(); // elevation
    obstacle_state.centroids_polar.push_back(polar_point);

    obstacle_state.point_counts.push_back(cluster.point_count);
    obstacle_state.confidence_levels.push_back(cluster.confidence_level);

    // Add bounding box information
    geometry_msgs::msg::Point bbox_min;
    bbox_min.x = cluster.bounding_box_min.x();
    bbox_min.y = cluster.bounding_box_min.y();
    bbox_min.z = cluster.bounding_box_min.z();
    obstacle_state.bounding_box_min.push_back(bbox_min);

    geometry_msgs::msg::Point bbox_max;
    bbox_max.x = cluster.bounding_box_max.x();
    bbox_max.y = cluster.bounding_box_max.y();
    bbox_max.z = cluster.bounding_box_max.z();
    obstacle_state.bounding_box_max.push_back(bbox_max);

    geometry_msgs::msg::Vector3 bbox_dims;
    bbox_dims.x = cluster.bounding_box_dimensions.x();  // depth (forward-backward dimension, along X-axis)
    bbox_dims.y = cluster.bounding_box_dimensions.y();  // width (left-right dimension, along Y-axis)
    bbox_dims.z = cluster.bounding_box_dimensions.z();  // height (up-down dimension, along Z-axis)
    obstacle_state.bounding_box_dimensions.push_back(bbox_dims);
  }

  // Update obstacle state based on filtered (grounded) clusters
  obstacle_state.state = (grounded_clusters.size() > 0) ? civ_interfaces::msg::ObstacleState::OBSTACLE : civ_interfaces::msg::ObstacleState::FREE;

  obstacle_state_publisher_->publish(obstacle_state);

  // Publish bounding box markers for Foxglove visualization
  auto bounding_box_markers = CreateBoundingBoxMarkers(grounded_clusters, header);
  bounding_box_publisher_->publish(bounding_box_markers);
}

void GroundSegmentationServer::OutputProfilingStatistics() {
  if (!enable_profiling_) return;

  // Check if it's time for periodic output
  auto now = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration<double>(now - profiling_data_.last_output_time).count();

  if (profiling_output_interval_ > 0.0 && elapsed >= profiling_output_interval_) {
    profiling_data_.last_output_time = now;

    RCLCPP_INFO(this->get_logger(), "\n=== PERFORMANCE PROFILING STATISTICS ===");
    RCLCPP_INFO(this->get_logger(), "Window size: %d frames, Time period: %.1f seconds",
                profiling_window_size_, elapsed);
    RCLCPP_INFO(this->get_logger(), "Processing mode: %s",
                use_simple_obstacle_detection_ ? "Simple Obstacle Detection" : "Patchwork++");
    RCLCPP_INFO(this->get_logger(), "Persistent tracking: %s",
                enable_persistent_tracking_ ? "Enabled" : "Disabled");

    // Print timing statistics (Average / Max in milliseconds)
    RCLCPP_INFO(this->get_logger(), "Component Timings (Avg/Max ms):");
    RCLCPP_INFO(this->get_logger(), "  Transform:          %6.2f / %6.2f",
                profiling_data_.GetAverage(profiling_data_.transform_times),
                profiling_data_.GetMax(profiling_data_.transform_times));
    RCLCPP_INFO(this->get_logger(), "  FOV Filter:         %6.2f / %6.2f",
                profiling_data_.GetAverage(profiling_data_.fov_filter_times),
                profiling_data_.GetMax(profiling_data_.fov_filter_times));

    if (use_simple_obstacle_detection_) {
      RCLCPP_INFO(this->get_logger(), "  Simple Detection:   %6.2f / %6.2f",
                  profiling_data_.GetAverage(profiling_data_.ground_segmentation_times),
                  profiling_data_.GetMax(profiling_data_.ground_segmentation_times));
    } else {
      RCLCPP_INFO(this->get_logger(), "  Ground Segmentation:%6.2f / %6.2f",
                  profiling_data_.GetAverage(profiling_data_.ground_segmentation_times),
                  profiling_data_.GetMax(profiling_data_.ground_segmentation_times));
    }

    RCLCPP_INFO(this->get_logger(), "  Clustering:         %6.2f / %6.2f",
                profiling_data_.GetAverage(profiling_data_.clustering_times),
                profiling_data_.GetMax(profiling_data_.clustering_times));

    if (enable_persistent_tracking_ && !profiling_data_.persistent_tracking_times.empty()) {
      RCLCPP_INFO(this->get_logger(), "  Persistent Tracking:%6.2f / %6.2f",
                  profiling_data_.GetAverage(profiling_data_.persistent_tracking_times),
                  profiling_data_.GetMax(profiling_data_.persistent_tracking_times));
    }

    RCLCPP_INFO(this->get_logger(), "  Floating Filter:    %6.2f / %6.2f",
                profiling_data_.GetAverage(profiling_data_.floating_filter_times),
                profiling_data_.GetMax(profiling_data_.floating_filter_times));

    RCLCPP_INFO(this->get_logger(), "  Publishing:         %6.2f / %6.2f",
                profiling_data_.GetAverage(profiling_data_.publishing_times),
                profiling_data_.GetMax(profiling_data_.publishing_times));

    RCLCPP_INFO(this->get_logger(), "  TOTAL FRAME:        %6.2f / %6.2f",
                profiling_data_.GetAverage(profiling_data_.total_frame_times),
                profiling_data_.GetMax(profiling_data_.total_frame_times));

    // Calculate frame rate
    double avg_frame_time_seconds = profiling_data_.GetAverage(profiling_data_.total_frame_times) / 1000.0;
    double fps = (avg_frame_time_seconds > 0.0) ? (1.0 / avg_frame_time_seconds) : 0.0;

    RCLCPP_INFO(this->get_logger(), "Average FPS: %.1f Hz (target: 10 Hz)", fps);

    if (fps < 9.0) {
      RCLCPP_WARN(this->get_logger(), "Processing speed below target! Consider optimization or parameter tuning.");
    }

    // Memory usage statistics
    if (enable_memory_profiling_ && !profiling_data_.memory_snapshots.empty()) {
      RCLCPP_INFO(this->get_logger(), "Memory Usage (Avg/Window Peak/Absolute Peak MB):");
      RCLCPP_INFO(this->get_logger(), "  Average RSS:        %6.1f MB",
                  profiling_data_.GetAverageMemoryUsage());
      RCLCPP_INFO(this->get_logger(), "  Window Peak RSS:    %6zu MB",
                  profiling_data_.GetPeakMemoryUsage());
      RCLCPP_INFO(this->get_logger(), "  Absolute Peak RSS:  %6zu MB",
                  profiling_data_.GetAbsolutePeakMemoryUsage());
      RCLCPP_INFO(this->get_logger(), "  Baseline RSS:       %6zu MB",
                  profiling_data_.baseline_rss_mb);
      RCLCPP_INFO(this->get_logger(), "  Memory Growth:      %6.1f MB",
                  profiling_data_.GetAverageMemoryUsage() - static_cast<double>(profiling_data_.baseline_rss_mb));
    }

    RCLCPP_INFO(this->get_logger(), "=======================================\n");
  }
}

GroundSegmentationServer::MemoryInfo GroundSegmentationServer::GetCurrentMemoryUsage() const {
  MemoryInfo info{};

  std::ifstream status("/proc/self/status");
  std::string line;

  while (std::getline(status, line)) {
    if (line.substr(0, 6) == "VmRSS:") {
      // Extract RSS in KB and convert to MB
      std::string value = line.substr(7);
      size_t pos = value.find_first_not_of(" \t");
      if (pos != std::string::npos) {
        value = value.substr(pos);
        size_t end_pos = value.find_first_of(" \t");
        if (end_pos != std::string::npos) {
          value = value.substr(0, end_pos);
        }
        try {
          info.rss_mb = std::stoull(value) / 1024;  // Convert KB to MB
        } catch (const std::exception&) {
          info.rss_mb = 0;
        }
      }
    } else if (line.substr(0, 7) == "VmSize:") {
      // Extract VMS in KB and convert to MB
      std::string value = line.substr(8);
      size_t pos = value.find_first_not_of(" \t");
      if (pos != std::string::npos) {
        value = value.substr(pos);
        size_t end_pos = value.find_first_of(" \t");
        if (end_pos != std::string::npos) {
          value = value.substr(0, end_pos);
        }
        try {
          info.vms_mb = std::stoull(value) / 1024;  // Convert KB to MB
        } catch (const std::exception&) {
          info.vms_mb = 0;
        }
      }
    }
  }

  return info;
}
}  // namespace patchworkpp_ros

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(patchworkpp_ros::GroundSegmentationServer)
