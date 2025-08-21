// Patchwork++
#include "patchwork/patchworkpp.h"

// ROS 2
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

// PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/kdtree/kdtree.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/header.hpp>
#include <civ_interfaces/msg/obstacle_state.hpp>

namespace patchworkpp_ros {

class GroundSegmentationServer : public rclcpp::Node {
 public:
  /// GroundSegmentationServer constructor
  GroundSegmentationServer() = delete;
  explicit GroundSegmentationServer(const rclcpp::NodeOptions &options);

 private:
  /// Register new frame
  void EstimateGround(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &msg);

  /// Stream the point clouds for visualization
  void PublishClouds(const Eigen::MatrixX3f &est_ground,
                     const Eigen::MatrixX3f &est_nonground,
                     const Eigen::MatrixX3f &est_obstacles,
                     const std_msgs::msg::Header header_msg);

  /// Filter point cloud based on FOV angle from positive x-axis (after transformation)
  Eigen::MatrixX3f FilterPointCloudByFOV(const Eigen::MatrixX3f &cloud, const std_msgs::msg::Header &header);

  /// Get the correct frame_id for published point clouds
  std::string GetOutputFrameId() const;

  /// Apply clustering to obstacle points and filter small clusters
  Eigen::MatrixX3f FilterObstaclesByClusterSize(const Eigen::MatrixX3f &obstacles);

  /// Track persistent clusters across frames and return validated obstacles
  Eigen::MatrixX3f TrackPersistentClusters(const std::vector<pcl::PointIndices> &cluster_indices,
                                           const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud);

  /// Simple obstacle detection using height and distance thresholds
  Eigen::MatrixX3f SimpleObstacleDetection(const Eigen::MatrixX3f &cloud);

 private:
  /// Data subscribers.
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;

  /// Data publishers.
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr ground_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr nonground_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr obstacles_publisher_;
  rclcpp::Publisher<civ_interfaces::msg::ObstacleState>::SharedPtr obstacle_state_publisher_;

  /// Patchwork++
  std::unique_ptr<patchwork::PatchWorkpp> Patchworkpp_;

  std::string base_frame_{"base_link"};

  /// FOV filtering parameters
  double fov_angle_deg_;
  double fov_angle_rad_;
  std::string target_frame_;

  /// Obstacle clustering parameters
  double cluster_tolerance_;
  int min_cluster_size_;
  bool enable_persistent_tracking_;
  int min_frames_for_obstacle_;
  double max_cluster_distance_;

  /// Simple obstacle detection parameters
  bool use_simple_obstacle_detection_;
  double simple_obstacle_min_height_;
  double simple_obstacle_max_height_;
  double simple_obstacle_max_distance_;
  double sensor_height_{1.45};  // Initialize with default value
  bool debug_logging_{false};   // Enable detailed debug logging

  /// TF2 for coordinate transformation
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  /// Cached transform for performance
  mutable Eigen::Isometry3d cached_transform_;
  mutable std::string cached_transform_key_;
  mutable bool transform_cached_;

  /// Persistent cluster tracking
  struct ClusterInfo {
    Eigen::Vector3f center;
    int frame_count;
    size_t point_count;
    int last_seen_frame;

    ClusterInfo(const Eigen::Vector3f& c, size_t count, int frame)
      : center(c), frame_count(1), point_count(count), last_seen_frame(frame) {}
  };
  
  int current_frame_id_;

  std::vector<ClusterInfo> persistent_clusters_;
};

}  // namespace patchworkpp_ros
