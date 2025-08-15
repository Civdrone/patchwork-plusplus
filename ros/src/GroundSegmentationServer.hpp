// Patchwork++
#include "patchwork/patchworkpp.h"

// ROS 2
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
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

  /// TF2 for coordinate transformation
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
};

}  // namespace patchworkpp_ros
