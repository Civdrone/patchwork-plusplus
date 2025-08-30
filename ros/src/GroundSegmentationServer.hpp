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
#include <visualization_msgs/msg/marker_array.hpp>
#include <chrono>
#include <deque>

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

  /// Extract cluster information for publishing
  struct ClusterDetail {
    uint32_t id;
    Eigen::Vector3f centroid_cartesian;
    Eigen::Vector3f centroid_polar;  // range, bearing, elevation
    uint32_t point_count;
    uint8_t confidence_level;
    Eigen::Vector3f bounding_box_min;   // Min corner of axis-aligned bounding box
    Eigen::Vector3f bounding_box_max;   // Max corner of axis-aligned bounding box
    Eigen::Vector3f bounding_box_dimensions; // Width (x), depth (y), height (z)
  };

  /// Store current frame cluster details for publishing
  std::vector<ClusterDetail> current_cluster_details_;

  /// Extract cluster details for a given cluster
  ClusterDetail ExtractClusterDetail(const std::vector<int>& cluster_indices,
                                     const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                                     uint32_t cluster_id, uint8_t confidence_level);

  /// Check if an obstacle is grounded (not floating)
  bool IsObstacleGrounded(const ClusterDetail& cluster_detail, double ground_level) const;

  /// Generate next available cluster ID, handling overflow and collision avoidance
  uint32_t GetNextClusterID();

  /// Create bounding box markers for Foxglove visualization
  visualization_msgs::msg::MarkerArray CreateBoundingBoxMarkers(
      const std::vector<ClusterDetail>& clusters,
      const std_msgs::msg::Header& header) const;

 private:
  /// Data subscribers.
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_sub_;

  /// Data publishers.
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr ground_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr nonground_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr obstacles_publisher_;
  rclcpp::Publisher<civ_interfaces::msg::ObstacleState>::SharedPtr obstacle_state_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr bounding_box_publisher_;

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

  /// Floating obstacle filtering parameters
  bool filter_floating_obstacles_{true};   // Enable floating obstacle filtering
  double max_ground_clearance_{0.5};       // Max distance from ground to be considered grounded

  /// Performance profiling parameters
  bool enable_profiling_{false};           // Enable detailed timing profiling
  bool enable_memory_profiling_{false};    // Enable memory usage tracking
  int profiling_window_size_{50};          // Number of frames for statistics
  double profiling_output_interval_{10.0}; // Seconds between profiling output

  /// Frame rate decimation parameters
  int frame_decimation_ratio_{1};          // Process every Nth frame (1=no decimation)
  int frame_counter_{0};                   // Frame counter for decimation

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
    uint32_t persistent_id;  // Unique ID that persists across frames

    ClusterInfo(const Eigen::Vector3f& c, size_t count, int frame, uint32_t id)
      : center(c), frame_count(1), point_count(count), last_seen_frame(frame), persistent_id(id) {}
  };

  int current_frame_id_;
  uint32_t next_cluster_id_{1};  // Global ID counter for new clusters

  std::vector<ClusterInfo> persistent_clusters_;

  /// Performance profiling infrastructure
  struct MemoryInfo {
    size_t rss_mb;      // Resident Set Size in MB
    size_t vms_mb;      // Virtual Memory Size in MB
    size_t frame_start_rss_mb; // RSS at frame start
    size_t frame_peak_rss_mb;  // Peak RSS during this frame
  };

  struct ProfilingData {
    std::deque<double> transform_times;
    std::deque<double> fov_filter_times;
    std::deque<double> ground_segmentation_times;
    std::deque<double> clustering_times;
    std::deque<double> persistent_tracking_times;
    std::deque<double> floating_filter_times;
    std::deque<double> publishing_times;
    std::deque<double> total_frame_times;

    // Memory profiling data
    std::deque<MemoryInfo> memory_snapshots;
    size_t baseline_rss_mb;      // Memory usage at startup
    size_t absolute_peak_rss_mb; // Absolute peak RSS ever observed

    std::chrono::high_resolution_clock::time_point last_output_time;

    void AddTiming(std::deque<double>& times, double duration, int window_size) {
      times.push_back(duration);
      if (static_cast<int>(times.size()) > window_size) {
        times.pop_front();
      }
    }

    double GetAverage(const std::deque<double>& times) const {
      if (times.empty()) return 0.0;
      double sum = 0.0;
      for (double time : times) sum += time;
      return sum / static_cast<double>(times.size());
    }

    double GetMax(const std::deque<double>& times) const {
      if (times.empty()) return 0.0;
      return *std::max_element(times.begin(), times.end());
    }

    void AddMemorySnapshot(const MemoryInfo& memory_info, int window_size) {
      memory_snapshots.push_back(memory_info);
      if (static_cast<int>(memory_snapshots.size()) > window_size) {
        memory_snapshots.pop_front();
      }

      // Track absolute peak across all frames
      absolute_peak_rss_mb = std::max(absolute_peak_rss_mb, memory_info.rss_mb);
    }

    double GetAverageMemoryUsage() const {
      if (memory_snapshots.empty()) return 0.0;
      double sum = 0.0;
      for (const auto& mem : memory_snapshots) {
        sum += static_cast<double>(mem.rss_mb);
      }
      return sum / static_cast<double>(memory_snapshots.size());
    }

    size_t GetPeakMemoryUsage() const {
      if (memory_snapshots.empty()) return 0;
      size_t peak = 0;
      for (const auto& mem : memory_snapshots) {
        peak = std::max(peak, mem.rss_mb);
      }
      return peak;
    }

    size_t GetAbsolutePeakMemoryUsage() const {
      return absolute_peak_rss_mb;
    }
  };

  ProfilingData profiling_data_;

  /// Helper class for automatic timing measurements
  class ScopedTimer {
  public:
    ScopedTimer(std::deque<double>& times, int window_size, bool enabled)
      : times_(times), window_size_(window_size), enabled_(enabled) {
      if (enabled_) {
        start_time_ = std::chrono::high_resolution_clock::now();
      }
    }

    ~ScopedTimer() {
      if (enabled_) {
        auto end_time = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double, std::milli>(end_time - start_time_).count();
        times_.push_back(duration);
        if (static_cast<int>(times_.size()) > window_size_) {
          times_.pop_front();
        }
      }
    }

  private:
    std::deque<double>& times_;
    int window_size_;
    bool enabled_;
    std::chrono::high_resolution_clock::time_point start_time_;
  };

  /// Print profiling statistics
  void OutputProfilingStatistics();

  /// Get current memory usage information
  MemoryInfo GetCurrentMemoryUsage() const;
};

}  // namespace patchworkpp_ros
