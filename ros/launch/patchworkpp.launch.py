from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

ROUGH_TERRAIN = True

# Configuration parameters not exposed through the launch system.
# To modify these values, create your own launch file and modify the 'parameters=' block.


def get_terrain_specific_params(rough_terrain=False):
    """Get terrain-specific parameters for different terrain types."""
    if rough_terrain:
        return {
            # Rough terrain specific parameters - more aggressive settings
            "num_lpr": 30,  # Increase "lowest point representative" seeds to stabilize PCA over sparser returns spread across fewer vertical channels.
            "th_dist": 0.2,  # Off-road terrain is much more uneven—allow seed clusters to spread and distance threshold to increase.
            "th_seeds_v": 0.4,  # threshold for lowest point representatives using in initial seeds selection of vertical structural points. Default: 0.25
            "th_dist_v": 0.15,  # Vertical structure detection (R-VPF) must tolerate larger gaps (e.g. rocky outcrops or stumps) before rejecting as non-ground.
            "uprightness_thr": 0.6,  # Default 0.707 (cos 45°) expects planar horizontal surfaces. Lowering to ~0.6 allows terrain up to ~55° tilt be recognized as ground when numerous vertical edges exist.
            "RNR_ver_angle_thr": -30,  # When scanning downward at 45°, ground returns sometimes appear "below" local horizon. Raising this threshold (more negative) prevents over-filtering in steep "look-down" angles.
            "RNR_intensity_thr": 0.15,  # Slightly more restrictive than default 0.2—useful when aggressive spurious reflections occur in dusty environments.
            "adaptive_seed_selection_margin": -1.0,  # Seeds slightly below the lowest classified ground space are allowed. Default −1.2 is conservative; off-road surfaces are unpredictable.
            "max_flatness_storage": 2000,  # Adaptive GLE benefit increases when you store history from multiple scans from varying viewpoints in complex terrain.
            "max_elevation_storage": 2000,
            "flatness_thr": [0.0012, 0.0016, 0.0020, 0.0025],  # Start higher to tolerate rough terrain cracks and grass tuft features.
            "elevation_thr": [0.9, 1.2, 1.5, 2.0],  # Elevation thresholds per ring accommodate hills or depressions; gains auto-adjust from the stored statistics.
        }
    else:
        return {
            # Standard terrain parameters - default settings suitable for flat, road-like terrain
            "num_lpr": 20,  # Maximum number of points to be selected as lowest points representative. Default: 20
            "th_dist": 0.125,  # threshold for thickness of ground. Default: 0.125
            "th_seeds_v": 0.25,  # threshold for lowest point representatives using in initial seeds selection of vertical structural points. Default: 0.25
            "th_dist_v": 0.9,  # threshold for thickness of vertical structure. Default: 0.9
            "uprightness_thr": 0.101,  # threshold of uprightness using in Ground Likelihood Estimation(GLE). Please refer paper for more information about GLE.
        }


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time", default="true")

    # tf tree configuration, these are the likely 3 parameters to change and nothing else
    base_frame = LaunchConfiguration("base_frame", default="os_lidar")

    # ROS configuration
    pointcloud_topic = LaunchConfiguration("cloud_topic")
    visualize = LaunchConfiguration("visualize", default="false")

    # Target frame for FOV filtering (can be different from lidar_target_frame)
    target_frame = LaunchConfiguration("target_frame", default="ouster_transformed")

    # FOV angle parameter
    fov_angle_deg = LaunchConfiguration("fov_angle_deg", default="120.0")

    # Processing mode parameters
    use_simple_obstacle_detection = LaunchConfiguration("use_simple_obstacle_detection", default="false")
    enable_persistent_tracking = LaunchConfiguration("enable_persistent_tracking", default="true")
    enable_voxel_downsampling = LaunchConfiguration("enable_voxel_downsampling", default="true")
    voxel_leaf_size = LaunchConfiguration("voxel_leaf_size", default="0.1")

    # Optional ros bag play
    bagfile = LaunchConfiguration("bagfile", default="")

    # LiDAR mounting compensation parameters
    lidar_compensate_mount = LaunchConfiguration("lidar_compensate_mount", default="false")
    lidar_mount_roll = LaunchConfiguration("lidar_mount_roll", default="0.0")  # degrees
    lidar_mount_pitch = LaunchConfiguration("lidar_mount_pitch", default="0.0")  # degrees
    lidar_mount_yaw = LaunchConfiguration("lidar_mount_yaw", default="0.0")  # degrees
    lidar_mount_x = LaunchConfiguration("lidar_mount_x", default="0.0")  # meters
    lidar_mount_y = LaunchConfiguration("lidar_mount_y", default="0.0")  # meters
    lidar_mount_z = LaunchConfiguration("lidar_mount_z", default="0.0")  # meters
    lidar_source_frame = LaunchConfiguration("lidar_source_frame", default="os_lidar")
    lidar_target_frame = LaunchConfiguration("lidar_target_frame", default="os_lidar_corrected")

    # Common parameters for all terrain types
    common_params = {
        # ROS node configuration
        "base_frame": base_frame,
        "use_sim_time": use_sim_time,
        # Patchwork++ configuration
        "sensor_height": 1.45, #0.7,  # Need to manually measure the height of the sensor from the ground.
        "num_iter": 3,  # Number of iterations for ground plane estimation using PCA.
        "num_min_pts": 0,  # Minimum number of points to be estimated as ground plane in each patch. Default: 0
                          # Some bins on terrain slopes will barely meet this minimum; avoid fitting in regions that are mostly air or sparsely filled.
                          # 0 is the default value, which means that the minimum number of points is not used.
        "th_seeds": 0.3,  # threshold for lowest point representatives using in initial seeds selection of ground points. Default: 0.3
        "max_range": 60.0,  # max_range of ground estimation area; Default: 80.0
                           # OS1-32 has a maximum range of 120 meters for 80% reflective targets and 40 meters for 10% reflective targets.
                           # OS1 "Gen2" op-range is 120 m, but for off-road speeds and loader workspace, 60 m balances density and compute.
        "min_range": 0.3,  # min_range of ground estimation area; Default: 1.0
                          # 0.3m is the minimum range at which the sensor can reliably detect objects and provide point cloud data.
        "obstacle_min_height": 0.001,  # minimum height above ground to consider as obstacle
        "obstacle_max_radius": 5.0,  # maximum distance from sensor to consider as obstacle
        "cluster_tolerance": 0.5,  # maximum distance between points in same cluster (meters)
        "min_cluster_size": 5,  # minimum number of points required to keep an obstacle cluster
        # Clustering optimization parameters
        "enable_voxel_downsampling": enable_voxel_downsampling,  # enable spatial downsampling before clustering for performance
        "voxel_leaf_size": voxel_leaf_size,  # voxel grid leaf size for downsampling (meters) - smaller=more detail, larger=faster
        "enable_persistent_tracking": enable_persistent_tracking,  # enable/disable persistent cluster tracking across frames
        "min_frames_for_obstacle": 3,  # minimum number of consecutive frames to confirm obstacle persistence
        "max_cluster_distance": 1.0,  # maximum distance to match clusters between frames (meters)
        "use_simple_obstacle_detection": use_simple_obstacle_detection,  # use simple height/distance thresholding instead of Patchwork++
        "simple_obstacle_min_height": 0.2,  # minimum height above ground for simple obstacle detection (meters)
        "simple_obstacle_max_height": 3.0,  # maximum height above ground for simple obstacle detection (meters)
        "simple_obstacle_max_distance": 5.0,  # maximum distance for simple obstacle detection (meters)
        "debug_logging": False,  # enable detailed debug logging for troubleshooting
        # Floating obstacle filtering parameters
        "filter_floating_obstacles": True,  # enable floating obstacle filtering
        "max_ground_clearance": 0.5,  # max distance from ground to be considered grounded (meters)
        # Performance profiling parameters
        "enable_profiling": True,  # enable detailed timing profiling for performance analysis
        "enable_memory_profiling": True,  # enable memory usage tracking and analysis
        "profiling_window_size": 50,  # number of frames to average for profiling statistics
        "profiling_output_interval": 10.0,  # seconds between profiling output (0 = disable periodic output)
        # Frame rate decimation parameters
        "frame_decimation_ratio": 1,  # process every Nth frame (1=no decimation, 2=half rate, 5=20% rate)
        "fov_angle_deg": fov_angle_deg,  # Field of view angle in degrees (±60° from positive x-axis)
        "num_sectors_each_zone": [12, 24, 36, 24],  # Setting of Concentric Zone Model(CZM); Default: [16, 32, 54, 32]
                                                    # 32 beams is half of the 64-beam systems tuned in the original tests.
                                                    # Fewer sectors will ensure adequate bin occupancy per sector even in irregular terrain.
        "num_rings_each_zone": [2, 3, 4, 5],  # Setting of Concentric Zone Model(CZM); Default: [2, 4, 4, 4]
                                               # Place more radial granularity toward the far zone to smooth terrain transitions better without exceeding bin-count.
        "verbose": True,  # display verbose info
    }

    # Merge common and terrain-specific parameters
    terrain_params = get_terrain_specific_params(ROUGH_TERRAIN)
    parameters = {**common_params, **terrain_params}

    # Convert degrees to radians for the static transform publisher
    roll_rad = PythonExpression([lidar_mount_roll, " * 3.14159265359 / 180.0"])
    pitch_rad = PythonExpression([lidar_mount_pitch, " * 3.14159265359 / 180.0"])
    yaw_rad = PythonExpression([lidar_mount_yaw, " * 3.14159265359 / 180.0"])

    # Static transform publisher for LiDAR mounting compensation
    # Using modern named parameter format to avoid argument order confusion
    lidar_transform_node = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="lidar_mount_compensation",
        arguments=[
            "--x", lidar_mount_x,
            "--y", lidar_mount_y,
            "--z", lidar_mount_z,
            "--roll", roll_rad,
            "--pitch", pitch_rad,
            "--yaw", yaw_rad,
            "--frame-id", lidar_source_frame,
            "--child-frame-id", lidar_target_frame
        ],
        condition=IfCondition(lidar_compensate_mount),
    )

    # Add target frame parameter for transformation and FOV filtering
    parameters["target_frame"] = target_frame

    patchworkpp_node = Node(
        package="patchworkpp",
        executable="patchworkpp_node",
        name="patchworkpp_node",
        output="screen",
        remappings=[
            ("pointcloud_topic", pointcloud_topic),
        ],
        parameters=[parameters],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        output="screen",
        arguments=[
            "-d",
            PathJoinSubstitution(
                [FindPackageShare("patchworkpp"), "rviz", "patchworkpp.rviz"]
            ),
        ],
        condition=IfCondition(visualize),
    )

    bagfile_play = ExecuteProcess(
        cmd=["ros2", "bag", "play", bagfile],
        output="screen",
        condition=IfCondition(PythonExpression(["'", bagfile, "' != ''"])),
    )

    return LaunchDescription([
        lidar_transform_node,
        patchworkpp_node,
        rviz_node,
        bagfile_play,
    ])