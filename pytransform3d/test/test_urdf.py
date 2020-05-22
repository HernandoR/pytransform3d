try:
    import matplotlib
    matplotlib_available = True
except ImportError:
    matplotlib_available = False
import warnings
import numpy as np
from pytransform3d.urdf import UrdfTransformManager, UrdfException
from pytransform3d.transformations import transform_from
from numpy.testing import assert_array_almost_equal
from nose.tools import assert_raises, assert_equal
from nose import SkipTest


COMPI_URDF = """
<?xml version="1.0"?>
  <robot name="compi">
    <link name="linkmount"/>
    <link name="link1"/>
    <link name="link2"/>
    <link name="link3"/>
    <link name="link4"/>
    <link name="link5"/>
    <link name="link6"/>
    <link name="tcp"/>

    <joint name="joint1" type="revolute">
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <parent link="linkmount"/>
      <child link="link1"/>
      <axis xyz="0 0 1.0"/>
      <limit lower="-1" upper="1"/>
    </joint>

    <joint name="joint2" type="revolute">
      <origin xyz="0 0 0.158" rpy="1.570796 0 0"/>
      <parent link="link1"/>
      <child link="link2"/>
      <axis xyz="0 0 -1.0"/>
      <limit lower="-1"/>
    </joint>

    <joint name="joint3" type="revolute">
      <origin xyz="0 0.28 0" rpy="0 0 0"/>
      <parent link="link2"/>
      <child link="link3"/>
      <axis xyz="0 0 -1.0"/>
      <limit upper="1"/>
    </joint>

    <joint name="joint4" type="revolute">
      <origin xyz="0 0 0" rpy="-1.570796 0 0"/>
      <parent link="link3"/>
      <child link="link4"/>
      <axis xyz="0 0 1.0"/>
    </joint>

    <joint name="joint5" type="revolute">
      <origin xyz="0 0 0.34" rpy="1.570796 0 0"/>
      <parent link="link4"/>
      <child link="link5"/>
      <axis xyz="0 0 -1.0"/>
    </joint>

    <joint name="joint6" type="revolute">
      <origin xyz="0 0.346 0" rpy="-1.570796 0 0"/>
      <parent link="link5"/>
      <child link="link6"/>
      <axis xyz="0 0 1.0"/>
    </joint>

    <joint name="jointtcp" type="fixed">
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <parent link="link6"/>
      <child link="tcp"/>
    </joint>
    
    <transmission name="joint1_trans">
      <type>transmission_interface/SimpleTransmission</type>
      <joint name="joint1">
        <hardwareInterface>PositionJointInterface</hardwareInterface>
      </joint>
      <actuator name="joint1_motor">
        <mechanicalReduction>1</mechanicalReduction>
      </actuator>
    </transmission>
  </robot>
"""


def test_missing_robot_tag():
    assert_raises(UrdfException, UrdfTransformManager().load_urdf, "")


def test_missing_robot_name():
    assert_raises(UrdfException, UrdfTransformManager().load_urdf, "<robot/>")


def test_missing_link_name():
    assert_raises(UrdfException, UrdfTransformManager().load_urdf,
                  "<robot name=\"robot_name\"><link/></robot>")


def test_missing_joint_name():
    urdf = """
    <robot name="robot_name">
    <link name="link0"/>
    <link name="link1"/>
    <joint type="fixed">
        <parent link="link0"/>
        <child link="link1"/>
    </joint>
    </robot>
    """
    assert_raises(UrdfException, UrdfTransformManager().load_urdf, urdf)


def test_missing_parent():
    urdf = """
    <robot name="robot_name">
    <link name="link0"/>
    <link name="link1"/>
    <joint name="joint0" type="fixed">
        <child link="link1"/>
    </joint>
    </robot>
    """
    assert_raises(UrdfException, UrdfTransformManager().load_urdf, urdf)


def test_missing_child():
    urdf = """
    <robot name="robot_name">
    <link name="link0"/>
    <link name="link1"/>
    <joint name="joint0" type="fixed">
        <parent link="link0"/>
    </joint>
    </robot>
    """
    assert_raises(UrdfException, UrdfTransformManager().load_urdf, urdf)


def test_missing_parent_link_name():
    urdf = """
    <robot name="robot_name">
    <link name="link0"/>
    <link name="link1"/>
    <joint name="joint0" type="fixed">
        <parent/>
        <child link="link1"/>
    </joint>
    </robot>
    """
    assert_raises(UrdfException, UrdfTransformManager().load_urdf, urdf)


def test_missing_child_link_name():
    urdf = """
    <robot name="robot_name">
    <link name="link0"/>
    <link name="link1"/>
    <joint name="joint0" type="fixed">
        <parent link="link0"/>
        <child/>
    </joint>
    </robot>
    """
    assert_raises(UrdfException, UrdfTransformManager().load_urdf, urdf)


def test_reference_to_unknown_child():
    urdf = """
    <robot name="robot_name">
    <link name="link0"/>
    <joint name="joint0" type="fixed">
        <parent link="link0"/>
        <child link="link1"/>
    </joint>
    </robot>
    """
    assert_raises(UrdfException, UrdfTransformManager().load_urdf, urdf)


def test_reference_to_unknown_parent():
    urdf = """
    <robot name="robot_name">
    <link name="link1"/>
    <joint name="joint0" type="fixed">
        <parent link="link0"/>
        <child link="link1"/>
    </joint>
    </robot>
    """
    assert_raises(UrdfException, UrdfTransformManager().load_urdf, urdf)


def test_missing_joint_type():
    urdf = """
    <robot name="robot_name">
    <link name="link0"/>
    <link name="link1"/>
    <joint name="joint0">
        <parent link="link0"/>
        <child link="link1"/>
    </joint>
    </robot>
    """
    assert_raises(UrdfException, UrdfTransformManager().load_urdf, urdf)


def test_without_origin():
    urdf = """
    <robot name="robot_name">
    <link name="link0"/>
    <link name="link1"/>
    <joint name="joint0" type="fixed">
        <parent link="link0"/>
        <child link="link1"/>
    </joint>
    </robot>
    """
    tm = UrdfTransformManager()
    tm.load_urdf(urdf)
    link1_to_link0 = tm.get_transform("link1", "link0")
    assert_array_almost_equal(link1_to_link0, np.eye(4))


def test_with_empty_origin():
    urdf = """
    <robot name="robot_name">
    <link name="link0"/>
    <link name="link1"/>
    <joint name="joint0" type="fixed">
        <parent link="link0"/>
        <child link="link1"/>
        <origin/>
    </joint>
    </robot>
    """
    tm = UrdfTransformManager()
    tm.load_urdf(urdf)
    link1_to_link0 = tm.get_transform("link1", "link0")
    assert_array_almost_equal(link1_to_link0, np.eye(4))


def test_unsupported_joint_type():
    urdf = """
    <robot name="robot_name">
    <link name="link0"/>
    <link name="link1"/>
    <joint name="joint0" type="prismatic">
        <parent link="link0"/>
        <child link="link1"/>
    </joint>
    </robot>
    """
    assert_raises(UrdfException, UrdfTransformManager().load_urdf, urdf)


def test_unknown_joint_type():
    urdf = """
    <robot name="robot_name">
    <link name="link0"/>
    <link name="link1"/>
    <joint name="joint0" type="does_not_exist">
        <parent link="link0"/>
        <child link="link1"/>
    </joint>
    </robot>
    """
    assert_raises(UrdfException, UrdfTransformManager().load_urdf, urdf)


def test_with_empty_axis():
    urdf = """
    <robot name="robot_name">
    <link name="link0"/>
    <link name="link1"/>
    <joint name="joint0" type="revolute">
        <parent link="link0"/>
        <child link="link1"/>
        <origin/>
        <axis/>
    </joint>
    </robot>
    """
    tm = UrdfTransformManager()
    tm.load_urdf(urdf)
    tm.set_joint("joint0", np.pi)
    link1_to_link0 = tm.get_transform("link1", "link0")
    assert_array_almost_equal(
        link1_to_link0,
        np.array([[1, 0, 0, 0],
                  [0, -1, 0, 0],
                  [0, 0, -1, 0],
                  [0, 0, 0, 1]])
    )


def test_ee_frame():
    tm = UrdfTransformManager()
    tm.load_urdf(COMPI_URDF)
    link7_to_linkmount = tm.get_transform("link6", "linkmount")
    assert_array_almost_equal(
        link7_to_linkmount,
        np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 1.124],
                  [0, 0, 0, 1]])
    )


def test_joint_angles():
    tm = UrdfTransformManager()
    tm.load_urdf(COMPI_URDF)
    for i in range(1, 7):
        tm.set_joint("joint%d" % i, 0.1 * i)
    link7_to_linkmount = tm.get_transform("link6", "linkmount")
    assert_array_almost_equal(
        link7_to_linkmount,
        np.array([[0.121698, -0.606672, 0.785582, 0.489351],
                  [0.818364, 0.509198, 0.266455, 0.114021],
                  [-0.561668, 0.610465, 0.558446, 0.924019],
                  [0., 0., 0., 1.]])
    )


def test_joint_limits():
    tm = UrdfTransformManager()
    tm.load_urdf(COMPI_URDF)

    assert_raises(KeyError, tm.get_joint_limits, "unknown_joint")
    assert_array_almost_equal(tm.get_joint_limits("joint1"), (-1, 1))
    assert_array_almost_equal(tm.get_joint_limits("joint2"), (-1, np.inf))
    assert_array_almost_equal(tm.get_joint_limits("joint3"), (-np.inf, 1))


def test_joint_limit_clipping():
    tm = UrdfTransformManager()
    tm.load_urdf(COMPI_URDF)

    tm.set_joint("joint1", 2.0)
    link7_to_linkmount = tm.get_transform("link6", "linkmount")
    assert_array_almost_equal(
        link7_to_linkmount,
        np.array([[0.5403023, -0.8414710, 0, 0],
                  [0.8414710, 0.5403023, 0, 0],
                  [0, 0, 1, 1.124],
                  [0, 0, 0, 1]])
    )

    tm.set_joint("joint1", -2.0)
    link7_to_linkmount = tm.get_transform("link6", "linkmount")
    assert_array_almost_equal(
        link7_to_linkmount,
        np.array([[0.5403023, 0.8414710, 0, 0],
                  [-0.8414710, 0.5403023, 0, 0],
                  [0, 0, 1, 1.124],
                  [0, 0, 0, 1]])
    )


def test_fixed_joint():
    tm = UrdfTransformManager()
    tm.load_urdf(COMPI_URDF)
    tcp_to_link0 = tm.get_transform("tcp", "linkmount")
    assert_array_almost_equal(
        tcp_to_link0,
        np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 1.174],
                  [0, 0, 0, 1]])
    )


def test_unknown_joint():
    tm = UrdfTransformManager()
    tm.load_urdf(COMPI_URDF)
    assert_raises(KeyError, tm.set_joint, "unknown_joint", 0)


def test_visual_without_geometry():
    urdf = """
    <robot name="robot_name">
    <link name="link0"/>
    <link name="link1">
        <visual name="link1_visual">
            <origin xyz="0 0 1"/>
        </visual>
    </link>
    <joint name="joint0" type="fixed">
        <parent link="link0"/>
        <child link="link1"/>
        <origin xyz="0 1 0"/>
    </joint>
    </robot>
    """
    assert_raises(UrdfException, UrdfTransformManager().load_urdf, urdf)


def test_visual():
    urdf = """
    <robot name="robot_name">
    <link name="link0"/>
    <link name="link1">
        <visual name="link1_visual">
            <origin xyz="0 0 1"/>
            <geometry/>
        </visual>
    </link>
    <joint name="joint0" type="fixed">
        <parent link="link0"/>
        <child link="link1"/>
        <origin xyz="0 1 0"/>
    </joint>
    </robot>
    """
    tm = UrdfTransformManager()
    tm.load_urdf(urdf)
    link1_to_link0 = tm.get_transform("link1", "link0")
    expected_link1_to_link0 = transform_from(np.eye(3), np.array([0, 1, 0]))
    assert_array_almost_equal(link1_to_link0, expected_link1_to_link0)

    link1_to_link0 = tm.get_transform("link1_visual", "link0")
    expected_link1_to_link0 = transform_from(np.eye(3), np.array([0, 1, 1]))
    assert_array_almost_equal(link1_to_link0, expected_link1_to_link0)


def test_collision():
    urdf = """
    <robot name="robot_name">
    <link name="link0"/>
    <link name="link1">
        <collision>
            <origin xyz="0 0 1"/>
            <geometry/>
        </collision>
    </link>
    <joint name="joint0" type="fixed">
        <parent link="link0"/>
        <child link="link1"/>
        <origin xyz="0 0 1"/>
    </joint>
    </robot>
    """
    tm = UrdfTransformManager()
    tm.load_urdf(urdf)

    link1_to_link0 = tm.get_transform("link1", "link0")
    expected_link1_to_link0 = transform_from(np.eye(3), np.array([0, 0, 1]))
    assert_array_almost_equal(link1_to_link0, expected_link1_to_link0)

    link1_to_link0 = tm.get_transform("link1/collision_0", "link0")
    expected_link1_to_link0 = transform_from(np.eye(3), np.array([0, 0, 2]))
    assert_array_almost_equal(link1_to_link0, expected_link1_to_link0)


def test_collision_box():
    urdf = """
    <robot name="robot_name">
    <link name="link0">
        <collision>
            <origin xyz="0 0 1"/>
            <geometry>
                <box size="2 3 4"/>
            </geometry>
        </collision>
    </link>
    </robot>
    """
    tm = UrdfTransformManager()
    tm.load_urdf(urdf)

    assert_equal(len(tm.collision_objects), 1)
    assert_array_almost_equal(tm.collision_objects[0].size,
                              np.array([2, 3, 4]))


def test_collision_box_without_size():
    urdf = """
    <robot name="robot_name">
    <link name="link0">
        <collision>
            <origin xyz="0 0 1"/>
            <geometry>
                <box/>
            </geometry>
        </collision>
    </link>
    </robot>
    """
    tm = UrdfTransformManager()
    tm.load_urdf(urdf)

    assert_equal(len(tm.collision_objects), 1)
    assert_array_almost_equal(tm.collision_objects[0].size, np.zeros(3))


def test_collision_sphere():
    urdf = """
    <robot name="robot_name">
    <link name="link0">
        <collision>
            <origin xyz="0 0 1"/>
            <geometry>
                <sphere radius="0.123"/>
            </geometry>
        </collision>
    </link>
    </robot>
    """
    tm = UrdfTransformManager()
    tm.load_urdf(urdf)

    assert_equal(len(tm.collision_objects), 1)
    assert_equal(tm.collision_objects[0].radius, 0.123)


def test_collision_sphere_without_radius():
    urdf = """
    <robot name="robot_name">
    <link name="link0">
        <collision>
            <origin xyz="0 0 1"/>
            <geometry>
                <sphere/>
            </geometry>
        </collision>
    </link>
    </robot>
    """
    assert_raises(UrdfException, UrdfTransformManager().load_urdf, urdf)


def test_collision_cylinder():
    urdf = """
    <robot name="robot_name">
    <link name="link0">
        <collision>
            <origin xyz="0 0 1"/>
            <geometry>
                <cylinder radius="0.123" length="1.234"/>
            </geometry>
        </collision>
    </link>
    </robot>
    """
    tm = UrdfTransformManager()
    tm.load_urdf(urdf)

    assert_equal(len(tm.collision_objects), 1)
    assert_equal(tm.collision_objects[0].radius, 0.123)
    assert_equal(tm.collision_objects[0].length, 1.234)


def test_collision_cylinder_without_radius():
    urdf = """
    <robot name="robot_name">
    <link name="link0">
        <collision>
            <origin xyz="0 0 1"/>
            <geometry>
                <cylinder length="1.234"/>
            </geometry>
        </collision>
    </link>
    </robot>
    """
    assert_raises(UrdfException, UrdfTransformManager().load_urdf, urdf)


def test_collision_cylinder_without_length():
    urdf = """
    <robot name="robot_name">
    <link name="link0">
        <collision>
            <origin xyz="0 0 1"/>
            <geometry>
                <cylinder radius="0.123"/>
            </geometry>
        </collision>
    </link>
    </robot>
    """
    assert_raises(UrdfException, UrdfTransformManager().load_urdf, urdf)


def test_multiple_collision_objects():
    urdf = """
    <robot name="robot_name">
    <link name="link0">
        <collision>
            <origin xyz="0 0 1"/>
            <geometry>
                <sphere radius="0.123"/>
            </geometry>
        </collision>
    </link>
    <link name="link1">
        <collision>
            <origin xyz="0 0 1"/>
            <geometry>
                <sphere radius="0.234"/>
            </geometry>
        </collision>
    </link>
    </robot>
    """
    tm = UrdfTransformManager()
    tm.load_urdf(urdf)

    assert_equal(len(tm.collision_objects), 2)


def test_multiple_visuals():
    urdf = """
    <robot name="robot_name">
    <link name="link0">
        <visual>
            <origin xyz="0 0 1"/>
            <geometry>
                <sphere radius="0.123"/>
            </geometry>
        </visual>
    </link>
    <link name="link1">
        <visual>
            <origin xyz="0 0 1"/>
            <geometry>
                <sphere radius="0.234"/>
            </geometry>
        </visual>
    </link>
    </robot>
    """
    tm = UrdfTransformManager()
    tm.load_urdf(urdf)

    assert_equal(len(tm.visuals), 2)


def test_multiple_parents():
    urdf = """
    <?xml version="1.0"?>
    <robot name="mmm">
        <link name="parent0"/>
        <link name="parent1"/>
        <link name="child"/>

        <joint name="joint0" type="revolute">
            <origin xyz="1 0 0" rpy="0 0 0"/>
            <parent link="parent0"/>
            <child link="child"/>
            <axis xyz="1 0 0"/>
        </joint>
        <joint name="joint1" type="revolute">
            <origin xyz="0 1 0" rpy="0 0 0"/>
            <parent link="parent1"/>
            <child link="child"/>
            <axis xyz="1 0 0"/>
        </joint>
    </robot>
    """
    tm = UrdfTransformManager()
    tm.load_urdf(urdf)

    p0c = tm.get_transform("parent0", "child")
    p1c = tm.get_transform("parent1", "child")
    assert_equal(p0c[0, 3], p1c[1, 3])


def test_mesh_missing_filename():
    urdf = """
    <?xml version="1.0"?>
    <robot name="simple_mechanism">
        <link name="upper_cone">
          <visual name="upper_cone">
            <origin xyz="0 0 0" rpy="0 1.5708 0"/>
            <geometry>
            <mesh scale="1 1 0.5"/>
            </geometry>
          </visual>
        </link>

        <link name="lower_cone">
          <visual name="lower_cone">
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
            <mesh scale="1 1 0.5"/>
            </geometry>
          </visual>
        </link>

        <joint name="joint" type="revolute">
          <origin xyz="0 0 0.2" rpy="0 0 0"/>
          <parent link="lower_cone"/>
          <child link="upper_cone"/>
          <axis xyz="1 0 0"/>
          <limit lower="-2.79253" upper="2.79253" effort="0" velocity="0"/>
        </joint>

    </robot>
    """
    tm = UrdfTransformManager()
    assert_raises(UrdfException, tm.load_urdf, urdf, mesh_path="")


def test_plot_mesh_smoke_without_scale():
    if not matplotlib_available:
        raise SkipTest("matplotlib is required for this test")

    urdf = """
    <?xml version="1.0"?>
    <robot name="simple_mechanism">
        <link name="upper_cone">
          <visual name="upper_cone">
            <origin xyz="0 0 0" rpy="0 1.5708 0"/>
            <geometry>
            <mesh filename="cone.stl"/>
            </geometry>
          </visual>
        </link>

        <link name="lower_cone">
          <visual name="lower_cone">
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
            <mesh filename="cone.stl"/>
            </geometry>
          </visual>
        </link>

        <joint name="joint" type="revolute">
          <origin xyz="0 0 0.2" rpy="0 0 0"/>
          <parent link="lower_cone"/>
          <child link="upper_cone"/>
          <axis xyz="1 0 0"/>
          <limit lower="-2.79253" upper="2.79253" effort="0" velocity="0"/>
        </joint>

    </robot>
    """
    BASE_DIR = "test/test_data/"
    tm = UrdfTransformManager()
    tm.load_urdf(urdf, mesh_path=BASE_DIR)
    tm.set_joint("joint", -1.1)
    ax = tm.plot_frames_in(
        "lower_cone", s=0.1, whitelist=["upper_cone", "lower_cone"], show_name=True)
    ax = tm.plot_connections_in("lower_cone", ax=ax)
    tm.plot_visuals("lower_cone", ax=ax)


def test_plot_mesh_smoke_with_scale():
    if not matplotlib_available:
        raise SkipTest("matplotlib is required for this test")

    BASE_DIR = "test/test_data/"
    tm = UrdfTransformManager()
    with open(BASE_DIR + "simple_mechanism.urdf", "r") as f:
        tm.load_urdf(f.read(), mesh_path=BASE_DIR)
    tm.set_joint("joint", -1.1)
    ax = tm.plot_frames_in(
        "lower_cone", s=0.1, whitelist=["upper_cone", "lower_cone"], show_name=True)
    ax = tm.plot_connections_in("lower_cone", ax=ax)
    tm.plot_visuals("lower_cone", ax=ax)


def test_plot_without_mesh():
    if not matplotlib_available:
        raise SkipTest("matplotlib is required for this test")

    BASE_DIR = "test/test_data/"
    tm = UrdfTransformManager()
    with open(BASE_DIR + "simple_mechanism.urdf", "r") as f:
        tm.load_urdf(f.read(), mesh_path=None)
    tm.set_joint("joint", -1.1)
    ax = tm.plot_frames_in(
        "lower_cone", s=0.1, whitelist=["upper_cone", "lower_cone"], show_name=True)
    ax = tm.plot_connections_in("lower_cone", ax=ax)

    with warnings.catch_warnings(record=True) as w:
        tm.plot_visuals("lower_cone", ax=ax)
        assert_equal(len(w), 1)
