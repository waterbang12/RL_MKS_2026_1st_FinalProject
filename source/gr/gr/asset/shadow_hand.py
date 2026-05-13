# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the dexterous hand from Shadow Robot.

The following configurations are available:

* :obj:`SHADOW_HAND_CFG`: Shadow Hand with implicit actuator model.

Reference:

* https://www.shadowrobot.com/dexterous-hand-series/

"""


from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


ASSET_DIR = Path(__file__).resolve().parent

SHADOW_HAND_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(ASSET_DIR / "usd" / "shadow_hand.usd"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            linear_damping = 100.0,
            angular_damping = 100.0,
            retain_accelerations=True,
            max_depenetration_velocity=1000.0,

            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_contact_impulse=1e32,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            fix_root_link=False,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
        fixed_tendons_props=sim_utils.FixedTendonPropertiesCfg(limit_stiffness=30.0, damping=0.2),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={".*": 0.0},
    ),
    actuators={
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=["robot0_(FF|MF|RF|LF|TH)J(3|2|1)", "robot0_(LF|TH)J4", "robot0_THJ0"],
            effort_limit_sim={
                "robot0_(FF|MF|RF|LF)J1": 100,
                "robot0_FFJ(3|2)": 100,
                "robot0_MFJ(3|2)": 100,
                "robot0_RFJ(3|2)": 100,
                "robot0_LFJ(4|3|2)": 100,
                "robot0_THJ4": 100,
                "robot0_THJ3": 100,
                "robot0_THJ(2|1)": 100,
                "robot0_THJ0": 100,
            },
            stiffness={
                "robot0_(FF|MF|RF|LF|TH)J(3|2|1)": 1.0,
                "robot0_(LF|TH)J4": 1.0,
                "robot0_THJ0": 1.0,
            },
            damping={
                "robot0_(FF|MF|RF|LF|TH)J(3|2|1)": 0.2,
                "robot0_(LF|TH)J4": 0.2,
                "robot0_THJ0": 0.2,
            },
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
