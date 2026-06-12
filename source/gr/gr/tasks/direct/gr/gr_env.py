# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_from_angle_axis, quat_mul, quat_apply, saturate, matrix_from_quat, quat_from_matrix, euler_xyz_from_quat, quat_from_euler_xyz
from .gr_env_cfg import GrEnvCfg
from pxr import Usd, UsdPhysics
import omni.usd


class GrEnv(DirectRLEnv):
    cfg: GrEnvCfg

    def __init__(self, cfg: GrEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.inputs = torch.load(cfg.seq_ref_path, map_location="cpu")

        self.num_hand_dof = self.hand.num_joints

        self.num_kpts = len(self.cfg.MANO_kpts)
        self.termination = not self.cfg.play
        self.play = self.cfg.play
        self.time_out = torch.zeros((self.num_envs, ), device=self.device).bool()
        self.episode_length = self.cfg.episode_length

        # list of joints, hand_bodies, fingertip_bodies, root, rigid bodies
        self.actuated_dof_indices = list()
        self.root_body = list()
        self.hand_bodies = list()
        self.hand_body_names = list()
        self.fingertip_bodies = list()

        for joint_name in self.cfg.actuated_joint_names:
            self.actuated_dof_indices.append(self.hand.joint_names.index(joint_name))
        for i in range(len(self.hand.data.body_names)):
            if self.hand.data.body_names[i] != 'robot0_hand_mount':
                self.hand_body_names.append(self.hand.data.body_names[i])
                self.hand_bodies.append(i)
                if self.hand.data.body_names[i] == 'robot0_palm':
                    self.root_body.append(i)
        for body_name in self.cfg.fingertip_body_names:
            self.fingertip_bodies.append(self.hand_body_names.index(body_name))

        # num of joints, hand_bodies, fingertip_bodies, rigid bodies
        self.num_actuated_dof = len(self.actuated_dof_indices)
        self.num_hand_bodies = len(self.hand_bodies)
        self.num_fingertips = len(self.fingertip_bodies)

        # ref parameters
        self.hand_pos_ref = torch.zeros((self.num_envs, 3), device=self.device)
        self.hand_rot_ref = torch.zeros((self.num_envs, 4), device=self.device)
        self.hand_dof_ref = torch.zeros((self.num_envs, self.num_hand_dof), device=self.device)
        self.obj_pos_ref = torch.zeros((self.num_envs, 3), device=self.device)
        self.obj_rot_ref = torch.zeros((self.num_envs, 4), device=self.device)
        self.hand_rot_ref[:,0] = 1.0
        self.obj_rot_ref[:,0] = 1.0

        # object parameters
        self.obj_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.obj_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.obj_linvel = torch.zeros((self.num_envs, 3), device=self.device)
        self.obj_angvel = torch.zeros((self.num_envs, 3), device=self.device)
        self.obj_pos_reset = torch.zeros((self.num_envs, 3), device=self.device)
        self.obj_rot_reset = torch.zeros((self.num_envs, 4), device=self.device)
        self.obj_rot_reset[:,0] = 1.0

        # hand parameters
        self.hand_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.hand_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.hand_linvel = torch.zeros((self.num_envs, 3), device=self.device)
        self.hand_angvel = torch.zeros((self.num_envs, 3), device=self.device)
        self.hand_pos_reset = torch.zeros((self.num_envs, 3), device=self.device)
        self.hand_rot_reset = torch.zeros((self.num_envs, 4), device=self.device)
        self.hand_rot_reset[:,0] = 1.0
        self.hand_dof_pos_reset = torch.zeros((self.num_envs, self.num_hand_dof), device=self.device)
        self.hand_dof_pos = torch.zeros((self.num_envs, self.num_hand_dof), device=self.device)
        self.hand_dof_vel = torch.zeros((self.num_envs, self.num_hand_dof), device=self.device)

        self.hand_bodies_pos = torch.zeros((self.num_envs,self.num_hand_bodies,3), device=self.device)
        self.hand_bodies_rot = torch.zeros((self.num_envs,self.num_hand_bodies,4), device=self.device)
        self.hand_bodies_linvel = torch.zeros((self.num_envs,self.num_hand_bodies,3), device=self.device)
        self.hand_bodies_angvel = torch.zeros((self.num_envs,self.num_hand_bodies,3), device=self.device)

        self.hand_kpts_pos = torch.zeros((self.num_envs, self.num_kpts, 3), device=self.device)

        # fingertip parameters
        self.fingertip_pos = torch.zeros((self.num_envs,self.num_fingertips,3), device=self.device)
        self.fingertip_normal = torch.zeros((self.num_envs,self.num_fingertips,3), device=self.device)
        self.fingertip_normal[:, 1:, 1] = -1
        self.fingertip_normal[:, 0, 0] = -1
        self.fingertip_rot = torch.zeros((self.num_envs,self.num_fingertips,4), device=self.device)
        self.fingertip_linvel = torch.zeros((self.num_envs,self.num_fingertips,3), device=self.device)
        self.fingertip_angvel = torch.zeros((self.num_envs,self.num_fingertips,3), device=self.device)

        # body to keypoints
        self.fingertip_offset = torch.zeros((self.num_envs,self.num_fingertips,3), device=self.device)
        self.fingertip_offset[:, 0, :] = torch.tensor([-0.0085, 0.0, 0.02], device=self.device)
        self.fingertip_offset[:, 1, :] = torch.tensor([0.0, -0.006, 0.0175], device=self.device)
        self.fingertip_offset[:, 2, :] = torch.tensor([0.0, -0.006, 0.0175], device=self.device)
        self.fingertip_offset[:, 3, :] = torch.tensor([0.0, -0.006, 0.0175], device=self.device)
        self.fingertip_offset[:, 4, :] = torch.tensor([0.0, -0.006, 0.0175], device=self.device)

        # fingertip force
        self.fingertip_contact_forces = torch.zeros((self.num_envs, self.num_fingertips,3), device=self.device)
        self.fingertip_contact_forces_buf = torch.zeros((self.num_envs, 3, self.num_fingertips), device=self.device)

        # joint limits
        joint_pos_limits = self.hand.root_physx_view.get_dof_limits().to(self.device)
        self.hand_dof_lower_limits = joint_pos_limits[..., 0]
        self.hand_dof_upper_limits = joint_pos_limits[..., 1]

        # delta
        self.delta_obj_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.delta_fingertip_pos = torch.zeros((self.num_envs, self.num_fingertips), device=self.device)

        # delta_value
        self.delta_obj_pos_value = torch.zeros((self.num_envs, ), device=self.device)

        # frame idx
        self.start_frame_idx = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)
        self.sampled_frame_idx = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)

        # buffers for dof actions
        self.prev_dof_actions = torch.zeros((self.num_envs, self.num_hand_dof), dtype=torch.float, device=self.device)
        self.cur_dof_actions = torch.zeros((self.num_envs, self.num_hand_dof), dtype=torch.float, device=self.device)
        # buffers for external force and torque
        self.prev_forces = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.prev_torques = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        # track goal resets
        self.hand_far_apart = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.obj_far_apart = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.early_terminate = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # markers
        self.goal_markers = VisualizationMarkers(self.cfg.goal_marker_cfg)
        self.debug_markers = VisualizationMarkers(self.cfg.debug_marker_cfg)

        # separate reward logging
        self.logs_dict = dict()

        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        # global action
        self.is_global = True
        self._debug_printed = False

        self._setup_data()

        # Override headless video camera (ViewerCfg only affects interactive GUI, not --video)
        self.sim.set_camera_view(
            eye=np.array([1.25, -1, 1.5]),
            target=np.array([1, -1.25, 0.45]),
        )



    def _setup_data(self):
        # Provided code. Do not modify.
        obj_bottom_offset = self.inputs['obj_bottom_offset'].to(self.device)
        obj_reset_pos = torch.zeros((1,3), dtype=torch.float, device=self.device)
        obj_reset_pos[0][2] = self.cfg.table_upper_z + obj_bottom_offset
        self.obj_rest_z: float = obj_reset_pos[0][2].item()
        obj_trans = self.inputs['obj_trans'].to(self.device)
        obj_rot = self.inputs['obj_rot'].to(self.device)
        obj_rot = quat_from_matrix(obj_rot)
        to_center_pos = (- obj_trans[0:1] + obj_reset_pos)

        self.obj_rot_reset[:] = obj_rot[0]
        self.obj_rot_seq = obj_rot
        self.obj_pos_seq = obj_trans + to_center_pos
        self.obj_linvel_seq = self.inputs['obj_vel'].to(self.device)
        self.obj_angvel_seq = self.inputs['obj_angvel'].to(self.device)
        self.obj_linvel_value_seq = torch.norm(self.obj_linvel_seq, p=2, dim=-1)
        self.obj_angvel_value_seq = torch.norm(self.obj_angvel_seq, p=2, dim=-1)

        mano_kpts_pos_seq = self.inputs["mano_kpts"][:, self.cfg.MANO_kpts].to(self.device)
        self.mano_kpts_pos_seq = mano_kpts_pos_seq + to_center_pos.unsqueeze(1)
        self.fingertip_pos_seq = self.mano_kpts_pos_seq[:, self.cfg.MANO_fingertips]

        self.obj_kpts_pos_seq_offset = self.mano_kpts_pos_seq - self.obj_pos_seq.unsqueeze(1)
        self.obj_fingertip_pos_seq_offset = self.obj_kpts_pos_seq_offset[:, self.cfg.MANO_fingertips]

        seq_len = self.obj_pos_seq.shape[0]

        self.hand_dof_seq = torch.zeros((seq_len, self.num_hand_dof), device=self.device)
        self.hand_dof_pos_reset[:] = self.hand_dof_seq[0]
        self.hand_rot_reset[:] = self.inputs['R_init'].to(self.device)
        self.hand_pos_reset[:] = (self.inputs['t_init']).to(self.device) + to_center_pos[0]
        self.hand_pos_reset[:,2] = self.hand_pos_reset[:,2] + 0.01

        # Pinch geometry diagnostic — check if thumb and index are on opposite sides of capsule
        mid = min(100, seq_len - 1)
        obj_p   = self.obj_pos_seq[mid].cpu()          # (3,)
        thumb_p = self.fingertip_pos_seq[mid, 0].cpu() # thumb tip (MANO_fingertips[0]=4)
        index_p = self.fingertip_pos_seq[mid, 1].cpu() # index tip (MANO_fingertips[1]=8)
        v_th = thumb_p - obj_p;  v_th = v_th / (v_th.norm() + 1e-8)
        v_ix = index_p - obj_p;  v_ix = v_ix / (v_ix.norm() + 1e-8)
        dot = (v_th * v_ix).sum().item()
        print(f"[PinchCheck] frame={mid}  thumb_offset={( thumb_p - obj_p).numpy()}  index_offset={(index_p - obj_p).numpy()}")
        print(f"[PinchCheck] cos_angle={dot:.3f}  ({'OPPOSITE SIDES = pinch OK' if dot < -0.2 else 'SAME SIDE = no pinch!' if dot > 0.3 else 'roughly orthogonal'})")


    def _setup_scene(self):
        # Provided code. Do not modify.

        self.hand = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        self.table = RigidObject(self.cfg.table_cfg)

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["object"] = self.object
        self.scene.rigid_objects["table"] = self.table

        self.contact_sensors = [
            self.scene.sensors[f"contact_sensor_{body}"]
            for body in self.cfg.fingertip_body_names
        ]

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        stage = omni.usd.get_context().get_stage()
        collisionGroupPaths = [
            "/World/collisionGroup0",
            "/World/collisionGroup1",
            "/World/collisionGroup2",
        ]
        collisionGroupIncludesRel = [None] * 3
        collisionGroupFilteredRels = [None] * 3

        for i in range(3):
            collisionGroup = UsdPhysics.CollisionGroup.Define(stage, collisionGroupPaths[i])
            collisionGroupPrim = collisionGroup.GetPrim()
            collectionAPI = Usd.CollectionAPI.Apply(
                collisionGroupPrim,
                UsdPhysics.Tokens.colliders
            )
            collisionGroupIncludesRel[i] = collectionAPI.CreateIncludesRel()
            collisionGroupFilteredRels[i] = collisionGroup.CreateFilteredGroupsRel()

        for i in range(self.num_envs):
            collisionGroupIncludesRel[0].AddTarget(f"/World/envs/env_{i}/Robot")
            collisionGroupIncludesRel[1].AddTarget(f"/World/envs/env_{i}/Object")
            collisionGroupIncludesRel[2].AddTarget(f"/World/envs/env_{i}/table")

        collisionGroupFilteredRels[1].AddTarget(collisionGroupPaths[1])
        collisionGroupFilteredRels[2].AddTarget(collisionGroupPaths[2])


    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # Provided code. Do not modify.
        self.actions = actions.clone()


    def _apply_action(self) -> None:
        # Provided code. Do not modify.
        pos_offset = self.actions[:, 0:3]
        rot_offset = self.actions[:, 3:9]
        finger_actions = self.actions[:, 9:]

        R_offset = rotation_6d_to_matrix(rot_offset)

        forces = pos_offset * self.cfg.action_dt * self.cfg.K_pos
        torques = matrix_to_axis_angle(R_offset) * self.cfg.action_dt * self.cfg.K_rot
        forces = (1.0 - self.cfg.global_moving_average) * self.prev_forces + self.cfg.global_moving_average * forces
        torques = (1.0 - self.cfg.global_moving_average) * self.prev_torques + self.cfg.global_moving_average * torques
        with torch.no_grad():
            self.prev_forces = forces.detach().clone()
            self.prev_torques = torques.detach().clone()
        full_forces = torch.zeros((self.num_envs, self.hand.num_bodies, 3), device=self.device)
        full_torques = torch.zeros((self.num_envs, self.hand.num_bodies, 3), device=self.device)

        full_forces[:, self.root_body[0], :] = forces
        full_torques[:, self.root_body[0], :] = torques
        self.hand.set_external_force_and_torque(
            full_forces,
            full_torques,
            is_global=True,
        )

        self.cur_dof_actions[:, self.actuated_dof_indices] = scale(
            finger_actions,
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )

        self.cur_dof_actions[:, self.actuated_dof_indices] = (
            self.cfg.act_moving_average * self.cur_dof_actions[:, self.actuated_dof_indices]
            + (1.0 - self.cfg.act_moving_average) * self.prev_dof_actions[:, self.actuated_dof_indices]
        )

        self.cur_dof_actions[:, self.actuated_dof_indices] = saturate(
            self.cur_dof_actions[:, self.actuated_dof_indices],
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )

        self.prev_dof_actions[:, self.actuated_dof_indices] = self.cur_dof_actions[:, self.actuated_dof_indices]
        self.hand.set_joint_position_target(
            self.cur_dof_actions[:, self.actuated_dof_indices],
            joint_ids=self.actuated_dof_indices
        )



    def _get_observations(self) -> dict:
        # Provided code. Do not modify.
        obs = self.compute_full_observations()
        observations = {"policy": obs}
        return observations


    def _get_rewards(self) -> torch.Tensor:
        (
            total_reward,
            logs_dict,
        ) = compute_rewards(
            self.obj_pos,
            self.obj_pos_ref,
            self.obj_rot,
            self.obj_rot_ref,
            self.fingertip_pos,
            self.fingertip_pos_ref,
            self.obj_fingertip_pos_seq_offset[30],
            self.hand_pos,
            self.mano_kpts_pos_ref[:, 0],
            self.actions,
            self.hand_dof_vel,
            self.fingertip_contact_forces,
            self.obj_linvel,
            self.cfg.action_penalty_scale,
            self.cfg.dof_penalty_scale,
            self.obj_rest_z,
            self.episode_length_buf.float(),
        )

        for key, value in logs_dict.items():
            if key not in self.logs_dict:
                self.logs_dict[key] = value.detach()
            else:
                self.logs_dict[key] += value.detach()

        if self.play:
            step = self.episode_length_buf[0].item()
            if step % 10 == 0:  # print every 10 steps
                ct = logs_dict["debug/contact_total"][0].item()
                cth = logs_dict["debug/contact_thumb"][0].item()
                cfi = logs_dict["debug/contact_fingers"][0].item()
                vz = logs_dict["debug/obj_linvel_z"][0].item()
                lh = (self.obj_pos[0, 2] - self.obj_rest_z).clamp(min=0).item()
                print(f"  step={step:3d}  contact: thumb={cth:.2f} fingers={cfi:.2f} total={ct:.2f} | obj_vz={vz:.4f} | lift_h={lh:.4f}")

        if "log" not in self.extras:
            self.extras["log"] = dict()

        return total_reward


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()

        self.time_out = self.episode_length_buf >= self.max_episode_length - 1

        early_terminate = self.early_terminate if self.termination else torch.zeros_like(self.early_terminate, device=self.device)
        return early_terminate, self.time_out


    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        super()._reset_idx(env_ids)
        self._reset_object(env_ids)
        self._reset_hand(env_ids)

        for key, value in self.logs_dict.items():
            self.extras["log"][key] = value.mean()

        if self.play and len(self.logs_dict) > 0:
            c_thumb   = self.logs_dict.get("debug/contact_thumb",   torch.zeros(1, device=self.device)).mean().item()
            c_fingers = self.logs_dict.get("debug/contact_fingers", torch.zeros(1, device=self.device)).mean().item()
            c_total   = self.logs_dict.get("debug/contact_total",   torch.zeros(1, device=self.device)).mean().item()
            vel_z     = self.logs_dict.get("debug/obj_linvel_z",    torch.zeros(1, device=self.device)).mean().item()
            lift      = self.logs_dict.get("reward/lift",           torch.zeros(1, device=self.device)).mean().item()
            print(f"[Episode] contact: thumb={c_thumb:.3f}  fingers={c_fingers:.3f}  total={c_total:.3f} | obj_vel_z={vel_z:.4f} | lift={lift:.4f}")
            # thumb position diagnostic — printed once per episode reset
            thumb_robot = self.fingertip_pos[0, 0].cpu()
            thumb_ref   = self.fingertip_pos_ref[0, 0].cpu()
            obj_p       = self.obj_pos[0].cpu()
            obj_ref_p   = self.obj_pos_ref[0].cpu()
            ref_to_obj     = (thumb_ref   - obj_p).norm().item()
            ref_to_obj_ref = (thumb_ref   - obj_ref_p).norm().item()
            print(f"[Thumb]   robot=({thumb_robot[0]:.3f},{thumb_robot[1]:.3f},{thumb_robot[2]:.3f})"
                  f"  ref=({thumb_ref[0]:.3f},{thumb_ref[1]:.3f},{thumb_ref[2]:.3f})"
                  f"  dist_to_ref={((thumb_robot-thumb_ref).norm()).item():.3f}m"
                  f"  ref→obj={ref_to_obj:.3f}m  ref→obj_ref={ref_to_obj_ref:.3f}m")

        self.logs_dict = dict()

        self.successes[env_ids] = 0
        self._compute_intermediate_values()


    def _set_object_state(self, pos, rot, env_ids, vel=None):
        default_states = self.object.data.default_root_state[env_ids].clone()
        default_states[:, :3] = pos + self.scene.env_origins[env_ids]
        default_states[:, 3:7] = rot

        if vel is not None:
            default_states[:, 7:13] = vel

        self.object.write_root_state_to_sim(default_states, env_ids=env_ids)

        self.obj_pos[env_ids] = self.obj_pos_reset[env_ids]
        self.obj_rot[env_ids] = self.obj_rot_reset[env_ids]


    def _reset_object(self, env_ids):
        self.obj_pos_reset[env_ids] = self.obj_pos_seq[0]
        self.obj_rot_reset[env_ids] = self.obj_rot_seq[0]
        self._set_object_state(self.obj_pos_reset[env_ids], self.obj_rot_reset[env_ids], env_ids)


    def _set_hand_state(self, pos, rot, dof_pos, dof_vel, root_vel, dof_target, ext_force, ext_torque, env_ids):
        hand_default_state = self.hand.data.default_root_state.clone()
        hand_default_state[env_ids, 0:3] = pos + self.scene.env_origins[env_ids]
        hand_default_state[env_ids, 3:7] = rot
        hand_default_state[env_ids, 7:13] = root_vel

        self.hand.write_root_pose_to_sim(hand_default_state[env_ids, :7], env_ids=env_ids)
        self.hand.write_root_velocity_to_sim(hand_default_state[env_ids, 7:13], env_ids=env_ids)
        self.hand.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)
        self.hand.set_joint_position_target(dof_target[:, self.actuated_dof_indices], self.actuated_dof_indices, env_ids=env_ids)
        self.hand.set_external_force_and_torque(ext_force, ext_torque, env_ids=env_ids, is_global=self.is_global)

        self.prev_dof_actions[env_ids] = dof_target.clone()
        self.cur_dof_actions[env_ids] = dof_target.clone()
        self.prev_forces[env_ids] = ext_force[:, self.root_body[0], :].clone()
        self.prev_torques[env_ids] = ext_torque[:, self.root_body[0], :].clone()

        self.hand_pos[env_ids] = self.hand_pos_reset[env_ids]
        self.hand_rot[env_ids] = self.hand_rot_reset[env_ids]


    def _reset_hand(self, env_ids):
        self.hand_pos_reset[env_ids] = self.hand_pos_reset[env_ids]
        self.hand_rot_reset[env_ids] = self.hand_rot_reset[env_ids]

        dof_pos = self.hand_dof_pos_reset[env_ids]
        dof_vel = torch.zeros_like(self.hand.data.default_joint_vel[env_ids])
        root_vel = torch.zeros_like(self.hand.data.default_root_state[env_ids, 7:13])

        hand_global_force = torch.zeros((len(env_ids), self.hand.num_bodies, 3), device=self.device)
        hand_global_torque = torch.zeros((len(env_ids), self.hand.num_bodies, 3), device=self.device)

        self._set_hand_state(self.hand_pos_reset[env_ids], self.hand_rot_reset[env_ids], dof_pos, dof_vel, root_vel, dof_pos, hand_global_force, hand_global_torque, env_ids)


    def _collect_target(self):
        t = self.episode_length_buf
        self.t = t
        t_next = torch.clamp(t + 1, max=(self.max_episode_length-1))

        self.obj_pos_ref = self.obj_pos_seq[t]
        self.obj_rot_ref = self.obj_rot_seq[t]
        self.obj_linvel_ref = self.obj_linvel_seq[t]
        self.obj_angvel_ref = self.obj_angvel_seq[t]
        self.obj_linvel_value_ref = self.obj_linvel_value_seq[t]
        self.obj_angvel_value_ref = self.obj_angvel_value_seq[t]

        self.fingertip_pos_ref = self.fingertip_pos_seq[t]
        self.mano_kpts_pos_ref = self.mano_kpts_pos_seq[t]

        self.obj_pos_next = self.obj_pos_seq[t_next]
        self.obj_rot_next = self.obj_rot_seq[t_next]
        self.obj_linvel_next = self.obj_linvel_seq[t_next]
        self.obj_angvel_next = self.obj_angvel_seq[t_next]
        self.obj_linvel_value_next = self.obj_linvel_value_seq[t_next]
        self.obj_angvel_value_next = self.obj_angvel_value_seq[t_next]

        self.hand_dof_next = self.hand_dof_seq[t_next]
        self.fingertip_pos_next = self.fingertip_pos_seq[t_next]
        self.mano_kpts_pos_next = self.mano_kpts_pos_seq[t_next]


    def _collect_state(self):
        object_state = self.object.data.root_state_w
        self.obj_pos = object_state[:,:3] - self.scene.env_origins
        self.obj_rot = object_state[:,3:7]
        self.obj_linvel = object_state[:,7:10]
        self.obj_angvel = object_state[:,10:13]

        hand_state = self.hand.data.root_state_w
        self.hand_pos = hand_state[:, :3] - self.scene.env_origins
        self.hand_rot = hand_state[:, 3:7]
        self.hand_linvel = hand_state[:,7:10]
        self.hand_angvel = hand_state[:,10:13]
        self.hand_dof_pos = self.hand.data.joint_pos
        self.hand_dof_vel = self.hand.data.joint_vel

        body_state = self.hand.data.body_state_w[:, self.hand_bodies]
        hand_bodies_pos = body_state[:, :, :3]
        self.hand_bodies_pos = hand_bodies_pos - self.scene.env_origins.unsqueeze(1)
        self.hand_bodies_rot = body_state[:, :, 3:7]
        self.hand_bodies_linvel = body_state[:, :, 7:10]
        self.hand_bodies_angvel = body_state[:, :, 10:13]

        fingertip_pos = self.hand_bodies_pos[:, self.fingertip_bodies]
        self.fingertip_rot = self.hand_bodies_rot[:, self.fingertip_bodies]
        self.fingertip_linvel = self.hand_bodies_linvel[:, self.fingertip_bodies]
        self.fingertip_angvel = self.hand_bodies_angvel[:, self.fingertip_bodies]

        self.normal = quat_apply(self.fingertip_rot, self.fingertip_normal)
        offset = quat_apply(self.fingertip_rot, self.fingertip_offset)
        self.hand_kpts_pos[:, self.cfg.MANO_kpts_except_fingertips] = self.hand_bodies_pos[:, self.cfg.body_to_kpts_except_fingertips]
        self.hand_kpts_pos[:, self.cfg.MANO_fingertips] = fingertip_pos + offset

        self.fingertip_pos = self.hand_kpts_pos[:, self.cfg.MANO_fingertips]

        for i in range(self.num_fingertips):
            force = self.contact_sensors[i].data.force_matrix_w
            self.fingertip_contact_forces[:, i] = force[:, 0, 0]
        self.fingertip_contact_forces_buf[:, 0] = torch.clamp_min((self.fingertip_contact_forces * (-self.normal)).sum(dim=-1), 0)


    def _compute_intermediate_values(self):
        self._collect_target()
        self._collect_state()

        self.delta_obj_pos = self.obj_pos - self.obj_pos_ref
        self.delta_obj_pos_value = torch.norm(self.delta_obj_pos, p=2, dim=-1)
        self.delta_fingertip_pos = torch.norm(
            self.fingertip_pos - self.fingertip_pos_ref, p=2, dim=-1
        )

        self.hand_far_apart = (
            torch.norm(self.hand_pos - self.mano_kpts_pos_ref[:, 0], p=2, dim=-1) > 0.5
        )
        self.obj_far_apart = self.delta_obj_pos_value > 0.3
        self.early_terminate = self.hand_far_apart | self.obj_far_apart

        debug_vis1 = self.mano_kpts_pos_ref[:, self.cfg.MANO_fingertips] + self.scene.env_origins.unsqueeze(1)
        self.goal_markers.visualize(debug_vis1.view(-1,3))
        debug_vis2 = self.hand_kpts_pos[:, self.cfg.MANO_fingertips] + self.scene.env_origins.unsqueeze(1)
        self.debug_markers.visualize(debug_vis2.view(-1,3))


    def compute_full_observations(self):
        obs = torch.cat(
            (
                self.hand_pos,                                                         # 3
                quat_to_6d(self.hand_rot),                                             # 6
                self.hand_linvel * self.cfg.vel_obs_scale,                             # 3
                self.hand_angvel * self.cfg.vel_obs_scale,                             # 3
                self.hand_dof_pos,                                                     # 24
                self.hand_dof_vel * self.cfg.vel_obs_scale,                            # 24
                self.obj_pos,                                                          # 3
                quat_to_6d(self.obj_rot),                                              # 6
                self.obj_linvel * self.cfg.vel_obs_scale,                              # 3
                self.obj_angvel * self.cfg.vel_obs_scale,                              # 3
                self.fingertip_pos.view(self.num_envs, -1),                            # 15 (5x3)
                self.fingertip_contact_forces_buf[:, 0],                               # 5
                self.obj_pos_ref,                                                      # 3
                quat_to_6d(self.obj_rot_ref),                                          # 6
                self.obj_linvel_ref * self.cfg.vel_obs_scale,                          # 3
                self.obj_angvel_ref * self.cfg.vel_obs_scale,                          # 3
                self.mano_kpts_pos_ref.view(self.num_envs, -1),                        # 63 (21x3)
                self.fingertip_pos_ref.view(self.num_envs, -1),                        # 15 (5x3)
                self.hand_dof_next,                                                    # 24
            ),
            dim=-1,
        )
        return obs


@torch.jit.script
def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower


@torch.jit.script
def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)


@torch.jit.script
def compute_rewards(
    obj_pos: torch.Tensor,
    obj_pos_ref: torch.Tensor,
    obj_rot: torch.Tensor,
    obj_rot_ref: torch.Tensor,
    fingertip_pos: torch.Tensor,
    fingertip_pos_ref: torch.Tensor,
    pregrasp_rel: torch.Tensor,
    hand_pos: torch.Tensor,
    wrist_pos_ref: torch.Tensor,
    actions: torch.Tensor,
    hand_dof_vel: torch.Tensor,
    fingertip_contact_forces: torch.Tensor,
    obj_linvel: torch.Tensor,
    action_penalty_scale: float,
    dof_penalty_scale: float,
    table_z: float,
    progress: torch.Tensor,
):
    obj_pos_err = torch.norm(obj_pos - obj_pos_ref, p=2, dim=-1)
    obj_pos_reward = torch.exp(-2.0 * obj_pos_err)

    rot_dot = torch.abs((obj_rot * obj_rot_ref).sum(dim=-1)).clamp(-1.0, 1.0)
    obj_rot_reward = torch.exp(-2.0 * (1.0 - rot_dot))

    # split direction: ref capsule -> ref fingertip (no lift bias); actual capsule -> robot fingertip
    obj_to_ref   = fingertip_pos_ref - obj_pos_ref.unsqueeze(1)  # (B, 5, 3)
    obj_to_robot = fingertip_pos     - obj_pos.unsqueeze(1)      # (B, 5, 3)

    obj_to_ref_n   = obj_to_ref   / (torch.norm(obj_to_ref,   p=2, dim=-1, keepdim=True) + 1e-8)
    obj_to_robot_n = obj_to_robot / (torch.norm(obj_to_robot, p=2, dim=-1, keepdim=True) + 1e-8)

    cos_sim    = (obj_to_ref_n * obj_to_robot_n).sum(dim=-1).clamp(-1.0, 1.0)  # (B, 5)
    dir_reward = torch.exp(-2.0 * (1.0 - cos_sim).sum(dim=-1))                 # (B,)

    # per-finger dir gate: thumb and fingers gated separately — mean can hide a bad thumb
    per_finger_dir  = torch.clamp(((cos_sim + 1.0) * 0.5 - 0.3) / 0.4, 0.0, 1.0)  # (B, 5)
    thumb_dir_gate  = per_finger_dir[:, 0]                                           # (B,)
    finger_dir_gate = per_finger_dir[:, 1:].mean(dim=-1)                             # (B,)
    dir_gate        = thumb_dir_gate * finger_dir_gate                               # (B,)

    # radial: press_margin shifts target 5mm inside reference surface to eliminate dead zone
    fingertip_dist = torch.norm(obj_to_robot, p=2, dim=-1)              # (B, 5)
    ref_dist       = torch.norm(obj_to_ref,   p=2, dim=-1)              # (B, 5)
    press_margin   = 0.005
    target_dist    = ref_dist - press_margin
    radial_over    = torch.clamp(fingertip_dist - target_dist, min=0.0) # (B, 5)

    radial_reward_mean = torch.exp(-15.0 * radial_over).mean(dim=-1)
    radial_reward_min  = torch.exp(-20.0 * radial_over).min(dim=-1).values
    radial_reward      = 0.7 * radial_reward_mean + 0.3 * radial_reward_min

    fingertip_reward = 0.5 * dir_reward + 1.5 * dir_reward * radial_reward

    near_gate = torch.clamp((radial_reward - 0.45) / 0.40, 0.0, 1.0)

    # approach reward: drives fingers to pregrasp pose (frame 30) in object-relative space,
    # active only before grip established and only for first 35 steps
    fingertip_rel    = obj_to_robot                                      # (B, 5, 3)
    approach_err     = torch.norm(fingertip_rel - pregrasp_rel.unsqueeze(0), p=2, dim=-1).mean(dim=-1)
    phase_gate       = (progress < 35.0).float()

    per_tip_err = torch.norm(fingertip_pos - fingertip_pos_ref, p=2, dim=-1)  # (B, 5)
    thumb_err   = per_tip_err[:, 0]
    other_tip_reward = torch.exp(-2.0 * per_tip_err[:, 1:]).min(dim=-1).values  # logging only

    thumb_reward_component = (
        0.3 * torch.exp(-2.0  * thumb_err) +
        0.7 * torch.exp(-20.0 * thumb_err)
    )

    wrist_err    = torch.norm(hand_pos - wrist_pos_ref, p=2, dim=-1)
    wrist_reward = torch.exp(-2.0 * wrist_err)

    contact_mag     = torch.norm(fingertip_contact_forces, p=2, dim=-1)  # (B, 5)
    contact_thumb   = contact_mag[:, 0]
    contact_fingers = contact_mag[:, 1:].sum(dim=-1)
    contact_total   = contact_mag.sum(dim=-1)

    thumb_contact_gate  = torch.tanh(contact_thumb   / 0.3)
    finger_contact_gate = torch.tanh(contact_fingers / 0.8)
    bilateral_gate      = thumb_contact_gate * finger_contact_gate

    # dir_gate on both contact and lift — tip grasping cannot unlock lift reward
    contact_reward = dir_gate * (
        2.0 * thumb_contact_gate
        + 2.0 * finger_contact_gate
        + 4.0 * bilateral_gate
    )

    lift_height  = (obj_pos[:, 2] - table_z).clamp(min=0.0)
    contact_gate = torch.clamp(contact_total / 0.1, 0.0, 1.0)

    grasp_gate   = dir_gate * near_gate * bilateral_gate
    lift_reward  = grasp_gate * 5.0 * torch.tanh(lift_height * 20.0)
    vel_z_reward = grasp_gate * 0.8 * contact_gate * torch.tanh(obj_linvel[:, 2] * 10.0)

    # approach reward: pregrasp frame target, off once bilateral contact established
    approach_reward = phase_gate * (1.0 - bilateral_gate) * torch.exp(-10.0 * approach_err)

    action_penalty  = action_penalty_scale * torch.sum(actions ** 2, dim=-1)
    dof_vel_penalty = dof_penalty_scale    * torch.sum(hand_dof_vel ** 2, dim=-1)

    reward = (
        obj_pos_reward
        + obj_rot_reward
        + fingertip_reward
        + approach_reward
        + thumb_reward_component
        + wrist_reward
        + contact_reward
        + lift_reward
        + vel_z_reward
        + action_penalty
        + dof_vel_penalty
    )
    reward = torch.clamp_min(reward, 0.0)

    logs_dict = {
        "reward/total":            reward,
        "reward/obj_pos":          obj_pos_reward,
        "reward/obj_rot":          obj_rot_reward,
        "reward/fingertip":        fingertip_reward,
        "reward/approach":         approach_reward,
        "reward/dir":              dir_reward,
        "reward/radial":           radial_reward,
        "reward/other_tip_min":    other_tip_reward,
        "reward/thumb_tip":        thumb_reward_component,
        "reward/wrist":            wrist_reward,
        "reward/contact":          contact_reward,
        "reward/lift":             lift_reward,
        "reward/vel_z":            vel_z_reward,
        "reward/action_penalty":   action_penalty,
        "reward/dof_vel_penalty":  dof_vel_penalty,
        "gate/dir":                dir_gate,
        "gate/near":               near_gate,
        "gate/bilateral":          bilateral_gate,
        "gate/grasp":              grasp_gate,
        "gate/thumb_dir":          thumb_dir_gate,
        "gate/finger_dir":         finger_dir_gate,
        "gate/thumb_contact":      thumb_contact_gate,
        "gate/finger_contact":     finger_contact_gate,
        "debug/approach_err":      approach_err,
        "debug/dir_mean":          ((cos_sim + 1.0) * 0.5).mean(dim=-1),
        "debug/radial_over_mean":  radial_over.mean(dim=-1),
        "debug/thumb_err":         thumb_err,
        "debug/contact_thumb":     contact_thumb,
        "debug/contact_fingers":   contact_fingers,
        "debug/contact_total":     contact_total,
        "debug/lift_height":       lift_height,
        "debug/obj_linvel_z":      obj_linvel[:, 2],
    }

    return reward, logs_dict


# Utils
def quat_to_6d(quat: torch.Tensor) -> torch.Tensor:
    return matrix_to_rotation_6d(matrix_from_quat(F.normalize(quat, dim=-1)))


def rotation_6d_to_matrix(rot_6d: torch.Tensor) -> torch.Tensor:
    a1 = rot_6d[..., 0:3]
    a2 = rot_6d[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    return matrix[..., :2, :].clone().reshape(*matrix.shape[:-2], 6)


def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    return axis_angle_from_quat(quat_from_matrix(matrix))
