import os
import pickle as pkl
import time

import imageio
import numpy as np
import pyrender
import trimesh
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from geometry import *
from smpl_tool import batch_hybrIK
from smplx import SMPL as _SMPL
from smplx.lbs import blend_shapes, vertices2joints

prj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
render_path = os.path.join(prj_root, 'render_res')


class SMPL(_SMPL):
    """ Extension of the official SMPL implementation """

    def __init__(self, gender, device='cpu', dtype=torch.float32, hybrik_joints=False, *args, **kwargs):
        assert gender in ['male', 'female', 'neutral']
        prj_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.smpl_model_dir = os.path.join(prj_path, r'data/smpl_file/')
        super(SMPL, self).__init__(self.smpl_model_dir, gender=gender, dtype=dtype, *args, **kwargs)

        j_regressor_h36m = np.load(os.path.join(self.smpl_model_dir, r'J_regressor_h36m_correct.npy'))
        h36m_j_regressor = torch.tensor(j_regressor_h36m, dtype=dtype)
        self.register_buffer('h36m_j_regressor', h36m_j_regressor, persistent=False)

        self.hybrik_joints_index = [411, 2445, 5905, 3216, 6617]
        self.use_hybrik_joints = hybrik_joints

        parents29 = torch.tensor(
            [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21,
             15, 22, 23, 10, 11])
        self.register_buffer('parents29', parents29, persistent=False)

        children29 = torch.tensor(
            [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 27, 28, 15, 16, 17, 24, 18, 19, 20, 21, 22, 23, 25, 26,
             -1, -1, -1, -1, -1], device=device)
        self.register_buffer('children29', children29, persistent=False)

        children24 = torch.tensor(
            [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, -1, -1, 15, 16, 17, -1, 18, 19, 20, 21, 22, 23, -1, -1],
            device=device)
        self.register_buffer('children24', children24, persistent=False)

        self.joints_with_twist = list(range(4, 12)) + list(range(15, 24))  # len==17
        self.h36m_j14_idx = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10]

        self.to(device)

    def get_device(self):
        return self.h36m_j_regressor.device

    def get_parents(self):
        return self.parents29 if self.use_hybrik_joints else self.parents

    def check(self, *inputs):
        results = []
        for inp in inputs:
            if inp is None:
                pass
            elif not torch.is_tensor(inp):
                inp = torch.tensor(inp, dtype=self.dtype, device=self.get_device())
            else:
                if inp.device != self.get_device():
                    inp = inp.to(self.get_device())
                if inp.dtype != self.dtype:
                    inp = inp.to(self.dtype)
            results.append(inp)
        if len(results) == 1:
            return results[0]
        return results

    def forward(self, beta=None, pose=None, transl=None, *args, **kwargs):
        """
        :param beta: shape(bs,10) or (10)
        :param pose: shape(bs,24,3) or (bs,72) or (bs,24,3,3) or (bs,24,9)
        :param transl: shape(bs,3)  unit:meter
        :return
        """
        beta, pose, transl = self.check(beta, pose, transl)
        bs = pose.shape[0]

        if transl is None:  # default smpl transl has grad
            transl = torch.zeros([bs, 3], dtype=self.dtype, device=self.get_device())
        if beta.shape == torch.Size([10]):
            beta = beta[None].expand(bs, -1)
        if pose.shape[1:] in [torch.Size([24, 3]), torch.Size([72])]:
            pose2rot = True
            global_orient = pose.reshape(-1, 72)[:, :3]
            body_pose = pose.reshape(-1, 72)[:, 3:]
        else:
            pose2rot = False
            global_orient = pose.reshape(-1, 24, 3, 3)[:, :1]
            body_pose = pose.reshape(-1, 24, 3, 3)[:, 1:]

        smpl_output = super(SMPL, self).forward(
            global_orient=global_orient, body_pose=body_pose, pose2rot=pose2rot,
            betas=beta, transl=transl, *args, **kwargs)
        joints, vertices = smpl_output.joints[:, :24], smpl_output.vertices

        if self.use_hybrik_joints:
            hybrik_joints = vertices[:, self.hybrik_joints_index].clone()
            joints = torch.cat([joints, hybrik_joints], dim=1)

        h36m_j_regressor_batch = self.h36m_j_regressor[None].expand(bs, -1, -1)
        mid = vertices.mean(dim=1, keepdim=True)  # bs, 1, 3
        joints17 = h36m_j_regressor_batch.matmul(vertices - mid) + mid

        # bone_dir=(joints-joints[:,parents])[:,1:]
        return joints, vertices, joints17

    def cal_rest_pose(self, beta, return_vertices=False):
        """
        :param beta: bs,10
        :return: smpl joints at rest pose
        """
        beta = self.check(beta)
        v_shaped = self.v_template + blend_shapes(beta, self.shapedirs)
        joints = vertices2joints(self.J_regressor, v_shaped)
        if self.use_hybrik_joints:
            hybrik_joints = v_shaped[:, self.hybrik_joints_index].clone()
            joints = torch.cat([joints, hybrik_joints], dim=1)
        if return_vertices:
            return joints, v_shaped
        else:
            return joints

    def cmp_beta(self, beta1, beta2):
        """ compare shape parameters at rest pose
        :param beta1: bs,10
        :param beta2: bs,10
        :return: jpe:bs vpe:bs
        """
        joints_1, vertices_1 = self.cal_rest_pose(beta1, return_vertices=True)
        joints_2, vertices_2 = self.cal_rest_pose(beta2, return_vertices=True)

        joints_1 = joints_1 - joints_1.mean(dim=-2, keepdim=True)
        vertices_1 = vertices_1 - vertices_1.mean(dim=-2, keepdim=True)
        joints_2 = joints_2 - joints_2.mean(dim=-2, keepdim=True)
        vertices_2 = vertices_2 - vertices_2.mean(dim=-2, keepdim=True)

        jpe = (joints_1 - joints_2).norm(dim=-1)[:, :24].mean(dim=-1) * 1000
        vpe = (vertices_1 - vertices_2).norm(dim=-1).mean(dim=-1) * 1000
        return jpe, vpe

    def cal_swing_twist(self, beta, pose):
        """
        :param beta:
        :param pose:
        :return: twist angle and swing of global rotations
        """
        beta, theta = self.check(beta, pose)
        bs = pose.shape[0]
        if beta.shape == torch.Size([10]):
            beta = beta[None].expand(bs, -1)

        rest_pose = self.cal_rest_pose(beta)
        rel_rest_pose = (rest_pose - rest_pose[:, self.get_parents()])
        pose[:, :9] = torch.eye(3).reshape(1, 9)
        global_rot = self.rotation_relative2global(pose)
        twist_radian, swing = rotation2swing_twist(
            rel_rest_pose.reshape(-1, 3),
            global_rot[:, self.get_parents()].reshape(-1, 3, 3))
        return twist_radian.reshape(bs, -1), swing.reshape(bs, -1, 3, 3)

    def visualize_vert(self, vert_idx):
        import trimesh
        vertices = self.cal_rest_pose(np.zeros((1, 10)), return_vertices=True)[1][0]
        # vertices[:, 1:] *= -1
        m = trimesh.Trimesh(vertices=vertices, faces=self.faces)
        face_vis = ((self.faces==vert_idx).sum(axis=-1)>0)
        f_color = torch.tensor(trimesh.visual.color.interpolate(face_vis, color_map="viridis")[:, :3])
        m.visual.face_colors = f_color
        m.show()

    def rotation_global2relative(self, global_rotation):
        """
        :param global_rotation: bs,N,3,3
        :return: relative rotation: bs,N,3,3
        """
        global_rotation = self.check(global_rotation)
        nest_rotation_root = global_rotation[:, :1]
        global_rotation_parents = global_rotation[:, self.parents[1:]]
        nest_rotation_other = global_rotation_parents.transpose(-1, -2).matmul(global_rotation[:, 1:])
        return torch.cat([nest_rotation_root, nest_rotation_other], dim=1)

    def rotation_relative2global(self, nest_rotation):
        """
        :param nest_rotation: bs,24,3,3 or bs,24,9 or bs,24,3 or bs,72
        :return: global rotation: bs,N,3,3
        """
        nest_rotation = self.check(nest_rotation)
        if nest_rotation.shape[1:] in [torch.Size([24, 3]), torch.Size([72])]:
            nest_rotation = batch_rodrigues(nest_rotation.reshape(-1, 3))
        nest_rotation = nest_rotation.reshape(-1, 24, 3, 3)

        transform_chain = [nest_rotation[:, 0]]
        parents = self.parents  # parents24
        for i in range(1, 24):
            current_global_rotation = transform_chain[parents[i]].matmul(nest_rotation[:, i])
            transform_chain.append(current_global_rotation)
        global_rotation = torch.stack(transform_chain, dim=1)
        return global_rotation

    def rotate_skeleton(self, rest_pose, global_rotation):
        """
        :param rest_pose: bs,N,3
        :param global_rotation: bs,24,3,3
        :return: result pose: bs,N,3
        """
        rest_pose, global_rotation = self.check(rest_pose, global_rotation)
        parents = self.get_parents()
        rel_rest_pose = rest_pose - rest_pose[:, parents]
        rel_pose = global_rotation[:, parents].matmul(rel_rest_pose.unsqueeze(-1))
        pose = [rest_pose[:, 0]]  # root_position is fixed in SMPL
        for joint_idx in range(1, rest_pose.shape[1]):
            pose.append(pose[parents[joint_idx]] + rel_pose[:, joint_idx].squeeze(-1))
        return torch.stack(pose, dim=1)

    def cal_joints(self, beta, pose, transl=None, hybrik_joints=False):
        """
        :param hybrik_joints: the position of hybrik-joints is not accurate! Average position error is about 9mm.
        :param beta: shape(bs,10) or (10)
        :param pose: shape(bs,24,3) or (bs,72) or (bs,24,3,3) or (bs,24,9)
        :param transl: shape(bs,3)
        :return smpl joints position
        """
        beta, pose, transl = self.check(beta, pose, transl)

        if beta.shape == torch.Size([10]):
            beta = beta[None].expand(pose.shape[0], -1)

        rest_pose = self.cal_rest_pose(beta)
        global_rotation = self.rotation_relative2global(pose)

        joints = self.rotate_skeleton(rest_pose, global_rotation)
        if not hybrik_joints and pose.shape[1] == 29:
            joints = joints[:, 24]
        if transl is not None:
            return joints + transl.unsqueeze(1)
        else:
            return joints

    def cal_ortho_twist(self, beta, pose):
        rest_pose = self.cal_rest_pose(beta)
        rest_pose_ot = self.cal_rest_pose_ortho_twist(rest_pose)
        global_rotation = self.rotation_relative2global(pose)
        ot = global_rotation[:, self.get_parents()].matmul(rest_pose_ot.unsqueeze(-1))
        return ot.squeeze(-1)

    def cal_rest_pose_ortho_twist(self, joints):
        """
        :param joints:   torch.tensor BxNx3
        :return: the ortho twist of rest pose skeleton(twists of some bones are useless)
        """
        joints = self.check(joints)
        bone_vector = joints - joints[:, self.get_parents()]
        Bx, By, Bz = bone_vector[:, :, 0], bone_vector[:, :, 1], bone_vector[:, :, 2]
        tmp = Bx ** 2 + By ** 2
        Tx = -1 * Bx * Bz / tmp
        Ty = -1 * By * Bz / tmp
        Tz = Tx * 0 + 1
        ot = torch.stack((Tx, Ty, Tz), dim=-1) * -1
        return F.normalize(ot, p=2, dim=-1)

    def pose_world2camera(self, cam_quat, cam_t, beta, theta_world, tran_world):
        """  convert smpl pose params from world coordinate system to camera coordinate system
        :param cam_quat: cam extrinsic parameters  bs,4
        :param cam_t: cam intrinsic parameters bs,3
        :param theta_world: smpl params, shape(bs,72) or bs,24,3
        :param beta: smpl params, shape(bs,10)
        :param tran_world: smpl params, shape(bs,3)
        :return: new smpl params
        """
        cam_quat, cam_t, beta, theta_world, tran_world = self.check(
            cam_quat, cam_t, beta, theta_world, tran_world)
        if beta.shape == torch.Size([10]):
            beta = beta[None].expand(theta_world.shape[0], -1)

        # cam_tran=world2cam(world_tran)
        rest_pose = self.cal_rest_pose(beta)
        root_position_world = tran_world + rest_pose[:, 0]
        tran_cam = qrot(cam_quat, root_position_world) + cam_t - rest_pose[:, 0]

        # cam_R*global_orientation*root_bone=new_orientation*root_bone
        theta_cam = theta_world.clone().reshape(-1, 24, 3)
        global_orientation_world = axisang2quat(theta_cam[:, 0])
        global_orientation_cam = quaternion_raw_multiply(cam_quat, global_orientation_world)
        theta_cam[:, 0] = quaternion_to_angle_axis(global_orientation_cam)

        return theta_cam.reshape(-1, 72), beta, tran_cam

    def render(self, beta, pose, transl, img_file=None, cam_intrinsics=None, suffix='.jpg', twist=False, res_path=render_path, prefix='rendered_'):
        """ mesh in the same batch need to be of the same gender
        :param beta: SMPL shape parameters
        :param pose: SMPL pose parameters in camera coordinate
        :param transl: shape(bs,3) SMPL pose parameters
        :param img_file: len(bs) list of image file path
        :param cam_intrinsics: shape(bs,3,3) or (3,3) to project mesh to image
        :param suffix: output image filename suffix e.g. '_scene1.jpg
        :return: None
        """
        beta, pose, transl, cam_intrinsics = self.check(beta, pose, transl, cam_intrinsics)
        vertex_color = [200, 200, 200]
        twist_len = 200  # mm
        twist_radius = 7  # mm
        light_intensity = 5

        bs = pose.shape[0]
        if cam_intrinsics is None:
            cam_intrinsics = np.array([[1.96185286e+03, 0.00000000e+00, 5.40000000e+02],
                                       [0.00000000e+00, 1.96923077e+03, 9.60000000e+02],
                                       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        else:
            cam_intrinsics = np.array(cam_intrinsics.cpu())
        if cam_intrinsics.shape == (3, 3):
            cam_intrinsics = cam_intrinsics[None].repeat(bs, 0)
        joints, vertices, _joints17 = self.forward(beta=beta, pose=pose, transl=transl)
        ortho_twist = self.cal_ortho_twist(beta, pose).cpu()
        bone_mid = (joints + joints[:, self.get_parents()]).cpu() / 2

        # print(joints[0, 0])

        def create_cylinder_between_points(point1, point2, radius=0.02):
            point1 = np.array(point1)
            point2 = np.array(point2)
            point1[1:] *= -1
            point2[1:] *= -1
            distance = np.linalg.norm(point1 - point2)
            direction = (point2 - point1) / distance

            # mesh = trimesh.Creation.cylinder(radius, distance)
            mesh_cylinder = trimesh.primitives.Cylinder(radius, distance)
            mesh_cylinder.visual.face_colors = np.array([235, 131, 52])

            orientation = np.array([0, 0, 1])
            rotation_matrix = trimesh.geometry.align_vectors(orientation, direction)
            rotation_matrix[:3, 3] = point1 + (point2 - point1) / 2
            mesh_cylinder.apply_transform(rotation_matrix)

            return mesh_cylinder

        # render each mesh
        for idx in range(bs):
            scene = pyrender.Scene()

            # add mesh to scene
            vs = vertices[idx].clone().cpu()
            vs[:, 1:] *= -1
            mesh = trimesh.Trimesh(vertices=vs, faces=self.faces, vertex_colors=vertex_color)
            mesh = pyrender.Mesh.from_trimesh(mesh)
            scene.add(mesh)

            # add ortho twist to scene
            if twist:
                for joint_idx in range(joints.shape[1]):  # 6,9,16,17
                    if joint_idx in [0, 1, 2, 3, 12, 13, 14]:
                        continue  # these bones can not twist.
                    if joint_idx in [28, 27, 26, 25, 24] + [6, 9, 16, 17, 15, 10, 11]:
                        continue  # for looking well
                    sp = bone_mid[idx][joint_idx].numpy()
                    twist_len_ = twist_len / 1000 + 0.05 * np.array(
                        [1, 1, 1, 1, 1.8, 1.8, 1.5, 0.5, 0.5, 1,
                         0.5, 0.5, 1, 1, 1, 1, 1.3, 1.3, 1, 1,
                         0.5, 0.5, 1, 1, 1.8, 0.5, 0.5, 0.2, 0.2])
                    displace = -1 * twist_len_[joint_idx] * ortho_twist[idx][joint_idx].numpy()
                    ep = sp + displace
                    cylinder = create_cylinder_between_points(sp, ep, radius=twist_radius / 1000)
                    cylinder_mesh = pyrender.Mesh.from_trimesh(cylinder, smooth=False)
                    scene.add(cylinder_mesh)

            camera = pyrender.IntrinsicsCamera(  # cam_fx, cam_fy, cam_cx, cam_cy
                cam_intrinsics[idx, 0, 0], cam_intrinsics[idx, 1, 1],
                cam_intrinsics[idx, 0, 2], cam_intrinsics[idx, 1, 2])
            scene.add(camera, pose=np.eye(4))

            scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=light_intensity))
            scene.bg_color = (255, 255, 255) if img_file is None else (0.0, 0.0, 0.0)

            # project scene to image
            # pip install PyOpenGL==3.1.5   # just ignore PyOpenGL==3.1.5 is incompatible with pyrender==0.1.45
            # sudo apt install *OSmesa*    # set environment variable PYOPENGL_PLATFORM=osmesa
            if img_file is not None:
                input_img_path = img_file[idx]
                output_img_name = prefix + os.path.basename(input_img_path) + suffix
                image = imageio.imread_v2(input_img_path)
                viewport_width, viewport_height = image.shape[1], image.shape[0]
                renderer = pyrender.OffscreenRenderer(viewport_width=viewport_width, viewport_height=viewport_height,
                                                      point_size=1.0)
                color, depth = renderer.render(scene)
                result = color.copy()
                zero_positions = np.sum(result, axis=-1) < 150
                result[zero_positions] = image[zero_positions]
            else:
                output_img_name = f'{prefix}_{idx:04d}{suffix}'
                viewport_width, viewport_height = 1080, 1920
                renderer = pyrender.OffscreenRenderer(viewport_width=viewport_width, viewport_height=viewport_height,
                                                      point_size=1.0)
                color, depth = renderer.render(scene)
                result = color.copy()
            os.makedirs(res_path, exist_ok=True)
            imageio.imwrite(os.path.join(res_path, output_img_name), result)

    def render_obj(self, beta, pose=None, res_path=render_path, prefix=''):
        beta = self.check(beta)
        if beta.shape == torch.Size([10]):
            beta = beta[None]
        bs = beta.shape[0]
        if pose is None:
            pose = torch.zeros(bs, 72)
        vertices = self.forward(beta, pose)[1]
        os.makedirs(res_path, exist_ok=True)
        for i in range(bs):
            obj_file_name = os.path.join(res_path, f'{prefix}_{i:03d}.obj')
            with open(obj_file_name, 'w') as f:
                for vertex in vertices[i]:
                    f.write("v {} {} {}\n".format(vertex[0], -1*vertex[1], -1*vertex[2]))
                for face in self.faces:
                    f.write('f {} {} {}\n'.format(face[0] + 1, face[1] + 1, face[2] + 1))  # 1-indexed

    def bone_dir2pose(self, beta, bone_dir, rot_root, rot_chest):
        """
        :param beta: bs,10
        :param bone_dir: bs,24,3 or bs,29,3
        :param rot_root: bs,3,3
        :param rot_chest: bs,3,3
        :return: joint position, use mid of j24[[1,2,3]] as center.
        """
        # beta, bone_dir, rot_root,rot_chest=beta_x,bd_x,r0_x,r9_x
        parents = self.parents29 if bone_dir.shape[1] == 29 else self.parents
        rest_pose = self.cal_rest_pose(beta)[:, :bone_dir.shape[1]]
        rel_rest_pose = rest_pose - rest_pose[:, parents]
        children_of_pelvis = rot_root.unsqueeze(1).matmul(rel_rest_pose[:, 1:4, :, None]).squeeze(-1)
        mid = children_of_pelvis.mean(dim=1, keepdim=True)

        bone_len = rel_rest_pose.norm(dim=-1)[:, :bone_dir.shape[1]]  # bs,29
        rel_result_pose = bone_len.unsqueeze(-1) * F.normalize(bone_dir, p=2, dim=-1)  # bs,29,3
        rel_result_pose[:, 1:4] = children_of_pelvis  # - mid
        rel_result_pose[:, 12:15] = rot_chest.unsqueeze(1).matmul(rel_rest_pose[:, 12:15, :, None]).squeeze(-1)

        pose = [-1 * mid]
        for joint_idx in range(1, rest_pose.shape[1]):
            pose.append(pose[parents[joint_idx]] + rel_result_pose[:, joint_idx].unsqueeze(1))
        return torch.cat(pose, dim=1)

    def optimize_pose(self, aim_joints, init_beta, init_rel_pose, init_rot_root, init_rot_chest):
        """ use torch.optim.LBFGS to search a better ik solution for SMPL
        :param aim_joints: bs,24,3 or bs,29,3 the aim pose
        :param init_beta: bs,10 initial value of smpl shape parameters
        :param init_rel_pose: bs,24,3 or bs,29,3 initial value of bone direction
        :param init_rot_root: bs,3,3 initial value of root rotation, use mid of j24[[1,2,3]] as center.
        :param init_rot_chest: bs,3,3 initial value of joint9 rotation
        :return:
        """
        lr = 1e-4  # 1e-4
        max_iter = int(3)
        tolerance_grad = 1e-14
        tolerance_change = 1e-14

        if self.dtype is not torch.float64: print(f'optimize datatype is not float64')
        aim_joints, init_beta, init_rel_pose, init_rot_root, init_rot_chest = self.check(
            aim_joints, init_beta, init_rel_pose, init_rot_root, init_rot_chest)
        mid = aim_joints[:, 1:4].mean(dim=1, keepdim=True)
        aim_joints_ = (aim_joints - mid)  # [:, :24] 24@1

        bd_x = init_rel_pose.clone().detach().requires_grad_(True)  # [:, :24] 24@2
        r0_x = mat2quat(init_rot_root).clone().detach().requires_grad_(True)
        r9_x = mat2quat(init_rot_chest).clone().detach().requires_grad_(True)

        optimizer = torch.optim.LBFGS(
            [bd_x, r0_x, r9_x], lr=lr, max_iter=max_iter,
            tolerance_grad=tolerance_grad, tolerance_change=tolerance_change,
            history_size=1000, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            pose_tmp = self.bone_dir2pose(init_beta, bd_x, quat2mat(r0_x), quat2mat(r9_x))
            jpe_tmp = (pose_tmp - aim_joints_)[:, 1:].norm(dim=-1).mean() * 1000

            loss = jpe_tmp + 1e-3 * (bd_x.norm(dim=-1) - 1).abs().mean()
            loss.backward()  # retain_graph=True)
            # if optimizer.state_dict()['state'][0]['func_evals'] % 10 == 0:
            #     print('\t', optimizer.state_dict()['state'][0]['func_evals'], 'func_evals', jpe_tmp.item(), 'mm')
            return loss

        st = time.time()
        optimizer.step(closure)
        print(f"{time.time() - st:.2f}s   func_evals:{optimizer.state_dict()['state'][0]['func_evals']}")

        res_bd, res_r0, res_r9 = bd_x, quat2mat(r0_x), quat2mat(r9_x)
        final_pose = self.bone_dir2pose(init_beta, res_bd, res_r0, res_r9)
        final_jpe = (final_pose - aim_joints_)[:, 1:].norm(dim=-1).mean() * 1000

        print(f'optimize jpe: {final_jpe.item():.2f}mm')

        # cal leaf direction  24@3
        # leaf_bd = (aim_joints - mid)[:, -5:] - final_pose[:, self.parents29[-5:]]
        # res_bd = torch.cat([res_bd, leaf_bd], dim=1)
        return res_bd, res_r0, res_r9

    def compose_rot(self, beta, bone_dir, aim_ot, root_orientation, chest_orientation, leaf_rot=None):
        """ calculate smpl pose parameters using bone_direction and ortho_twist
        :param bone_dir: bs,29,3  which provide bone direction
            6 children of root or chest are not used. Root has no parents.
        :param aim_ot: bs,17,3 or bs,24,3 or bs,29,3 #todo
            There are 17or22 effective ones.
            6 children of root or chest are not used. Root has no parents.
            Twist direction of 5 Hybrik extra joints is optional.
        :param beta: bs,10
        :param root_orientation: bs,3x3  global rotation
            root and chest have more than 1 child, their twists are determined by their own children's position.
        :param chest_orientation: bs,3x3  global rotation
        :param leaf_rot: dict keys==[15, 22, 23, 10, 11] value shape(bs,3,3)
            the relative rotation of leaf joints.
        :return: smpl pose parameters
        """
        beta, bone_dir, aim_ot, root_orientation, chest_orientation = self.check(
            beta, bone_dir, aim_ot, root_orientation, chest_orientation)
        bs = bone_dir.shape[0]
        inp_ot_num = aim_ot.shape[1]
        tmp = self.check(torch.ones_like(bone_dir))  # ortho-twist
        if inp_ot_num == 17:
            tmp[:, self.joints_with_twist] = aim_ot
        elif inp_ot_num == 24:
            tmp[:, :24] = aim_ot
        else:
            tmp = aim_ot
        aim_ot = tmp

        # calculate rest pose bone direction and ortho twist
        rest_pose = self.cal_rest_pose(beta)
        rest_pose_ot = self.cal_rest_pose_ortho_twist(rest_pose)

        parents = self.get_parents()
        rest_bone_normed = F.normalize(rest_pose - rest_pose[:, parents], p=2, dim=-1)
        aim_bone_normed = F.normalize(bone_dir, p=2, dim=-1)
        rest_twist_normed = F.normalize(rest_pose_ot, p=2, dim=-1)
        aim_twist_normed = F.normalize(aim_ot, p=2, dim=-1)

        # Gram–Schmidt orthonormalizing
        aim_twist_orthonormalized = aim_twist_normed - aim_bone_normed * torch.sum(
            aim_twist_normed * aim_bone_normed, dim=2, keepdim=True)
        aim_twist_final = F.normalize(aim_twist_orthonormalized, p=2, dim=-1)

        # calculate the transition matrix
        rest_pose_coordinate = torch.stack(
            [rest_bone_normed, rest_twist_normed, rest_bone_normed.cross(rest_twist_normed, dim=-1)], dim=-1)
        aim_pose_coordinate = torch.stack(
            [aim_bone_normed, aim_twist_final, aim_bone_normed.cross(aim_twist_final, dim=-1)], dim=-1)
        composed_global_rotation = aim_pose_coordinate.matmul(rest_pose_coordinate.transpose(-1, -2))

        children = self.children29 if self.use_hybrik_joints else self.children24
        global_rotation_list = []
        for joint_idx in range(24):  # rest_pose.shape[1]):
            if joint_idx == 0:
                global_rotation_list.append(root_orientation)
            elif joint_idx == 9:
                global_rotation_list.append(chest_orientation)
            else:
                child_idx = children[joint_idx]
                if child_idx == -1:
                    assert (leaf_rot is not None), "Leaf rotations need to be provided."
                    assert (joint_idx in [15, 22, 23, 10, 11]), f'{joint_idx} child error'
                    global_rotation_list.append(leaf_rot.get(joint_idx))  # fake operation
                else:
                    global_rotation_list.append(composed_global_rotation[:, child_idx])
        global_rotation = torch.stack(global_rotation_list, dim=1)  # bs,24,3,3
        theta_res = self.rotation_global2relative(global_rotation)
        for k in leaf_rot.keys():
            theta_res[:, k] = leaf_rot[k]

        # if hybrik joint positions are provided calculate relative swing rotation, else set leaf joint to be static.
        if self.use_hybrik_joints and inp_ot_num != 29:
            # children[[15, 22, 23, 10, 11]]
            template_vector = rest_bone_normed[:, children[[15, 22, 23, 10, 11]]]
            aim_vector = aim_bone_normed[:, children[[15, 22, 23, 10, 11]]]
            aim_vector_ = global_rotation[:, parents[[15, 22, 23, 10, 11]]].transpose(-1, -2).matmul(
                aim_vector.unsqueeze(-1))
            swing_mat = cal_swing(template_vector.reshape(-1, 3), aim_vector_.reshape(-1, 3)).reshape(-1, 5, 3, 3)
            theta_res[:, [15, 22, 23, 10, 11]] = swing_mat
        # theta_res[:, [15, 22, 23, 10, 11]] = self.check(torch.eye(3).reshape(1, 1, 3, 3))

        return theta_res

    def ik_w_ortho_twist(self, joints, ortho_twist, beta=None, optimize_pose=False, leaf_rot=None):
        """
        :param joints: bs,24,3  used to calculate average bone length
        :param ortho_twist: bs,17,3 or bs,24,3 or bs,29,3 which is in smpl coordinate
        :param beta: bs,10 initial smpl shape parameters, which will be modified
        :param optimize_pose: whether to optimize the hybrik results
        :return: beta_new  bone_dir_new  chest_rotation  root_translation
        """
        # 考虑j29与 j24 两种情况 #hybrik_joints=True
        bs = joints.shape[0]
        joints, ortho_twist, beta = self.check(joints, ortho_twist, beta)

        def search_beta0():  # simplified SFSR
            def inner(x):
                beta_ = beta.clone()
                beta_[:, 0] = x
                rest_pose = self.cal_rest_pose(beta_)
                result_pose, rel_result_pose, root_R, r9 = batch_hybrIK(joints, rest_pose, xy_priority=True, big_diff=False)
                return result_pose
            hybrik_res = torch.stack([inner((x - 20) / 10) for x in range(40)])  # 40,bs,24,3
            tmp = (hybrik_res - joints[None])[..., :2].norm(dim=-1).mean(dim=-1).argmin(dim=0)
            beta0 = (tmp - 20) / 10
            return beta0

        beta0_new = search_beta0()
        beta[:, 0] = beta0_new

        rest_pose = self.cal_rest_pose(beta)

        # j29+new_beta ->bone_dir_hybrik
        result_pose, rel_result_pose, root_R, r9 = batch_hybrIK(joints, rest_pose,xy_priority=False, big_diff=True)
        hybrik_jpe = (result_pose - joints).norm(dim=-1)[:, 1:].mean().item() * 1000
        # print(f'hybrik   jpe: {hybrik_jpe:.3f}mm')

        ortho_twist_global = ortho_twist
        if not optimize_pose:
            theta = self.compose_rot(beta, rel_result_pose, ortho_twist_global, root_R, r9, leaf_rot)
        else:
            # j29+beta + bone_dir_hybrik_res  -> beta_fitted+bone_dir_fitted
            fitted_bone_dir, fitted_rot_root, fitted_rot_chest = self.optimize_pose(
                joints, beta, rel_result_pose, root_R, r9)
            theta = self.compose_rot(
                beta, fitted_bone_dir, ortho_twist_global, fitted_rot_root, fitted_rot_chest, leaf_rot)

        transl = -1 * self.cal_joints(beta, theta)[:, 0] + result_pose[:, 0]
        return beta, theta, transl


if __name__ == '__main__':
    def test():
        """below is some testing code"""
        # import torch
        # from common.smpl_wrapper_new import SMPL
        # import torch

        bs = 4
        b = torch.randn(bs, 10)
        p = torch.randn(bs, 72)
        t = torch.randn(bs, 3)

        smpl = SMPL(gender='neutral', batch_size=bs, device='cpu', dtype=torch.float32, hybrik_joints=False)
        # default smpl transl has grad
        print(type(smpl(beta=b, pose=p, transl=t)[0].grad_fn))
        print(type(smpl(beta=b, pose=p)[0].grad_fn))

        joints, vertices, joints17 = smpl(beta=b, pose=p, transl=t)

        ot = smpl.cal_ortho_twist(b, p)

        joints_quick = smpl.cal_joints(beta=b, pose=p, transl=t, hybrik_joints=False)
        # the position of a joint is determined by parent's position, parent's rotation, and rest pose.
        # but hybrik defines 5 virtual joints, which work differently.
        print('quick results of hybrik-joints mpjpe(mm):',
              1000 * (joints - joints_quick)[:, 24:].norm(dim=-1).mean())
        print(1000 * (joints - joints_quick)[:, :24].norm(dim=-1).mean())

        tmp1 = smpl.rotation_relative2global(p)
        tmp2 = smpl.rotation_global2relative(tmp1)
        joints2, vertices2, joints172 = smpl(beta=b, pose=tmp2, transl=t)
        ot2 = smpl.cal_ortho_twist(b, tmp2)
        print((vertices - vertices2).abs().max())
        print((joints - joints2).abs().max())
        print((joints17 - joints172).abs().max())
        print((ot - ot2).abs().max())

        rest_pose1 = smpl.cal_rest_pose(b)
        rest_pose2, _, _ = smpl(beta=b, pose=p * 0)
        print((rest_pose1 - rest_pose2).abs().max())

        # from common.smpl_wrapper_old import SMPL as SMPL_OLD
        #
        # smpl_old = SMPL_OLD(batch_size=bs, device='cpu', dtype=torch.float32)
        # _joints, _vertices, _twist_dir, _bone_mid, _joints17 = smpl_old(
        #     betas=b, pose_axisang=p, transl=t)
        # _twist_dir_normed = F.normalize(_twist_dir, p=2, dim=-1)
        #
        # print((joints[:, :24] - _joints).abs().max())
        # print((_vertices - vertices).abs().max())
        # print((_joints17 - joints17).abs().max())
        # print((ot[:, 1:24, :] - _twist_dir_normed[:, 1:24]).abs().max())
        # bone_mid = (joints + joints[:, smpl.get_parents()]) / 2
        # print((_bone_mid - bone_mid[:, :24]).norm(dim=-1)[:, 1:].max())
        # print()

        # prepare smpl parameter and img
        seq_name = 'downtown_walkBridge_01'
        iModel = 0
        datasetDir = r'./data/3dpw'  # r'/data7/dataset/threedpw'  # r'/data2/haoxiaoyang/dataset/3dpw/'

        file = os.path.join(datasetDir, 'sequenceFiles', 'test', seq_name + '.pkl')
        seq = pkl.load(open(file, 'rb'), encoding='latin1')
        cam_intrinsics = seq['cam_intrinsics']

        frames, img_file = [], []
        for iFrame in list(range(0, 100, 10)):
            if seq['campose_valid'][iModel][iFrame]:
                frames.append(iFrame)
                img_path = os.path.join(datasetDir, 'imageFiles', seq['sequence'] + '/image_{:05d}.jpg'.format(iFrame))
                img_file.append(img_path)
        gender = 'female' if seq['genders'][iModel] == 'f' else 'male'

        cam_extrinsic = torch.tensor(seq['cam_poses'][frames])
        cam_quat = mat2quat(cam_extrinsic[:, :3, :3])
        cam_t = cam_extrinsic[:, :3, -1]
        beta_ = seq['betas'][iModel]
        beta = torch.zeros(10)
        beta[:min(len(beta_), 10)] = torch.tensor(beta_)[:min(len(beta_), 10)]
        theta_world = seq['poses'][iModel][frames]
        tran_world = seq['trans'][iModel][frames]

        # start to test
        smpl = SMPL(gender=gender, batch_size=len(frames), device='cpu', dtype=torch.float32, hybrik_joints=False)
        pose, beta, transl = smpl.pose_world2camera(cam_quat, cam_t, beta, theta_world, tran_world)

        joints, vertices, joints17 = smpl(beta=beta, pose=pose, transl=transl)

        # smpl_old = SMPL_OLD(gender=gender, batch_size=pose.shape[0], device='cpu', dtype=torch.float32)
        # _joints, _vertices, _twist_dir, _bone_mid, _joints17 = smpl_old(
        #    betas=beta, pose_axisang=theta_world, transl=tran_world, cam_quat=cam_quat.float(), cam_t=cam_t.float())
        #
        # print((_joints - joints[:, :24]).norm(dim=-1).max())

        # testing composing ortho twist and bone direction
        aim_pose = joints
        aim_ortho_twist = smpl.cal_ortho_twist(beta=beta, pose=pose)
        aim_ortho_twist += 0.2 * torch.randn(aim_ortho_twist.shape)
        root_orientation = smpl.rotation_relative2global(pose)[:, 0]
        chest_orientation = smpl.rotation_relative2global(pose)[:, 9]

        leaf_rotation = {}
        for idx in [15, 22, 23, 10, 11]:
            leaf_rotation[idx] = pose[:, idx]
        theta_res = smpl.compose_rot(beta, aim_pose - aim_pose[:, smpl.get_parents()],
                                     aim_ortho_twist,
                                     root_orientation, chest_orientation, leaf_rot=leaf_rotation)
        joints__, vertices__, joints17__ = smpl(beta=beta, pose=theta_res, transl=transl)
        print((joints - joints__).norm(dim=-1)[:, :24].max())
        if joints.shape[1] > 24:
            print(1000 * (joints - joints__).norm(dim=-1)[:, 24:].max(),
                  '(mm) the positions of virtual joint are not accurate')

        smpl.render(beta, theta_res, transl, None, cam_intrinsics, suffix='_random_twist.jpg')
        smpl.render(beta, pose, transl, None, cam_intrinsics)


    def render_beta_element():
        def write_obj(filename, vertices, faces):
            with open(filename, 'w') as f:
                for vertex in vertices:
                    f.write("v {} {} {}\n".format(vertex[0], vertex[1], vertex[2]))
                for face in faces:
                    f.write('f {} {} {}\n'.format(face[0] + 1, face[1] + 1, face[2] + 1))  # 1-indexed

        smpl = SMPL(gender='neutral', batch_size=3, device='cpu', dtype=torch.float32, hybrik_joints=False)
        bs = 11
        for beta_idx in range(10):
            beta = torch.zeros(bs, 10)
            for i in range(bs):
                beta[i][beta_idx] = 10 * (i / bs - 0.5)
            pose = torch.zeros(bs, 72)
            vertices = smpl(beta, pose)[1]
            faces = smpl.faces
            for i in range(bs):
                write_obj(f'./render_res/{beta_idx:02d}_{beta[i][beta_idx]:.3f}.obj', vertices[i], faces)

    def focal_length_matters():
        from common.geometry import batch_rodrigues

        smpl = SMPL(gender='neutral', batch_size=1)

        for f_coe in [1,4]:
            cam_intrinsics = np.array([[f_coe*800, 0.00, 540],
                                       [0.00, f_coe*800, 960],
                                       [0.00, 0.00, 1.]])
            transl = torch.tensor([[-0.8, 0.4, f_coe*2]])
            theta = batch_rodrigues(torch.zeros(24,3))[None]
            theta[0, :1] = batch_rodrigues(torch.tensor([0, 3.14159/2, 0])[None])@batch_rodrigues(torch.tensor([0, 0, 3.14159])[None])
            theta[0, :1] = batch_rodrigues(torch.tensor([0, -3.14159/180*25, 0])[None]) @ theta[0, :1]

            smpl.render(torch.zeros(1, 10), theta, transl, None, cam_intrinsics, suffix=f'{f_coe}.jpg')
            theta[0, :1] = batch_rodrigues(torch.tensor([3.14159 / 180 * 90, 0, 0])[None]) @ theta[0, :1]
            smpl.render(torch.zeros(1, 10), theta, transl, None, cam_intrinsics, suffix=f'{f_coe}_head.jpg')


    smpl = SMPL(gender='neutral', batch_size=1)
    smpl.visualize_vert(0)
    # render_beta_element()
    cam_intrinsics = np.array([[1.96185286e+03, 0.00000000e+00, 5.40000000e+02],
                               [0.00000000e+00, 1.96923077e+03, 9.60000000e+02],
                               [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    smpl.render(torch.zeros(1, 10), torch.zeros(1, 72), torch.zeros(1, 3), None, cam_intrinsics)
    test()
    print()
