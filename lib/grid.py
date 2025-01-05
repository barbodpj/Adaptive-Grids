import os
import time
import functools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.cpp_extension import load

parent_dir = os.path.dirname(os.path.abspath(__file__))
render_utils_cuda = load(
    name='render_utils_cuda',
    sources=[
        os.path.join(parent_dir, path)
        for path in ['cuda/render_utils.cpp', 'cuda/render_utils_kernel.cu']],
    verbose=True)

total_variation_cuda = load(
    name='total_variation_cuda',
    sources=[
        os.path.join(parent_dir, path)
        for path in ['cuda/total_variation.cpp', 'cuda/total_variation_kernel.cu']],
    verbose=True)


def create_grid(type, stage='coarse', **kwargs):
    if type == 'DenseGrid':
        return DenseGrid(stage=stage, **kwargs)
    elif type == 'TensoRFGrid':
        return TensoRFGrid(stage=stage, **kwargs)
    else:
        raise NotImplementedError


''' Dense 3D grid
'''


class DenseGrid(nn.Module):
    def __init__(self, channels, world_size, xyz_min, xyz_max, stage='coarse', **kwargs):
        super(DenseGrid, self).__init__()
        self.channels = channels
        self.world_size = world_size
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
        self.grid = nn.Parameter(torch.zeros([1, channels, *world_size]))
        if isinstance(self.world_size, list):  # Check if x is a list
            self.world_size = torch.tensor(self.world_size)  # Convert
        self.register_buffer('voxel_subsampling', torch.tensor(False))
        self.block_size = torch.tensor([16,16, 16])
        self.block_num = torch.floor(self.world_size/self.block_size).to(dtype=torch.long)
        self.important_num = 4**3 #4**3
        self.important_num_third_root = int(round(self.important_num**(1/3)))
        if stage == 'finea':
            # self.new_grid = (64,64,64)
            self.sub_sample = 2
            # self.important_voxels = nn.Parameter(torch.zeros([self.important_num, channels, *self.new_grid]),
            #                                      requires_grad=True)
            important_grid_size = []
            for i in range(3):
                important_grid_size.append(int(self.sub_sample*self.block_size[i]+1)*self.important_num_third_root)
            self.important_grid = nn.Parameter(torch.zeros([1, channels, *important_grid_size]))


            # self.register_buffer('importance_mask', torch.ones(*self.block_num.tolist()).to(torch.bool) * -1)
            self.register_buffer('importance_mask', torch.ones(*self.block_num.tolist(),3).to(torch.bool) * -1)

    def initialize_importance_subsampling(self, indices):
        print("Importance Training Started - number of important blocks: ", self.important_num)
        self.importance_mask = torch.ones(*self.block_num.tolist(),3,dtype=torch.long)* -1

        # interval [20,self.world_size[i]-20]
        self.voxel_subsampling = torch.tensor(True)
        indices[0] = indices[0].to(dtype=torch.long)
        indices[1] = indices[1].to(dtype=torch.long)
        indices[2] = indices[2].to(dtype=torch.long)
        a,b,c = torch.meshgrid(torch.arange(end=self.important_num_third_root), torch.arange(end=self.important_num_third_root), torch.arange(end=self.important_num_third_root))
        self.importance_mask[indices[0], indices[1], indices[2],0] = a.flatten()
        self.importance_mask[indices[0], indices[1], indices[2],1] = b.flatten()
        self.importance_mask[indices[0], indices[1], indices[2],2] = c.flatten()

        # self.importance_mask[indices[0], indices[1], indices[2]] = torch.arange(end=self.important_num)


        # indices[0] = indices[0] * self.block_size[0]
        # indices[1] = indices[1] * self.block_size[1]
        # indices[2] = indices[2] * self.block_size[2]

        # with torch.no_grad():
        #     for i in range(len(indices[0])):
        #         self.important_voxels[i] = torch.randn(self.important_voxels[i].shape)

        #         start = [0, 0, 0]
        #         end = [0, 0, 0]
        #         for j in range(3):
        #             start[j] = indices[j][i]
        #             end[j] = start[j] + self.block_size[j]
        #             if end[j] > self.world_size[j]:
        #                 end[j] = self.world_size[j]
        #                 start[j] = end[j] - self.block_size[j]
        #
        #
        #         g = self.grid[:, :, start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        #         interpolated = F.interpolate(g, size=self.new_grid, mode='trilinear', align_corners=True)
        #         self.important_voxels[i] = interpolated[0]

    def forward(self, xyz):
        '''
        xyz: global coordinates to query
        '''
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1,1,1,-1,3)
        ind_norm = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip((-1,)) * 2 - 1
        out = F.grid_sample(self.grid, ind_norm, mode='bilinear', align_corners=True)
        if self.voxel_subsampling:
            indices = ((xyz - self.xyz_min) / (self.xyz_max - self.xyz_min))[0, 0, 0]

            block_indices = (torch.floor(indices * (self.world_size-1)/self.block_size)).to(dtype=torch.long)
            block_indices[torch.min(indices, dim=-1)[0] < 0] = 0

            m1 = block_indices[:, 0] >= self.block_num[0]
            m2 = block_indices[:, 1] >= self.block_num[1]
            m3 = block_indices[:, 2] >= self.block_num[2]

            block_indices[m1] = 0
            block_indices[m2] = 0
            block_indices[m3] = 0

            locations = self.importance_mask[block_indices[:, 0], block_indices[:, 1], block_indices[:, 2]]

            locations[torch.min(indices, dim=-1)[0] < 0] = torch.tensor([-1000, -1000, -1000])
            locations[m1] = torch.tensor([-1000, -1000, -1000])
            locations[m2] = torch.tensor([-1000, -1000, -1000])
            locations[m3] = torch.tensor([-1000, -1000, -1000])

            new_indices = (indices * (self.world_size-1) - block_indices*self.block_size)
            # l1 = block_indices[:,0] == self.block_num[0]-1
            # l2 = block_indices[:,1] == self.block_num[1]-1
            # l3 = block_indices[:,2] == self.block_num[2]-1
            # new_indices[l1,0] = indices[l1,0] * (self.world_size[0]-1) - (self.world_size[0]-self.block_size[0])
            # new_indices[l2,1] = indices[l2,1] * (self.world_size[1]-1) - (self.world_size[1]-self.block_size[1])
            # new_indices[l3,2] = indices[l3,2] * (self.world_size[2]-1) - (self.world_size[2]-self.block_size[2])

            locations[torch.min(new_indices,-1) [0] < 0] = torch.tensor([-1000, -1000, -1000])
            locations[new_indices[:,0] > self.block_size[0] - 1] = torch.tensor([-1000, -1000, -1000])
            locations[new_indices[:,1] > self.block_size[1] - 1] = torch.tensor([-1000, -1000, -1000])
            locations[new_indices[:,2] > self.block_size[2] - 1] = torch.tensor([-1000, -1000, -1000])
            # if torch.sum(locations>-1) > 0:
            #     value, _ = torch.mode(locations[locations>-1].flatten())
            #     count = torch.sum(locations==value)
            # else:
            #     count=0
            # indices_grouped = torch.ones(self.important_num,1,1,count,3)*-2
            # indices_lengths = [-1]*self.important_num
            # new_indices = new_indices/(self.block_size-1)
            new_indices = (new_indices * (self.block_size-2))/(self.block_size-1)
            new_indices = (new_indices/(self.block_size-1) + locations)/self.important_num_third_root
            new_indices = (2 * new_indices - 1).flip((-1,))

            # f = torch.unique(locations[locations >= 0])

            # for j in range(f.shape[0]):
            #     q = new_indices[locations==f[j]]
            #     indices_lengths[f[j]] = q.shape[0]
            #     indices_grouped[f[j],0,0,0:q.shape[0],:] = q
            # s = F.grid_sample(self.important_voxels,indices_grouped,mode='bilinear', align_corners=True)

            # for j in range(f.shape[0]):
            #     out[0,...,locations==f[j]] += s[f[j],:,:,:,:indices_lengths[f[j]]]
            # for j in range(f.shape[0]):
            #     r = F.grid_sample(self.important_voxels[f[j]][None, ...],
            #                       new_indices[locations == f[j]].reshape(1, 1, 1, -1, 3), mode='bilinear',
            #                       align_corners=True)
            #
            #     out[..., locations == f[j]] = out[..., locations == f[j]]  +  r
            out = out + F.grid_sample(self.important_grid, new_indices.reshape(1,1,1,-1,3), mode='bilinear', align_corners=True)
        out = out.reshape(self.channels, -1).T.reshape(*shape, self.channels)
        if self.channels == 1:
            out = out.squeeze(-1)
        return out

    def scale_volume_grid(self, new_world_size):
        self.world_size = new_world_size
        if self.channels == 0:
            self.grid = nn.Parameter(torch.zeros([1, self.channels, *new_world_size]))
        else:
            self.grid = nn.Parameter(
                F.interpolate(self.grid.data, size=tuple(new_world_size), mode='trilinear', align_corners=True))

    def total_variation_add_grad(self, wx, wy, wz, dense_mode):
        '''Add gradients by total variation loss in-place'''
        total_variation_cuda.total_variation_add_grad(
            self.grid, self.grid.grad, wx, wy, wz, dense_mode)

    def get_dense_grid(self):
        return self.grid

    @torch.no_grad()
    def __isub__(self, val):
        print("isub")
        self.grid.data -= val
        return self

    def extra_repr(self):
        return f'channels={self.channels}, world_size={self.world_size.tolist()}'


''' Vector-Matrix decomposited grid
See TensoRF: Tensorial Radiance Fields (https://arxiv.org/abs/2203.09517)
'''


class TensoRFGrid(nn.Module):
    def __init__(self, channels, world_size, xyz_min, xyz_max, config, stage="coarse"):
        super(TensoRFGrid, self).__init__()
        self.channels = channels
        self.world_size = world_size
        self.config = config
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
        X, Y, Z = world_size
        R = config['n_comp']
        Rxy = config.get('n_comp_xy', R)
        self.xy_plane = nn.Parameter(torch.randn([1, Rxy, X, Y]) * 0.1)
        self.xz_plane = nn.Parameter(torch.randn([1, R, X, Z]) * 0.1)
        self.yz_plane = nn.Parameter(torch.randn([1, R, Y, Z]) * 0.1)
        self.x_vec = nn.Parameter(torch.randn([1, R, X, 1]) * 0.1)
        self.y_vec = nn.Parameter(torch.randn([1, R, Y, 1]) * 0.1)
        self.z_vec = nn.Parameter(torch.randn([1, Rxy, Z, 1]) * 0.1)


        if self.channels > 1:
            self.f_vec = nn.Parameter(torch.ones([R + R + Rxy, channels]))
            nn.init.kaiming_uniform_(self.f_vec, a=np.sqrt(5))

        self.register_buffer('voxel_subsampling', torch.tensor(False))
        self.block_size = torch.tensor([64, 64, 64])
        self.block_num = torch.floor(self.world_size / self.block_size).to(dtype=torch.long) + 1
        self.important_num = 10
        if stage == 'finea':
            self.additional_comps = 6
            self.important_xy_plane = nn.Parameter(torch.randn([self.important_num, self.additional_comps, self.block_size[0],  self.block_size[1]]) * 0.1,requires_grad=True)
            self.important_xz_plane = nn.Parameter(torch.randn([self.important_num, self.additional_comps,  self.block_size[0],  self.block_size[2]]) * 0.1, requires_grad=True)
            self.important_yz_plane = nn.Parameter(torch.randn([self.important_num, self.additional_comps,  self.block_size[1],  self.block_size[2]]) * 0.1, requires_grad=True)
            self.important_x_vec = nn.Parameter(torch.randn([self.important_num, self.additional_comps,  self.block_size[0], 1]) * 0.1, requires_grad=True)
            self.important_y_vec = nn.Parameter(torch.randn([self.important_num, self.additional_comps,  self.block_size[1], 1]) * 0.1, requires_grad=True)
            self.important_z_vec = nn.Parameter(torch.randn([self.important_num, self.additional_comps,  self.block_size[2], 1]) * 0.1,  requires_grad=True)
            if self.channels > 1:
                self.important_f_vec = nn.Parameter(torch.ones([self.important_num, 3*self.additional_comps, channels]))
                for i in range(self.important_num):
                    nn.init.kaiming_uniform_(self.important_f_vec[i], a=np.sqrt(5))

            self.register_buffer('importance_mask', torch.ones(*self.block_num.tolist()).to(torch.bool) * -1)

    def initialize_importance_subsampling(self, indices):
        self.block_num = torch.floor(self.world_size / self.block_size).to(dtype=torch.long) + 1

        print("Importance Training Started - number of important blocks: ", self.important_num)
        self.importance_mask = torch.ones(*self.block_num.tolist(), dtype=torch.long) * -1

        # interval [20,self.world_size[i]-20]
        self.voxel_subsampling = torch.tensor(True)
        indices[0] = indices[0].to(dtype=torch.long)
        indices[1] = indices[1].to(dtype=torch.long)
        indices[2] = indices[2].to(dtype=torch.long)
        self.importance_mask[indices[0], indices[1], indices[2]] = torch.arange(end=self.important_num)

    def forward(self, xyz):
        '''
        xyz: global coordinates to query
        '''
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1, 1, -1, 3)
        ind_norm = (xyz - self.xyz_min) / (self.xyz_max - self.xyz_min) * 2 - 1
        ind_norm = torch.cat([ind_norm, torch.zeros_like(ind_norm[..., [0]])], dim=-1)
        if self.channels > 1:
            out = self.compute_tensorf_feat(ind_norm)
            out = out.reshape(*shape, self.channels)
        else:
            out = self.compute_tensorf_val(ind_norm)
            out = out.reshape(*shape)
        return out

    def scale_volume_grid(self, new_world_size):
        self.world_size = new_world_size

        if self.channels == 0:
            return
        X, Y, Z = new_world_size
        self.xy_plane = nn.Parameter(F.interpolate(self.xy_plane.data, size=[X,Y], mode='bilinear', align_corners=True))
        self.xz_plane = nn.Parameter(F.interpolate(self.xz_plane.data, size=[X,Z], mode='bilinear', align_corners=True))
        self.yz_plane = nn.Parameter(F.interpolate(self.yz_plane.data, size=[Y,Z], mode='bilinear', align_corners=True))
        self.x_vec = nn.Parameter(F.interpolate(self.x_vec.data, size=[X,1], mode='bilinear', align_corners=True))
        self.y_vec = nn.Parameter(F.interpolate(self.y_vec.data, size=[Y,1], mode='bilinear', align_corners=True))
        self.z_vec = nn.Parameter(F.interpolate(self.z_vec.data, size=[Z,1], mode='bilinear', align_corners=True))

    def total_variation_add_grad(self, wx, wy, wz, dense_mode):
        '''Add gradients by total variation loss in-place'''
        loss = wx * F.smooth_l1_loss(self.xy_plane[:,:,1:], self.xy_plane[:,:,:-1], reduction='sum') +\
               wy * F.smooth_l1_loss(self.xy_plane[:,:,:,1:], self.xy_plane[:,:,:,:-1], reduction='sum') +\
               wx * F.smooth_l1_loss(self.xz_plane[:,:,1:], self.xz_plane[:,:,:-1], reduction='sum') +\
               wz * F.smooth_l1_loss(self.xz_plane[:,:,:,1:], self.xz_plane[:,:,:,:-1], reduction='sum') +\
               wy * F.smooth_l1_loss(self.yz_plane[:,:,1:], self.yz_plane[:,:,:-1], reduction='sum') +\
               wz * F.smooth_l1_loss(self.yz_plane[:,:,:,1:], self.yz_plane[:,:,:,:-1], reduction='sum') +\
               wx * F.smooth_l1_loss(self.x_vec[:,:,1:], self.x_vec[:,:,:-1], reduction='sum') +\
               wy * F.smooth_l1_loss(self.y_vec[:,:,1:], self.y_vec[:,:,:-1], reduction='sum') +\
               wz * F.smooth_l1_loss(self.z_vec[:,:,1:], self.z_vec[:,:,:-1], reduction='sum')
        loss /= 6
        loss.backward()

    def get_dense_grid(self):
        if self.channels > 1:
            feat = torch.cat([
                torch.einsum('rxy,rz->rxyz', self.xy_plane[0], self.z_vec[0, :, :, 0]),
                torch.einsum('rxz,ry->rxyz', self.xz_plane[0], self.y_vec[0, :, :, 0]),
                torch.einsum('ryz,rx->rxyz', self.yz_plane[0], self.x_vec[0, :, :, 0]),
            ])
            grid = torch.einsum('rxyz,rc->cxyz', feat, self.f_vec)[None]
        else:
            grid = torch.einsum('rxy,rz->xyz', self.xy_plane[0], self.z_vec[0, :, :, 0]) + \
                   torch.einsum('rxz,ry->xyz', self.xz_plane[0], self.y_vec[0, :, :, 0]) + \
                   torch.einsum('ryz,rx->xyz', self.yz_plane[0], self.x_vec[0, :, :, 0])
            grid = grid[None, None]
        return grid

    def extra_repr(self):
        return f'channels={self.channels}, world_size={self.world_size.tolist()}, n_comp={self.config["n_comp"]}'


    def compute_tensorf_feat(self, ind_norm):
        # Interp feature (feat shape: [n_pts, n_comp])
        xy_feat = F.grid_sample(self.xy_plane, ind_norm[:, :, :, [1, 0]], mode='bilinear', align_corners=True).flatten(0, 2).T
        xz_feat = F.grid_sample(self.xz_plane, ind_norm[:, :, :, [2, 0]], mode='bilinear', align_corners=True).flatten(0, 2).T
        yz_feat = F.grid_sample(self.yz_plane, ind_norm[:, :, :, [2, 1]], mode='bilinear', align_corners=True).flatten(0, 2).T
        x_feat = F.grid_sample(self.x_vec, ind_norm[:, :, :, [3, 0]], mode='bilinear', align_corners=True).flatten(0, 2).T
        y_feat = F.grid_sample(self.y_vec, ind_norm[:, :, :, [3, 1]], mode='bilinear', align_corners=True).flatten(0, 2).T
        z_feat = F.grid_sample(self.z_vec, ind_norm[:, :, :, [3, 2]], mode='bilinear', align_corners=True).flatten(0, 2).T
        # Aggregate components
        feat = torch.cat([
            xy_feat * z_feat,
            xz_feat * y_feat,
            yz_feat * x_feat,
        ], dim=-1)
        feat = torch.mm(feat, self.f_vec)

        if self.voxel_subsampling:
            indices = ((ind_norm+1)/2)[0,0,:,0:3]

            block_indices = (torch.floor(indices * (self.world_size-1)/self.block_size)).to(dtype=torch.long)
            block_indices[torch.min(indices, dim=-1)[0] < 0] = 0

            locations = self.importance_mask[block_indices[:, 0], block_indices[:, 1], block_indices[:, 2]]

            locations[torch.min(indices, dim=-1)[0] < 0] = -1


            new_indices = (indices * (self.world_size-1) - block_indices*self.block_size)
            l1 = block_indices[:,0] == self.block_num[0]-1
            l2 = block_indices[:,1] == self.block_num[1]-1
            l3 = block_indices[:,2] == self.block_num[2]-1
            new_indices[l1,0] = indices[l1,0] * (self.world_size[0]-1) - (self.world_size[0]-self.block_size[0])
            new_indices[l2,1] = indices[l2,1] * (self.world_size[1]-1) - (self.world_size[1]-self.block_size[1])
            new_indices[l3,2] = indices[l3,2] * (self.world_size[2]-1) - (self.world_size[2]-self.block_size[2])

            locations[torch.min(new_indices,-1) [0] < 1 ] = -1
            locations[new_indices[:,0] > self.block_size[0] - 1] = -1
            locations[new_indices[:,1] > self.block_size[1] - 1] = -1
            locations[new_indices[:,2] > self.block_size[2] - 1] = -1

            new_indices = new_indices/(self.block_size-1)
            new_indices = (2 * new_indices - 1)
            new_indices = new_indices[None, None, ...]
            new_indices = torch.cat([new_indices, torch.zeros_like(new_indices[..., [0]])], dim=-1)

            f = torch.unique(locations[locations >= 0])
            for j in range(f.shape[0]):
                p = new_indices[:, :, locations==f[j]]
                new_xy_feat = F.grid_sample(self.important_xy_plane[f[j]:f[j]+1], p[:,:,:, [1, 0]], mode='bilinear',
                                        align_corners=True).flatten(0, 2).T
                new_xz_feat = F.grid_sample(self.important_xz_plane[f[j]:f[j]+1], p[:,:,:, [2, 0]], mode='bilinear',
                                        align_corners=True).flatten(0, 2).T
                new_yz_feat = F.grid_sample(self.important_yz_plane[f[j]:f[j]+1], p[:,:,:, [2, 1]], mode='bilinear',
                                        align_corners=True).flatten(0, 2).T
                new_x_feat = F.grid_sample(self.important_x_vec[f[j]:f[j]+1], p[:,:,:, [3, 0]], mode='bilinear',
                                       align_corners=True).flatten(0, 2).T
                new_y_feat = F.grid_sample(self.important_y_vec[f[j]:f[j]+1], p[:,:,:, [3, 1]], mode='bilinear',
                                       align_corners=True).flatten(0, 2).T
                new_z_feat = F.grid_sample(self.important_z_vec[f[j]:f[j]+1], p[:,:,:, [3, 2]], mode='bilinear',
                                       align_corners=True).flatten(0, 2).T
                # Aggregate components
                new_feat = torch.cat([
                    new_xy_feat * new_z_feat,
                    new_xz_feat * new_y_feat,
                    new_yz_feat * new_x_feat,
                ], dim=-1)
                new_feat = torch.mm(new_feat, self.important_f_vec[f[j]])

                # new_feat = torch.ones_like(new_feat) * 100


                feat[locations == f[j]] = feat[locations == f[j]] + new_feat




        return feat


    def compute_tensorf_val(self,ind_norm):
        # Interp feature (feat shape: [n_pts, n_comp])
        xy_feat = F.grid_sample(self.xy_plane, ind_norm[:, :, :, [1, 0]], mode='bilinear', align_corners=True).flatten(0, 2).T
        xz_feat = F.grid_sample(self.xz_plane, ind_norm[:, :, :, [2, 0]], mode='bilinear', align_corners=True).flatten(0, 2).T
        yz_feat = F.grid_sample(self.yz_plane, ind_norm[:, :, :, [2, 1]], mode='bilinear', align_corners=True).flatten(0, 2).T
        x_feat = F.grid_sample(self.x_vec, ind_norm[:, :, :, [3, 0]], mode='bilinear', align_corners=True).flatten(0, 2).T
        y_feat = F.grid_sample(self.y_vec, ind_norm[:, :, :, [3, 1]], mode='bilinear', align_corners=True).flatten(0, 2).T
        z_feat = F.grid_sample(self.z_vec, ind_norm[:, :, :, [3, 2]], mode='bilinear', align_corners=True).flatten(0, 2).T
        # Aggregate components
        feat = (xy_feat * z_feat).sum(-1) + (xz_feat * y_feat).sum(-1) + (yz_feat * x_feat).sum(-1)

        if self.voxel_subsampling:
            indices = ((ind_norm+1)/2)[0,0,:,0:3]

            block_indices = (torch.floor(indices * (self.world_size-1)/self.block_size)).to(dtype=torch.long)
            block_indices[torch.min(indices, dim=-1)[0] < 0] = 0

            locations = self.importance_mask[block_indices[:, 0], block_indices[:, 1], block_indices[:, 2]]

            locations[torch.min(indices, dim=-1)[0] < 0] = -1


            new_indices = (indices * (self.world_size-1) - block_indices*self.block_size)
            l1 = block_indices[:,0] == self.block_num[0]-1
            l2 = block_indices[:,1] == self.block_num[1]-1
            l3 = block_indices[:,2] == self.block_num[2]-1
            new_indices[l1,0] = indices[l1,0] * (self.world_size[0]-1) - (self.world_size[0]-self.block_size[0])
            new_indices[l2,1] = indices[l2,1] * (self.world_size[1]-1) - (self.world_size[1]-self.block_size[1])
            new_indices[l3,2] = indices[l3,2] * (self.world_size[2]-1) - (self.world_size[2]-self.block_size[2])

            locations[torch.min(new_indices,-1) [0] < 1 ] = -1
            locations[new_indices[:,0] > self.block_size[0] - 1] = -1
            locations[new_indices[:,1] > self.block_size[1] - 1] = -1
            locations[new_indices[:,2] > self.block_size[2] - 1] = -1

            new_indices = new_indices/(self.block_size-1)
            new_indices = (2 * new_indices - 1)
            new_indices = new_indices[None, None, ...]
            new_indices = torch.cat([new_indices, torch.zeros_like(new_indices[..., [0]])], dim=-1)

            f = torch.unique(locations[locations >= 0])
            for j in range(f.shape[0]):
                p = new_indices[:, :, locations==f[j]]
                new_xy_feat = F.grid_sample(self.important_xy_plane[f[j]:f[j]+1], p[:,:,:, [1, 0]], mode='bilinear',
                                        align_corners=True).flatten(0, 2).T
                new_xz_feat = F.grid_sample(self.important_xz_plane[f[j]:f[j]+1], p[:,:,:, [2, 0]], mode='bilinear',
                                        align_corners=True).flatten(0, 2).T
                new_yz_feat = F.grid_sample(self.important_yz_plane[f[j]:f[j]+1], p[:,:,:, [2, 1]], mode='bilinear',
                                        align_corners=True).flatten(0, 2).T
                new_x_feat = F.grid_sample(self.important_x_vec[f[j]:f[j]+1], p[:,:,:, [3, 0]], mode='bilinear',
                                       align_corners=True).flatten(0, 2).T
                new_y_feat = F.grid_sample(self.important_y_vec[f[j]:f[j]+1], p[:,:,:, [3, 1]], mode='bilinear',
                                       align_corners=True).flatten(0, 2).T
                new_z_feat = F.grid_sample(self.important_z_vec[f[j]:f[j]+1], p[:,:,:, [3, 2]], mode='bilinear',
                                       align_corners=True).flatten(0, 2).T
                # Aggregate components
                new_feat = (new_xy_feat * new_z_feat).sum(-1) + (new_xz_feat * new_y_feat).sum(-1) + (new_yz_feat * new_x_feat).sum(-1)

                # new_feat = torch.ones_like(new_feat)  * 100

                feat[locations == f[j]] = feat[locations == f[j]] + new_feat
        return feat


''' Mask grid
It supports query for the known free space and unknown space.
'''


class MaskGrid(nn.Module):
    def __init__(self, path=None, mask_cache_thres=None, mask=None, xyz_min=None, xyz_max=None):
        super(MaskGrid, self).__init__()
        if path is not None:
            st = torch.load(path)
            self.mask_cache_thres = mask_cache_thres
            density = F.max_pool3d(st['model_state_dict']['density.grid'], kernel_size=3, padding=1, stride=1)
            alpha = 1 - torch.exp(
                -F.softplus(density + st['model_state_dict']['act_shift']) * st['model_kwargs']['voxel_size_ratio'])
            mask = (alpha >= self.mask_cache_thres).squeeze(0).squeeze(0)
            xyz_min = torch.Tensor(st['model_kwargs']['xyz_min'])
            xyz_max = torch.Tensor(st['model_kwargs']['xyz_max'])
        else:
            mask = mask.bool()
            xyz_min = torch.Tensor(xyz_min)
            xyz_max = torch.Tensor(xyz_max)

        self.register_buffer('mask', mask)
        xyz_len = xyz_max - xyz_min
        self.register_buffer('xyz2ijk_scale', (torch.Tensor(list(mask.shape)) - 1) / xyz_len)
        self.register_buffer('xyz2ijk_shift', -xyz_min * self.xyz2ijk_scale)

    @torch.no_grad()
    def forward(self, xyz):
        '''Skip know freespace
        @xyz:   [..., 3] the xyz in global coordinate.
        '''
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(-1, 3)
        mask = render_utils_cuda.maskcache_lookup(self.mask, xyz, self.xyz2ijk_scale, self.xyz2ijk_shift)
        mask = mask.reshape(shape)
        return mask

    def extra_repr(self):
        return f'mask.shape=list(self.mask.shape)'
