import os

import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
import yaml
from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F

from motionblur.motionblur import Kernel

from .jpeg_torch import jpeg_decode, jpeg_encode


class H_functions:
    """
    A class replacing the SVD of a matrix H, perhaps efficiently.
    All input vectors are of shape (Batch, ...).
    All output vectors are of shape (Batch, DataDimension).
    """

    def V(self, vec):
        """
        Multiplies the input vector by V
        """
        raise NotImplementedError()

    def Vt(self, vec):
        """
        Multiplies the input vector by V transposed
        """
        raise NotImplementedError()

    def U(self, vec):
        """
        Multiplies the input vector by U
        """
        raise NotImplementedError()

    def Ut(self, vec):
        """
        Multiplies the input vector by U transposed
        """
        raise NotImplementedError()

    def singulars(self):
        """
        Returns a vector containing the singular values. The shape of the vector should be the same as the smaller dimension (like U)
        """
        raise NotImplementedError()

    def add_zeros(self, vec):
        """
        Adds trailing zeros to turn a vector from the small dimension (U) to the big dimension (V)
        """
        raise NotImplementedError()

    def H(self, vec):
        """
        Multiplies the input vector by H
        """
        temp = self.Vt(vec)
        singulars = self.singulars()
        return self.U(singulars * temp[:, : singulars.shape[0]])

    def Ht(self, vec):
        """
        Multiplies the input vector by H transposed
        """
        temp = self.Ut(vec)
        singulars = self.singulars()
        return self.V(self.add_zeros(singulars * temp[:, : singulars.shape[0]]))

    def H_pinv(self, vec):
        """
        Multiplies the input vector by the pseudo inverse of H
        """
        temp = self.Ut(vec)  # (b, m) - > (b, m)
        singulars = self.singulars()  # (mxm, )
        # temp[:, :singulars.shape[0]] = temp[:, :singulars.shape[0]] / singulars
        nonzero_idx = singulars.nonzero().flatten()
        temp[:, nonzero_idx] = temp[:, nonzero_idx] / singulars[nonzero_idx]
        return self.V(self.add_zeros(temp))

    def H_pinvt(self, vec):
        temp = self.Vt(vec)
        singulars = self.singulars()
        nonzero_idx = singulars.nonzero().flatten()
        temp[:, nonzero_idx] = temp[:, nonzero_idx] / singulars[nonzero_idx]
        return self.U(temp[:, : singulars.shape[0]])


# a memory inefficient implementation for any general degradation H
class GeneralH(H_functions):
    def mat_by_vec(self, M, v):
        vshape = v.shape[1]
        if len(v.shape) > 2:
            vshape = vshape * v.shape[2]
        if len(v.shape) > 3:
            vshape = vshape * v.shape[3]
        return torch.matmul(M, v.view(v.shape[0], vshape, 1)).view(
            v.shape[0], M.shape[0]
        )

    def __init__(self, H):
        self._U, self._singulars, self._V = torch.svd(H, some=False)
        self._Vt = self._V.transpose(0, 1)
        self._Ut = self._U.transpose(0, 1)

        ZERO = 1e-3
        self._singulars[self._singulars < ZERO] = 0
        print(len([x.item() for x in self._singulars if x == 0]))

    def V(self, vec):
        return self.mat_by_vec(self._V, vec.clone())

    def Vt(self, vec):
        return self.mat_by_vec(self._Vt, vec.clone())

    def U(self, vec):
        return self.mat_by_vec(self._U, vec.clone())

    def Ut(self, vec):
        return self.mat_by_vec(self._Ut, vec.clone())

    def singulars(self):
        return self._singulars

    def add_zeros(self, vec):
        out = torch.zeros(vec.shape[0], self._V.shape[0], device=vec.device)
        out[:, : self._U.shape[0]] = vec.clone().reshape(vec.shape[0], -1)
        return out


class SparseH(GeneralH):
    def __init__(self, U, S, V):
        self._U = U.to_sparse()
        self._V = V.to_sparse()
        self._singulars = S
        ZERO = 1e-6
        self._singulars[self._singulars < ZERO] = 0

    def mat_by_vec(self, M, v):
        vshape = v.shape[1]
        if len(v.shape) > 2:
            vshape = vshape * v.shape[2]
        if len(v.shape) > 3:
            vshape = vshape * v.shape[3]

        return torch.sparse.mm(M, v.view(v.shape[0], vshape, 1)).view(
            v.shape[0], M.shape[0]
        )


class Inpainting2(H_functions):
    def __init__(self, channels, img_dim, dense_masks, device):
        self.channels = channels
        self.img_dim = img_dim
        self.dense_masks = dense_masks
        self.device = device

    def set_indices(self, idx):
        idx = torch.remainder(idx, self.dense_masks.size(0))
        _singulars = self.dense_masks[idx].clone().to(self.device)
        self._singulars = _singulars.reshape(_singulars.size(0), -1)

    def V(self, vec):
        return vec.reshape(vec.size(0), -1)

    def Vt(self, vec):
        return vec.reshape(vec.size(0), -1)

    def U(self, vec):
        return vec.reshape(vec.size(0), -1)

    def Ut(self, vec):
        return vec.reshape(vec.size(0), -1)

    def add_zeros(self, vec):
        return vec

    def singulars(self):
        return self._singulars.float()

    def H(self, vec):
        return vec.reshape(*self.singulars().size()) * self.singulars()

    def H_pinv(self, vec):
        return vec.reshape(*self.singulars().size()) * self.singulars()

    def H_pinvt(self, vec):
        return vec.reshape(*self.singulars().size()) * self.singulars()


# Inpainting
class Inpainting(H_functions):
    def __init__(self, channels, img_dim, dense_masks, device):
        self.channels = channels
        self.img_dim = img_dim
        self.dense_masks = dense_masks
        n = dense_masks.size(0)

        # xxx = [torch.nonzero(dense_masks[i] == 0).long() for i in range(n)]
        # print('xxx', len(xxx))
        # for i in range(len(xxx)):
        #     print('i', i)
        #     print(xxx[i].shape)

        # import pdb; pdb.set_trace()

        # self.missing_masks = torch.cat([torch.nonzero(dense_masks[i] == 0).long() for i in range(n)], dim=0).T
        # self.keep_masks = torch.cat([torch.nonzero(dense_masks[i] != 0).long() for i in range(n)], dim=0).T

        self.missing_masks = torch.cat(
            [torch.nonzero(dense_masks[i] == 0).long() for i in range(n)], dim=1
        ).T
        self.keep_masks = torch.cat(
            [torch.nonzero(dense_masks[i] != 0).long() for i in range(n)], dim=1
        ).T

        self.device = device

    def set_indices(self, idx):
        channels = self.channels
        img_dim = self.img_dim
        device = self.device
        idx = torch.remainder(idx, self.missing_masks.size(0))
        missing_masks = self.missing_masks[idx].clone().to(device)
        # ###
        # l = missing_masks.size(1)im
        # missing_masks = torch.div(missing_masks[:l//3], 4, rounding_mode="floor").unique()
        # missing_masks = torch.cat([missing_masks, missing_masks + channels * img_dim * img_dim // 4, missing_masks + channels * img_dim * img_dim // 4 * 2])
        # missing_masks = missing_masks.view(1, -1)
        # ###
        missing_masks = missing_masks
        print("missing_masks.shape", missing_masks.shape)
        self.missing_indices = missing_indices = (
            missing_masks  # self.missing_masks[idx].clone().to(device)
        )
        self._singulars = torch.ones(
            channels * img_dim**2 - missing_indices.shape[1]
        ).to(device)
        kept_masks = self.keep_masks[idx].clone().to(device)
        # l = kept_masks.size(1)
        # kept_masks = torch.div(kept_masks[:l//3], 4, rounding_mode="floor").unique()
        # kept_masks = torch.cat([kept_masks, kept_masks + channels * img_dim * img_dim // 4, kept_masks + channels * img_dim * img_dim // 4 * 2])
        # kept_masks = kept_masks.view(1, -1)
        self.kept_indices = kept_masks  # self.keep_masks[idx].clone().to(device)

    def V(self, vec):
        temp = vec.clone().reshape(vec.shape[0], -1)
        out = torch.zeros_like(temp)
        assert vec.size(0) == self.kept_indices.size(0)
        n = vec.size(0)
        for i in range(n):
            out[i, self.kept_indices[i]] = temp[i, : self.kept_indices[i].shape[0]]
            out[i, self.missing_indices[i]] = temp[i, self.kept_indices[i].shape[0] :]
        return out  # .reshape(vec.shape[0], -1, self.channels).permute(0, 2, 1).reshape(vec.shape[0], -1)

    def Vt(self, vec):
        temp = vec.clone().reshape(vec.shape[0], -1)
        out = torch.zeros_like(temp)
        assert vec.size(0) == self.kept_indices.size(0)
        n = vec.size(0)
        for i in range(n):
            out[i, : self.kept_indices[i].shape[0]] = temp[i, self.kept_indices[i]]
            out[i, self.kept_indices[i].shape[0] :] = temp[i, self.missing_indices[i]]
        return out

    def U(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def Ut(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def singulars(self):
        return self._singulars

    def add_zeros(self, vec):
        temp = torch.zeros(
            (vec.shape[0], self.channels * self.img_dim**2), device=vec.device
        )
        reshaped = vec.clone().reshape(vec.shape[0], -1)
        temp[:, : reshaped.shape[1]] = reshaped
        return temp

    def H(self, vec):
        """
        Multiplies the input vector by H
        """
        temp = self.Vt(vec)
        singulars = self.singulars()
        return self.U(singulars * temp[:, : singulars.shape[0]])

    def Ht(self, vec):
        """
        Multiplies the input vector by H transposed
        """
        temp = self.Ut(vec)
        singulars = self.singulars()
        return self.V(self.add_zeros(singulars * temp[:, : singulars.shape[0]]))

    def H_pinv(self, vec):
        """
        Multiplies the input vector by the pseudo inverse of H
        """
        temp = self.Ut(vec)  # (b, m) - > (b, m)
        singulars = self.singulars()  # (mxm, )
        # temp[:, :singulars.shape[0]] = temp[:, :singulars.shape[0]] / singulars
        nonzero_idx = singulars.nonzero().flatten()
        temp[:, nonzero_idx] = temp[:, nonzero_idx] / singulars[nonzero_idx]

        return self.V(self.add_zeros(temp))


# class Inpainting(H_functions):
#     def __init__(self, channels, img_dim, mask, device):
#         self.channels = channels
#         self.img_dim = img_dim
#         self.device = device
#         self.mask = mask.to(device)

#     def set_indices(self, ind):
#         pass

#     def H(self, vec):
#         vec = vec.reshape(vec.size(0), -1)
#         return vec * self.mask.reshape(1, -1)

#     def H_pinv(self, vec):
#         return vec


# Denoising
class Denoising(H_functions):
    def __init__(self, channels, img_dim, device):
        self._singulars = torch.ones(channels * img_dim**2, device=device)

    def V(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def Vt(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def U(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def Ut(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)

    def singulars(self):
        return self._singulars

    def add_zeros(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)


# Super Resolution
class SuperResolution(H_functions):
    def __init__(self, channels, img_dim, ratio, device):  # ratio = 2 or 4
        assert img_dim % ratio == 0
        self.img_dim = img_dim
        self.channels = channels
        self.y_dim = img_dim // ratio
        self.ratio = ratio
        H = torch.Tensor([[1 / ratio**2] * ratio**2]).to(device)
        self.U_small, self.singulars_small, self.V_small = torch.svd(H, some=False)
        self.Vt_small = self.V_small.transpose(0, 1)

    def V(self, vec):
        # reorder the vector back into patches (because singulars are ordered descendingly)
        temp = vec.clone().reshape(vec.shape[0], -1)
        patches = torch.zeros(
            vec.shape[0],
            self.channels,
            self.y_dim**2,
            self.ratio**2,
            device=vec.device,
        )
        patches[:, :, :, 0] = temp[:, : self.channels * self.y_dim**2].view(
            vec.shape[0], self.channels, -1
        )
        for idx in range(self.ratio**2 - 1):
            patches[:, :, :, idx + 1] = temp[
                :, (self.channels * self.y_dim**2 + idx) :: self.ratio**2 - 1
            ].view(vec.shape[0], self.channels, -1)
        # multiply each patch by the small V
        patches = torch.matmul(
            self.V_small, patches.reshape(-1, self.ratio**2, 1)
        ).reshape(vec.shape[0], self.channels, -1, self.ratio**2)
        # repatch the patches into an image
        patches_orig = patches.reshape(
            vec.shape[0], self.channels, self.y_dim, self.y_dim, self.ratio, self.ratio
        )
        recon = patches_orig.permute(0, 1, 2, 4, 3, 5).contiguous()
        recon = recon.reshape(vec.shape[0], self.channels * self.img_dim**2)
        return recon

    def Vt(self, vec):
        # extract flattened patches
        patches = vec.clone().reshape(
            vec.shape[0], self.channels, self.img_dim, self.img_dim
        )
        patches = patches.unfold(2, self.ratio, self.ratio).unfold(
            3, self.ratio, self.ratio
        )
        unfold_shape = patches.shape
        patches = patches.contiguous().reshape(
            vec.shape[0], self.channels, -1, self.ratio**2
        )
        # multiply each by the small V transposed
        patches = torch.matmul(
            self.Vt_small, patches.reshape(-1, self.ratio**2, 1)
        ).reshape(vec.shape[0], self.channels, -1, self.ratio**2)
        # reorder the vector to have the first entry first (because singulars are ordered descendingly)
        recon = torch.zeros(
            vec.shape[0], self.channels * self.img_dim**2, device=vec.device
        )
        recon[:, : self.channels * self.y_dim**2] = patches[:, :, :, 0].view(
            vec.shape[0], self.channels * self.y_dim**2
        )
        for idx in range(self.ratio**2 - 1):
            recon[:, (self.channels * self.y_dim**2 + idx) :: self.ratio**2 - 1] = (
                patches[:, :, :, idx + 1].view(
                    vec.shape[0], self.channels * self.y_dim**2
                )
            )
        return recon

    def U(self, vec):
        return self.U_small[0, 0] * vec.clone().reshape(vec.shape[0], -1)

    def Ut(self, vec):  # U is 1x1, so U^T = U
        return self.U_small[0, 0] * vec.clone().reshape(vec.shape[0], -1)

    def singulars(self):
        return self.singulars_small.repeat(self.channels * self.y_dim**2)

    def add_zeros(self, vec):
        reshaped = vec.clone().reshape(vec.shape[0], -1)
        temp = torch.zeros(
            (vec.shape[0], reshaped.shape[1] * self.ratio**2), device=vec.device
        )
        temp[:, : reshaped.shape[1]] = reshaped
        return temp


# Colorization
class Colorization(H_functions):
    def __init__(self, img_dim, device):
        self.channels = 3
        self.img_dim = img_dim
        # Do the SVD for the per-pixel matrix
        H = torch.Tensor([[0.3333, 0.3334, 0.3333]]).to(device)
        self.U_small, self.singulars_small, self.V_small = torch.svd(H, some=False)
        self.Vt_small = self.V_small.transpose(0, 1)

    def V(self, vec):
        # get the needles
        needles = (
            vec.clone().reshape(vec.shape[0], self.channels, -1).permute(0, 2, 1)
        )  # shape: B, WH, C'
        # multiply each needle by the small V
        needles = torch.matmul(
            self.V_small, needles.reshape(-1, self.channels, 1)
        ).reshape(
            vec.shape[0], -1, self.channels
        )  # shape: B, WH, C
        # permute back to vector representation
        recon = needles.permute(0, 2, 1)  # shape: B, C, WH
        return recon.reshape(vec.shape[0], -1)

    def Vt(self, vec):
        # get the needles
        needles = (
            vec.clone().reshape(vec.shape[0], self.channels, -1).permute(0, 2, 1)
        )  # shape: B, WH, C
        # multiply each needle by the small V transposed
        needles = torch.matmul(
            self.Vt_small, needles.reshape(-1, self.channels, 1)
        ).reshape(
            vec.shape[0], -1, self.channels
        )  # shape: B, WH, C'
        # reorder the vector so that the first entry of each needle is at the top
        recon = needles.permute(0, 2, 1).reshape(vec.shape[0], -1)
        return recon

    def U(self, vec):
        return self.U_small[0, 0] * vec.clone().reshape(vec.shape[0], -1)

    def Ut(self, vec):  # U is 1x1, so U^T = U
        return self.U_small[0, 0] * vec.clone().reshape(vec.shape[0], -1)

    def singulars(self):
        return self.singulars_small.repeat(self.img_dim**2)

    def add_zeros(self, vec):
        reshaped = vec.clone().reshape(vec.shape[0], -1)
        temp = torch.zeros(
            (vec.shape[0], self.channels * self.img_dim**2), device=vec.device
        )
        temp[:, : self.img_dim**2] = reshaped
        return temp


# Deblurring
class Deblurring(H_functions):
    def mat_by_img(self, M, v):
        return torch.matmul(
            M, v.reshape(v.shape[0] * self.channels, self.img_dim, self.img_dim)
        ).reshape(v.shape[0], self.channels, M.shape[0], self.img_dim)

    def img_by_mat(self, v, M):
        return torch.matmul(
            v.reshape(v.shape[0] * self.channels, self.img_dim, self.img_dim), M
        ).reshape(v.shape[0], self.channels, self.img_dim, M.shape[1])

    def __init__(self, kernel, channels, img_dim, device):
        self.img_dim = img_dim
        self.channels = channels
        # build 1D conv matrix
        H_small = torch.zeros(img_dim, img_dim, device=device)
        for i in range(img_dim):
            for j in range(i - kernel.shape[0] // 2, i + kernel.shape[0] // 2):
                if j < 0 or j >= img_dim:
                    continue
                H_small[i, j] = kernel[j - i + kernel.shape[0] // 2]
        # get the svd of the 1D conv
        self.U_small, self.singulars_small, self.V_small = torch.svd(
            H_small, some=False
        )
        ZERO = 3e-2
        self.singulars_small[self.singulars_small < ZERO] = 0
        # calculate the singular values of the big matrix
        self._singulars = torch.matmul(
            self.singulars_small.reshape(img_dim, 1),
            self.singulars_small.reshape(1, img_dim),
        ).reshape(img_dim**2)
        # sort the big matrix singulars and save the permutation
        self._singulars, self._perm = self._singulars.sort(
            descending=True
        )  # , stable=True)

    def V(self, vec):
        # invert the permutation
        temp = torch.zeros(
            vec.shape[0], self.img_dim**2, self.channels, device=vec.device
        )
        temp[:, self._perm, :] = vec.clone().reshape(
            vec.shape[0], self.img_dim**2, self.channels
        )
        temp = temp.permute(0, 2, 1)
        # multiply the image by V from the left and by V^T from the right
        out = self.mat_by_img(self.V_small, temp)
        out = self.img_by_mat(out, self.V_small.transpose(0, 1)).reshape(
            vec.shape[0], -1
        )
        return out

    def Vt(self, vec):
        # multiply the image by V^T from the left and by V from the right
        temp = self.mat_by_img(self.V_small.transpose(0, 1), vec.clone())
        temp = self.img_by_mat(temp, self.V_small).reshape(
            vec.shape[0], self.channels, -1
        )
        # permute the entries according to the singular values
        temp = temp[:, :, self._perm].permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def U(self, vec):
        # invert the permutation
        temp = torch.zeros(
            vec.shape[0], self.img_dim**2, self.channels, device=vec.device
        )
        temp[:, self._perm, :] = vec.clone().reshape(
            vec.shape[0], self.img_dim**2, self.channels
        )
        temp = temp.permute(0, 2, 1)
        # multiply the image by U from the left and by U^T from the right
        out = self.mat_by_img(self.U_small, temp)
        out = self.img_by_mat(out, self.U_small.transpose(0, 1)).reshape(
            vec.shape[0], -1
        )
        return out

    def Ut(self, vec):
        # multiply the image by U^T from the left and by U from the right
        temp = self.mat_by_img(self.U_small.transpose(0, 1), vec.clone())
        temp = self.img_by_mat(temp, self.U_small).reshape(
            vec.shape[0], self.channels, -1
        )
        # permute the entries according to the singular values
        temp = temp[:, :, self._perm].permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def singulars(self):
        return self._singulars.repeat(1, 3).reshape(-1)

    def add_zeros(self, vec):
        return vec.clone().reshape(vec.shape[0], -1)


class BDeblur(Deblurring):
    def H_pinv(self, vec):
        return vec


class JPEG(H_functions):
    def __init__(self, qf):
        self.qf = qf

    def H(self, image):
        """
        Here, we write H as decode(encode()) simply because the resulting y is easier to visualize.
        """
        return jpeg_decode(jpeg_encode(image, self.qf), self.qf)

    def H_pinv(self, x):
        return x


class Quantization(H_functions):
    def __init__(self, n_bits):
        self.n_bits = n_bits

    def H(self, image):
        # Assert that image is in range [-1, 1]
        x = (image + 1) / 2
        x = torch.floor(x * (2**self.n_bits - 1) + 0.5) / (2**self.n_bits - 1)
        x = x * 2 - 1
        return x

    def H_pinv(self, x):
        return x


class HDR(H_functions):
    def __init__(self):
        pass

    def H(self, image):
        # Assert that image is in range [-1, 1]
        x = image
        x = torch.clip(x / 0.5, -1, 1)
        return x

    def H_pinv(self, x):
        return x * 0.5


class SRConv(H_functions):
    def mat_by_img(self, M, v, dim):
        return torch.matmul(M, v.reshape(v.shape[0] * self.channels, dim, dim)).reshape(
            v.shape[0], self.channels, M.shape[0], dim
        )

    def img_by_mat(self, v, M, dim):
        return torch.matmul(v.reshape(v.shape[0] * self.channels, dim, dim), M).reshape(
            v.shape[0], self.channels, dim, M.shape[1]
        )

    def __init__(self, kernel, channels, img_dim, device, stride=1):
        self.img_dim = img_dim
        self.channels = channels
        self.ratio = stride
        small_dim = img_dim // stride
        self.small_dim = small_dim
        # build 1D conv matrix
        H_small = torch.zeros(small_dim, img_dim, device=device)
        for i in range(stride // 2, img_dim + stride // 2, stride):
            for j in range(i - kernel.shape[0] // 2, i + kernel.shape[0] // 2):
                j_effective = j
                # reflective padding
                if j_effective < 0:
                    j_effective = -j_effective - 1
                if j_effective >= img_dim:
                    j_effective = (img_dim - 1) - (j_effective - img_dim)
                # matrix building
                H_small[i // stride, j_effective] += kernel[
                    j - i + kernel.shape[0] // 2
                ]
        # get the svd of the 1D conv
        self.U_small, self.singulars_small, self.V_small = torch.svd(
            H_small, some=False
        )
        ZERO = 3e-2
        self.singulars_small[self.singulars_small < ZERO] = 0
        # calculate the singular values of the big matrix
        self._singulars = torch.matmul(
            self.singulars_small.reshape(small_dim, 1),
            self.singulars_small.reshape(1, small_dim),
        ).reshape(small_dim**2)
        # permutation for matching the singular values. See P_1 in Appendix D.5.
        self._perm = (
            torch.Tensor(
                [
                    self.img_dim * i + j
                    for i in range(self.small_dim)
                    for j in range(self.small_dim)
                ]
                + [
                    self.img_dim * i + j
                    for i in range(self.small_dim)
                    for j in range(self.small_dim, self.img_dim)
                ]
            )
            .to(device)
            .long()
        )

    def V(self, vec):
        # invert the permutation
        temp = torch.zeros(
            vec.shape[0], self.img_dim**2, self.channels, device=vec.device
        )
        temp[:, self._perm, :] = vec.clone().reshape(
            vec.shape[0], self.img_dim**2, self.channels
        )[:, : self._perm.shape[0], :]
        temp[:, self._perm.shape[0] :, :] = vec.clone().reshape(
            vec.shape[0], self.img_dim**2, self.channels
        )[:, self._perm.shape[0] :, :]
        temp = temp.permute(0, 2, 1)
        # multiply the image by V from the left and by V^T from the right
        out = self.mat_by_img(self.V_small, temp, self.img_dim)
        out = self.img_by_mat(out, self.V_small.transpose(0, 1), self.img_dim).reshape(
            vec.shape[0], -1
        )
        return out

    def Vt(self, vec):
        # multiply the image by V^T from the left and by V from the right
        temp = self.mat_by_img(self.V_small.transpose(0, 1), vec.clone(), self.img_dim)
        temp = self.img_by_mat(temp, self.V_small, self.img_dim).reshape(
            vec.shape[0], self.channels, -1
        )
        # permute the entries
        temp[:, :, : self._perm.shape[0]] = temp[:, :, self._perm]
        temp = temp.permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def U(self, vec):
        # invert the permutation
        temp = torch.zeros(
            vec.shape[0], self.small_dim**2, self.channels, device=vec.device
        )
        temp[:, : self.small_dim**2, :] = vec.clone().reshape(
            vec.shape[0], self.small_dim**2, self.channels
        )
        temp = temp.permute(0, 2, 1)
        # multiply the image by U from the left and by U^T from the right
        out = self.mat_by_img(self.U_small, temp, self.small_dim)
        out = self.img_by_mat(
            out, self.U_small.transpose(0, 1), self.small_dim
        ).reshape(vec.shape[0], -1)
        return out

    def Ut(self, vec):
        # multiply the image by U^T from the left and by U from the right
        temp = self.mat_by_img(
            self.U_small.transpose(0, 1), vec.clone(), self.small_dim
        )
        temp = self.img_by_mat(temp, self.U_small, self.small_dim).reshape(
            vec.shape[0], self.channels, -1
        )
        # permute the entries
        temp = temp.permute(0, 2, 1)
        return temp.reshape(vec.shape[0], -1)

    def singulars(self):
        return self._singulars.repeat_interleave(3).reshape(-1)

    def add_zeros(self, vec):
        reshaped = vec.clone().reshape(vec.shape[0], -1)
        temp = torch.zeros(
            (vec.shape[0], reshaped.shape[1] * self.ratio**2), device=vec.device
        )
        temp[:, : reshaped.shape[1]] = reshaped
        return temp


class Compose(H_functions):
    def __init__(self, Hs):
        self.Hs = Hs

    def set_indices(self, idx):
        # Assert that if there is inpainting, then it is at the last stage.
        self.Hs[-1].set_indices(idx)

    def H(self, image):
        n, c, h, w = image.size()
        x = image
        # import ipdb; ipdb.set_trace()
        for i in range(len(self.Hs)):
            c, h = self.Hs[i].channels, self.Hs[i].img_dim
            x = x.reshape(-1, c, h, h)
            x = self.Hs[i].H(x)
        return x

    def H_pinv(self, image):
        x = image
        n = x.size(0)
        for i in reversed(range(len(self.Hs))):
            x = self.Hs[i].H_pinv(x)
            c, h = self.Hs[i].channels, self.Hs[i].img_dim
            x = x.reshape(-1, c, h, h)
        return x


def pdf(x, sigma=10):
    return torch.exp(torch.tensor([-0.5 * (x / sigma) ** 2]))


def save_mask(f, mask):
    m_npy = mask.astype(bool)
    m = np.packbits(m_npy, axis=None)
    shape = m_npy.shape
    np.savez(f, m=m, shape=shape)


def load_mask(f):
    d = np.load(f)
    m = d["m"]
    shape = d["shape"]
    # m = np.unpackbits(m, count=np.prod(shape)).reshape(shape).view(bool)
    m = m.reshape(shape).view(bool)
    return m


def build_degredation_model(cfg: DictConfig):
    print(cfg.algo)
    deg = getattr(cfg.deg, "name", "deno")  #'inp_20ff'   #dfault value deno
    print("deg", deg)
    deg = deg.split("-")
    if len(deg) == 1:
        h, w, c = cfg.dataset.image_size, cfg.dataset.image_size, cfg.dataset.channels
        return build_one_degredation_model(cfg, h, w, c, deg[0])
    else:
        Hs = []
        h, w, c = cfg.dataset.image_size, cfg.dataset.image_size, cfg.dataset.channels
        for i in range(len(deg)):
            H = build_one_degredation_model(cfg, h, w, c, deg[i])
            Hs.append(H)

            if isinstance(H, SuperResolution) or isinstance(H, SRConv):
                h = h // H.ratio
                w = w // H.ratio

        return Compose(Hs)


def _generate_masks(n, c, h, w, device, prob=0.9):
    total = n * h * w
    mask_vec = torch.ones([1, total])
    samples = np.random.choice(total, int(total * prob), replace=False)
    mask_vec[:, samples] = 0
    mask_b = mask_vec.view(n, 1, h, w)
    mask_b = mask_b.repeat(1, c, 1, 1)
    return mask_b.view(n, -1).to(device)


def build_one_degredation_model(cfg, h, w, c, deg: str):
    device = torch.device("cuda")
    if deg == "deno":
        H = Denoising(c, w, device)
    elif "inp" in deg:
        # generate a random mask for random inpainting
        if "random" in deg:
            mask_prob = cfg.deg.mask_prob
            mask_count = cfg.deg.mask_count
            print(
                f"Generating random masks with prob {mask_prob} and count {mask_count}"
            )
            m = _generate_masks(mask_count, c, h, w, device, mask_prob)
            print(m.shape)
        else:
            exp_root = cfg.deg.mask_root
            deg_type = cfg.deg.deg_type
            masks_root = os.path.join(exp_root, f"{deg_type}.npz")
            m = load_mask(masks_root)
            m = torch.from_numpy(m)
            print(m.shape)
        H = Inpainting(c, w, m, device)
    elif "in2" in deg:
        if "random" in deg:
            mask_prob = cfg.deg.mask_prob
            mask_count = cfg.deg.mask_count
            print(
                f"Generating random masks with prob {mask_prob} and count {mask_count}"
            )
            m = _generate_masks(mask_count, c, h, w, device, mask_prob)
            print(m.shape)
        else:
            exp_root = cfg.deg.mask_root
            deg_type = cfg.deg.deg_type
            masks_root = os.path.join(exp_root, f"{deg_type}.npz")
            m = load_mask(masks_root)

            if h != 256 and w != 256:
                n = m.shape[0]
                m = np.reshape(m, (n, 3, 256, 256))
                m = np.repeat(m, w // 256, axis=2)
                m = np.repeat(m, h // 256, axis=3)
                m = np.reshape(m, (n, 3 * h * w))
                # print('m.shape', m.shape)
                # print('m', m)
                # import pdb; pdb.set_trace()

            m = torch.from_numpy(m)
        H = Inpainting2(c, w, m, device)
    elif deg[:10] == "deblur_uni":
        # Default was 13
        kernel_size = int(deg[10:])
        H = Deblurring(
            torch.Tensor([1 / kernel_size] * kernel_size).to("cuda"), c, w, device
        )
    elif deg == "bdeblur_uni":
        H = BDeblur(torch.Tensor([1 / 9] * 9).to("cuda"), c, w, device)
    elif deg == "deblur_gauss":
        # sigma = 25
        # window = 13
        sigma = 3
        window = 61
        kernel = [pdf(t, sigma) for t in range(-(window - 1) // 2, (window - 1) // 2)]
        # kernel = torch.Tensor([pdf(-2, sigma), pdf(-1, sigma), pdf(0, sigma), pdf(1, sigma), pdf(2, sigma)]).to(device)
        kernel = torch.Tensor(kernel).to(device)
        H = Deblurring(kernel / kernel.sum(), c, w, device)
    elif deg[:2] == "sr":
        downsample_by = int(deg[2:])
        H = SuperResolution(c, w, downsample_by, device)
    elif deg == "color":
        H = Colorization(w, device)
    elif "jpeg" in deg:
        qf = int(deg[4:])
        H = JPEG(qf)
    elif "quant" in deg:
        qf = int(deg[5:])
        H = Quantization(qf)
    elif deg == "hdr":
        H = HDR()
    elif "bicubic" in deg:
        factor = int(cfg.deg.factor)

        def bicubic_kernel(x, a=-0.5):
            if abs(x) <= 1:
                return (a + 2) * abs(x) ** 3 - (a + 3) * abs(x) ** 2 + 1
            elif 1 < abs(x) and abs(x) < 2:
                return a * abs(x) ** 3 - 5 * a * abs(x) ** 2 + 8 * a * abs(x) - 4 * a
            else:
                return 0

        k = np.zeros((factor * 4))
        for i in range(factor * 4):
            x = (1 / factor) * (i - np.floor(factor * 4 / 2) + 0.5)
            k[i] = bicubic_kernel(x)
        k = k / np.sum(k)
        kernel = torch.from_numpy(k).float().to(device)
        H = SRConv(kernel / kernel.sum(), c, h, device, stride=factor)
    elif deg == "deblur_nl":

        # task_config = load_yaml(args.task_config)
        # measure_config = task_config['measurement']
        # H = NonlinearBlurOperator(device, **measure_config['operator'])
        # current_dir = os.getcwd()

        # current_dir = '/lustre/fsw/nvresearch/mmardani/source/latent-diffusion-sampling/pgdm'
        current_dir = os.getcwd()  # "/home/fmaroufs/projects/c-pigdm"

        # opt_yml_path = os.path.join(
        #     current_dir, "bkse/options/generate_blur/default.yml"
        # )
        opt_yml_path = os.path.join(
            current_dir, cfg.deg.opt_yml_path
        )  #'bkse/options/generate_blur/default.yml'
        # opt_yml_path = os.path.join(current_dir, 'bkse/options/generic_deblur/default.yml')
        # print("opt_yml_path", opt_yml_path)
        H = NonlinearBlurOperator(
            device=device,
            opt_yml_path=opt_yml_path,
            current_dir=current_dir,
            cfg=cfg,
        )
    elif deg == "phase_retrieval":
        oversample = 2.0
        H = PhaseRetrievalOperator(oversample=oversample, device=device)

    elif deg == "bid":
        H = BlindImageDeblurring(cfg, device)
    else:
        raise ValueError(f"Degredation model {deg} does not exist.")

    return H


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def get_degreadation_image(y_0, H, cfg):
    c, w = cfg.dataset.channels, cfg.dataset.image_size
    pinv_y = H.H_pinv(y_0).reshape(-1, c, w, w)
    return pinv_y


# nonlinear operators

from abc import ABC, abstractmethod


class NonLinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        pass

    def project(self, data, measurement, **kwargs):
        return data + measurement - self.forward(data)


# @register_operator(name='nonlinear_blur')
class NonlinearBlurOperator(NonLinearOperator, H_functions):
    def __init__(self, opt_yml_path, current_dir, cfg, device):
        self.device = device
        self.blur_model = self.prepare_nonlinear_blur_model(
            opt_yml_path, current_dir, cfg
        )
        self.random_kernel = torch.randn(512, 2, 2).to(self.device) * 1.2

    def prepare_nonlinear_blur_model(self, opt_yml_path, current_dir, cfg):
        """
        Nonlinear deblur requires external codes (bkse).
        """
        from bkse.models.kernel_encoding.kernel_wizard import KernelWizard

        with open(opt_yml_path, "r") as f:
            opt = yaml.safe_load(f)
            opt = opt["KernelWizard"]
            model_path = cfg.deg.pretrained_model_path

            assert os.path.exists(
                model_path
            ), f"Pretrained model {model_path.split('/')[0]} does not exist."

        blur_model = KernelWizard(opt)
        blur_model.eval()
        blur_model.load_state_dict(torch.load(model_path, map_location=self.device))

        # Disable gradient computation for all parameters
        with torch.no_grad():
            for param in blur_model.parameters():
                param.requires_grad_(False)

        blur_model = blur_model.to(self.device)

        return blur_model

    def forward(self, data, **kwargs):
        # random_kernel = torch.randn(data.shape[0], 512, 2, 2).to(self.device) * 1.2
        data = (data + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        blurred = self.blur_model.adaptKernel(
            data, kernel=self.random_kernel.unsqueeze(0).repeat(data.shape[0], 1, 1, 1)
        )
        blurred = (blurred * 2.0 - 1.0).clamp(-1, 1)  # [0, 1] -> [-1, 1]
        return blurred

    def H(self, data, **kwargs):
        return self.forward(data, **kwargs)

    def H_pinv(self, x):
        return self.H(x)


from utils.fft_utils import fft2_m, ifft2_m


# @register_operator(name='phase_retrieval')
class PhaseRetrievalOperator(NonLinearOperator):
    def __init__(self, oversample, device):
        self.pad = int((oversample / 8.0) * 256)
        self.device = device

    def forward(self, data, **kwargs):
        padded = F.pad(data, (self.pad, self.pad, self.pad, self.pad))
        amplitude = fft2_m(padded).abs()
        return amplitude

    def H(self, data, **kwargs):
        return self.forward(data, **kwargs)

    def H_pinv(self, x):
        x = ifft2_m(x).abs()
        x = self.undo_padding(x, self.pad, self.pad, self.pad, self.pad)
        return x

    def undo_padding(self, tensor, pad_left, pad_right, pad_top, pad_bottom):
        # Assuming 'tensor' is the 4D tensor
        # 'pad_left', 'pad_right', 'pad_top', 'pad_bottom' are the padding values
        if tensor.dim() != 4:
            raise ValueError("Input tensor should have 4 dimensions.")
        return tensor[:, :, pad_top:-pad_bottom, pad_left:-pad_right]


# code adapted from DMPlug paper: https://arxiv.org/abs/2405.16749
class BlindImageDeblurring(NonLinearOperator):
    def __init__(self, cfg, device):
        self.device = device
        self.cfg = cfg
        self.kernel = self.prepare_kernel(cfg, device)

    def prepare_kernel(self, cfg, device):
        kernel_size = cfg.deg.kernel_size
        bs = cfg.loader.batch_size

        if cfg.deg.kernel == "gaussian":
            conv = Blurkernel("gaussian", kernel_size=kernel_size, device=device)
            kernel = conv.get_kernel().type(torch.float32)
            kernel = kernel.to(device).view(bs, 1, kernel_size, kernel_size)

        elif cfg.deg.kernel == "motion":
            kernel = Kernel(
                size=(kernel_size, kernel_size), intensity=cfg.deg.intensity
            ).kernelMatrix
            kernel = torch.from_numpy(kernel).type(torch.float32)
            kernel = kernel.to(device).view(bs, 1, kernel_size, kernel_size)

        return kernel

    def forward(self, data, kernel=None):
        if kernel is None:
            kernel = self.kernel
        b_img = torch.zeros_like(data).to(self.device)
        for i in range(3):  # per channel
            b_img[:, i, :, :] = F.conv2d(
                data[:, i : i + 1, :, :], kernel, padding="same"
            )
        return b_img

    def H(self, data, kernel=None, **kwargs):
        return self.forward(data, kernel, **kwargs)

    def H_pinv(self, x):
        return self.H(x)


class Blurkernel(nn.Module):
    def __init__(self, blur_type="gaussian", kernel_size=31, std=3.0, device=None):
        super().__init__()
        self.blur_type = blur_type
        self.kernel_size = kernel_size
        self.std = std
        self.device = device
        self.seq = nn.Sequential(
            nn.ReflectionPad2d(self.kernel_size // 2),
            nn.Conv2d(
                3, 3, self.kernel_size, stride=1, padding=0, bias=False, groups=3
            ),
        )

        self.weights_init()

    def forward(self, x):
        return self.seq(x)

    def weights_init(self):
        if self.blur_type == "gaussian":
            n = np.zeros((self.kernel_size, self.kernel_size))
            n[self.kernel_size // 2, self.kernel_size // 2] = 1
            k = scipy.ndimage.gaussian_filter(n, sigma=self.std)
            k = torch.from_numpy(k)
            self.k = k
            for name, f in self.named_parameters():
                f.data.copy_(k)
        elif self.blur_type == "motion":
            k = Kernel(
                size=(self.kernel_size, self.kernel_size), intensity=self.std
            ).kernelMatrix
            k = torch.from_numpy(k)
            self.k = k
            for name, f in self.named_parameters():
                f.data.copy_(k)

    def update_weights(self, k):
        if not torch.is_tensor(k):
            k = torch.from_numpy(k).to(self.device)
        for name, f in self.named_parameters():
            f.data.copy_(k)

    def get_kernel(self):
        return self.k
