#! /usr/bin/env python
#
# -*- coding: utf-8 -*-
#
#  Purpose:
#    Definitions of SPH kernel functions
#
#    Symbol:
#      W( |r - r'|, h) = w(q) / h^d
#
#        where q = |r - r'| / h
#              h : smoothing length
#              d : dimension
#      sigma is a normalization constant
#
#  Last Updated: 2020/11/29
#  He-Feng Hsieh


import os
import numpy as np
from bisect import bisect_left, bisect_right
from numpy.polynomial.polynomial import polyval
from scipy.spatial import cKDTree


__all__ = ["Kernel_CubicSpline",
           "Kernel_Harmonic",
           "Kernel_InteHarmonic",
           "calc_sph_radgrad",
           "calc_sph_radgrad_kdtree"]


# global constant
pihalf = 0.5 * np.pi
pwd = os.path.dirname(__file__)  # path to this file, for loading the sinx_x.dat file


class Kernel_CubicSpline():
    """
    Cubic Spline Kernel (Monaghan 1992)

    Parameters
    ----------
    xij: the distance in each coordinate
    rij: relative distance, | r - r' |
    h:   smoothing length
    dim: dimension
    """
    sigma_dim = {1: 2. / 3,
                 2: 10. / (7 * np.pi),
                 3: 1 / np.pi          }

    def __init__(self, dim = 3):
        assert dim in [1, 2, 3], "Invalid dim: {}".format(dim)

        self.update_par(dim = dim)

    def update_par(self, dim):
        # update parameters
        self.dim  = dim
        self.norm = self.sigma_dim[dim]

    def kernel(self, rij, h):
        # compute the kernel, W
        q = rij / h
        w = np.zeros(q.size)  # q > 2

        # 2 >= q > 1
        cond = (1.0 < q) & (q <= 2.0)
        if np.any(cond):
            q_cond  = q[cond]
            w[cond] = 0.25 * (2 - q_cond)**3

        # 1 >= q
        cond = (q <= 1)
        if np.any(cond):
            q_cond  = q[cond]
            w[cond] = 1 - 1.5 * q_cond * q_cond * (1 - 0.5 * q_cond)

        # multiply the normalization constant and h^d
        w *= self.norm / h**self.dim

        return w

    def dwdq(self, rij, h):
        # compute dW / dq
        q    = rij / h
        dwdq = np.zeros(q.size)  # q > 2

        # 2 >= q > 1
        cond = (1.0 < q) & (q <= 2.0)
        if np.any(cond):
            q_cond     = q[cond]
            dwdq[cond] = -0.75 * (2 - q_cond)**2

        # 1 >= q
        cond = (q <= 1)
        if np.any(cond):
            q_cond     = q[cond]
            dwdq[cond] = -3 * q_cond * (1 - 0.75 * q_cond)

        # multiply the normalization constant and h^d
        dwdq *= self.norm / h**self.dim

        return dwdq

    def gradient(self, xij, rij, h):
        # compute dW / dx_i = (dW / dq) * (1 / h) * (x_i / r)
        #                   = (dW / dq) * ( x_i / (q * h^2) )
        q    = rij / h
        dwdq = self.dwdq(rij, h)

        cond         = (rij != 0.0)
        dwdq[ cond] /= (rij * h)[cond]
        dwdq[~cond]  = 0.0

        # convert to numpy array if only one data is given
        if isinstance(dwdq, np.float64):
            dwdq = np.ones(1) * dwdq

        grad = dwdq[:, np.newaxis] * xij

        return grad


class Kernel_Harmonic():
    """
    Harmonic kernel (Ruben Cabezon et al., 2008, JCP, 227, 8523)

    Parameters
    ----------
    xij: the distance in each coordinate
    rij: relative distance, | r - r' |
    h:   smoothing length
    dim: dimension
    """
    sigma_coe_dim = {1: (2.645648550e-1, 1.824975074e-1, 2.426267234e-2,
                         3.112409973e-3, 2.404560325e-4, 8.032609314e-6 ),
                     2: (7.332472794e-2, 1.196424712e-1, 3.319286897e-3,
                         5.511884969e-4, 4.828286027e-5, 1.733765883e-6 ),
                     3: (2.719001858e-2, 5.469083261e-2, 1.711165670e-2,
                         1.237264940e-3, 8.193975215e-5, 2.552695164e-6 ) }

    def __init__(self, dim = 3, n = 3.0):
        assert dim in [1, 2, 3], "Invalid dim: {}".format(dim)
        self.update_par(dim = dim, n = n)

    def update_par(self, dim, n):
        # update parameters
        self.dim  = dim
        self.n    = n
        self.norm = self._get_norm()

    def _get_norm(self):
        return polyval(self.n, self.sigma_coe_dim[self.dim])

    def kernel(self, rij, h):
        # compute the kernel, W
        q = rij / h
        w = np.zeros(q.size)  # q > 2

        # 2 >= q
        cond = (q <= 2.0)
        if np.any(cond):
            q_cond  = q[cond]
            w[cond] = (np.sinc(0.5 * q_cond))**self.n

        # multiply the normalization constant and h^d
        w *= self.norm / h**self.dim

        return w

    def dwdq(self, rij, h):
        # compute dW / dq
        q    = rij / h
        dwdq = np.zeros(q.size)  # q > 2 and q = 0

        # 2 >= q > 0
        cond = (0.0 < q) & (q <= 2.0)
        if np.any(cond):
            q_cond     = q[cond]
            q_pihalf   = q_cond * pihalf
            dwdq[cond] = self.n * pihalf * (np.sinc(0.5 * q_cond))**self.n \
                       * (1 / np.tan(q_pihalf) - 1 / q_pihalf )

        # multiply the normalization constant and h^d
        dwdq *= self.norm / h**self.dim

        return dwdq

    def gradient(self, xij, rij, h):
        # compute dW / dx_i = (dW / dq) * (1 / h) * (x_i / r)
        #                   = (dW / dq) * ( x_i / (q * h^2) )
        q    = rij / h
        dwdq = self.dwdq(rij, h)

        cond        = (rij != 0.0)
        dwdq[cond] /= (rij * h)[cond]

        # convert to numpy array if only one data is given
        if isinstance(dwdq, np.float64):
            dwdq = np.ones(1) * dwdq

        grad = dwdq[:, np.newaxis] * xij

        return grad


class Kernel_InteHarmonic():
    """
    Interpolated Harmonic kernel (Ruben Cabezon et al., 2008, JCP, 227, 8523)

    Parameters
    ----------
    xij: the distance in each coordinate
    rij: relative distance, | r - r' |
    h:   smoothing length
    dim: dimension
    """
    sigma_coe_dim = {1: (2.645648550e-1, 1.824975074e-1, 2.426267234e-2,
                         3.112409973e-3, 2.404560325e-4, 8.032609314e-6 ),
                     2: (7.332472794e-2, 1.196424712e-1, 3.319286897e-3,
                         5.511884969e-4, 4.828286027e-5, 1.733765883e-6 ),
                     3: (2.719001858e-2, 5.469083261e-2, 1.711165670e-2,
                         1.237264940e-3, 8.193975215e-5, 2.552695164e-6 ) }

    uk, fk, fdk1, fdk2 = np.genfromtxt(os.path.join(pwd, "sinx_x.dat"),
                                       unpack = True)

    def __init__(self, dim = 3, n = 5.0):
        assert dim in [1, 2, 3], "Invalid dim: {}".format(dim)
        self.update_par(dim = dim, n = n)

    def update_par(self, dim, n):
        # update parameters
        self.dim  = dim
        self.n    = n
        self.norm = self._get_norm()

    def _get_norm(self):
        return polyval(self.n, self.sigma_coe_dim[self.dim])

    def kernel(self, rij, h):
        # compute the kernel, W
        q = rij / h
        w = np.zeros(q.size)  # q >= 2

        # q = 0
        cond_zero = (q == 0.0)
        if np.any(cond_zero):
            w[cond_zero] = 1.0

        # 2 > q > 0
        cond = ~cond_zero & (q < 2.0)
        if np.any(cond):
            q_cond  = q[cond]
            idx     = (q_cond * 10**4).astype(int)
            w[cond] = (self.fk[idx] + self.fdk1[idx] * (q_cond - self.uk[idx]))**self.n

        # insure the kernel is postive
        w = np.where(w < 0.0, 0.0, w)

        # multiply the normalization constant and h^d
        w *= self.norm / h**self.dim

        return w

    def dwdq(self, rij, h):
        # compute dW / dq
        q    = rij / h
        dwdq = np.zeros(q.size)  # q >= 2 and q = 0

        # 2 > q > 0
        cond = (0.0 < q) & (q < 2.0)
        if np.any(cond):
            q_cond  = q[cond]
            idx     = (q_cond * 10**4).astype(int)
            dwdq[cond] = self.n * self.fdk2[idx] \
                       * (self.fk[idx] + self.fdk1[idx] * (q_cond - self.uk[idx]))**self.n

        # multiply the normalization constant and h^d
        dwdq *= self.norm / h**self.dim

        return dwdq

    def gradient(self, xij, rij, h):
        # compute dW / dx_i = (dW / dq) * (1 / h) * (x_i / r)
        #                   = (dW / dq) * ( x_i / (q * h^2) )
        q    = rij / h
        dwdq = self.dwdq(rij, h)

        cond        = (rij != 0.0)
        dwdq[cond] /= (rij * h)[cond]

        # convert to numpy array if only one data is given
        if isinstance(dwdq, np.float64):
            dwdq = np.ones(1) * dwdq

        grad = dwdq[:, np.newaxis] * xij

        return grad


def calc_sph_radgrad(quant, xs, ys, zs, rs, hs, dens, mass, kernel):
    """
    Calculate radial gradient of given quantity 'quant' using SPH formalism:

      d(quant)      1
      -------- = ------- * \sum_j m_j * (rho_j - rho_i) * \nabla_i W
         dr       rho_i

    where rho is density
            m is particle mass
            W is kernel

    Parameters
    ----------
    quant: quantity to be computed
    xs, ys, zs: coordinate
    rs: radius
    hs: smoothing length
    dens: density
    mass: particle mass
    kernel: SPH kernel
    """
    xyzs = np.vstack([xs, ys, zs]).T

    def calc_deri_serial(q, xyz, r, d, h):
        ### calculate the derivative for one given particle
        # simple selection from r: find particle with radius in [r - 2h, r + 2h]
        idx_left  = bisect_left (rs, r - 2 * h)
        idx_right = bisect_right(rs, r + 2 * h)
        indices   = slice(idx_left, idx_right)

        # compute the relative distance xij and rij
        xij = xyz - xyzs[indices]
        rij = np.sqrt(np.sum(xij * xij, axis = 1))

        # selection again using dr: find particle with relative distance dr <= 2h
        cond = (rij < 2 * h)

        nabla_W = kernel.gradient(xij[cond], rij[cond], h)

        # compute dqdx, dqdy, dqdz, and dqdr
        dqdx_i = ( mass[indices][cond] * (quant[indices][cond] - q) )[:, np.newaxis] \
               * nabla_W
        dqdx_i = np.sum(dqdx_i, axis = 0) / d
        dqdr   = np.dot(dqdx_i, xyz) / r

        return dqdr

    dqdr = [calc_deri_serial(q, xyz, r, d, h)
            for q, xyz, r, d, h in zip(quant, xyzs, rs, dens, hs)]

    return np.array(dqdr)


def calc_sph_radgrad_kdtree(quant, xs, ys, zs, rs, hs, dens, mass, kernel):
    """
    Same as calc_sph_radgrad() but using KD-tree to find Fixed-radius (2h) near neighbors

    For 200k case, using KD-tree saves half of walltime.
    """
    xyzs = np.vstack([xs, ys, zs]).T

    # construct KD Tree
    kdtree = cKDTree(xyzs)

    def calc_deri_serial(q, xyz, r, d, h):
        ### calculate the derivative for one given particle
        # find particles with dr <= 2h
        indices = kdtree.query_ball_point(xyz, 2 * h)

        # compute the relative distance xij and rij
        xij = xyz - xyzs[indices]
        rij = np.sqrt(np.sum(xij * xij, axis = 1))

        nabla_W = kernel.gradient(xij, rij, h)

        # compute dqdx, dqdy, dqdz, and dqdr
        dqdx_i = ( mass[indices] * (quant[indices] - q) )[:, np.newaxis] \
               * nabla_W
        dqdx_i = np.sum(dqdx_i, axis = 0) / d
        dqdr   = np.dot(dqdx_i, xyz) / r

        return dqdr

    dqdr = [calc_deri_serial(q, xyz, r, d, h)
            for q, xyz, r, d, h in zip(quant, xyzs, rs, dens, hs)]

    return np.array(dqdr)

