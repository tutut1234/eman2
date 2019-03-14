from __future__ import print_function
from __future__ import division
from past.utils import old_div

import nose.tools as nt

import EMAN2_cppwrap as e2cpp
import EMAN2_cppwrap
import numpy
import copy
import math
# from ..libpy_py3 import sphire_fundamentals as fu
from ..libpy import fundamentals as fu
import fundamentals as oldfu
# from .sparx_lib import sparx_fundamentals as oldfu


import global_def
import unittest

def get_data(num):
    dim = 10
    data_list = []
    for i in range(num):
        a = e2cpp.EMData(dim, dim)
        data_a = a.get_3dview()
        data_a[...] = numpy.arange(dim * dim, dtype=numpy.float32).reshape(dim, dim) + i
        data_list.append(a)
    return data_list

class Test_lib_compare(unittest.TestCase):
    def test_ccf_true_should_return_equal_objects(self):
        a, b = get_data(2)
        return_new = fu.ccf(a, b)
        return_old = oldfu.ccf(a, b)
        self.assertTrue(numpy.array_equal(return_new.get_3dview(), return_old.get_3dview()))

    def test_scf_true_should_return_equal_objects(self):
        a, = get_data(1)
        return_new = fu.scf(a)
        return_old = oldfu.scf(a)
        self.assertTrue(numpy.array_equal(return_new.get_3dview(), return_old.get_3dview()))

    def test_cyclic_shift_should_return_equal_object(self):
        a, = get_data(1)
        return_new = fu.cyclic_shift(a)
        return_old = oldfu.cyclic_shift(a)
        self.assertTrue(numpy.array_equal(return_new.get_3dview(), return_old.get_3dview()))

    def test_mirror_should_return_equal_object(self):
        a, = get_data(1)
        return_new = fu.mirror(a)
        return_old = oldfu.mirror(a)
        self.assertTrue(numpy.array_equal(return_new.get_3dview(), return_old.get_3dview()))


    def test_fft_should_return_equal_object(self):
        a, = get_data(1)
        return_new = fu.fft(a)
        return_old = oldfu.fft(a)
        self.assertTrue(numpy.array_equal(return_new.get_3dview(), return_old.get_3dview()))


    def test_fftip_should_return_equal_object(self):
        a, = get_data(1)
        return_new = copy.deepcopy(a)
        return_old = copy.deepcopy(a)
        fu.fftip(return_new)
        oldfu.fftip(return_old)
        self.assertTrue(numpy.array_equal(return_new.get_3dview(), return_old.get_3dview()))

    def test_fpol_should_return_equal_object(self):
        a, = get_data(1)
        nx = a.get_xsize()
        print(a.get_xsize())
        return_new = fu.fpol(a,nx,nx)
        return_old = oldfu.fpol(a,nx,nx)
        self.assertTrue(numpy.array_equal(return_new.get_3dview(), return_old.get_3dview()))


    def test_fdecimate_should_return_equal_object(self):
        a, = get_data(1)
        nx = a.get_xsize()
        return_new = fu.fdecimate(a,nx)
        return_old = oldfu.fdecimate(a,nx)
        self.assertTrue(numpy.array_equal(return_new.get_3dview(), return_old.get_3dview()))

    def test_fshift_should_return_equal_object(self):
        a, = get_data(1)
        return_new = fu.fshift(a)
        return_old = oldfu.fshift(a)
        self.assertTrue(numpy.array_equal(return_new.get_3dview(), return_old.get_3dview()))

    def test_subsample_should_return_equal_object(self):
        a, = get_data(1)
        return_new = fu.subsample(a)
        return_old = oldfu.subsample(a)
        self.assertTrue(numpy.array_equal(return_new.get_3dview(), return_old.get_3dview()))

    def test_resample_should_return_equal_object(self):
        a, = get_data(1)
        return_new = fu.resample(a)
        return_old = oldfu.resample(a)
        self.assertTrue(numpy.array_equal(return_new.get_3dview(), return_old.get_3dview()))

    def test_prepi_should_return_equal_object(self):
        a, = get_data(1)
        return_new = fu.prepi(a)
        return_old = oldfu.prepi(a)
        self.assertTrue(numpy.array_equal(return_new[0].get_3dview(), return_old[0].get_3dview()))
        self.assertEqual(return_new[1].i0win(1), return_old[1].i0win(1))

    def test_prepi3D_should_return_equal_object(self):
        a, = get_data(1)
        return_new = fu.prepi3D(a)
        return_old = oldfu.prepi3D(a)
        self.assertTrue(numpy.array_equal(return_new[0].get_3dview(), return_old[0].get_3dview()))
        self.assertEqual(return_new[1].i0win(1), return_old[1].i0win(1))

    def test_ramp_should_return_equal_object(self):
        a, = get_data(1)
        return_new = fu.ramp(a)
        return_old = oldfu.ramp(a)
        self.assertTrue(numpy.array_equal(return_new.get_3dview(), return_old.get_3dview()))


    def test_rot_avg_table_should_return_equal_object(self):
        a, = get_data(1)
        return_new = fu.rot_avg_table(a)
        return_old = oldfu.rot_avg_table(a)
        self.assertTrue(numpy.array_equal(return_new, return_old))

    def test_roops_table_should_return_equal_object(self):
        a, = get_data(1)
        return_new = fu.rops_table(a)
        return_old = oldfu.rops_table(a)
        self.assertTrue(numpy.array_equal(return_new, return_old))

    def test_ramp_should_return_equal_object(self):
        a, = get_data(1)
        return_new = fu.ramp(a)
        return_old = oldfu.ramp(a)
        self.assertTrue(numpy.array_equal(return_new.get_3dview(), return_old.get_3dview()))


    def test_gridrot_shift2D_should_return_equal_object(self):
        a, = get_data(1)
        return_new = fu.gridrot_shift2D(a)
        return_old = oldfu.gridrot_shift2D(a)
        self.assertTrue(numpy.array_equal(return_new.get_3dview(), return_old.get_3dview()))


    def test_rot_shift2D_should_return_equal_object(self):
        a, = get_data(1)
        return_new = fu.rot_shift2D(a)
        return_old = oldfu.rot_shift2D(a)
        self.assertTrue(numpy.array_equal(return_new.get_3dview(), return_old.get_3dview()))


    def test_rot_shift3D_should_return_equal_object(self):
        a, = get_data(1)
        return_new = fu.rot_shift3D(a)
        return_old = oldfu.rot_shift3D(a)
        self.assertTrue(numpy.array_equal(return_new.get_3dview(), return_old.get_3dview()))


    def test_rtshg_should_return_equal_object(self):
        a, = get_data(1)
        return_new = fu.rtshg(a)
        return_old = oldfu.rtshg(a)
        self.assertTrue(numpy.array_equal(return_new.get_3dview(), return_old.get_3dview()))


    def test_smallprime_should_return_equal_object(self):
        return_new = fu.smallprime(5)
        return_old = oldfu.smallprime(5)
        self.assertTrue(numpy.array_equal(return_new, return_old))


    def test_tilemic_should_return_equal_object(self):
        a, = get_data(1)
        return_new = fu.tilemic(a)
        return_old = oldfu.tilemic(a)
        self.assertTrue(numpy.array_equal(return_new, return_old))


    def test_window2d_should_return_equal_object(self):
        a, = get_data(1)
        return_new = fu.window2d(a, 5,5)
        return_old = oldfu.window2d(a,5,5)
        self.assertTrue(numpy.array_equal(return_new.get_3dview(), return_old.get_3dview()))

    def test_goldsearch_should_return_equal_object(self):
        a, = get_data(1)
        def f(x):
            return 2
        return_new = fu.goldsearch(f,0,5)
        return_old = oldfu.goldsearch(f,0,5)
        self.assertTrue(numpy.array_equal(return_new, return_old))

    def test_rotate_params_should_return_equal_object(self):
        a = [[1, 2, 3], [3, 4, 5]]
        b = [1,5,6]
        return_new = fu.rotate_params(a, b)
        return_old = oldfu.rotate_params(a,b)
        self.assertTrue(numpy.array_equal(return_new, return_old))


    def test_rotmatrix_should_return_equal_object(self):
        return_new = fu.rotmatrix(1,5,6)
        return_old = oldfu.rotmatrix(1,5,6)
        self.assertTrue(numpy.array_equal(return_new, return_old))


    def test_mulmat_should_return_equal_object(self):
        a = [[1, 5, 2], [2, 5, 3], [3, 8, 6]]
        b = [[2, 3, 2], [3, 4, 3], [3, 4, 6]]
        return_new = fu.mulmat(a,b)
        return_old = oldfu.mulmat(a,b)
        self.assertTrue(numpy.array_equal(return_new, return_old))


    def test_recmat_should_return_equal_object(self):
        a = [[1, 5, 2], [2, 5, 3], [3, 8, 1]]
        return_new = fu.recmat(a)
        return_old = oldfu.recmat(a)
        self.assertTrue(numpy.array_equal(return_new, return_old))

class TestSymClassInitC(unittest.TestCase):

    def test_c0_sym_should_return_error(self):
        with self.assertRaises(ZeroDivisionError):
            fu.symclass('c0')
        with self.assertRaises(ZeroDivisionError):
            oldfu.symclass('c0')

    def test_lower_c1_sym_should_return_lower_c1_sym(self):
        self.assertEqual(fu.symclass('c1').sym, oldfu.symclass('c1').sym)

    def test_upper_c1_sym_should_return_lower_c1_sym(self):
        self.assertEqual(fu.symclass('C1').nsym , oldfu.symclass('C1').nsym)


    def test_c1_sym_should_return_nsym_1(self):
        self.assertEqual(fu.symclass('c1').nsym, 1)
        self.assertEqual(oldfu.symclass('c1').nsym,1)

    def test_c1_sym_should_return_nsym_5(self):
        self.assertEqual(fu.symclass('c5').nsym, 5)
        self.assertEqual(oldfu.symclass('c5').nsym,5)


    def test_c1_should_return_correct_brackets(self):
        nsym = 1
        fubrackets =  fu.symclass('c1').brackets == [
            [ old_div(360., nsym), 90.0, old_div(360., nsym), 90.0],
            [ old_div(360., nsym), 180.0, old_div(360., nsym), 180.0]
                    ]

        oldfubrackets =  oldfu.symclass('c1').brackets == [
            [ old_div(360., nsym), 90.0, old_div(360., nsym), 90.0],
            [ old_div(360., nsym), 180.0, old_div(360., nsym), 180.0]
                    ]

        self.assertEqual(fubrackets , oldfubrackets)


    def test_c5_should_return_correct_brackets(self):
        nsym = 1
        fubrackets =  fu.symclass('c5').brackets == [
            [ old_div(360., nsym), 90.0, old_div(360., nsym), 90.0],
            [ old_div(360., nsym), 180.0, old_div(360., nsym), 180.0]
                    ]

        oldfubrackets =  oldfu.symclass('c5').brackets == [
            [ old_div(360., nsym), 90.0, old_div(360., nsym), 90.0],
            [ old_div(360., nsym), 180.0, old_div(360., nsym), 180.0]
                    ]

        self.assertEqual(fubrackets , oldfubrackets)

    def test_c1_should_return_correct_symangles(self):
        nsym = 1
        symangles = []
        for i in range(nsym):
            symangles.append([0.0, 0.0, i * old_div(360., nsym)])

        self.assertEqual(fu.symclass('c1').symangles, symangles)
        self.assertEqual(oldfu.symclass('c1').symangles, symangles)

    def test_c5_should_return_correct_symangles(self):
        nsym = 5
        symangles = []
        for i in range(nsym):
            symangles.append([0.0, 0.0, i * old_div(360., nsym)])

        self.assertEqual(fu.symclass('c5').symangles, symangles)
        self.assertEqual(oldfu.symclass('c5').symangles, symangles)


    def test_c1_should_return_correct_transform(self):
        transform = []
        for args in fu.symclass('c1').symangles:
            transform.append(e2cpp.Transform({"type":"spider", "phi":args[0], "theta":args[1], "psi":args[2]}))
        self.assertEqual(fu.symclass('c1').transform, transform)
        transform = []
        for args in oldfu.symclass('c1').symangles:
            transform.append(e2cpp.Transform({"type":"spider", "phi":args[0], "theta":args[1], "psi":args[2]}))
        self.assertEqual(oldfu.symclass('c1').transform, transform)


    def test_c5_should_return_correct_transform(self):
        transform = []
        for args in fu.symclass('c5').symangles:
            transform.append(e2cpp.Transform({"type":"spider", "phi":args[0], "theta":args[1], "psi":args[2]}))
        self.assertEqual(fu.symclass('c5').transform, transform)
        transform = []
        for args in oldfu.symclass('c5').symangles:
            transform.append(e2cpp.Transform({"type":"spider", "phi":args[0], "theta":args[1], "psi":args[2]}))
        self.assertEqual(oldfu.symclass('c5').transform, transform)

    def test_c1_should_return_correct_symatrix(self):
        symatrix = []
        for args in fu.symclass('c1').symangles:
            symatrix.append(fu.rotmatrix(args[0],args[1],args[2]))
        self.assertEqual(fu.symclass('c1').symatrix, symatrix)

        symatrix = []
        for args in oldfu.symclass('c1').symangles:
            symatrix.append(oldfu.rotmatrix(args[0],args[1],args[2]))
        self.assertEqual(oldfu.symclass('c1').symatrix, symatrix)


    def test_c5_should_return_correct_symatrix(self):
        symatrix = []
        for args in fu.symclass('c5').symangles:
            symatrix.append(fu.rotmatrix(args[0],args[1],args[2]))
        self.assertEqual(fu.symclass('c5').symatrix, symatrix)

        symatrix = []
        for args in oldfu.symclass('c5').symangles:
            symatrix.append(oldfu.rotmatrix(args[0],args[1],args[2]))
        self.assertEqual(oldfu.symclass('c5').symatrix, symatrix)

    @staticmethod
    def rotmatrix(phi,theta,psi):
        rphi   = numpy.radians(phi)
        rtheta = numpy.radians(theta)
        rpsi   = numpy.radians(psi)
        cosphi = numpy.cos(rphi)
        sinphi = numpy.sin(rphi)
        costheta = numpy.cos(rtheta)
        sintheta = numpy.sin(rtheta)
        cospsi = numpy.cos(rpsi)
        sinpsi = numpy.sin(rpsi)
        mat = [[0.0]*3,[0.0]*3,[0.0]*3]

        mat[0][0] =  cospsi*costheta*cosphi - sinpsi*sinphi
        mat[1][0] = -sinpsi*costheta*cosphi - cospsi*sinphi
        mat[2][0] =            sintheta*cosphi


        mat[0][1] =  cospsi*costheta*sinphi + sinpsi*cosphi
        mat[1][1] = -sinpsi*costheta*sinphi + cospsi*cosphi
        mat[2][1] =            sintheta*sinphi


        mat[0][2] = -cospsi*sintheta
        mat[1][2] =  sinpsi*sintheta
        mat[2][2] =            costheta
        return mat


class symclass_mod(object):
    def __init__(self, sym):
        """
          sym: cn, dn, oct, tet, icos
        """
        pass  # IMPORTIMPORTIMPORT from math import degrees, radians, sin, cos, tan, atan, acos, sqrt, pi
        # from utilities import get_sym, get_symt
        pass  # IMPORTIMPORTIMPORT from string import lower
        self.sym = sym.lower()
        if (self.sym[0] == "c"):
            self.nsym = int(self.sym[1:])
            if (self.nsym < 1):  global_def.ERROR("For Cn symmetry, we need n>0", "symclass", 1)
            self.brackets = [[old_div(360., self.nsym), 90.0, old_div(360., self.nsym), 90.0],
                             [old_div(360., self.nsym), 180.0, old_div(360., self.nsym), 180.0]]
            self.symangles = []
            for i in range(self.nsym):
                self.symangles.append([0.0, 0.0, i * old_div(360., self.nsym)])

        elif (self.sym[0] == "d"):
            self.nsym = 2 * int(self.sym[1:])
            if (self.nsym < 1):  global_def.ERROR("For Dn symmetry, we need n>0", "symclass", 1)
            self.brackets = [[old_div(360., self.nsym), 90.0, old_div(360., self.nsym), 90.0],
                             [old_div(360., self.nsym) * 2, 90.0, old_div(360., self.nsym) * 2, 90.0]]
            self.symangles = []
            for i in range(old_div(self.nsym, 2)):
                self.symangles.append([0.0, 0.0, i * old_div(360., self.nsym) * 2])
            for i in range(old_div(self.nsym, 2)):
                self.symangles.append(
                    [0.0, 180.0, (i * old_div(360., self.nsym) * 2 + 180.0 * (int(self.sym[1:]) % 2)) % 360.0])

        elif (self.sym[:3] == "oct"):
            self.nsym = 24
            ncap = 4
            cap_sig = old_div(360.0, ncap)
            alpha = numpy.degrees(math.acos(
                old_div(1.0, (numpy.sqrt(3.0) * numpy.tan(
                    2 * old_div(old_div(numpy.pi, ncap), 2.0))))))  # also platonic_params["alt_max"]
            theta = numpy.degrees(0.5 * math.acos(old_div(numpy.cos(numpy.radians(cap_sig)),
                                                          (1.0 - numpy.cos(numpy.radians(
                                                              cap_sig))))))  # also platonic_params["theta_c_on_two"]
            self.brackets = [[old_div(180., ncap), theta, cap_sig, alpha], [old_div(360., ncap), theta, cap_sig, alpha]]
            self.symangles = [[0.0, 0.0, float(i)] for i in range(0, 271, 90)]
            for i in range(0, 271, 90):
                for j in range(0, 271, 90):
                    self.symangles.append([float(j), 90.0, float(i)])
            for i in range(0, 271, 90):  self.symangles.append([0.0, 180.0, float(i)])

        elif (self.sym[:3] == "tet"):
            self.nsym = 12
            ncap = 3
            cap_sig = old_div(360.0, ncap)
            alpha = numpy.degrees(math.acos(old_div(1.0, (numpy.sqrt(3.0) * numpy.tan(
                2 * old_div(old_div(numpy.pi, ncap), 2.0))))))  # also platonic_params["alt_max"]
            theta = numpy.degrees(0.5 * math.acos(old_div(numpy.cos(numpy.radians(cap_sig)),
                                                          (1.0 - numpy.cos(numpy.radians(
                                                              cap_sig))))))  # also platonic_params["theta_c_on_two"]
            self.brackets = [[old_div(360.0, ncap), theta, cap_sig, alpha],
                             [old_div(360.0, ncap), theta, cap_sig, alpha]]
            lvl1 = numpy.degrees(math.acos(old_div(-1.0, 3.0)))  # There  are 3 faces at this angle
            self.symangles = [[0., 0., 0.], [0., 0., 120.], [0., 0., 240.]]
            for l1 in range(0, 241, 120):
                for l2 in range(60, 301, 120):
                    self.symangles.append([float(l1), lvl1, float(l2)])

            """Multiline Comment4"""

        elif (self.sym[:4] == "icos"):
            self.nsym = 60
            ncap = 5
            cap_sig = old_div(360.0, ncap)
            alpha = numpy.degrees(math.acos(old_div(1.0, (numpy.sqrt(3.0) * numpy.tan(
                2 * old_div(old_div(numpy.pi, ncap), 2.0))))))  # also platonic_params["alt_max"]
            theta = numpy.degrees(0.5 * math.acos(old_div(numpy.cos(numpy.radians(cap_sig)),
                                                          (1.0 - numpy.cos(numpy.radians(
                                                              cap_sig))))))  # also platonic_params["theta_c_on_two"]
            self.brackets = [[36., theta, cap_sig, alpha], [72., theta, cap_sig, alpha]]
            lvl1 = numpy.degrees(math.atan(2.0))  # there are 5 pentagons with centers at this height (angle)
            lvl2 = 180.0 - lvl1  # there are 5 pentagons with centers at this height (angle)
            self.symangles = [[0.0, 0.0, float(i)] for i in range(0, 288 + 1, 72)]
            for l1 in range(0, 288 + 1, 72):
                for l2 in range(36, 324 + 1, 72):
                    self.symangles.append([float(l1), lvl1, float(l2)])
            for l1 in range(36, 324 + 1, 72):
                for l2 in range(0, 288 + 1, 72):
                    self.symangles.append([float(l1), lvl2, float(l2)])
            for i in range(0, 288 + 1, 72):  self.symangles.append([0.0, 180.0, float(i)])

        else:
            global_def.ERROR("Unknown symmetry", "symclass", 1)

        #
        self.transform = []
        for args in self.symangles:
            self.transform.append(
                EMAN2_cppwrap.Transform({"type": "spider", "phi": args[0], "theta": args[1], "psi": args[2]}))
        self.symatrix = self.rotmatrix(self.symangles)



    def symmetry_related(self, angles, return_mirror=0, tolistconv=True):

        mirror_list = []
        if return_mirror == 1:
            nsym = self.nsym
            mult = -1
            mask = numpy.ones(self.nsym, dtype=numpy.bool)
            mirror_list.append([mult, mask])
        elif return_mirror == 0:
            nsym = self.nsym
            mult = 1
            mask = numpy.ones(self.nsym, dtype=numpy.bool)
            mirror_list.append([mult, mask])
        else:
            nsym = 2 * self.nsym
            mult = 1
            mask = numpy.zeros(2*self.nsym, dtype=numpy.bool)
            for i in range(self.nsym):
                mask[i::2*self.nsym] = True

            angles_np = numpy.array(angles, numpy.float64)
            mirror_list.append([mult, mask])
            mirror_list.append([-mult, ~mask])


        if(self.sym[0] == "c"):
            inside_values = (
                (self.brackets[0][0], 0, 360.0, 0),
                (self.brackets[0][0], 180, 360.0, 0),
            )
        elif(self.sym[0] == "d"):
            inside_values = (
                (self.brackets[0][0], 0, 360.0, 0),
                (self.brackets[0][0], 180, 360.0, 0),
                (self.brackets[0][0], 0, 360.0, self.brackets[0][0]),
                (self.brackets[0][0], 180.0, 360.0,self.brackets[0][0]),
                (numpy.nan, 90.0, 180.0, 0),
            )
        elif (self.sym == "tet") :
            inside_values = (
                (self.brackets[0][0], self.brackets[0][1], 180, 0),
                (self.brackets[0][0], 180 - self.brackets[0][1], 180, 60),
                (self.brackets[0][0], 0, self.brackets[0][0], 0),
                (self.brackets[0][0], 180 - self.brackets[0][3], self.brackets[0][0], 0),
                (self.brackets[0][0], 180, self.brackets[0][0], 0),
                (self.brackets[0][0], self.brackets[0][3], self.brackets[0][0], 60),
            )
        elif(self.sym == "oct") :
            inside_values = (
                (self.brackets[0][2], 180, self.brackets[0][2], 0),
                (self.brackets[0][2], 0, self.brackets[0][2], 0),
                (self.brackets[0][2], 2 * self.brackets[0][1], self.brackets[0][2], 0),
                (self.brackets[0][2], 2 * self.brackets[0][1], 180, 45),
                (self.brackets[0][2], 3 * self.brackets[0][1], 180, 0),
                (self.brackets[0][2], self.brackets[0][1], 180, 0),
                (self.brackets[0][2], self.brackets[0][3], 120, 45),
                (self.brackets[0][2], 180 - self.brackets[0][3], 120, 45),
            )
        elif(self.sym == "icos"):
            inside_values = (
                (self.brackets[0][2], 180, self.brackets[0][2], 0),
                (self.brackets[0][2], 0, self.brackets[0][2], 0),
                (self.brackets[0][2], 2 * self.brackets[0][1], self.brackets[0][2], 0),
                (self.brackets[0][2], 180 - 2 * self.brackets[0][1], self.brackets[0][2], self.brackets[0][0]),
                (self.brackets[0][2], self.brackets[0][3], 60, self.brackets[0][0]),
                (self.brackets[0][2], self.brackets[0][3]+2*self.brackets[0][1], 120, 0),
                (self.brackets[0][2], 180 - self.brackets[0][3] - 2 * self.brackets[0][1], 120, self.brackets[0][0]),
                (self.brackets[0][2], 180 - self.brackets[0][3], 120, 0),
                (self.brackets[0][2], self.brackets[0][1], 180, 0),
                (self.brackets[0][2], 90 - self.brackets[0][1], 180, self.brackets[0][0]),
                (self.brackets[0][0], 90, 180, self.brackets[0][0]/2.0),
                (self.brackets[0][2], 180 - self.brackets[0][1], 180, self.brackets[0][0]),
                (self.brackets[0][2], 90 + self.brackets[0][1], 180, 0),
            )
        else :
            raise NameError("Symmetry unknown")

        sang_new_raw = numpy.atleast_2d(numpy.array(angles, numpy.float64)).repeat(nsym, axis=0)
        final_masks = []
        for multiplier, sang_mask in mirror_list:

            if return_mirror not in (0, 1) and self.sym[0] == 'd' and multiplier == -1:
                theta_0_or_180 = (sang_new_raw[:, 1] == 0) | (sang_new_raw[:, 1] == 180)
                sang_mask[theta_0_or_180] = False
            sang_mod = sang_new_raw[sang_mask]

            matrices = self.rotmatrix(sang_mod)

            matrices_mod = numpy.sum(
                numpy.transpose(matrices, (0, 2, 1)).reshape(
                    matrices.shape[0],
                    matrices.shape[1],
                    matrices.shape[2],
                    1
                ) *
                numpy.tile(
                    multiplier * numpy.array(self.symatrix, numpy.float64),
                    (sang_mod.shape[0] // self.nsym, 1, 1)).reshape(
                    matrices.shape[0], matrices.shape[1], 1, matrices.shape[2], ), -3)

            sang_new = self.recmat(matrices_mod)
            theta_0_or_180 = (sang_new[:,1] == 0) | (sang_new[:,1] == 180)
            if return_mirror not in (0, 1) and self.sym[0] != 'c' and multiplier == -1:
                print(sang_new[~theta_0_or_180, 2])
                sang_new[~theta_0_or_180, 2] += 180
                print(sang_new[~theta_0_or_180, 2])
                sang_new[~theta_0_or_180, 2] %= 360
                print(sang_new[~theta_0_or_180, 2])

            masks_good = []
            masks_bad = []

            for phi, theta, psi, offset in inside_values:

                if not numpy.isnan(phi):
                    phi_0_180 = numpy.round(sang_new[:, 0] - offset, 6) < numpy.round(phi, 6)
                    phi_not_0_180 = 0 == numpy.round(sang_new[:,0] - offset, 6 ) % numpy.round(phi, 6)
                    phi_good = numpy.logical_xor(
                        phi_0_180 & theta_0_or_180,
                        phi_not_0_180 & ~theta_0_or_180
                    )
                else:
                    phi_good = numpy.ones(sang_new.shape[0], dtype=numpy.bool)
                theta_good = numpy.round(sang_new[:,1], 6) == numpy.round(theta, 6)
                psi_good = numpy.round(sang_new[:,2], 6) < numpy.round(psi, 6)
                masks_good.append(phi_good & theta_good & psi_good)
                if not numpy.isnan(phi):
                    phi_bad_0_180 = numpy.round(sang_new[:, 0] - offset, 6) >= numpy.round(phi, 6)
                    phi_bad = numpy.logical_xor(
                        phi_bad_0_180 & theta_0_or_180,
                        phi_not_0_180 & ~theta_0_or_180
                    )
                else:
                    phi_bad = numpy.ones(sang_new.shape[0], dtype=numpy.bool)

                psi_bad_not_0_180 = numpy.round(sang_new[:,2], 6) >= numpy.round(psi, 6)
                psi_bad = numpy.logical_xor(
                    psi_good & theta_0_or_180,
                    psi_bad_not_0_180 & ~theta_0_or_180
                )

                masks_bad.append(phi_bad & theta_good & psi_bad)

            mask_good = numpy.zeros(sang_new.shape[0], numpy.bool)
            for entry in masks_good:
                mask_good = numpy.logical_or(mask_good, entry)

            mask_bad = numpy.zeros(sang_new.shape[0], numpy.bool)
            for entry in masks_bad:
                mask_bad = numpy.logical_or(mask_bad, entry)

            mask_not_special = ~numpy.logical_or(
                numpy.logical_xor(mask_good, mask_bad),
                numpy.logical_and(mask_good, mask_bad)
            )
            maski = numpy.logical_or(mask_good, mask_not_special)
            output_mask = numpy.zeros(sang_new_raw.shape[0], dtype=numpy.bool)
            output_mask[sang_mask] = maski

            sang_new_raw[sang_mask] = sang_new
            final_masks.append(output_mask)

        final_mask = numpy.zeros(nsym, dtype=numpy.bool)
        for entry in final_masks:
            final_mask = numpy.logical_or(final_mask, entry)

        sang_new = sang_new_raw[final_mask]
        sang_new %= 360

        if tolistconv:
            return sang_new.tolist()
        else:
            return sang_new


    def symmetry_neighbors(self, angles, tolistconv = True):
        if( self.sym[0] == "c" or self.sym[0] == "d" ):
            temp = e2cpp.Util.symmetry_neighbors(angles,self.sym)
            nt = old_div(len(temp), 3)
            mod_angles = numpy.array([[0,0,0]]).repeat(nt, axis=0)
            mod_angles[:,0] = temp[ 0:len(temp):3 ]
            mod_angles[:,1] = temp[ 1:len(temp):3 ]
            mod_angles[:,2] = 0.0
            return mod_angles.tolist()

        #  Note symmetry neighbors below refer to the particular order
        #   in which this class generates symmetry matrices
        neighbors = {}
        neighbors["oct"]  = [0,1,2,3,8,9,12,13]
        neighbors["tet"]  = [0,1,2,3,4,6,7]
        neighbors["icos"] = [0,1,2,3,4,6,7,11,12]
        sang_new = numpy.array(angles,numpy.float64).repeat(len(neighbors[self.sym]), axis=0 )
        matrices = self.rotmatrix(sang_new)
        matrices_mod = numpy.sum(
            numpy.transpose(matrices, (0, 2, 1)).reshape(
                matrices.shape[0],
                matrices.shape[1],
                matrices.shape[2],
                1
            ) *
            numpy.tile(
                numpy.array(self.symatrix,numpy.float64)[neighbors[self.sym]],
                (sang_new.shape[0] // len(neighbors[self.sym]), 1, 1)).reshape(
                matrices.shape[0],matrices.shape[1], 1, matrices.shape[2],), -3 )
        sang_mod = self.recmat(matrices_mod, sang_new )

        if tolistconv:
            return sang_mod.tolist()
        else:
            return sang_mod


    @staticmethod
    def recmat(mat, out=None):
        pass#IMPORTIMPORTIMPORT from math import acos,asin,atan2,degrees,pi
        def sign(x):
            return_array = numpy.sign(x)
            return_array[return_array == 0] = 1
            return return_array

        mask_2_2_1 = mat[:, 2, 2] == 1.0
        mask_0_0_0 = mat[:, 0, 0] == 0.0
        mask_2_2_m1 = mat[:, 2, 2] == -1.0
        mask_2_0_0 = mat[:, 2, 0] == 0.0
        mask_0_2_0 = mat[:, 0, 2] == 0.0
        # mask_2_2_0 = mat[:, 2, 2] == 0.0
        theta_2_2 = numpy.arccos(mat[:, 2, 2])
        st = sign(theta_2_2)
        if out is None:
            output_array = numpy.empty((mat.shape[0], 3), dtype=numpy.float64)
        else:
            output_array = out


        output_array[mask_2_2_1 & mask_0_0_0, 0] = numpy.degrees(numpy.arcsin(mat[mask_2_2_1 & mask_0_0_0, 0, 1]))
        output_array[mask_2_2_1 & ~mask_0_0_0, 0] = numpy.degrees(numpy.arctan2(
            mat[mask_2_2_1 & ~mask_0_0_0, 0, 1],
            mat[mask_2_2_1 & ~mask_0_0_0, 0, 0]
        ))
        output_array[mask_2_2_1 & ~mask_2_2_m1, 1] = numpy.degrees(0.0)  # theta
        output_array[mask_2_2_1 & ~mask_2_2_m1 , 2] = numpy.degrees(0.0)  # psi

        output_array[mask_2_2_m1 &  mask_0_0_0, 0] = numpy.degrees(numpy.arcsin(-mat[mask_2_2_m1 & mask_0_0_0, 0, 1] ))
        output_array[mask_2_2_m1 & ~mask_0_0_0, 0] = numpy.degrees(numpy.arctan2(
            -mat[mask_2_2_m1 & ~mask_0_0_0, 0, 1],
            -mat[mask_2_2_m1 & ~mask_0_0_0, 0, 0]
        ))
        output_array[mask_2_2_m1 & ~mask_2_2_1, 1] = numpy.degrees(numpy.pi)
        output_array[mask_2_2_m1 & ~mask_2_2_1, 2] = numpy.degrees(0.0)

        output_array[~mask_2_2_1 & ~mask_2_2_m1 & mask_2_0_0 & (st != sign(mat[:,2,1])), 0] = numpy.degrees(1.5*numpy.pi)
        output_array[~mask_2_2_1 & ~mask_2_2_m1 & mask_2_0_0 & (st == sign(mat[:,2,1])), 0] = numpy.degrees(0.5*numpy.pi)
        output_array[~mask_2_2_1 & ~mask_2_2_m1 & ~mask_2_0_0, 0] = numpy.degrees(numpy.arctan2(
            st[~mask_2_2_1 & ~mask_2_2_m1 & ~mask_2_0_0] * mat[~mask_2_2_1 & ~mask_2_2_m1 & ~mask_2_0_0, 2, 1],
            st[~mask_2_2_1 & ~mask_2_2_m1 & ~mask_2_0_0] * mat[~mask_2_2_1 & ~mask_2_2_m1 & ~mask_2_0_0, 2, 0]
        ))
        output_array[~mask_2_2_1 & ~mask_2_2_m1, 1] = numpy.degrees(theta_2_2[~mask_2_2_1 & ~mask_2_2_m1])

        output_array[~mask_2_2_1 & ~mask_2_2_m1 & mask_0_2_0 & (st != sign(mat[:,1,2])), 2] = numpy.degrees(1.5*numpy.pi)
        output_array[~mask_2_2_1 & ~mask_2_2_m1 & mask_0_2_0 & (st == sign(mat[:, 1, 2])), 2] = numpy.degrees(0.5 * numpy.pi)

        output_array[~mask_2_2_1 & ~mask_2_2_m1 & ~mask_0_2_0 , 2] = numpy.degrees(numpy.arctan2(
            st[~mask_2_2_1 & ~mask_2_2_m1 & ~mask_0_2_0]  * mat[~mask_2_2_1 & ~mask_2_2_m1 & ~mask_0_2_0,  1, 2],
            -st[~mask_2_2_1 & ~mask_2_2_m1 & ~mask_0_2_0] * mat[~mask_2_2_1 & ~mask_2_2_m1 & ~mask_0_2_0, 0, 2]))

        # sang_new[~mask_2_2_1 & ~mask_2_2_m1, 2] = numpy.degrees(0.0)

        numpy.round(output_array, 12, out=output_array)
        output_array %= 360.0
        return output_array


    @staticmethod
    def rotmatrix(angles):

        newmat = numpy.zeros((len(angles), 3, 3),dtype = numpy.float64)
        index = numpy.arange(len(angles))

        cosphi = numpy.cos(numpy.radians(numpy.array(angles)[:, 0]))
        costheta = numpy.cos(numpy.radians(numpy.array(angles)[:, 1]))
        cospsi = numpy.cos(numpy.radians(numpy.array(angles)[ : ,2] ))

        sinphi = numpy.sin(numpy.radians(numpy.array(angles)[:, 0]))
        sintheta = numpy.sin(numpy.radians(numpy.array(angles)[:, 1]))
        sinpsi = numpy.sin(numpy.radians(numpy.array(angles)[: ,2] ))

        newmat[:,0,0] =  cospsi[index]*costheta[index]*cosphi[index] - sinpsi[index]*sinphi[index]
        newmat[:,1,0] =     -sinpsi[index]*costheta[index]*cosphi[index] - cospsi[index]*sinphi[index]
        newmat[:,2,0] =           sintheta[index]*cosphi[index]
        newmat[:,0,1] =  cospsi[index]*costheta[index]*sinphi[index] + sinpsi[index]*cosphi[index]
        newmat[:,1,1] = -sinpsi[index]*costheta[index]*sinphi[index] + cospsi[index]*cosphi[index]
        newmat[:,2,1] =            sintheta[index]*sinphi[index]
        newmat[:,0,2] = -cospsi[index]*sintheta[index]
        newmat[:,1,2] =  sinpsi[index]*sintheta[index]
        newmat[:,2,2] =            costheta[index]

        return newmat

    def is_in_subunit(self, angles, inc_mirror =1, tolistconv = True):

        if (type(angles[0]) is list):
            print("lis of list")
        else:
            print("single list")

        """
        Input:  Before it was a projection direction specified by (phi, theta).
                Now it is a projection direction specified by phi(i) , theta(i) for all angles
                inc_mirror = 1 consider mirror directions as unique
                inc_mirror = 0 consider mirror directions as outside of unique range.
        Output: True if input projection direction is in the first asymmetric subunit,
                False otherwise.
        """
        pass  # IMPORTIMPORTIMPORT from math import degrees, radians, sin, cos, tan, atan, acos, sqrt

        angles = numpy.array(angles)
        condstat = numpy.zeros(numpy.shape(angles)[0], dtype = bool  )

        phi = angles[:,0]
        phi_0 =   phi >= 0.0
        phi_ld_br_inmirr_0 = phi < self.brackets[inc_mirror][0]
        phi_ld_br_1_0 = phi < self.brackets[1][0]
        theta = angles[:, 1]
        theta_ldeq_br_incmirr_1 = theta  <= self.brackets[inc_mirror][1]
        theta_ldeq_br_incmirr_3 = theta <= self.brackets[inc_mirror][3]
        theta_180 =  (numpy.logical_and(theta ==180 , inc_mirror))
        theta_0 = theta == 0


        if self.sym[0] == "c" :
            condstat[phi_0  & phi_ld_br_inmirr_0 & theta_ldeq_br_incmirr_1] = True
            condstat[theta_180] = True
            condstat[theta_0] = True
            condstat[~phi_0 & ~phi_ld_br_inmirr_0 & ~theta_ldeq_br_incmirr_1 & ~theta_180  & ~theta_0] = False

        elif self.sym[0] == "d" and (old_div(self.nsym, 2)) % 2 == 0:
            condstat[phi_0 & phi_ld_br_inmirr_0 & theta_ldeq_br_incmirr_1] = True
            condstat[theta_0] = True
            condstat[~phi_0 & ~phi_ld_br_inmirr_0 & ~theta_ldeq_br_incmirr_1 & ~theta_0] = False

        elif self.sym[0] == "d" and (old_div(self.nsym, 2)) % 2 == 1:
            phib = old_div(360.0, self.nsym)
            condstat[numpy.logical_and( (theta_ldeq_br_incmirr_1 & phi_0 & phi_ld_br_1_0), inc_mirror)] = True
            condstat[ theta_ldeq_br_incmirr_1 & phi_0 & phi_ld_br_1_0 & \
                ( numpy.logical_or(numpy.logical_and(  (phi >= old_div(phib, 2)) , (phi < phib)) , numpy.logical_and( (phi >= phib), (phi <= phib + old_div(phib, 2)) ))) ] = True
            condstat[theta_ldeq_br_incmirr_1 & phi_0 & phi_ld_br_1_0 & theta_0] = True
            condstat[~(theta_ldeq_br_incmirr_1 & phi_0 & phi_ld_br_1_0) & theta_0] = True
            condstat[~(theta_ldeq_br_incmirr_1 & phi_0 & phi_ld_br_1_0) &  ~theta_0] = False

        elif ((self.sym[:3] == "oct") or (self.sym[:4] == "icos")):
            tmphi = numpy.minimum(phi, self.brackets[inc_mirror][2] - phi)
            baldwin_lower_alt_bound = \
                old_div(
                    (old_div(numpy.sin(numpy.radians(old_div(self.brackets[inc_mirror][2], 2.0) - tmphi)), numpy.tan(
                        numpy.radians(self.brackets[inc_mirror][1])))
                     + old_div(numpy.sin(numpy.radians(tmphi)),
                               numpy.tan(numpy.radians(self.brackets[inc_mirror][3])))),
                    numpy.sin(numpy.radians(old_div(self.brackets[inc_mirror][2], 2.0))))
            baldwin_lower_alt_bound = numpy.degrees(numpy.arctan(old_div(1.0, baldwin_lower_alt_bound)))

            condstat[ phi_0 & phi_ld_br_inmirr_0  & theta_ldeq_br_incmirr_3 & (baldwin_lower_alt_bound > theta)] = True
            condstat[phi_0 & phi_ld_br_inmirr_0 & theta_ldeq_br_incmirr_3 & ~(baldwin_lower_alt_bound > theta)] = False
            condstat[theta_0] = True
            condstat[~phi_0 & ~phi_ld_br_inmirr_0 & ~theta_ldeq_br_incmirr_3 & ~theta_0 ] = False

        elif (self.sym[:3] == "tet"):
            tmphi = numpy.minimum(phi, self.brackets[inc_mirror][2] - phi)
            baldwin_lower_alt_bound_1 = \
                old_div(
                    (old_div(numpy.sin(numpy.radians(old_div(self.brackets[inc_mirror][2], 2.0) - tmphi)),
                             numpy.tan(numpy.radians(self.brackets[inc_mirror][1]))) \
                     + old_div(numpy.sin(numpy.radians(tmphi)), numpy.tan(numpy.radians(self.brackets[inc_mirror][3])))) \
                    , numpy.sin(numpy.radians(old_div(self.brackets[inc_mirror][2], 2.0))))
            baldwin_lower_alt_bound_1 = numpy.degrees(numpy.arctan(old_div(1.0, baldwin_lower_alt_bound_1)))

            baldwin_upper_alt_bound_2 = \
                old_div(
                    (old_div((numpy.sin(numpy.radians(old_div(self.brackets[inc_mirror][2], 2.0) - tmphi))),
                             (numpy.tan(numpy.radians(self.brackets[inc_mirror][1]))))
                     + old_div((numpy.sin(numpy.radians(tmphi))),
                               numpy.tan(numpy.radians(old_div(self.brackets[inc_mirror][3], 2.0))))) \
                    , (numpy.sin(numpy.radians(old_div(self.brackets[inc_mirror][2], 2.0)))))
            baldwin_upper_alt_bound_2 = numpy.degrees(numpy.arctan(old_div(1.0, baldwin_upper_alt_bound_2)))

            condstat[phi_0 & phi_ld_br_inmirr_0 & theta_ldeq_br_incmirr_3  & \
                      numpy.logical_and((baldwin_lower_alt_bound_1 > theta) , inc_mirror)] = True
            condstat[phi_0 & phi_ld_br_inmirr_0 & theta_ldeq_br_incmirr_3  & \
                     ~numpy.logical_and((baldwin_lower_alt_bound_1 > theta) , inc_mirror) & (baldwin_upper_alt_bound_2 < theta)] = False
            condstat[ phi_0 & phi_ld_br_inmirr_0 & theta_ldeq_br_incmirr_3  & \
                     ~numpy.logical_and((baldwin_lower_alt_bound_1 > theta) , inc_mirror) & ~(baldwin_upper_alt_bound_2 < theta)] = True
            condstat[phi_0 & phi_ld_br_inmirr_0 & theta_ldeq_br_incmirr_3  & ~(baldwin_lower_alt_bound_1 > theta) &  theta_0] = True
            condstat[phi_0 & phi_ld_br_inmirr_0 & theta_ldeq_br_incmirr_3  & ~(baldwin_lower_alt_bound_1 > theta) &  ~theta_0] = False
            condstat[~phi_0 & ~phi_ld_br_inmirr_0 & ~theta_ldeq_br_incmirr_3 &  theta_0] = True
            condstat[~phi_0 & ~phi_ld_br_inmirr_0 & ~theta_ldeq_br_incmirr_3 & ~theta_0] = False

        else:
            global_def.ERROR("unknown symmetry", "symclass: is_in_subunit", 1)

        if tolistconv:
            return condstat.tolist()
        else:
            return condstat


    def reduce_anglesets(self, angles,inc_mirror=1, tolistconv = True):
        """
          Input is either list ot lists [[phi,thet,psi],[],[]] or a triplet [phi,thet,psi]
                inc_mirror = 1 consider mirror directions as unique
                inc_mirror = 0 consider mirror directions as outside of unique range.
          It will map all triplets to the first asymmetric subunit.
        """

        sym_angles = self.symmetry_related(angles)
        print(sym_angles)
        subunits   = self.is_in_subunit(sym_angles, inc_mirror)
        reduced_anglesets = numpy.array(sym_angles)[subunits]

        if tolistconv:
            return reduced_anglesets.tolist()
        else:
            return reduced_anglesets




class TestSymClassIsInSubunitC(unittest.TestCase):
    output_template_angles = 'Got: {0} ; Expected: {1} ; Angle: {2}'

    def test_c1_sym_no_mirror_theta_smaller_equals_90_degrees_should_return_True(self):
        angles = [[entry, thet, 0] for entry in range(360) for thet in range(91)]
        [angles.append([entry, 0, 0]) for entry in range(360)]
        results = []
        for phi, theta, psi in angles:
            is_in = fu.symclass('c1').is_in_subunit(phi, theta, 0)
            results.append(is_in)
        expected_results = symclass_mod('c1').is_in_subunit(angles)
        for i, k, j in zip(results, expected_results, angles):
            if i != k:
                print(self.output_template_angles.format(i, k, j))
        self.assertEqual(results,expected_results)


        angles = [[entry, thet, 0] for entry in range(360) for thet in range(91)]
        [angles.append([entry, 0, 0]) for entry in range(360)]
        results = []
        for phi, theta, psi in angles:
            is_in = oldfu.symclass('c1').is_in_subunit(phi, theta, 0)
            results.append(is_in)
        expected_results = symclass_mod('c1').is_in_subunit(angles)
        for i, k, j in zip(results, expected_results, angles):
            if i != k:
                print(self.output_template_angles.format(i, k, j))
        self.assertEqual(results,expected_results)

    def test_c1_sym_no_mirror_theta_larger_90_degrees_should_return_False(self):
        angles = [[entry, thet, 0] for entry in range(360) for thet in range(90, 180)]
        [angles.append([entry, 180, 0]) for entry in range(360)]
        results = []
        for phi, theta, psi in angles:
            if theta != 180:
                theta = theta + 0.1
            is_in = fu.symclass('c1').is_in_subunit(phi, theta, 0)
            results.append(is_in)
        expected_results = symclass_mod('c1').is_in_subunit(angles)
        # for i, k, j in zip(results, expected_results, angles):
        #     if i != k:
        #         print(self.output_template_angles.format(i, k, j))
        self.assertTrue(results, expected_results)


        angles = [[entry, thet, 0] for entry in range(360) for thet in range(90, 180)]
        [angles.append([entry, 180, 0]) for entry in range(360)]
        results = []
        for phi, theta, psi in angles:
            if theta != 180:
                theta = theta + 0.1
            is_in = oldfu.symclass('c1').is_in_subunit(phi, theta, 0)
            results.append(is_in)
        expected_results = symclass_mod('c1').is_in_subunit(angles)
        # for i, k, j in zip(results, expected_results, angles):
        #     if i != k:
        #         print(self.output_template_angles.format(i, k, j))
        self.assertTrue(results, expected_results)

    def test_c1_sym_mirror_theta_smaller_equals_90_degrees_should_return_True(self):
        angles = [[entry, thet, 0] for entry in range(360) for thet in range(91)]
        [angles.append([entry, 0, 0]) for entry in range(360)]
        results = []
        for phi, theta, psi in angles:
            is_in = fu.symclass('c1').is_in_subunit(phi, theta, 1)
            results.append(is_in)
        expected_results = symclass_mod('c1').is_in_subunit(angles)
        for i, k, j in zip(results, expected_results, angles):
            if i != k:
                print(self.output_template_angles.format(i, k, j))
        self.assertEqual(results, expected_results)

        angles = [[entry, thet, 0] for entry in range(360) for thet in range(91)]
        [angles.append([entry, 0, 0]) for entry in range(360)]
        results = []
        for phi, theta, psi in angles:
            is_in = oldfu.symclass('c1').is_in_subunit(phi, theta, 1)
            results.append(is_in)
        expected_results = symclass_mod('c1').is_in_subunit(angles)
        for i, k, j in zip(results, expected_results, angles):
            if i != k:
                print(self.output_template_angles.format(i, k, j))
        self.assertEqual(results, expected_results)

    def test_c1_sym_mirror_theta_larger_90_degrees_should_return_False(self):
        angles = [[entry, thet, 0] for entry in range(360) for thet in range(90, 180)]
        [angles.append([entry, 180, 0]) for entry in range(360)]
        results = []
        for phi, theta, psi in angles:
            if theta != 180:
                theta = theta + 0.1
            is_in = fu.symclass('c1').is_in_subunit(phi, theta, 1)
            results.append(is_in)
        expected_results = symclass_mod('c1').is_in_subunit(angles)
        # for i, k, j in zip(results, expected_results, angles):
        #     if i != k:
        #         print(self.output_template_angles.format(i, k, j))
        self.assertTrue(results, expected_results)

        angles = [[entry, thet, 0] for entry in range(360) for thet in range(90, 180)]
        [angles.append([entry, 180, 0]) for entry in range(360)]
        results = []
        for phi, theta, psi in angles:
            if theta != 180:
                theta = theta + 0.1
            is_in = oldfu.symclass('c1').is_in_subunit(phi, theta, 1)
            results.append(is_in)
        expected_results = symclass_mod('c1').is_in_subunit(angles)
        # for i, k, j in zip(results, expected_results, angles):
        #     if i != k:
        #         print(self.output_template_angles.format(i, k, j))
        self.assertTrue(results, expected_results)

    def test_c5_sym_no_mirror_theta_smaller_equals_90_phi_smaller_72_degrees_should_return_True(self):
        angles = [[entry, thet, 0] for entry in range(72) for thet in range(91)]
        [angles.append([entry, 0, 0]) for entry in range(360)]
        results = []
        for phi, theta, psi in angles:
            is_in = fu.symclass('c5').is_in_subunit(phi, theta, 0)
            results.append(is_in)
        expected_results = symclass_mod('c5').is_in_subunit(angles)
        for i, k, j in zip(results, expected_results, angles):
            if i != k:
                print(self.output_template_angles.format(i, k, j))
        self.assertEqual(results, expected_results)


        angles = [[entry, thet, 0] for entry in range(72) for thet in range(91)]
        [angles.append([entry, 0, 0]) for entry in range(360)]
        results = []
        for phi, theta, psi in angles:
            is_in = oldfu.symclass('c5').is_in_subunit(phi, theta, 0)
            results.append(is_in)
        expected_results = symclass_mod('c5').is_in_subunit(angles)
        for i, k, j in zip(results, expected_results, angles):
            if i != k:
                print(self.output_template_angles.format(i, k, j))
        self.assertEqual(results, expected_results)

    def test_c5_sym_no_mirror_theta_larger_90_degrees_phi_smaller_72_should_return_False(self):
        angles = [[entry, thet, 0] for entry in range(72) for thet in range(90, 180)]
        [angles.append([entry, 180, 0]) for entry in range(360)]
        results = []
        for phi, theta, psi in angles:
            if theta != 180:
                theta = theta + 0.1
            is_in = fu.symclass('c5').is_in_subunit(phi, theta, 0)
            results.append(is_in)
        expected_results = symclass_mod('c5').is_in_subunit(angles)
        # for i, k, j in zip(results, expected_results, angles):
        #     if i != k:
        #         print(self.output_template_angles.format(i, k, j))
        self.assertTrue(results, expected_results)

        angles = [[entry, thet, 0] for entry in range(72) for thet in range(90, 180)]
        [angles.append([entry, 180, 0]) for entry in range(360)]
        results = []
        for phi, theta, psi in angles:
            if theta != 180:
                theta = theta + 0.1
            is_in = oldfu.symclass('c5').is_in_subunit(phi, theta, 0)
            results.append(is_in)
        expected_results = symclass_mod('c5').is_in_subunit(angles)
        # for i, k, j in zip(results, expected_results, angles):
        #     if i != k:
        #         print(self.output_template_angles.format(i, k, j))
        self.assertTrue(results, expected_results)

    def test_c5_sym_no_mirror_theta_larger_90_degrees_phi_smaller_72_should_return_False(self):
        angles = [[entry, thet, 0] for entry in range(72, 360) for thet in range(1, 91)]
        results = []
        for phi, theta, psi in angles:
            is_in = fu.symclass('c5').is_in_subunit(phi, theta, 0)
            results.append(is_in)
        expected_results = symclass_mod('c5').is_in_subunit(angles)
        # for i, k, j in zip(results, expected_results, angles):
        #     if i != k:
        #         print(self.output_template_angles.format(i, k, j))
        self.assertTrue(results, expected_results)


        angles = [[entry, thet, 0] for entry in range(72, 360) for thet in range(1, 91)]
        results = []
        for phi, theta, psi in angles:
            is_in = oldfu.symclass('c5').is_in_subunit(phi, theta, 0)
            results.append(is_in)
        expected_results = symclass_mod('c5').is_in_subunit(angles)
        # for i, k, j in zip(results, expected_results, angles):
        #     if i != k:
        #         print(self.output_template_angles.format(i, k, j))
        self.assertTrue(results, expected_results)


    def test_c5_sym_no_mirror_theta_larger_90_degrees_phi_bigger_equals_72_should_return_False(self):
        angles = [[entry, thet, 0] for entry in range(72, 360) for thet in range(90, 180)]
        [angles.append([entry, 180, 0]) for entry in range(360)]
        results = []
        for phi, theta, psi in angles:
            if theta != 180:
                theta = theta + 0.1
            is_in = fu.symclass('c5').is_in_subunit(phi, theta, 0)
            results.append(is_in)
        expected_results = symclass_mod('c5').is_in_subunit(angles)
        # for i, k, j in zip(results, expected_results, angles):
        #     if i != k:
        #         print(self.output_template_angles.format(i, k, j))
        self.assertTrue(results, expected_results)

        angles = [[entry, thet, 0] for entry in range(72, 360) for thet in range(90, 180)]
        [angles.append([entry, 180, 0]) for entry in range(360)]
        results = []
        for phi, theta, psi in angles:
            if theta != 180:
                theta = theta + 0.1
            is_in = oldfu.symclass('c5').is_in_subunit(phi, theta, 0)
            results.append(is_in)
        expected_results = symclass_mod('c5').is_in_subunit(angles)
        # for i, k, j in zip(results, expected_results, angles):
        #     if i != k:
        #         print(self.output_template_angles.format(i, k, j))
        self.assertTrue(results, expected_results)

    def test_c5_sym_mirror_theta_smaller_equals_90_phi_smaller_72_degrees_should_return_True(self):
        angles = [[entry, thet, 0] for entry in range(72) for thet in range(91)]
        [angles.append([entry, 0, 0]) for entry in range(360)]
        results = []
        for phi, theta, psi in angles:
            is_in = fu.symclass('c5').is_in_subunit(phi, theta, 1)
            results.append(is_in)
        expected_results = symclass_mod('c5').is_in_subunit(angles)
        for i, k, j in zip(results, expected_results, angles):
            if i != k:
                print(self.output_template_angles.format(i, k, j))
        self.assertEqual(results, expected_results)

        angles = [[entry, thet, 0] for entry in range(72) for thet in range(91)]
        [angles.append([entry, 0, 0]) for entry in range(360)]
        results = []
        for phi, theta, psi in angles:
            is_in = oldfu.symclass('c5').is_in_subunit(phi, theta, 1)
            results.append(is_in)
        expected_results = symclass_mod('c5').is_in_subunit(angles)
        for i, k, j in zip(results, expected_results, angles):
            if i != k:
                print(self.output_template_angles.format(i, k, j))
        self.assertEqual(results, expected_results)


    def test_c5_sym_mirror_theta_larger_90_degrees_phi_smaller_72_should_return_False(self):
        angles = [[entry, thet, 0] for entry in range(72) for thet in range(90, 180)]
        [angles.append([entry, 180, 0]) for entry in range(360)]
        results = []
        for phi, theta, psi in angles:
            if theta != 180:
                theta = theta + 0.1
            is_in = fu.symclass('c5').is_in_subunit(phi, theta, 1)
            results.append(is_in)
        expected_results = symclass_mod('c5').is_in_subunit(angles)
        # for i, k, j in zip(results, expected_results, angles):
        #     if i != k:
        #         print(self.output_template_angles.format(i, k, j))
        self.assertTrue(results, expected_results)

        angles = [[entry, thet, 0] for entry in range(72) for thet in range(90, 180)]
        [angles.append([entry, 180, 0]) for entry in range(360)]
        results = []
        for phi, theta, psi in angles:
            if theta != 180:
                theta = theta + 0.1
            is_in = oldfu.symclass('c5').is_in_subunit(phi, theta, 1)
            results.append(is_in)
        expected_results = symclass_mod('c5').is_in_subunit(angles)
        # for i, k, j in zip(results, expected_results, angles):
        #     if i != k:
        #         print(self.output_template_angles.format(i, k, j))
        self.assertTrue(results, expected_results)


    def test_c5_sym_mirror_theta_smaller_equals_90_degrees_phi_bigger_equals_72_should_return_False(self):
        angles = [[entry, thet, 0] for entry in range(72, 360) for thet in range(1, 91)]
        results = []
        for phi, theta, psi in angles:
            is_in = fu.symclass('c5').is_in_subunit(phi, theta, 1)
            results.append(is_in)
        expected_results = symclass_mod('c5').is_in_subunit(angles)
        for i, k, j in zip(results, expected_results, angles):
            if i != k:
                print(self.output_template_angles.format(i, k, j))
        self.assertEqual(results, expected_results)


        angles = [[entry, thet, 0] for entry in range(72, 360) for thet in range(1, 91)]
        results = []
        for phi, theta, psi in angles:
            is_in = oldfu.symclass('c5').is_in_subunit(phi, theta, 1)
            results.append(is_in)
        expected_results = symclass_mod('c5').is_in_subunit(angles)
        for i, k, j in zip(results, expected_results, angles):
            if i != k:
                print(self.output_template_angles.format(i, k, j))
        self.assertEqual(results, expected_results)

    def test_c5_sym_mirror_theta_larger_90_degrees_phi_bigger_equals_72_should_return_False(self):
        angles = [[entry, thet, 0] for entry in range(72, 360) for thet in range(90, 180)]
        results = []
        for phi, theta, psi in angles:
            if theta != 180:
                theta = theta + 0.1
            is_in = fu.symclass('c5').is_in_subunit(phi, theta, 1)
            results.append(is_in)
        expected_results = symclass_mod('c5').is_in_subunit(angles)
        # for i, k, j in zip(results, expected_results, angles):
        #     if i != k:
        #         print(self.output_template_angles.format(i, k, j))
        self.assertTrue(results, expected_results)

        angles = [[entry, thet, 0] for entry in range(72, 360) for thet in range(90, 180)]
        results = []
        for phi, theta, psi in angles:
            if theta != 180:
                theta = theta + 0.1
            is_in = oldfu.symclass('c5').is_in_subunit(phi, theta, 1)
            results.append(is_in)
        expected_results = symclass_mod('c5').is_in_subunit(angles)
        # for i, k, j in zip(results, expected_results, angles):
        #     if i != k:
        #         print(self.output_template_angles.format(i, k, j))
        self.assertTrue(results, expected_results)



class TestSymClassIsInSubunit(unittest.TestCase):



    output_template_angles = 'Got: {0} ; Expected: {1} ; Angle: {2}'

    def test_wrong_sym_crashes_problem(self):
        return_new = fu.symclass('c1')
        return_old = oldfu.symclass('c1')
        return_new.sym = 'foobar'
        return_old.sym = 'foobar'
        self.assertIsNone(return_new.is_in_subunit(0, 0))
        self.assertIsNone(return_old.is_in_subunit(0, 0))

    # -------------------------------------------------------[ Test Cases for C symmetry ]
    def test_newmethod_c1_sym_mirror_theta_smaller_equals_90_degrees_should_return_True(self):
        angles = [[entry, thet, 0] for entry in range(360) for thet in range(91)]
        # [angles.append([entry, 0, 0]) for entry in range(360)]
        results = []
        for phi, theta, psi in angles:
            is_in = fu.symclass('c1').is_in_subunit(phi, theta, inc_mirror=1)
            results.append(is_in)
        expected_results = symclass_mod('c1').is_in_subunit(angles,inc_mirror=1)
        for i, k, j in zip(results, expected_results, angles):
            if i != k:
                print(self.output_template_angles.format(i, k, j))
        self.assertEqual(results,expected_results)

        angles = [[entry, thet, 0] for entry in range(360) for thet in range(91)]
        # [angles.append([entry, 0, 0]) for entry in range(360)]
        results = []
        for phi, theta, psi in angles:
            is_in = oldfu.symclass('c1').is_in_subunit(phi, theta, inc_mirror=1)
            results.append(is_in)
        expected_results = symclass_mod('c1').is_in_subunit(angles,inc_mirror=1)
        for i, k, j in zip(results, expected_results, angles):
            if i != k:
                print(self.output_template_angles.format(i, k, j))
        self.assertEqual(results,expected_results)


    def test_newmethod_c1_sym_no_mirror_theta_smaller_equals_90_degrees_should_return_True(self):
        angles = [[entry, thet, 0] for entry in range(360) for thet in range(91)]
        # [angles.append([entry, 0, 0]) for entry in range(360)]
        results = []
        for phi, theta, psi in angles:
            is_in = fu.symclass('c1').is_in_subunit(phi, theta, inc_mirror=0)
            results.append(is_in)
        expected_results = symclass_mod('c1').is_in_subunit(angles,inc_mirror=0)
        for i, k, j in zip(results, expected_results, angles):
            if i != k:
                print(self.output_template_angles.format(i, k, j))
        self.assertEqual(results,expected_results)

        angles = [[entry, thet, 0] for entry in range(360) for thet in range(91)]
        # [angles.append([entry, 0, 0]) for entry in range(360)]
        results = []
        for phi, theta, psi in angles:
            is_in = oldfu.symclass('c1').is_in_subunit(phi, theta,inc_mirror=0)
            results.append(is_in)
        expected_results = symclass_mod('c1').is_in_subunit(angles, inc_mirror=0)
        for i, k, j in zip(results, expected_results, angles):
            if i != k:
                print(self.output_template_angles.format(i, k, j))
        self.assertEqual(results,expected_results)


    def test_newmethod_c4_sym_mirror_theta_smaller_equals_90_degrees_should_return_True(self):
        angles = [[entry, thet, 0] for entry in range(360) for thet in range(91)]
        # [angles.append([entry, 0, 0]) for entry in range(360)]
        results = []
        for phi, theta, psi in angles:
            is_in = fu.symclass('c4').is_in_subunit(phi, theta, inc_mirror=1)
            results.append(is_in)
        expected_results = symclass_mod('c4').is_in_subunit(angles,inc_mirror=1)
        for i, k, j in zip(results, expected_results, angles):
            if i != k:
                print(self.output_template_angles.format(i, k, j))
        self.assertEqual(results,expected_results)

        angles = [[entry, thet, 0] for entry in range(360) for thet in range(91)]
        # [angles.append([entry, 0, 0]) for entry in range(360)]
        results = []
        for phi, theta, psi in angles:
            is_in = oldfu.symclass('c4').is_in_subunit(phi, theta, inc_mirror=1)
            results.append(is_in)
        expected_results = symclass_mod('c4').is_in_subunit(angles, inc_mirror=1)
        for i, k, j in zip(results, expected_results, angles):
            if i != k:
                print(self.output_template_angles.format(i, k, j))
        self.assertEqual(results,expected_results)


    def test_newmethod_c4_sym_no_mirror_theta_smaller_equals_90_degrees_should_return_True(self):
        angles = [[entry, thet, 0] for entry in range(360) for thet in range(91)]
        # [angles.append([entry, 0, 0]) for entry in range(360)]
        results = []
        for phi, theta, psi in angles:
            is_in = fu.symclass('c4').is_in_subunit(phi, theta, inc_mirror=0)
            results.append(is_in)
        expected_results = symclass_mod('c4').is_in_subunit(angles, inc_mirror=0)
        for i, k, j in zip(results, expected_results, angles):
            if i != k:
                print(self.output_template_angles.format(i, k, j))
        self.assertEqual(results,expected_results)

        angles = [[entry, thet, 0] for entry in range(360) for thet in range(91)]
        # [angles.append([entry, 0, 0]) for entry in range(360)]
        results = []
        for phi, theta, psi in angles:
            is_in = oldfu.symclass('c4').is_in_subunit(phi, theta, inc_mirror=0)
            results.append(is_in)
        expected_results = symclass_mod('c4').is_in_subunit(angles, inc_mirror=0)
        for i, k, j in zip(results, expected_results, angles):
            if i != k:
                print(self.output_template_angles.format(i, k, j))
        self.assertEqual(results,expected_results)


    def test_newmethod_c1_sym_no_mirror_theta_larger_90_degrees_should_return_False(self):
        angles = [[entry, thet, 0] for entry in range(360) for thet in range(90, 180)]
        # [angles.append([entry, 180, 0]) for entry in range(360)]
        print (fu.symclass('c1').brackets)
        results = []
        for phi, theta, psi in angles:
            if theta != 180:
                theta = theta + 0.1
            is_in = fu.symclass('c1').is_in_subunit(phi, theta, 0)
            results.append(is_in)
        expected_results = symclass_mod('c1').is_in_subunit(angles,inc_mirror=0)
        self.assertTrue(results, expected_results)

        angles = [[entry, thet, 0] for entry in range(360) for thet in range(90, 180)]
        # [angles.append([entry, 180, 0]) for entry in range(360)]
        results = []
        for phi, theta, psi in angles:
            if theta != 180:
                theta = theta + 0.1
            is_in = oldfu.symclass('c1').is_in_subunit(phi, theta, 0)
            results.append(is_in)
        expected_results = symclass_mod('c1').is_in_subunit(angles,inc_mirror=0)
        self.assertTrue(results, expected_results)


    def test_newmethod_c1_sym_mirror_theta_larger_90_degrees_should_return_False(self):
        angles = [[entry, thet, 0] for entry in range(360) for thet in range(90, 180)]
        # [angles.append([entry, 180, 0]) for entry in range(360)]
        print (fu.symclass('c1').brackets)
        results = []
        for phi, theta, psi in angles:
            if theta != 180:
                theta = theta + 0.1
            is_in = fu.symclass('c1').is_in_subunit(phi, theta, 1)
            results.append(is_in)
        expected_results = symclass_mod('c1').is_in_subunit(angles,inc_mirror=1)
        self.assertTrue(results, expected_results)

        angles = [[entry, thet, 0] for entry in range(360) for thet in range(90, 180)]
        # [angles.append([entry, 180, 0]) for entry in range(360)]
        results = []
        for phi, theta, psi in angles:
            if theta != 180:
                theta = theta + 0.1
            is_in = oldfu.symclass('c1').is_in_subunit(phi, theta, 1)
            results.append(is_in)
        expected_results = symclass_mod('c1').is_in_subunit(angles,inc_mirror=1)
        self.assertTrue(results, expected_results)


    def test_newmethod_c5_sym_mirror_theta_smaller_equals_90_degrees_phi_bigger_equals_72_should_return_False(self):
        angles = [[entry, thet, 0] for entry in range(72, 360) for thet in range(1, 91)]
        results = []
        for phi, theta, psi in angles:
            is_in = fu.symclass('c5').is_in_subunit(phi, theta, 1)
            results.append(is_in)
        expected_results = symclass_mod('c5').is_in_subunit(angles,inc_mirror=1)
        for i, k, j in zip(results, expected_results, angles):
            if i != k:
                print(self.output_template_angles.format(i, k, j))
        self.assertEqual(results, expected_results)


        angles = [[entry, thet, 0] for entry in range(72, 360) for thet in range(1, 91)]
        results = []
        for phi, theta, psi in angles:
            is_in = oldfu.symclass('c5').is_in_subunit(phi, theta, 1)
            results.append(is_in)
        expected_results = symclass_mod('c5').is_in_subunit(angles,inc_mirror=1)
        for i, k, j in zip(results, expected_results, angles):
            if i != k:
                print(self.output_template_angles.format(i, k, j))
        self.assertEqual(results, expected_results)

    # -------------------------------------------------------[ Test Cases for D symmetry ]
    def test_newmethod_d1_sym_mirror_theta_smaller_equals_90_degrees_should_return_True(self):
        angles = [[entry, thet, 0] for entry in range(360) for thet in range(91)]
        # [angles.append([entry, 0, 0]) for entry in range(360)]
        print(fu.symclass('d1').brackets)
        print(fu.symclass('d1').nsym)

        results = []
        for phi, theta, psi in angles:
            is_in = fu.symclass('d1').is_in_subunit(phi, theta, inc_mirror=1)
            results.append(is_in)
        expected_results = symclass_mod('d1').is_in_subunit(angles,inc_mirror=1)
        for i, k, j in zip(results, expected_results, angles):
            if i != k:
                print(self.output_template_angles.format(i, k, j))
        self.assertEqual(results, expected_results)

        angles = [[entry, thet, 0] for entry in range(360) for thet in range(91)]
        # [angles.append([entry, 0, 0]) for entry in range(360)]
        results = []
        for phi, theta, psi in angles:
            is_in = oldfu.symclass('d1').is_in_subunit(phi, theta, inc_mirror=1)
            results.append(is_in)
        expected_results = symclass_mod('d1').is_in_subunit(angles, inc_mirror=1)
        for i, k, j in zip(results, expected_results, angles):
            if i != k:
                print(self.output_template_angles.format(i, k, j))
        self.assertEqual(results, expected_results)


    def test_newmethod_d1_sym_no_mirror_theta_smaller_equals_90_degrees_should_return_True(self):
        angles = [[entry, thet, 0] for entry in range(360) for thet in range(91)]
        # [angles.append([entry, 0, 0]) for entry in range(360)]
        print(fu.symclass('d1').brackets)
        print(fu.symclass('d1').nsym)

        results = []
        for phi, theta, psi in angles:
            is_in = fu.symclass('d1').is_in_subunit(phi, theta, inc_mirror=0)
            results.append(is_in)
        expected_results = symclass_mod('d1').is_in_subunit(angles,inc_mirror=0)
        for i, k, j in zip(results, expected_results, angles):
            if i != k:
                print(self.output_template_angles.format(i, k, j))
        self.assertEqual(results, expected_results)

        angles = [[entry, thet, 0] for entry in range(360) for thet in range(91)]
        # [angles.append([entry, 0, 0]) for entry in range(360)]
        results = []
        for phi, theta, psi in angles:
            is_in = oldfu.symclass('d1').is_in_subunit(phi, theta, inc_mirror=0)
            results.append(is_in)
        expected_results = symclass_mod('d1').is_in_subunit(angles,inc_mirror=0)
        for i, k, j in zip(results, expected_results, angles):
            if i != k:
                print(self.output_template_angles.format(i, k, j))
        self.assertEqual(results, expected_results)


    def test_newmethod_d5_sym_mirror_theta_smaller_equals_90_degrees_should_return_True(self):
        angles = [[entry, thet, 0] for entry in range(360) for thet in range(91)]
        # [angles.append([entry, 0, 0]) for entry in range(360)]
        print(fu.symclass('d5').brackets)
        print(fu.symclass('d5').nsym)

        results = []
        for phi, theta, psi in angles:
            is_in = fu.symclass('d5').is_in_subunit(phi, theta, inc_mirror=1)
            results.append(is_in)
        expected_results = symclass_mod('d5').is_in_subunit(angles,inc_mirror=1)
        for i, k, j in zip(results, expected_results, angles):
            if i != k:
                print(self.output_template_angles.format(i, k, j))
        self.assertEqual(results, expected_results)

        angles = [[entry, thet, 0] for entry in range(360) for thet in range(91)]
        # [angles.append([entry, 0, 0]) for entry in range(360)]
        results = []
        for phi, theta, psi in angles:
            is_in = oldfu.symclass('d5').is_in_subunit(phi, theta, inc_mirror=1)
            results.append(is_in)
        expected_results = symclass_mod('d5').is_in_subunit(angles,inc_mirror=1)
        for i, k, j in zip(results, expected_results, angles):
            if i != k:
                print(self.output_template_angles.format(i, k, j))
        self.assertEqual(results, expected_results)


    def test_newmethod_d5_sym_no_mirror_theta_smaller_equals_90_degrees_should_return_True(self):
        angles = [[entry, thet, 0] for entry in range(360) for thet in range(91)]
        # [angles.append([entry, 0, 0]) for entry in range(360)]
        print(fu.symclass('d5').brackets)
        print(fu.symclass('d5').nsym)

        results = []
        for phi, theta, psi in angles:
            is_in = fu.symclass('d5').is_in_subunit(phi, theta, inc_mirror=0)
            results.append(is_in)
        expected_results = symclass_mod('d5').is_in_subunit(angles,inc_mirror=0)
        for i, k, j in zip(results, expected_results, angles):
            if i != k:
                print(self.output_template_angles.format(i, k, j))
        self.assertEqual(results, expected_results)

        angles = [[entry, thet, 0] for entry in range(360) for thet in range(91)]
        # [angles.append([entry, 0, 0]) for entry in range(360)]
        results = []
        for phi, theta, psi in angles:
            is_in = oldfu.symclass('d5').is_in_subunit(phi, theta, inc_mirror=0)
            results.append(is_in)
        expected_results = symclass_mod('d5').is_in_subunit(angles,inc_mirror=0)
        for i, k, j in zip(results, expected_results, angles):
            if i != k:
                print(self.output_template_angles.format(i, k, j))
        self.assertEqual(results, expected_results)



    # -------------------------------------------------------[ Test Cases for tet symmetry ]
    def test_newmethod_tet_sym_mirror_theta_smaller_equals_90_degrees_should_return_True(self):
        angles = [[entry, thet, 0.0] for entry in range(120) for thet in range(55)]
        # [angles.append([entry, 0, 0]) for entry in range(360)]
        print(fu.symclass('tet').brackets)
        print(fu.symclass('tet').nsym)

        results = []
        for phi, theta, psi in angles:
            is_in = fu.symclass('tet').is_in_subunit(phi, theta, inc_mirror=1)
            results.append(is_in)
        expected_results = symclass_mod('tet').is_in_subunit(angles,inc_mirror=1)
        for i, k, j in zip(results, expected_results, angles):
            if i != k:
                print(self.output_template_angles.format(i, k, j))
        self.assertEqual(results, expected_results)

        angles = [[entry, thet, 0] for entry in range(120) for thet in range(54)]
        # [angles.append([entry, 0, 0]) for entry in range(360)]
        results = []
        for phi, theta, psi in angles:
            is_in = oldfu.symclass('tet').is_in_subunit(phi, theta, inc_mirror=1)
            results.append(is_in)
        expected_results = symclass_mod('tet').is_in_subunit(angles,inc_mirror=1)
        for i, k, j in zip(results, expected_results, angles):
            if i != k:
                print(self.output_template_angles.format(i, k, j))
        self.assertEqual(results, expected_results)



    def test_newmethod_tet_sym_no_mirror_theta_smaller_equals_90_degrees_should_return_True(self):
        angles = [[entry, thet, 0.0] for entry in range(120) for thet in range(55)]
        # [angles.append([entry, 0, 0]) for entry in range(360)]
        print(fu.symclass('tet').brackets)
        print(fu.symclass('tet').nsym)

        results = []
        for phi, theta, psi in angles:
            is_in = fu.symclass('tet').is_in_subunit(phi, theta, inc_mirror=0)
            results.append(is_in)
        expected_results = symclass_mod('tet').is_in_subunit(angles,inc_mirror=0)
        for i, k, j in zip(results, expected_results, angles):
            if i != k:
                print(self.output_template_angles.format(i, k, j))
        self.assertEqual(results, expected_results)

        angles = [[entry, thet, 0] for entry in range(120) for thet in range(54)]
        # [angles.append([entry, 0, 0]) for entry in range(360)]
        results = []
        for phi, theta, psi in angles:
            is_in = oldfu.symclass('tet').is_in_subunit(phi, theta, inc_mirror=0)
            results.append(is_in)
        expected_results = symclass_mod('tet').is_in_subunit(angles,inc_mirror=0)
        for i, k, j in zip(results, expected_results, angles):
            if i != k:
                print(self.output_template_angles.format(i, k, j))
        self.assertEqual(results, expected_results)


    # -------------------------------------------------------[ Test Cases for Oct symmetry ]
    def test_newmethod_oct_sym_mirror_theta_smaller_equals_90_degrees_should_return_True(self):
        angles = [[entry, thet, 0.0] for entry in range(120) for thet in range(55)]
        # [angles.append([entry, 0, 0]) for entry in range(360)]
        print(fu.symclass('oct').brackets)
        print(fu.symclass('oct').nsym)

        results = []
        for phi, theta, psi in angles:
            is_in = fu.symclass('oct').is_in_subunit(phi, theta, inc_mirror=1)
            results.append(is_in)
        expected_results = symclass_mod('oct').is_in_subunit(angles,inc_mirror=1)
        for i, k, j in zip(results, expected_results, angles):
            if i != k:
                print(self.output_template_angles.format(i, k, j))
        self.assertEqual(results, expected_results)

        angles = [[entry, thet, 0] for entry in range(120) for thet in range(54)]
        # [angles.append([entry, 0, 0]) for entry in range(360)]
        results = []
        for phi, theta, psi in angles:
            is_in = oldfu.symclass('oct').is_in_subunit(phi, theta, inc_mirror=1)
            results.append(is_in)
        expected_results = symclass_mod('oct').is_in_subunit(angles,inc_mirror=1)
        for i, k, j in zip(results, expected_results, angles):
            if i != k:
                print(self.output_template_angles.format(i, k, j))
        self.assertEqual(results, expected_results)


    def test_newmethod_oct_sym_no_mirror_theta_smaller_equals_90_degrees_should_return_True(self):
        angles = [[entry, thet, 0.0] for entry in range(120) for thet in range(55)]
        # [angles.append([entry, 0, 0]) for entry in range(360)]
        print(fu.symclass('oct').brackets)
        print(fu.symclass('oct').nsym)

        results = []
        for phi, theta, psi in angles:
            is_in = fu.symclass('oct').is_in_subunit(phi, theta, inc_mirror=0)
            results.append(is_in)
        expected_results = symclass_mod('oct').is_in_subunit(angles,inc_mirror=0)
        for i, k, j in zip(results, expected_results, angles):
            if i != k:
                print(self.output_template_angles.format(i, k, j))
        self.assertEqual(results, expected_results)

        angles = [[entry, thet, 0] for entry in range(120) for thet in range(54)]
        # [angles.append([entry, 0, 0]) for entry in range(360)]
        results = []
        for phi, theta, psi in angles:
            is_in = oldfu.symclass('oct').is_in_subunit(phi, theta, inc_mirror=0)
            results.append(is_in)
        expected_results = symclass_mod('oct').is_in_subunit(angles,inc_mirror=0)
        for i, k, j in zip(results, expected_results, angles):
            if i != k:
                print(self.output_template_angles.format(i, k, j))
        self.assertEqual(results, expected_results)


    # -------------------------------------------------------[ Test Cases for Icos symmetry ]
    def test_newmethod_icos_sym_mirror_theta_smaller_equals_90_degrees_should_return_True(self):
        angles = [[entry, thet, 0.0] for entry in range(120) for thet in range(55)]
        # [angles.append([entry, 0, 0]) for entry in range(360)]
        print(fu.symclass('icos').brackets)
        print(fu.symclass('icos').nsym)

        results = []
        for phi, theta, psi in angles:
            is_in = fu.symclass('icos').is_in_subunit(phi, theta, inc_mirror=1)
            results.append(is_in)
        expected_results = symclass_mod('icos').is_in_subunit(angles,inc_mirror=1)
        for i, k, j in zip(results, expected_results, angles):
            if i != k:
                print(self.output_template_angles.format(i, k, j))
        self.assertEqual(results, expected_results)

        angles = [[entry, thet, 0] for entry in range(120) for thet in range(54)]
        # [angles.append([entry, 0, 0]) for entry in range(360)]
        results = []
        for phi, theta, psi in angles:
            is_in = oldfu.symclass('icos').is_in_subunit(phi, theta, inc_mirror=1)
            results.append(is_in)
        expected_results = symclass_mod('icos').is_in_subunit(angles,inc_mirror=1)
        for i, k, j in zip(results, expected_results, angles):
            if i != k:
                print(self.output_template_angles.format(i, k, j))
        self.assertEqual(results, expected_results)


    def test_newmethod_icos_sym_no_mirror_theta_smaller_equals_90_degrees_should_return_True(self):
        angles = [[entry, thet, 0.0] for entry in range(120) for thet in range(55)]
        # [angles.append([entry, 0, 0]) for entry in range(360)]
        print(fu.symclass('icos').brackets)
        print(fu.symclass('icos').nsym)

        results = []
        for phi, theta, psi in angles:
            is_in = fu.symclass('icos').is_in_subunit(phi, theta, inc_mirror=0)
            results.append(is_in)
        expected_results = symclass_mod('icos').is_in_subunit(angles,inc_mirror=0)
        for i, k, j in zip(results, expected_results, angles):
            if i != k:
                print(self.output_template_angles.format(i, k, j))
        self.assertEqual(results, expected_results)

        angles = [[entry, thet, 0] for entry in range(120) for thet in range(54)]
        # [angles.append([entry, 0, 0]) for entry in range(360)]
        results = []
        for phi, theta, psi in angles:
            is_in = oldfu.symclass('icos').is_in_subunit(phi, theta, inc_mirror=0)
            results.append(is_in)
        expected_results = symclass_mod('icos').is_in_subunit(angles,inc_mirror=0)
        for i, k, j in zip(results, expected_results, angles):
            if i != k:
                print(self.output_template_angles.format(i, k, j))
        self.assertEqual(results, expected_results)



class TestSymClassSymmetryRelatedC(unittest.TestCase):

    def test_c1_sym_zero(self):
        new_result = fu.symclass('c1').symmetry_related([0, 0, 0])
        old_result = oldfu.symclass('c1').symmetry_related([0, 0, 0])

        self.assertEqual(new_result,[[0, 0, 0]])
        self.assertEqual(old_result, [[0, 0, 0]])

    def test_c5_sym_zero(self):
        new_result = fu.symclass('c5').symmetry_related([0, 0, 0])
        old_result = oldfu.symclass('c5').symmetry_related([0, 0, 0])
        self.assertEqual(new_result, [[0, 0, 0]])
        self.assertEqual(old_result, [[0, 0, 0]])


    def test_c6_sym_zero(self):
        new_result = fu.symclass('c6').symmetry_related([0, 0, 0])
        old_result = oldfu.symclass('c6').symmetry_related([0, 0, 0])
        self.assertEqual(new_result,[[0, 0, 0]])
        self.assertEqual(old_result, [[0, 0, 0]])


    def test_c1_sym_180(self):
        new_result = fu.symclass('c1').symmetry_related([0, 180, 0])
        old_result = oldfu.symclass('c1').symmetry_related([0, 180, 0])
        self.assertEqual(new_result, [[0, 180, 0]])
        self.assertEqual(old_result, [[0, 180, 0]])

    def test_c5_sym_180(self):
        new_result = fu.symclass('c5').symmetry_related([0, 180, 0])
        old_result = oldfu.symclass('c5').symmetry_related([0, 180, 0])
        self.assertEqual(new_result, [[0, 180, 0]])
        self.assertEqual(old_result, [[0, 180, 0]])

    def test_c6_sym_180(self):
        new_result = fu.symclass('c6').symmetry_related([0, 180, 0])
        old_result = oldfu.symclass('c6').symmetry_related([0, 180, 0])
        self.assertEqual(new_result,[[0, 180, 0]])
        self.assertEqual(old_result, [[0, 180, 0]])


    def test_c1_sym_90(self):
        new_result = fu.symclass('c1').symmetry_related([0, 90, 0])
        old_result = oldfu.symclass('c1').symmetry_related([0, 90, 0])
        self.assertEqual(new_result, [[0, 90, 0]])
        self.assertEqual(old_result, [[0, 90, 0]])

    def test_c5_sym_90(self):
        new_result = fu.symclass('c5').symmetry_related([0, 90, 0])
        old_result = oldfu.symclass('c5').symmetry_related([0, 90, 0])
        self.assertEqual(new_result, [[0, 90, 0], [72, 90, 0], [72*2, 90, 0], [72*3, 90, 0], [72*4, 90, 0]]   )
        self.assertEqual(old_result, [[0, 90, 0], [72, 90, 0], [72*2, 90, 0], [72*3, 90, 0], [72*4, 90, 0]]   )

    def test_c6_sym_90(self):
        new_result = fu.symclass('c6').symmetry_related([0, 90, 0])
        old_result = oldfu.symclass('c6').symmetry_related([0, 90, 0])
        self.assertEqual(new_result, [[0, 90, 0], [60, 90, 0], [60*2, 90, 0], [60*3, 90, 0], [60*4, 90, 0] , [60*5, 90, 0] ]   )
        self.assertEqual(old_result, [[0, 90, 0], [60, 90, 0], [60*2, 90, 0], [60*3, 90, 0], [60*4, 90, 0] , [60*5, 90, 0] ]   )


    def test_c1_sym_42(self):
        new_result = fu.symclass('c1').symmetry_related([0, 42, 0])
        old_result = oldfu.symclass('c1').symmetry_related([0, 42, 0])
        self.assertEqual(new_result, [[0, 42, 0]])
        self.assertEqual(old_result, [[0, 42, 0]])

    def test_c5_sym_42(self):
        new_result = symclass_mod('c5').symmetry_related([0, 42, 0])
        old_result = oldfu.symclass('c5').symmetry_related([0, 42, 0])
        self.assertEqual(new_result, [[0, 42, 0], [72, 42, 0], [72*2, 42, 0], [72*3, 42, 0], [72*4, 42, 0]]   )
        self.assertEqual(old_result, [[0, 42, 0], [72, 42, 0], [72*2, 42, 0], [72*3, 42, 0], [72*4, 42, 0]]   )

    def test_c6_sym_42(self):
        new_result = fu.symclass('c6').symmetry_related([0, 42, 0])
        old_result = oldfu.symclass('c6').symmetry_related([0, 42, 0])
        self.assertEqual(new_result, [[0, 42, 0], [60, 42, 0], [60*2, 42, 0], [60*3, 42, 0], [60*4, 42, 0] , [60*5, 42, 0] ]   )
        self.assertEqual(old_result, [[0, 42, 0], [60, 42, 0], [60*2, 42, 0], [60*3, 42, 0], [60*4, 42, 0] , [60*5, 42, 0] ]   )




class TestSymClassSymmetryRelatedD(unittest.TestCase):

    def test_d1_sym_zero(self):
        new_result = fu.symclass('d1').symmetry_related([0, 0, 0])
        old_result = oldfu.symclass('d1').symmetry_related([0, 0, 0])
        self.assertEqual(new_result, [[0, 0, 0], [0, 180, 180]] )
        # self.assertEqual(old_result, [[0, 0, 0], [0, 180, 0]] )


    def test_d5_sym_zero(self):
        new_result = fu.symclass('d5').symmetry_related([0, 0, 0])
        old_result = oldfu.symclass('d5').symmetry_related([0, 0, 0])
        self.assertEqual(new_result, [[0, 0, 0], [0, 180, 180]] )
        self.assertEqual(old_result, [[0, 0, 0], [0, 180, 180]] )


    def test_d6_sym_zero(self):
        new_result = fu.symclass('d6').symmetry_related([0, 0, 0])
        old_result = oldfu.symclass('d6').symmetry_related([0, 0, 0])
        self.assertEqual(new_result, [[0, 0, 0], [0, 180, 180]] )
        # self.assertEqual(old_result, [[0, 0, 0], [0, 180, 180]] )


    def test_d1_sym_90(self):
        new_result = fu.symclass('d1').symmetry_related([0, 90, 0])
        old_result = oldfu.symclass('d1').symmetry_related([0, 90, 0])
        self.assertEqual(new_result, [[0, 90, 0]])
        self.assertEqual(old_result, [[0, 90, 0]])

    def test_d5_sym_90(self):
        new_result = fu.symclass('d5').symmetry_related([0, 90, 0])
        old_result = oldfu.symclass('d5').symmetry_related([0, 90, 0])

        self.assertEqual(new_result, [[0, 90, 0], [72, 90, 0], [72*2, 90, 0], [72*3, 90, 0], [72*4, 90, 0]]   )
        self.assertEqual(old_result, [[0, 90, 0], [72, 90, 0], [72*2, 90, 0], [72*3, 90, 0], [72*4, 90, 0]]   )

    def test_d6_sym_90(self):
        new_result = fu.symclass('d6').symmetry_related([0, 90, 0])
        old_result = oldfu.symclass('d6').symmetry_related([0, 90, 0])
        self.assertEqual(new_result, [[0, 90, 0], [60, 90, 0], [60*2, 90, 0], [60*3, 90, 0], [60*4, 90, 0] , [60*5, 90, 0] ]   )
        self.assertEqual(old_result, [[0, 90, 0], [60, 90, 0], [60*2, 90, 0], [60*3, 90, 0], [60*4, 90, 0] , [60*5, 90, 0] ]   )

    """ Old version is wrong, have corrected it  """
    def test_d1_sym_180(self):
        new_result = fu.symclass('d1').symmetry_related([0, 180, 0])
        old_result = oldfu.symclass('d1').symmetry_related([0, 180, 0])
        self.assertEqual(new_result, [[0, 180, 0], [0, 0, 180]] )
        # self.assertEqual(old_result, [[0, 180, 0], [0,  180, 180]] )

    """ Old version is wrong, have corrected it  """
    def test_d5_sym_180(self):
        new_result = fu.symclass('d5').symmetry_related([0, 180, 0])
        old_result = oldfu.symclass('d5').symmetry_related([0, 180, 0])
        self.assertEqual(new_result, [[0, 180, 0], [0, 0, 180]] )
        # self.assertEqual(old_result, [[0, 180, 0], [0, 180, 180]] )


    def test_d6_sym_180(self):
        new_result = fu.symclass('d6').symmetry_related([0, 180, 0])
        old_result = oldfu.symclass('d6').symmetry_related([0, 180, 0])
        self.assertEqual(new_result, [[0, 180, 0], [0, 0, 180]] )
        # self.assertEqual(old_result, [[0, 0, 0], [0, 180, 0]] )

    def test_d1_sym_42(self):
        new_result = fu.symclass('d1').symmetry_related([0, 42, 0])
        old_result = oldfu.symclass('d1').symmetry_related([0, 42, 0])
        self.assertEqual(new_result, [[0, 42, 0], [0.0, 138.0, 180.0]])
        self.assertEqual(old_result, [[0, 42, 0], [0.0, 138.0, 180.0]])

    def test_d5_sym_42(self):
        new_result = fu.symclass('d5').symmetry_related([0, 42, 0])
        old_result = oldfu.symclass('d5').symmetry_related([0, 42, 0])
        self.assertEqual(new_result, [[0, 42, 0], [72.0, 42, 0],  [144.0, 42, 0],  [216.0, 42, 0],  [288.0, 42, 0], [0.0, 138.0, 180.0], \
                                      [288.0, 138.0, 180.0],  [216.0, 138.0, 180.0],  [144.0, 138.0, 180.0], [72.0, 138.0, 180.0]] )
        self.assertEqual(old_result, [[0, 42, 0], [72.0, 42, 0],  [144.0, 42, 0],  [216.0, 42, 0],  [288.0, 42, 0], [0.0, 138.0, 180.0], \
                                      [288.0, 138.0, 180.0],  [216.0, 138.0, 180.0],  [144.0, 138.0, 180.0], [72.0, 138.0, 180.0]] )
    def test_d6_sym_42(self):
        new_result = fu.symclass('d6').symmetry_related([0, 42, 0])
        old_result = oldfu.symclass('d6').symmetry_related([0, 42, 0])
        self.assertEqual(new_result, [[0, 42, 0], [60.0, 42, 0],  [120.0, 42, 0],  [180.0, 42, 0],  [240.0, 42, 0],  [300.0, 42, 0], \
                                     [0.0, 138.0, 0.0], [300.0, 138.0, 0.0],  [240.0, 138.0, 0.0],  [180.0, 138.0, 0.0],  [120.0, 138.0, 0.0], [60.0, 138.0, 0.0]]  )
        self.assertEqual(old_result, [[0, 42, 0], [60.0, 42, 0],  [120.0, 42, 0],  [180.0, 42, 0],  [240.0, 42, 0],  [300.0, 42, 0], \
                                     [0.0, 138.0, 0.0], [300.0, 138.0, 0.0],  [240.0, 138.0, 0.0],  [180.0, 138.0, 0.0],  [120.0, 138.0, 0.0], [60.0, 138.0, 0.0]]  )




class TestSymClassSymmetryNeighbors(unittest.TestCase):

    def test_c_sym_c4_neighbors(self):
        angles = [[idx1, idx2, 0] for idx1 in range(360) for idx2 in range(180)]
        expected_return_values = symclass_mod('c4').symmetry_neighbors(angles)
        return_values = fu.symclass('c4').symmetry_neighbors(angles)
        self.assertEqual(return_values,expected_return_values)

        angles = [[idx1, idx2, 0] for idx1 in range(360) for idx2 in range(180)]
        expected_return_values = symclass_mod('c4').symmetry_neighbors(angles)
        return_values = oldfu.symclass('c4').symmetry_neighbors(angles)
        self.assertEqual(return_values, expected_return_values)


    def test_c_sym_c5_neighbors(self):
        angles = [[idx1, idx2, 0] for idx1 in range(360) for idx2 in range(180)]
        expected_return_values = symclass_mod('c5').symmetry_neighbors(angles)
        return_values = fu.symclass('c5').symmetry_neighbors(angles)
        self.assertEqual(return_values,expected_return_values)

        angles = [[idx1, idx2, 0] for idx1 in range(360) for idx2 in range(180)]
        expected_return_values = symclass_mod('c5').symmetry_neighbors(angles)
        return_values = oldfu.symclass('c5').symmetry_neighbors(angles)
        self.assertEqual(return_values, expected_return_values)

    def test_d_sym_d4_neighbors(self):
        angles = [[idx1, idx2, 0] for idx1 in range(360) for idx2 in range(180)]
        expected_return_values = symclass_mod('d4').symmetry_neighbors(angles)
        return_values = fu.symclass('d4').symmetry_neighbors(angles)
        self.assertEqual(return_values, expected_return_values)

        angles = [[idx1, idx2, 0] for idx1 in range(360) for idx2 in range(180)]
        expected_return_values = symclass_mod('d4').symmetry_neighbors(angles)
        return_values = oldfu.symclass('d4').symmetry_neighbors(angles)
        self.assertEqual(return_values, expected_return_values)

    def test_d_sym_d5_neighbors(self):
        angles = [[idx1, idx2, 0] for idx1 in range(360) for idx2 in range(180)]
        expected_return_values = symclass_mod('d5').symmetry_neighbors(angles)
        return_values = fu.symclass('d5').symmetry_neighbors(angles)
        self.assertEqual(return_values, expected_return_values)

        angles = [[idx1, idx2, 0] for idx1 in range(360) for idx2 in range(180)]
        expected_return_values = symclass_mod('d5').symmetry_neighbors(angles)
        return_values = oldfu.symclass('d5').symmetry_neighbors(angles)
        self.assertEqual(return_values, expected_return_values)


    def test_tet_sym_neighbors(self):
        # angles = [[idx1, idx2, idx3] for idx1 in range(180) for idx2 in range(180) for idx3 in range(120) ]
        angles = [[0, 0 ,0], [0,180,0], [0,90,0], [90,0,0], [90,90,0], [90,180,0] ]
        expected_return_values = symclass_mod('tet').symmetry_neighbors(angles)
        return_values = fu.symclass('tet').symmetry_neighbors(angles)

        self.assertTrue(numpy.isclose(numpy.array(return_values), numpy.array(expected_return_values), atol = 1  ).all())

        # # angles = [[idx1, idx2, 0] for idx1 in range(360) for idx2 in range(180)]
        # angles = [[0, 0, 0], [0, 180, 0], [0, 90, 0], [90, 0, 0], [90, 90, 0], [90, 180, 0]]
        # expected_return_values = symclass_mod('tet').symmetry_neighbors(angles)
        # return_values = oldfu.symclass('tet').symmetry_neighbors(angles)
        # self.assertTrue(numpy.isclose(numpy.array(return_values), numpy.array(expected_return_values), atol = 1  ).all())


    def test_oct_sym_neighbors(self):
        # angles = [[idx1, idx2, 0] for idx1 in range(50) for idx2 in range(90)]
        angles = [[0, 0, 0], [0, 180, 0], [0, 90, 0], [90, 0, 0], [90, 90, 0], [90, 180, 0]]
        expected_return_values = symclass_mod('oct').symmetry_neighbors(angles)
        return_values = fu.symclass('oct').symmetry_neighbors(angles)
        self.assertTrue(numpy.isclose(numpy.array(return_values), numpy.array(expected_return_values), atol = 1  ).all())



        # angles = [[idx1, idx2, 0] for idx1 in range(50) for idx2 in range(90)]
        # expected_return_values = symclass_mod('oct').symmetry_neighbors(angles)
        # return_values = oldfu.symclass('oct').symmetry_neighbors(angles)
        # self.assertTrue(numpy.isclose(numpy.array(return_values), numpy.array(expected_return_values), atol = 1  ).all())
        #

    def test_icos_sym_neighbors(self):
        # angles = [[idx1, idx2, 0] for idx1 in range(50) for idx2 in range(90)]
        angles = [[0, 0, 0], [0, 180, 0], [0, 90, 0], [90, 0, 0], [90, 90, 0], [90, 180, 0]]
        expected_return_values = symclass_mod('icos').symmetry_neighbors(angles)
        return_values = fu.symclass('icos').symmetry_neighbors(angles)
        self.assertTrue(numpy.isclose(numpy.array(return_values), numpy.array(expected_return_values), atol = 1  ).all())

        # angles = [[idx1, idx2, 0] for idx1 in range(50) for idx2 in range(90)]
        # expected_return_values = symclass_mod('icos').symmetry_neighbors(angles)
        # return_values = oldfu.symclass('icos').symmetry_neighbors(angles)
        # self.assertTrue(numpy.isclose(numpy.array(return_values), numpy.array(expected_return_values), atol = 1  ).all())





class TestSymClassSymmetryRelated(unittest.TestCase):

    def test_c_sym_related_c4(self):
        angles = [[idx1, idx2, 0] for idx1 in range(90) for idx2 in range(90)]
        # angles = [[0, 0, 0], [0, 180, 0], [90, 45, 29]]

        expected_return_values = symclass_mod('c4').symmetry_related(angles)
        return_values = []
        for angle in angles:
            return_values.extend(fu.symclass('c4').symmetry_related(angle))

        print(angles)
        print(expected_return_values)

        self.assertEqual(return_values,expected_return_values)

        angles = [[idx1, idx2, 0] for idx1 in range(90) for idx2 in range(90)]

        expected_return_values = symclass_mod('c4').symmetry_related(angles)
        return_values = []
        for angle in angles:
            return_values.extend(oldfu.symclass('c4').symmetry_related(angle))

        self.assertEqual(return_values, expected_return_values)

    def test_c_sym_related_c5(self):
        angles = [[idx1, idx2, 0] for idx1 in range(90) for idx2 in range(90)]
        # angles = [[0, 0, 0], [0, 180, 0], [90, 45, 29]]

        expected_return_values = symclass_mod('c5').symmetry_related(angles)
        return_values = []
        for angle in angles:
            return_values.extend(fu.symclass('c5').symmetry_related(angle))

        print(angles)
        print(expected_return_values)

        self.assertEqual(return_values, expected_return_values)

        angles = [[idx1, idx2, 0] for idx1 in range(90) for idx2 in range(90)]

        expected_return_values = symclass_mod('c5').symmetry_related(angles)
        return_values = []
        for angle in angles:
            return_values.extend(oldfu.symclass('c5').symmetry_related(angle))

        self.assertEqual(return_values, expected_return_values)


    def test_d_sym_related_d4(self):
        angles = [[idx1, idx2, 0] for idx1 in range(90) for idx2 in range(90)]
        # angles = [[90, 0, 0], [90, 90, 29], [90, 45, 29]]
        # angles = [[idx1, idx2, 0] for idx1 in range(360) for idx2 in range(180)]

        import time; start=time.time()
        expected_return_values = symclass_mod('d4').symmetry_related(angles)
        print(time.time()-start); start=time.time()
        return_values = []
        for angle in angles:
            return_values.extend(fu.symclass('d4').symmetry_related(angle))
        print(time.time() - start);
        start = time.time()
        self.assertEqual(sorted(return_values),sorted(expected_return_values))

        angles = [[idx1, idx2, 0] for idx1 in range(360) for idx2 in range(180)]
        expected_return_values = symclass_mod('d4').symmetry_related(angles)
        return_values = []
        for angle in angles:
            return_values.extend(oldfu.symclass('d4').symmetry_related(angle))

        self.assertEqual(sorted(return_values),sorted(expected_return_values))

    def test_d_sym_related_d5(self):
        angles = [[idx1, idx2, 0] for idx1 in range(90) for idx2 in range(90)]
        # angles = [[90, 0, 0], [90, 90, 29], [90, 45, 29]]
        # angles = [[idx1, idx2, 0] for idx1 in range(360) for idx2 in range(180)]

        import time; start=time.time()
        expected_return_values = symclass_mod('d5').symmetry_related(angles)
        print(time.time()-start); start=time.time()
        return_values = []
        for angle in angles:
            return_values.extend(fu.symclass('d5').symmetry_related(angle))
        print(time.time() - start);
        start = time.time()
        self.assertEqual(sorted(return_values),sorted(expected_return_values))

        angles = [[idx1, idx2, 0] for idx1 in range(360) for idx2 in range(180)]
        expected_return_values = symclass_mod('d5').symmetry_related(angles)
        return_values = []
        for angle in angles:
            return_values.extend(oldfu.symclass('d5').symmetry_related(angle))

        self.assertEqual(sorted(return_values),sorted(expected_return_values))


    def test_tet_sym_related(self):
        # angles = [[idx1, idx2,idx3] for idx1 in range(90) for idx2 in range(90) for idx3 in range(5)]
        # angles = [[0, 0, 0], [0, 180, 0], [0, 90, 0], [90, 0, 0], [90, 90, 0], [90, 180, 0]]
        angles = [[idx1, idx2, 0] for idx1 in range(360) for idx2 in range(180)]
        # angles = [[90, 0, 0], [90, 90, 29], [90, 45, 29]]

        import time; start=time.time()
        expected_return_values = symclass_mod('tet').symmetry_related(angles)
        print(time.time()-start); start=time.time()
        return_values = []
        for angle in angles:
            return_values.extend(fu.symclass('tet').symmetry_related(angle))
        print(time.time() - start);
        start = time.time()

        # for i in range (len(return_values)):
        #      if ~(numpy.isclose(numpy.array(return_values)[i], numpy.array(expected_return_values)[i] , atol = 1 ).all())   :
        #          print(i)
        #          print(angles[i//symclass_mod('tet').nsym])
        #          print(expected_return_values[i])
        #          print(return_values[i])

        self.assertTrue(numpy.isclose(numpy.array(return_values), numpy.array(expected_return_values), atol=1).all())


        angles = [[idx1, idx2, 0] for idx1 in range(360) for idx2 in range(180)]
        # angles = [[90, 0, 0], [90, 90, 29], [90, 45, 29]]
        expected_return_values = symclass_mod('tet').symmetry_related(angles)
        return_values = []
        for angle in angles:
            return_values.extend(oldfu.symclass('tet').symmetry_related(angle))

        self.assertTrue(numpy.isclose(numpy.array(return_values), numpy.array(expected_return_values), atol=1).all())

    def test_oct_sym_related(self):
        angles = [[idx1, idx2, 0] for idx1 in range(360) for idx2 in range(180)]
        # angles = [[90, 0, 0], [90, 90, 29], [90, 45, 29]]

        import time; start=time.time()
        expected_return_values = symclass_mod('oct').symmetry_related(angles)
        print(time.time()-start); start=time.time()
        return_values = []
        for angle in angles:
            return_values.extend(fu.symclass('oct').symmetry_related(angle))
        print(time.time() - start);
        start = time.time()

        self.assertTrue(numpy.isclose(numpy.array(return_values), numpy.array(expected_return_values), atol=1).all())

        angles = [[idx1, idx2, 0] for idx1 in range(360) for idx2 in range(180)]
        # angles = [[90, 0, 0], [90, 90, 29], [90, 45, 29]]
        expected_return_values = symclass_mod('oct').symmetry_related(angles)
        return_values = []
        for angle in angles:
            return_values.extend(oldfu.symclass('oct').symmetry_related(angle))

        self.assertTrue(numpy.isclose(numpy.array(return_values), numpy.array(expected_return_values), atol=1).all())

    def test_icos_sym_related(self):
        angles = [[idx1, idx2, 0] for idx1 in range(360) for idx2 in range(180)]
        # angles = [[90, 0, 0], [90, 90, 29], [90, 45, 29]]

        import time; start=time.time()
        expected_return_values = symclass_mod('icos').symmetry_related(angles)
        print(time.time()-start); start=time.time()
        return_values = []
        for angle in angles:
            return_values.extend(fu.symclass('icos').symmetry_related(angle))
        print(time.time() - start);
        start = time.time()

        self.assertTrue(numpy.isclose(numpy.array(return_values), numpy.array(expected_return_values), atol=1).all())

        angles = [[idx1, idx2, 0] for idx1 in range(360) for idx2 in range(180)]
        # angles = [[90, 0, 0], [90, 90, 29], [90, 45, 29]]
        expected_return_values = symclass_mod('icos').symmetry_related(angles)
        return_values = []
        for angle in angles:
            return_values.extend(oldfu.symclass('icos').symmetry_related(angle))

        self.assertTrue(numpy.isclose(numpy.array(return_values), numpy.array(expected_return_values), atol=1).all())


class TestSymClassReduceAngleSets(unittest.TestCase):

    def test_reduce_anglesets_c1_should_return_equal_object(self):
        angles = [[phi, theta, 0] for phi in range(360) for theta in range(91)]

        oldangles = oldfu.symclass('c1').reduce_anglesets(angles)
        newangles = fu.symclass('c1').reduce_anglesets(angles)
        self.assertEqual(oldangles, newangles)

    def test_reduce_anglesets_c4_should_return_equal_object(self):
        angles = [[phi, theta, 0] for phi in range(360) for theta in range(91)]

        oldangles = oldfu.symclass('c4').reduce_anglesets(angles)
        newangles = fu.symclass('c4').reduce_anglesets(angles)
        self.assertEqual(oldangles, newangles)

    def test_reduce_anglesets_c5_should_return_equal_object(self):
        angles = [[phi, theta, 0] for phi in range(360) for theta in range(91)]

        oldangles = oldfu.symclass('c5').reduce_anglesets(angles)
        newangles = fu.symclass('c5').reduce_anglesets(angles)
        self.assertEqual(oldangles, newangles)

    def test_reduce_anglesets_d1_should_return_equal_object(self):
        angles = [[phi, theta, 0] for phi in range(360) for theta in range(91)]

        oldangles = oldfu.symclass('d1').reduce_anglesets(angles)
        newangles = fu.symclass('d1').reduce_anglesets(angles)
        self.assertEqual(oldangles, newangles)

    def test_reduce_anglesets_d4_should_return_equal_object(self):
        angles = [[phi, theta, 0] for phi in range(360) for theta in range(91)]

        oldangles = oldfu.symclass('d4').reduce_anglesets(angles)
        newangles = fu.symclass('d4').reduce_anglesets(angles)
        self.assertEqual(oldangles, newangles)

    def test_reduce_anglesets_d5_should_return_equal_object(self):
        angles = [[phi, theta, 0] for phi in range(360) for theta in range(91)]

        oldangles = oldfu.symclass('d5').reduce_anglesets(angles)
        newangles = fu.symclass('d5').reduce_anglesets(angles)
        self.assertEqual(oldangles, newangles)

    def test_reduce_anglesets_tet_should_return_equal_object(self):
        angles = [[phi, theta, 0] for phi in range(360) for theta in range(180)]

        oldangles = oldfu.symclass('tet').reduce_anglesets(angles, inc_mirror= 0)
        newangles = fu.symclass('tet').reduce_anglesets(angles, inc_mirror= 0)
        self.assertEqual(oldangles, newangles)

    def test_reduce_anglesets_icos_should_return_equal_object(self):
        angles = [[phi, theta, 0] for phi in range(50) for theta in range(91)]

        oldangles = oldfu.symclass('icos').reduce_anglesets(angles)
        newangles = fu.symclass('icos').reduce_anglesets(angles)
        self.assertEqual(oldangles, newangles)

    def test_reduce_anglesets_oct_should_return_equal_object(self):
        angles = [[phi, theta, 0] for phi in range(360) for theta in range(91)]

        oldangles = oldfu.symclass('oct').reduce_anglesets(angles, inc_mirror= 0)
        newangles = fu.symclass('oct').reduce_anglesets(angles, inc_mirror=0)
        self.assertEqual(oldangles, newangles)


    def test_reduce_anglesets_new_oct_sym_mirror_theta_smaller_equals_90_degrees_should_return_True(self):
        # angles = [[entry, thet, psi]for entry in range(120) for thet in range(0,55) for psi in range(20)   ]

        # angles = [[idx1, idx2, 0] for idx1 in range(120) for idx2 in range(55)]
        # angles = [[120.0, 54.735610317245346, 65]]

        # angles = [[0,0,0], [0,180,0], [120.0, 54.735610317245346, 65], [60.0, 180-54.735610317245346, 65]]
        angles = [[60.0, 70.528779365509308, 0]]
        # angles = [[0, 0, 0], [0, 180, 0], [0, 90, 0], [90, 0, 0], [90, 90, 0], [90, 180, 0]]

        # angles = [ [ 45,45,20]  , [ 25,90,45] ,[ 90,25,45] ]

        # results = fu.symclass('oct').reduce_anglesets(angles, inc_mirror=1)
        # print("first phase done ")
        # expected_results = symclass_mod('oct').reduce_anglesets(angles, inc_mirror=1)
        # print("Hello")
        results = []
        for ang in angles:
            results.extend(fu.symclass('oct').symmetry_related(ang))
        print("first phase done ")
        expected_results = symclass_mod('oct').symmetry_related(angles, return_mirror=-1)
        print("Hello")
        print(symclass_mod('oct').brackets)
        print(numpy.array(expected_results))


        # print(expected_results)
        # print(results)
        #
        # print(numpy.array(angles).shape)
        # print(numpy.array(results).shape)
        # print(numpy.array(expected_results).shape)
        #
        # for i in range(len(results)) :
        #     print(results[i])
        #     print(expected_results[i])
        #     print(" ")

        # with open('angles.txt', 'w') as write:
        #     for entry in angles:
        #         write.write('\t'.join([str(e) for e in entry]))
        #         write.write('\n')
        # with open('results.txt', 'w') as write:
        #     for entry in results:
        #         write.write('\t'.join([str(e) for e in entry]))
        #         write.write('\n')
        # with open('expected_results.txt', 'w') as write:
        #     for entry in expected_results:
        #         write.write('\t'.join([str(e) for e in entry]))
        #         write.write('\n')
        # import os
        # os.system('rm -r angles_dir')
        # os.system('sxpipe.py angular_distribution angles.txt angles_dir --delta 0.5 --sym=c1_full')
        # os.system('rm -r results_dir')
        # os.system('sxpipe.py angular_distribution results.txt results_dir --delta=0.5 --sym=c1_full')
        # os.system('rm -r expected_results_dir')
        # os.system('sxpipe.py angular_distribution expected_results.txt expected_results_dir --delta=0.5 --sym=c1_full')
        #




        # for i in range (len(return_values)):
        #      if ~(numpy.isclose(numpy.array(return_values)[i], numpy.array(expected_return_values)[i] , atol = 1 ).all())   :
        #          print(i)
        #          print(angles[i//symclass_mod('tet').nsym])
        #          print(expected_return_values[i])
        #          print(return_values[i])


        # for i in range(  len( results) ) :
        #     if results[i]  != expected_results[i] :
        #         print(i , angles[i], results[i] ,expected_results[i]  )


        self.assertTrue(numpy.isclose(numpy.array(results), numpy.array(expected_results), atol=1).all())

        print("knock knock")

        # angles = [[entry, thet, psi] for entry in range(120) for thet in range(1,55)  for psi in range(34)]
        # # [angles.append([entry, 0, 0]) for entry in range(360)]
        # results = []
        # for phi, theta, psi in angles:
        #     is_in = oldfu.symclass('oct').reduce_anglesets(angles, inc_mirror=1)
        #     results.append(is_in)
        # expected_results = self.reduce_anglesets_new(angles, oldfu.symclass('oct').sym, oldfu.symclass('oct').nsym,
        #                                           oldfu.symclass('oct').brackets, oldfu.symclass('oct').symatrix, inc_mirror=1)
        # self.assertTrue(numpy.array_equal(numpy.array(results), numpy.array(expected_results)))






class TestSymClassReduceAngles(unittest.TestCase):
    def test_reduce_angles_c1_should_return_equal_object(self):

        oldangles = oldfu.symclass('c1').reduce_angles(60,80,0)
        newangles = fu.symclass('c1').reduce_angles(60,80,0)
        self.assertEqual(oldangles, newangles)


if __name__ == '__main__':
    unittest.main()




""" Individual functions modified by adnan"""


    # def my_new_method(self, angles, sym, symatrix):
    #     if (sym[0] == "c" or sym[0] == "d"):
    #         temp = e2cpp.Util.symmetry_neighbors(angles, sym)
    #         nt = old_div(len(temp), 3)
    #         mod_angles = numpy.array([[0, 0, 0]]).repeat(nt, axis=0)
    #         mod_angles[:, 0] = temp[0:len(temp):3]
    #         mod_angles[:, 1] = temp[1:len(temp):3]
    #         mod_angles[:, 2] = 0.0
    #         return mod_angles.tolist()
    #
    #     #  Note symmetry neighbors below refer to the particular order
    #     #   in which this class generates symmetry matrices
    #     neighbors = {}
    #     neighbors["oct"] = [0, 1, 2, 3, 8, 9, 12, 13]
    #     neighbors["tet"] = [0, 1, 2, 3, 4, 6, 7]
    #     neighbors["icos"] = [0, 1, 2, 3, 4, 6, 7, 11, 12]
    #     sang_new = numpy.array(angles, numpy.float64).repeat(len(neighbors[sym]), axis=0)
    #     matrices = self.rotmatrix_new(sang_new)
    #     matrices_mod = numpy.sum(
    #         numpy.transpose(matrices, (0, 2, 1)).reshape(
    #             matrices.shape[0],
    #             matrices.shape[1],
    #             matrices.shape[2],
    #             1
    #         ) *
    #         numpy.tile(
    #             numpy.array(symatrix, numpy.float64)[neighbors[sym]],
    #             (sang_new.shape[0] // len(neighbors[sym]), 1, 1)).reshape(
    #             matrices.shape[0], matrices.shape[1], 1, matrices.shape[2], ), -3)
    #     sang_mod = self.recmat(matrices_mod, sang_new)
    #     return sang_mod.tolist()
    #
    #
    # @staticmethod
    # def recmat(mat, sang_new):
    #     pass  # IMPORTIMPORTIMPORT from math import acos,asin,atan2,degrees,pi
    #
    #     def sign(x):
    #         return_array = numpy.sign(x)
    #         return_array[return_array == 0] = 1
    #         return return_array
    #
    #     mask_2_2_1 = mat[:, 2, 2] == 1.0
    #     mask_0_0_0 = mat[:, 0, 0] == 0.0
    #     mask_2_2_m1 = mat[:, 2, 2] == -1.0
    #     mask_2_0_0 = mat[:, 2, 0] == 0.0
    #     mask_0_2_0 = mat[:, 0, 2] == 0.0
    #     mask_2_2_0 = mat[:, 2, 2] == 0.0
    #     theta_2_2 = numpy.arccos(mat[:, 2, 2])
    #     st = sign(theta_2_2)
    #
    #     sang_new[mask_2_2_1 & mask_0_0_0, 0] = numpy.degrees(numpy.arcsin(mat[mask_2_2_1 & mask_0_0_0, 0, 1]))
    #     sang_new[mask_2_2_1 & ~mask_0_0_0, 0] = numpy.degrees(numpy.arctan2(
    #         mat[mask_2_2_1 & ~mask_0_0_0, 0, 1],
    #         mat[mask_2_2_1 & ~mask_0_0_0, 0, 0]
    #     ))
    #     sang_new[mask_2_2_1 & ~mask_2_2_m1, 1] = numpy.degrees(0.0)  # theta
    #     sang_new[mask_2_2_1 & ~mask_2_2_m1, 2] = numpy.degrees(0.0)  # psi
    #     sang_new[mask_2_2_m1 & mask_0_0_0, 0] = numpy.degrees(numpy.arcsin(-mat[mask_2_2_m1 & mask_0_0_0, 0, 1]))
    #     sang_new[mask_2_2_m1 & ~mask_0_0_0, 0] = numpy.degrees(numpy.arctan2(
    #         -mat[mask_2_2_m1 & ~mask_0_0_0, 0, 1],
    #         -mat[mask_2_2_m1 & ~mask_0_0_0, 0, 0]
    #     ))
    #     sang_new[mask_2_2_m1 & ~mask_2_2_1, 1] = numpy.degrees(numpy.pi)
    #     sang_new[mask_2_2_m1 & ~mask_2_2_1, 2] = numpy.degrees(0.0)
    #     sang_new[~mask_2_2_1 & ~mask_2_2_m1 & mask_2_0_0 & (st != sign(mat[:, 2, 1])), 0] = numpy.degrees(1.5 * numpy.pi)
    #     sang_new[~mask_2_2_1 & ~mask_2_2_m1 & mask_2_0_0 & (st == sign(mat[:, 2, 1])), 0] = numpy.degrees(0.5 * numpy.pi)
    #     sang_new[~mask_2_2_1 & ~mask_2_2_m1 & ~mask_2_2_0, 0] = numpy.degrees(numpy.arctan2(
    #         st[~mask_2_2_1 & ~mask_2_2_m1 & ~mask_2_2_0] * mat[~mask_2_2_1 & ~mask_2_2_m1 & ~mask_2_2_0, 2, 1],
    #         st[~mask_2_2_1 & ~mask_2_2_m1 & ~mask_2_2_0] * mat[~mask_2_2_1 & ~mask_2_2_m1 & ~mask_2_2_0, 2, 0]
    #     ))
    #     sang_new[~mask_2_2_1 & ~mask_2_2_m1, 1] = numpy.degrees(theta_2_2[~mask_2_2_1 & ~mask_2_2_m1])
    #     sang_new[~mask_2_2_1 & ~mask_2_2_m1, 2] = numpy.degrees(0.0)
    #
    #     return sang_new % 360.0
    #
    #
    # @staticmethod
    # def rotmatrix_new(angles):
    #     newmat = numpy.zeros((len(angles), 3, 3), dtype=numpy.float64)
    #     index = numpy.arange(len(angles))
    #
    #     cosphi = numpy.cos(numpy.radians(numpy.array(angles)[:, 0]))
    #     costheta = numpy.cos(numpy.radians(numpy.array(angles)[:, 1]))
    #     cospsi = numpy.cos(numpy.radians(numpy.array(angles)[:, 2]))
    #
    #     sinphi = numpy.sin(numpy.radians(numpy.array(angles)[:, 0]))
    #     sintheta = numpy.sin(numpy.radians(numpy.array(angles)[:, 1]))
    #     sinpsi = numpy.sin(numpy.radians(numpy.array(angles)[:, 2]))
    #
    #     newmat[:, 0, 0] = cospsi[index] * costheta[index] * cosphi[index] - sinpsi[index] * sinphi[index]
    #     newmat[:, 1, 0] = -sinpsi[index] * costheta[index] * cosphi[index] - cospsi[index] * sinphi[index]
    #     newmat[:, 2, 0] = sintheta[index] * cosphi[index]
    #     newmat[:, 0, 1] = cospsi[index] * costheta[index] * sinphi[index] + sinpsi[index] * cosphi[index]
    #     newmat[:, 1, 1] = -sinpsi[index] * costheta[index] * sinphi[index] + cospsi[index] * cosphi[index]
    #     newmat[:, 2, 1] = sintheta[index] * sinphi[index]
    #     newmat[:, 0, 2] = -cospsi[index] * sintheta[index]
    #     newmat[:, 1, 2] = sinpsi[index] * sintheta[index]
    #     newmat[:, 2, 2] = costheta[index]
    #
    #     return newmat

    # def is_in_subunit_new(self, angles, sym, nsym, brackets, inc_mirror =1 ):
    #
    #     if (type(angles[0]) is list):
    #         print("lis of list")
    #     else:
    #         print("single list")
    #
    #     """
    #     Input:  Before it was a projection direction specified by (phi, theta).
    #             Now it is a projection direction specified by phi(i) , theta(i) for all angles
    #             inc_mirror = 1 consider mirror directions as unique
    #             inc_mirror = 0 consider mirror directions as outside of unique range.
    #     Output: True if input projection direction is in the first asymmetric subunit,
    #             False otherwise.
    #     """
    #     pass  # IMPORTIMPORTIMPORT from math import degrees, radians, sin, cos, tan, atan, acos, sqrt
    #
    #     angles = numpy.array(angles)
    #     condstat = numpy.zeros(numpy.shape(angles)[0], dtype = bool  )
    #
    #     phi = angles[:,0]
    #     phi_0 =   phi >= 0.0
    #     phi_ld_br_inmirr_0 = phi < brackets[inc_mirror][0]
    #     phi_ld_br_1_0 = phi < brackets[1][0]
    #     theta = angles[:, 1]
    #     theta_ldeq_br_incmirr_1 = theta  <= brackets[inc_mirror][1]
    #     theta_ldeq_br_incmirr_3 = theta <= brackets[inc_mirror][3]
    #     theta_180 =  (numpy.logical_and(theta ==180 , inc_mirror))
    #     theta_0 = theta == 0
    #
    #
    #     if sym[0] == "c" :
    #         condstat[phi_0  & phi_ld_br_inmirr_0 & theta_ldeq_br_incmirr_1] = True
    #         condstat[theta_180] = True
    #         condstat[theta_0] = True
    #         condstat[~phi_0 & ~phi_ld_br_inmirr_0 & ~theta_ldeq_br_incmirr_1 & ~theta_180  & ~theta_0] = False
    #
    #         return condstat.tolist()
    #
    #     elif sym[0] == "d" and (old_div(nsym, 2)) % 2 == 0:
    #         condstat[phi_0 & phi_ld_br_inmirr_0 & theta_ldeq_br_incmirr_1] = True
    #         condstat[theta_0] = True
    #         condstat[~phi_0 & ~phi_ld_br_inmirr_0 & ~theta_ldeq_br_incmirr_1 & ~theta_0] = False
    #
    #         return condstat.tolist()
    #
    #     elif sym[0] == "d" and (old_div(nsym, 2)) % 2 == 1:
    #         phib = old_div(360.0, nsym)
    #         condstat[numpy.logical_and( (theta_ldeq_br_incmirr_1 & phi_0 & phi_ld_br_1_0), inc_mirror)] = True
    #         condstat[ theta_ldeq_br_incmirr_1 & phi_0 & phi_ld_br_1_0 & \
    #             ( numpy.logical_or(numpy.logical_and(  (phi >= old_div(phib, 2)) , (phi < phib)) , numpy.logical_and( (phi >= phib), (phi <= phib + old_div(phib, 2)) ))) ] = True
    #         condstat[theta_ldeq_br_incmirr_1 & phi_0 & phi_ld_br_1_0 & theta_0] = True
    #         condstat[~(theta_ldeq_br_incmirr_1 & phi_0 & phi_ld_br_1_0) & theta_0] = True
    #         condstat[~(theta_ldeq_br_incmirr_1 & phi_0 & phi_ld_br_1_0) &  ~theta_0] = False
    #
    #         return condstat.tolist()
    #
    #     elif ((sym[:3] == "oct") or (sym[:4] == "icos")):
    #         tmphi = numpy.minimum(phi, brackets[inc_mirror][2] - phi)
    #         baldwin_lower_alt_bound = \
    #             old_div(
    #                 (old_div(numpy.sin(numpy.radians(old_div(brackets[inc_mirror][2], 2.0) - tmphi)), numpy.tan(
    #                     numpy.radians(brackets[inc_mirror][1])))
    #                  + old_div(numpy.sin(numpy.radians(tmphi)),
    #                            numpy.tan(numpy.radians(brackets[inc_mirror][3])))),
    #                 numpy.sin(numpy.radians(old_div(brackets[inc_mirror][2], 2.0))))
    #         baldwin_lower_alt_bound = numpy.degrees(numpy.arctan(old_div(1.0, baldwin_lower_alt_bound)))
    #
    #         condstat[ phi_0 & phi_ld_br_inmirr_0  & theta_ldeq_br_incmirr_3 & (baldwin_lower_alt_bound > theta)] = True
    #         condstat[phi_0 & phi_ld_br_inmirr_0 & theta_ldeq_br_incmirr_3 & ~(baldwin_lower_alt_bound > theta)] = False
    #         condstat[theta_0] = True
    #         condstat[~phi_0 & ~phi_ld_br_inmirr_0 & ~theta_ldeq_br_incmirr_3 & ~theta_0 ] = False
    #
    #         return condstat.tolist()
    #
    #     elif (sym[:3] == "tet"):
    #         tmphi = numpy.minimum(phi, brackets[inc_mirror][2] - phi)
    #         baldwin_lower_alt_bound_1 = \
    #             old_div(
    #                 (old_div(numpy.sin(numpy.radians(old_div(brackets[inc_mirror][2], 2.0) - tmphi)),
    #                          numpy.tan(numpy.radians(brackets[inc_mirror][1]))) \
    #                  + old_div(numpy.sin(numpy.radians(tmphi)), numpy.tan(numpy.radians(brackets[inc_mirror][3])))) \
    #                 , numpy.sin(numpy.radians(old_div(brackets[inc_mirror][2], 2.0))))
    #         baldwin_lower_alt_bound_1 = numpy.degrees(numpy.arctan(old_div(1.0, baldwin_lower_alt_bound_1)))
    #
    #         baldwin_upper_alt_bound_2 = \
    #             old_div(
    #                 (old_div((numpy.sin(numpy.radians(old_div(brackets[inc_mirror][2], 2.0) - tmphi))),
    #                          (numpy.tan(numpy.radians(brackets[inc_mirror][1]))))
    #                  + old_div((numpy.sin(numpy.radians(tmphi))),
    #                            numpy.tan(numpy.radians(old_div(brackets[inc_mirror][3], 2.0))))) \
    #                 , (numpy.sin(numpy.radians(old_div(brackets[inc_mirror][2], 2.0)))))
    #         baldwin_upper_alt_bound_2 = numpy.degrees(numpy.arctan(old_div(1.0, baldwin_upper_alt_bound_2)))
    #
    #         condstat[phi_0 & phi_ld_br_inmirr_0 & theta_ldeq_br_incmirr_3  & \
    #                   numpy.logical_and((baldwin_lower_alt_bound_1 > theta) , inc_mirror)] = True
    #         condstat[phi_0 & phi_ld_br_inmirr_0 & theta_ldeq_br_incmirr_3  & \
    #                  ~numpy.logical_and((baldwin_lower_alt_bound_1 > theta) , inc_mirror) & (baldwin_upper_alt_bound_2 < theta)] = False
    #         condstat[ phi_0 & phi_ld_br_inmirr_0 & theta_ldeq_br_incmirr_3  & \
    #                  ~numpy.logical_and((baldwin_lower_alt_bound_1 > theta) , inc_mirror) & ~(baldwin_upper_alt_bound_2 < theta)] = True
    #         condstat[phi_0 & phi_ld_br_inmirr_0 & theta_ldeq_br_incmirr_3  & ~(baldwin_lower_alt_bound_1 > theta) &  theta_0] = True
    #         condstat[phi_0 & phi_ld_br_inmirr_0 & theta_ldeq_br_incmirr_3  & ~(baldwin_lower_alt_bound_1 > theta) &  ~theta_0] = False
    #         condstat[~phi_0 & ~phi_ld_br_inmirr_0 & ~theta_ldeq_br_incmirr_3 &  theta_0] = True
    #         condstat[~phi_0 & ~phi_ld_br_inmirr_0 & ~theta_ldeq_br_incmirr_3 & ~theta_0] = False
    #
    #         return condstat.tolist()
    #     else:
    #         global_def.ERROR("unknown symmetry", "symclass: is_in_subunit", 1)









""" Pawel Code for unit test.  None of them are in working state but i kept them in case we may want to check something from them  """

# #!/usr/bin/env python
# from __future__ import print_function
# from __future__ import division
# from past.utils import old_div

# # imports from Pawel function
# from EMAN2 import EMData, display
# from utilities import model_blank
# from math import sqrt
# # from global_def import Util
# from builtins import range
# import unittest
# from optparse import OptionParser
# from fundamentals import cyclic_shift, mirror
# from utilities import model_circle
# from fundamentals import ccf, acf, acfn, acfnp, acfnpl, acfp, acfpl, ccfn, ccfnp, ccfnpl, ccfp, ccfpl
# from EMAN2 import Log

# #
# # Author: Piotr Pawliczek, 10/25/2012
# # Copyright (c) 2000-2006 The University of Texas - Houston Medical School
# #
# # This software is issued under a joint BSD/GNU license. You may use the
# # source code in this file under either license. However, note that the
# # complete EMAN2 and SPARX software packages have some GPL dependencies,
# # so you are responsible for compliance with the licenses of these packages
# # if you opt to use BSD licensing. The warranty disclaimer below holds
# # in either instance.
# #
# # This complete copyright notice must be included in any revised version of the
# # source code. Additional authorship citations may be added, but existing
# # author citations must be preserved.
# #
# # This program is free software; you can redistribute it and/or modify
# # it under the terms of the GNU General Public License as published by
# # the Free Software Foundation; either version 2 of the License, or
# # (at your option) any later version.
# #
# # This program is distributed in the hope that it will be useful,
# # but WITHOUT ANY WARRANTY; without even the implied warranty of
# # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# # GNU General Public License for more details.
# #
# # You should have received a copy of the GNU General Public License
# # along with this program; if not, write to the Free Software
# # Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  2111-1307 USA
# #
# #
#
#
# IS_TEST_EXCEPTION = False
#
# # ====================================================================================================================
# class TestCorrelationFunctions(unittest.TestCase):
# 	"""this is unit test for [acs]cf*(...) from fundamentals.py"""
# 	def internal_correlation(self, A, B, center, circulant, normalized, lag_normalization): # A, B - images, circulant - bool (False - zero padded), center - bool, normalized - bool
# 		anx = A.get_xsize()
# 		any = A.get_ysize()
# 		anz = A.get_zsize()
# 		self.assertEqual(anx, B.get_xsize())
# 		self.assertEqual(any, B.get_ysize())
# 		self.assertEqual(anz, B.get_zsize())
# 		snx = 2*anx
# 		sny = 2*any
# 		snz = 2*anz
# 		if normalized:
# 			A = A.copy()
# 			B = B.copy()
# 			A.sub(A.get_attr("mean"))
# 			B.sub(B.get_attr("mean"))
# 			A.div(A.get_attr("sigma") * sqrt(anz) * sqrt(any) * sqrt(anx))
# 			B.div(B.get_attr("sigma") * sqrt(anz) * sqrt(any) * sqrt(anx))
# 		S = model_blank(snx, sny, snz)
# 		if circulant:
# 			tx = snx
# 			ty = sny
# 			tz = snz
# 		else:
# 			tx = anx
# 			ty = any
# 			tz = anz
# 		for x in range(tx):
# 			for y in range(ty):
# 				for z in range(tz):
# 					S.set_value_at(x, y, z, A.get_value_at( (x)%anx, (y)%any, (z)%anz ))
# 		if center:
# 			S = cyclic_shift(S, anx/2, any/2, anz/2)
# 		R = model_blank(anx, any, anz)
# 		for x in range(anx):
# 			for y in range(any):
# 				for z in range(anz):
# 					s = 0.0
# 					for x2 in range(anx):
# 						for y2 in range(any):
# 							for z2 in range(anz):
# 								s += S.get_value_at(x+x2, y+y2, z+z2) * B.get_value_at(x2, y2, z2)
# 					R.set_value_at(x, y, z, s)
# 		if lag_normalization:
# 			cx = anx/2
# 			cy = any/2
# 			cz = anz/2
# 			for x in range(anx):
# 				x_center = abs(x-cx)
# 				x_lag = 1 + (x_center * 1.0) / (anx - x_center)
# 				for y in range(any):
# 					y_center = abs(y-cy)
# 					y_lag = 1 + (y_center * 1.0) / (any - y_center)
# 					for z in range(anz):
# 						z_center = abs(z-cz)
# 						z_lag = 1 + (z_center * 1.0) / (anz - z_center)
# 						R.set_value_at(x, y, z, R.get_value_at(x,y,z) * x_lag * y_lag * z_lag )
# 		return R
#
# 	def internal_assert_almostEquals(self, A, B):
# 		nx = A.get_xsize()
# 		ny = A.get_ysize()
# 		nz = A.get_zsize()
# 		self.assertEqual(nx, B.get_xsize())
# 		self.assertEqual(ny, B.get_ysize())
# 		self.assertEqual(nz, B.get_zsize())
# 		for x in range(nx):
# 			for y in range(ny):
# 				for z in range(nz):
# 					delta = abs(A.get_value_at(x,y,z)) / 100.0 # allowed error: 1% of value
# 					if delta < 0.001:
# 						delta = 0.001
# 					self.assertAlmostEqual(A.get_value_at(x,y,z), B.get_value_at(x,y,z), delta=delta)
#
# 	def internal_check_ccf_center(self, A, B, AB_circ, AB_circ_norm, AB_zero, AB_zero_norm, AB_lag, AB_lag_norm, center):
# 		R_circ          = self.internal_correlation(A, B, center, True , False, False)
# 		R_zero          = self.internal_correlation(A, B, center, False, False, False)
# 		R_circ_norm     = self.internal_correlation(A, B, center, True , True , False)
# 		R_zero_norm     = self.internal_correlation(A, B, center, False, True , False)
# 		R_zero_lag      = self.internal_correlation(A, B, center, False, False, True )
# 		R_zero_lag_norm = self.internal_correlation(A, B, center, False, True , True )
#
# 		self.internal_assert_almostEquals( R_circ         , AB_circ      )
# 		self.internal_assert_almostEquals( R_zero         , AB_zero      )
# 		self.internal_assert_almostEquals( R_circ_norm    , AB_circ_norm )
# 		self.internal_assert_almostEquals( R_zero_norm    , AB_zero_norm )
# 		self.internal_assert_almostEquals( R_zero_lag     , AB_lag       )
# 		self.internal_assert_almostEquals( R_zero_lag_norm, AB_lag_norm  )
#
# 	def internal_check_ccf(self, A, B, AB_circ, AB_circ_norm, AB_zero, AB_zero_norm, AB_lag, AB_lag_norm, cent_AB_circ, cent_AB_circ_norm, cent_AB_zero, cent_AB_zero_norm, cent_AB_lag, cent_AB_lag_norm):
# 		self.internal_check_ccf_center( A, B,      AB_circ,      AB_circ_norm,      AB_zero,      AB_zero_norm,      AB_lag,      AB_lag_norm, False )
# 		self.internal_check_ccf_center( A, B, cent_AB_circ, cent_AB_circ_norm, cent_AB_zero, cent_AB_zero_norm, cent_AB_lag, cent_AB_lag_norm, True  )
#
# 	def internal_test_image(self, nx, ny=1, nz=1):
# 		e = EMData()
# 		e.set_size(nx, ny, nz)
# 		e.process_inplace("testimage.tomo.objects")
# 		return  e
#
# 	def internal_test_image2(self, nx, ny=1, nz=1):
# 		e = EMData()
# 		e.set_size(nx, ny, nz)
# 		e.process_inplace("testimage.tomo.objects")
# 		e = cyclic_shift(e, nx/2, ny/3, nz/5)
# 		e = mirror(e)
# 		return  e
#
# 	# ======================= TESTS FOR acf* functions
#
# 	def test_acf_circle_2D_20x30(self):
# 		"""test acf*: circle 2D, 20x30.........................."""
#
# 		A = model_circle(7, 20, 30)
# 		self.internal_check_ccf(A, A, acf(A,False), acfn(A,False), acfp(A,False), acfnp(A,False), acfpl(A, False), acfnpl(A, False)
# 									, acf(A,True ), acfn(A,True ), acfp(A,True ), acfnp(A,True ), acfpl(A, True ), acfnpl(A, True ) )
#
# 	def test_acf_circle_2D_21x31(self):
# 		"""test acf*: circle 2D, 21x31.........................."""
#
# 		A = model_circle(7, 21, 31)
# 		self.internal_check_ccf(A, A, acf(A,False), acfn(A,False), acfp(A,False), acfnp(A,False), acfpl(A, False), acfnpl(A, False)
# 									, acf(A,True ), acfn(A,True ), acfp(A,True ), acfnp(A,True ), acfpl(A, True ), acfnpl(A, True ) )
#
# 	def test_acf_circle_2D_31x20(self):
# 		"""test acf*: circle 2D, 31x20.........................."""
# 		A = model_circle(7, 31, 20)
# 		self.internal_check_ccf(A, A, acf(A,False), acfn(A,False), acfp(A,False), acfnp(A,False), acfpl(A, False), acfnpl(A, False)
# 									, acf(A,True ), acfn(A,True ), acfp(A,True ), acfnp(A,True ), acfpl(A, True ), acfnpl(A, True ) )
#
# 	def test_acf_objects_2D_20x30(self):
# 		"""test acf*: objects 2D, 20x30.........................."""
# 		A = self.internal_test_image(20, 30)
# 		self.internal_check_ccf(A, A, acf(A,False), acfn(A,False), acfp(A,False), acfnp(A,False), acfpl(A, False), acfnpl(A, False)
# 									, acf(A,True ), acfn(A,True ), acfp(A,True ), acfnp(A,True ), acfpl(A, True ), acfnpl(A, True ) )
#
# 	def test_acf_objects_2D_21x31(self):
# 		"""test acf*: objects 2D, 21x31.........................."""
# 		A = self.internal_test_image(21, 31)
# 		self.internal_check_ccf(A, A, acf(A,False), acfn(A,False), acfp(A,False), acfnp(A,False), acfpl(A, False), acfnpl(A, False)
# 									, acf(A,True ), acfn(A,True ), acfp(A,True ), acfnp(A,True ), acfpl(A, True ), acfnpl(A, True ) )
#
# 	def test_acf_objects_2D_31x20(self):
# 		"""test acf*: objects 2D, 31x20.........................."""
# 		A = self.internal_test_image(31, 20)
# 		self.internal_check_ccf(A, A, acf(A,False), acfn(A,False), acfp(A,False), acfnp(A,False), acfpl(A, False), acfnpl(A, False)
# 									, acf(A,True ), acfn(A,True ), acfp(A,True ), acfnp(A,True ), acfpl(A, True ), acfnpl(A, True ) )
#
# 	# ======================= TESTS FOR ccf* functions
#
# 	def test_ccf_circle_2D_20x30(self):
# 		"""test ccf*: circle 2D, 20x30.........................."""
#
# 		A = model_circle(7, 20, 30)
# 		B = model_circle(4, 20, 30)
# 		self.internal_check_ccf(A, B, ccf(A,B,False), ccfn(A,B,False), ccfp(A,B,False), ccfnp(A,B,False), ccfpl(A,B,False), ccfnpl(A,B,False)
# 									, ccf(A,B,True ), ccfn(A,B,True ), ccfp(A,B,True ), ccfnp(A,B,True ), ccfpl(A,B,True ), ccfnpl(A,B,True ) )
#
# 	def test_ccf_circle_2D_21x31(self):
# 		"""test ccf*: circle 2D, 21x31.........................."""
# 		A = model_circle(7, 21, 31)
# 		B = model_circle(4, 21, 31)
# 		self.internal_check_ccf(A, B, ccf(A,B,False), ccfn(A,B,False), ccfp(A,B,False), ccfnp(A,B,False), ccfpl(A,B,False), ccfnpl(A,B,False)
# 									, ccf(A,B,True ), ccfn(A,B,True ), ccfp(A,B,True ), ccfnp(A,B,True ), ccfpl(A,B,True ), ccfnpl(A,B,True ) )
#
# 	def test_ccf_circle_2D_31x20(self):
# 		"""test ccf*: circle 2D, 31x20.........................."""
# 		A = model_circle(7, 31, 20)
# 		B = model_circle(4, 31, 20)
# 		self.internal_check_ccf(A, B, ccf(A,B,False), ccfn(A,B,False), ccfp(A,B,False), ccfnp(A,B,False), ccfpl(A,B,False), ccfnpl(A,B,False)
# 									, ccf(A,B,True ), ccfn(A,B,True ), ccfp(A,B,True ), ccfnp(A,B,True ), ccfpl(A,B,True ), ccfnpl(A,B,True ) )
#
# 	def test_ccf_objects_2D_20x30(self):
# 		"""test ccf*: objects 2D, 20x30.........................."""
# 		A = self.internal_test_image(20, 30)
# 		B = self.internal_test_image2(20, 30)
# 		self.internal_check_ccf(A, B, ccf(A,B,False), ccfn(A,B,False), ccfp(A,B,False), ccfnp(A,B,False), ccfpl(A,B,False), ccfnpl(A,B,False)
# 									, ccf(A,B,True ), ccfn(A,B,True ), ccfp(A,B,True ), ccfnp(A,B,True ), ccfpl(A,B,True ), ccfnpl(A,B,True ) )
#
# 	def test_ccf_objects_2D_21x31(self):
# 		"""test ccf*: objects 2D, 21x31.........................."""
# 		A = self.internal_test_image(21, 31)
# 		B = self.internal_test_image2(21, 31)
# 		self.internal_check_ccf(A, B, ccf(A,B,False), ccfn(A,B,False), ccfp(A,B,False), ccfnp(A,B,False), ccfpl(A,B,False), ccfnpl(A,B,False)
# 									, ccf(A,B,True ), ccfn(A,B,True ), ccfp(A,B,True ), ccfnp(A,B,True ), ccfpl(A,B,True ), ccfnpl(A,B,True ) )
#
# 	def test_ccf_objects_2D_31x20(self):
# 		"""test ccf*: objects 2D, 31x20.........................."""
# 		A = self.internal_test_image(31, 20)
# 		B = self.internal_test_image2(31, 20)
# 		self.internal_check_ccf(A, B, ccf(A,B,False), ccfn(A,B,False), ccfp(A,B,False), ccfnp(A,B,False), ccfpl(A,B,False), ccfnpl(A,B,False)
# 									, ccf(A,B,True ), ccfn(A,B,True ), ccfp(A,B,True ), ccfnp(A,B,True ), ccfpl(A,B,True ), ccfnpl(A,B,True ) )
#
# 	# ======================= TESTS FOR cnv* functions
# '''
# 	def test_cnv_circle_2D_20x30(self):
# 		"""test cnv*: circle 2D, 20x30.........................."""
# 		from utilities import model_circle
# 		from fundamentals import cnv, cnvn, cnvnp, cnvnpl, cnvp, cnvpl, mirror
# 		A = model_circle(7, 20, 30)
# 		B = model_circle(4, 20, 30)
# 		C = mirror(mirror(B,'x'),'y')
# 		self.internal_check_ccf(A, C, cnv(A,B,False), cnvn(A,B,False), cnvp(A,B,False), cnvnp(A,B,False), cnvpl(A,B,False), cnvnpl(A,B,False)
# 									, cnv(A,B,True ), cnvn(A,B,True ), cnvp(A,B,True ), cnvnp(A,B,True ), cnvpl(A,B,True ), cnvnpl(A,B,True ) )
#
# 	def test_cnv_circle_2D_21x31(self):
# 		"""test cnv*: circle 2D, 21x31.........................."""
# 		from utilities import model_circle
# 		from fundamentals import cnv, cnvn, cnvnp, cnvnpl, cnvp, cnvpl, mirror
# 		A = model_circle(7, 21, 31)
# 		B = model_circle(4, 21, 31)
# 		C = mirror(mirror(B,'x'),'y')
# 		self.internal_check_ccf(A, C, cnv(A,B,False), cnvn(A,B,False), cnvp(A,B,False), cnvnp(A,B,False), cnvpl(A,B,False), cnvnpl(A,B,False)
# 									, cnv(A,B,True ), cnvn(A,B,True ), cnvp(A,B,True ), cnvnp(A,B,True ), cnvpl(A,B,True ), cnvnpl(A,B,True ) )
#
# 	def test_cnv_circle_2D_31x20(self):
# 		"""test cnv*: circle 2D, 31x20.........................."""
# 		from utilities import model_circle
# 		from fundamentals import cnv, cnvn, cnvnp, cnvnpl, cnvp, cnvpl, mirror
# 		A = model_circle(7, 31, 20)
# 		B = model_circle(4, 31, 20)
# 		C = mirror(mirror(B,'x'),'y')
# 		self.internal_check_ccf(A, C, cnv(A,B,False), cnvn(A,B,False), cnvp(A,B,False), cnvnp(A,B,False), cnvpl(A,B,False), cnvnpl(A,B,False)
# 									, cnv(A,B,True ), cnvn(A,B,True ), cnvp(A,B,True ), cnvnp(A,B,True ), cnvpl(A,B,True ), cnvnpl(A,B,True ) )
#
# 	def test_cnv_objects_2D_20x30(self):
# 		"""test cnv*: objects 2D, 20x30.........................."""
# 		from fundamentals import cnv, cnvn, cnvnp, cnvnpl, cnvp, cnvpl, mirror
# 		A = self.internal_test_image(20, 30)
# 		B = self.internal_test_image2(20, 30)
# 		C = mirror(mirror(B,'x'),'y')
# 		self.internal_check_ccf(A, C, cnv(A,B,False), cnvn(A,B,False), cnvp(A,B,False), cnvnp(A,B,False), cnvpl(A,B,False), cnvnpl(A,B,False)
# 									, cnv(A,B,True ), cnvn(A,B,True ), cnvp(A,B,True ), cnvnp(A,B,True ), cnvpl(A,B,True ), cnvnpl(A,B,True ) )
#
# 	def test_cnv_objects_2D_21x31(self):
# 		"""test cnv*: objects 2D, 21x31.........................."""
# 		from fundamentals import cnv, cnvn, cnvnp, cnvnpl, cnvp, cnvpl, mirror
# 		A = self.internal_test_image(21, 31)
# 		B = self.internal_test_image2(21, 31)
# 		C = mirror(mirror(B,'x'),'y')
# 		self.internal_check_ccf(A, C, cnv(A,B,False), cnvn(A,B,False), cnvp(A,B,False), cnvnp(A,B,False), cnvpl(A,B,False), cnvnpl(A,B,False)
# 									, cnv(A,B,True ), cnvn(A,B,True ), cnvp(A,B,True ), cnvnp(A,B,True ), cnvpl(A,B,True ), cnvnpl(A,B,True ) )
#
# 	def test_cnv_objects_2D_31x20(self):
# 		"""test cnv*: objects 2D, 31x20.........................."""
# 		from fundamentals import cnv, cnvn, cnvnp, cnvnpl, cnvp, cnvpl, mirror
# 		A = self.internal_test_image(31, 20)
# 		B = self.internal_test_image2(31, 20)
# 		C = mirror(mirror(B,'x'),'y')
# 		self.internal_check_ccf(A, C, cnv(A,B,False), cnvn(A,B,False), cnvp(A,B,False), cnvnp(A,B,False), cnvpl(A,B,False), cnvnpl(A,B,False)
# 									, cnv(A,B,True ), cnvn(A,B,True ), cnvp(A,B,True ), cnvnp(A,B,True ), cnvpl(A,B,True ), cnvnpl(A,B,True ) )
# '''
#
# def test_main():
# 	p = OptionParser()
# 	p.add_option('--t', action='store_true', help='test exception', default=False )
# 	global IS_TEST_EXCEPTION
# 	opt, args = p.parse_args()
# 	if opt.t:
# 		IS_TEST_EXCEPTION = True
# 	Log.logger().set_level(-1)  #perfect solution for quenching the Log error information, thank Liwei
# 	suite = unittest.TestLoader().loadTestsFromTestCase(TestCorrelationFunctions)
# 	unittest.TextTestRunner(verbosity=2).run(suite)
#
# if __name__ == '__main__':
# 	unittest.main()
# 	# test_main()
