#!/bin/env python
from __future__ import print_function

from builtins import range


# apply random alignment parameters to 2D images
# invert the transformation and store parameters in headers
#  so transform2d will produce properly aligned image series.

import EMAN2_cppwrap
import random
import sparx_alignment
import sparx_fundamentals
import sparx_utilities
stack_data = "ang12_2.hdf"
outf = "rang12_2.hdf"
nima = EMAN2_cppwrap.EMUtil.get_image_count(stack_data)
attributes = ['alpha', 'sx', 'sy', 'mirror']
data = EMAN2_cppwrap.EMData()
data.read_image(stack_data, 0, True)
im = data.get_xsize()
kb = sparx_alignment.kbt(im)

for im in range(nima):
	data = EMAN2_cppwrap.EMData()
	data.read_image(stack_data, im)
	sx = (random.random()-0.5)*10.0
	sy = (random.random()-0.5)*10.0
	alpha = random.random()*360.0
	mir = random.randint(0,1)
	data = sparx_fundamentals.rot_shift2D(data,alpha,sx,sy,interpolation_method="gridding")
	#  invert the transformation.
	alphah, sxh, syh, sc = sparx_utilities.compose_transform2(0.0,-sx, -sy, 1.0 ,-alpha,0.,0.,1.)
	if(mir):
		data = sparx_fundamentals.mirror(data)
		alphah, sxh, syh, sc = sparx_utilities.combine_params2(0.,0.,0.,1.,alphah,sxh, syh,0.)
	sparx_utilities.set_arb_params(data, [alphah, sxh, syh, mir], attributes)
	data.write_image(outf, im)
