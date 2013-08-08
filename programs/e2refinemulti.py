#!/usr/bin/env python

#
# Author: Steve Ludtke 06/10/2013 (sludtke@bcm.edu)
# Copyright (c) 2013- Baylor College of Medicine
#
# This software is issued under a joint BSD/GNU license. You may use the
# source code in this file under either license. However, note that the
# complete EMAN2 and SPARX software packages have some GPL dependencies,
# so you are responsible for compliance with the licenses of these packages
# if you opt to use BSD licensing. The warranty disclaimer below holds
# in either instance.
#
# This complete copyright notice must be included in any revised version of the
# source code. Additional authorship citations may be added, but existing
# author citations must be preserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  2111-1307 USA
#
#


from EMAN2 import *
from optparse import OptionParser
from math import *
import os
import sys
import time
from numpy import array

# This is used to build the HTML status file. It's a global for convenience
output_html=[]
output_html_com=[]
output_path=None

def append_html(msg,com=False) :
	global output_html,output_html_com
	if com : output_html_com.append(str(msg))
	else : output_html.append(str(msg))
	write_html()


def write_html() :
	global output_html,output_html_com,output_path
	out=file(output_path+"/index.html","w")
	out.write("<html><head><title>EMAN2 Multi-model Refinement Analysis</title></head>\n<body>")
	out.write("\n".join(output_html))
	out.write("<h4>Detailed command log</h4>\n")
	out.write("\n".join(output_html_com))
	out.write("<br></br><hr></hr>Generated by {ver} {date}\n</body></html>".format(ver=EMANVERSION,date=CVSDATESTAMP))

def main():
	print "This program is still being converted to EMAN2.1. exiting"
	progname = os.path.basename(sys.argv[0])
	usage = """prog [options]

	This is the multiple model single particle refinement program in EMAN2.1+. It replaces the earlier e2refinemulti, offering features similar to e2refine_easy.
Major features of this program:

 * While a range of command-line options still exist. You should not normally specify more than the basic requirements. The rest will be auto-selected for you.
 * Unlike e2refine_easy, this program doesn't compute gold_standard resolution curves, since it's already splitting the data into multiple groups. After it completes, the particles are split, easy_refine is run on the fractions.
 * An HTML report file will be generated as this program runs, telling you exactly what it decided to do and why, as well as giving information about runtime, etc while the job is still running.
 * If you specify only one starting model it will be randomly perturbed N times, and results may be different with each run, depending on the nature of the heterogeneity in the data.
 * Many of the 'advanced' options are hidden in the e2projectmanager.py GUI, because most users should not need to specify them.

To run this program, you would normally specify only the following options:
  --model=<starting map to seed refinement>
  --nmodels=<number of starting models to generate from model>
  OR
  --models=<starting map 1>,<starting map 2>,...

  --input=<lst file referencing phase-flipped particles in HDF format>

  --targetres=<in A>     Resolution to target in Angstroms in this refinement run. Do not be overoptimistic !
                         Generally begin with something conservative like 25, then use --startfrom and reduce
                         to ~12, only after that try for high (3-8 A). Data permitting, of course. Low resolution
                         attempts will run MUCH faster due to more efficient parameters.
  --sym=<symmetry>       Symmetry to enforce during refinement (Cn, Dn, icos, oct, cub).
                         Default=c1 (no symmetry)
  --mass=<in kDa>        Putative mass of object in kDa, but as desired volume varies with resolution
                         actual number may vary by as much a ~2x from the true value. The goal is to
                         have a good isosurface in the final map with a threshold of 1.0.
  --parallel=<par spec>  While not strictly required, without this option the refinement will run on a single CPU
                         and you will likely wait a very long time. To use more than one core on a single computer,
                         just say thread:N (eg - thread:4). For other options, like MPI, see:
                         http://blake.bcm.edu/emanwiki/EMAN2/Parallel for details.

  Optional:
  --apix=<A/pix>         The value will normally come from the particle data if present. You can override with this.
  --sep=<classes/ptcl>   each particle will be put into N classes. Improves contrast at cost of rotational blur.
  --classkeep=<frac>     fraction of particles to use in final average. Default 90%%. Should be >50%%
  --m3dkeep=<frac>       fraction of class-averages to use in 3-D map. Default=auto
  --classautomask        applies an automask when aligning particles for improved alignment
  --m3dpostprocess       <name>:<parm>=<value>:...  An arbitrary processor
                         (e2help.py processors -v2) to apply to the 3-D map after each
                         iteration. Default=none
  --path=<path>          Normally the new directory will be named automatically. If you prefer your own convention
                         you can override, but it may cause minor GUI problems if you break the standard naming
                         convention.

========================================================================
  There are numerous additional options based on the original e2refine.py command. These options are not available from
the graphical interface, as it is generally best to let e2refine_easy pick these values for you. Normally you should
not need to specify any of the following other than the ones already listed above:

"""
	parser = EMArgumentParser(usage=usage,version=EMANVERSION)

	#options associated with e2refine.py
	#parser.add_header(name="multirefineheader", help='Options below this label are specific to e2refinemulti', title="### e2refinemulti options ###", row=1, col=0, rowspan=1, colspan=3, mode="refinement")
	#parser.add_header(name="multimodelheader", help='Options below this label are specific to e2refinemulti Model', title="### e2refinemulti model options ###", row=4, col=0, rowspan=1, colspan=3, mode="refinement")
	parser.add_argument("--model", dest="model", type=str,default=None, help="The map to use as a starting point for refinement", guitype='filebox', browser='EMModelsTable(withmodal=True,multiselect=False)', filecheck=False, row=1, col=0, rowspan=1, colspan=3, mode="refinement")
	parser.add_argument("--nmodels", dest = "nmodels", type = int, default=2, help = "The total number of refinement iterations to perform. Default=auto", guitype='intbox', row=3, col=2, rowspan=1, colspan=1, mode="refinement")
	parser.add_header(name="orblock", help='Just a visual separation', title="- OR -", row=5, col=0, rowspan=1, colspan=3, mode="refinement")
	parser.add_argument("--models", dest="models", type=str,default=None, help="The map to use as a starting point for refinement", guitype='filebox', browser='EMModelsTable(withmodal=True,multiselect=True)', filecheck=False, row=7, col=0, rowspan=1, colspan=3, mode="refinement")
	parser.add_argument("--input", dest="input", default=None,type=str, help="The name of the image file containing the particle data", guitype='filebox', browser='EMSetsTable(withmodal=True,multiselect=False)', filecheck=False, row=8, col=0, rowspan=1, colspan=3, mode="refinement")
	parser.add_header(name="required", help='Just a visual separation', title="Required:", row=9, col=0, rowspan=1, colspan=3, mode="refinement")
	parser.add_argument("--targetres", default=12.0, type=float,help="Target resolution in A of the final single-model refinements.", guitype='floatbox', row=10, col=0, rowspan=1, colspan=1, mode="refinement")
	parser.add_argument("--sym", dest = "sym", default="c1",help = "Specify symmetry - choices are: c<n>, d<n>, tet, oct, icos. You can specify either a single value or one for each model.", guitype='strbox', row=10, col=1, rowspan=1, colspan=1, mode="refinement")
	parser.add_argument("--iter", dest = "iter", type = int, default=6, help = "The total number of refinement iterations to perform. Default=auto", guitype='intbox', row=10, col=2, rowspan=1, colspan=1, mode="refinement")
	parser.add_argument("--mass", default=0, type=float,help="The ~mass of the particle in kilodaltons, used to run normalize.bymass. Due to resolution effects, not always the true mass.", guitype='floatbox', row=12, col=0, rowspan=1, colspan=1, mode="refinement['self.pm().getMass()']")
	parser.add_header(name="optional", help='Just a visual separation', title="Optional:", row=14, col=0, rowspan=1, colspan=3, mode="refinement")
	parser.add_argument("--apix", default=0, type=float,help="The angstrom per pixel of the input particles. This argument is required if you specify the --mass argument. If unspecified (set to 0), the convergence plot is generated using either the project apix, or if not an apix of 1.", guitype='floatbox', row=16, col=0, rowspan=1, colspan=1, mode="refinement['self.pm().getAPIX()']")
	parser.add_argument("--sep", type=int, help="The number of classes each particle can contribute towards (normally 1). Increasing will improve SNR, but produce rotational blurring.", default=1, guitype='intbox', row=16, col=1, rowspan=1, colspan=1, mode="refinement")
	parser.add_argument("--classkeep",type=float,help="The fraction of particles to keep in each class, based on the similarity score. (default=0.9 -> 90%%)", default=0.9, guitype='floatbox', row=16, col=2, rowspan=1, colspan=2, mode="refinement")
	parser.add_argument("--classautomask",default=False, action="store_true", help="This will apply an automask to the class-average during iterative alignment for better accuracy. The final class averages are unmasked.",guitype='boolbox', row=18, col=0, rowspan=1, colspan=1, mode="refinement")
	parser.add_argument("--prethreshold",default=False, action="store_true", help="Applies a threshold to the volume just before generating projections. A sort of aggressive solvent flattening for the reference.",guitype='boolbox', row=18, col=0, rowspan=1, colspan=1, mode="refinement")
	parser.add_argument("--m3dkeep", type=float, help="The fraction of slices to keep in e2make3d.py. Default=0.8 -> 80%%", default=0.8, guitype='floatbox', row=18, col=1, rowspan=1, colspan=1, mode="refinement")
	parser.add_argument("--nosingle",default=False, action="store_true", help="Normally the multi-model refinement will be followed by N single model refinements automatically. If this is set the job will finish after making the split data sets.",guitype='boolbox', row=18, col=2, rowspan=1, colspan=1, mode="refinement")
	parser.add_argument("--m3dpostprocess", type=str, default="", help="Default=none. An arbitrary post-processor to run after all other automatic processing.", guitype='comboparambox', choicelist='re_filter_list(dump_processors_list(),\'filter.lowpass|filter.highpass\')', row=20, col=0, rowspan=1, colspan=3, mode="refinement")
	parser.add_argument("--parallel","-P",type=str,help="Run in parallel, specify type:<option>=<value>:<option>=<value>. See http://blake.bcm.edu/emanwiki/EMAN2/Parallel",default=None, guitype='strbox', row=24, col=0, rowspan=1, colspan=2, mode="refinement")
	parser.add_argument("--path", default=None, type=str,help="The name of a directory where results are placed. Default = create new multi_xx")
	parser.add_argument("--verbose", "-v", dest="verbose", action="store", metavar="n", type=int, default=0, help="verbose level [0-9], higner number means higher level of verboseness")
#	parser.add_argument("--usefilt", dest="usefilt", type=str,default=None, help="Specify a particle data file that has been low pass or Wiener filtered. Has a one to one correspondence with your particle data. If specified will be used in projection matching routines, and elsewhere.")

	# options associated with e2project3d.py
#	parser.add_header(name="projectheader", help='Options below this label are specific to e2project', title="### e2project options ###", row=12, col=0, rowspan=1, colspan=3)
	parser.add_argument("--automask3d", default=None, type=str,help="Default=auto. Specify as a processor, eg - mask.auto3d:threshold=1.1:radius=30:nshells=5:nshellsgauss=5.", )
	parser.add_argument("--projector", dest = "projector", default = "standard",help = "Default=standard. Projector to use with parameters.")
	parser.add_argument("--orientgen", type = str, default=None,help = "Default=auto. Orientation generator for projections, eg - eman:delta=5.0:inc_mirror=0:perturb=1")

	# options associated with e2simmx.py
#	parser.add_header(name="simmxheader", help='Options below this label are specific to e2simmx', title="### e2simmx options ###", row=15, col=0, rowspan=1, colspan=3)
	parser.add_argument("--simalign",type=str,help="Default=auto. The name of an 'aligner' to use prior to comparing the images", default="rotate_translate_flip")
	parser.add_argument("--simaligncmp",type=str,help="Default=auto. Name of the aligner along with its construction arguments",default=None)
	parser.add_argument("--simralign",type=str,help="Default=auto. The name and parameters of the second stage aligner which refines the results of the first alignment", default=None)
	parser.add_argument("--simraligncmp",type=str,help="Default=auto. The name and parameters of the comparitor used by the second stage aligner.",default=None)
	parser.add_argument("--simcmp",type=str,help="Default=auto. The name of a 'cmp' to be used in comparing the aligned images", default=None)
	parser.add_argument("--simmask",type=str,help="Default=auto. A file containing a single 0/1 image to apply as a mask before comparison but after alignment", default=None)
	parser.add_argument("--shrink", dest="shrink", type = int, default=0, help="Default=auto. Optionally shrink the input particles by an integer amount prior to computing similarity scores. For speed purposes. 0 -> no shrinking", )
	parser.add_argument("--shrinks1", dest="shrinks1", type = int, help="The level of shrinking to apply in the first stage of the two-stage classification process. Default=0 (autoselect)",default=0)
	parser.add_argument("--prefilt",action="store_true",help="Default=auto. Filter each reference (c) to match the power spectrum of each particle (r) before alignment and comparison. Applies both to classification and class-averaging.",default=False)

	# options associated with e2classify.py

	# options associated with e2classaverage.py
#	parser.add_header(name="caheader", help='Options below this label are specific to e2classaverage', title="### e2classaverage options ###", row=22, col=0, rowspan=1, colspan=3, mode="refinement")
	parser.add_argument("--classkeepsig", default=False, action="store_true", help="Change the keep (\'--keep\') criterion from fraction-based to sigma-based.")
	parser.add_argument("--classiter", type=int, help="Default=auto. The number of iterations to perform.",default=-1)
	parser.add_argument("--classalign",type=str,default="rotate_translate_flip",help="Default=auto. If doing more than one iteration, this is the name and parameters of the 'aligner' used to align particles to the previous class average.")
	parser.add_argument("--classaligncmp",type=str,help="Default=auto. This is the name and parameters of the comparitor used by the fist stage aligner.",default=None)
	parser.add_argument("--classralign",type=str,help="Default=auto. The second stage aligner which refines the results of the first alignment in class averaging.", default=None)
	parser.add_argument("--classraligncmp",type=str,help="Default=auto. The comparitor used by the second stage aligner in class averageing.",default=None)
	parser.add_argument("--classaverager",type=str,help="Default=auto. The averager used to generate the class averages. Default is \'mean\'.",default=None)
	parser.add_argument("--classcmp",type=str,help="Default=auto. The name and parameters of the comparitor used to generate similarity scores, when class averaging.", default=None)
	parser.add_argument("--classnormproc",type=str,default="normalize.edgemean",help="Default=auto. Normalization applied during class averaging")
	parser.add_argument("--classrefsf",default=False, action="store_true", help="Default=True. Use the setsfref option in class averaging to produce better filtered averages.")


	#options associated with e2make3d.py
#	parser.add_header(name="make3dheader", help='Options below this label are specific to e2make3d', title="### e2make3d options ###", row=32, col=0, rowspan=1, colspan=3)
	parser.add_argument("--pad", type=int, dest="pad", default=0, help="Default=auto. To reduce Fourier artifacts, the model is typically padded by ~25 percent - only applies to Fourier reconstruction")
	parser.add_argument("--recon", dest="recon", default="fourier", help="Default=auto. Reconstructor to use see e2help.py reconstructors -v",)
	parser.add_argument("--m3dkeepsig", default=False, action="store_true", help="Default=auto. The standard deviation alternative to the --m3dkeep argument")
	parser.add_argument("--m3dsetsf", type=str,dest="m3dsetsf", default=None, help="Default=auto. Name of a file containing a structure factor to apply after refinement")
	parser.add_argument("--m3dpreprocess", type=str, default="normalize.edgemean", help="Default=auto. Normalization processor applied before 3D reconstruction")

	#lowmem!
	parser.add_argument("--lowmem", default=True, action="store_true",help="Default=auto. Make limited use of memory when possible - useful on lower end machines")
	parser.add_argument("--ppid", type=int, help="Set the PID of the parent process, used for cross platform PPID",default=-1)

	(options, args) = parser.parse_args()

	if options.model!=None and options.models!=None:
		print "ERROR : You may specify --model with --nmodels OR --models, not both"
		sys.exit(1)

	if options.input==None or options.input[-4:]!=".lst":
		print "ERROR : You must specify --input, which must be a .lst file\n"
		sys.exit(1)

	if options.path == None:
		fls=[int(i[-2:]) for i in os.listdir(".") if i[:6]=="multi_" and len(i)==9]
		if len(fls)==0 : fls=[0]
		options.path = "multi_{:02d}".format(max(fls)+1)

	global output_path
	output_path="{}/report".format(options.path)
	try: os.makedirs(output_path)
	except: pass

	# make randomized starting models
	if options.model!=None:
		model=EMData(options.model,0)
		for i in range(options.nmodels):
			model.process_inplace("filter.lowpass.randomphase",{"cutoff_freq":.03})
			model.write_image("{}/threed_00_{:02d}.hdf".format(options.path,i),0)
	else:
		# or copy the specified starting models
		options.nmodels=len(options.models)
		for i,m in enumerate(options.models):
			model=EMData(m,0)
			model.write_image("{}/threed_00_{:02d}.hdf".format(options.path,i))

	progress = 0.0
	total_procs = 5*options.iter

	if options.automask3d: automask_parms = parsemodopt(options.automask3d) # this is just so we only ever have to do it
	apix = get_apix_used(options)

	if options.targetres<apix*2:
		print "ERROR: Target resolution is smaller than 2*A/pix value. This is impossible."
		sys.exit(1)

	logid=E2init(sys.argv,options.ppid)

	###################################
	### This is where we fill in all of the undefined options, and analyse the data
	###################################
	append_html("<h1>e2refine_easy.py report</h1>\n")

	append_html("<h3>Warning - This is an alpha release version of EMAN2.1</h3> <p>This program is still experimental and planned functionality is not yet complete. \
While it will run refinements and do a decent job with parameters, this will improve substantially as we approach the actual 2.1 release, so please do not \
judge EMAN2.1 based on this preliminary version. As always, bug reports or reports of unexpected results are welcome.</p>\
<p>This analysis document is quite verbose. If you are just curious to see a list of the exact refinement parameters used, browse to the 0_refine_parms.json file\
in the refinement directory. You can use Info with the browser or just read the file directly (.json files are plain text)")


	### Prepare initial models
	# make sure the box sizes match
	hdr=EMData(options.input,0,True)
	xsize=hdr["nx"]
	apix=hdr["apix_x"]
	for i in range(options.nmodels):
		xsize3d=EMData("{}/threed_00_{:02d}.hdf".format(options.path,i),0,True)["nx"]
		if ( xsize3d != xsize ) :
			append_html("The dimensions of the particles ( {ptcl}x{ptcl} ) do not match the dimensions of initial model {n} ( {vol}x{vol}x{vol} ). I will assume A/pix is correct in the model and rescale/resize accordingly.".format(ptcl=xsize,vol=xsize3d,n=i))
			img3 = EMData("{}/threed_00_{:02d}".format(options.path,i),0,True)
			try:
				scale=img3["apix_x"]/apix
			except:
				print "A/pix unknown, assuming scale same as relative box size"
				scale=float(xsize)/xsize3d
			if scale>1 : cmd="e2proc3d.py {path}/threed_00_{i:02d} {path}/threed_00_{i:02d} --clip={cl},{cl},{cl} --scale={sca:1.4f}".format(path=options.path,i=i,cl=nx,sca=scale)
			else :       cmd="e2proc3d.py {path}/threed_00_{i:02d} {path}/threed_00_{i:02d} --scale={sca:1.4f} --clip={cl},{cl},{cl}".format(path=options.path,i=i,cl=nx,sca=scale)
			run(cmd)

	repim=EMData(options.input,0)		# read a representative image to get some basic info
	if repim.has_attr("ctf") : hasctf=True
	else: hasctf=False
	nx=repim["nx"]

	# Fill in optional parameters

	if hasctf:
		if os.path.exists("strucfac.txt") :
			append_html("<p>Several different methods can be used for final amplitude correction in cryoEM. For refinemulti, we base the filter on the FSC \
between the different output models. Second stage single-model refinement will use a different method.</p>")
			postprocess=""
			m3dsetsf="--setsf strucfac.txt"
		else :
			append_html("<p>No data-based structure factor was found in the project. Computing one during CTF correction is highly recommended. Falling back to two-stage filtration: \
'filter.lowpass.autob' which flattens the overall falloff of the structure factor in the 4-15 A range, \
and filter.wiener.byfsc which performs a low-pass Wiener filter based on the computed FSC curve between even/odd maps. \
While this filtration can work reasonably well, you may find that it over-exagerates low-resolution terms over multiple iterations. \
To avoid this, compute a structure factor.</p>")
			postprocess="--postprocess filter.lowpass.autob"
			m3dsetsf=""
	else:
		append_html("<p>No CTF information found in the input data. Note that EMAN2 cannot perform optimal reconstructions using phase-flipped particles from other \
software. Part of EMAN2's CTF correction process is measuring the SSNR of the particle data, an estimate of the low resolution structure factor and other parameters \
which are used to provide more accurate orientations, filters, etc. during processing. Since CTF information isn't present, we are limited to a basic refinement. \
output maps will be low-pass filtered based on resolution, but note that there will be no B-factor correction, so this will result in over-filtration of the final \
maps.")
		postprocess=""
		m3dsetsf=""


	if options.orientgen==None :
			astep=90.0/ceil(90.0/sqrt(4300/nx))		# This rounds to the best angular step divisible by 90 degrees
			options.orientgen="eman:delta={:1.3f}:inc_mirror=0".format(astep)
			append_html("<p>I will use an angular sampling of {} deg. For details, please see \
<a href=http://blake.bcm.edu/emanwiki/EMAN2/AngStep>http://blake.bcm.edu/emanwiki/EMAN2/AngStep</a></p>".format(options.targetres,apix,astep))
	else :
		append_html("<p>Using your specified orientation generator with angular step. You may consider reading this page: <a href=http://blake.bcm.edu/emanwiki/EMAN2/AngStep>http://blake.bcm.edu/emanwiki/EMAN2/AngStep</a></p></p>")

	if options.classiter<0 :
		options.classiter=3
		append_html("<p>Setting --classiter to 3 to give better convergence with multi-model refinement. In the followup single model refinements, this may be decreased automatically.</p>")

	if options.simaligncmp==None : options.simaligncmp="ccc"
	if options.simralign==None :
		if options.targetres>=7.0 :
			options.simralign="refine"
			options.simraligncmp="ccc"
		else :
			options.simralign="refine"
			options.simraligncmp="frc:zeromask=1:snrweight=1"
	simralign="--ralign {} --raligncmp {}".format(options.simralign,options.simraligncmp)

	if options.simcmp==None :
		if options.targetres>18.0 or not hasctf: options.simcmp="frc:maxres={}".format(options.targetres)
		elif options.targetres>7.0 : options.simcmp="frc:snrweight=1:maxres={}".format(options.targetres)	# no zeromask to avoid top/side errors at lower resolutions
		else : options.simcmp="frc:snrweight=1:maxres={}".format(max(7.0,options.targetres))

	if options.shrink==0 : shrink=""
	else : shrink="--shrink {}".format(options.shrink)
	if options.shrinks1==0 :
		if nx>=256 : shrinks1="--shrinks1 4"
		elif nx>=96: shrinks1="--shrinks1 2"
		else : shrinks1=""

	if options.classaligncmp==None :
		options.classaligncmp="ccc"

	if options.classralign==None :
		if options.targetres>15 or not hasctf :
			options.classralign="refine"
			options.classraligncmp="ccc"
		else :
			options.classralign="refine"
			options.classraligncmp="frc:snrweight=1:zeromask=1"
	classralign="--ralign {ralign} --raligncmp {raligncmp}".format(ralign=options.classralign,raligncmp=options.classraligncmp)

	if options.classaverager==None :
		if hasctf and options.targetres<15 : options.classaverager="ctfw.auto"
		else : options.classaverager="mean"

	if options.classcmp==None :
		if hasctf : options.classcmp="frc:snrweight=1"
		else : options.classcmp="ccc"

	if options.pad<nx :
		options.pad=good_size(nx*1.25)

	# deal with symmetry and alignment
	sym=options.sym.split(",")
	if len(sym)==1 or len(set(sym))==1: 
		if len(sym)==1 : sym=sym*options.nmodels
		if sym[0].lower() in ("icos","tet","oct") or sym[0][0].lower()=="d" : align="" 	# no alignment with higher symmetries
		elif sym[0][0].lower()=="c" and sym[0][1]!="1" : align=align=" --ralignz={path}/tmp0.hdf".format(path=options.path)		# z alignment only
		else: align="--alignref={path}/tmp0.hdf --align=refine_3d".format(path=options.path)	# full 3-D alignment for C1

	##################################
	### prepare for the run
	##################################
	### Convert some of the command-line options to more conventient strings for command generation
	if options.verbose>0 : verbose="--verbose {}".format(options.verbose)
	else: verbose=""

	if options.parallel!=None : parallel="--parallel {}".format(options.parallel)
	else: parallel=""

	if options.prefilt : prefilt="--prefilt"
	else: prefilt=""

	if options.simmask!=None :
		makesimmask=False
		simmask="--mask {}".format(options.simmask)
		append_html("<p>{simmask} was specified, so I will not automatically create a mask for each iteration.</p>".format(simmask=simmask))
	else:
		makesimmask=True
		simmask="--mask {path}/simmask.hdf".format(path=options.path)
		append_html("<p>As particles get translated during alignment, the total amount of noise present in the aligned particle can change\
 significantly. While this isn't a very large effect for similarity metrics like fsc, it can cause a bias in the reconstruction. Similarly\
 if we use a different mask derived for each projection to combat this problem (as with the zeromask=1 option in some comparators, then each\
 projection masks out a different fraction of the image causing some orientations to be preferred. To combat both effects, I will compute a\
 single aggregate mask from all of the projections in each iteration, and use it as {simmask}. The mask is autogenerated and overwritten\
 after each iteration. The only way to completely disable this behavior\
 is to specify --simmask yourself with a file containing all 1.0 pixels.</p>".format(simmask=simmask))

	if options.classrefsf : classrefsf="--setsfref"
	else: classrefsf=""

	if options.classautomask : classautomask="--automask"
	else: classautomask=""

	if options.classkeepsig : classkeepsig="--keepsig"
	else: classkeepsig=""

	if options.classralign!=None : classralign="--ralign {} --raligncmp {}".format(options.classralign,options.classraligncmp)
	else: classralign=""

	if options.m3dkeepsig : m3dkeepsig="--keepsig"
	else: m3dkeepsig=""

	if options.automask3d : amask3d = options.automask3d
	else : amask3d=""

	if options.prethreshold : prethreshold="--prethreshold"
	else : prethreshold=""

	# store the input arguments forever in the refinement directory
	db = js_open_dict(options.path+"/0_refine_parms.json")
	db.update(vars(options))

	print "NOTE: you can check the progress of the refinement at any time by opening this URL in your web-browser:  file://{}/index.html".format(os.path.abspath(output_path))

	### Actual refinement loop ###
	for it in range(1,options.iter+1) :
		append_html("<h4>Beginning iteration {} at {}</h4>".format(it,time.ctime(time.time())),True)

		### 3-D Projections
		models=["{path}/threed_{itrm1:02d}_{mdl:02d}.hdf".format(path=options.path,itrm1=it-1,mdl=mdl) for mdl in xrange(options.nmodels)]
		run("e2project3d.py {mdls} --outfile {path}/projections_{itr:02d}.hdf -f --projector {projector} --orientgen {orient} --sym {sym} --postprocess normalize.circlemean {prethr} {parallel} {verbose}".format(
			path=options.path,mdls=" ".join(models),itrm1=it-1,mdl=i,itr=it,projector=options.projector,orient=options.orientgen,sym=",".join(sym),prethr=prethreshold,parallel=parallel,verbose=verbose))

		progress += 1.0
		E2progress(logid,progress/total_procs)

		### We may need to make our own similarity mask file for more accurate particle classification
		if makesimmask :
			av=Averagers.get("minmax",{"max":1})
			nprj=EMUtil.get_image_count("{path}/projections_{itr:02d}.hdf".format(path=options.path,itr=it))
			print "Mask from {} projections".format(nprj)
			for i in xrange(nprj):
				a=EMData("{path}/projections_{itr:02d}.hdf".format(path=options.path,itr=it),i)
				av.add_image(a)
			msk=av.finish()
#			msk.process_inplace("threshold.binary",{"value":msk["sigma"]/50.0})
			msk.process_inplace("threshold.notzero")
			msk.write_image("{path}/simmask.hdf".format(path=options.path),0)

		### Simmx
		#FIXME - Need to combine simmx with classification !!!

		cmd = "e2simmx2stage.py {path}/projections_{itr:02d}.hdf {inputfile} {path}/simmx_{itr:02d}.hdf {path}/proj_simmx_{itr:02d}.hdf {path}/proj_stg1_{itr:02d}.hdf {path}/simmx_stg1_{itr:02d}.hdf --saveali --cmp {simcmp} \
--align {simalign} --aligncmp {simaligncmp} {simralign} {shrinks1} {shrink} {prefilt} {simmask} {verbose} {parallel}".format(
			path=options.path,itr=it,inputfile=options.input,simcmp=options.simcmp,simalign=options.simalign,simaligncmp=options.simaligncmp,simralign=simralign,
			shrinks1=shrinks1,shrink=shrink,prefilt=prefilt,simmask=simmask,verbose=verbose,parallel=parallel)
		run(cmd)
		progress += 1.0
		E2progress(logid,progress/total_procs)

		### Classify
		cmd = "e2classify.py {path}/simmx_{itr:02d}.hdf {path}/classmx_{itr:02d}.hdf -f --sep {sep} {verbose}".format(
			path=options.path,itr=it,sep=options.sep,verbose=verbose)
		run(cmd)
		progress += 1.0
		E2progress(logid,progress/total_procs)

		### Class-averaging
		cmd="e2classaverage.py --input {inputfile} --classmx {path}/classmx_{itr:02d}.hdf --storebad --output {path}/classes_{itr:02d}.hdf --ref {path}/projections_{itr:02d}.hdf --iter {classiter} \
-f --resultmx {path}/cls_result_{itr:02d}_even.hdf --normproc {normproc} --averager {averager} {classrefsf} {classautomask} --keep {classkeep} {classkeepsig} --cmp {classcmp} \
--align {classalign} --aligncmp {classaligncmp} {classralign} {prefilt} {verbose} {parallel}".format(
			inputfile=options.input, path=options.path, itr=it, classiter=options.classiter, normproc=options.classnormproc, averager=options.classaverager, classrefsf=classrefsf,
			classautomask=classautomask,classkeep=options.classkeep, classkeepsig=classkeepsig, classcmp=options.classcmp, classalign=options.classalign, classaligncmp=options.classaligncmp,
			classralign=classralign, prefilt=prefilt, verbose=verbose, parallel=parallel)
		run(cmd)
		progress += 1.0
		E2progress(logid,progress/total_procs)

		### 3-D Reconstruction
		# FIXME - --lowmem removed due to some tricky bug in e2make3d
		for mdl in xrange(options.nmodels):
			cmd="e2make3d.py --input {path}/classes_{itr:02d}.hdf --iter 2 -f --sym {sym} --output {path}/threed_{itr:02d}_{mdl:02d}.hdf --recon {recon} --preprocess {preprocess} \
{postprocess} --keep={m3dkeep} {keepsig} --apix={apix} --pad={m3dpad} {setsf} {verbose} --input_model {mdl}".format(
				path=options.path, itr=it, sym=sym[mdl], recon=options.recon, preprocess=options.m3dpreprocess, postprocess=postprocess, m3dkeep=options.m3dkeep, keepsig=m3dkeepsig,
				m3dpad=options.pad, setsf=m3dsetsf, apix=apix, verbose=verbose, mdl=mdl)
			run(cmd)

		progress += 1.0

		#######################
		### postprocessing, a bit different than e2refine_postprocess, and we need a lot of info, so we do it in-place
		
		# alignment

		run("e2proc3d.py {path}/threed_{itr:02d}_{mdl:02d}.hdf {path}/tmp0.hdf --process=filter.lowpass.gauss:cutoff_freq=.05".format(path=options.path,itr=it,mdl=0))
		o0=EMData("{path}/threed_{itr:02d}_{mdl:02d}.hdf".format(path=options.path,itr=it,mdl=0),0)
		for mdl in xrange(1,options.nmodels):
			if options.verbose>0 : print "Aligning map ",mdl
			map2="{path}/threed_{itr:02d}_{mdl:02d}.hdf".format(path=options.path,itr=it,mdl=mdl)
			run("e2proc3d.py {map2} {path}/tmp1.hdf --process=filter.lowpass.gauss:cutoff_freq=.05 {align}".format(path=options.path,map2=map2,align=align))
			run("e2proc3d.py {map2} {path}/tmp1f.hdf --process=filter.lowpass.gauss:cutoff_freq=.05 --process=xform.flip:axis=z {align}".format(path=options.path,map2=map2,align=align))
			
			# now we have to check which of the two handednesses produced the better alignment
			# Pick the best handedness
			a=EMData("{path}/tmp1.hdf".format(path=options.path),0)
			b=EMData("{path}/tmp1f.hdf".format(path=options.path),0)
			c=EMData("{path}/tmp0.hdf".format(path=options.path),0)
			ca=c.cmp("ccc",a)
			cb=c.cmp("ccc",b)
			o=EMData(map2,0)
			if ca<cb :
				try: ali=a["xform.align3d"]
				except: ali=Transform()
				if verbose>0 : print "correct hand detected ",ali
			else :
				try: ali=b["xform.align3d"]
				except: ali=Transform()
				o.process_inplace("xform.flip",{"axis":"z"})
				if verbose>0 : print "handedness flip required",ali
			o.transform(ali)

			os.unlink(map2)
			o.write_image(map2,0)
			os.unlink("{path}/tmp1.hdf".format(path=options.path))
			os.unlink("{path}/tmp1f.hdf".format(path=options.path))
			
			# now compute FSC
			f=o0.calc_fourier_shell_correlation(o)
			if mdl==1 : fm=array(f)
			else : fm+=array(f)

		# so we now have an average FSC curve between map 1 and each of the others (we should probably do all vs all, but don't)
		fm/=(options.nmodels-1.0)
		third = len(fm)/3
		xaxis = fsc[0:third]
		fsc = fm[third:2*third]
		saxis = [x/apix for x in xaxis]
		Util.save_data(saxis[1],saxis[1]-saxis[0],fsc[1:],"{path}/fsc_mutual_avg_{it:02d}.txt".format(path=options.path,it=it))
		
		models=["threed_{itr:02d}_{mdl:02d}.hdf".format(itr=it,mdl=mdl) for mdl in xrange(options.nmodels)]
		
		# we filter the maps, to a resolution ~20% higher than the FSC
		for m in models:
			run("e2proc3d.py {mod} {mod} {m3dsetsf} --process filter.wiener.byfsc:fscfile={path}/fsc_mutual_avg_{it:02d}.txt:sscale=1.2 --process normalize.bymass:thr=1:mass={mass}".format(
	m3dsetsf=m3dsetsf,mod=m,path=options.path,it=it,mass=options.mass,underfilter=underfilter))
		
		os.unlink("{path}/tmp0.hdf".format(path=options.path))

		db["last_map"]=models

		E2progress(logid,progress/total_procs)

	E2end(logid)

def get_apix_used(options):
	'''
	Just an encapsulation apix retrieval
	Basically, if the apix argument is in the options, that's what you get
	Else the project db is checked for the global.apix parameter
	Else you just get 1
	'''
	apix = 0.0
	if options.apix!=None and options.apix>0: return options.apix
	try:
		prj=js_open_dict("info/project.json")
		return prj["global.apix"]
	except: pass

	try:
		img=EMData(options.input,0,True)
		try: apix=img["ctf"].apix
		except: apix=img["apix_x"]
		return apix
	except: pass


	else:
		img=EMData(options.input,0,True)
		try: apix=img["ctf"].apix
		except: apix=img["apix_x"]

	print "ERROR: Could not find a valid A/pix value anywhere. Please specify."

	sys.exit(1)

def run(command):
	"Mostly here for debugging, allows you to control how commands are executed (os.system is normal)"

	print "{}: {}".format(time.ctime(time.time()),command)
	append_html("<p>{}: {}</p>".format(time.ctime(time.time()),command),True)

	ret=launch_childprocess(command)

	# We put the exit here since this is what we'd do in every case anyway. Saves replication of error detection code above.
	if ret !=0 :
		print "Error running: ",command
		sys.exit(1)

	return

if __name__ == "__main__":
    main()
