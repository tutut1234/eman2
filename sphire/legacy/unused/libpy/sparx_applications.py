"""1
		if CTF:
			ctf_params = data[im].get_attr("ctf")
			ctfimg = ctf_img(nx, ctf_params)
			if CUDA:
				all_ctf_params.extend([ctf_params.defocus, ctf_params.cs, ctf_params.voltage, ctf_params.apix, ctf_params.bfactor, ctf_params.ampcont])
			Util.add_img2(ctf_2_sum, ctfimg)
			Util.add_img_abs(ctf_abs_sum, ctfimg)
		if CUDA:
			alpha, sx, sy, mirror, scale = get_params2D(data[im])
			all_ali_params.extend([alpha, sx, sy, mirror])
		"""
'''2
def ORGali2d_c(stack, outdir, maskfile=None, ir=1, ou=-1, rs=1, xr="4 2 1 1", yr="-1", ts="2 1 0.5 0.25", center=-1, maxit=0, \
		CTF=False, snr=1.0, Fourvar = False, user_func_name="ref_ali2d", CUDA=False, GPU=0, MPI=False):
	if MPI:
		ali2d_c_MPI(stack, outdir, maskfile, ir, ou, rs, xr, yr, ts, center, maxit, CTF, snr, Fourvar, user_func_name, CUDA, GPU)
		return

	pass#IMPORTIMPORTIMPORT from utilities    import model_circle, drop_image, get_image, get_input_from_string, get_params2D
	pass#IMPORTIMPORTIMPORT from statistics   import fsc_mask, sum_oe, hist_list
	pass#IMPORTIMPORTIMPORT from alignment    import Numrinit, ringwe, ali2d_single_iter
	pass#IMPORTIMPORTIMPORT from pixel_error  import pixel_error_2D
	pass#IMPORTIMPORTIMPORT from filter       import filt_ctf, filt_table, filt_tophatb
	pass#IMPORTIMPORTIMPORT from fundamentals import fshift
	pass#IMPORTIMPORTIMPORT from utilities    import print_begin_msg, print_end_msg, print_msg
	pass#IMPORTIMPORTIMPORT from fundamentals import fft, rot_avg_table
	pass#IMPORTIMPORTIMPORT from utilities    import write_text_file, file_type
	pass#IMPORTIMPORTIMPORT import os
		
	print_begin_msg("ali2d_c")

	if os.path.exists(outdir):   ERROR('Output directory exists, please change the name and restart the program', " ORGali2d_c", 1)
	os.mkdir(outdir)

	pass#IMPORTIMPORTIMPORT import user_functions
	user_func = sparx_user_functions.factory[user_func_name]

	xrng        = get_input_from_string(xr)
	if  yr == "-1":  yrng = xrng
	else          :  yrng = get_input_from_string(yr)
	step        = get_input_from_string(ts)

	first_ring=int(ir); last_ring=int(ou); rstep=int(rs); max_iter=int(maxit);

	if max_iter == 0:
		max_iter  = 10
		auto_stop = True
	else:
		auto_stop = False

	print_msg("Input stack                 : %s\n"%(stack))
	print_msg("Output directory            : %s\n"%(outdir))
	print_msg("Inner radius                : %i\n"%(first_ring))

	if(file_type(stack) == "bdb"):
		pass#IMPORTIMPORTIMPORT from EMAN2db import db_open_dict
		dummy = db_open_dict(stack, True)
	active = EMUtil.get_all_attributes(stack, 'active')
	list_of_particles = []
	for im in xrange(len(active)):
		if active[im]:  list_of_particles.append(im)
	del active
	nima = len(list_of_particles)
	ima  = EMData()
	ima.read_image(stack, list_of_particles[0], True)
	nx = ima.get_xsize()

	# default value for the last ring
	if last_ring == -1:  last_ring = nx/2-2

	print_msg("Outer radius                : %i\n"%(last_ring))
	print_msg("Ring step                   : %i\n"%(rstep))
	print_msg("X search range              : %s\n"%(xrng))
	print_msg("Y search range              : %s\n"%(yrng))
	print_msg("Translational step          : %s\n"%(step))
	print_msg("Center type                 : %i\n"%(center))
	print_msg("Maximum iteration           : %i\n"%(max_iter))
	print_msg("Use Fourier variance        : %s\n"%(Fourvar))
	print_msg("CTF correction              : %s\n"%(CTF))
	print_msg("Signal-to-Noise Ratio       : %f\n"%(snr))
	if auto_stop:  print_msg("Stop iteration with         : criterion\n")
	else:           print_msg("Stop iteration with         : maxit\n")
	print_msg("User function               : %s\n"%(user_func_name))

	if maskfile:
		pass#IMPORTIMPORTIMPORT import	types
		if type(maskfile) is types.StringType:
			print_msg("Maskfile                    : %s\n\n"%(maskfile))
			mask=get_image(maskfile)
		else:
			print_msg("Maskfile                    : user provided in-core mask\n\n")
			mask = maskfile
	else :
		print_msg("Maskfile                    : default, a circle with radius %i\n\n"%(last_ring))
		mask = model_circle(last_ring, nx, nx)

	cnx = nx/2+1
 	cny = cnx
 	mode = "F"
	data = []
	if CTF:
		ctf_params = ima.get_attr("ctf")
		if ima.get_attr_default('ctf_applied', 0) > 0:	ERROR("data cannot be ctf-applied", "ORGali2d_c", 1)
		pass#IMPORTIMPORTIMPORT from morphology   import ctf_img
		ctf_2_sum = EMData(nx, nx, 1, False)
	else:
		ctf_2_sum = None
	if  Fourvar:
		pass#IMPORTIMPORTIMPORT from statistics   import add_ave_varf

	del ima
	data = EMData.read_images(stack, list_of_particles)
	for im in xrange(nima):
		data[im].set_attr('ID', list_of_particles[im])
		st = Util.infomask(data[im], mask, False)
		data[im] -= st[0]
		if CTF:
			ctf_params = data[im].get_attr("ctf")
	 		Util.add_img2(ctf_2_sum, ctf_img(nx, ctf_params))
	if(CTF and (not Fourvar)):  ctf_2_sum += 1.0/snr  # note this is complex addition (1.0/snr,0.0)
	# startup
	numr = Numrinit(first_ring, last_ring, rstep, mode) 	#precalculate rings
 	wr = ringwe(numr, mode)
	
	# initialize data for the reference preparation function
	#  mask can be modified in user_function
	ref_data = [mask, center, None, None]

	cs = [0.0]*2
	# iterate
	total_iter = 0
	a0 = -1e22
	sx_sum = 0.0
	sy_sum = 0.0
	for N_step in xrange(len(xrng)):
		msg = "\nX range = %5.2f   Y range = %5.2f   Step = %5.2f\n"%(xrng[N_step], yrng[N_step], step[N_step])
		print_msg(msg)
		for Iter in xrange(max_iter):
			total_iter += 1
			print_msg("Iteration #%4d\n"%(total_iter))
			if  Fourvar:  
				tavg, ave1, ave2, vav, sumsq = add_ave_varf(data, mask, "a", CTF, ctf_2_sum)
				# write the current average
				fft(tavg).write_image(os.path.join(outdir, "aqc.hdf"), total_iter-1)
				tavg    = fft(Util.divn_img(tavg, vav))

				vav_r	= Util.pack_complex_to_real(vav)
				vav_r.write_image(os.path.join(outdir, "varf.hdf"), total_iter-1)
				sumsq_r = Util.pack_complex_to_real(sumsq)
				rvar    = rot_avg_table(vav_r)
				rsumsq  = rot_avg_table(sumsq_r)
				frsc = []
				freq = []
				for i in xrange(len(rvar)):
					qt = max(0.0, rsumsq[i]/rvar[i] - 1.0)
					frsc.append(qt/(qt+1.0))
					freq.append(float(i)/(len(rvar)-1)*0.5)
				frsc = [freq, frsc]
				del freq
				write_text_file(frsc, os.path.join(outdir, "resolution%03d"%(total_iter)))
			else:
				ave1, ave2 = sum_oe(data, "a", CTF, ctf_2_sum)
				if CTF:  tavg = fft(Util.divn_img(fft(Util.addn_img(ave1, ave2)), ctf_2_sum))
				else:	 tavg = (ave1+ave2)/nima
				tavg.write_image(os.path.join(outdir, "aqc.hdf"), total_iter-1)
				frsc = fsc_mask(ave1, ave2, mask, 1.0, os.path.join(outdir, "resolution%03d"%(total_iter)))

			ref_data[2] = tavg
			ref_data[3] = frsc
			#  call user-supplied function to prepare reference image, i.e., center and filter it
			if center == -1:
				# When center = -1, which is by default, we use the average center method
				ref_data[1] = 0
				tavg, cs = user_func(ref_data)
				cs[0] = sx_sum/float(nima)
				cs[1] = sy_sum/float(nima)
				tavg = fshift(tavg, -cs[0], -cs[1])
				msg = "Average center x =      %10.3f        Center y       = %10.3f\n"%(cs[0], cs[1])
				print_msg(msg)
			else:
				tavg, cs = user_func(ref_data)

			# write the current filtered average
			tavg.write_image(os.path.join(outdir, "aqf.hdf"), total_iter-1)

			# a0 should increase; stop algorithm when it decreases.
			if Fourvar:  
				Util.div_filter(sumsq, vav)
				sumsq = filt_tophatb(sumsq, 0.01, 0.49)
				a1 = Util.infomask(sumsq, None, True)
				a1 = a1[0]
			else:
				a1 = tavg.cmp("dot", tavg, dict(negative = 0, mask = ref_data[0]))
			msg = "Criterion = %15.8e\n"%(a1)
			print_msg(msg)
			if total_iter == len(xrng)*max_iter: break
			if a1 < a0:
				if auto_stop == True: break
			else:	a0 = a1

			old_ali_params = []
		        for im in xrange(nima):
		        	alphan, sxn, syn, mirror, scale = get_params2D(data[im])
		        	old_ali_params.append([alphan, sxn, syn, mirror, scale])

			sx_sum, sy_sum = ali2d_single_iter(data, numr, wr, cs, tavg, cnx, cny, xrng[N_step], yrng[N_step], step[N_step], mode, CTF)

		        pixel_error = 0.0
		        mirror_changed = 0
			pixel_error_list = []
		        for im in xrange(nima):
		        	alphan, sxn, syn, mirror, scale = get_params2D(data[im]) 
		        	if old_ali_params[im][3] == mirror:
		        		this_error = pixel_error_2D(old_ali_params[im][0], old_ali_params[im][1], old_ali_params[im][2], alphan, sxn, syn, last_ring)
		        		pixel_error += this_error
					pixel_error_list.append(this_error)
		        	else:
		        		mirror_changed += 1
			print_msg("Mirror changed = %6.4f%%\n"%(float(mirror_changed)/nima*100))
			print_msg("Among the mirror consistent images, average pixel error is %0.4f, their distribution is:\n"%(pixel_error/float(nima-mirror_changed)))
 			region, hist = hist_list(pixel_error_list, 20)	
			for p in xrange(20):
				print_msg("      %8.4f: %5d\n"%(region[p], hist[p]))
			print_msg("\n\n\n")
			
	drop_image(tavg, os.path.join(outdir, "aqfinal.hdf"))
	# write out headers
	pass#IMPORTIMPORTIMPORT from utilities import write_headers
	write_headers(stack, data, list_of_particles)
	print_end_msg("ali2d_c")

def ORGali2d_c_MPI(stack, outdir, maskfile=None, ir=1, ou=-1, rs=1, xr="4 2 1 1", yr="-1", ts="2 1 0.5 0.25", center=-1, maxit=0, CTF=False, snr=1.0, \
			Fourvar = False, user_func_name="ref_ali2d", CUDA=False, GPU=0):

	pass#IMPORTIMPORTIMPORT from utilities    import model_circle, model_blank, drop_image, get_image, get_input_from_string
	pass#IMPORTIMPORTIMPORT from utilities    import reduce_EMData_to_root, bcast_EMData_to_all, send_attr_dict, file_type, bcast_number_to_all, bcast_list_to_all
	pass#IMPORTIMPORTIMPORT from statistics   import fsc_mask, sum_oe, add_ave_varf_MPI, hist_list
	pass#IMPORTIMPORTIMPORT from alignment    import Numrinit, ringwe, ali2d_single_iter
	pass#IMPORTIMPORTIMPORT from pixel_error  import pixel_error_2D
	pass#IMPORTIMPORTIMPORT from filter       import filt_table, filt_ctf, filt_tophatb
	pass#IMPORTIMPORTIMPORT from numpy        import reshape, shape
	pass#IMPORTIMPORTIMPORT from fundamentals import fshift, fft, rot_avg_table
	pass#IMPORTIMPORTIMPORT from utilities    import write_text_file, get_params2D, set_params2D
	pass#IMPORTIMPORTIMPORT from utilities    import print_msg, print_begin_msg, print_end_msg
	pass#IMPORTIMPORTIMPORT import os
	pass#IMPORTIMPORTIMPORT import sys
	pass#IMPORTIMPORTIMPORT from mpi 	  import mpi_init, mpi_comm_size, mpi_comm_rank, MPI_COMM_WORLD
	pass#IMPORTIMPORTIMPORT from mpi 	  import mpi_reduce, mpi_bcast, mpi_barrier, mpi_gatherv
	pass#IMPORTIMPORTIMPORT from mpi 	  import MPI_SUM, MPI_FLOAT, MPI_INT

	number_of_proc = mpi_comm_size(MPI_COMM_WORLD)
	myid = mpi_comm_rank(MPI_COMM_WORLD)
	main_node = 0
	
	ftp = file_type(stack)

	if os.path.exists(outdir):  ERROR('Output directory exists, please change the name and restart the program', "ORGali2d_c_MPI ", 1, myid)
	mpi_barrier(MPI_COMM_WORLD)

	if myid == main_node:
		print_begin_msg("ali2d_c_MPI")
		os.mkdir(outdir)
		pass#IMPORTIMPORTIMPORT import global_def
		sparx_global_def.LOGFILE =  os.path.join(outdir, sparx_global_def.LOGFILE)

	xrng        = get_input_from_string(xr)
	if  yr == "-1":  yrng = xrng
	else          :  yrng = get_input_from_string(yr)
	step        = get_input_from_string(ts)
	
	first_ring=int(ir); last_ring=int(ou); rstep=int(rs); max_iter=int(maxit);
	
	if max_iter == 0:
		max_iter = 10
		auto_stop = True
	else:
		auto_stop = False

	if myid == main_node:
		pass#IMPORTIMPORTIMPORT import user_functions
		user_func = sparx_user_functions.factory[user_func_name]
		print_msg("Input stack                 : %s\n"%(stack))
		print_msg("Output directory            : %s\n"%(outdir))
		print_msg("Inner radius                : %i\n"%(first_ring))
	
	if myid == main_node:
       		if(file_type(stack) == "bdb"):
			pass#IMPORTIMPORTIMPORT from EMAN2db import db_open_dict
			dummy = db_open_dict(stack, True)
		active = EMUtil.get_all_attributes(stack, 'active')
		list_of_particles = []
		for im in xrange(len(active)):
			if active[im]:  list_of_particles.append(im)
		del active
		nima = len(list_of_particles)
	else:
		nima = 0
	nima = bcast_number_to_all(nima, source_node = main_node)
	
	if myid != main_node:
		list_of_particles = [-1]*nima
	list_of_particles = bcast_list_to_all(list_of_particles, myid,  source_node = main_node)
	
	image_start, image_end = MPI_start_end(nima, number_of_proc, myid)
	list_of_particles = list_of_particles[image_start: image_end]

	ima = EMData()
	for i in xrange(number_of_proc):
		if myid == i:
			ima.read_image(stack, list_of_particles[0], True)
		if ftp == "bdb": mpi_barrier(MPI_COMM_WORLD)
	nx = ima.get_xsize()
	# default value for the last ring
	if last_ring == -1: last_ring = nx/2-2

	if myid == main_node:
		print_msg("Outer radius                : %i\n"%(last_ring))
		print_msg("Ring step                   : %i\n"%(rstep))
		print_msg("X search range              : %s\n"%(xrng))
		print_msg("Y search range              : %s\n"%(yrng))
		print_msg("Translational step          : %s\n"%(step))
		print_msg("Center type                 : %i\n"%(center))
		print_msg("Maximum iteration           : %i\n"%(max_iter))
		print_msg("Use Fourier variance        : %s\n"%(Fourvar))
		print_msg("CTF correction              : %s\n"%(CTF))
		print_msg("Signal-to-Noise Ratio       : %f\n"%(snr))
		if auto_stop:
			print_msg("Stop iteration with         : criterion\n")
		else:
			print_msg("Stop iteration with         : maxit\n")
		print_msg("User function               : %s\n"%(user_func_name))
		print_msg("Number of processors used   : %d\n"%(number_of_proc))
		print_msg("Using CUDA                  : %s\n"%(CUDA))
		print_msg("Number of GPUs              : %d\n"%(GPU))

		
	if maskfile:
		pass#IMPORTIMPORTIMPORT import  types
		if type(maskfile) is types.StringType:  
			if myid == main_node:		print_msg("Maskfile                    : %s\n\n"%(maskfile))
			mask = get_image(maskfile)
		else:
			if myid == main_node: 		print_msg("Maskfile                    : user provided in-core mask\n\n")
			mask = maskfile
	else : 
		if myid == main_node: 	print_msg("Maskfile                    : default, a circle with radius %i\n\n"%(last_ring))
		mask = model_circle(last_ring, nx, nx)

	cnx  = nx/2+1
 	cny  = cnx
 	mode = "F"
	data = []
	if CTF:
		ctf_params = ima.get_attr("ctf")
		if ima.get_attr_default('ctf_applied', 0) > 0:	ERROR("data cannot be ctf-applied", "ORGali2d_c_MPI", 1,myid)
		pass#IMPORTIMPORTIMPORT from filter import filt_ctf
		pass#IMPORTIMPORTIMPORT from morphology   import ctf_img
		ctf_2_sum = EMData(nx, nx, 1, False)
	else:
		ctf_2_sum = None
	if  Fourvar:
		pass#IMPORTIMPORTIMPORT from statistics   import add_ave_varf

	del ima

	for i in xrange(number_of_proc):
		if myid == i: 
			data = EMData.read_images(stack, list_of_particles)
		if ftp == "bdb": mpi_barrier(MPI_COMM_WORLD)
	
	if CUDA:
		GPUID = myid%GPU
		all_ali_params = []
		all_ctf_params = []
	for im in xrange(len(data)):
		data[im].set_attr('ID', list_of_particles[im])
		st = Util.infomask(data[im], mask, False)
		data[im] -= st[0]
		if CTF:
			ctf_params = data[im].get_attr("ctf")
			Util.add_img2(ctf_2_sum, ctf_img(nx, ctf_params))
			if CUDA:
				all_ctf_params.append(ctf_params.defocus)
				all_ctf_params.append(ctf_params.cs)
				all_ctf_params.append(ctf_params.voltage)
				all_ctf_params.append(ctf_params.apix)
				all_ctf_params.append(ctf_params.bfactor)
				all_ctf_params.append(ctf_params.ampcont)
		if CUDA:
			alpha, sx, sy, mirror, scale = get_params2D(data[im])
			all_ali_params.append(alpha)
			all_ali_params.append(sx)
			all_ali_params.append(sy)
			all_ali_params.append(mirror)

	if CTF:
		reduce_EMData_to_root(ctf_2_sum, myid, main_node)
		if(not Fourvar):  ctf_2_sum += 1.0/snr  # note this is complex addition (1.0/snr,0.0)
	else:  ctf_2_sum = None
	# startup
	numr = Numrinit(first_ring, last_ring, rstep, mode) 	#precalculate rings
 	wr = ringwe(numr, mode)
	
	if myid == main_node:
		# initialize data for the reference preparation function
		ref_data = [mask, center, None, None]
		sx_sum = 0.0
		sy_sum = 0.0
		a0 = -1.0e22
		
	recvcount = []
	disp = []
	for i in xrange(number_of_proc):
		ib, ie = MPI_start_end(nima, number_of_proc, i)
		recvcount.append(ie-ib)
		if i == 0:
			disp.append(0)
		else:
			disp.append(disp[i-1]+recvcount[i-1])

	again = True
	total_iter = 0
	cs = [0.0]*2

	for N_step in xrange(len(xrng)):

		if CUDA:
			R = CUDA_Aligner()
			R.setup(len(data), nx, nx, 256, 32, last_ring, step[N_step], int(xrng[N_step]/step[N_step]+0.5), int(yrng[N_step]/step[N_step]+0.5), CTF)
			for im in xrange(len(data)):	R.insert_image(data[im], im)
			if CTF:  R.filter_stack(all_ctf_params, GPUID)
					
		msg = "\nX range = %5.2f   Y range = %5.2f   Step = %5.2f\n"%(xrng[N_step], yrng[N_step], step[N_step])
		if myid == main_node: print_msg(msg)
		for Iter in xrange(max_iter):
			total_iter += 1
			if  Fourvar:  
				tavg, ave1, ave2, vav, sumsq = add_ave_varf_MPI(myid, data, mask, "a", CTF, ctf_2_sum)
			else:
				if CUDA:
					ave1 = model_blank(nx, nx)
					ave2 = model_blank(nx, nx)
					R.sum_oe(all_ctf_params, all_ali_params, ave1, ave2, GPUID)
				else:
					ave1, ave2 = sum_oe(data, "a", CTF, EMData())  # pass empty object to prevent calculation of ctf^2
				reduce_EMData_to_root(ave1, myid, main_node)
				reduce_EMData_to_root(ave2, myid, main_node)

			if myid == main_node:
				print_msg("Iteration #%4d\n"%(total_iter))
				if Fourvar:
					fft(tavg).write_image(os.path.join(outdir, "aqc.hdf"), total_iter-1)
					tavg    = fft(Util.divn_img(tavg, vav))
					vav_r	= Util.pack_complex_to_real(vav)
					vav_r.write_image(os.path.join(outdir, "varf.hdf"), total_iter-1)
					rvar	= rot_avg_table(vav_r)
					rsumsq  = rot_avg_table(Util.pack_complex_to_real(sumsq))
					frsc = []
					freq = []
					for i in xrange(len(rvar)):
						qt = max(0.0, rsumsq[i]/rvar[i] - 1.0)
						frsc.append(qt/(qt+1.0))
						freq.append(float(i)/(len(rvar)-1)*0.5)
					frsc = [freq, frsc]
					del freq
					write_text_file(frsc, os.path.join(outdir, "resolution%03d"%(total_iter)) )
				else:
					if CTF:  tavg = fft(Util.divn_img(fft(Util.addn_img(ave1, ave2)), ctf_2_sum))
					else:	 tavg = (ave1+ave2)/nima
					tavg.write_image(os.path.join(outdir, "aqc.hdf"), total_iter-1)
					frsc = fsc_mask(ave1, ave2, mask, 1.0, os.path.join(outdir, "resolution%03d"%(total_iter)))

				ref_data[2] = tavg
				ref_data[3] = frsc
				
				#  call user-supplied function to prepare reference image, i.e., center and filter it
				if center == -1:
					# When center = -1, which is by default, we use the average center method
					ref_data[1] = 0
					tavg, cs = user_func(ref_data)
					cs[0] = float(sx_sum)/nima
					cs[1] = float(sy_sum)/nima
					tavg = fshift(tavg, -cs[0], -cs[1])
					msg = "Average center x =      %10.3f        Center y       = %10.3f\n"%(cs[0], cs[1])
					print_msg(msg)
				else:
					tavg, cs = user_func(ref_data)

				# write the current filtered average
				tavg.write_image(os.path.join(outdir, "aqf.hdf"), total_iter-1)

				# a0 should increase; stop algorithm when it decreases.    
				if Fourvar:  
					Util.div_filter(sumsq, vav)
					sumsq = filt_tophatb(sumsq, 0.01, 0.49)
					a1 = Util.infomask(sumsq, None, True)
					a1 = a1[0]
				else:
					a1 = tavg.cmp("dot", tavg, dict(negative = 0, mask = ref_data[0]))
				msg = "Criterion %d = %15.8e\n"%(total_iter, a1)
				print_msg(msg)
				
				if a1 < a0:
					if auto_stop: 
						again = False
						break
				else:	a0 = a1
			else:
				tavg = EMData(nx, nx, 1, True)
				cs = [0.0]*2

			bcast_EMData_to_all(tavg, myid, main_node)
			cs = mpi_bcast(cs, 2, MPI_FLOAT, main_node, MPI_COMM_WORLD)
			cs = map(float, cs)
			if auto_stop:
				again = mpi_bcast(again, 1, MPI_INT, main_node, MPI_COMM_WORLD)
				if not again: break
			if total_iter != max_iter*len(xrng):
				if CUDA:
					old_ali_params = all_ali_params[:]
				else:
					old_ali_params = []
				        for im in xrange(len(data)):  
						alpha, sx, sy, mirror, scale = get_params2D(data[im])
						old_ali_params.append(alpha)
						old_ali_params.append(sx)
						old_ali_params.append(sy)
						old_ali_params.append(mirror)

				if CUDA:
					all_ali_params = R.ali2d_single_iter(tavg, all_ali_params, cs[0], cs[1], GPUID, 1)
					sx_sum = all_ali_params[-2]
					sy_sum = all_ali_params[-1]
					for im in xrange(len(data)):  all_ali_params[im*4+3] = int(all_ali_params[im*4+3])
				else:	
					sx_sum, sy_sum, nope = ali2d_single_iter(data, numr, wr, cs, tavg, cnx, cny, xrng[N_step], yrng[N_step], step[N_step], mode, CTF)
					
				sx_sum = mpi_reduce(sx_sum, 1, MPI_FLOAT, MPI_SUM, main_node, MPI_COMM_WORLD)
				sy_sum = mpi_reduce(sy_sum, 1, MPI_FLOAT, MPI_SUM, main_node, MPI_COMM_WORLD)

			        pixel_error = 0.0
			        mirror_consistent = 0
				pixel_error_list = []
			        for im in xrange(len(data)):
			        	if CUDA:
						alpha = all_ali_params[im*4]
						sx = all_ali_params[im*4+1]
						sy = all_ali_params[im*4+2]
						mirror = all_ali_params[im*4+3]
					else:
						alpha, sx, sy, mirror, scale = get_params2D(data[im]) 
			        	if old_ali_params[im*4+3] == mirror:
		        			this_error = pixel_error_2D(old_ali_params[im*4:im*4+3], [alpha, sx, sy], last_ring)
		        			pixel_error += this_error
						pixel_error_list.append(this_error)
						mirror_consistent += 1
					else:
						pixel_error_list.append(-1)
				mirror_consistent = mpi_reduce(mirror_consistent, 1, MPI_INT, MPI_SUM, main_node, MPI_COMM_WORLD)
				pixel_error = mpi_reduce(pixel_error, 1, MPI_FLOAT, MPI_SUM, main_node, MPI_COMM_WORLD)
				pixel_error_list = mpi_gatherv(pixel_error_list, len(data), MPI_FLOAT, recvcount, disp, MPI_FLOAT, main_node, MPI_COMM_WORLD)
				if myid == main_node:
					print_msg("Mirror consistent rate = %8.4f%%\n"%(float(mirror_consistent)/nima*100))
					print_msg("Among the mirror consistent images, average pixel error is %0.4f, their distribution is:\n"%(float(pixel_error)/mirror_consistent))
					pixel_error_list = map(float, pixel_error_list)
					for i in xrange(nima-1, -1, -1):
						if pixel_error_list[i] < 0:  del pixel_error_list[i]
					region, hist = hist_list(pixel_error_list, 20)	
					for p in xrange(20):
						print_msg("      %10.6f: %5d\n"%(region[p], hist[p]))
					print_msg("\n\n\n")
		if CUDA: R.finish()

	if CUDA:
		for im in xrange(len(data)):
			set_params2D(data[im], [all_ali_params[im*4], all_ali_params[im*4+1], all_ali_params[im*4+2], all_ali_params[im*4+3], 1.0])

	if myid == main_node:  drop_image(tavg, os.path.join(outdir, "aqfinal.hdf"))
	# write out headers and STOP, under MPI writing has to be done sequentially
	mpi_barrier(MPI_COMM_WORLD)
	par_str = ["xform.align2d", "ID"]
	if myid == main_node:
		pass#IMPORTIMPORTIMPORT from utilities import file_type
		if(file_type(stack) == "bdb"):
			pass#IMPORTIMPORTIMPORT from utilities import recv_attr_dict_bdb
			recv_attr_dict_bdb(main_node, stack, data, par_str, image_start, image_end, number_of_proc)
		else:
			pass#IMPORTIMPORTIMPORT from utilities import recv_attr_dict
			recv_attr_dict(main_node, stack, data, par_str, image_start, image_end, number_of_proc)
	else:           send_attr_dict(main_node, data, par_str, image_start, image_end)
	if myid == main_node: print_end_msg("ali2d_c_MPI")
'''
'''3
def ali3d_abandoned(stack, ref_vol, outdir, maskfile = None, ir = 1, ou = -1, rs = 1, 
            xr = "4 2 2 1", yr = "-1", ts = "1 1 0.5 0.25", delta = "10 6 4 4", an = "-1", apsi = "-1", deltapsi = "-1", startpsi = "-1",
	    center = -1, maxit = 5, CTF = False, snr = 1.0,  ref_a = "S", sym = "c1",
	    user_func_name = "ref_ali3d", fourvar = True, npad = 4, debug = False, MPI = False, termprec = 0.0):
	"""
		Name
			ali3d - Perform 3-D projection matching given initial reference volume and image series
		Input
			stack: set of 2-D images in a stack file, images have to be squares
			ref_vol: initial reference volume
			outdir: directory name into which the results will be written
			maskfile: filename of the file containing 3D mask.
			ir: inner radius for rotational correlation > 0 
			ou: outer radius for rotational correlation <int(nx/2)-1 
			rs: steps between rings in rotational correlation >0
			xr: range for translation search in x direction in each iteration, search is +/xr
			yr: range for translation search in y direction in each iteration, search is +/yr
			ts: step size of the translation search in both directions, search is -xr, -xr+ts, 0, xr-ts, xr, can be fractional.
			delta: angular step for the reference projections in respective iterations
			an: angular neighborhood for local searches
			center: average center method
			max_iter: maximum iterations at each angle step
			CTF: if the flag is present, program will use the CTF information stored in file headers
			snr: signal noise ratio used in the 3D reconstruction
			ref_a: method for creating quasi-uniform distribution of the projection directions of reference projections: "S" - spiral
			sym: symmetry of the refined structure
			function: name of the user-supplied-function
			MPI: if presetm use MPI version
		Output
			output_directory: directory name into which the output files will be written.
			header: the alignment parameters are stored in the headers of input files as Transform Object xform.proj
	"""
	if MPI:
		ali3d_MPI(stack, ref_vol, outdir, maskfile, ir, ou, rs, xr, yr, ts,
	        	delta, an, apsi, deltapsi, startpsi, center, maxit, CTF, snr, ref_a, sym, user_func_name,
			fourvar, npad, debug, termprec)
		return

	pass#IMPORTIMPORTIMPORT from alignment      import proj_ali_incore, proj_ali_incore_local
	pass#IMPORTIMPORTIMPORT from utilities      import model_circle, drop_image, get_image, get_input_from_string
	pass#IMPORTIMPORTIMPORT from utilities      import get_params_proj
	pass#IMPORTIMPORTIMPORT from utilities      import estimate_3D_center, rotate_3D_shift
	pass#IMPORTIMPORTIMPORT from filter         import filt_params, fit_tanh, filt_tanl, filt_ctf
	pass#IMPORTIMPORTIMPORT from statistics     import fsc_mask
	pass#IMPORTIMPORTIMPORT from utilities      import print_begin_msg, print_end_msg, print_msg
	pass#IMPORTIMPORTIMPORT from alignment      import Numrinit, prepare_refrings
	pass#IMPORTIMPORTIMPORT from projection     import prep_vol

	pass#IMPORTIMPORTIMPORT import user_functions
	pass#IMPORTIMPORTIMPORT import os
	pass#IMPORTIMPORTIMPORT import types
	pass#IMPORTIMPORTIMPORT from math			import radians, sin, cos

	user_func = sparx_user_functions.factory[user_func_name]

	if os.path.exists(outdir):  ERROR('Output directory exists, please change the name and restart the program', "ali3d", 1)
	os.mkdir(outdir)
	pass#IMPORTIMPORTIMPORT import global_def
	sparx_global_def.LOGFILE =  os.path.join(outdir, sparx_global_def.LOGFILE)
	print_begin_msg("ali3d")

	xrng        = get_input_from_string(xr)
	if  yr == "-1":  yrng = xrng
	else          :  yrng = get_input_from_string(yr)
	step        = get_input_from_string(ts)
	delta       = get_input_from_string(delta)
	lstp = min(len(xrng), len(yrng), len(step), len(delta))
	if an == "-1":
		an = [-1] * lstp
	else:
		an = get_input_from_string(an)
	first_ring  = int(ir)
	rstep       = int(rs)
	last_ring   = int(ou)
	max_iter    = int(maxit)
	center      = int(center)

	print_msg("Input stack                 : %s\n"%(stack))
	print_msg("Reference volume            : %s\n"%(ref_vol))	
	print_msg("Output directory            : %s\n"%(outdir))
	print_msg("Maskfile                    : %s\n"%(maskfile))
	print_msg("Inner radius                : %i\n"%(first_ring))

	vol     = EMData()
	vol.read_image(ref_vol)
	nx      = vol.get_xsize()
	if last_ring == -1:	last_ring = nx/2 - 2

	print_msg("Outer radius                : %i\n"%(last_ring))
	print_msg("Ring step                   : %i\n"%(rstep))
	print_msg("X search range              : %s\n"%(xrng))
	print_msg("Y search range              : %s\n"%(yrng))
	print_msg("Translational step          : %s\n"%(step))
	print_msg("Angular step                : %s\n"%(delta))
	print_msg("Angular search range        : %s\n"%(an))
	print_msg("Maximum iteration           : %i\n"%(max_iter))
	print_msg("Center type                 : %i\n"%(center))
	print_msg("CTF correction              : %s\n"%(CTF))
	print_msg("Signal-to-Noise Ratio       : %f\n"%(snr))
	print_msg("Reference projection method : %s\n"%(ref_a))
	print_msg("Symmetry group              : %s\n\n"%(sym))
	print_msg("User function               : %s\n"%(user_func_name))

	if maskfile :
		if type(maskfile) is types.StringType: mask3D = get_image(maskfile)
		else                                  : mask3D = maskfile
	else          :   mask3D = model_circle(last_ring, nx, nx, nx)
	mask2D = model_circle(last_ring, nx, nx) - model_circle(first_ring, nx, nx)
	numr   = Numrinit(first_ring, last_ring, rstep, "F")

	if CTF:
		pass#IMPORTIMPORTIMPORT from reconstruction import recons3d_4nn_ctf
		pass#IMPORTIMPORTIMPORT from filter         import filt_ctf
	pass#IMPORTIMPORTIMPORT else: from reconstruction import recons3d_4nn

	if debug:  outf = file(os.path.join(outdir, "progress"), "w")
	else:      outf = None

	active = EMUtil.get_all_attributes(stack, 'active')
	list_of_particles = []
	for im in xrange(len(active)):
		if(active[im]):  list_of_particles.append(im)
	del active
	data = EMData.read_images(stack, list_of_particles)
        for im in xrange(len(data)):
                data[im].set_attr('ID', list_of_particles[im])
		if CTF:
			ctf_params = data[im].get_attr("ctf")
			st = Util.infomask(data[im], mask2D, False)
			data[im] -= st[0]
			data[im] = filt_ctf(data[im], ctf_params)
			data[im].set_attr('ctf_applied', 1)

	nima = len(data)
	# initialize data for the reference preparation function
	ref_data = [ mask3D, max(center,0), None, None ]#  for center -1 switch of centering by user function

	cs = [0.0]*3
	# do the projection matching
	for N_step in xrange(lstp):
 		for Iter in xrange(max_iter):
			print_msg("\nITERATION #%3d\n"%(N_step*max_iter+Iter+1))
			ref_angles = even_angles(delta[N_step], symmetry=sym, method = ref_a, phiEqpsi = "Minus")

			pixer  = [0.0]*nima
			neworient = [[0.0, 0.0, 0.0, 0.0, 0.0, -2.0e23] for i in xrange(nima)]
			Torg = []
			pikorg = [0.0]*nima
			for im in xrange( nima ):
				Torg.append(data[im].get_attr('xform.projection'))
				pikorg[im] = data[im].get_attr_default("previousmax",-1.0e23)

			for refang in ref_angles:
				n1 = sin(radians(refang[1]))*cos(radians(refang[0]))
				n2 = sin(radians(refang[1]))*sin(radians(refang[0]))
				n3 = cos(radians(refang[1]))

				for im in xrange(nima):
					volft, kb = prep_vol(vol)
					refrings = [None]
				
					if an[N_step] == -1:
						if(refrings[0] == None): refrings = refprojs( volft, kb, [refang], numr, mode, wr )
						peak, pixel_error = proj_ali_incore(data[im], refrings, numr, xrng[N_step], yrng[N_step], step[N_step], sym=sym)
						if(peak > neworient[im][-1]):
							# I stopped here realizing it would include conversion of data[im] to polar coords at the bottom of the loop,
							#    thus significantly slowing down the code.
							pass
					else:
						if(comparedirections):
							if(refrings[0] == None):
								refrings = refprojs( volft, kb, [refang], numr, mode, wr )
						peak, pixel_error = proj_ali_incore_local(data[im],refrings,numr,xrng[N_step],yrng[N_step],step[N_step],an[N_step],sym=sym)
					data[im].set_attr("previousmax", peak)
			if center == -1 and sym[0] == 'c':
				cs[0], cs[1], cs[2], dummy, dummy = estimate_3D_center(data)
				msg = "Average center x = %10.3f        Center y = %10.3f        Center z = %10.3f\n"%(cs[0], cs[1], cs[2])
				print_msg(msg)
				if int(sym[1]) > 1:
					cs[0] = cs[1] = 0.0
					print_msg("For symmetry group cn (n>1), we only center the volume in z-direction\n")
				rotate_3D_shift(data, [-cs[0], -cs[1], -cs[2]])

			del volft, kb

			if CTF:   vol1 = recons3d_4nn_ctf(data, range(0, nima, 2), snr, 1, sym)
			else:	   vol1 = recons3d_4nn(data, range(0, nima, 2), sym)
			if CTF:   vol2 = recons3d_4nn_ctf(data, range(1, nima, 2), snr, 1, sym)
			else:	   vol2 = recons3d_4nn(data, range(1, nima, 2), sym)

			fscc = fsc_mask(vol1, vol2, mask3D, 1.0, os.path.join(outdir, "resolution%04d"%(N_step*max_iter+Iter+1)))
			del vol1
			del vol2

			# calculate new and improved 3D
			if CTF:  vol = recons3d_4nn_ctf(data, range(nima), snr, 1, sym)
			else:	 vol = recons3d_4nn(data, range(nima), sym)
			# store the reference volume
			drop_image(vol, os.path.join(outdir, "vol%04d.hdf"%(N_step*max_iter+Iter+1)))
			ref_data[2] = vol
			ref_data[3] = fscc

			#  call user-supplied function to prepare reference image, i.e., center and filter it
			vol, dummy = user_func(ref_data)

			drop_image(vol, os.path.join(outdir, "volf%04d.hdf"%(N_step*max_iter+Iter+1)))
			#  here we write header info
			pass#IMPORTIMPORTIMPORT from utilities import write_headers
			#from utilities import write_select_headers
			if CTF:
				for dat in data:  dat.set_attr('ctf_applied',0)
			write_headers(stack, data, list_of_particles)
			#list_params= ['ID','xform.projection']
			#write_select_headers(stack, data, list_params)
			if CTF:
				for dat in data:  dat.set_attr('ctf_applied', 1)
	print_end_msg("ali3d")
'''
'''4
def Xali3d_MPI_chunks(stack, ref_vol, outdir, maskfile = None, ir = 1, ou = -1, rs = 1, 
            xr = "4 2 2 1", yr = "-1", ts = "1 1 0.5 0.25", delta = "10 6 4 4", an = "-1", apsi = "-1", deltapsi = "-1", startpsi = "-1",
	    center = -1, maxit = 5, CTF = False, snr = 1.0,  ref_a = "S", sym = "c1",  user_func_name = "ref_ali3d",
	    fourvar = True, npad = 4, debug = False, termprec = 0.0):

	pass#IMPORTIMPORTIMPORT from alignment       import Numrinit, prepare_refrings, proj_ali_incore, proj_ali_incore_local, proj_ali_incore_local_psi
	pass#IMPORTIMPORTIMPORT from utilities       import model_circle, get_image, drop_image, get_input_from_string
	pass#IMPORTIMPORTIMPORT from utilities       import bcast_list_to_all, bcast_number_to_all, reduce_EMData_to_root, bcast_EMData_to_all
	pass#IMPORTIMPORTIMPORT from utilities       import send_attr_dict
	pass#IMPORTIMPORTIMPORT from utilities       import get_params_proj, file_type
	pass#IMPORTIMPORTIMPORT from fundamentals    import rot_avg_image
	pass#IMPORTIMPORTIMPORT import os
	pass#IMPORTIMPORTIMPORT import types
	pass#IMPORTIMPORTIMPORT from utilities       import print_begin_msg, print_end_msg, print_msg
	pass#IMPORTIMPORTIMPORT from mpi             import mpi_bcast, mpi_comm_size, mpi_comm_rank, MPI_FLOAT, MPI_COMM_WORLD, mpi_barrier
	pass#IMPORTIMPORTIMPORT from mpi             import mpi_reduce, MPI_INT, MPI_SUM
	pass#IMPORTIMPORTIMPORT from filter          import filt_ctf
	pass#IMPORTIMPORTIMPORT from projection      import prep_vol, prgs
	pass#IMPORTIMPORTIMPORT from statistics      import hist_list, varf3d_MPI
	pass#IMPORTIMPORTIMPORT from applications    import MPI_start_end


	number_of_proc = mpi_comm_size(MPI_COMM_WORLD)
	myid           = mpi_comm_rank(MPI_COMM_WORLD)
	main_node = 0
	
	if os.path.exists(outdir):  ERROR('Output directory exists, please change the name and restart the program', "ali3d_MPI", 1, myid)
	mpi_barrier(MPI_COMM_WORLD)

	if myid == main_node:
		os.mkdir(outdir)
		pass#IMPORTIMPORTIMPORT import global_def
		sparx_global_def.LOGFILE =  os.path.join(outdir, sparx_global_def.LOGFILE)
		print_begin_msg("ali3d_MPI")
	mpi_barrier(MPI_COMM_WORLD)

	if debug:
		pass#IMPORTIMPORTIMPORT from time import sleep
		while not os.path.exists(outdir):
			print  "Node ",myid,"  waiting..."
			sleep(5)

		info_file = os.path.join(outdir, "progress%04d"%myid)
		finfo = open(info_file, 'w')
	else:
		finfo = None

	xrng        = get_input_from_string(xr)
	if  yr == "-1":  yrng = xrng
	else          :  yrng = get_input_from_string(yr)
	step        = get_input_from_string(ts)
	delta       = get_input_from_string(delta)
	lstp = min(len(xrng), len(yrng), len(step), len(delta))
	if an == "-1":
		an = [-1] * lstp
	else:
		an = get_input_from_string(an)

	if apsi == "-1":
		apsi = [-1] * lstp
	else:
		apsi = get_input_from_string(apsi)

	if deltapsi == "-1":
		deltapsi = [-1] * lstp
	else:
		deltapsi = get_input_from_string(deltapsi)

	if startpsi == "-1":
		startpsi = [-1] * lstp
	else:
		startpsi = get_input_from_string(startpsi)

	first_ring  = int(ir)
	rstep       = int(rs)
	last_ring   = int(ou)
	max_iter    = int(maxit)
	center      = int(center)

	vol     = EMData()
	vol.read_image(ref_vol)
	nx      = vol.get_xsize()
	if last_ring < 0:	last_ring = int(nx/2) - 2

	if myid == main_node:
		pass#IMPORTIMPORTIMPORT import user_functions
		user_func = sparx_user_functions.factory[user_func_name]

		print_msg("Input stack                 : %s\n"%(stack))
		print_msg("Reference volume            : %s\n"%(ref_vol))	
		print_msg("Output directory            : %s\n"%(outdir))
		print_msg("Maskfile                    : %s\n"%(maskfile))
		print_msg("Inner radius                : %i\n"%(first_ring))
		print_msg("Outer radius                : %i\n"%(last_ring))
		print_msg("Ring step                   : %i\n"%(rstep))
		print_msg("X search range              : %s\n"%(xrng))
		print_msg("Y search range              : %s\n"%(yrng))
		print_msg("Translational step          : %s\n"%(step))
		print_msg("Angular step                : %s\n"%(delta))
		print_msg("Angular search range (phi and theta)       : %s\n"%(an))
		print_msg("Angular search range (psi)                 : %s\n"%(apsi))
		print_msg("Delta psi                   : %s\n"%(deltapsi))
		print_msg("Start psi                   : %s\n"%(startpsi))
		print_msg("Maximum iteration           : %i\n"%(max_iter))
		print_msg("Percentage of change for termination: %f\n"%(termprec))
		print_msg("Center type                 : %i\n"%(center))
		print_msg("CTF correction              : %s\n"%(CTF))
		print_msg("Signal-to-Noise Ratio       : %f\n"%(snr))
		print_msg("Reference projection method : %s\n"%(ref_a))
		print_msg("Symmetry group              : %s\n\n"%(sym))
		print_msg("User function               : %s\n"%(user_func_name))

	if maskfile:
		if type(maskfile) is types.StringType: mask3D = get_image(maskfile)
		else:                                  mask3D = maskfile
	else: mask3D = model_circle(last_ring, nx, nx, nx)

	numr	= Numrinit(first_ring, last_ring, rstep, "F")
	mask2D  = model_circle(last_ring,nx,nx) - model_circle(first_ring,nx,nx)

	fscmask = mask3D  #model_circle(last_ring,nx,nx,nx)  For a fancy mask circle would work better  PAP 7/21/11
	if CTF:
		pass#IMPORTIMPORTIMPORT from reconstruction import rec3D_MPI
		pass#IMPORTIMPORTIMPORT from filter         import filt_ctf
	pass#IMPORTIMPORTIMPORT else:	 from reconstruction import rec3D_MPI_noCTF

	if myid == main_node:
       		if file_type(stack) == "bdb":
			pass#IMPORTIMPORTIMPORT from EMAN2db import db_open_dict
			dummy = db_open_dict(stack, True)
		active = EMUtil.get_all_attributes(stack, 'active')
		list_of_particles = []
		for im in xrange(len(active)):
			if active[im]:  list_of_particles.append(im)
		del active
		nima = len(list_of_particles)
	else:
		nima = 0
	total_nima = bcast_number_to_all(nima, source_node = main_node)

	if myid != main_node:
		list_of_particles = [-1]*total_nima
	list_of_particles = bcast_list_to_all(list_of_particles, myid,  source_node = main_node)

	image_start, image_end = MPI_start_end(total_nima, number_of_proc, myid)
	# create a list of images for each node
	list_of_particles = list_of_particles[image_start: image_end]
	nima = len(list_of_particles)
	if debug:
		finfo.write("image_start, image_end: %d %d\n" %(image_start, image_end))
		finfo.flush()

	data = EMData.read_images(stack, list_of_particles)
	if fourvar:  original_data = []
	for im in xrange(nima):
		data[im].set_attr('ID', list_of_particles[im])
		if fourvar: original_data.append(data[im].copy())
		if CTF:
			ctf_params = data[im].get_attr("ctf")
			st = Util.infomask(data[im], mask2D, False)
			data[im] -= st[0]
			data[im] = filt_ctf(data[im], ctf_params)
			data[im].set_attr('ctf_applied', 1)

	if debug:
		finfo.write( '%d loaded  \n' % nima )
		finfo.flush()
	if myid == main_node:
		# initialize data for the reference preparation function
		ref_data = [ mask3D, max(center,0), None, None, None, None ]
		# for method -1, switch off centering in user function

	pass#IMPORTIMPORTIMPORT from time import time

	#  this is needed for gathering of pixel errors
	disps = []
	recvcount = []
	for im in xrange(number_of_proc):
		if im == main_node :  disps.append(0)
		else:                 disps.append(disps[im-1] + recvcount[im-1])
		ib, ie = MPI_start_end(total_nima, number_of_proc, im)
		recvcount.append(ie-ib)

	pixer = [0.0]*nima
	cs = [0.0]*3
	total_iter = 0
	# do the projection matching
	for N_step in xrange(lstp):
		terminate = 0
		Iter = -1
 		while Iter < max_iter-1 and terminate == 0:
			Iter += 1
			total_iter += 1
			if myid == main_node:
				start_time = time()
				print_msg("\nITERATION #%3d,  inner iteration #%3d\nDelta = %4.1f, an = %5.2f, xrange = %5.2f, yrange = %5.2f, step = %5.2f, delta psi = %5.2f, start psi = %5.2f\n"%(total_iter, Iter, delta[N_step], an[N_step], xrng[N_step],yrng[N_step],step[N_step],deltapsi[N_step],startpsi[N_step]))

			volft, kb = prep_vol(vol)
			refrings = prepare_refrings_chunks(volft, kb, nx, delta[N_step], ref_a, sym, numr, True, ant = max(an[N_step],0.0)*1.1)  # 1.1 is to have extra safety
			del volft, kb
			if myid == main_node:
				print_msg("Time to prepare rings: %d\n" % (time()-start_time))
				start_time = time()

			for im in xrange(nima):
				if deltapsi[N_step] > 0.0:
					pass#IMPORTIMPORTIMPORT from alignment import proj_ali_incore_delta
					peak, pixer[im] = proj_ali_incore_delta(data[im],refrings,numr,xrng[N_step],yrng[N_step],step[N_step],startpsi[N_step],deltapsi[N_step],finfo)						
				elif an[N_step] == -1:
					peak, pixer[im] = proj_ali_incore_chunks(data[im],refrings,numr,xrng[N_step],yrng[N_step],step[N_step],finfo)
				else:
					if apsi[N_step] == -1:
						peak, pixer[im] = proj_ali_incore_local_chunks(data[im],refrings,numr,xrng[N_step],yrng[N_step],step[N_step],an[N_step],finfo, sym = sym)
					else:
						peak, pixer[im] = proj_ali_incore_local_psi(data[im],refrings,numr,xrng[N_step],yrng[N_step],step[N_step],an[N_step],apsi[N_step],finfo)
				data[im].set_attr("previousmax", peak)

			if myid == main_node:
				print_msg("Time of alignment = %d\n"%(time()-start_time))
				start_time = time()

			#output pixel errors
			pass#IMPORTIMPORTIMPORT from mpi import mpi_gatherv
			recvbuf = mpi_gatherv(pixer, nima, MPI_FLOAT, recvcount, disps, MPI_FLOAT, main_node, MPI_COMM_WORLD)
			mpi_barrier(MPI_COMM_WORLD)
			terminate = 0
			if myid == main_node:
				recvbuf = map(float, recvbuf)
				pass#IMPORTIMPORTIMPORT from statistics import hist_list
				lhist = 20
				region, histo = hist_list(recvbuf, lhist)
				if region[0] < 0.0:  region[0] = 0.0
				msg = "      Histogram of pixel errors\n      ERROR       number of particles\n"
				print_msg(msg)
				for lhx in xrange(lhist):
					msg = " %10.3f     %7d\n"%(region[lhx], histo[lhx])
					print_msg(msg)
				# Terminate if 95% within 1 pixel error
				im = 0
				for lhx in xrange(lhist):
					if region[lhx] > 1.0: break
					im += histo[lhx]
				precn = 100*float(total_nima-im)/float(total_nima)
				msg = " Number of particles that changed orientations %7d, percentage of total: %5.1f\n"%(total_nima-im, precn)
				print_msg(msg)
				if precn <= termprec:  terminate = 1
				del region, histo
			del recvbuf
			terminate = mpi_bcast(terminate, 1, MPI_INT, 0, MPI_COMM_WORLD)
			terminate = int(terminate[0])

			if center == -1 and sym[0] == 'c':
				pass#IMPORTIMPORTIMPORT from utilities      import estimate_3D_center_MPI, rotate_3D_shift
				cs[0], cs[1], cs[2], dummy, dummy = estimate_3D_center_MPI(data, total_nima, myid, number_of_proc, main_node)
				if myid == main_node:
					msg = " Average center x = %10.3f        Center y = %10.3f        Center z = %10.3f\n"%(cs[0], cs[1], cs[2])
					print_msg(msg)
				if int(sym[1]) > 1:
					cs[0] = cs[1] = 0.0
					if myid == main_node:
						print_msg("For symmetry group cn (n>1), we only center the volume in z-direction\n")
				cs = mpi_bcast(cs, 3, MPI_FLOAT, main_node, MPI_COMM_WORLD)
				cs = [-float(cs[0]), -float(cs[1]), -float(cs[2])]
				rotate_3D_shift(data, cs)

			# write out headers, under MPI writing has to be done sequentially
			mpi_barrier(MPI_COMM_WORLD)
			par_str = ['xform.projection', 'previousmax', 'ID']
			if myid == main_node:
	   			if(file_type(stack) == "bdb"):
	        			pass#IMPORTIMPORTIMPORT from utilities import recv_attr_dict_bdb
	        			recv_attr_dict_bdb(main_node, stack, data, par_str, image_start, image_end, number_of_proc)
	        		else:
	        			pass#IMPORTIMPORTIMPORT from utilities import recv_attr_dict
	        			recv_attr_dict(main_node, stack, data, par_str, image_start, image_end, number_of_proc)
				print_msg("Time to write header information= %d\n"%(time()-start_time))
				start_time = time()
	        	else:	       send_attr_dict(main_node, data, par_str, image_start, image_end)

			if CTF: vol, fscc = rec3D_MPI(data, snr, sym, fscmask, os.path.join(outdir, "resolution%04d"%(total_iter)), myid, main_node, npad = npad)
			else:   vol, fscc = rec3D_MPI_noCTF(data, sym, fscmask, os.path.join(outdir, "resolution%04d"%(total_iter)), myid, main_node, npad = npad)

			if myid == main_node:
				print_msg("3D reconstruction time = %d\n"%(time()-start_time))
				start_time = time()

			if fourvar:
			#  Compute Fourier variance
				for im in xrange(nima):
					original_data[im].set_attr( 'xform.projection', data[im].get_attr('xform.projection') )
				varf = varf3d_MPI(original_data, ssnr_text_file = os.path.join(outdir, "ssnr%04d"%(total_iter)), mask2D = None, reference_structure = vol, ou = last_ring, rw = 1.0, npad = 1, CTF = CTF, sign = 1, sym =sym, myid = myid)
				if myid == main_node:
					print_msg("Time to calculate 3D Fourier variance= %d\n"%(time()-start_time))
					start_time = time()
					varf = 1.0/varf
			else:  varf = None

			if myid == main_node:
				drop_image(vol, os.path.join(outdir, "vol%04d.hdf"%(total_iter)))
				ref_data[2] = vol
				ref_data[3] = fscc
				ref_data[4] = varf
				#  call user-supplied function to prepare reference image, i.e., center and filter it
				vol, cs = user_func(ref_data)
				drop_image(vol, os.path.join(outdir, "volf%04d.hdf"%(total_iter)))

			del varf
			bcast_EMData_to_all(vol, myid, main_node)


	if myid == main_node: print_end_msg("ali3d_MPI")
'''
"""5
		# finfo = None
		pass#IMPORTIMPORTIMPORT import os
		outdir = "./"
		# info_file = os.path.join(outdir, "progress%04d"%myid)
		
		# finfo = open(info_file, 'w')
		if "filename_first_time" not in sali3d_base.__dict__:
			pass#IMPORTIMPORTIMPORT import datetime
			sali3d_base.filename_first_time = os.path.join(outdir, "progress%04d_%s"%(myid, datetime.datetime.now().strftime('%Y-%m-%d--%H-%M-%S-%f')[:-3]))
			finfo = open(sali3d_base.filename_first_time, "w")
		else: 
			finfo = open(sali3d_base.filename_first_time, "a")
			finfo.write("=======================================================================================\n")
		# finfo = open(info_file, 'a')
		"""
"""6
	user_func = Tracker["constants"]["user_func"]
	if ref_vol:
		#vol = do_volume_mrk01(ref_vol, Tracker, 0, mpi_comm)
		ref_data = [ref_vol, Tracker, 0, mpi_comm]
		vol = user_func(ref_data)
	else:
		#vol = do_volume_mrk01(data, Tracker, 0, mpi_comm)
		ref_data = [data, Tracker, 0, mpi_comm]
		vol = user_func(ref_data)
	"""
"""  Have to think about it PAP7
				if(  changes > saturatecrit ):
					if( Iter == 1 ):
						log.add("Will continue even though %4.2f images did not find better orientations"%saturatecrit)
					else:
						terminate = 1
						log.add("...............")
						log.add(">>>>>>>>>>>>>>>   Will terminate as %4.2f images did not find better orientations"%saturatecrit)
				if( terminate == 0 ):
					historyofchanges.append(changes)
					historyofchanges = historyofchanges[:3]
					historyofchanges.sort()
					if( (historyofchanges[-1]-historyofchanges[0])/2/(historyofchanges[-1]+historyofchanges[0]) <0.05 ):
						terminate = 1
						log.add("...............")
						log.add(">>>>>>>>>>>>>>>   Will terminate as orientations do not improve anymore")
				"""
"""8
					if( lhx > saturatecrit):
						if( Iter == 1 ):
							log.add("Will continue even though %4.2f images had pixel error < %5.2f"%(saturatecrit,pixercutoff))
						else:
							terminate = 1
							log.add("...............")
							log.add(">>>>>>>>>>>>>>>   Will terminate as %4.2f images had pixel error < %5.2f"%(saturatecrit,pixercutoff))
					"""
"""9
			#=========================================================================
			if(False):  #total_iter%1 == 5 or terminate):
				# gather parameters
				params = []
				previousmax = []
				for im in data:
					t = get_params_proj(im)
					params.append( [t[0], t[1], t[2], t[3]/shrinkage, t[4]/shrinkage] )
					#if(t[3] >0.0 or t[4]>0.0):  print  "  ERRROR  ",t
					previousmax.append(im.get_attr("previousmax"))
				assert(nima == len(params))
				params = wrap_mpi_gatherv(params, 0, mpi_comm)
				if myid == 0:
					assert(total_nima == len(params))
				previousmax = wrap_mpi_gatherv(previousmax, 0, mpi_comm)
				if myid == main_node:
					pass#IMPORTIMPORTIMPORT from utilities import write_text_row, write_text_file
					write_text_row(params, "soft/params%04d.txt"%total_iter)
					write_text_file(previousmax, "soft/previousmax%04d.txt"%total_iter)


				del previousmax, params
				i = 1
				while data[0].has_attr("xform.projection" + str(i)):
					params = []
					previousmax = []
					for im in data:

						try:
							#print  im.get_attr("xform.projection" + str(i))
							t = get_params_proj(im,"xform.projection" + str(i))
						except:
							print " NO XFORM  ",myid, i,im.get_attr('ID')
							pass#IMPORTIMPORTIMPORT from sys import exit
							exit()

						params.append( [t[0], t[1], t[2], t[3]/shrinkage, t[4]/shrinkage] )
					assert(nima == len(params))
					params = wrap_mpi_gatherv(params, 0, mpi_comm)
					if myid == 0:
						assert(total_nima == len(params))
					if myid == main_node:
						write_text_row(params, "soft/params-%04d-%04d.txt"%(i,total_iter))
					del previousmax, params
					i+=1


			if( ( terminate or (Iter == max_iter) ) and (myid == main_node) ):
				if( type(stack) is types.StringType ):
					pass#IMPORTIMPORTIMPORT from EMAN2 import Vec2f, Transform
					pass#IMPORTIMPORTIMPORT from EMAN2db import db_open_dict
					DB = db_open_dict(stack)
					for im in xrange(len(params)):
						t = Transform({"type":"spider","phi":params[im][0],"theta":params[im][1],"psi":params[im][2]})
						t.set_trans(Vec2f(-params[im][3], -params[im][4]))
						DB.set_attr(particle_ids[im], "xform.projection", t)
					DB.close()
				else:
					for im in xrange(len(params)): set_params_proj(stack[particle_ids[im]], params[im])
			"""
'''10
def sali3d_base_horatio_01(stack, ref_vol = None, Tracker = None, rangle = 0.0, rshift = 0.0, mpi_comm = None, log = None):	
	"""
		parameters: list of (all) projections | reference volume is optional, the data is shrank, 
		  the program does not know anything about shrinking| ...
		Data is assumed to be CTF multiplied and the ctf_applied flag to be set.
		The alignment done depends on nsoft:
					 nsoft = 0 & an = -1: exhaustive deterministic
					 nsoft = 0 & an > 0 : local deterministic
					 nsoft = 1 shc
					 nsoft >1  shc_multi
		
	"""

	pass#IMPORTIMPORTIMPORT from alignment       import Numrinit, prepare_refrings, generate_indices_and_refrings
	pass#IMPORTIMPORTIMPORT from alignment       import proj_ali_incore,  proj_ali_incore_zoom,  proj_ali_incore_local, proj_ali_incore_local_zoom
	pass#IMPORTIMPORTIMPORT from alignment       import shc, center_projections_3D, ringwe
	pass#IMPORTIMPORTIMPORT from utilities       import bcast_number_to_all, bcast_EMData_to_all, 	wrap_mpi_gatherv, wrap_mpi_bcast, model_blank, print_from_process
	pass#IMPORTIMPORTIMPORT from utilities       import get_im, file_type, model_circle, get_input_from_string, get_params_proj, set_params_proj, pad
	pass#IMPORTIMPORTIMPORT from utilities       import even_angles
	pass#IMPORTIMPORTIMPORT from mpi             import mpi_bcast, mpi_comm_size, mpi_comm_rank, MPI_FLOAT, MPI_COMM_WORLD, mpi_barrier, mpi_reduce, MPI_INT, MPI_SUM
	pass#IMPORTIMPORTIMPORT from projection      import prep_vol
	pass#IMPORTIMPORTIMPORT from statistics      import hist_list
	pass#IMPORTIMPORTIMPORT from applications    import MPI_start_end
	pass#IMPORTIMPORTIMPORT from filter          import filt_ctf, filt_table
	pass#IMPORTIMPORTIMPORT from global_def      import Util
	pass#IMPORTIMPORTIMPORT from fundamentals    import resample, fshift
	pass#IMPORTIMPORTIMPORT from multi_shc       import shc_multi
	#from development     import do_volume_mrk01
	pass#IMPORTIMPORTIMPORT import user_functions
	pass#IMPORTIMPORTIMPORT from EMAN2           import EMUtil, EMData
	pass#IMPORTIMPORTIMPORT import types
	pass#IMPORTIMPORTIMPORT from time            import time

	nsoft            = Tracker["nsoft"]
	saturatecrit     = Tracker["saturatecrit"]
	pixercutoff      = Tracker["pixercutoff"]
	zoom             = Tracker["zoom"]
	center           = Tracker["constants"]["center"]
	CTF              = Tracker["constants"]["CTF"]
	ref_a            = Tracker["constants"]["ref_a"]
	rstep            = Tracker["constants"]["rs"]
	sym              = Tracker["constants"]["sym"]
	first_ring       = 1
	last_ring        = Tracker["radius"]
	xr               = Tracker["xr"]
	yr               = Tracker["yr"]
	ts               = Tracker["ts"]
	an               = Tracker["an"]
	delta            = Tracker["delta"]
	max_iter         = Tracker["maxit"]

	if mpi_comm == None: mpi_comm = MPI_COMM_WORLD

	if log == None:
		pass#IMPORTIMPORTIMPORT from logger import Logger
		log = Logger()

	number_of_proc = mpi_comm_size(mpi_comm)
	myid           = mpi_comm_rank(mpi_comm)
	main_node      = 0

	if myid == main_node:
		log.add("Start sali3d_base_h_01, nsoft = %1d"%nsoft)

	xrng        = get_input_from_string(xr)
	if  yr == "-1":  yrng = xrng
	else          :  yrng = get_input_from_string(yr)
	step        = get_input_from_string(ts)
	delta       = get_input_from_string(delta)
	lstp = min(len(xrng), len(yrng), len(step), len(delta))
	if an == "-1":
		an = [-1] * lstp
	else:
		an = get_input_from_string(an)

	if( type(stack) is types.StringType ):
		if myid == main_node:
			total_nima = EMUtil.get_image_count( stack )
		else:
			total_nima = 0
		total_nima = wrap_mpi_bcast(total_nima, main_node, mpi_comm)
		list_of_particles = range(total_nima)
		image_start, image_end = MPI_start_end(total_nima, number_of_proc, myid)
		# create a list of images for each node
		list_of_particles = list_of_particles[image_start: image_end]
		nima = len(list_of_particles)

	else:
		list_of_particles = range(len(stack))
		nima = len(list_of_particles)
		total_nima = len(list_of_particles)
		total_nima = mpi_reduce(total_nima, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD)
		total_nima = mpi_bcast(total_nima, 1, MPI_INT, 0, MPI_COMM_WORLD)
		total_nima = int(total_nima[0])


	if myid == 0 or myid == 5 or myid == 14:
		# finfo = None
		pass#IMPORTIMPORTIMPORT import os
		outdir = "./"
		# info_file = os.path.join(outdir, "progress%04d"%myid)
		
		# finfo = open(info_file, 'w')
		if "filename_first_time" not in sali3d_base_h_01.__dict__:
			pass#IMPORTIMPORTIMPORT import datetime
			sali3d_base_h_01.filename_first_time = os.path.join(outdir, "progress%04d_%s"%(myid, datetime.datetime.now().strftime('%Y-%m-%d--%H-%M-%S-%f')[:-3]))
			finfo = open(sali3d_base_h_01.filename_first_time, 'w')
		else: 
			finfo = open(sali3d_base_h_01.filename_first_time, 'a')
			finfo.write("=======================================================================================\n")
		# finfo = open(info_file, 'a')
	else:
		finfo = None

	if( myid == main_node):
		if( type(stack) is types.StringType ):  mask2D = get_im(stack, list_of_particles[0])
		else:                                   mask2D = stack[list_of_particles[0]]
		nx = mask2D.get_xsize()
	else:  nx = 0
	nx  = bcast_number_to_all(nx, source_node = main_node)
	mx = 2*nx
	if last_ring < 0:	last_ring = int(nx/2) - 2

	numr	= Numrinit(first_ring, last_ring, rstep, "F")

	data = [None]*nima
	for im in xrange(nima):
		if( type(stack) is types.StringType ):  data[im] = get_im(stack, list_of_particles[im])
		else:                                   data[im] = stack[list_of_particles[im]]
	mpi_barrier(mpi_comm)


	if myid == main_node:
		start_time = time()

	#  Read	template volume if provided or reconstruct it
	"""
	user_func = Tracker["constants"]["user_func"]
	if ref_vol:
		#vol = do_volume_mrk01(ref_vol, Tracker, 0, mpi_comm)
		ref_data = [ref_vol, Tracker, 0, mpi_comm]
		vol = user_func(ref_data)
	else:
		#vol = do_volume_mrk01(data, Tracker, 0, mpi_comm)
		ref_data = [data, Tracker, 0, mpi_comm]
		vol = user_func(ref_data)
	"""
	vol = ref_vol
	# log
	if myid == main_node:
		log.add("Setting of reference 3D reconstruction time = %10.1f\n"%(time()-start_time))
		start_time = time()


	pixer = [0.0]*nima
	#historyofchanges = [0.0, 0.5, 1.0]
	#par_r = [[] for im in list_of_particles ]
	cs = [0.0]*3
	total_iter = 0
	# do the projection matching,  it has a loop over iterations here, 
	#  but it can only do one iteration as many settings are done in meridien.  Perturbations are a good example, there is one per each iteration.
	if zoom: lstp = 1
	
	for N_step in xrange(lstp):
		# calculate_number_of_cones(volft, kb, delta, sym, cnx, cny, numr, mode, wr_four)
		cnx = cny = nx//2 + 1
		# numr = Numrinit(1,15)
		mode = "F"
		# wr_four = ringwe(numr, mode)
		# volft, kb = prep_vol(vol)

		terminate = 0
		Iter = 0
		while Iter < max_iter and terminate == 0:

			Iter += 1
			total_iter += 1

			mpi_barrier(mpi_comm)
			if myid == main_node:
				log.add("ITERATION #%3d,  inner iteration #%3d"%(total_iter, Iter))
				log.add("Delta = %7.4f, an = %7.4f, xrange = %7.4f, yrange = %7.4f, step = %7.4f   %7.4f  %7.4f\n"%\
							(delta[N_step], an[N_step], xrng[N_step], yrng[N_step], step[N_step], rshift, rangle))
				start_time = time()


			#=========================================================================
			# build references
			volft, kb = prep_vol(vol)
			projangles = [[] for i in xrange(nima)]
			for im in xrange(nima):
				projangles[im] = get_params_proj(data[im])[:3]
					

			cone_count = 0
			# start loop here ZCcw2oL8ZbcGbU		
			for image_indices, refrings, list_of_reference_angles in generate_indices_and_refrings(nima, projangles, volft, kb, nx, delta[N_step], an[N_step],	rangle, ref_a, sym, numr, MPI=mpi_comm, phiEqpsi = "Zero"):
				
				cone_count += 1
				print "cone_count", cone_count

				if myid == main_node:
					log.add("Time to prepare rings: %10.1f\n" % (time()-start_time))
					start_time = time()
	
				#=========================================================================
				#  there is no need for previousmax for deterministic searches
				if total_iter == 1 and nsoft > 0:
					if(an[N_step] < 0.0):
						# adjust params to references, calculate psi+shifts, calculate previousmax
						# for im in xrange(nima):
						for im in image_indices:
							previousmax = data[im].get_attr_default("previousmax", -1.0e23)
							if(previousmax == -1.0e23):
								peak, pixer[im] = proj_ali_incore_local(data[im], refrings, list_of_reference_angles, numr, \
										xrng[N_step], yrng[N_step], step[N_step], delta[N_step]*2.5, sym = sym)
								data[im].set_attr("previousmax", peak)
					else:
						#  Here it is supposed to be shake and bake for local SHC, but it would have to be signaled somehow
						for im in xrange(nima):
							data[im].set_attr("previousmax", -1.0e23)
					if myid == main_node:
						log.add("Time to calculate first psi+shifts+previousmax: %10.1f\n" % (time()-start_time))
						start_time = time()
				#=========================================================================
	
				# cannot have barriers in this loop because some cones might not have images assigned to them! In this case 'image_indices' is empty.
				# mpi_barrier(mpi_comm)
				if myid == main_node:  start_time = time()
				#=========================================================================
				# alignment
				#number_of_checked_refs = 0
				par_r = [0]*max(2,(nsoft+1))
				if(an[N_step] > 0):
					pass
					# these are already calculated by the generator at the top of the loop
					# generate list of angles
					# from alignment import generate_list_of_reference_angles_for_search
					# list_of_reference_angles = \
					# generate_list_of_reference_angles_for_search([[refrings[lr].get_attr("phi"), refrings[lr].get_attr("theta")] for lr in xrange(len(refrings))], sym=sym)			
				else:  list_of_reference_angles = [[1.0,1.0]]
				error_status = 0
				pass#IMPORTIMPORTIMPORT from utilities import if_error_then_all_processes_exit_program

				# for im in xrange(nima):
				for im in image_indices:
					if Tracker["constants"]["pwsharpening"] :
						#  High-pass filtration of data[im]
						try:
							stmp = data[im].get_attr("ptcl_source_image")
						except:
							try:
								stmp = data[im].get_attr("ctf")
								stmp = round(stmp.defocus,4)
							except:
								ERROR("Either ptcl_source_image or ctf has to be present in the header.","meridien",1, myid)
						try:
							indx = Tracker["bckgnoise"][1].index(stmp)
						except:
							ERROR("Problem with indexing ptcl_source_image.","meridien",1, myid)
	
						tempdata = Util.window(pad(filt_table(data[im],[Tracker["bckgnoise"][0][i,indx] for i in xrange(nx)]), mx, mx,1,0.0), nx, nx)
					else:  tempdata = data[im].copy()
					if(nsoft == 0):
						if(an[N_step] == -1):
							#  In zoom option each projection goes through shift zoom alignment
							if  zoom: peak, pixer[im] = proj_ali_incore_zoom(tempdata, refrings, numr, \
															xrng, yrng, step, finfo = finfo, sym=sym)
							else:  peak, pixer[im] = proj_ali_incore(tempdata, refrings, numr, \
													xrng[N_step], yrng[N_step], step[N_step], finfo = finfo, sym=sym, delta_psi = delta[N_step], rshift = rshift*xrng[N_step])
						else:
							if  zoom: peak, pixer[im] = proj_ali_incore_local_zoom(tempdata, refrings, list_of_reference_angles, numr, \
										xrng, yrng, step, an, finfo = finfo, sym=sym)
							else:  
								
								peak, pixer[im] = proj_ali_incore_local(tempdata, refrings, list_of_reference_angles, numr, \
										xrng[N_step], yrng[N_step], step[N_step], an[N_step], finfo = finfo, sym=sym, delta_psi = delta[N_step], rshift = rshift)
						if(pixer[im] == 0.0):  par_r[0] += 1
					elif(nsoft == 1):
						tempdata.set_attr("previousmax", data[im].get_attr("previousmax"))
						peak, pixer[im], number_of_checked_refs, iref = \
							shc(tempdata, refrings, list_of_reference_angles, numr, xrng[N_step], yrng[N_step], step[N_step], an[N_step], sym, finfo = finfo)
						if(pixer[im] == 0.0):  par_r[0] += 1
						data[im].set_attr("previousmax", tempdata.get_attr("previousmax"))
					elif(nsoft > 1):
						#  This is not functional
						peak, pixer[im], checked_refs, number_of_peaks = shc_multi(data[im], refrings, numr, \
													xrng[N_step], yrng[N_step], step[N_step], an[N_step], nsoft, sym, finfo = finfo)
						par_r[number_of_peaks] += 1
						#number_of_checked_refs += checked_refs
					data[im].set_attr("xform.projection", tempdata.get_attr("xform.projection"))
				if len(image_indices)>0: del tempdata
				if(an[N_step] > 0):  del list_of_reference_angles
				#=========================================================================
				# if_error_then_all_processes_exit_program(error_status)
				# cannot have barriers in this loop because some cones might not have images assigned to them! In this case 'image_indices' is empty.
				# mpi_barrier(mpi_comm)
				if myid == main_node:
					#print  data[0].get_attr_dict()
					log.add("Time of alignment = %10.1f\n"%(time()-start_time))
					start_time = time()
				
				# end loop here ZCcw2oL8ZbcGbU
				
			del volft, kb
			mpi_barrier(mpi_comm)
			print_from_process(0, "passed")
			#=========================================================================
			#  Pixer errors available here are useless as they are done for shifts on the reduced image size.
			#output pixel errors, check stop criterion
			all_pixer = wrap_mpi_gatherv(pixer, 0, mpi_comm)
			par_r = mpi_reduce(par_r, len(par_r), MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD)
			#total_checked_refs = wrap_mpi_gatherv([number_of_checked_refs], main_node, mpi_comm)
			terminate = 0
			if myid == main_node:
				#total_checked_refs = sum(total_checked_refs)
				if(nsoft < 2):  par_r[1] = total_nima - par_r[0]
				log.add("=========== Number of better orientations found ==============")
				for lhx in xrange(len(par_r)):
					msg = "            %5d     %7d"%(lhx, par_r[lhx])
					log.add(msg)
				log.add("_______________________________________________________")
				changes = par_r[0]/float(total_nima)
				"""  Have to think about it PAP
				if(  changes > saturatecrit ):
					if( Iter == 1 ):
						log.add("Will continue even though %4.2f images did not find better orientations"%saturatecrit)
					else:
						terminate = 1
						log.add("...............")
						log.add(">>>>>>>>>>>>>>>   Will terminate as %4.2f images did not find better orientations"%saturatecrit)
				if( terminate == 0 ):
					historyofchanges.append(changes)
					historyofchanges = historyofchanges[:3]
					historyofchanges.sort()
					if( (historyofchanges[-1]-historyofchanges[0])/2/(historyofchanges[-1]+historyofchanges[0]) <0.05 ):
						terminate = 1
						log.add("...............")
						log.add(">>>>>>>>>>>>>>>   Will terminate as orientations do not improve anymore")
				"""

				lhist = 20
				region, histo = hist_list(all_pixer, lhist)
				log.add("=========== Histogram of pixel errors ==============")
				for lhx in xrange(lhist):
					msg = "          %10.3f     %7d"%(region[lhx], histo[lhx])
					log.add(msg)
				log.add("____________________________________________________")
				if(nsoft<2 and terminate == 0):
					lhx = 0
					for msg in all_pixer:
						if(msg < pixercutoff): lhx += 1
					lhx = float(lhx)/float(total_nima)
					log.add(">>> %4.2f images had pixel error <%5.2f"%(lhx,pixercutoff))
					"""
					if( lhx > saturatecrit):
						if( Iter == 1 ):
							log.add("Will continue even though %4.2f images had pixel error < %5.2f"%(saturatecrit,pixercutoff))
						else:
							terminate = 1
							log.add("...............")
							log.add(">>>>>>>>>>>>>>>   Will terminate as %4.2f images had pixel error < %5.2f"%(saturatecrit,pixercutoff))
					"""
			terminate = True  #wrap_mpi_bcast(terminate, main_node, mpi_comm)
			#=========================================================================
			mpi_barrier(mpi_comm)
			if myid == main_node:
				#print  data[0].get_attr_dict()
				log.add("Time to compute histograms = %10.1f\n"%(time()-start_time))
				start_time = time()


			#=========================================================================
			mpi_barrier(mpi_comm)
			if( terminate or (Iter == max_iter) ):
				# gather parameters
				params = []
				for im in xrange(nima):
					t = get_params_proj(data[im])
					params.append( [t[0], t[1], t[2], t[3], t[4]] )
				params = wrap_mpi_gatherv(params, main_node, mpi_comm)
			# centering and volume reconstruction if not terminating
			else:
				#=========================================================================
				# centering
				if center == -1 and sym[0] == 'c':
					pass#IMPORTIMPORTIMPORT from utilities      import estimate_3D_center_MPI, rotate_3D_shift
					cs[0], cs[1], cs[2], dummy, dummy = estimate_3D_center_MPI(data, total_nima, myid, number_of_proc, main_node, mpi_comm=mpi_comm)
					if myid == main_node:
						msg = " Average center x = %10.3f        Center y = %10.3f        Center z = %10.3f\n"%(cs[0], cs[1], cs[2])
						log.add(msg)
					if int(sym[1]) > 1:
						cs[0] = cs[1] = 0.0
						if myid == main_node:
							log.add("For symmetry group cn (n>1), we only center the volume in z-direction\n")
					cs = mpi_bcast(cs, 3, MPI_FLOAT, main_node, mpi_comm)
					cs = [-float(cs[0]), -float(cs[1]), -float(cs[2])]
					rotate_3D_shift(data, cs)
				#=========================================================================
				if myid == main_node:
					start_time = time()
				#vol = do_volume_mrk01(data, Tracker, total_iter, mpi_comm)
				ref_data = [data, Tracker, total_iter, mpi_comm]
				user_func = Tracker["constants"] ["user_func"]
				vol = user_func(ref_data)
				#if myid == main_node:  vol.write_image('soft/smvol%04d.hdf'%total_iter)
				# log
				if myid == main_node:
					log.add("3D reconstruction time = %10.1f\n"%(time()-start_time))
					start_time = time()
			#=========================================================================

			"""
			#=========================================================================
			if(False):  #total_iter%1 == 5 or terminate):
				# gather parameters
				params = []
				previousmax = []
				for im in data:
					t = get_params_proj(im)
					params.append( [t[0], t[1], t[2], t[3]/shrinkage, t[4]/shrinkage] )
					#if(t[3] >0.0 or t[4]>0.0):  print  "  ERRROR  ",t
					previousmax.append(im.get_attr("previousmax"))
				assert(nima == len(params))
				params = wrap_mpi_gatherv(params, 0, mpi_comm)
				if myid == 0:
					assert(total_nima == len(params))
				previousmax = wrap_mpi_gatherv(previousmax, 0, mpi_comm)
				if myid == main_node:
					pass#IMPORTIMPORTIMPORT from utilities import write_text_row, write_text_file
					write_text_row(params, "soft/params%04d.txt"%total_iter)
					write_text_file(previousmax, "soft/previousmax%04d.txt"%total_iter)


				del previousmax, params
				i = 1
				while data[0].has_attr("xform.projection" + str(i)):
					params = []
					previousmax = []
					for im in data:

						try:
							#print  im.get_attr("xform.projection" + str(i))
							t = get_params_proj(im,"xform.projection" + str(i))
						except:
							print " NO XFORM  ",myid, i,im.get_attr('ID')
							pass#IMPORTIMPORTIMPORT from sys import exit
							exit()

						params.append( [t[0], t[1], t[2], t[3]/shrinkage, t[4]/shrinkage] )
					assert(nima == len(params))
					params = wrap_mpi_gatherv(params, 0, mpi_comm)
					if myid == 0:
						assert(total_nima == len(params))
					if myid == main_node:
						write_text_row(params, "soft/params-%04d-%04d.txt"%(i,total_iter))
					del previousmax, params
					i+=1


			if( ( terminate or (Iter == max_iter) ) and (myid == main_node) ):
				if( type(stack) is types.StringType ):
					pass#IMPORTIMPORTIMPORT from EMAN2 import Vec2f, Transform
					pass#IMPORTIMPORTIMPORT from EMAN2db import db_open_dict
					DB = db_open_dict(stack)
					for im in xrange(len(params)):
						t = Transform({"type":"spider","phi":params[im][0],"theta":params[im][1],"psi":params[im][2]})
						t.set_trans(Vec2f(-params[im][3], -params[im][4]))
						DB.set_attr(particle_ids[im], "xform.projection", t)
					DB.close()
				else:
					for im in xrange(len(params)): set_params_proj(stack[particle_ids[im]], params[im])
			"""


	if myid == main_node:
		log.add("Finish sali3d_base, nsoft = %1d"%nsoft)
	return params
'''
"""11
	if myid == main_node:
		os.mkdir(outdir)
		pass#IMPORTIMPORTIMPORT import global_def
		sparx_global_def.LOGFILE =  os.path.join(outdir, sparx_global_def.LOGFILE)
		print_begin_msg("local_ali3d_MPI")
		pass#IMPORTIMPORTIMPORT import user_functions
		user_func = sparx_user_functions.factory[user_func_name]
		if CTF:
			ima = EMData()
			ima.read_image(stack, 0)
			ctf_applied = ima.get_attr_default("ctf_applied", 0)
			del ima
			if ctf_applied == 1:  ERROR("local_ali3d does not work for CTF-applied data", "local_ali3d_MPI", 1,myid)
	mpi_barrier(mpi_comm)
	"""
"""12
	if myid == main_node:
		pass#IMPORTIMPORTIMPORT import user_functions
		user_func = sparx_user_functions.factory[user_func_name]

		print_msg("Input stack                 : %s\n"%(stack))
		print_msg("Output directory            : %s\n"%(outdir))
		print_msg("Maskfile                    : %s\n"%(Tracker["constants"]["mask3D"]))
		print_msg("Outer radius                : %i\n"%(last_ring))
		print_msg("Angular search range        : %s\n"%(delta))
		print_msg("Shift search range          : %f\n"%(ts))
		print_msg("Maximum iteration           : %i\n"%(max_iter))
		print_msg("Center type                 : %i\n"%(center))
		print_msg("CTF correction              : %s\n"%(CTF))
		print_msg("Signal-to-Noise Ratio       : %f\n"%(snr))
		print_msg("Symmetry group              : %s\n"%(sym))
		print_msg("Chunk size                  : %f\n\n"%(chunk))
		print_msg("User function               : %s\n"%(user_func_name))
	"""
"""13
			if( lhx > saturatecrit):
				if( iteration == 1 ):
					log.add("First iteration, will continue even though %4.2f images did not find better orientations"%saturatecrit)
				else:
					terminate = 1
					log.add("..............................................................")
					log.add(">>>>>>>>>>>>>>>   Will terminate as %4.2f images did not find better orientations"%saturatecrit)
			"""
"""14
		if( type(stack) is types.StringType ):
			pass#IMPORTIMPORTIMPORT from EMAN2 import Vec2f, Transform
			pass#IMPORTIMPORTIMPORT from EMAN2db import db_open_dict
			DB = db_open_dict(stack)
			for im in xrange(len(params)):
				t = Transform({"type":"spider","phi":params[im][0],"theta":params[im][1],"psi":params[im][2]})
				t.set_trans(Vec2f(-params[im][3], -params[im][4]))
				DB.set_attr(particle_ids[im], "xform.projection", t)
			DB.close()
		else:
			for im in xrange(len(params)): set_params_proj(stack[particle_ids[im]], params[im])

		log.add("Time to write header information= %d\n"%(time()-start_time))
		"""
'''15
#  This does not handle symmetries properly

def ali3dlocal_MPI(stack, ref_vol, outdir, maskfile = None, ir = 1, ou = -1, rs = 1, 
            xr = "4 2 2 1", yr = "-1", ts = "1 1 0.5 0.25", delta = "10 6 4 4", an = "-1", apsi = "-1", deltapsi = "-1", startpsi = "-1",
	    center = -1, maxit = 5, CTF = False, snr = 1.0,  ref_a = "S", sym = "c1",  user_func_name = "ref_ali3d",
	    fourvar = True, npad = 4, debug = False, termprec = 0.0):

	pass#IMPORTIMPORTIMPORT from alignment       import Numrinit, ringwe, proj_ali_incore, proj_ali_incore_local, proj_ali_incore_local_psi
	pass#IMPORTIMPORTIMPORT from utilities       import model_circle, get_image, drop_image, get_input_from_string
	pass#IMPORTIMPORTIMPORT from utilities       import bcast_list_to_all, bcast_number_to_all, reduce_EMData_to_root, bcast_EMData_to_all
	pass#IMPORTIMPORTIMPORT from utilities       import send_attr_dict
	pass#IMPORTIMPORTIMPORT from utilities       import get_params_proj, file_type
	pass#IMPORTIMPORTIMPORT from fundamentals    import rot_avg_image
	pass#IMPORTIMPORTIMPORT from utilities       import print_begin_msg, print_end_msg, print_msg
	pass#IMPORTIMPORTIMPORT from mpi             import mpi_bcast, mpi_comm_size, mpi_comm_rank, MPI_FLOAT, MPI_COMM_WORLD, mpi_barrier
	pass#IMPORTIMPORTIMPORT from mpi             import mpi_reduce, MPI_INT, MPI_SUM
	pass#IMPORTIMPORTIMPORT from filter          import filt_ctf
	pass#IMPORTIMPORTIMPORT from projection      import prep_vol, prgs
	pass#IMPORTIMPORTIMPORT from statistics      import hist_list, varf3d_MPI
	pass#IMPORTIMPORTIMPORT from applications    import MPI_start_end
	pass#IMPORTIMPORTIMPORT import os
	pass#IMPORTIMPORTIMPORT import types


	number_of_proc = mpi_comm_size(MPI_COMM_WORLD)
	myid           = mpi_comm_rank(MPI_COMM_WORLD)
	main_node = 0
	
	if os.path.exists(outdir):  ERROR('Output directory exists, please change the name and restart the program', "ali3dlocal_MPI", 1, myid)
	mpi_barrier(MPI_COMM_WORLD)

	if myid == main_node:
		os.mkdir(outdir)
		pass#IMPORTIMPORTIMPORT import global_def
		sparx_global_def.LOGFILE =  os.path.join(outdir, sparx_global_def.LOGFILE)
		print_begin_msg("ali3dlocal_MPI")
	mpi_barrier(MPI_COMM_WORLD)

	if debug:
		pass#IMPORTIMPORTIMPORT from time import sleep
		while not os.path.exists(outdir):
			print  "Node ",myid,"  waiting..."
			sleep(5)

		info_file = os.path.join(outdir, "progress%04d"%myid)
		finfo = open(info_file, 'w')
	else:
		finfo = None

	xrng        = get_input_from_string(xr)
	if  yr == "-1":  yrng = xrng
	else          :  yrng = get_input_from_string(yr)
	step        = get_input_from_string(ts)
	delta       = get_input_from_string(delta)
	lstp = min(len(xrng), len(yrng), len(step), len(delta))
	if an == "-1":
		an = [-1] * lstp
	else:
		an = get_input_from_string(an)

	if apsi == "-1":
		apsi = [-1] * lstp
	else:
		apsi = get_input_from_string(apsi)

	if deltapsi == "-1":
		deltapsi = [-1] * lstp
	else:
		deltapsi = get_input_from_string(deltapsi)

	if startpsi == "-1":
		startpsi = [-1] * lstp
	else:
		startpsi = get_input_from_string(startpsi)

	first_ring  = int(ir)
	rstep       = int(rs)
	last_ring   = int(ou)
	max_iter    = int(maxit)
	center      = int(center)

	vol     = EMData()
	vol.read_image(ref_vol)
	nx      = vol.get_xsize()
	if last_ring < 0:	last_ring = int(nx/2) - 2

	if myid == main_node:
		pass#IMPORTIMPORTIMPORT import user_functions
		user_func = sparx_user_functions.factory[user_func_name]

		print_msg("Input stack                 : %s\n"%(stack))
		print_msg("Reference volume            : %s\n"%(ref_vol))	
		print_msg("Output directory            : %s\n"%(outdir))
		print_msg("Maskfile                    : %s\n"%(maskfile))
		print_msg("Inner radius                : %i\n"%(first_ring))
		print_msg("Outer radius                : %i\n"%(last_ring))
		print_msg("Ring step                   : %i\n"%(rstep))
		print_msg("X search range              : %s\n"%(xrng))
		print_msg("Y search range              : %s\n"%(yrng))
		print_msg("Translational step          : %s\n"%(step))
		print_msg("Angular step                : %s\n"%(delta))
		print_msg("Angular search range (phi and theta)       : %s\n"%(an))
		#print_msg("Angular search range (psi)                 : %s\n"%(apsi))
		#print_msg("Delta psi                   : %s\n"%(deltapsi))
		#print_msg("Start psi                   : %s\n"%(startpsi))
		print_msg("Maximum iteration           : %i\n"%(max_iter))
		print_msg("Percentage of change for termination: %f\n"%(termprec))
		print_msg("Center type                 : %i\n"%(center))
		print_msg("CTF correction              : %s\n"%(CTF))
		print_msg("Signal-to-Noise Ratio       : %f\n"%(snr))
		print_msg("Reference projection method : %s\n"%(ref_a))
		print_msg("Symmetry group              : %s\n\n"%(sym))
		print_msg("User function               : %s\n"%(user_func_name))

	if maskfile:
		if type(maskfile) is types.StringType: mask3D = get_image(maskfile)
		else:                                  mask3D = maskfile
	else: mask3D = model_circle(last_ring, nx, nx, nx)

	numr	= Numrinit(first_ring, last_ring, rstep, "F")
	wr_four = ringwe(numr, "F")
	cx = cy = nx//2
	mask2D  = model_circle(last_ring,nx,nx) - model_circle(first_ring,nx,nx)

	fscmask = mask3D  #model_circle(last_ring,nx,nx,nx)  For a fancy mask circle would work better  PAP 7/21/11
	if CTF:
		pass#IMPORTIMPORTIMPORT from reconstruction import rec3D_MPI
		pass#IMPORTIMPORTIMPORT from filter         import filt_ctf
	pass#IMPORTIMPORTIMPORT else:	 from reconstruction import rec3D_MPI_noCTF

	if myid == main_node:
		if file_type(stack) == "bdb":
			pass#IMPORTIMPORTIMPORT from EMAN2db import db_open_dict
			dummy = db_open_dict(stack, True)
		# horatio active_refactoring Jy51i1EwmLD4tWZ9_00000_1
		# active = EMUtil.get_all_attributes(stack, 'active')
		# list_of_particles = []
		# for im in xrange(len(active)):
		# 	if active[im]:  list_of_particles.append(im)
		# del active
		# nima = len(list_of_particles)

		nima = EMUtil.get_image_count(stack)
		list_of_particles = range(nima)
		
	else:
		nima = 0
	total_nima = bcast_number_to_all(nima, source_node = main_node)

	if myid != main_node:
		list_of_particles = [-1]*total_nima
	list_of_particles = bcast_list_to_all(list_of_particles, myid,  source_node = main_node)

	image_start, image_end = MPI_start_end(total_nima, number_of_proc, myid)
	# create a list of images for each node
	list_of_particles = list_of_particles[image_start: image_end]
	nima = len(list_of_particles)
	if debug:
		finfo.write("image_start, image_end: %d %d\n" %(image_start, image_end))
		finfo.flush()

	data = EMData.read_images(stack, list_of_particles)
	if fourvar:  original_data = []
	for im in xrange(nima):
		data[im].set_attr('ID', list_of_particles[im])
		if fourvar: original_data.append(data[im].copy())
		if CTF:
			ctf_params = data[im].get_attr("ctf")
			st = Util.infomask(data[im], mask2D, False)
			data[im] -= st[0]
			data[im] = filt_ctf(data[im], ctf_params)
			data[im].set_attr('ctf_applied', 1)

	if debug:
		finfo.write( '%d loaded  \n' % nima )
		finfo.flush()
	if myid == main_node:
		# initialize data for the reference preparation function
		ref_data = [ mask3D, max(center,0), None, None, None, None ]
		# for method -1, switch off centering in user function

	pass#IMPORTIMPORTIMPORT from time import time

	#  this is needed for gathering of pixel errors
	disps = []
	recvcount = []
	for im in xrange(number_of_proc):
		if im == main_node :  disps.append(0)
		else:                 disps.append(disps[im-1] + recvcount[im-1])
		ib, ie = MPI_start_end(total_nima, number_of_proc, im)
		recvcount.append(ie-ib)

	#  Number of reference image that fit into the memory.  Will have to be hardwired.
	numberofrefs = 500


	pixer = [0.0]*nima
	cs = [0.0]*3
	total_iter = 0
	# do the projection matching
	for N_step in xrange(lstp):
		terminate = 0
		Iter = -1
 		while Iter < max_iter-1 and terminate == 0:
			Iter += 1
			total_iter += 1
			if myid == main_node:
				start_time = time()
				print_msg("\nITERATION #%3d,  inner iteration #%3d\nDelta = %4.1f, an = %5.2f, xrange = %5.2f, yrange = %5.2f, step = %5.2f, delta psi = %5.2f, start psi = %5.2f\n"%\
						(total_iter, Iter, delta[N_step], an[N_step], xrng[N_step],yrng[N_step],step[N_step],deltapsi[N_step],startpsi[N_step]))

			totrefs = len(even_angles(delta[N_step], method = ref_a, symmetry = sym))
			numberofcones = max(totrefs/numberofrefs,1)
			if myid == main_node:
				print_msg("\n   Number of references permitted in memory = %d  , total number of references = %d , number of cones = %d \n"%(numberofrefs, totrefs, numberofcones))


			volft, kb = prep_vol(vol)
			if( numberofcones == 1):
				# One cone good enough, use the original code
				refrings = prepare_refrings(volft, kb, nx, delta[N_step], ref_a, sym, numr, True)

				if myid == main_node:
					print_msg("Time to prepare rings: %d\n" % (time()-start_time))
					start_time = time()

				for im in xrange(nima):
					#  It needs list_of_reference_angles
					peak, pixer[im] = proj_ali_incore_local(data[im],refrings,numr,xrng[N_step],yrng[N_step],step[N_step],an[N_step],finfo, sym = sym)
					data[im].set_attr("previousmax", peak)

			else:
				pass#IMPORTIMPORTIMPORT from morphology import  bracket_def
				pass#IMPORTIMPORTIMPORT from utilities  import  assign_projangles, cone_ang
				pass#IMPORTIMPORTIMPORT from alignment  import  refprojs

				h = 1.0
				dat = [sym, numberofcones, ref_a]
				def1, def2 = bracket_def(computenumberofrefs,dat, rs, h)
				def1, val  = goldsearch_astigmatism(computenumberofrefs, dat, def1, def2, tol=1.0)
				coneangles = even_angles(def1, method = "S", symmetry = sym)
				if myid == main_node:
					print_msg("\n   Computed cone delta = %f  , and the number of cones = %d \n"%(def1, len(coneangles)))
				assignments = assign_projangles(projangles, coneangles)
				for k in xrange(len(coneangles)):
					if(len(assignements[k]) > 0):
						refsincone = even_angles(delta, method = ref_a, symmetry = sym)
						ant = 1.5*an[N_step]
						refsincone = cone_ang( refsincone, coneangles[k][0], coneangles[k][1], ant )
						refrings = refprojs( volf, kb, refsincone, cnx, cny, numr, "F", wr_four )
						#    match projections to its cone using an as a distance.
						for im in assignments[k]:
							#  It needs list_of_reference_angles
							peak, pixer[im] = proj_ali_incore_local(data[im], refrings, numr, xrng[N_step], yrng[N_step], step[N_step], an[N_step], finfo, sym = sym)
							data[im].set_attr("previousmax", peak)

			del volft, kb

			if myid == main_node:
				print_msg("Time of alignment = %d\n"%(time()-start_time))
				start_time = time()

			#output pixel errors
			pass#IMPORTIMPORTIMPORT from mpi import mpi_gatherv
			recvbuf = mpi_gatherv(pixer, nima, MPI_FLOAT, recvcount, disps, MPI_FLOAT, main_node, MPI_COMM_WORLD)
			mpi_barrier(MPI_COMM_WORLD)
			terminate = 0
			if myid == main_node:
				recvbuf = map(float, recvbuf)
				pass#IMPORTIMPORTIMPORT from statistics import hist_list
				lhist = 20
				region, histo = hist_list(recvbuf, lhist)
				if region[0] < 0.0:  region[0] = 0.0
				msg = "      Histogram of pixel errors\n      ERROR       number of particles\n"
				print_msg(msg)
				for lhx in xrange(lhist):
					msg = " %10.3f     %7d\n"%(region[lhx], histo[lhx])
					print_msg(msg)
				# Terminate if 95% within 1 pixel error
				im = 0
				for lhx in xrange(lhist):
					if region[lhx] > 1.0: break
					im += histo[lhx]
				precn = 100*float(total_nima-im)/float(total_nima)
				msg = " Number of particles that changed orientations %7d, percentage of total: %5.1f\n"%(total_nima-im, precn)
				print_msg(msg)
				if precn <= termprec:  terminate = 1
				del region, histo
			del recvbuf
			terminate = mpi_bcast(terminate, 1, MPI_INT, 0, MPI_COMM_WORLD)
			terminate = int(terminate[0])

			if center == -1 and sym[0] == 'c':
				pass#IMPORTIMPORTIMPORT from utilities      import estimate_3D_center_MPI, rotate_3D_shift
				cs[0], cs[1], cs[2], dummy, dummy = estimate_3D_center_MPI(data, total_nima, myid, number_of_proc, main_node)
				if myid == main_node:
					msg = " Average center x = %10.3f        Center y = %10.3f        Center z = %10.3f\n"%(cs[0], cs[1], cs[2])
					print_msg(msg)
				if int(sym[1]) > 1:
					cs[0] = cs[1] = 0.0
					if myid == main_node:
						print_msg("For symmetry group cn (n>1), we only center the volume in z-direction\n")
				cs = mpi_bcast(cs, 3, MPI_FLOAT, main_node, MPI_COMM_WORLD)
				cs = [-float(cs[0]), -float(cs[1]), -float(cs[2])]
				rotate_3D_shift(data, cs)

			# write out headers, under MPI writing has to be done sequentially
			mpi_barrier(MPI_COMM_WORLD)
			par_str = ['xform.projection', 'previousmax', 'ID']
			if myid == main_node:
	   			if(file_type(stack) == "bdb"):
	        			pass#IMPORTIMPORTIMPORT from utilities import recv_attr_dict_bdb
	        			recv_attr_dict_bdb(main_node, stack, data, par_str, image_start, image_end, number_of_proc)
	        		else:
	        			pass#IMPORTIMPORTIMPORT from utilities import recv_attr_dict
	        			recv_attr_dict(main_node, stack, data, par_str, image_start, image_end, number_of_proc)
				print_msg("Time to write header information= %d\n"%(time()-start_time))
				start_time = time()
	        	else:	       send_attr_dict(main_node, data, par_str, image_start, image_end)

			if CTF: vol, fscc = rec3D_MPI(data, snr, sym, fscmask, os.path.join(outdir, "resolution%04d"%(total_iter)), myid, main_node, npad = npad)
			else:   vol, fscc = rec3D_MPI_noCTF(data, sym, fscmask, os.path.join(outdir, "resolution%04d"%(total_iter)), myid, main_node, npad = npad)

			if myid == main_node:
				print_msg("3D reconstruction time = %d\n"%(time()-start_time))
				start_time = time()

			if fourvar:
			#  Compute Fourier variance
				for im in xrange(nima):
					original_data[im].set_attr( 'xform.projection', data[im].get_attr('xform.projection') )
				varf = varf3d_MPI(original_data, ssnr_text_file = os.path.join(outdir, "ssnr%04d"%(total_iter)), mask2D = None, reference_structure = vol, ou = last_ring, rw = 1.0, npad = 1, CTF = CTF, sign = 1, sym =sym, myid = myid)
				if myid == main_node:
					print_msg("Time to calculate 3D Fourier variance= %d\n"%(time()-start_time))
					start_time = time()
					varf = 1.0/varf
			else:  varf = None

			if myid == main_node:
				drop_image(vol, os.path.join(outdir, "vol%04d.hdf"%(total_iter)))
				ref_data[2] = vol
				ref_data[3] = fscc
				ref_data[4] = varf
				#  call user-supplied function to prepare reference image, i.e., center and filter it
				vol, cs = user_func(ref_data)
				drop_image(vol, os.path.join(outdir, "volf%04d.hdf"%(total_iter)))

			del varf
			bcast_EMData_to_all(vol, myid, main_node)


	if myid == main_node: print_end_msg("ali3dlocal_MPI")
'''
'''16
def Xali3d_shc0MPI(stack, ref_vol, outdir, maskfile = None, ir = 1, ou = -1, rs = 1, 
            xr = "4 2 2 1", yr = "-1", ts = "1 1 0.5 0.25", delta = "10 6 4 4", an = "-1", apsi = "-1", deltapsi = "-1", startpsi = "-1",
	    center = -1, maxit = 5, CTF = False, snr = 1.0,  ref_a = "S", sym = "c1",  user_func_name = "ref_ali3d",
	    fourvar = True, npad = 4, debug = False, termprec = 0.0, gamma=-1):

	pass#IMPORTIMPORTIMPORT from alignment       import Numrinit, prepare_refrings, proj_ali_incore, proj_ali_incore_local, proj_ali_incore_local_psi, shc
	pass#IMPORTIMPORTIMPORT from utilities       import model_circle, get_image, drop_image, get_input_from_string
	pass#IMPORTIMPORTIMPORT from utilities       import bcast_list_to_all, bcast_number_to_all, bcast_EMData_to_all
	pass#IMPORTIMPORTIMPORT from utilities       import send_attr_dict, get_params_proj, file_type
	pass#IMPORTIMPORTIMPORT import os
	pass#IMPORTIMPORTIMPORT import types
	pass#IMPORTIMPORTIMPORT from utilities       import print_begin_msg, print_end_msg, print_msg
	pass#IMPORTIMPORTIMPORT from mpi             import mpi_bcast, mpi_comm_size, mpi_comm_rank, MPI_FLOAT, MPI_COMM_WORLD, mpi_barrier, MPI_INT
	pass#IMPORTIMPORTIMPORT from projection      import prep_vol, prgs
	pass#IMPORTIMPORTIMPORT from statistics      import hist_list, varf3d_MPI
	pass#IMPORTIMPORTIMPORT from applications    import MPI_start_end
	pass#IMPORTIMPORTIMPORT from math            import sqrt, acos, radians
	pass#IMPORTIMPORTIMPORT from random          import shuffle

	if gamma > 0:
		gamma = radians(gamma)

	number_of_proc = mpi_comm_size(MPI_COMM_WORLD)
	myid           = mpi_comm_rank(MPI_COMM_WORLD)
	main_node = 0
	
	if os.path.exists(outdir):  ERROR('Output directory exists, please change the name and restart the program', "ali3d_MPI", 1, myid)
	mpi_barrier(MPI_COMM_WORLD)

	if myid == main_node:
		os.mkdir(outdir)
		pass#IMPORTIMPORTIMPORT import global_def
		sparx_global_def.LOGFILE =  os.path.join(outdir, sparx_global_def.LOGFILE)
		print_begin_msg("ali3d_shcMPI")
	mpi_barrier(MPI_COMM_WORLD)

	if debug:
		pass#IMPORTIMPORTIMPORT from time import sleep
		while not os.path.exists(outdir):
			print  "Node ",myid,"  waiting..."
			sleep(5)

		info_file = os.path.join(outdir, "progress%04d"%myid)
		finfo = open(info_file, 'w')
	else:
		finfo = None

	xrng        = get_input_from_string(xr)
	if  yr == "-1":  yrng = xrng
	else          :  yrng = get_input_from_string(yr)
	step        = get_input_from_string(ts)
	delta       = get_input_from_string(delta)
	lstp = min(len(xrng), len(yrng), len(step), len(delta))
	if an == "-1":
		an = [-1] * lstp
	else:
		an = get_input_from_string(an)

	if apsi == "-1":
		apsi = [-1] * lstp
	else:
		apsi = get_input_from_string(apsi)

	if deltapsi == "-1":
		deltapsi = [-1] * lstp
	else:
		deltapsi = get_input_from_string(deltapsi)

	if startpsi == "-1":
		startpsi = [-1] * lstp
	else:
		startpsi = get_input_from_string(startpsi)

	first_ring  = int(ir)
	rstep       = int(rs)
	last_ring   = int(ou)
	max_iter    = int(maxit)
	center      = int(center)

	vol     = EMData()
	vol.read_image(ref_vol)
	nx      = vol.get_xsize()
	if last_ring < 0:	last_ring = int(nx/2) - 2

	if myid == main_node:
		pass#IMPORTIMPORTIMPORT import user_functions
		user_func = sparx_user_functions.factory[user_func_name]

		print_msg("Input stack                 : %s\n"%(stack))
		print_msg("Reference volume            : %s\n"%(ref_vol))	
		print_msg("Output directory            : %s\n"%(outdir))
		print_msg("Maskfile                    : %s\n"%(maskfile))
		print_msg("Inner radius                : %i\n"%(first_ring))
		print_msg("Outer radius                : %i\n"%(last_ring))
		print_msg("Ring step                   : %i\n"%(rstep))
		print_msg("X search range              : %s\n"%(xrng))
		print_msg("Y search range              : %s\n"%(yrng))
		print_msg("Translational step          : %s\n"%(step))
		print_msg("Angular step                : %s\n"%(delta))
		print_msg("Angular search range (phi and theta)       : %s\n"%(an))
		#print_msg("Angular search range (psi)                 : %s\n"%(apsi))
		#print_msg("Delta psi                   : %s\n"%(deltapsi))
		#print_msg("Start psi                   : %s\n"%(startpsi))
		print_msg("Maximum number of iterations : %i\n"%(max_iter))
		print_msg("Percentage of change for termination: %f\n"%(termprec))
		print_msg("Center type                 : %i\n"%(center))
		print_msg("CTF correction              : %s\n"%(CTF))
		print_msg("Signal-to-Noise Ratio       : %f\n"%(snr))
		print_msg("Reference projection method : %s\n"%(ref_a))
		print_msg("Symmetry group              : %s\n\n"%(sym))

	if maskfile:
		if type(maskfile) is types.StringType: mask3D = get_image(maskfile)
		else:                                  mask3D = maskfile
	else: mask3D = model_circle(last_ring, nx, nx, nx)

	numr	= Numrinit(first_ring, last_ring, rstep, "F")
	mask2D  = model_circle(last_ring,nx,nx) - model_circle(first_ring,nx,nx)

	fscmask = mask3D  #model_circle(last_ring,nx,nx,nx)  For a fancy mask circle would work better  PAP 7/21/11
	if CTF:
		pass#IMPORTIMPORTIMPORT from reconstruction import rec3D_MPI
		pass#IMPORTIMPORTIMPORT from filter         import filt_ctf
	pass#IMPORTIMPORTIMPORT else:	 from reconstruction import rec3D_MPI_noCTF

	if myid == main_node:
		if file_type(stack) == "bdb":
			pass#IMPORTIMPORTIMPORT from EMAN2db import db_open_dict
			dummy = db_open_dict(stack, True)
		active = EMUtil.get_all_attributes(stack, 'active')
		list_of_particles = []
		for im in xrange(len(active)):
			if active[im]:  list_of_particles.append(im)
		del active
		nima = len(list_of_particles)
	else:
		nima = 0
	total_nima = bcast_number_to_all(nima, source_node = main_node)

	if myid != main_node:
		list_of_particles = [-1]*total_nima
	list_of_particles = bcast_list_to_all(list_of_particles, myid,  source_node = main_node)

	image_start, image_end = MPI_start_end(total_nima, number_of_proc, myid)
	# create a list of images for each node
	list_of_particles = list_of_particles[image_start: image_end]
	nima = len(list_of_particles)
	if debug:
		finfo.write("image_start, image_end: %d %d\n" %(image_start, image_end))
		finfo.flush()

	data = EMData.read_images(stack, list_of_particles)
	if fourvar:  original_data = []
	for im in xrange(nima):
		data[im].set_attr('ID', list_of_particles[im])
		if fourvar: original_data.append(data[im].copy())
		if CTF:
			ctf_params = data[im].get_attr("ctf")
			st = Util.infomask(data[im], mask2D, False)
			data[im] -= st[0]
			data[im] = filt_ctf(data[im], ctf_params)
			data[im].set_attr('ctf_applied', 1)

	
	if debug:
		finfo.write( '%d loaded  \n' % nima )
		finfo.flush()
	if myid == main_node:
		# initialize data for the reference preparation function
		ref_data = [ mask3D, max(center,0), None, None, None, None ]
		# for method -1, switch off centering in user function

	pass#IMPORTIMPORTIMPORT from time import time

	#  this is needed for gathering of pixel errors
	disps = []
	recvcount = []
	for im in xrange(number_of_proc):
		if im == main_node :  disps.append(0)
		else:                 disps.append(disps[im-1] + recvcount[im-1])
		ib, ie = MPI_start_end(total_nima, number_of_proc, im)
		recvcount.append(ie-ib)

	
	mode = "F"
	pixer = [0.0]*nima
	cs = [0.0]*3
	total_iter = 0
	final_params = None
	final_volume = None
	final_volume_filtered = None
	# do the projection matching
	for N_step in xrange(lstp):
		##convert data into polar coordinates.  @ming.
		ky = int(2*yrng[N_step]/step[N_step]+0.5)/2;
		kx = int(2*xrng[N_step]/step[N_step]+0.5)/2;
		cimages = []
		for im in xrange(nima):
			nx = data[im].get_xsize()
			ny = data[im].get_ysize()
			#  center is in SPIDER convention
			cnx  = nx//2 + 1
			cny  = ny//2 + 1
			ims = [None]*(2*ky+1)*(2*kx+1)
			
			nring = len(numr)/3
			inr = numr[3*(nring-1)]
			for i in xrange(-ky, ky+1):
				iy = i*step[N_step]
				if inr+int(cny+iy) <= ny-1 and -inr + int(cny+iy) >=1:
					for j in xrange(-kx, kx+1):
						ix = j*step[N_step]
						if inr+int(cnx+ix) <= nx-1 and -inr + int(cnx+ix) >=1:
							cimage = Util.Polar2Dm(data[im], cnx+ix, cny+iy, numr, mode)
							Util.Normalize_ring( cimage, numr, 0)
							Util.Frngs(cimage, numr)
							ims[(i+ky)*(2*kx+1)+j+kx] = cimage
			cimages.append(ims)
			
	
		terminate = 0
		Iter = 0
		while Iter < max_iter and terminate == 0:

			Iter += 1
			total_iter += 1

			if myid == main_node:
				start_time = time()
				print_msg("\nITERATION #%3d,  inner iteration #%3d\nDelta = %5.2f, an = %5.2f, xrange = %5.2f, yrange = %5.2f,translational step = %5.2f\n"%(total_iter, Iter, delta[N_step], an[N_step], xrng[N_step], yrng[N_step], step[N_step]))
				#print_msg("\nITERATION #%3d,  inner iteration #%3d\nDelta = %4.1f, an = %5.2f, xrange = %5.2f, yrange = %5.2f, step = %5.2f, delta psi = %5.2f, start psi = %5.2f\n"%(total_iter, Iter, delta[N_step], an[N_step], xrng[N_step],yrng[N_step],step[N_step],deltapsi[N_step],startpsi[N_step]))

			#=========================================================================
			# build references
			volft, kb = prep_vol(vol)
			refrings = prepare_refrings(volft, kb, nx, delta[N_step], ref_a, sym, numr, True)
			lastdelta = delta[N_step]
			del volft, kb
			#=========================================================================

			if myid == main_node:
				print_msg("Time to prepare rings: %d\n" % (time()-start_time))
				start_time = time()

			#=========================================================================
			#  We assume previousmax exists
			"""
			if total_iter == 1:
				# adjust params to references, calculate psi+shifts, calculate previousmax
				for im in xrange(nima):
					stable = data[im].get_attr_default("stable", 0)
					if stable == 0:
						data[im].set_attr("previousmax", -1.0e23)
						data[im].set_attr("stable", 1)
					else:
						#print "  params  ",get_params_proj(data[im])
						peak, pixer[im] = proj_ali_incore_local(data[im],refrings,numr,0.0,0.0,1.0,delta[N_step]/4.0,finfo)
						data[im].set_attr("previousmax", peak)
						print "peak  ",im,peak,get_params_proj(data[im])
				if myid == main_node:
					print_msg("Time to calculate first psi+shifts+previousmax: %d\n" % (time()-start_time))
					start_time = time()
			"""
			#=========================================================================

			#=========================================================================
			# alignment
			iter_indexes = range(nima)
			shuffle(iter_indexes)
			for im in iter_indexes:
				pass#IMPORTIMPORTIMPORT from utilities import get_params_proj
				#print "  IN  ",im,get_params_proj(data[im]),data[im].get_attr("previousmax")
				peak, pixer[im], number_of_checked_refs, iref = \
					shc0(data[im], cimages[im], refrings, numr, xrng[N_step], yrng[N_step], step[N_step], an[N_step], sym, finfo)
				#print "  OU  ",im,get_params_proj(data[im]),data[im].get_attr("previousmax")
				if gamma > 0:
					n1 = refrings[iref].get_attr("n1")
					n2 = refrings[iref].get_attr("n2")
					n3 = refrings[iref].get_attr("n3")
					to_be_deleted = []
					for irr in xrange(len(refrings)):
						nn1 = refrings[irr].get_attr("n1")
						nn2 = refrings[irr].get_attr("n2")
						nn3 = refrings[irr].get_attr("n3")
						if abs(acos(n1*nn1 + n2*nn2 * n3*nn3)) < gamma:
							to_be_deleted.append( irr )
					if len(to_be_deleted) > 0:
						to_be_deleted.sort(reverse=True)
						for irr in to_be_deleted:
							del refrings[irr]
				elif gamma == 0:
						del refrings[iref]
			#=========================================================================
			del refrings

			if myid == main_node:
				print_msg("Time of alignment = %d\n"%(time()-start_time))
				start_time = time()
			#=========================================================================
			#output pixel errors, check stop criterion
			pass#IMPORTIMPORTIMPORT from mpi import mpi_gatherv
			recvbuf = mpi_gatherv(pixer, nima, MPI_FLOAT, recvcount, disps, MPI_FLOAT, main_node, MPI_COMM_WORLD)
			mpi_barrier(MPI_COMM_WORLD)
			terminate = 0
			if myid == main_node:
				recvbuf = map(float, recvbuf)
				pass#IMPORTIMPORTIMPORT from statistics import hist_list
				lhist = 20
				region, histo = hist_list(recvbuf, lhist)
				if region[0] < 0.0:  region[0] = 0.0
				msg = "      Histogram of pixel errors\n      ERROR       number of particles\n"
				print_msg(msg)
				for lhx in xrange(lhist):
					msg = " %10.3f     %7d\n"%(region[lhx], histo[lhx])
					print_msg(msg)
				# Terminate if 95% within 0.1 pixel error
				im = 0
				for lhx in xrange(lhist):
					if region[lhx] > 0.1: break
					im += histo[lhx]
				precn = 100*float(total_nima-im)/float(total_nima)
				msg = " Number of particles that changed orientations %7d, percentage of total: %5.1f\n"%(total_nima-im, precn)
				print_msg(msg)
				if precn <= termprec:  terminate = 1
				del region, histo
			del recvbuf
			terminate = mpi_bcast(terminate, 1, MPI_INT, 0, MPI_COMM_WORLD)
			terminate = int(terminate[0])
			#=========================================================================

			#=========================================================================
			# centering
			if center == -1 and sym[0] == 'c':
				pass#IMPORTIMPORTIMPORT from utilities      import estimate_3D_center_MPI, rotate_3D_shift
				cs[0], cs[1], cs[2], dummy, dummy = estimate_3D_center_MPI(data, total_nima, myid, number_of_proc, main_node)
				if myid == main_node:
					msg = " Average center x = %10.3f        Center y = %10.3f        Center z = %10.3f\n"%(cs[0], cs[1], cs[2])
					print_msg(msg)
				if int(sym[1]) > 1:
					cs[0] = cs[1] = 0.0
					if myid == main_node:
						print_msg("For symmetry group cn (n>1), we only center the volume in z-direction\n")
				cs = mpi_bcast(cs, 3, MPI_FLOAT, main_node, MPI_COMM_WORLD)
				cs = [-float(cs[0]), -float(cs[1]), -float(cs[2])]
				rotate_3D_shift(data, cs)
			#=========================================================================

			#=========================================================================
			# write out headers, under MPI writing has to be done sequentially
			mpi_barrier(MPI_COMM_WORLD)
			# It takes plenty of times, so do it once in awhile...
			if( True ): #total_iter>3 and total_iter%5 == 0 ):
				par_str = ['xform.projection', 'previousmax', 'ID']
				if myid == main_node:
					if(file_type(stack) == "bdb"):
						pass#IMPORTIMPORTIMPORT from utilities import recv_attr_dict_bdb
						recv_attr_dict_bdb(main_node, stack, data, par_str, image_start, image_end, number_of_proc)
					else:
						pass#IMPORTIMPORTIMPORT from utilities import recv_attr_dict
						recv_attr_dict(main_node, stack, data, par_str, image_start, image_end, number_of_proc)
					"""
					# save parameters to file
					paro = [None]*total_nima
					projs_headers = EMData.read_images(stack, range(total_nima), True)
					for im in xrange(total_nima):
						a1,a2,a3,a4,a5 = get_params_proj(projs_headers[im])
						previousmax = projs_headers[im].get_attr("previousmax")
						paro[im] = [a1,a2,a3,a4,a5,previousmax]
					pass#IMPORTIMPORTIMPORT from utilities import write_text_row
					write_text_row(paro,os.path.join(outdir, "params%04d.txt"%(total_iter)))
					final_params = paro
					del projs_headers
					del paro
					# ------- end of saving parameters to file
					print_msg("Time to write header information= %d\n"%(time()-start_time))
					start_time = time()
					"""
				else:
					send_attr_dict(main_node, data, par_str, image_start, image_end)
			#=========================================================================

			#=========================================================================
			# volume reconstruction
			#vol_previous = vol
			if CTF: vol, fscc = rec3D_MPI(data, snr, sym, fscmask, os.path.join(outdir, "resolution%04d"%(total_iter)), myid, main_node, npad = npad)
			else:   vol, fscc = rec3D_MPI_noCTF(data, sym, fscmask, os.path.join(outdir, "resolution%04d"%(total_iter)), myid, main_node, npad = npad)
			# log
			if myid == main_node:
				print_msg("3D reconstruction time = %d\n"%(time()-start_time))
				start_time = time()
			if fourvar:
			#  Compute Fourier variance
				for im in xrange(nima):
					original_data[im].set_attr( 'xform.projection', data[im].get_attr('xform.projection') )
				varf = varf3d_MPI(original_data, ssnr_text_file = os.path.join(outdir, "ssnr%04d"%(total_iter)), mask2D = None, reference_structure = vol, ou = last_ring, rw = 1.0, npad = 1, CTF = CTF, sign = 1, sym =sym, myid = myid)
				if myid == main_node:
					print_msg("Time to calculate 3D Fourier variance= %d\n"%(time()-start_time))
					start_time = time()
					varf = 1.0/varf
			else:
				varf = None
			# user functions + save volume
			if myid == main_node:
				#drop_image(vol, os.path.join(outdir, "vol%04d.hdf"%(total_iter)))
				ref_data[2] = vol
				ref_data[3] = fscc
				ref_data[4] = varf
				#  call user-supplied function to prepare reference image, i.e., center and filter it
				vol, cs = user_func(ref_data)
				drop_image(vol, os.path.join(outdir, "volf%04d.hdf"%(total_iter)))
				#print_msg("Euclidean distance between the current and the previous volume: " + str(sqrt(vol.cmp("SqEuclidean",vol_previous,{"mask":mask3D,"zeromask":0,"normto":0}))) + "\n")
				print_msg("L2 norm of the volume: " + str(vol.cmp("dot", vol, {"negative":0, "mask":mask3D})) + "\n")
			del varf
			# broadcast volume
			bcast_EMData_to_all(vol, myid, main_node)
			#=========================================================================
		del cimages

	if myid == main_node: 
		print_end_msg("ali3d_shcMPI")
'''
"""17
			if total_iter == 1:
				# adjust params to references, calculate psi+shifts, calculate previousmax
				for im in xrange(nima):
					stable = data[im].get_attr_default("stable", 0)
					if stable == 0:
						data[im].set_attr("previousmax", -1.0e23)
						data[im].set_attr("stable", 1)
					else:
						#print "  params  ",get_params_proj(data[im])
						peak, pixer[im] = proj_ali_incore_local(data[im],refrings,numr,0.0,0.0,1.0,delta[N_step]/4.0,finfo)
						data[im].set_attr("previousmax", peak)
						print "peak  ",im,peak,get_params_proj(data[im])
				if myid == main_node:
					print_msg("Time to calculate first psi+shifts+previousmax: %d\n" % (time()-start_time))
					start_time = time()
			"""
'''18
					# save parameters to file
					paro = [None]*total_nima
					projs_headers = EMData.read_images(stack, range(total_nima), True)
					for im in xrange(total_nima):
						a1,a2,a3,a4,a5 = get_params_proj(projs_headers[im])
						previousmax = projs_headers[im].get_attr("previousmax")
						paro[im] = [a1,a2,a3,a4,a5,previousmax]
					pass#IMPORTIMPORTIMPORT from utilities import write_text_row
					write_text_row(paro,os.path.join(outdir, "params%04d.txt"%(total_iter)))
					final_params = paro
					del projs_headers
					del paro
					# ------- end of saving parameters to file
					print_msg("Time to write header information= %d\n"%(time()-start_time))
					start_time = time()
					'''
'''19
	if(myid == main_node):	
		total_nima = EMUtil.get_image_count(stack)
		list_of_particles = range(total_nima)
	
	else:
		total_nima =0

	total_nima = bcast_number_to_all(total_nima, source_node = main_node)

	if(myid != main_node):
		list_of_particles = [-1]*total_nima

	list_of_particles = bcast_list_to_all(list_of_particles, myid,  source_node = main_node)

	image_start, image_end = MPI_start_end(total_nima, number_of_proc, myid)
	# create a list of images for each node
	list_of_particles = list_of_particles[image_start: image_end]
	nima = len(list_of_particles)
	'''
"""20
		#  Trying to use ISAC code for EQ-Kmeans  PAP 03/21/2015
		if myid == main_node:

			for imrefa in xrange(numrefang):
				pass#IMPORTIMPORTIMPORT from utilities import findall
				N = findall(imrefa, assigntorefa)
				current_nima = len(N)
				if( current_nima >= numref and report_error == 0):
					tasi = [[] for iref in xrange(numref)]
					maxasi = current_nima//numref
					nt = current_nima
					kt = numref
					K = range(numref)

					d = empty( (numref, current_nima), dtype = float32)
					for ima in xrange(current_nima):
						for iref in xrange(numref):  d[iref][ima] = dtot[iref][N[ima]]

			d = empty( (numref, total_nima), dtype = float32)
			for ima in xrange(total_nima):
				for iref in xrange(numref):  d[iref][ima] = dtot[iref][N[ima]]
			id_list_long = Util.assign_groups(str(d.__array_interface__['data'][0]), numref, nima) # string with memory address is passed as parameters
			del d
			id_list = [[] for i in xrange(numref)]
			maxasi = total_nima/numref
			for i in xrange(maxasi*numref):
				id_list[i/maxasi].append(id_list_long[i])
			for i in xrange(total_nima%maxasi):
				id_list[id_list_long[-1]].append(id_list_long[maxasi*numref+i])
			for iref in xrange(numref):
				id_list[iref].sort()

			assignment = [0]*total_nima
			for iref in xrange(numref):
				for im in id_list[iref]: assignment[im] = iref
		else:
			assignment = [0]*total_nima
		mpi_barrier(MPI_COMM_WORLD)
		#belongsto = mpi_bcast(belongsto, nima, MPI_INT, main_node, MPI_COMM_WORLD)
		#belongsto = map(int, belongsto)
		"""
"""21
		if myid == main_node:
			assignment = [0]*total_nima
			for iref in xrange(numref):
				for im in xrange(len(asi[iref])):
					assignment[asi[iref][im]] = iref
			del asi
		"""
'''22
		if myid == main_node:
			SA = False
			maxasi = total_nima//numref
			asi = [[] for iref in xrange(numref)]
			nt = total_nima
			kt = numref
			K = range(numref)
			N = range(total_nima)

			while nt > 0 and kt > 1:
				l = d.argmax()
				group = l//total_nima
				ima   = l-total_nima*group
				if SA:
					J = [0.0]*numref
					sJ = 0
					Jc = [0.0]*numref
					for iref in xrange(numref):
						J[iref] = exp(d[iref][ima]/T)
						sJ += J[iref]
					for iref in xrange(numref):
						J[iref] /= sJ
					Jc[0] = J[0]
					for iref in xrange(1, numref):
						Jc[iref] = Jc[iref-1]+J[iref]
					sss = random()
					for group in xrange(numref):
						if( sss <= Jc[group]): break
				asi[group].append(N[ima])
				for iref in xrange(numref):  d[iref][ima] = -1.e10
				nt -= 1
				masi = len(asi[group])
				if masi == maxasi:
					for im in xrange(total_nima):  d[group][im] = -1.e10
					kt -= 1
			else:
				mas = [len(asi[iref]) for iref in xrange(numref)]
				group = mas.index(min(mas))
				del mas
				for im in xrange(total_nima):
					kt = 0
					go = True
					while(go and kt < numref):
						if d[kt][im] > -1.e10:
							asi[group].append(im)
							go = False
						kt += 1

			assignment = [0]*total_nima
			for iref in xrange(numref):
				for im in xrange(len(asi[iref])):
					assignment[asi[iref][im]] = iref

			del asi, d, N, K
			if  SA:  del J, Jc


		else:
			assignment = []
		'''
"""23
	if runtype=="REFINEMENT":
		par_str = ['xform.projection', 'ID', 'group']
	else:
		par_str = ['group', 'ID' ]
	if myid == main_node:
		pass#IMPORTIMPORTIMPORT from utilities import file_type
		if file_type(stack) == "bdb":
			pass#IMPORTIMPORTIMPORT from utilities import recv_attr_dict_bdb
			recv_attr_dict_bdb(main_node, stack, data, par_str, image_start, image_end, number_of_proc)
		else:
			pass#IMPORTIMPORTIMPORT from utilities import recv_attr_dict
			recv_attr_dict(main_node, stack, data, par_str, image_start, image_end, number_of_proc)
	else:		send_attr_dict(main_node, data, par_str, image_start, image_end)
	"""
"""24
			if CTF:
				previous_defocus = -1.0
				if runtype=="REFINEMENT":
					start_time = time()
					prjref = prgq( volft, kb, nx, delta[N_step], ref_a, sym, MPI=True)
					if myid == main_node:
						log.add( "Calculation of projections: %d" % (time()-start_time) );start_time = time()
					del volft, kb
			else:
				if runtype=="REFINEMENT":
					start_time = time()
					refrings = prepare_refrings( volft, kb, nx, delta[N_step], ref_a, syms[0], numr)
					if myid == main_node:
						log.add( "Initial time to prepare rings: %d" % (time()-start_time) );start_time = time()
					del volft, kb
			"""
"""25
				if CTF:
					ctf = data[im].get_attr( "ctf" )
					if runtype=="REFINEMENT":
						if ctf.defocus != previous_defocus:
							previous_defocus = ctf.defocus
							rstart_time = time()
							refrings = gen_rings_ctf( prjref, nx, ctf, numr)
							if myid == main_node:
								log.add( "Repeated time to prepare rings: %d" % (time()-rstart_time) );rstart_time = time()
				"""
"""26
	if runtype=="REFINEMENT":
		par_str = ['xform.projection', 'ID', 'group']
	else:
		par_str = ['group', 'ID' ]
	if myid == main_node:
		pass#IMPORTIMPORTIMPORT from utilities import file_type
		if file_type(stack) == "bdb":
			pass#IMPORTIMPORTIMPORT from utilities import recv_attr_dict_bdb
			recv_attr_dict_bdb(main_node, stack, data, par_str, image_start, image_end, number_of_proc)
		else:
			pass#IMPORTIMPORTIMPORT from utilities import recv_attr_dict
			recv_attr_dict(main_node, stack, data, par_str, image_start, image_end, number_of_proc)
	else:		send_attr_dict(main_node, data, par_str, image_start, image_end)
	"""
"""27
	if myid == main_node:
		os.mkdir(outdir)
		pass#IMPORTIMPORTIMPORT import global_def
		sparx_global_def.LOGFILE =  os.path.join(outdir, sparx_global_def.LOGFILE)
		print_begin_msg("local_ali3d_MPI")
		pass#IMPORTIMPORTIMPORT import user_functions
		user_func = sparx_user_functions.factory[user_func_name]
		if CTF:
			ima = EMData()
			ima.read_image(stack, 0)
			ctf_applied = ima.get_attr_default("ctf_applied", 0)
			del ima
			if ctf_applied == 1:  ERROR("local_ali3d does not work for CTF-applied data", "local_ali3d_MPI", 1,myid)
	mpi_barrier(mpi_comm)
	"""
"""28
	if myid == main_node:
		pass#IMPORTIMPORTIMPORT import user_functions
		user_func = sparx_user_functions.factory[user_func_name]

		print_msg("Input stack                 : %s\n"%(stack))
		print_msg("Output directory            : %s\n"%(outdir))
		print_msg("Maskfile                    : %s\n"%(ali3d_options.mask3D))
		print_msg("Outer radius                : %i\n"%(last_ring))
		print_msg("Angular search range        : %s\n"%(delta))
		print_msg("Shift search range          : %f\n"%(ts))
		print_msg("Maximum iteration           : %i\n"%(max_iter))
		print_msg("Center type                 : %i\n"%(center))
		print_msg("CTF correction              : %s\n"%(CTF))
		print_msg("Signal-to-Noise Ratio       : %f\n"%(snr))
		print_msg("Symmetry group              : %s\n"%(sym))
		print_msg("Chunk size                  : %f\n\n"%(chunk))
		print_msg("User function               : %s\n"%(user_func_name))
	"""
"""29
		if( type(stack) is types.StringType ):
			pass#IMPORTIMPORTIMPORT from EMAN2 import Vec2f, Transform
			pass#IMPORTIMPORTIMPORT from EMAN2db import db_open_dict
			DB = db_open_dict(stack)
			for im in xrange(len(params)):
				t = Transform({"type":"spider","phi":params[im][0],"theta":params[im][1],"psi":params[im][2]})
				t.set_trans(Vec2f(-params[im][3], -params[im][4]))
				DB.set_attr(particle_ids[im], "xform.projection", t)
			DB.close()
		else:
			for im in xrange(len(params)): set_params_proj(stack[particle_ids[im]], params[im])

		log.add("Time to write header information= %d\n"%(time()-start_time))
		"""
'''30
					if myid == main_node:
						print_msg("\nphi = %5.2f, theta = %5.2f, psi=%5.2f\n"%( refrings[i].get_attr('phi'), refrings[i].get_attr('theta'), refrings[i].get_attr('psi') ) )
					if myid == main_node:
						print_msg("\nlen(ref1) = %4d, len(ref2) = %4d\n"%(len(refrings1), len(refrings2)) )
					'''		
"""31
					Logic of searches:
						refrings1 (even point-group symmetry AND theta = 90.), and refrings2 ( odd point-group symmetry AND any theta (including theta=90), OR even point-group symmetry AND theta <> 90.)
						an == -1 exhaustive search
						psi_max - how far rotation in plane can can deviate from 90 or 270 degrees
						psi_max - is used in both exhaustive and local searches
						:::
						>even point-group symmetry AND theta = 90. 
							exhaustive   proj_ali_helical_90   psi_max
											Util.multiref_polar_ali_helical_90  psi_max
											Crosrng_sm_psi  (in util_sparx.cpp)  psi_max - WILL SEARCH for BOTH PSI=0 AND 180 NO MATTER WHAT
																					flag - no mirror or mirror, DOES NOT CHECK MIRRORED
											
							local       reference projections phi= [0,180,delta], theta=90
										proj_ali_helical_90_local   an[N_step], psi_max,  (alignment.py)
											Util.multiref_polar_ali_helical_90_local   psi_max
											Uses the following construct
											if ((psi-90.0f) < 90.0f) retvals = Crosrng_sm_psi(crefim[iref], cimage, numr,   0, 0, psi_max);
											else                     retvals = Crosrng_sm_psi(crefim[iref], cimage, numr, 180, 0, psi_max);
										where Crosrng_sm_psi will do search for psi around one angle and mirror is NOT checked.

						>odd point-group symmetry AND any theta (including theta=90), OR even point-group symmetry AND theta <> 90.
							exhaustive   proj_ali_helical           psi_max
											Util.multiref_polar_ali_helical  psi_max
												CALLS Crosrng_pi twice, for 0 and for 180, thus duplicates the work!!
											Crosrng_psi  checks MIRROR
											
							local        proj_ali_helical_local   psi_max
											Util.multiref_polar_ali_helical_local  psi_max
												Uses the following construct
												if (mirror_only == true) {
													if ((psi-90) < 90) retvals = Crosrng_sm_psi(crefim[iref], cimage, numr,   0, 1, psi_max);
													else               retvals = Crosrng_sm_psi(crefim[iref], cimage, numr, 180, 1, psi_max); 
												} else {
													if ((psi-90) < 90) retvals = Crosrng_sm_psi(crefim[iref], cimage, numr,   0, 0, psi_max);
													else               retvals = Crosrng_sm_psi(crefim[iref], cimage, numr, 180, 0, psi_max);
												}
											where Crosrng_sm_psi will do search for psi around one angle and do mirror or not.

				"""
"""32
			if fourvar:
			#  Compute Fourier variance
				for im in xrange(nima):
					original_data[im].set_attr( 'xform.projection', data[im].get_attr('xform.projection') )
				varf = varf3d_MPI(original_data, ssnr_text_file = os.path.join(outdir, "ssnr%04d"%(total_iter)), mask2D = None, reference_structure = vol, ou = last_ring, rw = 1.0, npad = 1, CTF = CTF, sign = 1, sym =sym, myid = myid)
				if myid == main_node:
					print_msg("Time to calculate 3D Fourier variance= %d\n"%(time()-start_time))
					start_time = time()
					varf = 1.0/varf
					drop_image(varf, os.path.join(outdir, "varf%04d.hdf"%(total_iter)))
			else:  varf = None
			"""
'''33
def gchelix_MPI(stack, ref_vol, outdir, maskfile, ir, ou, rs, xr, ynumber,\
	txs, delta, initial_theta, delta_theta, an, maxit, CTF, snr, dp, ndp, dp_step, dphi, ndphi, dphi_step, psi_max,\
	rmin, rmax, fract, nise, npad, sym, user_func_name, datasym,\
	pixel_size, debug, y_restrict, WRAP):

	pass#IMPORTIMPORTIMPORT from alignment      import Numrinit, prepare_refrings, proj_ali_helical, proj_ali_helical_90, proj_ali_helical_local, proj_ali_helical_90_local, helios,helios7
	pass#IMPORTIMPORTIMPORT from utilities      import model_circle, get_image, drop_image, get_input_from_string, pad, model_blank
	pass#IMPORTIMPORTIMPORT from utilities      import bcast_list_to_all, bcast_number_to_all, reduce_EMData_to_root, bcast_EMData_to_all
	pass#IMPORTIMPORTIMPORT from utilities      import send_attr_dict
	pass#IMPORTIMPORTIMPORT from utilities      import get_params_proj, set_params_proj, file_type
	pass#IMPORTIMPORTIMPORT from fundamentals   import rot_avg_image
	pass#IMPORTIMPORTIMPORT from pixel_error    import max_3D_pixel_error
	pass#IMPORTIMPORTIMPORT import os
	pass#IMPORTIMPORTIMPORT import types
	pass#IMPORTIMPORTIMPORT from utilities      import print_begin_msg, print_end_msg, print_msg
	pass#IMPORTIMPORTIMPORT from mpi            import mpi_bcast, mpi_comm_size, mpi_comm_rank, MPI_FLOAT, MPI_COMM_WORLD, mpi_barrier
	pass#IMPORTIMPORTIMPORT from mpi            import mpi_recv,  mpi_send
	pass#IMPORTIMPORTIMPORT from mpi            import mpi_reduce, MPI_INT, MPI_SUM
	pass#IMPORTIMPORTIMPORT from filter         import filt_ctf
	pass#IMPORTIMPORTIMPORT from projection     import prep_vol, prgs
	pass#IMPORTIMPORTIMPORT from statistics     import hist_list, varf3d_MPI
	pass#IMPORTIMPORTIMPORT from applications   import MPI_start_end
	pass#IMPORTIMPORTIMPORT from EMAN2 import Vec2f
	pass#IMPORTIMPORTIMPORT from string    import lower,split
	pass#IMPORTIMPORTIMPORT from math import cos, pi

	number_of_proc = mpi_comm_size(MPI_COMM_WORLD)
	myid           = mpi_comm_rank(MPI_COMM_WORLD)
	main_node = 0

	if myid == 0:
		if os.path.exists(outdir):  nx = 1
		else:  nx = 0
	else:  nx = 0
	ny = bcast_number_to_all(nx, source_node = main_node)

	if ny == 1:  ERROR('Output directory exists, please change the name and restart the program', "ihrsr_MPI", 1,myid)
	mpi_barrier(MPI_COMM_WORLD)

	if myid == main_node:
		os.mkdir(outdir)
		pass#IMPORTIMPORTIMPORT import global_def
		sparx_global_def.LOGFILE =  os.path.join(outdir, sparx_global_def.LOGFILE)
	mpi_barrier(MPI_COMM_WORLD)


	nlprms = (2*ndp+1)*(2*ndphi+1)
	if nlprms< number_of_proc:
		ERROR('number of CPUs is larger than the number of helical search, please reduce it or at this moment modify ndp,dphi in the program', "ihrsr_MPI", 1,myid)


	if debug:
		pass#IMPORTIMPORTIMPORT from time import sleep
		while not os.path.exists(outdir):
			print  "Node ",myid,"  waiting..."
			sleep(5)

		info_file = os.path.join(outdir, "progress%04d"%myid)
		finfo = open(info_file, 'w')
	else:
		finfo = None

	symref = "s"+sym

	ref_a= "P"
	symmetryLower = sym.lower()
	symmetry_string = split(symmetryLower)[0]

	xrng        = get_input_from_string(xr)
	y_restrict       = get_input_from_string(y_restrict)
	ynumber	    = get_input_from_string(ynumber)
	for i in xrange(len(ynumber)):
		if(ynumber[i]%2==1): ynumber[i]=ynumber[i]+1
	yrng =[]

	for i in xrange(len(xrng)): yrng.append(dp/2)
	
	stepx        = get_input_from_string(txs)
	delta       = get_input_from_string(delta)
	lstp = min(len(xrng), len(yrng), len(stepx), len(delta))
	if an == "-1": an = [-1] * lstp
	else:          an = get_input_from_string(an)

	first_ring  = int(ir)
	rstep       = int(rs)
	last_ring   = int(ou)
	max_iter    = int(maxit)

	vol     = EMData()
	vol.read_image(ref_vol)
	nx      = vol.get_xsize()
	ny      = vol.get_ysize()
	nz      = vol.get_zsize()
	nmax = max(nx, ny, nz)
	if ( nx == ny ):
		if( nx == nz):
			xysize = -1
			zsize = -1
		elif( nx < nz):
			xysize = nx
			zsize = -1
		else:
			zsize = nz
			xysize = -1
	
	else:
		ERROR('the x and y size have to be same, please change the reference volume and restart the program', "ihrsr_MPI", 1,myid)

	if last_ring < 0:	last_ring = int(nx/2) - 2

	if myid == main_node:
		pass#IMPORTIMPORTIMPORT import user_functions
		user_func = sparx_user_functions.factory[user_func_name]

		print_msg("Input stack                               : %s\n"%(stack))
		print_msg("Reference volume                          : %s\n"%(ref_vol))	
		print_msg("Output directory                          : %s\n"%(outdir))
		print_msg("Maskfile                                  : %s\n"%(maskfile))
		print_msg("Inner radius                              : %i\n"%(first_ring))
		print_msg("Outer radius                              : %i\n"%(last_ring))
		print_msg("Ring step                                 : %i\n"%(rstep))
		print_msg("X search range                            : %s\n"%(xrng))
		print_msg("Y number                                  : %s\n"%(ynumber))
		print_msg("Translational stepx                       : %s\n"%(stepx))
		print_msg("Angular step                              : %s\n"%(delta))
		print_msg("Angular search range                      : %s\n"%(an))
		print_msg("Initial Theta                             : %s\n"%(initial_theta))
		print_msg("min radius for helical search (in pix)    : %5.4f\n"%(rmin))
		print_msg("max radius for helical search (in pix)    : %5.4f\n"%(rmax))
		print_msg("fraction of volume used for helical search: %5.4f\n"%(fract))
		print_msg("initial symmetry - angle                  : %5.4f\n"%(dphi))
		print_msg("initial symmetry - axial rise             : %5.4f\n"%(dp))
		print_msg("Maximum iteration                         : %i\n"%(max_iter))
		print_msg("Data with CTF                             : %s\n"%(CTF))
		print_msg("Signal-to-Noise Ratio                     : %5.4f\n"%(snr))
		print_msg("symmetry output doc file                  : %s\n"%(datasym))
		print_msg("number of times to impose initial symmetry: %i\n"%(nise))
		print_msg("npad                                      : %i\n"%(npad))
		print_msg("User function                             : %s\n"%(user_func_name))

	if maskfile:
		if type(maskfile) is types.StringType: mask3D = get_image(maskfile)
		else:                                  mask3D = maskfile
	else: mask3D = None
	#else: mask3D = model_circle(last_ring, nx, nx, nx)

	numr	= Numrinit(first_ring, last_ring, rstep, "F")

	if CTF:
		pass#IMPORTIMPORTIMPORT from reconstruction import recons3d_4nn_ctf_MPI
		pass#IMPORTIMPORTIMPORT from filter         import filt_ctf
	pass#IMPORTIMPORTIMPORT else:	 from reconstruction import recons3d_4nn_MPI

	if myid == main_node:
       		if(file_type(stack) == "bdb"):
			pass#IMPORTIMPORTIMPORT from EMAN2db import db_open_dict
			dummy = db_open_dict(stack, True)
		active = EMUtil.get_all_attributes(stack, 'active')
		list_of_particles = []
		for im in xrange(len(active)):
			if active[im]:  list_of_particles.append(im)
		del active
		nima = len(list_of_particles)
	else:
		nima = 0
	total_nima = bcast_number_to_all(nima, source_node = main_node)

	if myid != main_node:
		list_of_particles = [-1]*total_nima
	list_of_particles = bcast_list_to_all(list_of_particles, myid,  source_node = main_node)

	image_start, image_end = MPI_start_end(total_nima, number_of_proc, myid)
	# create a list of images for each node
	list_of_particles = list_of_particles[image_start: image_end]
	nima = len(list_of_particles)
	if debug:
		finfo.write("image_start, image_end: %d %d\n" %(image_start, image_end))
		finfo.flush()
	
	mask2D = pad( model_blank( int(nmax-20),nmax,1,bckg=1.0), nmax, nmax, 1,0.0)

	data = EMData.read_images(stack, list_of_particles)
	#if fourvar:  original_data = []
	for im in xrange(nima):
		data[im].set_attr('ID', list_of_particles[im])
		sttt = Util.infomask(data[im], mask2D, False)
		data[im] = data[im] - sttt[0]
		#if fourvar: original_data.append(data[im].copy())
		if CTF:
			st = data[im].get_attr_default("ctf_applied", 0)
			if(st == 0):
				ctf_params = data[im].get_attr("ctf")
				data[im] = filt_ctf(data[im], ctf_params)
				data[im].set_attr('ctf_applied', 1)
	del mask2D

	if debug:
		finfo.write( '%d loaded  \n' % nima )
		finfo.flush()
	
	for i in xrange(len(xrng)): yrng[i]=dp/(2*pixel_size)
	pass#IMPORTIMPORTIMPORT from math import sin, pi
	if ( ou > ( nmax/2.0)*sin( initial_theta*pi/180) - dp/2.0/pixel_size -1.0 ):
		ERROR('ou should be less than or equal to ----( nmax/2.0)*sin( initial_theta*pi/180) - dp/2.0/pixel_size -1.0 ', "ihrsr_MPI", 1,myid)

	if myid == main_node:
		print_msg("Pixel size in Angstroms                   : %5.4f\n\n"%(pixel_size))
		print_msg("Y search range (pix) initialized as       : %s\n\n"%(yrng))

	pass#IMPORTIMPORTIMPORT from time import time

	#  this is needed for gathering of pixel errors
	disps = []
	recvcount = []
	for im in xrange(number_of_proc):
		if( im == main_node ):  disps.append(0)
		else:                   disps.append(disps[im-1] + recvcount[im-1])
		ib, ie = MPI_start_end(total_nima, number_of_proc, im)
		recvcount.append( ie - ib )

	total_iter = 0
	# do the projection matching
	for N_step in xrange(lstp):
		terminate = 0
		Iter = 0
 		while(Iter < max_iter and terminate == 0):
			yrng[N_step]=float(dp)/(2*pixel_size) #will change it later according to dp
			if(ynumber[N_step]==0): stepy = 0.0
			else:                   stepy = (2*yrng[N_step]/ynumber[N_step])

			pixer  = [0.0]*nima
			modphi = [0.0]*nima
			Iter += 1
			total_iter += 1
			if myid == main_node:
				start_time = time()
				print_msg("\nITERATION #%3d,  inner iteration #%3d\nDelta = %4.1f, an = %5.4f, xrange (Pixels) = %5.4f,stepx (Pixels) = %5.4f, yrng (Pixels) = %5.4f,  stepy (Pixels) = %5.4f, y_restrict (Pixels)=%5.4f, ynumber = %3d\n"%(total_iter, Iter, delta[N_step], an[N_step], xrng[N_step],stepx[N_step],yrng[N_step],stepy,y_restrict[N_step], ynumber[N_step]))
			if( xysize == -1 and zsize==-1 ):
				volft,kb = prep_vol( vol )
				refrings = prepare_refrings( volft, kb, nmax, delta[N_step], ref_a, symref, numr, MPI = True, phiEqpsi = "Zero", initial_theta =initial_theta, delta_theta = delta_theta)
				del volft,kb
			else:
				volft, kbx, kby, kbz = prep_vol( vol )
				refrings = prepare_refrings( volft, kbz, nmax, delta[N_step], ref_a, symref, numr, MPI = True, phiEqpsi = "Zero", kbx = kbx, kby = kby, initial_theta =initial_theta, delta_theta = delta_theta)
				del volft, kbx, kby, kbz

			if myid== main_node:
				print_msg( "Time to prepare rings: %d\n" % (time()-start_time) )
				start_time = time()
			#split refrings to two list: refrings1 (even point-group symmetry AND theta = 90. ), 
			#   or refrings2 ( odd point-group symmetry AND any theta (including theta=90), OR even point-group symmetry AND theta <> 90.
			refrings1= []
			refrings2= []
			sn = int(symmetry_string[1:])
			for i in xrange( len(refrings) ):
				if( sn%2 ==0 and abs( refrings[i].get_attr('n3') ) <1.0e-6 ):
					#  even point-group symmetry AND theta = 90. 
					refrings1.append( refrings[i] )
				else:
					# odd point-group symmetry AND any theta (including theta=90), OR even point-group symmetry AND theta <> 90.
					refrings2.append( refrings[i] )
					"""
					if myid == main_node:
						print_msg("\nphi = %5.2f, theta = %5.2f, psi=%5.2f\n"%( refrings[i].get_attr('phi'), refrings[i].get_attr('theta'), refrings[i].get_attr('psi') ) )
					if myid == main_node:
						print_msg("\nlen(ref1) = %4d, len(ref2) = %4d\n"%(len(refrings1), len(refrings2)) )
					"""		
			del refrings
			pass#IMPORTIMPORTIMPORT from numpy import float32
			dpp = float32(float(dp)/pixel_size)
			dpp = float( dpp )
			dpp_half = dpp/2.0

			for im in xrange( nima ):
				"""
					Logic of searches:
						refrings1 (even point-group symmetry AND theta = 90.), and refrings2 ( odd point-group symmetry AND any theta (including theta=90), OR even point-group symmetry AND theta <> 90.)
						an == -1 exhaustive search
						psi_max - how far rotation in plane can can deviate from 90 or 270 degrees
						psi_max -s used in both exhaustive and local searches
						:::
						>even point-group symmetry AND theta = 90. 
							exhaustive   proj_ali_helical_90   psi_max
											Util.multiref_polar_ali_helical_90  psi_max
											Crosrng_sm_psi  (in util_sparx.cpp)  psi_max - WILL SEARCH for BOTH PSI=0 AND 180 NO MATTER WHAT					
																					flag - no mirror or mirror, DOES NOT CHECK MIRRORED
											
							local       reference projections phi= [0,180,delta], theta=90 
										proj_ali_helical_90_local   an[N_step], psi_max,  (alignment.py)
											Util.multiref_polar_ali_helical_90_local   psi_max
											Crosrng_sm_psi		(in util_sparx.cpp)   psi_max - WILL SEARCH AROUND BOTH PSI=0 AND 180 NO MATTER WHAT					

						>odd point-group symmetry AND any theta (including theta=90), OR even point-group symmetry AND theta <> 90.
							exhaustive   proj_ali_helical           psi_max
											Util.multiref_polar_ali_helical  psi_max
												CALLS Crosrng_pi twice, for 0 and for 180, thus duplicates the work!!
											Crosrng_psi  checks MIRROR
											
							local        proj_ali_helical_local   psi_max
											Util.multiref_polar_ali_helical_local  psi_max
												Uses the following construct
												if (mirror_only == true) {
													if ((psi-90) < 90) retvals = Crosrng_sm_psi(crefim[iref], cimage, numr,   0, 1, psi_max);
													else               retvals = Crosrng_sm_psi(crefim[iref], cimage, numr, 180, 1, psi_max); 
												} else {
													if ((psi-90) < 90) retvals = Crosrng_sm_psi(crefim[iref], cimage, numr,   0, 0, psi_max);
													else               retvals = Crosrng_sm_psi(crefim[iref], cimage, numr, 180, 0, psi_max);
												}
											where Crosrng_sm_psi will do search for psi around one angle and do mirror or not.

				"""
				peak1 = None
				peak2 = None
				#print im, get_params_proj(data[im])
				if( len(refrings1) > 0):
					if  an[N_step] == -1:
						peak1, phihi1, theta1, psi1, sxi1, syi1, t11 = \
						proj_ali_helical_90(data[im], refrings1, numr, xrng[N_step], yrng[N_step], stepx[N_step], ynumber[N_step], psi_max, finfo)
					else:
						peak1, phihi1, theta1, psi1, sxi1, syi1, t11 = \
						proj_ali_helical_90_local(data[im], refrings1, numr, xrng[N_step], yrng[N_step], stepx[N_step], ynumber[N_step], an[N_step], psi_max, finfo, yrnglocal=y_restrict[N_step])
					#print "  1  ",im, peak1, phihi1, theta1, psi1, sxi1, syi1
				if( len(refrings2) > 0):
					if  an[N_step] == -1:
						peak2, phihi2, theta2, psi2, sxi2, syi2, t12 = \
						proj_ali_helical(data[im], refrings2, numr, xrng[N_step], yrng[N_step], stepx[N_step], ynumber[N_step], psi_max, finfo)
					else:
						peak2, phihi2, theta2, psi2, sxi2, syi2, t12 = \
						proj_ali_helical_local(data[im], refrings2, numr, xrng[N_step], yrng[N_step], stepx[N_step], ynumber[N_step], an[N_step], psi_max, finfo, yrnglocal=y_restrict[N_step])
					#print "  2  ",im, peak2, phihi2, theta2, psi2, sxi2, syi2
				if peak1 is None: 
					peak = peak2
					phihi = phihi2
					theta = theta2
					psi = psi2
					sxi = sxi2
					syi = syi2
					t1 = t12
				elif peak2 is None:
					peak = peak1
					phihi = phihi1
					theta = theta1
					psi = psi1
					sxi = sxi1
					syi = syi1
					t1 = t11
				else:
					if(peak1 >= peak2):
						peak = peak1
						phihi = phihi1
						theta = theta1
						psi = psi1
						sxi = sxi1
						syi = syi1
						t1 = t11
					else:
						peak = peak2
						phihi = phihi2
						theta = theta2
						psi = psi2
						sxi = sxi2
						syi = syi2
						t1 = t12
				#print "  3  ",im, peak, phihi, theta, psi, sxi, syi
				if(peak > -1.0e22):
					if WRAP == 1:
						#Guozhi Tao: wrap y-shifts back into box within rise of one helical unit by changing phi
						tp = Transform({"type":"spider","phi":phihi,"theta":theta,"psi":psi})
						tp.set_trans( Vec2f( -sxi, -syi ) )
						dtp = tp.get_params("spider")
						dtp_ty = float( dtp["ty"] )
						del dtp
						if( abs(dtp_ty) >dpp_half):
							dtp_ty_temp = dtp_ty
							if( abs(psi-90) < 90  ):
								sign_psi = 1
							else:
								sign_psi = -1
							if( dtp_ty > 0):
								period_step = -1*sign_psi
							else:
								period_step = 1*sign_psi
							nperiod = 0
							while( abs( dtp_ty_temp ) > dpp_half ):
								nperiod += period_step
								th = Transform({"type":"spider","phi": -nperiod*dphi, "tz":nperiod*dpp})
								tfinal = tp*th
								df = tfinal.get_params("spider")
								dtp_ty_temp = float( df["ty"] )

							phihi = float(df["phi"])
							sxi   = float(-df["tx"])
							syi   = float(-df["ty"])
					#print "  4  ",im, peak, phihi, theta, psi, sxi, syi

					# unique ranges of azimuthal angle for ortho-axial and non-ortho-axial projection directions are identified by [k0,k1) and [k2,k3), where k0, k1, k2, k3 are floats denoting azimuthal angles.
					# Eulerian angles whose azimuthal angles are mapped into [k2, k3) are related to Eulerian angles whose azimuthal angles are mapped into [k0, k1) by an in-plane mirror operaton along the x-axis.

					tp = Transform({"type":"spider","phi":phihi,"theta":theta,"psi":psi})
					tp.set_trans( Vec2f( -sxi, -syi ) )

					k0 =   0.0
					k2 = 180.0
					if( abs( tp.at(2,2) )<1.0e-6 ):
						if (symmetry_string[0] =="c"):
							if sn%2 == 0:  k1=360.0/sn
							else:          k1=360.0/2/sn
						elif (symmetry_string[0] =="d"):
							if sn%2 == 0:  k1=360.0/2/sn
							else:          k1=360.0/4/sn
					else:
						if (symmetry_string[0] =="c"):  k1=360.0/sn
						if (symmetry_string[0] =="d"):  k1=360.0/2/sn
					k3 = k1 +180.0

					pass#IMPORTIMPORTIMPORT from utilities import get_sym
					T = get_sym(symmetry_string[0:])

					d1tp = tp.get_params('spider')
					sxnew    = -d1tp["tx"]
					synew    = -d1tp["ty"]
					phinew   =  d1tp['phi']
					thetanew =  d1tp["theta"]
					psinew   =  d1tp["psi"]
					del d1tp

					#print "  5  ",im, phinew, thetanew, psinew, sxnew, synew
					#print k0,k1,k2,k3

					for i in xrange( len(T) ):
						ttt = tp*Transform({"type":"spider","phi":T[i][0],"theta":T[i][1],"psi":T[i][2]})
						d1  = ttt.get_params("spider")

						if ( abs( tp.at(2,2) )<1.0e-6 ):
							if( sn%2==1 ): # theta=90 and n odd, only one of the two region match

								if( ( d1['phi'] >= k0 and d1['phi'] < k1 ) or ( d1['phi'] >= k2 and d1['phi'] < k3 )):

									sxnew    = -d1["tx"]
									synew    = -d1["ty"]
									phinew   =  d1['phi']
									thetanew =  d1["theta"]
									psinew   =  d1["psi"]

									# For boundary cases where phihi is exactly on the boundary of the unique range, there may be two symmetry related Eulerian angles which are both in the unique 
									# range but whose psi differ by 180. 
									# For example, (180,90,270) has two symmetry related angles in unique range: (180,90,270) and (180, 90, 90)
									# In local search, psi should stay within neighborhood of original value, so take the symmetry related
									# Eulerian angles in unique range which does not change psi by 180.
									if an[N_step] != -1:
										if abs(psinew - psi) < 90:
											break
							else: #for theta=90 and n even, there is no mirror version during aligment, so only consider region [k0,k1]

								if( d1['phi'] >= float(k0) and d1['phi'] < float(k1)  ) :

									sxnew    = - d1["tx"]
									synew    = - d1["ty"]
									phinew   = d1['phi']
									thetanew = d1["theta"]
									psinew   = d1["psi"]

						else: #theta !=90, # if theta >90, put the projection into [k2,k3]. Otherwise put it into the region [k0,k1]

							if( sn==1):
								sxnew    = sxi
								synew    = syi
								phinew   = phihi
								thetanew = theta
								psinew   = psi
							else:

								if (tp.at(2,2) >0.0): #theta <90
									
									if(  d1['phi'] >= float(k0) and d1['phi'] < float(k1)):
										if( cos( pi*float( d1['theta'] )/180.0 )>0.0 ):
			
											sxnew    = - d1["tx"]
											synew    = - d1["ty"]
											phinew   = d1['phi']
											thetanew = d1["theta"]
											psinew   = d1["psi"]
		
								else:
									if(  d1['phi'] >= float(k2) and d1['phi'] < float(k3)):
										if( cos( pi*float( d1['theta'] )/180.0 )<0.0 ):

											sxnew    = - d1["tx"]
											synew    = - d1["ty"]
											phinew   = d1['phi']
											thetanew = d1["theta"]
											psinew   = d1["psi"]

						del ttt,d1

					t2 = Transform({"type":"spider","phi":phinew,"theta":thetanew,"psi":psinew})
					t2.set_trans(Vec2f(-sxnew, -synew))
					data[im].set_attr("xform.projection", t2)
					pixer[im]  = max_3D_pixel_error(t1, t2, numr[-3])
					modphi[im] = phinew

				else:
					# peak not found, parameters not modified
					pixer[im]  = 0.0
					phihi, theta, psi, sxi, syi = get_params_proj(data[im])
					modphi[im] = phihi
				#print "  6  ",im, phinew, thetanew, psinew, sxnew, synew

				#if(im==2):
				#	from sys import exit
				#	exit()

			del refrings1, refrings2
			if myid == main_node:
				print_msg("Time of alignment = %d\n"%(time()-start_time))
				start_time = time()

			#output pixel errors
			pass#IMPORTIMPORTIMPORT from mpi import mpi_gatherv
			recvbuf = mpi_gatherv(pixer, nima, MPI_FLOAT, recvcount, disps, MPI_FLOAT, main_node, MPI_COMM_WORLD)
			del pixer		
			mpi_barrier(MPI_COMM_WORLD)
			terminate = 0
			if(myid == main_node):
				recvbuf = map(float, recvbuf)
				pass#IMPORTIMPORTIMPORT from utilities import write_text_file
				write_text_file([range(len(recvbuf)), recvbuf], os.path.join(outdir, "pixer_%04d_%04d.txt"%(N_step+1,Iter)) )
				pass#IMPORTIMPORTIMPORT from statistics import hist_list
				lhist = 20
				region, histo = hist_list(recvbuf, lhist)
				if(region[0] < 0.0):  region[0] = 0.0
				msg = "      Histogram of pixel errors\n      ERROR       number of particles\n"
				print_msg(msg)
				for lhx in xrange(lhist):
					msg = " %10.2f     %7d\n"%(region[lhx], histo[lhx])
					print_msg(msg)
				# Terminate if 95% within 1 pixel error
				im = 0
				for lhx in xrange(lhist):
					if(region[lhx] > 1.0): break
					im += histo[lhx]
				#if(im/float(total_nima) > 0.95):  terminate = 1
				del region, histo
			#output distribution of phi
			#jeanmod
			recvbuf = mpi_gatherv(modphi, nima, MPI_FLOAT, recvcount, disps, MPI_FLOAT, main_node, MPI_COMM_WORLD)
			#end jeanmod
			mpi_barrier(MPI_COMM_WORLD)
			del modphi
			if(myid == main_node):
				recvbuf = map(float, recvbuf)
				phi_value_0 = []
				phi_value_180 = []
				for i in xrange ( len ( recvbuf ) ):
					if ( recvbuf[i] < 180.0):
						phi_value_0.append( recvbuf[i] )
					else:
						phi_value_180.append( recvbuf[i] ) 
 				lhist = int( round(max(phi_value_0)/delta[N_step]) )
								# if delta is big, number of bins (lhist) will be small, leave it as it is
				# if delta is small, number of bins (lhist) will be big, adjust lhist = lhist/n such as the total 
				# number of bins close to 30, thus most likely we can see each bin contains particles.
				pass#IMPORTIMPORTIMPORT from math import ceil
				if ( len( phi_value_180) > 0):
					if lhist > 15:
						lhist = int(   lhist/ceil((lhist/15.0))  ) 
				else:
					if lhist > 30:
						lhist = int(   lhist/ceil((lhist/30.0))  )  
				region, histo = hist_list(phi_value_0, lhist)
				msg = "\n      Distribution of phi\n      phi         number of particles\n"
				print_msg(msg)
				for lhx in xrange(lhist):
					msg = " %10.2f     %7d\n"%(region[lhx], histo[lhx])
					print_msg(msg)
				del region, histo, phi_value_0
				if ( len( phi_value_180) > 0):
					region, histo = hist_list(phi_value_180, lhist)
					for lhx in xrange(lhist):
						msg = " %10.2f     %7d\n"%(region[lhx], histo[lhx])
						print_msg(msg)
					del region, histo, phi_value_180			
			del recvbuf
			terminate = mpi_bcast(terminate, 1, MPI_INT, 0, MPI_COMM_WORLD)
			terminate = int(terminate[0])
			if myid == main_node:
				print_msg("Time to compute pixer = %d\n"%(time()-start_time))
				start_time = time()
			# write out headers, under MPI writing has to be done sequentially
			mpi_barrier(MPI_COMM_WORLD)
			m = 5
			pass#IMPORTIMPORTIMPORT from mpi import mpi_recv, mpi_send, MPI_COMM_WORLD, MPI_FLOAT
			if myid == main_node:
				
				fexp = open(os.path.join(outdir, "parameters_%04d_%04d.txt"%(N_step+1,Iter)),"w")
				for n in xrange(number_of_proc):
					if n!=main_node:
						t = mpi_recv(recvcount[n]*m,MPI_FLOAT, n, SPARX_MPI_TAG_UNIVERSAL, MPI_COMM_WORLD)
						for i in xrange(recvcount[n]):
							for j in xrange(m):
								fexp.write(" %15.5f  "%t[j+i*m])
							fexp.write("\n")
					else:
						t = [0.0]*m
						for i in xrange(recvcount[myid]):
							t = get_params_proj(data[i])
							for j in xrange(m):
								fexp.write(" %15.5f  "%t[j])
							fexp.write("\n")
				fexp.close()
				del t
	        	else:
				nvalue = [0.0]*m*recvcount[myid]
				t = [0.0]*m
				for i in xrange(recvcount[myid]):
					t = get_params_proj(data[i])
					for j in xrange(m):
						nvalue[j + i*m] = t[j]
				mpi_send(nvalue, recvcount[myid]*m, MPI_FLOAT, main_node, SPARX_MPI_TAG_UNIVERSAL, MPI_COMM_WORLD)
				del nvalue
			if myid == main_node:
				print_msg("Time to write parameters = %d\n"%(time()-start_time))
				start_time = time()

			if CTF: vol = recons3d_4nn_ctf_MPI(myid, data, symmetry=sym, snr = snr, npad = npad, xysize = xysize, zsize = zsize)
			else:    vol = recons3d_4nn_MPI(myid, data, symmetry=sym, snr = snr, npad = npad, xysize = xysize, zsize = zsize)

			if myid == main_node:
				print_msg("\n3D reconstruction time = %d\n"%(time()-start_time))
				start_time = time()

			"""
			if fourvar:
			#  Compute Fourier variance
				for im in xrange(nima):
					original_data[im].set_attr( 'xform.projection', data[im].get_attr('xform.projection') )
				varf = varf3d_MPI(original_data, ssnr_text_file = os.path.join(outdir, "ssnr%04d"%(total_iter)), mask2D = None, reference_structure = vol, ou = last_ring, rw = 1.0, npad = 1, CTF = CTF, sign = 1, sym =sym, myid = myid)
				if myid == main_node:
					print_msg("Time to calculate 3D Fourier variance= %d\n"%(time()-start_time))
					start_time = time()
					varf = 1.0/varf
					drop_image(varf, os.path.join(outdir, "varf%04d.hdf"%(total_iter)))
			else:  varf = None
			"""
			#search for helical symmetry
			if myid == main_node:
				drop_image(vol, os.path.join(outdir, "vol%04d.hdf"%(total_iter)))

			if(total_iter > nise):
				bcast_EMData_to_all(vol, myid, main_node)
				#from filter import filt_gaussl
				#vol = filt_gaussl(vol, 0.25)

				if myid == main_node:
					lprms = []
					for i in xrange(-ndp,ndp+1,1):
						for j in xrange(-ndphi,ndphi+1,1):
							lprms.append( dp   + i*dp_step)
							lprms.append( dphi + j*dphi_step)
					#print "lprms===",lprms
					recvpara = []
					for im in xrange(number_of_proc):
						helic_ib, helic_ie = MPI_start_end(nlprms, number_of_proc, im)
						recvpara.append(helic_ib )
						recvpara.append(helic_ie )

				para_start, para_end = MPI_start_end(nlprms, number_of_proc, myid)

				list_dps     = [0.0]*((para_end-para_start)*2)
				list_fvalues = [-1.0]*((para_end-para_start)*1)

				if myid == main_node:
					for n in xrange(number_of_proc):
						if n!=main_node: mpi_send(lprms[2*recvpara[2*n]:2*recvpara[2*n+1]], 2*(recvpara[2*n+1]-recvpara[2*n]), MPI_FLOAT, n, SPARX_MPI_TAG_UNIVERSAL, MPI_COMM_WORLD)
						else:    list_dps = lprms[2*recvpara[2*0]:2*recvpara[2*0+1]]
				else:
					list_dps = mpi_recv((para_end-para_start)*2, MPI_FLOAT, main_node, SPARX_MPI_TAG_UNIVERSAL, MPI_COMM_WORLD)

				list_dps = map(float, list_dps)

				local_pos = [0.0, 0.0, -1.0e20]
				for i in xrange(para_end-para_start):
					fvalue = helios7(vol, pixel_size, list_dps[i*2], list_dps[i*2+1], fract, rmax, rmin)
					if(fvalue >= local_pos[2]):
						local_pos = [list_dps[i*2], list_dps[i*2+1], fvalue ]
				if myid == main_node:
					list_return = [0.0]*(3*number_of_proc)
					for n in xrange(number_of_proc):
						if n != main_node: list_return[3*n:3*n+3]                 = mpi_recv(3,MPI_FLOAT, n, SPARX_MPI_TAG_UNIVERSAL, MPI_COMM_WORLD)
 						else:              list_return[3*main_node:3*main_node+3]  = local_pos[:]
				else:
					mpi_send(local_pos, 3, MPI_FLOAT, main_node, SPARX_MPI_TAG_UNIVERSAL, MPI_COMM_WORLD)

				if myid == main_node:	
					maxvalue = list_return[2]
					for i in xrange(number_of_proc):
						if( list_return[i*3+2] >= maxvalue ):
							maxvalue = list_return[i*3+2]
							dp       = list_return[i*3+0]
							dphi     = list_return[i*3+1]
					dp   = float(dp)
					dphi = float(dphi)
					#print  "  GOT dp dphi",dp,dphi

					vol  = vol.helicise(pixel_size, dp, dphi, fract, rmax, rmin)
					print_msg("New delta z and delta phi      : %s,    %s\n\n"%(dp,dphi))		
			
			else:
				if myid==main_node:
					#  in the first nise steps the symmetry is imposed
					vol = vol.helicise(pixel_size, dp, dphi, fract, rmax, rmin)
					print_msg("Imposed delta z and delta phi      : %s,    %s\n\n"%(dp,dphi))
			if(myid==main_node):
				fofo = open(os.path.join(outdir,datasym),'a')
				fofo.write('  %12.4f   %12.4f\n'%(dp,dphi))
				fofo.close()
				ref_data = [vol, mask3D]
				#if  fourvar:  ref_data.append(varf)
				vol = user_func(ref_data)
				vol = vol.helicise(pixel_size, dp, dphi, fract, rmax, rmin)

				drop_image(vol, os.path.join(outdir, "volf%04d.hdf"%(total_iter)))
				print_msg("\nSymmetry search and user function time = %d\n"%(time()-start_time))
				start_time = time()

			bcast_EMData_to_all(vol, myid, main_node)
			dp   = bcast_number_to_all(dp,   source_node = main_node)
			dphi = bcast_number_to_all(dphi, source_node = main_node)
			# del varf
	par_str = ["xform.projection"]
	if myid == main_node:
	   	if(file_type(stack) == "bdb"):
	        	pass#IMPORTIMPORTIMPORT from utilities import recv_attr_dict_bdb
	        	recv_attr_dict_bdb(main_node, stack, data, par_str, image_start, image_end, number_of_proc)
	        else:
	        	pass#IMPORTIMPORTIMPORT from utilities import recv_attr_dict
	        	recv_attr_dict(main_node, stack, data, par_str, image_start, image_end, number_of_proc)
		print_msg("Time to write header information= %d\n"%(time()-start_time))
		start_time = time()
	else:	       send_attr_dict(main_node, data, par_str, image_start, image_end)
	if myid == main_node: print_end_msg("ihrsr_MPI")
'''
"""34

	if(myid == 0):
		pid_list = read_text_file("main000/chunk0.txt")
		nima = len(pid_list)
	else: nima = 0
	nima = bcast_number_to_all(nima, source_node = 0)
	if(myid != 0):
		pid = [-1]*nima
	pid_list = mpi_bcast(pid_list, nima, MPI_INT, 0, MPI_COMM_WORLD)
	pid_list = map(int, pid_list)
	
	image_start, image_end = MPI_start_end(nima, nproc, myid)
	prjlist = [EMData.read_images(prj_stack, pid_list[image_start:image_end])]
	"""
"""35
	if(myid == 0):
		pid_list = read_text_file("main000/chunk1.txt")
		nima = len(pid_list)
	else: nima = 0
	nima = bcast_number_to_all(nima, source_node = 0)
	if(myid != 0):
		pid = [-1]*nima
	"""
"""36
	pass#IMPORTIMPORTIMPORT from fundamentals import fdecimate
	pass#IMPORTIMPORTIMPORT from utilities import get_params_proj,set_params_proj
	scale = 384./54.
	for i in xrange(len(prjlist)):
		prjlist[k][i] = fdecimate(prjlist[i],54,54)
		ctf_params = prjlist[i].get_attr("ctf")
		ctf_params.apix *= scale
		prjlist[i].set_attr('ctf', ctf_params)
		phi,theta,psi,sx,sy = get_params_proj(prjlist[i])
		set_params_proj(prjlist[i],[phi,theta,psi,sx/scale,sy/scale])
	"""
"""37


	bckgnoise = get_im("bckgnoise.hdf")
	nlx = bckgnoise.get_xsize()
	datastamp = read_text_file("defgroup_stamp.txt")
	#nnnx = 200#prjlist[0].get_ysize()
	pass#IMPORTIMPORTIMPORT from utilities import get_params_proj,set_params_proj
	for i in xrange(len(prjlist)):
		#phi,theta,psi,sxs,sys = get_params_proj(prjlist[i])
		try:
			stmp = prjlist[i].get_attr("ptcl_source_image")
		except:
			try:
				stmp = prjlist[i].get_attr("ctf")
				stmp = round(stmp.defocus,4)
			except:
				ERROR("Either ptcl_source_image or ctf has to be present in the header.","get_shrink_data",1, myid)
		try:
			indx = datastamp.index(stmp)
		except:
			ERROR("Problem with indexing ptcl_source_image.","get_shrink_data",1, myid)


		prjlist[i] = fft(prjlist[i])
		#prjlist[i] = fshift(prjlist[i],sxs,sys)
		prjlist[i].set_attr("padffted",1)
		prjlist[i].set_attr("npad",1)
		#set_params_proj(prjlist[i] ,[phi,theta,psi,0.0,0.0])
		prjlist[i].set_attr("bckgnoise", [bckgnoise[k,indx] for k in xrange(nlx)]) #  This is index on bckgnoise table



	"""
"""38
	if(myid == 0):
		if(listfile):
			pass#IMPORTIMPORTIMPORT from utilities import read_text_file
			pid_list = read_text_file(listfile, 0)
			pid_list = map(int, pid_list)
		elif(group > -1):
			tmp_list = EMUtil.get_all_attributes(prj_stack, 'group')
			pid_list = []
			for i in xrange(len(tmp_list)):
				if(tmp_list[i] == group):  pid_list.append(i)
			del tmp_list
		nima = len(pid_list)
	else:
		nima = 0
	nima = bcast_number_to_all(nima, source_node = 0)

	if(listfile or group > -1):
		if myid != 0:
			pid_list = [-1]*nima
		pid_list = mpi_bcast(pid_list, nima, MPI_INT, 0, MPI_COMM_WORLD)
		pid_list = map(int, pid_list)
	else:
		if(not pid_list):  pid_list = range(nima)
	"""
'''39
	qt = 0.0
	for i in xrange(len(ssnr1)):
		tqt = ssnr1[i][1] - ssnr2[i][1]
		if( tqt<qt ): qt = tqt
	for i in xrange(len(ssnr1)): ssnr1[i][1] -= (ssnr2[i][1] + qt)
	pass#IMPORTIMPORTIMPORT from utilities import dropSpiderDoc19289
	dropSpiderDoc(ssnr_text_file+".doc", ssnr1)
	dropImage(vol_ssnr2, output_volume+"2.spi", "s")
	'''
"""40
		qt = 0.0
		for i in xrange(len(ssnr2)):
			tqt = ssnr1[i][1] - ssnr2[i][1]
			if( tqt<qt ): qt = tqt
		for i in xrange(len(ssnr1)): ssnr1[i][1] -= (ssnr2[i][1] + qt)
		
		dropSpiderDoc(ssnr_text_file+".doc", ssnr1)
		vol_ssnr2, output_volume+"2.spi", "s")
		"""
"""41
def incvar(prefix, nfile, nprj, output, fl, fh, radccc, writelp, writestack):
	pass#IMPORTIMPORTIMPORT from statistics import variancer, ccc
	pass#IMPORTIMPORTIMPORT from string     import atoi, replace, split, atof
	pass#IMPORTIMPORTIMPORT from utilities  import get_im, circumference, model_circle, drop_image
	pass#IMPORTIMPORTIMPORT from filter     import filt_btwl
	pass#IMPORTIMPORTIMPORT from math       import sqrt
	pass#IMPORTIMPORTIMPORT import os

	all_varer = variancer()
	odd_varer = variancer()
	eve_varer = variancer()

	filname = prefix + "0000.hdf"
	img = get_im(filname, 0)
	n = img.get_xsize()
	
	if os.path.exists(output):		os.system("rm -f "+output)
	if os.path.exists('stack_'+output):	os.system("rm -f stack_"+output)
	if os.path.exists('odd_stack_'+output):	os.system("rm -f odd_stack_"+output)
	if os.path.exists('eve_stack_'+output):	os.system("rm -f eve_stack_"+output)
	if os.path.exists('avg_'+output):	os.system("rm -f avg_"+output)

	cccmask = model_circle(radccc, n, n, n)
	scale = sqrt( nprj )
	totnimg = 0
	iwrite = 0
	radcir = n/2-3
	for i in xrange(nfile):
		filename = prefix + '%04d.hdf'%i 

		print 'loading file ', filename
		nimg = EMUtil.get_image_count( filename )
		for j in xrange(nimg):
			
			img = get_im( filename, j )
			img *= scale
			img = circumference( img, radcir, radcir+2 )
			img = filt_btwl(img, fl, fh)
						
			if writelp:
				img.write_image("btwl_cir_"+filename, j)

			if totnimg%2==0: odd_varer.insert(img)
			else: eve_varer.insert(img)

			all_varer.insert(img)

			totnimg += 1

			if totnimg%100==0:
				odd_var = odd_varer.getvar()
				eve_var = eve_varer.getvar()
				all_var = all_varer.getvar()
			
				if writestack:
					odd_var.write_image( 'odd_stack_' + output, iwrite )
					eve_var.write_image( 'eve_stack_' + output, iwrite )
					all_var.write_image( 'stack_' + output, iwrite )
					iwrite += 1  
				print 'ntot, ccc: %6d %10.3f' % (totnimg, ccc(odd_var, eve_var, cccmask))  

	all_var = all_varer.getvar()
	odd_var = odd_varer.getvar()
	eve_var = eve_varer.getvar()
	print 'ntot, ccc: %6d %10.3f' % (totnimg, ccc(odd_var, eve_var, cccmask))  

	avg = all_varer.getavg()
	avg.write_image( 'avg_' + output, 0 )

	if writestack:
		all_var.write_image( 'stack_' + output, iwrite )
		odd_var.write_image( 'odd_stack_' + output, iwrite )
		eve_var.write_image( 'eve_stack_' + output, iwrite )

	all_var = circumference( all_var, radcir, radcir+2 )
	all_var.write_image( output, 0 )
"""
"""42
	Util.mul_scalar(avg1, 1.0/float(total_img//2+total_img%2 - 1 ))
	avg1.write_image(avgfileE)
	Util.mul_scalar(avg2, 1.0/float(total_img//2 - 1) )
	avg2.write_image(avgfileO)
	"""
"""43
	Util.mul_scalar(var1, 1.0/float(total_img//2+total_img%2 - 1 ))
	var1.write_image(varfileE)
	Util.mul_scalar(var2, 1.0/float(total_img//2 - 1) )
	var2.write_image(varfileO)
	"""
"""44
	  writelp means overwrite original stacks with repaired ones
	  writestack means write new stacks with low-pass filtered data
	"""
'''45
#  06-12-2014 code lifted
def within_group_refinement(data, maskfile, randomize, ir, ou, rs, xrng, yrng, step, dst, maxit, FH, FF):

	# Comment by Zhengfan Yang 03/11/11
	# This is a simple version of ali2d_data (down to the bone), no output dir, no logfile, no CTF, no MPI or CUDA, no Fourvar, no auto stop, no user function
	
	pass#IMPORTIMPORTIMPORT from alignment    import Numrinit, ringwe, ali2d_single_iter
	pass#IMPORTIMPORTIMPORT from filter	  import filt_tanl
	pass#IMPORTIMPORTIMPORT from fundamentals import fshift
	pass#IMPORTIMPORTIMPORT from random	  import randint, random
	pass#IMPORTIMPORTIMPORT from statistics   import ave_series
	pass#IMPORTIMPORTIMPORT from utilities    import get_input_from_string, model_circle, set_params2D, get_params2D, combine_params2, inverse_transform2

	first_ring=int(ir); last_ring=int(ou); rstep=int(rs); max_iter=int(maxit);
	nima = len(data)
	nx = data[0].get_xsize()
	if last_ring == -1:  last_ring = nx/2-2
	if maskfile: mask = maskfile
	else: mask = model_circle(last_ring, nx, nx)

	if randomize:
		for im in data:
			alpha, sx, sy, mirror, scale = get_params2D(im)
			alphai, sxi, syi, mirrori = inverse_transform2(alpha, sx, sy)
			alphan, sxn, syn, mirrorn = combine_params2(0.0, -sxi, -syi, 0, numpy.random.random()*360.0, 0.0, 0.0, randint(0, 1))
			set_params2D(im, [alphan, sxn, syn, mirrorn, 1.0])

	cnx = nx/2+1
	cny = cnx
	mode = "F"
	numr = Numrinit(first_ring, last_ring, rstep, mode)
	wr = ringwe(numr, mode)

	sx_sum = 0.0
	sy_sum = 0.0
	cs = [0.0]*2
	total_iter = 0
	for N_step in xrange(len(xrng)):
		for Iter in xrange(max_iter):
			total_iter += 1
			tavg = ave_series(data)
			if( FH > 0.0):
				fl = 0.1+(FH-0.1)*Iter/float(max_iter-1)
				tavg = filt_tanl(tavg, fl, FF)
			if total_iter == len(xrng)*max_iter:  return tavg
			#if( xrng[0] > 0.0 ): cs[0] = sx_sum/float(nima)
			#if( yrng[0] > 0.0 ): cs[1] = sy_sum/float(nima)
			#tavg = fshift(tavg, -cs[0], -cs[1])
			if Iter%4 != 0 or total_iter > max_iter*len(xrng)-10: delta = 0.0
			else: delta = dst
			sx_sum, sy_sum = ali2d_single_iter(data, numr, wr, cs, tavg, cnx, cny, xrng[N_step], yrng[N_step], step[N_step], mode=mode, CTF=False, delta=delta)

'''
"""46
		tavg.write_image('tata.hdf')
		for Iter in xrange(0):#max_iter):  # large number
			total_iter += 1
			tavg = ave_series(data)
			if( FH > 0.0):
				fl = FH
				tavg = filt_tanl(fft(tavg), fl, FF)
				if total_iter == len(xrng)*max_iter:  return fft(tavg)
				if( xrng[0] > 0.0 ): cs[0] = sx_sum/float(nima)
				if( yrng[0] > 0.0 ): cs[1] = sy_sum/float(nima)
				tavg = fft(fshift(tavg, -cs[0], -cs[1]))
			else:
				if total_iter == len(xrng)*max_iter:  return tavg
				if( xrng[0] > 0.0 ): cs[0] = sx_sum/float(nima)
				if( yrng[0] > 0.0 ): cs[1] = sy_sum/float(nima)
				tavg = fshift(tavg, -cs[0], -cs[1])
			
			print  "  iteration  ***   %03d   %7.2f    %7.2f  "%(total_iter,cs[0],cs[1])
			if Iter%4 != 0 or total_iter > max_iter*len(xrng)-10: delta = 0.0
			else:                                                 delta = dst
			sx_sum, sy_sum, nope = ali2d_single_iter(data, numr, wr, cs, tavg, cnx, cny, \
														xrng[N_step], yrng[N_step], step[N_step], \
														mode=mode, CTF=False, delta=delta)
			if( (abs(cs[0]) + abs(cs[1])) < 0.01 and Iter > 1):  break
		"""
"""47
					tavg = filt_tanl(fft(tavg), fl, FF)
					if total_iter == len(xrng)*max_iter:  return fft(tavg)
					if( xrng[0] > 0.0 ): cs[0] = sx_sum/float(nima)
					if( yrng[0] > 0.0 ): cs[1] = sy_sum/float(nima)
					tavg = fft(fshift(tavg, -cs[0], -cs[1]))
					"""
"""48
					if total_iter == len(xrng)*max_iter:  return tavg
					if( xrng[0] > 0.0 ): cs[0] = sx_sum/float(nima)
					if( yrng[0] > 0.0 ): cs[1] = sy_sum/float(nima)
					tavg = fshift(tavg, -cs[0], -cs[1])
					"""
"""49
					tavg = filt_tanl(fft(tavg), fl, FF)
					if total_iter == len(xrng)*max_iter:  return fft(tavg)
					if( xrng[0] > 0.0 ): cs[0] = sx_sum/float(nima)
					if( yrng[0] > 0.0 ): cs[1] = sy_sum/float(nima)
					tavg = fft(fshift(tavg, -cs[0], -cs[1]))
				else:
					if total_iter == len(xrng)*max_iter:  return tavg
					if( xrng[0] > 0.0 ): cs[0] = sx_sum/float(nima)
					if( yrng[0] > 0.0 ): cs[1] = sy_sum/float(nima)
					tavg = fshift(tavg, -cs[0], -cs[1])
				"""
"""50
		tavg.write_image('tata.hdf')
		for Iter in xrange(0):#max_iter):  # large number
			total_iter += 1
			tavg = ave_series(data)
			if( FH > 0.0):
				fl = FH
				tavg = filt_tanl(fft(tavg), fl, FF)
				if total_iter == len(xrng)*max_iter:  return fft(tavg)
				if( xrng[0] > 0.0 ): cs[0] = sx_sum/float(nima)
				if( yrng[0] > 0.0 ): cs[1] = sy_sum/float(nima)
				tavg = fft(fshift(tavg, -cs[0], -cs[1]))
			else:
				if total_iter == len(xrng)*max_iter:  return tavg
				if( xrng[0] > 0.0 ): cs[0] = sx_sum/float(nima)
				if( yrng[0] > 0.0 ): cs[1] = sy_sum/float(nima)
				tavg = fshift(tavg, -cs[0], -cs[1])
			
			print  "  iteration  ***   %03d   %7.2f    %7.2f  "%(total_iter,cs[0],cs[1])
			if Iter%4 != 0 or total_iter > max_iter*len(xrng)-10: delta = 0.0
			else:                                                 delta = dst
			sx_sum, sy_sum, nope = ali2d_single_iter(data, numr, wr, cs, tavg, cnx, cny, \
														xrng[N_step], yrng[N_step], step[N_step], \
														mode=mode, CTF=False, delta=delta)
			if( (abs(cs[0]) + abs(cs[1])) < 0.01 and Iter > 1):  break
		"""
"""51
					tavg = filt_tanl(fft(tavg), fl, FF)
					if total_iter == len(xrng)*max_iter:  return fft(tavg)
					if( xrng[0] > 0.0 ): cs[0] = sx_sum/float(nima)
					if( yrng[0] > 0.0 ): cs[1] = sy_sum/float(nima)
					tavg = fft(fshift(tavg, -cs[0], -cs[1]))
					"""
"""52
					if total_iter == len(xrng)*max_iter:  return tavg
					if( xrng[0] > 0.0 ): cs[0] = sx_sum/float(nima)
					if( yrng[0] > 0.0 ): cs[1] = sy_sum/float(nima)
					tavg = fshift(tavg, -cs[0], -cs[1])
					"""
"""53
					tavg = filt_tanl(fft(tavg), fl, FF)
					if total_iter == len(xrng)*max_iter:  return fft(tavg)
					if( xrng[0] > 0.0 ): cs[0] = sx_sum/float(nima)
					if( yrng[0] > 0.0 ): cs[1] = sy_sum/float(nima)
					tavg = fft(fshift(tavg, -cs[0], -cs[1]))
				else:
					if total_iter == len(xrng)*max_iter:  return tavg
					if( xrng[0] > 0.0 ): cs[0] = sx_sum/float(nima)
					if( yrng[0] > 0.0 ): cs[1] = sy_sum/float(nima)
					tavg = fshift(tavg, -cs[0], -cs[1])
				"""
'''54


#  commented out to prevent problems 03/02/2015
def within_group_refinement_fast(data, dimage, maskfile, randomize, ir, ou, rs, xrng, yrng, step, maxrange, dst, maxit, FH, FF):
	#  It is not used anywhere, however, the check of boundaries has to be added or the code removed
	# Comment by Zhengfan Yang 03/11/11
	# This is a simple version of ali2d_data (down to the bone), no output dir, no logfile, no CTF, no MPI or CUDA, no Fourvar, no auto stop, no user function
	
	pass#IMPORTIMPORTIMPORT from alignment    import Numrinit, ringwe, ali2d_single_iter_fast
	pass#IMPORTIMPORTIMPORT from filter	      import filt_tanl
	pass#IMPORTIMPORTIMPORT from fundamentals import fshift, rot_shift2D, cyclic_shift
	pass#IMPORTIMPORTIMPORT from random	      import randint, random
	pass#IMPORTIMPORTIMPORT from math         import cos, sin, radians
	pass#IMPORTIMPORTIMPORT from statistics   import ave_series
	pass#IMPORTIMPORTIMPORT from utilities    import get_input_from_string, model_circle, model_blank, set_params2D, get_params2D, combine_params2, inverse_transform2

	first_ring=int(ir); last_ring=int(ou); rstep=int(rs); max_iter=int(maxit);
	nima = len(data)
	nx = data[0].get_xsize()
	if last_ring == -1:  last_ring = nx/2-2
	if maskfile: mask = maskfile
	else: mask = model_circle(last_ring, nx, nx)

	params = [[0.,0.,0.,0] for im in xrange(nima) ]
	if randomize:
		for im in xrange(nima):
			#alpha, sx, sy, mirror, scale = get_params2D(data[im])
			#alphai, sxi, syi, mirrori = inverse_transform2(alpha, sx, sy)
			#alphan, sxn, syn, mirrorn = combine_params2(0.0, -sxi, -syi, 0, random()*360.0, 0,0, randint(0, 1))
			params[im] = [numpy.random.random()*360.0, 0, 0, randint(0, 1)]
	else:
		for im in xrange(nima):
			alpha, sx, sy, mirror, scale = get_params2D(data[im])
			params[im] = [alpha, 0, 0, mirror]

	tavg = model_blank(nx,nx)
	for im in xrange(nima):
		Util.add_img( tavg, rot_shift2D(data[im], params[im][0], params[im][1], params[im][2], params[im][3]) )
	tavg /= nima
	tavg = filt_tanl(tavg, 0.1, FF)
	#tavg = EMData('image.hdf')
	#for im in xrange(nima):  print  im,params[im]

	cnx = nx/2+1
	cny = cnx
	mode = "F"
	numr = Numrinit(first_ring, last_ring, rstep, mode)
	wr = ringwe(numr, mode)

	cs = [0.0]*2
	total_iter = 0
	for Iter in xrange(max_iter):
		for N_step in xrange(len(xrng)):
			total_iter += 1
			if Iter%4 != 0 or total_iter > max_iter*len(xrng)-10: delta = 0.0
			else:                                                 delta = dst
			#delta=0.0
			#for im in xrange(nima):		print  " sx, sy:  %2d   %2d"%(params[im][1], params[im][2]) 
			ali2d_single_iter_fast(data, dimage, params, numr, wr, cs, tavg, cnx, cny, \
					xrng[N_step], yrng[N_step], step[N_step], maxrange = maxrange, mode=mode, delta=delta)
			tavg = model_blank(nx,nx)
			sx_sum = 0.0
			sy_sum = 0.0
			for im in xrange(nima):
				sxi = -params[im][1]
				syi = -params[im][2]
				"""
				co =  cos(radians(alphai))
				so = -sin(radians(alphai))
				"""
				co =  cos(radians(params[im][0]))
				so = -sin(radians(params[im][0]))
				sxs = sxi*co - syi*so
				sys = sxi*so + syi*co
				Util.add_img( tavg, rot_shift2D(data[im], params[im][0], sxs, sys, params[im][3]) )

				if params[im][3] == 0: sx_sum += sxs
				else:                  sx_sum -= sxs
				sy_sum += sys

			tavg /= nima
			if( FH > 0.0):
				fl = 0.1+(FH-0.1)*Iter/float(max_iter-1)
			tavg = filt_tanl(tavg, fl, FF)
			"""
			if( xrng[0] > 0.0 ): sx_sum = int(sx_sum/float(nima)+0.5)
			if( yrng[0] > 0.0 ): sy_sum = int(sy_sum/float(nima)+0.5)
			#print ' ave shift',sx_sum, sy_sum
			tavg = cyclic_shift(tavg, -sx_sum, -sy_sum)
			"""
			if( xrng[0] > 0.0 ): sx_sum = sx_sum/float(nima)
			if( yrng[0] > 0.0 ): sy_sum = sy_sum/float(nima)
			tavg = fshift(tavg, -sx_sum, -sy_sum)
			#tavg.write_image('tavg.hdf',total_iter-1)
	for im in xrange(nima):
		sxi = -params[im][1]
		syi = -params[im][2]
		"""
		co =  cos(radians(alphai))
		so = -sin(radians(alphai))
		"""
		co =  cos(radians(params[im][0]))
		so = -sin(radians(params[im][0]))
		sxs = sxi*co - syi*so
		sys = sxi*so + syi*co
		params[im][1] = sxs
		params[im][2] = sys

	return tavg, params
'''
"""55
	ndat = 0
	for i in xrange(0, nfils, 2): ndat += len(filaments[i])
	td = [None]*ndat
	k = 0
	for i in xrange(0, nfils, 2):
		for j in xrange(indcs[i][0],indcs[i][1]):
			td[k] = data[j]
			k += 1

	if CTF:  vol_even = recons3d_4nn_ctf_MPI(myid, td, symmetry=sym, snr = snr, npad = npad, xysize = xysize, zsize = zsize)
	else:    vol_even = recons3d_4nn_MPI(myid, td, symmetry=sym, snr = snr, npad = npad, xysize = xysize, zsize = zsize)

	ndat = 0
	for i in xrange(1, nfils, 2): ndat += len(filaments[i])
	td = [None]*ndat
	k = 0
	for i in xrange(1, nfils, 2):
		for j in xrange(indcs[i][0],indcs[i][1]):
			td[k] = data[j]
			k += 1

	if CTF:  vol_odd = recons3d_4nn_ctf_MPI(myid, td, symmetry=sym, snr = snr, npad = npad, xysize = xysize, zsize = zsize)
	else:    vol_odd = recons3d_4nn_MPI(myid, td, symmetry=sym, snr = snr, npad = npad, xysize = xysize, zsize = zsize)

	del td

	if  myid == main_node  :
		vol_even = vol_even.helicise(pixel_size, dp, dphi, fract, rmax, rmin)
		vol_even = sym_vol(vol_even, symmetry=sym)
		vol_odd  = vol_odd.helicise(pixel_size, dp, dphi, fract, rmax, rmin)
		vol_odd = sym_vol(vol_odd, symmetry=sym)
	
		f = fsc_mask( vol_even, vol_odd, fscmask, 1.0, os.path.join(outdir, "hfsc.txt"))

	del vol_even, vol_odd
	"""
"""56
		pass#IMPORTIMPORTIMPORT from filter import filt_tanl
		fullvol0 = filt_tanl(fullvol0, 0.3, 0.2)
		fullvol0 = fullvol0.helicise(pixel_size, dp, dphi, fract, rmax, rmin)
		fullvol0 = sym_vol(fullvol0, symmetry=sym)
		"""
"""57
	forg = []
	for ivol in xrange(nfils):
		for im in xrange(inds[ivol][0],inds[ivol][1]):
				T = data[im].get_attr("xform.projection")
				d = T.get_params('spider')
				forg.append([d["phi"],d["theta"],d["psi"],-d["tx"],-d["ty"],data[im].get_attr('ID')])
	write_text_row(forg,"pheader%03d.txt"%myid)
	"""
'''58
	def rot2pad(imi, alpha=0.0, sx=0.0, sy=0.0):
		pass#IMPORTIMPORTIMPORT from utilities    import pad
		pass#IMPORTIMPORTIMPORT from fundamentals import rot_shift2D
		lnx = imi.get_xsize()
		lny = imi.get_ysize()
		ln = max(lnx,lny)
		if lnx == lny: return rot_shift2D(imi,alpha,sx,sy)
		else:          return Util.window(rot_shift2D(pad(imi,ln,ln,1,"circumference"), alpha,sx,sy), lnx, lny,1, 0,0,0)
	'''
'''59
		# check previous max and if does not exist set it to -1.e23
		p = data[im].get_attr_default('previousmax',-1.0e23)
		if( p == -1.0e23 ):
			resetatone = True
			data[im].set_attr('previousmax', p)
		'''
'''60
			if doExhaustive:
				Util.constrained_helix_exhaustive(ldata, fdata[indcs[ifil][0]:indcs[ifil][1]], refproj, rotproj, [float(dp), float(dphi), float(rise), float(delta)], [int(nphi), int(phiwobble), int(rng), int(ywobble), int(Dsym), int(nwx), int(nwy), int(nwxc), int(nwyc)], FindPsi, float(psi_max), crefim, numr, int(maxrin), mode, int(cnx), int(cny))
				terminate = 0
			else:
				tempch = Util.constrained_helix(ldata, fdata[indcs[ifil][0]:indcs[ifil][1]], refproj, rotproj, [float(dp), float(dphi), float(rise), float(delta)], [int(nphi), int(phiwobble), int(rng), int(ywobble), int(Dsym), int(nwx), int(nwy), int(nwxc), int(nwyc)], FindPsi, float(psi_max), crefim, numr, int(maxrin), mode, int(cnx), int(cny))
				#print "tempch, Iter, myid: ", tempch, Iter, myid
				#tempch = constrained_helix_SHC(ldata, fdata[indcs[ifil][0]:indcs[ifil][1]], refproj, rotproj,  dp, dphi, rise, delta ,  nphi, phiwobble, rng, ywobble, Dsym, nwx, nwy, nwxc, nwyc , FindPsi, psi_max, crefim, numr, maxrin, mode, cnx, cny, myid, main_node)
				if tempch > -1:
					#if myid == main_node:
					#	print_msg("tempch %d\n"%tempch)
					terminate = 0
			'''
"""61
		#  Should we continue??
		terminate = mpi_reduce(terminate, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD)
		terminate = mpi_bcast(terminate, 1, MPI_INT, 0, MPI_COMM_WORLD)
		terminate = int(terminate[0])
		if terminate == nproc:
			if myid == main_node: print_end_msg("helicon_MPI")
			return
		"""
"""62
			# If the previous iteration did a reconstruction, then generate new refrings
			if ( (Iter - 1) % search_iter == 0):


				volft,kb = prep_vol( vol )
				#  What about cushion for a neighborhood?  PAP 06/04/2014
				refrings = prepare_refffts( volft, kb, data_nn,data_nn,nz, segmask, delta[N_step], \
					MPI=True, psimax=psi_max, psistep=psistep, initial_theta =initial_theta, delta_theta = delta_theta)
				#refrings = prepare_refrings2(  volft, kb, nmax, segmask, delta[N_step], ref_a, symref, numr, MPI = True, phiEqpsi = "Zero", initial_theta =initial_theta, delta_theta = delta_theta)
				del volft,kb

				if myid== main_node:
					print_msg( "Time to prepare rings: %d\n" % (time()-start_time) )
					start_time = time()
			"""
"""63
			# If the previous iteration did a reconstruction, then generate new refrings
			if ( (Iter - 1) % search_iter == 0):


				volft,kb = prep_vol( vol )
				#  What about cushion for a neighborhood?  PAP 06/04/2014
				refrings = prepare_refffts( volft, kb, data_nn,data_nn,nz, segmask, delta[N_step], \
					MPI=True, psimax=psi_max, psistep=psistep, initial_theta =initial_theta, delta_theta = delta_theta)
				#refrings = prepare_refrings2(  volft, kb, nmax, segmask, delta[N_step], ref_a, symref, numr, MPI = True, phiEqpsi = "Zero", initial_theta =initial_theta, delta_theta = delta_theta)
				del volft,kb

				if myid== main_node:
					print_msg( "Time to prepare rings: %d\n" % (time()-start_time) )
					start_time = time()
			"""
"""64
def setfilori_MA(fildata, pixel_size, dp, dphi):
	pass#IMPORTIMPORTIMPORT from utilities		import get_params_proj, set_params_proj, get_dist
	pass#IMPORTIMPORTIMPORT from applications	import filamentupdown
	pass#IMPORTIMPORTIMPORT from copy 			import copy
	pass#IMPORTIMPORTIMPORT from math 			import atan2, sin, cos, pi

	#if sym != 'c1':
	#	ERROR("does not handle any point-group symmetry other than c1 for the time being.", 'setfilori_MA')

	rise 	= dp/pixel_size
	ddphi   = pixel_size/dp*dphi
	ns 		= len(fildata)
	qv 		= pi/180.0

	phig 	= [0.0]*ns # given phi
	psig 	= [0.0]*ns # given psi
	yg 		= [0.0]*ns # given y
	xg 		= [0.0]*ns # given x
	thetag	= [0.0]*ns # given theta

	coords = [[] for im in xrange(ns)]
	for i in xrange(ns):
		coords[i] = fildata[i].get_attr('ptcl_source_coord')
		phig[i], thetag[i], psig[i] , xg[i], yg[i] = get_params_proj(fildata[i])
		#print "%3d  %5.1f   %5.1f   %5.1f   %5.1f   %5.1f"%(i,phig[i], thetag[i], psig[i] , xg[i], yg[i])
		if( abs(psig[i] - psig[0]) )> 90.0:
			ERROR('PSI should be pointing in the same direction for all segments belonging to same filament', 'setfilori_MA')

	# forward predicted psi of segment i is current psi of segment i + 1
	# backward predicted psi of segment i is current psi of segment i - 1
	# For now set the new psi to the psi closest to the two predicted psi
	conspsi = [0.0]*ns
	for i in xrange(1,ns-1):
		ff = psig[i+1]*qv
		bb = psig[i-1]*qv
		conspsi[i] = (atan2(  (sin(ff)+sin(bb)) , (cos(ff)+cos(bb)) )/qv)%360.0
	#  border psi values
	ff = psig[1]*qv
	bb = psig[0]*qv
	conspsi[0]    = (atan2(  (sin(ff)+sin(bb)) , (cos(ff)+cos(bb)) )/qv)%360.0
	ff = psig[ns-1]*qv
	bb = psig[ns-2]*qv
	conspsi[ns-1] = (atan2(  (sin(ff)+sin(bb)) , (cos(ff)+cos(bb)) )/qv)%360.0

	# predict theta in same way as psi. 
	constheta = [0.0]*ns
	for i in xrange(1,ns-1):
		ff = thetag[i+1]*qv
		bb = thetag[i-1]*qv
		constheta[i] = (atan2(  (sin(ff)+sin(bb)) , (cos(ff)+cos(bb)) )/qv)%360.0
	#  border theta values
	ff = thetag[1]*qv
	bb = thetag[0]*qv
	constheta[0]    = (atan2(  (sin(ff)+sin(bb)) , (cos(ff)+cos(bb)) )/qv)%360.0
	ff = thetag[ns-1]*qv
	bb = thetag[ns-2]*qv
	constheta[ns-1] = (atan2(  (sin(ff)+sin(bb)) , (cos(ff)+cos(bb)) )/qv)%360.0


	#  X-shift (straightforward)
	consx = [0.0]*ns
	for i in xrange(1,ns-1): consx[i] = (xg[i+1] + xg[i-1])/2.0
	consx[0]    = (xg[1] + xg[0])/2.0
	consx[ns-1] = (xg[ns-1] + xg[ns-2])/2.0

	#  Y-shift - include non-integer distance.

	# Predict phi angles based on approximate distances calculated using theta=90 between nearby segments.
	# (The other option is to calculate distance between segments using predicted theta. 
	#   But for now, just use theta=90 since for reasonable deviations of theta from 90 (say +/- 10 degrees), 
	#   and an intersegment distance of say >= 15 pixels, the difference theta makes is minimal.)

	sgn = (1 - fildata[0].get_attr("updown")*2)

	consy   = [0.0]*ns
	consphi = [0.0]*ns
	
	
	#per = 0.0
	#yer = 0.0
	
	for i in xrange(ns):
		if( i>0 ):
			# from i-1 predict i
			dist = get_dist(coords[i-1], coords[i])
			qd = round((yg[i-1] + dist)/rise)
			yb   = yg[i-1] + dist - rise*qd
			#print  " YB ",yg[i-1],dist,qd,yb
			phib = qv*((phig[i-1] + sgn*dphi*qd)%360.0)
		else:
			yb   = yg[0]
			phib = qv*phig[0]

		if( i < ns-1 ):
			# from i+1 predict i
			dist = get_dist(coords[i+1], coords[i])
			qd = round((yg[i+1] - dist)/rise)
			yf   = yg[i+1] - dist - rise*qd
			#print  " YF ",yg[i+1],dist,qd,yf
			phif = qv*((phig[i+1] + sgn*dphi*qd)%360.0)
		else:
			yf   = yg[-1]
			phif = qv*phig[-1]

		consy[i] = (yb + yf)/2.0
		#print "    YP",yg[max(0,i-1)], yg[min(i+1,ns-1)], yb,yf, yg[i],consy[i],"   >>>   ",yg[i]-consy[i]
		consphi[i] = (atan2(  (sin(phib)+sin(phif)) , (cos(phib)+cos(phif)) )/qv)%360.0
		#print " PHI ",i, round(phig[max(0,i-1)],1), round(phig[i],1), round(phig[min(i+1,ns-1)],1), \
		#	round(phib/qv,1), round(phif/qv,1), round(consphi[i],1),\
		#	"    >><>>>>> ",round((round((phig[i]-consphi[i])*100)/100.)%360.,1)
		#yer += abs(yg[i]-consy[i])
		#per += abs(phig[i]-consphi[i])
		set_params_proj(fildata[i], [consphi[i], constheta[i], conspsi[i], consx[i], consy[i]])
	#print yer, per
	return
"""
"""65
		vol = sym_vol(vol, symmetry=sym)
		ref_data = [vol, mask3D]
		#if  fourvar:  ref_data.append(varf)
		vol = user_func(ref_data)
		vol = vol.helicise(pixel_size, dp, dphi, fract, rmax, rmin)
		vol = sym_vol(vol, symmetry=sym)
		drop_image(vol, os.path.join(outdir, "volf.hdf"))
		print_msg("\nSymmetry search and user function time = %d\n"%(time()-start_time))
		start_time = time()
		"""
"""66
		pass#IMPORTIMPORTIMPORT import os
		outdir = "./"
		info_file = os.path.join(outdir, "progress%04d"%myid)
		finfo = open(info_file, 'w')
		"""
"""  Have to think about it PAP67
					if( (historyofchanges[-1]-historyofchanges[0])/2/(historyofchanges[-1]+historyofchanges[0]) <0.05 ):
						terminate = 1
						log.add("...............")
						log.add(">>>>>>>>>>>>>>>   Will terminate as orientations do not improve anymore")
					"""
"""68
			#=========================================================================
			if(False):  #total_iter%1 == 5 or terminate):
				# gather parameters
				params = []
				previousmax = []
				for im in data:
					t = get_params_proj(im)
					params.append( [t[0], t[1], t[2], t[3]/shrinkage, t[4]/shrinkage] )
					#if(t[3] >0.0 or t[4]>0.0):  print  "  ERRROR  ",t
					previousmax.append(im.get_attr("previousmax"))
				assert(nima == len(params))
				params = wrap_mpi_gatherv(params, 0, mpi_comm)
				if myid == 0:
					assert(total_nima == len(params))
				previousmax = wrap_mpi_gatherv(previousmax, 0, mpi_comm)
				if myid == main_node:
					pass#IMPORTIMPORTIMPORT from utilities import write_text_row, write_text_file
					write_text_row(params, "soft/params%04d.txt"%total_iter)
					write_text_file(previousmax, "soft/previousmax%04d.txt"%total_iter)


				del previousmax, params
				i = 1
				while data[0].has_attr("xform.projection" + str(i)):
					params = []
					previousmax = []
					for im in data:

						try:
							#print  im.get_attr("xform.projection" + str(i))
							t = get_params_proj(im,"xform.projection" + str(i))
						except:
							print " NO XFORM  ",myid, i,im.get_attr('ID')
							pass#IMPORTIMPORTIMPORT from sys import exit
							exit()

						params.append( [t[0], t[1], t[2], t[3]/shrinkage, t[4]/shrinkage] )
					assert(nima == len(params))
					params = wrap_mpi_gatherv(params, 0, mpi_comm)
					if myid == 0:
						assert(total_nima == len(params))
					if myid == main_node:
						write_text_row(params, "soft/params-%04d-%04d.txt"%(i,total_iter))
					del previousmax, params
					i+=1


			if( ( terminate or (Iter == max_iter) ) and (myid == main_node) ):
				if( type(stack) is types.StringType ):
					pass#IMPORTIMPORTIMPORT from EMAN2 import Vec2f, Transform
					pass#IMPORTIMPORTIMPORT from EMAN2db import db_open_dict
					DB = db_open_dict(stack)
					for im in xrange(len(params)):
						t = Transform({"type":"spider","phi":params[im][0],"theta":params[im][1],"psi":params[im][2]})
						t.set_trans(Vec2f(-params[im][3], -params[im][4]))
						DB.set_attr(particle_ids[im], "xform.projection", t)
					DB.close()
				else:
					for im in xrange(len(params)): set_params_proj(stack[particle_ids[im]], params[im])
			"""
"""69
	if myid == main_node:
		os.mkdir(outdir)
		pass#IMPORTIMPORTIMPORT import global_def
		sparx_global_def.LOGFILE =  os.path.join(outdir, sparx_global_def.LOGFILE)
		print_begin_msg("local_ali3d_MPI")
		pass#IMPORTIMPORTIMPORT import user_functions
		user_func = sparx_user_functions.factory[user_func_name]
		if CTF:
			ima = EMData()
			ima.read_image(stack, 0)
			ctf_applied = ima.get_attr_default("ctf_applied", 0)
			del ima
			if ctf_applied == 1:  ERROR("local_ali3d does not work for CTF-applied data", "local_ali3d_MPI", 1,myid)
	mpi_barrier(mpi_comm)
	"""
"""70
	if myid == main_node:
		pass#IMPORTIMPORTIMPORT import user_functions
		user_func = sparx_user_functions.factory[user_func_name]

		print_msg("Input stack                 : %s\n"%(stack))
		print_msg("Output directory            : %s\n"%(outdir))
		print_msg("Maskfile                    : %s\n"%(Tracker["constants"]["mask3D"]))
		print_msg("Outer radius                : %i\n"%(last_ring))
		print_msg("Angular search range        : %s\n"%(delta))
		print_msg("Shift search range          : %f\n"%(ts))
		print_msg("Maximum iteration           : %i\n"%(max_iter))
		print_msg("Center type                 : %i\n"%(center))
		print_msg("CTF correction              : %s\n"%(CTF))
		print_msg("Signal-to-Noise Ratio       : %f\n"%(snr))
		print_msg("Symmetry group              : %s\n"%(sym))
		print_msg("Chunk size                  : %f\n\n"%(chunk))
		print_msg("User function               : %s\n"%(user_func_name))
	"""
"""71
		if( type(stack) is types.StringType ):
			pass#IMPORTIMPORTIMPORT from EMAN2 import Vec2f, Transform
			pass#IMPORTIMPORTIMPORT from EMAN2db import db_open_dict
			DB = db_open_dict(stack)
			for im in xrange(len(params)):
				t = Transform({"type":"spider","phi":params[im][0],"theta":params[im][1],"psi":params[im][2]})
				t.set_trans(Vec2f(-params[im][3], -params[im][4]))
				DB.set_attr(particle_ids[im], "xform.projection", t)
			DB.close()
		else:
			for im in xrange(len(params)): set_params_proj(stack[particle_ids[im]], params[im])

		log.add("Time to write header information= %d\n"%(time()-start_time))
		"""
'''72
	if fourvar:
		pass#IMPORTIMPORTIMPORT from reconstruction import rec3D_MPI
		pass#IMPORTIMPORTIMPORT from statistics     import varf3d_MPI
		#  Compute Fourier variance
		vol, fscc = rec3D_MPI(data,snr,sym,fscmask,os.path.join(outdir, "resolution0000"), myid, main_node, finfo=frec, npad=npad)
		varf = varf3d_MPI(data, os.path.join(outdir, "ssnr0000"), None, vol, last_ring, 1.0, 1, CTF, 1, sym, myid)
		if myid == main_node:   
			varf = 1.0/varf
			varf.write_image( os.path.join(outdir,"varf0000.hdf") )
	else:
		varf = None
	'''
'''73
					if CTF:
						ref = fft(filt_ctf( prgl( volft, [phi,tht,psi,-s2x,-s2y],1,False), ctf ))
					else:
						ref = prgl( volft, [phi,tht,psi,-s2x,-s2y],1)
					peak = ref.cmp("ccc",data[im],{"mask":mask2D, "negative":0})
					'''
'''74
					#  FSC distance
					#  Ref is in reciprocal space
					if CTF:
						ref = filt_ctf( prgl( volft, [phi,tht,psi,-s2x,-s2y], 1, False), ctf )
					else:
						ref = prgl( volft, [phi,tht,psi,-s2x,-s2y], 1, False)
					pass#IMPORTIMPORTIMPORT from statistics import fsc
					if(focus):
						mask2D = binarize( prgl( focus, [phi,tht,psi,-s2x,-s2y]), 1)
						tempx = fsc(ref, fft(data[im]*mask2D))[1]
					else:
						tempx = fsc(ref, data[im])[1]
					peak = sum(tempx[1:highres[iref]])/highres[iref]

					if not(finfo is None):
						finfo.write( "ID,iref,peak: %6d %d %8.5f\n" % (list_of_particles[im],iref,peak) )
				else:
					if an[N_step] == -1:
						peak, pixel_error = proj_ali_incore(data[im],refrings,numr,xrng[N_step],yrng[N_step],step[N_step])
					else:
						peak, pixel_error = proj_ali_incore_local(data[im], refrings, list_of_reference_angles, numr,\
																	xrng[N_step], yrng[N_step], step[N_step], an[N_step])
					if not(finfo is None):
						phi,tht,psi,s2x,s2y = get_params_proj(data[im])
						finfo.write( "ID,iref,peak,trans: %6d %d %f %f %f %f %f %f\n"%(list_of_particles[im],iref,peak,phi,tht,psi,s2x,s2y) )
						finfo.flush()
					'''
"""75
		if myid == main_node:
			refdata    = [None]*7
			refdata[0] = numref
			refdata[1] = outdir
			refdata[2] = None
			refdata[3] = total_iter
			refdata[4] = varf
			refdata[5] = mask3D
			refdata[6] = Tracker["low_pass_filter"]#(runtype=="REFINEMENT") # whether align on 50S, this only happens at refinement step
			user_func( refdata )
		"""
"""76
	if myid == main_node:
		pass#IMPORTIMPORTIMPORT from utilities import file_type
		if file_type(stack) == "bdb":
			pass#IMPORTIMPORTIMPORT from utilities import recv_attr_dict_bdb
			recv_attr_dict_bdb(main_node, stack, data, par_str, image_start, image_end, number_of_proc)
		else:
			pass#IMPORTIMPORTIMPORT from utilities import recv_attr_dict
			recv_attr_dict(main_node, stack, data, par_str, image_start, image_end, number_of_proc)
	else:		send_attr_dict(main_node, data, par_str, image_start, image_end)
	"""
'''77
	if(myid == main_node):	
		total_nima = EMUtil.get_image_count(stack)
		list_of_particles = range(total_nima)
	
	else:
		total_nima =0

	total_nima = bcast_number_to_all(total_nima, source_node = main_node)

	if(myid != main_node):
		list_of_particles = [-1]*total_nima

	list_of_particles = bcast_list_to_all(list_of_particles, myid,  source_node = main_node)

	image_start, image_end = MPI_start_end(total_nima, number_of_proc, myid)
	# create a list of images for each node
	list_of_particles = list_of_particles[image_start: image_end]
	nima = len(list_of_particles)
	'''
'''78
					if CTF:
						ref = fft(filt_ctf( prgl( volft, [phi,tht,psi,-s2x,-s2y],1,False), ctf ))
					else:
						ref = prgl( volft, [phi,tht,psi,-s2x,-s2y],1)
					if(focus != None):  mask2D = binarize( prgl( focus, [phi,tht,psi,-s2x,-s2y]),1)  #  Should be precalculated!!
					peak = ref.cmp("ccc",data[im],{"mask":mask2D, "negative":0})
					'''
'''79
					#  FSC distance
					#  Ref is in reciprocal space
					if CTF:
						ref = filt_ctf( prgl( volft, [phi,tht,psi,-s2x,-s2y], 1, False), ctf )
					else:
						ref = prgl( volft, [phi,tht,psi,-s2x,-s2y], 1, False)
					if(focus):
						mask2D = binarize( prgl( focus, [phi,tht,psi,-s2x,-s2y]), 1)
						tempx = fsc(ref, fft(data[im]*mask2D))[1]
					else:
						tempx = fsc(ref, data[im])[1]
					peak = sum(tempx[1:highres[iref]])/highres[iref]
					'''
"""80
		#  Trying to use ISAC code for EQ-Kmeans  PAP 03/21/2015
		if myid == main_node:

			for imrefa in xrange(numrefang):
				pass#IMPORTIMPORTIMPORT from utilities import findall
				N = findall(imrefa, assigntorefa)
				current_nima = len(N)
				if( current_nima >= numref and report_error == 0):
					tasi = [[] for iref in xrange(numref)]
					maxasi = current_nima//numref
					nt = current_nima
					kt = numref
					K = range(numref)

					d = empty( (numref, current_nima), dtype = float32)
					for ima in xrange(current_nima):
						for iref in xrange(numref):  d[iref][ima] = dtot[iref][N[ima]]

			e b = empty( (numref, total_nima), dtype = float32)
			for ima in xrange(total_nima):
				for iref in xrange(numref):  d[iref][ima] = dtot[iref][N[ima]]
			id_list_long = Util.assign_groups(str(d.__array_interface__['data'][0]), numref, nima) # string with memory address is passed as parameters
			del d
			id_list = [[] for i in xrange(numref)]
			maxasi = total_nima/numref
			for i in xrange(maxasi*numref):
				id_list[i/maxasi].append(id_list_long[i])
			for i in xrange(total_nima%maxasi):
				id_list[id_list_long[-1]].append(id_list_long[maxasi*numref+i])
			for iref in xrange(numref):
				id_list[iref].sort()

			assignment = [0]*total_nima
			for iref in xrange(numref):
				for im in id_list[iref]: assignment[im] = iref
		else:
			assignment = [0]*total_nima
		mpi_barrier(MPI_COMM_WORLD)
		#belongsto = mpi_bcast(belongsto, nima, MPI_INT, main_node, MPI_COMM_WORLD)
		#belongsto = map(int, belongsto)
		"""
"""81
		if myid == main_node:
			assignment = [0]*total_nima
			for iref in xrange(numref):
				for im in xrange(len(asi[iref])):
					assignment[asi[iref][im]] = iref
			del asi
		"""
'''82
		if myid == main_node:
			SA = False
			maxasi = total_nima//numref
			asi = [[] for iref in xrange(numref)]
			nt = total_nima
			kt = numref
			K = range(numref)
			N = range(total_nima)

			while nt > 0 and kt > 1:
				l = d.argmax()
				group = l//total_nima
				ima   = l-total_nima*group
				if SA:
					J = [0.0]*numref
					sJ = 0
					Jc = [0.0]*numref
					for iref in xrange(numref):
						J[iref] = exp(d[iref][ima]/T)
						sJ += J[iref]
					for iref in xrange(numref):
						J[iref] /= sJ
					Jc[0] = J[0]
					for iref in xrange(1, numref):
						Jc[iref] = Jc[iref-1]+J[iref]
					sss = random()
					for group in xrange(numref):
						if( sss <= Jc[group]): break
				asi[group].append(N[ima])
				for iref in xrange(numref):  d[iref][ima] = -1.e10
				nt -= 1
				masi = len(asi[group])
				if masi == maxasi:
					for im in xrange(total_nima):  d[group][im] = -1.e10
					kt -= 1
			else:
				mas = [len(asi[iref]) for iref in xrange(numref)]
				group = mas.index(min(mas))
				del mas
				for im in xrange(total_nima):
					kt = 0
					go = True
					while(go and kt < numref):
						if d[kt][im] > -1.e10:
							asi[group].append(im)
							go = False
						kt += 1

			assignment = [0]*total_nima
			for iref in xrange(numref):
				for im in xrange(len(asi[iref])):
					assignment[asi[iref][im]] = iref

			del asi, d, N, K
			if  SA:  del J, Jc


		else:
			assignment = []
		'''
"""83
		if runtype=="REFINEMENT":
			if fourvar:
				varf = varf3d_MPI(data, os.path.join(outdir, "ssnr%04d"%total_iter), None,sumvol,last_ring, 1.0, 1, CTF, 1, sym, myid)
				if myid == main_node:   
					varf = 1.0/varf
					varf.write_image( os.path.join(outdir,"varf%04d.hdf"%total_iter) )
		"""
"""84
			frcs={}
			for iref in xrange(numref):
				frc=read_text_file(os.path.join(outdir, "resolution_%02d_%04d"%(iref, total_iter)),-1)
				frcs[iref]=frc
			refdata = [None]*7
			refdata[0] = numref
			refdata[1] = outdir
			refdata[2] = None #frequency_low_pass
			refdata[3] = total_iter
			refdata[4] = None
			refdata[5] = mask3D
			refdata[6] = Tracker["low_pass_filter"]#(runtype=="REFINEMENT") # whether to align on 50S, this only happens at refinement step
		"""
"""85
	        	if myid == main_node:
				pass#IMPORTIMPORTIMPORT from utilities import file_type
	        		if(file_type(stack) == "bdb"):
	        			pass#IMPORTIMPORTIMPORT from utilities import recv_attr_dict_bdb
	        			recv_attr_dict_bdb(main_node, stack, data, par_str, image_start, image_end, number_of_proc)
	        		else:
	        			pass#IMPORTIMPORTIMPORT from utilities import recv_attr_dict
	        			recv_attr_dict(main_node, stack, data, par_str, image_start, image_end, number_of_proc)
	        	else:		send_attr_dict(main_node, data, par_str, image_start, image_end)
			"""
'''86
	if(myid == main_node):	
		total_nima = EMUtil.get_image_count(stack)
		list_of_particles = range(total_nima)
	
	else:
		total_nima =0

	total_nima = bcast_number_to_all(total_nima, source_node = main_node)

	if(myid != main_node):
		list_of_particles = [-1]*total_nima

	list_of_particles = bcast_list_to_all(list_of_particles, myid,  source_node = main_node)

	image_start, image_end = MPI_start_end(total_nima, number_of_proc, myid)
	# create a list of images for each node
	list_of_particles = list_of_particles[image_start: image_end]
	nima = len(list_of_particles)
	'''
'''87
					if CTF:
						ref = fft(filt_ctf( prgl( volft, [phi,tht,psi,-s2x,-s2y],1,False), ctf ))
					else:
						ref = prgl( volft, [phi,tht,psi,-s2x,-s2y],1)
					if(focus != None):  mask2D = binarize( prgl( focus, [phi,tht,psi,-s2x,-s2y]),1)  #  Should be precalculated!!
					peak = ref.cmp("ccc",data[im],{"mask":mask2D, "negative":0})
					'''
'''88
					#  FSC distance
					#  Ref is in reciprocal space
					if CTF:
						ref = filt_ctf( prgl( volft, [phi,tht,psi,-s2x,-s2y], 1, False), ctf )
					else:
						ref = prgl( volft, [phi,tht,psi,-s2x,-s2y], 1, False)
					if(focus):
						mask2D = binarize( prgl( focus, [phi,tht,psi,-s2x,-s2y]), 1)
						tempx = fsc(ref, fft(data[im]*mask2D))[1]
					else:
						tempx = fsc(ref, data[im])[1]
					peak = sum(tempx[1:highres[iref]])/highres[iref]
					'''
"""89
		#  Trying to use ISAC code for EQ-Kmeans  PAP 03/21/2015
		if myid == main_node:

			for imrefa in xrange(numrefang):
				pass#IMPORTIMPORTIMPORT from utilities import findall
				N = findall(imrefa, assigntorefa)
				current_nima = len(N)
				if( current_nima >= numref and report_error == 0):
					tasi = [[] for iref in xrange(numref)]
					maxasi = current_nima//numref
					nt = current_nima
					kt = numref
					K = range(numref)

					d = empty( (numref, current_nima), dtype = float32)
					for ima in xrange(current_nima):
						for iref in xrange(numref):  d[iref][ima] = dtot[iref][N[ima]]

			e b = empty( (numref, total_nima), dtype = float32)
			for ima in xrange(total_nima):
				for iref in xrange(numref):  d[iref][ima] = dtot[iref][N[ima]]
			id_list_long = Util.assign_groups(str(d.__array_interface__['data'][0]), numref, nima) # string with memory address is passed as parameters
			del d
			id_list = [[] for i in xrange(numref)]
			maxasi = total_nima/numref
			for i in xrange(maxasi*numref):
				id_list[i/maxasi].append(id_list_long[i])
			for i in xrange(total_nima%maxasi):
				id_list[id_list_long[-1]].append(id_list_long[maxasi*numref+i])
			for iref in xrange(numref):
				id_list[iref].sort()

			assignment = [0]*total_nima
			for iref in xrange(numref):
				for im in id_list[iref]: assignment[im] = iref
		else:
			assignment = [0]*total_nima
		mpi_barrier(MPI_COMM_WORLD)
		#belongsto = mpi_bcast(belongsto, nima, MPI_INT, main_node, MPI_COMM_WORLD)
		#belongsto = map(int, belongsto)
		"""
"""90
		if myid == main_node:
			assignment = [0]*total_nima
			for iref in xrange(numref):
				for im in xrange(len(asi[iref])):
					assignment[asi[iref][im]] = iref
			del asi
		"""
'''91
		if myid == main_node:
			SA = False
			maxasi = total_nima//numref
			asi = [[] for iref in xrange(numref)]
			nt = total_nima
			kt = numref
			K = range(numref)
			N = range(total_nima)

			while nt > 0 and kt > 1:
				l = d.argmax()
				group = l//total_nima
				ima   = l-total_nima*group
				if SA:
					J = [0.0]*numref
					sJ = 0
					Jc = [0.0]*numref
					for iref in xrange(numref):
						J[iref] = exp(d[iref][ima]/T)
						sJ += J[iref]
					for iref in xrange(numref):
						J[iref] /= sJ
					Jc[0] = J[0]
					for iref in xrange(1, numref):
						Jc[iref] = Jc[iref-1]+J[iref]
					sss = random()
					for group in xrange(numref):
						if( sss <= Jc[group]): break
				asi[group].append(N[ima])
				for iref in xrange(numref):  d[iref][ima] = -1.e10
				nt -= 1
				masi = len(asi[group])
				if masi == maxasi:
					for im in xrange(total_nima):  d[group][im] = -1.e10
					kt -= 1
			else:
				mas = [len(asi[iref]) for iref in xrange(numref)]
				group = mas.index(min(mas))
				del mas
				for im in xrange(total_nima):
					kt = 0
					go = True
					while(go and kt < numref):
						if d[kt][im] > -1.e10:
							asi[group].append(im)
							go = False
						kt += 1

			assignment = [0]*total_nima
			for iref in xrange(numref):
				for im in xrange(len(asi[iref])):
					assignment[asi[iref][im]] = iref

			del asi, d, N, K
			if  SA:  del J, Jc


		else:
			assignment = []
		'''
"""92
		if runtype=="REFINEMENT":
			if fourvar:
				varf = varf3d_MPI(data, os.path.join(outdir, "ssnr%04d"%total_iter), None,sumvol,last_ring, 1.0, 1, CTF, 1, sym, myid)
				if myid == main_node:   
					varf = 1.0/varf
					varf.write_image( os.path.join(outdir,"varf%04d.hdf"%total_iter) )
		"""
"""93
			frcs={}
			for iref in xrange(numref):
				frc=read_text_file(os.path.join(outdir, "resolution_%02d_%04d"%(iref, total_iter)),-1)
				frcs[iref]=frc
			refdata = [None]*7
			refdata[0] = numref
			refdata[1] = outdir
			refdata[2] = None #frequency_low_pass
			refdata[3] = total_iter
			refdata[4] = None
			refdata[5] = mask3D
			refdata[6] = Tracker["low_pass_filter"]#(runtype=="REFINEMENT") # whether to align on 50S, this only happens at refinement step
		"""
"""94
	        	if myid == main_node:
				pass#IMPORTIMPORTIMPORT from utilities import file_type
	        		if(file_type(stack) == "bdb"):
	        			pass#IMPORTIMPORTIMPORT from utilities import recv_attr_dict_bdb
	        			recv_attr_dict_bdb(main_node, stack, data, par_str, image_start, image_end, number_of_proc)
	        		else:
	        			pass#IMPORTIMPORTIMPORT from utilities import recv_attr_dict
	        			recv_attr_dict(main_node, stack, data, par_str, image_start, image_end, number_of_proc)
	        	else:		send_attr_dict(main_node, data, par_str, image_start, image_end)
			"""
def ali2d(stack, outdir, maskfile=None, ir=1, ou=-1, rs=1, xr="4 2 1 1", yr="-1", ts="2 1 0.5 0.25", \
			nomirror=False, dst=0.0, center=-1, maxit=0, \
			CTF=False, snr=1.0, Fourvar=False, Ng=-1, user_func_name="ref_ali2d", \
			CUDA=False, GPUID="", MPI=False, template=None, random_method = ""):
	"""
		Name
			ali2d - Perform 2-D reference-free alignment of an image series
		Input
			stack: set of 2-D images in a stack file (format hdf or bdb), images have to be square (nx=ny)
			maskfile: optional mask file to be used internally during alignment
			inner_radius: inner radius for rotational correlation > 0 (set to 1)
			outer_radius: outer radius for rotational correlation < nx/2-1 (set to nx/2-2, should be set to the radius of the particle)
			ring_step: step between rings in rotational correlation > 0 (set to 1)
			x_range: range for translation search in x direction, search is +/xr 
			y_range: range for translation search in y direction, search is +/yr
			translation_step: step of translation search in both directions, 
			center_type
			max_iteration: maximum number of iterations the program will perform 
			CTF: if this flag is set, the program will use CTF information provided in file headers (for details see I_O). (default no CTF)
			snr: signal-to-noise ratio of the data (default SNR=1.0)
			Fourvar: use Fourier variance to weight the reference (recommended, default False)
			function: name of the user-supplied-function that prepares reference image for each iteration (default ref_ali2d)
			rand_alpha: if this flag is set, the program start with random alpha angle (default no rand_alpha)
			MPI: if this flag is set, use MPI version
		Output
			output_directory: directory name into which the output files will be written.
			header: the alignment parameters are stored in the headers of input files as 'xform.align2d'.
	"""
	if MPI:
		ali2d_MPI(stack, outdir, maskfile, ir, ou, rs, xr, yr, ts, nomirror, dst, \
					center, maxit, CTF, snr, Fourvar, Ng, user_func_name, CUDA, GPUID, \
					random_method = random_method)
		return

	pass#IMPORTIMPORTIMPORT from utilities    import print_begin_msg, print_end_msg, print_msg
	pass#IMPORTIMPORTIMPORT from utilities    import file_type
	pass#IMPORTIMPORTIMPORT import os

	# Comment by Zhengfan Yang on 09/03/10
	# I have decided that outdir should be able to take None as parameter, if the user does not need an output directory.
	# This is particularly useful in the ISAC program, nobody will care what is in the alignment and realignment directory.
	# It takes a lot of disk space and time to write them, too. It may not be much for one iteration, but will be enormous 
	# for several hundred iterations. This change will not affect the normal use.
	if outdir:
		if os.path.exists(outdir):   sparx_global_def.ERROR('Output directory exists, please change the name and restart the program', "ali2d", 1)
		os.mkdir(outdir)
		pass#IMPORTIMPORTIMPORT import global_def
		sparx_global_def.LOGFILE =  os.path.join(outdir, sparx_global_def.LOGFILE)
	sparx_utilities.print_begin_msg("ali2d")
	

	if sparx_utilities.file_type(stack) == "bdb":
		pass#IMPORTIMPORTIMPORT from EMAN2db import db_open_dict
		dummy = EMAN2db.db_open_dict(stack, True)
	# horatio active_refactoring Jy51i1EwmLD4tWZ9_00000_1
	# active = EMUtil.get_all_attributes(stack, 'active')
	# list_of_particles = []
	# for im in xrange(len(active)):
	# 	if active[im]:  list_of_particles.append(im)
	# del active
	# nima = len(list_of_particles)

	nima = EMAN2_cppwrap.EMUtil.get_image_count(stack)
	list_of_particles = list(range(nima))
		
	data = EMAN2_cppwrap.EMData.read_images(stack, list_of_particles)
	for im in range(nima):
		data[im].set_attr('ID', list_of_particles[im])
	try:
		tt = data[0].get_attr("xform.align2d")
		not_set = False
	except:
		print("xform.align2d params are not set yet!")
		not_set = True
	if not_set:
		pass#IMPORTIMPORTIMPORT from utilities import set_params2D
		for index in range(len(data)):
			p=[0.0, 0.0, 0.0, 0, 1]
			sparx_utilities.set_params2D(data[index],p,xform = "xform.align2d")
			
	sparx_utilities.print_msg("Input stack                 : %s\n"%(stack))

	ali2d_data(data, outdir, maskfile, ir, ou, rs, xr, yr, ts, nomirror, dst, \
				center, maxit, CTF, snr, Fourvar, Ng, user_func_name, \
				CUDA, GPUID, True, template, random_method = random_method)

	# write out headers
	pass#IMPORTIMPORTIMPORT from utilities import write_headers
	sparx_utilities.write_headers(stack, data, list_of_particles)
	sparx_utilities.print_end_msg("ali2d")


def ali2d_data(data, outdir, maskfile=None, ir=1, ou=-1, rs=1, xr="4 2 1 1", yr="-1", ts="2 1 0.5 0.25", \
				nomirror=False, dst=0.0, center=-1, maxit=0, \
				CTF=False, snr=1.0, Fourvar=False, Ng=-1, user_func_name="ref_ali2d", \
				CUDA=False, GPUID="", from_ali2d=False, template=None, random_method = ""):

	# Comment by Zhengfan Yang 02/25/11
	# This is where ali2d() actually runs, the reason I divided it into two parts is that
	# this alignment program supports list of EMData not a stack.

	pass#IMPORTIMPORTIMPORT from utilities    import drop_image, get_image, get_input_from_string, get_params2D, set_params2D
	pass#IMPORTIMPORTIMPORT from statistics   import fsc_mask, sum_oe, hist_list
	pass#IMPORTIMPORTIMPORT from alignment    import Numrinit, ringwe, ali2d_single_iter
	pass#IMPORTIMPORTIMPORT from pixel_error  import pixel_error_2D
	pass#IMPORTIMPORTIMPORT from fundamentals import fshift, fft, rot_avg_table
	pass#IMPORTIMPORTIMPORT from utilities    import print_begin_msg, print_end_msg, print_msg
	pass#IMPORTIMPORTIMPORT from utilities    import model_blank, model_circle
	pass#IMPORTIMPORTIMPORT import os

	if from_ali2d == False: sparx_utilities.print_begin_msg("ali2d_data")
	
	if outdir and from_ali2d == False:
		if os.path.exists(outdir):   sparx_global_def.ERROR('Output directory exists, please change the name and restart the program', "ali2d", 1)
		os.mkdir(outdir)

	xrng        = sparx_utilities.get_input_from_string(xr)
	if  yr == "-1":  yrng = xrng
	else          :  yrng = sparx_utilities.get_input_from_string(yr)
	step        = sparx_utilities.get_input_from_string(ts)

	first_ring=int(ir); last_ring=int(ou); rstep=int(rs); max_iter=int(maxit);

	if max_iter == 0:
		max_iter  = 10
		auto_stop = True
	else:
		auto_stop = False

	nima = len(data)

	if Ng == -1:
		Ng = nima
	elif Ng == -2:
		Ng = int(0.98*nima)

	nx = data[0].get_xsize()

	# default value for the last ring
	if last_ring == -1:  last_ring = nx/2-2

	if last_ring + max([max(xrng), max(yrng)]) > (nx-1) // 2:
		sparx_global_def.ERROR('Shift or radius is too large - particle crosses image boundary', "ali2d", 1)

	pass#IMPORTIMPORTIMPORT import user_functions
	user_func = sparx_user_functions.factory[user_func_name]
	# horatio active_refactoring Jy51i1EwmLD4tWZ9_00000_1
	# print_msg("Number of active images     : %s\n"%(nima))
	sparx_utilities.print_msg("Number of images            : %s\n"%(nima))
	sparx_utilities.print_msg("Output directory            : %s\n"%(outdir))
	sparx_utilities.print_msg("Inner radius                : %i\n"%(first_ring))
	sparx_utilities.print_msg("Outer radius                : %i\n"%(last_ring))
	sparx_utilities.print_msg("Ring step                   : %i\n"%(rstep))
	sparx_utilities.print_msg("X search range              : %s\n"%(xrng))
	sparx_utilities.print_msg("Y search range              : %s\n"%(yrng))
	sparx_utilities.print_msg("Translational step          : %s\n"%(step))
	sparx_utilities.print_msg("Disable checking mirror     : %s\n"%(nomirror))
	sparx_utilities.print_msg("Discrete angle used         : %d\n"%(dst))
	sparx_utilities.print_msg("Center type                 : %i\n"%(center))
	sparx_utilities.print_msg("Maximum iteration           : %i\n"%(max_iter))
	sparx_utilities.print_msg("Use Fourier variance        : %s\n"%(Fourvar))
	sparx_utilities.print_msg("Number of groups            : %d\n"%(Ng))
	sparx_utilities.print_msg("CTF correction              : %s\n"%(CTF))
	sparx_utilities.print_msg("Signal-to-Noise Ratio       : %f\n"%(snr))
	if auto_stop:
		sparx_utilities.print_msg("Stop iteration with         : criterion\n")
	else:
		sparx_utilities.print_msg("Stop iteration with         : maxit\n")
	sparx_utilities.print_msg("User function               : %s\n"%(user_func_name))
	sparx_utilities.print_msg("Using CUDA                  : %s\n"%(CUDA))
	if CUDA:
		GPUID = sparx_utilities.get_input_from_string(GPUID)
		GPUID = int(GPUID[0])
		R = CUDA_Aligner(GPUID)
		sparx_utilities.print_msg("GPU ID                      : %d\n"%(GPUID))

	if maskfile:
		pass#IMPORTIMPORTIMPORT import	types
		if type(maskfile) is bytes:
			sparx_utilities.print_msg("Maskfile                    : %s\n\n"%(maskfile))
			mask = sparx_utilities.get_image(maskfile)
		else:
			sparx_utilities.print_msg("Maskfile                    : user provided in-core mask\n\n")
			mask = maskfile
	else :
		sparx_utilities.print_msg("Maskfile                    : default, a circle with radius %i\n\n"%(last_ring))
		mask = sparx_utilities.model_circle(last_ring, nx, nx)

	cnx = nx/2+1
	cny = cnx
	mode = "F"
	if CTF:
		if data[0].get_attr_default('ctf_applied', 0) > 0:	sparx_global_def.ERROR("data cannot be ctf-applied", "ali2d", 1)
		pass#IMPORTIMPORTIMPORT from filter import filt_ctf
		flip_phases = True
		CTF = False
		#from morphology import ctf_img
		#ctf_abs_sum = EMData(nx, nx, 1, False)
		ctf_2_sum = EMAN2_cppwrap.EMData(nx, nx, 1, False)
	else:
		flip_phases = False
		ctf_2_sum = None

	if Fourvar:
		if CTF:
			pass#IMPORTIMPORTIMPORT from statistics   import varf2d
		else:
			pass#IMPORTIMPORTIMPORT from statistics   import varf

	if CUDA:
		all_ali_params = []
		all_ctf_params = []

	for im in range(nima):
		#  Subtract averages outside of mask from all input data
		st = EMAN2_cppwrap.Util.infomask(data[im], mask, False)
		data[im] -= st[0]
		if flip_phases: data[im] = sparx_filter.filt_ctf(data[im], data[im].get_attr("ctf"), binary = 1)
		"""Multiline Comment0"""
		if( random_method == "SHC" ):  data[im].set_attr('previousmax',1.0e-23)
	if CTF: 
		adw_img = EMAN2_cppwrap.Util.mult_scalar(ctf_2_sum, snr)
		EMAN2_cppwrap.Util.div_filter(adw_img, ctf_abs_sum)
		EMAN2_cppwrap.Util.mul_scalar(adw_img, float(Ng-1)/(nima-1))
		adw_img += float(nima-Ng)/(nima-1)

	# startup
	numr = sparx_alignment.Numrinit(first_ring, last_ring, rstep, mode) 	#precalculate rings
	wr = sparx_alignment.ringwe(numr, mode)

	ref_data = [mask, center, None, None]
	sx_sum = 0.0
	sy_sum = 0.0
	a0 = -1.0e22

	cs = [0.0]*2
	total_iter = 0

	if CUDA:
		pass#IMPORTIMPORTIMPORT from math import log, pi
		RING_LENGTH = 2**(int(numpy.log(2*numpy.pi*last_ring)/numpy.log(2))+1)
		NRING = 2**(int(numpy.log(last_ring)/numpy.log(2))+1)

	for N_step in range(len(xrng)):

		if CUDA:
			R.setup(len(data), nx, nx, RING_LENGTH, NRING, last_ring, step[N_step], int(xrng[N_step]/step[N_step]+0.5), int(yrng[N_step]/step[N_step]+0.5), CTF)
			for im in range(len(data)):	R.insert_image(data[im], im)
			if CTF:  R.filter_stack(all_ctf_params)

		msg = "\nX range = %5.2f   Y range = %5.2f   Step = %5.2f\n"%(xrng[N_step], yrng[N_step], step[N_step])
		sparx_utilities.print_msg(msg)
		for Iter in range(max_iter):
			total_iter += 1
			sparx_utilities.print_msg("Iteration #%4d\n"%(total_iter))
			if( total_iter ==1 and template != None):
				pass#IMPORTIMPORTIMPORT from utilities import get_im
				tavg = sparx_utilities.get_im(template)
				old_ali_params = []
				for im in range(nima):  old_ali_params.extend([0.0,0.0,0.0,0])
			else:
				if CUDA:
					ave1 = sparx_utilities.model_blank(nx, nx)
					ave2 = sparx_utilities.model_blank(nx, nx)
					R.sum_oe(all_ctf_params, all_ali_params, ave1, ave2)
					# Comment by Zhengfan Yang on 02/01/10
					# The reason for this step is that in CUDA 2-D FFT, the image is multipled by NX*NY times after
					# FFT and IFFT, so we want to decrease it such that the criterion is in line with non-CUDA version
					# However, this step is not mandatory.
					if CTF:
						ave1 /= (nx*2)**2
						ave2 /= (nx*2)**2
				else:
					ave1, ave2 = sparx_statistics.sum_oe(data, "a", CTF, EMAN2_cppwrap.EMData())  # pass empty object to prevent calculation of ctf^2
				if CTF:
					tavg_Ng = sparx_fundamentals.fft(EMAN2_cppwrap.Util.divn_filter(EMAN2_cppwrap.Util.muln_img(sparx_fundamentals.fft(EMAN2_cppwrap.Util.addn_img(ave1, ave2)), adw_img), ctf_2_sum))
					tavg = sparx_fundamentals.fft(EMAN2_cppwrap.Util.divn_filter(sparx_fundamentals.fft(EMAN2_cppwrap.Util.addn_img(ave1, ave2)), ctf_2_sum))
				else: tavg = (ave1+ave2)/nima

				if outdir:
					tavg.write_image(os.path.join(outdir, "aqc.hdf"), total_iter-1)
					if CTF:
						tavg_Ng.write_image(os.path.join(outdir, "aqc_view.hdf"), total_iter-1)
					frsc = sparx_statistics.fsc_mask(ave1, ave2, mask, 1.0, os.path.join(outdir, "resolution%03d"%(total_iter)))
				else:
					frsc = sparx_statistics.fsc_mask(ave1, ave2, mask, 1.0)

				if Fourvar:
					if CTF: vav, rvar = sparx_statistics.varf2d(data, tavg, mask, "a")
					else: vav, rvar = sparx_statistics.varf(data, tavg, mask, "a")
					tavg = sparx_fundamentals.fft(EMAN2_cppwrap.Util.divn_img(sparx_fundamentals.fft(tavg), vav))
					vav_r	= EMAN2_cppwrap.Util.pack_complex_to_real(vav)
					if outdir:
						vav_r.write_image(os.path.join(outdir, "varf.hdf"), total_iter-1)

				ref_data[2] = tavg
				ref_data[3] = frsc
				#  call user-supplied function to prepare reference image, i.e., center and filter it
				if center == -1:
					# When center = -1, which is by default, we use the average center method
					ref_data[1] = 0
					tavg, cs = user_func(ref_data)
					cs[0] = sx_sum/float(nima)
					cs[1] = sy_sum/float(nima)
					tavg = sparx_fundamentals.fshift(tavg, -cs[0], -cs[1])
					msg = "Average center x =      %10.3f        Center y       = %10.3f\n"%(cs[0], cs[1])
					sparx_utilities.print_msg(msg)
				else:
					tavg, cs = user_func(ref_data)

				# write the current filtered average
				if outdir:
					tavg.write_image(os.path.join(outdir, "aqf.hdf"), total_iter-1)

				# a0 should increase; stop algorithm when it decreases.  However, it will depend on filtration, so it is not quite right.
				a1 = tavg.cmp("dot", tavg, dict(negative = 0, mask = ref_data[0]))
				msg = "Criterion %d = %15.8e\n"%(total_iter, a1)
				sparx_utilities.print_msg(msg)
				if total_iter == len(xrng)*max_iter: break
				if a1 < a0:
					if auto_stop == True: break
				else:	a0 = a1

				if CUDA:
					old_ali_params = all_ali_params[:]
				else:
					old_ali_params = []
					for im in range(nima):
						alphan, sxn, syn, mirror, scale = sparx_utilities.get_params2D(data[im])
						old_ali_params.extend([alphan, sxn, syn, mirror])

			if Iter%4 != 0 or total_iter > max_iter*len(xrng)-10: delta = 0.0 
			else: delta = dst
			if CUDA:
				all_ali_params = R.ali2d_single_iter(tavg, all_ali_params, cs[0], cs[1], 1, delta)
				sx_sum = all_ali_params[-2]
				sy_sum = all_ali_params[-1]
				for im in range(len(data)):  all_ali_params[im*4+3] = int(all_ali_params[im*4+3])
			else:
				sx_sum, sy_sum, nope = sparx_alignment.ali2d_single_iter(data, numr, wr, cs, tavg, \
												cnx, cny, xrng[N_step], yrng[N_step], step[N_step], \
												nomirror=nomirror, mode=mode, CTF=CTF, delta=delta, \
												random_method = random_method)

			if( random_method == "SHC" and nope == nima):  break
			pixel_error = 0.0
			mirror_consistent = 0
			pixel_error_list = []
			for im in range(nima):
				if CUDA:
					alpha = all_ali_params[im*4]
					sx = all_ali_params[im*4+1]
					sy = all_ali_params[im*4+2]
					mirror = all_ali_params[im*4+3]
				else:
					alpha, sx, sy, mirror, scale = sparx_utilities.get_params2D(data[im]) 
				if old_ali_params[im*4+3] == mirror:
					this_error = sparx_pixel_error.pixel_error_2D(old_ali_params[im*4:im*4+3], [alpha, sx, sy], last_ring)
					pixel_error += this_error
					pixel_error_list.append(this_error)
					mirror_consistent += 1
			sparx_utilities.print_msg("Mirror consistent rate = %6.4f%%\n"%(float(mirror_consistent)/nima*100))
			if mirror_consistent != 0:
				sparx_utilities.print_msg("Among the mirror consistent images, average pixel error is %0.4f, their distribution is:\n"%(float(pixel_error)/float(mirror_consistent)))
				region, hist = sparx_statistics.hist_list(pixel_error_list, 20)	
				for p in range(20):
					sparx_utilities.print_msg("      %8.4f: %5d\n"%(region[p], hist[p]))
			sparx_utilities.print_msg("\n\n\n")
		if CUDA: R.finish()

	if CUDA:
		for im in range(nima):
			sparx_utilities.set_params2D(data[im], [all_ali_params[im*4], all_ali_params[im*4+1], all_ali_params[im*4+2], all_ali_params[im*4+3], 1.0])

	if outdir:
		sparx_utilities.drop_image(tavg, os.path.join(outdir, "aqfinal.hdf"))

	if from_ali2d == False: sparx_utilities.print_end_msg("ali2d_data")

def local_ali2d(stack, outdir, maskfile = None, ou = -1, br = 1.75, center = 1, eps = 0.001, maxit = 10, CTF = False, snr = 1.0, user_func_name="ref_ali2d"):
	"""
		Name
			local_ali2d - Perform local refinement of 2-D alignment of an image series
		Input
			stack: set of 2-D images in a stack file (format hdf), images have to be squares 
			maskfile: optional mask file to use internally during alignment
			outer_radius: outer radius for rotational correlation < int(nx/2)-1 
			br: brackets for the search of orientation parameters
			center_type: center the average
			epsilon: stopping criterion
			max_iter: maximum number of iterations the program will perform 
			CTF: default no CTF
			snr: Signal-to-Noise Ratio of the data
			function: name of the user-supplied function
		Output
			output_directory: directory name into which the output files will be written.
			header: the alignment parameters are stored in the headers of input files as 'alpha', 'sx', 'sy', 'mirror'
	"""
# 2D alignment using amoeba and gridding interpolation
	pass#IMPORTIMPORTIMPORT from alignment    	import kbt
	pass#IMPORTIMPORTIMPORT from utilities    	import model_circle, amoeba, compose_transform2, drop_image, get_arb_params, get_image, get_params2D, set_params2D
	pass#IMPORTIMPORTIMPORT from alignment    	import fine_2D_refinement, crit2d
	pass#IMPORTIMPORTIMPORT from statistics   	import add_oe_series, fsc_mask
	pass#IMPORTIMPORTIMPORT from filter 		import filt_from_fsc_bwt,filt_table
	pass#IMPORTIMPORTIMPORT from morphology         import ctf_2, ctf_1d
	pass#IMPORTIMPORTIMPORT import os
	pass#IMPORTIMPORTIMPORT import sys
	pass#IMPORTIMPORTIMPORT import types
	output = sys.stdout
	
	pass#IMPORTIMPORTIMPORT from utilities import print_begin_msg, print_end_msg, print_msg
	
	# create the output directory, if it does not existm
	if os.path.exists(outdir):  sparx_global_def.ERROR('Output directory exists, please change the name and restart the program', "local_ali2d", 1)
	os.mkdir(outdir)
	pass#IMPORTIMPORTIMPORT import global_def
	sparx_global_def.LOGFILE =  os.path.join(outdir, sparx_global_def.LOGFILE)
	
	max_iter  = int(maxit)
	last_ring = int(ou)

	sparx_utilities.print_begin_msg("local_ali2d")	
	
	pass#IMPORTIMPORTIMPORT import user_functions
	user_func = sparx_user_functions.factory[user_func_name]
	
	sparx_utilities.print_msg("Input stack                 : %s\n"%(stack))
	sparx_utilities.print_msg("Output directory            : %s\n"%(outdir))

	ima = EMAN2_cppwrap.EMData()
	ima.read_image(stack, 0, True)
	nx = ima.get_xsize()
	# default value for the last ring
	if (last_ring == -1): last_ring=nx//2-2

	if maskfile:
		if(type(maskfile) is bytes):
			sparx_utilities.print_msg("Maskfile                    : %s\n"%(maskfile))
			mask = sparx_utilities.get_image(maskfile)
		else:
			sparx_utilities.print_msg("Maskfile                    : user provided in-core mask\n")
			mask = maskfile
	else :
		sparx_utilities.print_msg("Maskfile                    : default, a circle with radius %i\n"%(last_ring))
		mask = sparx_utilities.model_circle(last_ring, nx, nx)
	# initialize data for the reference preparation function
	ref_data = [mask, center]

	sparx_utilities.print_msg("Outer radius                : %i\n"%(last_ring))
	sparx_utilities.print_msg("Search range                : %-5.2f\n"%(br))
	sparx_utilities.print_msg("Center type                 : %i\n"%(center))
	sparx_utilities.print_msg("Error tolerance             : %f\n"%(eps))
	sparx_utilities.print_msg("Maximum iteration           : %i\n"%(max_iter))
	sparx_utilities.print_msg("CTF correction              : %s\n"%(CTF))
	sparx_utilities.print_msg("Signal-to-Noise Ratio       : %f\n\n"%(snr))
	
	low = 0.5

	# read images
	del ima
	data = EMAN2_cppwrap.EMData.read_images(stack)
	nima = len(data)
	if(CTF):
		ctf_params = data[0].get_attr("ctf")
		data_had_ctf = data[0].get_attr("ctf_applied")
		ctm = sparx_morphology.ctf_1d(nx, ctf_params)
		lctf = len(ctm)
		ctf2 = []
		ctf2.append([0.0]*lctf)
		ctf2.append([0.0]*lctf)
		ctfb2 = [0.0]*lctf
		for im in range(nima):
			ctf_params = data[im].get_attr( "ctf" )
			ctm = sparx_morphology.ctf_2(nx, ctf_params)
			k = im%2
			for i in range(lctf):  ctf2[k][i] += ctm[i]
			if(data[im].get_attr("ctf_applied") == 0):
				st = EMAN2_cppwrap.Util.infomask(data[im], mask, False)
				data[im] -= st[0]
				pass#IMPORTIMPORTIMPORT from filter import filt_ctf
				data[im] = sparx_filter.filt_ctf(data[im], ctf_params)
				data[im].set_attr('ctf_applied', 1)
		for i in range(lctf):
			ctfb2[i] = 1.0/(ctf2[0][i] + ctf2[1][i] + 1.0/snr)
			for k in range(2):
				ctf2[k][i] = 1.0/(ctf2[k][i] + 1.0/snr)
	#calculate averages
	av1, av2 = sparx_statistics.add_oe_series(data)
	tavg = EMAN2_cppwrap.Util.addn_img(av1, av2)
	if(CTF):
		pass#IMPORTIMPORTIMPORT from filter import filt_table
		tavg = sparx_filter.filt_table(tavg, ctfb2)
	else:
		tavg /= nima

	EMAN2_cppwrap.Util.mul_img( tavg, mask)
	a0 = tavg.cmp("dot", tavg, {"negative":0, "mask":mask})
	msg = "Initial criterion = %-20.7e\n"%(a0)	
	sparx_utilities.print_msg(msg)
	# do the alignment

	for Iter in range(max_iter):
		again = False
		sparx_alignment.fine_2D_refinement(data, br, ref_data[0], tavg)

		# calculate total average using current alignment parameters
		av1, av2 = sparx_statistics.add_oe_series(data)
		if(CTF):
			tavg = sparx_filter.filt_table(EMAN2_cppwrap.Util.addn_img(av1, av2), ctfb2)
			av1  = sparx_filter.filt_table(av1, ctf2[0])
			av2  = sparx_filter.filt_table(av2, ctf2[1])
		else:
			tavg = (av1 + av2)/nima
		sparx_utilities.drop_image(tavg, os.path.join(outdir, "aqe_%03d.hdf"%(Iter)))

		frsc = sparx_statistics.fsc_mask(av1, av2, ref_data[0], 1.0, os.path.join(outdir, "dre%03d"%Iter))

		ref_data.append(tavg)
		ref_data.append(frsc)
		#  call user-supplied function to prepare reference image, i.e., center and filter it
		tavg, cs = user_func( ref_data )
		del ref_data[3]
		del ref_data[2]

		if center:
			#  apply centering parameters to shifts
			for im in range(nima):
				alpha, sx, sy, mirror, scale    = sparx_utilities.get_params2D(data[im])
				alphan, sxn, syn, scale = sparx_utilities.compose_transform2(alpha, sx, sy, 1.0, 0.0, -cs[0], -cs[1], 1.0)
				sparx_utilities.set_params2D( data[im], [alphan, sxn, syn, mirror, scale])

		# write current average
		a1 = tavg.cmp("dot", tavg, {"negative":0, "mask":ref_data[0]})
		msg = "ITERATION #%3d        criterion = %20.7e\n"%(Iter+1,a1)
		sparx_utilities.print_msg(msg)
		# write the current average
		sparx_utilities.drop_image(tavg, os.path.join(outdir, "aqf_%03d.hdf"%(Iter)))

	if(CTF and data_had_ctf == 0):
		for im in range(nima): data[im].set_attr('ctf_applied', 0)
	pass#IMPORTIMPORTIMPORT from utilities import write_headers
	sparx_utilities.write_headers(stack, data, list(range(nima)))
	sparx_utilities.print_end_msg("local_ali2d")


def mref_ali2d(stack, refim, outdir, maskfile=None, ir=1, ou=-1, rs=1, xrng=0, yrng=0, step=1, center=1, maxit=0, CTF=False, snr=1.0, user_func_name="ref_ali2d", rand_seed=1000, MPI=False):
	"""
		Name
			mref_ali2d - Perform 2-D multi-reference alignment of an image series
		Input
			stack: set of 2-D images in a stack file, images have to be squares
			refim: set of initial reference 2-D images in a stack file 
			maskfile: optional maskfile to be used in the alignment
			inner_radius: inner radius for rotational correlation > 0
			outer_radius: outer radius for rotational correlation < nx/2-1
			ring_step: step between rings in rotational correlation >0
			x_range: range for translation search in x direction, search is +/xr 
			y_range: range for translation search in y direction, search is +/yr 
			translation_step: step of translation search in both directions
			center: center the average
			max_iter: maximum number of iterations the program will perform
			CTF: if this flag is set, the program will use CTF information provided in file headers
			snr: signal-to-noise ratio of the data
			rand_seed: the seed used for generating random numbers
			MPI: whether to use MPI version
		Output
			output_directory: directory name into which the output files will be written.
			header: the alignment parameters are stored in the headers of input files as 'xform.align2d'.
	"""
# 2D multi-reference alignment using rotational ccf in polar coordinates and quadratic interpolation
	if MPI:
		mref_ali2d_MPI(stack, refim, outdir, maskfile, ir, ou, rs, xrng, yrng, step, center, maxit, CTF, snr, user_func_name, rand_seed)
		return

	pass#IMPORTIMPORTIMPORT from utilities      import   model_circle, combine_params2, inverse_transform2, drop_image, get_image
	pass#IMPORTIMPORTIMPORT from utilities	    import   center_2D, get_im, get_params2D, set_params2D
	pass#IMPORTIMPORTIMPORT from statistics     import   fsc
	pass#IMPORTIMPORTIMPORT from alignment      import   Numrinit, ringwe, fine_2D_refinement, search_range
	pass#IMPORTIMPORTIMPORT from fundamentals   import   rot_shift2D, fshift
	pass#IMPORTIMPORTIMPORT from morphology     import   ctf_2
	pass#IMPORTIMPORTIMPORT from filter         import   filt_btwl, filt_params
	pass#IMPORTIMPORTIMPORT from random         import   seed, randint
	pass#IMPORTIMPORTIMPORT import os
	pass#IMPORTIMPORTIMPORT import sys

	pass#IMPORTIMPORTIMPORT from utilities      import   print_begin_msg, print_end_msg, print_msg
	
	# create the output directory, if it does not exist
	if os.path.exists(outdir):  sparx_global_def.ERROR('Output directory exists, please change the name and restart the program', "mref_ali2d", 1)
	os.mkdir(outdir)
	pass#IMPORTIMPORTIMPORT import global_def
	sparx_global_def.LOGFILE =  os.path.join(outdir, sparx_global_def.LOGFILE)
	
	first_ring=int(ir); last_ring=int(ou); rstep=int(rs); max_iter=int(maxit)
	if max_iter == 0:
		max_iter  = 10
		auto_stop = True
	else:
		auto_stop = False

	sparx_utilities.print_begin_msg("mref_ali2d")

	sparx_utilities.print_msg("Input stack                 : %s\n"%(stack))
	sparx_utilities.print_msg("Reference stack             : %s\n"%(refim))	
	sparx_utilities.print_msg("Output directory            : %s\n"%(outdir))
	sparx_utilities.print_msg("Maskfile                    : %s\n"%(maskfile))
	sparx_utilities.print_msg("Inner radius                : %i\n"%(first_ring))

	ima = EMAN2_cppwrap.EMData()
	ima.read_image(stack, 0)
	nx = ima.get_xsize()
	# default value for the last ring
	if last_ring == -1: last_ring = nx/2-2

	sparx_utilities.print_msg("Outer radius                : %i\n"%(last_ring))
	sparx_utilities.print_msg("Ring step                   : %i\n"%(rstep))
	sparx_utilities.print_msg("X search range              : %i\n"%(xrng))
	sparx_utilities.print_msg("Y search range              : %i\n"%(yrng))
	sparx_utilities.print_msg("Translational step          : %i\n"%(step))
	sparx_utilities.print_msg("Center type                 : %i\n"%(center))
	sparx_utilities.print_msg("Maximum iteration           : %i\n"%(max_iter))
	sparx_utilities.print_msg("CTF correction              : %s\n"%(CTF))
	sparx_utilities.print_msg("Signal-to-Noise Ratio       : %f\n"%(snr))
	sparx_utilities.print_msg("Random seed                 : %i\n\n"%(rand_seed))
	sparx_utilities.print_msg("User function               : %s\n"%(user_func_name))
	output = sys.stdout

	pass#IMPORTIMPORTIMPORT import user_functions
	user_func = sparx_user_functions.factory[user_func_name]

	if maskfile:
		pass#IMPORTIMPORTIMPORT import types
		if type(maskfile) is bytes:  mask = sparx_utilities.get_image(maskfile)
		else: mask = maskfile
	else: mask = sparx_utilities.model_circle(last_ring, nx, nx)
	#  references
	refi = []
	numref = EMAN2_cppwrap.EMUtil.get_image_count(refim)
	#  CTF stuff
	if CTF:
		ctf_params = ima.get_attr("ctf")
		data_had_ctf = ima.get_attr("ctf_applied")
		ctm = sparx_morphology.ctf_2(nx, ctf_params)
		lctf = len(ctm)
		ctf2 = [[[0.0]*lctf for k in range(2)] for j in range(numref)]

	# IMAGES ARE SQUARES! center is in SPIDER convention
	cnx = nx/2+1
	cny = cnx

	mode = "F"
	#precalculate rings
	numr = sparx_alignment.Numrinit(first_ring, last_ring, rstep, mode)
	wr = sparx_alignment.ringwe(numr, mode)
	# reference images
	params = []
	#read all data
	data = EMAN2_cppwrap.EMData.read_images(stack)
	nima = len(data)
	# prepare the reference
	ima.to_zero()
	for j in range(numref):
		temp = EMAN2_cppwrap.EMData()
		temp.read_image(refim, j)
		#  eve, odd, numer of even, number of images.  After frc, totav
		refi.append([temp, ima.copy(), 0])
	random.seed(rand_seed)
	a0 = -1.
	again = True
	Iter = 0

	ref_data = [mask, center, None, None]

	while Iter < max_iter and again:
		#again = False
		ringref = []
		#print "numref",numref
		mashi = cnx-last_ring-2
		for j in range(numref):
			refi[j][0].process_inplace("normalize.mask", {"mask":mask, "no_sigma":1})
			cimage = EMAN2_cppwrap.Util.Polar2Dm(refi[j][0], cnx, cny, numr, mode)
			EMAN2_cppwrap.Util.Frngs(cimage, numr)
			EMAN2_cppwrap.Util.Applyws(cimage, numr, wr)
			ringref.append(cimage)
			# zero refi
			refi[j][0].to_zero()
			refi[j][1].to_zero()
			refi[j][2] = 0
			if CTF:
				for i in range(lctf): 
					ctf2[j][0][i] = 0.0
					ctf2[j][1][i] = 0.0
		assign = [[] for i in range(numref)]
		sx_sum = [0.0]*numref
		sy_sum = [0.0]*numref
		for im in range(nima):
			if CTF:
				ctf_params = data[im].get_attr("ctf")
				if data[im].get_attr("ctf_applied") == 0:
					st = EMAN2_cppwrap.Util.infomask(data[im], mask, False)
					data[im] -= st[0]
					pass#IMPORTIMPORTIMPORT from filter import filt_ctf
					data[im] = sparx_filter.filt_ctf(data[im], ctf_params)
					data[im].set_attr('ctf_applied', 1)
			alpha, sx, sy, mirror, scale = sparx_utilities.get_params2D(data[im])
			#  Why inverse?  07/11/2015  PAP
			alphai, sxi, syi, scalei = sparx_utilities.inverse_transform2(alpha, sx, sy)
			# normalize
			data[im].process_inplace("normalize.mask", {"mask":mask, "no_sigma":0})
			# If shifts are outside of the permissible range, reset them
			if(abs(sxi)>mashi or abs(syi)>mashi):
				sxi = 0.0
				syi = 0.0
				sparx_utilities.set_params2D(data[im],[0.0,0.0,0.0,0,1.0])
			ny = nx
			txrng = sparx_alignment.search_range(nx, last_ring, sxi, xrng, "mref_ali2d")
			txrng = [txrng[1],txrng[0]]
			tyrng = sparx_alignment.search_range(ny, last_ring, syi, yrng, "mref_ali2d")
			tyrng = [tyrng[1],tyrng[0]]
			# align current image to the reference
			[angt, sxst, syst, mirrort, xiref, peakt] = EMAN2_cppwrap.Util.multiref_polar_ali_2d(data[im], 
				ringref, txrng, tyrng, step, mode, numr, cnx+sxi, cny+syi)
			iref = int(xiref)
			# combine parameters and set them to the header, ignore previous angle and mirror
			[alphan, sxn, syn, mn] = sparx_utilities.combine_params2(0.0, -sxi, -syi, 0, angt, sxst, syst, int(mirrort))
			sparx_utilities.set_params2D(data[im], [alphan, sxn, syn, int(mn), scale])
			if mn == 0: sx_sum[iref] += sxn
			else: sx_sum[iref] -= sxn
			sy_sum[iref] += syn
			data[im].set_attr('assign', iref)
			# apply current parameters and add to the average
			temp = sparx_fundamentals.rot_shift2D(data[im], alphan, sxn, syn, mn)
			it = im%2
			EMAN2_cppwrap.Util.add_img(refi[iref][it], temp)
			if CTF:
				ctm = sparx_morphology.ctf_2(nx, ctf_params)
				for i in range(lctf):  ctf2[iref][it][i] += ctm[i]
			assign[iref].append(im)
			refi[iref][2] += 1
		del ringref
		if again:
			a1 = 0.0
			for j in range(numref):
				msg = "   group #%3d   number of particles = %7d\n"%(j, refi[j][2])
				sparx_utilities.print_msg(msg)
				if refi[j][2] < 4:
					#ERROR("One of the references vanished","mref_ali2d",1)
					#  if vanished, put a random image there
					assign[j] = []
					assign[j].append(random.randint(0, nima-1))
					refi[j][0] = data[assign[j][0]].copy()
				else:
					max_inter = 0  # switch off fine refi.
					br = 1.75
					#  the loop has to 
					for INter in range(max_inter+1):
						# Calculate averages at least ones, meaning even if no within group refinement was requested
						if CTF:
							for i in range(lctf):  ctm[i] = 1.0 / (ctf2[j][0][i] + 1.0/snr)
							pass#IMPORTIMPORTIMPORT from filter import filt_table
							av1 = sparx_filter.filt_table(refi[j][0], ctm)
							for i in range(lctf):  ctm[i] = 1.0 / (ctf2[j][1][i] + 1.0/snr)
							av2 = sparx_filter.filt_table(refi[j][1], ctm)
							frsc = sparx_statistics.fsc(av1, av2, 1.0, os.path.join(outdir,"drm_%03d_%04d.txt"%(Iter, j)))
							#Now the total average
							for i in range(lctf):  ctm[i] = 1.0 / (ctf2[j][0][i] + ctf2[j][1][i] + 1.0/snr)
							refi[j][0] = sparx_filter.filt_table(EMAN2_cppwrap.Util.addn_img(refi[j][0], refi[j][1]), ctm)
						else:
							frsc = sparx_statistics.fsc(refi[j][0], refi[j][1], 1.0, os.path.join(outdir,"drm_%03d_%04d.txt"%(Iter, j)))
							EMAN2_cppwrap.Util.add_img(refi[j][0], refi[j][1])
							EMAN2_cppwrap.Util.mul_scalar(refi[j][0], 1.0/float(refi[j][2]))
							
						ref_data[2] = refi[j][0]
						ref_data[3] = frsc						
						refi[j][0], cs = user_func(ref_data)
						if center == -1:
							cs[0] = sx_sum[j]/len(assign[j])
							cs[1] = sy_sum[j]/len(assign[j])
							refi[j][0] = sparx_fundamentals.fshift(refi[j][0], -cs[0], -cs[1])
						for i in range(len(assign[j])):
							im = assign[j][i]
							alpha, sx, sy, mirror, scale =  sparx_utilities.get_params2D(data[im])
							alphan, sxn, syn, mirrorn = sparx_utilities.combine_params2(alpha, sx, sy, mirror, 0.0, -cs[0], -cs[1], 0)
							sparx_utilities.set_params2D(data[im], [alphan, sxn, syn, int(mirrorn), scale])
						# refine images within the group
						#  Do the refinement only if max_inter>0, but skip it for the last iteration.
						if INter < max_inter:
							sparx_alignment.fine_2D_refinement(data, br, mask, refi[j][0], j)
							#  Calculate updated average
							refi[j][0].to_zero()
							refi[j][1].to_zero()
							for i in range(len(assign[j])):
								im = assign[j][i]
								alpha, sx, sy, mirror, scale = sparx_utilities.get_params2D(data[im])
								# apply current parameters and add to the average
								temp = sparx_fundamentals.rot_shift2D(data[im], alpha, sx, sy, mn)
								it = im%2
								EMAN2_cppwrap.Util.add_img(refi[j][it], temp)

				# write the current average
				TMP = []
				for i_tmp in range(len(assign[j])):  TMP.append(float(assign[j][i_tmp]))
				TMP.sort()
				refi[j][0].set_attr_dict({'ave_n': refi[j][2], 'members': TMP })
				del TMP
				# replace the name of the stack with reference with the current one
				newrefim = os.path.join(outdir,"aqm%03d.hdf"%Iter)
				refi[j][0].write_image(newrefim, j)
				a1 += refi[j][0].cmp("dot", refi[j][0], {"negative":0, "mask":mask})
			Iter += 1
			msg = "ITERATION #%3d        criterion = %20.7e\n"%(Iter,a1)
			sparx_utilities.print_msg(msg)
			if a1 < a0:
				if auto_stop == True:	break
			else:	a0 = a1

	newrefim = os.path.join(outdir,"multi_ref.hdf")
	for j in range(numref):  refi[j][0].write_image(newrefim, j)
	if CTF:
		if data_had_ctf == 0:
			for im in range(nima): data[im].set_attr('ctf_applied', 0)
	pass#IMPORTIMPORTIMPORT from utilities import write_headers
	sparx_utilities.write_headers(stack, data, list(range(nima)))
	sparx_utilities.print_end_msg("mref_ali2d")


def mref_ali2d_MPI(stack, refim, outdir, maskfile = None, ir=1, ou=-1, rs=1, xrng=0, yrng=0, step=1, center=1, maxit=10, CTF=False, snr=1.0, user_func_name="ref_ali2d", rand_seed=1000):
# 2D multi-reference alignment using rotational ccf in polar coordinates and quadratic interpolation

	pass#IMPORTIMPORTIMPORT from utilities      import   model_circle, combine_params2, inverse_transform2, drop_image, get_image, get_im
	pass#IMPORTIMPORTIMPORT from utilities      import   reduce_EMData_to_root, bcast_EMData_to_all, bcast_number_to_all
	pass#IMPORTIMPORTIMPORT from utilities      import   send_attr_dict
	pass#IMPORTIMPORTIMPORT from utilities	    import   center_2D
	pass#IMPORTIMPORTIMPORT from statistics     import   fsc_mask
	pass#IMPORTIMPORTIMPORT from alignment      import   Numrinit, ringwe, search_range
	pass#IMPORTIMPORTIMPORT from fundamentals   import   rot_shift2D, fshift
	pass#IMPORTIMPORTIMPORT from utilities      import   get_params2D, set_params2D
	pass#IMPORTIMPORTIMPORT from random         import   seed, randint
	pass#IMPORTIMPORTIMPORT from morphology     import   ctf_2
	pass#IMPORTIMPORTIMPORT from filter         import   filt_btwl, filt_params
	pass#IMPORTIMPORTIMPORT from numpy          import   reshape, shape
	pass#IMPORTIMPORTIMPORT from utilities      import   print_msg, print_begin_msg, print_end_msg
	pass#IMPORTIMPORTIMPORT import os
	pass#IMPORTIMPORTIMPORT import sys
	pass#IMPORTIMPORTIMPORT from mpi 	  import mpi_comm_size, mpi_comm_rank, MPI_COMM_WORLD
	pass#IMPORTIMPORTIMPORT from mpi 	  import mpi_reduce, mpi_bcast, mpi_barrier, mpi_recv, mpi_send
	pass#IMPORTIMPORTIMPORT from mpi 	  import MPI_SUM, MPI_FLOAT, MPI_INT

	number_of_proc = mpi.mpi_comm_size(mpi.MPI_COMM_WORLD)
	myid = mpi.mpi_comm_rank(mpi.MPI_COMM_WORLD)
	main_node = 0
	
	# create the output directory, if it does not exist
	if os.path.exists(outdir):  sparx_global_def.ERROR('Output directory exists, please change the name and restart the program', "mref_ali2d_MPI ", 1, myid)
	mpi.mpi_barrier(mpi.MPI_COMM_WORLD)

	if myid == main_node:
		os.mkdir(outdir)
		pass#IMPORTIMPORTIMPORT import global_def
		sparx_global_def.LOGFILE =  os.path.join(outdir, sparx_global_def.LOGFILE)
		sparx_utilities.print_begin_msg("mref_ali2d_MPI")

	nima = EMAN2_cppwrap.EMUtil.get_image_count(stack)
	
	image_start, image_end = MPI_start_end(nima, number_of_proc, myid)

	nima = EMAN2_cppwrap.EMUtil.get_image_count(stack)
	ima = EMAN2_cppwrap.EMData()
	ima.read_image(stack, image_start)

	first_ring=int(ir); last_ring=int(ou); rstep=int(rs); max_iter=int(maxit)

	if max_iter == 0:
		max_iter  = 10
		auto_stop = True
	else:
		auto_stop = False

	if myid == main_node:
		sparx_utilities.print_msg("Input stack                 : %s\n"%(stack))
		sparx_utilities.print_msg("Reference stack             : %s\n"%(refim))	
		sparx_utilities.print_msg("Output directory            : %s\n"%(outdir))
		sparx_utilities.print_msg("Maskfile                    : %s\n"%(maskfile))
		sparx_utilities.print_msg("Inner radius                : %i\n"%(first_ring))

	nx = ima.get_xsize()
	# default value for the last ring
	if last_ring == -1: last_ring=nx/2-2
	
	if myid == main_node:
		sparx_utilities.print_msg("Outer radius                : %i\n"%(last_ring))
		sparx_utilities.print_msg("Ring step                   : %i\n"%(rstep))
		sparx_utilities.print_msg("X search range              : %f\n"%(xrng))
		sparx_utilities.print_msg("Y search range              : %f\n"%(yrng))
		sparx_utilities.print_msg("Translational step          : %f\n"%(step))
		sparx_utilities.print_msg("Center type                 : %i\n"%(center))
		sparx_utilities.print_msg("Maximum iteration           : %i\n"%(max_iter))
		sparx_utilities.print_msg("CTF correction              : %s\n"%(CTF))
		sparx_utilities.print_msg("Signal-to-Noise Ratio       : %f\n"%(snr))
		sparx_utilities.print_msg("Random seed                 : %i\n\n"%(rand_seed))	
		sparx_utilities.print_msg("User function               : %s\n"%(user_func_name))
	pass#IMPORTIMPORTIMPORT import user_functions
	user_func = sparx_user_functions.factory[user_func_name]

	if maskfile:
		pass#IMPORTIMPORTIMPORT import  types
		if type(maskfile) is bytes:  mask = sparx_utilities.get_image(maskfile)
		else: mask = maskfile
	else : mask = sparx_utilities.model_circle(last_ring, nx, nx)
	#  references, do them on all processors...
	refi = []
	numref = EMAN2_cppwrap.EMUtil.get_image_count(refim)
	#  CTF stuff
	if CTF:
		ctf_params = ima.get_attr("ctf")
		data_had_ctf = ima.get_attr("ctf_applied")
		ctm = sparx_morphology.ctf_2(nx, ctf_params)
		lctf = len(ctm)

	# IMAGES ARE SQUARES! center is in SPIDER convention
	cnx = nx/2+1
	cny = cnx

	mode = "F"
	#precalculate rings
	numr = sparx_alignment.Numrinit(first_ring, last_ring, rstep, mode)
	wr = sparx_alignment.ringwe(numr, mode)
	# reference images
	again = True
	params = []
	# prepare reference images on all nodes
	ima.to_zero()
	for j in range(numref):
		#  even, odd, numer of even, number of images.  After frc, totav
		refi.append([sparx_utilities.get_im(refim,j), ima.copy(), 0])
	#  for each node read its share of data
	data = EMAN2_cppwrap.EMData.read_images(stack, list(range(image_start, image_end)))
	for im in range(image_start, image_end):
		data[im-image_start].set_attr('ID', im)
		if CTF:
			ctf_params = data[im-image_start].get_attr( "ctf" )
			if data[im-image_start].get_attr("ctf_applied") == 0:
				st = EMAN2_cppwrap.Util.infomask(data[im-image_start], mask, False)
				data[im-image_start] -= st[0]
				pass#IMPORTIMPORTIMPORT from filter import filt_ctf
				data[im-image_start] = sparx_filter.filt_ctf(data[im-image_start], ctf_params)
				data[im-image_start].set_attr('ctf_applied', 1)
	if myid == main_node:  random.seed(rand_seed)

	a0 = -1.0
	again = True
	Iter = 0

	ref_data = [mask, center, None, None]

	while Iter < max_iter and again:
		ringref = []
		mashi = cnx-last_ring-2
		for j in range(numref):
			refi[j][0].process_inplace("normalize.mask", {"mask":mask, "no_sigma":1}) # normalize reference images to N(0,1)
			cimage = EMAN2_cppwrap.Util.Polar2Dm(refi[j][0] , cnx, cny, numr, mode)
			EMAN2_cppwrap.Util.Frngs(cimage, numr)
			EMAN2_cppwrap.Util.Applyws(cimage, numr, wr)
			ringref.append(cimage)
			# zero refi
			refi[j][0].to_zero()
			refi[j][1].to_zero()
			refi[j][2] = 0
		if CTF: ctf2 = [[[0.0]*lctf for k in range(2)] for j in range(numref)]
		assign = [[] for i in range(numref)]
		# begin MPI section
		for im in range(image_start, image_end):
			alpha, sx, sy, mirror, scale = sparx_utilities.get_params2D(data[im-image_start])
			#  Why inverse?  07/11/2015 PAP
			alphai, sxi, syi, scalei = sparx_utilities.inverse_transform2(alpha, sx, sy)
			# normalize
			data[im-image_start].process_inplace("normalize.mask", {"mask":mask, "no_sigma":0}) # subtract average under the mask
			# If shifts are outside of the permissible range, reset them
			if(abs(sxi)>mashi or abs(syi)>mashi):
				sxi = 0.0
				syi = 0.0
				sparx_utilities.set_params2D(data[im-image_start],[0.0,0.0,0.0,0,1.0])
			ny = nx
			txrng = sparx_alignment.search_range(nx, last_ring, sxi, xrng, "mref_ali2d_MPI")
			txrng = [txrng[1],txrng[0]]
			tyrng = sparx_alignment.search_range(ny, last_ring, syi, yrng, "mref_ali2d_MPI")
			tyrng = [tyrng[1],tyrng[0]]
			# align current image to the reference
			[angt, sxst, syst, mirrort, xiref, peakt] = EMAN2_cppwrap.Util.multiref_polar_ali_2d(data[im-image_start], 
				ringref, txrng, tyrng, step, mode, numr, cnx+sxi, cny+syi)
			iref = int(xiref)
			# combine parameters and set them to the header, ignore previous angle and mirror
			[alphan, sxn, syn, mn] = sparx_utilities.combine_params2(0.0, -sxi, -syi, 0, angt, sxst, syst, (int)(mirrort))
			sparx_utilities.set_params2D(data[im-image_start], [alphan, sxn, syn, int(mn), scale])
			data[im-image_start].set_attr('assign',iref)
			# apply current parameters and add to the average
			temp = sparx_fundamentals.rot_shift2D(data[im-image_start], alphan, sxn, syn, mn)
			it = im%2
			EMAN2_cppwrap.Util.add_img( refi[iref][it], temp)
			assign[iref].append(im)
			if CTF:
				#  I wonder whether params are still there....
				ctf_params = data[im-image_start].get_attr("ctf")
				ctm = sparx_morphology.ctf_2(nx, ctf_params)
				for i in range(lctf):  ctf2[iref][it][i] += ctm[i]
			#assign[im] = iref
			refi[iref][2] += 1.0
		del ringref
		# end MPI section, bring partial things together, calculate new reference images, broadcast them back
		if CTF:
		# bring ctf2 together on main node
			s = numpy.shape(ctf2)
			ctf2  = mpi.mpi_reduce(ctf2, 2*lctf*numref, mpi.MPI_FLOAT, mpi.MPI_SUM, main_node, mpi.MPI_COMM_WORLD)
			if myid == main_node: ctf2 = numpy.reshape(ctf2, s)
		for j in range(numref):
			sparx_utilities.reduce_EMData_to_root(refi[j][0], myid, main_node)
			sparx_utilities.reduce_EMData_to_root(refi[j][1], myid, main_node)
			refi[j][2] = mpi.mpi_reduce(refi[j][2], 1, mpi.MPI_FLOAT, mpi.MPI_SUM, main_node, mpi.MPI_COMM_WORLD)
			if(myid == main_node): refi[j][2] = int(refi[j][2][0])
		# gather assignements
		for j in range(numref):
			if myid == main_node:
				for n in range(number_of_proc):
					if n != main_node:
						ln =  mpi.mpi_recv(1, mpi.MPI_INT, n, sparx_global_def.SPARX_MPI_TAG_UNIVERSAL, mpi.MPI_COMM_WORLD)
						lis = mpi.mpi_recv(ln[0], mpi.MPI_INT, n, sparx_global_def.SPARX_MPI_TAG_UNIVERSAL, mpi.MPI_COMM_WORLD)
						for l in range(ln[0]): assign[j].append(int(lis[l]))
			else:
				mpi.mpi_send(len(assign[j]), 1, mpi.MPI_INT, main_node, sparx_global_def.SPARX_MPI_TAG_UNIVERSAL, mpi.MPI_COMM_WORLD)
				mpi.mpi_send(assign[j], len(assign[j]), mpi.MPI_INT, main_node, sparx_global_def.SPARX_MPI_TAG_UNIVERSAL, mpi.MPI_COMM_WORLD)

		if myid == main_node:
			# replace the name of the stack with reference with the current one
			refim = os.path.join(outdir,"aqm%03d.hdf"%Iter)
			a1 = 0.0
			ave_fsc = []
			for j in range(numref):
				if refi[j][2] < 4:
					#ERROR("One of the references vanished","mref_ali2d_MPI",1)
					#  if vanished, put a random image (only from main node!) there
					assign[j] = []
					assign[j].append( random.randint(image_start, image_end-1) - image_start )
					refi[j][0] = data[assign[j][0]].copy()
					#print 'ERROR', j
				else:
					if CTF:
						for i in range(lctf):  ctm[i] = 1.0 / (ctf2[j][0][i] + 1.0/snr)
						pass#IMPORTIMPORTIMPORT from filter import filt_table
						av1 = sparx_filter.filt_table( refi[j][0], ctm)
						for i in range(lctf):  ctm[i] = 1.0 / (ctf2[j][1][i] + 1.0/snr)
						av2 = sparx_filter.filt_table( refi[j][1], ctm)
						pass#IMPORTIMPORTIMPORT from statistics import fsc
						#frsc = fsc_mask(av1, av2, mask, 1.0, os.path.join(outdir,"drm%03d%04d"%(Iter, j)))
						frsc = sparx_statistics.fsc(av1, av2, 1.0, os.path.join(outdir,"drm%03d%04d.txt"%(Iter, j)))
						#Now the total average
						for i in range(lctf):  ctm[i] = 1.0 / (ctf2[j][0][i] + ctf2[j][1][i] + 1.0/snr)
						refi[j][0] = sparx_filter.filt_table( EMAN2_cppwrap.Util.addn_img( refi[j][0], refi[j][1] ), ctm)
					else:
						#frsc = fsc_mask(refi[j][0], refi[j][1], mask, 1.0, os.path.join(outdir,"drm%03d%04d"%(Iter, j)))
						pass#IMPORTIMPORTIMPORT from statistics import fsc
						frsc = sparx_statistics.fsc(refi[j][0], refi[j][1], 1.0, os.path.join(outdir,"drm%03d%04d.txt"%(Iter,j)))
						EMAN2_cppwrap.Util.add_img( refi[j][0], refi[j][1] )
						EMAN2_cppwrap.Util.mul_scalar( refi[j][0], 1.0/float(refi[j][2]) )
				        	
					if ave_fsc == []:
						for i in range(len(frsc[1])): ave_fsc.append(frsc[1][i])
						c_fsc = 1
					else:
						for i in range(len(frsc[1])): ave_fsc[i] += frsc[1][i]
						c_fsc += 1
					#print 'OK', j, len(frsc[1]), frsc[1][0:5], ave_fsc[0:5]			


			#print 'sum', sum(ave_fsc)
			if sum(ave_fsc) != 0:		
				for i in range(len(ave_fsc)):
					ave_fsc[i] /= float(c_fsc)
					frsc[1][i]  = ave_fsc[i]
			
			for j in range(numref):
				ref_data[2]    = refi[j][0]
				ref_data[3]    = frsc
				refi[j][0], cs = user_func(ref_data)	

				# write the current average
				TMP = []
				for i_tmp in range(len(assign[j])): TMP.append(float(assign[j][i_tmp]))
				TMP.sort()
				refi[j][0].set_attr_dict({'ave_n': refi[j][2],  'members': TMP })
				del TMP
				refi[j][0].process_inplace("normalize.mask", {"mask":mask, "no_sigma":1})
				refi[j][0].write_image(refim, j)
				a1 += refi[j][0].cmp("dot", refi[j][0], {"negative":0, "mask":mask})
				# send refi[j][0]  back!

			Iter += 1
			msg = "ITERATION #%3d        criterion = %20.7e\n\n"%(Iter,a1)
			sparx_utilities.print_msg(msg)
			for j in range(numref):
				msg = "   group #%3d   number of particles = %7d\n"%(j, refi[j][2])
				sparx_utilities.print_msg(msg)
			
			if a1 < a0:
				if (auto_stop == True):	again = False
			else:	a0 = a1
		#again = mpi_bcast(again, 1, MPI_INT, main_node, MPI_COMM_WORLD)
		Iter  = sparx_utilities.bcast_number_to_all(Iter, main_node)
		if CTF:  del  ctf2
		if again:
			for j in range(numref):
				sparx_utilities.bcast_EMData_to_all(refi[j][0], myid, main_node)

	#  clean up
	del assign
	# write out headers  and STOP, under MPI writing has to be done sequentially
	mpi.mpi_barrier(mpi.MPI_COMM_WORLD)
	if CTF and data_had_ctf == 0:
		for im in range(len(data)): data[im].set_attr('ctf_applied', 0)
	par_str = ['xform.align2d', 'assign', 'ID']
	if myid == main_node:
		pass#IMPORTIMPORTIMPORT from utilities import file_type
		if(sparx_utilities.file_type(stack) == "bdb"):
			pass#IMPORTIMPORTIMPORT from utilities import recv_attr_dict_bdb
			sparx_utilities.recv_attr_dict_bdb(main_node, stack, data, par_str, image_start, image_end, number_of_proc)
		else:
			pass#IMPORTIMPORTIMPORT from utilities import recv_attr_dict
			sparx_utilities.recv_attr_dict(main_node, stack, data, par_str, image_start, image_end, number_of_proc)
	else:           sparx_utilities.send_attr_dict(main_node, data, par_str, image_start, image_end)

	if myid == main_node:
		newrefim = os.path.join(outdir, "multi_ref.hdf")
		for j in range(numref):
			refi[j][0].write_image(newrefim, j)
		sparx_utilities.print_end_msg("mref_ali2d_MPI")


def ali2d_ra(stack, maskfile = None, ir = 1, ou = -1, rs = 1, maxit = 10, check_mirror = False, CTF = False, rand_seed = 1000):
# 2D rotational alignment using ccf in polar coordinates

	pass#IMPORTIMPORTIMPORT from utilities    import model_circle, compose_transform2, combine_params2, drop_image, get_im, get_arb_params, get_params2D, set_params2D
	pass#IMPORTIMPORTIMPORT from alignment    import Numrinit, ringwe, ang_n
	pass#IMPORTIMPORTIMPORT from statistics   import kmn, kmn_ctf
	pass#IMPORTIMPORTIMPORT from morphology   import ctf_2
	pass#IMPORTIMPORTIMPORT from statistics   import add_series
	pass#IMPORTIMPORTIMPORT from applications import transform2d
	pass#IMPORTIMPORTIMPORT from random       import random
	pass#IMPORTIMPORTIMPORT from utilities    import print_begin_msg, print_end_msg, print_msg

	sparx_utilities.print_begin_msg("ali2d_ra")

	first_ring=int(ir); last_ring=int(ou); rstep=int(rs); max_iter=int(maxit); 

	sparx_utilities.print_msg("Input stack                 : %s\n"%(stack))
	sparx_utilities.print_msg("Maskfile                    : %s\n"%(maskfile))
	sparx_utilities.print_msg("Inner radius                : %i\n"%(first_ring))

	temp = EMAN2_cppwrap.EMData()
	temp.read_image(stack, 0)
	nx = temp.get_xsize()
	ny = nx
	# default value for the last ring
	if (last_ring == -1): last_ring=nx//2-2

	sparx_utilities.print_msg("Outer radius                : %i\n"%(last_ring))
	sparx_utilities.print_msg("Ring step                   : %i\n"%(rstep))
	sparx_utilities.print_msg("Maximum iteration           : %i\n"%(max_iter))
	sparx_utilities.print_msg("Consider Mirror             : %s\n"%(check_mirror))	
	sparx_utilities.print_msg("CTF correction              : %s\n"%(CTF))
	sparx_utilities.print_msg("Random seed                 : %i\n\n"%(rand_seed))
	
	# consider mirror
	# somebody dedicated could write a version with option "H" for half rings that would work for ACF functions.
	mode = "F"

	nima = EMAN2_cppwrap.EMUtil.get_image_count(stack)

	# precalculate rings
	numr = sparx_alignment.Numrinit(first_ring, last_ring, rstep, mode)
	wr   = sparx_alignment.ringwe(numr, mode)
	lnumr = numr[len(numr)-1]
	# prepare 2-D mask for normalization
	if maskfile:
		pass#IMPORTIMPORTIMPORT import  types
		if(type(maskfile) is bytes):  mask2D = sparx_utilities.get_im(maskfile)
		else: mask2D = maskfile
	else : mask2D = sparx_utilities.model_circle(last_ring, nx, nx)
	if (first_ring > 0):
		tave = sparx_utilities.model_circle(first_ring-1, nx, ny)
		mask2D -= tave

	# read images and resample them into polar coordinates
	data = []
	#  center is in SPIDER convention
	cnx = int(nx/2) + 1
	cny = int(ny/2) + 1

	if(CTF):
		# for rotational alignment with CTF correction, we have more complicated strategies.
		#   We need two series of images in polar coordinates: multiplied by the ctf, and 'ctf-corrected', i.e., also divided by ctf^2+1/snr
		#   Thus, is alignment, the reference is always computed from the second series, while the first is used for alignment 
		parnames = ["Pixel_size", "defocus", "voltage", "Cs", "amp_contrast", "B_factor",  "ctf_applied"]
		#                        0                1              2            3              4             5                6
		for im in range(nima):
			if(im>0):
				temp = EMAN2_cppwrap.EMData()
				temp.read_image(stack, im)
			ctf_params = temp.get_attr( "ctf" )
			if(im == 0):  data_had_ctf = temp.get_attr('ctf_applied')
			ctf = sparx_morphology.ctf_2(nx, ctf_params)
			if(im == 0):
				lctf = len(ctf)
				ctf2 = [0.0]*lctf
			for i in range(lctf):  ctf2[i] += ctf[i]
		del ctf
		ref_data = []
		for im in range(nima):
			temp = EMAN2_cppwrap.EMData()
			temp.read_image(stack, im)
			st = EMAN2_cppwrap.Util.infomask(temp, mask2D, False)
			ctf_params = temp.get_attr( "ctf" )
			temp -= st[0]
			if(temp.get_attr("ctf_applied") == 0):
				pass#IMPORTIMPORTIMPORT from filter import filt_ctf
				temp = sparx_filter.filt_ctf(temp, ctf_params)
				temp.set_attr('ctf_applied', 1)
			pass#IMPORTIMPORTIMPORT from filter       import filt_table
			refc = sparx_filter.filt_table(temp, ctf2)
			
			alpha_original, sx, sy, miri, scale = sparx_utilities.get_params2D(temp)
			#tempg = prepg(temp, kb)
			#cimage = Util.Polar2Dmi(tempg, cnx+sx, cny+sy, numr, mode, kb)
			alphan, sxn, syn, mir = sparx_utilities.combine_params2(0, -sx, -sy, 0, -alpha_original, 0,0,0)

			nring = len(numr)/3
			inr = numr[3*(nring-1)]
			#here the centers of the image cny and cnx use Spider convention which means the index of the image array starts from 1 to nx (ny). 02-24-2015
			if ((inr+int(cny+syn) <= ny-1 and -inr + int(cny+syn) >=1) and (inr+int(cnx+sxn) <= nx-1 and -inr + int(cnx+sxn) >=1)):
				cimage = EMAN2_cppwrap.Util.Polar2Dm(temp, cnx+sxn, cny+syn, numr, mode)
				EMAN2_cppwrap.Util.Frngs(cimage, numr)
				data.append(cimage)
				#  Here alpha is position of the peak (i.e., starts from 1), just put as a place holder, will be determined in kmn
				data[im].set_attr_dict({'alpha':1.0, 'alpha_original':alpha_original, 'sx':sx, 'sy':sy, 'mirror': 0})
				cimage = EMAN2_cppwrap.Util.Polar2Dm(refc, cnx+sxn, cny+syn, numr, mode)
				EMAN2_cppwrap.Util.Frngs(cimage, numr)
				#  We do not need any attributes for ref_data, as they are going to be taken from data
				ref_data.append(cimage)
			else:
				sparx_global_def.ERROR("ali2d_ra","Particle radius given too large for particle shifts found in the header",1) 
	


		del temp
		del refc
		del mask2D
		del ctf2
		sparx_statistics.kmn_ctf(data, ref_data, numr, wr, check_mirror, max_iter, rand_seed)
	else:
		for im in range(nima):
			if (im>0):
				temp = EMAN2_cppwrap.EMData()
				temp.read_image(stack, im)
			alpha_original, sx, sy, miri, scale = sparx_utilities.get_params2D(temp)
			temp.process_inplace("normalize.mask", {"mask":mask2D, "no_sigma":0})
			#tempg = prepg(temp, kb)
			#cimage = Util.Polar2Dmi(tempg, cnx+sx, cny+sy, numr, mode, kb)
			alphan, sxn, syn, mir = sparx_utilities.combine_params2(0.0, -sx, -sy, 0, -alpha_original, 0.0, 0.0, 0)
			nring = len(numr)/3
			inr = numr[3*(nring-1)]
			#here the centers of the image cny and cnx use Spider convention which means the index of the image array starts from 1 to nx (ny). 02-24-2015
			if ((inr+int(cny+syn) <= ny-1 and -inr + int(cny+syn) >=1) and (inr+int(cnx+sxn) <= nx-1 and -inr + int(cnx+sxn) >=1)):
				cimage = EMAN2_cppwrap.Util.Polar2Dm(temp, cnx+sxn, cny+syn, numr, mode)
				EMAN2_cppwrap.Util.Frngs(cimage, numr)
				data.append(cimage)
				#  Here alpha is position of the peak (i.e., starts from 1), just put as a place holder, will be determined in kmn
				data[im].set_attr_dict({'alpha':1.0, 'alpha_original':alpha_original, 'sx':sx, 'sy':sy, 'mirror':0})
			else: 
				sparx_global_def.ERROR("ali2d_ra","Particle radius given too large for particle shifts found in the header",1)
			
		del temp
		#del tempg
		del mask2D
		sparx_statistics.kmn(data, numr, wr, check_mirror, max_iter, rand_seed)
	#  write out the alignment parameters to headers
	pass#IMPORTIMPORTIMPORT from utilities import write_header, file_type
	temp = EMAN2_cppwrap.EMData()
	for im in range(nima):
		alpha_original   = data[im].get_attr('alpha_original')
		alpha = data[im].get_attr('alpha')
		sx    =  data[im].get_attr('sx')
		sy    =  data[im].get_attr('sy')
		mirror =  data[im].get_attr('mirror')
		alpha = sparx_alignment.ang_n(alpha, mode, lnumr)
		#  here the original angle is irrelevant, used only to determine proper shifts
		alpha_original_n, sxn, syn, mir = sparx_utilities.combine_params2(0, -sx, -sy, 0, -alpha_original, 0,0,0)
		alphan, sxn, syn, mir           = sparx_utilities.combine_params2(0, -sxn, -syn, 0, alpha, 0,0, mirror)
		temp.read_image(stack, im, True)
		if(CTF and data_had_ctf == 0):   temp.set_attr('ctf_applied', 0)
		sparx_utilities.set_params2D(temp, [alphan, sxn, syn, mir, 1.0])
		sparx_utilities.write_header(stack, temp, im)
		#temp.write_image(stack, im, EMUtil.ImageType.IMAGE_HDF, True)
	sparx_utilities.print_end_msg("ali2d_ra")

def ali2d_rag(stack, maskfile = None, ir = 1, ou = -1, rs = 1, maxit = 10, check_mirror = False, CTF = False, rand_seed = 1000):
# 2D rotational alignment using ccf in polar coordinates and gridding-based interpolation

	pass#IMPORTIMPORTIMPORT from utilities    import model_circle, compose_transform2, combine_params2, drop_image, get_im, get_arb_params
	pass#IMPORTIMPORTIMPORT from alignment    import Numrinit, ringwe, ang_n
	pass#IMPORTIMPORTIMPORT from statistics   import kmn_g, kmn_ctf
	pass#IMPORTIMPORTIMPORT from morphology   import ctf_2
	pass#IMPORTIMPORTIMPORT from statistics   import add_series
	pass#IMPORTIMPORTIMPORT from applications import transform2d
	pass#IMPORTIMPORTIMPORT from random       import random
	pass#IMPORTIMPORTIMPORT from utilities    import print_begin_msg, print_end_msg, print_msg
	pass#IMPORTIMPORTIMPORT from fundamentals import prepi

	sparx_utilities.print_begin_msg("ali2d_rag")

	first_ring=int(ir); last_ring=int(ou); rstep=int(rs); max_iter=int(maxit); 

	sparx_utilities.print_msg("Input stack                 : %s\n"%(stack))
	sparx_utilities.print_msg("Maskfile                    : %s\n"%(maskfile))
	sparx_utilities.print_msg("Inner radius                : %i\n"%(first_ring))

	temp = EMAN2_cppwrap.EMData()
	temp.read_image(stack, 0)
	nx = temp.get_xsize()
	ny = nx
	# default value for the last ring
	if (last_ring == -1): last_ring=nx//2-2

	sparx_utilities.print_msg("Outer radius                : %i\n"%(last_ring))
	sparx_utilities.print_msg("Ring step                   : %i\n"%(rstep))
	sparx_utilities.print_msg("Maximum iteration           : %i\n"%(max_iter))
	sparx_utilities.print_msg("Consider Mirror             : %s\n"%(check_mirror))	
	sparx_utilities.print_msg("CTF correction              : %s\n"%(CTF))
	sparx_utilities.print_msg("Random seed                 : %i\n\n"%(rand_seed))
	
	# consider mirror
	# somebody dedicated could write a version with option "H" for half rings that would work for ACF functions.
	mode = "F"

	nima = EMAN2_cppwrap.EMUtil.get_image_count(stack)

	# precalculate rings
	numr = sparx_alignment.Numrinit(first_ring, last_ring, rstep, mode)
	wr   = sparx_alignment.ringwe(numr, mode)
	maxrin = numr[len(numr)-1]
	# prepare 2-D mask for normalization
	if maskfile:
		pass#IMPORTIMPORTIMPORT import  types
		if(type(maskfile) is bytes):  mask2D = sparx_utilities.get_im(maskfile)
		else: mask2D = maskfile
	else : mask2D = sparx_utilities.model_circle(last_ring, nx, nx)
	if (first_ring > 0):
		tave = sparx_utilities.model_circle(first_ring-1, nx, ny)
		mask2D -= tave

	# read images and resample them into polar coordinates
	data = []
	#  center is in SPIDER convention
	cnx = int(nx/2) + 1
	cny = int(ny/2) + 1

	if(CTF):
		# for rotational alignment with CTF correction, we have more complicated strategies.
		#   We need two series of images in polar coordinates: multiplied by the ctf, and 'ctf-corrected', i.e., also divided by ctf^2+1/snr
		#   Thus, is alignment, the reference is always computed from the second series, while the first is used for alignment 
		parnames = ["Pixel_size", "defocus", "voltage", "Cs", "amp_contrast", "B_factor",  "ctf_applied"]
		#                        0                1              2            3              4             5                6
		for im in range(nima):
			if(im>0):
				temp = EMAN2_cppwrap.EMData()
				temp.read_image(stack, im)
			ctf_params = temp.get_attr( "ctf" )
			if(im == 0):  data_had_ctf = temp.get_attr('ctf_applied')
			ctf = sparx_morphology.ctf_2(nx, ctf_params)
			if(im == 0):
				lctf = len(ctf)
				ctf2 = [0.0]*lctf
			for i in range(lctf):  ctf2[i] += ctf[i]
		del ctf
		ref_data = []
		for im in range(nima):
			temp = EMAN2_cppwrap.EMData()
			temp.read_image(stack, im)
			st = EMAN2_cppwrap.Util.infomask(temp, mask2D, False)
					
			temp -= st[0]
			if(temp.get_attr("ctf_applied") == 0):
				pass#IMPORTIMPORTIMPORT from filter import filt_ctf
				ctf_params = temp.get_attr( "ctf" )
				temp = sparx_filter.filt_ctf(temp, ctf_params)
				temp.set_attr('ctf_applied', 1)
			pass#IMPORTIMPORTIMPORT from filter       import filt_table
			refc = sparx_filter.filt_table(temp, ctf2)
			
			alpha_original = temp.get_attr('alpha')
			sx =  temp.get_attr('sx')
			sy =  temp.get_attr('sy')
			miri = temp.get_attr('mirror')
			alphan, sxn, syn, mir = sparx_utilities.combine_params2(0, -sx, -sy, 0, -alpha_original, 0,0,0)
			tempg, kb = sparx_fundamentals.prepi(temp)
			cimage = EMAN2_cppwrap.Util.Polar2Dmi(tempg, cnx+sxn, cny+syn, numr, mode, kb)
			EMAN2_cppwrap.Util.Frngs(cimage, numr)
			data.append(cimage)

			#  Here alpha is position of the peak (i.e., starts from 1), just put as a place holder, will be determined in kmn
			data[im].set_attr_dict({'alpha':1.0, 'alpha_original':alpha_original, 'sx':sx, 'sy':sy, 'mirror': 0})
			cimage = EMAN2_cppwrap.Util.Polar2Dmi(refc, cnx+sxn, cny+syn, numr, mode, kb)
			EMAN2_cppwrap.Util.Frngs(cimage, numr)
			#  We do not need any attributes for ref_data, as they are going to be taken from data
			ref_data.append(cimage)

		del temp
		del refc
		del mask2D
		del ctf2
		sparx_statistics.kmn_ctf(data, ref_data, numr, wr, check_mirror, max_iter, rand_seed)
	else:
		for im in range(nima):
			if (im>0):
				temp = EMAN2_cppwrap.EMData()
				temp.read_image(stack, im)
			alpha_original = temp.get_attr('alpha')
			sx =  temp.get_attr('sx')
			sy =  temp.get_attr('sy')
			miri = temp.get_attr('mirror')
			temp.process_inplace("normalize.mask", {"mask":mask2D, "no_sigma":0})
			alphan, sxn, syn, mir = sparx_utilities.combine_params2(0.0, -sx, -sy, 0, -alpha_original, 0.0, 0.0, 0)
			tempg, kb = sparx_fundamentals.prepi(temp)
			cimage = EMAN2_cppwrap.Util.Polar2Dmi(tempg, cnx+sxn, cny+syn, numr, mode, kb)
			EMAN2_cppwrap.Util.Frngs(cimage, numr)
			data.append(cimage)
			#  Here alpha is position of the peak (i.e., starts from 1), just put as a place holder, will be determined in kmn
			data[im].set_attr_dict({'alpha':1.0, 'alpha_original':alpha_original, 'sx':sx, 'sy':sy, 'mirror':0})
		del temp
		del tempg
		del mask2D
		sparx_statistics.kmn_g(data, numr, wr, stack, check_mirror, max_iter, rand_seed)
	#  write out the alignment parameters to headers
	pass#IMPORTIMPORTIMPORT from utilities import write_header, file_type
	ext = sparx_utilities.file_type(stack)
	if(ext == "bdb"):
		pass#IMPORTIMPORTIMPORT from EMAN2db import EMAN2DB
		DB = EMAN2db.EMAN2DB()
		DB = EMAN2db.EMAN2DB.open_db(ipath)
	temp = EMAN2_cppwrap.EMData()
	for im in range(nima):
		alpha_original   = data[im].get_attr('alpha_original')
		alpha = data[im].get_attr('alpha')
		sx    =  data[im].get_attr('sx')
		sy    =  data[im].get_attr('sy')
		mirror =  data[im].get_attr('mirror')
		alpha = sparx_alignment.ang_n(alpha+1, mode, maxrin)
		#  here the original angle is irrelevant, used only to determine proper shifts
		alpha_original_n, sxn, syn, mir = sparx_utilities.combine_params2(0, -sx, -sy, 0, -alpha_original, 0,0,0)
		alphan, sxn, syn, mir           = sparx_utilities.combine_params2(0, -sxn, -syn, 0, alpha, 0,0, mirror)
		temp.read_image(stack, im, True)
		if(CTF and data_had_ctf == 0):   temp.set_attr('ctf_applied', 0)
		temp.set_attr_dict({'alpha':alphan, 'sx':sxn, 'sy':syn, 'mirror': mir})
		sparx_utilities.write_header(stack, temp, im)
		#temp.write_image(stack, im, EMUtil.ImageType.IMAGE_HDF, True)
	if(ext == "bdb"):
		DB.close_dict(ipath)
	sparx_utilities.print_end_msg("ali2d_rag")
	
def ali2d_rac(stack, maskfile = None, ir = 1, ou = -1, rs = 1, nclass = 2, maxit = 10, maxin = 10, check_mirror = False, rand_seed = 1000, MPI=False):
# 2D rotational classification and alignment using ccf in polar coords, no CTF
	test = True
	if MPI:
		if test:
			pass#IMPORTIMPORTIMPORT from development import ali2d_rac_MPI
			ali2d_rac_MPI(stack, maskfile, ir, ou, rs, nclass, maxit, maxin, check_mirror, rand_seed)
			return
		else:
			print('ali2d_rac: no mpi version')
			return

	pass#IMPORTIMPORTIMPORT from utilities    import model_circle, combine_params2, drop_image
	pass#IMPORTIMPORTIMPORT from alignment    import Numrinit, ringwe, ang_n
	pass#IMPORTIMPORTIMPORT from statistics   import kmnr, kmn, add_series_class
	pass#IMPORTIMPORTIMPORT from random       import seed, randint

	pass#IMPORTIMPORTIMPORT from utilities    import info, ttime, print_list_format
	pass#IMPORTIMPORTIMPORT import time

	random.seed(rand_seed)
	first_ring=int(ir); last_ring=int(ou); rstep=int(rs); kc=int(nclass); max_iter=int(maxit); max_internal=int(maxin);

	# somebody dedicated could write a version with option "H" for half rings that would work for ACF functions.
	mode = 'F'
	nima = EMAN2_cppwrap.EMUtil.get_image_count(stack)

	pass#IMPORTIMPORTIMPORT from utilities import print_begin_msg, print_end_msg, print_msg
	sparx_utilities.print_begin_msg('ali2d_rac')
	sparx_utilities.print_msg("Input stack                 : %s\n"%(stack))
	sparx_utilities.print_msg('Number of images            : %i\n' % nima)
	sparx_utilities.print_msg("Maskfile                    : %s\n"%(maskfile))
	sparx_utilities.print_msg("Inner radius                : %i\n"%(first_ring))
	sparx_utilities.print_msg("Outer radius                : %i\n"%(last_ring))
	sparx_utilities.print_msg("Ring step                   : %i\n"%(rstep))
	sparx_utilities.print_msg("Maximum iteration           : %i\n"%(max_iter))
	sparx_utilities.print_msg('Maximum intern iteration    : %i\n' % max_internal)
	sparx_utilities.print_msg("Consider Mirror             : %s\n"%(check_mirror))
	sparx_utilities.print_msg('Number of classes           : %i\n' % kc)
	sparx_utilities.print_msg("Random seed                 : %i\n"%(rand_seed))
	sparx_utilities.print_msg('Ouput stack                 : %s\n\n' % stack)
	t_start = time.time()

	# create the output directory, if it does not exist
	#if os.path.exists(outdir) is False: os.mkdir(outdir)
	temp = EMAN2_cppwrap.EMData()
	temp.read_image(stack, 0)
	nx = temp.get_xsize()
	ny = nx
	
	# default value for the last ring
	if(last_ring==-1): last_ring=nx//2-2
	
	# precalculate rings
	numr = sparx_alignment.Numrinit(first_ring, last_ring, rstep, mode)
	# norm to criterion
	norm_rsd = 0
	for n in range(1, len(numr), 3): norm_rsd += numr[n]
		
	wr    = sparx_alignment.ringwe(numr ,mode)
	lnumr = numr[len(numr)-1]
	# prepare 2-D ask for normalization
	if maskfile:
		pass#IMPORTIMPORTIMPORT import  types
		if(type(maskfile) is bytes):  mask2D = sparx_utilities.get_image(maskfile)
		else: mask2D = maskfile
	else : mask2D = sparx_utilities.model_circle(last_ring,nx,nx)
	if(first_ring > 0):
		tave = sparx_utilities.model_circle(first_ring-1, nx, ny)
		mask2D -= tave


	# read images and resample them into polar coordinates
	data = []
	org  = []
	#  center 
	cnx = nx//2+1
	cny = ny//2+1
	
	for im in range(nima):
		if(im>0):
			temp = EMAN2_cppwrap.EMData()
			temp.read_image(stack, im)
		sx =  temp.get_attr('sx')
		sy =  temp.get_attr('sy')
		alpha_original = temp.get_attr('alpha')
		miri = temp.get_attr('mirror')
		[mean, sigma, qn, qm] = EMAN2_cppwrap.Util.infomask(temp, mask2D, True)
		temp = (temp - mean)/sigma
		alpha_original_n,sxn,syn,mir = sparx_utilities.combine_params2(0, -sx, -sy, 0, -alpha_original,0,0,0)
		
		nring = len(numr)/3
		inr = numr[3*(nring-1)]
		if ((inr+int(cny+syn) <= ny-1 and -inr + int(cny+syn) >=1) and (inr+int(cnx+sxn) <= nx-1 and -inr + int(cnx+sxn) >=1)):
			cimage = EMAN2_cppwrap.Util.Polar2Dm(temp, cnx+sxn, cny+syn, numr, mode)
			EMAN2_cppwrap.Util.Frngs(cimage, numr)
			data.append(cimage)
			#  Here alpha is postion of the pick (i.e., starts from 1)
			data[im].set_attr_dict({'alpha':1.0, 'alpha_original':alpha_original, 'sx':sx, 'sy':sy, 'mirror': 0})
		else: 
			sparx_global_def.ERROR("ali2d_ra","Particle radius given too large for particle shifts found in the header",1)		

	del temp
	del mask2D
	
	# initial classification
	assign = [0]*nima
	nclass = [0]*kc
	init = "Random"
	#init = "Even"
	#init = "Batch"
	if(init == "Random"):
		again = True
		while(again):
			for i in range (nima):
				assign[i] = random.randint(0,kc-1)
				nclass[assign[i]] += 1
			# make sure than no class is zero
			again = False
			for k in range(kc):
				if(nclass[k] == 0): again = True
	elif(init == "Even"):
		for i in range (nima):
			assign[i] = i%kc
			nclass[assign[i]] += 1
	elif(init == "Batch"):
		kt = (nima+kc)//kc
		for i in range (nima):
			assign[i] = i//kt
			nclass[assign[i]] += 1

	
	print('%-20s'%("Initialization"))
	sparx_utilities.print_list_format(nclass)

	again = True
	it = -1

	t= time.time()
	while(again and it < (max_iter - 1)):
		it += 1
		# averages
		tave = []
		for k in range(kc):
			temp = sparx_statistics.kmnr(data, assign, nclass[k], k, numr, wr, check_mirror, max_internal, rand_seed+(it * 10))
			temp /= nclass[k]
			tave.append(temp)
					
		#  assign objects to averages, angles are irrelevant
		for k in range(kc):  nclass[k] = 0
		again = False

		# compute norm to distance
		norm = []
		for k in range(kc):
			retval = EMAN2_cppwrap.Util.Crosrng_ew(tave[k], tave[k], numr, wr, 0)
			sq1    = EMAN2_cppwrap.Util.ener(tave[k], numr)
			norm.append(2*sq1 / float(retval['qn']))
			
		Je_rsd = 0
		for im in range (nima):
			dmin = 1.0e20
			g    = -1
			for k in range(kc):
				
				if (check_mirror):
					retvals = EMAN2_cppwrap.Util.Crosrng_ew(tave[k], data[im], numr, wr, 0)
					qn  = retvals["qn"]
					retvals = EMAN2_cppwrap.Util.Crosrng_ew(tave[k], data[im], numr, wr, 1)
					qm  = retvals["qn"]
					qn = max(qn,qm)
				else:
					retvals = EMAN2_cppwrap.Util.Crosrng_ew(tave[k], data[im], numr, wr, 0)
					qn      = retvals["qn"]

				q1 = EMAN2_cppwrap.Util.ener(tave[k], numr)
				q2 = EMAN2_cppwrap.Util.ener(data[im], numr)
				qn = q1 + q2 - (qn * norm[k])
									
				if(qn < dmin):
					dmin = qn
					g = k

			if( g <0 ):
				print("  Error in assignment of objects to averages ")
				break
			if(assign[im] != g):
				again = True
				assign[im] = g

			nclass[g] += 1
			Je_rsd    += dmin / float(norm_rsd)

		#print 'before remove class'
		#print_list_format(nclass)
			
		# eliminate empty classes
		rep = True
		while(kc > 1 and rep):
			rep = False
			for k in range(kc):
				if(nclass[k] == 0):
					del nclass[k]
					del tave[k]
					#  fix the assign list
					for i in range(nima):
						if(assign[i] > k):   assign[i] -= 1
						elif assign[i] == k: assign[i] = random.randint(0, kc - 2)
					kc -= 1
					print("  Empty class was eliminated, new number of classes =",kc)
					if(kc == 1): again = False
					rep = True
					break

		print('%-20s %5d'%("ITERATION #",it+1))
		sparx_utilities.print_list_format(nclass)

		sparx_utilities.print_msg('> iteration %d      criterion %5.3e\n' % (it+1, Je_rsd))
		print('> iteration %d      criterion %5.3e\n' % (it+1, Je_rsd))

	sparx_utilities.print_msg('\n')
	for k in range(kc): sparx_utilities.print_msg('Cls[%3d]: %d\n' % (k, nclass[k]))
	sparx_utilities.print_msg('\n')

	print('time ite:', time.time() - t)
	print('%30s %5d %10s'%("Numbers of objects in ",kc," classes:"))
	sparx_utilities.print_list_format(nclass)
	#align class averages and transfer parameters to individual images
	for k in range(kc):
		tave[k].set_attr_dict({'alpha':1.0, 'mirror':0})
	
	sparx_statistics.kmn(tave, numr, wr, check_mirror, max_iter)

	#print ttime()

	talpha = [0]*kc
	tmir   = [0]*kc
	for k in range(kc):
		talpha[k] = sparx_alignment.ang_n(tave[k].get_attr('alpha'), mode, lnumr)
		tmir[k]   = tave[k].get_attr('mirror')

	del tave
	#  write out the alignment parameters to headers
	del temp
	pass#IMPORTIMPORTIMPORT from utilities import write_header, file_type
	ext = sparx_utilities.file_type(stack)
	if(ext == "bdb"):
		pass#IMPORTIMPORTIMPORT from EMAN2db import EMAN2DB
		DB = EMAN2db.EMAN2DB()
		DB = EMAN2db.EMAN2DB.open_db(ipath)
	temp = EMAN2_cppwrap.EMData()
	for im in range(nima):
				
		#  First combine with angle of the average
		alpha = sparx_alignment.ang_n(data[im].get_attr('alpha'), mode, lnumr)
		mirror =  data[im].get_attr('mirror')
		alpha, k,it, mirror = sparx_utilities.combine_params2(alpha, 0,0, mirror, talpha[assign[im]], 0,0, tmir[assign[im]])

		#  Second combine with given alignment
		alpha_original   =  data[im].get_attr('alpha_original')
		sx    =  data[im].get_attr('sx')
		sy    =  data[im].get_attr('sy')
		alpha_original_n, sxn, syn, mir = sparx_utilities.combine_params2(0, -sx, -sy, 0, -alpha_original,0,0,0)
		alphan, sxn, syn, mir           = sparx_utilities.combine_params2(0, -sxn, -syn, 0, alpha, 0,0,mirror)
		temp.read_image(stack, im, True)
		temp.set_attr_dict({'alpha':alphan, 'sx':sxn, 'sy':syn, 'mirror': mir, 'nclass':kc, 'assign':assign[im]})
		
		#if(data_had_ctf == 0):   temp.set_attr('ctf_applied', 0)
		sparx_utilities.write_header(stack, temp, im)
		#temp.write_image(stack ,im, EMUtil.ImageType.IMAGE_HDF, True)
	if(ext == "bdb"):
		DB.close_dict(ipath)
	del temp
	del data
	#for im in xrange(nima): data[im].write_image(stack, im, EMUtil.ImageType.IMAGE_HDF, True)
	#transform2d(stack, "dala.hdf")
	#ave, var, nclass = add_series_class("dala.hdf")
	#for k in xrange(kc):
	#		ave[k].set_attr_dict({'Class_average':1, 'nobjects':nclass[k]})
	#		var[k].set_attr_dict({'Class_variance':1, 'nobjects':nclass[k]})
	#		ave[k].write_image("class_ave.hdf", k)
	#		var[k].write_image("class_var.hdf", k)

	sparx_utilities.print_msg("\nTime: %f s\n" % (time.time() - t_start))
	sparx_utilities.print_end_msg("ali2d_rac")


def ali2d_ras(data2d, randomize = False, ir = 1, ou = -1, rs = 1, step = 1.0, dst = 0.0, maxit = 10, check_mirror = True, FH = 0.0, FF =0.0):
# stripped down 2D rotational alignment in polar coordinates
#  I did not check the version with no check mirror, I doubt it works.

	pass#IMPORTIMPORTIMPORT from utilities    import compose_transform2, combine_params2, get_arb_params, get_params2D, set_params2D, inverse_transform2
	pass#IMPORTIMPORTIMPORT from alignment    import Numrinit, ringwe, ang_n
	pass#IMPORTIMPORTIMPORT from statistics   import ave_series
	pass#IMPORTIMPORTIMPORT from filter       import filt_tanl
	pass#IMPORTIMPORTIMPORT from random       import random, randint

	first_ring=int(ir); last_ring=int(ou); rstep=int(rs); max_iter=int(maxit); 

	nx = data2d[0].get_xsize()
	ny = nx
	# default value for the last ring
	if (last_ring == -1): last_ring=nx//2-2
	mode = "F"

	nima = len(data2d)

	# precalculate rings
	numr = sparx_alignment.Numrinit(first_ring, last_ring, rstep, mode)
	wr   = sparx_alignment.ringwe(numr, mode)
	maxrin = numr[len(numr)-1]

	#  center is in SPIDER convention
	cnx = int(nx/2) + 1
	cny = int(ny/2) + 1
	# resample images into polar coordinates
	data = []
	if not check_mirror: mirror=0
	params = []
	for im in range(nima):
		if randomize:
			alpha, sx, sy, miri, scale = sparx_utilities.get_params2D(data2d[im])
			#  Check this 07/11/2015 PAP
			alphai, sxi, syi, mirrori = sparx_utilities.inverse_transform2(alpha, sx, sy)
			if check_mirror: alphan, sxn, syn, mirrorn = sparx_utilities.combine_params2(0.0, -sxi, -syi, 0, numpy.random.random()*360.0, 0.0, 0.0, random.randint(0, 1))
			else:            alphan, sxn, syn, mirrorn = sparx_utilities.combine_params2(0.0, -sxi, -syi, 0, numpy.random.random()*360.0, 0.0, 0.0, 0)			
			sparx_utilities.set_params2D(data2d[im], [alphan, sxn, syn, mirrorn, 1.0] )
		else:
			alphan, sxn, syn, mirrorn, scale = sparx_utilities.get_params2D(data2d[im])
		#  Here we need inverse transformation shifts for resampling into polar  WHY inverse ?  07/11/PAP
		alphai, sxn, syn, mirrori = sparx_utilities.inverse_transform2(alphan, sxn, syn)
		params.append([sxn, syn])
		nring = len(numr)/3
		inr = numr[3*(nring-1)]
		#here the centers of the image cny and cnx use Spider convention which means the index of the image array starts from 1 to nx (ny). 02-24-2015
		if ((inr+int(cny+syn) <= ny-1 and -inr + int(cny+syn) >=1) and (inr+int(cnx+sxn) <= nx-1 and -inr + int(cnx+sxn) >=1)):
			cimage = EMAN2_cppwrap.Util.Polar2Dm(data2d[im], cnx+sxn, cny+syn, numr, mode)
			EMAN2_cppwrap.Util.Frngs(cimage, numr)
			data.append(cimage)
		else: 
			sparx_global_def.ERROR("ali2d_ra","Particle radius given too large for particle shifts found in the header",1)
			
	total_iter = 0
	for Iter in range(max_iter):
		total_iter += 1
		tavg = sparx_statistics.ave_series(data2d)
		if( FH > 0.0):
			fl = 0.1+(FH-0.1)*Iter/float(max_iter-1)
			tavg = sparx_filter.filt_tanl(tavg, fl, FF)
		if total_iter == max_iter:  return tavg
		if Iter%4 != 0 or total_iter > max_iter-10: delta = 0.0
		else:                                       delta = dst
		#  Convert average to polar
		cimage = EMAN2_cppwrap.Util.Polar2Dm(tavg, cnx, cny, numr, mode)
		EMAN2_cppwrap.Util.Frngs(cimage, numr)
		EMAN2_cppwrap.Util.Applyws(cimage, numr, wr)
		for im in range(nima):
			# align current image to the reference 
			if(check_mirror):
				if delta == 0.0: retvals = EMAN2_cppwrap.Util.Crosrng_ms(cimage, data[im], numr, 0.0)
				else:            retvals = EMAN2_cppwrap.Util.Crosrng_ms_delta(cimage, data[im], numr, 0.0, delta)
				qn = retvals["qn"]
				qm = retvals["qm"]
				if (qn >= qm):
					ang = sparx_alignment.ang_n(retvals["tot"], mode, numr[-1])
					mirror = 0
				else:
					ang = sparx_alignment.ang_n(retvals["tmt"], mode, numr[-1])
					mirror = 1
			else:
				retvals = EMAN2_cppwrap.Util.Crosrng_e(cimage, data[im], numr, 0)
				ang = sparx_alignment.ang_n(retvals["tot"], mode, numr[-1])
			# combine parameters and store in data2d header
			alphan, sxn, syn, mir = sparx_utilities.combine_params2(0.0, -params[im][0], -params[im][1], 0, ang, 0.0 ,0.0, mirror)
			sparx_utilities.set_params2D(data2d[im], [alphan, sxn, syn, mir, 1.0])


def ali2d_rotationaltop(outdir, stack, randomize = False, orient=True, ir = 4, ou = -1, rs = 1, psi_max = 180.0, mode = "F", maxit = 10):
	# calling program for rotational alignment of power spectra
	pass#IMPORTIMPORTIMPORT from utilities    import print_begin_msg, print_end_msg, print_msg
	pass#IMPORTIMPORTIMPORT from utilities    import file_type
	pass#IMPORTIMPORTIMPORT import os

	
	if os.path.exists(outdir):   sparx_global_def.ERROR('Output directory exists, please change the name and restart the program', "ali2d_friedel", 1)
	os.mkdir(outdir)
	
	first_ring=int(ir); last_ring=int(ou); rstep=int(rs); max_iter=int(maxit);

	data2d = EMAN2_cppwrap.EMData.read_images(stack)
	nima = len(data2d)

	# default value for the last ring
	if last_ring == -1:  
		nx = data2d[0].get_xsize()
		last_ring = nx/2-2
	
	tavg = ali2d_rotational(data2d, randomize, orient, first_ring, last_ring, rstep, psi_max, mode, max_iter)
	tavg.write_image(os.path.join(outdir, "aqfinal.hdf"))
	# write out headers
	pass#IMPORTIMPORTIMPORT from utilities import write_headers
	sparx_utilities.write_headers(stack, data2d, list(range(nima)))
	

def ali2d_rotational(data2d, randomize = False, orient=True, ir = 1, ou = -1, rs = 1, psi_max = 180.0, mode = "F", maxit = 10):
# 2D rotational alignment of power spectra in polar coordinates

	pass#IMPORTIMPORTIMPORT from utilities    import get_params2D, set_params2D, model_blank, model_circle
	pass#IMPORTIMPORTIMPORT from alignment    import Numrinit, ringwe, ang_n
	pass#IMPORTIMPORTIMPORT from fundamentals import rot_shift2D, mirror
	pass#IMPORTIMPORTIMPORT from statistics   import ave_series
	pass#IMPORTIMPORTIMPORT from random       import randint

	first_ring=int(ir); last_ring=int(ou); rstep=int(rs); max_iter=int(maxit); 

	nx = data2d[0].get_xsize()
	ny = nx
	# default value for the last ring
	if (last_ring == -1): last_ring=nx//2-2

	nima = len(data2d)

	# precalculate rings
	numr = sparx_alignment.Numrinit(first_ring, last_ring, rstep, mode)
	wr   = sparx_alignment.ringwe(numr, mode)
	maxrin = numr[len(numr)-1]

	#  center is in SPIDER convention
	cnx = int(nx/2) + 1
	cny = int(ny/2) + 1
	# resample images into polar coordinates
	data = []
	if randomize:  angle = [float(random.randint(1,maxrin)) for i in range(nima)]
	else:          angle = [0.0]*nima
	for im in range(nima):
		#  Here we need inverse transformation shifts for resampling into polar
		cimage = EMAN2_cppwrap.Util.Polar2Dm(data2d[im], cnx, cny, numr, mode)
		EMAN2_cppwrap.Util.Frngs(cimage, numr)
		data.append(cimage.copy())

	change = True
	for Iter in range(max_iter+1):
		if Iter == max_iter or not change:
			tavg = sparx_utilities.model_blank(nx,ny)
			#compute average
			for im in range(nima):
				angle[im] = sparx_alignment.ang_n(angle[im], mode, numr[-1])
				sparx_utilities.set_params2D(data2d[im], [angle[im], 0.0, 0.0, 0, 1.0])
			tavg = sparx_statistics.ave_series(data2d)
			if orient:
				qet = -1.e23
				mask = sparx_utilities.model_circle(ou,nx,ny)-sparx_utilities.model_circle(ir,nx,ny)
				for i in range(360):
					temp = sparx_fundamentals.rot_shift2D(tavg,i/2.0)
					qt = sparx_fundamentals.mirror(temp,'y').cmp("dot", temp, {"negative":0, "mask":mask})
					if(qt > qet):
						qet = qt
						mang = i/2.0
				if( mang != 0.0 ):
					for im in range(nima):
						angle[im] += mang
						sparx_utilities.set_params2D(data2d[im], [angle[im], 0.0, 0.0, 0, 1.0])
					tavg = sparx_statistics.ave_series(data2d)
			return tavg
		else:
			cimage.to_zero()
			for im in range(nima):  EMAN2_cppwrap.Util.update_fav(cimage, data[im], angle[im], 0, numr)
			EMAN2_cppwrap.Util.Applyws(cimage, numr, wr)
			EMAN2_cppwrap.Util.mul_scalar(cimage, 1.0/float(nima))
		change = False
		for im in range(nima):
			# align current image to the reference 
			#retvals = Util.Crosrng_e(cimage, data[im], numr, 0)
			retvals = EMAN2_cppwrap.Util.Crosrng_sm_psi(cimage, data[im], numr, 0.0, 0, psi_max)
			if( abs(retvals["tot"] - angle[im]) > 1.e-2):
				change = True
				angle[im] = retvals["tot"]

def ali2d_cross_res(stack, outdir, maskfile=None, ir=1, ou=-1, rs=1, xr="4 2 1 1", yr="-1", ts="2 1 0.5 0.25", center=1, maxit=0, CTF=False, snr=1.0, user_func_name="ref_ali2d"):
	"""
		Split data into odd and even data sets and align them seperately
		Cross resolution alignment
	"""
	pass#IMPORTIMPORTIMPORT import os
	pass#IMPORTIMPORTIMPORT from utilities 		import model_circle, combine_params2, drop_image
	pass#IMPORTIMPORTIMPORT from utilities          import get_input_from_string, get_image, get_arb_params, set_arb_params
	pass#IMPORTIMPORTIMPORT from fundamentals 	import rot_shift2D
	pass#IMPORTIMPORTIMPORT from statistics 	      import add_oe_series, ave_series_ctf, ave_series, fsc_mask
	pass#IMPORTIMPORTIMPORT from alignment 		import Numrinit, ringwe, ali2d_single_iter, align2d
	pass#IMPORTIMPORTIMPORT from filter 		import filt_table, filt_ctf
	pass#IMPORTIMPORTIMPORT from morphology     import ctf_2

	pass#IMPORTIMPORTIMPORT from utilities import print_begin_msg, print_end_msg, print_msg
	pass#IMPORTIMPORTIMPORT import	types
	
	if os.path.exists(outdir):  sparx_global_def.ERROR('Output directory exists, please change the name and restart the program', "ali2d_cross_res", 1)
	os.mkdir(outdir)
	pass#IMPORTIMPORTIMPORT import global_def
	sparx_global_def.LOGFILE =  os.path.join(outdir, sparx_global_def.LOGFILE)

	sparx_utilities.print_begin_msg("ali2d_cross_res")

	pass#IMPORTIMPORTIMPORT import user_functions
	user_func = sparx_user_functions.factory[user_func_name]

	xrng        = sparx_utilities.get_input_from_string(xr)
	if  yr == "-1":  yrng = xrng
	else          :  yrng = sparx_utilities.get_input_from_string(yr)
	step        = sparx_utilities.get_input_from_string(ts)

	first_ring=int(ir); last_ring=int(ou); rstep=int(rs); max_iter=int(maxit);

	if max_iter == 0:
		max_iter  = 10
		auto_stop = True
	else:
		auto_stop = False

	sparx_utilities.print_msg("Input stack                 : %s\n"%(stack))
	sparx_utilities.print_msg("Output directory            : %s\n"%(outdir))
	sparx_utilities.print_msg("Inner radius                : %i\n"%(first_ring))

	nima = EMAN2_cppwrap.EMUtil.get_image_count(stack)
	ima = EMAN2_cppwrap.EMData()
	ima.read_image(stack, 0)
	nx = ima.get_xsize()
	# default value for the last ring
	if (last_ring == -1):  last_ring = nx//2-2

	sparx_utilities.print_msg("Outer radius                : %i\n"%(last_ring))
	sparx_utilities.print_msg("Ring step                   : %i\n"%(rstep))
	sparx_utilities.print_msg("X search range              : %s\n"%(xrng))
	sparx_utilities.print_msg("Y search range              : %s\n"%(yrng))
	sparx_utilities.print_msg("Translational step          : %s\n"%(step))
	sparx_utilities.print_msg("Center type                 : %i\n"%(center))
	sparx_utilities.print_msg("Maximum iteration           : %i\n"%(max_iter))
	sparx_utilities.print_msg("CTF correction              : %s\n"%(CTF))
	sparx_utilities.print_msg("Signal-to-Noise Ratio       : %f\n"%(snr))
	if auto_stop:  sparx_utilities.print_msg("Stop iteration with         : criterion\n")
	else:           sparx_utilities.print_msg("Stop iteration with         : maxit\n")


	if maskfile:
		if(type(maskfile) is bytes):
			sparx_utilities.print_msg("Maskfile                    : %s\n\n"%(maskfile))
			mask=sparx_utilities.get_image(maskfile)
		else:
			sparx_utilities.print_msg("Maskfile                    : user provided in-core mask\n\n")
			mask = maskfile
	else :
		sparx_utilities.print_msg("Maskfile                    : default, a circle with radius %i\n\n"%(last_ring))
		mask = sparx_utilities.model_circle(last_ring, nx, nx)

	#  ODD-EVEN is controlled by setting NG to 2
	NG = 2
	cnx = int(nx/2)+1
	cny = cnx
	mode = "F"
	if(CTF):
		ctf_params = ima.get_attr( "ctf" )
		data_had_ctf = ima.get_attr( "ctf_applied" )
		ctm = sparx_morphology.ctf_2(nx, ctf_params)
		lctf = len(ctm)
		ctf2 = [[[0.0]*lctf for j in range(2)] for i in range(NG)]
		ctfb2 = [[0.0]*lctf for i in range(NG)]
	del ima
	all_data = EMAN2_cppwrap.EMData.read_images(stack)
	for im in range(nima):
		all_data[im].set_attr('ID', im)
		k = im%NG
		if(CTF):
			ctf_params = all_data[im].get_attr( "ctf" )
			ctm = sparx_morphology.ctf_2(nx, ctf_params)

			kl = (im//2)%NG  # not sure it will work for NG>2
			for i in range(lctf):
				ctf2[k][kl][i] += ctm[i]
	
			if(all_data[im].get_attr("ctf_applied") == 0):
				st = EMAN2_cppwrap.Util.infomask(all_data[im], mask, False)
				all_data[im] -= st[0]
				all_data[im] = sparx_filter.filt_ctf(all_data[im], ctf_params)
				all_data[im].set_attr('ctf_applied', 1)

	#  create to lists of images in groups.
	data = [[] for i in range(NG)]
	for im in range(nima):
		k = im%NG
		data[k].append(all_data[im])

	if(CTF):
		ctf_tot = [0.0]*lctf
		for i in range(lctf):
			for k in range(NG):
				ctf_tot[i] += ctf2[k][0][i] + ctf2[k][1][i]
			ctf_tot[i] = 1.0/(ctf_tot[i] + 1.0/snr)
		for k in range(NG):
			for i in range(lctf):
				ctfb2[k][i] = 1.0/(ctf2[k][0][i] + ctf2[k][1][i] + 1.0/snr)
				for kl in range(2):
					ctf2[k][kl][i] = 1.0/(ctf2[k][kl][i] + 1.0/snr)
	#precalculate rings
	numr = sparx_alignment.Numrinit(first_ring, last_ring, rstep, mode)
	wr = sparx_alignment.ringwe(numr, mode)
	tavg = [None]*NG
	for k in range(NG):
		av1, av2 = sparx_statistics.add_oe_series(data[k])
		EMAN2_cppwrap.Util.add_img(av1, av2)
		if(CTF):  tavg[k] = sparx_filter.filt_table(av1, ctfb2[k])
		else:	  tavg[k] = av1/len(data[k])
		sparx_utilities.drop_image(tavg[k],os.path.join(outdir, "aqc_%03d_%03d.hdf"%(k, 0)))
	fscross = sparx_statistics.fsc_mask(tavg[0], tavg[1], mask, 1.0, os.path.join(outdir, "drcross_%03d"%(0)))

	if(CTF): total_ave = sparx_statistics.ave_series_ctf(all_data, ctf_tot)
	else:    total_ave = sparx_statistics.ave_series(all_data)
	sparx_utilities.drop_image(total_ave, os.path.join(outdir, "total_ave_%03d.hdf"%(0)))
	a0 = total_ave.cmp("dot", total_ave, dict(negative = 0, mask = mask))
	msg = "Initial criterion = %-20.7e\n"%(a0)
	sparx_utilities.print_msg(msg)
	params = ["alpha", "sx", "sy", "mirror"]
	# initialize data for the reference preparation function
	#  mask can be modified in user_function
	ref_data = [mask, center, None, None]
	cs=[[0.0,0.0]]*NG
	total_iter = 0
	for N_step in range(len(xrng)):
		msg = "\nX range = %5.2f   Y range = %5.2f   Step = %5.2f\n"%(xrng[N_step], yrng[N_step], step[N_step])
		sparx_utilities.print_msg(msg)
		for Iter in range(max_iter):
			total_iter += 1
			frsc = []
			ktavg = [None]*NG
			for k in range(NG):
				sxsum, sysum, nope = sparx_alignment.ali2d_single_iter(data[k], numr, wr, cs[k], tavg[k], cnx, cny, xrng[N_step], yrng[N_step], step[N_step], mode=mode)
				av1, av2 = sparx_statistics.add_oe_series(data[k])
				if(CTF):
					tavg[k] = sparx_filter.filt_table(EMAN2_cppwrap.Util.addn_img(av1, av2), ctfb2[k])
					av1    = sparx_filter.filt_table(av1, ctf2[k][0])
					av2    = sparx_filter.filt_table(av2, ctf2[k][1])
				else:
					tavg[k] = (av1+av2)/len(data[k])
				sparx_utilities.drop_image(tavg[k], os.path.join(outdir, "aqc_%03d_%03d.hdf"%(k, total_iter)))

				frsc.append(sparx_statistics.fsc_mask(av1, av2, ref_data[0], 1.0, os.path.join(outdir, "resolution_%03d_%03d"%(k, total_iter))))
				#  prepare averages for alignment
				kref_data = [mask, 0, tavg[k], frsc[k]]
				#  call the user-supplied function to prepare reference image, i.e., filter it, but do not center!
				ktavg[k], cs[k] = user_func( kref_data )
				del kref_data
			#  This should be done only for estimation of resolution, nothing else!
			alpha, sx, sy, mirror, peak = sparx_alignment.align2d(ktavg[0], ktavg[1], xrng[0], yrng[0], step=0.25, first_ring = first_ring, last_ring = last_ring, rstep=1, mode = mode)
			#  apply parameters to the original average
			favg2 = sparx_fundamentals.rot_shift2D(tavg[0], alpha, sx, sy, mn, interpolation_method="gridding")
			fscross = sparx_statistics.fsc_mask(favg2, tavg[1], ref_data[0], 1.0, os.path.join(outdir, "drcross_%03d"%(total_iter)))
			del favg2
			# Here one may want to apply rot-shift of the first average to all images in its group
			for im in range(len(data[0])):
				ps = sparx_utilities.get_arb_params(data[0][im], params)
				an,sxn,syn,mrn = sparx_utilities.combine_params2(ps[0], ps[1], ps[2], ps[3], alpha, sx, sy, mirror)
				sparx_utilities.set_arb_params(data[0][im], [an,sxn,syn,mrn], params)
			k = 0
			av1, av2 = sparx_statistics.add_oe_series(data[k])
			if(CTF):
				tavg[k] = sparx_filter.filt_table(EMAN2_cppwrap.Util.addn_img(av1, av2), ctfb2[k])
			else:
				tavg[k] = (av1+av2)/len(data[k])
			#  Here we have to change fsc values.  The reason is that we have crossresolution, so snr can be calculated directly,
			#        while in user function the fit to fsc is done under assumption that is was calculated by splitting the dataset, so it has a factor of 2
			for i in range(len(fscross[1])):   fscross[1][i] = fscross[1][i]/(2.0-fscross[1][i])
			for k in range(NG):
				#  Apply the same filtration to all averages
				ref_data[2] = tavg[k]
				ref_data[3] = fscross
				#  call the user-supplied function to prepare reference image, i.e., filter and center!
				tavg[k], cs[k] = user_func( ref_data )
				sparx_utilities.drop_image(tavg[k], os.path.join(outdir, "aqf_%03d_%03d.hdf"%(k, total_iter)))

			if(CTF): total_ave = sparx_statistics.ave_series_ctf(all_data, ctf_tot)
			else:    total_ave = sparx_statistics.ave_series(all_data)
			sparx_utilities.drop_image(total_ave, os.path.join(outdir, "total_ave_%03d.hdf"%(total_iter)))
			# a0 should increase; stop algorithm when it decreases.
			a1 = total_ave.cmp("dot", total_ave, dict(negative = 0, mask = ref_data[0]))
			msg = "ITERATION #%3d	     criterion = %20.7e\n"%(total_iter,a1)
			sparx_utilities.print_msg(msg)
			if(a1 < a0):
				if (auto_stop == True): break
			else:	a0 = a1
	# write out headers
	if(CTF and data_had_ctf == 0):
		for k in range(NG):
			for im in range(len(data[k])):
				data[k][im].set_attr('ctf_applied', 0)
	pass#IMPORTIMPORTIMPORT from utilities import write_header, file_type
	ext = sparx_utilities.file_type(stack)
	if(ext == "bdb"):
		pass#IMPORTIMPORTIMPORT from EMAN2db import EMAN2DB
		DB = EMAN2db.EMAN2DB()
		DB = EMAN2db.EMAN2DB.open_db(ipath)
	for im in range(nima):
		k=im%NG
		imm = im//NG
		sparx_utilities.write_header(stack, data[k][imm], im)
		#data[k][imm].write_image(stack, im, EMUtil.ImageType.IMAGE_HDF, True)
	if(ext == "bdb"):
		DB.close_dict(ipath)
	sparx_utilities.print_end_msg("ali2d_cross_res")


"""Multiline Comment2"""

"""Multiline Comment3"""


def ali3d(stack, ref_vol, outdir, maskfile = None, ir = 1, ou = -1, rs = 1, 
            xr = "4 2 2 1", yr = "-1", ts = "1 1 0.5 0.25", delta = "10 6 4 4", an = "-1", apsi = "-1", deltapsi = "-1", startpsi = "-1",
	    center = -1, maxit = 5, CTF = False, snr = 1.0,  ref_a = "S", sym = "c1",
	    user_func_name = "ref_ali3d", fourvar = True, npad = 4, debug = False, MPI = False, termprec = 0.0):
	"""
		Name
			ali3d - Perform 3-D projection matching given initial reference volume and image series
		Input
			stack: set of 2-D images in a stack file, images have to be squares
			ref_vol: initial reference volume
			outdir: directory name into which the results will be written
			maskfile: filename of the file containing 3D mask.
			ir: inner radius for rotational correlation > 0 
			ou: outer radius for rotational correlation <int(nx/2)-1 
			rs: steps between rings in rotational correlation >0
			xr: range for translation search in x direction in each iteration, search is +/xr
			yr: range for translation search in y direction in each iteration, search is +/yr
			ts: step size of the translation search in both directions, search is -xr, -xr+ts, 0, xr-ts, xr, can be fractional.
			delta: angular step for the reference projections in respective iterations
			an: angular neighborhood for local searches
			center: average center method
			max_iter: maximum iterations at each angle step
			CTF: if the flag is present, program will use the CTF information stored in file headers
			snr: signal noise ratio used in the 3D reconstruction
			ref_a: method for creating quasi-uniform distribution of the projection directions of reference projections: "S" - spiral
			sym: symmetry of the refined structure
			function: name of the user-supplied-function
			MPI: if presetm use MPI version
		Output
			output_directory: directory name into which the output files will be written.
			header: the alignment parameters are stored in the headers of input files as Transform Object xform.proj
	"""
	if MPI:
		ali3d_MPI(stack, ref_vol, outdir, maskfile, ir, ou, rs, xr, yr, ts,
	        	delta, an, apsi, deltapsi, startpsi, center, maxit, CTF, snr, ref_a, sym, user_func_name,
			fourvar, npad, debug, termprec)
		return

	pass#IMPORTIMPORTIMPORT from alignment      import proj_ali_incore, proj_ali_incore_local
	pass#IMPORTIMPORTIMPORT from utilities      import model_circle, drop_image, get_image, get_input_from_string
	pass#IMPORTIMPORTIMPORT from utilities      import get_params_proj
	pass#IMPORTIMPORTIMPORT from utilities      import estimate_3D_center, rotate_3D_shift
	pass#IMPORTIMPORTIMPORT from filter         import filt_params, fit_tanh, filt_tanl, filt_ctf
	pass#IMPORTIMPORTIMPORT from statistics     import fsc_mask
	pass#IMPORTIMPORTIMPORT import os
	pass#IMPORTIMPORTIMPORT import types
	pass#IMPORTIMPORTIMPORT from utilities      import print_begin_msg, print_end_msg, print_msg
	pass#IMPORTIMPORTIMPORT from alignment      import Numrinit, prepare_refrings
	pass#IMPORTIMPORTIMPORT from projection     import prep_vol

	pass#IMPORTIMPORTIMPORT import user_functions
	user_func = sparx_user_functions.factory[user_func_name]

	if os.path.exists(outdir):  sparx_global_def.ERROR('Output directory exists, please change the name and restart the program', "ali3d", 1)
	os.mkdir(outdir)
	pass#IMPORTIMPORTIMPORT import global_def
	sparx_global_def.LOGFILE =  os.path.join(outdir, sparx_global_def.LOGFILE)
	sparx_utilities.print_begin_msg("ali3d")

	xrng        = sparx_utilities.get_input_from_string(xr)
	if  yr == "-1":  yrng = xrng
	else          :  yrng = sparx_utilities.get_input_from_string(yr)
	step        = sparx_utilities.get_input_from_string(ts)
	delta       = sparx_utilities.get_input_from_string(delta)
	lstp = min(len(xrng), len(yrng), len(step), len(delta))
	if an == "-1":
		an = [-1] * lstp
	else:
		an = sparx_utilities.get_input_from_string(an)
	first_ring  = int(ir)
	rstep       = int(rs)
	last_ring   = int(ou)
	max_iter    = int(maxit)
	center      = int(center)

	sparx_utilities.print_msg("Input stack                 : %s\n"%(stack))
	sparx_utilities.print_msg("Reference volume            : %s\n"%(ref_vol))	
	sparx_utilities.print_msg("Output directory            : %s\n"%(outdir))
	sparx_utilities.print_msg("Maskfile                    : %s\n"%(maskfile))
	sparx_utilities.print_msg("Inner radius                : %i\n"%(first_ring))

	vol     = EMAN2_cppwrap.EMData()
	vol.read_image(ref_vol)
	nx      = vol.get_xsize()
	if last_ring == -1:	last_ring = nx/2 - 2

	sparx_utilities.print_msg("Outer radius                : %i\n"%(last_ring))
	sparx_utilities.print_msg("Ring step                   : %i\n"%(rstep))
	sparx_utilities.print_msg("X search range              : %s\n"%(xrng))
	sparx_utilities.print_msg("Y search range              : %s\n"%(yrng))
	sparx_utilities.print_msg("Translational step          : %s\n"%(step))
	sparx_utilities.print_msg("Angular step                : %s\n"%(delta))
	sparx_utilities.print_msg("Angular search range        : %s\n"%(an))
	sparx_utilities.print_msg("Maximum iteration           : %i\n"%(max_iter))
	sparx_utilities.print_msg("Center type                 : %i\n"%(center))
	sparx_utilities.print_msg("CTF correction              : %s\n"%(CTF))
	sparx_utilities.print_msg("Signal-to-Noise Ratio       : %f\n"%(snr))
	sparx_utilities.print_msg("Reference projection method : %s\n"%(ref_a))
	sparx_utilities.print_msg("Symmetry group              : %s\n\n"%(sym))
	sparx_utilities.print_msg("User function               : %s\n"%(user_func_name))

	if maskfile :
		if type(maskfile) is bytes: mask3D = sparx_utilities.get_image(maskfile)
		else                                  : mask3D = maskfile
	else          :   mask3D = sparx_utilities.model_circle(last_ring, nx, nx, nx)
	mask2D = sparx_utilities.model_circle(last_ring, nx, nx) - sparx_utilities.model_circle(first_ring, nx, nx)
	numr   = sparx_alignment.Numrinit(first_ring, last_ring, rstep, "F")

	if CTF:
		pass#IMPORTIMPORTIMPORT from reconstruction import recons3d_4nn_ctf
		pass#IMPORTIMPORTIMPORT from filter         import filt_ctf
	pass#IMPORTIMPORTIMPORT else: from reconstruction import recons3d_4nn

	if debug:  outf = open(os.path.join(outdir, "progress"), "w")
	else:      outf = None

	# horatio active_refactoring Jy51i1EwmLD4tWZ9_00000_1
	# active = EMUtil.get_all_attributes(stack, 'active')
	# list_of_particles = []
	# for im in xrange(len(active)):
	# 	if(active[im]):  list_of_particles.append(im)
	# del active
	nima = EMAN2_cppwrap.EMUtil.get_image_count(stack)
	list_of_particles = list(range(nima))
		
	data = EMAN2_cppwrap.EMData.read_images(stack, list_of_particles)
	for im in range(len(data)):
		data[im].set_attr('ID', list_of_particles[im])
		if CTF:
			ctf_params = data[im].get_attr("ctf")
			st = EMAN2_cppwrap.Util.infomask(data[im], mask2D, False)
			data[im] -= st[0]
			data[im] = sparx_filter.filt_ctf(data[im], ctf_params)
			data[im].set_attr('ctf_applied', 1)

	nima = len(data)
	# initialize data for the reference preparation function
	ref_data = [ mask3D, max(center,0), None, None ]#  for center -1 switch of centering by user function

	cs = [0.0]*3
	# do the projection matching
	for N_step in range(lstp):
		for Iter in range(max_iter):
			sparx_utilities.print_msg("\nITERATION #%3d\n"%(N_step*max_iter+Iter+1))

			volft, kb = sparx_projection.prep_vol(vol)
			refrings = sparx_alignment.prepare_refrings( volft, kb, nx, delta[N_step], ref_a, sym, numr, MPI=False)
			del volft, kb
			if( an[N_step] > 0):
				# generate list of angles
				pass#IMPORTIMPORTIMPORT from alignment import generate_list_of_reference_angles_for_search
				list_of_reference_angles = \
					sparx_alignment.generate_list_of_reference_angles_for_search([[refrings[lr].get_attr("phi"), refrings[lr].get_attr("theta")] for lr in range(len(refrings))], sym=sym)			


			for im in range(nima):
				if an[N_step] == -1:
					peak, pixel_error = sparx_alignment.proj_ali_incore(data[im],refrings,numr,xrng[N_step],yrng[N_step],step[N_step], sym=sym)
				else:
					peak, pixel_error = sparx_alignment.proj_ali_incore_local(data[im],refrings,list_of_reference_angles,numr,xrng[N_step],yrng[N_step],step[N_step],an[N_step],sym=sym)
				data[im].set_attr("previousmax", peak)
			if( an[N_step] > 0): del list_of_reference_angles

			if center == -1 and sym[0] == 'c':
				cs[0], cs[1], cs[2], dummy, dummy = sparx_utilities.estimate_3D_center(data)
				msg = "Average center x = %10.3f        Center y = %10.3f        Center z = %10.3f\n"%(cs[0], cs[1], cs[2])
				sparx_utilities.print_msg(msg)
				if int(sym[1]) > 1:
					cs[0] = cs[1] = 0.0
					sparx_utilities.print_msg("For symmetry group cn (n>1), we only center the volume in z-direction\n")
				sparx_utilities.rotate_3D_shift(data, [-cs[0], -cs[1], -cs[2]])

			if CTF:   vol1 = sparx_reconstruction.recons3d_4nn_ctf(data, list(range(0, nima, 2)), snr, 1, sym)
			else:	   vol1 = sparx_reconstruction.recons3d_4nn(data, list(range(0, nima, 2)), sym, snr = snr)
			if CTF:   vol2 = sparx_reconstruction.recons3d_4nn_ctf(data, list(range(1, nima, 2)), snr, 1, sym)
			else:	   vol2 = sparx_reconstruction.recons3d_4nn(data, list(range(1, nima, 2)), sym, snr = snr)

			fscc = sparx_statistics.fsc_mask(vol1, vol2, mask3D, 1.0, os.path.join(outdir, "resolution%04d"%(N_step*max_iter+Iter+1)))
			del vol1
			del vol2

			# calculate new and improved 3D
			if CTF:  vol = sparx_reconstruction.recons3d_4nn_ctf(data, list(range(nima)), snr, 1, sym)
			else:	 vol = sparx_reconstruction.recons3d_4nn(data, list(range(nima)), sym, snr = snr)
			# store the reference volume
			sparx_utilities.drop_image(vol, os.path.join(outdir, "vol%04d.hdf"%(N_step*max_iter+Iter+1)))
			ref_data[2] = vol
			ref_data[3] = fscc

			#  call user-supplied function to prepare reference image, i.e., center and filter it
			vol, dummy = user_func(ref_data)

			sparx_utilities.drop_image(vol, os.path.join(outdir, "volf%04d.hdf"%(N_step*max_iter+Iter+1)))
			#  here we write header info
			pass#IMPORTIMPORTIMPORT from utilities import write_headers
			#from utilities import write_select_headers
			if CTF:
				for dat in data:  dat.set_attr('ctf_applied',0)
			sparx_utilities.write_headers(stack, data, list_of_particles)
			#list_params= ['ID','xform.projection']
			#write_select_headers(stack, data, list_params)
			if CTF:
				for dat in data:  dat.set_attr('ctf_applied', 1)
	sparx_utilities.print_end_msg("ali3d")

def ali3d_MPI(stack, ref_vol, outdir, maskfile = None, ir = 1, ou = -1, rs = 1, 
            xr = "4 2 2 1", yr = "-1", ts = "1 1 0.5 0.25", delta = "10 6 4 4", an = "-1", apsi = "-1", deltapsi = "-1", startpsi = "-1",
	    center = -1, maxit = 5, CTF = False, snr = 1.0,  ref_a = "S", sym = "c1",  user_func_name = "ref_ali3d",
	    fourvar = True, npad = 2, debug = False, termprec = 0.0):

	pass#IMPORTIMPORTIMPORT from alignment       import Numrinit, prepare_refrings, proj_ali_incore, proj_ali_incore_local, proj_ali_incore_local_psi
	pass#IMPORTIMPORTIMPORT from utilities       import model_circle, get_image, drop_image, get_input_from_string
	pass#IMPORTIMPORTIMPORT from utilities       import bcast_list_to_all, bcast_number_to_all, reduce_EMData_to_root, bcast_EMData_to_all
	pass#IMPORTIMPORTIMPORT from utilities       import send_attr_dict
	pass#IMPORTIMPORTIMPORT from utilities       import get_params_proj, file_type
	pass#IMPORTIMPORTIMPORT from fundamentals    import rot_avg_image
	pass#IMPORTIMPORTIMPORT import os
	pass#IMPORTIMPORTIMPORT import types
	pass#IMPORTIMPORTIMPORT from utilities       import print_begin_msg, print_end_msg, print_msg
	pass#IMPORTIMPORTIMPORT from mpi             import mpi_bcast, mpi_comm_size, mpi_comm_rank, MPI_FLOAT, MPI_COMM_WORLD, mpi_barrier
	pass#IMPORTIMPORTIMPORT from mpi             import mpi_reduce, MPI_INT, MPI_SUM
	pass#IMPORTIMPORTIMPORT from filter          import filt_ctf
	pass#IMPORTIMPORTIMPORT from projection      import prep_vol, prgs
	pass#IMPORTIMPORTIMPORT from statistics      import hist_list, varf3d_MPI
	pass#IMPORTIMPORTIMPORT from applications    import MPI_start_end


	number_of_proc = mpi.mpi_comm_size(mpi.MPI_COMM_WORLD)
	myid           = mpi.mpi_comm_rank(mpi.MPI_COMM_WORLD)
	main_node = 0
	
	if os.path.exists(outdir):  sparx_global_def.ERROR('Output directory exists, please change the name and restart the program', "ali3d_MPI", 1, myid)
	mpi.mpi_barrier(mpi.MPI_COMM_WORLD)

	if myid == main_node:
		os.mkdir(outdir)
		pass#IMPORTIMPORTIMPORT import global_def
		sparx_global_def.LOGFILE =  os.path.join(outdir, sparx_global_def.LOGFILE)
		sparx_utilities.print_begin_msg("ali3d_MPI")
	mpi.mpi_barrier(mpi.MPI_COMM_WORLD)

	if debug:
		pass#IMPORTIMPORTIMPORT from time import sleep
		while not os.path.exists(outdir):
			print("Node ",myid,"  waiting...")
			time.sleep(5)

		info_file = os.path.join(outdir, "progress%04d"%myid)
		finfo = open(info_file, 'w')
	else:
		finfo = None

	xrng        = sparx_utilities.get_input_from_string(xr)
	if  yr == "-1":  yrng = xrng
	else          :  yrng = sparx_utilities.get_input_from_string(yr)
	step        = sparx_utilities.get_input_from_string(ts)
	delta       = sparx_utilities.get_input_from_string(delta)
	lstp = min(len(xrng), len(yrng), len(step), len(delta))
	if an == "-1":
		an = [-1.0] * lstp
	else:
		an = sparx_utilities.get_input_from_string(an)

	if apsi == "-1":
		apsi = [-1] * lstp
	else:
		apsi = sparx_utilities.get_input_from_string(apsi)

	if deltapsi == "-1":
		deltapsi = [-1] * lstp
	else:
		deltapsi = sparx_utilities.get_input_from_string(deltapsi)

	if startpsi == "-1":
		startpsi = [-1] * lstp
	else:
		startpsi = sparx_utilities.get_input_from_string(startpsi)

	first_ring  = int(ir)
	rstep       = int(rs)
	last_ring   = int(ou)
	max_iter    = int(maxit)
	center      = int(center)

	vol     = EMAN2_cppwrap.EMData()
	vol.read_image(ref_vol)
	nx      = vol.get_xsize()
	if last_ring < 0:	last_ring = int(nx/2) - 2

	if myid == main_node:
		pass#IMPORTIMPORTIMPORT import user_functions
		user_func = sparx_user_functions.factory[user_func_name]

		sparx_utilities.print_msg("Input stack                 : %s\n"%(stack))
		sparx_utilities.print_msg("Reference volume            : %s\n"%(ref_vol))	
		sparx_utilities.print_msg("Output directory            : %s\n"%(outdir))
		sparx_utilities.print_msg("Maskfile                    : %s\n"%(maskfile))
		sparx_utilities.print_msg("Inner radius                : %i\n"%(first_ring))
		sparx_utilities.print_msg("Outer radius                : %i\n"%(last_ring))
		sparx_utilities.print_msg("Ring step                   : %i\n"%(rstep))
		sparx_utilities.print_msg("X search range              : %s\n"%(xrng))
		sparx_utilities.print_msg("Y search range              : %s\n"%(yrng))
		sparx_utilities.print_msg("Translational step          : %s\n"%(step))
		sparx_utilities.print_msg("Angular step                : %s\n"%(delta))
		sparx_utilities.print_msg("Angular search range (phi and theta)       : %s\n"%(an))
		sparx_utilities.print_msg("Angular search range (psi)                 : %s\n"%(apsi))
		sparx_utilities.print_msg("Delta psi                   : %s\n"%(deltapsi))
		sparx_utilities.print_msg("Start psi                   : %s\n"%(startpsi))
		sparx_utilities.print_msg("Maximum iteration           : %i\n"%(max_iter))
		sparx_utilities.print_msg("Percentage of change for termination: %f\n"%(termprec))
		sparx_utilities.print_msg("Center type                 : %i\n"%(center))
		sparx_utilities.print_msg("CTF correction              : %s\n"%(CTF))
		sparx_utilities.print_msg("Signal-to-Noise Ratio       : %f\n"%(snr))
		sparx_utilities.print_msg("Reference projection method : %s\n"%(ref_a))
		sparx_utilities.print_msg("Symmetry group              : %s\n\n"%(sym))
		sparx_utilities.print_msg("User function               : %s\n"%(user_func_name))

	if maskfile:
		if type(maskfile) is bytes: mask3D = sparx_utilities.get_image(maskfile)
		else:                                  mask3D = maskfile
	else: mask3D = sparx_utilities.model_circle(last_ring, nx, nx, nx)

	numr	= sparx_alignment.Numrinit(first_ring, last_ring, rstep, "F")
	mask2D  = sparx_utilities.model_circle(last_ring,nx,nx) - sparx_utilities.model_circle(first_ring,nx,nx)

	fscmask = mask3D  #model_circle(last_ring,nx,nx,nx)  For a fancy mask circle would work better  PAP 7/21/11
	if CTF:
		pass#IMPORTIMPORTIMPORT from reconstruction import rec3D_MPI
		pass#IMPORTIMPORTIMPORT from filter         import filt_ctf
	pass#IMPORTIMPORTIMPORT else:	 from reconstruction import rec3D_MPI_noCTF

	if myid == main_node:
		if sparx_utilities.file_type(stack) == "bdb":
			pass#IMPORTIMPORTIMPORT from EMAN2db import db_open_dict
			dummy = EMAN2db.db_open_dict(stack, True)
		# horatio active_refactoring Jy51i1EwmLD4tWZ9_00000_1
		# active = EMUtil.get_all_attributes(stack, 'active')
		# list_of_particles = []
		# for im in xrange(len(active)):
		# 	if active[im]:  list_of_particles.append(im)
		# del active
		# nima = len(list_of_particles)
			
		nima = EMAN2_cppwrap.EMUtil.get_image_count(stack)
		list_of_particles = list(range(nima))
		
	else:
		nima = 0
	total_nima = sparx_utilities.bcast_number_to_all(nima, source_node = main_node)

	if myid != main_node:
		list_of_particles = [-1]*total_nima
	list_of_particles = sparx_utilities.bcast_list_to_all(list_of_particles, myid,  source_node = main_node)

	image_start, image_end = MPI_start_end(total_nima, number_of_proc, myid)
	# create a list of images for each node
	list_of_particles = list_of_particles[image_start: image_end]
	nima = len(list_of_particles)
	if debug:
		finfo.write("image_start, image_end: %d %d\n" %(image_start, image_end))
		finfo.flush()

	data = EMAN2_cppwrap.EMData.read_images(stack, list_of_particles)
	if fourvar:  original_data = []
	for im in range(nima):
		data[im].set_attr('ID', list_of_particles[im])
		if fourvar: original_data.append(data[im].copy())
		if CTF:
			ctf_params = data[im].get_attr("ctf")
			st = EMAN2_cppwrap.Util.infomask(data[im], mask2D, False)
			data[im] -= st[0]
			data[im] = sparx_filter.filt_ctf(data[im], ctf_params)
			data[im].set_attr('ctf_applied', 1)

	if debug:
		finfo.write( '%d loaded  \n' % nima )
		finfo.flush()
	if myid == main_node:
		# initialize data for the reference preparation function
		ref_data = [ mask3D, max(center,0), None, None, None, None ]
		# for method -1, switch off centering in user function

	pass#IMPORTIMPORTIMPORT from time import time

	#  this is needed for gathering of pixel errors
	disps = []
	recvcount = []
	for im in range(number_of_proc):
		if im == main_node :  disps.append(0)
		else:                 disps.append(disps[im-1] + recvcount[im-1])
		ib, ie = MPI_start_end(total_nima, number_of_proc, im)
		recvcount.append(ie-ib)

	pixer = [0.0]*nima
	cs = [0.0]*3
	total_iter = 0
	# do the projection matching
	for N_step in range(lstp):
		terminate = 0
		Iter = -1
		while Iter < max_iter-1 and terminate == 0:
			Iter += 1
			total_iter += 1
			if myid == main_node:
				start_time = time.time()
				sparx_utilities.print_msg("\nITERATION #%3d,  inner iteration #%3d\nDelta = %4.1f, an = %5.2f, xrange = %5.2f, yrange = %5.2f, step = %5.2f, delta psi = %5.2f, start psi = %5.2f\n"%(total_iter, Iter, delta[N_step], an[N_step], xrng[N_step],yrng[N_step],step[N_step],deltapsi[N_step],startpsi[N_step]))

			volft, kb = sparx_projection.prep_vol(vol)
			refrings = sparx_alignment.prepare_refrings(volft, kb, nx, delta[N_step], ref_a, sym, numr, True)
			del volft, kb
			if(an[N_step] > 0):
				# generate list of angles
				pass#IMPORTIMPORTIMPORT from alignment import generate_list_of_reference_angles_for_search
				list_of_reference_angles = \
				sparx_alignment.generate_list_of_reference_angles_for_search([[refrings[lr].get_attr("phi"), refrings[lr].get_attr("theta")] for lr in range(len(refrings))], sym=sym)			
			else:  list_of_reference_angles = [[1.0,1.0]]
			if myid == main_node:
				sparx_utilities.print_msg("Time to prepare rings: %d\n" % (time.time()-start_time))
				start_time = time.time()

			for im in range(nima):
				if deltapsi[N_step] > 0.0:
					pass#IMPORTIMPORTIMPORT from alignment import proj_ali_incore_delta
					peak, pixer[im] = sparx_alignment.proj_ali_incore_delta(data[im],refrings,numr,xrng[N_step],yrng[N_step],step[N_step],startpsi[N_step],deltapsi[N_step],finfo)						
				elif an[N_step] == -1:
					peak, pixer[im] = sparx_alignment.proj_ali_incore(data[im],refrings,numr,xrng[N_step],yrng[N_step],step[N_step],finfo, sym=sym)
				else:
					if apsi[N_step] == -1:
						peak, pixer[im] = sparx_alignment.proj_ali_incore_local(data[im],refrings, list_of_reference_angles, numr,xrng[N_step],yrng[N_step],step[N_step],an[N_step],finfo, sym = sym)
					else:
						peak, pixer[im] = sparx_alignment.proj_ali_incore_local_psi(data[im],refrings,numr,xrng[N_step],yrng[N_step],step[N_step],an[N_step],apsi[N_step],finfo)
				data[im].set_attr("previousmax", peak)

			if(an[N_step] > 0):  del list_of_reference_angles
			#=========================================================================

			if myid == main_node:
				sparx_utilities.print_msg("Time of alignment = %d\n"%(time.time()-start_time))
				start_time = time.time()

			#output pixel errors
			pass#IMPORTIMPORTIMPORT from mpi import mpi_gatherv
			recvbuf = mpi.mpi_gatherv(pixer, nima, mpi.MPI_FLOAT, recvcount, disps, mpi.MPI_FLOAT, main_node, mpi.MPI_COMM_WORLD)
			mpi.mpi_barrier(mpi.MPI_COMM_WORLD)
			terminate = 0
			if myid == main_node:
				recvbuf = list(map(float, recvbuf))
				pass#IMPORTIMPORTIMPORT from statistics import hist_list
				lhist = 20
				region, histo = sparx_statistics.hist_list(recvbuf, lhist)
				if region[0] < 0.0:  region[0] = 0.0
				msg = "      Histogram of pixel errors\n      ERROR       number of particles\n"
				sparx_utilities.print_msg(msg)
				for lhx in range(lhist):
					msg = " %10.3f     %7d\n"%(region[lhx], histo[lhx])
					sparx_utilities.print_msg(msg)
				# Terminate if 95% within 1 pixel error
				im = 0
				for lhx in range(lhist):
					if region[lhx] > 1.0: break
					im += histo[lhx]
				precn = 100*float(total_nima-im)/float(total_nima)
				msg = " Number of particles that changed orientations %7d, percentage of total: %5.1f\n"%(total_nima-im, precn)
				sparx_utilities.print_msg(msg)
				if precn <= termprec:  terminate = 1
				del region, histo
			del recvbuf
			terminate = mpi.mpi_bcast(terminate, 1, mpi.MPI_INT, 0, mpi.MPI_COMM_WORLD)
			terminate = int(terminate[0])

			if center == -1 and sym[0] == 'c':
				pass#IMPORTIMPORTIMPORT from utilities      import estimate_3D_center_MPI, rotate_3D_shift
				cs[0], cs[1], cs[2], dummy, dummy = sparx_utilities.estimate_3D_center_MPI(data, total_nima, myid, number_of_proc, main_node)
				if myid == main_node:
					msg = " Average center x = %10.3f        Center y = %10.3f        Center z = %10.3f\n"%(cs[0], cs[1], cs[2])
					sparx_utilities.print_msg(msg)
				if int(sym[1]) > 1:
					cs[0] = cs[1] = 0.0
					if myid == main_node:
						sparx_utilities.print_msg("For symmetry group cn (n>1), we only center the volume in z-direction\n")
				cs = mpi.mpi_bcast(cs, 3, mpi.MPI_FLOAT, main_node, mpi.MPI_COMM_WORLD)
				cs = [-float(cs[0]), -float(cs[1]), -float(cs[2])]
				sparx_utilities.rotate_3D_shift(data, cs)

			# write out headers, under MPI writing has to be done sequentially
			mpi.mpi_barrier(mpi.MPI_COMM_WORLD)
			par_str = ['xform.projection', 'previousmax', 'ID']
			if myid == main_node:
				if(sparx_utilities.file_type(stack) == "bdb"):
					pass#IMPORTIMPORTIMPORT from utilities import recv_attr_dict_bdb
					sparx_utilities.recv_attr_dict_bdb(main_node, stack, data, par_str, image_start, image_end, number_of_proc)
				else:
					pass#IMPORTIMPORTIMPORT from utilities import recv_attr_dict
					sparx_utilities.recv_attr_dict(main_node, stack, data, par_str, image_start, image_end, number_of_proc)
				sparx_utilities.print_msg("Time to write header information= %d\n"%(time.time()-start_time))
				start_time = time.time()
			else:	sparx_utilities.send_attr_dict(main_node, data, par_str, image_start, image_end)

			if CTF: vol, fscc = sparx_reconstruction.rec3D_MPI(data, snr, sym, fscmask, os.path.join(outdir, "resolution%04d"%(total_iter)), myid, main_node, npad = npad)
			else:   vol, fscc = sparx_reconstruction.rec3D_MPI_noCTF(data, sym, fscmask, os.path.join(outdir, "resolution%04d"%(total_iter)), myid, main_node, npad = npad)

			if myid == main_node:
				sparx_utilities.print_msg("3D reconstruction time = %d\n"%(time.time()-start_time))
				start_time = time.time()

			if fourvar:
			#  Compute Fourier variance
				for im in range(nima):
					original_data[im].set_attr( 'xform.projection', data[im].get_attr('xform.projection') )
				varf = sparx_statistics.varf3d_MPI(original_data, ssnr_text_file = os.path.join(outdir, "ssnr%04d"%(total_iter)), mask2D = None, reference_structure = vol, ou = last_ring, rw = 1.0, npad = 1, CTF = CTF, sign = 1, sym =sym, myid = myid)
				if myid == main_node:
					sparx_utilities.print_msg("Time to calculate 3D Fourier variance= %d\n"%(time.time()-start_time))
					start_time = time.time()
					varf = 1.0/varf
			else:  varf = None

			if myid == main_node:
				sparx_utilities.drop_image(vol, os.path.join(outdir, "vol%04d.hdf"%(total_iter)))
				ref_data[2] = vol
				ref_data[3] = fscc
				ref_data[4] = varf
				#  call user-supplied function to prepare reference image, i.e., center and filter it
				vol, cs = user_func(ref_data)
				sparx_utilities.drop_image(vol, os.path.join(outdir, "volf%04d.hdf"%(total_iter)))
				sparx_utilities.drop_image(vol, os.path.join(outdir, "volf.hdf"))

			del varf
			sparx_utilities.bcast_EMData_to_all(vol, myid, main_node)


	if myid == main_node: sparx_utilities.print_end_msg("ali3d_MPI")


def sali3d_base(stack, ref_vol = None, Tracker = None, rangle = 0.0, rshift = 0.0, mpi_comm = None, log = None):
	"""
		parameters: list of (all) projections | reference volume is optional, the data is shrank, 
		  the program does not know anything about shrinking| ...
		Data is assumed to be CTF multiplied and the ctf_applied flag to be set.
		The alignment done depends on nsoft:
					 nsoft = 0 & an = -1: exhaustive deterministic
					 nsoft = 0 & an > 0 : local deterministic
					 nsoft = 1 shc
					 nsoft >1  shc_multi
		
	"""

	pass#IMPORTIMPORTIMPORT from alignment       import Numrinit, prepare_refrings
	pass#IMPORTIMPORTIMPORT from alignment       import proj_ali_incore,  proj_ali_incore_zoom,  proj_ali_incore_local, proj_ali_incore_local_zoom
	pass#IMPORTIMPORTIMPORT from alignment       import shc, center_projections_3D
	pass#IMPORTIMPORTIMPORT from utilities       import bcast_number_to_all, bcast_EMData_to_all, 	wrap_mpi_gatherv, wrap_mpi_bcast, model_blank
	pass#IMPORTIMPORTIMPORT from utilities       import get_im, file_type, model_circle, get_input_from_string, get_params_proj, set_params_proj, pad
	pass#IMPORTIMPORTIMPORT from utilities       import even_angles
	pass#IMPORTIMPORTIMPORT from mpi             import mpi_bcast, mpi_comm_size, mpi_comm_rank, MPI_FLOAT, MPI_COMM_WORLD, mpi_barrier, mpi_reduce, MPI_INT, MPI_SUM
	pass#IMPORTIMPORTIMPORT from projection      import prep_vol
	pass#IMPORTIMPORTIMPORT from statistics      import hist_list
	pass#IMPORTIMPORTIMPORT from applications    import MPI_start_end
	pass#IMPORTIMPORTIMPORT from filter          import filt_ctf, filt_table
	pass#IMPORTIMPORTIMPORT from global_def      import Util
	pass#IMPORTIMPORTIMPORT from fundamentals    import resample, fshift
	pass#IMPORTIMPORTIMPORT from multi_shc       import shc_multi
	#from development     import do_volume_mrk01
	pass#IMPORTIMPORTIMPORT import user_functions
	pass#IMPORTIMPORTIMPORT from EMAN2           import EMUtil, EMData
	pass#IMPORTIMPORTIMPORT import types
	pass#IMPORTIMPORTIMPORT from time            import time

	nsoft            = Tracker["nsoft"]
	saturatecrit     = Tracker["saturatecrit"]
	pixercutoff      = Tracker["pixercutoff"]
	zoom             = Tracker["zoom"]
	center           = Tracker["constants"]["center"]
	CTF              = Tracker["constants"]["CTF"]
	ref_a            = Tracker["constants"]["ref_a"]
	rstep            = Tracker["constants"]["rs"]
	sym              = Tracker["constants"]["sym"]
	first_ring       = 1
	last_ring        = Tracker["radius"]
	xr               = Tracker["xr"]
	yr               = Tracker["yr"]
	ts               = Tracker["ts"]
	an               = Tracker["an"]
	delta            = Tracker["delta"]
	max_iter         = Tracker["maxit"]

	if mpi_comm == None: mpi_comm = mpi.MPI_COMM_WORLD

	if log == None:
		pass#IMPORTIMPORTIMPORT from logger import Logger
		log = sparx_logger.Logger()

	number_of_proc = mpi.mpi_comm_size(mpi_comm)
	myid           = mpi.mpi_comm_rank(mpi_comm)
	main_node      = 0

	if myid == main_node:
		log.add("Start sali3d_base, nsoft = %1d"%nsoft)

	xrng        = sparx_utilities.get_input_from_string(xr)
	if  yr == "-1":  yrng = xrng
	else          :  yrng = sparx_utilities.get_input_from_string(yr)
	step        = sparx_utilities.get_input_from_string(ts)
	delta       = sparx_utilities.get_input_from_string(delta)
	lstp = min(len(xrng), len(yrng), len(step), len(delta))
	if an == "-1":
		an = [-1] * lstp
	else:
		an = sparx_utilities.get_input_from_string(an)

	if( type(stack) is bytes ):
		if myid == main_node:
			total_nima = EMAN2_cppwrap.EMUtil.get_image_count( stack )
		else:
			total_nima = 0
		total_nima = sparx_utilities.wrap_mpi_bcast(total_nima, main_node, mpi_comm)
		list_of_particles = list(range(total_nima))
		image_start, image_end = MPI_start_end(total_nima, number_of_proc, myid)
		# create a list of images for each node
		list_of_particles = list_of_particles[image_start: image_end]
		nima = len(list_of_particles)

	else:
		list_of_particles = list(range(len(stack)))
		nima = len(list_of_particles)
		total_nima = len(list_of_particles)
		total_nima = mpi.mpi_reduce(total_nima, 1, mpi.MPI_INT, mpi.MPI_SUM, 0, mpi.MPI_COMM_WORLD)
		total_nima = mpi.mpi_bcast(total_nima, 1, mpi.MPI_INT, 0, mpi.MPI_COMM_WORLD)
		total_nima = int(total_nima[0])


	if myid == 0 or myid == 5 or myid == 14:
		"""Multiline Comment4"""
	else:
		finfo = None

	if( myid == main_node):
		if( type(stack) is bytes ):  mask2D = sparx_utilities.get_im(stack, list_of_particles[0])
		else:                                   mask2D = stack[list_of_particles[0]]
		nx = mask2D.get_xsize()
	else:  nx = 0
	nx  = sparx_utilities.bcast_number_to_all(nx, source_node = main_node)
	mx = 2*nx
	if last_ring < 0:	last_ring = int(nx/2) - 2

	numr	= sparx_alignment.Numrinit(first_ring, last_ring, rstep, "F")

	data = [None]*nima
	for im in range(nima):
		if( type(stack) is bytes ):  data[im] = sparx_utilities.get_im(stack, list_of_particles[im])
		else:                                   data[im] = stack[list_of_particles[im]]
	mpi.mpi_barrier(mpi_comm)


	if myid == main_node:
		start_time = time.time()

	#  Read	template volume if provided or reconstruct it
	"""Multiline Comment5"""
	vol = ref_vol
	# log
	if myid == main_node:
		log.add("Setting of reference 3D reconstruction time = %10.1f\n"%(time.time()-start_time))
		start_time = time.time()


	pixer = [0.0]*nima
	#historyofchanges = [0.0, 0.5, 1.0]
	#par_r = [[] for im in list_of_particles ]
	cs = [0.0]*3
	total_iter = 0
	# do the projection matching,  it has a loop over iterations here, 
	#  but it can only do one iteration as many settings are done in meridien.  Perturbations are a good example, there is one per each iteration.
	if zoom: lstp = 1
	for N_step in range(lstp):

		terminate = 0
		Iter = 0
		while Iter < max_iter and terminate == 0:

			Iter += 1
			total_iter += 1


			mpi.mpi_barrier(mpi_comm)
			if myid == main_node:
				log.add("ITERATION #%3d,  inner iteration #%3d"%(total_iter, Iter))
				log.add("Delta = %7.4f, an = %7.4f, xrange = %7.4f, yrange = %7.4f, step = %7.4f   %7.4f  %7.4f\n"%\
							(delta[N_step], an[N_step], xrng[N_step], yrng[N_step], step[N_step], rshift, rangle))
				start_time = time.time()

			#=========================================================================
			# prepare reference angles
			ref_angles = sparx_utilities.even_angles(delta[N_step], symmetry=sym, method = ref_a, phiEqpsi = "Zero")
			#  Modify 0,0,0 s it can be properly inverted
			if( ref_angles[0][0] == 0.0  and ref_angles[0][1] == 0.0 ):
				ref_angles[0][0] = 0.01
				ref_angles[0][1] = 0.01
			if( rangle > 0.0 ):
				# shake
				pass#IMPORTIMPORTIMPORT from utilities import rotate_shift_params
				ref_angles = sparx_utilities.rotate_shift_params(anglelist, [ delta[N_step]*rangle, delta[N_step]*rangle, delta[N_step]*rangle ])

			#=========================================================================
			# build references
			volft, kb = sparx_projection.prep_vol(vol)
			refrings = sparx_alignment.prepare_refrings(volft, kb, nx, delta[N_step], ref_angles, sym, numr, MPI=mpi_comm, phiEqpsi = "Zero")
			del volft, kb
			#=========================================================================		

			if myid == main_node:
				log.add("Time to prepare rings: %10.1f\n" % (time.time()-start_time))
				start_time = time.time()

			#=========================================================================
			#  there is no need for previousmax for deterministic searches
			if total_iter == 1 and nsoft > 0:
				if(an[N_step] < 0.0):
					# adjust params to references, calculate psi+shifts, calculate previousmax
					# generate list of angles
					pass#IMPORTIMPORTIMPORT from alignment import generate_list_of_reference_angles_for_search
					list_of_reference_angles = \
					sparx_alignment.generate_list_of_reference_angles_for_search([[refrings[lr].get_attr("phi"), refrings[lr].get_attr("theta")] for lr in range(len(refrings))], sym=sym)			
					for im in range(nima):
						previousmax = data[im].get_attr_default("previousmax", -1.0e23)
						if(previousmax == -1.0e23):
							peak, pixer[im] = sparx_alignment.proj_ali_incore_local(data[im], refrings, list_of_reference_angles, numr, \
									xrng[N_step], yrng[N_step], step[N_step], delta[N_step]*2.5, sym = sym)
							data[im].set_attr("previousmax", peak)
					del list_of_reference_angles
				else:
					#  Here it is supposed to be shake and bake for local SHC, but it would have to be signaled somehow
					for im in range(nima):
						data[im].set_attr("previousmax", -1.0e23)
				if myid == main_node:
					log.add("Time to calculate first psi+shifts+previousmax: %10.1f\n" % (time.time()-start_time))
					start_time = time.time()
			#=========================================================================

			mpi.mpi_barrier(mpi_comm)
			if myid == main_node:  start_time = time.time()
			#=========================================================================
			# alignment
			#number_of_checked_refs = 0
			par_r = [0]*max(2,(nsoft+1))
			if(an[N_step] > 0):
				# generate list of angles
				pass#IMPORTIMPORTIMPORT from alignment import generate_list_of_reference_angles_for_search
				list_of_reference_angles = \
				sparx_alignment.generate_list_of_reference_angles_for_search([[refrings[lr].get_attr("phi"), refrings[lr].get_attr("theta")] for lr in range(len(refrings))], sym=sym)			
			else:  list_of_reference_angles = [[1.0,1.0]]
			for im in range(nima):
				if Tracker["constants"]["pwsharpening"] :
					#  High-pass filtration of data[im]
					try:
						stmp = data[im].get_attr("ptcl_source_image")
					except:
						try:
							stmp = data[im].get_attr("ctf")
							stmp = round(stmp.defocus,4)
						except:
							sparx_global_def.ERROR("Either ptcl_source_image or ctf has to be present in the header.","meridien",1, myid)
					try:
						indx = Tracker["bckgnoise"][1].index(stmp)
					except:
						sparx_global_def.ERROR("Problem with indexing ptcl_source_image.","meridien",1, myid)

					tempdata = EMAN2_cppwrap.Util.window(sparx_utilities.pad(sparx_filter.filt_table(data[im],[Tracker["bckgnoise"][0][i,indx] for i in range(nx)]), mx, mx,1,0.0), nx, nx)
				else:  tempdata = data[im].copy()
				if(nsoft == 0):
					if(an[N_step] == -1):
						#  In zoom option each projection goes through shift zoom alignment
						if  zoom: peak, pixer[im] = sparx_alignment.proj_ali_incore_zoom(tempdata, refrings, numr, \
														xrng, yrng, step, finfo = finfo, sym=sym)
						else:  peak, pixer[im] = sparx_alignment.proj_ali_incore(tempdata, refrings, numr, \
												xrng[N_step], yrng[N_step], step[N_step], finfo = finfo, sym=sym, delta_psi = delta[N_step], rshift = rshift*xrng[N_step])
					else:
						if  zoom: peak, pixer[im] = sparx_alignment.proj_ali_incore_local_zoom(tempdata, refrings, list_of_reference_angles, numr, \
									xrng, yrng, step, an, finfo = finfo, sym=sym)
						else:  peak, pixer[im] = sparx_alignment.proj_ali_incore_local(tempdata, refrings, list_of_reference_angles, numr, \
									xrng[N_step], yrng[N_step], step[N_step], an[N_step], finfo = finfo, sym=sym, delta_psi = delta[N_step], rshift = rshift)
					if(pixer[im] == 0.0):  par_r[0] += 1
				elif(nsoft == 1):
					tempdata.set_attr("previousmax", data[im].get_attr("previousmax"))
					peak, pixer[im], number_of_checked_refs, iref = \
						sparx_alignment.shc(tempdata, refrings, list_of_reference_angles, numr, xrng[N_step], yrng[N_step], step[N_step], an[N_step], sym, finfo = finfo)
					if(pixer[im] == 0.0):  par_r[0] += 1
					data[im].set_attr("previousmax", tempdata.get_attr("previousmax"))
				elif(nsoft > 1):
					#  This is not functional
					peak, pixer[im], checked_refs, number_of_peaks = sparx_multi_shc.shc_multi(data[im], refrings, numr, \
												xrng[N_step], yrng[N_step], step[N_step], an[N_step], nsoft, sym, finfo = finfo)
					par_r[number_of_peaks] += 1
					#number_of_checked_refs += checked_refs
				data[im].set_attr("xform.projection", tempdata.get_attr("xform.projection"))
			del tempdata
			if(an[N_step] > 0):  del list_of_reference_angles
			#=========================================================================
			mpi.mpi_barrier(mpi_comm)
			if myid == main_node:
				#print  data[0].get_attr_dict()
				log.add("Time of alignment = %10.1f\n"%(time.time()-start_time))
				start_time = time.time()
			#=========================================================================
			#  Pixer errors available here are useless as they are done for shifts on the reduced image size.
			#output pixel errors, check stop criterion
			all_pixer = sparx_utilities.wrap_mpi_gatherv(pixer, 0, mpi_comm)
			par_r = mpi.mpi_reduce(par_r, len(par_r), mpi.MPI_INT, mpi.MPI_SUM, 0, mpi.MPI_COMM_WORLD)
			#total_checked_refs = wrap_mpi_gatherv([number_of_checked_refs], main_node, mpi_comm)
			terminate = 0
			if myid == main_node:
				#total_checked_refs = sum(total_checked_refs)
				if(nsoft < 2):  par_r[1] = total_nima - par_r[0]
				log.add("=========== Number of better orientations found ==============")
				for lhx in range(len(par_r)):
					msg = "            %5d     %7d"%(lhx, par_r[lhx])
					log.add(msg)
				log.add("_______________________________________________________")
				changes = par_r[0]/float(total_nima)
				"""Multiline Comment6"""

				lhist = 20
				region, histo = sparx_statistics.hist_list(all_pixer, lhist)
				log.add("=========== Histogram of pixel errors ==============")
				for lhx in range(lhist):
					msg = "          %10.3f     %7d"%(region[lhx], histo[lhx])
					log.add(msg)
				log.add("____________________________________________________")
				if(nsoft<2 and terminate == 0):
					lhx = 0
					for msg in all_pixer:
						if(msg < pixercutoff): lhx += 1
					lhx = float(lhx)/float(total_nima)
					log.add(">>> %4.2f images had pixel error <%5.2f"%(lhx,pixercutoff))
					"""Multiline Comment7"""
			terminate = True  #wrap_mpi_bcast(terminate, main_node, mpi_comm)
			#=========================================================================
			mpi.mpi_barrier(mpi_comm)
			if myid == main_node:
				#print  data[0].get_attr_dict()
				log.add("Time to compute histograms = %10.1f\n"%(time.time()-start_time))
				start_time = time.time()


			#=========================================================================
			mpi.mpi_barrier(mpi_comm)
			if( terminate or (Iter == max_iter) ):
				# gather parameters
				params = []
				for im in range(nima):
					t = sparx_utilities.get_params_proj(data[im])
					params.append( [t[0], t[1], t[2], t[3], t[4]] )
				params = sparx_utilities.wrap_mpi_gatherv(params, main_node, mpi_comm)
			# centering and volume reconstruction if not terminating
			else:
				#=========================================================================
				# centering
				if center == -1 and sym[0] == 'c':
					pass#IMPORTIMPORTIMPORT from utilities      import estimate_3D_center_MPI, rotate_3D_shift
					cs[0], cs[1], cs[2], dummy, dummy = sparx_utilities.estimate_3D_center_MPI(data, total_nima, myid, number_of_proc, main_node, mpi_comm=mpi_comm)
					if myid == main_node:
						msg = " Average center x = %10.3f        Center y = %10.3f        Center z = %10.3f\n"%(cs[0], cs[1], cs[2])
						log.add(msg)
					if int(sym[1]) > 1:
						cs[0] = cs[1] = 0.0
						if myid == main_node:
							log.add("For symmetry group cn (n>1), we only center the volume in z-direction\n")
					cs = mpi.mpi_bcast(cs, 3, mpi.MPI_FLOAT, main_node, mpi_comm)
					cs = [-float(cs[0]), -float(cs[1]), -float(cs[2])]
					sparx_utilities.rotate_3D_shift(data, cs)
				#=========================================================================
				if myid == main_node:
					start_time = time.time()
				#vol = do_volume_mrk01(data, Tracker, total_iter, mpi_comm)
				ref_data = [data, Tracker, total_iter, mpi_comm]
				user_func = Tracker["constants"] ["user_func"]
				vol = user_func(ref_data)
				#if myid == main_node:  vol.write_image('soft/smvol%04d.hdf'%total_iter)
				# log
				if myid == main_node:
					log.add("3D reconstruction time = %10.1f\n"%(time.time()-start_time))
					start_time = time.time()
			#=========================================================================

			"""Multiline Comment8"""


	if myid == main_node:
		log.add("Finish sali3d_base, nsoft = %1d"%nsoft)
	return params


"""Multiline Comment9"""

# from development import sali3d_base_h_01
# sali3d_base = sali3d_base_horatio_01


def slocal_ali3d_base(stack, templatevol, Tracker, mpi_comm = None, log= None, chunk = -1.0, debug = False ):
	"""

	"""
	pass#IMPORTIMPORTIMPORT from alignment        import eqproj_cascaded_ccc
	pass#IMPORTIMPORTIMPORT from filter           import filt_ctf
	pass#IMPORTIMPORTIMPORT from projection       import prep_vol
	pass#IMPORTIMPORTIMPORT from fundamentals     import resample
	pass#IMPORTIMPORTIMPORT from utilities        import bcast_number_to_all, model_circle, get_params_proj, set_params_proj
	pass#IMPORTIMPORTIMPORT from utilities        import bcast_EMData_to_all, bcast_list_to_all, send_attr_dict, wrap_mpi_bcast, wrap_mpi_gatherv
	pass#IMPORTIMPORTIMPORT from utilities        import get_image, drop_image, file_type, get_im, get_input_from_string, model_blank
	pass#IMPORTIMPORTIMPORT from utilities        import amoeba_multi_level, rotate_3D_shift, estimate_3D_center_MPI
	pass#IMPORTIMPORTIMPORT from utilities        import print_begin_msg, print_end_msg, print_msg
	#from development      import do_volume_mrk01
	pass#IMPORTIMPORTIMPORT import user_functions
	pass#IMPORTIMPORTIMPORT from statistics       import varf3d_MPI
	pass#IMPORTIMPORTIMPORT from math             import pi
	pass#IMPORTIMPORTIMPORT from mpi              import mpi_bcast, mpi_comm_size, mpi_comm_rank, MPI_FLOAT, MPI_COMM_WORLD, mpi_barrier
	pass#IMPORTIMPORTIMPORT from mpi              import mpi_reduce, MPI_INT, MPI_SUM
	pass#IMPORTIMPORTIMPORT from EMAN2 import Processor
	pass#IMPORTIMPORTIMPORT from EMAN2 import Vec2f, Transform
	pass#IMPORTIMPORTIMPORT import os
	pass#IMPORTIMPORTIMPORT import sys
	pass#IMPORTIMPORTIMPORT import types

	maxit         = Tracker["maxit"]
	saturatecrit  = Tracker["saturatecrit"]
	pixercutoff   = Tracker["pixercutoff"]
	ou            = Tracker["radius"]
	ts            = sparx_utilities.get_input_from_string(Tracker["ts"])[0]
	delta         = sparx_utilities.get_input_from_string(Tracker["delta"])[0]
	sym           = Tracker["constants"]["sym"]
	sym           = sym[0].lower() + sym[1:]
	center        = Tracker["constants"]["center"]
	CTF           = Tracker["constants"]["CTF"]
	fourvar = False

	if log == None:
		pass#IMPORTIMPORTIMPORT from logger import Logger
		log = sparx_logger.Logger()


	if mpi_comm == None: mpi_comm = mpi.MPI_COMM_WORLD

	number_of_proc = mpi.mpi_comm_size(mpi_comm)
	myid           = mpi.mpi_comm_rank(mpi_comm)
	main_node = 0

	if myid == main_node:
		log.add("Start slocal_ali3d_base")

	"""Multiline Comment10"""
	if debug:
		outdir = "debug_outdir"
		if myid == main_node:  os.system("mkdir   "+outdir)
		pass#IMPORTIMPORTIMPORT from time import sleep
		while not os.path.exists(outdir):
			print("Node ",myid,"  waiting...")
			time.sleep(5)

		finfo = open(os.path.join(outdir, "progress%04d"%myid), 'w')
	else:
		finfo = None

	last_ring   = int(ou)
	center      = int(center)

	if( type(stack) is bytes ):
		if myid == main_node:
			if(sparx_utilities.file_type(stack) == "bdb"):
				pass#IMPORTIMPORTIMPORT from EMAN2db import db_open_dict
				dummy = EMAN2db.db_open_dict(stack, True)

			nima = EMAN2_cppwrap.EMUtil.get_image_count(stack)
			list_of_particles = list(range(nima))
			total_nima = len(list_of_particles)
		else:
			list_of_particles = None
			total_nima = 0
		total_nima = sparx_utilities.wrap_mpi_bcast(total_nima, main_node, mpi_comm)
		total_nima = int(total_nima[0])
		list_of_particles = sparx_utilities.wrap_mpi_bcast(list_of_particles, main_node, mpi_comm)
		if myid == main_node:
			particle_ids = [0]*total_nima
			for i in range(total_nima):  particle_ids[i] = list_of_particles[i]
		image_start, image_end = MPI_start_end(total_nima, number_of_proc, myid)
		# create a list of images for each node
		list_of_particles = list_of_particles[image_start: image_end]
		nima = len(list_of_particles)

	else:
		list_of_particles = list(range(len(stack)))
		nima = len(list_of_particles)
		total_nima = len(list_of_particles)
		total_nima = mpi.mpi_reduce(total_nima, 1, mpi.MPI_INT, mpi.MPI_SUM, 0, mpi.MPI_COMM_WORLD)
		total_nima = mpi.mpi_bcast(total_nima, 1, mpi.MPI_INT, 0, mpi.MPI_COMM_WORLD)
		total_nima = int(total_nima[0])
		image_start = 0
		image_end   = nima

	if(myid == main_node):
		if( type(stack) is bytes ):  dataim = sparx_utilities.get_im(stack, list_of_particles[0])
		else:                                   dataim = stack[list_of_particles[0]]
		nx      = dataim.get_xsize()
		if CTF :
			ctf_applied = dataim.get_attr_default('ctf_applied', 0)
			if ctf_applied >0 :  sparx_global_def.ERROR("Projection data cannot be CTF-applied","local_ali3d_base",1,myid)
	else:
		nx = 0

	nx  = sparx_utilities.bcast_number_to_all(nx, source_node = main_node)

	if last_ring < 0:	last_ring = int(nx/2) - 2
	mask2D  = sparx_utilities.model_circle(last_ring, nx, nx)

	dataim = [None]*nima
	for im in range(nima):
		if( type(stack) is bytes ):  dataim[im] = sparx_utilities.get_im(stack, list_of_particles[im])
		else:                                   dataim[im] = stack[list_of_particles[im]]
		dataim[im].set_attr('ID', list_of_particles[im])
		if CTF :
			st = EMAN2_cppwrap.Util.infomask(dataim[im], mask2D, False)
			dataim[im] -= st[0]


	if chunk <= 0.0:  chunk = 1.0
	n_of_chunks = int(1.0/chunk)

	"""Multiline Comment11"""

	pass#IMPORTIMPORTIMPORT import  types
	if Tracker["constants"]["mask3D"]:
		if type(Tracker["constants"]["mask3D"]) is bytes:
			if myid == main_node:
				mask3D = sparx_utilities.get_im(Tracker["constants"]["mask3D"])
			else:
				mask3D = sparx_utilities.model_blank(nx, nx, nx)
		else:
			mask3D = Tracker["constants"]["mask3D"].copy()
		if myid == main_node:
			nxm = mask3D.get_xsize()
			if( nxm > nx ):
				pass#IMPORTIMPORTIMPORT from fundamentals import rot_shift3D
				mask3D = EMAN2_cppwrap.Util.window(sparx_fundamentals.rot_shift3D(mask3D,scale=float(nx)/float(nxm)),nx,nx,nx)
				nxm = mask3D.get_xsize()
				assert(nx == nxm)
			else:
				mask3D = sparx_utilities.model_blank(nx, nx, nx)
		sparx_utilities.bcast_EMData_to_all(mask3D, myid, main_node)
	else:
		mask3D = sparx_utilities.model_circle(last_ring, nx, nx, nx)

	#  Read	template volume if provided
	if templatevol:
		if type(templatevol) is bytes:
			if myid == main_node:
				vol = sparx_utilities.get_im(templatevol)
				nxm = vol.get_xsize()
				if( nxm > nx ):
					pass#IMPORTIMPORTIMPORT from fundamentals import rot_shift3D
					vol = EMAN2_cppwrap.Util.window(sparx_fundamentals.rot_shift3D(vol,scale=float(nx)/float(nxm)),nx,nx,nx)
					nxm = vol.get_xsize()
					assert(nx == nxm)
			else:
				vol = sparx_utilities.model_blank(nx, nx, nx)
		else:
			if myid == main_node:
				nxm = templatevol.get_xsize()
				if( nxm > nx ):
					pass#IMPORTIMPORTIMPORT from fundamentals import rot_shift3D
					vol = EMAN2_cppwrap.Util.window(sparx_fundamentals.rot_shift3D(templatevol,scale=float(nx)/float(nxm)),nx,nx,nx)
					nxm = vol.get_xsize()
					assert(nx == nxm)
				else:
					vol = templatevol.copy()
			else:
				vol = sparx_utilities.model_blank(nx, nx, nx)
		sparx_utilities.bcast_EMData_to_all(vol, myid, main_node)
		del templatevol
		#  Do the 3D
		#vol = do_volume_mrk01(vol, Tracker, 0, mpi_comm)
		ref_data = [vol, Tracker, 0, mpi_comm]
		user_func = Tracker["constants"] ["user_func"]
		vol = user_func(ref_data)
	else:
		vol = None

	if debug:
		finfo.write( "image_start, image_end: %d %d\n" %(image_start, image_end) )
		finfo.flush()

	if debug:
		finfo.write("  chunk = "+str(chunk)+"   ")
		finfo.write("\n")
		finfo.flush()
		finfo.write("  Number of chunks = "+str(n_of_chunks)+"   ")
		finfo.write("\n")
		finfo.flush()

	del list_of_particles

	if debug:
		finfo.write("  First image on this processor: "+str(image_start)+"   ")
		finfo.write("  Last  image on this processor: "+str(image_end)+"   ")
		finfo.write("\n")
		finfo.flush()

	if myid == main_node:
		# initialize data for the reference preparation function
		ref_data = [ mask3D, max(center,0), None, None, None ]
		# for method -1, switch off centering in user function
		ref_data.append( None )

	pass#IMPORTIMPORTIMPORT from time import time
	if myid == main_node:
		log.add("Dimensions used (nx, last_ring)  %5d    %5d\n"%(nx, last_ring))
		start_time = time.time()


		
	M = nx
	npad = 2
	N = M*npad
	K = 6
	alpha = 1.75
	r = M/2
	v = K/2.0/N
	params = {"filter_type": EMAN2_cppwrap.Processor.fourier_filter_types.KAISER_SINH_INVERSE, "alpha":alpha, "K":K, "r":r, "v":v, "N":N}

	data = [None]*7
	data[3] = mask2D
	cs = [0.0]*3

	for iteration in range(maxit):
		if myid == main_node:
			start_time = time.time()
			log.add("ITERATION #%3d\n"%(iteration+1))
		if debug:
			finfo.write("  iteration = "+str(iteration)+"   ")
			finfo.write("\n")
			finfo.flush()
		pixer = [0.0]*nima
		for ic in range(n_of_chunks):
			# In the very first step the volume has to be computed if it was not provided by the user
			if( ((iteration > 0) and (ic > 0)) or vol == None):
				if(center == -1 and sym[0] == 'c'):
					if debug:
						finfo.write("  begin centering \n")
						finfo.flush()
					cs[0], cs[1], cs[2], dummy, dummy = sparx_utilities.estimate_3D_center_MPI(dataim, total_nima, myid, number_of_proc, main_node)
					cs = mpi.mpi_bcast(cs, 3, mpi.MPI_FLOAT, main_node, mpi_comm)
					cs = [-float(cs[0]), -float(cs[1]), -float(cs[2])]
					if int(sym[1]) > 1:
						cs[0] = cs[1] = 0.0
					sparx_utilities.rotate_3D_shift(dataim, cs)
					if myid == main_node:
						msg = "Average center x = %10.3f        Center y = %10.3f        Center z = %10.3f\n"%(cs[0], cs[1], cs[2])
						log.add(msg)
						log.add("Time to center = %d\n"%(time.time()-start_time))
						start_time = time.time()
					# compute updated 3D before each chunk
					# resolution
				if debug:
					finfo.write("  begin reconstruction = "+str(image_start))
					finfo.write("\n")
					finfo.flush()

				#  Do the 3D
				ref_data = [dataim, Tracker, iteration, mpi_comm]
				user_func = Tracker["constants"] ["user_func"]
				vol = user_func(ref_data)

				if myid == main_node:
					#drop_image(vol, os.path.join(outdir, "vol%03d_%03d.hdf"%(iteration, ic) ))
					log.add("3D reconstruction time = %d"%(time.time()-start_time))
					start_time = time.time()
				if debug:
					finfo.write("  done reconstruction = "+str(image_start))
					finfo.write("\n")
					finfo.flush()

				if fourvar:
				#  Compute Fourier variance
					varf = sparx_statistics.varf3d_MPI(dataim, ssnr_text_file = os.path.join(outdir, "ssnr%03d_%03d"%(iteration, ic)), mask2D = None, reference_structure = vol, ou = ou, rw = 1.0, npad = 1, CTF = CTF, sign = 1, sym =sym, myid = myid)
					if myid == main_node:
						varf = 1.0/varf
						sparx_utilities.print_msg("Time to calculate 3D Fourier variance= %d\n"%(time.time()-start_time))
						start_time = time.time()
				else:  varf = None

			if CTF:
				previous_defocus = -1.0
				#vol = fft(pad(vol, N, N, N))
			else:
				data[0], data[1] = sparx_projection.prep_vol(vol)

			image_start_in_chunk = ic*nima/n_of_chunks
			image_end_in_chunk   = (ic+1)*nima/n_of_chunks
			if debug:
				finfo.write("Chunk "+str(ic)+"   Number of images in this chunk: "+str(image_end_in_chunk-image_start_in_chunk)+"\n")
				finfo.write("First image in this chunk: "+str(image_start_in_chunk)+"   Last image in this chunk: "+str(image_end_in_chunk-1)+"\n")
				finfo.flush()
			for imn in range(image_start_in_chunk, image_end_in_chunk):
				if CTF:
					ctf_params = dataim[imn].get_attr( "ctf" )
					if ctf_params.defocus != previous_defocus:
						previous_defocus = ctf_params.defocus
						data[0], data[1] = sparx_projection.prep_vol(sparx_filter.filt_ctf(vol, ctf_params))

				#  Insert high-pass filtration of data[im]
				###data[2] = Util.window(pad(filt_table(dataimn[im],bckgnoise), mx, mx,1,0.0), nx, nx)
				data[2] = dataim[imn]
				if ts > 0.0:
					refi = dataim[imn].FourInterpol(nx*2, nx*2, 1, True)
					data[4] = EMAN2_cppwrap.Processor.EMFourierFilter(refi, params)

				#phi, theta, psi, tx, ty = get_params_proj(dataim[imn])
				t1 = dataim[imn].get_attr("xform.projection")
				dp = t1.get_params("spider")
				atparams = [dp["phi"], dp["theta"], dp["psi"]]
				data[5]  = [dp["tx"], dp["ty"]]
				if debug:
					# we have to distiguish between no shift situation, which is done through ccc, and shift, which is done using gridding in 2D
					if(ts == 0.0):  data[6] = 0.0
					else:           data[6] = -1.0#ts#-1.0
					initial, dummy = sparx_alignment.eqproj_cascaded_ccc(atparams, data)  # this is if we need initial discrepancy
					finfo.write("Image "+str(imn)+"\n")
					finfo.write('Old  %6.1f  %6.1f  %6.1f   %5.2f  %5.2f  %11.4e\n'%(atparams[0],atparams[1],atparams[2], -dummy[0], -dummy[1], initial))
				# change signs of shifts for projections
				data[6] = ts
				#from random import random
				#data[5] = [(random()-0.5)*2,(random()-0.5)*2]  #  HERE !!!!!!!!!!!

				weight_phi = max(delta, delta*abs((atparams[1]-90.0)/180.0*numpy.pi))
				optm_params = sparx_utilities.amoeba_multi_level(atparams, [weight_phi, delta, weight_phi], sparx_alignment.eqproj_cascaded_ccc, 1.0, 1.e-2, 500, data)
				optm_params[0].append(optm_params[3][0])
				optm_params[0].append(optm_params[3][1])
				optm_params[0][3] *= -1
				optm_params[0][4] *= -1

				if debug:
					finfo.write('New  %6.1f  %6.1f  %6.1f   %5.2f  %5.2f  %11.4e  %4d\n'%(optm_params[0][0], optm_params[0][1], optm_params[0][2], optm_params[0][3], optm_params[0][4], optm_params[1], optm_params[2]))
					finfo.flush()

				#from sys import exit
				#exit()
				t2 = EMAN2_cppwrap.Transform({"type":"spider","phi":optm_params[0][0],"theta":optm_params[0][1],"psi":optm_params[0][2]})
				t2.set_trans(EMAN2_cppwrap.Vec2f(-optm_params[0][3], -optm_params[0][4]))
				dataim[imn].set_attr("xform.projection", t2)
				pass#IMPORTIMPORTIMPORT from pixel_error import max_3D_pixel_error
				pixer[imn] = sparx_pixel_error.max_3D_pixel_error(t1, t2, last_ring)
				#set_params_proj(dataim[imn], optm_params[0])
				#if( myid == main_node and imn%4 == 0):
				#	log.add( "Time to process %6d particles : %d\n" % (imn, time()-start_time) )
				#	start_time = time()
			if( myid == main_node ):
				log.add( "Time to process %6d particles : %d" % (image_end_in_chunk-image_start_in_chunk, time.time()-start_time) )
				start_time = time.time()

			# release memory
			data[0] = None


		#output pixel errors after all headers were processed
		pass#IMPORTIMPORTIMPORT from mpi import mpi_gatherv
		pixer = sparx_utilities.wrap_mpi_gatherv(pixer, main_node, mpi_comm)
		mpi.mpi_barrier(mpi_comm)
		terminate = 0
		if(myid == main_node):
			pixer = list(map(float, pixer))
			pass#IMPORTIMPORTIMPORT from statistics import hist_list
			lhist = 20
			region, histo = sparx_statistics.hist_list(pixer, lhist)
			log.add(" ")
			log.add("=========== Histogram of pixel errors ==============")
			for lhx in range(lhist):
				msg = "          %10.3f     %7d"%(region[lhx], histo[lhx])
				log.add(msg)
			log.add("____________________________________________________\n")


			# Terminate if saturatecrit% within pixercutoff pixel error
			im = 0
			for lhx in range(lhist):
				if(region[lhx] > pixercutoff): break
				im += histo[lhx]
			lhx = im/float(total_nima)
			"""Multiline Comment12"""
			del region, histo
		del pixer
		#terminate = mpi_bcast(terminate, 1, MPI_INT, 0, mpi_comm)
		#terminate = int(terminate[0])
		#if terminate:  break
		break

	del vol
	# gather parameters
	params = []
	for im in dataim:
		t = sparx_utilities.get_params_proj(im)
		params.append( [t[0], t[1], t[2], t[3], t[4]] )
	params = sparx_utilities.wrap_mpi_gatherv(params, main_node, mpi_comm)

	if( myid == main_node ):
		"""Multiline Comment13"""
		log.add("Finish local_ ali3d_base")

	return  params


"""Multiline Comment14"""

# Auxiliary function to compute number of cones in ali3dlocal
def computenumberofrefs(x, dat):
	pass#IMPORTIMPORTIMPORT from utilities import even_angles
	#  dat = [sym, desired number of refs, ref_a]
	return (len(sparx_utilities.even_angles(x, method = dat[2], symmetry = dat[0])) - dat[1])**2


def ali3dpsi_MPI(stack, ref_vol, outdir, maskfile = None, ir = 1, ou = -1, rs = 1, 
            xr = "4 2 2 1", yr = "-1", ts = "1 1 0.5 0.25", delta = "10 6 4 4", an = "-1", apsi = "-1", deltapsi = "-1", startpsi = "-1",
	    center = -1, maxit = 5, CTF = False, snr = 1.0,  ref_a = "S", sym = "c1",  user_func_name = "ref_ali3d",
	    fourvar = True, npad = 4, debug = False, termprec = 0.0):

	pass#IMPORTIMPORTIMPORT from alignment       import Numrinit, prepare_refrings, proj_ali_incore, proj_ali_incore_local, proj_ali_incore_local_psi
	pass#IMPORTIMPORTIMPORT from utilities       import model_circle, get_image, drop_image, get_input_from_string
	pass#IMPORTIMPORTIMPORT from utilities       import bcast_list_to_all, bcast_number_to_all, reduce_EMData_to_root, bcast_EMData_to_all
	pass#IMPORTIMPORTIMPORT from utilities       import send_attr_dict
	pass#IMPORTIMPORTIMPORT from utilities       import get_params_proj, set_params_proj, file_type
	pass#IMPORTIMPORTIMPORT from fundamentals    import rot_avg_image
	pass#IMPORTIMPORTIMPORT import os
	pass#IMPORTIMPORTIMPORT import types
	pass#IMPORTIMPORTIMPORT from utilities       import print_begin_msg, print_end_msg, print_msg
	pass#IMPORTIMPORTIMPORT from mpi             import mpi_bcast, mpi_comm_size, mpi_comm_rank, MPI_FLOAT, MPI_COMM_WORLD, mpi_barrier
	pass#IMPORTIMPORTIMPORT from mpi             import mpi_reduce, MPI_INT, MPI_SUM
	pass#IMPORTIMPORTIMPORT from filter          import filt_ctf
	pass#IMPORTIMPORTIMPORT from projection      import prep_vol, prgs
	pass#IMPORTIMPORTIMPORT from statistics      import hist_list, varf3d_MPI
	pass#IMPORTIMPORTIMPORT from applications    import MPI_start_end


	number_of_proc = mpi.mpi_comm_size(mpi.MPI_COMM_WORLD)
	myid           = mpi.mpi_comm_rank(mpi.MPI_COMM_WORLD)
	main_node = 0
	
	if os.path.exists(outdir):  sparx_global_def.ERROR('Output directory exists, please change the name and restart the program', "ali3d_MPI", 1, myid)
	mpi.mpi_barrier(mpi.MPI_COMM_WORLD)
	
	if myid == main_node:
		os.mkdir(outdir)
		pass#IMPORTIMPORTIMPORT import global_def
		sparx_global_def.LOGFILE =  os.path.join(outdir, sparx_global_def.LOGFILE)
		sparx_utilities.print_begin_msg("ali3dpsi_MPI")
	mpi.mpi_barrier(mpi.MPI_COMM_WORLD)

	if debug:
		pass#IMPORTIMPORTIMPORT from time import sleep
		while not os.path.exists(outdir):
			print("Node ",myid,"  waiting...")
			time.sleep(5)

		info_file = os.path.join(outdir, "progress%04d"%myid)
		finfo = open(info_file, 'w')
	else:
		finfo = None

	xrng        = sparx_utilities.get_input_from_string(xr)
	if  yr == "-1":  yrng = xrng
	else          :  yrng = sparx_utilities.get_input_from_string(yr)
	step        = sparx_utilities.get_input_from_string(ts)
	delta       = sparx_utilities.get_input_from_string(delta)
	lstp = 1 # min(len(xrng), len(yrng), len(step), len(delta))
	if an == "-1":
		an = [-1] * lstp
	else:
		an = sparx_utilities.get_input_from_string(an)

	if apsi == "-1":
		apsi = [-1] * lstp
	else:
		apsi = sparx_utilities.get_input_from_string(apsi)

	if deltapsi == "-1":
		deltapsi = [-1] * lstp
	else:
		deltapsi = sparx_utilities.get_input_from_string(deltapsi)

	if startpsi == "-1":
		startpsi = [-1] * lstp
	else:
		startpsi = sparx_utilities.get_input_from_string(startpsi)

	first_ring  = int(ir)
	rstep       = int(rs)
	last_ring   = int(ou)
	max_iter    = int(maxit)
	center      = int(center)

	vol     = EMAN2_cppwrap.EMData()
	vol.read_image(ref_vol)
	nx      = vol.get_xsize()
	if last_ring < 0:	last_ring = int(nx/2) - 2

	if myid == main_node:
		pass#IMPORTIMPORTIMPORT import user_functions
		user_func = sparx_user_functions.factory[user_func_name]

		sparx_utilities.print_msg("Input stack                 : %s\n"%(stack))
		sparx_utilities.print_msg("Reference volume            : %s\n"%(ref_vol))	
		sparx_utilities.print_msg("Output directory            : %s\n"%(outdir))
		sparx_utilities.print_msg("Maskfile                    : %s\n"%(maskfile))
		sparx_utilities.print_msg("Inner radius                : %i\n"%(first_ring))
		sparx_utilities.print_msg("Outer radius                : %i\n"%(last_ring))
		sparx_utilities.print_msg("Ring step                   : %i\n"%(rstep))
		sparx_utilities.print_msg("X search range              : %s\n"%(xrng))
		sparx_utilities.print_msg("Y search range              : %s\n"%(yrng))
		sparx_utilities.print_msg("Translational step          : %s\n"%(step))
		sparx_utilities.print_msg("Angular step                : %s\n"%(delta))
		sparx_utilities.print_msg("Angular search range (phi and theta)       : %s\n"%(an))
		sparx_utilities.print_msg("Angular search range (psi)                 : %s\n"%(apsi))
		sparx_utilities.print_msg("Delta psi                   : %s\n"%(deltapsi))
		sparx_utilities.print_msg("Start psi                   : %s\n"%(startpsi))
		sparx_utilities.print_msg("Maximum iteration           : %i\n"%(max_iter))
		sparx_utilities.print_msg("Percentage of change for termination: %f\n"%(termprec))
		sparx_utilities.print_msg("Center type                 : %i\n"%(center))
		sparx_utilities.print_msg("CTF correction              : %s\n"%(CTF))
		sparx_utilities.print_msg("Signal-to-Noise Ratio       : %f\n"%(snr))
		sparx_utilities.print_msg("Reference projection method : %s\n"%(ref_a))
		sparx_utilities.print_msg("Symmetry group              : %s\n\n"%(sym))

	if maskfile:
		if type(maskfile) is bytes: mask3D = sparx_utilities.get_image(maskfile)
		else:                                  mask3D = maskfile
	else: mask3D = sparx_utilities.model_circle(last_ring, nx, nx, nx)

	numr	= sparx_alignment.Numrinit(first_ring, last_ring, rstep, "F")
	mask2D  = sparx_utilities.model_circle(last_ring,nx,nx) - sparx_utilities.model_circle(first_ring,nx,nx)

	fscmask = mask3D  #model_circle(last_ring,nx,nx,nx)  For a fancy mask circle would work better  PAP 7/21/11
	if CTF:
		pass#IMPORTIMPORTIMPORT from reconstruction import rec3D_MPI
		pass#IMPORTIMPORTIMPORT from filter         import filt_ctf
	pass#IMPORTIMPORTIMPORT else:	 from reconstruction import rec3D_MPI_noCTF

	if myid == main_node:
		if sparx_utilities.file_type(stack) == "bdb":
			pass#IMPORTIMPORTIMPORT from EMAN2db import db_open_dict
			dummy = EMAN2db.db_open_dict(stack, True)
		# horatio active_refactoring Jy51i1EwmLD4tWZ9_00000_1
		# active = EMUtil.get_all_attributes(stack, 'active')
		# list_of_particles = []
		# for im in xrange(len(active)):
		# 	if active[im]:  list_of_particles.append(im)
		# del active
		# nima = len(list_of_particles)
		
		nima = EMAN2_cppwrap.EMUtil.get_image_count(stack)
		list_of_particles = list(range(nima))
	
	else:
		nima = 0
	total_nima = sparx_utilities.bcast_number_to_all(nima, source_node = main_node)

	if myid != main_node:
		list_of_particles = [-1]*total_nima
	list_of_particles = sparx_utilities.bcast_list_to_all(list_of_particles, myid,  source_node = main_node)

	image_start, image_end = MPI_start_end(total_nima, number_of_proc, myid)
	# create a list of images for each node
	list_of_particles = list_of_particles[image_start: image_end]
	nima = len(list_of_particles)
	if debug:
		finfo.write("image_start, image_end: %d %d\n" %(image_start, image_end))
		finfo.flush()

	data = EMAN2_cppwrap.EMData.read_images(stack, list_of_particles)
	if fourvar:  original_data = []
	for im in range(nima):
		data[im].set_attr('ID', list_of_particles[im])
		if fourvar: original_data.append(data[im].copy())
		if CTF:
			ctf_params = data[im].get_attr("ctf")
			st = EMAN2_cppwrap.Util.infomask(data[im], mask2D, False)
			data[im] -= st[0]
			data[im] = sparx_filter.filt_ctf(data[im], ctf_params)
			data[im].set_attr('ctf_applied', 1)

	if debug:
		finfo.write( '%d loaded  \n' % nima )
		finfo.flush()
	if myid == main_node:
		# initialize data for the reference preparation function
		ref_data = [ mask3D, max(center,0), None, None, None, None ]
		# for method -1, switch off centering in user function

	pass#IMPORTIMPORTIMPORT from time import time

	#  this is needed for gathering of pixel errors
	disps = []
	recvcount = []
	for im in range(number_of_proc):
		if im == main_node :  disps.append(0)
		else:                  disps.append(disps[im-1] + recvcount[im-1])
		ib, ie = MPI_start_end(total_nima, number_of_proc, im)
		recvcount.append(ie-ib)

	pixer = [0.0]*nima
	cs = [0.0]*3
	cnx = cny = nx//2
	numr = sparx_alignment.Numrinit(first_ring, last_ring, 1, "F")

	total_iter = 0
	# do the projection matching
	for N_step in range(lstp):
		terminate = 0
		Iter = -1
		while Iter < max_iter-1 and terminate == 0:
			Iter += 1
			total_iter += 1
			if myid == main_node:
				start_time = time.time()
				sparx_utilities.print_msg("\nITERATION #%3d,  inner iteration #%3d\nDelta = %4.1f, an = %5.2f, xrange = %5.2f, yrange = %5.2f, step = %5.2f, delta psi = %5.2f, start psi = %5.2f\n"%(total_iter, Iter, delta[N_step], an[N_step], xrng[N_step],yrng[N_step],step[N_step],deltapsi[N_step],startpsi[N_step]))

			volft,kb = sparx_projection.prep_vol(vol)

			for im in range(nima):
				phi,tht,psi,s2x,s2y = sparx_utilities.get_params_proj(data[im])
				refim = sparx_projection.prgs( volft,kb,[phi,tht,0.0,0.0,0.0] )
				pass#IMPORTIMPORTIMPORT from alignment import align2d
				ang, sxs, sys, mirror, peak = sparx_alignment.align2d(data[im], refim, xrng=0.0, yrng=0.0, step=1, first_ring=first_ring, last_ring=last_ring, rstep=1, mode = "F")
				if mirror > 0:
					phi   = (540.0 + phi)%360.0
					tht   = 180.0  - tht
					psi   = (540.0 - ang)%360.0
				else:
					psi   = (720.0 - ang)%360.0
				sparx_utilities.set_params_proj(data[im],[phi,tht,psi,0.0,0.0])
				data[im].set_attr("previousmax", peak)

			if myid == main_node:
				sparx_utilities.print_msg("Time of alignment = %d\n"%(time.time()-start_time))
				start_time = time.time()

			if CTF:  vol, fscc = sparx_reconstruction.rec3D_MPI(data, snr, sym, fscmask, os.path.join(outdir, "resolution%04d"%(total_iter)), myid, main_node, npad = npad)
			else:    vol, fscc = sparx_reconstruction.rec3D_MPI_noCTF(data, sym, fscmask, os.path.join(outdir, "resolution%04d"%(total_iter)), myid, main_node, npad = npad)

			if myid == main_node:
				sparx_utilities.print_msg("3D reconstruction time = %d\n"%(time.time()-start_time))
				start_time = time.time()

			if myid == main_node:
				sparx_utilities.drop_image(vol, os.path.join(outdir, "vol%04d.hdf"%(total_iter)))
				ref_data[2] = vol
				ref_data[3] = fscc
				ref_data[4] = None
				#  call user-supplied function to prepare reference image, i.e., center and filter it
				vol, cs = user_func(ref_data)
				sparx_utilities.drop_image(vol, os.path.join(outdir, "volf%04d.hdf"%(total_iter)))

			sparx_utilities.bcast_EMData_to_all(vol, myid, main_node)
			# write out headers, under MPI writing has to be done sequentially
			mpi.mpi_barrier(mpi.MPI_COMM_WORLD)
			par_str = ['xform.projection', 'previousmax', 'ID']
			if myid == main_node:
				if(sparx_utilities.file_type(stack) == "bdb"):
					pass#IMPORTIMPORTIMPORT from utilities import recv_attr_dict_bdb
					sparx_utilities.recv_attr_dict_bdb(main_node, stack, data, par_str, image_start, image_end, number_of_proc)
				else:
					pass#IMPORTIMPORTIMPORT from utilities import recv_attr_dict
					sparx_utilities.recv_attr_dict(main_node, stack, data, par_str, image_start, image_end, number_of_proc)
				sparx_utilities.print_msg("Time to write header information= %d\n"%(time.time()-start_time))
				start_time = time.time()
			else:	       sparx_utilities.send_attr_dict(main_node, data, par_str, image_start, image_end)
	if myid == main_node: sparx_utilities.print_end_msg("ali3dpsi_MPI")

# =================== SHC
"""Multiline Comment15"""
def ali3d_shcMPI(stack, ref_vol, outdir, maskfile = None, ir = 1, ou = -1, rs = 1, 
            xr = "4 2 2 1", yr = "-1", ts = "1 1 0.5 0.25", delta = "10 6 4 4", an = "-1", apsi = "-1", deltapsi = "-1", startpsi = "-1",
	    center = -1, maxit = 5, CTF = False, snr = 1.0,  ref_a = "S", sym = "c1",  user_func_name = "ref_ali3d",
	    fourvar = True, npad = 4, debug = False, termprec = 0.0, gamma=-1):

	pass#IMPORTIMPORTIMPORT from alignment       import Numrinit, prepare_refrings, proj_ali_incore, proj_ali_incore_local, proj_ali_incore_local_psi, shc
	pass#IMPORTIMPORTIMPORT from utilities       import model_circle, get_image, drop_image, get_input_from_string
	pass#IMPORTIMPORTIMPORT from utilities       import bcast_list_to_all, bcast_number_to_all, bcast_EMData_to_all
	pass#IMPORTIMPORTIMPORT from utilities       import send_attr_dict, get_params_proj, file_type
	pass#IMPORTIMPORTIMPORT import os
	pass#IMPORTIMPORTIMPORT import types
	pass#IMPORTIMPORTIMPORT from utilities       import print_begin_msg, print_end_msg, print_msg
	pass#IMPORTIMPORTIMPORT from mpi             import mpi_bcast, mpi_comm_size, mpi_comm_rank, MPI_FLOAT, MPI_COMM_WORLD, mpi_barrier, MPI_INT
	pass#IMPORTIMPORTIMPORT from projection      import prep_vol, prgs
	pass#IMPORTIMPORTIMPORT from statistics      import hist_list, varf3d_MPI
	pass#IMPORTIMPORTIMPORT from applications    import MPI_start_end
	pass#IMPORTIMPORTIMPORT from math            import sqrt, acos, radians
	pass#IMPORTIMPORTIMPORT from random          import shuffle

	if gamma > 0:
		gamma = numpy.radians(gamma)

	number_of_proc = mpi.mpi_comm_size(mpi.MPI_COMM_WORLD)
	myid           = mpi.mpi_comm_rank(mpi.MPI_COMM_WORLD)
	main_node = 0
	
	if os.path.exists(outdir):  sparx_global_def.ERROR('Output directory exists, please change the name and restart the program', "ali3d_MPI", 1, myid)
	mpi.mpi_barrier(mpi.MPI_COMM_WORLD)

	if myid == main_node:
		os.mkdir(outdir)
		pass#IMPORTIMPORTIMPORT import global_def
		sparx_global_def.LOGFILE =  os.path.join(outdir, sparx_global_def.LOGFILE)
		sparx_utilities.print_begin_msg("ali3d_shcMPI")
	mpi.mpi_barrier(mpi.MPI_COMM_WORLD)

	if debug:
		pass#IMPORTIMPORTIMPORT from time import sleep
		while not os.path.exists(outdir):
			print("Node ",myid,"  waiting...")
			time.sleep(5)

		info_file = os.path.join(outdir, "progress%04d"%myid)
		finfo = open(info_file, 'w')
	else:
		finfo = None

	xrng        = sparx_utilities.get_input_from_string(xr)
	if  yr == "-1":  yrng = xrng
	else          :  yrng = sparx_utilities.get_input_from_string(yr)
	step        = sparx_utilities.get_input_from_string(ts)
	delta       = sparx_utilities.get_input_from_string(delta)
	lstp = min(len(xrng), len(yrng), len(step), len(delta))
	if an == "-1":
		an = [-1.0] * lstp
	else:
		an = sparx_utilities.get_input_from_string(an)

	if apsi == "-1":
		apsi = [-1] * lstp
	else:
		apsi = sparx_utilities.get_input_from_string(apsi)

	if deltapsi == "-1":
		deltapsi = [-1] * lstp
	else:
		deltapsi = sparx_utilities.get_input_from_string(deltapsi)

	if startpsi == "-1":
		startpsi = [-1] * lstp
	else:
		startpsi = sparx_utilities.get_input_from_string(startpsi)

	first_ring  = int(ir)
	rstep       = int(rs)
	last_ring   = int(ou)
	max_iter    = int(maxit)
	center      = int(center)

	vol     = EMAN2_cppwrap.EMData()
	vol.read_image(ref_vol)
	nx      = vol.get_xsize()
	if last_ring < 0:	last_ring = int(nx/2) - 2

	if myid == main_node:
		pass#IMPORTIMPORTIMPORT import user_functions
		user_func = sparx_user_functions.factory[user_func_name]

		sparx_utilities.print_msg("Input stack                 : %s\n"%(stack))
		sparx_utilities.print_msg("Reference volume            : %s\n"%(ref_vol))	
		sparx_utilities.print_msg("Output directory            : %s\n"%(outdir))
		sparx_utilities.print_msg("Maskfile                    : %s\n"%(maskfile))
		sparx_utilities.print_msg("Inner radius                : %i\n"%(first_ring))
		sparx_utilities.print_msg("Outer radius                : %i\n"%(last_ring))
		sparx_utilities.print_msg("Ring step                   : %i\n"%(rstep))
		sparx_utilities.print_msg("X search range              : %s\n"%(xrng))
		sparx_utilities.print_msg("Y search range              : %s\n"%(yrng))
		sparx_utilities.print_msg("Translational step          : %s\n"%(step))
		sparx_utilities.print_msg("Angular step                : %s\n"%(delta))
		sparx_utilities.print_msg("Angular search range (phi and theta)       : %s\n"%(an))
		#print_msg("Angular search range (psi)                 : %s\n"%(apsi))
		#print_msg("Delta psi                   : %s\n"%(deltapsi))
		#print_msg("Start psi                   : %s\n"%(startpsi))
		sparx_utilities.print_msg("Maximum number of iterations : %i\n"%(max_iter))
		sparx_utilities.print_msg("Percentage of change for termination: %f\n"%(termprec))
		sparx_utilities.print_msg("Center type                 : %i\n"%(center))
		sparx_utilities.print_msg("CTF correction              : %s\n"%(CTF))
		sparx_utilities.print_msg("Signal-to-Noise Ratio       : %f\n"%(snr))
		sparx_utilities.print_msg("Reference projection method : %s\n"%(ref_a))
		sparx_utilities.print_msg("Symmetry group              : %s\n\n"%(sym))

	if maskfile:
		if type(maskfile) is bytes: mask3D = sparx_utilities.get_image(maskfile)
		else:                                  mask3D = maskfile
	else: mask3D = sparx_utilities.model_circle(last_ring, nx, nx, nx)

	numr	= sparx_alignment.Numrinit(first_ring, last_ring, rstep, "F")
	mask2D  = sparx_utilities.model_circle(last_ring,nx,nx) - sparx_utilities.model_circle(first_ring,nx,nx)

	fscmask = mask3D  #model_circle(last_ring,nx,nx,nx)  For a fancy mask circle would work better  PAP 7/21/11
	if CTF:
		pass#IMPORTIMPORTIMPORT from reconstruction import rec3D_MPI
		pass#IMPORTIMPORTIMPORT from filter         import filt_ctf
	pass#IMPORTIMPORTIMPORT else:	 from reconstruction import rec3D_MPI_noCTF

	if myid == main_node:
		if sparx_utilities.file_type(stack) == "bdb":
			pass#IMPORTIMPORTIMPORT from EMAN2db import db_open_dict
			dummy = EMAN2db.db_open_dict(stack, True)
		# horatio active_refactoring Jy51i1EwmLD4tWZ9_00000_1
		# active = EMUtil.get_all_attributes(stack, 'active')
		# list_of_particles = []
		# for im in xrange(len(active)):
		# 	if active[im]:  list_of_particles.append(im)
		# del active
		# nima = len(list_of_particles)
		
		nima = EMAN2_cppwrap.EMUtil.get_image_count(stack)
		list_of_particles = list(range(nima))
	
	else:
		nima = 0
	total_nima = sparx_utilities.bcast_number_to_all(nima, source_node = main_node)

	if myid != main_node:
		list_of_particles = [-1]*total_nima
	list_of_particles = sparx_utilities.bcast_list_to_all(list_of_particles, myid,  source_node = main_node)

	image_start, image_end = MPI_start_end(total_nima, number_of_proc, myid)
	# create a list of images for each node
	list_of_particles = list_of_particles[image_start: image_end]
	nima = len(list_of_particles)
	if debug:
		finfo.write("image_start, image_end: %d %d\n" %(image_start, image_end))
		finfo.flush()

	data = EMAN2_cppwrap.EMData.read_images(stack, list_of_particles)
	if fourvar:  original_data = []
	for im in range(nima):
		data[im].set_attr('ID', list_of_particles[im])
		if( an[0] > -1 ):
			#  These are local searches, set xform.anchor to the current projection direction
			data[im].set_attr("xform.anchor", data[im].get_attr("xform.projection"))
		if fourvar: original_data.append(data[im].copy())
		if CTF:
			ctf_params = data[im].get_attr("ctf")
			st = EMAN2_cppwrap.Util.infomask(data[im], mask2D, False)
			data[im] -= st[0]
			data[im] = sparx_filter.filt_ctf(data[im], ctf_params)
			data[im].set_attr('ctf_applied', 1)

	if debug:
		finfo.write( '%d loaded  \n' % nima )
		finfo.flush()
	if myid == main_node:
		# initialize data for the reference preparation function
		ref_data = [ mask3D, max(center,0), None, None, None, None ]
		# for method -1, switch off centering in user function

	pass#IMPORTIMPORTIMPORT from time import time

	#  this is needed for gathering of pixel errors
	disps = []
	recvcount = []
	for im in range(number_of_proc):
		if im == main_node :  disps.append(0)
		else:                 disps.append(disps[im-1] + recvcount[im-1])
		ib, ie = MPI_start_end(total_nima, number_of_proc, im)
		recvcount.append(ie-ib)

	pixer = [0.0]*nima
	cs = [0.0]*3
	total_iter = 0
	final_params = None
	final_volume = None
	final_volume_filtered = None
	# do the projection matching
	for N_step in range(lstp):

		terminate = 0
		Iter = 0
		while Iter < max_iter and terminate == 0:

			Iter += 1
			total_iter += 1

			if myid == main_node:
				start_time = time.time()
				sparx_utilities.print_msg("\nITERATION #%3d,  inner iteration #%3d\nDelta = %5.2f, an = %5.2f, xrange = %5.2f, yrange = %5.2f,translational step = %5.2f\n"%(total_iter, Iter, delta[N_step], an[N_step], xrng[N_step], yrng[N_step], step[N_step]))

			#=========================================================================
			# build references
			volft, kb = sparx_projection.prep_vol(vol)
			refrings = sparx_alignment.prepare_refrings(volft, kb, nx, delta[N_step], ref_a, sym, numr, True)
			lastdelta = delta[N_step]
			del vol, volft, kb
			if(an[N_step] > 0):
				# generate list of angles
				pass#IMPORTIMPORTIMPORT from alignment import generate_list_of_reference_angles_for_search
				list_of_reference_angles = \
				sparx_alignment.generate_list_of_reference_angles_for_search([[refrings[lr].get_attr("phi"), refrings[lr].get_attr("theta")] for lr in range(len(refrings))], sym=sym)			
			else:  list_of_reference_angles = [[1.0,1.0]]
			#=========================================================================

			if myid == main_node:
				sparx_utilities.print_msg("Time to prepare rings: %d\n" % (time.time()-start_time))
				start_time = time.time()

			#=========================================================================
			#  We assume previousmax exists
			"""Multiline Comment16"""
			#=========================================================================

			#=========================================================================
			# alignment
			iter_indexes = list(range(nima))
			random.shuffle(iter_indexes)
			for im in iter_indexes:
				pass#IMPORTIMPORTIMPORT from utilities import get_params_proj
				#print "  IN  ",im,get_params_proj(data[im]),data[im].get_attr("previousmax")
				peak, pixer[im], number_of_checked_refs, iref = \
					sparx_alignment.shc(data[im], refrings, list_of_reference_angles, numr, xrng[N_step], yrng[N_step], step[N_step], an[N_step], sym, finfo)
				#print "  OU  ",im,get_params_proj(data[im]),data[im].get_attr("previousmax")
				if gamma > 0:
					n1 = refrings[iref].get_attr("n1")
					n2 = refrings[iref].get_attr("n2")
					n3 = refrings[iref].get_attr("n3")
					to_be_deleted = []
					for irr in range(len(refrings)):
						nn1 = refrings[irr].get_attr("n1")
						nn2 = refrings[irr].get_attr("n2")
						nn3 = refrings[irr].get_attr("n3")
						if abs(math.acos(n1*nn1 + n2*nn2 * n3*nn3)) < gamma:
							to_be_deleted.append( irr )
					if len(to_be_deleted) > 0:
						to_be_deleted.sort(reverse=True)
						for irr in to_be_deleted:
							del refrings[irr]
				elif gamma == 0:
						del refrings[iref]
			#=========================================================================
			del refrings

			if myid == main_node:
				sparx_utilities.print_msg("Time of alignment = %d\n"%(time.time()-start_time))
				start_time = time.time()
			#=========================================================================
			#output pixel errors, check stop criterion
			pass#IMPORTIMPORTIMPORT from mpi import mpi_gatherv
			recvbuf = mpi.mpi_gatherv(pixer, nima, mpi.MPI_FLOAT, recvcount, disps, mpi.MPI_FLOAT, main_node, mpi.MPI_COMM_WORLD)
			mpi.mpi_barrier(mpi.MPI_COMM_WORLD)
			terminate = 0
			if myid == main_node:
				recvbuf = list(map(float, recvbuf))
				pass#IMPORTIMPORTIMPORT from statistics import hist_list
				lhist = 20
				region, histo = sparx_statistics.hist_list(recvbuf, lhist)
				if region[0] < 0.0:  region[0] = 0.0
				msg = "      Histogram of pixel errors\n      ERROR       number of particles\n"
				sparx_utilities.print_msg(msg)
				for lhx in range(lhist):
					msg = " %10.3f     %7d\n"%(region[lhx], histo[lhx])
					sparx_utilities.print_msg(msg)
				# Terminate if 95% within 0.1 pixel error
				im = 0
				for lhx in range(lhist):
					if region[lhx] > 0.1: break
					im += histo[lhx]
				precn = 100*float(total_nima-im)/float(total_nima)
				msg = " Number of particles that changed orientations %7d, percentage of total: %5.1f\n"%(total_nima-im, precn)
				sparx_utilities.print_msg(msg)
				if precn <= termprec:  terminate = 1
				del region, histo
			del recvbuf
			terminate = mpi.mpi_bcast(terminate, 1, mpi.MPI_INT, 0, mpi.MPI_COMM_WORLD)
			terminate = int(terminate[0])
			#=========================================================================

			#=========================================================================
			# centering
			if center == -1 and sym[0] == 'c':
				pass#IMPORTIMPORTIMPORT from utilities      import estimate_3D_center_MPI, rotate_3D_shift
				cs[0], cs[1], cs[2], dummy, dummy = sparx_utilities.estimate_3D_center_MPI(data, total_nima, myid, number_of_proc, main_node)
				if myid == main_node:
					msg = " Average center x = %10.3f        Center y = %10.3f        Center z = %10.3f\n"%(cs[0], cs[1], cs[2])
					sparx_utilities.print_msg(msg)
				if int(sym[1]) > 1:
					cs[0] = cs[1] = 0.0
					if myid == main_node:
						sparx_utilities.print_msg("For symmetry group cn (n>1), we only center the volume in z-direction\n")
				cs = mpi.mpi_bcast(cs, 3, mpi.MPI_FLOAT, main_node, mpi.MPI_COMM_WORLD)
				cs = [-float(cs[0]), -float(cs[1]), -float(cs[2])]
				sparx_utilities.rotate_3D_shift(data, cs)
			#=========================================================================

			#=========================================================================
			# write out headers, under MPI writing has to be done sequentially
			mpi.mpi_barrier(mpi.MPI_COMM_WORLD)
			# It takes plenty of times, so do it once in awhile...
			if( True ): #total_iter>3 and total_iter%5 == 0 ):
				par_str = ['xform.projection', 'previousmax', 'ID']
				if myid == main_node:
					if(sparx_utilities.file_type(stack) == "bdb"):
						pass#IMPORTIMPORTIMPORT from utilities import recv_attr_dict_bdb
						sparx_utilities.recv_attr_dict_bdb(main_node, stack, data, par_str, image_start, image_end, number_of_proc)
					else:
						pass#IMPORTIMPORTIMPORT from utilities import recv_attr_dict
						sparx_utilities.recv_attr_dict(main_node, stack, data, par_str, image_start, image_end, number_of_proc)

					"""Multiline Comment17"""


				else:
					sparx_utilities.send_attr_dict(main_node, data, par_str, image_start, image_end)
			#=========================================================================

			#=========================================================================
			# volume reconstruction
			#vol_previous = vol
			if CTF: vol, fscc = sparx_reconstruction.rec3D_MPI(data, snr, sym, fscmask, os.path.join(outdir, "resolution%04d"%(total_iter)), myid, main_node, npad = npad)
			else:   vol, fscc = sparx_reconstruction.rec3D_MPI_noCTF(data, sym, fscmask, os.path.join(outdir, "resolution%04d"%(total_iter)), myid, main_node, npad = npad)
			# log
			if myid == main_node:
				sparx_utilities.print_msg("3D reconstruction time = %d\n"%(time.time()-start_time))
				start_time = time.time()
			if fourvar:
			#  Compute Fourier variance
				for im in range(nima):
					original_data[im].set_attr( 'xform.projection', data[im].get_attr('xform.projection') )
				varf = sparx_statistics.varf3d_MPI(original_data, ssnr_text_file = os.path.join(outdir, "ssnr%04d"%(total_iter)), mask2D = None, reference_structure = vol, ou = last_ring, rw = 1.0, npad = 1, CTF = CTF, sign = 1, sym =sym, myid = myid)
				if myid == main_node:
					sparx_utilities.print_msg("Time to calculate 3D Fourier variance= %d\n"%(time.time()-start_time))
					start_time = time.time()
					varf = 1.0/varf
			else:
				varf = None
			# user functions + save volume
			if myid == main_node:
				#drop_image(vol, os.path.join(outdir, "vol%04d.hdf"%(total_iter)))
				ref_data[2] = vol
				ref_data[3] = fscc
				ref_data[4] = varf
				#  call user-supplied function to prepare reference image, i.e., center and filter it
				vol, cs = user_func(ref_data)
				sparx_utilities.drop_image(vol, os.path.join(outdir, "volf%04d.hdf"%(total_iter)))
				#print_msg("Euclidean distance between the current and the previous volume: " + str(sqrt(vol.cmp("SqEuclidean",vol_previous,{"mask":mask3D,"zeromask":0,"normto":0}))) + "\n")
				sparx_utilities.print_msg("L2 norm of the volume: " + str(vol.cmp("dot", vol, {"negative":0, "mask":mask3D})) + "\n")
			del varf
			# broadcast volume
			sparx_utilities.bcast_EMData_to_all(vol, myid, main_node)
			#=========================================================================

	if myid == main_node: 
		sparx_utilities.print_end_msg("ali3d_shcMPI")

def mref_ali3d(stack, ref_vol, outdir, maskfile=None, focus = None, maxit=1, ir=1, ou=-1, rs=1, 
           xr = "4 2 2 1", yr = "-1", ts = "1 1 0.5 0.25", delta="10 6 4 4", an="-1", 
	     center = 1.0, nassign = 3, nrefine = 1, CTF = False, snr = 1.0,  ref_a = "S", sym="c1",
	     user_func_name="ref_ali3d", MPI=False, npad = 4, debug = False, fourvar=False, termprec = 0.0):
	pass#IMPORTIMPORTIMPORT from utilities      import model_circle, drop_image, get_image, get_input_from_string
	pass#IMPORTIMPORTIMPORT from utilities      import get_arb_params, set_arb_params, get_im, write_headers
	pass#IMPORTIMPORTIMPORT from projection     import prep_vol, prgs
	pass#IMPORTIMPORTIMPORT from utilities      import get_params_proj, estimate_3D_center
	pass#IMPORTIMPORTIMPORT from alignment      import proj_ali_incore, proj_ali_incore_local, Numrinit, prepare_refrings
	pass#IMPORTIMPORTIMPORT from filter	    import filt_params, filt_tanl
	pass#IMPORTIMPORTIMPORT from fundamentals   import fshift
	pass#IMPORTIMPORTIMPORT from statistics     import fsc_mask
	pass#IMPORTIMPORTIMPORT from utilities      import print_begin_msg, print_end_msg, print_msg

	pass#IMPORTIMPORTIMPORT import os
	pass#IMPORTIMPORTIMPORT import types
	# 2D alignment using rotational ccf in polar coords and linear
	# interpolation	

	pass#IMPORTIMPORTIMPORT import user_functions
	user_func = sparx_user_functions.factory[user_func_name]

	
	if os.path.exists(outdir):  sparx_global_def.ERROR('Output directory exists, please change the name and restart the program', "mref_ali3d", 1)
	os.mkdir(outdir)
	pass#IMPORTIMPORTIMPORT import global_def
	sparx_global_def.LOGFILE =  os.path.join(outdir, sparx_global_def.LOGFILE)
	sparx_utilities.print_begin_msg("mref_ali3d")

	xrng        = sparx_utilities.get_input_from_string(xr)
	if  yr == "-1":  yrng = xrng
	else          :  yrng = sparx_utilities.get_input_from_string(yr)
	step        = sparx_utilities.get_input_from_string(ts)
	delta       = sparx_utilities.get_input_from_string(delta)
	lstp = min( len(xrng), len(yrng), len(step), len(delta) )
	if (an == "-1"):
		an = []
		for i in range(len(xrng)):   an.append(-1)
		pass#IMPORTIMPORTIMPORT from alignment	  import proj_ali_incore
	else:
		an = sparx_utilities.get_input_from_string(an)
		pass#IMPORTIMPORTIMPORT from alignment	  import proj_ali_incore_local

	first_ring  = int(ir)
	rstep       = int(rs)
	last_ring   = int(ou)

	numref = EMAN2_cppwrap.EMUtil.get_image_count(ref_vol)
	for  iref in range(numref):
		volref     = EMAN2_cppwrap.EMData()
		volref.read_image(ref_vol, iref)
		volref.write_image(os.path.join(outdir, "volf0000.hdf"), iref)

	nx      = volref.get_xsize()
	if last_ring < 0:	last_ring = nx//2 - 2

	fscmask = sparx_utilities.model_circle(last_ring, nx, nx, nx)

	pass#IMPORTIMPORTIMPORT import user_functions
	user_func = sparx_user_functions.factory[user_func_name]

	sparx_utilities.print_msg("Input stack                 : %s\n"%(stack))
	sparx_utilities.print_msg("Reference volume            : %s\n"%(ref_vol))	
	sparx_utilities.print_msg("Number of reference volumes : %i\n"%(numref))
	sparx_utilities.print_msg("Output directory            : %s\n"%(outdir))
	sparx_utilities.print_msg("Maskfile                    : %s\n"%(maskfile))
	sparx_utilities.print_msg("Inner radius                : %i\n"%(first_ring))
	sparx_utilities.print_msg("Outer radius                : %i\n"%(last_ring))
	sparx_utilities.print_msg("Ring step                   : %i\n"%(rstep))
	sparx_utilities.print_msg("X search range              : %s\n"%(xrng))
	sparx_utilities.print_msg("Y search range              : %s\n"%(yrng))
	sparx_utilities.print_msg("Translational step          : %s\n"%(step))
	sparx_utilities.print_msg("Angular step                : %s\n"%(delta))
	sparx_utilities.print_msg("Angular search range        : %s\n"%(an))
	sparx_utilities.print_msg("Maximum number of reassignment iterations   : %i\n"%(nassign))
	sparx_utilities.print_msg("Maximum number of alignment iterations      : %i\n"%(nrefine))
	sparx_utilities.print_msg("Center type                 : %i\n"%(center))
	sparx_utilities.print_msg("CTF correction              : %s\n"%(CTF))
	sparx_utilities.print_msg("Signal-to-Noise Ratio       : %f\n"%(snr))
	sparx_utilities.print_msg("Reference projection method : %s\n"%(ref_a))
	sparx_utilities.print_msg("Symmetry group              : %s\n\n"%(sym))
	sparx_utilities.print_msg("User function               : %s\n"%(user_func_name))

	if(maskfile):
		if(type(maskfile) is bytes):	 mask3D = sparx_utilities.get_image(maskfile)
		else: 	                                 mask3D = maskfile
	else        :   mask3D = sparx_utilities.model_circle(last_ring, nx, nx, nx)
	
	numr = sparx_alignment.Numrinit(first_ring, last_ring, rstep, "F")
	mask2D = sparx_utilities.model_circle(last_ring,nx,nx) - sparx_utilities.model_circle(first_ring,nx,nx)


	if debug:  finfo = open(os.path.join(outdir, "progress"), "w")
	else:      finfo = None

	# horatio active_refactoring Jy51i1EwmLD4tWZ9_00000_1
	# active = EMUtil.get_all_attributes(stack, 'active')
	# list_of_particles = []
	# for im in xrange(len(active)):
	# 	if(active[im]):  list_of_particles.append(im)
	# del active
	
	nima = EMAN2_cppwrap.EMUtil.get_image_count(stack)
	list_of_particles = list(range(nima))
	
	data = EMAN2_cppwrap.EMData.read_images(stack, list_of_particles)
	nima = len(data)

	if CTF :
		#  ERROR if ctf applied
		if data[0].get_attr("ctf_applied") > 0:  sparx_global_def.ERROR("mref_ali3d does not work for CTF-applied data", "mref_ali3d", 1)
		pass#IMPORTIMPORTIMPORT from reconstruction import recons3d_4nn_ctf
		pass#IMPORTIMPORTIMPORT from filter import filt_ctf
	pass#IMPORTIMPORTIMPORT else   : from reconstruction import recons3d_4nn

	# initialize data for the reference preparation function
	ref_data = [mask3D, center, None, None]

	# do the projection matching
	total_iter = 0
	tr_dummy = EMAN2_cppwrap.Transform({"type":"spider"})

	Niter = int(lstp*maxit*(nassign+nrefine))
	for Iter in range(Niter):
		N_step = (Iter%(lstp*(nassign+nrefine)))/(nassign+nrefine)
		if Iter%(nassign+nrefine) < nassign:
			runtype = "ASSIGNMENT"
		else:
			runtype = "REFINEMENT"


		total_iter += 1
		sparx_utilities.print_msg("%s ITERATION #%3d\n"%(runtype,total_iter))
		peaks = [-1.0e23]*nima
		trans = [tr_dummy]*nima
		if(an[N_step] > 0):
			pass#IMPORTIMPORTIMPORT from utilities    import even_angles
			ref_angles = sparx_utilities.even_angles(delta[N_step], symmetry=sym, method = ref_a, phiEqpsi = "Zero")
			# generate list of angles
			pass#IMPORTIMPORTIMPORT from alignment import generate_list_of_reference_angles_for_search
			list_of_reference_angles = \
			sparx_alignment.generate_list_of_reference_angles_for_search(ref_angles, sym=sym)
			del ref_angles
		else:  list_of_reference_angles = [[1.0,1.0]]

		cs = [0.0]*3
		for iref in range(numref):
			vol = sparx_utilities.get_im(os.path.join(outdir, "volf%04d.hdf"%( total_iter-1)), iref)
			if(CTF):
				previous_defocus = -1.0
			else:
				volft, kb = sparx_projection.prep_vol(vol)
				if runtype=="REFINEMENT":
					refrings = sparx_alignment.prepare_refrings(volft,kb,nx,delta[N_step],ref_a,sym,numr)

			for im in range(nima):
				if(CTF):
					ctf_params = data[im].get_attr("ctf")
					if(ctf_params.defocus != previous_defocus):
						previous_defocus = ctf_params.defocus
						volft,kb = sparx_projection.prep_vol(sparx_filter.filt_ctf(vol, ctf_params))
					if runtype=="REFINEMENT":
						refrings = sparx_alignment.prepare_refrings(volft,kb,nx,delta[N_step],ref_a,sym,numr)

				if runtype=="ASSIGNMENT":
					phi,tht,psi,s2x,s2y = sparx_utilities.get_params_proj(data[im])
					ref = sparx_projection.prgs( volft,kb,[phi,tht,psi,-s2x,-s2y] )
					peak = ref.cmp("ccc", data[im], {"mask":mask2D, "negative":0})
					if not(finfo is None):
						finfo.write( "ID,iref,peak: %6d %d %8.5f" % (list_of_particles[im],iref,peak) )
						finfo.flush()
				else:		
					if(an[N_step] == -1):
						peak, pixel_error = sparx_alignment.proj_ali_incore(data[im],refrings,numr,xrng[N_step],yrng[N_step],step[N_step])
					else:
						peak, pixel_error = sparx_alignment.proj_ali_incore_local(data[im],refrings,list_of_reference_angles,\
												numr,xrng[N_step],yrng[N_step],step[N_step],an[N_step],sym=sym)
					if not(finfo is None):
						phi,tht,psi,s2x,s2y = sparx_utilities.get_params_proj(data[im])
						finfo.write( "ID,iref,peak,trans: %6d %d %f %f %f %f %f %f"%(list_of_particles[im],iref,peak,phi,tht,psi,s2x,s2y) )
						finfo.flush()


				if(peak > peaks[im]):
					peaks[im] = peak
					data[im].set_attr('group', iref)
					if runtype=="REFINEMENT":
						trans[im] = data[im].get_attr( "xform.projection" )
					if not(finfo is None):
						finfo.write( " current best\n" )
						finfo.flush()
				else:
					if not(finfo is None):
						finfo.write( "\n" )
						finfo.flush()


		if runtype=="REFINEMENT":
			if(an[N_step] > 0):  del  list_of_reference_angles
			for im in range(nima):
				data[im].set_attr('xform.projection', trans[im])

			if center == -1:
				cs[0], cs[1], cs[2], dummy, dummy = sparx_utilities.estimate_3D_center(data, total_nima, myid, number_of_proc, main_node)
				msg = " Average center x = %10.3f        Center y = %10.3f        Center z = %10.3f\n"%(cs[0], cs[1], cs[2])
				sparx_utilities.print_msg(msg)
				cs = [-float(cs[0]), -float(cs[1]), -float(cs[2])]
				sparx_utilities.rotate_3D_shift(data, cs)


		fscc = [None]*numref
		for iref in range(numref):
			list_p = []
			for im in range(nima):
				if(iref == data[im].get_attr('group')):
					list_p.append(im)
			sparx_utilities.print_msg("Group number : %i"%(iref) + ",  number of objects: %i\n"%(len(list_p)))
			#  3D stuff
			if(CTF): vodd = sparx_reconstruction.recons3d_4nn_ctf(data, [list_p[im] for im in range(0,len(list_p), 2)], snr, 1, sym)
			else:    vodd = sparx_reconstruction.recons3d_4nn(data, [list_p[im] for im in range(1,len(list_p), 2)], sym, snr = snr)
			if(CTF): veve = sparx_reconstruction.recons3d_4nn_ctf(data, [list_p[im] for im in range(0,len(list_p), 2)], snr, 1, sym)
			else:    veve = sparx_reconstruction.recons3d_4nn(data,[list_p[im] for im in range(1,len(list_p), 2)], sym, snr = snr)

			fscc[iref] = sparx_statistics.fsc_mask(vodd, veve, mask3D, 1.0, os.path.join(outdir, "resolution_%02d_%04d"%(iref, total_iter)))
			
			# calculate new and improved 3D
			if(CTF): volref = sparx_reconstruction.recons3d_4nn_ctf(data, list_p, snr, 1, sym)
			else:	 volref = sparx_reconstruction.recons3d_4nn(data, list_p, sym, snr = snr)
			volref.write_image(os.path.join(outdir, "vol%04d.hdf"%( total_iter)), iref)
			

		refdata = [None]*7
		refdata[0] = numref
		refdata[1] = outdir
		refdata[2] = fscc
		refdata[3] = total_iter
		refdata[4] = None #varf
		refdata[5] = fscmask
		refdata[6] = False
		user_func( refdata )

		#  here we  write header info
		sparx_utilities.write_headers( stack, data, list_of_particles)

	sparx_utilities.print_end_msg("mref_ali3d")


# This is version with the same number of images per group.
def Kmref2_ali3d_MPI(stack, ref_vol, outdir, maskfile=None, focus = None, maxit=1, ir=1, ou=-1, rs=1, 
            xr ="4 2  2  1", yr="-1", ts="1 1 0.5 0.25",   delta="10  6  4  4", an="-1",
	      center = -1, nassign = 3, nrefine= 1, CTF = False, snr = 1.0,  ref_a="S", sym="c1",
	      user_func_name="ref_ali3d", npad = 4, debug = False, fourvar=False, termprec = 0.0, mpi_comm = None, log = None): 
	pass#IMPORTIMPORTIMPORT from utilities      import model_circle, reduce_EMData_to_root, bcast_EMData_to_all, bcast_number_to_all, drop_image
	pass#IMPORTIMPORTIMPORT from utilities      import bcast_list_to_all, get_image, get_input_from_string, get_im
	pass#IMPORTIMPORTIMPORT from utilities      import get_arb_params, set_arb_params, drop_spider_doc, send_attr_dict
	pass#IMPORTIMPORTIMPORT from utilities      import get_params_proj, set_params_proj, model_blank, write_text_row, write_text_file
	pass#IMPORTIMPORTIMPORT from filter         import filt_params, filt_btwl, filt_ctf, filt_table, fit_tanh, filt_tanl
	pass#IMPORTIMPORTIMPORT from utilities      import rotate_3D_shift,estimate_3D_center_MPI
	pass#IMPORTIMPORTIMPORT from alignment      import Numrinit, prepare_refrings, proj_ali_incore
	pass#IMPORTIMPORTIMPORT from random         import randint
	pass#IMPORTIMPORTIMPORT from filter         import filt_ctf
	pass#IMPORTIMPORTIMPORT from utilities      import print_begin_msg, print_end_msg, print_msg
	pass#IMPORTIMPORTIMPORT from projection     import prep_vol, prgs, project, prgq, gen_rings_ctf
	pass#IMPORTIMPORTIMPORT from utilities      import wrap_mpi_recv, wrap_mpi_send
	pass#IMPORTIMPORTIMPORT from copy           import deepcopy
	pass#IMPORTIMPORTIMPORT import string as sting
	pass#IMPORTIMPORTIMPORT import os
	pass#IMPORTIMPORTIMPORT import types
	pass#IMPORTIMPORTIMPORT from mpi            import mpi_bcast, mpi_comm_size, mpi_comm_rank, MPI_FLOAT, MPI_COMM_WORLD, mpi_barrier
	pass#IMPORTIMPORTIMPORT from mpi            import mpi_reduce, MPI_INT, MPI_SUM
	
	if mpi_comm == None: mpi_comm = mpi.MPI_COMM_WORLD
	number_of_proc = mpi.mpi_comm_size(mpi_comm)
	myid           = mpi.mpi_comm_rank(mpi_comm)
	main_node = 0
	if log == None:
		pass#IMPORTIMPORTIMPORT from logger import Logger
		log =sparx_logger.Logger()

	if os.path.exists(outdir): sparx_global_def.ERROR('Output directory exists, please change the name and restart the program', "Kmref_ali3d_MPI ", 1, myid)
	mpi.mpi_barrier(mpi.MPI_COMM_WORLD)

	if myid == main_node:	
		os.mkdir(outdir)
		pass#IMPORTIMPORTIMPORT import global_def
		sparx_global_def.LOGFILE =  os.path.join(outdir, sparx_global_def.LOGFILE)
		log.add("Kmref_ali3d_MPI - Traditional Kmeans clustering  !")
	mpi.mpi_barrier(mpi.MPI_COMM_WORLD)

	pass#IMPORTIMPORTIMPORT from time import time

	if debug:
		pass#IMPORTIMPORTIMPORT from time import sleep
		while not os.path.exists(outdir):
			print("Node ",myid,"  waiting...")
			time.sleep(5)

		finfo = open(os.path.join(outdir, "progress%04d"%myid), 'w')
		frec  = open( os.path.join(outdir, "recons%04d"%myid), "w" )
	else:
		finfo = None
		frec  = None

	xrng        = sparx_utilities.get_input_from_string(xr)
	if  yr == "-1":  yrng = xrng
	else          :  yrng = sparx_utilities.get_input_from_string(yr)
	step        = sparx_utilities.get_input_from_string(ts)
	delta       = sparx_utilities.get_input_from_string(delta)
	lstp = min( len(xrng), len(yrng), len(step), len(delta) )
	if an == "-1":
		an = []
		for i in range(len(xrng)):   an.append(-1)
	else:
		pass#IMPORTIMPORTIMPORT from  alignment	    import proj_ali_incore_local
		an      = sparx_utilities.get_input_from_string(an)

	first_ring  = int(ir)
	rstep       = int(rs)
	last_ring   = int(ou)
	center      = int(center)

	numref = EMAN2_cppwrap.EMUtil.get_image_count(ref_vol)
	volref     = EMAN2_cppwrap.EMData()
	volref.read_image(stack, 0)
	nx      = volref.get_xsize()
	if last_ring < 0:	last_ring = nx//2 - 2

	fscmask = sparx_utilities.model_circle(last_ring, nx, nx, nx)

	# ignore fourvar
	fourvar = False

	if myid == main_node:
		pass#IMPORTIMPORTIMPORT import user_functions
		user_func = sparx_user_functions.factory[user_func_name]
		log.add("Input stack                 : %s"%(stack))
		log.add("Reference volumes           : %s"%(ref_vol))	
		log.add("Number of reference volumes : %i"%(numref))
		log.add("Output directory            : %s"%(outdir))
		log.add("User function               : %s"%(user_func_name))
		log.add("Maskfile                    : %s"%(maskfile))
		log.add("Inner radius                : %i"%(first_ring))
		log.add("Outer radius                : %i"%(last_ring))
		log.add("Ring step                   : %i"%(rstep))
		log.add("X search range              : %s"%(xrng))
		log.add("Y search range              : %s"%(yrng))
		log.add("Translational step          : %s"%(step))
		log.add("Angular step                : %s"%(delta))
		log.add("Angular search range        : %s"%(an))
		#log.add("Number of assignments in each iteration   : %i"%(nassign))
		#log.add("Number of alignments in each iteration    : %i"%(nrefine))
		log.add("Number of iterations                      : %i"%(lstp*maxit) )
		log.add("Center type                 : %i"%(center))
		log.add("CTF correction              : %s"%(CTF))
		log.add("fourvar                     : %s"%(fourvar))
		log.add("Signal-to-Noise Ratio       : %f"%(snr))
		log.add("Reference projection method : %s"%(ref_a))
		log.add("Symmetry group              : %s"%(sym))
		log.add("Percentage of change for termination: %f"%(termprec))
		log.add("User function               : %s"%(user_func_name))

	if maskfile:
		if type(maskfile) is bytes: mask3D = sparx_utilities.get_image(maskfile)
		else: 	                                mask3D = maskfile
	else        :  mask3D = sparx_utilities.model_circle(last_ring, nx, nx, nx)

	numr     = sparx_alignment.Numrinit(first_ring, last_ring, rstep, "F")
	mask2D   = sparx_utilities.model_circle(last_ring, nx, nx) - sparx_utilities.model_circle(first_ring, nx, nx)

	if myid == main_node:
		nima =EMAN2_cppwrap.EMUtil.get_image_count( stack )
		list_of_particles=list(range(nima))
	else:
		nima = 0

	nima = sparx_utilities.bcast_number_to_all(nima, source_node = main_node)

	if myid != main_node:
		list_of_particles = [-1]*nima

	list_of_particles = sparx_utilities.bcast_list_to_all(list_of_particles, myid,  source_node = main_node)

	image_start, image_end = MPI_start_end(nima, number_of_proc, myid)
	# create a list of images for each node
	total_nima = nima
	list_of_particles = list_of_particles[image_start: image_end]
	nima = len(list_of_particles)

	if debug:
		finfo.write( "Image_start, image_end: %d %d\n" %(image_start, image_end) )
		finfo.flush()

	syms = sting.split(sym, ",")
	if(len(syms) != numref):
		sparx_global_def.ERROR('Number of symmetries (%d)  different from the number of reference volumes  (%d)'%(len(syms) , numref), "Kmref2_ali3d_MPI", 1,myid)


	start_time = time.time()
	data = EMAN2_cppwrap.EMData.read_images(stack, list_of_particles)
	if myid == main_node:
		log.add( "Time to read data: %d\n" % (time.time()-start_time) );start_time = time.time()
	#  Initialize Particle ID and set group number to non-existant -1
	assignment = [-1]*len(data)
	for im in range(len(data)):
		data[im].set_attr_dict({'ID':list_of_particles[im], 'group':-1})
		if CTF:
			ctf_params = data[im].get_attr("ctf")
			st = EMAN2_cppwrap.Util.infomask(data[im], mask2D, False)
			data[im] -= st[0]
			data[im] = sparx_filter.filt_ctf(data[im], ctf_params)
			data[im].set_attr('ctf_applied', 1)

	if fourvar:
		pass#IMPORTIMPORTIMPORT from reconstruction import rec3D_MPI
		pass#IMPORTIMPORTIMPORT from statistics     import varf3d_MPI
		#  Compute Fourier variance
		vol, fscc = sparx_reconstruction.rec3D_MPI(data, snr, sym, fscmask, os.path.join(outdir, "resolution0000"), myid, main_node, finfo=frec, npad=npad)
		varf = sparx_statistics.varf3d_MPI(data, os.path.join(outdir, "ssnr0000"), None, vol, last_ring, 1.0, 1, CTF, 1, sym, myid)
		if myid == main_node:   
			varf = 1.0/varf
			varf.write_image( os.path.join(outdir,"varf0000.hdf") )
	else:
		varf = None

	if myid == main_node:
		for  iref in range(numref):
			sparx_utilities.get_im(ref_vol, iref).write_image(os.path.join(outdir, "volf0000.hdf"), iref)
	mpi.mpi_barrier( mpi.MPI_COMM_WORLD )

	if CTF:
		#if(data[0].get_attr("ctf_applied") > 0.0):  ERROR("Kmref_ali3d_MPI does not work for CTF-applied data", "Kmref_ali3d_MPI", 1, myid)
		pass#IMPORTIMPORTIMPORT from reconstruction import rec3D_MPI
	else:
		pass#IMPORTIMPORTIMPORT from reconstruction import rec3D_MPI_noCTF

	if debug:
		finfo.write( '%d loaded  \n' % len(data) )
		finfo.flush()

	#  this is needed for gathering of pixel errors
	disps = []
	recvcount = []
	for im in range(number_of_proc):
		if im == main_node:  disps.append(0)
		else:                  disps.append(disps[im-1] + recvcount[im-1])
		ib, ie = MPI_start_end(total_nima, number_of_proc, im)
		recvcount.append( ie - ib )

	total_iter = 0
	tr_dummy = EMAN2_cppwrap.Transform({"type":"spider"})

	Niter = int(lstp*maxit*(nassign + nrefine) )
	for Iter in range(Niter):
		N_step = (Iter%(lstp*(nassign+nrefine)))/(nassign+nrefine)
		if Iter%(nassign+nrefine) < nassign:
			runtype = "ASSIGNMENT"
		else:
			runtype = "REFINEMENT"

		total_iter += 1
		if myid == main_node:
			log.add("  %s ITERATION #%3d,  inner iteration #%3d\nDelta = %4.1f, an = %5.2f, xrange = %5.2f, yrange = %5.2f, step = %5.2f"%(runtype, total_iter, Iter, delta[N_step], an[N_step], xrng[N_step],yrng[N_step],step[N_step]))
			start_ime = time.time()
		peaks = [ -1.0e23]*nima
		if runtype=="REFINEMENT":
			pass#IMPORTIMPORTIMPORT from utilities    import even_angles
			#  We assume first symmetry is the lowest one
			refang = sparx_utilities.even_angles(delta[N_step], symmetry=syms[0], method = ref_a, phiEqpsi = "Zero")
			pass#IMPORTIMPORTIMPORT from random import uniform
			pass#IMPORTIMPORTIMPORT from fundamentals import rotate_params
			shake = 0.5
			if(myid == main_node):
				shakenumber = random.uniform( -shake, shake)
			else:
				shakenumber = 0.0
			shakenumber = sparx_utilities.bcast_number_to_all(shakenumber, source_node = main_node)
			# it has to be rounded as the number written to the disk is rounded,
			#  so if there is discrepancy one cannot reproduce iteration.
			shakenumber  = round(shakenumber,5)

			rangle = shakenumber*delta[N_step]
			rshift = shakenumber*step[N_step]
			refang = sparx_fundamentals.rotate_params(refang, [-rangle,-rangle,-rangle])

			trans = [tr_dummy]*nima
			pixer = [0.0]*nima
			if(an[N_step] > 0):
				#  These are local searches.
				# generate list of angles
				pass#IMPORTIMPORTIMPORT from alignment import generate_list_of_reference_angles_for_search
				list_of_reference_angles = sparx_alignment.generate_list_of_reference_angles_for_search(refang, sym=syms[0])
				#del ref_angles
			else:  list_of_reference_angles = [[1.0,1.0]]
 
		cs = [0.0]*3
		for iref in range(numref):
			start_time = time.time()
			if myid==main_node:
				volft = sparx_utilities.get_im(os.path.join(outdir, "volf%04d.hdf"%(total_iter-1)), iref)
			else:
				volft=sparx_utilities.model_blank(nx,nx,nx)
			sparx_utilities.bcast_EMData_to_all(volft, myid, main_node)
			volft, kb = sparx_projection.prep_vol(volft)
			refrings = sparx_alignment.prepare_refrings(volft, kb, nx, delta[N_step], refang, syms[0], numr, True)
			del volft, kb
			if myid == main_node:
				log.add( "Time to prepare %d rings: %g" % (len(refrings),time.time()-start_time) );start_time = time.time()

			"""Multiline Comment23"""

			for im in range(nima):
				"""Multiline Comment24"""

				if runtype=="ASSIGNMENT":
					phi,tht,psi,s2x,s2y = sparx_utilities.get_params_proj(data[im])
					ref = sparx_projection.prgs( volft, kb, [phi,tht,psi,-s2x,-s2y])
					if CTF:  ref = sparx_filter.filt_ctf( ref, ctf )
					peak = ref.cmp("ccc",data[im],{"mask":mask2D, "negative":0})
					if not(finfo is None):
						finfo.write( "ID,iref,peak: %6d %d %8.5f\n" % (list_of_particles[im],iref,peak) )
				else:
					if an[N_step] == -1:
						peak, pixel_error = sparx_alignment.proj_ali_incore(data[im], refrings, numr, xrng[N_step], yrng[N_step], step[N_step], rshift = rshift)
					else:
						peak, pixel_error = sparx_alignment.proj_ali_incore_local(data[im], refrings, list_of_reference_angles, numr,\
																	xrng[N_step], yrng[N_step], step[N_step], an[N_step])
					if not(finfo is None):
						phi,tht,psi,s2x,s2y = sparx_utilities.get_params_proj(data[im])
						finfo.write( "ID,iref,peak,trans: %6d %d %f %f %f %f %f %f\n"%(list_of_particles[im],iref,peak,phi,tht,psi,s2x,s2y) )
						finfo.flush()

				if peak > peaks[im]:
					peaks[im] = peak
					data[im].set_attr('group', iref)
					if runtype=="REFINEMENT":
						pixer[im] = pixel_error
						trans[im] = data[im].get_attr( "xform.projection" )
					if not(finfo is None):
						finfo.write( " current best\n" )
						finfo.flush()
				else:
					if not(finfo is None):
						finfo.write( "\n" )
						finfo.flush()
			if myid == main_node:
				log.add( "Time to process particles for reference %3d: %g" % (iref, time.time()-start_time) );start_time = time.time()


		del peaks
		if runtype=="ASSIGNMENT":  del volft, kb, ref
		else:
			#if CTF: del prjref
			del refrings
			if an[N_step] > 0: del list_of_reference_angles


		#  compute number of particles that changed assignment and how man are in which group
		nchng = 0
		npergroup = [0]*numref
		for im in range(nima):
			iref = data[im].get_attr('group')
			npergroup[iref] += 1
			if iref != assignment[im]:
				assignment[im] = iref
				nchng += 1
		nchng = mpi.mpi_reduce(nchng, 1, mpi.MPI_INT, mpi.MPI_SUM, 0, mpi.MPI_COMM_WORLD)
		npergroup = mpi.mpi_reduce(npergroup, numref, mpi.MPI_INT, mpi.MPI_SUM, 0, mpi.MPI_COMM_WORLD)
		npergroup = list(map(int, npergroup))
		terminate  = 0
		empty_group =0
		if myid == main_node:
			nchng = int(nchng[0])
			precn = 100*float(nchng)/float(total_nima)
			msg = " Number of particles that changed assignments %7d, percentage of total: %5.1f"%(nchng, precn)
			log.add(msg)
			msg = " Group       number of particles"
			log.add(msg)
			for iref in range(numref):
				msg = " %5d       %7d"%(iref+1, npergroup[iref])
				log.add(msg)
				if npergroup[iref]==0:
					empty_group =1
			if precn <= termprec:  
				terminate = 1
			if empty_group ==1:
				terminate = 1
		terminate = mpi.mpi_bcast(terminate, 1, mpi.MPI_INT, 0, mpi.MPI_COMM_WORLD)
		terminate = int(terminate[0])
		empty_group = mpi.mpi_bcast(empty_group, 1, mpi.MPI_INT, 0, mpi.MPI_COMM_WORLD)
		empty_group = int(empty_group[0])
		if empty_group ==1: break # program stops whenever empty_group appears!
		if runtype=="REFINEMENT":
			for im in range(nima):
				data[im].set_attr('xform.projection', trans[im])


			if center == -1:
				cs[0], cs[1], cs[2], dummy, dummy = sparx_utilities.estimate_3D_center_MPI(data, total_nima, myid, number_of_proc, main_node)				
				if myid == main_node:
					msg = " Average center x = %10.3f        Center y = %10.3f        Center z = %10.3f"%(cs[0], cs[1], cs[2])
					log.add(msg)
				cs = mpi.mpi_bcast(cs, 3, mpi.MPI_FLOAT, main_node, mpi.MPI_COMM_WORLD)
				cs = [-float(cs[0]), -float(cs[1]), -float(cs[2])]
				sparx_utilities.rotate_3D_shift(data, cs)
			#output pixel errors
			pass#IMPORTIMPORTIMPORT from mpi import mpi_gatherv
			recvbuf = mpi.mpi_gatherv(pixer, nima, mpi.MPI_FLOAT, recvcount, disps, mpi.MPI_FLOAT, main_node, mpi.MPI_COMM_WORLD)
			mpi.mpi_barrier(mpi.MPI_COMM_WORLD)
			if myid == main_node:
				recvbuf = list(map(float, recvbuf))
				pass#IMPORTIMPORTIMPORT from statistics import hist_list
				lhist = 20
				region, histo = sparx_statistics.hist_list(recvbuf, lhist)
				if region[0] < 0.0:  region[0] = 0.0
				msg = "      Histogram of pixel errors\n      ERROR       number of particles"
				log.add(msg)
				for lhx in range(lhist):
					msg = " %10.3f      %7d"%(region[lhx], histo[lhx])
					log.add(msg)
				del region, histo
			del recvbuf


			for im in range(nima):
				phi,theta,psi,tx,ty = sparx_utilities.get_params_proj(data[im])
				trans[im] = [phi,theta,psi,tx,ty]
			if myid == main_node:
				all_trans = []
				for klm in range(number_of_proc):
					if(klm == main_node):  all_trans.extend(copy.deepcopy(trans))
					else:  all_trans.extend(sparx_utilities.wrap_mpi_recv(klm, mpi.MPI_COMM_WORLD))
			else:  sparx_utilities.wrap_mpi_send(trans, main_node, mpi.MPI_COMM_WORLD)
			if myid == main_node:
				sparx_utilities.write_text_row(all_trans, os.path.join(outdir, "params_%04d.txt"%(total_iter)) )
				del all_trans


		if myid == main_node:
			all_trans = []
			for klm in range(number_of_proc):
				if(klm == main_node):  all_trans.extend(copy.deepcopy(assignment))
				else:  all_trans.extend(sparx_utilities.wrap_mpi_recv(klm, mpi.MPI_COMM_WORLD))
		else:  sparx_utilities.wrap_mpi_send(assignment, main_node, mpi.MPI_COMM_WORLD)
		if myid == main_node:
			sparx_utilities.write_text_file(all_trans, os.path.join(outdir, "assignment_%04d.txt"%(total_iter)) )
			del all_trans

		#if CTF: del vol
		fscc = [None]*numref

		if fourvar and runtype=="REFINEMENT":
			sumvol = sparx_utilities.model_blank(nx, nx, nx)

		sart_time = time.time()
		for iref in range(numref):
			#  3D stuff
			pass#IMPORTIMPORTIMPORT from time import localtime, strftime
			if CTF: volref, fscc[iref] = sparx_reconstruction.rec3D_MPI(data, snr, syms[iref], fscmask, os.path.join(outdir, "resolution_%04d_%02d.txt"%(total_iter, iref)), myid, main_node, index = iref, npad = npad, finfo=frec)
			else:    volref, fscc[iref] = sparx_reconstruction.rec3D_MPI_noCTF(data, syms[iref], fscmask, os.path.join(outdir, "resolution_%04d_%02d.txt"%(total_iter, iref)), myid, main_node, index = iref, npad = npad, finfo=frec)
			if myid == main_node:
				log.add( "Time to compute 3D: %d" % (time.time()-start_time) );start_time = time.time()

			if myid == main_node:
				volref.write_image(os.path.join(outdir, "vol%04d.hdf"%( total_iter)), iref)
				if fourvar and runtype=="REFINEMENT":
					sumvol += volref
			del volref

		if runtype=="REFINEMENT":
			if fourvar:
				varf = sparx_statistics.varf3d_MPI(data, os.path.join(outdir, "ssnr%04d"%total_iter), None,sumvol,last_ring, 1.0, 1, CTF, 1, sym, myid)
				if myid == main_node:   
					varf = 1.0/varf
					varf.write_image( os.path.join(outdir,"varf%04d.hdf"%total_iter) )

		if myid == main_node:
			refdata = [None]*7
			refdata[0] = numref
			refdata[1] = outdir
			refdata[2] = None
			refdata[3] = total_iter
			refdata[4] = varf
			refdata[5] = mask3D
			refdata[6] = (runtype=="REFINEMENT") # whether align on 50S, this only happens at refinement step
			user_func( refdata )

		#  here we  write header info
		mpi.mpi_barrier(mpi.MPI_COMM_WORLD)
		#start_time = time()
		#if runtype=="REFINEMENT":
		#	par_str = ['xform.projection', 'ID', 'group']
		#else:
		#	par_str = ['group', 'ID' ]
	        #if myid == main_node:
		#	from utilities import file_type
	        #	if file_type(stack) == "bdb":
	        #		from utilities import recv_attr_dict_bdb
	        #		recv_attr_dict_bdb(main_node, stack, data, par_str, image_start, image_end, number_of_proc)
	        #	else:
	        #		from utilities import recv_attr_dict
	        #		recv_attr_dict(main_node, stack, data, par_str, image_start, image_end, number_of_proc)
	        #else:		send_attr_dict(main_node, data, par_str, image_start, image_end)
		if terminate == 1:
			if myid == main_node:
				log.add("Kmref_ali3d_MPI terminated due to small number of objects changing assignments")
			break
		#if myid == main_node:
		#	log.add( "Time to write headers: %d\n" % (time()-start_time) );start_time = time()
	######writing paritition only in the end of the program
	mpi.mpi_barrier(mpi.MPI_COMM_WORLD)
	"""Multiline Comment25"""
	if myid == main_node:
		log.add("Kmref_ali3d_MPI is done!")
	#return empty_group

def get_refiparams(nx):
	pass#IMPORTIMPORTIMPORT from EMAN2 import Processor
	M = nx
	npad = 2
	N = M*npad
	K = 6
	alpha = 1.75
	r = M/2
	v = K/2.0/N
	return {"filter_type": EMAN2_cppwrap.Processor.fourier_filter_types.KAISER_SINH_INVERSE, "alpha":alpha, "K":K, "r":r, "v":v, "N":N}

def local_ali3dm_MPI_(stack, refvol, outdir, maskfile, ou=-1,  delta=2, ts=0.25, maxit=10, nassign=4, nrefine=1, CTF = None,
                snr=1.0, sym="c1", user_func_name="ref_ali3d", fourvar=False, debug=False, termprec = 0.0 ):
	"""
	  Focus on intersubunit region	
	"""
	pass#IMPORTIMPORTIMPORT from alignment	    import eqproj_cascaded_ccc
	pass#IMPORTIMPORTIMPORT from filter         import filt_ctf, filt_params, filt_table, filt_from_fsc, filt_btwl, filt_tanl, filt_vols
	pass#IMPORTIMPORTIMPORT from fundamentals   import fshift, rot_avg_image
	pass#IMPORTIMPORTIMPORT from projection     import prep_vol, prgs, project
	pass#IMPORTIMPORTIMPORT from utilities      import amoeba_multi_level, model_circle, get_arb_params, set_arb_params, drop_spider_doc
	pass#IMPORTIMPORTIMPORT from utilities      import bcast_number_to_all, bcast_list_to_all,get_image, drop_image, bcast_EMData_to_all, send_attr_dict
	pass#IMPORTIMPORTIMPORT from utilities      import get_params_proj, set_params_proj, get_im
	pass#IMPORTIMPORTIMPORT from utilities      import model_blank, print_begin_msg, print_msg, print_end_msg, file_type
	pass#IMPORTIMPORTIMPORT from reconstruction import rec3D_MPI
	pass#IMPORTIMPORTIMPORT from statistics     import ccc
	pass#IMPORTIMPORTIMPORT from pixel_error    import max_3D_pixel_error
	pass#IMPORTIMPORTIMPORT from math           import pi, sqrt
	pass#IMPORTIMPORTIMPORT from string         import replace
	pass#IMPORTIMPORTIMPORT from mpi            import mpi_bcast, mpi_comm_size, mpi_comm_rank, MPI_FLOAT, MPI_COMM_WORLD, mpi_barrier
	pass#IMPORTIMPORTIMPORT from mpi            import mpi_reduce, MPI_INT, MPI_SUM
	pass#IMPORTIMPORTIMPORT from utilities      import estimate_3D_center_MPI, rotate_3D_shift
	pass#IMPORTIMPORTIMPORT from EMAN2 import Processor
	pass#IMPORTIMPORTIMPORT import os
	pass#IMPORTIMPORTIMPORT import sys
	pass#IMPORTIMPORTIMPORT from EMAN2 import Vec2f


	number_of_proc = mpi.mpi_comm_size(mpi.MPI_COMM_WORLD)
	myid           = mpi.mpi_comm_rank(mpi.MPI_COMM_WORLD)
	main_node = 0

	if os.path.exists(outdir):  sparx_global_def.ERROR('Output directory exists, please change the name and restart the program', "local_ali3dm_MPI_ ", 1,myid)
	mpi.mpi_barrier(mpi.MPI_COMM_WORLD)
	
	if myid == main_node:
		os.mkdir(outdir)
		pass#IMPORTIMPORTIMPORT import global_def
		sparx_global_def.LOGFILE =  os.path.join(outdir, sparx_global_def.LOGFILE)
	
	mpi.mpi_barrier(mpi.MPI_COMM_WORLD)



	if debug:
		pass#IMPORTIMPORTIMPORT from time import sleep
		while not os.path.exists(outdir):
			print("Node ",myid,"  waiting...")
			time.sleep(5)

		finfo = open(os.path.join(outdir, "progress%04d"%myid), 'w')
	else:
		finfo = None

	pass#IMPORTIMPORTIMPORT from time import time


	# refine step define on which step refinement will be carried
	# if set to -1, no (??) refinement only assignment 

	nx  = sparx_utilities.get_image( refvol ).get_xsize()
	ou = int(ou)
	if(ou <= 0):  ou = nx//2-2
	if maskfile:
		pass#IMPORTIMPORTIMPORT import  types
		if(type(maskfile) is bytes): 
			mask3D = sparx_utilities.get_image(maskfile)
		else:   
			mask3D = maskfile
	else:
		mask3D = sparx_utilities.model_circle(ou, nx, nx, nx)

	mask2D  = sparx_utilities.model_circle(ou, nx, nx)
	fscmask = sparx_utilities.model_circle(ou, nx, nx, nx)

	numref = EMAN2_cppwrap.EMUtil.get_image_count(refvol)
	if myid==main_node:
		pass#IMPORTIMPORTIMPORT import user_functions
		user_func = sparx_user_functions.factory[user_func_name]
		for krf in range(numref):
			vol = sparx_utilities.get_im(refvol, krf)
			vol.write_image( os.path.join(outdir, "volf0000.hdf"), krf )
			vol = None
		sparx_utilities.print_begin_msg("local_ali3dm_MPI")
		sparx_utilities.print_msg("Input stack                 : %s\n"%(stack))
		sparx_utilities.print_msg("Reference volume            : %s\n"%(refvol))	
		sparx_utilities.print_msg("Number of reference volumes : %i\n"%(numref))
		sparx_utilities.print_msg("Output directory            : %s\n"%(outdir))
		sparx_utilities.print_msg("Maskfile                    : %s\n"%(maskfile))
		sparx_utilities.print_msg("Angular step                : %f\n"%(delta))
		sparx_utilities.print_msg("Shift search range          : %f\n"%(ts))

	if(myid == main_node):
		if(sparx_utilities.file_type(stack) == "bdb"):
			pass#IMPORTIMPORTIMPORT from EMAN2db import db_open_dict
			dummy = EMAN2db.db_open_dict(stack, True)
		# horatio active_refactoring Jy51i1EwmLD4tWZ9_00000_1
		# active = EMUtil.get_all_attributes(stack, "active")
		# list_of_particles = []
		# for im in xrange( len(active) ):
		# 	if( active[im] ) : list_of_particles.append(im)
		# del active
		# nima = len( list_of_particles )
		
		nima = EMAN2_cppwrap.EMUtil.get_image_count(stack)
		list_of_particles = list(range(nima))
	
		start_time = time.time()
	else:
		nima = 0

	nima = sparx_utilities.bcast_number_to_all( nima, source_node = main_node )

	if(myid != main_node):
		list_of_particles = [-1]*nima

	list_of_particles = sparx_utilities.bcast_list_to_all(list_of_particles, myid,  source_node = main_node)

	image_start, image_end = MPI_start_end(nima, number_of_proc, myid)
	# create a list of images for each node
	total_nima = nima
	list_of_particles = list_of_particles[image_start: image_end]
	nima = len(list_of_particles)

	if debug:
		finfo.write( "image_start, image_end: %d %d\n" % (image_start, image_end) )
		finfo.flush()

	data = EMAN2_cppwrap.EMData.read_images(stack, list_of_particles)	
	if(debug) :
		finfo.write( '%d loaded  \n' % len(data) )	
		finfo.flush()

	#  Initialize Particle ID and set group number to non-existant -1
	assignment = [-1]*len(data)
	for im in range(len(data)):
		data[im].set_attr('ID', list_of_particles[im])

	if fourvar:
		#  I am not sure why it is here!  PAP 09/26/09
		pass#IMPORTIMPORTIMPORT from reconstruction import rec3D_MPI
		pass#IMPORTIMPORTIMPORT from statistics     import varf3d_MPI
		#  Compute Fourier variance
		vol, fscc = sparx_reconstruction.rec3D_MPI(data, snr, sym, fscmask, os.path.join(outdir, "resolution0000"), myid, main_node, finfo=finfo)
		varf = sparx_statistics.varf3d_MPI(data, os.path.join(outdir, "ssnr0000"), None, vol, int(ou), 1.0, 1, CTF, 1, sym, myid)
		if myid == main_node:
			varf = 1.0/varf
			varf.write_image( os.path.join(outdir,"varf0000.hdf") )
			sparx_utilities.print_msg("Time to calculate 3D Fourier variance = %d\n"%(time.time()-start_time))
			start_time = time.time()
	else:
		varf = None


	if(CTF):
		if(data[0].get_attr("ctf_applied") > 0):
			sparx_global_def.ERROR( "local_ali3dm does not work on ctf_applied data", "local_ali3dm_MPI_", 1,myid)
		pass#IMPORTIMPORTIMPORT from reconstruction import rec3D_MPI
	else:
		pass#IMPORTIMPORTIMPORT from reconstruction import rec3D_MPI_noCTF
	

	#  this is needed for gathering of pixel errors
	disps = []
	recvcount = []
	for im in range(number_of_proc):
		if( im == main_node ):  disps.append(0)
		else:                  disps.append(disps[im-1] + recvcount[im-1])
		ib, ie = MPI_start_end(total_nima, number_of_proc, im)
		recvcount.append( ie - ib )



	if(myid == main_node):
		momm = sparx_utilities.get_im("varmask.hdf")
	else:
		momm =  sparx_utilities.model_blank(nx, nx, nx)
	sparx_utilities.bcast_EMData_to_all(momm, myid, main_node)
	pass#IMPORTIMPORTIMPORT from projection import project
	


	refiparams = get_refiparams(nx)

	maxit = maxit*(nassign+nrefine)
	Iter = -1
	iteration = 0
	while(Iter < maxit - 1):
		#for Iter in xrange(maxit):
		Iter += 1
		if Iter%(nassign+nrefine) < nassign :
			runtype = "ASSIGNMENT"
		else:
			runtype = "REFINEMENT"

		iteration += 1
		if(myid == main_node) :
			start_time = time.time()
			sparx_utilities.print_msg( runtype + (" ITERATION #%3d\n"%iteration) )

		peaks = [-1.0e23] * nima
		if runtype=="REFINEMENT":  pixer = [0.0]*nima

		for krf in range(numref):
			vol = sparx_utilities.get_im(os.path.join(outdir, "volf%04d.hdf"%(iteration-1)), krf)
			if CTF:
				previous_defocus = -1
			else:
				volft,kb = sparx_projection.prep_vol(vol)

			for im in range(nima):
				img = data[im]
				if CTF:
					ctf = img.get_attr( "ctf" )
					if ctf.defocus != previous_defocus:
						ctfvol = sparx_filter.filt_ctf( vol, ctf )
						volft, kb = sparx_projection.prep_vol( ctfvol )
						previous_defocus = ctf.defocus

				phi,tht,psi,s2x,s2y = sparx_utilities.get_params_proj(img)
				t1 = img.get_attr("xform.projection")
				dp = t1.get_params("spider")
				phi =  dp["phi"]
				tht =  dp["theta"]
				psi =  dp["psi"]
				s2x = -dp["tx"]
				s2y = -dp["ty"]
				if runtype=="ASSIGNMENT":
					pmomm = EMAN2_cppwrap.Util.addn_img(sparx_projection.project(momm, [phi,tht,psi,-s2x,-s2y], ou), mask2D)
					ref   = EMAN2_cppwrap.Util.muln_img(sparx_projection.prgs( volft, kb, [phi,tht,psi,-s2x,-s2y] ), pmomm)
					peak = ref.cmp("ccc",img,{"mask":mask2D, "negative":0})
					if not(finfo is None):
						finfo.write( "ID,iref,peak: %6d %d %f"%(list_of_particles[im],krf,peak) )
						finfo.flush()
				else:
					refi = img.process( "normalize.mask", {"mask":mask2D, "no_sigma":0} )
					refi = refi.FourInterpol(nx*2, nx*2, 1, True)
					refi = EMAN2_cppwrap.Processor.EMFourierFilter(refi, refiparams)
					refdata = [None]*7
					refdata[0] = volft
					refdata[1] = kb
					refdata[2] = img
					refdata[3] = mask2D
					refdata[4] = refi
					refdata[5] = [-s2x,-s2y]
					refdata[6] = ts
					weight_phi = max(delta, delta*abs((tht-90.0)/180.0*numpy.pi))
					[ang,peak,qiter,sft] = sparx_utilities.amoeba_multi_level([phi,tht,psi],[weight_phi,delta,weight_phi],sparx_alignment.eqproj_cascaded_ccc, 1.0,1.e-2, 500, refdata)
					if not(finfo is None):
						finfo.write( "ID,iref,peak,trans: %6d %d %f %f %f %f %f %f"%(list_of_particles[im],krf,peak,ang[0],ang[1],ang[2],-sft[0],-sft[1]) )
						finfo.flush()

				if(peak > peaks[im]):
					peaks[im] = peak
					data[im].set_attr( "group",  krf )
					if runtype=="REFINEMENT":
						t2 = EMAN2_cppwrap.Transform({"type":"spider","phi":ang[0],"theta":ang[1],"psi":ang[2]})
						t2.set_trans(EMAN2_cppwrap.Vec2f(sft[0], sft[1]))
						data[im].set_attr("xform.projection", t2)
						pixer[im] = sparx_pixel_error.max_3D_pixel_error(t1, t2, ou)

					if not(finfo is None):
						finfo.write( " current best" )

				if not(finfo is None):
					finfo.write( "\n" )
				if( myid== main_node and (im>0) and ( ((im)%(nima//2) == 0) or (im == nima-1) ) ):
					sparx_utilities.print_msg( "Time to process %6d particles : %d\n" % (nima//2, time.time()-start_time) )
					start_time = time.time()

		del peaks
		del vol, volft
		#  compute number of particles that changed assignments and how many are in which group
		nchng = 0
		npergroup = [0]*numref
		for im in range(nima):
			iref = data[im].get_attr('group')
			npergroup[iref] += 1
			if( iref != assignment[im]):
				assignment[im] = iref
				nchng += 1
		nchng = mpi.mpi_reduce(nchng, 1, mpi.MPI_INT, mpi.MPI_SUM, 0, mpi.MPI_COMM_WORLD)
		npergroup = mpi.mpi_reduce(npergroup, numref, mpi.MPI_INT, mpi.MPI_SUM, 0, mpi.MPI_COMM_WORLD)
		npergroup = list(map(int, npergroup))
		terminate = 0
		if( myid == 0 ):
			nchng = int(nchng[0])
			precn = 100*float(nchng)/float(total_nima)
			msg = " Number of particles that changed assignment %7d, percentage of total: %5.1f\n"%(nchng, precn)
			sparx_utilities.print_msg(msg)
			msg = " Group       number of particles\n"
			sparx_utilities.print_msg(msg)
			for iref in range(numref):
				msg = " %5d       %7d\n"%(iref+1, npergroup[iref])
				sparx_utilities.print_msg(msg)
			if(precn <= termprec):  terminate = 1
		terminate = mpi.mpi_bcast(terminate, 1, mpi.MPI_INT, 0, mpi.MPI_COMM_WORLD)
		terminate = int(terminate[0])


		if runtype=="REFINEMENT":
			if(True):
				cs = [0.0]*3
				cs[0],cs[1],cs[2],dummy,dummy = sparx_utilities.estimate_3D_center_MPI(data, total_nima, myid, number_of_proc, main_node)				
				if myid == main_node:
					msg = " Average center x = %10.3f        Center y = %10.3f        Center z = %10.3f\n"%(cs[0], cs[1], cs[2])
					sparx_utilities.print_msg(msg)
				cs = mpi.mpi_bcast(cs, 3, mpi.MPI_FLOAT, main_node, mpi.MPI_COMM_WORLD)
				cs = [-float(cs[0]), -float(cs[1]), -float(cs[2])]
				sparx_utilities.rotate_3D_shift(data, cs)
			#output pixel errors
			pass#IMPORTIMPORTIMPORT from mpi import mpi_gatherv
			recvbuf = mpi.mpi_gatherv(pixer, nima, mpi.MPI_FLOAT, recvcount, disps, mpi.MPI_FLOAT, main_node, mpi.MPI_COMM_WORLD)
			mpi.mpi_barrier(mpi.MPI_COMM_WORLD)
			if(myid == main_node):
				recvbuf = list(map(float, recvbuf))
				pass#IMPORTIMPORTIMPORT from statistics import hist_list
				lhist = 20
				region, histo = sparx_statistics.hist_list(recvbuf, lhist)
				if(region[0] < 0.0):  region[0] = 0.0
				msg = "      Histogram of pixel errors\n      ERROR       number of particles\n"
				sparx_utilities.print_msg(msg)
				for lhx in range(lhist):
					msg = " %10.3f     %7d\n"%(region[lhx], histo[lhx])
					sparx_utilities.print_msg(msg)
				del region, histo
			del recvbuf


		fscc = [None]*numref
		if fourvar and runtype=="REFINEMENT":
			sumvol = sparx_utilities.model_blank(nx, nx, nx)
		for krf in range(numref):
			if CTF:
				vol, fscc[krf] = sparx_reconstruction.rec3D_MPI(data, snr, sym, fscmask, os.path.join(outdir, "resolution%02d_%04d"%(krf, iteration)), myid, main_node, index = krf)
			else:
				vol, fscc[krf] = sparx_reconstruction.rec3D_MPI_noCTF(data, snr, sym, fscmask, os.path.join(outdir, "resolution%02d_%04d"%(krf, iteration)), myid, main_node, index = krf)

			if(myid==main_node):
				vol.write_image(os.path.join(outdir,"vol%04d.hdf"%iteration),krf)
				sparx_utilities.print_msg("3D reconstruction time = %d\n"%(time.time()-start_time))
				start_time = time.time()
				if fourvar and runtype=="REFINEMENT":
					sumvol += vol

		if runtype=="REFINEMENT":
			if fourvar:
				varf = sparx_statistics.varf3d_MPI(data, os.path.join(outdir, "ssnr%04d"%iteration), None, sumvol, int(ou), 1.0, 1, CTF, 1, sym, myid)
				if myid == main_node:   
					varf = 1.0/varf
					varf.write_image( os.path.join(outdir,"varf%04d.hdf"%iteration) )
					sparx_utilities.print_msg("Time to calculate 3D Fourier variance= %d\n"%(time.time()-start_time))
					start_time = time.time()


		if(myid == main_node):
			refdata = [None]*7
			refdata[0] = numref
			refdata[1] = outdir
			refdata[2] = None # fscc
			refdata[3] = iteration
			refdata[4] = varf
			refdata[5] = mask3D
			refdata[6] = (runtype=="REFINEMENT") # whether align on 50S, this only happens at refinement step
			user_func( refdata )


		mpi.mpi_barrier(mpi.MPI_COMM_WORLD)
		# here we should write header info, just in case the program crashes...
		# write out headers  and STOP, under MPI writing has to be done sequentially

		if runtype=="REFINEMENT":
			par_str = ["xform.projection", "group", "ID"]
		else:
			par_str = ["group", "ID"]

		if myid == main_node:
			pass#IMPORTIMPORTIMPORT from utilities import file_type
			if(sparx_utilities.file_type(stack) == "bdb"):
				pass#IMPORTIMPORTIMPORT from utilities import recv_attr_dict_bdb
				sparx_utilities.recv_attr_dict_bdb(main_node, stack, data, par_str, image_start, image_end, number_of_proc)
			else:
				pass#IMPORTIMPORTIMPORT from utilities import recv_attr_dict
				sparx_utilities.recv_attr_dict(main_node, stack, data, par_str, image_start, image_end, number_of_proc)
		else:		sparx_utilities.send_attr_dict(main_node, data, par_str, image_start, image_end)
		if myid == main_node:
			sparx_utilities.print_msg("Time to write header information= %d\n"%(time.time()-start_time))
			start_time = time.time()
		if(terminate == 1  and runtype=="ASSIGNMENT"):
			if myid==main_node:
				#print_end_msg("local_ali3dm_MPI terminated due to small number of objects changing assignments")
				sparx_utilities.print_msg("local_ali3dm_MPI abandoned assignments due to small number of objects changing assignments\n")
			pass#IMPORTIMPORTIMPORT from sys import exit
			exit()
			while(runtype == "ASSIGNMENT"):
				Iter += 1
				if Iter%(nassign+nrefine) < nassign :
					runtype = "ASSIGNMENT"
				else:
					runtype = "REFINEMENT"
			Iter += -1

	if myid==main_node:
		sparx_utilities.print_end_msg("local_ali3dm_MPI")


def local_ali3dm_MPI(stack, refvol, outdir, maskfile, ou=-1,  delta=2, ts=0.25, maxit=10, nassign=4, nrefine=1, CTF = None,
                snr=1.0, sym="c1", user_func_name="ref_ali3d", fourvar=False, npad = 4, debug=False, termprec = 0.0 ):
	"""
	  The original, fully operational version	
	"""
	pass#IMPORTIMPORTIMPORT from alignment	    import eqproj_cascaded_ccc
	pass#IMPORTIMPORTIMPORT from filter         import filt_ctf, filt_params, filt_table, filt_from_fsc, filt_btwl, filt_tanl, filt_vols
	pass#IMPORTIMPORTIMPORT from fundamentals   import fshift, rot_avg_image
	pass#IMPORTIMPORTIMPORT from projection     import prep_vol, prgs, project
	pass#IMPORTIMPORTIMPORT from utilities      import amoeba_multi_level, model_circle, get_arb_params, set_arb_params, drop_spider_doc
	pass#IMPORTIMPORTIMPORT from utilities      import bcast_number_to_all, bcast_list_to_all,get_image, drop_image, bcast_EMData_to_all, send_attr_dict
	pass#IMPORTIMPORTIMPORT from utilities      import get_params_proj, set_params_proj, get_im
	pass#IMPORTIMPORTIMPORT from utilities      import model_blank, print_begin_msg, print_msg, print_end_msg, file_type
	pass#IMPORTIMPORTIMPORT from reconstruction import rec3D_MPI
	pass#IMPORTIMPORTIMPORT from statistics     import ccc
	pass#IMPORTIMPORTIMPORT from pixel_error    import max_3D_pixel_error
	pass#IMPORTIMPORTIMPORT from math           import pi, sqrt
	pass#IMPORTIMPORTIMPORT from string         import replace
	pass#IMPORTIMPORTIMPORT from mpi            import mpi_bcast, mpi_comm_size, mpi_comm_rank, MPI_FLOAT, MPI_COMM_WORLD, mpi_barrier
	pass#IMPORTIMPORTIMPORT from mpi            import mpi_reduce, MPI_INT, MPI_SUM
	pass#IMPORTIMPORTIMPORT from utilities      import estimate_3D_center_MPI, rotate_3D_shift
	pass#IMPORTIMPORTIMPORT from EMAN2 import Processor
	pass#IMPORTIMPORTIMPORT import os
	pass#IMPORTIMPORTIMPORT import sys
	pass#IMPORTIMPORTIMPORT from EMAN2 import Vec2f


	number_of_proc = mpi.mpi_comm_size(mpi.MPI_COMM_WORLD)
	myid           = mpi.mpi_comm_rank(mpi.MPI_COMM_WORLD)
	main_node = 0

	if os.path.exists(outdir):  sparx_global_def.ERROR('Output directory exists, please change the name and restart the program', "local_ali3dm_MPI ", 1, myid)
	mpi.mpi_barrier(mpi.MPI_COMM_WORLD)

	if(myid == main_node):
		pass#IMPORTIMPORTIMPORT import global_def
		sparx_global_def.LOGFILE =  os.path.join(outdir, sparx_global_def.LOGFILE)
		os.mkdir(outdir)
		sparx_utilities.print_begin_msg("local_ali3dm_MPI")
	
	mpi.mpi_barrier(mpi.MPI_COMM_WORLD)



	if debug:
		pass#IMPORTIMPORTIMPORT from time import sleep
		while not os.path.exists(outdir):
			print("Node ",myid,"  waiting...")
			time.sleep(5)

		finfo = open(os.path.join(outdir, "progress%04d"%myid), 'w')
	else:
		finfo = None

	pass#IMPORTIMPORTIMPORT from time import time


	# refine step define on which step refinement will be carried
	# if set to -1, no (??) refinement only assignment 

	nx  = sparx_utilities.get_image( refvol ).get_xsize()
	ou = int(ou)
	if(ou <= 0):  ou = nx//2-2
	if maskfile:
		pass#IMPORTIMPORTIMPORT import  types
		if(type(maskfile) is bytes): 
			mask3D = sparx_utilities.get_image(maskfile)
		else:   
			mask3D = maskfile
	else:
		mask3D = sparx_utilities.model_circle(ou, nx, nx, nx)

	mask2D  = sparx_utilities.model_circle(ou, nx, nx)
	fscmask = sparx_utilities.model_circle(ou, nx, nx, nx)

	numref = EMAN2_cppwrap.EMUtil.get_image_count(refvol)
	if myid==main_node:
		pass#IMPORTIMPORTIMPORT import user_functions
		user_func = sparx_user_functions.factory[user_func_name]
		for krf in range(numref):
			vol = sparx_utilities.get_im(refvol, krf)
			vol.write_image( os.path.join(outdir, "volf0000.hdf"), krf )
			vol = None
		sparx_utilities.print_begin_msg("local_ali3dm_MPI")
		sparx_utilities.print_msg("Input stack                 : %s\n"%(stack))
		sparx_utilities.print_msg("Reference volume            : %s\n"%(refvol))	
		sparx_utilities.print_msg("Number of reference volumes : %i\n"%(numref))
		sparx_utilities.print_msg("Output directory            : %s\n"%(outdir))
		sparx_utilities.print_msg("Maskfile                    : %s\n"%(maskfile))
		sparx_utilities.print_msg("Angular step                : %f\n"%(delta))
		sparx_utilities.print_msg("Shift search range          : %f\n"%(ts))
		sparx_utilities.print_msg("User function               : %s\n"%(user_func_name))

	if(myid == main_node):
		if(sparx_utilities.file_type(stack) == "bdb"):
			pass#IMPORTIMPORTIMPORT from EMAN2db import db_open_dict
			dummy = EMAN2db.db_open_dict(stack, True)
		# horatio active_refactoring Jy51i1EwmLD4tWZ9_00000_1
		# active = EMUtil.get_all_attributes(stack, "active")
		# list_of_particles = []
		# for im in xrange( len(active) ):
		# 	if( active[im] ) : list_of_particles.append(im)
		# del active
		# nima = len( list_of_particles )
		
		nima = EMAN2_cppwrap.EMUtil.get_image_count(stack)
		list_of_particles = list(range(nima))
	
		start_time = time.time()
	else:
		nima = 0

	nima = sparx_utilities.bcast_number_to_all( nima, source_node = main_node )

	if(myid != main_node):
		list_of_particles = [-1]*nima

	list_of_particles = sparx_utilities.bcast_list_to_all(list_of_particles, myid,  source_node = main_node)

	image_start, image_end = MPI_start_end(nima, number_of_proc, myid)
	# create a list of images for each node
	total_nima = nima
	list_of_particles = list_of_particles[image_start: image_end]
	nima = len(list_of_particles)

	if debug:
		finfo.write( "image_start, image_end: %d %d\n" % (image_start, image_end) )
		finfo.flush()

	data = EMAN2_cppwrap.EMData.read_images(stack, list_of_particles)	
	if(debug) :
		finfo.write( '%d loaded  \n' % len(data) )	
		finfo.flush()

	#  Initialize Particle ID and set group number to non-existant -1
	assignment = [-1]*len(data)
	for im in range(len(data)):
		data[im].set_attr('ID', list_of_particles[im])

	if fourvar:
		#  I am not sure why it is here!  PAP 09/26/09
		pass#IMPORTIMPORTIMPORT from reconstruction import rec3D_MPI
		pass#IMPORTIMPORTIMPORT from statistics     import varf3d_MPI
		#  Compute Fourier variance
		vol, fscc = sparx_reconstruction.rec3D_MPI(data, snr, sym, fscmask, os.path.join(outdir, "resolution0000"), myid, main_node, finfo=finfo, npad = npad)
		varf = sparx_statistics.varf3d_MPI(data, os.path.join(outdir, "ssnr0000"), None, vol, int(ou), 1.0, 1, CTF, 1, sym, myid)
		if myid == main_node:
			varf = 1.0/varf
			varf.write_image( os.path.join(outdir,"varf0000.hdf") )
			sparx_utilities.print_msg("Time to calculate 3D Fourier variance = %d\n"%(time.time()-start_time))
			start_time = time.time()
	else:
		varf = None


	if(CTF):
		if(data[0].get_attr("ctf_applied") > 0):
			sparx_global_def.ERROR( "local_ali3dm does not work on ctf_applied data", "local_ali3dm_MPI", 1,myid)
		pass#IMPORTIMPORTIMPORT from reconstruction import rec3D_MPI
	else:
		pass#IMPORTIMPORTIMPORT from reconstruction import rec3D_MPI_noCTF


	#  this is needed for gathering of pixel errors
	disps = []
	recvcount = []
	for im in range(number_of_proc):
		if( im == main_node ):  disps.append(0)
		else:                  disps.append(disps[im-1] + recvcount[im-1])
		ib, ie = MPI_start_end(total_nima, number_of_proc, im)
		recvcount.append( ie - ib )
	
	refiparams = get_refiparams(nx)

	maxit = maxit*(nassign+nrefine)
	Iter = -1
	iteration = 0
	while(Iter < maxit - 1):
		#for Iter in xrange(maxit):
		Iter += 1
		if Iter%(nassign+nrefine) < nassign :
			runtype = "ASSIGNMENT"
		else:
			runtype = "REFINEMENT"

		iteration += 1
		if(myid == main_node) :
			start_time = time.time()
			sparx_utilities.print_msg( runtype + (" ITERATION #%3d\n"%iteration) )

		peaks = [-1.0e23] * nima
		if runtype=="REFINEMENT":  pixer = [0.0]*nima

		for krf in range(numref):
			vol = sparx_utilities.get_im(os.path.join(outdir, "volf%04d.hdf"%(iteration-1)), krf)
			if CTF:
				previous_defocus = -1
			else:
				volft,kb = sparx_projection.prep_vol(vol)

			for im in range(nima):
				img = data[im]
				if CTF:
					ctf = img.get_attr( "ctf" )
					if ctf.defocus != previous_defocus:
						ctfvol = sparx_filter.filt_ctf( vol, ctf )
						volft, kb = sparx_projection.prep_vol( ctfvol )
						previous_defocus = ctf.defocus

				phi,tht,psi,s2x,s2y = sparx_utilities.get_params_proj(img)
				t1 = img.get_attr("xform.projection")
				dp = t1.get_params("spider")
				phi =  dp["phi"]
				tht =  dp["theta"]
				psi =  dp["psi"]
				s2x = -dp["tx"]
				s2y = -dp["ty"]
				if runtype=="ASSIGNMENT":
					ref  = sparx_projection.prgs( volft, kb, [phi,tht,psi,-s2x,-s2y] )
					peak = ref.cmp("ccc",img,{"mask":mask2D, "negative":0})
					if not(finfo is None):
						finfo.write( "ID,iref,peak: %6d %d %f"%(list_of_particles[im],krf,peak) )
						finfo.flush()
				else:
					refi = img.process( "normalize.mask", {"mask":mask2D, "no_sigma":0} )
					refi = refi.FourInterpol(nx*2, nx*2, 1, True)
					refi = EMAN2_cppwrap.Processor.EMFourierFilter(refi, refiparams)
					refdata = [None]*7
					refdata[0] = volft
					refdata[1] = kb
					refdata[2] = img
					refdata[3] = mask2D
					refdata[4] = refi
					refdata[5] = [-s2x,-s2y]
					refdata[6] = ts
					weight_phi = max(delta, delta*abs((tht-90.0)/180.0*numpy.pi))
					[ang,peak,qiter,sft] = sparx_utilities.amoeba_multi_level([phi,tht,psi],[weight_phi,delta,weight_phi],sparx_alignment.eqproj_cascaded_ccc, 1.0,1.e-2, 500, refdata)
					if not(finfo is None):
						finfo.write( "ID,iref,peak,trans: %6d %d %f %f %f %f %f %f"%(list_of_particles[im],krf,peak,ang[0],ang[1],ang[2],-sft[0],-sft[1]) )
						finfo.flush()

				if(peak > peaks[im]):
					peaks[im] = peak
					data[im].set_attr( "group",  krf )
					if runtype=="REFINEMENT":
						t2 = EMAN2_cppwrap.Transform({"type":"spider","phi":ang[0],"theta":ang[1],"psi":ang[2]})
						t2.set_trans(EMAN2_cppwrap.Vec2f(sft[0], sft[1]))
						data[im].set_attr("xform.projection", t2)
						pixer[im] = sparx_pixel_error.max_3D_pixel_error(t1, t2, ou)

					if not(finfo is None):
						finfo.write( " current best" )

				if not(finfo is None):
					finfo.write( "\n" )
				if( myid== main_node and (im>0) and ( ((im)%(nima//2) == 0) or (im == nima-1) ) ):
					sparx_utilities.print_msg( "Time to process %6d particles : %d\n" % (nima//2, time.time()-start_time) )
					start_time = time.time()

		del peaks
		del vol, volft
		#  compute number of particles that changed assignments and how many are in which group
		nchng = 0
		npergroup = [0]*numref
		for im in range(nima):
			iref = data[im].get_attr('group')
			npergroup[iref] += 1
			if( iref != assignment[im]):
				assignment[im] = iref
				nchng += 1
		nchng = mpi.mpi_reduce(nchng, 1, mpi.MPI_INT, mpi.MPI_SUM, 0, mpi.MPI_COMM_WORLD)
		npergroup = mpi.mpi_reduce(npergroup, numref, mpi.MPI_INT, mpi.MPI_SUM, 0, mpi.MPI_COMM_WORLD)
		npergroup = list(map(int, npergroup))
		terminate = 0
		if( myid == 0 ):
			nchng = int(nchng[0])
			precn = 100*float(nchng)/float(total_nima)
			msg = " Number of particles that changed assignment %7d, percentage of total: %5.1f\n"%(nchng, precn)
			sparx_utilities.print_msg(msg)
			msg = " Group       number of particles\n"
			sparx_utilities.print_msg(msg)
			for iref in range(numref):
				msg = " %5d       %7d\n"%(iref+1, npergroup[iref])
				sparx_utilities.print_msg(msg)
			if(precn <= termprec):  terminate = 1
		terminate = mpi.mpi_bcast(terminate, 1, mpi.MPI_INT, 0, mpi.MPI_COMM_WORLD)
		terminate = int(terminate[0])


		if runtype=="REFINEMENT":
			if(True):
				cs = [0.0]*3
				cs[0],cs[1],cs[2],dummy,dummy = sparx_utilities.estimate_3D_center_MPI(data, total_nima, myid, number_of_proc, main_node)				
				if myid == main_node:
					msg = " Average center x = %10.3f        Center y = %10.3f        Center z = %10.3f\n"%(cs[0], cs[1], cs[2])
					sparx_utilities.print_msg(msg)
				cs = mpi.mpi_bcast(cs, 3, mpi.MPI_FLOAT, main_node, mpi.MPI_COMM_WORLD)
				cs = [-float(cs[0]), -float(cs[1]), -float(cs[2])]
				sparx_utilities.rotate_3D_shift(data, cs)
			#output pixel errors
			pass#IMPORTIMPORTIMPORT from mpi import mpi_gatherv
			recvbuf = mpi.mpi_gatherv(pixer, nima, mpi.MPI_FLOAT, recvcount, disps, mpi.MPI_FLOAT, main_node, mpi.MPI_COMM_WORLD)
			mpi.mpi_barrier(mpi.MPI_COMM_WORLD)
			if(myid == main_node):
				recvbuf = list(map(float, recvbuf))
				pass#IMPORTIMPORTIMPORT from statistics import hist_list
				lhist = 20
				region, histo = sparx_statistics.hist_list(recvbuf, lhist)
				if(region[0] < 0.0):  region[0] = 0.0
				msg = "      Histogram of pixel errors\n      ERROR       number of particles\n"
				sparx_utilities.print_msg(msg)
				for lhx in range(lhist):
					msg = " %10.3f     %7d\n"%(region[lhx], histo[lhx])
					sparx_utilities.print_msg(msg)
				del region, histo
			del recvbuf


		fscc = [None]*numref
		if fourvar and runtype=="REFINEMENT":
			sumvol = sparx_utilities.model_blank(nx, nx, nx)
		for krf in range(numref):
			if CTF:
				vol, fscc[krf] = sparx_reconstruction.rec3D_MPI(data, snr, sym, fscmask, os.path.join(outdir, "resolution%02d_%04d"%(krf, iteration)), myid, main_node, index = krf, npad = npad)
			else:
				vol, fscc[krf] = sparx_reconstruction.rec3D_MPI_noCTF(data, snr, sym, fscmask, os.path.join(outdir, "resolution%02d_%04d"%(krf, iteration)), myid, main_node, index = krf, npad = npad)
				

			if(myid==main_node):
				vol.write_image(os.path.join(outdir,"vol%04d.hdf"%iteration),krf)
				sparx_utilities.print_msg("3D reconstruction time = %d\n"%(time.time()-start_time))
				start_time = time.time()
				if fourvar and runtype=="REFINEMENT":
					sumvol += vol

		if runtype=="REFINEMENT":
			if fourvar:
				varf = sparx_statistics.varf3d_MPI(data, os.path.join(outdir, "ssnr%04d"%iteration), None, sumvol, int(ou), 1.0, 1, CTF, 1, sym, myid)
				if myid == main_node:   
					varf = 1.0/varf
					varf.write_image( os.path.join(outdir,"varf%04d.hdf"%iteration) )
					sparx_utilities.print_msg("Time to calculate 3D Fourier variance= %d\n"%(time.time()-start_time))
					start_time = time.time()


		if(myid == main_node):
			refdata = [None]*7
			refdata[0] = numref
			refdata[1] = outdir
			refdata[2] = fscc
			refdata[3] = iteration
			refdata[4] = varf
			refdata[5] = mask3D
			refdata[6] = (runtype=="REFINEMENT") # whether align on 50S, this only happens at refinement step
			user_func( refdata )


		mpi.mpi_barrier(mpi.MPI_COMM_WORLD)
		# here we should write header info, just in case the program crashes...
		# write out headers  and STOP, under MPI writing has to be done sequentially

		if runtype=="REFINEMENT":
			par_str = ["xform.projection", "group", "ID"]
		else:
			par_str = ["group", "ID"]

		if myid == main_node:
			pass#IMPORTIMPORTIMPORT from utilities import file_type
			if(sparx_utilities.file_type(stack) == "bdb"):
				pass#IMPORTIMPORTIMPORT from utilities import recv_attr_dict_bdb
				sparx_utilities.recv_attr_dict_bdb(main_node, stack, data, par_str, image_start, image_end, number_of_proc)
			else:
				pass#IMPORTIMPORTIMPORT from utilities import recv_attr_dict
				sparx_utilities.recv_attr_dict(main_node, stack, data, par_str, image_start, image_end, number_of_proc)
		else:		sparx_utilities.send_attr_dict(main_node, data, par_str, image_start, image_end)
		if myid == main_node:
			sparx_utilities.print_msg("Time to write header information= %d\n"%(time.time()-start_time))
			start_time = time.time()
		if(terminate == 1  and runtype=="ASSIGNMENT"):
			if myid==main_node:
				#print_end_msg("local_ali3dm_MPI terminated due to small number of objects changing assignments")
				sparx_utilities.print_msg("local_ali3dm_MPI abandoned assignments due to small number of objects changing assignments\n")

			while(runtype == "ASSIGNMENT"):
				Iter += 1
				if Iter%(nassign+nrefine) < nassign :
					runtype = "ASSIGNMENT"
				else:
					runtype = "REFINEMENT"
			Iter += -1

	if myid==main_node:
		sparx_utilities.print_end_msg("local_ali3dm_MPI")


def local_ali3d(stack, outdir, maskfile = None, ou = -1,  delta = 2, ts=0.25, center = -1, maxit = 10, 
           CTF = False, snr = 1.0, sym = "c1", chunk = -1.0, user_func_name = "ref_ali3d",
	     fourvar = True, npad = 4, debug = False, MPI = False):
	"""
		
	"""

	if MPI:
		local_ali3d_MPI(stack, outdir, maskfile, ou, delta, ts, center, maxit,
				CTF, snr, sym, chunk, user_func_name, 
				fourvar, npad, debug)
		return

	pass#IMPORTIMPORTIMPORT from alignment      import eqproj_cascaded_ccc
	pass#IMPORTIMPORTIMPORT from projection     import prep_vol
	pass#IMPORTIMPORTIMPORT from utilities      import model_circle, get_params_proj, set_params_proj
	pass#IMPORTIMPORTIMPORT from utilities      import get_image, drop_image
	pass#IMPORTIMPORTIMPORT from utilities      import amoeba_multi_level, rotate_3D_shift, estimate_3D_center
	pass#IMPORTIMPORTIMPORT from math           import pi
	pass#IMPORTIMPORTIMPORT from statistics     import fsc_mask
	pass#IMPORTIMPORTIMPORT from utilities      import print_begin_msg, print_end_msg, print_msg
	pass#IMPORTIMPORTIMPORT from EMAN2 import Processor
	pass#IMPORTIMPORTIMPORT import os
	pass#IMPORTIMPORTIMPORT import sys

	if os.path.exists(outdir): sparx_global_def.ERROR('Output directory exists, please change the name and restart the program', "local_ali3d", 1)
	os.mkdir(outdir)
	pass#IMPORTIMPORTIMPORT import global_def
	sparx_global_def.LOGFILE =  os.path.join(outdir, sparx_global_def.LOGFILE)
	
	sparx_utilities.print_begin_msg('local_ali3d')

	pass#IMPORTIMPORTIMPORT import user_functions
	user_func = sparx_user_functions.factory[user_func_name]

	if CTF:
		ima = EMAN2_cppwrap.EMData()
		ima.read_image(stack, 0)
		ctf_applied = ima.get_attr("ctf_applied")
		del ima
		if ctf_applied == 1:  sparx_global_def.ERROR("local_ali3d does not work for CTF-applied data", "local_ali3d", 1)
		pass#IMPORTIMPORTIMPORT from reconstruction import recons3d_4nn_ctf
		pass#IMPORTIMPORTIMPORT from filter         import filt_ctf
	pass#IMPORTIMPORTIMPORT else   : from reconstruction import recons3d_4nn

	last_ring   = int(ou)
	max_iter    = int(maxit)
	center      = int(center)

	ima     = EMAN2_cppwrap.EMData()
	ima.read_image(stack, 0)
	nx      = ima.get_xsize()
	del ima
	if last_ring == -1:	last_ring = nx//2 - 2

	sparx_utilities.print_msg("Input stack                 : %s\n"%(stack))
	sparx_utilities.print_msg("Output directory            : %s\n"%(outdir))
	sparx_utilities.print_msg("Maskfile                    : %s\n"%(maskfile))
	sparx_utilities.print_msg("Outer radius                : %i\n"%(last_ring))
	sparx_utilities.print_msg("Angular search range        : %s\n"%(delta))
	sparx_utilities.print_msg("Translation search range    : %f\n"%(ts))
	sparx_utilities.print_msg("Maximum iteration           : %i\n"%(max_iter))
	sparx_utilities.print_msg("Center type                 : %i\n"%(center))
	sparx_utilities.print_msg("CTF correction              : %s\n"%(CTF))
	sparx_utilities.print_msg("Signal-to-Noise Ratio       : %f\n"%(snr))
	sparx_utilities.print_msg("Symmetry group              : %s\n"%(sym))
	if chunk <= 0.0:  chunk = 1.0
	sparx_utilities.print_msg("Chunk size                  : %f\n\n"%(chunk))
	sparx_utilities.print_msg("User function               : %s\n"%(user_func_name))
	
	if maskfile:
		pass#IMPORTIMPORTIMPORT import  types
		if type(maskfile) is bytes:  mask3D = sparx_utilities.get_image(maskfile)
		else:                                  mask3D = maskfile
	else:
		mask3D = sparx_utilities.model_circle(last_ring, nx, nx, nx)
	mask2D = sparx_utilities.model_circle(last_ring, nx, nx)


	if debug:  outf = open(os.path.join(outdir, "progress"), "w")
	else:      outf = None

	# horatio active_refactoring Jy51i1EwmLD4tWZ9_00000_1
	# active = EMUtil.get_all_attributes(stack, 'active')
	# list_of_particles = []
	# for im in xrange(len(active)):
	# 	if(active[im]):  list_of_particles.append(im)
	# del active
	
	nima = EMAN2_cppwrap.EMUtil.get_image_count(stack)
	list_of_particles = list(range(nima))
	
	dataim = EMAN2_cppwrap.EMData.read_images(stack, list_of_particles)
	nima = len(dataim)

	if debug:
		outf.write("  data read")
		outf.write("\n")
		outf.flush()

	n_of_chunks = int(1.0/chunk)
	
	if debug:
		outf = open(os.path.join(outdir, "progress"), "w")
		outf.write("  chunk = "+str(chunk)+"   ")
		outf.write("\n")
		outf.flush()
		outf.write("  chunk = "+str(n_of_chunks)+"   ")
		outf.write("\n")
		outf.flush()

	# initialize data for the reference preparation function
	ref_data = [mask3D, center, None, None]

	M = nx
	npad = 2
	N = M*npad
	K = 6
	alpha = 1.75
	r = M/2
	v = K/2.0/N
	params = {"filter_type": EMAN2_cppwrap.Processor.fourier_filter_types.KAISER_SINH_INVERSE, "alpha":alpha, "K":K, "r":r, "v":v, "N":N}

	data = [None]*7
	data[3] = mask2D
	cs = [0.0]*3

	for iteration in range(maxit+1):
		sparx_utilities.print_msg("ITERATION #%3d\n"%(iteration+1))
		for ic in range(n_of_chunks):
			if(center == -1):
				cs[0], cs[1], cs[2], dummy, dummy = sparx_utilities.estimate_3D_center(dataim)				
				sparx_utilities.rotate_3D_shift(dataim, [-cs[0], -cs[1], -cs[2]])
				msg = "Average center x = %10.3f        Center y = %10.3f        Center z = %10.3f\n"%(cs[0], cs[1], cs[2])
				sparx_utilities.print_msg(msg)				
			# compute updated 3D at the beginning of each chunk
			#  3D stuff
			if CTF: vol1 = sparx_reconstruction.recons3d_4nn_ctf(dataim, list(range(0, nima, 2)), snr, 1, sym)
			else:	   vol1 = sparx_reconstruction.recons3d_4nn(dataim, list(range(0, nima, 2)), sym, snr = snr)

			if CTF: vol2 = sparx_reconstruction.recons3d_4nn_ctf(dataim, list(range(1, nima, 2)), snr, 1, sym)
			else:	   vol2 = sparx_reconstruction.recons3d_4nn(dataim, list(range(1, nima, 2)), sym, snr = snr)

			# resolution
			fscc = sparx_statistics.fsc_mask(vol1, vol2, mask3D, 1.0, os.path.join(outdir, "resolution%04d"%(iteration*n_of_chunks+ic+1)))
			del vol1
			del vol2

			# calculate new and improved 3D
			if CTF: vol = sparx_reconstruction.recons3d_4nn_ctf(dataim, list(range(nima)), snr, 1, sym)
			else:	   vol = sparx_reconstruction.recons3d_4nn(dataim, list(range(nima)), sym, snr = snr)

			# store the reference volume
			sparx_utilities.drop_image(vol, os.path.join(outdir, "vol%04d.hdf"%(iteration*n_of_chunks+ic+1)))
			ref_data[2] = vol
			ref_data[3] = fscc

			#  call user-supplied function to prepare reference image, i.e., center and filter it
			vol, dummy = user_func(ref_data)

			sparx_utilities.drop_image(vol, os.path.join(outdir, "volf%04d.hdf"%(iteration*n_of_chunks+ic+1)))
			if(iteration == maxit):
				#  in last iteration quit here
				sparx_utilities.print_end_msg("local_ali3d")
				return

			if not CTF:
				data[0], data[1] = sparx_projection.prep_vol(vol)

			image_start_in_chunk = ic*nima/n_of_chunks
			image_end_in_chunk   = (ic+1)*nima/n_of_chunks
			if debug:
				outf.write("image_start_in_chunk "+str(image_start_in_chunk)+"\n")
				outf.write("\n")
				outf.write("image_end_in_chunk "+str(image_end_in_chunk)+"\n")
				outf.write("\n")
				outf.flush()
			if CTF:  previous_defocus = -1.0
			for imn in range(image_start_in_chunk, image_end_in_chunk):
				if CTF:
					ctf_params = dataim[imn].get_attr( "ctf" )
					if ctf_params.defocus != previous_defocus:
						previous_defocus = ctf_params.defocus
						data[0], data[1] = sparx_projection.prep_vol(sparx_filter.filt_ctf(vol, ctf_params, dopad=False))

				data[2] = dataim[imn]

				if ts > 0.0:
					refi = dataim[imn].FourInterpol(nx*2, nx*2, 1, False)
					data[4] = EMAN2_cppwrap.Processor.EMFourierFilter(refi, params)
				
				phi, theta, psi, tx, ty = sparx_utilities.get_params_proj(dataim[imn])
				atparams = [phi, theta, psi]
				data[5] = [tx, ty]
				data[6] = ts
				data[5][0] *= -1
				data[5][1] *= -1

				if debug:
					initial, dummy  = sparx_alignment.eqproj_cascaded_ccc(atparams, data)  # this is if we need initial discrepancy
					outf.write("Image "+str(imn)+"\n")
					outf.write('Old  %8.3f  %8.3f  %8.3f  %8.3f  %8.3f  %11.4f'%(phi,theta,psi,tx,ty,initial))
					outf.write("\n")
					
				# change signs of shifts for projections
			
				weight_phi = max(delta, delta*abs((atparams[1]-90.0)/180.0*numpy.pi))

				optm_params = sparx_utilities.amoeba_multi_level(atparams, [weight_phi, delta, weight_phi], sparx_alignment.eqproj_cascaded_ccc, 1.e-4, 1.e-4, 500, data)
				optm_params[0].append(optm_params[3][0])
				optm_params[0].append(optm_params[3][1])
				optm_params[0][3] *= -1
				optm_params[0][4] *= -1
				
				if debug:
					outf.write('New  %8.3f  %8.3f  %8.3f  %8.3f  %8.3f  %11.4f  %4d'%(optm_params[0][0], optm_params[0][1], optm_params[0][2], optm_params[0][3], optm_params[0][4], optm_params[1], optm_params[2]))
					outf.write("\n")
					outf.flush()

				sparx_utilities.set_params_proj(dataim[imn], optm_params[0])

			#  here we write header infomation
			pass#IMPORTIMPORTIMPORT from utilities import write_headers
			#write_headers(stack, dataim, list_of_particles)


def local_ali3d_MPI(stack, outdir, maskfile, ou = -1,  delta = 2, ts=0.25, center = -1, maxit = 10, 
                CTF = False, snr = 1.0, sym = "c1", chunk = -1.0, user_func_name = "ref_ali3d",
		    fourvar = True, npad = 4, debug = False):
	"""
		
	"""
	pass#IMPORTIMPORTIMPORT from alignment        import eqproj_cascaded_ccc
	pass#IMPORTIMPORTIMPORT from filter           import filt_ctf
	pass#IMPORTIMPORTIMPORT from projection       import prep_vol
	pass#IMPORTIMPORTIMPORT from utilities        import bcast_number_to_all, model_circle, get_params_proj, set_params_proj
	pass#IMPORTIMPORTIMPORT from utilities        import bcast_EMData_to_all, bcast_list_to_all, send_attr_dict
	pass#IMPORTIMPORTIMPORT from utilities        import get_image, drop_image, file_type
	pass#IMPORTIMPORTIMPORT from utilities        import amoeba_multi_level, rotate_3D_shift, estimate_3D_center_MPI
	pass#IMPORTIMPORTIMPORT from utilities        import print_begin_msg, print_end_msg, print_msg
	pass#IMPORTIMPORTIMPORT from reconstruction   import rec3D_MPI, rec3D_MPI_noCTF
	pass#IMPORTIMPORTIMPORT from statistics       import varf3d_MPI
	pass#IMPORTIMPORTIMPORT from math             import pi
	pass#IMPORTIMPORTIMPORT from mpi              import mpi_bcast, mpi_comm_size, mpi_comm_rank, MPI_FLOAT, MPI_COMM_WORLD, mpi_barrier
	pass#IMPORTIMPORTIMPORT from mpi              import mpi_reduce, MPI_INT, MPI_SUM
	pass#IMPORTIMPORTIMPORT from EMAN2 import Processor
	pass#IMPORTIMPORTIMPORT from EMAN2 import Vec2f
	pass#IMPORTIMPORTIMPORT import os
	pass#IMPORTIMPORTIMPORT import sys


	number_of_proc = mpi.mpi_comm_size(mpi.MPI_COMM_WORLD)
	myid = mpi.mpi_comm_rank(mpi.MPI_COMM_WORLD)

	if CTF:
		pass#IMPORTIMPORTIMPORT from filter import filt_ctf

	main_node = 0
	
	if os.path.exists(outdir):  sparx_global_def.ERROR('Output directory exists, please change the name and restart the program', "local_ali3d_MPI ", 1,myid)
	mpi.mpi_barrier(mpi.MPI_COMM_WORLD)

	if myid == main_node:
		os.mkdir(outdir)
		pass#IMPORTIMPORTIMPORT import global_def
		sparx_global_def.LOGFILE =  os.path.join(outdir, sparx_global_def.LOGFILE)
		sparx_utilities.print_begin_msg("local_ali3d_MPI")
		if CTF:
			ima = EMAN2_cppwrap.EMData()
			ima.read_image(stack, 0)
			ctf_applied = ima.get_attr_default("ctf_applied", 0)
			del ima
			if ctf_applied == 1:  sparx_global_def.ERROR("local_ali3d does not work for CTF-applied data", "local_ali3d_MPI", 1,myid)
	mpi.mpi_barrier(mpi.MPI_COMM_WORLD)

	if debug:
		pass#IMPORTIMPORTIMPORT from time import sleep
		while not os.path.exists(outdir):
			print("Node ",myid,"  waiting...")
			time.sleep(5)

		finfo = open(os.path.join(outdir, "progress%04d"%myid), 'w')
	else:
		finfo = None

	last_ring   = int(ou)
	max_iter    = int(maxit)
	center      = int(center)

	if myid == main_node:
		if(sparx_utilities.file_type(stack) == "bdb"):
			pass#IMPORTIMPORTIMPORT from EMAN2db import db_open_dict
			dummy = EMAN2db.db_open_dict(stack, True)
		# horatio active_refactoring Jy51i1EwmLD4tWZ9_00000_1
		# active = EMUtil.get_all_attributes(stack, 'active')
		# list_of_particles = []
		# for im in xrange(len(active)):
		# 	if(active[im]):  list_of_particles.append(im)
		# del active
		# nima = len(list_of_particles)
			
		nima = EMAN2_cppwrap.EMUtil.get_image_count(stack)
		list_of_particles = list(range(nima))
			
		ima     = EMAN2_cppwrap.EMData()
		ima.read_image(stack, 0)
		nx      = ima.get_xsize()
		del ima
	else:
		nima = 0
		nx = 0
	total_nima = sparx_utilities.bcast_number_to_all(nima, source_node = main_node)
	nx = sparx_utilities.bcast_number_to_all(nx, source_node = main_node)

	if myid != main_node:
		list_of_particles = [-1]*total_nima
	list_of_particles = sparx_utilities.bcast_list_to_all(list_of_particles, myid,  source_node = main_node)

	image_start, image_end = MPI_start_end(total_nima, number_of_proc, myid)
	# create a list of images for each node
	list_of_particles = list_of_particles[image_start: image_end]
	nima = len(list_of_particles)

	if last_ring < 0:	last_ring = int(nx/2) - 2

	if chunk <= 0.0:  chunk = 1.0
	n_of_chunks = int(1.0/chunk)

	if myid == main_node:
		pass#IMPORTIMPORTIMPORT import user_functions
		user_func = sparx_user_functions.factory[user_func_name]

		sparx_utilities.print_msg("Input stack                 : %s\n"%(stack))
		sparx_utilities.print_msg("Output directory            : %s\n"%(outdir))
		sparx_utilities.print_msg("Maskfile                    : %s\n"%(maskfile))
		sparx_utilities.print_msg("Outer radius                : %i\n"%(last_ring))
		sparx_utilities.print_msg("Angular search range        : %s\n"%(delta))
		sparx_utilities.print_msg("Shift search range          : %f\n"%(ts))
		sparx_utilities.print_msg("Maximum iteration           : %i\n"%(max_iter))
		sparx_utilities.print_msg("Center type                 : %i\n"%(center))
		sparx_utilities.print_msg("CTF correction              : %s\n"%(CTF))
		sparx_utilities.print_msg("Signal-to-Noise Ratio       : %f\n"%(snr))
		sparx_utilities.print_msg("Symmetry group              : %s\n"%(sym))
		sparx_utilities.print_msg("Chunk size                  : %f\n"%(chunk))
		sparx_utilities.print_msg("User function               : %s\n\n"%(user_func_name))

	if maskfile:
		pass#IMPORTIMPORTIMPORT import  types
		if type(maskfile) is bytes:  mask3D = sparx_utilities.get_image(maskfile)
		else:                                  mask3D = maskfile
	else:
		mask3D = sparx_utilities.model_circle(last_ring, nx, nx, nx)
	mask2D = sparx_utilities.model_circle(last_ring, nx, nx)

	if debug:
		finfo.write( "image_start, image_end: %d %d\n" %(image_start, image_end) )
		finfo.flush()

	if debug:
		finfo.write("  chunk = "+str(chunk)+"   ")
		finfo.write("\n")
		finfo.flush()
		finfo.write("  Number of chunks = "+str(n_of_chunks)+"   ")
		finfo.write("\n")
		finfo.flush()

	dataim = EMAN2_cppwrap.EMData.read_images(stack, list_of_particles)
	for im in range(len(dataim)):
		dataim[im].set_attr('ID', list_of_particles[im])
	del list_of_particles

	if debug:
		finfo.write("  First image on this processor: "+str(image_start)+"   ")
		finfo.write("  Last  image on this processor: "+str(image_end)+"   ")
		finfo.write("\n")
		finfo.flush()

	if myid == main_node:
		# initialize data for the reference preparation function
		ref_data = [ mask3D, max(center,0), None, None, None ]
		# for method -1, switch off centering in user function
		ref_data.append( None )

	pass#IMPORTIMPORTIMPORT from time import time
		
	M = nx
	npad = 2
	N = M*npad
	K = 6
	alpha = 1.75
	r = M/2
	v = K/2.0/N
	params = {"filter_type": EMAN2_cppwrap.Processor.fourier_filter_types.KAISER_SINH_INVERSE, "alpha":alpha, "K":K, "r":r, "v":v, "N":N}


	disps = []
	recvcount = []
	for im in range(number_of_proc):
		if( im == main_node ):  disps.append(0)
		else:                  disps.append(disps[im-1] + recvcount[im-1])
		ib, ie = MPI_start_end(total_nima, number_of_proc, im)
		recvcount.append( ie - ib )

	#  this is needed for gathering of pixel errors
	pixer = [0.0]*nima
	data = [None]*7
	data[3] = mask2D
	cs = [0.0]*3

	for iteration in range(maxit+1):
		if myid == main_node:
			start_time = time.time()
			sparx_utilities.print_msg("ITERATION #%3d\n"%(iteration+1))
		if debug:
			finfo.write("  iteration = "+str(iteration)+"   ")
			finfo.write("\n")
			finfo.flush()
		for ic in range(n_of_chunks):
			if(center == -1 and sym[0] == 'c'):
				if debug:
					finfo.write("  begin centering \n")
					finfo.flush()
				cs[0], cs[1], cs[2], dummy, dummy = sparx_utilities.estimate_3D_center_MPI(dataim, total_nima, myid, number_of_proc, main_node)
				cs = mpi.mpi_bcast(cs, 3, mpi.MPI_FLOAT, main_node, mpi_comm)
				cs = [-float(cs[0]), -float(cs[1]), -float(cs[2])]
				if int(sym[1]) > 1:
					cs[0] = cs[1] = 0.0
				sparx_utilities.rotate_3D_shift(dataim, cs)
				if myid == main_node:
					msg = "Average center x = %10.3f        Center y = %10.3f        Center z = %10.3f\n"%(cs[0], cs[1], cs[2])
					sparx_utilities.print_msg(msg)
					sparx_utilities.print_msg("Time to center = %d\n"%(time.time()-start_time))
					start_time = time.time()
			# compute updated 3D before each chunk
 	    		# resolution
			if debug:
				finfo.write("  begin reconstruction = "+str(image_start))
				finfo.write("\n")
				finfo.flush()

			if CTF: vol, fscc = sparx_reconstruction.rec3D_MPI(dataim, snr, sym, mask3D, os.path.join(outdir, "resolution%03d_%03d"%(iteration, ic)), myid, main_node, npad = npad)
			else:   vol, fscc = sparx_reconstruction.rec3D_MPI_noCTF(dataim, sym, mask3D, os.path.join(outdir, "resolution%03d_%03d"%(iteration, ic)), myid, main_node, npad = npad)

			if myid == main_node:
				sparx_utilities.drop_image(vol, os.path.join(outdir, "vol%03d_%03d.hdf"%(iteration, ic) ))
				sparx_utilities.print_msg("3D reconstruction time = %d\n"%(time.time()-start_time))
				start_time = time.time()
			if debug:
				finfo.write("  done reconstruction = "+str(image_start))
				finfo.write("\n")
				finfo.flush()

			if fourvar:
			#  Compute Fourier variance
				varf = sparx_statistics.varf3d_MPI(dataim, ssnr_text_file = os.path.join(outdir, "ssnr%03d_%03d"%(iteration, ic)), mask2D = None, reference_structure = vol, ou = ou, rw = 1.0, npad = 1, CTF = CTF, sign = 1, sym =sym, myid = myid)
				if myid == main_node:
					varf = 1.0/varf
					sparx_utilities.print_msg("Time to calculate 3D Fourier variance= %d\n"%(time.time()-start_time))
					start_time = time.time()

			else:  varf = None
			if myid == main_node:
				ref_data[2] = vol
				ref_data[3] = fscc
				ref_data[4] = varf
				#  call user-supplied function to prepare reference image, i.e., filter it
				# When center = -1, which is by default, we use the average center method
				ref_data[1] = 0
				vol, dummy = user_func(ref_data)
				sparx_utilities.drop_image(vol, os.path.join(outdir, "volf%03d_%03d.hdf"%(iteration, ic)))
			del varf

			# in last iteration return here
			if(iteration == maxit):
				if myid == main_node: sparx_utilities.print_end_msg("local_ali3d_MPI")
				return
			sparx_utilities.bcast_EMData_to_all(vol, myid, main_node)

			if not CTF:
				data[0], data[1] = sparx_projection.prep_vol(vol)

			image_start_in_chunk = image_start + ic*nima/n_of_chunks
			image_end_in_chunk   = image_start + (ic+1)*nima/n_of_chunks
			if debug:
				finfo.write("Chunk "+str(ic)+"   Number of images in this chunk: "+str(image_end_in_chunk-image_start_in_chunk)+"\n")
				finfo.write("First image in this chunk: "+str(image_start_in_chunk)+"   Last image in this chunk: "+str(image_end_in_chunk-1)+"\n")
				finfo.flush()
			if CTF:  previous_defocus = -1.0
			for imn in range(image_start_in_chunk, image_end_in_chunk):
				if CTF:
					ctf_params = dataim[imn-image_start].get_attr( "ctf" )
					if ctf_params.defocus != previous_defocus:
						previous_defocus = ctf_params.defocus
						data[0], data[1] = sparx_projection.prep_vol(sparx_filter.filt_ctf(vol, ctf_params, dopad = False))

				data[2] = dataim[imn-image_start]
				if ts > 0.0:
					refi = dataim[imn-image_start].FourInterpol(nx*2, nx*2, 1, True)
					data[4] = EMAN2_cppwrap.Processor.EMFourierFilter(refi, params)

				#phi, theta, psi, tx, ty = get_params_proj(dataim[imn-image_start])
				t1 = dataim[imn-image_start].get_attr("xform.projection")
				dp = t1.get_params("spider")
				atparams = [dp["phi"], dp["theta"], dp["psi"]]
				data[5] = [dp["tx"], dp["ty"]]
				if debug:
					# we have to distiguish between no shift situation, which is done through ccc, and shift, which is done using gridding in 2D
					if(ts == 0.0):  data[6] = 0.0
					else:           data[6] = -1.0#ts#-1.0
					initial, dummy = sparx_alignment.eqproj_cascaded_ccc(atparams, data)  # this is if we need initial discrepancy
					finfo.write("Image "+str(imn)+"\n")
					finfo.write('Old  %6.1f  %6.1f  %6.1f   %5.2f  %5.2f  %11.4e\n'%(atparams[0],atparams[1],atparams[2], -dummy[0], -dummy[1], initial))
				# change signs of shifts for projections
				data[6] = ts
				#from random import random
				#data[5] = [(random()-0.5)*2,(random()-0.5)*2]  #  HERE !!!!!!!!!!!

				weight_phi = max(delta, delta*abs((atparams[1]-90.0)/180.0*numpy.pi))
				optm_params = sparx_utilities.amoeba_multi_level(atparams, [weight_phi, delta, weight_phi], sparx_alignment.eqproj_cascaded_ccc, 1.0, 1.e-2, 500, data)
				optm_params[0].append(optm_params[3][0])
				optm_params[0].append(optm_params[3][1])
				optm_params[0][3] *= -1
				optm_params[0][4] *= -1

				if debug:
					finfo.write('New  %6.1f  %6.1f  %6.1f   %5.2f  %5.2f  %11.4e  %4d\n'%(optm_params[0][0], optm_params[0][1], optm_params[0][2], optm_params[0][3], optm_params[0][4], optm_params[1], optm_params[2]))
					finfo.flush()

				#from sys import exit
				#exit()
				t2 = EMAN2_cppwrap.Transform({"type":"spider","phi":optm_params[0][0],"theta":optm_params[0][1],"psi":optm_params[0][2]})
				t2.set_trans(EMAN2_cppwrap.Vec2f(-optm_params[0][3], -optm_params[0][4]))
				dataim[imn-image_start].set_attr("xform.projection", t2)
				pass#IMPORTIMPORTIMPORT from pixel_error import max_3D_pixel_error
				pixer[imn-image_start] = sparx_pixel_error.max_3D_pixel_error(t1, t2, last_ring)
				#set_params_proj(dataim[imn-image_start], optm_params[0])
				if( myid == main_node ):
					sparx_utilities.print_msg( "Time to process %6d particles : %d\n" % (image_end_in_chunk-image_start_in_chunk, time.time()-start_time) )
					start_time = time.time()
			# release memory of volft
			data[0] = None

			# write out headers, under MPI writing has to be done sequentially
			mpi.mpi_barrier(mpi.MPI_COMM_WORLD)
			par_str = ['xform.projection', 'ID']
			if myid == main_node:
				pass#IMPORTIMPORTIMPORT from utilities import file_type
				if(sparx_utilities.file_type(stack) == "bdb"):
					pass#IMPORTIMPORTIMPORT from utilities import recv_attr_dict_bdb
					sparx_utilities.recv_attr_dict_bdb(main_node, stack, dataim, par_str, image_start, image_end, number_of_proc)
				else:
					pass#IMPORTIMPORTIMPORT from utilities import recv_attr_dict
					sparx_utilities.recv_attr_dict(main_node, stack, dataim, par_str, image_start, image_end, number_of_proc)
			else:	        sparx_utilities.send_attr_dict(main_node, dataim, par_str, image_start, image_end)
			if myid == main_node:
				sparx_utilities.print_msg("Time to write header information= %d\n"%(time.time()-start_time))
				start_time = time.time()

		#output pixel errors after all headers were processed
		pass#IMPORTIMPORTIMPORT from mpi import mpi_gatherv
		recvbuf = mpi.mpi_gatherv(pixer, nima, mpi.MPI_FLOAT, recvcount, disps, mpi.MPI_FLOAT, main_node, mpi.MPI_COMM_WORLD)
		mpi.mpi_barrier(mpi.MPI_COMM_WORLD)
		terminate = 0
		if(myid == main_node):
			recvbuf = list(map(float, recvbuf))
			pass#IMPORTIMPORTIMPORT from statistics import hist_list
			lhist = 20
			region, histo = sparx_statistics.hist_list(recvbuf, lhist)
			if(region[0] < 0.0):  region[0] = 0.0
			msg = "      Histogram of pixel errors\n      ERROR       number of particles\n"
			sparx_utilities.print_msg(msg)
			for lhx in range(lhist):
				msg = " %10.3f     %7d\n"%(region[lhx], histo[lhx])
				sparx_utilities.print_msg(msg)
			# Terminate if 95% within 1 pixel error
			im = 0
			for lhx in range(lhist):
				if(region[lhx] > 1.0): break
				im += histo[lhx]
			if(im/float(total_nima) > 0.95):  terminate = 1
			del region, histo
		del recvbuf
		terminate = mpi.mpi_bcast(terminate, 1, mpi.MPI_INT, 0, mpi.MPI_COMM_WORLD)
		terminate = int(terminate[0])


def local_ali3d_MPI_scipy_minimization(stack, outdir, maskfile, ou = -1,  delta = 2, ts=0.25, center = -1, maxit = 10, 
                CTF = False, snr = 1.0, sym = "c1", chunk = -1.0, user_func_name = "ref_ali3d",
		    fourvar = True, npad = 4, debug = False):
	"""
		
	"""
	pass#IMPORTIMPORTIMPORT from scipy.optimize import minimize
	pass#IMPORTIMPORTIMPORT from fundamentals import fft
	pass#IMPORTIMPORTIMPORT from alignment        import eqproj_cascaded_ccc, objective_function_just_ccc_has_maximum, objective_function_just_ccc_has_minimum, objective_function_just_ccc_has_minimum_reduced
	pass#IMPORTIMPORTIMPORT from filter           import filt_ctf
	pass#IMPORTIMPORTIMPORT from projection       import prep_vol
	pass#IMPORTIMPORTIMPORT from utilities        import bcast_number_to_all, model_circle, get_params_proj, set_params_proj
	pass#IMPORTIMPORTIMPORT from utilities        import bcast_EMData_to_all, bcast_list_to_all, send_attr_dict
	pass#IMPORTIMPORTIMPORT from utilities        import get_image, drop_image, file_type
	pass#IMPORTIMPORTIMPORT from utilities        import amoeba_multi_level, rotate_3D_shift, estimate_3D_center_MPI
	pass#IMPORTIMPORTIMPORT from utilities        import print_begin_msg, print_end_msg, print_msg
	pass#IMPORTIMPORTIMPORT from reconstruction   import rec3D_MPI, rec3D_MPI_noCTF
	pass#IMPORTIMPORTIMPORT from statistics       import varf3d_MPI
	pass#IMPORTIMPORTIMPORT from math             import pi
	pass#IMPORTIMPORTIMPORT from mpi              import mpi_bcast, mpi_comm_size, mpi_comm_rank, MPI_FLOAT, MPI_COMM_WORLD, mpi_barrier
	pass#IMPORTIMPORTIMPORT from mpi              import mpi_reduce, MPI_INT, MPI_SUM
	pass#IMPORTIMPORTIMPORT from EMAN2 import Processor
	pass#IMPORTIMPORTIMPORT from EMAN2 import Vec2f
	pass#IMPORTIMPORTIMPORT import os
	pass#IMPORTIMPORTIMPORT import sys


	number_of_proc = mpi.mpi_comm_size(mpi.MPI_COMM_WORLD)
	myid = mpi.mpi_comm_rank(mpi.MPI_COMM_WORLD)

	if CTF:
		pass#IMPORTIMPORTIMPORT from filter import filt_ctf

	main_node = 0
	
	if os.path.exists(outdir):  sparx_global_def.ERROR('Output directory exists, please change the name and restart the program', "local_ali3d_MPI ", 1,myid)
	mpi.mpi_barrier(mpi.MPI_COMM_WORLD)

	if myid == main_node:
		os.mkdir(outdir)
		pass#IMPORTIMPORTIMPORT import global_def
		sparx_global_def.LOGFILE =  os.path.join(outdir, sparx_global_def.LOGFILE)
		sparx_utilities.print_begin_msg("local_ali3d_MPI")
		if CTF:
			ima = EMAN2_cppwrap.EMData()
			ima.read_image(stack, 0)
			ctf_applied = ima.get_attr_default("ctf_applied", 0)
			del ima
			if ctf_applied == 1:  sparx_global_def.ERROR("local_ali3d does not work for CTF-applied data", "local_ali3d_MPI", 1,myid)
	mpi.mpi_barrier(mpi.MPI_COMM_WORLD)

	########## I52DqDNCvGHVn4uZ
	# debug = myid == 7
	# debug = myid in [7,8,9,10]
	# debug = myid in list(range(number_of_proc))

	if debug:
		pass#IMPORTIMPORTIMPORT from time import sleep
		while not os.path.exists(outdir):
			print("Node ",myid,"  waiting...")
			time.sleep(5)

		finfo = open(os.path.join(outdir, "progress%04d"%myid), 'w')
	else:
		finfo = None

	last_ring   = int(ou)
	max_iter    = int(maxit)
	center      = int(center)

	if myid == main_node:
		if(sparx_utilities.file_type(stack) == "bdb"):
			pass#IMPORTIMPORTIMPORT from EMAN2db import db_open_dict
			dummy = EMAN2db.db_open_dict(stack, True)
			
		nima = EMAN2_cppwrap.EMUtil.get_image_count(stack)
		list_of_particles = list(range(nima))
			
		ima     = EMAN2_cppwrap.EMData()
		ima.read_image(stack, 0)
		nx      = ima.get_xsize()
		del ima
	else:
		nima = 0
		nx = 0
	total_nima = sparx_utilities.bcast_number_to_all(nima, source_node = main_node)
	nx = sparx_utilities.bcast_number_to_all(nx, source_node = main_node)

	if myid != main_node:
		list_of_particles = [-1]*total_nima
	list_of_particles = sparx_utilities.bcast_list_to_all(list_of_particles, myid,  source_node = main_node)

	image_start, image_end = MPI_start_end(total_nima, number_of_proc, myid)
	# create a list of images for each node
	list_of_particles = list_of_particles[image_start: image_end]
	nima = len(list_of_particles)

	if last_ring < 0:	last_ring = int(nx/2) - 2

	if chunk <= 0.0:  chunk = 1.0
	n_of_chunks = int(1.0/chunk)

	if myid == main_node:
		pass#IMPORTIMPORTIMPORT import user_functions
		user_func = sparx_user_functions.factory[user_func_name]

		sparx_utilities.print_msg("Input stack                 : %s\n"%(stack))
		sparx_utilities.print_msg("Output directory            : %s\n"%(outdir))
		sparx_utilities.print_msg("Maskfile                    : %s\n"%(maskfile))
		sparx_utilities.print_msg("Outer radius                : %i\n"%(last_ring))
		sparx_utilities.print_msg("Angular search range        : %s\n"%(delta))
		sparx_utilities.print_msg("Shift search range          : %f\n"%(ts))
		sparx_utilities.print_msg("Maximum iteration           : %i\n"%(max_iter))
		sparx_utilities.print_msg("Center type                 : %i\n"%(center))
		sparx_utilities.print_msg("CTF correction              : %s\n"%(CTF))
		sparx_utilities.print_msg("Signal-to-Noise Ratio       : %f\n"%(snr))
		sparx_utilities.print_msg("Symmetry group              : %s\n"%(sym))
		sparx_utilities.print_msg("Chunk size                  : %f\n"%(chunk))
		sparx_utilities.print_msg("User function               : %s\n\n"%(user_func_name))

	if maskfile:
		pass#IMPORTIMPORTIMPORT import  types
		if type(maskfile) is bytes:  mask3D = sparx_utilities.get_image(maskfile)
		else:                                  mask3D = maskfile
	else:
		mask3D = sparx_utilities.model_circle(last_ring, nx, nx, nx)
	mask2D = sparx_utilities.model_circle(last_ring, nx, nx)

	if debug:
		finfo.write( "image_start, image_end: %d %d\n" %(image_start, image_end) )
		finfo.flush()

	if debug:
		finfo.write("  chunk = "+str(chunk)+"   ")
		finfo.write("\n")
		finfo.flush()
		finfo.write("  Number of chunks = "+str(n_of_chunks)+"   ")
		finfo.write("\n")
		finfo.flush()

	dataim = EMAN2_cppwrap.EMData.read_images(stack, list_of_particles)
	for im in range(len(dataim)):
		dataim[im].set_attr('ID', list_of_particles[im])
		if CTF:
			ctf_params = dataim[im].get_attr("ctf")
			st = EMAN2_cppwrap.Util.infomask(dataim[im], mask2D, False)
			dataim[im] -= st[0]
			dataim[im] = sparx_filter.filt_ctf(dataim[im], ctf_params)
			dataim[im].set_attr('ctf_applied', 1)
	del list_of_particles

	if debug:
		finfo.write("  First image on this processor: "+str(image_start)+"   ")
		finfo.write("  Last  image on this processor: "+str(image_end)+"   ")
		finfo.write("\n")
		finfo.flush()

	if myid == main_node:
		# initialize data for the reference preparation function
		ref_data = [ mask3D, max(center,0), None, None, None ]
		# for method -1, switch off centering in user function
		ref_data.append( None )

	pass#IMPORTIMPORTIMPORT from time import time
		
	M = nx
	npad = 2
	N = M*npad
	K = 6
	alpha = 1.75
	r = M/2
	v = K/2.0/N
	params = {"filter_type": EMAN2_cppwrap.Processor.fourier_filter_types.KAISER_SINH_INVERSE, "alpha":alpha, "K":K, "r":r, "v":v, "N":N}


	disps = []
	recvcount = []
	for im in range(number_of_proc):
		if( im == main_node ):  disps.append(0)
		else:                  disps.append(disps[im-1] + recvcount[im-1])
		ib, ie = MPI_start_end(total_nima, number_of_proc, im)
		recvcount.append( ie - ib )

	#  this is needed for gathering of pixel errors
	pixer = [0.0]*nima
	data = [None]*7
	data[3] = mask2D
	cs = [0.0]*3

	for iteration in range(maxit+1):
		if myid == main_node:
			start_time = time.time()
			sparx_utilities.print_msg("ITERATION #%3d\n"%(iteration+1))
		if debug:
			finfo.write("  iteration = "+str(iteration)+"   ")
			finfo.write("\n")
			finfo.flush()
		for ic in range(n_of_chunks):
			if(center == -1 and sym[0] == 'c'):
				if debug:
					finfo.write("  begin centering \n")
					finfo.flush()
				cs[0], cs[1], cs[2], dummy, dummy = sparx_utilities.estimate_3D_center_MPI(dataim, total_nima, myid, number_of_proc, main_node)
				cs = mpi.mpi_bcast(cs, 3, mpi.MPI_FLOAT, main_node, mpi.MPI_COMM_WORLD)
				cs = [-float(cs[0]), -float(cs[1]), -float(cs[2])]
				if int(sym[1]) > 1:
					cs[0] = cs[1] = 0.0
				sparx_utilities.rotate_3D_shift(dataim, cs)
				if myid == main_node:
					msg = "Average center x = %10.3f        Center y = %10.3f        Center z = %10.3f\n"%(cs[0], cs[1], cs[2])
					sparx_utilities.print_msg(msg)
					sparx_utilities.print_msg("Time to center = %d\n"%(time.time()-start_time))
					start_time = time.time()
			# compute updated 3D before each chunk
				# resolution
			if debug:
				finfo.write("  begin reconstruction = "+str(image_start))
				finfo.write("\n")
				finfo.flush()

			if CTF: vol, fscc = sparx_reconstruction.rec3D_MPI(dataim, snr, sym, mask3D, os.path.join(outdir, "resolution%03d_%03d"%(iteration, ic)), myid, main_node, npad = npad)
			else:   vol, fscc = sparx_reconstruction.rec3D_MPI_noCTF(dataim, sym, mask3D, os.path.join(outdir, "resolution%03d_%03d"%(iteration, ic)), myid, main_node, npad = npad)

			if myid == main_node:
				sparx_utilities.drop_image(vol, os.path.join(outdir, "vol%03d_%03d.hdf"%(iteration, ic) ))
				sparx_utilities.print_msg("3D reconstruction time = %d\n"%(time.time()-start_time))
				start_time = time.time()
			if debug:
				finfo.write("  done reconstruction = "+str(image_start))
				finfo.write("\n")
				finfo.flush()

			if fourvar:
			#  Compute Fourier variance
				varf = sparx_statistics.varf3d_MPI(dataim, ssnr_text_file = os.path.join(outdir, "ssnr%03d_%03d"%(iteration, ic)), mask2D = None, reference_structure = vol, ou = ou, rw = 1.0, npad = 1, CTF = CTF, sign = 1, sym =sym, myid = myid)
				if myid == main_node:
					varf = 1.0/varf
					sparx_utilities.print_msg("Time to calculate 3D Fourier variance= %d\n"%(time.time()-start_time))
					start_time = time.time()

			else:  varf = None
			if myid == main_node:
				ref_data[2] = vol
				ref_data[3] = fscc
				ref_data[4] = varf
				#  call user-supplied function to prepare reference image, i.e., filter it
				# When center = -1, which is by default, we use the average center method
				ref_data[1] = 0
				vol, dummy = user_func(ref_data)
				sparx_utilities.drop_image(vol, os.path.join(outdir, "volf%03d_%03d.hdf"%(iteration, ic)))
			del varf

			# in last iteration return here
			if(iteration == maxit):
				if myid == main_node: sparx_utilities.print_end_msg("local_ali3d_MPI")
				return
			sparx_utilities.bcast_EMData_to_all(vol, myid, main_node)

			data[0] = sparx_projection.prep_vol(vol, 1, 1)

			image_start_in_chunk = image_start + ic*nima/n_of_chunks
			image_end_in_chunk   = image_start + (ic+1)*nima/n_of_chunks
			if debug:
				finfo.write("Chunk "+str(ic)+"   Number of images in this chunk: "+str(image_end_in_chunk-image_start_in_chunk)+"\n")
				finfo.write("First image in this chunk: "+str(image_start_in_chunk)+"   Last image in this chunk: "+str(image_end_in_chunk-1)+"\n")
				finfo.flush()

			for imn in range(image_start_in_chunk, image_end_in_chunk):
				data[2] = sparx_fundamentals.fft(dataim[imn-image_start])
				data[2].set_attr("is_complex",0)
	
				#phi, theta, psi, tx, ty = get_params_proj(dataim[imn-image_start])
				t1 = dataim[imn-image_start].get_attr("xform.projection")
				dp = t1.get_params("spider")
				atparams = [dp["phi"], dp["theta"], dp["psi"]]
				data[5] = [dp["tx"], dp["ty"]]
				if debug:
					# we have to distiguish between no shift situation, which is done through ccc, and shift, which is done using gridding in 2D
					if(ts == 0.0):  data[6] = 0.0
					# else:           data[6] = -1.0#ts#-1.0
					else:           data[6] = ts
	
					finfo.write('Old  %6.2f  %6.2f  %6.2f   %5.2f  %5.2f\n'%(atparams[0],atparams[1],atparams[2], data[5][0], data[5][1]))
	
				# change signs of shifts for projections
				data[6] = ts
				#from random import random
				#data[5] = [(random()-0.5)*2,(random()-0.5)*2]  #  HERE !!!!!!!!!!!
	
				# region cobyla
				atparams_for_minimize = atparams + data[5]
				
				total_number_of_function_calls = 0
				res = scipy.optimize.minimize(sparx_alignment.objective_function_just_ccc_has_minimum, atparams_for_minimize, (data,), method='COBYLA')
				total_number_of_function_calls += res.nfev
				result = res.x.tolist()
				data[5] = result[3:5]
				res = scipy.optimize.minimize(sparx_alignment.objective_function_just_ccc_has_minimum_reduced, result[0:3], (data,), method='COBYLA')
				total_number_of_function_calls += res.nfev
				result = res.x.tolist() + data[5]
				res = scipy.optimize.minimize(sparx_alignment.objective_function_just_ccc_has_minimum, result, (data,), method='COBYLA')
				total_number_of_function_calls += res.nfev

				optm_params = [[], 0, 0, 0]
				optm_params[0] = res.x.tolist()
				optm_params[0][3] *= -1
				optm_params[0][4] *= -1
				
				optm_params[1] = res.fun
				# optm_params[2] = res.nfev
				optm_params[2] = total_number_of_function_calls
				# endregion
	
	

	
				if debug:
					# finfo.write('New  %6.1f  %6.1f  %6.1f   %5.2f  %5.2f  %11.4e  %4d, time_dur=%4d  %6.1f\n'%(optm_params[0][0], optm_params[0][1], optm_params[0][2], optm_params[0][3], optm_params[0][4], optm_params[1], optm_params[2], time_duration_user, optm_params[3]))
					finfo.write('New  %6.2f  %6.2f  %6.2f   %5.2f  %5.2f  %11.4e  %4d\n'%(optm_params[0][0], optm_params[0][1], optm_params[0][2], optm_params[0][3], optm_params[0][4], optm_params[1], optm_params[2]))
					finfo.flush()
				#from sys import exit
				#exit()
				t2 = EMAN2_cppwrap.Transform({"type":"spider","phi":optm_params[0][0],"theta":optm_params[0][1],"psi":optm_params[0][2]})
				t2.set_trans(EMAN2_cppwrap.Vec2f(-optm_params[0][3], -optm_params[0][4]))
				dataim[imn-image_start].set_attr("xform.projection", t2)
				pass#IMPORTIMPORTIMPORT from pixel_error import max_3D_pixel_error
				pixer[imn-image_start] = sparx_pixel_error.max_3D_pixel_error(t1, t2, last_ring)
				#set_params_proj(dataim[imn-image_start], optm_params[0])
				if( myid == main_node ):
					sparx_utilities.print_msg( "Time to process %6d particles : %d\n" % (image_end_in_chunk-image_start_in_chunk, time.time()-start_time) )
					start_time = time.time()
			# release memory of volft
			data[0] = None

			# write out headers, under MPI writing has to be done sequentially
			mpi.mpi_barrier(mpi.MPI_COMM_WORLD)
			par_str = ['xform.projection', 'ID']
			if myid == main_node:
				pass#IMPORTIMPORTIMPORT from utilities import file_type
				if(sparx_utilities.file_type(stack) == "bdb"):
					pass#IMPORTIMPORTIMPORT from utilities import recv_attr_dict_bdb
					sparx_utilities.recv_attr_dict_bdb(main_node, stack, dataim, par_str, image_start, image_end, number_of_proc)
				else:
					pass#IMPORTIMPORTIMPORT from utilities import recv_attr_dict
					sparx_utilities.recv_attr_dict(main_node, stack, dataim, par_str, image_start, image_end, number_of_proc)
			else:	        sparx_utilities.send_attr_dict(main_node, dataim, par_str, image_start, image_end)
			if myid == main_node:
				sparx_utilities.print_msg("Time to write header information= %d\n"%(time.time()-start_time))
				start_time = time.time()

		#output pixel errors after all headers were processed
		pass#IMPORTIMPORTIMPORT from mpi import mpi_gatherv
		recvbuf = mpi.mpi_gatherv(pixer, nima, mpi.MPI_FLOAT, recvcount, disps, mpi.MPI_FLOAT, main_node, mpi.MPI_COMM_WORLD)
		mpi.mpi_barrier(mpi.MPI_COMM_WORLD)
		terminate = 0
		if(myid == main_node):
			recvbuf = list(map(float, recvbuf))
			pass#IMPORTIMPORTIMPORT from statistics import hist_list
			lhist = 20
			region, histo = sparx_statistics.hist_list(recvbuf, lhist)
			if(region[0] < 0.0):  region[0] = 0.0
			msg = "      Histogram of pixel errors\n      ERROR       number of particles\n"
			sparx_utilities.print_msg(msg)
			for lhx in range(lhist):
				msg = " %10.3f     %7d\n"%(region[lhx], histo[lhx])
				sparx_utilities.print_msg(msg)
			# Terminate if 95% within 1 pixel error
			im = 0
			for lhx in range(lhist):
				if(region[lhx] > 1.0): break
				im += histo[lhx]
			if(im/float(total_nima) > 0.95):  terminate = 1
			del region, histo
		del recvbuf
		terminate = mpi.mpi_bcast(terminate, 1, mpi.MPI_INT, 0, mpi.MPI_COMM_WORLD)
		terminate = int(terminate[0])


def local_ali3d_base_MPI(stack, templatevol, ali3d_options, shrinkage = 1.0,
		    	mpi_comm = None, log= None, chunk = -1.0, saturatecrit = 0.95, pixercutoff = 1.0, debug = False ):
	"""
		
	"""
	pass#IMPORTIMPORTIMPORT from alignment        import eqproj_cascaded_ccc
	pass#IMPORTIMPORTIMPORT from filter           import filt_ctf
	pass#IMPORTIMPORTIMPORT from projection       import prep_vol
	pass#IMPORTIMPORTIMPORT from fundamentals     import resample
	pass#IMPORTIMPORTIMPORT from utilities        import bcast_number_to_all, model_circle, get_params_proj, set_params_proj
	pass#IMPORTIMPORTIMPORT from utilities        import bcast_EMData_to_all, bcast_list_to_all, send_attr_dict, wrap_mpi_bcast, wrap_mpi_gatherv
	pass#IMPORTIMPORTIMPORT from utilities        import get_image, drop_image, file_type, get_im, get_input_from_string, model_blank
	pass#IMPORTIMPORTIMPORT from utilities        import amoeba_multi_level, rotate_3D_shift, estimate_3D_center_MPI
	pass#IMPORTIMPORTIMPORT from utilities        import print_begin_msg, print_end_msg, print_msg
	pass#IMPORTIMPORTIMPORT from multi_shc        import do_volume
	pass#IMPORTIMPORTIMPORT from statistics       import varf3d_MPI
	pass#IMPORTIMPORTIMPORT from math             import pi
	pass#IMPORTIMPORTIMPORT from mpi              import mpi_bcast, mpi_comm_size, mpi_comm_rank, MPI_FLOAT, MPI_COMM_WORLD, mpi_barrier
	pass#IMPORTIMPORTIMPORT from mpi              import mpi_reduce, MPI_INT, MPI_SUM
	pass#IMPORTIMPORTIMPORT from EMAN2 import Processor
	pass#IMPORTIMPORTIMPORT from EMAN2 import Vec2f, Transform
	pass#IMPORTIMPORTIMPORT import os
	pass#IMPORTIMPORTIMPORT import sys
	pass#IMPORTIMPORTIMPORT import types

	maxit  = ali3d_options.maxit
	ou     = ali3d_options.ou
	ts     = sparx_utilities.get_input_from_string(ali3d_options.ts)[0]
	delta  = sparx_utilities.get_input_from_string(ali3d_options.delta)[0]
	sym    = ali3d_options.sym
	sym    = sym[0].lower() + sym[1:]
	center = ali3d_options.center
	CTF    = ali3d_options.CTF
	fourvar = False



	if log == None:
		pass#IMPORTIMPORTIMPORT from logger import Logger
		log = sparx_logger.Logger()


	if mpi_comm == None:
		mpi_comm = mpi_comm

	number_of_proc = mpi.mpi_comm_size(mpi_comm)
	myid           = mpi.mpi_comm_rank(mpi_comm)
	main_node = 0

	if myid == main_node:
		log.add("Start local_ali3d_base")

	#if os.path.exists(outdir):  ERROR('Output directory exists, please change the name and restart the program', "local_ali3d_MPI ", 1,myid)
	#mpi_barrier(mpi_comm)

	"""Multiline Comment26"""
	if debug:
		pass#IMPORTIMPORTIMPORT from time import sleep
		while not os.path.exists(outdir):
			print("Node ",myid,"  waiting...")
			time.sleep(5)

		finfo = open(os.path.join(outdir, "progress%04d"%myid), 'w')
	else:
		finfo = None

	last_ring   = int(ou)
	center      = int(center)

	if( type(stack) is bytes ):
		if myid == main_node:
			if(sparx_utilities.file_type(stack) == "bdb"):
				pass#IMPORTIMPORTIMPORT from EMAN2db import db_open_dict
				dummy = EMAN2db.db_open_dict(stack, True)
			# horatio active_refactoring Jy51i1EwmLD4tWZ9_00000_1
			# active = EMUtil.get_all_attributes(stack, 'active')
			# list_of_particles = []
			# for im in xrange(len(active)):
			# 	if(active[im]):  list_of_particles.append(im)
			# del active
		
			nima = EMAN2_cppwrap.EMUtil.get_image_count(stack)
			list_of_particles = list(range(nima))
	
			total_nima = len(list_of_particles)
		else:
			list_of_particles = None
			total_nima = 0
		total_nima = sparx_utilities.wrap_mpi_bcast(total_nima, main_node, mpi_comm)
		total_nima = int(total_nima[0])
		list_of_particles = sparx_utilities.wrap_mpi_bcast(list_of_particles, main_node, mpi_comm)
		if myid == main_node:
			particle_ids = [0]*total_nima
			for i in range(total_nima):  particle_ids[i] = list_of_particles[i]
		image_start, image_end = MPI_start_end(total_nima, number_of_proc, myid)
		# create a list of images for each node
		list_of_particles = list_of_particles[image_start: image_end]
		nima = len(list_of_particles)

	else:
		list_of_particles = list(range(len(stack)))
		nima = len(list_of_particles)
		total_nima = len(list_of_particles)
		total_nima = mpi.mpi_reduce(total_nima, 1, mpi.MPI_INT, mpi.MPI_SUM, 0, mpi.MPI_COMM_WORLD)
		total_nima = mpi.mpi_bcast(total_nima, 1, mpi.MPI_INT, 0, mpi.MPI_COMM_WORLD)
		total_nima = int(total_nima[0])
		image_start = 0
		image_and   = nima

	if(myid == main_node):
		if( type(stack) is bytes ):  dataim = sparx_utilities.get_im(stack, list_of_particles[0])
		else:                                   dataim = stack[list_of_particles[0]]
		onx      = dataim.get_xsize()
		if(shrinkage == 1.0):  nx = onx
		else:		
			st = sparx_fundamentals.resample(dataim, shrinkage)
			nx = st.get_xsize()
	else:
		nx = 0
		onx = 0
	nx  = sparx_utilities.bcast_number_to_all(nx, source_node = main_node)
	onx = sparx_utilities.bcast_number_to_all(onx, source_node = main_node)


	if last_ring < 0:	last_ring = int(onx/2) - 2
	mask2D  = sparx_utilities.model_circle(last_ring, onx, onx)
	if(shrinkage < 1.0):
		last_ring  = int(last_ring*shrinkage)
		ali3d_options.ou = last_ring


	dataim = [None]*nima
	for im in range(nima):
		if( type(stack) is bytes ):  dataim[im] = sparx_utilities.get_im(stack, list_of_particles[im])
		else:                                   dataim[im] = stack[list_of_particles[im]]
		dataim[im].set_attr('ID', list_of_particles[im])
		ctf_applied = dataim[im].get_attr_default('ctf_applied', 0)
		if CTF :
			ctf_params = dataim[im].get_attr("ctf")
			if ctf_applied == 0:
				st = EMAN2_cppwrap.Util.infomask(dataim[im], mask2D, False)
				dataim[im] -= st[0]
			else:
				sparx_global_def.ERROR("Projection data cannot be CTF-applied","local_ali3d_base",1,myid)
		if(shrinkage != 1.0):
			phi,theta,psi,sx,sy = sparx_utilities.get_params_proj(dataim[im])
			dataim[im] = sparx_fundamentals.resample(dataim[im], shrinkage)
			sx *= shrinkage
			sy *= shrinkage
			sparx_utilities.set_params_proj(dataim[im], [phi,theta,psi,sx,sy])
			if CTF :
				ctf_params.apix /= shrinkage
				dataim[im].set_attr('ctf', ctf_params)

	mask2D  = sparx_utilities.model_circle(last_ring, nx, nx)


	if chunk <= 0.0:  chunk = 1.0
	n_of_chunks = int(1.0/chunk)

	"""Multiline Comment27"""

	pass#IMPORTIMPORTIMPORT import  types
	if ali3d_options.mask3D:
		if type(ali3d_options.mask3D) is bytes:
			if myid == main_node:
				mask3D = sparx_utilities.get_im(ali3d_options.mask3D)
			else:
				mask3D = sparx_utilities.model_blank(nx, nx, nx)
		else:
			mask3D = ali3d_options.mask3D.copy()
		if myid == main_node:
			i = mask3D.get_xsize()
			if( shrinkage != 1.0 ):
				if( i != nx ):
					mask3D = sparx_fundamentals.resample(mask3D, shrinkage)
		sparx_utilities.bcast_EMData_to_all(mask3D, myid, main_node)
	else:
		mask3D = sparx_utilities.model_circle(last_ring, nx, nx, nx)

	#  Read	template volume if provided
	if templatevol:
		if type(templatevol) is bytes:
			if myid == main_node:
				vol = sparx_utilities.get_im(templatevol)
				i = vol.get_xsize()
				if( shrinkage != 1.0 ):
					if( i != nx ):
						vol = sparx_fundamentals.resample(vol, shrinkage)
			else:
				vol = sparx_utilities.model_blank(nx, nx, nx)
		else:
			if myid == main_node:
				i = templatevol.get_xsize()
				if( shrinkage != 1.0 ):
					if( i != nx ):
						vol = sparx_fundamentals.resample(templatevol, shrinkage)
				else:
					vol = templatevol.copy()
			else:
				vol = sparx_utilities.model_blank(nx, nx, nx)
		sparx_utilities.bcast_EMData_to_all(vol, myid, main_node)
		del templatevol
	else:
		vol = None

	if debug:
		finfo.write( "image_start, image_end: %d %d\n" %(image_start, image_end) )
		finfo.flush()

	if debug:
		finfo.write("  chunk = "+str(chunk)+"   ")
		finfo.write("\n")
		finfo.flush()
		finfo.write("  Number of chunks = "+str(n_of_chunks)+"   ")
		finfo.write("\n")
		finfo.flush()

	del list_of_particles

	if debug:
		finfo.write("  First image on this processor: "+str(image_start)+"   ")
		finfo.write("  Last  image on this processor: "+str(image_end)+"   ")
		finfo.write("\n")
		finfo.flush()

	if myid == main_node:
		# initialize data for the reference preparation function
		ref_data = [ mask3D, max(center,0), None, None, None ]
		# for method -1, switch off centering in user function
		ref_data.append( None )

	pass#IMPORTIMPORTIMPORT from time import time
	if myid == main_node:
		log.add("Dimensions used (nx, onx, last_ring, shrinkage)  %5d    %5d     %5d     %6.3f\n"%(nx, onx, last_ring, shrinkage))
		start_time = time.time()


		
	M = nx
	npad = 2
	N = M*npad
	K = 6
	alpha = 1.75
	r = M/2
	v = K/2.0/N
	params = {"filter_type": EMAN2_cppwrap.Processor.fourier_filter_types.KAISER_SINH_INVERSE, "alpha":alpha, "K":K, "r":r, "v":v, "N":N}

	data = [None]*7
	data[3] = mask2D
	cs = [0.0]*3

	for iteration in range(maxit):
		if myid == main_node:
			start_time = time.time()
			log.add("ITERATION #%3d\n"%(iteration+1))
		if debug:
			finfo.write("  iteration = "+str(iteration)+"   ")
			finfo.write("\n")
			finfo.flush()
		pixer = [0.0]*nima
		for ic in range(n_of_chunks):
			# In the very first step the volume has to be computed if it was not provided by the user
			if( ((iteration > 0) and (ic > 0)) or vol == None):
				if(center == -1 and sym[0] == 'c'):
					if debug:
						finfo.write("  begin centering \n")
						finfo.flush()
					cs[0], cs[1], cs[2], dummy, dummy = sparx_utilities.estimate_3D_center_MPI(dataim, total_nima, myid, number_of_proc, main_node)
					cs = mpi.mpi_bcast(cs, 3, mpi.MPI_FLOAT, main_node, mpi_comm)
					cs = [-float(cs[0]), -float(cs[1]), -float(cs[2])]
					if int(sym[1]) > 1:
						cs[0] = cs[1] = 0.0
					sparx_utilities.rotate_3D_shift(dataim, cs)
					if myid == main_node:
						msg = "Average center x = %10.3f        Center y = %10.3f        Center z = %10.3f\n"%(cs[0], cs[1], cs[2])
						log.add(msg)
						log.add("Time to center = %d\n"%(time.time()-start_time))
						start_time = time.time()
				# compute updated 3D before each chunk
					# resolution
				if debug:
					finfo.write("  begin reconstruction = "+str(image_start))
					finfo.write("\n")
					finfo.flush()

				#  Do the 3D
				vol = sparx_multi_shc.do_volume(dataim, ali3d_options, iteration, mpi_comm)

				if myid == main_node:
					#drop_image(vol, os.path.join(outdir, "vol%03d_%03d.hdf"%(iteration, ic) ))
					log.add("3D reconstruction time = %d"%(time.time()-start_time))
					start_time = time.time()
				if debug:
					finfo.write("  done reconstruction = "+str(image_start))
					finfo.write("\n")
					finfo.flush()

				if fourvar:
				#  Compute Fourier variance
					varf = sparx_statistics.varf3d_MPI(dataim, ssnr_text_file = os.path.join(outdir, "ssnr%03d_%03d"%(iteration, ic)), mask2D = None, reference_structure = vol, ou = ou, rw = 1.0, npad = 1, CTF = CTF, sign = 1, sym =sym, myid = myid)
					if myid == main_node:
						varf = 1.0/varf
						sparx_utilities.print_msg("Time to calculate 3D Fourier variance= %d\n"%(time.time()-start_time))
						start_time = time.time()
				else:  varf = None

			if CTF:
				previous_defocus = -1.0
				#vol = fft(pad(vol, N, N, N))
			else:
				data[0], data[1] = sparx_projection.prep_vol(vol)

			image_start_in_chunk = ic*nima/n_of_chunks
			image_end_in_chunk   = (ic+1)*nima/n_of_chunks
			if debug:
				finfo.write("Chunk "+str(ic)+"   Number of images in this chunk: "+str(image_end_in_chunk-image_start_in_chunk)+"\n")
				finfo.write("First image in this chunk: "+str(image_start_in_chunk)+"   Last image in this chunk: "+str(image_end_in_chunk-1)+"\n")
				finfo.flush()
			for imn in range(image_start_in_chunk, image_end_in_chunk):
				if CTF:
					ctf_params = dataim[imn].get_attr( "ctf" )
					if ctf_params.defocus != previous_defocus:
						previous_defocus = ctf_params.defocus
						data[0], data[1] = sparx_projection.prep_vol(sparx_filter.filt_ctf(vol, ctf_params, dopad = False))

				data[2] = dataim[imn]
				if ts > 0.0:
					refi = dataim[imn].FourInterpol(nx*2, nx*2, 1, True)
					data[4] = EMAN2_cppwrap.Processor.EMFourierFilter(refi, params)

				#phi, theta, psi, tx, ty = get_params_proj(dataim[imn])
				t1 = dataim[imn].get_attr("xform.projection")
				dp = t1.get_params("spider")
				atparams = [dp["phi"], dp["theta"], dp["psi"]]
				data[5]  = [dp["tx"], dp["ty"]]
				if debug:
					# we have to distiguish between no shift situation, which is done through ccc, and shift, which is done using gridding in 2D
					if(ts == 0.0):  data[6] = 0.0
					else:           data[6] = -1.0#ts#-1.0
					initial, dummy = sparx_alignment.eqproj_cascaded_ccc(atparams, data)  # this is if we need initial discrepancy
					finfo.write("Image "+str(imn)+"\n")
					finfo.write('Old  %6.1f  %6.1f  %6.1f   %5.2f  %5.2f  %11.4e\n'%(atparams[0],atparams[1],atparams[2], -dummy[0], -dummy[1], initial))
				# change signs of shifts for projections
				data[6] = ts
				#from random import random
				#data[5] = [(random()-0.5)*2,(random()-0.5)*2]  #  HERE !!!!!!!!!!!

				weight_phi = max(delta, delta*abs((atparams[1]-90.0)/180.0*numpy.pi))
				optm_params = sparx_utilities.amoeba_multi_level(atparams, [weight_phi, delta, weight_phi], sparx_alignment.eqproj_cascaded_ccc, 1.0, 1.e-2, 500, data)
				optm_params[0].append(optm_params[3][0])
				optm_params[0].append(optm_params[3][1])
				optm_params[0][3] *= -1
				optm_params[0][4] *= -1

				if debug:
					finfo.write('New  %6.1f  %6.1f  %6.1f   %5.2f  %5.2f  %11.4e  %4d\n'%(optm_params[0][0], optm_params[0][1], optm_params[0][2], optm_params[0][3], optm_params[0][4], optm_params[1], optm_params[2]))
					finfo.flush()

				#from sys import exit
				#exit()
				t2 = EMAN2_cppwrap.Transform({"type":"spider","phi":optm_params[0][0],"theta":optm_params[0][1],"psi":optm_params[0][2]})
				t2.set_trans(EMAN2_cppwrap.Vec2f(-optm_params[0][3], -optm_params[0][4]))
				dataim[imn].set_attr("xform.projection", t2)
				pass#IMPORTIMPORTIMPORT from pixel_error import max_3D_pixel_error
				pixer[imn] = sparx_pixel_error.max_3D_pixel_error(t1, t2, last_ring)
				#set_params_proj(dataim[imn], optm_params[0])
				#if( myid == main_node and imn%4 == 0):
				#	log.add( "Time to process %6d particles : %d\n" % (imn, time()-start_time) )
				#	start_time = time()
			if( myid == main_node ):
				log.add( "Time to process %6d particles : %d" % (image_end_in_chunk-image_start_in_chunk, time.time()-start_time) )
				start_time = time.time()

			# release memory
			data[0] = None


		#output pixel errors after all headers were processed
		pass#IMPORTIMPORTIMPORT from mpi import mpi_gatherv
		pixer = sparx_utilities.wrap_mpi_gatherv(pixer, main_node, mpi_comm)
		mpi.mpi_barrier(mpi_comm)
		terminate = 0
		if(myid == main_node):
			pixer = list(map(float, pixer))
			pass#IMPORTIMPORTIMPORT from statistics import hist_list
			lhist = 20
			region, histo = sparx_statistics.hist_list(pixer, lhist)
			log.add(" ")
			log.add("=========== Histogram of pixel errors ==============")
			for lhx in range(lhist):
				msg = "          %10.3f     %7d"%(region[lhx], histo[lhx])
				log.add(msg)
			log.add("____________________________________________________\n")


			# Terminate if saturatecrit% within pixercutoff pixel error
			im = 0
			for lhx in range(lhist):
				if(region[lhx] > pixercutoff): break
				im += histo[lhx]
			lhx = im/float(total_nima)
			if( lhx > saturatecrit):
				if( iteration == 1 ):
					log.add("First iteration, will continue even though %4.2f images did not find better orientations"%saturatecrit)
				else:
					terminate = 1
					log.add("..............................................................")
					log.add(">>>>>>>>>>>>>>>   Will terminate as %4.2f images did not find better orientations"%saturatecrit)
			del region, histo
		del pixer
		terminate = mpi.mpi_bcast(terminate, 1, mpi.MPI_INT, 0, mpi_comm)
		terminate = int(terminate[0])
		if terminate:  break


	del vol
	# gather parameters
	params = []
	for im in dataim:
		t = sparx_utilities.get_params_proj(im)
		params.append( [t[0], t[1], t[2], t[3]/shrinkage, t[4]/shrinkage] )
	params = sparx_utilities.wrap_mpi_gatherv(params, main_node, mpi_comm)

	if( myid == main_node ):
		"""Multiline Comment28"""
		log.add("Finish local_ ali3d_base")

	return  params


def autowin(indir,outdir, noisedoc, noisemic, templatefile, deci, CC_method, p_size, sigma, hf_p, n_peak_max, contrast_invert=1, CTF = False, prm = "micrograph", MPI=False):
	""" 
		automated particle picking program
	   	CC_method=1  Applying high pass filter to detect protein particles
		sigma can be adjusted to perform particle picking. 
		CC_method=2  Using fast local normalization method to detect protein particles
		The output files are detected coordinates and windowed particles
	"""
	if MPI:
		autowin_MPI(indir, outdir, noisedoc, noisemic, templatefile, deci, CC_method, p_size, sigma, hf_p, n_peak_max, contract_invert, prm)
		return
	pass#IMPORTIMPORTIMPORT from utilities 		import get_image, model_circle, ce_fit, drop_image,info
	pass#IMPORTIMPORTIMPORT from utilities          import get_arb_params, set_arb_params
	pass#IMPORTIMPORTIMPORT from fundamentals 	import smallprime, window2d, ccf, ramp, fft
	pass#IMPORTIMPORTIMPORT from filter 		import filt_gaussh, filt_tanl
	pass#IMPORTIMPORTIMPORT from string 		import split
	pass#IMPORTIMPORTIMPORT from morphology 	import flcc
	pass#IMPORTIMPORTIMPORT import os
	if os.path.exists(indir)  is False: sparx_global_def.ERROR("micrograph directory does not exsit", "autowin",1)
	else                              : flist=os.listdir(indir)
	if os.path.exists(outdir)         : sparx_global_def.ERROR('Output directory exists, please change the name and restart the program', "autowin", 1) 
	os.mkdir(outdir)
	t       = EMAN2_cppwrap.EMData()
	e_n     = EMAN2_cppwrap.EMData()
	e       = sparx_utilities.get_image(noisemic)
	i_tem   = EMAN2_cppwrap.EMUtil.get_image_count(templatefile)
	e_n.read_image(templatefile, 0)
	nx      = e_n.get_xsize()
	ny      = e_n.get_ysize()
	rad     = int(nx/2)-1
	mask    = sparx_utilities.model_circle(rad, nx, ny)
	f       = open(noisedoc, "r")
	rstring = f.readlines() 
	f.close
	xs      = rstring[0]
	tmp     = string.split(xs)
	x       = tmp[0]
	y       = tmp[1]
	x_noi   = int(x)-p_size/2
	y_noi   = int(y)-p_size/2
	reg     = EMAN2_cppwrap.Region(x_noi,y_noi, p_size, p_size)# Get the reference noise from the input mic and noise coordinates
	e_n     = e.get_clip(reg)
	if CTF : ctf_dicts = ["defocus", "Pixel_size", "voltage", "Cs", "amp_contrast", "B_factor", "sign"]
	for i, v in enumerate(flist):
		(filename, filextension) = os.path.splitext(v)
		out1_file     = "coord_" + filename + ".txt"
		out2_file     = "particle_" + filename + ".hdf"
		print("micrograph being processed:", filename)
		f_coord       = os.path.join(outdir, out1_file)
		file_particle = os.path.join(outdir, out2_file) #  parse output file names 			
		peaks         = []
		file_input    = os.path.join(indir, v)
		img1 = EMAN2_cppwrap.EMData()
		img1.read_image(file_input)
		if CTF : ctf_params    = img1.get_attr( "ctf" )
		img1         *= contrast_invert
		nx            = img1.get_xsize()
		ny            = img1.get_ysize()
		N_ptl         = int(nx*ny/p_size/p_size) # number of possible particles
		if(N_ptl >= n_peak_max):	N_ptl = n_peak_max
		sigma_win     = float(float(sigma)/float(p_size)) # filter radius
		nx_d          = int(nx/deci)
		ny_d          = int(ny/deci)
		nx_fft_p      = sparx_fundamentals.smallprime(nx_d)
		ny_fft_p      = sparx_fundamentals.smallprime(ny_d)
		nx_fft_m      = nx_fft_p*int(deci)
		ny_fft_m      = ny_fft_p*int(deci)
		img1          = sparx_fundamentals.window2d(img1, nx_fft_m, ny_fft_m, "l")
		if(CC_method == 1):
			if(int(deci) == 1):
				img1         = sparx_filter.filt_gaussh(sparx_fundamentals.fft(img1), sigma_win)
			else:
				feq_deci = 0.5/deci
				img1       = EMAN2_cppwrap.Util.decimate(sparx_fundamentals.fft(sparx_filter.filt_tanl(sparx_filter.filt_gaussh(sparx_fundamentals.fft(img1), sigma_win), feq_deci, 0.04)), int(deci), int(deci),1)
				img1       = sparx_fundamentals.fft(img1)
		else:
			feq_deci           = 0.5/deci
			img2               = EMAN2_cppwrap.Util.decimate(sparx_filter.filt_tanl(img1, feq_deci, 0.04), int(deci), int(deci), 1)
			img2               = sparx_fundamentals.fft(img2)
			img1               = EMAN2_cppwrap.Util.decimate(sparx_fundamentals.fft(sparx_filter.filt_tanl(sparx_filter.filt_gaussh(sparx_fundamentals.fft(img1,sigma_win)), feq_deci, 0.04)), int(deci),int(deci), 1)
		for j in range(i_tem):
			t.read_image(templatefile, j)
			if(int(CC_method) == 2):	cc_map = sparx_morphology.flcc(t, img2)
			else:
				t_pad  = EMAN2_cppwrap.Util.pad(t, nx_fft_p, ny_fft_p, 1,0,0,0, "circumference")
				cc_map = sparx_fundamentals.ccf(img1, t_pad)
				del t_pad
			peaks.insert(0, cc_map.peak_ccf(p_size/2-1.0))
		if(int(CC_method) == 2): del img2
		peak = peaks[0]
		for j in range(1,i_tem):#output results
			peak1 = EMAN2_cppwrap.Util.merge_peaks(peak, peaks[j], hf_p)
			del peak
			peak  = peak1
		n_peak = int(len(peak)/3)
		if n_peak <= N_ptl :	N_wi=int(n_peak)
		else:			N_wi=int(N_ptl )
		out = open(f_coord, "w")
		out.write("#Coordinates: %s\n")
		if N_wi == 0 :	sparx_global_def.ERROR("Number of particles is zero", "autowin", 0)
		if(CC_method == 1):  img1 = sparx_fundamentals.fft(img1)
		for k in range(N_wi):
			x       = peak[k*3+1] -p_size/2
			y       = peak[k*3+2] -p_size/2
			# print "x==",x, "y===",y, " ccf==",peak[k*3]
			out.write("%d\t%f\t%f\n" % (k+1,x,y))
			reg     = EMAN2_cppwrap.Region(x,y, p_size, p_size)
			wi      = img1.get_clip(reg)
			ra      = sparx_fundamentals.ramp(wi)
			outlist = sparx_utilities.ce_fit(ra, e_n, mask)
			outlist[2].set_attr_dict({'xp':peak[k*3+1], 'yp': peak[k*3+2], 'mic':filename})
			if CTF: outlist[2].set( "ctf", ctf_params)
			outlist[2].write_image(file_particle, k)
		out.close()

def autowin_MPI(indir,outdir, noisedoc, noisemic, templatefile, deci, CC_method, p_size, sigma, hf_p,n_peak_max, contrast_invert=1, prefix_of_micrograph="micrograph"):
	""" 
		automated particle picking program
	   	CC_method=1  Applying high pass filter to detect protein particles
		sigma can be adjusted to perform particle picking. 
		CC_method=2  Using fast local normalization method to detect protein particles
		The output files are detected coordinates and windowed particles
	"""
	pass#IMPORTIMPORTIMPORT from utilities 		import get_image, model_circle,ce_fit,drop_image,info
	pass#IMPORTIMPORTIMPORT from fundamentals 	import smallprime, window2d, ccf, ramp, fft
	pass#IMPORTIMPORTIMPORT from filter 		import filt_gaussh, filt_tanl
	pass#IMPORTIMPORTIMPORT from string 		import split
	pass#IMPORTIMPORTIMPORT from morphology 	import flcc
	pass#IMPORTIMPORTIMPORT from random     	import randint
	pass#IMPORTIMPORTIMPORT import sys
	pass#IMPORTIMPORTIMPORT import os
	pass#IMPORTIMPORTIMPORT from mpi 	    import mpi_comm_size, mpi_comm_rank, MPI_COMM_WORLD
	pass#IMPORTIMPORTIMPORT from mpi 	    import mpi_barrier, mpi_bcast
	pass#IMPORTIMPORTIMPORT from mpi 	    import MPI_INT

	number_of_proc = mpi.mpi_comm_size(mpi.MPI_COMM_WORLD)
	myid           = mpi.mpi_comm_rank(mpi.MPI_COMM_WORLD)
	#  chose a random node as a main one...
	main_node      = 0
	if(myid == 0): main_node = random.randint(0,number_of_proc-1)
	main_node      = mpi.mpi_bcast(main_node, 1, mpi.MPI_INT, 0, mpi.MPI_COMM_WORLD)
	if os.path.exists(indir)  is False: sparx_global_def.ERROR("micrograph directory does not exsit", "autowin_MPI",1,myid)	
	flist = os.listdir(indir)
	nima          = 0
	mic_name_list = []
	for i, v in enumerate(flist):
		micname                  = os.path.join(indir,v)
		(filename, filextension) = os.path.splitext(v  )
		if(filename[0:len(prefix_of_micrograph)] == prefix_of_micrograph):
			mic_name_list.append(micname)
			nima += 1
	
	if os.path.exists(outdir):  sparx_global_def.ERROR('Output directory exists, please change the name and restart the program', "autowin_MPI ", 1,myid)
	mpi.mpi_barrier(mpi.MPI_COMM_WORLD)
	
	if myid == int(main_node): # directory cleaning only performed by main node
		sparx_utilities.print_begin_msg("autowin_MPI")
		os.mkdir(outdir)
	mpi.mpi_barrier(mpi.MPI_COMM_WORLD)
	
	#if(number_of_proc <= nima ):	nimage_per_node = nima/number_of_proc
	#else: 				nimage_per_node = 1 
	#image_start     = myid * nimage_per_node
	#if(myid == number_of_proc-1):  image_end = nima
	#else:                          image_end = image_start + nimage_per_node
	image_start, image_end = MPI_start_end(nima, number_of_proc, myid)
	if image_start == image_end: 
		if image_end == nima:	image_start -= 1
		else: image_end += 1
	
	t       = EMAN2_cppwrap.EMData()
	e_n     = EMAN2_cppwrap.EMData()
	e       = sparx_utilities.get_image(noisemic)
	i_tem   = EMAN2_cppwrap.EMUtil.get_image_count(templatefile)
	e_n.read_image(templatefile,0)
	nx      = e_n.get_xsize()
	ny      = e_n.get_ysize()
	rad     = int(nx/2)-1
	mask    = sparx_utilities.model_circle(rad, nx, ny)
	f       = open(noisedoc, "r")
	rstring = f.readlines() 
	f.close
	xs      = rstring[0]
	tmp     = string.split(xs)
	x       = tmp[0]
	y       = tmp[1]
	x_noi   = int(x)-p_size/2
	y_noi   = int(y)-p_size/2
	reg     = EMAN2_cppwrap.Region(x_noi,y_noi, p_size, p_size)# Get the reference noise from the input mic and noise coordinates
	e_n     = e.get_clip(reg)
	
	for i in range(image_start,image_end):
		filename=mic_name_list[i] 
		print('%-15s%-30s'%("micrographs # ",filename))
		(f_nam, filextension) = os.path.splitext(filename)
		out1_file     = "coord_"+f_nam[len(prefix_of_micrograph)+len(indir)+2:]+".txt"
		out2_file     = "particle_"+f_nam[len(prefix_of_micrograph)+len(indir)+2:]+filextension
		f_coord       = os.path.join(outdir, out1_file)
		file_particle = os.path.join(outdir, out2_file) #  parse output file names 			
		peaks         = []
		img1 = EMAN2_cppwrap.EMData()
		img1.read_image(filename)
		img1          *= contrast_invert
		nx            = img1.get_xsize()
		ny            = img1.get_ysize()
		N_ptl         = int(nx*ny/p_size/p_size) # number of possible particles
		if(N_ptl >= n_peak_max):	N_ptl = n_peak_max
		sigma_win     = float(float(sigma)/float(p_size)) # filter radius
		nx_d          = int(nx/deci)
		ny_d          = int(ny/deci)
		nx_fft_p      = sparx_fundamentals.smallprime(nx_d)
		ny_fft_p      = sparx_fundamentals.smallprime(ny_d)
		nx_fft_m      = nx_fft_p*int(deci)
		ny_fft_m      = ny_fft_p*int(deci)
		img1          = sparx_fundamentals.window2d(img1, nx_fft_m, ny_fft_m, "l")
		if(CC_method == 1):
			if(int(deci) == 1):
				img1         = sparx_filter.filt_gaussh(sparx_fundamentals.fft(img1), sigma_win)
			else:
				feq_deci = 0.5/deci
				img1       = EMAN2_cppwrap.Util.decimate(sparx_fundamentals.fft(sparx_filter.filt_tanl(sparx_filter.filt_gaussh(sparx_fundamentals.fft(img1), sigma_win), feq_deci, 0.04)), int(deci), int(deci),1)
				img1       = sparx_fundamentals.fft(img1)
		else:
			feq_deci           = 0.5/deci
			img2               = EMAN2_cppwrap.Util.decimate(sparx_filter.filt_tanl(img1, feq_deci, 0.04), int(deci), int(deci), 1)
			img2               = sparx_fundamentals.fft(img2)
			img1               = EMAN2_cppwrap.Util.decimate(sparx_fundamentals.fft(sparx_filter.filt_tanl(sparx_filter.filt_gaussh(sparx_fundamentals.fft(img1,sigma_win)), feq_deci, 0.04)), int(deci),int(deci), 1)
		for j in range(i_tem):
			t.read_image(templatefile, j)
			if(int(CC_method) == 2):	cc_map = sparx_morphology.flcc(t, img2)
			else:
				t_pad  = EMAN2_cppwrap.Util.pad(t, nx_fft_p, ny_fft_p, 1,0,0,0, "circumference")
				cc_map = sparx_fundamentals.ccf(img1, t_pad)
				del t_pad
			peaks.insert(0,cc_map.peak_ccf(p_size/2-1.0))
		if(int(CC_method) == 2): del img2
		peak = peaks[0]
		for j in range(1,i_tem):#output results
			peak1 = EMAN2_cppwrap.Util.merge_peaks(peak, peaks[j], hf_p)
			del peak
			peak  = peak1
		n_peak = int(len(peak)/3)
		if n_peak <= N_ptl :	N_wi=int(n_peak)
		else:			N_wi=int(N_ptl )			
		out = open(f_coord, "w")
		out.write("#Coordinates: %s\n")
		if N_wi == 0 :	sparx_global_def.ERROR("Number of particles is zero","autowin.py",0,myid)
		if(CC_method == 1):img1 = sparx_fundamentals.fft(img1)			
		for k in range(N_wi):
			x       = peak[k*3+1] -p_size/2
			y       = peak[k*3+2] -p_size/2
			# print "x==",x, "y===",y, " ccf==",peak[k*3]
			out.write("%d\t%f\t%f\n" % (k+1,x,y))
			reg     = EMAN2_cppwrap.Region(x, y, p_size, p_size)
			wi      = img1.get_clip(reg)
			ra      = sparx_fundamentals.ramp(wi)
			outlist = sparx_utilities.ce_fit(ra,e_n,mask)
			outlist[2].set_attr_dict({'xp':peak[k*3+1], 'yp': peak[k*3+2],'mic':filename})
			outlist[2].write_image(file_particle, k)
		out.close()


def ihrsr(stack, ref_vol, outdir, maskfile, ir, ou, rs, xr, ynumber, \
		txs, delta, initial_theta, delta_theta, an, maxit, CTF, snr, dp, ndp, dp_step, dphi, ndhpi, dphi_step, psi_max, \
		rmin, rmax, fract, nise, npad, sym, user_func_name, datasym, \
		pixel_size, debug = False, MPI = False, WRAP = 1, y_restrict=-1.0):

	ihrsr_MPI(stack, ref_vol, outdir, maskfile, ir, ou, rs, xr, ynumber, 
			txs, delta, initial_theta, delta_theta, an, maxit, CTF, snr, dp, ndp, dp_step, dphi, ndhpi, dphi_step, psi_max,
			rmin, rmax, fract, nise, npad, sym, user_func_name, datasym,
			pixel_size, debug, y_restrict, WRAP)

	return

def ihrsr_MPI(stack, ref_vol, outdir, maskfile, ir, ou, rs, xr, ynumber,\
	txs, delta, initial_theta, delta_theta, an, maxit, CTF, snr, dp, ndp, dp_step, dphi, ndphi, dphi_step, psi_max,\
	rmin, rmax, fract, nise, npad, sym, user_func_name, datasym,\
	pixel_size, debug, y_restrict, WRAP):

	pass#IMPORTIMPORTIMPORT from alignment      import Numrinit, prepare_refrings, proj_ali_helical, proj_ali_helical_90, proj_ali_helical_local, proj_ali_helical_90_local, helios,helios7
	pass#IMPORTIMPORTIMPORT from utilities      import model_circle, get_image, drop_image, get_input_from_string, pad, model_blank, sym_vol
	pass#IMPORTIMPORTIMPORT from utilities      import bcast_list_to_all, bcast_number_to_all, reduce_EMData_to_root, bcast_EMData_to_all
	pass#IMPORTIMPORTIMPORT from utilities      import send_attr_dict
	pass#IMPORTIMPORTIMPORT from utilities      import get_params_proj, set_params_proj, file_type
	pass#IMPORTIMPORTIMPORT from fundamentals   import rot_avg_image
	pass#IMPORTIMPORTIMPORT from pixel_error    import max_3D_pixel_error
	pass#IMPORTIMPORTIMPORT import os
	pass#IMPORTIMPORTIMPORT import types
	pass#IMPORTIMPORTIMPORT from utilities      import print_begin_msg, print_end_msg, print_msg
	pass#IMPORTIMPORTIMPORT from mpi            import mpi_bcast, mpi_comm_size, mpi_comm_rank, MPI_FLOAT, MPI_COMM_WORLD, mpi_barrier
	pass#IMPORTIMPORTIMPORT from mpi            import mpi_recv,  mpi_send
	pass#IMPORTIMPORTIMPORT from mpi            import mpi_reduce, MPI_INT, MPI_SUM
	pass#IMPORTIMPORTIMPORT from filter         import filt_ctf
	pass#IMPORTIMPORTIMPORT from projection     import prep_vol, prgs
	pass#IMPORTIMPORTIMPORT from statistics     import hist_list, varf3d_MPI
	pass#IMPORTIMPORTIMPORT from applications   import MPI_start_end
	pass#IMPORTIMPORTIMPORT from EMAN2 import Vec2f
	pass#IMPORTIMPORTIMPORT from string    import lower,split
	pass#IMPORTIMPORTIMPORT from math import cos, pi

	number_of_proc = mpi.mpi_comm_size(mpi.MPI_COMM_WORLD)
	myid           = mpi.mpi_comm_rank(mpi.MPI_COMM_WORLD)
	main_node = 0

	if myid == 0:
		if os.path.exists(outdir):  nx = 1
		else:  nx = 0
	else:  nx = 0
	ny = sparx_utilities.bcast_number_to_all(nx, source_node = main_node)

	if ny == 1:  sparx_global_def.ERROR('Output directory exists, please change the name and restart the program', "ihrsr_MPI", 1,myid)
	mpi.mpi_barrier(mpi.MPI_COMM_WORLD)

	if myid == main_node:
		os.mkdir(outdir)
		pass#IMPORTIMPORTIMPORT import global_def
		sparx_global_def.LOGFILE =  os.path.join(outdir, sparx_global_def.LOGFILE)
	mpi.mpi_barrier(mpi.MPI_COMM_WORLD)


	nlprms = (2*ndp+1)*(2*ndphi+1)
	if nlprms< number_of_proc:
		sparx_global_def.ERROR('number of CPUs is larger than the number of helical search, please reduce it or at this moment modify ndp,dphi in the program', "ihrsr_MPI", 1,myid)


	if debug:
		pass#IMPORTIMPORTIMPORT from time import sleep
		while not os.path.exists(outdir):
			print("Node ",myid,"  waiting...")
			time.sleep(5)

		info_file = os.path.join(outdir, "progress%04d"%myid)
		finfo = open(info_file, 'w')
	else:
		finfo = None

	symref = "s"+sym

	ref_a= "P"
	symmetryLower = sym.lower()
	symmetry_string = string.split(symmetryLower)[0]

	xrng        = sparx_utilities.get_input_from_string(xr)
	y_restrict       = sparx_utilities.get_input_from_string(y_restrict)
	ynumber	    = sparx_utilities.get_input_from_string(ynumber)
	for i in range(len(ynumber)):
		if ynumber[i] >= 0:
			if(ynumber[i]%2==1): ynumber[i]=ynumber[i]+1
	yrng =[]

	for i in range(len(xrng)): yrng.append(dp/2)

	stepx        = sparx_utilities.get_input_from_string(txs)
	delta       = sparx_utilities.get_input_from_string(delta)
	lstp = min(len(xrng), len(yrng), len(stepx), len(delta))
	if an == "-1": an = [-1] * lstp
	else:          an = sparx_utilities.get_input_from_string(an)

	first_ring  = int(ir)
	rstep       = int(rs)
	last_ring   = int(ou)
	max_iter    = int(maxit)

	vol     = EMAN2_cppwrap.EMData()
	vol.read_image(ref_vol)
	nx      = vol.get_xsize()
	ny      = vol.get_ysize()
	nz      = vol.get_zsize()
	nmax = max(nx, ny, nz)
	if ( nx == ny ):
		if( nx == nz):
			xysize = -1
			zsize = -1
		elif( nx < nz):
			xysize = nx
			zsize = -1
		else:
			zsize = nz
			xysize = -1
	else:
		sparx_global_def.ERROR('the x and y size have to be same, please change the reference volume and restart the program', "ihrsr_MPI", 1,myid)

	if last_ring < 0:	last_ring = int(nx/2) - 2

	if myid == main_node:
		pass#IMPORTIMPORTIMPORT import user_functions
		user_func = sparx_user_functions.factory[user_func_name]

		sparx_utilities.print_msg("Input stack                               : %s\n"%(stack))
		sparx_utilities.print_msg("Reference volume                          : %s\n"%(ref_vol))	
		sparx_utilities.print_msg("Output directory                          : %s\n"%(outdir))
		sparx_utilities.print_msg("Maskfile                                  : %s\n"%(maskfile))
		sparx_utilities.print_msg("Inner radius                              : %i\n"%(first_ring))
		sparx_utilities.print_msg("Outer radius                              : %i\n"%(last_ring))
		sparx_utilities.print_msg("Ring step                                 : %i\n"%(rstep))
		sparx_utilities.print_msg("X search range                            : %s\n"%(xrng))
		sparx_utilities.print_msg("Y number                                  : %s\n"%(ynumber))
		sparx_utilities.print_msg("Translational stepx                       : %s\n"%(stepx))
		sparx_utilities.print_msg("Angular step                              : %s\n"%(delta))
		sparx_utilities.print_msg("Angular search range                      : %s\n"%(an))
		sparx_utilities.print_msg("Initial Theta                             : %s\n"%(initial_theta))
		sparx_utilities.print_msg("Maximum range for psi search              : %s\n"%(psi_max))
		sparx_utilities.print_msg("min radius for helical search (in pix)    : %5.4f\n"%(rmin))
		sparx_utilities.print_msg("max radius for helical search (in pix)    : %5.4f\n"%(rmax))
		sparx_utilities.print_msg("fraction of volume used for helical search: %5.4f\n"%(fract))
		sparx_utilities.print_msg("initial symmetry - angle                  : %5.4f\n"%(dphi))
		sparx_utilities.print_msg("initial symmetry - axial rise             : %5.4f\n"%(dp))
		sparx_utilities.print_msg("Maximum iteration                         : %i\n"%(max_iter))
		sparx_utilities.print_msg("Data with CTF                             : %s\n"%(CTF))
		sparx_utilities.print_msg("Signal-to-Noise Ratio                     : %5.4f\n"%(snr))
		sparx_utilities.print_msg("symmetry output doc file                  : %s\n"%(datasym))
		sparx_utilities.print_msg("number of times to impose initial symmetry: %i\n"%(nise))
		sparx_utilities.print_msg("npad                                      : %i\n"%(npad))
		sparx_utilities.print_msg("User function                             : %s\n"%(user_func_name))

	if maskfile:
		if type(maskfile) is bytes: mask3D = sparx_utilities.get_image(maskfile)
		else:                                  mask3D = maskfile
	else: mask3D = None
	#else: mask3D = model_circle(last_ring, nx, nx, nx)

	numr	= sparx_alignment.Numrinit(first_ring, last_ring, rstep, "F")

	if CTF:
		pass#IMPORTIMPORTIMPORT from reconstruction import recons3d_4nn_ctf_MPI
		pass#IMPORTIMPORTIMPORT from filter         import filt_ctf
	pass#IMPORTIMPORTIMPORT else:	 from reconstruction import recons3d_4nn_MPI

	if myid == main_node:
		if(sparx_utilities.file_type(stack) == "bdb"):
			pass#IMPORTIMPORTIMPORT from EMAN2db import db_open_dict
			dummy = EMAN2db.db_open_dict(stack, True)
		# horatio active_refactoring Jy51i1EwmLD4tWZ9_00000_1
		# active = EMUtil.get_all_attributes(stack, 'active')
		# list_of_particles = []
		# for im in xrange(len(active)):
		# 	if active[im]:  list_of_particles.append(im)
		# del active
		# nima = len(list_of_particles)
		
		nima = EMAN2_cppwrap.EMUtil.get_image_count(stack)
		list_of_particles = list(range(nima))
		
	else:
		nima = 0
	total_nima = sparx_utilities.bcast_number_to_all(nima, source_node = main_node)

	if myid != main_node:
		list_of_particles = [-1]*total_nima
	list_of_particles = sparx_utilities.bcast_list_to_all(list_of_particles, myid,  source_node = main_node)

	image_start, image_end = MPI_start_end(total_nima, number_of_proc, myid)
	# create a list of images for each node
	list_of_particles = list_of_particles[image_start: image_end]
	nima = len(list_of_particles)
	if debug:
		finfo.write("image_start, image_end: %d %d\n" %(image_start, image_end))
		finfo.flush()
	
	data = EMAN2_cppwrap.EMData.read_images(stack, list_of_particles)
	
	data_nx = data[0].get_xsize()
	data_ny = data[0].get_ysize()
	data_nn = max(data_nx, data_ny)
	mask2D  = sparx_utilities.pad(sparx_utilities.model_blank(2*int(rmax), data_ny, 1, 1.0), data_nx, data_ny, 1, 0.0)
	
	#if fourvar:  original_data = []
	for im in range(nima):
		data[im].set_attr('ID', list_of_particles[im])
		sttt = EMAN2_cppwrap.Util.infomask(data[im], mask2D, False)
		data[im] = data[im] - sttt[0]
		#if fourvar: original_data.append(data[im].copy())
		if CTF:
			st = data[im].get_attr_default("ctf_applied", 0)
			if(st == 0):
				ctf_params = data[im].get_attr("ctf")
				data[im] = sparx_filter.filt_ctf(data[im], ctf_params)
				data[im].set_attr('ctf_applied', 1)
	del mask2D

	if debug:
		finfo.write( '%d loaded  \n' % nima )
		finfo.flush()

	for i in range(len(xrng)): yrng[i]=dp/(2*pixel_size)
	pass#IMPORTIMPORTIMPORT from math import sin, pi
	if ( ou > ( nmax/2.0)*numpy.sin( initial_theta*numpy.pi/180) - dp/2.0/pixel_size -1.0 ):
		sparx_global_def.ERROR('ou should be less than or equal to ----( nmax/2.0)*sin( initial_theta*pi/180) - dp/2.0/pixel_size -1.0 ', "ihrsr_MPI", 1,myid)

	if myid == main_node:
		sparx_utilities.print_msg("Pixel size in Angstroms                   : %5.4f\n\n"%(pixel_size))
		sparx_utilities.print_msg("Y search range (pix) initialized as       : %s\n\n"%(yrng))

	pass#IMPORTIMPORTIMPORT from time import time

	#  this is needed for gathering of pixel errors
	disps = []
	recvcount = []
	for im in range(number_of_proc):
		if( im == main_node ):  disps.append(0)
		else:                   disps.append(disps[im-1] + recvcount[im-1])
		ib, ie = MPI_start_end(total_nima, number_of_proc, im)
		recvcount.append( ie - ib )
	
	for ii in range(lstp):
		if stepx[ii] == 0.0:
			if xrng[ii] != 0.0:
				sparx_global_def.ERROR('xrange step size cannot be zero', "ihrsr_MPI", 1,myid)
			else:
				stepx[ii] = 1.0 # this is to prevent division by zero in c++ code
	total_iter = 0
	# do the projection matching
	for N_step in range(lstp):
		terminate = 0
		Iter = 0
		while(Iter < max_iter and terminate == 0):
			yrng[N_step]=float(dp)/(2*pixel_size) #will change it later according to dp
			if(ynumber[N_step]==0): stepy = 0.0
			else:                   stepy = (2*yrng[N_step]/ynumber[N_step])

			pixer  = [0.0]*nima
			modphi = [0.0]*nima
			Iter += 1
			total_iter += 1
			if myid == main_node:
				start_time = time.time()
				if an[N_step] == -1:
					sparx_utilities.print_msg("\nITERATION #%3d,  inner iteration #%3d\nDelta = %5.2f, an = %5.4f, xrange (Pixels) = %5.4f,stepx (Pixels) = %5.4f, yrng (Pixels) = %5.4f,  stepy (Pixels) = %5.4f, ynumber = %3d\n"%(total_iter, Iter, delta[N_step], an[N_step], xrng[N_step],stepx[N_step],yrng[N_step],stepy, ynumber[N_step]))
				else:
					sparx_utilities.print_msg("\nITERATION #%3d,  inner iteration #%3d\nDelta = %5.2f, an = %5.4f, xrange (Pixels) = %5.4f,stepx (Pixels) = %5.4f, yrng (Pixels) = %5.4f,  stepy (Pixels) = %5.4f, y_restrict (Pixels)=%5.4f, ynumber = %3d\n"%(total_iter, Iter, delta[N_step], an[N_step], xrng[N_step],stepx[N_step],yrng[N_step],stepy,y_restrict[N_step], ynumber[N_step]))
			if( xysize == -1 and zsize==-1 ):
				volft,kb = sparx_projection.prep_vol( vol )
				refrings = sparx_alignment.prepare_refrings( volft, kb, nmax, delta[N_step], ref_a, symref, numr, MPI = True, phiEqpsi = "Zero", initial_theta =initial_theta, delta_theta = delta_theta)
				del volft,kb
			else:
				volft, kbx, kby, kbz = sparx_projection.prep_vol( vol )
				refrings = sparx_alignment.prepare_refrings( volft, kbz, nmax, delta[N_step], ref_a, symref, numr, MPI = True, phiEqpsi = "Zero", kbx = kbx, kby = kby, initial_theta =initial_theta, delta_theta = delta_theta)
				del volft, kbx, kby, kbz

			if myid== main_node:
				sparx_utilities.print_msg( "Time to prepare rings: %d\n" % (time.time()-start_time) )
				start_time = time.time()
			#split refrings to two list: refrings1 (even point-group symmetry AND theta = 90. ), 
			#   or refrings2 ( odd point-group symmetry AND any theta (including theta=90), OR even point-group symmetry AND theta <> 90.
			refrings1= []
			refrings2= []
			sn = int(symmetry_string[1:])
			for i in range( len(refrings) ):
				if( sn%2 ==0 and abs( refrings[i].get_attr('n3') ) <1.0e-6 ):
					#  even point-group symmetry AND theta = 90. 
					refrings1.append( refrings[i] )
				else:
					# odd point-group symmetry AND any theta (including theta=90), OR even point-group symmetry AND theta <> 90.
					refrings2.append( refrings[i] )
					"""Multiline Comment29"""
			del refrings
			pass#IMPORTIMPORTIMPORT from numpy import float32
			dpp = numpy.float32(float(dp)/pixel_size)
			dpp = float( dpp )
			dpp_half = dpp/2.0

			for im in range( nima ):
				"""Multiline Comment30"""
				peak1 = None
				peak2 = None
				t1 = data[im].get_attr("xform.projection")
				if( len(refrings1) > 0):
					if  an[N_step] == -1:
						peak1, phihi1, theta1, psi1, sxi1, syi1 = \
							sparx_alignment.proj_ali_helical_90(data[im], refrings1, numr, xrng[N_step], yrng[N_step], stepx[N_step], ynumber[N_step], psi_max, finfo)
					else:
						peak1, phihi1, theta1, psi1, sxi1, syi1 = \
							sparx_alignment.proj_ali_helical_90_local(data[im], refrings1, numr, xrng[N_step], yrng[N_step], stepx[N_step], ynumber[N_step], an[N_step], psi_max, finfo, yrnglocal=y_restrict[N_step])
					#print "  1  ",im, peak1, phihi1, theta1, psi1, sxi1, syi1
				if( len(refrings2) > 0):
					if  an[N_step] == -1:
						peak2, phihi2, theta2, psi2, sxi2, syi2 = \
							sparx_alignment.proj_ali_helical(data[im], refrings2, numr, xrng[N_step], yrng[N_step], stepx[N_step], ynumber[N_step], psi_max, finfo)
					else:
						peak2, phihi2, theta2, psi2, sxi2, syi2 = \
							sparx_alignment.proj_ali_helical_local(data[im], refrings2, numr, xrng[N_step], yrng[N_step], stepx[N_step], ynumber[N_step], an[N_step], psi_max, finfo, yrnglocal=y_restrict[N_step])
					#print "  2  ",im, peak2, phihi2, theta2, psi2, sxi2, syi2
				if peak1 is None: 
					peak = peak2
					phihi = phihi2
					theta = theta2
					psi = psi2
					sxi = sxi2
					syi = syi2
				elif peak2 is None:
					peak = peak1
					phihi = phihi1
					theta = theta1
					psi = psi1
					sxi = sxi1
					syi = syi1
				else:
					if(peak1 >= peak2):
						peak = peak1
						phihi = phihi1
						theta = theta1
						psi = psi1
						sxi = sxi1
						syi = syi1
					else:
						peak = peak2
						phihi = phihi2
						theta = theta2
						psi = psi2
						sxi = sxi2
						syi = syi2
				#print "  3  ",im, peak, phihi, theta, psi, sxi, syi
				if(peak > -1.0e22):
					if WRAP == 1:
						#Guozhi Tao: wrap y-shifts back into box within rise of one helical unit by changing phi
						tp = EMAN2_cppwrap.Transform({"type":"spider","phi":phihi,"theta":theta,"psi":psi})
						tp.set_trans( EMAN2_cppwrap.Vec2f( -sxi, -syi ) )
						dtp = tp.get_params("spider")
						dtp_ty = float( dtp["ty"] )
						del dtp
						if( abs(dtp_ty) >dpp_half):
							dtp_ty_temp = dtp_ty
							if( abs(psi-90) < 90  ):
								sign_psi = 1
							else:
								sign_psi = -1
							if( dtp_ty > 0):
								period_step = -1*sign_psi
							else:
								period_step = 1*sign_psi
							nperiod = 0
							while( abs( dtp_ty_temp ) > dpp_half ):
								nperiod += period_step
								th = EMAN2_cppwrap.Transform({"type":"spider","phi": -nperiod*dphi, "tz":nperiod*dpp})
								tfinal = tp*th
								df = tfinal.get_params("spider")
								dtp_ty_temp = float( df["ty"] )

							phihi = float(df["phi"])
							sxi   = float(-df["tx"])
							syi   = float(-df["ty"])
					#print "  4  ",im, peak, phihi, theta, psi, sxi, syi

					# unique ranges of azimuthal angle for ortho-axial and non-ortho-axial projection directions are identified by [k0,k1) and [k2,k3), where k0, k1, k2, k3 are floats denoting azimuthal angles.
					# Eulerian angles whose azimuthal angles are mapped into [k2, k3) are related to Eulerian angles whose azimuthal angles are mapped into [k0, k1) by an in-plane mirror operaton along the x-axis.

					tp = EMAN2_cppwrap.Transform({"type":"spider","phi":phihi,"theta":theta,"psi":psi})
					tp.set_trans( EMAN2_cppwrap.Vec2f( -sxi, -syi ) )

					k0 =   0.0
					k2 = 180.0
					if( abs( tp.at(2,2) )<1.0e-6 ):
						if (symmetry_string[0] =="c"):
							if sn%2 == 0:  k1=360.0/sn
							else:          k1=360.0/2/sn
						elif (symmetry_string[0] =="d"):
							if sn%2 == 0:  k1=360.0/2/sn
							else:          k1=360.0/4/sn
					else:
						if (symmetry_string[0] =="c"):  k1=360.0/sn
						if (symmetry_string[0] =="d"):  k1=360.0/2/sn
					k3 = k1 +180.0

					pass#IMPORTIMPORTIMPORT from utilities import get_sym
					T = sparx_utilities.get_sym(symmetry_string[0:])

					d1tp = tp.get_params('spider')
					sxnew    = -d1tp["tx"]
					synew    = -d1tp["ty"]
					phinew   =  d1tp['phi']
					thetanew =  d1tp["theta"]
					psinew   =  d1tp["psi"]
					del d1tp

					#print "  5  ",im, phinew, thetanew, psinew, sxnew, synew
					#print k0,k1,k2,k3

					for i in range( len(T) ):
						ttt = tp*EMAN2_cppwrap.Transform({"type":"spider","phi":T[i][0],"theta":T[i][1],"psi":T[i][2]})
						d1  = ttt.get_params("spider")

						if ( abs( tp.at(2,2) )<1.0e-6 ):
							if( sn%2==1 ): # theta=90 and n odd, only one of the two region match

								if( ( d1['phi'] >= k0 and d1['phi'] < k1 ) or ( d1['phi'] >= k2 and d1['phi'] < k3 )):

									sxnew    = -d1["tx"]
									synew    = -d1["ty"]
									phinew   =  d1['phi']
									thetanew =  d1["theta"]
									psinew   =  d1["psi"]

									# For boundary cases where phihi is exactly on the boundary of the unique range, there may be two symmetry related Eulerian angles which are both in the unique 
									# range but whose psi differ by 180. 
									# For example, (180,90,270) has two symmetry related angles in unique range: (180,90,270) and (180, 90, 90)
									# In local search, psi should stay within neighborhood of original value, so take the symmetry related
									# Eulerian angles in unique range which does not change psi by 180.
									if an[N_step] != -1:
										if abs(psinew - psi) < 90:
											break
							else: #for theta=90 and n even, there is no mirror version during aligment, so only consider region [k0,k1]

								if( d1['phi'] >= float(k0) and d1['phi'] < float(k1)  ) :

									sxnew    = - d1["tx"]
									synew    = - d1["ty"]
									phinew   = d1['phi']
									thetanew = d1["theta"]
									psinew   = d1["psi"]

						else: #theta !=90, # if theta >90, put the projection into [k2,k3]. Otherwise put it into the region [k0,k1]

							if( sn==1):
								sxnew    = sxi
								synew    = syi
								phinew   = phihi
								thetanew = theta
								psinew   = psi
							else:

								if (tp.at(2,2) >0.0): #theta <90
									
									if(  d1['phi'] >= float(k0) and d1['phi'] < float(k1)):
										if( numpy.cos( numpy.pi*float( d1['theta'] )/180.0 )>0.0 ):
			
											sxnew    = - d1["tx"]
											synew    = - d1["ty"]
											phinew   = d1['phi']
											thetanew = d1["theta"]
											psinew   = d1["psi"]
		
								else:
									if(  d1['phi'] >= float(k2) and d1['phi'] < float(k3)):
										if( numpy.cos( numpy.pi*float( d1['theta'] )/180.0 )<0.0 ):

											sxnew    = - d1["tx"]
											synew    = - d1["ty"]
											phinew   = d1['phi']
											thetanew = d1["theta"]
											psinew   = d1["psi"]

						del ttt,d1

					t2 = EMAN2_cppwrap.Transform({"type":"spider","phi":phinew,"theta":thetanew,"psi":psinew})
					t2.set_trans(EMAN2_cppwrap.Vec2f(-sxnew, -synew))
					data[im].set_attr("xform.projection", t2)
					pixer[im]  = sparx_pixel_error.max_3D_pixel_error(t1, t2, numr[-3])
					modphi[im] = phinew

				else:
					# peak not found, parameters not modified
					pixer[im]  = 0.0
					phihi, theta, psi, sxi, syi = sparx_utilities.get_params_proj(data[im])
					modphi[im] = phihi
				#print "  6  ",im, phinew, thetanew, psinew, sxnew, synew

				#if(im==2):
				#	from sys import exit
				#	exit()

			del refrings1, refrings2
			if myid == main_node:
				sparx_utilities.print_msg("Time of alignment = %d\n"%(time.time()-start_time))
				start_time = time.time()

			#output pixel errors
			pass#IMPORTIMPORTIMPORT from mpi import mpi_gatherv
			recvbuf = mpi.mpi_gatherv(pixer, nima, mpi.MPI_FLOAT, recvcount, disps, mpi.MPI_FLOAT, main_node, mpi.MPI_COMM_WORLD)
			del pixer		
			mpi.mpi_barrier(mpi.MPI_COMM_WORLD)
			terminate = 0
			if(myid == main_node):
				recvbuf = list(map(float, recvbuf))
				pass#IMPORTIMPORTIMPORT from utilities import write_text_file
				sparx_utilities.write_text_file([list(range(len(recvbuf))), recvbuf], os.path.join(outdir, "pixer_%04d_%04d.txt"%(N_step+1,Iter)) )
				pass#IMPORTIMPORTIMPORT from statistics import hist_list
				lhist = 20
				region, histo = sparx_statistics.hist_list(recvbuf, lhist)
				if(region[0] < 0.0):  region[0] = 0.0
				msg = "      Histogram of pixel errors\n      ERROR       number of particles\n"
				sparx_utilities.print_msg(msg)
				for lhx in range(lhist):
					msg = " %10.2f     %7d\n"%(region[lhx], histo[lhx])
					sparx_utilities.print_msg(msg)
				# Terminate if 95% within 1 pixel error
				im = 0
				for lhx in range(lhist):
					if(region[lhx] > 1.0): break
					im += histo[lhx]
				#if(im/float(total_nima) > 0.95):  terminate = 1
				del region, histo
			#output distribution of phi
			#jeanmod
			recvbuf = mpi.mpi_gatherv(modphi, nima, mpi.MPI_FLOAT, recvcount, disps, mpi.MPI_FLOAT, main_node, mpi.MPI_COMM_WORLD)
			#end jeanmod
			mpi.mpi_barrier(mpi.MPI_COMM_WORLD)
			del modphi
			if(myid == main_node):
				recvbuf = list(map(float, recvbuf))
				phi_value_0 = []
				phi_value_180 = []
				for i in range ( len ( recvbuf ) ):
					if ( recvbuf[i] < 180.0):
						phi_value_0.append( recvbuf[i] )
					else:
						phi_value_180.append( recvbuf[i] ) 
				lhist = int( round(max(phi_value_0)/delta[N_step]) )
								# if delta is big, number of bins (lhist) will be small, leave it as it is
				# if delta is small, number of bins (lhist) will be big, adjust lhist = lhist/n such as the total 
				# number of bins close to 30, thus most likely we can see each bin contains particles.
				pass#IMPORTIMPORTIMPORT from math import ceil
				if ( len( phi_value_180) > 0):
					if lhist > 15:
						lhist = int(   lhist/numpy.ceil((lhist/15.0))  ) 
				else:
					if lhist > 30:
						lhist = int(   lhist/numpy.ceil((lhist/30.0))  )  
				region, histo = sparx_statistics.hist_list(phi_value_0, lhist)
				msg = "\n      Distribution of phi\n      phi         number of particles\n"
				sparx_utilities.print_msg(msg)
				for lhx in range(lhist):
					msg = " %10.2f     %7d\n"%(region[lhx], histo[lhx])
					sparx_utilities.print_msg(msg)
				del region, histo, phi_value_0
				if ( len( phi_value_180) > 0):
					region, histo = sparx_statistics.hist_list(phi_value_180, lhist)
					for lhx in range(lhist):
						msg = " %10.2f     %7d\n"%(region[lhx], histo[lhx])
						sparx_utilities.print_msg(msg)
					del region, histo, phi_value_180			
			del recvbuf
			terminate = mpi.mpi_bcast(terminate, 1, mpi.MPI_INT, 0, mpi.MPI_COMM_WORLD)
			terminate = int(terminate[0])
			if myid == main_node:
				sparx_utilities.print_msg("Time to compute pixer = %d\n"%(time.time()-start_time))
				start_time = time.time()
			# write out headers, under MPI writing has to be done sequentially
			mpi.mpi_barrier(mpi.MPI_COMM_WORLD)
			m = 5
			pass#IMPORTIMPORTIMPORT from mpi import mpi_recv, mpi_send, MPI_COMM_WORLD, MPI_FLOAT
			if myid == main_node:
				
				fexp = open(os.path.join(outdir, "parameters_%04d_%04d.txt"%(N_step+1,Iter)),"w")
				for n in range(number_of_proc):
					if n!=main_node:
						t = mpi.mpi_recv(recvcount[n] * m, mpi.MPI_FLOAT, n, sparx_global_def.SPARX_MPI_TAG_UNIVERSAL, mpi.MPI_COMM_WORLD)
						for i in range(recvcount[n]):
							for j in range(m):
								fexp.write(" %15.5f  "%t[j+i*m])
							fexp.write("\n")
					else:
						t = [0.0]*m
						for i in range(recvcount[myid]):
							t = sparx_utilities.get_params_proj(data[i])
							for j in range(m):
								fexp.write(" %15.5f  "%t[j])
							fexp.write("\n")
				fexp.close()
				del t
			else:
				nvalue = [0.0]*m*recvcount[myid]
				t = [0.0]*m
				for i in range(recvcount[myid]):
					t = sparx_utilities.get_params_proj(data[i])
					for j in range(m):
						nvalue[j + i*m] = t[j]
				mpi.mpi_send(nvalue, recvcount[myid] * m, mpi.MPI_FLOAT, main_node, sparx_global_def.SPARX_MPI_TAG_UNIVERSAL, mpi.MPI_COMM_WORLD)
				del nvalue
			if myid == main_node:
				sparx_utilities.print_msg("Time to write parameters = %d\n"%(time.time()-start_time))
				start_time = time.time()

			if CTF:  vol = sparx_reconstruction.recons3d_4nn_ctf_MPI(myid, data, symmetry=sym, snr = snr, npad = npad, xysize = nx, zsize = zsize)
			else:    vol = sparx_reconstruction.recons3d_4nn_MPI(myid, data, symmetry=sym, snr = snr, npad = npad, xysize = nx, zsize = zsize)

			if myid == main_node:
				sparx_utilities.print_msg("\n3D reconstruction time = %d\n"%(time.time()-start_time))
				start_time = time.time()

			"""Multiline Comment31"""
			#search for helical symmetry
			if myid == main_node:
				sparx_utilities.drop_image(vol, os.path.join(outdir, "vol%04d.hdf"%(total_iter)))

			if(total_iter > nise):
				sparx_utilities.bcast_EMData_to_all(vol, myid, main_node)
				#from filter import filt_gaussl
				#vol = filt_gaussl(vol, 0.25)

				if myid == main_node:
					lprms = []
					for i in range(-ndp,ndp+1,1):
						for j in range(-ndphi,ndphi+1,1):
							lprms.append( dp   + i*dp_step)
							lprms.append( dphi + j*dphi_step)
					#print "lprms===",lprms
					recvpara = []
					for im in range(number_of_proc):
						helic_ib, helic_ie = MPI_start_end(nlprms, number_of_proc, im)
						recvpara.append(helic_ib )
						recvpara.append(helic_ie )

				para_start, para_end = MPI_start_end(nlprms, number_of_proc, myid)

				list_dps     = [0.0]*((para_end-para_start)*2)
				list_fvalues = [-1.0]*((para_end-para_start)*1)

				if myid == main_node:
					for n in range(number_of_proc):
						if n!=main_node: mpi.mpi_send(lprms[2*recvpara[2*n]:2*recvpara[2*n+1]], 2 * (recvpara[2*n+1]-recvpara[2*n]), mpi.MPI_FLOAT, n, sparx_global_def.SPARX_MPI_TAG_UNIVERSAL, mpi.MPI_COMM_WORLD)
						else:    list_dps = lprms[2*recvpara[2*0]:2*recvpara[2*0+1]]
				else:
					list_dps = mpi.mpi_recv((para_end-para_start) * 2, mpi.MPI_FLOAT, main_node, sparx_global_def.SPARX_MPI_TAG_UNIVERSAL, mpi.MPI_COMM_WORLD)

				list_dps = list(map(float, list_dps))

				local_pos = [0.0, 0.0, -1.0e20]
				for i in range(para_end-para_start):
					fvalue = sparx_alignment.helios7(vol, pixel_size, list_dps[i*2], list_dps[i*2+1], fract, rmax, rmin)
					if(fvalue >= local_pos[2]):
						local_pos = [list_dps[i*2], list_dps[i*2+1], fvalue ]
				if myid == main_node:
					list_return = [0.0]*(3*number_of_proc)
					for n in range(number_of_proc):
						if n != main_node: list_return[3*n:3*n+3]                 = mpi.mpi_recv(3, mpi.MPI_FLOAT, n, sparx_global_def.SPARX_MPI_TAG_UNIVERSAL, mpi.MPI_COMM_WORLD)
						else:              list_return[3*main_node:3*main_node+3]  = local_pos[:]
				else:
					mpi.mpi_send(local_pos, 3, mpi.MPI_FLOAT, main_node, sparx_global_def.SPARX_MPI_TAG_UNIVERSAL, mpi.MPI_COMM_WORLD)

				if myid == main_node:	
					maxvalue = list_return[2]
					for i in range(number_of_proc):
						if( list_return[i*3+2] >= maxvalue ):
							maxvalue = list_return[i*3+2]
							dp       = list_return[i*3+0]
							dphi     = list_return[i*3+1]
					dp   = float(dp)
					dphi = float(dphi)
					#print  "  GOT dp dphi",dp,dphi

					vol  = vol.helicise(pixel_size, dp, dphi, fract, rmax, rmin)
					sparx_utilities.print_msg("New delta z and delta phi      : %s,    %s\n\n"%(dp,dphi))		
			
			else:
				if myid==main_node:
					#  in the first nise steps the symmetry is imposed
					vol = vol.helicise(pixel_size, dp, dphi, fract, rmax, rmin)
					sparx_utilities.print_msg("Imposed delta z and delta phi      : %s,    %s\n\n"%(dp,dphi))
			if(myid==main_node):
				fofo = open(os.path.join(outdir,datasym),'a')
				fofo.write('  %12.4f   %12.4f\n'%(dp,dphi))
				fofo.close()
				vol = vol.helicise(pixel_size, dp, dphi, fract, rmax, rmin)
				vol = sparx_utilities.sym_vol(vol, symmetry=sym)
				ref_data = [vol, mask3D]
				#if  fourvar:  ref_data.append(varf)
				vol = user_func(ref_data)
				vol = sparx_utilities.sym_vol(vol, symmetry=sym)
				vol = vol.helicise(pixel_size, dp, dphi, fract, rmax, rmin)

				sparx_utilities.drop_image(vol, os.path.join(outdir, "volf%04d.hdf"%(total_iter)))
				sparx_utilities.print_msg("\nSymmetry search and user function time = %d\n"%(time.time()-start_time))
				start_time = time.time()

			sparx_utilities.bcast_EMData_to_all(vol, myid, main_node)
			dp   = sparx_utilities.bcast_number_to_all(dp,   source_node = main_node)
			dphi = sparx_utilities.bcast_number_to_all(dphi, source_node = main_node)
			# del varf
	par_str = ["xform.projection"]
	if myid == main_node:
		if(sparx_utilities.file_type(stack) == "bdb"):
			pass#IMPORTIMPORTIMPORT from utilities import recv_attr_dict_bdb
			sparx_utilities.recv_attr_dict_bdb(main_node, stack, data, par_str, image_start, image_end, number_of_proc)
		else:
			pass#IMPORTIMPORTIMPORT from utilities import recv_attr_dict
			sparx_utilities.recv_attr_dict(main_node, stack, data, par_str, image_start, image_end, number_of_proc)
		sparx_utilities.print_msg("Time to write header information= %d\n"%(time.time()-start_time))
		start_time = time.time()
	else:	       sparx_utilities.send_attr_dict(main_node, data, par_str, image_start, image_end)
	if myid == main_node: sparx_utilities.print_end_msg("ihrsr_MPI")

"""Multiline Comment32"""

def copyfromtif(indir, outdir=None, input_extension="tif", film_or_CCD="f", output_extension="hdf", contrast_invert=1, Pixel_size=1, scanner_param_a=1,scanner_param_b=1, scan_step=63.5, magnification=40, MPI=False):
	"""
		purpose: convert raw image into SPIDER format images
		1. film_or_CCD =f (scanned film, density will be converted into OD )or c (CCD)
		2. contrast_invert =1 or -1  invert contrast or not 
		3. The default vale of input_extension is tiff.
		4. Pixel_size denotes the final converted image pixel size. When it set as 
		   negative, the program will switch back to image_decimate reduce image size
		   integer times. 
	"""
	if MPI:
		copyfromtif_MPI(indir, outdir, input_extension, film_or_CCD, output_extension, contrast_invert, Pixel_size, scanner_param_a, scanner_param_b, scan_step, magnification)
		return

	pass#IMPORTIMPORTIMPORT from utilities 		import get_image, drop_image
	pass#IMPORTIMPORTIMPORT from fundamentals 	import smallprime, window2d, resample, image_decimate
	pass#IMPORTIMPORTIMPORT from filter 		import filt_btwl
	pass#IMPORTIMPORTIMPORT import types
	pass#IMPORTIMPORTIMPORT import os
	if os.path.exists(indir) is False: sparx_global_def.ERROR("Input directory doesn't exist","copyfromtif",1)
	else                             : flist          = os.listdir(indir)
	if(type(outdir)          is bytes):
		if os.path.exists(outdir) :   sparx_global_def.ERROR("Output directory exists, please change the name and restart the program","copyfromtif",1)
		os.mkdir(outdir)	
		pass#IMPORTIMPORTIMPORT import global_def
		sparx_global_def.LOGFILE =  os.path.join(outdir, sparx_global_def.LOGFILE)
	else: 	
		outdir = ("micrographs")# default output directory
		os.mkdir(outdir)
		pass#IMPORTIMPORTIMPORT import global_def
		sparx_global_def.LOGFILE =  os.path.join(outdir, sparx_global_def.LOGFILE)
		
	gridding       = False
	Pixel_size_raw = scan_step/magnification
	if Pixel_size == 0 : Pixel_size = Pixel_size_raw
	if Pixel_size <  0 : 
		scaling_ratio  = - int(Pixel_size)
		gridding       = False
		Pixel_size     = Pixel_size_raw*scaling_ratio
	else               : scaling_ratio   = Pixel_size/Pixel_size_raw	
	e              = EMAN2_cppwrap.EMData()
	e1             = EMAN2_cppwrap.EMData()
	X19            = 2**16-1.0
	for i, v in enumerate(flist):
		tifname                 = os.path.join(indir, v)
		(rawname, rawextension) = os.path.splitext(v)
		if(rawextension == "."+input_extension):
			print("The raw file under processing is", tifname)
			e = sparx_utilities.get_image(tifname)
			if(film_or_CCD == "f"):
				e  -= 1.
				e  /= X19
				e1  = e.log10()
				if(scanner_param_a == 1):
					e1  *= -1.883	# scanned by the scanner in Pawel's lab
					e1  += 0.06698
				elif(scanner_param_a == 2):
					e1  *= -1.6090  # scanned by the scanner in Dowhan's lab
					e1  += 0.012
				else:
					e1  *= scanner_param_a # Check the local scanner
					e1  += scanner_param_b
			else:   e1   =e*contrast_invert
			#if  gridding : e1    = resample(e, scaling_ratio, 1) # resample will pad image to four times 
			#else         : e1    = image_decimate(e, scaling_ratio, 1)
			f_micrograph         = "micrograph_"+rawname+"."+ output_extension
			f_micname            = os.path.join(outdir, f_micrograph)
			if output_extension == "spi": sparx_utilities.drop_image(e1,f_micname,"s")
			else:   e1.write_image(f_micname)

def copyfromtif_MPI(indir, outdir=None, input_extension="tif", film_or_CCD="f", output_extension="hdf", contrast_invert=1, Pixel_size=1, scanner_param_a=1,scanner_param_b=1, scan_step=63.5, magnification=40):
	"""
		purpose: convert raw image into SPIDER format images
		1. film_or_CCD =f (scanned film, density will be converted into OD )or c (CCD)
		2. contrast_invert =1 or -1  invert contrast or not 
		3. The default vale of input_extension is tiff.
		4. Pixel_size denotes the final converted image pixel size. When it set as 
		   negative, the program will switch back to image_decimate reduce image size
		   integer times. 
	"""
	pass#IMPORTIMPORTIMPORT from utilities 		import get_image, drop_image
	pass#IMPORTIMPORTIMPORT from fundamentals 	import smallprime, window2d, resample, image_decimate
	pass#IMPORTIMPORTIMPORT from filter 		import filt_btwl
	pass#IMPORTIMPORTIMPORT from random     	import randint
	pass#IMPORTIMPORTIMPORT import types
	pass#IMPORTIMPORTIMPORT import os
	pass#IMPORTIMPORTIMPORT import sys
	pass#IMPORTIMPORTIMPORT from mpi 	    import mpi_comm_size, mpi_comm_rank, MPI_COMM_WORLD
	pass#IMPORTIMPORTIMPORT from mpi 	    import mpi_barrier, mpi_bcast
	pass#IMPORTIMPORTIMPORT from mpi 	    import MPI_INT

	if os.path.exists(indir) is False: sparx_global_def.ERROR("Input directory doesn't exist","copyfromtif_MPI",1,myid)
	else                             : flist = os.listdir(indir)
	if(type(outdir)          is bytes):
		if os.path.exists(outdir): sparx_global_def.ERROR("Output directory exists, please change the name and restart the program","copyfromtif_MPI",1,myid)
		os.mkdir(outdir)	
		pass#IMPORTIMPORTIMPORT import global_def
		sparx_global_def.LOGFILE =  os.path.join(outdir, sparx_global_def.LOGFILE)
	else:
		os.system(" rm -rf micrograph ")
		outdir ="micrograph"# default output directory
		os.mkdir(outdir)	
		pass#IMPORTIMPORTIMPORT import global_def
		sparx_global_def.LOGFILE =  os.path.join(outdir, sparx_global_def.LOGFILE)
	gridding       = True
	Pixel_size_raw = scan_step/magnification
	if Pixel_size == 0 : Pixel_size = Pixel_size_raw
	if Pixel_size <  0 : 
		scaling_ratio  = - int(Pixel_size)
		gridding       = False
		Pixel_size     = Pixel_size_raw*scaling_ratio
	else               : scaling_ratio   = Pixel_size/Pixel_size_raw
	#	
	nima           = 0
	mic_name_list  = []
	for i, v in enumerate(flist):
		micname  		= os.path.join(indir,v)
		(filename, filextension)= os.path.splitext(v  )
		if(filextension == "."+input_extension):
			mic_name_list.append(micname)
			nima += 1
	if nima < 1:    sparx_global_def.ERROR("No micrograph is found, check either directory or prefix of micrographs is correctly given","copyfromtif_MPI",1,myid)
       
	number_of_proc = mpi.mpi_comm_size(mpi.MPI_COMM_WORLD)
	myid	      = mpi.mpi_comm_rank(mpi.MPI_COMM_WORLD)
	#  chose a random node as a main one...
	main_node      = 0
	if(myid == 0): main_node = random.randint(0,number_of_proc-1)
	main_node      = mpi.mpi_bcast(main_node, 1, mpi.MPI_INT, 0, mpi.MPI_COMM_WORLD)

	#nimage_per_node = nima/number_of_proc
	#image_start     = myid * nimage_per_node
	#if(myid == number_of_proc-1):  image_end = nima
	#else:			      image_end = image_start + nimage_per_node
	image_start, image_end = MPI_start_end(nima, number_of_proc, myid)

	e              = EMAN2_cppwrap.EMData()
	e1             = EMAN2_cppwrap.EMData()
	X19            = 2**16-1.0
	#
	for i in range(image_start,image_end):
		filename               = mic_name_list[i]
		(tifname, filextension)= os.path.splitext(filename)
		print('%-30s%-30s'%("micrographs under proccessing : ",filename[len(indir)+1:]))
		e = sparx_utilities.get_image(filename)
		if(film_or_CCD == "f"):
			e  -= 1.
			e  /= X19
			e1  = e.log10()
			if(scanner_param_a == 1):
				e1  *= -1.883	# scanned by the scanner in Pawel's lab
				e1  += 0.06698
			elif(scanner_param_a == 2):
				e1  *= -1.6090  # scanned by the scanner in Dowhan's lab
				e1  += 0.012
			else:
				e1  *= scanner_param_a # Check the local scanner
				e1  += scanner_param_b
		e                    = e1*contrast_invert
		if  gridding : e1    = sparx_fundamentals.resample(e, scaling_ratio) # resample will pad image two times 
		else         : e1    = sparx_fundamentals.image_decimate(e, scaling_ratio, 1)
		f_micrograph         = "micrograph_"+tifname[len(indir)+1:]+"."+ output_extension
		f_micname            = os.path.join(outdir, f_micrograph)
		e1.write_image(f_micname)

# reworked cpy is able to process lists instead of single files. ins_list can
#    be either single file names like "test.hdf" or "bdb:image1", but also lists
#    of these names. note that mixed lists (containing both raw file names and
#    db objects) also work.

def dele_flist(flist):
	""" 
		delete a series files listed in a document file
	"""
	delist=[]
	inf = open(flist, "r")
	strg=inf.readline()
	while (len(strg)>0):
		sh_com="rm -f"+strg
		print(sh_com)
		delist.append(sh_com)
		strg=inf.readline()
	for i in range(len(delist)):  os.system(delist[i])

def defocus_calc(roodir, method, writetodoc="w", Pixel_size=1, voltage=120, Cs=1, amp_contrast=.1, round_off=100, dz_max=50000., frequency_low=30, frequency_high=5, polynomial_rank_baseline=5, polynomial_rank_envelope=5, prefix="roo", format="spider", skip_comment="#", micdir = "no", print_screen="no"):	
	pass#IMPORTIMPORTIMPORT from morphology import defocus_get_slow, defocus_get_fast
	if( method == "s"): 	sparx_morphology.defocus_get_slow(roodir, writetodoc, Pixel_size, voltage, Cs, amp_contrast, round_off, dz_max, frequency_low, frequency_high, prefix, format, skip_comment, micdir, print_screen)
	else: 			sparx_morphology.defocus_get_fast(roodir, writetodoc, Pixel_size, voltage, Cs, amp_contrast, round_off, dz_max, frequency_low, frequency_high, polynomial_rank_baseline,polynomial_rank_envelope, prefix, format, skip_comment,micdir, print_screen)

'''
def iso_kmeans(images, out_dir, parameter, K=None, mask=None, init_method="Random"):
	pass#IMPORTIMPORTIMPORT from statistics import init_Kmeans,Kmeans_step,kmeans_ave_var,iso_kmeans_rm_cluster,iso_kmeans_split,iso_kmeans_merge
	pass#IMPORTIMPORTIMPORT import os
	
	e=EMData()
	if(type(images) == str):
		flag = 1
		N = EMUtil.get_image_count(images)
		e.read_image(images,0)
	else:
		if(type(images) == list):
			flag = 0
			N = len(images)
			e = images[0].copy()
		else:
			return 'Invalid Input file format --- The input images can only be a stack file or a list of images'

	nx = e.get_xsize()
	ny = e.get_ysize()
	nz = e.get_zsize()
	size = nx*ny*nz
	if(mask == None):
		mask = model_blank(nx, ny, nz, 1.0)

	class Cluster:
		def __init__(self,a,n,C,SSE,V):
			self.a   = a
			self.n   = n
			self.C   = C
			self.SSE = SSE
			self.V   = V

	m = "%s" % (parameter)
	parameter *= size

	out  = open(out_dir+"/Isodata_kmeans_chart_"+m+".txt",'w')

	[assign,TE,Cls] = init_Kmeans(images, N, K, e, flag, mask, init_method)

	[assign,TE,Cls] = Kmeans_step(images,N,len(Cls),assign,TE,Cls,size,mask,flag)

	out.write("%s\t%d\n" % ("After K means the Total Error is",TE))

	for k in xrange(len(Cls)):
		out.write("\n%s\t%d\t%s\t%g\t%s\t%d\n" % ("Cluster",k,"variance = ",Cls[k].V/size,"No of Object = ",Cls[k].n))
		out.write("\n%s\t%d\t\t%s\n" % ("Cluster",k,"assignment"))
		Cls[k].a.sort()
		for kk in xrange(len(Cls[k].a)):
			out.write("\t%d" % (Cls[k].a[kk]))
			if(float(kk+1)/10==int(kk+1)/10):
				out.write("\n")
		out.write("\n")

	cnt = 0
	change = True
	while(change):

		print " ITERATION #",cnt,"  Total Error = ",TE
		change=False

		[assign,Cls] = iso_kmeans_rm_cluster(images,N,assign,Cls)

		[assign,TE,Cls] = iso_kmeans_split(images,N,assign,TE,Cls,Cluster,size,parameter,mask,init_method,flag)

		length_split = len(Cls)

		[assign,Cls] = iso_kmeans_rm_cluster(images,N,assign,Cls)

		[assign,TE,Cls] = iso_kmeans_merge(images,N,assign,TE,Cls,Cluster,size,parameter,mask,init_method,flag)

		length_merge =  len(Cls)
		
		if(length_merge != length_split):
			change = True
		cnt +=1

	out.write("%s\t%d\n" % ("After Isodata Clustering the Total Error is",TE))
		
	for k in xrange(len(Cls)):
		out.write("\n%s\t%d\t%s\t%g\t%s\t%d\n" % ("Cluster",k,"variance = ",Cls[k].V/size,"No of Object = ",Cls[k].n))
		out.write("\n%s\t%d\t\t%s\n" % ("Cluster",k,"assignment"))
		Cls[k].a.sort()
		for kk in xrange(len(Cls[k].a)):
			out.write("\t%d" % (Cls[k].a[kk]))
			if(float(kk+1)/10==int(kk+1)/10):
				out.write("\n")
		out.write("\n")
	
	os.system('rm -f '+out_dir+'/Isodata_kmeans_average_'+m+'.spi')
	os.system('rm -f '+out_dir+'/Isodata_kmeans_variance_'+m+'.spi')
	for k in xrange(len(Cls)):
		res = kmeans_ave_var(Cls[k].a,images,flag)
		res[0].write_image(out_dir+"/Isodata_kmeans_average_"+m+".spi",k)
		res[1].write_image(out_dir+"/Isodata_kmeans_variance_"+m+".spi",k)
'''

def pw2sp(indir, outdir = None, w =256, xo =50, yo = 50, xd = 0, yd = 0, r = 0, prefix_of_micrograph="micrograph", MPI=False):
	""" 
		Calculate power spectra of a list of micrographs in a given directory using Welch's periodogram
		The input options enable one to select an area in micrographs to calculate overlapped periodogram.
	"""
	if MPI:
		pw2sp_MPI(indir, outdir, w, xo, yo, xd, yd, r, prefix_of_micrograph)
		return

	pass#IMPORTIMPORTIMPORT from utilities    import get_image,drop_image, model_circle, info
	pass#IMPORTIMPORTIMPORT from fundamentals import welch_pw2, ro_textfile
	pass#IMPORTIMPORTIMPORT import sys
	pass#IMPORTIMPORTIMPORT import os
	pass#IMPORTIMPORTIMPORT import types
	if os.path.exists(indir) is False: sparx_global_def.ERROR("Input directory doesn't exist","pw2sp",1)
	else				 : flist    = os.listdir(indir)
	if(type(outdir)          is bytes):
		if os.path.exists(outdir) is True: sparx_global_def.ERROR("Output directory exists, please change the name and restart the program","pw2sp",1)
		os.mkdir(outdir)	
		pass#IMPORTIMPORTIMPORT import global_def
		sparx_global_def.LOGFILE =  os.path.join(outdir, sparx_global_def.LOGFILE)
	else: 	os.system("mkdir power")
	mask     = sparx_utilities.model_circle(int(r), int(w), int(w), nz=1)
	mask    -= 1
	mask    *= -1
	pw2_mask = EMAN2_cppwrap.EMData()
	ncount   = 0
	for i, v in enumerate(flist):
		micname=os.path.join(indir,v)
		(filename, filextension) = os.path.splitext(v)
		if(filename[0:len(prefix_of_micrograph)] == prefix_of_micrograph):
			print("micrographs :",filename)
			ncount   += 1
			roofile   = "roo"+filename[len(prefix_of_micrograph):]+".txt"
			pw2file   = "pw2"+filename[len(prefix_of_micrograph):]+filextension
			e         = sparx_utilities.get_image(micname)
			pw2       = sparx_fundamentals.welch_pw2(e,int(w),int(xo),int(yo),int(xd),int(yd))
			pw2_mask  = pw2*mask
			pw2name   = os.path.join(outdir,pw2file)
			sparx_utilities.drop_image(pw2_mask, pw2name)
			rotxtname = os.path.join(outdir,roofile)
			sparx_fundamentals.ro_textfile(pw2, rotxtname)
	if ncount < 1: 	sparx_global_def.ERROR("No micrograph is found, check either directory or prefix of micrographs is correctly given","pw2sp",1)

def pw2sp_MPI(indir, outdir, w =256, xo =50, yo = 50, xd = 0, yd = 0, r = 0, prefix_of_micrograph="micrograph"):
	""" 
		Calculate power spectra of a list of micrographs in a given directory using Welch's periodogram
		The input options enable one to select an area in micrographs to calculate overlapped periodogram.
	"""
	pass#IMPORTIMPORTIMPORT from utilities    	import get_image,drop_image, model_circle, info
	pass#IMPORTIMPORTIMPORT from fundamentals 	import welch_pw2, ro_textfile
	pass#IMPORTIMPORTIMPORT from random     	import randint
	pass#IMPORTIMPORTIMPORT import sys
	pass#IMPORTIMPORTIMPORT import os
	pass#IMPORTIMPORTIMPORT import types
	pass#IMPORTIMPORTIMPORT from mpi 	    import mpi_comm_size, mpi_comm_rank, MPI_COMM_WORLD
	pass#IMPORTIMPORTIMPORT from mpi 	    import mpi_barrier, mpi_bcast
	pass#IMPORTIMPORTIMPORT from mpi 	    import MPI_INT

	number_of_proc = mpi.mpi_comm_size(mpi.MPI_COMM_WORLD)
	myid           = mpi.mpi_comm_rank(mpi.MPI_COMM_WORLD)
	#  chose a random node as a main one...
	main_node      = 0
	if(myid == 0): main_node = random.randint(0,number_of_proc-1)
	main_node      = mpi.mpi_bcast(main_node, 1, mpi.MPI_INT, 0, mpi.MPI_COMM_WORLD)	
	
	if os.path.exists(outdir): sparx_global_def.ERROR("Output directory exists, please change the name and restart the program","pw2sp_MPI ",1,myid)
	mpi.mpi_barrier(mpi.MPI_COMM_WORLD)

	if myid == int(main_node):  # only main node do cleaning & creating jobs
		os.mkdir(outdir)
		pass#IMPORTIMPORTIMPORT import global_def
		sparx_global_def.LOGFILE =  os.path.join(outdir, sparx_global_def.LOGFILE)
		sparx_utilities.print_begin_msg("pw2sp_MPI")
	mpi.mpi_barrier(mpi.MPI_COMM_WORLD)
	
	# get the total micrograph number 
	if os.path.exists(indir) is False: sparx_global_def.ERROR("Input directory doesn't exist","pw2sp_MPI",1,myid)
	else:	                           flist = os.listdir(indir)
	nima          = 0
	mic_name_list = []
	for i, v in enumerate(flist):
		micname                  = os.path.join(indir,v)
		(filename, filextension) = os.path.splitext(v)
		if(filename[0:len(prefix_of_micrograph)] == prefix_of_micrograph):
			mic_name_list.append(micname)
			nima += 1
	if nima < 1: 	sparx_global_def.ERROR("No micrograph is found, check spelling of either directory or micrographs","pw2sp_MPI",1,myid)	

	#if nima < number_of_proc: nimage_per_node = 1
	#else                    : nimage_per_node = nima/number_of_proc
	#image_start     = myid *  nimage_per_node
	#if(myid == number_of_proc-1):  image_end = nima
	#else:                          image_end = image_start + nimage_per_node				
	image_start, image_end = MPI_start_end(nima, number_of_proc, myid)
	if image_start == image_end: 
		if image_end == nima:	image_start -= 1
		else: image_end += 1

	mask     =  sparx_utilities.model_circle(int(r), int(w), int(w), nz=1)
	mask    -=  1
	mask    *= -1
	pw2_mask =  EMAN2_cppwrap.EMData() 	
	for i in range(image_start,image_end):
		filename=mic_name_list[i] 
		print('%-15s%-30s'%("micrographs : ",filename))
		(f_nam, filextension) = os.path.splitext(filename)
		roofile   = "roo_"+f_nam[len(prefix_of_micrograph)+len(indir)+2:]+".txt"
		pw2file   = "pw2_"+f_nam[len(prefix_of_micrograph)+len(indir)+2:]+filextension
		e         = sparx_utilities.get_image(filename)
		pw2       = sparx_fundamentals.welch_pw2(e,int(w),int(xo),int(yo),int(xd),int(yd))
		pw2_mask  = pw2*mask
		pw2name   = os.path.join(outdir,pw2file)
		sparx_utilities.drop_image(pw2_mask, pw2name)
		rotxtname = os.path.join(outdir,roofile)
		sparx_fundamentals.ro_textfile(pw2, rotxtname)

def ra_cef(indir, noise, outdir, prf, num):
	"""
		Apply ramp and cefit to detected particles
		1. Fits a least-squares plane to the picture, 
		   and subtracts the plane from the picture.(ramp) 
		2. Fit the histogram of the input image under 
		   mask with the reference image (ce fit)
	"""
	flist  = os.listdir(indir)
	e      = EMAN2_cppwrap.EMData()
	e_n    = sparx_utilities.get_image(noise)
	nx     = e_n.get_xsize()
	ny     = e_n.get_ysize()
	radius = nx//2-1
	mask   = sparx_utilities.model_circle(radius, nx, ny)
	for i, v in enumerate(flist):
		(filename,filextension) = os.path.splitext(v)
		if(filename[0:4] == "ptl_"):
			infile  = os.path.join(indir, v)
			outfile = os.path.join(outdir,prf+v)
			for j in range(num):
				e.read_image(infile, j)
				ra=sparx_fundamentals.ramp(e)
				e=sparx_utilities.ce_fit(ra, e_n, mask)
				e.write_image(outfile, j)

def ali_vol_2(vol, refv, ang_scale, shift_scale, radius=None, discrepancy = "ccc"):
	#rotation and shift
	pass#IMPORTIMPORTIMPORT from alignment    import ali_vol_func
	pass#IMPORTIMPORTIMPORT from utilities    import get_im, model_circle
	pass#IMPORTIMPORTIMPORT from utilities    import amoeba
	pass#IMPORTIMPORTIMPORT from fundamentals import rot_shift3D

	nx = refv.get_xsize()
	ny = refv.get_ysize()
	nz = refv.get_zsize()
	if(radius != None):   mask = sparx_utilities.model_circle(radius, nx, ny, nz)
	else:                 mask = sparx_utilities.model_circle(float(min(nx, ny, nz)//2-2), nx, ny, nz)

	scale = [ang_scale, ang_scale, ang_scale, shift_scale, shift_scale, shift_scale]
	data  = [vol, refv, mask]
	new_params = [0.0]*6
	new_params = sparx_utilities.amoeba(new_params, scale, sparx_alignment.ali_vol_func, 1.e-4, 1.e-4, 500, data)
	vol = sparx_fundamentals.rot_shift3D(vol, new_params[0][0], new_params[0][1], new_params[0][2], new_params[0][3], new_params[0][4], new_params[0][5],1.0)
	return vol

def ali_vol_3(vol, refv, ang_scale, shift_scale, radius=None, discrepancy = "ccc", mask=None):
	#rotation and shift
	pass#IMPORTIMPORTIMPORT from alignment    import ali_vol_func
	pass#IMPORTIMPORTIMPORT from utilities    import model_circle, amoeba

	nx = refv.get_xsize()
	ny = refv.get_ysize()
	nz = refv.get_zsize()
	if mask is None:
		if(radius != None): mask = sparx_utilities.model_circle(radius, nx, ny, nz)
		else:               mask = sparx_utilities.model_circle(float(min(nx, ny, nz)//2-2), nx, ny, nz)

	scale = [ang_scale, ang_scale, ang_scale, shift_scale, shift_scale, shift_scale]
	data=[vol, refv, mask, discrepancy]
	new_params = [0.0]*6
	opt_params,funval,niter = sparx_utilities.amoeba(new_params, scale, sparx_alignment.ali_vol_func, 1.e-4, 1.e-4, 500, data)
	return opt_params

def ali_vol_n(vol, refv, ang_scale, shift_scale, radius=None, discrepancy="ccc", rsdec=1):
	"""
		Name
			sxali_vol_n - align a 3D structure with respect of a 3D reference structure.
				Like sxali_vol, but bypassing the composition of transformations.
		Input
			aligned_volume.hdf: 3D structure to be aligned.
		reference_volume.hdf
			3D reference structure.
		ang_scale
			correct angles are expected to be within +/-ang_scale of the values preset in the header of the structure to be aligned
		shift_scale
			correct shifts are expected to be within +/-shift_scale of the values preset in the header of the structure to be aligned
		mag_scale
			correct magnification is expected to be within +/-mag_scale of the value preset in the header of the structure to be aligned
		radius
			radius of a spherical mask centered at nx/2, ny/2, nz/2
		rsdec
			(int) if given and >1, the map, after being transformed, is convolved with a gaussian kernel, and then decimated rsdec times.
		Note - there are no defaults for three scale parameters. At least one has to appear.
	"""


	#rotation and shift
	pass#IMPORTIMPORTIMPORT from alignment    import ali_vol_func
	pass#IMPORTIMPORTIMPORT from utilities    import get_image, model_circle
	pass#IMPORTIMPORTIMPORT from utilities    import amoeba, get_params3D, set_params3D
	pass#IMPORTIMPORTIMPORT from utilities    import get_arb_params, set_arb_params
	
	ref = sparx_utilities.get_image(refv)
	nx = ref.get_xsize()
	ny = ref.get_ysize()
	nz = ref.get_zsize()
	if(radius != None):    mask = sparx_utilities.model_circle(radius, nx, ny, nz)
	else:                  mask = sparx_utilities.model_circle(float(min(nx, ny, nz)//2-2), nx, ny, nz)

	names_params = ["phi", "theta", "psi", "s3x", "s3y", "s3z"]

	e = sparx_utilities.get_image(vol)
	params =  sparx_utilities.get_arb_params(e, names_params)
	data = [e, ref, mask, None, discrepancy, rsdec]

	new_params = sparx_utilities.amoeba(params, [ang_scale, ang_scale, ang_scale, shift_scale, shift_scale, shift_scale], sparx_alignment.ali_vol_func, 1.e-5, 1.e-4, 500, data)

	sparx_utilities.set_arb_params(e, [new_params[0][0], new_params[0][1], new_params[0][2], new_params[0][3], new_params[0][4], new_params[0][5]], names_params)
	if type(vol)==type(""):
		pass#IMPORTIMPORTIMPORT from utilities import write_headers
		sparx_utilities.write_headers( vol, [e], [0])
	else:
		return e

def ali_vol_grid(vol, params, refv, ang_scale, shift_scale, radius=None, discrepancy="dot", kb=None, wrap=False):
	"""
		Name
			sxali_vol_grid - align a 3D structure with respect of a 3D reference structure.
				Like ali_vol_n, but using gridding transformations.

		Arguments:
		vol
			3D structure to be aligned
		params
			starting parameters to be applied to vol
		refv
			3D reference structure
		ang_scale
			correct angles are expected to be within +/-ang_scale of the values preset in the header of the structure to be aligned
		shift_scale
			correct shifts are expected to be within +/-shift_scale of the values preset in the header of the structure to be aligned
		radius
			radius of a spherical mask centered at nx/2, ny/2, nz/2
		discrepancy
			discrepancy (or similarity) measure
		kb
			if given, then (vol,kb) must be the output from prepi3D. If not given, then prepi3D is called here.
		wrap
			if True, use wraparound pixels during translations
	"""


	#rotation and shift
	pass#IMPORTIMPORTIMPORT from alignment    import ali_vol_func_grid
	pass#IMPORTIMPORTIMPORT from utilities    import get_image, model_circle
	pass#IMPORTIMPORTIMPORT from utilities    import amoeba, get_params3D, set_params3D
	pass#IMPORTIMPORTIMPORT from utilities    import get_arb_params, set_arb_params
	
	ref = sparx_utilities.get_image(refv)
	nx = ref.get_xsize()
	ny = ref.get_ysize()
	nz = ref.get_zsize()
	if(radius != None):    mask = sparx_utilities.model_circle(radius, nx, ny, nz)
	else:                  mask = sparx_utilities.model_circle(float(min(nx, ny, nz)//2-2), nx, ny, nz)

	e = sparx_utilities.get_image(vol)
	
	if kb==None:
		ee, kb = sparx_fundamentals.prepi3D(e)
		data = [ee, ref, mask, None, discrepancy, kb, wrap]
	else:
		data = [e, ref, mask, None, discrepancy, kb, wrap]
	
	new_params = sparx_utilities.amoeba(params, [ang_scale, ang_scale, ang_scale, shift_scale, shift_scale, shift_scale], sparx_alignment.ali_vol_func_grid, 1.e-5, 1.e-4, 500, data)

	return [new_params[0][0], new_params[0][1], new_params[0][2], new_params[0][3], new_params[0][4], new_params[0][5]]


def ali_vol_M(vol, refv, ang_scale, shift_scale, mask=None, discrepancy = "ccc"):
	"""
		Name
			sxali_vol_m - align a 3D structure with respect of a 3D reference structure.
				Like sxali_vol_n, but taking a specified mask instead of a radius.
		Input
			aligned_volume.hdf: 3D structure to be aligned.
		reference_volume.hdf
			3D reference structure.
		ang_scale
			correct angles are expected to be within +/-ang_scale of the values preset in the header of the structure to be aligned
		shift_scale
			correct shifts are expected to be within +/-shift_scale of the values preset in the header of the structure to be aligned
		mag_scale
			correct magnification is expected to be within +/-mag_scale of the value preset in the header of the structure to be aligned
		radius
			radius of a spherical mask centered at nx/2, ny/2, nz/2
		Note - there are no defaults for three scale parameters. At least one has to appear.
	"""


	#rotation and shift
	pass#IMPORTIMPORTIMPORT from alignment    import ali_vol_func
	pass#IMPORTIMPORTIMPORT from utilities    import get_image, model_circle
	pass#IMPORTIMPORTIMPORT from utilities    import amoeba, get_params3D, set_params3D
	
	ref = sparx_utilities.get_image(refv)
	nx = ref.get_xsize()
	ny = ref.get_ysize()
	nz = ref.get_zsize()

	e = sparx_utilities.get_image(vol)

	if(mask == None):
		mask = sparx_utilities.model_circle(float(min(nx, ny, nz)//2-2), nx, ny, nz)
		minval = None
	elif(mask == "tight"):
		minval = 0.1*EMAN2_cppwrap.Util.infomask(e,None,True)[0]   # threshold for binarizing
	elif(isinstance(mask, (int))):
		minval = -mask    # negative denotes mask is a moving sphere
		mask = sparx_utilities.model_circle(-minval, nx, ny, nz)
	else:
		minval = None

	params = sparx_utilities.get_params3D(e)
	params = [params[i] for i in range(6)]
	data = [e, ref, mask, minval, discrepancy]
	
	maxiter = 500
	new_params = sparx_utilities.amoeba(params, [ang_scale, ang_scale, ang_scale, shift_scale, shift_scale, shift_scale], sparx_alignment.ali_vol_func, 1.e-5, 1.e-4, maxiter, data)

	if new_params[2]>=maxiter:
		print("Warning: amoeba reached the max number of iterations allowed.")

	sparx_utilities.set_params3D(e, [new_params[0][0], new_params[0][1], new_params[0][2], new_params[0][3], new_params[0][4], new_params[0][5], 0, 1.0])
	if type(vol)==type(""):
		pass#IMPORTIMPORTIMPORT from utilities import write_headers
		sparx_utilities.write_headers( vol, [e], [0])
	else:
		return e


def ali_vol_nopsi(vol, refv, ang_scale, shift_scale, radius=None, discrepancy = "ccc"):
	"""
		Name
			sxali_vol_nopsi - align a 3D structure with respect of a 3D reference structure,
				like sxali_vol, but keeping psi fixed at 0.
		Input
			aligned_volume.hdf: 3D structure to be aligned.
		reference_volume.hdf
			3D reference structure.
		ang_scale
			correct angles are expected to be within +/-ang_scale of the values preset in the header of the structure to be aligned
		shift_scale
			correct shifts are expected to be within +/-shift_scale of the values preset in the header of the structure to be aligned
		mag_scale
			correct magnification is expected to be within +/-mag_scale of the value preset in the header of the structure to be aligned
		radius
			radius of a spherical mask centered at nx/2, ny/2, nz/2
		Note - there are no defaults for three scale parameters. At least one has to appear.
	"""


	#rotation and shift
	pass#IMPORTIMPORTIMPORT from alignment    import ali_vol_func_nopsi
	pass#IMPORTIMPORTIMPORT from utilities    import get_image, model_circle
	pass#IMPORTIMPORTIMPORT from utilities    import amoeba, get_params3D, set_params3D
	pass#IMPORTIMPORTIMPORT from utilities    import get_arb_params, set_arb_params
	
	ref = sparx_utilities.get_image(refv)
	nx = ref.get_xsize()
	ny = ref.get_ysize()
	nz = ref.get_zsize()
	if(radius != None):    mask = sparx_utilities.model_circle(radius, nx, ny, nz)
	else:                  mask = sparx_utilities.model_circle(float(min(nx, ny, nz)//2-2), nx, ny, nz)

	names_params = ["phi", "theta", "s3x", "s3y", "s3z"]

	e = sparx_utilities.get_image(vol)
	params = sparx_utilities.get_arb_params(e, names_params)
	data = [e, ref, mask, None, discrepancy]
	
	new_params = sparx_utilities.amoeba(params, [ang_scale, ang_scale, shift_scale, shift_scale, shift_scale], sparx_alignment.ali_vol_func_nopsi, 1.e-5, 1.e-4, 500, data)

	sparx_utilities.set_arb_params(e, [new_params[0][0], new_params[0][1], new_params[0][2], new_params[0][3], new_params[0][4]], names_params)
	if type(vol)==type(""):
		pass#IMPORTIMPORTIMPORT from utilities import write_headers
		sparx_utilities.write_headers( vol, [e], [0])
	else:
		return e


def ali_vol_rotate(vol, refv, ang_scale, radius=None, discrepancy = "ccc"):
	#rotation 
	pass#IMPORTIMPORTIMPORT from alignment    import ali_vol_func_rotate
	pass#IMPORTIMPORTIMPORT from utilities    import get_image, model_circle, get_params3D, set_params3D
	pass#IMPORTIMPORTIMPORT from utilities    import amoeba, compose_transform3
	pass#IMPORTIMPORTIMPORT from fundamentals import rot_shift3D

	ref = sparx_utilities.get_image(refv)
	nx = ref.get_xsize()
	ny = ref.get_ysize()
	nz = ref.get_zsize()
	if(radius != None):    mask = sparx_utilities.model_circle(radius, nx, ny, nz)
	else:                  mask = sparx_utilities.model_circle(float(min(nx, ny, nz)//2-2), nx, ny, nz)

	#names_params = ["phi", "theta", "psi", "s3x", "s3y", "s3z", "scale"]
	params = sparx_utilities.get_params3D(ref)
	#print  " params of the reference volume",params
	ref = sparx_fundamentals.rot_shift3D(ref, params[0], params[1], params[2], params[3], params[4], params[5], params[7])

	e = sparx_utilities.get_image(vol)
	params = sparx_utilities.get_params3D(e)
	#e = rot_shift3D(e, params[0], params[1], params[2], params[3], params[4], params[5], params[7])
	#print  " input params ", params
	data = [e, ref, mask, params, discrepancy]
	new_params = [0.0, 0.0, 0.0]
	new_params = sparx_utilities.amoeba(new_params, [ang_scale, ang_scale, ang_scale], sparx_alignment.ali_vol_func_rotate, 1.e-4, 1.e-4, 500, data)
	cphi, ctheta, cpsi, cs2x, cs2y, cs2z, cscale= sparx_utilities.compose_transform3(params[0], params[1], params[2], params[3], params[4], params[5], params[7], new_params[0][0], new_params[0][1], new_params[0][2],0.0,0.0,0.0,1.0)
	#print  " new params ", cphi, ctheta, cpsi, cs2x, cs2y, cs2z, cscale, new_params[1]
	sparx_utilities.set_params3D(e, [cphi, ctheta, cpsi, cs2x, cs2y, cs2z, 0, cscale])
	if type(vol)==type(""):
		pass#IMPORTIMPORTIMPORT from utilities import write_headers
		sparx_utilities.write_headers( vol, [e], [0])
	else:
		return e

def ali_vol_shift(vol, refv, shift_scale, radius=None, discrepancy = "ccc"):
	# shift
	pass#IMPORTIMPORTIMPORT from alignment    import ali_vol_func_shift
	pass#IMPORTIMPORTIMPORT from utilities    import get_image, model_circle, get_params3D, set_params3D
	pass#IMPORTIMPORTIMPORT from utilities    import amoeba, compose_transform3
	pass#IMPORTIMPORTIMPORT from fundamentals import rot_shift3D

	ref = sparx_utilities.get_image(refv)
	nx = ref.get_xsize()
	ny = ref.get_ysize()
	nz = ref.get_zsize()
	if(radius != None):    mask = sparx_utilities.model_circle(radius, nx, ny, nz)
	else:                  mask = sparx_utilities.model_circle(float(min(nx, ny, nz)//2-2), nx, ny, nz)

	#names_params = ["phi", "theta", "psi", "s3x", "s3y", "s3z", "scale"]
	params = sparx_utilities.get_params3D(ref)
	#print  " params of the reference volume",params
	ref = sparx_fundamentals.rot_shift3D(ref, params[0], params[1], params[2], params[3], params[4], params[5], params[7])

	e = sparx_utilities.get_image(vol)
	params = sparx_utilities.get_params3D(e)
	#e = rot_shift3D(e, params[0], params[1], params[2], params[3], params[4], params[5], params[7])
	#print  " input params ",params
	data = [e, ref, mask, params, discrepancy]
	new_params = [0.0, 0.0, 0.0]
	new_params = sparx_utilities.amoeba(new_params, [shift_scale, shift_scale, shift_scale], sparx_alignment.ali_vol_func_shift, 1.e-4, 1.e-4, 500, data)
	cphi, ctheta, cpsi, cs3x, cs3y, cs3z, cscale= sparx_utilities.compose_transform3(params[0], params[1], params[2], params[3], params[4], params[5], params[7], 0.0,0.0,0.0, new_params[0][0], new_params[0][1], new_params[0][2],1.0)
	#print  " new params ", cphi, ctheta, cpsi, cs3x, cs3y, cs3z, cscale, new_params[1]
	sparx_utilities.set_params3D(e, [cphi, ctheta, cpsi, cs3x, cs3y, cs3z, 0, cscale])
	if type(vol)==type(""):
		pass#IMPORTIMPORTIMPORT from utilities import write_headers
		sparx_utilities.write_headers( vol, [e], [0])
	else:
		return e

def ali_vol_scale(vol, refv, ang_scale, shift_scale, mag_scale, radius=None, discrepancy = "ccc"):
	# rotation shift and scale
	pass#IMPORTIMPORTIMPORT from alignment    import ali_vol_func_scale
	pass#IMPORTIMPORTIMPORT from utilities    import get_image, model_circle, get_params3D, set_params3D
	pass#IMPORTIMPORTIMPORT from utilities    import amoeba, compose_transform3
	pass#IMPORTIMPORTIMPORT from fundamentals import rot_shift3D
	ref = sparx_utilities.get_image(refv)
	nx = ref.get_xsize()
	ny = ref.get_ysize()
	nz = ref.get_zsize()
	if(radius != None):    mask = sparx_utilities.model_circle(radius, nx, ny, nz)
	else:                  mask = sparx_utilities.model_circle(float(min(nx, ny, nz)//2-2), nx, ny, nz)

	#names_params = ["phi", "theta", "psi", "s3x", "s3y", "s3z", "scale"]
	params = sparx_utilities.get_params3D(ref)
	print(" params of the reference volume",params)
	ref = sparx_fundamentals.rot_shift3D(ref, params[0], params[1], params[2], params[3], params[4], params[5], params[7])

	e = sparx_utilities.get_image(vol)
	params = sparx_utilities.get_params3D(e)
	print(" input params ",params)
	data = [e, ref, mask, params, discrepancy]
	new_params = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
	new_params = sparx_utilities.amoeba(new_params, [ang_scale, ang_scale, ang_scale, shift_scale, shift_scale, shift_scale, mag_scale], sparx_alignment.ali_vol_func_scale, 1.e-4, 1.e-4, 500, data)
	cphi, ctheta, cpsi, cs2x, cs2y, cs2z, cscale= sparx_utilities.compose_transform3(params[0], params[1], params[2], params[3], params[4], params[5], params[7], new_params[0][0], new_params[0][1], new_params[0][2], new_params[0][3], new_params[0][4], new_params[0][5], new_params[0][6])
	print(" new params ", cphi, ctheta, cpsi, cs2x, cs2y, cs2z, cscale, new_params[1])
	sparx_utilities.set_params3D(e, [cphi, ctheta, cpsi, cs2x, cs2y, cs2z, 0, cscale])
	
	if type(vol)==type(""):
		pass#IMPORTIMPORTIMPORT from utilities import write_headers
		sparx_utilities.write_headers( vol, [e], [0])
	else:
		return e

def ali_vol_only_scale(vol, refv, mag_scale, radius=None, discrepancy = "ccc"):
	# scale
	pass#IMPORTIMPORTIMPORT from alignment    import ali_vol_func_only_scale
	pass#IMPORTIMPORTIMPORT from utilities    import get_image, model_circle, get_params3D, set_params3D
	pass#IMPORTIMPORTIMPORT from utilities    import amoeba, compose_transform3
	pass#IMPORTIMPORTIMPORT from fundamentals import rot_shift3D
	ref = sparx_utilities.get_image(refv)
	nx = ref.get_xsize()
	ny = ref.get_ysize()
	nz = ref.get_zsize()
	if(radius != None):    mask = sparx_utilities.model_circle(radius, nx, ny, nz)
	else:                  mask = sparx_utilities.model_circle(float(min(nx, ny, nz)//2-2), nx, ny, nz)


	#names_params = ["phi", "theta", "psi", "s3x", "s3y", "s3z", "scale"]
	params = sparx_utilities.get_params3D(ref)
	print(" params of the reference volume",params)
	ref = sparx_fundamentals.rot_shift3D(ref, params[0], params[1], params[2], params[3], params[4], params[5], params[7])

	e = sparx_utilities.get_image(vol)
	params = sparx_utilities.get_params3D(e)
	print(" input params ",params)
	data = [e, ref, mask, params, discrepancy]
	new_params = [1.0]
	new_params = sparx_utilities.amoeba(new_params, [mag_scale], sparx_alignment.ali_vol_func_only_scale, 1.e-4, 1.e-4, 500, data)
	cphi, ctheta, cpsi, cs2x, cs2y, cs2z, cscale= sparx_utilities.compose_transform3(params[0], params[1], params[2], params[3], params[4], params[5], params[7], 0.0,0.0,0.0,0.0,0.0,0.0, new_params[0][0])
	print(" new params ", cphi, ctheta, cpsi, cs2x, cs2y, cs2z, cscale, new_params[1])
	sparx_utilities.set_params3D(e, [cphi, ctheta, cpsi, cs2x, cs2y, cs2z, 0, cscale])
	
	if type(vol) == type(""):
		pass#IMPORTIMPORTIMPORT from utilities import write_headers
		sparx_utilities.write_headers( vol, [e], [0])
	else:
		return e

def rot_sym(infile, outfile, sym_gp="d4", \
			radius=None, phi=0, theta=0, psi=0, phirange=20, thetarange=20, psirange=20, ftolerance=1.e-4, xtolerance=1.e-4):

	pass#IMPORTIMPORTIMPORT from alignment     import find_symm
	pass#IMPORTIMPORTIMPORT from utilities     import drop_image, model_circle, sym_vol
	pass#IMPORTIMPORTIMPORT from fundamentals  import  rot_shift3D

	e=EMAN2_cppwrap.EMData()
	e.read_image(infile)
	mask = EMAN2_cppwrap.EMData()
	if radius == None: radius = e.get_xsize()/2.0
	mask  = sparx_utilities.model_circle(radius, e.get_xsize(), e.get_ysize(), e.get_zsize())
	scale = [phirange, thetarange, psirange]
	res = sparx_alignment.find_symm(e, mask, sym_gp, phi, theta, psi, scale, ftolerance, xtolerance)
	print("  RESULTS: ",res)

	sym = sparx_utilities.sym_vol(sparx_fundamentals.rot_shift3D(e, res[0][0], res[0][1], res[0][2] ), sym_gp)

	sparx_utilities.drop_image(sym, outfile)

def transform2d(stack_data, stack_data_ali, shift = False, ignore_mirror = False, method = "quadratic"):
# apply 2D alignment parameters stored in the header of the input stack file using gridding interpolation and create an output stack file
	pass#IMPORTIMPORTIMPORT from fundamentals   import rot_shift2D
	pass#IMPORTIMPORTIMPORT from utilities 	    import set_params2D, get_params2D, get_im
	pass#IMPORTIMPORTIMPORT import os
	if  shift:
		pass#IMPORTIMPORTIMPORT from utilities     import compose_transform2m
		pass#IMPORTIMPORTIMPORT from fundamentals  import fshift, mirror

	t = EMAN2_cppwrap.Transform({"type":"2D"})
	nima = EMAN2_cppwrap.EMUtil.get_image_count(stack_data)
	for im in range(nima):
		data = sparx_utilities.get_im(stack_data, im)
		al2d = sparx_utilities.get_params2D(data)
		if(shift):
			angb, sxb, syb, nm, ct = sparx_utilities.compose_transform2m(0.0, al2d[1], al2d[2], 0, 1.0, -al2d[0], 0.0, 0.0, al2d[3], 1.0)
			data = sparx_fundamentals.fshift(data, sxb, syb)
			if ignore_mirror: nm = 0
			if(nm == 1):  data = sparx_fundamentals.mirror(data)
		else:
			if ignore_mirror: al2d[3] = 0
			data = sparx_fundamentals.rot_shift2D(data, al2d[0], al2d[1], al2d[2], al2d[3], al2d[4], interpolation_method = method)
		data.set_attr("xform.align2d", t)
		data.write_image(stack_data_ali, im)

def recons3d_n(prj_stack, pid_list, vol_stack, CTF=False, snr=1.0, sign=1, npad=4, sym="c1", listfile = "", group = -1, verbose=0, MPI = False, xysize=-1, zsize = -1, smearstep = 0.0, upweighted = False, compensate = False, chunk_id = -1):
	if MPI:
		recons3d_n_MPI(prj_stack, pid_list, vol_stack, CTF, snr, 1, npad, sym, listfile, group, verbose, xysize, zsize, smearstep)
		##newrecons3d_n_MPI(prj_stack, pid_list, vol_stack, CTF, snr, 1, npad, sym, listfile, group, verbose,xysize, zsize)
		return

	pass#IMPORTIMPORTIMPORT from reconstruction import recons3d_4nn_ctf, recons3d_4nn
	pass#IMPORTIMPORTIMPORT from utilities import drop_image

	if(listfile):
		pass#IMPORTIMPORTIMPORT from utilities import read_text_file
		pid_list = sparx_utilities.read_text_file(listfile, 0)
		pid_list = list(map(int, pid_list))
	elif(group > -1):
		tmp_list = EMAN2_cppwrap.EMUtil.get_all_attributes(prj_stack, 'group')
		pid_list = []
		for i in range(len(tmp_list)):
			if(tmp_list[i] == group):  pid_list.append(i)
		del tmp_list

	if CTF: vol = sparx_reconstruction.recons3d_4nn_ctf(prj_stack, pid_list, snr, 1, sym, verbose, npad, xysize=xysize, zsize=zsize)
	else:   vol = sparx_reconstruction.recons3d_4nn(prj_stack,  pid_list, sym, npad, snr = snr, xysize=xysize, zsize = zsize)
	if(vol_stack[-3:] == "spi"):
		sparx_utilities.drop_image(vol, vol_stack, "s")
	else:
		sparx_utilities.drop_image(vol, vol_stack)

def recons3d_n_MPI(prj_stack, pid_list, vol_stack, CTF=False, snr=1.0, sign=1, npad=2, sym="c1", listfile="", group=-1, verbose=0, xysize=-1, zsize=-1, smearstep = 0.0):
	pass#IMPORTIMPORTIMPORT from reconstruction import recons3d_4nn_ctf_MPI, recons3d_4nn_MPI
	pass#IMPORTIMPORTIMPORT from utilities      import get_im, drop_image, bcast_number_to_all
	pass#IMPORTIMPORTIMPORT from string         import replace
	pass#IMPORTIMPORTIMPORT from time           import time
	pass#IMPORTIMPORTIMPORT from utilities      import iterImagesStack
	pass#IMPORTIMPORTIMPORT from mpi            import mpi_comm_size, mpi_comm_rank, mpi_bcast, MPI_INT, MPI_COMM_WORLD

	myid  = mpi.mpi_comm_rank(mpi.MPI_COMM_WORLD)
	nproc = mpi.mpi_comm_size(mpi.MPI_COMM_WORLD)
	time_start = time.time()

	if(myid == 0):
		if(listfile):
			pass#IMPORTIMPORTIMPORT from utilities import read_text_file
			pid_list = sparx_utilities.read_text_file(listfile, 0)
		elif(group > -1):
			tmp_list = EMAN2_cppwrap.EMUtil.get_all_attributes(prj_stack, 'group')
			pid_list = []
			for i in range(len(tmp_list)):
				if(tmp_list[i] == group):  pid_list.append(i)
			del tmp_list
		nima = len(pid_list)
	else:
		nima = 0

	nima = sparx_utilities.bcast_number_to_all(nima, source_node = 0)

	if(listfile or group > -1):
		if myid != 0:
			pid_list = [-1]*nima
		pid_list = mpi.mpi_bcast(pid_list, nima, mpi.MPI_INT, 0, mpi.MPI_COMM_WORLD)
		pid_list = list(map(int, pid_list))
	else:
		if(not pid_list):  pid_list = list(range(nima))

	if verbose==0:
		finfo = None
	else:
		infofile = "progress%04d.txt"%(myid+1)
		finfo = open( infofile, 'w' )

	image_start, image_end = MPI_start_end(nima, nproc, myid)

	prjlist = sparx_utilities.iterImagesStack(prj_stack, pid_list[image_start:image_end])
	del pid_list

	if CTF: vol = sparx_reconstruction.recons3d_4nn_ctf_MPI(myid, prjlist, snr, sign, sym, finfo, npad, xysize, zsize, smearstep = smearstep)
	else:	vol = sparx_reconstruction.recons3d_4nn_MPI(myid, prjlist, sym, finfo, snr, npad, xysize, zsize)
	if myid == 0 :
		if(vol_stack[-3:] == "spi"):
			sparx_utilities.drop_image(vol, vol_stack, "s")
		else:
			sparx_utilities.drop_image(vol, vol_stack)
		if not(finfo is None):
			finfo.write( "result written to " + vol_stack + "\n")
			finfo.write( "Total time: %10.3f\n" % (time.time()-time_start) )
			finfo.flush()

def recons3d_trl_MPI(prj_stack, pid_list, vol_stack, CTF, snr, sign, npad, sym, verbose = None, niter =10, compensate = False, target_window_size=-1):
	# unregularized reconstruction  flags reconstruct(Iunreg(), gridding_nr_iter,false, 1., dummy, dummy, dummy, dummy, 1., false, true, nr_threads, -1
	pass#IMPORTIMPORTIMPORT from reconstruction import recons3d_4nn_ctf_MPI, recons3d_4nn_MPI, recons3d_4nnf_MPI
	pass#IMPORTIMPORTIMPORT from utilities      import get_im, drop_image, bcast_number_to_all, write_text_file, read_text_file, info
	pass#IMPORTIMPORTIMPORT from string         import replace
	pass#IMPORTIMPORTIMPORT from time           import time
	pass#IMPORTIMPORTIMPORT from mpi            import mpi_comm_size, mpi_comm_rank, mpi_bcast, MPI_INT, MPI_COMM_WORLD, mpi_barrier
	pass#IMPORTIMPORTIMPORT from EMAN2          import Reconstructors
	pass#IMPORTIMPORTIMPORT from fundamentals   import fftip, fft
	myid       = mpi.mpi_comm_rank(mpi.MPI_COMM_WORLD)
	nproc      = mpi.mpi_comm_size(mpi.MPI_COMM_WORLD)
	mpi_comm   = mpi.MPI_COMM_WORLD
	time_start = time.time()
	nnxo       = 0	
	if(myid == 0):
		nnxo = sparx_utilities.get_im(prj_stack).get_xsize()
		print("  trilinear interpolation used in reconstruction")
		nima = len(pid_list)
	else:
		nima = 0
	nima = sparx_utilities.bcast_number_to_all(nima, source_node = 0)
	nnxo = sparx_utilities.bcast_number_to_all(nnxo, source_node = 0)
	if target_window_size ==-1: target_size = 2*nnxo+3
	else:                       target_size = target_window_size*2+3
	
	if verbose==0: finfo = None
	else:
		infofile = "progress%04d.txt"%(myid+1)
		finfo = open( infofile, 'w' )
	
	image_start, image_end = MPI_start_end(nima, nproc, myid)
	prjlist                = EMAN2_cppwrap.EMData.read_images(prj_stack, pid_list[image_start:image_end])
	
	#nnnx = ((prjlist[0].get_ysize())*2+3)	

	pass#IMPORTIMPORTIMPORT from utilities      import read_text_file, read_text_row, write_text_file, info, model_blank, get_im
	pass#IMPORTIMPORTIMPORT from fundamentals   import fft,fshift
	pass#IMPORTIMPORTIMPORT from reconstruction import insert_slices, insert_slices_pdf
	pass#IMPORTIMPORTIMPORT from utilities      import reduce_EMData_to_root, model_blank
	pass#IMPORTIMPORTIMPORT from filter         import filt_table
	# reconstruction step
	refvol = sparx_utilities.model_blank(target_size)
	refvol.set_attr("fudge", 1.0)
	if CTF: do_ctf = 1
	else:   do_ctf = 0
	if not (finfo is None): nimg = 0
	
	fftvol = EMAN2_cppwrap.EMData()
	weight = EMAN2_cppwrap.EMData()
	
	params = {"size":target_size, "npad":2, "snr":1.0, "sign":1, "symmetry":"c1", "refvol":refvol, "fftvol":fftvol, "weight":weight, "do_ctf": do_ctf}
	r      = EMAN2_cppwrap.Reconstructors.get( "nn4_ctfw", params )
	r.setup()
	m = [1.0]*target_size
	is_complex = prjlist[0].get_attr("is_complex")
	pass#IMPORTIMPORTIMPORT from filter		import filt_ctf
	for image in prjlist:
		if not is_complex: image = sparx_fundamentals.fft(image)
		if CTF: image =sparx_filter.filt_ctf(image, image.get_attr("ctf"))
		image.set_attr("padffted",1)
		image.set_attr("npad",1)
		image.set_attr("bckgnoise",m)
		r.insert_slice(image, image.get_attr_default("xform.projection", None), 1.0)
	if not (finfo is None): 
		finfo.write( "begin reduce\n" )
		finfo.flush()

	sparx_utilities.reduce_EMData_to_root(fftvol, myid, 0, comm=mpi_comm)
	sparx_utilities.reduce_EMData_to_root(weight, myid, 0, comm=mpi_comm)
	
	
	if not (finfo is None): 
		finfo.write( "after reduce\n" )
		finfo.flush()

	if myid == 0: dummy = r.finish(compensate)
	mpi.mpi_barrier(mpi_comm)

	if myid == 0 : # post-insertion operations, done only in main_node
		fftvol.set_attr("is_complex",1)
		# no regularization here.
		if( sym != "c1" ):
			fftvol    = fftvol.symfvol(sym, -1)
			weight    = weight.symfvol(sym, -1)  # symmetrize if not asymmetric
		maxr2 = ((target_window_size//-1)*2)**2 
		EMAN2_cppwrap.Util.iterefa(fftvol, weight, maxr2, target_window_size)
		pass#IMPORTIMPORTIMPORT from morphology   import cosinemask
		pass#IMPORTIMPORTIMPORT from fundamentals import fshift, fpol, fdecimate
		fftvol = sparx_fundamentals.fft(sparx_fundamentals.fshift(fftvol,target_window_size, target_window_size, target_window_size))
		fftvol = EMAN2_cppwrap.Util.window(fftvol, target_window_size, target_window_size, target_window_size)
		fftvol = sparx_morphology.cosinemask(fftvol, target_window_size//2-1,5, None)
		fftvol.div_sinc(1)
		fftvol.write_image(vol_stack)
		
def newsrecons3d_n_MPI(prj_stack, pid_list, vol_stack, CTF, snr, sign, npad, sym, listfile, group, verbose):
	pass#IMPORTIMPORTIMPORT from reconstruction import recons3d_4nn_ctf_MPI, recons3d_4nn_MPI, recons3d_4nnf_MPI
	pass#IMPORTIMPORTIMPORT from utilities      import get_im, drop_image, bcast_number_to_all, write_text_file, read_text_file, info
	pass#IMPORTIMPORTIMPORT from string         import replace
	pass#IMPORTIMPORTIMPORT from time           import time
	pass#IMPORTIMPORTIMPORT from mpi            import mpi_comm_size, mpi_comm_rank, mpi_bcast, MPI_INT, MPI_COMM_WORLD

	myid  = mpi.mpi_comm_rank(mpi.MPI_COMM_WORLD)
	nproc = mpi.mpi_comm_size(mpi.MPI_COMM_WORLD)
	time_start = time.time()
	if(myid == 0):
		print("  news ")
		if(listfile):
			pass#IMPORTIMPORTIMPORT from utilities import read_text_file
			pid_list = sparx_utilities.read_text_file(listfile, 0)
			pid_list = list(map(int, pid_list))
		elif(group > -1):
			tmp_list = EMAN2_cppwrap.EMUtil.get_all_attributes(prj_stack, 'group')
			pid_list = []
			for i in range(len(tmp_list)):
				if(tmp_list[i] == group):  pid_list.append(i)
			del tmp_list
		nima = len(pid_list)
	else:
		nima = 0
	nima = sparx_utilities.bcast_number_to_all(nima, source_node = 0)

	if(listfile or group > -1):
		if myid != 0:
			pid_list = [-1]*nima
		pid_list = mpi.mpi_bcast(pid_list, nima, mpi.MPI_INT, 0, mpi.MPI_COMM_WORLD)
		pid_list = list(map(int, pid_list))
	else:
		if(not pid_list):  pid_list = list(range(nima))

	if verbose==0:
		finfo = None
	else:
		infofile = "progress%04d.txt"%(myid+1)
		finfo = open( infofile, 'w' )
	"""Multiline Comment33"""
	"""Multiline Comment34"""
	pid_list = mpi.mpi_bcast(pid_list, nima, mpi.MPI_INT, 0, mpi.MPI_COMM_WORLD)
	pid_list = list(map(int, pid_list))
	
	image_start, image_end = MPI_start_end(nima, nproc, myid)
	prjlist = EMAN2_cppwrap.EMData.read_images(prj_stack, pid_list[image_start:image_end])

	#if myid == 0 :  print "  NEW  "
	#if CTF: vol = recons3d_4nn_ctf_MPI(myid, prjlist, snr, sign, sym, finfo, npad,xysize, zsize)
	pass#IMPORTIMPORTIMPORT from utilities import model_blank, get_im
	#from reconstruction import recons3d_4nnw_MPI
	#if myid == 0 :  print  sym,finfo,npad
	"""Multiline Comment35"""
	nnnx = ((prjlist[0].get_ysize())*2+3)

	pass#IMPORTIMPORTIMPORT from utilities import read_text_file, read_text_row, write_text_file
	pass#IMPORTIMPORTIMPORT from fundamentals import fft,fshift

	"""Multiline Comment36"""
	m = [1.0]*600
	for i in range(len(prjlist)):
		prjlist[i] = sparx_fundamentals.fft(prjlist[i])
		#prjlist[i] = fshift(prjlist[i],sxs,sys)
		prjlist[i].set_attr("padffted",1)
		prjlist[i].set_attr("npad",1)
		#set_params_proj(prjlist[i] ,[phi,theta,psi,0.0,0.0])
		prjlist[i].set_attr("bckgnoise",m)
	pass#IMPORTIMPORTIMPORT from reconstruction import recons3d_4nnfs_MPI


	vol,wei,reg = sparx_reconstruction.recons3d_4nnfs_MPI(myid, 0, prjlist,  upweighted = False, CTF = CTF, compensate = False, target_size = nnnx)

	if myid == 0 :
		#print  info(reg)
		#for i in xrange(reg.get_xsize()):  print i,reg[i]
		if(vol_stack[-3:] == "spi"):
			sparx_utilities.drop_image(vol, vol_stack, "s")
		else:
			sparx_fundamentals.fft(vol).write_image( vol_stack )
			wei.write_image("w"+vol_stack)
			reg.write_image("r"+vol_stack)
		#drop_image(vol1, "nvol0.hdf")
		#drop_image(vol2, "nvol1.hdf")
		#write_text_file(fff,"nfsc.txt")
		if not(finfo is None):
			finfo.write( "result written to " + vol_stack + "\n")
			finfo.write( "Total time: %10.3f\n" % (time.time()-time_start) )
			finfo.flush()


def newrecons3d_n_MPI(prj_stack, pid_list, vol_stack, CTF, snr, sign, npad, sym, listfile, group, verbose,xysize, zsize):
	pass#IMPORTIMPORTIMPORT from reconstruction import recons3d_4nn_ctf_MPI, recons3d_4nn_MPI, recons3d_4nnf_MPI
	pass#IMPORTIMPORTIMPORT from utilities      import get_im, drop_image, bcast_number_to_all, write_text_file, read_text_file
	pass#IMPORTIMPORTIMPORT from string         import replace
	pass#IMPORTIMPORTIMPORT from time           import time
	pass#IMPORTIMPORTIMPORT from mpi            import mpi_comm_size, mpi_comm_rank, mpi_bcast, MPI_INT, MPI_COMM_WORLD

	myid  = mpi.mpi_comm_rank(mpi.MPI_COMM_WORLD)
	nproc = mpi.mpi_comm_size(mpi.MPI_COMM_WORLD)
	time_start = time.time()
	"""Multiline Comment37"""

	if verbose==0:
		finfo = None
	else:
		infofile = "progress%04d.txt"%(myid+1)
		finfo = open( infofile, 'w' )

	if(myid == 0):
		#pid_list = read_text_file("main000/chunk0.txt")
		pid_list = sparx_utilities.read_text_file("main000/indexes.txt")
		nima = len(pid_list)
	else: nima = 0
	nima = sparx_utilities.bcast_number_to_all(nima, source_node = 0)
	if(myid != 0):
		pid = [-1]*nima
	pid_list = mpi.mpi_bcast(pid_list, nima, mpi.MPI_INT, 0, mpi.MPI_COMM_WORLD)
	pid_list = list(map(int, pid_list))
	
	image_start, image_end = MPI_start_end(nima, nproc, myid)
	prjlist = [EMAN2_cppwrap.EMData.read_images(prj_stack, pid_list[image_start:image_end])]


	if(myid == 0):
		#pid_list = read_text_file("main000/chunk1.txt")
		pid_list = list(range(320))
		nima = len(pid_list)
	else: nima = 0
	nima = sparx_utilities.bcast_number_to_all(nima, source_node = 0)
	if(myid != 0):
		pid = [-1]*nima
	pid_list = mpi.mpi_bcast(pid_list, nima, mpi.MPI_INT, 0, mpi.MPI_COMM_WORLD)
	pid_list = list(map(int, pid_list))
	
	image_start, image_end = MPI_start_end(nima, nproc, myid)
	prjlist += [EMAN2_cppwrap.EMData.read_images(prj_stack, pid_list[image_start:image_end])]

	pass#IMPORTIMPORTIMPORT from fundamentals import fdecimate
	pass#IMPORTIMPORTIMPORT from utilities import get_params_proj,set_params_proj
	scale = 384./54.
	for k in range(2):
		for i in range(len(prjlist[k])):
			prjlist[k][i] = sparx_fundamentals.fdecimate(prjlist[k][i],54,54)
			ctf_params = prjlist[k][i].get_attr("ctf")
			ctf_params.apix *= scale
			prjlist[k][i].set_attr('ctf', ctf_params)
			phi,theta,psi,sx,sy = sparx_utilities.get_params_proj(prjlist[k][i])
			sparx_utilities.set_params_proj(prjlist[k][i],[phi,theta,psi,sx/scale,sy/scale])
			
	if myid == 0 :  print("  NEW  ")
	#if CTF: vol = recons3d_4nn_ctf_MPI(myid, prjlist, snr, sign, sym, finfo, npad,xysize, zsize)
	pass#IMPORTIMPORTIMPORT from utilities import model_blank, get_im
	pass#IMPORTIMPORTIMPORT from reconstruction import recons3d_4nnw_MPI
	pass#IMPORTIMPORTIMPORT from utilities import read_text_file, read_text_row, write_text_file
	bckgnoise = [sparx_utilities.get_im("bckgnoise.hdf"), sparx_utilities.read_text_file("defgroup_stamp.txt")]#model_blank(1000,1,1,1.0)
	if myid == 0 :  print(sym,finfo,npad)

	if CTF: vol1, vol2, fff = sparx_reconstruction.recons3d_4nnf_MPI(myid, prjlist, bckgnoise, symmetry = sym, info = finfo, npad = npad,\
									 smearstep = 0.0)
	else:	vol = sparx_reconstruction.recons3d_4nn_MPI(myid, prjlist, sym, finfo, snr, npad, xysize, zsize)
	if myid == 0 :
		sparx_utilities.drop_image(vol1, "nvol0.hdf")
		sparx_utilities.drop_image(vol2, "nvol1.hdf")
		sparx_utilities.write_text_file(fff,"nfsc.txt")
		if not(finfo is None):
			finfo.write( "result written to " + vol_stack + "\n")
			finfo.write( "Total time: %10.3f\n" % (time.time()-time_start) )
			finfo.flush()

def recons3d_f(prj_stack, vol_stack, fsc_file, mask=None, CTF=True, snr=1.0, sym="c1", listfile = "", group = -1, npad = 4, verbose=1, MPI=False):
	if MPI:
		recons3d_f_MPI(prj_stack, vol_stack, fsc_file, mask, CTF, snr, sym, listfile, group, npad, verbose)
		return

	nima = EMAN2_cppwrap.EMUtil.get_image_count( prj_stack )

	pass#IMPORTIMPORTIMPORT from reconstruction import recons3d_4nn_ctf, recons3d_4nn
	pass#IMPORTIMPORTIMPORT from statistics     import fsc_mask
	pass#IMPORTIMPORTIMPORT from utilities      import drop_image
	if(listfile):
		pass#IMPORTIMPORTIMPORT from utilities import read_text_file
		pid_list = sparx_utilities.read_text_file(listfile, 0)
		pid_list = list(map(int, pid_list))
	elif(group > -1):
			tmp_list = EMAN2_cppwrap.EMUtil.get_all_attributes(prj_stack, 'group')
			pid_list = []
			for i in range(len(tmp_list)):
				if(tmp_list[i] == group):  pid_list.append(i)
			del tmp_list
	else:
		pid_list = list(range(nima))
	if CTF:
		volodd = sparx_reconstruction.recons3d_4nn_ctf(prj_stack, [ pid_list[i] for i in range(0, len(pid_list), 2) ], snr, 1, sym, verbose, npad)
		voleve = sparx_reconstruction.recons3d_4nn_ctf(prj_stack, [ pid_list[i] for i in range(1, len(pid_list), 2) ], snr, 1, sym, verbose, npad)
		t = sparx_statistics.fsc_mask( volodd, voleve, mask, filename=fsc_file)
		del volodd, voleve
		volall = sparx_reconstruction.recons3d_4nn_ctf(prj_stack, pid_list,                                          snr, 1, sym, verbose, npad)
	else:
		volodd = sparx_reconstruction.recons3d_4nn(prj_stack, [ pid_list[i] for i in range(0, len(pid_list), 2) ], sym, npad, snr = snr)
		voleve = sparx_reconstruction.recons3d_4nn(prj_stack, [ pid_list[i] for i in range(1, len(pid_list), 2) ], sym, npad, snr = snr)
		t = sparx_statistics.fsc_mask( volodd, voleve, mask, filename=fsc_file)
		del volodd, voleve
		volall = sparx_reconstruction.recons3d_4nn(prj_stack, pid_list,                                          sym, npad, snr = snr)
	if(vol_stack[-3:] == "spi"):
		sparx_utilities.drop_image(volall, vol_stack, "s")
	else:
		sparx_utilities.drop_image(volall, vol_stack)

def recons3d_f_MPI(prj_stack, vol_stack, fsc_file, mask, CTF=True, snr=1.0, sym="c1", listfile="", group=-1, npad = 4, verbose=1):

	pass#IMPORTIMPORTIMPORT from mpi       import mpi_comm_size, mpi_comm_rank, MPI_COMM_WORLD, mpi_bcast, MPI_INT
	pass#IMPORTIMPORTIMPORT from utilities import drop_image, bcast_number_to_all
	nproc = mpi.mpi_comm_size( mpi.MPI_COMM_WORLD )
	myid  = mpi.mpi_comm_rank( mpi.MPI_COMM_WORLD )

	if(myid == 0):
		if(listfile):
			pass#IMPORTIMPORTIMPORT from utilities import read_text_file
			pid_list = sparx_utilities.read_text_file(listfile, 0)
			pid_list = list(map(int, pid_list))
			nima = len(pid_list)
		elif(group > -1):
			tmp_list = EMAN2_cppwrap.EMUtil.get_all_attributes(prj_stack, 'group')
			pid_list = []
			for i in range(len(tmp_list)):
				if(tmp_list[i] == group):  pid_list.append(i)
			del tmp_list
			nima = len(pid_list)
		else:
			nima = EMAN2_cppwrap.EMUtil.get_image_count(prj_stack)
			pid_list = list(range(nima))
	else:
		nima = 0

	nima = sparx_utilities.bcast_number_to_all(nima, source_node = 0)

	if myid != 0:
		pid_list = [-1]*nima
	pid_list = mpi.mpi_bcast(pid_list, nima, mpi.MPI_INT, 0, mpi.MPI_COMM_WORLD)
	pid_list = list(map(int, pid_list))

	image_start, image_end = MPI_start_end(nima, nproc, myid)

	imgdata = EMAN2_cppwrap.EMData.read_images(prj_stack, pid_list[image_start:image_end])
	del pid_list

	if verbose==0:
		finfo = None
	else:
		infofile = "progress%04d.txt" % (myid)
		finfo = open( infofile, 'w' )

	odd_start = image_start%2
	eve_start = (odd_start+1)%2
	if CTF:
		pass#IMPORTIMPORTIMPORT from reconstruction import rec3D_MPI
		vol,fsc = sparx_reconstruction.rec3D_MPI(imgdata, snr, sym, mask, fsc_file, myid, 0, 1.0, odd_start, eve_start, finfo, npad = npad)
	else :
		pass#IMPORTIMPORTIMPORT from reconstruction import rec3D_MPI_noCTF
		vol,fsc = sparx_reconstruction.rec3D_MPI_noCTF(imgdata, sym, mask, fsc_file, myid, 0, 1.0, odd_start, eve_start, finfo, npad = npad)
	if myid == 0:
		if(vol_stack[-3:] == "spi"):
			sparx_utilities.drop_image(vol, vol_stack, "s")
		else:
			sparx_utilities.drop_image(vol, vol_stack)

def ssnr3d(stack, output_volume = None, ssnr_text_file = None, mask = None, reference_structure = None, ou = -1, rw = 1.0,  npad = 1, CTF = False, sign = 1, sym ="c1", MPI = False, random_angles = 0):
	'''
	        Perform 3D reconstruction using selected particle images, 
	        and calculate spectrum signal noise ratio (SSNR).
	        1. The selection file is supposed to be in SPIDER text format.
	        2. The 3D alignment parameters have been written in headers of the particle images.
	''' 
	if MPI:
		ssnr3d_MPI(stack, output_volume, ssnr_text_file, mask, reference_structure, ou, rw, npad, CTF, sign, sym, random_angles)
		return

	pass#IMPORTIMPORTIMPORT from utilities               import model_circle, get_im
	pass#IMPORTIMPORTIMPORT from filter                  import filt_ctf
	pass#IMPORTIMPORTIMPORT from reconstruction          import recons3d_nn_SSNR, recons3d_4nn, recons3d_4nn_ctf
	pass#IMPORTIMPORTIMPORT from projection              import prep_vol, prgs
	
	fring_width = float(rw)
	if mask:
		pass#IMPORTIMPORTIMPORT import  types
		if type(mask) is bytes:
			mask2D=sparx_utilities.get_im(mask)
		else:
			mask2D = mask
	else:
		mask2D = None

	[ssnr1, vol_ssnr1] = sparx_reconstruction.recons3d_nn_SSNR(stack, mask2D, rw, npad, sign, sym, CTF, random_angles)
	vol_ssnr1.write_image(output_volume, 0)
	del vol_ssnr1
	pass#IMPORTIMPORTIMPORT from sys import exit
	#exit()

	# perform 3D reconstruction
	if reference_structure == None:
		nima = EMAN2_cppwrap.EMUtil.get_image_count(stack)
		if CTF:
			snr = 1.0e20
			vol = sparx_reconstruction.recons3d_4nn_ctf(stack, list(range(nima)), snr, sign, sym)
		else:   vol = sparx_reconstruction.recons3d_4nn(stack, list(range(nima)), sym, snr = snr)
	else:
		vol = sparx_utilities.get_im(reference_structure)

	# re-project the reconstructed volume
	nx = vol.get_xsize()
	if int(ou) == -1: radius = nx//2 - 1
	else :            radius = int(ou)
	#
	vol *= sparx_utilities.model_circle(radius, nx, nx, nx)
	volft, kb = sparx_projection.prep_vol(vol)
	del vol
	prjlist = []
	pass#IMPORTIMPORTIMPORT from utilities import get_params_proj
	for i in range(nima):
		e = EMAN2_cppwrap.EMData()
		e.read_image(stack, i, True)
		e.set_attr('sign', 1)
		phi, theta, psi, tx, ty = sparx_utilities.get_params_proj(e)
		proj = sparx_projection.prgs(volft, kb, [phi, theta, psi, -tx, -ty])
		if CTF :
			ctf_params = e.get_attr("ctf")			
			proj = sparx_filter.filt_ctf(proj, ctf_params)
		prjlist.append(proj)
	del volft
	[ssnr2, vol_ssnr2] = sparx_reconstruction.recons3d_nn_SSNR(prjlist, mask2D, rw, npad, sign, sym, CTF, random_angles)
	vol_ssnr2.write_image(output_volume, 1)
	outf = open(ssnr_text_file, "w")
	for i in range(len(ssnr2[0])):
		datstrings = []
		datstrings.append("  %15f" % ssnr1[0][i])    # have to subtract 0.5 as in C code there is round.
		datstrings.append("  %15e" % ssnr1[1][i])    # SSNR
		datstrings.append("  %15e" % ssnr1[2][i])    # variance
		datstrings.append("  %15f" % ssnr1[3][i])    # number of points in the shell
		datstrings.append("  %15f" % ssnr1[4][i])    # number of added Fourier points
		datstrings.append("  %15e" % ssnr1[5][i])    # square of signal
		datstrings.append("  %15e" % ssnr2[1][i])    # SSNR
		datstrings.append("  %15e" % ssnr2[2][i])    # variance
		datstrings.append("  %15e" % ssnr2[5][i])    # square of signal
		datstrings.append("\n")
		outf.write("".join(datstrings))
	outf.close()

	
	"""Multiline Comment38"""

def ssnr3d_MPI(stack, output_volume = None, ssnr_text_file = None, mask = None, reference_structure = None, ou = -1, rw = 1.0, npad = 1, CTF = False, sign = 1, sym ="c1", random_angles = 0):
	pass#IMPORTIMPORTIMPORT from reconstruction import recons3d_nn_SSNR_MPI, recons3d_4nn_MPI, recons3d_4nn_ctf_MPI
	pass#IMPORTIMPORTIMPORT from utilities      import bcast_EMData_to_all, model_blank, model_circle, get_im
	pass#IMPORTIMPORTIMPORT from projection     import prep_vol, prgs
	pass#IMPORTIMPORTIMPORT from mpi            import mpi_comm_size, mpi_comm_rank, MPI_COMM_WORLD

	nima = EMAN2_cppwrap.EMUtil.get_image_count(stack)
	nproc = mpi.mpi_comm_size(mpi.MPI_COMM_WORLD)
	myid  = mpi.mpi_comm_rank(mpi.MPI_COMM_WORLD)

	image_start, image_end = MPI_start_end(nima, nproc, myid)

	if mask:
		pass#IMPORTIMPORTIMPORT import  types
		if type(mask) is bytes: mask2D = sparx_utilities.get_im(mask)
		else: mask2D = mask
	else:
		mask2D = None

	prjlist = EMAN2_cppwrap.EMData.read_images(stack, list(range(image_start, image_end)))
	if random_angles > 0:
		for prj in prjlist:
			# horatio active_refactoring Jy51i1EwmLD4tWZ9_00000_1
			# active = prj.get_attr_default('active', 1)
			# if active == 1:
			if random_angles == 2:
				pass#IMPORTIMPORTIMPORT from  random import  random
				phi	 = 360.0*numpy.random.random()
				theta	 = 180.0*numpy.random.random()
				psi	 = 360.0*numpy.random.random()
				xform_proj = EMAN2_cppwrap.Transform( {"type":"spider", "phi":phi, "theta":theta, "psi":psi} )
				prj.set_attr("xform.projection", xform_proj)
			elif random_angles == 3:
				pass#IMPORTIMPORTIMPORT from  random import  random
				phi    = 360.0*numpy.random.random()
				theta  = 180.0*numpy.random.random()
				psi    = 360.0*numpy.random.random()
				tx     = 6.0*(numpy.random.random() - 0.5)
				ty     = 6.0*(numpy.random.random() - 0.5)
				xform_proj = EMAN2_cppwrap.Transform( {"type":"spider", "phi":phi, "theta":theta, "psi":psi, "tx":tx, "ty":ty} )
				prj.set_attr("xform.projection", xform_proj)
			elif random_angles  == 1:
				pass#IMPORTIMPORTIMPORT from  random import  random
				old_xform_proj = prj.get_attr( "xform.projection" )
				dict = old_xform_proj.get_rotation( "spider" )
				dict["psi"] = 360.0*numpy.random.random()
				xform_proj = EMAN2_cppwrap.Transform( dict )
				prj.set_attr("xform.projection", xform_proj)
		random_angles = 0
	if myid == 0: [ssnr1, vol_ssnr1] = sparx_reconstruction.recons3d_nn_SSNR_MPI(myid, prjlist, mask2D, rw, npad, sign, sym, CTF, random_angles)  
	else:	                           sparx_reconstruction.recons3d_nn_SSNR_MPI(myid, prjlist, mask2D, rw, npad, sign, sym, CTF, random_angles)
	if myid == 0:
		vol_ssnr1.write_image( output_volume, 0)
		del vol_ssnr1

	nx  = prjlist[0].get_xsize()
	if ou == -1: radius = int(nx/2) - 1
	else:        radius = int(ou)
	if(reference_structure == None):
		vol = sparx_utilities.model_blank(nx, nx, nx)
		if CTF:
			snr = 1.0e20
			if myid == 0 : vol = sparx_reconstruction.recons3d_4nn_ctf_MPI(myid, prjlist, snr, sign, sym)
			else :  	     sparx_reconstruction.recons3d_4nn_ctf_MPI(myid, prjlist, snr, sign, sym)
		else:
			if myid == 0 : vol = sparx_reconstruction.recons3d_4nn_MPI(myid, prjlist, sym, snr=snr)
			else:		     sparx_reconstruction.recons3d_4nn_MPI(myid, prjlist, sym, snr = snr)
	else:
		if myid == 0: vol = sparx_utilities.get_im(reference_structure)

	sparx_utilities.bcast_EMData_to_all(vol, myid, 0)
	re_prjlist = []
	#vol *= model_circle(radius, nx, nx, nx)
	volft, kb = sparx_projection.prep_vol(vol)
	del vol
	pass#IMPORTIMPORTIMPORT from utilities import get_params_proj
	pass#IMPORTIMPORTIMPORT if CTF: from filter import filt_ctf
	for prj in prjlist:
		phi, theta, psi, tx, ty = sparx_utilities.get_params_proj(prj)
		proj = sparx_projection.prgs(volft, kb, [phi, theta, psi, -tx, -ty])
		if CTF:
			ctf_params = prj.get_attr("ctf")			
			proj = sparx_filter.filt_ctf(proj, ctf_params)
			proj.set_attr('sign', 1)
		re_prjlist.append(proj)
	del volft, prjlist
	if myid == 0: [ssnr2, vol_ssnr2] = sparx_reconstruction.recons3d_nn_SSNR_MPI(myid, re_prjlist, mask2D, rw, npad, sign, sym, CTF, random_angles)
	else:                              sparx_reconstruction.recons3d_nn_SSNR_MPI(myid, re_prjlist, mask2D, rw, npad, sign, sym, CTF, random_angles)
	if myid == 0:
		vol_ssnr2.write_image( output_volume, 1)
		outf = open(ssnr_text_file, "w")
		for i in range(len(ssnr2[0])):
			datstrings = []
			datstrings.append("  %15f" % ssnr1[0][i])    #  have to subtract 0.5 as in C code there is round.
			datstrings.append("  %15e" % ssnr1[1][i])    # SSNR
			datstrings.append("  %15e" % ssnr1[2][i])    # variance
			datstrings.append("  %15f" % ssnr1[3][i])    # number of points in the shell
			datstrings.append("  %15f" % ssnr1[4][i])    # number of added Fourier points
			datstrings.append("  %15e" % ssnr1[5][i])    # square of signal
			datstrings.append("  %15e" % ssnr2[1][i])    # SSNR
			datstrings.append("  %15e" % ssnr2[2][i])    # variance
			datstrings.append("  %15e" % ssnr2[5][i])    # square of signal
			datstrings.append("\n")
			outf.write("".join(datstrings))
		outf.close()
		"""Multiline Comment39"""

def varimax(input_stack, imglist, output_stack, maskfile, mask_radius, verbose ) :
	pass#IMPORTIMPORTIMPORT from utilities import get_im, model_circle
	pass#IMPORTIMPORTIMPORT from EMAN2     import Analyzers

	data = sparx_utilities.get_im( input_stack )

	if maskfile:
		pass#IMPORTIMPORTIMPORT import types
		if type(maskfile) is bytes: mask = sparx_utilities.get_im(maskfile)
		else:                                  mask = maskfile
	else:
		if(mask_radius < 1):  mask_radius = data.get_xsize()//2-2
		mask = sparx_utilities.model_circle( mask_radius, data.get_xsize(), data.get_ysize(), data.get_zsize() )

	ana = EMAN2_cppwrap.Analyzers.get( "varimax", {"mask":mask} )
	sumeig =0.0
	#from utilities import info
	#from math import sqrt
	for i in imglist:
		data = sparx_utilities.get_im( input_stack, i)
		eigval = data.get_attr_default('eigval', 1.0)
		sumeig += eigval
		#Util.mul_scalar(data, sqrt(eigval))
		#info(data)
		ana.insert_image( data )
		#print "Inserting image %4d" % i
	del data
	vecs = ana.analyze()

	sumeig /= len(vecs)
	#print  sumeig
	for iout in range(len(vecs)):
		#info(vecs[iout])
		vecs[iout].set_attr('eigval', sumeig)
		vecs[iout].write_image( output_stack, iout)

def bootstrap_genbuf(prj_stack, buf_prefix, npad, verbose, CTF=False):
	pass#IMPORTIMPORTIMPORT from EMAN2 import newfile_store
	pass#IMPORTIMPORTIMPORT import os
	size = 1
	myid = 0
	print()  
	if os.path.exists( buf_prefix + ".bin" ):
		sparx_global_def.ERROR('Output file exists, please change the name and restart the program', "bootstrap_genbuf", 1)

	if(verbose == 1):  finfo=open( os.path.join(outdir, "progress%04d.txt" % myid), "w" )
	else:              finfo = None

	store = EMAN2_cppwrap.newfile_store(buf_prefix, npad, CTF)

	nimage = EMAN2_cppwrap.EMUtil.get_image_count( prj_stack )
	for i in range(nimage):
		proj = EMAN2_cppwrap.EMData()
		proj.read_image( prj_stack, i )
		store.add_image( proj, proj.get_attr("xform.projection") )

		if( verbose == 1 and ((i+1) % 100 == 0  or i==nimage-1)) :
			finfo.write( "projection %6d buffered\n" % (i+1) )
			finfo.flush()
 
def bootstrap_run(prj_stack, media, outdir, nvol, CTF, snr, sym, verbose, MPI=False):

	pass#IMPORTIMPORTIMPORT import string
	pass#IMPORTIMPORTIMPORT from mpi import mpi_comm_size, mpi_comm_rank, MPI_COMM_WORLD, mpi_init, mpi_barrier
	pass#IMPORTIMPORTIMPORT import os

	if MPI:
		size = mpi.mpi_comm_size(mpi.MPI_COMM_WORLD)
		myid = mpi.mpi_comm_rank(mpi.MPI_COMM_WORLD)
	else:
		size = 1
		myid = 0

	if myid==0:
		if os.path.exists(outdir): sparx_global_def.ERROR('Output directory exists, please change the name and restart the program', "bootstrap_run", 1,myid)
		os.system( "mkdir " + outdir )
	if MPI:
		mpi.mpi_barrier( mpi.MPI_COMM_WORLD )	


	myvolume_file = "%s/bsvol%04d.hdf" % (outdir, myid)
	if verbose != 0 :
		mystatus_file = "%s/status%04d.inf" % (outdir, myid)
		mystatus = open( mystatus_file, 'w' )
	else:
		mystatus = None

	mynvol = nvol/size

	if myid==(size-1) : mynvol = mynvol + (nvol%size)


	nproj = EMAN2_cppwrap.EMUtil.get_image_count(prj_stack)
	if verbose != 0 : mystatus.write( "# of projs: %d\n" % nproj )

	npad = 4
	sign = 1
	list_proj = list(range(nproj))
	pass#IMPORTIMPORTIMPORT from reconstruction import bootstrap_nn
	sparx_reconstruction.bootstrap_nn( prj_stack, myvolume_file, list_proj, mynvol, media, npad, sym, mystatus, CTF, snr, sign)
	
def wrapper_params_2D_to_3D(stack):
	pass#IMPORTIMPORTIMPORT from utilities import params_2D_3D, print_begin_msg, print_end_msg, print_msg, get_params2D, set_params_proj, write_header
	
	#print_begin_msg("params_2D_to_3D")
	#print_msg("Input stack                 : %s\n\n"%(stack))

	nima = EMAN2_cppwrap.EMUtil.get_image_count(stack)
	ima = EMAN2_cppwrap.EMData()
	for im in range(nima):
		ima.read_image(stack, im, True)
		p = sparx_utilities.get_params2D(ima)
		p = sparx_utilities.params_2D_3D(p[0], p[1], p[2], int(p[3]))
		sparx_utilities.set_params_proj(ima, p)
		sparx_utilities.write_header(stack, ima, im)
	#print_end_msg("params_2D_to_3D")

def wrapper_params_3D_to_2D(stack):
	pass#IMPORTIMPORTIMPORT from utilities import params_3D_2D, print_begin_msg, print_end_msg, print_msg, set_params2D, write_header
	
	#print_begin_msg("params_3D_to_2D")
	#print_msg("Input stack                 : %s\n\n"%(stack))

	nima = EMAN2_cppwrap.EMUtil.get_image_count(stack)
	ima = EMAN2_cppwrap.EMData()
	for im in range(nima):
		ima.read_image(stack, im, True)
		pass#IMPORTIMPORTIMPORT from utilities import set_params_proj, get_params_proj
		phi,theta,psi,s2x,s2y = sparx_utilities.get_params_proj( ima )
		alpha, sx, sy, mirror = sparx_utilities.params_3D_2D(phi, theta, psi, s2x, s2y)
		sparx_utilities.set_params2D(ima, [alpha, sx, sy, mirror, 1.0])
		sparx_utilities.write_header(stack, ima, im)
	#print_end_msg("params_3D_to_2D")


# application find structure
def cml_find_structure_main(stack, out_dir, ir, ou, delta, dpsi, lf, hf, rand_seed, maxit, given = False, first_zero = False, flag_weights = False, debug = False, trials = 1):
	pass#IMPORTIMPORTIMPORT from projection import cml_open_proj, cml_init_global_var, cml_head_log, cml_disc, cml_export_txtagls
	pass#IMPORTIMPORTIMPORT from projection import cml_find_structure, cml_export_struc, cml_end_log
	pass#IMPORTIMPORTIMPORT from utilities  import print_begin_msg, print_msg, print_end_msg, start_time, running_time
	pass#IMPORTIMPORTIMPORT from copy       import deepcopy
	pass#IMPORTIMPORTIMPORT from random     import seed, random
	pass#IMPORTIMPORTIMPORT import time, sys, os

	# logfile
	t_start = sparx_utilities.start_time()

	out_dir = out_dir.rstrip('/')
	if os.path.exists(out_dir): sparx_global_def.ERROR('Output directory exists, please change the name and restart the program', "cml_find_structure_main", 1)
	os.mkdir(out_dir)
	pass#IMPORTIMPORTIMPORT import global_def
	sparx_global_def.LOGFILE =  os.path.join(out_dir, sparx_global_def.LOGFILE)
	sparx_utilities.print_begin_msg('find_struct')

	if rand_seed > 0: random.seed(rand_seed)
	else:             random.seed()

	# Open and transform projections
	Prj, Ori = sparx_projection.cml_open_proj(stack, ir, ou, lf, hf, dpsi)
	# Init the global vars
	sparx_projection.cml_init_global_var(dpsi, delta, len(Prj), debug)
	# Update logfile
	sparx_projection.cml_head_log(stack, out_dir, delta, ir, ou, lf, hf, rand_seed, maxit, given, flag_weights, trials, 1)

	ibest    = -1
	bestdisc = 1.0e20
	MEM      = []
	for itrial in range(trials):
		# if not angles given select randomly orientation for each projection
		if not given:
			j = 0
			for n in range(len(Prj)):
				if first_zero and n == 0:
					Ori[j]   = 0.0
					Ori[j+1] = 0.0
					Ori[j+2] = 0.0
				else:
					Ori[j]   = numpy.random.random() * 360  # phi
					Ori[j+1] = numpy.random.random() * 180  # theta
					Ori[j+2] = numpy.random.random() * 360  # psi
				j += 4

		# prepare rotation matrix
		Rot = EMAN2_cppwrap.Util.cml_init_rot(Ori)
		# Compute the first disc
		disc_init = sparx_projection.cml_disc(Prj, Ori, Rot, flag_weights)
		# Update progress file
		sparx_projection.cml_export_txtagls(out_dir, 'angles_%03i' % itrial, Ori, disc_init, 'Init')
		# Find structure
		Ori, disc, ite = sparx_projection.cml_find_structure(Prj, Ori, Rot, out_dir, 'angles_%03i' % itrial, maxit, first_zero, flag_weights)
		sparx_utilities.print_msg('Trial %03i\tdiscrepancy init: %10.7f\tnb ite: %i\tdiscrepancy end: %10.7f\n' % (itrial, disc_init, ite + 1, disc))
		if disc < bestdisc:
			bestdisc = disc
			ibest    = itrial
			MEM      = copy.deepcopy(Ori)

		# Export structure
		sparx_projection.cml_export_struc(stack, out_dir, itrial, Ori)

	sparx_utilities.print_msg('\n Selected trial #%03i with disc %10.7f\n' % (ibest, bestdisc))
	os.system('cp %s/structure_%03i.hdf %s/structure.hdf' % (out_dir, ibest, out_dir))
	sparx_projection.cml_end_log(MEM)
	sparx_utilities.running_time(t_start)
	sparx_utilities.print_end_msg('find_struct')

# application find structure
def cml_find_structure_MPI2(stack, out_dir, ir, ou, delta, dpsi, lf, hf, rand_seed, maxit, given = False, first_zero = False, flag_weights = False, debug = False, trials = 1):
	pass#IMPORTIMPORTIMPORT from projection import cml_open_proj, cml_init_global_var, cml_head_log, cml_disc, cml_export_txtagls
	pass#IMPORTIMPORTIMPORT from projection import cml_find_structure2, cml_export_struc, cml_end_log
	pass#IMPORTIMPORTIMPORT from utilities  import print_begin_msg, print_msg, print_end_msg, start_time, running_time
	pass#IMPORTIMPORTIMPORT from copy       import deepcopy
	pass#IMPORTIMPORTIMPORT from random     import seed, random
	pass#IMPORTIMPORTIMPORT import time, sys, os
	pass#IMPORTIMPORTIMPORT from mpi 	  import mpi_init, mpi_comm_size, mpi_comm_rank, MPI_COMM_WORLD
	pass#IMPORTIMPORTIMPORT from mpi 	  import mpi_reduce, mpi_bcast, mpi_barrier, mpi_gatherv
	pass#IMPORTIMPORTIMPORT from mpi 	  import MPI_SUM, MPI_FLOAT, MPI_INT

	number_of_proc = mpi.mpi_comm_size(mpi.MPI_COMM_WORLD)
	myid = mpi.mpi_comm_rank(mpi.MPI_COMM_WORLD)
	main_node = 0

	# logfile
	if myid == main_node:
		t_start = sparx_utilities.start_time()

		out_dir = out_dir.rstrip('/')
		if os.path.exists(out_dir): sparx_global_def.ERROR('Output directory exists, please change the name and restart the program', "cml_find_structure_main", 1)
		os.mkdir(out_dir)
		pass#IMPORTIMPORTIMPORT import global_def
		sparx_global_def.LOGFILE =  os.path.join(out_dir, sparx_global_def.LOGFILE)
		sparx_utilities.print_begin_msg('find_struct')

	if rand_seed > 0: random.seed(rand_seed)
	else:             random.seed()

	# Open and transform projections
	Prj, Ori = sparx_projection.cml_open_proj(stack, ir, ou, lf, hf, dpsi)
	# Init the global vars
	sparx_projection.cml_init_global_var(dpsi, delta, len(Prj), debug)
	# Update logfile
	if myid == main_node:
		sparx_projection.cml_head_log(stack, out_dir, delta, ir, ou, lf, hf, rand_seed, maxit, given, flag_weights, trials, number_of_proc)

	ibest    = -1
	bestdisc = 1.0e20
	MEM      = []
	for itrial in range(trials):
		# if not angles given select randomly orientation for each projection
		if not given:
			j = 0
			for n in range(len(Prj)):
				if first_zero and n == 0:
					Ori[j]   = 0.0
					Ori[j+1] = 0.0
					Ori[j+2] = 0.0
				else:
					Ori[j]   = numpy.random.random() * 360  # phi
					Ori[j+1] = numpy.random.random() * 180  # theta
					Ori[j+2] = numpy.random.random() * 360  # psi
				j += 4

		# prepare rotation matrix
		Rot = EMAN2_cppwrap.Util.cml_init_rot(Ori)
		# Compute the first disc
		disc_init = sparx_projection.cml_disc(Prj, Ori, Rot, flag_weights)
		# Update progress file
		if myid == main_node:
			sparx_projection.cml_export_txtagls(out_dir, 'angles_%03i' % itrial, Ori, disc_init, 'Init')
		# Find structure
		Ori, disc, ite = sparx_projection.cml_find_structure2(Prj, Ori, Rot, out_dir, 'angles_%03i' % itrial, maxit, first_zero, flag_weights, myid, main_node, number_of_proc)
		if myid == main_node:
			sparx_utilities.print_msg('Trial %03i\tdiscrepancy init: %10.7f\tnb ite: %i\tdiscrepancy end: %10.7f\n' % (itrial, disc_init, ite + 1, disc))
		if disc < bestdisc:
			bestdisc = disc
			ibest    = itrial
			MEM      = copy.deepcopy(Ori)

		# Export structure
		if myid == main_node:
			sparx_projection.cml_export_struc(stack, out_dir, itrial, Ori)

	if myid == main_node:
		sparx_utilities.print_msg('\n Selected trial #%03i with disc %10.7f\n' % (ibest, bestdisc))
		os.system('cp %s/structure_%03i.hdf %s/structure.hdf' % (out_dir, ibest, out_dir))
		sparx_projection.cml_end_log(MEM)
		sparx_utilities.running_time(t_start)
		sparx_utilities.print_end_msg('find_struct')

# application find structure
def cml_find_structure_MPI(stack, out_dir, ir, ou, delta, dpsi, lf, hf, rand_seed, maxit, given = False, first_zero = False, flag_weights = False, debug = False, trials = 10):
	pass#IMPORTIMPORTIMPORT from projection import cml_open_proj, cml_init_global_var, cml_head_log, cml_disc, cml_export_txtagls
	pass#IMPORTIMPORTIMPORT from projection import cml_find_structure, cml_export_struc, cml_end_log, cml_init_rnd, cml2_ori_collinearity
	pass#IMPORTIMPORTIMPORT from utilities  import print_begin_msg, print_msg, print_end_msg, start_time, running_time
	pass#IMPORTIMPORTIMPORT from random     import seed, random
	pass#IMPORTIMPORTIMPORT from mpi        import mpi_init, mpi_comm_size, mpi_comm_rank, mpi_bcast
	pass#IMPORTIMPORTIMPORT from mpi        import mpi_barrier, MPI_COMM_WORLD, mpi_reduce, MPI_FLOAT, MPI_INT, MPI_SUM
	pass#IMPORTIMPORTIMPORT import time, sys, os

	# init
	sys.argv  = mpi.mpi_init(len(sys.argv),sys.argv)
	ncpu      = mpi.mpi_comm_size(mpi.MPI_COMM_WORLD)
	myid      = mpi.mpi_comm_rank(mpi.MPI_COMM_WORLD)
	main_node = 0
	mpi.mpi_barrier(mpi.MPI_COMM_WORLD)

	if os.path.exists(out_dir): sparx_global_def.ERROR('Output directory exists, please change the name and restart the program', "cml_find_structure_MPI ", 1,myid)
	mpi.mpi_barrier(mpi.MPI_COMM_WORLD)

	if myid == main_node:
		t_start = sparx_utilities.start_time()
		os.mkdir(out_dir)
		pass#IMPORTIMPORTIMPORT import global_def
		sparx_global_def.LOGFILE =  os.path.join(out_dir, sparx_global_def.LOGFILE)
		sparx_utilities.print_begin_msg('find_struct')


	flag = 0
	if myid == main_node:
		if ncpu > trials:
			print('** WARNING **')
			print('Find structure MPI: number of trials must be larger than the number of processors.')
			flag = 1
	flag = mpi.mpi_bcast(flag, 1, mpi.MPI_INT, 0, mpi.MPI_COMM_WORLD)
	flag = int(flag[0])
	if flag != 0: sys.exit()

	N_start, N_stop = MPI_start_end(trials, ncpu, myid)
	lrnd    = sparx_projection.cml_init_rnd(trials, rand_seed)
	out_dir = out_dir.rstrip('/')

	# Open and transform projections
	Prj, Ori = sparx_projection.cml_open_proj(stack, ir, ou, lf, hf, dpsi)

	# Init the global vars
	sparx_projection.cml_init_global_var(dpsi, delta, len(Prj), debug)

	# Update logfile
	if myid == main_node: sparx_projection.cml_head_log(stack, out_dir, delta, ir, ou, lf, hf, rand_seed, maxit, given, flag_weights, trials, ncpu)

	disc_init = [0.0] * trials
	disc_end  = [0.0] * trials
	ite       = [0]   * trials
	coll      = [0.0] * trials
	for itrial in range(N_start, N_stop):

		# if not angles given select randomly orientation for each projection
		if not given:
			random.seed(lrnd[itrial])
			j = 0
			for n in range(len(Prj)):
				if first_zero and n == 0:
					Ori[j]   = 0.0
					Ori[j+1] = 0.0
					Ori[j+2] = 0.0
				else:
					Ori[j]   = numpy.random.random() * 360  # phi
					Ori[j+1] = numpy.random.random() * 180  # theta
					Ori[j+2] = numpy.random.random() * 360  # psi
				j += 4

		# prepare rotation matrix
		Rot = EMAN2_cppwrap.Util.cml_init_rot(Ori)
		# Compute the first disc
		disc_init[itrial] = sparx_projection.cml_disc(Prj, Ori, Rot, flag_weights)
		# Update progress file
		sparx_projection.cml_export_txtagls(out_dir, 'angles_%03i' % itrial, Ori, disc_init[itrial], 'Init')
		# Find structure
		Ori, disc_end[itrial], ite[itrial] = sparx_projection.cml_find_structure(Prj, Ori, Rot, out_dir, 'angles_%03i' % itrial, maxit, first_zero, flag_weights)
		# Export structure
		sparx_projection.cml_export_struc(stack, out_dir, itrial, Ori)

	#from development import cml2_ori_collinearity
	coll[itrial] = sparx_projection.cml2_ori_collinearity(Ori)

	mpi.mpi_barrier(mpi.MPI_COMM_WORLD)
	disc_init = mpi.mpi_reduce(disc_init, trials, mpi.MPI_FLOAT, mpi.MPI_SUM, main_node, mpi.MPI_COMM_WORLD)
	disc_init = disc_init.tolist()
	disc_end  = mpi.mpi_reduce(disc_end, trials, mpi.MPI_FLOAT, mpi.MPI_SUM, main_node, mpi.MPI_COMM_WORLD)
	disc_end  = disc_end.tolist()
	ite       = mpi.mpi_reduce(ite, trials, mpi.MPI_INT, mpi.MPI_SUM, main_node, mpi.MPI_COMM_WORLD)
	ite       = ite.tolist()
	coll      = mpi.mpi_reduce(coll, trials, mpi.MPI_FLOAT, mpi.MPI_SUM, main_node, mpi.MPI_COMM_WORLD)
	coll      = coll.tolist()

	if myid == main_node:
		score = [0.0] * trials
		sparx_utilities.print_msg('\n')
		for i in range(trials):
			score[i] = disc_end[i] * (1 - coll[i])
			sparx_utilities.print_msg('Trial  %03i\trnd %10i discrepnacy init: %10.7f\tnb ite: %i\tdiscrepancy end: %10.7f\tcollinearity: %f\tscore: %f\n' % (i, lrnd[i], disc_init[i], ite[i] + 1, disc_end[i], coll[i], score[i]))
			
		ibest = disc_end.index(min(disc_end))
		#ibest = score.index(min(score))
		sparx_utilities.print_msg('\n Selected trial #%03i with discrepancy %10.7f\n' % (ibest, disc_end[ibest]))
		os.system('cp %s/structure_%03i.hdf %s/structure.hdf' % (out_dir, ibest, out_dir))

		sparx_utilities.running_time(t_start)
		sparx_utilities.print_end_msg('find_struct')

def imgstat_ccc( stacks, rad ):
	pass#IMPORTIMPORTIMPORT from EMAN2 import EMUtil
	pass#IMPORTIMPORTIMPORT from utilities import get_im, model_circle
	pass#IMPORTIMPORTIMPORT from statistics import ccc
	pass#IMPORTIMPORTIMPORT from projection import prep_vol,prgs
	pass#IMPORTIMPORTIMPORT from utilities	import get_params_proj

	if len(stacks)>3: sparx_global_def.ERROR("Error: ccc should be run on two stacks","imgstat_ccc",1)

	nimg1 = EMAN2_cppwrap.EMUtil.get_image_count( stacks[0] )
	nimg2 = EMAN2_cppwrap.EMUtil.get_image_count( stacks[1] )


	if nimg2==1 and sparx_utilities.get_im(stacks[0]).get_zsize()==1 and sparx_utilities.get_im(stacks[1]).get_zsize() > 1:
		print("ccc between prj and volume")
		volccc = True
		volft,kb = sparx_projection.prep_vol( sparx_utilities.get_im(stacks[1]) )
	else:
		volccc = False


	
	nimg = max(nimg1,nimg2)
	if(nimg2<nimg1): nimg2 = 1

	imgtmp = sparx_utilities.get_im( stacks[0] )

	if rad==-1:
		if len(stacks) == 3:  mask = sparx_utilities.get_im(stacks[2])
		else:                 mask = None
	else:
		if len(stacks) == 3:    sparx_global_def.ERROR("Error: Mask radius and mask file canot be given simultaneously","imgstat_ccc",1)
		else:
			nx = imgtmp.get_xsize()
			ny = imgtmp.get_ysize()
			nz = imgtmp.get_zsize()
			mask = sparx_utilities.model_circle( rad, nx, ny, nz )

	for i in range(nimg):
		img1 = sparx_utilities.get_im( stacks[0], i )

		if nimg2==1:
			if volccc:
				phi,tht,psi,s2x,s2y = sparx_utilities.get_params_proj( img1 )
				img2 = sparx_projection.prgs( volft,kb, [phi,tht,psi,-s2x,-s2y] )
			else:
				img2 = sparx_utilities.get_im( stacks[1] )
		else:
			img2 = sparx_utilities.get_im( stacks[1], i )

		val = sparx_statistics.ccc(img1, img2, mask)

		print("%6d: %10.5f" % (i, val))

def imgstat_fsc( stacks, fscfile, rad ):
	pass#IMPORTIMPORTIMPORT from utilities import get_im, model_circle
	pass#IMPORTIMPORTIMPORT from statistics import fsc_mask

	if len(stacks)>3: sparx_global_def.ERROR("Error: fsc should be run on two images","imgstat_fsc",1)

	img1 = sparx_utilities.get_im( stacks[0] )
	img2 = sparx_utilities.get_im( stacks[1] )

	nx = img1.get_xsize()
	ny = img1.get_ysize()
	nz = img1.get_zsize()

	if  EMAN2_cppwrap.EMUtil.get_image_count(stacks[0])>1: sparx_global_def.ERROR("Error: %s is an stack, fsc should be run on images","imgstat_fsc",1)

	if img2.get_xsize() != nx or img2.get_ysize() != ny or img2.get_zsize() != nz: sparx_global_def.ERROR("Error: input images has different sizes","imgstat_fsc",1)

	if rad==-1:
		if len(stacks) == 3: mask = sparx_utilities.get_im(stacks[2])
		else:                mask = None
	else:
		if len(stacks) == 3:  sparx_global_def.ERROR("Error: Mask radius and mask file canot be given simultaneously","imgstat_fsc",1)
		else:    mask = sparx_utilities.model_circle( rad, nx, ny, nz )

	sparx_statistics.fsc_mask( img1, img2, mask, filename=fscfile )

def imgstat_inf( stacks, rad ):
	pass#IMPORTIMPORTIMPORT from EMAN2 import EMUtil
	pass#IMPORTIMPORTIMPORT from utilities import get_im, model_circle
	if len(stacks)>2: sparx_global_def.ERROR("Error: inf should be run on one file","imgstat_inf",1)

	nimg = EMAN2_cppwrap.EMUtil.get_image_count( stacks[0] )
	img1 = sparx_utilities.get_im( stacks[0] )

	nx = img1.get_xsize()
	ny = img1.get_ysize()
	nz = img1.get_zsize()

	if rad==-1:
		if len(stacks) == 2:  mask = sparx_utilities.get_im(stacks[1])
		else:                 mask = None
	else:
		if len(stacks) == 2:    sparx_global_def.ERROR("Error: Mask radius and mask file canot be given simultaneously","imgstat_inf",1)
		else:			mask = sparx_utilities.model_circle( rad, nx, ny, nz )


	for i in range(nimg):

		img = sparx_utilities.get_im( stacks[0], i )

		[avg,sigma,fmin,fmax] = EMAN2_cppwrap.Util.infomask( img, mask, True )
		if mask == None:    L2 = img.cmp("dot", img, dict(negative = 0))
		else:               L2 = img.cmp("dot", img, dict(negative = 0, mask = mask))

		print("nx,ny,nz,avg,sigma,min,max, L2: %6d %6d %6d %11.4e %10.5f %10.5f %10.5f %11.4e" % (nx, ny, nz, avg, sigma, fmin, fmax, L2 ))

def imgstat( stacks, ifccc, fscfile, pinf, rad ):
	if ifccc:
		imgstat_ccc( stacks, rad )
		return

	if len(fscfile)>0:
		imgstat_fsc( stacks, fscfile, rad )
		return

	if pinf:
		imgstat_inf( stacks, rad )
		return

def normal_prj( prj_stack, outdir, refvol, weights, r, niter, snr, sym, verbose = 0, CTF = False, MPI=False ):
	def peak_range( nx, ctf ):
		"""
		  Find first maximum of the CTF, use CTF^2, so the sign will be ignored
		"""
		pass#IMPORTIMPORTIMPORT from morphology import ctf_2
		ctf = sparx_morphology.ctf_2( nx, ctf )

		for i in range( 1, len(ctf)-1 ):
			prev = ctf[i-1]
			curt = ctf[i]
			next = ctf[i+1]

			if curt > prev and curt > next:
				freq = float(i)/nx
				return [freq-0.03, freq+0.02]

		assert false

	pass#IMPORTIMPORTIMPORT from utilities     import get_image, get_im, model_circle, drop_spider_doc
	pass#IMPORTIMPORTIMPORT from utilities     import drop_image, get_params_proj
	pass#IMPORTIMPORTIMPORT from projection    import prep_vol, prgs
	pass#IMPORTIMPORTIMPORT from filter        import filt_ctf, filt_btwo, filt_tophatb
	pass#IMPORTIMPORTIMPORT from fundamentals  import fft
	pass#IMPORTIMPORTIMPORT from statistics    import ccc
	pass#IMPORTIMPORTIMPORT import os

	if(MPI and not (weights is None)):
		sparx_global_def.ERROR('Application of weights does not have MPI version', "normal_prj", 1,myid)

	if MPI:
		pass#IMPORTIMPORTIMPORT from mpi import mpi_comm_size, mpi_comm_rank, mpi_barrier, mpi_init, mpi_reduce, mpi_bcast, MPI_COMM_WORLD, MPI_FLOAT, MPI_SUM
		pass#IMPORTIMPORTIMPORT from utilities     import bcast_EMData_to_all
		nproc = mpi.mpi_comm_size( mpi.MPI_COMM_WORLD )
		myid  = mpi.mpi_comm_rank( mpi.MPI_COMM_WORLD )
	else:
		nproc = 1
		myid  = 0
		if( not (weights is None) ):
			#  This section is application of weights
			pass#IMPORTIMPORTIMPORT from utilities import read_text_file
			s = sparx_utilities.read_text_file(weights)
			img_number     = EMAN2_cppwrap.EMUtil.get_image_count( prj_stack )
			if(len(s) != img_number):  sparx_global_def.ERROR('Number of images does not agree with number of weights', "normal_prj", 1,myid)
			for i in range(img_number):
				img = sparx_utilities.get_im(prj_stack, i)
				EMAN2_cppwrap.Util.mul_scalar(img, s[i])
				img.write_image(outdir, i)
			return
	
	if myid== 0:
		if os.path.exists(outdir): sparx_global_def.ERROR('Output directory exists, please change the name and restart the program', "normal_prj", 1,myid)
		os.mkdir(outdir)

	if MPI:
		mpi.mpi_barrier( mpi.MPI_COMM_WORLD )


	img = sparx_utilities.get_image( prj_stack )
	nx  = img.get_xsize()
	ny  = img.get_ysize()
	del img

	if r <= 0:  r = nx//2-1

	img_number     = EMAN2_cppwrap.EMUtil.get_image_count( prj_stack )
	img_node_start, img_node_end = MPI_start_end(img_number, nproc, myid )


	if(verbose == 1):  info=open( os.path.join(outdir, "progress%04d.txt" % myid), "w" )
	else:              info = None

	imgdata = EMAN2_cppwrap.EMData.read_images(prj_stack, list(range(img_node_start, img_node_end)))

	if(verbose == 1):
		info.write( ' all images loaded\n' )
		info.flush( )

	pred = [1.0]* len(imgdata)

	pass#IMPORTIMPORTIMPORT from reconstruction import rec3D_MPI,rec3D_MPI_noCTF,rec3D_MPI_noCTF, recons3d_4nn_ctf, recons3d_4nn
	if refvol is None:
		fsc_file = os.path.join(outdir, "fsc_init.dat")
		vol_file = os.path.join(outdir, "vol_init.hdf")
		if  MPI:
			if  CTF:  refvol, fscc = sparx_reconstruction.rec3D_MPI( imgdata, snr, sym, None, fsc_file, myid )
			else:     refvol, fscc = sparx_reconstruction.rec3D_MPI_noCTF( imgdata, sym, None, fsc_file, myid )
			sparx_utilities.bcast_EMData_to_all( refvol, myid )
		else:
			if CTF:   refvol = sparx_reconstruction.recons3d_4nn_ctf( imgdata, list(range(len(imgdata))), snr, 1, sym)
			else:	   refvol = sparx_reconstruction.recons3d_4nn( imgdata, list(range(len(imgdata))), sym, snr = snr)
		if myid==0:
			refvol.write_image( vol_file )
		if(verbose == 1):
			info.write( "inital reconstructed volume written to " + vol_file + "\n" )
			info.flush()


	mask = sparx_utilities.model_circle( r, nx, ny )
	for iter in range(niter) :
		refvol, kb = sparx_projection.prep_vol( refvol )

		scales = []
		for i in range( len(imgdata) ) :
			exp_prj = imgdata[i].copy()

			phi,theta,psi,s2x,s2y = sparx_utilities.get_params_proj( exp_prj )

			ref_prj = sparx_filter.filt_btwo( sparx_fundamentals.fft( sparx_projection.prgs( refvol, kb, [phi, theta, psi, -s2x, -s2y] ) ), 0.01, 0.1, 0.2)

			if  CTF:
				ctf = exp_prj.get_attr( "ctf" )
				ref_prj = sparx_filter.filt_ctf( sparx_filter.filt_ctf( ref_prj, ctf ), ctf )
				frange = peak_range( nx, ctf)

				if exp_prj.get_attr('ctf_applied')==0.0:
					exp_prj = sparx_filter.filt_ctf( sparx_fundamentals.fft(exp_prj), ctf )
				else:
					exp_prj = sparx_fundamentals.fft(exp_prj)
				ref_prj = sparx_filter.filt_tophatb( ref_prj, frange[0], frange[1], False )
				exp_prj = sparx_filter.filt_tophatb( exp_prj, frange[0], frange[1], False )
			else:
				exp_prj = sparx_fundamentals.fft(exp_prj)

			ref_prj = sparx_fundamentals.fft(ref_prj)
			exp_prj = sparx_fundamentals.fft(exp_prj)
			EMAN2_cppwrap.Util.mul_img(ref_prj, mask)
			EMAN2_cppwrap.Util.mul_img(exp_prj, mask)
			curtccc = sparx_statistics.ccc( ref_prj, exp_prj, mask )

			try:
				a = exp_prj.dot( ref_prj ) / exp_prj.dot(exp_prj)
			except:
				print('exception at myid, i:', myid, i)
				a = 1.0

			scales.append( a )
			if(verbose == 1):
				info.write( "i, a, ccc:  %4d %10.5f  %10.5f\n" %(i, a, curtccc) )
				info.flush()

		sum_scale = sum( scales )

		if  MPI:
			total_sum_scale = mpi.mpi_reduce( sum_scale, 1, mpi.MPI_FLOAT, mpi.MPI_SUM, 0, mpi.MPI_COMM_WORLD )
			total_sum_scale = mpi.mpi_bcast( total_sum_scale, 1, mpi.MPI_FLOAT, 0, mpi.MPI_COMM_WORLD)
			sum_scale = float(total_sum_scale[0])

		avg_scale = sum_scale/img_number

		assert( len(imgdata)==len(scales) )

		for i in range( len(imgdata) ):
			s = scales[i] / avg_scale
			imgdata[i] *= s
			pred[i] *= s

			scale_file = os.path.join(outdir, "newscale%04d_%04d.txt" % (myid, iter))
			sparx_utilities.drop_spider_doc( scale_file, pred )

		fsc_file = os.path.join(outdir, ( "fsc_%04d.dat" % iter ))
		vol_file = os.path.join(outdir, ( "vol_%04d.hdf" % iter ))
		if(verbose == 1):
			info.write( 'running reconstruction\n' )
			info.flush()
		if  MPI:
			if  CTF:  refvol, fscc = sparx_reconstruction.rec3D_MPI( imgdata, snr, sym, None, fsc_file, myid )
			else:     refvol, fscc = sparx_reconstruction.rec3D_MPI_noCTF( imgdata, sym, None, fsc_file, myid )
			sparx_utilities.bcast_EMData_to_all( refvol, myid )
		else:
			if CTF:   refvol = sparx_reconstruction.recons3d_4nn_ctf( imgdata, list(range(len(imgdata))), snr, 1, sym)
			else:	   refvol = sparx_reconstruction.recons3d_4nn( imgdata, list(range(len(imgdata))), sym, snr = snr)
		if(verbose == 1):
			info.write( 'reconstruction finished\n' )
			info.flush()

		if myid==0:
			sparx_utilities.drop_image( refvol, vol_file )
			if(verbose == 1):
				info.write( "reconstructed volume written to " + vol_file  + "\n")
				info.flush()

		if  MPI:  sparx_utilities.bcast_EMData_to_all( refvol, myid )
		[mean,sigma,fmin,fmax] = EMAN2_cppwrap.Util.infomask( refvol, None, True )
		if(verbose == 1):
			info.write( 'vol all after reconstruction, myid: %d %10.3e %10.3e %10.3e %10.3e\n' % ( myid, mean, sigma, fmin, fmax ) )
			info.flush()

	if(myid == 0):
		foutput = open( os.path.join(outdir,"weights.txt"), "w" )
	if  MPI:
		pass#IMPORTIMPORTIMPORT from mpi import MPI_INT, mpi_recv, mpi_send
		if myid == 0:
			ltot = 0
			base = 0
			for iq in range( nproc ):
				if(iq == 0):
					ltot = spill_out(ltot, base, pred, 1, foutput)
				else:
					lend = mpi.mpi_recv(1, mpi.MPI_INT, iq, sparx_global_def.SPARX_MPI_TAG_UNIVERSAL, mpi.MPI_COMM_WORLD)
					lend = int(lend[0])
					pred = mpi.mpi_recv(lend, mpi.MPI_FLOAT, iq, sparx_global_def.SPARX_MPI_TAG_UNIVERSAL, mpi.MPI_COMM_WORLD)
					ltot = spill_out(ltot, base, pred, 1, foutput)
				base += len(pred)
		else:
			mpi.mpi_send([len(pred)], 1, mpi.MPI_INT, 0, sparx_global_def.SPARX_MPI_TAG_UNIVERSAL, mpi.MPI_COMM_WORLD)
			mpi.mpi_send(pred, len(pred), mpi.MPI_FLOAT, 0, sparx_global_def.SPARX_MPI_TAG_UNIVERSAL, mpi.MPI_COMM_WORLD)
	else:
		ltot = 0
		base = 0
		ltot = spill_out(ltot, base, pred, 1, foutput)

"""Multiline Comment40"""

def defvar(files, outdir, fl, aa, radccc, frepa = "default", pca=False, pcamask=None, pcanvec=None):
	pass#IMPORTIMPORTIMPORT from utilities  import get_im, get_image, circumference, model_blank
	pass#IMPORTIMPORTIMPORT from filter     import filt_tanl
	pass#IMPORTIMPORTIMPORT from math       import sqrt
	pass#IMPORTIMPORTIMPORT import os
	
	if os.path.exists(outdir): sparx_global_def.ERROR('Output directory exists, please change the name and restart the program', " defvar", 1)
	os.mkdir(outdir)

	#finf = open( outdir + "/var_progress.txt", "w" )
	print("  START ")

	img = sparx_utilities.get_image( files[0] )
	nx = img.get_xsize()
	ny = img.get_ysize()
	nz = img.get_zsize()
	if(frepa == "None"):  repair = False
	else:
		repair = True
		if(frepa == "default"):
			#from math import sqrt
			#from utilities import model_gauss
			#rota = model_gauss(sqrt(2.0)*nx, nx,ny,nz)
			#Util.mul_scalar( rota, 1.0/(rota.get_value_at(nx//2, ny//2, nz//2)) )
			pass#IMPORTIMPORTIMPORT from utilities import model_blank
			rota = sparx_utilities.model_blank(nx, ny, nz, 1.0)
		else:   rota = sparx_utilities.get_im(frepa)

	radcir = min(nx,ny,nz)//2 - 2

	if pca :
		pass#IMPORTIMPORTIMPORT from statistics import pcanalyzer
		pcamask = sparx_utilities.get_im( pcamask)
		pcaer = sparx_statistics.pcanalyzer(pcamask, pcanvec, False)

	avgfile  = os.path.join(outdir, "avg.hdf")
	varfile  = os.path.join(outdir, "var.hdf")
	varfileE = os.path.join(outdir, "varE.hdf")
	avgfileE = os.path.join(outdir, "avgE.hdf")
	varfileO = os.path.join(outdir, "varO.hdf")
	avgfileO = os.path.join(outdir, "avgO.hdf")

	if(radccc < 1):  radcir = min(nx,ny,nz)//2-2
	else:            radcir = radccc

	nfiles = len( files )

	avg1 = sparx_utilities.model_blank(nx,ny,nz)
	avg2 = sparx_utilities.model_blank(nx,ny,nz)

	total_img = 0
	mf = 0
	for f in files:
		nimg = EMAN2_cppwrap.EMUtil.get_image_count( f )
		#print f," A  ",nimg
		mf += 1
		for i in range(nimg):
			img = sparx_utilities.get_im( f, i )
			if(fl > 0.0):
				img = sparx_filter.filt_tanl( img, fl, aa )
			if(repair):
				EMAN2_cppwrap.Util.div_img(img, rota) #img = circumference(Util.divn_img(img, rota), radcir)
				if pca:
					pc   = EMAN2_cppwrap.Util.infomask(img, pcamask, True)
					img -= pc[0]
					img *= (refstat[1]/pc[1])
			if(total_img%2 == 0):	EMAN2_cppwrap.Util.add_img(avg1, img)
			else:			EMAN2_cppwrap.Util.add_img(avg2, img)
			total_img += 1

	avg = EMAN2_cppwrap.Util.addn_img(avg1, avg2)
	EMAN2_cppwrap.Util.mul_scalar(avg, 1.0/float(total_img))
	"""Multiline Comment41"""
	avg.write_image( avgfile)

	del avg1, avg2

	pass#IMPORTIMPORTIMPORT from utilities import model_circle
	#cccmask = model_circle(radccc, nx, ny, nz)
	var1 = sparx_utilities.model_blank(nx,ny,nz)
	var2 = sparx_utilities.model_blank(nx,ny,nz)
	for f in files:
		nimg = EMAN2_cppwrap.EMUtil.get_image_count( f )
		for i in range(nimg):
			img = sparx_utilities.get_im( f, i )
			#img = circumference( img, radcir )
			if(fl > 0.0): img = sparx_filter.filt_tanl( img, fl, aa)
			if(repair):  EMAN2_cppwrap.Util.div_img(img, rota) #img = circumference(Util.divn_img(img, rota), radcir)
			if pca:
				pc = EMAN2_cppwrap.Util.infomask(img, pcamask, True)
				img -= pc[0]
				img *= (refstat[1]/pc[1])
				#img += refstat[1]
			if pca : pcaer.insert(img)
			EMAN2_cppwrap.Util.sub_img(img, avg)
			if(total_img%2 == 0): EMAN2_cppwrap.Util.add_img2(var1, img)
			else:                 EMAN2_cppwrap.Util.add_img2(var2 , img)

	var = EMAN2_cppwrap.Util.addn_img(var1, var2)
	EMAN2_cppwrap.Util.mul_scalar(var, 1.0/float(total_img-1) )
	"""Multiline Comment42"""
	sparx_utilities.circumference(var, radcir-1).write_image( varfile )
	del var, var1, var2#, cccmask

	if pca:
		assert not(avg is None)

		pcaer.setavg( avg )

		eigs = pcaer.analyze()
		eigfile = os.path.join(outdir, "eigvol.hdf")
		for i in range( len(eigs) ):
			eigs[i].write_image( eigfile, i )

def var_mpi(files, outdir, fl, aa, radccc, frepa = "default", pca=False, pcamask=None, pcanvec=None):
	pass#IMPORTIMPORTIMPORT from string     import atoi, replace, split, atof
	pass#IMPORTIMPORTIMPORT from utilities  import get_im, circumference, model_circle, model_blank
	pass#IMPORTIMPORTIMPORT from utilities  import bcast_EMData_to_all, reduce_EMData_to_root
	pass#IMPORTIMPORTIMPORT from filter     import filt_tanl
	pass#IMPORTIMPORTIMPORT from mpi        import mpi_comm_rank, mpi_comm_size, mpi_barrier, mpi_bcast, mpi_reduce
	pass#IMPORTIMPORTIMPORT from mpi        import MPI_COMM_WORLD, MPI_INT, MPI_FLOAT, MPI_SUM
	pass#IMPORTIMPORTIMPORT import os
	"""Multiline Comment43"""

	myid = mpi.mpi_comm_rank( mpi.MPI_COMM_WORLD )
	ncpu = mpi.mpi_comm_size( mpi.MPI_COMM_WORLD )
	main_node=0

	if os.path.exists(outdir):  sparx_global_def.ERROR('Output directory exists, please change the name and restart the program', "mref_ali2d_MPI ", 1)
	mpi.mpi_barrier(mpi.MPI_COMM_WORLD)
	if myid== main_node:   os.mkdir(outdir)

	mpi.mpi_barrier( mpi.MPI_COMM_WORLD )

	if( myid == main_node ):
		print("  START ")
		img = sparx_utilities.get_im(files[0])
		nx = img.get_xsize()
		ny = img.get_ysize()
		nz = img.get_zsize()
		#del img
	else:
		nx = 0
		ny = 0
		nz = 0
	nx = mpi.mpi_bcast(nx, 1, mpi.MPI_INT, 0, mpi.MPI_COMM_WORLD)
	nx = int(nx[0])
	ny = mpi.mpi_bcast(ny, 1, mpi.MPI_INT, 0, mpi.MPI_COMM_WORLD)
	ny = int(ny[0])
	nz = mpi.mpi_bcast(nz, 1, mpi.MPI_INT, 0, mpi.MPI_COMM_WORLD)
	nz = int(nz[0])
	if(frepa == "None"):  repair = False
	else:
		repair = True
		if(frepa == "default"):
			#from math import sqrt
			#from utilities import model_gauss
			#rota = model_gauss(sqrt(2.0)*nx, nx,ny,nz)
			#Util.mul_scalar( rota, 1.0/(rota.get_value_at(nx//2, ny//2, nz//2)) )
			pass#IMPORTIMPORTIMPORT from utilities import model_blank
			rota = sparx_utilities.model_blank(nx, ny, nz, 1.0)
		else:   rota = sparx_utilities.get_im(frepa)

	if pca:
		pass#IMPORTIMPORTIMPORT from statistics import pcanalyzer
		if(myid == 0):  pcamask = sparx_utilities.get_im( pcamask)
		else:           pcamask = sparx_utilities.model_blank(nx,ny,nz)
		sparx_utilities.bcast_EMData_to_all(pcamask, myid)
		pcaer = sparx_statistics.pcanalyzer(pcamask, pcanvec, True)
		if( myid == 0 ):  refstat = EMAN2_cppwrap.Util.infomask(img, pcamask, True)
		else:             refstat = [0.0,0.0,0.0,0.0]
		refstat = mpi.mpi_bcast(refstat, 4, mpi.MPI_FLOAT, 0, mpi.MPI_COMM_WORLD)
		refstat = list(map(float, refstat))

	avgfile  = os.path.join(outdir, "avg.hdf")
	varfile  = os.path.join(outdir, "var.hdf")
	varfileE = os.path.join(outdir, "varE.hdf")
	avgfileE = os.path.join(outdir, "avgE.hdf")
	varfileO = os.path.join(outdir, "varO.hdf")
	avgfileO = os.path.join(outdir, "avgO.hdf")

	if(radccc < 1):  radcir = min(nx,ny,nz)//2-2
	else:            radcir = radccc

	nfiles = len( files )
	if(nfiles < ncpu):
		sparx_global_def.ERROR('Number of files less than number of processors specified, reduce number of processors', " var_mpi", 1, myid)
		
	file_start, file_end = MPI_start_end(nfiles, ncpu, myid)

	#ndump = 100 # write out ccc after each 100 steps

	iwrite = 0
	istack = 0
	iprint = 0
	iadded = 0

	avg1 = sparx_utilities.model_blank(nx,ny,nz)
	avg2 = sparx_utilities.model_blank(nx,ny,nz)

	total_img = 0
	for ifile in range(file_start, file_end):
		nimg = EMAN2_cppwrap.EMUtil.get_image_count( files[ifile] )
		#print myid," A  ",files[ifile],"   ",nimg
		for i in range(nimg):
			img = sparx_utilities.get_im( files[ifile], i )
			#img = circumference( img, radcir )
			if(fl > 0.0):
				img = sparx_filter.filt_tanl( img, fl, aa )
			if(repair):
				EMAN2_cppwrap.Util.div_img(img, rota) #img = circumference(Util.divn_img(img, rota), radcir)
				if pca:
					pc   = EMAN2_cppwrap.Util.infomask(img, pcamask, True)
					img -= pc[0]
					img *= (refstat[1]/pc[1])
			if(total_img%2 == 0):	EMAN2_cppwrap.Util.add_img(avg1, img)
			else:			EMAN2_cppwrap.Util.add_img(avg2, img)
			total_img += 1
	sparx_utilities.reduce_EMData_to_root(avg1, myid)
	sparx_utilities.reduce_EMData_to_root(avg2, myid)
	total_img = mpi.mpi_reduce(total_img, 1, mpi.MPI_INT, mpi.MPI_SUM, 0, mpi.MPI_COMM_WORLD)
	if( myid == 0) :
		total_img = int(total_img[0])
		avg = EMAN2_cppwrap.Util.addn_img(avg1, avg2)
		EMAN2_cppwrap.Util.mul_scalar(avg, 1.0/float(total_img))
	else:    avg = sparx_utilities.model_blank(nx,ny,nz)
	sparx_utilities.bcast_EMData_to_all( avg, myid )
	if( myid == 0 ):
		#Util.mul_scalar(avg1, 1.0/float(total_img//2+total_img%2 - 1 ))
		#avg1.write_image(avgfileE)
		#Util.mul_scalar(avg2, 1.0/float(total_img//2 - 1) )
		#avg2.write_image(avgfileO)
		avg.write_image( avgfile)

	del avg1, avg2

	var1 = sparx_utilities.model_blank(nx,ny,nz)
	var2 = sparx_utilities.model_blank(nx,ny,nz)
	for ifile in range(file_start, file_end):
		nimg = EMAN2_cppwrap.EMUtil.get_image_count( files[ifile] )
		#print myid," V  ",files[ifile],"   ",nimg
		for i in range(nimg):
			img = sparx_utilities.get_im( files[ifile], i )
			#img = circumference( img, radcir )
			if(fl > 0.0): img = sparx_filter.filt_tanl( img, fl, aa)
			if(repair):  EMAN2_cppwrap.Util.div_img(img, rota) #img = circumference(Util.divn_img(img, rota), radcir)
			if pca:
				pc = EMAN2_cppwrap.Util.infomask(img, pcamask, True)
				img -= pc[0]
				img *= (refstat[1]/pc[1])
				#img += refstat[1]
			if pca : pcaer.insert(img)
			EMAN2_cppwrap.Util.sub_img(img, avg)
			if(total_img%2 == 0): EMAN2_cppwrap.Util.add_img2(var1, img)
			else:                 EMAN2_cppwrap.Util.add_img2(var2, img)

	sparx_utilities.reduce_EMData_to_root(var1, myid)
	sparx_utilities.reduce_EMData_to_root(var2, myid)
	if( myid == 0):
		var = EMAN2_cppwrap.Util.addn_img(var1, var2)
		EMAN2_cppwrap.Util.mul_scalar(var, 1.0/float(total_img-1) )
	else:    var = sparx_utilities.model_blank(nx,ny,nz)
	sparx_utilities.bcast_EMData_to_all( var, myid )
	if(  (myid == 0)):
		#Util.mul_scalar(var1, 1.0/float(total_img//2+total_img%2 - 1 ))
		#circumference(var1, radcir-1).write_image(varfileE)
		#Util.mul_scalar(var2, 1.0/float(total_img//2 - 1) )
		#circumference(var2, radcir-1).write_image(varfileO)

		sparx_utilities.circumference(var, radcir-1).write_image( varfile )
		del var1, var2
	del var

	if pca:
		assert not(avg is None)

		pcaer.setavg( avg )

		eigs = pcaer.analyze()

		if myid==0:
			eigfile = os.path.join(outdir, "eigvol.hdf")
			for i in range( len(eigs) ):
				eigs[i].write_image( eigfile, i )

def factcoords_vol( vol_stacks, avgvol_stack, eigvol_stack, prefix, rad = -1, neigvol = -1, fl=0.0, aa=0.0, MPI=False):
	pass#IMPORTIMPORTIMPORT from utilities import get_im, model_circle, model_blank

	if MPI:
		pass#IMPORTIMPORTIMPORT from mpi import mpi_comm_rank, mpi_comm_size, MPI_COMM_WORLD
		ncpu = mpi.mpi_comm_size(mpi.MPI_COMM_WORLD)
		myid = mpi.mpi_comm_rank(mpi.MPI_COMM_WORLD)
	else:
		ncpu = 1
		myid = 0


	if( myid == 0 ):
		foutput = open( prefix+".txt", "w" )

	
	if(neigvol < 0):
		eigvols = EMAN2_cppwrap.EMData.read_images(eigvol_stack)
		neigvol = len(eigvols)
	else:
		eigvols = EMAN2_cppwrap.EMData.read_images(eigvol_stack,list(range(neigvol)))

	pass#IMPORTIMPORTIMPORT from math import sqrt
	eigvals = [0.0]*neigvol
	for j in range(neigvol):
		eigvals[j] = numpy.sqrt( eigvols[j].get_attr_default('eigval',1.0) )
		EMAN2_cppwrap.Util.mul_scalar(eigvols[j] , eigvals[j])
	if( avgvol_stack != None):
		avgvol = sparx_utilities.get_im( avgvol_stack )

	nx = eigvols[0].get_xsize()
	ny = eigvols[0].get_ysize()
	nz = eigvols[0].get_zsize()

	m = sparx_utilities.model_circle( rad, nx, ny, nz )
	files = file_set( vol_stacks )
	vol_bgn,vol_end = MPI_start_end( files.nimg(), ncpu, myid )

	d = []
	for i in range( vol_bgn, vol_end ):
		fname,imgid = files.get( i )
		exp_vol = sparx_utilities.get_im( fname, imgid )
		if(avgvol_stack != None):  
			EMAN2_cppwrap.Util.sub_img(exp_vol, avgvol)

		for j in range( neigvol ):
			d.append( exp_vol.cmp( "ccc", eigvols[j], {"negative":0, "mask":m} )*eigvals[j] )

	if  MPI:
		pass#IMPORTIMPORTIMPORT from mpi import MPI_INT, MPI_FLOAT, MPI_COMM_WORLD, mpi_recv, mpi_send
		if myid == 0:
			ltot = 0
			base = 0
			for iq in range( ncpu ):
				if(iq == 0):
					ltot = spill_out(ltot, base, d, neigvol, foutput)
				else:
					lend = mpi.mpi_recv(1, mpi.MPI_INT, iq, sparx_global_def.SPARX_MPI_TAG_UNIVERSAL, mpi.MPI_COMM_WORLD)
					lend = int(lend[0])
					d = mpi.mpi_recv(lend, mpi.MPI_FLOAT, iq, sparx_global_def.SPARX_MPI_TAG_UNIVERSAL, mpi.MPI_COMM_WORLD)
					ltot = spill_out(ltot, base, d, neigvol, foutput)
				base += len(d)/neigvol
		else:
			mpi.mpi_send([len(d)], 1, mpi.MPI_INT, 0, sparx_global_def.SPARX_MPI_TAG_UNIVERSAL, mpi.MPI_COMM_WORLD)
			mpi.mpi_send(d, len(d), mpi.MPI_FLOAT, 0, sparx_global_def.SPARX_MPI_TAG_UNIVERSAL, mpi.MPI_COMM_WORLD)
	else:
		ltot = 0
		base = 0
		ltot = spill_out(ltot, base, d, neigvol, foutput)

def factcoords_prj( prj_stacks, avgvol_stack, eigvol_stack, prefix, rad, neigvol, fl=0.0, aa=0.0, CTF = False, MPI=False):
	pass#IMPORTIMPORTIMPORT from utilities    import get_im, get_image, model_circle, model_blank, get_params_proj
	pass#IMPORTIMPORTIMPORT from projection   import prgs, prep_vol
	pass#IMPORTIMPORTIMPORT from filter       import filt_ctf, filt_tanl
	pass#IMPORTIMPORTIMPORT from statistics   import im_diff

	if MPI:
		pass#IMPORTIMPORTIMPORT from mpi import mpi_comm_rank, mpi_comm_size, MPI_COMM_WORLD
		ncpu = mpi.mpi_comm_size( mpi.MPI_COMM_WORLD )
		myid = mpi.mpi_comm_rank( mpi.MPI_COMM_WORLD )
	else:
		ncpu = 1
		myid = 0

	if(myid == 0):
		foutput = open( prefix+".txt", "w" )

	nx = sparx_utilities.get_im( prj_stacks[0] ).get_xsize()
	ny = nx

	avgvol = sparx_utilities.get_im( avgvol_stack )
	a = EMAN2_cppwrap.Util.infomask(avgvol, sparx_utilities.model_circle(int(rad), nx, nx, nx), False)
	avgvol -= a[0]
	avgvol,kb = sparx_projection.prep_vol( avgvol )

	if neigvol==-1:  neigvol = EMAN2_cppwrap.EMUtil.get_image_count( eigvol_stack )
	# average volumes and eigen volumes.
	eigvols = [None]*neigvol
	eigvals = [0.0]*neigvol
	pass#IMPORTIMPORTIMPORT from math import sqrt
	for j in range(neigvol):
		eigvols[j] = sparx_utilities.get_im(eigvol_stack, j)
		eigvals[j] = numpy.sqrt( eigvols[j].get_attr('eigval') )
		eigvols[j], kb = sparx_projection.prep_vol( eigvols[j] )

	m = sparx_utilities.model_circle( int(rad), nx, ny )

	files = file_set( prj_stacks )
	nprj  = files.nimg()
	img_bgn, img_end = MPI_start_end( nprj, ncpu, myid )
	#ltot = -1
	d = []
	for i in range( img_bgn, img_end ):
		fname,imgid = files.get(i)
		#if(i%1000 == 0):  print  "  ",myid,"   ",i
		exp_prj = sparx_utilities.get_im( fname, imgid )

		phi,theta,psi,s2x,s2y = sparx_utilities.get_params_proj(exp_prj)
		if CTF:  ctf = exp_prj.get_attr("ctf")

		ref_prj = sparx_projection.prgs( avgvol, kb, [phi, theta, psi, -s2x, -s2y] )
		if  CTF:  ref_prj = sparx_filter.filt_ctf( ref_prj, ctf )
		if(fl > 0.0):  exp_prj = sparx_filter.filt_tanl(exp_prj, fl, aa)
		#ltot += 1
		#ref_prj.write_image("projection.hdf",ltot)
		diff,a,b = sparx_statistics.im_diff( ref_prj, exp_prj, m)
		#nrmd = diff.cmp( "dot", diff, {"negative":0, "mask":m} )  #CHANGED HERE
		#diff.write_image("difference.hdf",ltot)

		for j in range( neigvol ) :

			ref_eigprj = sparx_projection.prgs( eigvols[j], kb, [phi, theta, psi, -s2x, -s2y] )
			if  CTF:  ref_eigprj = sparx_filter.filt_ctf( ref_eigprj, ctf )
			#ref_eigprj.write_image("eigen.hdf",ltot)

			#d.append( diff.cmp( "dot", ref_eigprj, {"negative":0, "mask":m} )*eigvals[j]/nrmd )   CHANGED HERE
			d.append( diff.cmp( "ccc", ref_eigprj, {"negative":0, "mask":m} ))#*eigvals[j] )
		#print  i,j,d[-1]
	if  MPI:
		pass#IMPORTIMPORTIMPORT from mpi import MPI_INT, MPI_FLOAT, MPI_COMM_WORLD, mpi_recv, mpi_send
		if myid == 0:
			ltot = 0
			base = 0
			for iq in range( ncpu ):
				if(iq == 0):
					ltot = spill_out(ltot, base, d, neigvol, foutput)
				else:
					lend = mpi.mpi_recv(1, mpi.MPI_INT, iq, sparx_global_def.SPARX_MPI_TAG_UNIVERSAL, mpi.MPI_COMM_WORLD)
					lend = int(lend[0])
					d = mpi.mpi_recv(lend, mpi.MPI_FLOAT, iq, sparx_global_def.SPARX_MPI_TAG_UNIVERSAL, mpi.MPI_COMM_WORLD)
					ltot = spill_out(ltot, base, d, neigvol, foutput)
				base += len(d)/neigvol
		else:
			mpi.mpi_send([len(d)], 1, mpi.MPI_INT, 0, sparx_global_def.SPARX_MPI_TAG_UNIVERSAL, mpi.MPI_COMM_WORLD)
			mpi.mpi_send(d, len(d), mpi.MPI_FLOAT, 0, sparx_global_def.SPARX_MPI_TAG_UNIVERSAL, mpi.MPI_COMM_WORLD)
	else:
		ltot = 0
		base = 0
		ltot = spill_out(ltot, base, d, neigvol, foutput)

def spill_out(ltot, base, d, neigvol, foutput):
	loc = 0
	for i in range(len(d)//neigvol):
		for j in range( neigvol):
			foutput.write( "    %e" % d[loc] )
			ltot += 1
			loc  += 1
		foutput.write( "    %d" % (i+base) )
		foutput.write( "\n" )
	foutput.flush()
	return  ltot

def k_means_main(stack, out_dir, maskname, opt_method, K, rand_seed, maxit, trials, critname,
		 CTF = False, F = 0, T0 = 0, MPI = False, CUDA = False, DEBUG = False, flagnorm = False,
		 init_method = 'rnd'):
	# Common
	pass#IMPORTIMPORTIMPORT from utilities   import print_begin_msg, print_end_msg, print_msg, file_type, running_time
	pass#IMPORTIMPORTIMPORT from statistics  import k_means_locasg2glbasg
	pass#IMPORTIMPORTIMPORT from time        import time
	pass#IMPORTIMPORTIMPORT import sys, os
	#import time
	if MPI:
		pass#IMPORTIMPORTIMPORT from mpi        import mpi_init, mpi_comm_size, mpi_comm_rank, mpi_barrier
		pass#IMPORTIMPORTIMPORT from mpi        import MPI_COMM_WORLD, MPI_INT, mpi_bcast
		pass#IMPORTIMPORTIMPORT from mpi	import MPI_FLOAT, MPI_INT, mpi_recv, mpi_send
		pass#IMPORTIMPORTIMPORT from utilities  import bcast_number_to_all, recv_EMData, send_EMData
		
	if CUDA:
		pass#IMPORTIMPORTIMPORT from statistics import k_means_cuda_init_open_im, k_means_cuda_headlog
		pass#IMPORTIMPORTIMPORT from statistics import k_means_cuda_export
		pass#IMPORTIMPORTIMPORT if MPI: from statistics import k_means_CUDA_MPI
		pass#IMPORTIMPORTIMPORT else:   from statistics import k_means_CUDA, k_means_SSE_CUDA
	else:
		pass#IMPORTIMPORTIMPORT from statistics import k_means_init_open_im, k_means_open_im, k_means_headlog
		pass#IMPORTIMPORTIMPORT from statistics import k_means_criterion, k_means_export
		pass#IMPORTIMPORTIMPORT if MPI: from statistics import k_means_cla, k_means_SSE_MPI
		pass#IMPORTIMPORTIMPORT else:   from statistics import k_means_cla, k_means_SSE

	ext = sparx_utilities.file_type(stack)
	if ext == 'txt': TXT = True
	else:            TXT = False

	if (T0 == 0 and F != 0) or (T0 != 0 and F == 0):
		sparx_global_def.ERROR('Ambigues parameters F=%f T0=%f' % (F, T0), 'k_means_main', 1)
		sys.exit()

	if MPI:
		sys.argv  = mpi.mpi_init(len(sys.argv), sys.argv)
		ncpu      = mpi.mpi_comm_size(mpi.MPI_COMM_WORLD)
		myid      = mpi.mpi_comm_rank(mpi.MPI_COMM_WORLD)
		main_node = 0
		mpi.mpi_barrier(mpi.MPI_COMM_WORLD)

		if os.path.exists(out_dir): sparx_global_def.ERROR('Output directory exists, please change the name and restart the program', "k_means_main ", 1)
		mpi.mpi_barrier(mpi.MPI_COMM_WORLD)

	else:
		if os.path.exists(out_dir): sparx_global_def.ERROR('Output directory exists, please change the name and restart the program', "k_means_main ", 1)

	if MPI and not CUDA:
		
		if myid == main_node:	sparx_utilities.print_begin_msg('k-means')
		mpi.mpi_barrier(mpi.MPI_COMM_WORLD)
		
		LUT, mask, N, m, Ntot = sparx_statistics.k_means_init_open_im(stack, maskname)
		N_min = N
		
		
		
		IM, ctf, ctf2         = sparx_statistics.k_means_open_im(stack, mask, CTF, LUT, flagnorm)
		
		
		if myid == main_node: 
			sparx_statistics.k_means_headlog(stack, out_dir, opt_method, N, K, 
						      critname, maskname, ncpu, maxit, CTF, T0, 
						      F, rand_seed, ncpu, m)
			t_start = time.time()
		
		[Cls, assign, Je] = sparx_statistics.k_means_SSE_MPI(IM, mask, K, rand_seed, maxit, 
					1, [CTF, ctf, ctf2], F, T0, DEBUG, init_method, myid = myid, main_node = main_node, jumping = 1)
					
				
		
		pass#IMPORTIMPORTIMPORT from statistics import k_means_SSE_combine
		[ assign_return, r_Cls, je_return, n_best] = sparx_statistics.k_means_SSE_combine(Cls, assign, Je, N, K, ncpu, myid, main_node)
		mpi.mpi_barrier(mpi.MPI_COMM_WORLD)
		if myid == main_node:
		
			if n_best == -1:
				sparx_utilities.print_msg('>>> WARNING: All trials resulted in empty clusters, STOP k-means.\n\n')
				sparx_utilities.print_end_msg('k-means MPI end')
				sparx_utilities.running_time(t_start)	
			#print "assign_return===", assign_return[10:20], "cls_n return==", r_Cls['n'], "Ji==", r_Cls['Ji'], "ave size ==", r_Cls['ave'][0].get_xsize()
			else:
				for i in range( ncpu ):
					if( je_return[i] <0 ):
						sparx_utilities.print_msg('> Trials: %5d    resulted in empty clusters  \n' % (i) )
					else:
						sparx_utilities.print_msg('> Trials: %5d    criterion: %11.6e  \n' % (i, je_return[i]) )
				sparx_utilities.running_time(t_start)
				crit = sparx_statistics.k_means_criterion(r_Cls, critname)
				glb_assign = sparx_statistics.k_means_locasg2glbasg(assign_return, LUT, Ntot)
				sparx_statistics.k_means_export(r_Cls, crit, glb_assign, out_dir, -1, TXT)
				sparx_utilities.print_end_msg('k-means MPI end')
	
	
	
	#don't touch below code

	elif CUDA and not MPI: # added 2009-02-20 16:27:26 # modify 2009-09-23 13:52:29
		sparx_utilities.print_begin_msg('k-means')
		LUT, mask, N, m, Ntot = sparx_statistics.k_means_cuda_init_open_im(stack, maskname)
		sparx_statistics.k_means_cuda_headlog(stack, out_dir, 'cla', N, K, maskname, maxit, T0, F, rand_seed, 1, m)
		if   opt_method == 'cla':
			sparx_statistics.k_means_CUDA(stack, mask, LUT, m, N, Ntot, K, maxit, F, T0, rand_seed, out_dir, TXT, 1, flagnorm=flagnorm)
		else:
			sparx_statistics.k_means_SSE_CUDA(stack, mask, LUT, m, N, Ntot, K, maxit, F, T0, rand_seed, out_dir, TXT, 1, flagnorm=flagnorm)
		sparx_utilities.print_end_msg('k-means')
	#don't touch below code
	elif MPI and CUDA: # added 2009-09-22 14:34:45
		print("tao mpi and cuda")
		LUT, mask, N, m, Ntot = sparx_statistics.k_means_cuda_init_open_im(stack, maskname)
		if myid == main_node:
			sparx_utilities.print_begin_msg('k-means')
			sparx_statistics.k_means_cuda_headlog(stack, out_dir, 'cuda', N, K, maskname, maxit, T0, F, rand_seed, ncpu, m)
		sparx_statistics.k_means_CUDA_MPI(stack, mask, LUT, m, N, Ntot, K, maxit, F, T0, rand_seed, myid, main_node, ncpu, out_dir, TXT, 1, flagnorm=flagnorm)
		if myid == main_node:
			sparx_utilities.print_end_msg('k-means')

		mpi.mpi_barrier(mpi.MPI_COMM_WORLD)

	else:
		sparx_utilities.print_begin_msg('k-means')
		LUT, mask, N, m, Ntot = sparx_statistics.k_means_init_open_im(stack, maskname)
		IM, ctf, ctf2         = sparx_statistics.k_means_open_im(stack, mask, CTF, LUT, flagnorm)
		sparx_statistics.k_means_headlog(stack, out_dir, opt_method, N, K, critname, maskname, trials, maxit, 
					CTF, T0, F, rand_seed, 1, m, init_method)
		
		if   opt_method == 'cla':
			[Cls, assign] = sparx_statistics.k_means_cla(IM, mask, K, rand_seed, maxit, 
					trials, [CTF, ctf, ctf2], F, T0, DEBUG, init_method)
		elif opt_method == 'SSE':
			[Cls, assign] = sparx_statistics.k_means_SSE(IM, mask, K, rand_seed, maxit, 
					trials, [CTF, ctf, ctf2], F, T0, DEBUG, init_method)
		else:
			sparx_global_def.ERROR('opt_method %s unknown!' % opt_method, 'k_means_main', 1,myid)
			sys.exit()
		crit = sparx_statistics.k_means_criterion(Cls, critname)
		glb_assign = sparx_statistics.k_means_locasg2glbasg(assign, LUT, Ntot)
		sparx_statistics.k_means_export(Cls, crit, glb_assign, out_dir, -1, TXT)

		sparx_utilities.print_end_msg('k-means')


# -- K-means groups ---------------------------------------------------------------------------
			
# K-means groups driver
def k_means_groups(stack, out_file, maskname, opt_method, K1, K2, rand_seed, maxit, trials, CTF=False, F=0.0, T0=0.0, MPI=False, CUDA=False, DEBUG=False, flagnorm=False):

	#import os
	#if os.path.exists(out_file):
		#ERROR('Output directory exists, please change the name and restart the program', "k_means_groups", 1)
	
	if MPI:
		#print "MPI version of kmeans group is under development"
		#sys.exit()
		pass#IMPORTIMPORTIMPORT from statistics import k_means_groups_MPI
		sparx_statistics.k_means_groups_MPI(stack, out_file, maskname, opt_method, K1, K2, rand_seed, maxit, trials, CTF, F, T0, flagnorm)
	elif CUDA:
		pass#IMPORTIMPORTIMPORT import os
		if os.path.exists(out_file):
			sparx_global_def.ERROR('Output directory exists, please change the name and restart the program', "k_means_groups_CUDA", 1)
		pass#IMPORTIMPORTIMPORT from statistics import k_means_groups_CUDA
		sparx_statistics.k_means_groups_CUDA(stack, out_file, maskname, K1, K2, rand_seed, maxit, F, T0)
	else:
		pass#IMPORTIMPORTIMPORT import os
		if os.path.exists(out_file):
			sparx_global_def.ERROR('Output directory exists, please change the name and restart the program', "k_means_groups_serial", 1)
		pass#IMPORTIMPORTIMPORT from statistics import k_means_groups_serial
		sparx_statistics.k_means_groups_serial(stack, out_file, maskname, opt_method, K1, K2, rand_seed, maxit, trials, CTF, F, T0, DEBUG, flagnorm)



# 2008-12-08 12:46:11 JB
# Plot angles distribution on a hemisphere from a list of given projection
def plot_projs_distrib(stack, outplot, wnx = 256):
	pass#IMPORTIMPORTIMPORT from projection import plot_angles
	pass#IMPORTIMPORTIMPORT from utilities  import get_params_proj, file_type, read_text_row
	pass#IMPORTIMPORTIMPORT import sys

	ext  = sparx_utilities.file_type(stack)
	if ext == 'txt' :
		agls = sparx_utilities.read_text_row(stack)
	elif ext == 'bdb' :
		pass#IMPORTIMPORTIMPORT from EMAN2db import db_open_dict
		N  = EMAN2_cppwrap.EMUtil.get_image_count(stack)
		DB = EMAN2db.db_open_dict(stack)
		agls = []
		for n in range(N):
			im = EMAN2_cppwrap.EMData()
			im.read_image(stack, n, True)
			try:
				p0, p1, p2, p3, p4 = sparx_utilities.get_params_proj(im)
			except RuntimeError:
				print('Projection #%d from %s has no angles set!' % (n, stack))
				sys.exit()
			agls.append([p0, p1])

		if ext == 'bdb': DB.close()

	sparx_projection.plot_angles(agls, wnx).write_image(outplot, 0)

# 2008-12-08 12:46:46 JB
# Wrap for the HAC part of py_cluster in the statistics.py file
def HAC_clustering(stack, dendoname, maskname, kind_link, kind_dist, flag_diss):
	pass#IMPORTIMPORTIMPORT from statistics   import ccc, py_cluster_HierarchicalClustering
	pass#IMPORTIMPORTIMPORT from copy         import deepcopy
	pass#IMPORTIMPORTIMPORT from utilities    import get_im, get_params2D, get_params3D
	pass#IMPORTIMPORTIMPORT from fundamentals import rot_shift2D, rot_shift3D

	N    = EMAN2_cppwrap.EMUtil.get_image_count(stack)
	if maskname != None: mask = sparx_utilities.get_im(maskname)
	else:                mask = None

	IM = EMAN2_cppwrap.EMData.read_images(stack)
	ny = IM[0].get_ysize()
	nz = IM[0].get_zsize()
	for n in range(N):
		# 3D object
		if nz > 1:
			phi, theta, psi, s3x, s3y, s3z, mirror, scale = sparx_utilities.get_params3D(IM[n])
			IM[n]  = sparx_fundamentals.rot_shift3D(IM[n], phi, theta, psi, s3x, s3y, s3z, scale)
			if mirror: IM[n].process_inplace('xform.mirror', {'axis':'x'})
		# 2D object
		elif ny > 1:
			alpha, sx, sy, mirror, scale = sparx_utilities.get_params2D(IM[n])
			IM[n] = sparx_fundamentals.rot_shift2D(IM[n], alpha, sx, sy, mirror, scale)

		if mask != None: EMAN2_cppwrap.Util.mul_img(IM[n], mask)
		IM[n].set_attr('ID_hclus', n)

	if kind_dist   == 'SqEuc':
		if flag_diss: cl = sparx_statistics.py_cluster_HierarchicalClustering(IM, lambda x,y: -x.cmp("SqEuclidean", y), linkage = kind_link)
		else:        cl = sparx_statistics.py_cluster_HierarchicalClustering(IM, lambda x,y: x.cmp("SqEuclidean", y), linkage = kind_link)
	elif kind_dist == 'CCC':
		if flag_diss: cl = sparx_statistics.py_cluster_HierarchicalClustering(IM, lambda x,y: -sparx_statistics.ccc(x, y, mask), linkage = kind_link)
		else:        cl = sparx_statistics.py_cluster_HierarchicalClustering(IM, lambda x,y: sparx_statistics.ccc(x, y, mask), linkage = kind_link)

	k     = N
	Dendo = {}
	doc   = open(dendoname + '.txt', 'w')
	for val in range(0, 10000):
		if flag_diss: th  = -(val / 1000.0)
		else:         th  = val / 1000.0
	
		res = cl.getlevel(th)

		GP = []
		for gp in res:
			OBJ = []
			for obj in gp:
				OBJ.append(obj.get_attr('ID_hclus'))
		GP.append(OBJ)

		newk = len(GP)
		if newk != k:
			k = newk
			doc.write('%7.3f %d\n' % (th, k))
			Dendo[k] = copy.deepcopy(GP)
	doc.close()

	pass#IMPORTIMPORTIMPORT import pickle
	f = open(dendoname + '.dendo', 'w')
	pickle.dump(Dendo, f)
	f.close()

# 2008-12-08 15:20:24 JB
# Compute the averages from the dendogram given by the function HAC_clustering
def HAC_averages(stack, dendoname, avename, K):
	pass#IMPORTIMPORTIMPORT from utilities    import get_im, get_params2D, get_params3D
	pass#IMPORTIMPORTIMPORT from fundamentals import rot_shift2D, rot_shift3D
	pass#IMPORTIMPORTIMPORT from utilities    import model_blank
	pass#IMPORTIMPORTIMPORT import sys
	
	N    = EMAN2_cppwrap.EMUtil.get_image_count(stack)

	try:
		pass#IMPORTIMPORTIMPORT import pickle
		f = open(dendoname, 'r')
		Dendo = pickle.load(f)
		f.close()
	except:
		print('Impossible to read dendogram structure.')
		sys.exit()

	list_k = list(Dendo.keys())
	if K not in list_k:
		print('Dendogram does not contain the draw for K=%d' % K)
		sys.exit()

	IM = EMAN2_cppwrap.EMData.read_images(stack)
	nx = IM[0].get_xsize()
	ny = IM[0].get_ysize()
	nz = IM[0].get_zsize()
	for n in range(N):
		# 3D object
		if nz > 1:
			phi, theta, psi, s3x, s3y, s3z, mirror, scale = sparx_utilities.get_params3D(IM[n])
			IM[n]  = sparx_fundamentals.rot_shift3D(IM[n], phi, theta, psi, s3x, s3y, s3z, scale)
			if mirror: IM[n].process_inplace('xform.mirror', {'axis':'x'})
		# 2D object
		elif ny > 1:
			alpha, sx, sy, mirror, scale = sparx_utilities.get_params2D(IM[n])
			IM[n] = sparx_fundamentals.rot_shift2D(IM[n], alpha, sx, sy, mirror, scale)

	part = Dendo[K]
	AVE  = []
	ct   = 0
	for k in range(K):
		AVE.append(sparx_utilities.model_blank(nx, ny, nz))
		nobj = len(part[k])
		if nobj > 1:
			for id in part[k]: EMAN2_cppwrap.Util.add_img(AVE[k], IM[id])
			EMAN2_cppwrap.Util.mul_scalar(AVE[k], 1 / float(len(part[k])))
			AVE[k].set_attr('nobjects', len(part[k]))
			AVE[k].set_attr('members',  part[k])
			AVE[k].write_image(avename, k)
		else:
			im = IM[part[k][0]].copy()
			im.set_attr('nobjects', 1)
			im.set_attr('members', part[k][0])
			im.write_image(avename, k)


def tomo(box):
	print("box: ", box)
	cmd = "your_executable -box " + box

	print("cmd: ", cmd)
	pass#IMPORTIMPORTIMPORT from os import system
	status = os.system( cmd )
	print('status: ', status)

# Calculate averages of a given stack (wrap for ave_var in statistics)
def ave_ali(name_stack, name_out = None, ali = False, param_to_save_size = None, set_as_member_id = None):
	pass#IMPORTIMPORTIMPORT from statistics import ave_var, add_ave_varf, k_means_list_active
	pass#IMPORTIMPORTIMPORT from utilities  import file_type
	"""
	   This function is called by sxave_ali.py
	"""
	N = EMAN2_cppwrap.EMUtil.get_image_count(name_stack)
	if ali:
		mode = 'a'
	else:
		mode = ''

	# # horatio active_refactoring Jy51i1EwmLD4tWZ9_00002_1	
	# if active: 
	# 	listID, N = k_means_list_active(name_stack)
	# else:
	# 	listID    = range(N)

	# # horatio active_refactoring Jy51i1EwmLD4tWZ9_00002_2	
	listID    = list(range(N))

	
	ave, var = sparx_statistics.ave_var(name_stack, mode, listID)
	nlistID = len(listID)
	if param_to_save_size:
		ave.set_attr(param_to_save_size, nlistID)
	
	if set_as_member_id:
		members = []
		for i in range(nlistID):
			in_img = sparx_utilities.get_im(name_stack, listID[i])
			members.append( int(in_img.get_attr(set_as_member_id)) )
			
		ave.set_attr("members", members)

	ext = sparx_utilities.file_type(name_stack)
	if name_out is None:
		if ext == 'bdb': name = name_stack.split(':')[1] + '.hdf'
		else:            name = name_stack
		ave.write_image('ave_' + name, 0)
	else:
		ave.write_image(name_out, 0)

"""Multiline Comment44"""
def Xwithin_group_refinement(data, maskfile, randomize, ir, ou, rs, xrng, yrng, step, dst, maxit, FH, FF, method = ""):

	# Comment by Zhengfan Yang 03/11/11
	# This is a simple version of ali2d_data (down to the bone), no output dir, no logfile, no CTF, no MPI or CUDA, no Fourvar, no auto stop, no user function

	pass#IMPORTIMPORTIMPORT from alignment    import Numrinit, ringwe, ali2d_single_iter
	pass#IMPORTIMPORTIMPORT from filter	      import filt_tanl
	pass#IMPORTIMPORTIMPORT from fundamentals import fshift, fft
	pass#IMPORTIMPORTIMPORT from random	      import randint, random
	pass#IMPORTIMPORTIMPORT from statistics   import ave_series
	pass#IMPORTIMPORTIMPORT from utilities    import get_input_from_string, model_circle, center_2D
	pass#IMPORTIMPORTIMPORT from utilities    import set_params2D, get_params2D, combine_params2, inverse_transform2

	first_ring=int(ir); last_ring=int(ou); rstep=int(rs); max_iter=int(maxit);
	nima = len(data)
	nx = data[0].get_xsize()
	if last_ring == -1:  last_ring = nx/2-2
	if maskfile: mask = maskfile
	else:        mask = sparx_utilities.model_circle(last_ring, nx, nx)

	lx = [0]*nima
	ly = [0]*nima
	for im in range(nima):
		alpha, sx, sy, mirrorn, dummy = sparx_utilities.get_params2D(data[im])
		alphai, sxi, syi, dummy    = sparx_utilities.combine_params2(0.0, sx, sy, 0, -alpha, 0.,0.,0)
		lx[im] = int(round(sxi,0))
		ly[im] = int(round(syi,0))
		EMAN2_cppwrap.Util.cyclicshift(data[im] , {"dx":lx[im],"dy":ly[im]})
		sxi -= lx[im]
		syi -= ly[im]
		if randomize :
			alphan, sxn, syn, mirrorn    = sparx_utilities.combine_params2(0.0, sxi, syi, 0, numpy.random.random()*360.0, 0.0, 0.0, random.randint(0, 1))
			#alphan, sxn, syn, mirrorn = combine_params2(0.0, randint(-xrng[0],xrng[0]), randint(-xrng[0],xrng[0]), 0, random()*360.0, 0, 0, randint(0, 1))
		else:
			alphan, sxn, syn, dummy = sparx_utilities.combine_params2(0.0, sxi, syi, 0, alpha, 0.0, 0.0, 0)
		sparx_utilities.set_params2D(data[im], [alpha, sxn, syn, mirrorn, 1.0])
		

	cnx = nx/2+1
	cny = cnx
	mode = "F"
	numr = sparx_alignment.Numrinit(first_ring, last_ring, rstep, mode)
	wr = sparx_alignment.ringwe(numr, mode)

	sx_sum = 0.0
	sy_sum = 0.0
	cs = [0.0]*2
	total_iter = 0
	if(method == "SHC"):
		#  This is my failed attempt to use SHC for 2D alignment.  
		#    Inexplicably, it did not do all that well.  While initially it converges fast
		#     and generally yields a very good solution, it converges to cluster of images scattered
		#     around the 'best' solution, i.e., the shifts are within a fraction of a pixel of what they
		#     should be and, as a result, some are in wrong positions and overall pixel error is large.
		#     Overall, Yang's method works much better, so I am leaving it at that.  PAP 01/22/2015
		for im in data:  im.set_attr('previousmax', -1.0e23)
		tavg = sparx_statistics.ave_series(data)
		for N_step in range(len(xrng)):
			nope = 0
			Iter = 0
			while(nope < len(data)//1 and Iter < max_iter ):
				total_iter += 1
				Iter += 1
				if( FH > 0.0):
					tavg = sparx_filter.filt_tanl(sparx_fundamentals.fft(tavg), FH, FF)
					if( xrng[0] > 0.0 ): cs[0] = sx_sum/float(nima)
					if( yrng[0] > 0.0 ): cs[1] = sy_sum/float(nima)
					tavg = sparx_fundamentals.fft(sparx_fundamentals.fshift(tavg, -cs[0], -cs[1]))
				else:
					tavg = sparx_filter.filt_tanl(tavg, FH, FF)
				sx_sum, sy_sum, nope = sparx_alignment.ali2d_single_iter(data, numr, wr, cs, tavg, cnx, cny, \
															xrng[N_step], yrng[N_step], step[N_step], \
															mode=mode, CTF=False, random_method = method)
				#print  "  iteration  shc   %03d   %03d   %7.2f    %7.2f  "%(total_iter,nope,cs[0],cs[1])
				#print total_iter,nope
				#for i in data:  print "  ",i.get_attr('previousmax'),
				#print "  "
				#tavg.write_image('tata.hdf',total_iter-1)
				tavg = sparx_statistics.ave_series(data)
		"""Multiline Comment45"""


	elif( method == "PCP"):
		pass#IMPORTIMPORTIMPORT from isac import prepref
		pass#IMPORTIMPORTIMPORT from utilities import model_circle
		stp = step[-1]
		rings = sparx_alignment.prepref(data, sparx_utilities.model_circle(nx//2-1,nx,nx), cnx, cnx, numr, mode, xrng[0], xrng[0], stp)
		print(" rings  ",len(rings))
		for im in range(len(data)):
			rings[im][0][0].set_attr("sxi",0)
			rings[im][0][0].set_attr("syi",0)
			rings[im][0][0].set_attr("inx",nx)
		tavg = sparx_statistics.ave_series(data)
		for N_step in range(len(xrng)):
			print(" xrng ",xrng[N_step])
			for Iter in range(max_iter):
				total_iter += 1
				if( FH > 0.0):
					fl = 0.1+(FH-0.1)*Iter/float(max_iter-1)
					tavg = sparx_filter.filt_tanl(tavg, fl, FF)
					"""Multiline Comment46"""
				else:
					"""Multiline Comment47"""
				cs = [0,0]
				#print  "  iteration  std   %03d   %7.2f    %7.2f  "%(total_iter,cs[0],cs[1])
				if Iter%4 != 0 or total_iter > max_iter*len(xrng)-10: delta = 0.0
				else:                                                 delta = dst
				sx_sum, sy_sum, nope = sparx_alignment.ali2d_single_iter(rings, numr, wr, cs, tavg, cnx, cny, \
															xrng[N_step], yrng[N_step], step[N_step], \
															mode=mode, CTF=False, delta=delta, random_method = method)
				for im in range(len(data)):
					alpha, tx, ty, mir, scale = sparx_utilities.get_params2D(rings[im][0][0])
					sparx_utilities.set_params2D(data[im],[alpha, tx, ty, mir, scale])
				tavg = sparx_statistics.ave_series(data)
				#print  "tata ",total_iter-1,Util.infomask(tavg,None,True)
				#tavg.write_image('tata.hdf',total_iter-1)
	else:
		tavg = sparx_statistics.ave_series(data)
		for N_step in range(len(xrng)):
			for Iter in range(max_iter):
				total_iter += 1
				cs = EMAN2_cppwrap.Util.infomask(tavg, mask, False)
				tavg -= cs[0]
				if( FH > 0.0):
					fl = 0.1+(FH-0.1)*Iter/float(max_iter-1)
					tavg = sparx_filter.filt_tanl(tavg, fl, FF)
				"""Multiline Comment48"""
				cs = [0,0]
				if Iter%4 != 0 or total_iter > max_iter*len(xrng)-10:
					delta = 0.0
				else:
					delta = dst
					asx = 0
					asy = 0
					#tavg, asx,asy = \
					#	center_2D(tavg, center_method = 7, searching_range = cnx//2, self_defined_reference = mask)
					if(asx != 0 or asy != 0):
						#  Shift images by this additional amount
						for im in range(nima):
							alpha, sx, sy, mir, scale = sparx_utilities.get_params2D(data[im])
							if mir == 0:  sxn = sx-asx
							else:  sxn = sx+asx
							syn = sy-asy
							alphai, sxn, syn, dummy  = sparx_utilities.combine_params2(0, sxn, syn, 0, -alpha, 0,0, 0)
							sxn += asx
							syn += asy
							EMAN2_cppwrap.Util.cyclicshift(data[im] , {"dx":-asx,"dy":-asy})
							lx[im] += asx
							ly[im] += asy
							alphai, sxn, syn, dummy  = sparx_utilities.combine_params2(0, sxn, syn, 0, alpha, 0,0, 0)
							sparx_utilities.set_params2D(data[im], [alpha, sxn, syn, mir, 1.0])

				sx_sum, sy_sum, nope = sparx_alignment.ali2d_single_iter(data, numr, wr, cs, tavg, cnx, cny, \
															xrng[N_step], yrng[N_step], step[N_step], \
															mode=mode, CTF=False, delta=delta)

				tavg = sparx_statistics.ave_series(data)
				#for im in data:  print get_params2D(im)
				#print  "tata ",total_iter-1,Util.infomask(tavg,None,True)
				#tavg.write_image('tata.hdf',total_iter-1)

		#  Shift data back and adjust parameters
		for im in range(nima):
			alpha, sx, sy, mir, scale = sparx_utilities.get_params2D(data[im])
			alphai, sxn, syn, dummy  = sparx_utilities.combine_params2(0, sx, sy, 0, -alpha, 0,0, 0)
			EMAN2_cppwrap.Util.cyclicshift(data[im] , {"dx":-lx[im],"dy":-ly[im]})
			alphai, sxn, syn, dummy  = sparx_utilities.combine_params2(0, sxn-lx[im], syn-ly[im], 0, alpha, 0,0, 0)
			sparx_utilities.set_params2D(data[im], [alpha, sxn, syn, mir, 1.0])

	return tavg




#  Version before I started messing with centering of averages  PAP 07/10/2015
def volalixshift_MPI(stack, ref_vol, outdir, search_rng, pixel_size, dp, dphi, fract, rmax, rmin, maskfile = None, \
	    maxit = 1, CTF = False, snr = 1.0, sym = "c1",  user_func_name = "helical", \
	    npad = 2, debug = False, nearby=3):

	pass#IMPORTIMPORTIMPORT from alignment       import Numrinit, prepare_refrings, proj_ali_incore, proj_ali_incore_local, proj_ali_incore_local_psi
	pass#IMPORTIMPORTIMPORT from utilities       import model_circle, get_image, drop_image, get_input_from_string, peak_search, model_cylinder, pad, model_blank
	pass#IMPORTIMPORTIMPORT from utilities       import bcast_list_to_all, bcast_number_to_all, reduce_EMData_to_root, bcast_EMData_to_all
	pass#IMPORTIMPORTIMPORT from utilities       import send_attr_dict, sym_vol
	pass#IMPORTIMPORTIMPORT from utilities       import get_params_proj, set_params_proj, file_type
	pass#IMPORTIMPORTIMPORT from fundamentals    import rot_avg_image, ccf
	pass#IMPORTIMPORTIMPORT import os
	pass#IMPORTIMPORTIMPORT import types
	pass#IMPORTIMPORTIMPORT from utilities       import print_begin_msg, print_end_msg, print_msg
	pass#IMPORTIMPORTIMPORT from mpi             import mpi_bcast, mpi_comm_size, mpi_comm_rank, MPI_FLOAT, MPI_COMM_WORLD, mpi_barrier
	pass#IMPORTIMPORTIMPORT from mpi             import mpi_reduce, MPI_INT, MPI_SUM
	pass#IMPORTIMPORTIMPORT from filter          import filt_ctf
	pass#IMPORTIMPORTIMPORT from projection      import prep_vol, prgs
	pass#IMPORTIMPORTIMPORT from statistics      import hist_list, varf3d_MPI, fsc_mask
	pass#IMPORTIMPORTIMPORT from applications	 import MPI_start_end, ordersegments
	pass#IMPORTIMPORTIMPORT from time            import time
	
	nproc     = mpi.mpi_comm_size(mpi.MPI_COMM_WORLD)
	myid      = mpi.mpi_comm_rank(mpi.MPI_COMM_WORLD)
	main_node = 0

	search_rng    = int(search_rng)
	if(search_rng < 1 ):  sparx_global_def.ERROR('Search range has to be provided', "volalixshift_MPI", 1, myid)
	if(pixel_size < 0.0 or dp < 0.0 ):  sparx_global_def.ERROR('Helical symmetry parameters have to be provided', "volalixshift_MPI", 1, myid)

	if os.path.exists(outdir):  sparx_global_def.ERROR('Output directory %s  exists, please change the name and restart the program'%outdir, "volalixshift_MPI", 1, myid)
	mpi.mpi_barrier(mpi.MPI_COMM_WORLD)
	
	if myid == main_node:
		os.mkdir(outdir)
		pass#IMPORTIMPORTIMPORT import global_def
		sparx_global_def.LOGFILE =  os.path.join(outdir, sparx_global_def.LOGFILE)
		sparx_utilities.print_begin_msg("volalixshift_MPI")
	mpi.mpi_barrier(mpi.MPI_COMM_WORLD)

	if debug:
		pass#IMPORTIMPORTIMPORT from time import sleep
		while not os.path.exists(outdir):
			print("Node ",myid,"  waiting...")
			time.sleep(5)

		info_file = os.path.join(outdir, "progress%04d"%myid)
		finfo = open(info_file, 'w')
	else:
		finfo = None
	max_iter    = int(maxit)
	
	vol     = EMAN2_cppwrap.EMData()
	vol.read_image(ref_vol)
	nx      = vol.get_xsize()
	ny      = vol.get_ysize()
	nz      = vol.get_zsize()

	xysize = nx
	zsize = -1

	if myid == main_node:
		pass#IMPORTIMPORTIMPORT import user_functions
		user_func = sparx_user_functions.factory[user_func_name]

		sparx_utilities.print_msg("Input stack                 : %s\n"%(stack))
		sparx_utilities.print_msg("Reference volume            : %s\n"%(ref_vol))	
		sparx_utilities.print_msg("Output directory            : %s\n"%(outdir))
		sparx_utilities.print_msg("Maskfile                    : %s\n"%(maskfile))
		sparx_utilities.print_msg("Search range                : %f\n"%(search_rng))
		sparx_utilities.print_msg("Pixel size [A]              : %f\n"%(pixel_size))
		sparx_utilities.print_msg("dp [A]                      : %f\n"%(dp))
		sparx_utilities.print_msg("dphi                        : %f\n"%(dphi))
		sparx_utilities.print_msg("Maximum iteration           : %i\n"%(max_iter))
		sparx_utilities.print_msg("CTF correction              : %s\n"%(CTF))
		sparx_utilities.print_msg("Signal-to-Noise Ratio       : %f\n"%(snr))
		sparx_utilities.print_msg("Symmetry group              : %s\n\n"%(sym))
	
	
	if maskfile:
		if type(maskfile) is bytes: mask3D = sparx_utilities.get_image(maskfile)
		else:                                  mask3D = maskfile
	else: mask3D = sparx_utilities.model_cylinder(rmax, nx, ny, nz)

	fscmask = mask3D  #model_circle(last_ring,nx,nx,nx)  For a fancy mask circle would work better  PAP 7/21/11
	if CTF:
		pass#IMPORTIMPORTIMPORT from reconstruction import recons3d_4nn_ctf_MPI
		pass#IMPORTIMPORTIMPORT from filter         import filt_ctf
	pass#IMPORTIMPORTIMPORT else:	 from reconstruction import recons3d_4nn_MPI

	filaments = sparx_pixel_error.ordersegments(stack, filament_attr = 'filament', verify=False)
	
	total_nfils = len(filaments)

	if total_nfils< nproc:
		sparx_global_def.ERROR('number of CPUs (%i) is larger than the number of filaments (%i), please reduce the number of CPUs used'%(nproc, total_nfils), "volalixshift_MPI", 1,myid)

	fstart, fend = MPI_start_end(total_nfils, nproc, myid)
	filaments = filaments[fstart:fend]
	nfils = len(filaments)

	list_of_particles = []
	indcs = []
	k = 0
	for i in range(nfils):
		list_of_particles += filaments[i]
		k1 = k+len(filaments[i])
		indcs.append([k,k1])
		k = k1

	data = EMAN2_cppwrap.EMData.read_images(stack, list_of_particles)
	nima = len(data)

	data_nx = data[0].get_xsize()
	data_ny = data[0].get_xsize()
	mask2D  = sparx_utilities.pad(sparx_utilities.model_blank(2*int(rmax), data_ny, 1, 1.0), data_nx, data_ny, 1, 0.0)
	for im in range(nima):
		data[im].set_attr('ID', list_of_particles[im])
		if CTF:
			ctf_params = data[im].get_attr("ctf")
			st = EMAN2_cppwrap.Util.infomask(data[im], mask2D, False)
			data[im] -= st[0]
			data[im] = sparx_filter.filt_ctf(data[im], ctf_params)
			data[im].set_attr('ctf_applied', 1)
	del list_of_particles
	
	if debug:
		finfo.write( '%d loaded  \n' % nima )
		finfo.flush()
	if myid == main_node:
		# initialize data for the reference preparation function
		# No centering for helical reconstruction
		ref_data = [None, mask3D, None, None, None ]

	pass#IMPORTIMPORTIMPORT from time import time

	nwx = 2*search_rng+1
	terminate = 0
	Iter = -1
	while Iter < max_iter-1 and terminate == 0:
		Iter += 1
		if myid == main_node:
			start_time = time.time()
			sparx_utilities.print_msg("\nITERATION #%3d\n"%(Iter))

		volft, kbx, kby, kbz = sparx_projection.prep_vol( vol )
		del vol

		for ifil in range(nfils):
			ctxsum = sparx_utilities.model_blank(data_nx, data_ny)
			segsctx = []
			start = indcs[ifil][0]
			for im in range(start, indcs[ifil][1]):
				phi,tht,psi,s2x,s2y = sparx_utilities.get_params_proj(data[im])
				refim = (sparx_projection.prgs( volft, kbz, [phi, tht, psi, 0.0, s2y], kbx, kby ))*mask2D
				ctx = sparx_fundamentals.ccf(data[im],refim)
				EMAN2_cppwrap.Util.add_img(ctxsum, ctx)
				ct1 = sparx_utilities.model_blank(data_nx, 1)
				for ii in range(data_nx):
					for jj in range(data_ny):
						ct1[ii] += ctx[ii,jj]
				ct1 = EMAN2_cppwrap.Util.window(ct1, nwx+2, 1)
				segsctx.append(ct1)

			# find overall peak
			ct1 = sparx_utilities.model_blank(data_nx, 1)
			for ii in range(data_nx):
				for jj in range(data_ny):
					ct1[ii] += ctxsum[ii,jj]
			ct1 = EMAN2_cppwrap.Util.window(ct1, nwx+2, 1)
			sump1 = sparx_utilities.peak_search(ct1)
			peakval = sump1[0][0]/(indcs[ifil][1] - start)
			sump1   = int(sump1[0][1])

			for im in range(start, indcs[ifil][1]):
				phi,tht,psi,s2x,s2y = sparx_utilities.get_params_proj(data[im])
				loc = sump1
				cim = im - start
				for k in range(max(1,sump1-nearby), min(nwx+1, sump1+nearby)):
					if( segsctx[cim].get_value_at(k) > segsctx[cim].get_value_at(k-1) and segsctx[cim].get_value_at(k) > segsctx[cim].get_value_at(k+1) and segsctx[cim].get_value_at(k) > peakval):
						peakval = segsctx[cim].get_value_at(k)
						loc = k

				sparx_utilities.set_params_proj(data[im], [phi, tht, psi, float(loc - (nwx+2)//2), s2y])

		del volft, refim, segsctx

		if myid == main_node:
			sparx_utilities.print_msg("Time of alignment = %d\n"%(time.time()-start_time))
			start_time = time.time()

		if CTF:  vol = sparx_reconstruction.recons3d_4nn_ctf_MPI(myid, data, symmetry=sym, snr = snr, npad = npad, xysize = xysize, zsize = zsize)
		else:    vol = sparx_reconstruction.recons3d_4nn_MPI(myid, data, symmetry=sym, snr = snr, npad = npad, xysize = xysize, zsize = zsize)

		if myid == main_node:
			sparx_utilities.print_msg("3D reconstruction time = %d\n"%(time.time()-start_time))
			start_time = time.time()

		if myid == main_node:
			vol = vol.helicise(pixel_size, dp, dphi, fract, rmax, rmin)
			vol = sparx_utilities.sym_vol(vol, symmetry=sym)
			ref_data[0] = vol
			vol = user_func(ref_data)
			vol = vol.helicise(pixel_size, dp, dphi, fract, rmax, rmin)
			vol = sparx_utilities.sym_vol(vol, symmetry=sym)
			#vol.write_image(os.path.join(outdir, "volfshift%03d.hdf"%Iter))
			if(Iter == max_iter-1):  sparx_utilities.drop_image(vol, os.path.join(outdir, "volfshift.hdf"))

		sparx_utilities.bcast_EMData_to_all(vol, myid, main_node)
	#In the last iteration calculate hfsc
	"""Multiline Comment54"""
	# write out headers, under MPI writing has to be done sequentially
	mpi.mpi_barrier(mpi.MPI_COMM_WORLD)
	par_str = ['xform.projection', 'ID']
	if myid == main_node:
		if(sparx_utilities.file_type(stack) == "bdb"):
			pass#IMPORTIMPORTIMPORT from utilities import recv_attr_dict_bdb
			sparx_utilities.recv_attr_dict_bdb(main_node, stack, data, par_str, 0, nima, nproc)
		else:
			pass#IMPORTIMPORTIMPORT from utilities import recv_attr_dict
			sparx_utilities.recv_attr_dict(main_node, stack, data, par_str, 0, nima, nproc)
		sparx_utilities.print_msg("Time to write header information= %d\n"%(time.time()-start_time))
		start_time = time.time()
	else:		sparx_utilities.send_attr_dict(main_node, data, par_str, 0, nima)
	if myid == main_node: sparx_utilities.print_end_msg("volalixshift_MPI")


def diskali_MPI(stack, ref_vol, outdir, maskfile, dp, dphi, pixel_size, user_func_name, zstep=1.0, fract=0.67, rmax=70, rmin=0, \
					 CTF=False, maxit=1, sym = "c1"):
	pass#IMPORTIMPORTIMPORT from mpi              import mpi_bcast, mpi_comm_size, mpi_comm_rank, MPI_COMM_WORLD, mpi_barrier, MPI_INT, mpi_recv, mpi_send
	pass#IMPORTIMPORTIMPORT from utilities        import get_params_proj, read_text_row, model_cylinder,pad, set_params3D, get_params3D, reduce_EMData_to_root, bcast_EMData_to_all, bcast_number_to_all, drop_image, bcast_EMData_to_all, model_blank
	pass#IMPORTIMPORTIMPORT from utilities        import send_attr_dict, file_type, sym_vol, get_image
	pass#IMPORTIMPORTIMPORT from fundamentals     import resample, rot_shift3D
	pass#IMPORTIMPORTIMPORT from applications     import MPI_start_end, ordersegments
	pass#IMPORTIMPORTIMPORT from math             import fmod, atan, pi
	pass#IMPORTIMPORTIMPORT from utilities        import model_blank
	pass#IMPORTIMPORTIMPORT from filter           import filt_tanl, filt_ctf
	pass#IMPORTIMPORTIMPORT import os
	pass#IMPORTIMPORTIMPORT from statistics       import fsc_mask
	pass#IMPORTIMPORTIMPORT from copy             import copy
	pass#IMPORTIMPORTIMPORT from os               import sys
	pass#IMPORTIMPORTIMPORT from time             import time
	pass#IMPORTIMPORTIMPORT from alignment        import Numrinit, ringwe
	pass#IMPORTIMPORTIMPORT from reconstruction   import recons3d_wbp
	pass#IMPORTIMPORTIMPORT from morphology       import ctf_2
	pass#IMPORTIMPORTIMPORT import types
	
	myid  = mpi.mpi_comm_rank(mpi.MPI_COMM_WORLD)
	nproc = mpi.mpi_comm_size(mpi.MPI_COMM_WORLD)
	main_node = 0

	mpi.mpi_barrier(mpi.MPI_COMM_WORLD)

	if myid == main_node:
		if not(os.path.exists(outdir)):
			os.mkdir(outdir)
	mpi.mpi_barrier(mpi.MPI_COMM_WORLD)

	dpp = (float(dp)/pixel_size)
	rise = int(dpp)
	winxy = int(rmax)*2 + 4
	if(float(rise) != dpp):
		print("  dpp has to be integer multiplicity of the pixel size")
		os.sys.exit()
	rise3 = 3*rise
	# for resampling to polar rmin>1
	rminpolar = max(1,rmin)

	pass#IMPORTIMPORTIMPORT from utilities import get_im
	refvol = sparx_utilities.get_im(ref_vol)
	ref_nx = refvol.get_xsize()
	ref_ny = refvol.get_ysize()
	ref_nz = refvol.get_zsize()
	if(ref_nz < rise3):
		print("  reference volumes has to be at least 3*rise long ")
		os.sys.exit()
	
	if maskfile:
		if type(maskfile) is bytes: mask3D = sparx_utilities.get_image(maskfile)
		else:                                  mask3D = maskfile
	else:
		if rmin > 0:
			mask3D = sparx_utilities.model_cylinder(rmax, ref_nx, ref_ny, ref_nz) - sparx_utilities.model_cylinder(rmin, ref_nx, ref_ny, ref_nz)
		else:
			mask3D = sparx_utilities.model_cylinder(rmax, ref_nx, ref_ny, ref_nz)
	
	filaments = sparx_pixel_error.ordersegments(stack, filament_attr = 'filament', verify=False)
	
	total_nfils = len(filaments)
	
	if total_nfils< nproc:
		sparx_global_def.ERROR('number of CPUs (%i) is larger than the number of filaments (%i), please reduce the number of CPUs used'%(nproc, total_nfils), "diskali_MPI", 1,myid)

	fstart, fend = MPI_start_end(total_nfils, nproc, myid)
	filaments = filaments[fstart:fend]
	nfils = len(filaments)

	list_of_particles = []
	indcs = []
	k = 0
	for i in range(nfils):
		list_of_particles += filaments[i]
		k1 = k+len(filaments[i])
		indcs.append([k,k1])
		k = k1

	if( myid == 0 ):
		# Build a 3D dedicated correction function
		if CTF:
			pass#IMPORTIMPORTIMPORT from morphology import ctf_2
			cc = EMAN2_cppwrap.EMUtil.get_all_attributes(stack, 'ctf')
			ctf2 = sparx_morphology.ctf_2(ref_nz, cc[0])
			ncc = len(ctf2)
			for i in range(1,len(cc)):
				temp = sparx_morphology.ctf_2(ref_nz, cc[i])
				for k in range(ncc): ctf2[k] += temp[k]
			del temp
		pass#IMPORTIMPORTIMPORT from math import sqrt
		rrc = sparx_utilities.model_blank(ref_nz, ref_nz, ref_nz)
		rc = ref_nz//2
		for i in range(ref_nz):
			ic = (i-rc)**2
			for j in range(ref_nz):
				jc = (j-rc)**2
				dc = numpy.sqrt(ic+jc)
				for k in range(ref_nz):
					if CTF:
						rr = numpy.sqrt((k-rc)**2 + ic + jc)
						rin = int(rr)
						drin = rr-rin
						rrc.set_value_at(i,j,k, dc/(1.0+(1.0-drin)*ctf2[rin] + drin*ctf2[rin+1]) )
					else:
						rrc.set_value_at(i,j,k, dc )

	data = EMAN2_cppwrap.EMData.read_images(stack, list_of_particles)
	nima = len(data)

	data_nx = data[0].get_xsize()
	data_ny = data[0].get_xsize()
	mask2D  = sparx_utilities.pad(sparx_utilities.model_blank(2*int(rmax), data_ny, 1, 1.0), data_nx, data_ny, 1, 0.0)
	for im in range(nima):
		data[im].set_attr('ID', list_of_particles[im])
		if CTF:
			ctf_params = data[im].get_attr("ctf")
			st = EMAN2_cppwrap.Util.infomask(data[im], mask2D, False)
			data[im] -= st[0]
			data[im] = sparx_filter.filt_ctf(data[im], ctf_params)
			data[im].set_attr('ctf_applied', 1)
	del list_of_particles, mask2D

	# This is the part sufficient to get polar resampling
	refvol = EMAN2_cppwrap.Util.window(refvol, winxy, winxy, rise)
	rr = ref_nz//2-2
	# do full sized reconstruction with the projection parameters BEFORE they are modified by disk alignment step
	fullvolsum0 = sparx_utilities.model_blank(ref_nx, ref_ny, ref_nz)
	filvols = []
	data_slices = []
	start = time.time()
	Torg = [None]*len(data)
	for i in range(len(data)):  Torg[i] = data[i].get_attr("xform.projection")
	msk = sparx_utilities.model_cylinder(rmax-1, ref_nx, ref_ny, ref_nz) - sparx_utilities.model_cylinder(rmax-2, ref_nx, ref_ny, ref_nz)
	ms3 = sparx_utilities.model_cylinder(rmax, ref_nx, ref_ny, ref_nz)
	for ivol in range(nfils):
		fullvol0 = EMAN2_cppwrap.Util.window(sparx_reconstruction.recons3d_wbp(data, list_proj=list(range(indcs[ivol][0],indcs[ivol][1])), method = None, symmetry=sym, radius=rr), ref_nx, ref_ny, ref_nz, 0, 0, 0)
		fullvol0 = fullvol0.helicise(pixel_size, dp, dphi, fract, rmax, rmin)
		fullvol0 = sparx_utilities.sym_vol(fullvol0, symmetry=sym)
		stat = EMAN2_cppwrap.Util.infomask(fullvol0, msk, True)
		fullvol0 -= stat[0]
		fullvol0 *= ms3
		"""Multiline Comment55"""
		EMAN2_cppwrap.Util.add_img(fullvolsum0, fullvol0)

		filvols.append(EMAN2_cppwrap.Util.window(fullvol0, winxy, winxy, rise3))

		data_slices.append(cylindrical_trans(filvols[ivol], rminpolar, rmax, rise, True))
		#print  " DONE ", myid,ivol
	#  original projection data no longer needed
	del data, fullvol0, ms3, msk
	if myid == main_node:
		tt = time.time()
		print(" TIME to do initial reconstructions :", tt-start)
		start = tt
	sparx_utilities.reduce_EMData_to_root(fullvolsum0, myid)
	if myid == main_node and False:
		fullvolsum0.write_image("verify0.hdf")
		rrc.write_image("verify1.hdf")
		fullvolsum0 = EMAN2_cppwrap.Util.window( sparx_utilities.pad(fullvolsum0, ref_nz, ref_nz, ref_nz, 0.0).filter_by_image(rrc), ref_nx, ref_ny, ref_nz, 0, 0, 0)
		fullvolsum0 = fullvolsum0.helicise(pixel_size, dp, dphi, fract, rmax, rmin)
		fullvolsum0 = sparx_utilities.sym_vol(fullvolsum0, symmetry=sym)
		stat = EMAN2_cppwrap.Util.infomask(fullvolsum0,  sparx_utilities.model_cylinder(rmax-1, ref_nx, ref_ny, ref_nz) - sparx_utilities.model_cylinder(rmax-2, ref_nx, ref_ny, ref_nz), True)
		fullvolsum0 -= stat[0]
		fullvolsum0 *= sparx_utilities.model_cylinder(rmax-1, ref_nx, ref_ny, ref_nz)
		fullvolsum0.write_image("verify2.hdf")
		#for i in xrange(len(filvols)):
		#	filvols[i].write_image("verify7.hdf",i)
	mpi.mpi_barrier(mpi.MPI_COMM_WORLD)
	#sys.exit()

	T_filament = [None]*nfils
	for iter in range(1,maxit+1):
		if myid == main_node: print(" ITERATION:  ", iter)
		refslices = cylindrical_trans(refvol, rminpolar, rmax, rise)

		for ivol in range(nfils):
			#if myid == main_node: print "  ALI  ",myid,ivol,Util.infomask(data_slices[ivol][0], None, True)
			T_filament[ivol] = alihelical3(data_slices[ivol], refslices, zstep, dphi, rise, rminpolar, rmax, sym)

		refvol = sparx_utilities.model_blank(winxy, winxy, rise3)
		for ivol in range(nfils):
			d = T_filament[ivol].get_params('spider')
			#if myid == main_node: print  d["phi"], d["theta"], d["psi"], d["tz"]
			#if myid == main_node:
			#	pad(rot_shift3D(filvols[ivol],  d["phi"], d["theta"], d["psi"], sz=d["tz"]), ref_nx, ref_ny, ref_nz, 0.0).write_image("verify1.hdf")
			EMAN2_cppwrap.Util.add_img(refvol, sparx_fundamentals.rot_shift3D(filvols[ivol],  d["phi"], d["theta"], d["psi"], sz=d["tz"]))
		sparx_utilities.reduce_EMData_to_root(refvol, myid)
			
		if myid == main_node:
			#refvol.write_image("verify3.hdf")
			#Util.window(refvol, winxy, winxy, rise).write_image("verify4.hdf")
			refvol =  stack_disks(EMAN2_cppwrap.Util.window(refvol, winxy, winxy, rise), winxy, winxy, ref_nz, dphi, rise)
			#refvol.write_image("verify5.hdf")
			refvol = sparx_utilities.sym_vol(refvol, symmetry=sym)
			#refvol.write_image("verify6.hdf")
			if CTF:
				refvol = EMAN2_cppwrap.Util.window( sparx_utilities.pad(refvol, ref_nz, ref_nz, ref_nz, 0.0).filter_by_image(rrc), ref_nx, ref_ny, ref_nz, 0, 0, 0)
				refvol = refvol.helicise(pixel_size, dp, dphi, fract, rmax, rmin)
				refvol = sparx_utilities.sym_vol(refvol, symmetry=sym)
				stat = EMAN2_cppwrap.Util.infomask(refvol,  sparx_utilities.model_cylinder(rmax-1, ref_nx, ref_ny, ref_nz) - sparx_utilities.model_cylinder(rmax-2, ref_nx, ref_ny, ref_nz), True)
				refvol -= stat[0]
				refvol *= sparx_utilities.model_cylinder(rmax-1, ref_nx, ref_ny, ref_nz)

			pass#IMPORTIMPORTIMPORT import user_functions
			user_func = sparx_user_functions.factory[user_func_name]

			ref_data = [refvol, mask3D]
			refvol = user_func(ref_data)
			refvol = refvol.helicise( pixel_size , dp, dphi, fract, rmax, rmin)
			refvol = sparx_utilities.sym_vol(refvol, symmetry=sym)
			if(iter == maxit):  sparx_utilities.drop_image(refvol, os.path.join(outdir, 'fvheli.hdf'))
			refvol = EMAN2_cppwrap.Util.window(refvol, winxy, winxy, rise)

			tt = time.time()
			print(" TIME of one iteration :", tt-start)
			start = tt
		else:
			refvol = sparx_utilities.model_blank(winxy, winxy, rise)

		sparx_utilities.bcast_EMData_to_all(refvol, myid, main_node)

	del refvol, data_slices

	# update rotations of individual images
	data = [EMAN2_cppwrap.EMData() for i in range(nima)]
	forg = []
	helisym = EMAN2_cppwrap.Transform({"type":"spider","phi":dphi,"tz":dpp})
	ihelisym = helisym.inverse()
	pass#IMPORTIMPORTIMPORT from utilities import get_params_proj, set_params_proj
	permitrange = rise/2.0
	
	for ivol in range(nfils):
		#  This is for printout
		d = T_filament[ivol].get_params('spider')
		#print  d["phi"],d["theta"],d["psi"],d["tx"],d["ty"],d["tz"]
		if( d["tz"] < 0.0 ):  d = (helisym*T_filament[ivol]).get_params('spider')
		#print  d["phi"],d["theta"],d["psi"],d["tx"],d["ty"],d["tz"]
		#forg.append([d["phi"],d["theta"],d["psi"],d["tx"],d["ty"],d["tz"]])
		#  Use inverse transformation to modify projection directions.
		Tinv = T_filament[ivol].inverse()
		for im in range(indcs[ivol][0],indcs[ivol][1]):
			#d = Torg[im].get_params('spider')
			#print  d["phi"],d["theta"],d["psi"],d["tx"],d["ty"],d["tz"]
			Torg[im] = Torg[im]*Tinv
			d = Torg[im].get_params('spider')
			d["tx"] = -d["tx"]
			d["ty"] = -d["ty"]
			#if myid == 0: print  " A ",d["phi"],d["theta"],d["psi"],d["tx"],d["ty"]
			#finfo.write(" %d  %f  %f7.2  %f  %f  %f\n" %(im,d["phi"],d["theta"],d["psi"],d["tx"],d["ty"]))
			#finfo.flush()
			psi = d["psi"]
			while( d["ty"] < -permitrange ):
				if( psi > 180.0):  Torg[im] = Torg[im]*helisym
				else:              Torg[im] = Torg[im]*ihelisym
				d = Torg[im].get_params('spider')
				d["tx"] = -d["tx"]
				d["ty"] = -d["ty"]
			while( d["ty"] > permitrange ):
				if( psi > 180.0):  Torg[im] = Torg[im]*ihelisym
				else:              Torg[im] = Torg[im]*helisym
				d = Torg[im].get_params('spider')
				d["tx"] = -d["tx"]
				d["ty"] = -d["ty"]
			#finfo.write(" %d  %f  %f7.2  %f  %f  %f\n" %(im,d["phi"],d["theta"],d["psi"],d["tx"],d["ty"]))
			#finfo.flush()

			#if myid == 0: print  " B ",d["phi"],d["theta"],d["psi"],d["tx"],d["ty"]
			sparx_utilities.set_params_proj(data[im], [d["phi"],d["theta"],d["psi"],d["tx"],d["ty"]])
			data[im].set_attr('ID', filaments[ivol][im-indcs[ivol][0]])
	#from utilities import write_text_row
	#write_text_row(forg,"forg%03d.txt"%myid)
	"""Multiline Comment56"""
	mpi.mpi_barrier(mpi.MPI_COMM_WORLD)

	# write out headers, under MPI writing has to be done sequentially
	pass#IMPORTIMPORTIMPORT from mpi import mpi_recv, mpi_send, MPI_COMM_WORLD, MPI_FLOAT
	par_str = ['xform.projection', 'ID']
	if myid == main_node:
		if(sparx_utilities.file_type(stack) == "bdb"):
			pass#IMPORTIMPORTIMPORT from utilities import recv_attr_dict_bdb
			sparx_utilities.recv_attr_dict_bdb(main_node, stack, data, par_str, 0, nima, nproc)
		else:
			pass#IMPORTIMPORTIMPORT from utilities import recv_attr_dict
			sparx_utilities.recv_attr_dict(main_node, stack, data, par_str, 0, nima, nproc)
	else:
		sparx_utilities.send_attr_dict(main_node, data, par_str, 0, nima)

def cylindrical_trans(vol, rmin, rmax, rise, apply_weights = False):
	pass#IMPORTIMPORTIMPORT from alignment import Numrinit, ringwe
	refslices=[]
	mode="F"
	numr = sparx_alignment.Numrinit(rmin, rmax, 1, mode)
	maxrin = numr[len(numr)-1]
	wr_four  = sparx_alignment.ringwe(numr, mode)
	ref_nx2 = vol.get_xsize()
	ref_ny2 = vol.get_ysize()
	cnx = ref_nx2//2 + 1
	cny = ref_ny2//2 + 1
	cnz = rise//2
	for i in range(rise):
		cimage = EMAN2_cppwrap.Util.Polar2Dm(EMAN2_cppwrap.Util.window(vol, ref_nx2, ref_ny2, 1, 0, 0, i-cnz), cnx, cny, numr, mode)
		EMAN2_cppwrap.Util.Frngs(cimage, numr)
		if apply_weights:  EMAN2_cppwrap.Util.Applyws(cimage, numr, wr_four)
		refslices.append(cimage)
	return refslices

def alihelical3(slices, refslices, zstep, dphi, rise, rmin, rmax, sym="c1"):
	pass#IMPORTIMPORTIMPORT from applications import  alihelical4

	T,qn = alihelical4(slices, refslices, zstep, dphi, rise, rmin, rmax, theta=0.0)
	if( sym[:1] != "d" and  sym[:1] != "D" ):
		R,qr = alihelical4(slices, refslices, zstep, dphi, rise, rmin, rmax, theta=180.0)
		if( qr > qn ):
			qn = qr
			T = R
	# The transformation returned is to be applied to the disk.  To projection data apply the inverse.
	return T

# standalone code for aligning two disks 
def alihelical4(slices, refslices, zstep, dphi, rise, rmin, rmax, theta=0.0):

	pass#IMPORTIMPORTIMPORT from fundamentals import rot_shift3D, cyclic_shift
	pass#IMPORTIMPORTIMPORT from statistics import ccc
	pass#IMPORTIMPORTIMPORT from math import ceil, fmod
	pass#IMPORTIMPORTIMPORT from utilities import pad, model_cylinder
	pass#IMPORTIMPORTIMPORT from random import randrange, randint
	pass#IMPORTIMPORTIMPORT from math         import fmod
	pass#IMPORTIMPORTIMPORT from alignment    import Numrinit, ringwe, ang_n
	pass#IMPORTIMPORTIMPORT from utilities    import model_blank


	pdphi = -dphi
	if(pdphi < 0.0):  pdphi = 360.0 - pdphi

	mode="F"
	numr = sparx_alignment.Numrinit(rmin, rmax, 1, mode) # pull this to the calling program
	maxrin = numr[len(numr)-1]

	lren = refslices[0].get_xsize()

	maxqn = -1.0e23
	#slices = cylindrical_trans(vol, rmax, rise)
	cyclic_slices = [None]*rise
	local_slices  = [None]*rise
	if( theta == 0.0 ):
		for j in range(rise):  local_slices[j] = slices[j]
	else:
		# inverse the order
		for j in range(rise):  local_slices[j] = slices[rise-1 - j]
	for j in range(rise):
		iz = j * zstep
		for k in range(rise):
			kiz = k + iz
			if( kiz < rise ):
				cyclic_slices[k] = local_slices[kiz]
			else:
				cslice = kiz%rise
				cyclic_slices[k] = local_slices[cslice]  # shifted have to be rotated in Fourier space
				refim = sparx_utilities.model_blank(lren)
				if(theta == 0.0):  EMAN2_cppwrap.Util.update_fav(refim, local_slices[cslice],  iang( pdphi, maxrin), 0, numr)
				else:              EMAN2_cppwrap.Util.update_fav(refim, local_slices[cslice],  iang( 360.0 - pdphi, maxrin), 0, numr)
				cyclic_slices[k] = refim.copy()

		linesum = sparx_utilities.model_blank(maxrin)

		for i in range(rise):
			if(theta == 0.0):  temp = EMAN2_cppwrap.Util.Crosrng_msg_s(refslices[i], cyclic_slices[i], numr)
			else:              temp = EMAN2_cppwrap.Util.Crosrng_msg_m(refslices[i], cyclic_slices[i], numr)
			linesum += temp

		qn  = -1.0e20
		tot = 0
		for i in range(maxrin):
			if (linesum[i] >= qn):
				qn  = linesum[i]
				tot = i+1

		if qn > maxqn:
			maxqn = qn
			#  It is somewhat strange that both theta cases have the same setting
			maxphi = sparx_alignment.ang_n(tot, mode, maxrin)
			maxz = -iz

		
	T = EMAN2_cppwrap.Transform({"type":"spider","phi":maxphi,"theta":theta,"tz":maxz})
	return T, maxqn

def iang(alpha, maxrin):
	# compute position on maxrin  
	return  alpha*maxrin/360.+1

def stack_disks(v, nx, ny, ref_nz, dphi, rise):
	pass#IMPORTIMPORTIMPORT from utilities    import model_blank
	pass#IMPORTIMPORTIMPORT from fundamentals import rot_shift3D
	refc = ref_nz//2
	rsc = rise//2
	
	heli = sparx_utilities.model_blank(nx, ny, ref_nz)

	lb = -((refc-rsc)/rise)
	if(lb*rise+refc-rsc > 0):  lb -= 1
	
	le = (ref_nz-refc-rsc)/rise
	if((le+1)*rise+refc-rsc < ref_nz): le +=1
	
	for i in range(lb,le+1):
		#print i,refc + i*rise - rsc
		heli.insert_clip(sparx_fundamentals.rot_shift3D(v, i*dphi),(0,0,refc + i*rise - rsc))

	return heli

def imgstat_hfsc( stack, file_prefix, fil_attr='filament'):
	pass#IMPORTIMPORTIMPORT from utilities    import write_text_file, chunks_distribution
	pass#IMPORTIMPORTIMPORT from pixel_error  import ordersegments
	
	infils = EMAN2_cppwrap.EMUtil.get_all_attributes(stack, fil_attr)
	ptlcoords = EMAN2_cppwrap.EMUtil.get_all_attributes(stack, 'ptcl_source_coord')
	filaments = sparx_pixel_error.ordersegments(infils, ptlcoords)
	
	
	temp = sparx_utilities.chunks_distribution([[len(filaments[i]), i] for i in range(len(filaments))], 2)
	tempeven = temp[0:1][0]
	tempodd = temp[1:2][0]
	filaments_even = [filaments[tempeven[i][1]] for i in range(len(tempeven))]
	filaments_odd = [filaments[tempodd[i][1]] for i in range(len(tempodd))]
	
	nfileven = len(filaments_even)
	nfilodd = len(filaments_odd)
	
	even_segs = []
	odd_segs = []
	
	for ifil in range(nfileven):
		even_segs += filaments_even[ifil]
	for ifil in range(nfilodd):
		odd_segs += filaments_odd[ifil]
	sparx_utilities.write_text_file(even_segs, file_prefix + '_even.txt')
	sparx_utilities.write_text_file(odd_segs, file_prefix + '_odd.txt')	

def match_pixel_rise(dz,px, nz=-1, ndisk=-1, rele=0.1, stop=900000):
	'''
	Error is calculated as:
		error = ndisk*((( int( (dz/q/px) ) - dz/q/px))**2)
	instead of:
		error = ((int(ndisk*dz/q/px) - ndisk*dz/q/px)/px)**2
		
	If ndisk is such that ndisk * (dz/q/px) is an integer, then the error
	will come out zero when using the second equation even if dz/q/px is not an integer.
	
	E.g. say dz/q/px=1.2 and ndisk=5, then ndisk*dz/q/px = int(ndisk*dz/q/px) = 6, and
	the error will come out zero using the second equation.
	'''
	if nz < 0 and ndisk < 0:
		print("either nz or ndisk must be specified")
		os.sys.exit()
		
	if ndisk < 0: # calculate ndisk from nz
		dnz = nz*px
		ndisk = (int(dnz/dz)-1)//2
	
	q=1.0
	for i in range(0, stop):
		q  = 1.0 - 0.000001*i
		q1 = 1.0 + 0.000001*i
		error  = ndisk*((( int( (dz/q/px) ) - dz/q/px))**2)
		error1 = ndisk*((( int( (dz/q1/px)) - dz/q1/px))**2)
		if error1 < error:
			error = error1
			q = q1
		if (error < rele):
			return q, error
	return -1.0, -1.0

def gendisks_MPI(stack, mask3d, ref_nx, pixel_size, dp, dphi, fract=0.67, rmax=70, rmin=0, CTF=False, user_func_name = "helical", sym = "c1", dskfilename='bdb:disks', maxerror=0.01, new_pixel_size = -1, do_match_pixel_rise=False):
	pass#IMPORTIMPORTIMPORT from mpi              import mpi_bcast, mpi_comm_size, mpi_comm_rank, MPI_COMM_WORLD, mpi_barrier, MPI_INT, MPI_FLOAT, mpi_recv, mpi_send, mpi_reduce, MPI_MAX
	pass#IMPORTIMPORTIMPORT from utilities        import get_params_proj, read_text_row, model_cylinder,pad, set_params3D, get_params3D, model_blank, drop_image
	pass#IMPORTIMPORTIMPORT from utilities        import reduce_EMData_to_root, bcast_EMData_to_all, bcast_number_to_all, bcast_EMData_to_all, send_EMData, recv_EMData, bcast_list_to_all
	pass#IMPORTIMPORTIMPORT from utilities        import send_attr_dict, file_type, sym_vol, get_im, chunks_distribution
	pass#IMPORTIMPORTIMPORT from fundamentals     import resample, rot_shift3D
	pass#IMPORTIMPORTIMPORT from applications     import MPI_start_end, match_pixel_rise
	pass#IMPORTIMPORTIMPORT from pixel_error	  import ordersegments
	pass#IMPORTIMPORTIMPORT from math             import fmod, atan, pi
	pass#IMPORTIMPORTIMPORT from utilities        import model_blank
	pass#IMPORTIMPORTIMPORT from filter           import filt_tanl, filt_ctf
	pass#IMPORTIMPORTIMPORT import os
	pass#IMPORTIMPORTIMPORT from statistics       import fsc_mask
	pass#IMPORTIMPORTIMPORT from copy             import copy
	pass#IMPORTIMPORTIMPORT from os               import sys
	pass#IMPORTIMPORTIMPORT from time             import time
	pass#IMPORTIMPORTIMPORT from alignment        import Numrinit, ringwe
	pass#IMPORTIMPORTIMPORT from reconstruction   import recons3d_4nn,recons3d_4nn_ctf
	pass#IMPORTIMPORTIMPORT from morphology       import ctf_2

	myid  = mpi.mpi_comm_rank(mpi.MPI_COMM_WORLD)
	nproc = mpi.mpi_comm_size(mpi.MPI_COMM_WORLD)
	main_node = 0

	mpi.mpi_barrier(mpi.MPI_COMM_WORLD)

	if do_match_pixel_rise and (new_pixel_size > 0):
		sparx_global_def.ERROR( "If resampling is desired, either set do_match_pixel_rise to True OR specify new_pixel_size, but not both at the same time.\n If do_match_pixel_rise=True, the program will automatically calculate new pixel size of the output disks such that the rise will be ~ integer number of pixels in new pixel size.\n If new_pixel_size is specified, then the output disks will be resampled so that resulting pixel size is new_pixel_size.", "gendisks_MPI", 1, myid)
	pass#IMPORTIMPORTIMPORT from math import ceil
	dpp = float(dp)/pixel_size
	rise = int(numpy.ceil(dpp))
	
	if do_match_pixel_rise:
		# Calculate new pixel size such that dp/new_pixel_size is approximately an
		# integer.
		nsteps = 100000
		stepsize = (float(maxerror)/nsteps)
		for i in range(1, nsteps + 1):
			err_thr = i * stepsize
			q, error = match_pixel_rise(dp, pixel_size, ndisk=1, rele=err_thr)
			if q > 0:
				new_pixel_size = q*pixel_size
				break
		#print "new pixel size by match_pixel_rise: ",new_pixel_size
		if new_pixel_size < 0:  sparx_global_def.ERROR('match_pixel_size was not able to find a new pixel size with the desired maxerror', "gendisks_MPI", 1, myid)

	mpi.mpi_barrier(mpi.MPI_COMM_WORLD)

	if myid == 0:
		if new_pixel_size > 0:
			print("Output disks will be resampled to pixel size: ", new_pixel_size)

	if new_pixel_size > 0:
		dpp = (float(dp)/new_pixel_size)
		rise = int(numpy.ceil(dpp))
		ratio = pixel_size/new_pixel_size

	pass#IMPORTIMPORTIMPORT import user_functions
	user_func = sparx_user_functions.factory[user_func_name]
	
	if( myid == 0):
		infils = EMAN2_cppwrap.EMUtil.get_all_attributes(stack, "filament")
		ptlcoords = EMAN2_cppwrap.EMUtil.get_all_attributes(stack, 'ptcl_source_coord')
		filaments = sparx_pixel_error.ordersegments(infils, ptlcoords)
		total_nfils = len(filaments)
		inidl = [0]*total_nfils
		for i in range(total_nfils):  inidl[i] = len(filaments[i])
		linidl = sum(inidl)
		tfilaments = []
		for i in range(total_nfils):  tfilaments += filaments[i]
		del filaments
	else:
		total_nfils = 0
		linidl = 0
	total_nfils = sparx_utilities.bcast_number_to_all(total_nfils, source_node = main_node)
	if myid != main_node:
		inidl = [-1]*total_nfils
	inidl = sparx_utilities.bcast_list_to_all(inidl, myid, source_node = main_node)
	linidl = sparx_utilities.bcast_number_to_all(linidl, source_node = main_node)
	if myid != main_node:
		tfilaments = [-1]*linidl
	tfilaments = sparx_utilities.bcast_list_to_all(tfilaments, myid, source_node = main_node)
	filaments = []
	iendi = 0
	for i in range(total_nfils):
		isti = iendi
		iendi = isti+inidl[i]
		filaments.append(tfilaments[isti:iendi])
	del tfilaments,inidl

	if myid == main_node:
		print("total number of filaments: ", total_nfils)
	if total_nfils< nproc:
		sparx_global_def.ERROR('number of CPUs (%i) is larger than the number of filaments (%i), please reduce the number of CPUs used'%(nproc, total_nfils), "ehelix_MPI", 1,myid)

	#  balanced load
	chunks = sparx_utilities.chunks_distribution([[len(filaments[i]), i] for i in range(len(filaments))], nproc)

	# make a table associating filament name with processor id
	if myid == main_node:
		filatable = [[] for i in range(nproc)]
		for mid in range(nproc):
			tmp = chunks[mid:mid+1][0]
			tmpfilaments = [filaments[tmp[i][1]] for i in range(len(tmp))]
			nfil = len(tmpfilaments)
			filatable[mid] = [[] for j in range(nfil)]
			for i in range(nfil):
				a = sparx_utilities.get_im(stack, tmpfilaments[i][0])
				filname = a.get_attr('filament')
				filatable[mid][i] = filname

	#print filatable
	temp = chunks[myid:myid+1][0]
	filaments = [filaments[temp[i][1]] for i in range(len(temp))]
	nfils     = len(filaments)
	#  Get the maximum number of filaments on any node
	mfils = nfils
	mfils = mpi.mpi_reduce(mfils, 1, mpi.MPI_INT, mpi.MPI_MAX, 0, mpi.MPI_COMM_WORLD)
	if myid == main_node:  mfils = int(mfils[0])
	mfils = mpi.mpi_bcast(mfils, 1, mpi.MPI_INT, 0, mpi.MPI_COMM_WORLD)
	mfils = int(mfils[0])
	if mfils < nfils:	
		sparx_global_def.ERROR('Maximum number of filaments %d should not be less than nfils %d!'%(mfils, nfils), "gendisks_MPI", 1,myid)
	list_of_particles = []
	indcs = []
	k = 0
	for i in range(nfils):
		list_of_particles += filaments[i]
		k1 = k+len(filaments[i])
		indcs.append([k,k1])
		k = k1
	data = EMAN2_cppwrap.EMData.read_images(stack, list_of_particles)
	nima = len(data)

	data_nx = data[0].get_xsize()
	data_ny = data[0].get_ysize()
	data_nn = max(data_nx, data_ny)
	mask2D  = sparx_utilities.pad(sparx_utilities.model_blank(2*int(rmax), data_nn, 1, 1.0), data_nn, data_nn, 1, 0.0)
	for im in range(nima):
		data[im] = sparx_utilities.pad(data[im], data_nn, data_nn, 1, 'circumference')
		data[im].set_attr('ID', list_of_particles[im])
		if CTF:
			ctf_params = data[im].get_attr("ctf")
			st = EMAN2_cppwrap.Util.infomask(data[im], mask2D, False)
			data[im] -= st[0]
			data[im] = sparx_filter.filt_ctf(data[im], ctf_params)
			data[im].set_attr('ctf_applied', 1)
	del list_of_particles, mask2D

	ref_data = [None, mask3d, None, None, None ]

	if new_pixel_size > 0:
		if(ref_nx <0): ref_nx = int(data_nn*ratio+0.5)
	else:
		if(ref_nx <0): ref_nx = data_nn

	if myid == main_node:  outvol = 0
	start_time = time.time()
	for ivol in range(mfils):
		if( ivol < nfils ):
			#print myid, ivol, data[indcs[ivol][0]].get_attr('filament')
			if CTF:
				fullvol0 = sparx_reconstruction.recons3d_4nn_ctf(data, list_proj=list(range(indcs[ivol][0],indcs[ivol][1])), symmetry="c1", npad=2)
			else:
				fullvol0 = sparx_reconstruction.recons3d_4nn(data, list_proj=list(range(indcs[ivol][0],indcs[ivol][1])), symmetry="c1", npad=2, snr = snr)

			fullvol0 = fullvol0.helicise(pixel_size, dp, dphi, fract, rmax, rmin)
			fullvol0 = sparx_utilities.sym_vol(fullvol0, symmetry=sym)
			ref_data[0] = fullvol0
			fullvol0 = user_func(ref_data)
			fullvol0 = fullvol0.helicise(pixel_size, dp, dphi, fract, rmax, rmin)
			fullvol0 = sparx_utilities.sym_vol(fullvol0, symmetry=sym)
			if new_pixel_size > 0:
				# resample the volume using ratio such that resulting pixel size is new_pixel_size
				fullvol0 = sparx_fundamentals.resample(fullvol0, ratio)
			fullvol0 = EMAN2_cppwrap.Util.window(fullvol0, ref_nx, ref_nx, rise)
			if mask3d != None:  EMAN2_cppwrap.Util.mul_img(fullvol0, mask3d)
			gotfil = 1
		else:
			gotfil = 0
		#print "did volume ",myid,gotfil,ref_nx, ref_ny, rise
		if(myid == main_node):
			for i in range(nproc):
				if(i != main_node):
					didfil = mpi.mpi_recv(1, mpi.MPI_INT, i, sparx_global_def.SPARX_MPI_TAG_UNIVERSAL, mpi.MPI_COMM_WORLD)
					didfil = int(didfil[0])
					if(didfil == 1):
						fil = sparx_utilities.recv_EMData(i, ivol+i+70000)
						fil.set_attr('filament',filatable[i][ivol])
						fil.write_image(dskfilename, outvol)
						#print outvol, filatable[i][ivol]
						outvol += 1
				else:
					if( gotfil == 1 ):
						fullvol0.set_attr('filament',filatable[main_node][ivol])
						fullvol0.write_image(dskfilename, outvol)
						#print outvol, filatable[main_node][ivol]
						outvol += 1
		else:
			mpi.mpi_send(gotfil, 1, mpi.MPI_INT, main_node, sparx_global_def.SPARX_MPI_TAG_UNIVERSAL, mpi.MPI_COMM_WORLD)
			if(gotfil == 1):
				sparx_utilities.send_EMData(fullvol0, main_node, ivol+myid+70000)

def ehelix_MPI(stack, ref_vol, outdir, seg_ny, delta, phiwobble, psi_max, search_rng, rng, ywobble, ystep, pixel_size, dp, dphi, fract, rmax, rmin, FindPsi = True, maskfile = None, \
	    maxit = 1, CTF = False, snr = 1.0, sym = "c1",  user_func_name = "helical", npad = 2, debug = False, slowIO = False):

	pass#IMPORTIMPORTIMPORT from alignment       import Numrinit, prepare_refrings, proj_ali_incore, proj_ali_incore_local, proj_ali_incore_local_psi
	pass#IMPORTIMPORTIMPORT from alignment       import ringwe, ang_n
	pass#IMPORTIMPORTIMPORT from utilities       import model_circle, get_image, drop_image, get_input_from_string, peak_search, model_cylinder, pad, model_blank
	pass#IMPORTIMPORTIMPORT from utilities       import bcast_list_to_all, bcast_number_to_all, reduce_EMData_to_root, bcast_EMData_to_all
	pass#IMPORTIMPORTIMPORT from utilities       import send_attr_dict, sym_vol, get_input_from_string
	pass#IMPORTIMPORTIMPORT from utilities       import get_params_proj, set_params_proj, file_type, compose_transform2
	pass#IMPORTIMPORTIMPORT from fundamentals    import rot_avg_image, ccf, fft, rot_shift2D
	pass#IMPORTIMPORTIMPORT from utilities       import print_begin_msg, print_end_msg, print_msg, chunks_distribution
	pass#IMPORTIMPORTIMPORT from mpi             import mpi_bcast, mpi_comm_size, mpi_comm_rank, MPI_FLOAT, MPI_COMM_WORLD, mpi_barrier
	pass#IMPORTIMPORTIMPORT from mpi             import mpi_reduce, MPI_INT, MPI_SUM
	pass#IMPORTIMPORTIMPORT from filter          import filt_ctf
	pass#IMPORTIMPORTIMPORT from projection      import prep_vol, prgs
	#from statistics      import hist_list, varf3d_MPI, fsc_mask
	pass#IMPORTIMPORTIMPORT from applications	 import MPI_start_end, header
	pass#IMPORTIMPORTIMPORT from pixel_error     import ordersegments

	pass#IMPORTIMPORTIMPORT from time            import time
	pass#IMPORTIMPORTIMPORT from copy 			 import copy
	pass#IMPORTIMPORTIMPORT from math 			 import sqrt
	pass#IMPORTIMPORTIMPORT import os
	pass#IMPORTIMPORTIMPORT import types

	"""Multiline Comment57"""

	nproc     = mpi.mpi_comm_size(mpi.MPI_COMM_WORLD)
	myid      = mpi.mpi_comm_rank(mpi.MPI_COMM_WORLD)
	main_node = 0

	search_rng   = int(search_rng)
	rng        = int(rng)
	if(pixel_size < 0.0 or dp < 0.0 ):  sparx_global_def.ERROR('Helical symmetry parameters have to be provided', "helicon_MPI", 1, myid)

	if os.path.exists(outdir):  sparx_global_def.ERROR('Output directory %s  exists, please change the name and restart the program'%outdir, "helicon_MPI", 1, myid)
	mpi.mpi_barrier(mpi.MPI_COMM_WORLD)

	if myid == main_node:
		os.mkdir(outdir)
		pass#IMPORTIMPORTIMPORT import global_def
		sparx_global_def.LOGFILE =  os.path.join(outdir, sparx_global_def.LOGFILE)
		sparx_utilities.print_begin_msg("helicon_MPI")
	mpi.mpi_barrier(mpi.MPI_COMM_WORLD)

	if debug:
		pass#IMPORTIMPORTIMPORT from time import sleep
		while not os.path.exists(outdir):
			print("Node ",myid,"  waiting...")
			time.sleep(5)

		info_file = os.path.join(outdir, "progress%04d"%myid)
		finfo = open(info_file, 'w')
	else:
		finfo = None
	max_iter = int(maxit)
	
	vol     = EMAN2_cppwrap.EMData()
	vol.read_image(ref_vol)
	nx      = vol.get_xsize()
	ny      = vol.get_ysize()
	nz      = vol.get_zsize()
	
	# Only handle square volumes now!
	if nz < nx:
		sparx_global_def.ERROR('Only handles square volumes .... nz cannot be less than nx', "helicon_MPI", 1, myid)
	
	# Pad to square
	if nz > nx:
		nx = nz
		ny = nz	
		vol = sparx_utilities.pad(vol, nx, ny,nz,background=0.0)
	
	
	if(sym[0] == "d"  or sym[0] == "D"):  Dsym = True
	else:                                 Dsym = False
	# restriction of phi search for the first segment
	symrestrict = int(sym[1])

	#  For the time being only one delta!!!
	delta       = sparx_utilities.get_input_from_string(delta)[0]

	if myid == main_node:
		pass#IMPORTIMPORTIMPORT import user_functions
		user_func = sparx_user_functions.factory[user_func_name]

		sparx_utilities.print_msg("Input stack                               : %s\n"%(stack))
		sparx_utilities.print_msg("Reference volume                          : %s\n"%(ref_vol))	
		sparx_utilities.print_msg("Output directory                          : %s\n"%(outdir))
		sparx_utilities.print_msg("Maskfile                                  : %s\n"%(maskfile))
		sparx_utilities.print_msg("Angular step                              : %s\n"%(delta))
		sparx_utilities.print_msg("Search for psi                            : %s\n"%(FindPsi))
		if FindPsi:
			sparx_utilities.print_msg("Maximum range for psi search              : %s\n"%(psi_max))
		sparx_utilities.print_msg("X-search range [pixels]                   : %f\n"%(search_rng))
		sparx_utilities.print_msg("X-search wobble [pixels]                  : %f\n"%(rng))
		sparx_utilities.print_msg("Y-search wobble	[pixels]                 : %f\n"%(ywobble))
		sparx_utilities.print_msg("Y-search step [pixels]                    : %f\n"%(ystep))
		sparx_utilities.print_msg("Pixel size [A]                            : %f\n"%(pixel_size))
		sparx_utilities.print_msg("dp [A]     (helical symmetry)             : %f\n"%(dp))
		sparx_utilities.print_msg("dphi       (helical symmetry)             : %f\n"%(dphi))
		sparx_utilities.print_msg("Maximum iteration                         : %i\n"%(max_iter))
		sparx_utilities.print_msg("CTF correction                            : %s\n"%(CTF))
		sparx_utilities.print_msg("Signal-to-Noise Ratio                     : %f\n"%(snr))
		sparx_utilities.print_msg("Symmetry group                            : %s\n"%(sym))
		sparx_utilities.print_msg("Fraction of the volume used to helicise   : %f\n"%(fract))
		sparx_utilities.print_msg("Segment height seg_ny                     : %s\n\n"%(seg_ny))
		sparx_utilities.print_msg("User function                             : %s\n"%(user_func_name))

	if maskfile:
		if type(maskfile) is bytes: mask3D = sparx_utilities.get_image(maskfile)
		else:                                  mask3D = maskfile
	else: mask3D = sparx_utilities.model_cylinder(rmax, nx, ny, nz)

	fscmask = mask3D
	if CTF:
		pass#IMPORTIMPORTIMPORT from reconstruction import recons3d_4nn_ctf_MPI
		pass#IMPORTIMPORTIMPORT from filter         import filt_ctf
	pass#IMPORTIMPORTIMPORT else:	 from reconstruction import recons3d_4nn_MPI

	if( myid == 0):
		infils = EMAN2_cppwrap.EMUtil.get_all_attributes(stack, "filament")
		ptlcoords = EMAN2_cppwrap.EMUtil.get_all_attributes(stack, 'ptcl_source_coord')
		filaments = sparx_pixel_error.ordersegments(infils, ptlcoords)
		total_nfils = len(filaments)
		inidl = [0]*total_nfils
		for i in range(total_nfils):  inidl[i] = len(filaments[i])
		linidl = sum(inidl)
		tfilaments = []
		for i in range(total_nfils):  tfilaments += filaments[i]
		del filaments
	else:
		total_nfils = 0
		linidl = 0
	total_nfils = sparx_utilities.bcast_number_to_all(total_nfils, source_node = main_node)
	if myid != main_node:
		inidl = [-1]*total_nfils
	inidl = sparx_utilities.bcast_list_to_all(inidl, myid, source_node = main_node)
	linidl = sparx_utilities.bcast_number_to_all(linidl, source_node = main_node)
	if myid != main_node: tfilaments = [-1]*linidl
	tfilaments = sparx_utilities.bcast_list_to_all(tfilaments, myid, source_node = main_node)
	filaments = []
	iendi = 0
	for i in range(total_nfils):
		isti = iendi
		iendi = isti+inidl[i]
		filaments.append(tfilaments[isti:iendi])
	del tfilaments,inidl

	if myid == main_node:
		sparx_utilities.print_msg("Total number of filaments:    : %i\n"%(total_nfils))
	if total_nfils< nproc:
		sparx_global_def.ERROR('number of CPUs (%i) is larger than the number of filaments (%i), please reduce the number of CPUs used'%(nproc, total_nfils), "helicon_MPI", 1,myid)

	#  balanced load
	temp = sparx_utilities.chunks_distribution([[len(filaments[i]), i] for i in range(len(filaments))], nproc)[myid:myid+1][0]
	filaments = [filaments[temp[i][1]] for i in range(len(temp))]
	nfils     = len(filaments)

	#filaments = [[0,1]]
	#print "filaments",filaments
	list_of_particles = []
	indcs = []
	k = 0
	for i in range(nfils):
		list_of_particles += filaments[i]
		k1 = k+len(filaments[i])
		indcs.append([k,k1])
		k = k1
		
	if slowIO:
		for iproc in range(number_of_proc):
			if myid ==iproc:
				data = EMAN2_cppwrap.EMData.read_images(stack, list_of_particles)
				print("Read %6d images on process  : %4d"%(len(list_of_particles),myid))
			mpi.mpi_barrier(mpi.MPI_COMM_WORLD)
	else: data = EMAN2_cppwrap.EMData.read_images(stack, list_of_particles)
		
	nima = len(data)
	#print  " READ IMAGES ", myid,nima,nproc

	#  Was integer, now it is float, in PIXELS!
	rise = dp/pixel_size

	data_nx = data[0].get_xsize()
	data_ny = data[0].get_ysize()
	
	if data_nx != data_ny:
		sparx_global_def.ERROR('Input projections must be square.', "helicon_MPI", 1,myid)

	if(nx != data_ny):
		sparx_global_def.ERROR('Height of reference volume must be same as dimension of input projections', "helicon_MPI", 1,myid)

	data_nn = max(data_nx, data_ny)
	segmask = sparx_utilities.pad(sparx_utilities.model_blank(2*int(rmax), seg_ny, 1, 1.0), data_nx, data_ny, 1, 0.0)
	fdata = [None]*nima
	resetatone = False
	for im in range(nima):
		data[im].set_attr('ID', list_of_particles[im])
		st = EMAN2_cppwrap.Util.infomask(data[im], segmask, False)
		data[im] -= st[0]
		data[im] /= st[1]
		if CTF:
			qctf = data[im].get_attr_default("ctf_applied", 0)
			if qctf == 0:
				ctf_params = data[im].get_attr("ctf")
				data[im] = sparx_filter.filt_ctf(data[im], ctf_params)
				data[im].set_attr('ctf_applied', 1)
			elif qctf != 1:
				sparx_global_def.ERROR('Incorrectly set ctf flag', "helicon_MPI", 1,myid)
		#  if FindPsi,  apply the angle to data[im], do fft and put in fdata[im]
		if FindPsi:
			phi,theta,psi,tsx,tsy = sparx_utilities.get_params_proj(data[im])
			if( theta != 0.0):
				if(abs(psi - 90.) < abs(psi - 270.0)):  gamma =  90.0
				else:                                   gamma = 270.0
				fdata[im] = sparx_fundamentals.fft( segmask*sparx_fundamentals.rot_shift2D(data[im], gamma-psi) )
			else:
				sparx_utilities.set_params_proj(data[im], [0.0, 90.0, 90.0, 0.0,0.0])
				fdata[im] = sparx_fundamentals.fft( segmask*data[im] )
		else:
			sparx_utilities.set_params_proj(data[im], [0.0, 90.0, 90.0, 0.0,0.0])
			fdata[im] = sparx_fundamentals.fft( segmask*data[im] )
		"""Multiline Comment58"""
	del list_of_particles

	if debug:
		finfo.write( '%d loaded  \n' % nima )
		finfo.flush()
	if myid == main_node:
		# initialize data for the reference preparation function
		# No centering for helical reconstruction
		ref_data = [None, mask3D, None, None, None ]

	#phiwobble = int(float(ywobble)/rise*dphi/delta+0.5)  # phiwobble is NOT in degrees, it is in nphi units
	phiwobble = int(float(phiwobble)/delta+0.5) #  convert phiwobble to nphi units

	pass#IMPORTIMPORTIMPORT from math import ceil
	nwx = 2*search_rng+3
	nwy = int(numpy.ceil(rise/2)*2+1+2*numpy.ceil(ywobble)+2)
	nwxc = nwx//2
	nwyc = nwy//2
	nphi = int(360.0/delta + 0.5)
	#print  "  params  ",nwx,nwy,nwxc,nwyc,nphi
	if FindPsi:
		mode = "F"
		cnx = data_nn//2+1
		cny = cnx
		numr = sparx_alignment.Numrinit(1, data_nn//2-max(nwxc,nwyc)-1, 1, mode)
		if myid == main_node:  print("  Radius Used  ",data_nn//2-max(nwxc,nwyc)-1)
		wr   = sparx_alignment.ringwe(numr, mode)
		maxrin = numr[len(numr)-1]
		crefim = [None]*nphi
	else:
		#  have to initialize them, otherwise problem with passing the arguments
		mode = "F"
		cnx = data_nx//2+1
		cny = cnx
		numr = []
		wr   = []
		maxrin = 0
		crefim = []

	terminate = 0
	Iter = 0
	while Iter < max_iter:
		Iter += 1
		if myid == main_node:
			start_time = time.time()
			sparx_utilities.print_msg("\nITERATION #%3d\n"%(Iter))

		volft, kbz = sparx_projection.prep_vol( vol )
		del vol

		refproj = [None]*nphi
		if( not Dsym):  rotproj = [None]*nphi
		else:           rotproj = []
		for iphi in range(nphi):
			refproj[iphi] = sparx_projection.prgs( volft, kbz, [delta*iphi, 90.0, 90.0, 0.0, 0.0])
			st = EMAN2_cppwrap.Util.infomask(refproj[iphi] , segmask, True)
			refproj[iphi] -= st[0]
			refproj[iphi] /= st[1]
			refproj[iphi] = EMAN2_cppwrap.Util.muln_img(refproj[iphi], segmask )

			if FindPsi:
				temp = EMAN2_cppwrap.Util.Polar2Dm(refproj[iphi], cnx, cny, numr, mode)
				EMAN2_cppwrap.Util.Frngs(temp, numr)
				EMAN2_cppwrap.Util.Applyws(temp, numr, wr)
				crefim[iphi] = temp
			#  rotated in-plane by 180 are equivalent to rot_shift3D(vol,-90,180.0,90) with phi running as phi
			if(not Dsym):  rotproj[iphi] = sparx_fundamentals.fft( segmask * (sparx_fundamentals.rot_shift2D(refproj[iphi],180.0)) )
			refproj[iphi] = sparx_fundamentals.fft( segmask*(refproj[iphi]) )
		del volft
		#exit()
		#if myid == main_node:  
		#astart_time = time()
		if myid == main_node:  start_time = time.time()
		terminate = 0
		for ifil in range(nfils):
			ldata = [data[im] for im in range(indcs[ifil][0],indcs[ifil][1])]
			#for im in xrange(len(ldata)):  ldata[im].set_attr("bestang", 10000.0)
			EMAN2_cppwrap.Util.constrained_helix_exhaustive(ldata, fdata[indcs[ifil][0]:indcs[ifil][1]], refproj, rotproj, [float(dp), float(dphi), rise, float(delta), ywobble, ystep], [int(nphi), symrestrict, int(phiwobble), int(rng), int(Dsym), int(nwx), int(nwy), int(nwxc), int(nwyc)], FindPsi, float(psi_max), crefim, numr, int(maxrin), mode, int(cnx), int(cny))
			#from development import constrained_helix
			#constrained_helix(ldata, fdata[indcs[ifil][0]:indcs[ifil][1]], refproj, rotproj, dp, dphi, rise, delta, nphi, symrestrict, phiwobble, rng, ywobble, ystep, Dsym, nwx, nwy, nwxc, nwyc, FindPsi, psi_max, crefim, numr, maxrin, mode, cnx, cny, myid, main_node)
			"""Multiline Comment59"""
			for im in range(indcs[ifil][0], indcs[ifil][1]):
				temp = EMAN2_cppwrap.Util.get_transform_params(ldata[im-indcs[ifil][0]], "xform.projection", "spider")
				sparx_utilities.set_params_proj(data[im],[temp["phi"],temp["theta"],temp["psi"],-temp["tx"],-temp["ty"]])
				#if not(doExhaustive):
				#	if Iter == 1 and resetatone:  data[im].set_attr('previousmax',-1.0e23)

			if FindPsi:
				for im in range(indcs[ifil][0], indcs[ifil][1]):
					fdata[im] = sparx_fundamentals.fft( segmask*sparx_fundamentals.rot_shift2D(data[im], ldata[im-indcs[ifil][0]].get_attr("bestang") ) )
					#bestang = ldata[im-indcs[ifil][0]].get_attr("bestang")
					#if( bestang < 10000.0): fdata[im] = fft( segmask*rot_shift2D(data[im], bestang ) )
			#print  "Parameters computed for filament",myid,ifil,time()-start_time;start_time = time()
		if myid == main_node:
			sparx_utilities.print_msg("Alignment time = %d\n"%(time.time()-start_time));start_time = time.time()

		del ldata
		del refproj
		if(not Dsym):  del rotproj
		#mpi_barrier(MPI_COMM_WORLD)

		"""Multiline Comment60"""

		# write out headers, under MPI writing has to be done sequentially
		mpi.mpi_barrier(mpi.MPI_COMM_WORLD)
		par_str = ['xform.projection', 'ID']   #, 'previousmax']
		if myid == main_node:
			start_time = time.time()
			if(sparx_utilities.file_type(stack) == "bdb"):
				pass#IMPORTIMPORTIMPORT from utilities import recv_attr_dict_bdb
				sparx_utilities.recv_attr_dict_bdb(main_node, stack, data, par_str, 0, nima, nproc)
			else:
				pass#IMPORTIMPORTIMPORT from utilities import recv_attr_dict
				sparx_utilities.recv_attr_dict(main_node, stack, data, par_str, 0, nima, nproc)
			sparx_utilities.print_msg("Time to write header information= %d\n"%(time.time()-start_time))
			start_time = time.time()
		else:		sparx_utilities.send_attr_dict(main_node, data, par_str, 0, nima)
		if myid == main_node:
			# write params to text file
			header(stack, params='xform.projection', fexport=os.path.join(outdir, "parameters%04d.txt"%Iter))
			#header(stack, params='previousmax', fexport=os.path.join(outdir, "previousmax%04d.txt"%Iter))
		mpi.mpi_barrier(mpi.MPI_COMM_WORLD)

		#if myid == main_node:
		#	print_msg("Time of alignment = %\n"%(time()-astart_time));start_time = time()
		if CTF:  vol = sparx_reconstruction.recons3d_4nn_ctf_MPI(myid, data, symmetry=sym, snr = snr, npad = npad)
		else:    vol = sparx_reconstruction.recons3d_4nn_MPI(myid, data, symmetry=sym, snr = snr, npad = npad)
		if myid == main_node:
			sparx_utilities.print_msg("3D reconstruction time = %d\n"%(time.time()-start_time));start_time = time.time()

		if myid == main_node:
			#vol.write_image(os.path.join(outdir, "vol%03d.hdf"%Iter))
			vol = vol.helicise(pixel_size, dp, dphi, fract, rmax, rmin)
			vol = sparx_utilities.sym_vol(vol, symmetry=sym)
			ref_data[0] = vol
			vol = user_func(ref_data)
			vol = vol.helicise(pixel_size, dp, dphi, fract, rmax, rmin)
			vol = sparx_utilities.sym_vol(vol, symmetry=sym)
			vol.write_image(os.path.join(outdir, "volf%03d.hdf"%Iter))
		sparx_utilities.bcast_EMData_to_all(vol, myid, main_node)
	if myid == main_node: sparx_utilities.print_end_msg("helicon_MPI")

def localhelicon_MPInew(stack, ref_vol, outdir, seg_ny, maskfile, ir, ou, rs, xr, ynumber,\
						txs, delta, initial_theta, delta_theta, an, maxit, CTF, snr, dp, dphi, psi_max,\
						rmin, rmax, fract,  npad, sym, user_func_name, \
						pixel_size, debug, y_restrict, search_iter, slowIO):

	pass#IMPORTIMPORTIMPORT from alignment      import Numrinit, proj_ali_helicon_local, proj_ali_helicon_90_local_direct, directaligridding1, directaligriddingconstrained
	pass#IMPORTIMPORTIMPORT from utilities      import model_circle, get_image, drop_image, get_input_from_string, pad, model_blank
	pass#IMPORTIMPORTIMPORT from utilities      import bcast_list_to_all, bcast_number_to_all, reduce_EMData_to_root, bcast_EMData_to_all
	pass#IMPORTIMPORTIMPORT from utilities      import send_attr_dict, read_text_row, sym_vol
	pass#IMPORTIMPORTIMPORT from utilities      import get_params_proj, set_params_proj, file_type, chunks_distribution
	pass#IMPORTIMPORTIMPORT from fundamentals   import rot_avg_image
	pass#IMPORTIMPORTIMPORT from applications 	import setfilori_SP, filamentupdown, prepare_refffts
	pass#IMPORTIMPORTIMPORT from pixel_error    import max_3D_pixel_error, ordersegments
	pass#IMPORTIMPORTIMPORT from utilities      import print_begin_msg, print_end_msg, print_msg
	pass#IMPORTIMPORTIMPORT from mpi            import mpi_bcast, mpi_comm_size, mpi_comm_rank, MPI_FLOAT, MPI_COMM_WORLD, mpi_barrier
	pass#IMPORTIMPORTIMPORT from mpi            import mpi_recv,  mpi_send, MPI_TAG_UB
	pass#IMPORTIMPORTIMPORT from mpi            import mpi_reduce, MPI_INT, MPI_SUM
	pass#IMPORTIMPORTIMPORT from filter         import filt_ctf
	pass#IMPORTIMPORTIMPORT from projection     import prep_vol, prgs
	pass#IMPORTIMPORTIMPORT from statistics     import hist_list, varf3d_MPI
	pass#IMPORTIMPORTIMPORT from applications   import MPI_start_end, header, prepare_helical_refangles, prepare_reffft1
	pass#IMPORTIMPORTIMPORT from EMAN2          import Vec2f, Processor
	pass#IMPORTIMPORTIMPORT from math			import sin, cos, radians
	pass#IMPORTIMPORTIMPORT from string         import lower, split
	pass#IMPORTIMPORTIMPORT from copy           import copy
	pass#IMPORTIMPORTIMPORT import  os
	pass#IMPORTIMPORTIMPORT import  types

	number_of_proc = mpi.mpi_comm_size(mpi.MPI_COMM_WORLD)
	myid           = mpi.mpi_comm_rank(mpi.MPI_COMM_WORLD)
	main_node = 0

	if myid == 0:
		if os.path.exists(outdir):  nx = 1
		else:  nx = 0
	else:  nx = 0
	ny = sparx_utilities.bcast_number_to_all(nx, source_node = main_node)

	if ny == 1:  sparx_global_def.ERROR('Output directory exists, please change the name and restart the program', "localhelicon_MPI", 1,myid)
	mpi.mpi_barrier(mpi.MPI_COMM_WORLD)

	if myid == main_node:
		os.mkdir(outdir)
		pass#IMPORTIMPORTIMPORT import global_def
		sparx_global_def.LOGFILE =  os.path.join(outdir, sparx_global_def.LOGFILE)
		sparx_utilities.print_begin_msg("localhelicon_MPI NEW")
	mpi.mpi_barrier(mpi.MPI_COMM_WORLD)

	if debug:
		pass#IMPORTIMPORTIMPORT from time import sleep
		while not os.path.exists(outdir):
			print("Node ",myid,"  waiting...")
			time.sleep(5)

		info_file = os.path.join(outdir, "progress%04d"%myid)
		finfo = open(info_file, 'w')
	else:
		finfo = None

	sym    = sym.lower()
	symref = "s"+sym

	ref_a           = "P"
	symmetry_string = string.split(sym)[0]

	xrng        = sparx_utilities.get_input_from_string(xr)
	y_restrict  = sparx_utilities.get_input_from_string(y_restrict)
	ynumber	    = sparx_utilities.get_input_from_string(ynumber)
	for i in range(len(ynumber)):
		if ynumber[i] > 0:
			if(ynumber[i]%2==1): ynumber[i]=ynumber[i]+1
	yrng = []

	for i in range(len(xrng)): yrng.append(dp/2)

	stepx       = sparx_utilities.get_input_from_string(txs)
	delta       = sparx_utilities.get_input_from_string(delta)
	lstp = min(len(xrng), len(yrng), len(stepx), len(delta))
	an = sparx_utilities.get_input_from_string(an)

	if len(an) == 1:
		an = [an[0] for ii in range(lstp)]
	y_restrict = y_restrict[0:lstp]
	for i in range(lstp):
		if an[i] < 0 and y_restrict[i] < 0: 
			sparx_global_def.ERROR('This is a local search, an and y_restrict should not both be -1', "localhelicon_MPI", 1,myid)
		if y_restrict[i] < 0:   y_restrict[i] = (an[i]/dphi)*(dp/pixel_size)/2.0
		if an[i] < 0:           an[i] = ((2.0*y_restrict[i])/(dp/pixel_size)) * dphi

	first_ring  = int(ir)
	rstep       = int(rs)
	last_ring   = int(ou)
	max_iter    = int(maxit)
	search_iter = int(search_iter)
	totmax_iter = max_iter * search_iter

	vol     = EMAN2_cppwrap.EMData()
	vol.read_image(ref_vol)
	nx      = vol.get_xsize()
	ny      = vol.get_ysize()
	nz      = vol.get_zsize()

	if nz < nx:
		sparx_global_def.ERROR('Do not handle squat volumes .... nz cannot be less than nx', "localhelicon_MPI", 1, myid)

	# Pad to square
	if nz > nx:
		nx = nz
		ny = nz	
		vol = sparx_utilities.pad(vol, nx, ny,nz,background=0.0)	
	nmax = max(nx, ny, nz)

	if myid == main_node:
		pass#IMPORTIMPORTIMPORT import user_functions
		user_func = sparx_user_functions.factory[user_func_name]

		sparx_utilities.print_msg("Input stack                               : %s\n"%(stack))
		sparx_utilities.print_msg("Reference volume                          : %s\n"%(ref_vol))	
		sparx_utilities.print_msg("Output directory                          : %s\n"%(outdir))
		sparx_utilities.print_msg("Maskfile                                  : %s\n"%(maskfile))
		sparx_utilities.print_msg("Inner radius for psi angle search         : %i\n"%(first_ring))
		sparx_utilities.print_msg("Outer radius for psi angle search         : %i\n"%(last_ring))
		sparx_utilities.print_msg("Ring step                                 : %i\n"%(rstep))
		sparx_utilities.print_msg("X search range                            : %s\n"%(xrng))
		sparx_utilities.print_msg("Y search range                            : %s\n"%(y_restrict))
		sparx_utilities.print_msg("Y number                                  : %s\n"%(ynumber))
		sparx_utilities.print_msg("Translational stepx                       : %s\n"%(stepx))
		sparx_utilities.print_msg("Angular step                              : %s\n"%(delta))
		sparx_utilities.print_msg("Angular search range                      : %s\n"%(an))
		sparx_utilities.print_msg("Intial theta for out-of-plane tilt search : %s\n"%(initial_theta))
		sparx_utilities.print_msg("Delta theta for out-of-plane tilt search  : %s\n"%(delta_theta))
		sparx_utilities.print_msg("Min radius for application of helical symmetry (in pix)    : %5.4f\n"%(rmin))
		sparx_utilities.print_msg("Max radius for application of helical symmetry (in pix)    : %5.4f\n"%(rmax))
		sparx_utilities.print_msg("Fraction of volume used for application of helical symmetry: %5.4f\n"%(fract))
		sparx_utilities.print_msg("Helical symmetry - axial rise   [A]       : %5.4f\n"%(dp))
		sparx_utilities.print_msg("Helical symmetry - angle                  : %5.4f\n"%(dphi))
		sparx_utilities.print_msg("Maximum number of iterations              : %i\n"%(max_iter))
		sparx_utilities.print_msg("Number of iterations to predict/search before doing reconstruction and updating reference volume : %i\n"%(search_iter))
		sparx_utilities.print_msg("Data with CTF                             : %s\n"%(CTF))
		sparx_utilities.print_msg("Signal-to-Noise Ratio                     : %5.4f\n"%(snr))
		sparx_utilities.print_msg("npad                                      : %i\n"%(npad))
		sparx_utilities.print_msg("User function                             : %s\n"%(user_func_name))
		sparx_utilities.print_msg("Pixel size [A]                            : %f\n"%(pixel_size))
		sparx_utilities.print_msg("Point-group symmetry group                : %s\n"%(sym))
		sparx_utilities.print_msg("Segment height seg_ny                     : %s\n\n"%(seg_ny))

	if maskfile:
		if type(maskfile) is bytes: mask3D = sparx_utilities.get_image(maskfile)
		else:                                  mask3D = maskfile
	else: mask3D = None
	#else: mask3D = model_circle(last_ring, nx, nx, nx)

	if CTF:
		pass#IMPORTIMPORTIMPORT from reconstruction import recons3d_4nn_ctf_MPI
		pass#IMPORTIMPORTIMPORT from filter         import filt_ctf
	pass#IMPORTIMPORTIMPORT else:	 from reconstruction import recons3d_4nn_MPI

	if( myid == 0):
		infils = EMAN2_cppwrap.EMUtil.get_all_attributes(stack, "filament")
		ptlcoords = EMAN2_cppwrap.EMUtil.get_all_attributes(stack, 'ptcl_source_coord')
		filaments = sparx_pixel_error.ordersegments(infils, ptlcoords)
		total_nfils = len(filaments)
		inidl = [0]*total_nfils
		for i in range(total_nfils):  inidl[i] = len(filaments[i])
		linidl = sum(inidl)
		tfilaments = []
		for i in range(total_nfils):  tfilaments += filaments[i]
		del filaments
	else:
		total_nfils = 0
		linidl = 0
	total_nfils = sparx_utilities.bcast_number_to_all(total_nfils, source_node = main_node)
	if myid != main_node:
		inidl = [-1]*total_nfils
	inidl = sparx_utilities.bcast_list_to_all(inidl, myid, source_node = main_node)
	linidl = sparx_utilities.bcast_number_to_all(linidl, source_node = main_node)
	if myid != main_node:
		tfilaments = [-1]*linidl
	tfilaments = sparx_utilities.bcast_list_to_all(tfilaments, myid, source_node = main_node)
	filaments = []
	iendi = 0
	for i in range(total_nfils):
		isti = iendi
		iendi = isti+inidl[i]
		filaments.append(tfilaments[isti:iendi])
	del tfilaments,inidl

	if myid == main_node:
		sparx_utilities.print_msg("total number of filaments in the data:  %i\n"%(total_nfils))
	if total_nfils< number_of_proc:
		sparx_global_def.ERROR('number of CPUs (%i) is larger than the number of filaments (%i), please reduce the number of CPUs used'%(number_of_proc, total_nfils), "localhelicon_MPI", 1,myid)

	#  balanced load
	temp = sparx_utilities.chunks_distribution([[len(filaments[i]), i] for i in range(len(filaments))], number_of_proc)[myid:myid+1][0]
	filaments = [filaments[temp[i][1]] for i in range(len(temp))]
	nfils     = len(filaments)
	list_of_particles = []
	indcs = []
	k = 0
	for i in range(nfils):
		list_of_particles += filaments[i]
		k1 = k+len(filaments[i])
		indcs.append([k,k1])
		k = k1

	if slowIO:
		for iproc in range(number_of_proc):
			if myid ==iproc:
				data = EMAN2_cppwrap.EMData.read_images(stack, list_of_particles)
				print("Read %6d images on process  : %4d"%(len(list_of_particles),myid))
			mpi.mpi_barrier(mpi.MPI_COMM_WORLD)
	else: data = EMAN2_cppwrap.EMData.read_images(stack, list_of_particles)

	
	nima = len(data)
	data_nx = data[0].get_xsize()
	data_ny = data[0].get_ysize()
	if ((nx < data_nx) or (data_nx != data_ny)):
		sparx_global_def.ERROR('Images should be square with nx and ny equal to nz of reference volume', "localhelicon_MPI", 1, myid)
	data_nn = max(data_nx, data_ny)

	segmask = sparx_utilities.pad(sparx_utilities.model_blank(2*rmax+1, seg_ny, 1, 1.0), data_nx, data_ny, 1, 0.0)
	
	if last_ring < 0:
		last_ring = (max(seg_ny, 2*int(rmax)))//2 - 2
		
	numr	= sparx_alignment.Numrinit(first_ring, last_ring, rstep, "F")

	maxrin = numr[len(numr)-1]
	psistep = 360./maxrin
	print("psistep", psistep)
	#if fourvar:  original_data = []
	for im in range(nima):
		data[im].set_attr('ID', list_of_particles[im])
		sttt = EMAN2_cppwrap.Util.infomask(data[im], segmask, False)
		data[im] -= sttt[0]
		#if fourvar: original_data.append(data[im].copy())
		if CTF:
			st = data[im].get_attr_default("ctf_applied", 0)
			if(st == 0):
				ctf_params = data[im].get_attr("ctf")
				data[im] = sparx_filter.filt_ctf(data[im], ctf_params)
				data[im].set_attr('ctf_applied', 1)


	M = data_nn
	alpha = 1.75
	K = 6
	N = M*2  # npad*image size
	r = M/2
	v = K/2.0/N
	params = {"filter_type" : EMAN2_cppwrap.Processor.fourier_filter_types.KAISER_SINH_INVERSE,
	          "alpha" : alpha, "K":K,"r":r,"v":v,"N":N}
	kb = EMAN2_cppwrap.Util.KaiserBessel(alpha, K, r, v, N)
	dataft = [None]*nima
	for im in range(nima):
		dataft[im] = data[im].FourInterpol(N, N, 1,0)
		dataft[im] = EMAN2_cppwrap.Processor.EMFourierFilter(dataft[im] ,params)

	if debug:
		finfo.write( '%d loaded  \n' % nima )
		finfo.flush()

	for i in range(len(xrng)): yrng[i]=max(int(dp/(2*pixel_size)+0.5),1)
	for i in range(len(xrng)): xrng[i]=max(int(xrng[i]),1)

	if myid == main_node:
		sparx_utilities.print_msg("Pixel size in Angstroms                   : %5.4f\n"%(pixel_size))
		sparx_utilities.print_msg("Y search range (pix) initialized as       : %s\n\n"%(yrng))

	#  set attribute updown for each filament, up will be 0, down will be 1
	for ivol in range(nfils):
		seg_start = indcs[ivol][0]
		seg_end   = indcs[ivol][1]
		filamentupdown(data[seg_start: seg_end], pixel_size, dp, dphi)


	if debug:
		finfo.write("seg_start, seg_end: %d %d\n" %(seg_start, seg_end))
		finfo.flush()

	pass#IMPORTIMPORTIMPORT from time import time

	total_iter = 0
	for ii in range(lstp):
		if stepx[ii] == 0.0:
			if xrng[ii] != 0.0:
				sparx_global_def.ERROR('xrange step size cannot be zero', "localhelicon_MPI", 1,myid)
			else:
				stepx[ii] = 1.0 # this is to prevent division by zero in c++ code

	#  TURN INTO PARAMETER OR COMPUTE FROM OU
	#psistep=0.5
	# do the projection matching
	startl = time.time()
	ooiter = 0
	for N_step in range(lstp):
		terminate = 0
		Iter = 0
		ant = numpy.cos(numpy.radians(an[N_step]))
		while(Iter < totmax_iter and terminate == 0):
			yrng[N_step]=float(dp)/(2*pixel_size) #will change it later according to dp
			#yrng[N_step]=max(int(yrng[N_step]+0.5),1)
			if(ynumber[N_step]==0): 
				yrng[N_step]= 0
				stepy = 1.0
			else:                   stepy = (2*yrng[N_step]/ynumber[N_step])
			
			#stepx = stepy
			if stepy < 0.1:
				sparx_global_def.ERROR('yrange step size cannot be lower than 0.1', "localhelicon_MPInew", 1,myid)
			pixer  = [0.0]*nima

			neworient = [[0.0, 0.0, 0.0, 0.0, 0.0, -2.0e23] for i in range(nima)]
			ooiter += 1
			Iter += 1
			if Iter%search_iter == 0:  total_iter += 1
			if myid == main_node:
				start_time = time.time()
				sparx_utilities.print_msg("\n (localhelicon_MPI) ITERATION #%3d,  inner iteration #%3d\nDelta = %4.1f, an = %5.4f, xrange (Pixels) = %5.4f,stepx (Pixels) = %5.4f, yrng (Pixels) = %5.4f,  stepy (Pixels) = %5.4f, y_restrict (Pixels)=%5.4f, ynumber = %3d\n"\
				%(total_iter, Iter, delta[N_step], an[N_step], xrng[N_step], stepx[N_step], yrng[N_step], stepy, y_restrict[N_step], ynumber[N_step]))
				print("ITERATION   ",total_iter)

			volft,kbv = sparx_projection.prep_vol( vol )


			"""Multiline Comment61"""
			#  WHAT DOES IT DO?
			pass#IMPORTIMPORTIMPORT from numpy import float32
			dpp = numpy.float32(float(dp)/pixel_size)
			dpp = float( dpp )
			dpp_half = dpp/2.0

			Torg = []
			for ivol in range(nfils):

				seg_start = indcs[ivol][0]
				seg_end   = indcs[ivol][1]
				for im in range( seg_start, seg_end ):
					Torg.append(data[im].get_attr('xform.projection'))

				#Fit predicted locations as new starting points
#  				if (seg_end - seg_start) > 1:
#  					setfilori_SP(data[seg_start: seg_end], pixel_size, dp, dphi)
				
			#  Generate list of reference angles, all nodes have the entire list
			ref_angles = prepare_helical_refangles(delta[N_step], initial_theta =initial_theta, delta_theta = delta_theta)
			#  count how many projections did not have a peak.  If too many, something is wrong
			nopeak = 0
			#  DO ORIENTATION SEARCHES
			#print "len_refang", len(ref_angles)

			reftime = 0
			searchtime = 0
	
			for refang in ref_angles:
				n1 = numpy.sin(numpy.radians(refang[1]))*numpy.cos(numpy.radians(refang[0]))
				n2 = numpy.sin(numpy.radians(refang[1]))*numpy.sin(numpy.radians(refang[0]))
				n3 = numpy.cos(numpy.radians(refang[1]))

				refrings = [None]

				for ivol in range(nfils):

					seg_start = indcs[ivol][0]
					seg_end   = indcs[ivol][1]

					for im in range( seg_start, seg_end ):

						#  Here I have to figure for local search whether given image has to be matched with this refproj dir
						ID = data[im].get_attr("ID")
						phi, theta, psi, tx, ty = sparx_utilities.get_params_proj(data[im])
						if finfo:
							finfo.write("Image id: %6d\n"%(ID))
							finfo.write("Old parameters: %9.4f %9.4f %9.4f %9.4f %9.4f\n"%(phi, theta, psi, tx, ty))
							finfo.flush()
						#  Determine whether segment is up and down and search for psi in one orientation only.
						imn1 = numpy.sin(numpy.radians(theta))*numpy.cos(numpy.radians(phi))
						imn2 = numpy.sin(numpy.radians(theta))*numpy.sin(numpy.radians(phi))
						imn3 = numpy.cos(numpy.radians(theta))
						if( (n1*imn1 + n2*imn2 + n3*imn3)>=ant ):

						

							if psi < 180.0 :  direction = "up"
							else:             direction = "down"
							
							if(refrings[0] == None):
			
								startref = time.time()
								refrings = prepare_reffft2(volft, kbv, refang, segmask, psi_max, psistep)
								endref = time.time()
								#print " compute ref timing",  endref-startref,  refang,  idd
								reftime = reftime + endref-startref
							#angb, tx, ty, pik = directaligridding1(dataft[im], kb, refrings, \
							#	psi_max, psistep, xrng[N_step], yrng[N_step], stepx, stepy, direction)



							#  Constrained search methodology
							#		x - around previously found location tx +/- xrng[N_step] in stepx
							#		y - around previously found location ty +/- yrng[N_step] in stepy
							#		psi - around previously found position psi +/- psi_max in steps psistep
							#		phi and theta are restricted by parameter an above.
							#
							#
							
							
							#tyrng = max(stepy,min(yrng[N_step],abs(y_restrict[N_step]-ty),abs(-y_restrict[N_step]-ty)))
							tyrng = min(yrng[N_step],abs(y_restrict[N_step]))
							#print  "IMAGE  ",im
		
							startsearch = time.time()
							angb, tx, ty, pik, = sparx_alignment.directaligriddingconstrained(dataft[im], kb, refrings, \
								psi_max, psistep, xrng[N_step], tyrng, stepx[N_step], stepy, psi, tx, ty, direction)
							endsearch = time.time()	
							#print "image id, searching time is ", im, endsearch-startsearch, iddi
							searchtime = searchtime +endsearch-startsearch 
							if(pik > -1.0e23):
								if(pik > neworient[im][-1]):
									neworient[im][-1] = pik
									neworient[im][:4] = [angb, tx, ty, refang]
						
			print("total time of calculating references", reftime)
			print("total time of searching", searchtime)
			
			for im in range(nima):
				if(neworient[im][-1] > -1.0e23):
					#from utilities import inverse_transform2
					#t1, t2, t3, tp = inverse_transform2(neworient[im][3][1]+neworient[im][0])
					tp = EMAN2_cppwrap.Transform({"type":"spider","phi":neworient[im][3][0],"theta":neworient[im][3][1],"psi":neworient[im][3][2]+neworient[im][0]})
					tp.set_trans( EMAN2_cppwrap.Vec2f( neworient[im][1], neworient[im][2] ) )
					data[im].set_attr("xform.projection", tp)
					pass#IMPORTIMPORTIMPORT from utilities import get_params_proj
					#print  "  PARAMS ",im,get_params_proj(data[im])
					pixer[im]  = sparx_pixel_error.max_3D_pixel_error(Torg[im], tp, last_ring)
					data[im].set_attr("pixerr", pixer[im])

				else:
					# peak not found, parameters not modified
					print("peak not found image id", im)
					nopeak += 1
					pixer[im]  = 0.0
					data[im].set_attr("pixerr", pixer[im])
					#data[im].set_attr('xform.projection', Torg[im-seg_start])

			nopeak = mpi.mpi_reduce(nopeak, 1, mpi.MPI_INT, mpi.MPI_SUM, main_node, mpi.MPI_COMM_WORLD)
			if myid == main_node:
				sparx_utilities.print_msg("Time of alignment = %d\n"%(time.time()-start_time))
				sparx_utilities.print_msg("Number of segments without a peak = %d\n"%(nopeak))
				start_time = time.time()

			mpi.mpi_barrier(mpi.MPI_COMM_WORLD)

			terminate = mpi.mpi_bcast(terminate, 1, mpi.MPI_INT, 0, mpi.MPI_COMM_WORLD)
			terminate = int(terminate[0])

			mpi.mpi_barrier(mpi.MPI_COMM_WORLD)

			if (Iter-1) % search_iter == 0 :

				if CTF:  vol = sparx_reconstruction.recons3d_4nn_ctf_MPI(myid, data, symmetry=sym, snr = snr, npad = npad)
				else:    vol = sparx_reconstruction.recons3d_4nn_MPI(myid, data, symmetry=sym, npad = npad)

				if myid == main_node:
					sparx_utilities.print_msg("3D reconstruction time = %d\n"%(time.time()-start_time))
					start_time = time.time()

					#drop_image(vol, os.path.join(outdir, "vol%04d.hdf"%(total_iter)))

					#  symmetry is imposed
					vol = vol.helicise(pixel_size, dp, dphi, fract, rmax, rmin)
					sparx_utilities.print_msg("Imposed delta z and delta phi      : %s,    %s\n"%(dp,dphi))
					vol = sparx_utilities.sym_vol(vol, symmetry=sym)
					ref_data = [vol, mask3D]
					#if  fourvar:  ref_data.append(varf)
					vol = user_func(ref_data)
					vol = vol.helicise(pixel_size, dp, dphi, fract, rmax, rmin)
					vol = sparx_utilities.sym_vol(vol, symmetry=sym)
					sparx_utilities.drop_image(vol, os.path.join(outdir, "volf%04d.hdf"%(total_iter)))
					sparx_utilities.print_msg("Symmetry enforcement and user function time = %d\n"%(time.time()-start_time))
					start_time = time.time()

				# using current volume
				sparx_utilities.bcast_EMData_to_all(vol, myid, main_node)

			mpi.mpi_barrier(mpi.MPI_COMM_WORLD)

			# write out headers, under MPI writing has to be done sequentially
			pass#IMPORTIMPORTIMPORT from mpi import mpi_recv, mpi_send, MPI_TAG_UB, MPI_COMM_WORLD, MPI_FLOAT
			par_str = ['xform.projection', 'ID','pixerr']
			if myid == main_node:
				if(sparx_utilities.file_type(stack) == "bdb"):
					pass#IMPORTIMPORTIMPORT from utilities import recv_attr_dict_bdb
					sparx_utilities.recv_attr_dict_bdb(main_node, stack, data, par_str, 0, nima, number_of_proc)
				else:
					pass#IMPORTIMPORTIMPORT from utilities import recv_attr_dict
					sparx_utilities.recv_attr_dict(main_node, stack, data, par_str, 0, nima, number_of_proc)
			else:
				sparx_utilities.send_attr_dict(main_node, data, par_str, 0, nima)

			if myid == main_node:
				# write params to text file
				header(stack, params='xform.projection', fexport=os.path.join(outdir, "parameters%04d.txt"%(ooiter)))
				header(stack, params='pixerr', fexport=os.path.join(outdir, "pixelerror%04d.txt"%(ooiter)))
	print("new method runing time", time.time()-startl)
def localhelicon_MPIming(stack, ref_vol, outdir, seg_ny, maskfile, ir, ou, rs, xr, ynumber,\
						txs, delta, initial_theta, delta_theta, an, maxit, CTF, snr, dp, dphi, psi_max,\
						rmin, rmax, fract,  npad, sym, user_func_name, \
						pixel_size, debug, y_restrict, search_iter, snakeknots, slowIO):
	pass#IMPORTIMPORTIMPORT from alignment      import proj_ali_helicon_local, proj_ali_helicon_90_local_direct, directaligridding1, directaligriddingconstrained, directaligriddingconstrained3dccf, alignment3Dsnake
	pass#IMPORTIMPORTIMPORT from utilities      import model_circle, get_image, drop_image, get_input_from_string, pad, model_blank
	pass#IMPORTIMPORTIMPORT from utilities      import bcast_list_to_all, bcast_number_to_all, reduce_EMData_to_root, bcast_EMData_to_all
	pass#IMPORTIMPORTIMPORT from utilities      import send_attr_dict, read_text_row, sym_vol
	pass#IMPORTIMPORTIMPORT from utilities      import get_params_proj, set_params_proj, file_type, chunks_distribution
	pass#IMPORTIMPORTIMPORT from fundamentals   import rot_avg_image
	pass#IMPORTIMPORTIMPORT from applications 	import setfilori_SP, filamentupdown, prepare_refffts
	pass#IMPORTIMPORTIMPORT from pixel_error    import max_3D_pixel_error, ordersegments
	pass#IMPORTIMPORTIMPORT from utilities      import print_begin_msg, print_end_msg, print_msg
	pass#IMPORTIMPORTIMPORT from mpi            import mpi_bcast, mpi_comm_size, mpi_comm_rank, MPI_FLOAT, MPI_COMM_WORLD, mpi_barrier
	pass#IMPORTIMPORTIMPORT from mpi            import mpi_recv,  mpi_send
	pass#IMPORTIMPORTIMPORT from mpi            import mpi_reduce, MPI_INT, MPI_SUM
	pass#IMPORTIMPORTIMPORT from filter         import filt_ctf
	pass#IMPORTIMPORTIMPORT from projection     import prep_vol, prgs
	pass#IMPORTIMPORTIMPORT from statistics     import hist_list, varf3d_MPI
	pass#IMPORTIMPORTIMPORT from applications   import MPI_start_end, header, prepare_helical_refangles, prepare_reffft1
	pass#IMPORTIMPORTIMPORT from EMAN2          import Vec2f, Processor
	pass#IMPORTIMPORTIMPORT from math			import sin, cos, radians
	pass#IMPORTIMPORTIMPORT from string         import lower, split
	pass#IMPORTIMPORTIMPORT from copy           import copy
	pass#IMPORTIMPORTIMPORT import  os
	pass#IMPORTIMPORTIMPORT import  types

	number_of_proc = mpi.mpi_comm_size(mpi.MPI_COMM_WORLD)
	myid           = mpi.mpi_comm_rank(mpi.MPI_COMM_WORLD)
	main_node = 0

	if myid == 0:
		if os.path.exists(outdir):  nx = 1
		else:  nx = 0
	else:  nx = 0
	ny = sparx_utilities.bcast_number_to_all(nx, source_node = main_node)

	if ny == 1:  sparx_global_def.ERROR('Output directory exists, please change the name and restart the program', "localhelicon_MPI", 1,myid)
	mpi.mpi_barrier(mpi.MPI_COMM_WORLD)

	if myid == main_node:
		os.mkdir(outdir)
		pass#IMPORTIMPORTIMPORT import global_def
		sparx_global_def.LOGFILE =  os.path.join(outdir, sparx_global_def.LOGFILE)
		sparx_utilities.print_begin_msg("localhelicon_MPI NEW")
	mpi.mpi_barrier(mpi.MPI_COMM_WORLD)

	if debug:
		pass#IMPORTIMPORTIMPORT from time import sleep
		while not os.path.exists(outdir):
			print("Node ",myid,"  waiting...")
			time.sleep(5)

		info_file = os.path.join(outdir, "progress%04d"%myid)
		finfo = open(info_file, 'w')
	else:
		finfo = None

	sym    = sym.lower()
	symref = "s"+sym

	ref_a           = "P"
	symmetry_string = string.split(sym)[0]

	xrng        = sparx_utilities.get_input_from_string(xr)
	y_restrict  = sparx_utilities.get_input_from_string(y_restrict)
	ynumber	    = sparx_utilities.get_input_from_string(ynumber)
	for i in range(len(ynumber)):
		if ynumber[i] > 0:
			if(ynumber[i]%2==1): ynumber[i]=ynumber[i]+1
	yrng = []

	for i in range(len(xrng)): yrng.append(dp/2)

	stepx       = sparx_utilities.get_input_from_string(txs)
	delta       = sparx_utilities.get_input_from_string(delta)
	lstp = min(len(xrng), len(yrng), len(stepx), len(delta))
	an = sparx_utilities.get_input_from_string(an)

	if len(an) == 1:
		an = [an[0] for ii in range(lstp)]
	y_restrict = y_restrict[0:lstp]
	for i in range(lstp):
		if an[i] < 0 and y_restrict[i] < 0: 
			sparx_global_def.ERROR('This is a local search, an and y_restrict should not both be -1', "localhelicon_MPI", 1,myid)
		if y_restrict[i] < 0:   y_restrict[i] = (an[i]/dphi)*(dp/pixel_size)/2.0
		if an[i] < 0:           an[i] = ((2.0*y_restrict[i])/(dp/pixel_size)) * dphi

	first_ring  = int(ir)
	rstep       = int(rs)
	last_ring   = int(ou)
	max_iter    = int(maxit)
	search_iter = int(search_iter)
	totmax_iter = max_iter * search_iter

	vol     = EMAN2_cppwrap.EMData()
	vol.read_image(ref_vol)
	nx      = vol.get_xsize()
	ny      = vol.get_ysize()
	nz      = vol.get_zsize()

	if nz < nx:
		sparx_global_def.ERROR('Do not handle squat volumes .... nz cannot be less than nx', "localhelicon_MPI", 1, myid)

	# Pad to square
	if nz > nx:
		nx = nz
		ny = nz	
		vol = sparx_utilities.pad(vol, nx, ny,nz,background=0.0)	
	nmax = max(nx, ny, nz)

	if myid == main_node:
		pass#IMPORTIMPORTIMPORT import user_functions
		user_func = sparx_user_functions.factory[user_func_name]

		sparx_utilities.print_msg("Input stack                               : %s\n"%(stack))
		sparx_utilities.print_msg("Reference volume                          : %s\n"%(ref_vol))	
		sparx_utilities.print_msg("Output directory                          : %s\n"%(outdir))
		sparx_utilities.print_msg("Maskfile                                  : %s\n"%(maskfile))
		sparx_utilities.print_msg("Inner radius for psi angle search         : %i\n"%(first_ring))
		sparx_utilities.print_msg("Outer radius for psi angle search         : %i\n"%(last_ring))
		sparx_utilities.print_msg("Ring step                                 : %i\n"%(rstep))
		sparx_utilities.print_msg("X search range                            : %s\n"%(xrng))
		sparx_utilities.print_msg("Y search range                            : %s\n"%(y_restrict))
		sparx_utilities.print_msg("Y number                                  : %s\n"%(ynumber))
		sparx_utilities.print_msg("Translational stepx                       : %s\n"%(stepx))
		sparx_utilities.print_msg("Angular step                              : %s\n"%(delta))
		sparx_utilities.print_msg("Angular search range                      : %s\n"%(an))
		sparx_utilities.print_msg("Intial theta for out-of-plane tilt search : %s\n"%(initial_theta))
		sparx_utilities.print_msg("Delta theta for out-of-plane tilt search  : %s\n"%(delta_theta))
		sparx_utilities.print_msg("Min radius for application of helical symmetry (in pix)    : %5.4f\n"%(rmin))
		sparx_utilities.print_msg("Max radius for application of helical symmetry (in pix)    : %5.4f\n"%(rmax))
		sparx_utilities.print_msg("Fraction of volume used for application of helical symmetry: %5.4f\n"%(fract))
		sparx_utilities.print_msg("Helical symmetry - axial rise   [A]       : %5.4f\n"%(dp))
		sparx_utilities.print_msg("Helical symmetry - angle                  : %5.4f\n"%(dphi))
		sparx_utilities.print_msg("Maximum number of iterations              : %i\n"%(max_iter))
		sparx_utilities.print_msg("Number of iterations to predict/search before doing reconstruction and updating reference volume : %i\n"%(search_iter))
		sparx_utilities.print_msg("Data with CTF                             : %s\n"%(CTF))
		sparx_utilities.print_msg("Signal-to-Noise Ratio                     : %5.4f\n"%(snr))
		sparx_utilities.print_msg("npad                                      : %i\n"%(npad))
		sparx_utilities.print_msg("User function                             : %s\n"%(user_func_name))
		sparx_utilities.print_msg("Pixel size [A]                            : %f\n"%(pixel_size))
		sparx_utilities.print_msg("Point-group symmetry group                : %s\n"%(sym))
		sparx_utilities.print_msg("Segment height seg_ny                     : %s\n\n"%(seg_ny))

	if maskfile:
		if type(maskfile) is bytes: mask3D = sparx_utilities.get_image(maskfile)
		else:                                  mask3D = maskfile
	else: mask3D = None
	#else: mask3D = model_circle(last_ring, nx, nx, nx)

	if CTF:
		pass#IMPORTIMPORTIMPORT from reconstruction import recons3d_4nn_ctf_MPI
		pass#IMPORTIMPORTIMPORT from filter         import filt_ctf
	pass#IMPORTIMPORTIMPORT else:	 from reconstruction import recons3d_4nn_MPI

	if( myid == 0):
		infils = EMAN2_cppwrap.EMUtil.get_all_attributes(stack, "filament")
		ptlcoords = EMAN2_cppwrap.EMUtil.get_all_attributes(stack, 'ptcl_source_coord')
		filaments = sparx_pixel_error.ordersegments(infils, ptlcoords)
		total_nfils = len(filaments)
		inidl = [0]*total_nfils
		for i in range(total_nfils):  inidl[i] = len(filaments[i])
		linidl = sum(inidl)
		tfilaments = []
		for i in range(total_nfils):  tfilaments += filaments[i]
		del filaments
	else:
		total_nfils = 0
		linidl = 0
	total_nfils = sparx_utilities.bcast_number_to_all(total_nfils, source_node = main_node)
	if myid != main_node:
		inidl = [-1]*total_nfils
	inidl = sparx_utilities.bcast_list_to_all(inidl, myid, source_node = main_node)
	linidl = sparx_utilities.bcast_number_to_all(linidl, source_node = main_node)
	if myid != main_node:
		tfilaments = [-1]*linidl
	tfilaments = sparx_utilities.bcast_list_to_all(tfilaments, myid, source_node = main_node)
	filaments = []
	iendi = 0
	for i in range(total_nfils):
		isti = iendi
		iendi = isti+inidl[i]
		filaments.append(tfilaments[isti:iendi])
	del tfilaments,inidl

	if myid == main_node:
		sparx_utilities.print_msg("total number of filaments in the data:  %i\n"%(total_nfils))
	if total_nfils< number_of_proc:
		sparx_global_def.ERROR('number of CPUs (%i) is larger than the number of filaments (%i), please reduce the number of CPUs used'%(number_of_proc, total_nfils), "localhelicon_MPI", 1,myid)

	#  balanced load
	temp = sparx_utilities.chunks_distribution([[len(filaments[i]), i] for i in range(len(filaments))], number_of_proc)[myid:myid+1][0]
	filaments = [filaments[temp[i][1]] for i in range(len(temp))]
	nfils     = len(filaments)
	list_of_particles = []
	indcs = []
	k = 0
	for i in range(nfils):
		list_of_particles += filaments[i]
		k1 = k+len(filaments[i])
		indcs.append([k,k1])
		k = k1
	if slowIO:
		for iproc in range(number_of_proc):
			if myid ==iproc:
				data = EMAN2_cppwrap.EMData.read_images(stack, list_of_particles)
				print("Read %6d images on process  : %4d"%(len(list_of_particles),myid))
			mpi.mpi_barrier(mpi.MPI_COMM_WORLD)
	else: data = EMAN2_cppwrap.EMData.read_images(stack, list_of_particles)	
	nima = len(data)
	data_nx = data[0].get_xsize()
	data_ny = data[0].get_ysize()
	if ((nx < data_nx) or (data_nx != data_ny)):
		sparx_global_def.ERROR('Images should be square with nx and ny equal to nz of reference volume', "localhelicon_MPI", 1, myid)
	data_nn = max(data_nx, data_ny)

	segmask = sparx_utilities.pad(sparx_utilities.model_blank(2*rmax+1, seg_ny, 1, 1.0), data_nx, data_ny, 1, 0.0)
	
	if last_ring < 0:
		last_ring = (max(seg_ny, 2*int(rmax)))//2 - 2

	#if fourvar:  original_data = []
	for im in range(nima):
		data[im].set_attr('ID', list_of_particles[im])
		sttt = EMAN2_cppwrap.Util.infomask(data[im], segmask, False)
		data[im] -= sttt[0]
		#if fourvar: original_data.append(data[im].copy())
		if CTF:
			st = data[im].get_attr_default("ctf_applied", 0)
			if(st == 0):
				ctf_params = data[im].get_attr("ctf")
				data[im] = sparx_filter.filt_ctf(data[im], ctf_params)
				data[im].set_attr('ctf_applied', 1)


	M = data_nn
	alpha = 1.75
	K = 6
	N = M*2  # npad*image size
	r = M/2
	v = K/2.0/N
	params = {"filter_type" : EMAN2_cppwrap.Processor.fourier_filter_types.KAISER_SINH_INVERSE,
	          "alpha" : alpha, "K":K,"r":r,"v":v,"N":N}
	kb = EMAN2_cppwrap.Util.KaiserBessel(alpha, K, r, v, N)
	dataft = [None]*nima
	for im in range(nima):
		dataft[im] = data[im].FourInterpol(N, N, 1,0)
		dataft[im] = EMAN2_cppwrap.Processor.EMFourierFilter(dataft[im] ,params)

	if debug:
		finfo.write( '%d loaded  \n' % nima )
		finfo.flush()

	for i in range(len(xrng)): yrng[i]=max(int(dp/(2*pixel_size)+0.5),1)
	for i in range(len(xrng)): xrng[i]=max(int(xrng[i]),1)

	if myid == main_node:
		sparx_utilities.print_msg("Pixel size in Angstroms                   : %5.4f\n"%(pixel_size))
		sparx_utilities.print_msg("Y search range (pix) initialized as       : %s\n\n"%(yrng))

	#  set attribute updown for each filament, up will be 0, down will be 1
	for ivol in range(nfils):
		seg_start = indcs[ivol][0]
		seg_end   = indcs[ivol][1]
		filamentupdown(data[seg_start: seg_end], pixel_size, dp, dphi)


	if debug:
		finfo.write("seg_start, seg_end: %d %d\n" %(seg_start, seg_end))
		finfo.flush()

	pass#IMPORTIMPORTIMPORT from time import time

	total_iter = 0
	for ii in range(lstp):
		if stepx[ii] == 0.0:
			if xrng[ii] != 0.0:
				sparx_global_def.ERROR('xrange step size cannot be zero', "localhelicon_MPIming", 1,myid)
			else:
				stepx[ii] = 1.0 # this is to prevent division by zero in c++ code

	#  TURN INTO PARAMETER OR COMPUTE FROM OU
	psistep=0.5
	# do the projection matching
	ooiter = 0
	for N_step in range(lstp):
		terminate = 0
		Iter = 0
		ant = numpy.cos(numpy.radians(an[N_step]))
		while(Iter < totmax_iter and terminate == 0):
			yrng[N_step]=float(dp)/(2*pixel_size) #will change it later according to dp
			if(ynumber[N_step]==0): 
				yrng[N_step]= 0
				stepy = 1.0
			else:                   stepy = (2*yrng[N_step]/ynumber[N_step])
					
			if stepy < 0.1:
				sparx_global_def.ERROR('yrange step size cannot be lower than 0.1', "localhelicon_MPIming", 1,myid)
 		
 		
			
			

			pixer  = [0.0]*nima

			neworient = [[0.0, 0.0, 0.0, 0.0, 0.0, -2.0e23] for i in range(nima)]

			ooiter += 1
			Iter += 1
			if Iter%search_iter == 0:  total_iter += 1
			if myid == main_node:
				start_time = time.time()
				sparx_utilities.print_msg("\n (localhelicon_MPI) ITERATION #%3d,  inner iteration #%3d\nDelta = %4.1f, an = %5.4f, xrange (Pixels) = %5.4f,stepx (Pixels) = %5.4f, yrng (Pixels) = %5.4f,  stepy (Pixels) = %5.4f, y_restrict (Pixels)=%5.4f, ynumber = %3d\n"\
				%(total_iter, Iter, delta[N_step], an[N_step], xrng[N_step], stepx[N_step], yrng[N_step], stepy, y_restrict[N_step], ynumber[N_step]))
				print("ITERATION   ",total_iter)

			volft,kbv = sparx_projection.prep_vol( vol )


			"""Multiline Comment62"""
			#  WHAT DOES IT DO?
			pass#IMPORTIMPORTIMPORT from numpy import float32
			dpp = numpy.float32(float(dp)/pixel_size)
			dpp = float( dpp )
			dpp_half = dpp/2.0

			Torg = []
			for ivol in range(nfils):

				seg_start = indcs[ivol][0]
				seg_end   = indcs[ivol][1]
				for im in range( seg_start, seg_end ):
					Torg.append(data[im].get_attr('xform.projection'))

				#  Fit predicted locations as new starting points
				if (seg_end - seg_start) > 1:
					setfilori_SP(data[seg_start: seg_end], pixel_size, dp, dphi)

			#  Generate list of reference angles, all nodes have the entire list
			ref_angles = prepare_helical_refangles(delta[N_step], initial_theta =initial_theta, delta_theta = delta_theta)
			#  count how many projections did not have a peak.  If too many, something is wrong
			print("finish generating list of reference angles.")
			nopeak = 0
			#  DO ORIENTATION SEARCHES
			for ivol in range(nfils):
				seg_start = indcs[ivol][0]
				seg_end   = indcs[ivol][1]
				ctx = [None]*(seg_end-seg_start)
				txtol = [0.0]*(seg_end-seg_start)
				tytol = [0.0]*(seg_end-seg_start)	
				for im in range( seg_start, seg_end ):
					#print "for %dth segment"%im 
					#  Here I have to figure for local search whether given image has to be matched with this refproj dir
					ID = data[im].get_attr("ID")
					phi, theta, psi, tx, ty = sparx_utilities.get_params_proj(data[im])
					txtol[im-seg_start] = tx
					tytol[im-seg_start] = ty
					if finfo:
						finfo.write("Image id: %6d\n"%(ID))
						finfo.write("Old parameters: %9.4f %9.4f %9.4f %9.4f %9.4f\n"%(phi, theta, psi, tx, ty))
						finfo.flush()
					#  Determine whether segment is up and down and search for psi in one orientation only.
					
					imn1 = numpy.sin(numpy.radians(theta))*numpy.cos(numpy.radians(phi))
					imn2 = numpy.sin(numpy.radians(theta))*numpy.sin(numpy.radians(phi))
					imn3 = numpy.cos(numpy.radians(theta))	
					for refang in ref_angles:
						n1 = numpy.sin(numpy.radians(refang[1]))*numpy.cos(numpy.radians(refang[0]))
						n2 = numpy.sin(numpy.radians(refang[1]))*numpy.sin(numpy.radians(refang[0]))
						n3 = numpy.cos(numpy.radians(refang[1]))
						refrings = [None]
						if psi < 180.0 :  direction = "up"
						else:             direction = "down"
										
						if( (n1*imn1 + n2*imn2 + n3*imn3)>=ant ):
							if(refrings[0] == None):
								#print  "  reffft1  ",im,refang
								refrings = prepare_reffft1(volft, kbv, refang, segmask, psi_max, psistep)
								
				
							#  Constrained snake search methodology
							#		x - around previously found location tx +/- xrng[N_step] in stepx
							#		y - around previously found location ty +/- yrng[N_step] in stepy
							#		psi - around previously found position psi +/- psi_max in steps psistep
							#		phi and theta are restricted by parameter an above.
							#
							#
							#print  "IMAGE  ",im
							#print "AAAAAAAAAAA, refang=", refang
							tyrng = max(1,min(yrng[N_step],abs(y_restrict[N_step]-ty),abs(-y_restrict[N_step]-ty)))
							
							angb, newtx, newty, pik, ccf3dimg = sparx_alignment.directaligriddingconstrained3dccf(dataft[im], kb, refrings, \
								psi_max, psistep, xrng[N_step], tyrng, stepx[N_step], stepy, psi, tx, ty, direction)
					
							if(pik > -1.0e23):
								if(pik > neworient[im][-1]):
									neworient[im][-1] = pik
									neworient[im][:4] = [0, 0, 0, refang] #[angb, newtx, newty, refang]
									#print "im", im-seg_start
									ctx[im-seg_start]=ccf3dimg
					#print "im peak", im, pik			
				
				##3D snake search.
				#print "before refine: neworient", neworient[seg_start:seg_end]
				nc = (int(2*psi_max/psistep)+1)//2
				rnx   = int(round(xrng[N_step]/stepx[N_step]))
				rny   = int(round(yrng[N_step]/stepy))
				neworientsnake=sparx_alignment.alignment3Dsnake(1, snakeknots, seg_end-seg_start, neworient[seg_start:seg_end], ctx, psistep, stepx[N_step], stepy, txtol, tytol, nc, rnx, rny, direction)
				for im in range( seg_start, seg_end ):
					neworient[im][:3] = neworientsnake[im- seg_start]
				#print "after refine: neworient", neworient[seg_start:seg_end]	
			for im in range(nima):
				if(neworient[im][-1] > -1.0e23):
					#print " neworient  ",im,neworient[im]
					#from utilities import inverse_transform2
					#t1, t2, t3, tp = inverse_transform2(neworient[im][3][1]+neworient[im][0])
					tp = EMAN2_cppwrap.Transform({"type":"spider","phi":neworient[im][3][0],"theta":neworient[im][3][1],"psi":neworient[im][3][2]+neworient[im][0]})
					tp.set_trans( EMAN2_cppwrap.Vec2f( neworient[im][1], neworient[im][2] ) )
					data[im].set_attr("xform.projection", tp)
					pass#IMPORTIMPORTIMPORT from utilities import get_params_proj
					#print  "  PARAMS ",im,get_params_proj(data[im])
					pixer[im]  = sparx_pixel_error.max_3D_pixel_error(Torg[im], tp, last_ring)
					data[im].set_attr("pixerr", pixer[im])

				else:
					# peak not found, parameters not modified
					nopeak += 1
					pixer[im]  = 0.0
					data[im].set_attr("pixerr", pixer[im])
					#data[im].set_attr('xform.projection', Torg[im-seg_start])

			nopeak = mpi.mpi_reduce(nopeak, 1, mpi.MPI_INT, mpi.MPI_SUM, main_node, mpi.MPI_COMM_WORLD)
			if myid == main_node:
				sparx_utilities.print_msg("Time of alignment = %d\n"%(time.time()-start_time))
				sparx_utilities.print_msg("Number of segments without a peak = %d\n"%(nopeak))
				start_time = time.time()

			mpi.mpi_barrier(mpi.MPI_COMM_WORLD)

			terminate = mpi.mpi_bcast(terminate, 1, mpi.MPI_INT, 0, mpi.MPI_COMM_WORLD)
			terminate = int(terminate[0])

			mpi.mpi_barrier(mpi.MPI_COMM_WORLD)

			if (Iter-1) % search_iter == 0 :

				if CTF:  vol = sparx_reconstruction.recons3d_4nn_ctf_MPI(myid, data, symmetry=sym, snr = snr, npad = npad)
				else:    vol = sparx_reconstruction.recons3d_4nn_MPI(myid, data, symmetry=sym, snr = snr, npad = npad)

				if myid == main_node:
					sparx_utilities.print_msg("3D reconstruction time = %d\n"%(time.time()-start_time))
					start_time = time.time()

					#drop_image(vol, os.path.join(outdir, "vol%04d.hdf"%(total_iter)))

					#  symmetry is imposed
					vol = vol.helicise(pixel_size, dp, dphi, fract, rmax, rmin)
					sparx_utilities.print_msg("Imposed delta z and delta phi      : %s,    %s\n"%(dp,dphi))
					vol = sparx_utilities.sym_vol(vol, symmetry=sym)
					ref_data = [vol, mask3D]
					#if  fourvar:  ref_data.append(varf)
					vol = user_func(ref_data)
					vol = vol.helicise(pixel_size, dp, dphi, fract, rmax, rmin)
					vol = sparx_utilities.sym_vol(vol, symmetry=sym)
					sparx_utilities.drop_image(vol, os.path.join(outdir, "volf%04d.hdf"%(total_iter)))
					sparx_utilities.print_msg("Symmetry enforcement and user function time = %d\n"%(time.time()-start_time))
					start_time = time.time()

				# using current volume
				sparx_utilities.bcast_EMData_to_all(vol, myid, main_node)

			mpi.mpi_barrier(mpi.MPI_COMM_WORLD)

			# write out headers, under MPI writing has to be done sequentially
			pass#IMPORTIMPORTIMPORT from mpi import mpi_recv, mpi_send, MPI_COMM_WORLD, MPI_FLOAT
			par_str = ['xform.projection', 'ID','pixerr']
			if myid == main_node:
				if(sparx_utilities.file_type(stack) == "bdb"):
					pass#IMPORTIMPORTIMPORT from utilities import recv_attr_dict_bdb
					sparx_utilities.recv_attr_dict_bdb(main_node, stack, data, par_str, 0, nima, number_of_proc)
				else:
					pass#IMPORTIMPORTIMPORT from utilities import recv_attr_dict
					sparx_utilities.recv_attr_dict(main_node, stack, data, par_str, 0, nima, number_of_proc)
			else:
				sparx_utilities.send_attr_dict(main_node, data, par_str, 0, nima)

			if myid == main_node:
				# write params to text file
				header(stack, params='xform.projection', fexport=os.path.join(outdir, "parameters%04d.txt"%(ooiter)))
				header(stack, params='pixerr', fexport=os.path.join(outdir, "pixelerror%04d.txt"%(ooiter)))



def localhelicon_MPInew_fullrefproj(stack, ref_vol, outdir, seg_ny, maskfile, ir, ou, rs, xr, ynumber,\
						txs, delta, initial_theta, delta_theta, an, maxit, CTF, snr, dp, dphi, psi_max,\
						rmin, rmax, fract,  npad, sym, user_func_name, \
						pixel_size, debug, y_restrict, search_iter, slowIO):

	pass#IMPORTIMPORTIMPORT from alignment      import proj_ali_helicon_local, proj_ali_helicon_90_local_direct
	pass#IMPORTIMPORTIMPORT from utilities      import model_circle, get_image, drop_image, get_input_from_string, pad, model_blank
	pass#IMPORTIMPORTIMPORT from utilities      import bcast_list_to_all, bcast_number_to_all, reduce_EMData_to_root, bcast_EMData_to_all
	pass#IMPORTIMPORTIMPORT from utilities      import send_attr_dict, read_text_row, sym_vol
	pass#IMPORTIMPORTIMPORT from utilities      import get_params_proj, set_params_proj, file_type, chunks_distribution
	pass#IMPORTIMPORTIMPORT from fundamentals   import rot_avg_image
	pass#IMPORTIMPORTIMPORT from applications 	import setfilori_SP, filamentupdown, prepare_refffts
	pass#IMPORTIMPORTIMPORT from pixel_error    import max_3D_pixel_error, ordersegments
	pass#IMPORTIMPORTIMPORT from utilities      import print_begin_msg, print_end_msg, print_msg
	pass#IMPORTIMPORTIMPORT from mpi            import mpi_bcast, mpi_comm_size, mpi_comm_rank, MPI_FLOAT, MPI_COMM_WORLD, mpi_barrier
	pass#IMPORTIMPORTIMPORT from mpi            import mpi_recv,  mpi_send
	pass#IMPORTIMPORTIMPORT from mpi            import mpi_reduce, MPI_INT, MPI_SUM
	pass#IMPORTIMPORTIMPORT from filter         import filt_ctf
	pass#IMPORTIMPORTIMPORT from projection     import prep_vol, prgs
	pass#IMPORTIMPORTIMPORT from statistics     import hist_list, varf3d_MPI
	pass#IMPORTIMPORTIMPORT from applications   import MPI_start_end, header
	pass#IMPORTIMPORTIMPORT from EMAN2          import Vec2f
	pass#IMPORTIMPORTIMPORT from string         import lower, split
	pass#IMPORTIMPORTIMPORT from copy           import copy
	pass#IMPORTIMPORTIMPORT import os
	pass#IMPORTIMPORTIMPORT import types

	number_of_proc = mpi.mpi_comm_size(mpi.MPI_COMM_WORLD)
	myid           = mpi.mpi_comm_rank(mpi.MPI_COMM_WORLD)
	main_node = 0

	if myid == 0:
		if os.path.exists(outdir):  nx = 1
		else:  nx = 0
	else:  nx = 0
	ny = sparx_utilities.bcast_number_to_all(nx, source_node = main_node)

	if ny == 1:  sparx_global_def.ERROR('Output directory exists, please change the name and restart the program', "localhelicon_MPI", 1,myid)
	mpi.mpi_barrier(mpi.MPI_COMM_WORLD)

	if myid == main_node:
		os.mkdir(outdir)
		pass#IMPORTIMPORTIMPORT import global_def
		sparx_global_def.LOGFILE =  os.path.join(outdir, sparx_global_def.LOGFILE)
		sparx_utilities.print_begin_msg("localhelicon_MPI NEW")
	mpi.mpi_barrier(mpi.MPI_COMM_WORLD)

	if debug:
		pass#IMPORTIMPORTIMPORT from time import sleep
		while not os.path.exists(outdir):
			print("Node ",myid,"  waiting...")
			time.sleep(5)

		info_file = os.path.join(outdir, "progress%04d"%myid)
		finfo = open(info_file, 'w')
	else:
		finfo = None

	sym    = sym.lower()
	symref = "s"+sym

	ref_a           = "P"
	symmetry_string = string.split(sym)[0]

	xrng        = sparx_utilities.get_input_from_string(xr)
	y_restrict  = sparx_utilities.get_input_from_string(y_restrict)
	ynumber	    = sparx_utilities.get_input_from_string(ynumber)
	for i in range(len(ynumber)):
		if ynumber[i] > 0:
			if(ynumber[i]%2==1): ynumber[i]=ynumber[i]+1
	yrng = []

	for i in range(len(xrng)): yrng.append(dp/2)

	stepx       = sparx_utilities.get_input_from_string(txs)
	delta       = sparx_utilities.get_input_from_string(delta)
	lstp = min(len(xrng), len(yrng), len(stepx), len(delta))
	an = sparx_utilities.get_input_from_string(an)

	if len(an) == 1:
		aan = an[0]
		an = [aan for ii in range(lstp)]
	y_restrict = y_restrict[0:lstp]
	for i in range(lstp):
		if an[i] < 0 and y_restrict[i] < 0: 
			sparx_global_def.ERROR('This is a local search, an and y_restrict should not both be -1', "localhelicon_MPI", 1,myid)
		if y_restrict[i] < 0:   y_restrict[i] = (an[i]/dphi)*(dp/pixel_size)/2.0
		if an[i] < 0:           an[i] = ((2.0*y_restrict[i])/(dp/pixel_size)) * dphi

	first_ring  = int(ir)
	rstep       = int(rs)
	last_ring   = int(ou)
	max_iter    = int(maxit)
	search_iter = int(search_iter)
	totmax_iter = max_iter * search_iter

	vol     = EMAN2_cppwrap.EMData()
	vol.read_image(ref_vol)
	nx      = vol.get_xsize()
	ny      = vol.get_ysize()
	nz      = vol.get_zsize()

	if nz < nx:
		sparx_global_def.ERROR('Do not handle squat volumes .... nz cannot be less than nx', "localhelicon_MPI", 1, myid)

	# Pad to square
	if nz > nx:
		nx = nz
		ny = nz	
		vol = sparx_utilities.pad(vol, nx, ny,nz,background=0.0)	
	nmax = max(nx, ny, nz)

	if myid == main_node:
		pass#IMPORTIMPORTIMPORT import user_functions
		user_func = sparx_user_functions.factory[user_func_name]

		sparx_utilities.print_msg("Input stack                               : %s\n"%(stack))
		sparx_utilities.print_msg("Reference volume                          : %s\n"%(ref_vol))	
		sparx_utilities.print_msg("Output directory                          : %s\n"%(outdir))
		sparx_utilities.print_msg("Maskfile                                  : %s\n"%(maskfile))
		sparx_utilities.print_msg("Inner radius for psi angle search         : %i\n"%(first_ring))
		sparx_utilities.print_msg("Outer radius for psi angle search         : %i\n"%(last_ring))
		sparx_utilities.print_msg("Ring step                                 : %i\n"%(rstep))
		sparx_utilities.print_msg("X search range                            : %s\n"%(xrng))
		sparx_utilities.print_msg("Y search range                            : %s\n"%(y_restrict))
		sparx_utilities.print_msg("Y number                                  : %s\n"%(ynumber))
		sparx_utilities.print_msg("Translational stepx                       : %s\n"%(stepx))
		sparx_utilities.print_msg("Angular step                              : %s\n"%(delta))
		sparx_utilities.print_msg("Angular search range                      : %s\n"%(an))
		sparx_utilities.print_msg("Intial theta for out-of-plane tilt search : %s\n"%(initial_theta))
		sparx_utilities.print_msg("Delta theta for out-of-plane tilt search  : %s\n"%(delta_theta))
		sparx_utilities.print_msg("Min radius for application of helical symmetry (in pix)    : %5.4f\n"%(rmin))
		sparx_utilities.print_msg("Max radius for application of helical symmetry (in pix)    : %5.4f\n"%(rmax))
		sparx_utilities.print_msg("Fraction of volume used for application of helical symmetry: %5.4f\n"%(fract))
		sparx_utilities.print_msg("Helical symmetry - axial rise   [A]       : %5.4f\n"%(dp))
		sparx_utilities.print_msg("Helical symmetry - angle                  : %5.4f\n"%(dphi))
		sparx_utilities.print_msg("Maximum number of iterations              : %i\n"%(max_iter))
		sparx_utilities.print_msg("Number of iterations to predict/search before doing reconstruction and updating reference volume : %i\n"%(search_iter))
		sparx_utilities.print_msg("Data with CTF                             : %s\n"%(CTF))
		sparx_utilities.print_msg("Signal-to-Noise Ratio                     : %5.4f\n"%(snr))
		sparx_utilities.print_msg("npad                                      : %i\n"%(npad))
		sparx_utilities.print_msg("User function                             : %s\n"%(user_func_name))
		sparx_utilities.print_msg("Pixel size [A]                            : %f\n"%(pixel_size))
		sparx_utilities.print_msg("Point-group symmetry group                : %s\n"%(sym))
		sparx_utilities.print_msg("Segment height seg_ny                     : %s\n\n"%(seg_ny))

	if maskfile:
		if type(maskfile) is bytes: mask3D = sparx_utilities.get_image(maskfile)
		else:                                  mask3D = maskfile
	else: mask3D = None
	#else: mask3D = model_circle(last_ring, nx, nx, nx)

	if CTF:
		pass#IMPORTIMPORTIMPORT from reconstruction import recons3d_4nn_ctf_MPI
		pass#IMPORTIMPORTIMPORT from filter         import filt_ctf
	pass#IMPORTIMPORTIMPORT else:	 from reconstruction import recons3d_4nn_MPI

	if( myid == 0):
		infils = EMAN2_cppwrap.EMUtil.get_all_attributes(stack, "filament")
		ptlcoords = EMAN2_cppwrap.EMUtil.get_all_attributes(stack, 'ptcl_source_coord')
		filaments = sparx_pixel_error.ordersegments(infils, ptlcoords)
		total_nfils = len(filaments)
		inidl = [0]*total_nfils
		for i in range(total_nfils):  inidl[i] = len(filaments[i])
		linidl = sum(inidl)
		tfilaments = []
		for i in range(total_nfils):  tfilaments += filaments[i]
		del filaments
	else:
		total_nfils = 0
		linidl = 0
	total_nfils = sparx_utilities.bcast_number_to_all(total_nfils, source_node = main_node)
	if myid != main_node:
		inidl = [-1]*total_nfils
	inidl = sparx_utilities.bcast_list_to_all(inidl, myid, source_node = main_node)
	linidl = sparx_utilities.bcast_number_to_all(linidl, source_node = main_node)
	if myid != main_node:
		tfilaments = [-1]*linidl
	tfilaments = sparx_utilities.bcast_list_to_all(tfilaments, myid, source_node = main_node)
	filaments = []
	iendi = 0
	for i in range(total_nfils):
		isti = iendi
		iendi = isti+inidl[i]
		filaments.append(tfilaments[isti:iendi])
	del tfilaments,inidl

	if myid == main_node:
		sparx_utilities.print_msg("total number of filaments in the data:  %i\n"%(total_nfils))
	if total_nfils< number_of_proc:
		sparx_global_def.ERROR('number of CPUs (%i) is larger than the number of filaments (%i), please reduce the number of CPUs used'%(number_of_proc, total_nfils), "localhelicon_MPI", 1,myid)

	#  balanced load
	temp = sparx_utilities.chunks_distribution([[len(filaments[i]), i] for i in range(len(filaments))], number_of_proc)[myid:myid+1][0]
	filaments = [filaments[temp[i][1]] for i in range(len(temp))]
	nfils     = len(filaments)
	list_of_particles = []
	indcs = []
	k = 0
	for i in range(nfils):
		list_of_particles += filaments[i]
		k1 = k+len(filaments[i])
		indcs.append([k,k1])
		k = k1

	if slowIO:
		for iproc in range(number_of_proc):
			if myid ==iproc:
				data = EMAN2_cppwrap.EMData.read_images(stack, list_of_particles)
				print("Read %6d images on process  : %4d"%(len(list_of_particles),myid))
			mpi.mpi_barrier(mpi.MPI_COMM_WORLD)
	else: data = EMAN2_cppwrap.EMData.read_images(stack, list_of_particles)
	nima = len(data)
	data_nx = data[0].get_xsize()
	data_ny = data[0].get_ysize()
	if ((nx < data_nx) or (data_nx != data_ny)):
		sparx_global_def.ERROR('Images should be square with nx and ny equal to nz of reference volume', "localhelicon_MPI", 1, myid)
	data_nn = max(data_nx, data_ny)

	segmask = sparx_utilities.pad(sparx_utilities.model_blank(2*rmax+1, seg_ny, 1, 1.0), data_nx, data_ny, 1, 0.0)
	
	if last_ring < 0:
		last_ring = (max(seg_ny, 2*int(rmax)))//2 - 2

	#if fourvar:  original_data = []
	for im in range(nima):
		data[im].set_attr('ID', list_of_particles[im])
		sttt = EMAN2_cppwrap.Util.infomask(data[im], segmask, False)
		data[im] -= sttt[0]
		#if fourvar: original_data.append(data[im].copy())
		if CTF:
			st = data[im].get_attr_default("ctf_applied", 0)
			if(st == 0):
				ctf_params = data[im].get_attr("ctf")
				data[im] = sparx_filter.filt_ctf(data[im], ctf_params)
				data[im].set_attr('ctf_applied', 1)

	if debug:
		finfo.write( '%d loaded  \n' % nima )
		finfo.flush()

	for i in range(len(xrng)): yrng[i]=max(int(dp/(2*pixel_size)+0.5),1)
	for i in range(len(xrng)): xrng[i]=max(int(xrng[i]),1)

	if myid == main_node:
		sparx_utilities.print_msg("Pixel size in Angstroms                   : %5.4f\n"%(pixel_size))
		sparx_utilities.print_msg("Y search range (pix) initialized as       : %s\n\n"%(yrng))

	#  set attribute updown for each filament, up will be 0, down will be 1
	for ivol in range(nfils):
		seg_start = indcs[ivol][0]
		seg_end   = indcs[ivol][1]
		filamentupdown(data[seg_start: seg_end], pixel_size, dp, dphi)


	if debug:
		finfo.write("seg_start, seg_end: %d %d\n" %(seg_start, seg_end))
		finfo.flush()

	pass#IMPORTIMPORTIMPORT from time import time

	total_iter = 0
	for ii in range(lstp):
		if stepx[ii] == 0.0:
			if xrng[ii] != 0.0:
				sparx_global_def.ERROR('xrange step size cannot be zero', "localhelicon_MPI", 1,myid)
			else:
				stepx[ii] = 1.0 # this is to prevent division by zero in c++ code

	#  TURN INTO PARAMETER OR COMPUTE FROM OU
	psistep=0.5
	# do the projection matching
	ooiter = 0
	for N_step in range(lstp):
		terminate = 0
		Iter = 0
		while(Iter < totmax_iter and terminate == 0):
			yrng[N_step]=float(dp)/(2*pixel_size) #will change it later according to dp
			#yrng[N_step]=max(int(yrng[N_step]+0.5),1)
			if(ynumber[N_step]==0): stepy = 0.0
			else:                   stepy = (2*yrng[N_step]/ynumber[N_step])
			stepx = stepy

			pixer  = [0.0]*nima
			modphi = [0.0]*nima
			ooiter += 1
			Iter += 1
			if Iter%search_iter == 0:  total_iter += 1

			# If the previous iteration did a reconstruction, then generate new refrings
			if ( (Iter - 1) % search_iter == 0):

				if myid == main_node:
					start_time = time.time()
					sparx_utilities.print_msg("\n (localhelicon_MPI) ITERATION #%3d,  inner iteration #%3d\nDelta = %4.1f, an = %5.4f, xrange (Pixels) = %5.4f,stepx (Pixels) = %5.4f, yrng (Pixels) = %5.4f,  stepy (Pixels) = %5.4f, y_restrict (Pixels)=%5.4f, ynumber = %3d\n"\
					%(total_iter, Iter, delta[N_step], an[N_step], xrng[N_step], stepx, yrng[N_step], stepy, y_restrict[N_step], ynumber[N_step]))

				volft,kb = sparx_projection.prep_vol( vol )
				#  What about cushion for a neighborhood?  PAP 06/04/2014
				refrings = prepare_refffts( volft, kb, data_nn,data_nn,nz, segmask, delta[N_step], \
					MPI=True, psimax=psi_max, psistep=psistep, initial_theta =initial_theta, delta_theta = delta_theta)
				#refrings = prepare_refrings2(  volft, kb, nmax, segmask, delta[N_step], ref_a, symref, numr, MPI = True, phiEqpsi = "Zero", initial_theta =initial_theta, delta_theta = delta_theta)
				del volft,kb

				if myid== main_node:
					sparx_utilities.print_msg( "Time to prepare rings: %d\n" % (time.time()-start_time) )
					start_time = time.time()

			pass#IMPORTIMPORTIMPORT from numpy import float32
			dpp = numpy.float32(float(dp)/pixel_size)
			dpp = float( dpp )
			dpp_half = dpp/2.0

			for ivol in range(nfils):

				seg_start = indcs[ivol][0]
				seg_end   = indcs[ivol][1]
				Torg = []
				for im in range( seg_start, seg_end ):
					Torg.append(data[im].get_attr('xform.projection'))

				#  Fit predicted locations as new starting points
				#  LOCKED FOR TESTING  PAP 12/30/2014
				#if (seg_end - seg_start) > 1:
				#	setfilori_SP(data[seg_start: seg_end], pixel_size, dp, dphi)

				for im in range( seg_start, seg_end ):

					peak, phihi, theta, psi, sxi, syi = \
						sparx_alignment.proj_ali_helicon_90_local_direct(data[im], refrings, xrng[N_step], yrng[N_step], \
						an[N_step], psi_max, psistep, stepx, stepy, finfo, yrnglocal=y_restrict[N_step])

					if(peak > -1.0e23):

						tp = EMAN2_cppwrap.Transform({"type":"spider","phi":phihi,"theta":theta,"psi":psi})
						tp.set_trans( EMAN2_cppwrap.Vec2f( -sxi, -syi ) )
						data[im].set_attr("xform.projection", tp)
						pixer[im]  = sparx_pixel_error.max_3D_pixel_error(Torg[im-seg_start], tp, last_ring)
						data[im].set_attr("pixerr", pixer[im])

					else:
						# peak not found, parameters not modified
						print(" peak not found, something is wrong!")
						pixer[im]  = 0.0
						data[im].set_attr("pixerr", pixer[im])
						data[im].set_attr('xform.projection', Torg[im-seg_start])

			if myid == main_node:
				sparx_utilities.print_msg("Time of alignment = %d\n"%(time.time()-start_time))
				start_time = time.time()

			mpi.mpi_barrier(mpi.MPI_COMM_WORLD)

			terminate = mpi.mpi_bcast(terminate, 1, mpi.MPI_INT, 0, mpi.MPI_COMM_WORLD)
			terminate = int(terminate[0])

			mpi.mpi_barrier(mpi.MPI_COMM_WORLD)

			if (Iter-1) % search_iter == 0:

				if CTF:  vol = sparx_reconstruction.recons3d_4nn_ctf_MPI(myid, data, symmetry=sym, snr = snr, npad = npad)
				else:    vol = sparx_reconstruction.recons3d_4nn_MPI(myid, data, symmetry=sym, snr = snr, npad = npad)

				if myid == main_node:
					sparx_utilities.print_msg("3D reconstruction time = %d\n"%(time.time()-start_time))
					start_time = time.time()

					#drop_image(vol, os.path.join(outdir, "vol%04d.hdf"%(total_iter)))

					#  symmetry is imposed
					vol = vol.helicise(pixel_size, dp, dphi, fract, rmax, rmin)
					sparx_utilities.print_msg("Imposed delta z and delta phi      : %s,    %s\n"%(dp,dphi))
					vol = sparx_utilities.sym_vol(vol, symmetry=sym)
					ref_data = [vol, mask3D]
					#if  fourvar:  ref_data.append(varf)
					vol = user_func(ref_data)
					vol = vol.helicise(pixel_size, dp, dphi, fract, rmax, rmin)
					vol = sparx_utilities.sym_vol(vol, symmetry=sym)
					sparx_utilities.drop_image(vol, os.path.join(outdir, "volf%04d.hdf"%(total_iter)))
					sparx_utilities.print_msg("Symmetry enforcement and user function time = %d\n"%(time.time()-start_time))
					start_time = time.time()

				# using current volume
				sparx_utilities.bcast_EMData_to_all(vol, myid, main_node)

			mpi.mpi_barrier(mpi.MPI_COMM_WORLD)

			# write out headers, under MPI writing has to be done sequentially
			pass#IMPORTIMPORTIMPORT from mpi import mpi_recv, mpi_send, MPI_COMM_WORLD, MPI_FLOAT
			par_str = ['xform.projection', 'ID','pixerr']
			if myid == main_node:
				if(sparx_utilities.file_type(stack) == "bdb"):
					pass#IMPORTIMPORTIMPORT from utilities import recv_attr_dict_bdb
					sparx_utilities.recv_attr_dict_bdb(main_node, stack, data, par_str, 0, nima, number_of_proc)
				else:
					pass#IMPORTIMPORTIMPORT from utilities import recv_attr_dict
					sparx_utilities.recv_attr_dict(main_node, stack, data, par_str, 0, nima, number_of_proc)
			else:
				sparx_utilities.send_attr_dict(main_node, data, par_str, 0, nima)

			if myid == main_node:
				# write params to text file
				header(stack, params='xform.projection', fexport=os.path.join(outdir, "parameters%04d.txt"%(ooiter)))
				header(stack, params='pixerr', fexport=os.path.join(outdir, "pixelerror%04d.txt"%(ooiter)))


def localhelicon_MPI(stack, ref_vol, outdir, seg_ny, maskfile, ir, ou, rs, xr, ynumber,\
						txs, delta, initial_theta, delta_theta, an, maxit, CTF, snr, dp, dphi, psi_max,\
						rmin, rmax, fract,  npad, sym, user_func_name, \
						pixel_size, debug, y_restrict, search_iter, slowIO):

	pass#IMPORTIMPORTIMPORT from alignment      import Numrinit, prepare_refrings2, prepare_refrings
	pass#IMPORTIMPORTIMPORT from alignment      import proj_ali_helicon_local, proj_ali_helicon_90_local
	pass#IMPORTIMPORTIMPORT from utilities      import model_circle, get_image, drop_image, get_input_from_string, pad, model_blank
	pass#IMPORTIMPORTIMPORT from utilities      import bcast_list_to_all, bcast_number_to_all, reduce_EMData_to_root, bcast_EMData_to_all
	pass#IMPORTIMPORTIMPORT from utilities      import send_attr_dict, read_text_row, sym_vol
	pass#IMPORTIMPORTIMPORT from utilities      import get_params_proj, set_params_proj, file_type, chunks_distribution
	pass#IMPORTIMPORTIMPORT from fundamentals   import rot_avg_image
	pass#IMPORTIMPORTIMPORT from applications 	import setfilori_SP, filamentupdown
	pass#IMPORTIMPORTIMPORT from pixel_error    import max_3D_pixel_error, ordersegments
	pass#IMPORTIMPORTIMPORT from utilities      import print_begin_msg, print_end_msg, print_msg
	pass#IMPORTIMPORTIMPORT from mpi            import mpi_bcast, mpi_comm_size, mpi_comm_rank, MPI_FLOAT, MPI_COMM_WORLD, mpi_barrier
	pass#IMPORTIMPORTIMPORT from mpi            import mpi_recv,  mpi_send
	pass#IMPORTIMPORTIMPORT from mpi            import mpi_reduce, MPI_INT, MPI_SUM
	pass#IMPORTIMPORTIMPORT from filter         import filt_ctf
	pass#IMPORTIMPORTIMPORT from projection     import prep_vol, prgs
	pass#IMPORTIMPORTIMPORT from statistics     import hist_list, varf3d_MPI
	pass#IMPORTIMPORTIMPORT from applications   import MPI_start_end, header
	pass#IMPORTIMPORTIMPORT from EMAN2          import Vec2f
	pass#IMPORTIMPORTIMPORT from string         import lower,split
	pass#IMPORTIMPORTIMPORT from copy           import copy
	pass#IMPORTIMPORTIMPORT import os
	pass#IMPORTIMPORTIMPORT import types

	number_of_proc = mpi.mpi_comm_size(mpi.MPI_COMM_WORLD)
	myid           = mpi.mpi_comm_rank(mpi.MPI_COMM_WORLD)
	main_node = 0

	if myid == 0:
		if os.path.exists(outdir):  nx = 1
		else:  nx = 0
	else:  nx = 0
	ny = sparx_utilities.bcast_number_to_all(nx, source_node = main_node)

	if ny == 1:  sparx_global_def.ERROR('Output directory exists, please change the name and restart the program', "localhelicon_MPI", 1,myid)
	mpi.mpi_barrier(mpi.MPI_COMM_WORLD)

	if myid == main_node:
		os.mkdir(outdir)
		pass#IMPORTIMPORTIMPORT import global_def
		sparx_global_def.LOGFILE =  os.path.join(outdir, sparx_global_def.LOGFILE)
		sparx_utilities.print_begin_msg("localhelicon_MPI")
	mpi.mpi_barrier(mpi.MPI_COMM_WORLD)

	if debug:
		pass#IMPORTIMPORTIMPORT from time import sleep
		while not os.path.exists(outdir):
			print("Node ",myid,"  waiting...")
			time.sleep(5)

		info_file = os.path.join(outdir, "progress%04d"%myid)
		finfo = open(info_file, 'w')
	else:
		finfo = None

	sym = sym.lower()
	symref = "s"+sym

	ref_a= "P"
	symmetry_string = string.split(sym)[0]

	xrng        = sparx_utilities.get_input_from_string(xr)
	y_restrict  = sparx_utilities.get_input_from_string(y_restrict)
	ynumber	    = sparx_utilities.get_input_from_string(ynumber)
	for i in range(len(ynumber)):
		if ynumber[i] > 0:
			if(ynumber[i]%2==1): ynumber[i]=ynumber[i]+1
	yrng =[]

	for i in range(len(xrng)): yrng.append(dp/2)

	stepx        = sparx_utilities.get_input_from_string(txs)
	delta       = sparx_utilities.get_input_from_string(delta)
	lstp = min(len(xrng), len(yrng), len(stepx), len(delta))
	an = sparx_utilities.get_input_from_string(an)
	
	if len(an) == 1:
		aan = an[0]
		an = [aan for ii in range(lstp)]
	y_restrict = y_restrict[0:lstp]
	for i in range(lstp):
		if an[i] < 0 and y_restrict[i] < 0: 
			sparx_global_def.ERROR('This is a local search, an and y_restrict should not both be -1', "localhelicon_MPI", 1,myid)
		if y_restrict[i] < 0:  y_restrict[i] = (an[i]/dphi)*(dp/pixel_size)/2.0
		if an[i] < 0:           an[i] = ((2.0*y_restrict[i])/(dp/pixel_size)) * dphi

	first_ring  = int(ir)
	rstep       = int(rs)
	last_ring   = int(ou)
	max_iter    = int(maxit)
	search_iter = int(search_iter)
	totmax_iter = max_iter * search_iter

	vol     = EMAN2_cppwrap.EMData()
	vol.read_image(ref_vol)
	nx      = vol.get_xsize()
	ny      = vol.get_ysize()
	nz      = vol.get_zsize()

	if nz < nx:
		sparx_global_def.ERROR('Do not handle squat volumes .... nz cannot be less than nx', "localhelicon_MPI", 1, myid)

	# Pad to square
	if nz > nx:
		nx = nz
		ny = nz	
		vol = sparx_utilities.pad(vol, nx, ny,nz,background=0.0)	
	nmax = max(nx, ny, nz)

	if myid == main_node:
		pass#IMPORTIMPORTIMPORT import user_functions
		user_func = sparx_user_functions.factory[user_func_name]

		sparx_utilities.print_msg("Input stack                               : %s\n"%(stack))
		sparx_utilities.print_msg("Reference volume                          : %s\n"%(ref_vol))	
		sparx_utilities.print_msg("Output directory                          : %s\n"%(outdir))
		sparx_utilities.print_msg("Maskfile                                  : %s\n"%(maskfile))
		sparx_utilities.print_msg("Inner radius for psi angle search         : %i\n"%(first_ring))
		sparx_utilities.print_msg("Outer radius for psi angle search         : %i\n"%(last_ring))
		sparx_utilities.print_msg("Ring step                                 : %i\n"%(rstep))
		sparx_utilities.print_msg("X search range                            : %s\n"%(xrng))
		sparx_utilities.print_msg("Y search range                            : %s\n"%(y_restrict))
		sparx_utilities.print_msg("Y number                                  : %s\n"%(ynumber))
		sparx_utilities.print_msg("Translational stepx                       : %s\n"%(stepx))
		sparx_utilities.print_msg("Angular step                              : %s\n"%(delta))
		sparx_utilities.print_msg("Angular search range                      : %s\n"%(an))
		sparx_utilities.print_msg("Intial theta for out-of-plane tilt search : %s\n"%(initial_theta))
		sparx_utilities.print_msg("Delta theta for out-of-plane tilt search  : %s\n"%(delta_theta))
		sparx_utilities.print_msg("Min radius for application of helical symmetry (in pix)    : %5.4f\n"%(rmin))
		sparx_utilities.print_msg("Max radius for application of helical symmetry (in pix)    : %5.4f\n"%(rmax))
		sparx_utilities.print_msg("Fraction of volume used for application of helical symmetry: %5.4f\n"%(fract))
		sparx_utilities.print_msg("Helical symmetry - axial rise   [A]       : %5.4f\n"%(dp))
		sparx_utilities.print_msg("Helical symmetry - angle                  : %5.4f\n"%(dphi))
		sparx_utilities.print_msg("Maximum number of iterations              : %i\n"%(max_iter))
		sparx_utilities.print_msg("Number of iterations to predict/search before doing reconstruction and updating reference volume : %i\n"%(search_iter))
		sparx_utilities.print_msg("Data with CTF                             : %s\n"%(CTF))
		sparx_utilities.print_msg("Signal-to-Noise Ratio                     : %5.4f\n"%(snr))
		sparx_utilities.print_msg("npad                                      : %i\n"%(npad))
		sparx_utilities.print_msg("User function                             : %s\n"%(user_func_name))
		sparx_utilities.print_msg("Pixel size [A]                            : %f\n"%(pixel_size))
		sparx_utilities.print_msg("Point-group symmetry group                : %s\n"%(sym))
		sparx_utilities.print_msg("Segment height seg_ny                     : %s\n\n"%(seg_ny))

	if maskfile:
		if type(maskfile) is bytes: mask3D = sparx_utilities.get_image(maskfile)
		else:                                  mask3D = maskfile
	else: mask3D = None
	#else: mask3D = model_circle(last_ring, nx, nx, nx)

	if CTF:
		pass#IMPORTIMPORTIMPORT from reconstruction import recons3d_4nn_ctf_MPI
		pass#IMPORTIMPORTIMPORT from filter         import filt_ctf
	pass#IMPORTIMPORTIMPORT else:	 from reconstruction import recons3d_4nn_MPI

	if( myid == 0):
		infils = EMAN2_cppwrap.EMUtil.get_all_attributes(stack, "filament")
		ptlcoords = EMAN2_cppwrap.EMUtil.get_all_attributes(stack, 'ptcl_source_coord')
		filaments = sparx_pixel_error.ordersegments(infils, ptlcoords)
		total_nfils = len(filaments)
		inidl = [0]*total_nfils
		for i in range(total_nfils):  inidl[i] = len(filaments[i])
		linidl = sum(inidl)
		tfilaments = []
		for i in range(total_nfils):  tfilaments += filaments[i]
		del filaments
	else:
		total_nfils = 0
		linidl = 0
	total_nfils = sparx_utilities.bcast_number_to_all(total_nfils, source_node = main_node)
	if myid != main_node:
		inidl = [-1]*total_nfils
	inidl = sparx_utilities.bcast_list_to_all(inidl, myid, source_node = main_node)
	linidl = sparx_utilities.bcast_number_to_all(linidl, source_node = main_node)
	if myid != main_node:
		tfilaments = [-1]*linidl
	tfilaments = sparx_utilities.bcast_list_to_all(tfilaments, myid, source_node = main_node)
	filaments = []
	iendi = 0
	for i in range(total_nfils):
		isti = iendi
		iendi = isti+inidl[i]
		filaments.append(tfilaments[isti:iendi])
	del tfilaments,inidl

	if myid == main_node:
		sparx_utilities.print_msg("total number of filaments in the data:  %i\n"%(total_nfils))
	if total_nfils< number_of_proc:
		sparx_global_def.ERROR('number of CPUs (%i) is larger than the number of filaments (%i), please reduce the number of CPUs used'%(number_of_proc, total_nfils), "localhelicon_MPI", 1,myid)

	#  balanced load
	temp = sparx_utilities.chunks_distribution([[len(filaments[i]), i] for i in range(len(filaments))], number_of_proc)[myid:myid+1][0]
	filaments = [filaments[temp[i][1]] for i in range(len(temp))]
	nfils     = len(filaments)
	list_of_particles = []
	indcs = []
	k = 0
	for i in range(nfils):
		list_of_particles += filaments[i]
		k1 = k+len(filaments[i])
		indcs.append([k,k1])
		k = k1
	if slowIO:
		for iproc in range(number_of_proc):
			if myid ==iproc:
				data = EMAN2_cppwrap.EMData.read_images(stack, list_of_particles)
				print("Read %6d images on process  : %4d"%(len(list_of_particles),myid))
			mpi.mpi_barrier(mpi.MPI_COMM_WORLD)
	else: data = EMAN2_cppwrap.EMData.read_images(stack, list_of_particles)	
	nima = len(data)
	data_nx = data[0].get_xsize()
	data_ny = data[0].get_ysize()
	if ((nx < data_nx) or (data_nx != data_ny)):
		sparx_global_def.ERROR('Images should be square with nx and ny equal to nz of reference volume', "localhelicon_MPI", 1, myid)
	data_nn = max(data_nx, data_ny)

	segmask = sparx_utilities.pad(sparx_utilities.model_blank(2*rmax+1, seg_ny, 1, 1.0), data_nx, data_ny, 1, 0.0)
	
	if last_ring < 0:
		last_ring = (max(seg_ny, 2*int(rmax)))//2 - 2

	numr	= sparx_alignment.Numrinit(first_ring, last_ring, rstep, "F")

	#if fourvar:  original_data = []
	for im in range(nima):
		data[im].set_attr('ID', list_of_particles[im])
		sttt = EMAN2_cppwrap.Util.infomask(data[im], segmask, False)
		data[im] -= sttt[0]
		#if fourvar: original_data.append(data[im].copy())
		if CTF:
			st = data[im].get_attr_default("ctf_applied", 0)
			if(st == 0):
				ctf_params = data[im].get_attr("ctf")
				data[im] = sparx_filter.filt_ctf(data[im], ctf_params)
				data[im].set_attr('ctf_applied', 1)

	if debug:
		finfo.write( '%d loaded  \n' % nima )
		finfo.flush()

	for i in range(len(xrng)): yrng[i]=dp/(2*pixel_size)

	if myid == main_node:
		sparx_utilities.print_msg("Pixel size in Angstroms                   : %5.4f\n"%(pixel_size))
		sparx_utilities.print_msg("Y search range (pix) initialized as       : %s\n\n"%(yrng))

	#  set attribute updown for each filament, up will be 0, down will be 1
	for ivol in range(nfils):
		seg_start = indcs[ivol][0]
		seg_end   = indcs[ivol][1]
		filamentupdown(data[seg_start: seg_end], pixel_size, dp, dphi)


	if debug:
		finfo.write("seg_start, seg_end: %d %d\n" %(seg_start, seg_end))
		finfo.flush()

	pass#IMPORTIMPORTIMPORT from time import time

	total_iter = 0
	for ii in range(lstp):
		if stepx[ii] == 0.0:
			if xrng[ii] != 0.0:
				sparx_global_def.ERROR('xrange step size cannot be zero', "localhelicon_MPI", 1,myid)
			else:
				stepx[ii] = 1.0 # this is to prevent division by zero in c++ code
	# do the projection matching
	ooiter = 0
	for N_step in range(lstp):
		terminate = 0
		Iter = 0
		while(Iter < totmax_iter and terminate == 0):
			yrng[N_step]=float(dp)/(2*pixel_size) #will change it later according to dp
			if(ynumber[N_step]==0): stepy = 0.0
			else:                   stepy = (2*yrng[N_step]/ynumber[N_step])

			pixer  = [0.0]*nima
			modphi = [0.0]*nima
			ooiter += 1
			Iter += 1
			if Iter%search_iter == 0:  total_iter += 1

			# If the previous iteration did a reconstruction, then generate new refrings
			if ( (Iter - 1) % search_iter == 0):

				if myid == main_node:
					start_time = time.time()
					sparx_utilities.print_msg("\n (localhelicon_MPI) ITERATION #%3d,  inner iteration #%3d\nDelta = %4.1f, an = %5.4f, xrange (Pixels) = %5.4f,stepx (Pixels) = %5.4f, yrng (Pixels) = %5.4f,  stepy (Pixels) = %5.4f, y_restrict (Pixels)=%5.4f, ynumber = %3d\n"%(total_iter, Iter, delta[N_step], an[N_step], xrng[N_step],stepx[N_step],yrng[N_step],stepy,y_restrict[N_step], ynumber[N_step]))

				volft,kb = sparx_projection.prep_vol( vol )
				#  What about cushion for a neighborhood?  PAP 06/04/2014
				refrings = sparx_alignment.prepare_refrings2(  volft, kb, nmax, segmask, delta[N_step], ref_a, symref, numr, \
							MPI = True, phiEqpsi = "Zero", initial_theta =initial_theta, delta_theta = delta_theta)
				del volft,kb

				if myid== main_node:
					sparx_utilities.print_msg( "Time to prepare rings: %d\n" % (time.time()-start_time) )
					start_time = time.time()

			pass#IMPORTIMPORTIMPORT from numpy import float32
			dpp = numpy.float32(float(dp)/pixel_size)
			dpp = float( dpp )
			dpp_half = dpp/2.0

			for ivol in range(nfils):

				seg_start = indcs[ivol][0]
				seg_end   = indcs[ivol][1]
				Torg = []
				for im in range( seg_start, seg_end ):
					Torg.append(data[im].get_attr('xform.projection'))

				#  Fit predicted locations as new starting points					
				if (seg_end - seg_start) > 1:
					setfilori_SP(data[seg_start: seg_end], pixel_size, dp, dphi)

				for im in range( seg_start, seg_end ):

					peak, phihi, theta, psi, sxi, syi = \
						sparx_alignment.proj_ali_helicon_90_local(data[im], refrings, numr, xrng[N_step], yrng[N_step], stepx[N_step], ynumber[N_step], \
								an[N_step], psi_max, finfo, yrnglocal=y_restrict[N_step])

					if(peak > -1.0e23):

						tp = EMAN2_cppwrap.Transform({"type":"spider","phi":phihi,"theta":theta,"psi":psi})
						tp.set_trans( EMAN2_cppwrap.Vec2f( -sxi, -syi ) )
						data[im].set_attr("xform.projection", tp)
						pixer[im]  = sparx_pixel_error.max_3D_pixel_error(Torg[im-seg_start], tp, numr[-3])
						data[im].set_attr("pixerr", pixer[im])

					else:
						# peak not found, parameters not modified
						print(" peak not found, something's wrong!")
						pixer[im]  = 0.0
						data[im].set_attr("pixerr", pixer[im])
						data[im].set_attr('xform.projection', Torg[im-seg_start])

			if myid == main_node:
				sparx_utilities.print_msg("Time of alignment = %d\n"%(time.time()-start_time))
				start_time = time.time()

			mpi.mpi_barrier(mpi.MPI_COMM_WORLD)

			terminate = mpi.mpi_bcast(terminate, 1, mpi.MPI_INT, 0, mpi.MPI_COMM_WORLD)
			terminate = int(terminate[0])

			mpi.mpi_barrier(mpi.MPI_COMM_WORLD)

			if (Iter-1) % search_iter == 0:

				if CTF:  vol = sparx_reconstruction.recons3d_4nn_ctf_MPI(myid, data, symmetry=sym, snr = snr, npad = npad)
				else:    vol = sparx_reconstruction.recons3d_4nn_MPI(myid, data, symmetry=sym, snr = snr, npad = npad)

				if myid == main_node:
					sparx_utilities.print_msg("3D reconstruction time = %d\n"%(time.time()-start_time))
					start_time = time.time()

					#drop_image(vol, os.path.join(outdir, "vol%04d.hdf"%(total_iter)))

					#  symmetry is imposed
					vol = vol.helicise(pixel_size, dp, dphi, fract, rmax, rmin)
					sparx_utilities.print_msg("Imposed delta z and delta phi      : %s,    %s\n"%(dp,dphi))
					vol = sparx_utilities.sym_vol(vol, symmetry=sym)
					ref_data = [vol, mask3D]
					#if  fourvar:  ref_data.append(varf)
					vol = user_func(ref_data)
					vol = vol.helicise(pixel_size, dp, dphi, fract, rmax, rmin)
					vol = sparx_utilities.sym_vol(vol, symmetry=sym)
					sparx_utilities.drop_image(vol, os.path.join(outdir, "volf%04d.hdf"%(total_iter)))
					sparx_utilities.print_msg("Symmetry enforcement and user function time = %d\n"%(time.time()-start_time))
					start_time = time.time()

				# using current volume
				sparx_utilities.bcast_EMData_to_all(vol, myid, main_node)

			mpi.mpi_barrier(mpi.MPI_COMM_WORLD)

			# write out headers, under MPI writing has to be done sequentially
			pass#IMPORTIMPORTIMPORT from mpi import mpi_recv, mpi_send, MPI_COMM_WORLD, MPI_FLOAT
			par_str = ['xform.projection', 'ID','pixerr']
			if myid == main_node:
				if(sparx_utilities.file_type(stack) == "bdb"):
					pass#IMPORTIMPORTIMPORT from utilities import recv_attr_dict_bdb
					sparx_utilities.recv_attr_dict_bdb(main_node, stack, data, par_str, 0, nima, number_of_proc)
				else:
					pass#IMPORTIMPORTIMPORT from utilities import recv_attr_dict
					sparx_utilities.recv_attr_dict(main_node, stack, data, par_str, 0, nima, number_of_proc)
			else:
				sparx_utilities.send_attr_dict(main_node, data, par_str, 0, nima)

			if myid == main_node:
				# write params to text file
				header(stack, params='xform.projection', fexport=os.path.join(outdir, "parameters%04d.txt"%(ooiter)))
				header(stack, params='pixerr', fexport=os.path.join(outdir, "pixelerror%04d.txt"%(ooiter)))


def filamentupdown(fildata, pixel_size, dp, dphi):
	pass#IMPORTIMPORTIMPORT from utilities import get_params_proj, get_dist

	rise  = dp/pixel_size
	ns = len(fildata)
	phig   = [0.0]*ns # given phi
	s2y    = [0.0]*ns
	coords = []*ns
	for i in range(ns):
		phig[i], theta, psi, s2x, s2y[i] = sparx_utilities.get_params_proj(fildata[i])
		coords.append(fildata[i].get_attr('ptcl_source_coord'))
		#print i,phig[i], theta, psi, s2x, s2y[i]
	terr = [0.0]*2 # total error between predicted angles and given angles
	##serr = [0.0]*2  # shift error not needed, blocked with ##
	for i in range(1, ns):
		dist = sparx_utilities.get_dist(coords[0], coords[i])
		qd = round((s2y[0] + dist)/rise)
		##yn   = s2y[0] + dist - rise*qd
		kl = -1
		for sgn in range(-1,2,2):
			phin   = (phig[0] + sgn*dphi*qd)%360.0
			err    = (phin - phig[i])%360.0
			##srr = abs(s2y[i]-yn)
			kl += 1
			##serr[kl] += srr
			terr[kl] += min(err, 360.0 - err)
			#print "    %3d   %2d   %7.1f    %7.1f   %7.1f   %6.1f    %8.2f    %8.1f"%\
			#(i,kl, round(s2y[i]), round(yn,1), round(srr,1), round(phig[i],1), round(phin,1), round(min(err, 360.0 - err),1))

	#print " ERRORS :",terr," \n        ",serr
	if(terr[0] > terr[1]):  updown = 0
	else:                   updown = 1
	#print "updown=%d"%updown	
	for i in range(ns):  fildata[i].set_attr("updown",updown)
	return
"""Multiline Comment63"""

def setfilori_SP(fildata, pixel_size, dp, dphi):
	pass#IMPORTIMPORTIMPORT from utilities		import get_params_proj, set_params_proj, get_dist
	pass#IMPORTIMPORTIMPORT from pixel_error 	import angle_diff
	pass#IMPORTIMPORTIMPORT from applications	import filamentupdown
	pass#IMPORTIMPORTIMPORT from copy 			import deepcopy
	pass#IMPORTIMPORTIMPORT from math 			import atan2, sin, cos, pi, radians
	
	#if sym != 'c1':
	#	ERROR("does not handle any point-group symmetry other than c1 for the time being.", 'setfilori_SP')

	rise 	= dp/pixel_size
	#ddphi   = pixel_size/dp*dphi
	ns 		= len(fildata)
	qv 		= numpy.pi/180.0

	phig 	= [0.0]*ns # given phi
	psig 	= [0.0]*ns # given psi
	yg 		= [0.0]*ns # given y
	xg 		= [0.0]*ns # given x
	thetag	= [0.0]*ns # given theta
	gxyz    = [[0.0 for i in range(3)]for k in range(ns) ]

	dist = [0.0]*ns
	coords0 = fildata[0].get_attr('ptcl_source_coord')
	for i in range(ns):
		coordsi = fildata[i].get_attr('ptcl_source_coord')
		dist[i] = sparx_utilities.get_dist(coords0, coordsi)
		phig[i], thetag[i], psig[i] , xg[i], yg[i] = sparx_utilities.get_params_proj(fildata[i])
		#print "before setfil_SP: %3d  %5.1f   %5.1f   %5.1f   %5.1f   %5.1f"%(i,phig[i], thetag[i], psig[i] , xg[i], yg[i])
		gxyz [i][0] = numpy.cos(numpy.radians(phig[i]))
		gxyz [i][1] = numpy.sin(numpy.radians(phig[i]))
		gxyz [i][2] = yg[i]
		if( abs(psig[i] - psig[0]) )> 90.0:
			sparx_global_def.ERROR('PSI should be pointing in the same direction for all segments belonging to same filament', 'setfilori_SP',1)
	
	# Generate a spring starting from shift and phi equal zero
	sgn = (1 - fildata[0].get_attr("updown")*2)
	s2y = [0.0]*ns
	phi = [0.0]*ns
	bys = [0.0]*ns
	bang = [0.0]*ns
	cxyz = [[0.0 for i in range(3)]for k in range(ns) ]

	i= 0
	s2y[i] = 0.0
	phi[i] = 0.0
	step = 0.1
	qshift = -rise/2
	toto = 1.0e23
	qshifm = 0
	while( qshift < rise/2 ):
		#print qshift
		i= 0
		s2y[i] = qshift
		phi[i] = 0.0

		qd = round((s2y[0] + dist[i])/rise)

		for i in range(1, ns):
			qd     = round((s2y[0] + dist[i])/rise)
			s2y[i] = s2y[0] + dist[i] - rise*qd
			phi[i] = (phi[0] + sgn*dphi*qd)%360.0
		phidiff = sparx_pixel_error.angle_diff(phi, phig)
		#print  phidiff

		for i in range(ns):
			phi[i] = phi[i]+phidiff
			temp = numpy.radians(phi[i])
			cxyz [i][0] = numpy.cos(temp)
			cxyz [i][1] = numpy.sin(temp)
			cxyz [i][2] = s2y[i]

		qdst = 0.0
		for i in range(ns):
			for k in range(3):
				qdst += (gxyz[i][k]-cxyz[i][k])**2
		#print qdst,toto
		if(qdst<toto):
			toto = qdst
			for i in range(ns):
				bys[i]   = cxyz[i][2]
				bang[i]  = phi[i]%360.0
				
			#print "found better", phidiff,qshift
		qshift += step


	#print  " phidiff,shift", bang,bshift
	for i in range(ns): phi[i] = (phi[i]+phidiff)%360.0
	for i in range(ns):
		sparx_utilities.set_params_proj(fildata[i], [bang[i], thetag[i], psig[i] , xg[i], bys[i]])
		#print    "    %3d  %7.1f    %9.3f"%(i,bang[i]-phig[i],bys[i]-yg[i])
	#print yer, per
	
	return

def prepare_refffts( volft, kb, nx,ny,nz, segmask, delta,  \
		MPI=False, psimax=1.0, psistep=1.0, kbx = None, kby = None, initial_theta = None, delta_theta = None):

	pass#IMPORTIMPORTIMPORT from projection   import prgs
	pass#IMPORTIMPORTIMPORT from math         import sin, cos, radians
	pass#IMPORTIMPORTIMPORT from applications import MPI_start_end
	pass#IMPORTIMPORTIMPORT from utilities    import even_angles
	pass#IMPORTIMPORTIMPORT from alignment	  import preparerefsgrid, ringwe
	pass#IMPORTIMPORTIMPORT from fundamentals import fft, rot_shift2D

	# generate list of Eulerian angles for reference projections
	#  phi, theta, psi
	mode = "F"
	ref_angles = []
	if initial_theta is None:
		phiphi = 0.0
		while( phiphi < 360.0 ):
			ref_angles.append([phiphi, 90.0, 90.0])
			phiphi += delta
	else:
		if delta_theta is None: delta_theta = 1.0
		ththt = 90.0
		while(ththt >= initial_theta ):
			phiphi = 0.0
			while( phiphi < 360.0 ):
				ref_angles.append([phiphi, ththt, 90.0])
				if(ththt != 90.0): ref_angles.append([phiphi, 180.0 - ththt, 90.0])
				phiphi += delta
			ththt -= delta_theta

	num_ref = len(ref_angles)
	for q in ref_angles:  print(q)
	if MPI:
		pass#IMPORTIMPORTIMPORT from mpi import mpi_comm_rank, mpi_comm_size, MPI_COMM_WORLD
		myid = mpi.mpi_comm_rank( mpi.MPI_COMM_WORLD )
		ncpu = mpi.mpi_comm_size( mpi.MPI_COMM_WORLD )
	else:
		ncpu = 1
		myid = 0
	pass#IMPORTIMPORTIMPORT from applications import MPI_start_end
	ref_start, ref_end = MPI_start_end(num_ref, ncpu, myid)

	refrings = []     # list of (image objects) reference projections in Fourier representation

	nr = int(2*psimax/psistep)+1

	for i in range(num_ref):
		refrings.append([EMAN2_cppwrap.EMData(nx,ny,1,False) for j in range(nr)])

	if kbx is None:
		for i in range(ref_start, ref_end):
			prjref = sparx_projection.prgs(volft, kb, [ref_angles[i][0], ref_angles[i][1], ref_angles[i][2], 0.0, 0.0])
			EMAN2_cppwrap.Util.mul_img(prjref, segmask )
			refrings[i] = sparx_alignment.preparerefsgrid(prjref, psimax, psistep)
	else:
		sparx_global_def.ERROR("do not handle this case","prepare_refffts",1)
		os.sys.exit()
	if MPI:
		pass#IMPORTIMPORTIMPORT from utilities import bcast_EMData_to_all
		for i in range(num_ref):
			for j in range(ncpu):
				ref_start, ref_end = MPI_start_end(num_ref, ncpu, j)
				if i >= ref_start and i < ref_end: rootid = j
			for j in range(nr):
				sparx_utilities.bcast_EMData_to_all(refrings[i][j], myid, rootid)

	for i in range(num_ref):
		q0  = numpy.radians(ref_angles[i][0])
		q1  = numpy.radians(ref_angles[i][1])
		sq1 = numpy.sin(q1)
		n1 = sq1*numpy.cos(q0)
		n2 = sq1*numpy.sin(q0)
		n3 = numpy.cos(q1)
		refrings[i][0].set_attr_dict( {"n1":n1, "n2":n2, "n3":n3} )
		refrings[i][0].set_attr("phi",   ref_angles[i][0])
		refrings[i][0].set_attr("theta", ref_angles[i][1])
		refrings[i][0].set_attr("psi",   ref_angles[i][2])

	return refrings

def prepare_helical_refangles(delta, initial_theta = None, delta_theta = None):
	# generate list of Eulerian angles for reference projections
	#  phi, theta, psi
	mode = "F"
	ref_angles = []
	if initial_theta is None:
		phiphi = 0.0
		while( phiphi < 360.0 ):
			ref_angles.append([phiphi, 90.0, 90.0])
			phiphi += delta
	else:
		if delta_theta is None: delta_theta = 1.0
		ththt = 90.0
		while(ththt >= initial_theta ):
			phiphi = 0.0
			while( phiphi < 360.0 ):
				ref_angles.append([phiphi, ththt, 90.0])
				if(ththt != 90.0): ref_angles.append([phiphi, 180.0 - ththt, 90.0])
				phiphi += delta
			ththt -= delta_theta
	return ref_angles

def prepare_reffft1( volft, kb, ref_angles, segmask, psimax=1.0, psistep=1.0, kbx = None, kby = None):

	pass#IMPORTIMPORTIMPORT from projection   import prgs
	pass#IMPORTIMPORTIMPORT from alignment    import preparerefsgrid
	pass#IMPORTIMPORTIMPORT from math         import sin, cos, radians

	#refrings = []     # list of (image objects) reference projections in Fourier representation

	nr = int(2*psimax/psistep)+1

	if kbx is None:
		prjref = sparx_projection.prgs(volft, kb, [ref_angles[0], ref_angles[1], ref_angles[2], 0.0, 0.0])
		EMAN2_cppwrap.Util.mul_img(prjref, segmask )
		#  EVENTUALLY PASS kb inside
		refrings = sparx_alignment.preparerefsgrid(prjref, psimax, psistep)
	else:
		sparx_global_def.ERROR("do not handle this case","prepare_refffts",1)
		os.sys.exit()

	q0  = numpy.radians(ref_angles[0])
	q1  = numpy.radians(ref_angles[1])
	sq1 = numpy.sin(q1)
	n1 = sq1*numpy.cos(q0)
	n2 = sq1*numpy.sin(q0)
	n3 = numpy.cos(q1)
	refrings[0].set_attr_dict( {"n1":n1, "n2":n2, "n3":n3} )
	refrings[0].set_attr("phi",   ref_angles[0])
	refrings[0].set_attr("theta", ref_angles[1])
	refrings[0].set_attr("psi",   ref_angles[2])

	return refrings


def prepare_reffft2( volft, kb, ref_angles, segmask, psimax=1.0, psistep=1.0, kbx = None, kby = None):

	pass#IMPORTIMPORTIMPORT from projection   import prgs
	pass#IMPORTIMPORTIMPORT from alignment    import preparerefsgrid, preparerefsgrid1
	pass#IMPORTIMPORTIMPORT from math         import sin, cos, radians

	#refrings = []     # list of (image objects) reference projections in Fourier representation

	
	if kbx is None:
		prjref = sparx_projection.prgs(volft, kb, [ref_angles[0], ref_angles[1], ref_angles[2], 0.0, 0.0])
		EMAN2_cppwrap.Util.mul_img(prjref, segmask )
		#  EVENTUALLY PASS kb inside
		refrings = sparx_alignment.preparerefsgrid1(prjref, psimax, psistep)
	else:
		sparx_global_def.ERROR("do not handle this case","prepare_refffts",1)
		os.sys.exit()

	q0  = numpy.radians(ref_angles[0])
	q1  = numpy.radians(ref_angles[1])
	sq1 = numpy.sin(q1)
	n1 = sq1*numpy.cos(q0)
	n2 = sq1*numpy.sin(q0)
	n3 = numpy.cos(q1)
	refrings[0].set_attr_dict( {"n1":n1, "n2":n2, "n3":n3} )
	refrings[0].set_attr("phi",   ref_angles[0])
	refrings[0].set_attr("theta", ref_angles[1])
	refrings[0].set_attr("psi",   ref_angles[2])

	return refrings
	
def symsearch_MPI(ref_vol, outdir, maskfile, dp, ndp, dp_step, dphi, ndphi, dphi_step,\
	rmin, rmax, fract, sym, user_func_name, datasym,\
	pixel_size, debug):

	pass#IMPORTIMPORTIMPORT from alignment      import helios, helios7
	pass#IMPORTIMPORTIMPORT from utilities      import model_circle, get_image, drop_image, get_input_from_string, pad, model_blank, sym_vol
	pass#IMPORTIMPORTIMPORT from utilities      import bcast_list_to_all, bcast_number_to_all, reduce_EMData_to_root, bcast_EMData_to_all
	pass#IMPORTIMPORTIMPORT from utilities      import send_attr_dict
	pass#IMPORTIMPORTIMPORT from utilities      import get_params_proj, set_params_proj, file_type
	pass#IMPORTIMPORTIMPORT from fundamentals   import rot_avg_image
	pass#IMPORTIMPORTIMPORT from pixel_error    import max_3D_pixel_error
	pass#IMPORTIMPORTIMPORT import os
	pass#IMPORTIMPORTIMPORT import types
	pass#IMPORTIMPORTIMPORT from utilities      import print_begin_msg, print_end_msg, print_msg
	pass#IMPORTIMPORTIMPORT from mpi            import mpi_bcast, mpi_comm_size, mpi_comm_rank, MPI_FLOAT, MPI_COMM_WORLD, mpi_barrier
	pass#IMPORTIMPORTIMPORT from mpi            import mpi_recv,  mpi_send
	pass#IMPORTIMPORTIMPORT from mpi            import mpi_reduce, MPI_INT, MPI_SUM
	pass#IMPORTIMPORTIMPORT from filter         import filt_ctf
	pass#IMPORTIMPORTIMPORT from projection     import prep_vol, prgs
	pass#IMPORTIMPORTIMPORT from statistics     import hist_list, varf3d_MPI
	pass#IMPORTIMPORTIMPORT from applications   import MPI_start_end
	pass#IMPORTIMPORTIMPORT from EMAN2          import Vec2f
	pass#IMPORTIMPORTIMPORT from string         import lower,split
	pass#IMPORTIMPORTIMPORT from math           import cos, pi

	number_of_proc = mpi.mpi_comm_size(mpi.MPI_COMM_WORLD)
	myid           = mpi.mpi_comm_rank(mpi.MPI_COMM_WORLD)
	main_node = 0

	if myid == 0:
		if os.path.exists(outdir):  nx = 1
		else:  nx = 0
	else:  nx = 0
	ny = sparx_utilities.bcast_number_to_all(nx, source_node = main_node)

	if ny == 1:  sparx_global_def.ERROR('Output directory exists, please change the name and restart the program', "symsearch_MPI", 1,myid)
	mpi.mpi_barrier(mpi.MPI_COMM_WORLD)

	if myid == main_node:
		os.mkdir(outdir)
		pass#IMPORTIMPORTIMPORT import global_def
		sparx_global_def.LOGFILE =  os.path.join(outdir, sparx_global_def.LOGFILE)
		sparx_utilities.print_begin_msg("symsearch_MPI")
	mpi.mpi_barrier(mpi.MPI_COMM_WORLD)

	nlprms = (2*ndp+1)*(2*ndphi+1)
	if nlprms< number_of_proc:
		sparx_global_def.ERROR('number of CPUs is larger than the number of helical search, please reduce it or at this moment modify ndp,dphi in the program', "ihrsr_MPI", 1,myid)

	if debug:
		pass#IMPORTIMPORTIMPORT from time import sleep
		while not os.path.exists(outdir):
			print("Node ",myid,"  waiting...")
			time.sleep(5)

		info_file = os.path.join(outdir, "progress%04d"%myid)
		finfo = open(info_file, 'w')
	else:
		finfo = None

	vol     = EMAN2_cppwrap.EMData()
	vol.read_image(ref_vol)

	if myid == main_node:
		pass#IMPORTIMPORTIMPORT import user_functions
		user_func = sparx_user_functions.factory[user_func_name]

		sparx_utilities.print_msg("Reference volume                          : %s\n"%(ref_vol))	
		sparx_utilities.print_msg("Output directory                          : %s\n"%(outdir))
		sparx_utilities.print_msg("Maskfile                                  : %s\n"%(maskfile))
		sparx_utilities.print_msg("min radius for helical search (in pix)    : %5.4f\n"%(rmin))
		sparx_utilities.print_msg("max radius for helical search (in pix)    : %5.4f\n"%(rmax))
		sparx_utilities.print_msg("fraction of volume used for helical search: %5.4f\n"%(fract))
		sparx_utilities.print_msg("initial symmetry - angle                  : %5.4f\n"%(dphi))
		sparx_utilities.print_msg("initial symmetry - axial rise             : %5.4f\n"%(dp))
		sparx_utilities.print_msg("symmetry output doc file                  : %s\n"%(datasym))
		sparx_utilities.print_msg("User function                             : %s\n"%(user_func_name))

	if maskfile:
		if type(maskfile) is bytes: mask3D = sparx_utilities.get_image(maskfile)
		else:                                  mask3D = maskfile
	else: mask3D = None
	#else: mask3D = model_circle(last_ring, nx, nx, nx)

	pass#IMPORTIMPORTIMPORT from time import time
	#from filter import filt_gaussl
	#vol = filt_gaussl(vol, 0.25)
	start_time = time.time()
	if myid == main_node:
		lprms = []
		for i in range(-ndp,ndp+1,1):
			for j in range(-ndphi,ndphi+1,1):
				lprms.append( dp   + i*dp_step)
				lprms.append( dphi + j*dphi_step)
		#print "lprms===",lprms
		recvpara = []
		for im in range(number_of_proc):
			helic_ib, helic_ie = MPI_start_end(nlprms, number_of_proc, im)
			recvpara.append(helic_ib )
			recvpara.append(helic_ie )

	para_start, para_end = MPI_start_end(nlprms, number_of_proc, myid)

	list_dps     = [0.0]*((para_end-para_start)*2)
	list_fvalues = [-1.0]*((para_end-para_start)*1)

	if myid == main_node:
		for n in range(number_of_proc):
			if n!=main_node: mpi.mpi_send(lprms[2*recvpara[2*n]:2*recvpara[2*n+1]], 2 * (recvpara[2*n+1]-recvpara[2*n]), mpi.MPI_FLOAT, n, sparx_global_def.SPARX_MPI_TAG_UNIVERSAL, mpi.MPI_COMM_WORLD)
			else:            list_dps = lprms[2*recvpara[2*0]:2*recvpara[2*0+1]]
	else:
		list_dps = mpi.mpi_recv((para_end-para_start) * 2, mpi.MPI_FLOAT, main_node, sparx_global_def.SPARX_MPI_TAG_UNIVERSAL, mpi.MPI_COMM_WORLD)

	list_dps = list(map(float, list_dps))

	local_pos = [0.0, 0.0, -1.0e20]
	for i in range(para_end-para_start):
		fvalue = sparx_alignment.helios7(vol, pixel_size, list_dps[i*2], list_dps[i*2+1], fract, rmax, rmin)
		if(fvalue >= local_pos[2]):
			local_pos = [list_dps[i*2], list_dps[i*2+1], fvalue ]
	if myid == main_node:
		list_return = [0.0]*(3*number_of_proc)
		for n in range(number_of_proc):
			if n != main_node: list_return[3*n:3*n+3]                 = mpi.mpi_recv(3, mpi.MPI_FLOAT, n, sparx_global_def.SPARX_MPI_TAG_UNIVERSAL, mpi.MPI_COMM_WORLD)
			else:              list_return[3*main_node:3*main_node+3]  = local_pos[:]
	else:
		mpi.mpi_send(local_pos, 3, mpi.MPI_FLOAT, main_node, sparx_global_def.SPARX_MPI_TAG_UNIVERSAL, mpi.MPI_COMM_WORLD)

	if myid == main_node:	
		maxvalue = list_return[2]
		for i in range(number_of_proc):
			if( list_return[i*3+2] >= maxvalue ):
				maxvalue = list_return[i*3+2]
				dp       = list_return[i*3+0]
				dphi     = list_return[i*3+1]
		dp   = float(dp)
		dphi = float(dphi)
		#print  "  GOT dp dphi",dp,dphi

		vol  = vol.helicise(pixel_size, dp, dphi, fract, rmax, rmin)
		sparx_utilities.drop_image(vol, os.path.join(outdir, "vol.hdf"))

		sparx_utilities.print_msg("New delta z and delta phi      : %s,    %s\n\n"%(dp,dphi))		


	if(myid==main_node):
		fofo = open(os.path.join(outdir,datasym),'a')
		fofo.write('  %12.4f   %12.4f\n'%(dp,dphi))
		fofo.close()
		"""Multiline Comment64"""

	# del varf
	if myid == main_node: sparx_utilities.print_end_msg("symsearch_MPI")




#  What follows is old code to make old meridian to run  01/27/2016

def sali3d_base_old(stack, ref_vol = None, Tracker = None, mpi_comm = None, log = None):
	"""
		parameters: list of (all) projections | reference volume is optional, the data is shrank, 
		  the program does not know anything about shrinking| ...
		Data is assumed to be CTF multiplied and the ctf_applied flag to be set.
		The alignment done depends on nsoft:
					 nsoft = 0 & an = -1: exhaustive deterministic
					 nsoft = 0 & an > 0 : local deterministic
					 nsoft = 1 shc
					 nsoft >1  shc_multi
		
	"""

	pass#IMPORTIMPORTIMPORT from alignment       import Numrinit, prepare_refrings
	pass#IMPORTIMPORTIMPORT from alignment       import proj_ali_incore,  proj_ali_incore_zoom,  proj_ali_incore_local, proj_ali_incore_local_zoom
	pass#IMPORTIMPORTIMPORT from alignment       import shc, center_projections_3D
	pass#IMPORTIMPORTIMPORT from utilities       import bcast_number_to_all, bcast_EMData_to_all, 	wrap_mpi_gatherv, wrap_mpi_bcast, model_blank
	pass#IMPORTIMPORTIMPORT from utilities       import get_im, file_type, model_circle, get_input_from_string, get_params_proj, set_params_proj
	pass#IMPORTIMPORTIMPORT from mpi             import mpi_bcast, mpi_comm_size, mpi_comm_rank, MPI_FLOAT, MPI_COMM_WORLD, mpi_barrier, mpi_reduce, MPI_INT, MPI_SUM
	pass#IMPORTIMPORTIMPORT from projection      import prep_vol
	pass#IMPORTIMPORTIMPORT from statistics      import hist_list
	pass#IMPORTIMPORTIMPORT from applications    import MPI_start_end
	pass#IMPORTIMPORTIMPORT from filter          import filt_ctf
	pass#IMPORTIMPORTIMPORT from global_def      import Util
	pass#IMPORTIMPORTIMPORT from fundamentals    import resample, fshift
	pass#IMPORTIMPORTIMPORT from multi_shc       import shc_multi
	#from development     import do_volume_mrk01
	pass#IMPORTIMPORTIMPORT import user_functions
	pass#IMPORTIMPORTIMPORT from EMAN2           import EMUtil, EMData
	pass#IMPORTIMPORTIMPORT import types
	pass#IMPORTIMPORTIMPORT from time            import time

	nsoft            = Tracker["nsoft"]
	saturatecrit     = Tracker["saturatecrit"]
	pixercutoff      = Tracker["pixercutoff"]
	zoom             = Tracker["zoom"]
	center           = Tracker["constants"]["center"]
	CTF              = Tracker["constants"]["CTF"]
	ref_a            = Tracker["constants"]["ref_a"]
	rstep            = Tracker["constants"]["rs"]
	sym              = Tracker["constants"]["sym"]
	first_ring       = 1
	last_ring        = Tracker["radius"]
	xr               = Tracker["xr"]
	yr               = Tracker["yr"]
	ts               = Tracker["ts"]
	an               = Tracker["an"]
	delta            = Tracker["delta"]
	max_iter         = Tracker["maxit"]

	if mpi_comm == None: mpi_comm = mpi.MPI_COMM_WORLD

	if log == None:
		pass#IMPORTIMPORTIMPORT from logger import Logger
		log = sparx_logger.Logger()

	number_of_proc = mpi.mpi_comm_size(mpi_comm)
	myid           = mpi.mpi_comm_rank(mpi_comm)
	main_node      = 0

	if myid == main_node:
		log.add("Start sali3d_base, nsoft = %1d"%nsoft)

	xrng        = sparx_utilities.get_input_from_string(xr)
	if  yr == "-1":  yrng = xrng
	else          :  yrng = sparx_utilities.get_input_from_string(yr)
	step        = sparx_utilities.get_input_from_string(ts)
	delta       = sparx_utilities.get_input_from_string(delta)
	lstp = min(len(xrng), len(yrng), len(step), len(delta))
	if an == "-1":
		an = [-1] * lstp
	else:
		an = sparx_utilities.get_input_from_string(an)

	if( type(stack) is bytes ):
		if myid == main_node:
			total_nima = EMAN2_cppwrap.EMUtil.get_image_count( stack )
		else:
			total_nima = 0
		total_nima = sparx_utilities.wrap_mpi_bcast(total_nima, main_node, mpi_comm)
		list_of_particles = list(range(total_nima))
		image_start, image_end = MPI_start_end(total_nima, number_of_proc, myid)
		# create a list of images for each node
		list_of_particles = list_of_particles[image_start: image_end]
		nima = len(list_of_particles)

	else:
		list_of_particles = list(range(len(stack)))
		nima = len(list_of_particles)
		total_nima = len(list_of_particles)
		total_nima = mpi.mpi_reduce(total_nima, 1, mpi.MPI_INT, mpi.MPI_SUM, 0, mpi.MPI_COMM_WORLD)
		total_nima = mpi.mpi_bcast(total_nima, 1, mpi.MPI_INT, 0, mpi.MPI_COMM_WORLD)
		total_nima = int(total_nima[0])


	if myid == 0:
		finfo = None
		"""Multiline Comment65"""
	else:
		finfo = None

	if( myid == main_node):
		if( type(stack) is bytes ):  mask2D = sparx_utilities.get_im(stack, list_of_particles[0])
		else:                                   mask2D = stack[list_of_particles[0]]
		nx = mask2D.get_xsize()
	else:  nx = 0
	nx  = sparx_utilities.bcast_number_to_all(nx, source_node = main_node)
	if last_ring < 0:	last_ring = int(nx/2) - 2

	numr	= sparx_alignment.Numrinit(first_ring, last_ring, rstep, "F")

	data = [None]*nima
	for im in range(nima):
		if( type(stack) is bytes ):  data[im] = sparx_utilities.get_im(stack, list_of_particles[im])
		else:                                   data[im] = stack[list_of_particles[im]]
	mpi.mpi_barrier(mpi_comm)


	if myid == main_node:
		start_time = time.time()

	#  Read	template volume if provided or reconstruct it
	#  Apply initfl first, meaning true fl has to be preserved
	#fl = Tracker["lowpass"]
	#Tracker["lowpass"] = Tracker["initialfl"]
	user_func = Tracker["constants"]["user_func"]
	if ref_vol:
		#vol = do_volume_mrk01(ref_vol, Tracker, 0, mpi_comm)
		ref_data = [ref_vol, Tracker, 0, mpi_comm]
		vol = user_func(ref_data)
	else:
		#vol = do_volume_mrk01(data, Tracker, 0, mpi_comm)
		ref_data = [data, Tracker, 0, mpi_comm]
		vol = user_func(ref_data)
	#  Restore desired fl
	#Tracker["lowpass"] = fl

	# log
	if myid == main_node:
		log.add("Setting of reference 3D reconstruction time = %10.1f\n"%(time.time()-start_time))
		start_time = time.time()


	pixer = [0.0]*nima
	historyofchanges = [0.0, 0.5, 1.0]
	#par_r = [[] for im in list_of_particles ]
	cs = [0.0]*3
	total_iter = 0
	# do the projection matching
	if zoom: lstp = 1
	for N_step in range(lstp):

		terminate = 0
		Iter = 0
		while Iter < max_iter and terminate == 0:

			Iter += 1
			total_iter += 1

			mpi.mpi_barrier(mpi_comm)
			if myid == main_node:
				log.add("ITERATION #%3d,  inner iteration #%3d"%(total_iter, Iter))
				log.add("Delta = %5.2f, an = %5.2f, xrange = %5d, yrange = %5d, step = %5.2f\n"%\
							(delta[N_step], an[N_step], xrng[N_step], yrng[N_step], step[N_step]))
				start_time = time.time()



			#=========================================================================
			# build references
			volft, kb = sparx_projection.prep_vol(vol)
			refrings = sparx_alignment.prepare_refrings(volft, kb, nx, delta[N_step], ref_a, sym, numr, MPI=mpi_comm, phiEqpsi = "Zero")
			del volft, kb
			#=========================================================================

			if myid == main_node:
				log.add("Time to prepare rings: %10.1f\n" % (time.time()-start_time))
				start_time = time.time()

			#=========================================================================
			#  there is no need for previousmax for deterministic searches
			if total_iter == 1 and nsoft > 0:
				if(an[N_step] < 0.0):
					# adjust params to references, calculate psi+shifts, calculate previousmax
					# generate list of angles
					pass#IMPORTIMPORTIMPORT from alignment import generate_list_of_reference_angles_for_search
					list_of_reference_angles = \
					sparx_alignment.generate_list_of_reference_angles_for_search([[refrings[lr].get_attr("phi"), refrings[lr].get_attr("theta")] for lr in range(len(refrings))], sym=sym)			
					for im in range(nima):
						previousmax = data[im].get_attr_default("previousmax", -1.0e23)
						if(previousmax == -1.0e23):
							peak, pixer[im] = sparx_alignment.proj_ali_incore_local(data[im], refrings, list_of_reference_angles, numr, \
									xrng[N_step], yrng[N_step], step[N_step], delta[N_step]*2.5, sym = sym)
							data[im].set_attr("previousmax", peak)
					del list_of_reference_angles
				else:
					#  Here it is supposed to be shake and bake for local SHC, but it would have to be signaled somehow
					for im in range(nima):
						data[im].set_attr("previousmax", -1.0e23)
				if myid == main_node:
					log.add("Time to calculate first psi+shifts+previousmax: %10.1f\n" % (time.time()-start_time))
					start_time = time.time()
			#=========================================================================

			mpi.mpi_barrier(mpi_comm)
			if myid == main_node:  start_time = time.time()
			#=========================================================================
			# alignment
			#number_of_checked_refs = 0
			par_r = [0]*max(2,(nsoft+1))
			if(an[N_step] > 0):
				# generate list of angles
				pass#IMPORTIMPORTIMPORT from alignment import generate_list_of_reference_angles_for_search
				list_of_reference_angles = \
				sparx_alignment.generate_list_of_reference_angles_for_search([[refrings[lr].get_attr("phi"), refrings[lr].get_attr("theta")] for lr in range(len(refrings))], sym=sym)			
			else:  list_of_reference_angles = [[1.0,1.0]]
			for im in range(nima):
				if(nsoft == 0):
					if(an[N_step] == -1):
						#  In zoom option each projection goes through shift zoom alignment
						if  zoom: peak, pixer[im] = sparx_alignment.proj_ali_incore_zoom(data[im], refrings, numr, \
														xrng, yrng, step, sym=sym)
						else:  peak, pixer[im] = sparx_alignment.proj_ali_incore(data[im], refrings, numr, \
														xrng[N_step], yrng[N_step], step[N_step], sym=sym)
					else:
						if  zoom: peak, pixer[im] = sparx_alignment.proj_ali_incore_local_zoom(data[im], refrings, list_of_reference_angles, numr, \
									xrng, yrng, step, an, finfo = finfo, sym=sym)
						else:  peak, pixer[im] = sparx_alignment.proj_ali_incore_local(data[im], refrings, list_of_reference_angles, numr, \
									xrng[N_step], yrng[N_step], step[N_step], an[N_step], finfo = finfo, sym=sym)
					if(pixer[im] == 0.0):  par_r[0] += 1
				elif(nsoft == 1):
					peak, pixer[im], number_of_checked_refs, iref = \
						sparx_alignment.shc(data[im], refrings, list_of_reference_angles, numr, xrng[N_step], yrng[N_step], step[N_step], an[N_step], sym, finfo = finfo)
					if(pixer[im] == 0.0):  par_r[0] += 1
				elif(nsoft > 1):
					#  This is not functional
					peak, pixer[im], checked_refs, number_of_peaks = sparx_multi_shc.shc_multi(data[im], refrings, numr, \
												xrng[N_step], yrng[N_step], step[N_step], an[N_step], nsoft, sym, finfo = finfo)
					par_r[number_of_peaks] += 1
					#number_of_checked_refs += checked_refs
			if(an[N_step] > 0):  del list_of_reference_angles
			#=========================================================================
			mpi.mpi_barrier(mpi_comm)
			if myid == main_node:
				#print  data[0].get_attr_dict()
				log.add("Time of alignment = %10.1f\n"%(time.time()-start_time))
				start_time = time.time()
			#=========================================================================
			#output pixel errors, check stop criterion
			all_pixer = sparx_utilities.wrap_mpi_gatherv(pixer, 0, mpi_comm)
			par_r = mpi.mpi_reduce(par_r, len(par_r), mpi.MPI_INT, mpi.MPI_SUM, 0, mpi.MPI_COMM_WORLD)
			#total_checked_refs = wrap_mpi_gatherv([number_of_checked_refs], main_node, mpi_comm)
			terminate = 0
			if myid == main_node:
				#total_checked_refs = sum(total_checked_refs)
				if(nsoft < 2):  par_r[1] = total_nima - par_r[0]
				log.add("=========== Number of better orientations found ==============")
				for lhx in range(len(par_r)):
					msg = "            %5d     %7d"%(lhx, par_r[lhx])
					log.add(msg)
				log.add("_______________________________________________________")
				changes = par_r[0]/float(total_nima)
				if(  changes > saturatecrit ):
					if( Iter == 1 ):
						log.add("Will continue even though %4.2f images did not find better orientations"%saturatecrit)
					else:
						terminate = 1
						log.add("...............")
						log.add(">>>>>>>>>>>>>>>   Will terminate as %4.2f images did not find better orientations"%saturatecrit)
				if( terminate == 0 ):
					historyofchanges.append(changes)
					historyofchanges = historyofchanges[:3]
					historyofchanges.sort()
					"""Multiline Comment66"""

				lhist = 20
				region, histo = sparx_statistics.hist_list(all_pixer, lhist)
				log.add("=========== Histogram of pixel errors ==============")
				for lhx in range(lhist):
					msg = "          %10.3f     %7d"%(region[lhx], histo[lhx])
					log.add(msg)
				log.add("____________________________________________________")
				if(nsoft<2 and terminate == 0):
					lhx = 0
					for msg in all_pixer:
						if(msg < pixercutoff): lhx += 1
					lhx = float(lhx)/float(total_nima)
					log.add(">>> %4.2f images had pixel error <%5.2f"%(lhx,pixercutoff))
					if( lhx > saturatecrit):
						if( Iter == 1 ):
							log.add("Will continue even though %4.2f images had pixel error < %5.2f"%(saturatecrit,pixercutoff))
						else:
							terminate = 1
							log.add("...............")
							log.add(">>>>>>>>>>>>>>>   Will terminate as %4.2f images had pixel error < %5.2f"%(saturatecrit,pixercutoff))
			terminate = sparx_utilities.wrap_mpi_bcast(terminate, main_node, mpi_comm)
			#=========================================================================
			mpi.mpi_barrier(mpi_comm)
			if myid == main_node:
				#print  data[0].get_attr_dict()
				log.add("Time to compute histograms = %10.1f\n"%(time.time()-start_time))
				start_time = time.time()


			#=========================================================================
			mpi.mpi_barrier(mpi_comm)
			if( terminate or (Iter == max_iter) ):
				# gather parameters
				params = []
				for im in range(nima):
					t = sparx_utilities.get_params_proj(data[im])
					params.append( [t[0], t[1], t[2], t[3], t[4]] )
				params = sparx_utilities.wrap_mpi_gatherv(params, main_node, mpi_comm)
			# centering and volume reconstruction if not terminating
			else:
				#=========================================================================
				# centering
				if center == -1 and sym[0] == 'c':
					pass#IMPORTIMPORTIMPORT from utilities      import estimate_3D_center_MPI, rotate_3D_shift
					cs[0], cs[1], cs[2], dummy, dummy = sparx_utilities.estimate_3D_center_MPI(data, total_nima, myid, number_of_proc, main_node, mpi_comm=mpi_comm)
					if myid == main_node:
						msg = " Average center x = %10.3f        Center y = %10.3f        Center z = %10.3f\n"%(cs[0], cs[1], cs[2])
						log.add(msg)
					if int(sym[1]) > 1:
						cs[0] = cs[1] = 0.0
						if myid == main_node:
							log.add("For symmetry group cn (n>1), we only center the volume in z-direction\n")
					cs = mpi.mpi_bcast(cs, 3, mpi.MPI_FLOAT, main_node, mpi_comm)
					cs = [-float(cs[0]), -float(cs[1]), -float(cs[2])]
					sparx_utilities.rotate_3D_shift(data, cs)
				#=========================================================================
				if myid == main_node:
					start_time = time.time()
				#vol = do_volume_mrk01(data, Tracker, total_iter, mpi_comm)
				ref_data = [data, Tracker, total_iter, mpi_comm]
				user_func = Tracker["constants"] ["user_func"]
				vol = user_func(ref_data)
				#if myid == main_node:  vol.write_image('soft/smvol%04d.hdf'%total_iter)
				# log
				if myid == main_node:
					log.add("3D reconstruction time = %10.1f\n"%(time.time()-start_time))
					start_time = time.time()
			#=========================================================================

			"""Multiline Comment67"""


	if myid == main_node:
		log.add("Finish sali3d_base, nsoft = %1d"%nsoft)
	return params


# MUST RETURN
# import sys
# sys.path.insert(0, "~/EMAN2/bin")
# sys.path.insert(0, "~/EMAN2/lib")
# print 
# print sys.path
# print
# import development
# reload(development)
# from development import sali3d_base_h_01
# 
# sali3d_base = sali3d_base_h_01


def slocal_ali3d_base_old(stack, templatevol, Tracker, mpi_comm = None, log= None, chunk = -1.0, debug = False ):
	"""

	"""
	pass#IMPORTIMPORTIMPORT from alignment        import eqproj_cascaded_ccc
	pass#IMPORTIMPORTIMPORT from filter           import filt_ctf
	pass#IMPORTIMPORTIMPORT from projection       import prep_vol
	pass#IMPORTIMPORTIMPORT from fundamentals     import resample
	pass#IMPORTIMPORTIMPORT from utilities        import bcast_number_to_all, model_circle, get_params_proj, set_params_proj
	pass#IMPORTIMPORTIMPORT from utilities        import bcast_EMData_to_all, bcast_list_to_all, send_attr_dict, wrap_mpi_bcast, wrap_mpi_gatherv
	pass#IMPORTIMPORTIMPORT from utilities        import get_image, drop_image, file_type, get_im, get_input_from_string, model_blank
	pass#IMPORTIMPORTIMPORT from utilities        import amoeba_multi_level, rotate_3D_shift, estimate_3D_center_MPI
	pass#IMPORTIMPORTIMPORT from utilities        import print_begin_msg, print_end_msg, print_msg
	#from development      import do_volume_mrk01
	pass#IMPORTIMPORTIMPORT import user_functions
	pass#IMPORTIMPORTIMPORT from statistics       import varf3d_MPI
	pass#IMPORTIMPORTIMPORT from math             import pi
	pass#IMPORTIMPORTIMPORT from mpi              import mpi_bcast, mpi_comm_size, mpi_comm_rank, MPI_FLOAT, MPI_COMM_WORLD, mpi_barrier
	pass#IMPORTIMPORTIMPORT from mpi              import mpi_reduce, MPI_INT, MPI_SUM
	pass#IMPORTIMPORTIMPORT from EMAN2 import Processor
	pass#IMPORTIMPORTIMPORT from EMAN2 import Vec2f, Transform
	pass#IMPORTIMPORTIMPORT import os
	pass#IMPORTIMPORTIMPORT import sys
	pass#IMPORTIMPORTIMPORT import types

	maxit         = Tracker["maxit"]
	saturatecrit  = Tracker["saturatecrit"]
	pixercutoff   = Tracker["pixercutoff"]
	ou            = Tracker["radius"]
	ts            = sparx_utilities.get_input_from_string(Tracker["ts"])[0]
	delta         = sparx_utilities.get_input_from_string(Tracker["delta"])[0]
	sym           = Tracker["constants"]["sym"]
	sym           = sym[0].lower() + sym[1:]
	center        = Tracker["constants"]["center"]
	CTF           = Tracker["constants"]["CTF"]
	fourvar = False

	if log == None:
		pass#IMPORTIMPORTIMPORT from logger import Logger
		log = sparx_logger.Logger()


	if mpi_comm == None: mpi_comm = mpi.MPI_COMM_WORLD

	number_of_proc = mpi.mpi_comm_size(mpi_comm)
	myid           = mpi.mpi_comm_rank(mpi_comm)
	main_node = 0

	if myid == main_node:
		log.add("Start slocal_ali3d_base")

	"""Multiline Comment68"""
	if debug:
		outdir = "debug_outdir"
		if myid == main_node:  os.system("mkdir   "+outdir)
		pass#IMPORTIMPORTIMPORT from time import sleep
		while not os.path.exists(outdir):
			print("Node ",myid,"  waiting...")
			time.sleep(5)

		finfo = open(os.path.join(outdir, "progress%04d"%myid), 'w')
	else:
		finfo = None

	last_ring   = int(ou)
	center      = int(center)

	if( type(stack) is bytes ):
		if myid == main_node:
			if(sparx_utilities.file_type(stack) == "bdb"):
				pass#IMPORTIMPORTIMPORT from EMAN2db import db_open_dict
				dummy = EMAN2db.db_open_dict(stack, True)

			nima = EMAN2_cppwrap.EMUtil.get_image_count(stack)
			list_of_particles = list(range(nima))
			total_nima = len(list_of_particles)
		else:
			list_of_particles = None
			total_nima = 0
		total_nima = sparx_utilities.wrap_mpi_bcast(total_nima, main_node, mpi_comm)
		total_nima = int(total_nima[0])
		list_of_particles = sparx_utilities.wrap_mpi_bcast(list_of_particles, main_node, mpi_comm)
		if myid == main_node:
			particle_ids = [0]*total_nima
			for i in range(total_nima):  particle_ids[i] = list_of_particles[i]
		image_start, image_end = MPI_start_end(total_nima, number_of_proc, myid)
		# create a list of images for each node
		list_of_particles = list_of_particles[image_start: image_end]
		nima = len(list_of_particles)

	else:
		list_of_particles = list(range(len(stack)))
		nima = len(list_of_particles)
		total_nima = len(list_of_particles)
		total_nima = mpi.mpi_reduce(total_nima, 1, mpi.MPI_INT, mpi.MPI_SUM, 0, mpi.MPI_COMM_WORLD)
		total_nima = mpi.mpi_bcast(total_nima, 1, mpi.MPI_INT, 0, mpi.MPI_COMM_WORLD)
		total_nima = int(total_nima[0])
		image_start = 0
		image_end   = nima

	if(myid == main_node):
		if( type(stack) is bytes ):  dataim = sparx_utilities.get_im(stack, list_of_particles[0])
		else:                                   dataim = stack[list_of_particles[0]]
		nx      = dataim.get_xsize()
		if CTF :
			ctf_applied = dataim.get_attr_default('ctf_applied', 0)
			if ctf_applied >0 :  sparx_global_def.ERROR("Projection data cannot be CTF-applied","local_ali3d_base",1,myid)
	else:
		nx = 0

	nx  = sparx_utilities.bcast_number_to_all(nx, source_node = main_node)

	if last_ring < 0:	last_ring = int(nx/2) - 2
	mask2D  = sparx_utilities.model_circle(last_ring, nx, nx)

	dataim = [None]*nima
	for im in range(nima):
		if( type(stack) is bytes ):  dataim[im] = sparx_utilities.get_im(stack, list_of_particles[im])
		else:                                   dataim[im] = stack[list_of_particles[im]]
		dataim[im].set_attr('ID', list_of_particles[im])
		if CTF :
			st = EMAN2_cppwrap.Util.infomask(dataim[im], mask2D, False)
			dataim[im] -= st[0]


	if chunk <= 0.0:  chunk = 1.0
	n_of_chunks = int(1.0/chunk)

	"""Multiline Comment69"""

	pass#IMPORTIMPORTIMPORT import  types
	if Tracker["constants"]["mask3D"]:
		if type(Tracker["constants"]["mask3D"]) is bytes:
			if myid == main_node:
				mask3D = sparx_utilities.get_im(Tracker["constants"]["mask3D"])
			else:
				mask3D = sparx_utilities.model_blank(nx, nx, nx)
		else:
			mask3D = Tracker["constants"]["mask3D"].copy()
		if myid == main_node:
			nxm = mask3D.get_xsize()
			if( nxm > nx ):
				pass#IMPORTIMPORTIMPORT from fundamentals import rot_shift3D
				mask3D = EMAN2_cppwrap.Util.window(sparx_fundamentals.rot_shift3D(mask3D,scale=float(nx)/float(nxm)),nx,nx,nx)
				nxm = mask3D.get_xsize()
				assert(nx == nxm)
			else:
				mask3D = sparx_utilities.model_blank(nx, nx, nx)
		sparx_utilities.bcast_EMData_to_all(mask3D, myid, main_node)
	else:
		mask3D = sparx_utilities.model_circle(last_ring, nx, nx, nx)

	#  Read	template volume if provided
	if templatevol:
		if type(templatevol) is bytes:
			if myid == main_node:
				vol = sparx_utilities.get_im(templatevol)
				nxm = vol.get_xsize()
				if( nxm > nx ):
					pass#IMPORTIMPORTIMPORT from fundamentals import rot_shift3D
					vol = EMAN2_cppwrap.Util.window(sparx_fundamentals.rot_shift3D(vol,scale=float(nx)/float(nxm)),nx,nx,nx)
					nxm = vol.get_xsize()
					assert(nx == nxm)
			else:
				vol = sparx_utilities.model_blank(nx, nx, nx)
		else:
			if myid == main_node:
				nxm = templatevol.get_xsize()
				if( nxm > nx ):
					pass#IMPORTIMPORTIMPORT from fundamentals import rot_shift3D
					vol = EMAN2_cppwrap.Util.window(sparx_fundamentals.rot_shift3D(templatevol,scale=float(nx)/float(nxm)),nx,nx,nx)
					nxm = vol.get_xsize()
					assert(nx == nxm)
				else:
					vol = templatevol.copy()
			else:
				vol = sparx_utilities.model_blank(nx, nx, nx)
		sparx_utilities.bcast_EMData_to_all(vol, myid, main_node)
		del templatevol
		#  Do the 3D
		#vol = do_volume_mrk01(vol, Tracker, 0, mpi_comm)
		ref_data = [vol, Tracker, 0, mpi_comm]
		user_func = Tracker["constants"] ["user_func"]
		vol = user_func(ref_data)
	else:
		vol = None

	if debug:
		finfo.write( "image_start, image_end: %d %d\n" %(image_start, image_end) )
		finfo.flush()

	if debug:
		finfo.write("  chunk = "+str(chunk)+"   ")
		finfo.write("\n")
		finfo.flush()
		finfo.write("  Number of chunks = "+str(n_of_chunks)+"   ")
		finfo.write("\n")
		finfo.flush()

	del list_of_particles

	if debug:
		finfo.write("  First image on this processor: "+str(image_start)+"   ")
		finfo.write("  Last  image on this processor: "+str(image_end)+"   ")
		finfo.write("\n")
		finfo.flush()

	if myid == main_node:
		# initialize data for the reference preparation function
		ref_data = [ mask3D, max(center,0), None, None, None ]
		# for method -1, switch off centering in user function
		ref_data.append( None )

	pass#IMPORTIMPORTIMPORT from time import time
	if myid == main_node:
		log.add("Dimensions used (nx, last_ring)  %5d    %5d\n"%(nx, last_ring))
		start_time = time.time()


		
	M = nx
	npad = 2
	N = M*npad
	K = 6
	alpha = 1.75
	r = M/2
	v = K/2.0/N
	params = {"filter_type": EMAN2_cppwrap.Processor.fourier_filter_types.KAISER_SINH_INVERSE, "alpha":alpha, "K":K, "r":r, "v":v, "N":N}

	data = [None]*7
	data[3] = mask2D
	cs = [0.0]*3

	for iteration in range(maxit):
		if myid == main_node:
			start_time = time.time()
			log.add("ITERATION #%3d\n"%(iteration+1))
		if debug:
			finfo.write("  iteration = "+str(iteration)+"   ")
			finfo.write("\n")
			finfo.flush()
		pixer = [0.0]*nima
		for ic in range(n_of_chunks):
			# In the very first step the volume has to be computed if it was not provided by the user
			if( ((iteration > 0) and (ic > 0)) or vol == None):
				if(center == -1 and sym[0] == 'c'):
					if debug:
						finfo.write("  begin centering \n")
						finfo.flush()
					cs[0], cs[1], cs[2], dummy, dummy = sparx_utilities.estimate_3D_center_MPI(dataim, total_nima, myid, number_of_proc, main_node)
					cs = mpi.mpi_bcast(cs, 3, mpi.MPI_FLOAT, main_node, mpi_comm)
					cs = [-float(cs[0]), -float(cs[1]), -float(cs[2])]
					if int(sym[1]) > 1:
						cs[0] = cs[1] = 0.0
					sparx_utilities.rotate_3D_shift(dataim, cs)
					if myid == main_node:
						msg = "Average center x = %10.3f        Center y = %10.3f        Center z = %10.3f\n"%(cs[0], cs[1], cs[2])
						log.add(msg)
						log.add("Time to center = %d\n"%(time.time()-start_time))
						start_time = time.time()
					# compute updated 3D before each chunk
					# resolution
				if debug:
					finfo.write("  begin reconstruction = "+str(image_start))
					finfo.write("\n")
					finfo.flush()

				#  Do the 3D
				#vol = do_volume_mrk01(dataim, Tracker, iteration, mpi_comm)
				ref_data = [dataim, Tracker, iteration, mpi_comm]
				user_func = Tracker["constants"] ["user_func"]
				vol = user_func(ref_data)

				if myid == main_node:
					#drop_image(vol, os.path.join(outdir, "vol%03d_%03d.hdf"%(iteration, ic) ))
					log.add("3D reconstruction time = %d"%(time.time()-start_time))
					start_time = time.time()
				if debug:
					finfo.write("  done reconstruction = "+str(image_start))
					finfo.write("\n")
					finfo.flush()

				if fourvar:
				#  Compute Fourier variance
					varf = sparx_statistics.varf3d_MPI(dataim, ssnr_text_file = os.path.join(outdir, "ssnr%03d_%03d"%(iteration, ic)), mask2D = None, reference_structure = vol, ou = ou, rw = 1.0, npad = 1, CTF = CTF, sign = 1, sym =sym, myid = myid)
					if myid == main_node:
						varf = 1.0/varf
						sparx_utilities.print_msg("Time to calculate 3D Fourier variance= %d\n"%(time.time()-start_time))
						start_time = time.time()
				else:  varf = None

			if CTF:
				previous_defocus = -1.0
				#vol = fft(pad(vol, N, N, N))
			else:
				data[0], data[1] = sparx_projection.prep_vol(vol)

			image_start_in_chunk = ic*nima/n_of_chunks
			image_end_in_chunk   = (ic+1)*nima/n_of_chunks
			if debug:
				finfo.write("Chunk "+str(ic)+"   Number of images in this chunk: "+str(image_end_in_chunk-image_start_in_chunk)+"\n")
				finfo.write("First image in this chunk: "+str(image_start_in_chunk)+"   Last image in this chunk: "+str(image_end_in_chunk-1)+"\n")
				finfo.flush()
			for imn in range(image_start_in_chunk, image_end_in_chunk):
				if CTF:
					ctf_params = dataim[imn].get_attr( "ctf" )
					if ctf_params.defocus != previous_defocus:
						previous_defocus = ctf_params.defocus
						data[0], data[1] = sparx_projection.prep_vol(sparx_filter.filt_ctf(vol, ctf_params))

				data[2] = dataim[imn]
				if ts > 0.0:
					refi = dataim[imn].FourInterpol(nx*2, nx*2, 1, True)
					data[4] = EMAN2_cppwrap.Processor.EMFourierFilter(refi, params)

				#phi, theta, psi, tx, ty = get_params_proj(dataim[imn])
				t1 = dataim[imn].get_attr("xform.projection")
				dp = t1.get_params("spider")
				atparams = [dp["phi"], dp["theta"], dp["psi"]]
				data[5]  = [dp["tx"], dp["ty"]]
				if debug:
					# we have to distiguish between no shift situation, which is done through ccc, and shift, which is done using gridding in 2D
					if(ts == 0.0):  data[6] = 0.0
					else:           data[6] = -1.0#ts#-1.0
					initial, dummy = sparx_alignment.eqproj_cascaded_ccc(atparams, data)  # this is if we need initial discrepancy
					finfo.write("Image "+str(imn)+"\n")
					finfo.write('Old  %6.1f  %6.1f  %6.1f   %5.2f  %5.2f  %11.4e\n'%(atparams[0],atparams[1],atparams[2], -dummy[0], -dummy[1], initial))
				# change signs of shifts for projections
				data[6] = ts
				#from random import random
				#data[5] = [(random()-0.5)*2,(random()-0.5)*2]  #  HERE !!!!!!!!!!!

				weight_phi = max(delta, delta*abs((atparams[1]-90.0)/180.0*numpy.pi))
				optm_params = sparx_utilities.amoeba_multi_level(atparams, [weight_phi, delta, weight_phi], sparx_alignment.eqproj_cascaded_ccc, 1.0, 1.e-2, 500, data)
				optm_params[0].append(optm_params[3][0])
				optm_params[0].append(optm_params[3][1])
				optm_params[0][3] *= -1
				optm_params[0][4] *= -1

				if debug:
					finfo.write('New  %6.1f  %6.1f  %6.1f   %5.2f  %5.2f  %11.4e  %4d\n'%(optm_params[0][0], optm_params[0][1], optm_params[0][2], optm_params[0][3], optm_params[0][4], optm_params[1], optm_params[2]))
					finfo.flush()

				#from sys import exit
				#exit()
				t2 = EMAN2_cppwrap.Transform({"type":"spider","phi":optm_params[0][0],"theta":optm_params[0][1],"psi":optm_params[0][2]})
				t2.set_trans(EMAN2_cppwrap.Vec2f(-optm_params[0][3], -optm_params[0][4]))
				dataim[imn].set_attr("xform.projection", t2)
				pass#IMPORTIMPORTIMPORT from pixel_error import max_3D_pixel_error
				pixer[imn] = sparx_pixel_error.max_3D_pixel_error(t1, t2, last_ring)
				#set_params_proj(dataim[imn], optm_params[0])
				#if( myid == main_node and imn%4 == 0):
				#	log.add( "Time to process %6d particles : %d\n" % (imn, time()-start_time) )
				#	start_time = time()
			if( myid == main_node ):
				log.add( "Time to process %6d particles : %d" % (image_end_in_chunk-image_start_in_chunk, time.time()-start_time) )
				start_time = time.time()

			# release memory
			data[0] = None


		#output pixel errors after all headers were processed
		pass#IMPORTIMPORTIMPORT from mpi import mpi_gatherv
		pixer = sparx_utilities.wrap_mpi_gatherv(pixer, main_node, mpi_comm)
		mpi.mpi_barrier(mpi_comm)
		terminate = 0
		if(myid == main_node):
			pixer = list(map(float, pixer))
			pass#IMPORTIMPORTIMPORT from statistics import hist_list
			lhist = 20
			region, histo = sparx_statistics.hist_list(pixer, lhist)
			log.add(" ")
			log.add("=========== Histogram of pixel errors ==============")
			for lhx in range(lhist):
				msg = "          %10.3f     %7d"%(region[lhx], histo[lhx])
				log.add(msg)
			log.add("____________________________________________________\n")


			# Terminate if saturatecrit% within pixercutoff pixel error
			im = 0
			for lhx in range(lhist):
				if(region[lhx] > pixercutoff): break
				im += histo[lhx]
			lhx = im/float(total_nima)
			if( lhx > saturatecrit):
				if( iteration == 1 ):
					log.add("First iteration, will continue even though %4.2f images did not find better orientations"%saturatecrit)
				else:
					terminate = 1
					log.add("..............................................................")
					log.add(">>>>>>>>>>>>>>>   Will terminate as %4.2f images did not find better orientations"%saturatecrit)
			del region, histo
		del pixer
		terminate = mpi.mpi_bcast(terminate, 1, mpi.MPI_INT, 0, mpi_comm)
		terminate = int(terminate[0])
		if terminate:  break

	del vol
	# gather parameters
	params = []
	for im in dataim:
		t = sparx_utilities.get_params_proj(im)
		params.append( [t[0], t[1], t[2], t[3], t[4]] )
	params = sparx_utilities.wrap_mpi_gatherv(params, main_node, mpi_comm)

	if( myid == main_node ):
		"""Multiline Comment70"""
		log.add("Finish local_ ali3d_base")

	return  params

### from sxsort3d
			
def mref_ali3d_EQ_Kmeans_circular(ref_list, outdir, particle_list_file, Tracker):
	pass#IMPORTIMPORTIMPORT from utilities      import model_circle, reduce_EMData_to_root, bcast_EMData_to_all, bcast_number_to_all, drop_image
	pass#IMPORTIMPORTIMPORT from utilities      import  bcast_list_to_all, get_image, get_input_from_string, get_im
	pass#IMPORTIMPORTIMPORT from utilities      import get_arb_params, set_arb_params, drop_spider_doc, send_attr_dict
	pass#IMPORTIMPORTIMPORT from utilities      import get_params_proj, set_params_proj, model_blank, wrap_mpi_bcast, write_text_file
	pass#IMPORTIMPORTIMPORT from filter         import filt_params, filt_btwl, filt_ctf, filt_table, fit_tanh, filt_tanl
	pass#IMPORTIMPORTIMPORT from utilities      import rotate_3D_shift,estimate_3D_center_MPI, get_shrink_data_huang, get_im
	####-------
	pass#IMPORTIMPORTIMPORT from utilities      import get_attr_stack, get_sorting_attr_stack, get_sorting_params, get_sorting_params_refine
	pass#IMPORTIMPORTIMPORT from utilities      import parsing_sorting_params, fill_in_mpi_list, wrap_mpi_bcast, get_groups_from_partition
	pass#IMPORTIMPORTIMPORT from utilities      import remove_small_groups, set_filter_parameters_from_adjusted_fsc
	###-----------
	pass#IMPORTIMPORTIMPORT from alignment      import Numrinit, prepare_refrings, proj_ali_incore
	pass#IMPORTIMPORTIMPORT from random         import randint, random
	pass#IMPORTIMPORTIMPORT from filter         import filt_ctf
	pass#IMPORTIMPORTIMPORT from utilities      import print_begin_msg, print_end_msg, print_msg, read_text_file
	pass#IMPORTIMPORTIMPORT from projection     import prep_vol, prgs, prgl, project, prgq, gen_rings_ctf
	pass#IMPORTIMPORTIMPORT from morphology     import binarize, get_shrink_3dmask
	pass#IMPORTIMPORTIMPORT from statistics		import fsc

	pass#IMPORTIMPORTIMPORT import os
	pass#IMPORTIMPORTIMPORT import types
	pass#IMPORTIMPORTIMPORT from mpi            import mpi_bcast, mpi_comm_size, mpi_comm_rank, MPI_FLOAT, MPI_COMM_WORLD, mpi_barrier
	pass#IMPORTIMPORTIMPORT from mpi            import mpi_reduce, mpi_gatherv, mpi_scatterv, MPI_INT, MPI_SUM
	pass#IMPORTIMPORTIMPORT from applications   import MPI_start_end
	pass#IMPORTIMPORTIMPORT from reconstruction import rec3D_two_chunks_MPI, rec3D_MPI_noCTF
	pass#IMPORTIMPORTIMPORT from fundamentals   import fftip, rops_table, fft
	mpi_comm = mpi.MPI_COMM_WORLD
	#####  reconstruction parameters, no need to change.
	fourvar   = False
	snr       = 1.0
	debug     = False
	ref_a     = "S"
	npad      = 2
	#####################################################
	pass#IMPORTIMPORTIMPORT from logger import Logger,BaseLogger_Files
	log       = sparx_logger.Logger()
	log   =sparx_logger.Logger(sparx_logger.BaseLogger_Files())
	log.prefix=outdir+"/"
	myid           = Tracker["constants"]["myid"]
	main_node      = Tracker["constants"]["main_node"]
	number_of_proc = Tracker["constants"]["nproc"]
	shrinkage      = Tracker["shrinkage"]
	### input parameters
	maxit          = Tracker["constants"]["maxit"]
	ou             = Tracker["constants"]["radius"]
	ir             = Tracker["constants"]["ir"]
	rs             = Tracker["constants"]["rs"]
	xr             = Tracker["constants"]["xr"]
	yr             = Tracker["constants"]["yr"]
	ts             = Tracker["constants"]["ts"]
	delta          = Tracker["constants"]["delta"]
	an             = Tracker["constants"]["an"]
	center         = Tracker["constants"]["center"]
	nassign        = Tracker["constants"]["nassign"]
	nrefine        = Tracker["constants"]["nrefine"]
	CTF            = Tracker["constants"]["CTF"]
	sym            = Tracker["constants"]["sym"]
	termprec       = Tracker["constants"]["stoprnct"]
	if myid == main_node:
		weight         = Tracker["constants"]["weight"]
		m = len(weight)-1
		#s= sum(weight)
	else:
		weight         = 0.0
		m              = 0
	weight = sparx_utilities.wrap_mpi_bcast(weight, main_node, mpi_comm)
	if myid ==main_node:
		print(weight)
	m = sparx_utilities.bcast_number_to_all(m, main_node)
	if Tracker["constants"]["mask3D"]:
		maskfile            = Tracker["mask3D"]
	else:
		maskfile            = None
	if Tracker["constants"]["focus3Dmask"]: focus = Tracker["constants"]["focus3Dmask"]
	else:                                   focus = None
	partstack           = Tracker["constants"]["partstack"]
	user_func_name      = Tracker["constants"]["user_func"]
	Tracker["lowpass"]  = Tracker["low_pass_filter"]
	Tracker["falloff"]  = .1
	if Tracker["constants"]["PWadjustment"]: Tracker["PWadjustment"] = Tracker["PW_dict"][Tracker["constants"]["nxinit"]]
	else:                                    Tracker["PWadjustment"] = None	
	####################################################
	pass#IMPORTIMPORTIMPORT from time import sleep
	#Tracker["applyctf"] = True # 
	data, old_shifts = sparx_utilities.get_shrink_data_huang(Tracker, Tracker["nxinit"], particle_list_file, partstack, myid,main_node,number_of_proc,preshift=True)
	if myid ==main_node:
		list_of_particles = sparx_utilities.read_text_file(particle_list_file)
		total_nima        =len(list_of_particles)
	else:
		total_nima = 0
		list_of_particles = 0
	total_nima            = sparx_utilities.bcast_number_to_all(total_nima,main_node)
	list_of_particles     = sparx_utilities.wrap_mpi_bcast(list_of_particles, main_node)
	
	if myid == main_node:
		if os.path.exists(outdir):  nx = 1
		else:  nx = 0
	else:  nx = 0
	ny = sparx_utilities.bcast_number_to_all(nx, source_node = main_node)
	#image_start, image_end = MPI_start_end(total_data,number_of_proc, myid)
	
	if ny == 1:  sparx_global_def.ERROR('Output directory exists, please change the name and restart the program', "mref_ali3d_iter", 1,myid)
	mpi.mpi_barrier(mpi.MPI_COMM_WORLD)
	if myid == main_node:	
		os.mkdir(outdir)
		pass#IMPORTIMPORTIMPORT import global_def
		sparx_global_def.LOGFILE =  os.path.join(outdir, sparx_global_def.LOGFILE)
		log.add("Equal K-means  ")
	mpi.mpi_barrier(mpi.MPI_COMM_WORLD)
	pass#IMPORTIMPORTIMPORT from time import time
	if debug:
		pass#IMPORTIMPORTIMPORT from time import sleep
		while not os.path.exists(outdir):
			print("Node ",myid,"  waiting...")
			time.sleep(5)
		finfo = open(os.path.join(outdir, "progress%04d"%myid), 'w')
		frec  = open( os.path.join(outdir, "recons%04d"%myid), "w" )
	else:
		finfo = None
		frec  = None
	xrng        = sparx_utilities.get_input_from_string(xr)
	if  yr == "-1":  yrng = xrng
	else          :  yrng = sparx_utilities.get_input_from_string(yr)
	step        = sparx_utilities.get_input_from_string(ts)
	delta       = sparx_utilities.get_input_from_string(delta)
	lstp = min( len(xrng), len(yrng), len(step), len(delta) )
	if (an == "-1"):
		an = []
		for i in range(len(xrng)):   an.append(-1)
	else:
		pass#IMPORTIMPORTIMPORT from  alignment	    import proj_ali_incore_local
		an      = sparx_utilities.get_input_from_string(an)

	first_ring  = int(ir)
	rstep       = int(rs)
	last_ring   = int(int(ou)*shrinkage+.5)
	center      = int(center)
	numref  = len(ref_list)
	nx      = ref_list[0].get_xsize()
	if last_ring < 0:last_ring = nx//2 - 2
	pass#IMPORTIMPORTIMPORT import user_functions
	user_func = sparx_user_functions.factory[user_func_name]
	if (myid == main_node):
		#import user_functions
		#user_func = sparx_user_functions.factory[user_func_name]
		log.add("mref_ali3d_MPI")
		log.add("Input stack                               : %s"%(Tracker["constants"]["stack"]))
		#log.add("Reference volumes                         : %s"%(ref_vol))	
		log.add("Number of reference volumes               : %i"%(numref))
		log.add("Output directory                          : %s"%(outdir))
		log.add("User function                             : %s"%(user_func_name))
		if(focus != None):  \
		log.add("Maskfile 3D for focused clustering        : %s"%(focus))
		log.add("Overall 3D mask applied in user function  : %s"%(maskfile))
		log.add("Inner radius                              : %i"%(first_ring))
		log.add("Outer radius                              : %i"%(last_ring))
		log.add("Ring step                                 : %i"%(rstep))
		log.add("X search range                            : %s"%(xrng))
		log.add("Y search range                            : %s"%(yrng))
		log.add("Translational step                        : %s"%(step))
		log.add("Angular step                              : %s"%(delta))
		log.add("Angular search range                      : %s"%(an))
		log.add("Number of assignments in each iteration   : %i"%(nassign))
		log.add("Number of alignments in each iteration    : %i"%(nrefine))
		log.add("Number of iterations                      : %i"%(lstp*maxit) )
		log.add("Center type                               : %i"%(center))
		log.add("CTF correction                            : %s"%(CTF))
		log.add("Symmetry group                            : %s"%(sym))
		log.add("Percentage of change for termination      : %f"%(termprec))
		log.add("User function                             : %s"%(user_func_name))
		log.add("total number of particles                 : %d"%len(Tracker["this_data_list"]))
		log.add("shrinkage is                              : %f"%shrinkage)
		log.add("the particle id files for get_shrink_dat is %s"%particle_list_file) 
	if(maskfile):
		if(type(maskfile) is bytes): mask3D = sparx_morphology.get_shrink_3dmask(Tracker["nxinit"],maskfile) 
		else: 	                                mask3D = maskfile
	else        :  mask3D = sparx_utilities.model_circle(last_ring, nx, nx, nx)
	numr     = sparx_alignment.Numrinit(first_ring, last_ring, rstep, "F")
	mask2D   = sparx_utilities.model_circle(last_ring, nx, nx)
	if(first_ring > 1):  mask2D -= sparx_utilities.model_circle(first_ring,nx,nx)
	nima       = len(data) # local number of particles
	
	"""Multiline Comment85"""
	if debug:
		finfo.write( "Image_start, image_end: %d %d\n" %(image_start,image_end))
		finfo.flush()
	start_time = time.time()
	#  Here the assumption is that input are always volumes.  It should be most likely be changed so optionally these are group assignments.
	#  Initialize Particle ID and set group number to non-existant -1
	for im in range(nima):
		data[im].set_attr_dict({'group':-1})
	if(myid == 0):
		log.add( "Time to read data: %d" % (time.time()-start_time) );start_time = time.time()

	if myid == main_node:
		refdata = [None]*7
		for  iref in range(numref):
			ref_list[iref].write_image(os.path.join(outdir, "vol0000.hdf"), iref)
		#refdata[0] = 
		#user_func(refdata)
	refdata        =[None]*4
	res = 0.5
	for i in range(len(Tracker["global_fsc"][0])-1,0,-1):
		if Tracker["global_fsc"][1][i] > 0.5:
				res=fsc_in[0][i]
				break
	highres = []
	for  iref in range(numref): highres.append(int(res*Tracker["nxinit"] + 0.5))
	Tracker["lowpass"] = min(0.45, res-0.05)
	Tracker["lowpass"] = max(0.11, res)
	Tracker["falloff"] = 0.1
	##-----------------------------------------------
	if myid == main_node:  ### 3-D mask, low pass filter, and power spectrum adjustment
		for iref in range(numref):
		
			#set_filter_parameters_from_adjusted_fsc(Tracker["constants"]["total_stack"],Tracker["number_of_ref_class"][iref],Tracker)
			log.add("%d  low pass filter   %f %f  %d"%(iref,Tracker["lowpass"],Tracker["falloff"],Tracker["number_of_ref_class"][iref]))
			log.add("%d highres                   %f"%(iref, highres[iref]))
			
			if Tracker["mask3D"]:
				mask3D = sparx_utilities.get_im(Tracker["mask3D"])
				stat = EMAN2_cppwrap.Util.infomask(ref_list[iref], mask3D, False)
				ref_list[iref] -= stat[0]
				EMAN2_cppwrap.Util.mul_scalar(ref_list[iref], 1.0/stat[1])
				
			if(Tracker["constants"]["PWadjustment"] !=""):
				rt = sparx_utilities.read_text_file(Tracker["PW_dict"][Tracker["constants"]["nxinit"]])
				ro = sparx_fundamentals.rops_table(ref_list[iref])
				for i in range(1,len(ro)):  ro[i] = (rt[i]/ro[i])**Tracker["constants"]["upscale"]
				ref_list[iref] =sparx_filter.filt_table(ref_list[iref],ro)
				
			if (Tracker["constants"]["low_pass_filter"]==-1.):  ref_list[iref] = sparx_filter.filt_tanl(ref_list[iref], Tracker["lowpass"], Tracker["falloff"])                                       # low pass from resolution 
			else:                                               ref_list[iref] = sparx_filter.filt_tanl(ref_list[iref], min(Tracker["constants"]["low_pass_filter"]/Tracker["shrinkage"],0.45), Tracker["falloff"]) # user define filter
							

			if Tracker["mask3D"]: EMAN2_cppwrap.Util.mul_img(ref_list[iref], mask3D)
			ref_list[iref].write_image(os.path.join(outdir,"volf0000.hdf"),iref)
			
	mpi.mpi_barrier( mpi.MPI_COMM_WORLD )
	if CTF:
		#if(data[0].get_attr_default("ctf_applied",0) > 0):  ERROR("mref_ali3d_MPI does not work for CTF-applied data", "mref_ali3d_MPI", 1, myid)
		pass#IMPORTIMPORTIMPORT from reconstruction import rec3D_MPI
	else:
		pass#IMPORTIMPORTIMPORT from reconstruction import rec3D_MPI_noCTF
	if debug:
		finfo.write( '%d loaded  \n' % len(data) )
		finfo.flush()
	#  this is needed for gathering of pixel errors
	disps = []
	recvcount = []
	for im in range(number_of_proc):
		if( im == main_node ):  disps.append(0)
		else:                   disps.append(disps[im-1] + recvcount[im-1])
		ib, ie = MPI_start_end(total_nima, number_of_proc, im)
		recvcount.append( ie - ib )
	total_iter = 0
	tr_dummy = EMAN2_cppwrap.Transform({"type":"spider"})

	if(focus != None):
		if(myid == main_node):
			focus = sparx_morphology.get_shrink_3dmask(Tracker["nxinit"], focus)
		else:
			focus =  sparx_utilities.model_blank(Tracker["nxinit"], Tracker["nxinit"], Tracker["nxinit"])
		sparx_utilities.bcast_EMData_to_all(focus, myid, main_node)
		st = EMAN2_cppwrap.Util.infomask(focus, None, True)
		if( st[0] == 0.0 ):  sparx_global_def.ERROR("sxrsort3d","incorrect focused mask, after binarize all values zero",1, myid)
		focus = sparx_projection.prep_vol(focus, 1, 1)

	Niter = int(lstp*maxit*(nassign + nrefine) )
	for Iter in range(Niter):
		N_step = (Iter%(lstp*(nassign+nrefine)))/(nassign+nrefine)
		if Iter%(nassign+nrefine) < nassign:
			runtype = "ASSIGNMENT"
		else:
			runtype = "REFINEMENT"

		total_iter += 1
		if(myid == main_node):
			log.add("\n%s ITERATION #%3d,  inner iteration #%3d\nDelta = %4.1f, an = %5.2f, xrange = %5.2f, yrange = %5.2f, step = %5.2f  \
			"%(runtype, total_iter, Iter, delta[N_step], an[N_step], xrng[N_step],yrng[N_step],step[N_step]))
			start_ime = time.time()
		peaks =  [ [ -1.0e23 for im in range(nima) ] for iref in range(numref) ]
		if runtype=="REFINEMENT":
			trans = [ [ tr_dummy for im in range(nima) ] for iref in range(numref) ]
			pixer = [ [  0.0     for im in range(nima) ] for iref in range(numref) ]
			if(an[N_step] > 0):
				pass#IMPORTIMPORTIMPORT from utilities    import even_angles
				ref_angles = sparx_utilities.even_angles(delta[N_step], symmetry=sym, method = ref_a, phiEqpsi = "Zero")
				# generate list of angles
				pass#IMPORTIMPORTIMPORT from alignment import generate_list_of_reference_angles_for_search
				list_of_reference_angles = \
				sparx_alignment.generate_list_of_reference_angles_for_search(ref_angles, sym=sym)
				del ref_angles
			else:  list_of_reference_angles = [[1.0,1.0]]
		cs = [0.0]*3
		pass#IMPORTIMPORTIMPORT from fundamentals import fft
		if( not focus ):
			for im in range(nima):  data[im] = sparx_fundamentals.fft(data[im])

		for iref in range(numref):
			if(myid == main_node):	volft = sparx_utilities.get_im(os.path.join(outdir, "volf%04d.hdf"%(total_iter-1)), iref)
			else: 					volft =  sparx_utilities.model_blank(nx, nx, nx)
			sparx_utilities.bcast_EMData_to_all(volft, myid, main_node)
			volft = sparx_projection.prep_vol(volft, 1, 1)
			if CTF:
				previous_defocus=-1.0
				if runtype=="REFINEMENT":
					start_time = time.time()
					prjref = sparx_projection.prgq( volft, kb, nx, delta[N_step], ref_a, sym, MPI=True)
					if(myid == 0): log.add( "Calculation of projections: %d" % (time.time()-start_time) );start_time = time.time()
					del volft, kb
			else:
				if runtype=="REFINEMENT":
					start_time = time.time()
					refrings = sparx_alignment.prepare_refrings( volft, kb, nx, delta[N_step], ref_a, sym, numr)
					if(myid == 0): log.add( "Initial time to prepare rings: %d" % (time.time()-start_time) );start_time = time.time()
					del volft, kb
			start_time = time.time()
			for im in range(nima):
				if(CTF):
					ctf = data[im].get_attr("ctf")
					if runtype=="REFINEMENT":
						if(ctf.defocus != previous_defocus):
							previous_defocus = ctf.defocus
							rstart_time = time.time()
							refrings = sparx_projection.gen_rings_ctf( prjref, nx, ctf, numr)
							if(myid == 0): log.add( "Repeated time to prepare rings: %d" % (time.time()-rstart_time) );rstart_time = time.time()

				if runtype=="ASSIGNMENT":
					phi,tht,psi,s2x,s2y = sparx_utilities.get_params_proj(data[im])
					"""Multiline Comment86"""
					#  Standard distance
					#  Ref is in reciprocal space
					if CTF:
						ref = sparx_filter.filt_ctf( sparx_projection.prgl( volft, [phi,tht,psi,-s2x,-s2y], 1, False), ctf )
					else:
						ref = sparx_projection.prgl( volft, [phi,tht,psi,-s2x,-s2y], 1, False)
					pass#IMPORTIMPORTIMPORT from filter import filt_tophatl
					pass#IMPORTIMPORTIMPORT from math import sqrt
					ref = sparx_filter.filt_tophatl(ref, float(highres[iref])/(ref.get_ysize()))
					ref.set_attr("is_complex",0)
					ref.set_value_at(0,0,0.0)
					nrmref = numpy.sqrt(EMAN2_cppwrap.Util.innerproduct(ref, ref, None))
					if(focus):
						mask2D = sparx_morphology.binarize( sparx_projection.prgl( focus, [phi,tht,psi,-s2x,-s2y]), 1)
						tempx = sparx_fundamentals.fft(data[im]*mask2D)
						tempx.set_attr("is_complex",0)
						peak = EMAN2_cppwrap.Util.innerproduct(ref, tempx, None)
					else:
						data[im].set_attr("is_complex",0)
						peak = EMAN2_cppwrap.Util.innerproduct(ref, data[im], None)
						data[im].set_attr("is_complex",1)
					peak /= nrmref

					"""Multiline Comment87"""



					if not(finfo is None):
						finfo.write( "ID, iref, peak: %6d %d %8.5f\n" % (list_of_particles[im],iref,peak) )
				else:
					if(an[N_step] == -1):
						peak, pixel_error = sparx_alignment.proj_ali_incore(data[im], refrings, numr, xrng[N_step], yrng[N_step], step[N_step])
					else:
						peak, pixel_error = sparx_alignment.proj_ali_incore_local(data[im], refrings, list_of_reference_angles, numr, \
																	xrng[N_step], yrng[N_step], step[N_step], an[N_step],sym=sym)
					if not(finfo is None):
						phi,tht,psi,s2x,s2y = sparx_utilities.get_params_proj(data[im])
						finfo.write( "ID, iref, peak,t rans: %6d %d %f %f %f %f %f %f\n"%(list_of_particles[im],iref,peak,phi,tht,psi,s2x,s2y) )
						finfo.flush()

				peaks[iref][im] = peak
				if runtype=="REFINEMENT":
					pixer[iref][im] = pixel_error
					trans[iref][im] = data[im].get_attr( "xform.projection" )
			if(myid == 0):log.add( "Time to process particles for reference %3d: %d" % (iref, time.time()-start_time) );start_time = time.time()
		#newpeaks = [nima*[0.0]]*numref
		### add circular conditions
		# m = len(weight)//2
		#for im in xrange(nima):
		#	for iref in xrange(numref):
		#		for j in xrange(-m, m):
		#			newpeaks[iref][im] += peaks[(iref+j)%numref][im]*weight[j+m]
		#peaks = newpeaks	
		if runtype=="ASSIGNMENT":  del volft, ref #kb, ref
		else:
			if CTF: del prjref
			del refrings
			if(an[N_step] > 0): del list_of_reference_angles
		#  send peak values to the main node, do the assignments, and bring them back
		pass#IMPORTIMPORTIMPORT from numpy import float32, empty, inner, abs
		if( myid == 0 ):
			dtot = numpy.empty( (numref, total_nima), dtype = numpy.float32)
		for  iref in range(numref):
			recvbuf = mpi.mpi_gatherv(peaks[iref], nima, mpi.MPI_FLOAT, recvcount, disps, mpi.MPI_FLOAT, main_node, mpi.MPI_COMM_WORLD)
			if( myid == 0 ): dtot[iref] = recvbuf
		del recvbuf
		#  The while loop over even angles delta should start here.
		#  prepare reference directions
		pass#IMPORTIMPORTIMPORT from utilities import even_angles, getvec
		refa = sparx_utilities.even_angles(60.0)
		numrefang = len(refa)
		refanorm = numpy.empty( (numrefang, 3), dtype = numpy.float32)
		for i in range(numrefang):
			tmp = sparx_utilities.getvec(refa[i][0], refa[i][1])
			for j in range(3):
				refanorm[i][j] = tmp[j]
		del  refa, tmp
		transv = numpy.empty( (nima, 3), dtype = numpy.float32)
		if runtype=="ASSIGNMENT":
			for im in range(nima):
				trns = data[im].get_attr("xform.projection")
				for j in range(3):
					transv[im][j] = trns.at(2,j)
		else:
			# For REFINEMENT we have a problem, as the exact angle is known only after the next step of assigning projections.
			# So, we will assume it is the one with max peak
			for im in range(nima):
				qt = -1.0e23
				it = -1
				for iref in range(numref):
					pt = peaks[iref][im]
					if(pt > qt):
						qt = pt
						it = iref
				for j in range(3):
					transv[im][j] = trans[it][im].at(2,j)
		#  We have all vectors, now create a list of assignments of images to references
		refassign = [-1]*nima
		for im in range(nima):
			refassign[im] = abs(numpy.inner(refanorm,transv[im])).argmax()
		assigntorefa = mpi.mpi_gatherv(refassign, nima, mpi.MPI_INT, recvcount, disps, mpi.MPI_INT, main_node, mpi.MPI_COMM_WORLD)
		assigntorefa = list(map(int, assigntorefa))
		del refassign, refanorm, transv
		"""Multiline Comment88"""
		if myid == main_node:
			SA = False
			asi = [[] for iref in range(numref)]
			report_error = 0
			for imrefa in range(numrefang):
				pass#IMPORTIMPORTIMPORT from utilities import findall
				N = sparx_utilities.findall(imrefa, assigntorefa)
				current_nima = len(N)
				if( current_nima >= numref and report_error == 0):
					tasi = [[] for iref in range(numref)]
					maxasi = current_nima//numref
					nt = current_nima
					kt = numref
					K = list(range(numref))

					d = numpy.empty( (numref, current_nima), dtype = numpy.float32)
					for ima in range(current_nima):
						for iref in range(numref):  d[iref][ima] = dtot[iref][N[ima]]

					while nt > 0 and kt > 0:
						l = d.argmax()
						group = l//current_nima
						ima   = l-current_nima*group
						if SA:
							J = [0.0]*numref
							sJ = 0
							Jc = [0.0]*numref
							for iref in range(numref):
								J[iref] = numpy.exp(d[iref][ima]/T)
								sJ += J[iref]
							for iref in range(numref):
								J[iref] /= sJ
							Jc[0] = J[0]
							for iref in range(1, numref):
								Jc[iref] = Jc[iref-1]+J[iref]
							sss = numpy.random.random()
							for group in range(numref):
								if( sss <= Jc[group]): break
						tasi[group].append(N[ima])
						N[ima] = -1
						for iref in range(numref):  d[iref][ima] = -1.e10
						nt -= 1
						masi = len(tasi[group])
						if masi == maxasi:
							for im in range(current_nima):  d[group][im] = -1.e10
							kt -= 1
					else:
						for ima in range(current_nima):
							if N[ima] > -1:
								qm = -1.e10
								for iref in range(numref):
									qt = dtot[iref][N[ima]]
									if( qt > qm ):
										qm = qt
										group = iref
								tasi[group].append(N[ima])
					del d, N, K
					if  SA:  del J, Jc
					for iref in range(numref):
						asi[iref] += tasi[iref]
					del tasi
				else:
					report_error = 1
			#  This should be deleted only once we know that the number of images is sufficiently large, see below.
			del dtot
		else:
			assignment = []
			report_error = 0
		report_error = sparx_utilities.bcast_number_to_all(report_error, source_node = main_node)
		if report_error == 1:  sparx_global_def.ERROR('Number of images within a group too small', "mref_ali3d_MPI", 1, myid)
		if myid == main_node:
			assignment = [0]*total_nima
			for iref in range(numref):
				for im in range(len(asi[iref])):
					assignment[asi[iref][im]] = iref
			del asi
		
		"""Multiline Comment89"""

		"""Multiline Comment90"""
		assignment = mpi.mpi_scatterv(assignment, recvcount, disps, mpi.MPI_INT, recvcount[myid], mpi.MPI_INT, main_node, mpi.MPI_COMM_WORLD)
		assignment = list(map(int, assignment))
		#  compute number of particles that changed assignment and how many are in which group
		nchng = 0
		npergroup = [0]*numref
		for im in range(nima):
			iref = data[im].get_attr('group')
			npergroup[assignment[im]] += 1
			if( iref != assignment[im]): nchng += 1
			data[im].set_attr('group', assignment[im])
		nchng = mpi.mpi_reduce(nchng, 1, mpi.MPI_INT, mpi.MPI_SUM, 0, mpi.MPI_COMM_WORLD)
		npergroup = mpi.mpi_reduce(npergroup, numref, mpi.MPI_INT, mpi.MPI_SUM, 0, mpi.MPI_COMM_WORLD)
		npergroup = list(map(int, npergroup))
		terminate = 0
		if( myid == 0 ):
			ngroup=[]
			nchng = int(nchng[0])
			precn = 100*float(nchng)/float(total_nima)
			msg = " Number of particles that changed assignments %7d, percentage of total: %5.1f"%(nchng, precn)
			log.add(msg)
			msg = " Group       number of particles"
			log.add(msg)
			for iref in range(numref):
				msg = " %5d       %7d"%(iref+1, npergroup[iref])
				log.add(msg)
				ngroup.append(int(npergroup[iref]))
			if(precn <= termprec):  terminate = 1
		else:
			ngroup = 0
		terminate = mpi.mpi_bcast(terminate, 1, mpi.MPI_INT, 0, mpi.MPI_COMM_WORLD)
		terminate = int(terminate[0])
		ngroup = sparx_utilities.wrap_mpi_bcast(ngroup,main_node)

		if runtype=="REFINEMENT":
			for im in range(nima):
				data[im].set_attr('xform.projection', trans[assignment[im]][im])
				pixer[0][im] = pixer[assignment[im]][im]
			pixer = pixer[0]
			if(center == -1):
				cs[0], cs[1], cs[2], dummy, dummy = sparx_utilities.estimate_3D_center_MPI(data, total_nima, myid, number_of_proc, main_node)				
				if myid == main_node:
					msg = " Average center x = %10.3f        Center y = %10.3f        Center z = %10.3f"%(cs[0], cs[1], cs[2])
					log.add(msg)
				cs = mpi.mpi_bcast(cs, 3, mpi.MPI_FLOAT, main_node, mpi.MPI_COMM_WORLD)
				cs = [-float(cs[0]), -float(cs[1]), -float(cs[2])]
				sparx_utilities.rotate_3D_shift(data, cs)
			#output pixel errors
			recvbuf = mpi.mpi_gatherv(pixer, nima, mpi.MPI_FLOAT, recvcount, disps, mpi.MPI_FLOAT, main_node, mpi.MPI_COMM_WORLD)
			mpi.mpi_barrier(mpi.MPI_COMM_WORLD)
			if(myid == main_node):
				recvbuf = list(map(float, recvbuf))
				pass#IMPORTIMPORTIMPORT from statistics import hist_list
				lhist = 20
				region, histo = sparx_statistics.hist_list(recvbuf, lhist)
				if(region[0] < 0.0):  region[0] = 0.0
				msg = "      Histogram of pixel errors\n      ERROR       number of particles"
				log.add(msg)
				for lhx in range(lhist):
					msg = " %10.3f      %7d"%(region[lhx], histo[lhx])
					log.add(msg)
				del region, histo
			del recvbuf
		fscc = [None]*numref
		if fourvar and runtype=="REFINEMENT":
			sumvol = sparx_utilities.model_blank(nx, nx, nx)
		start_time = time.time()

		if( not focus ):
			for im in range(nima):  data[im] = sparx_fundamentals.fft(data[im])
		highres     = []
		lowpass_tmp = []
		tmpref      = []
		for iref in range(numref):
			#  3D stuff
			pass#IMPORTIMPORTIMPORT from time import localtime, strftime
			if(CTF): volref, fscc[iref] = sparx_reconstruction.rec3D_two_chunks_MPI(data, snr, sym, mask3D,\
			 	os.path.join(outdir, "resolution_%02d_%04d"%(iref, total_iter)), myid, main_node, index=iref, npad=npad, finfo=frec)
			else:    volref, fscc[iref] = sparx_reconstruction.rec3D_MPI_noCTF(data, sym,mask3D,\
			 	os.path.join(outdir, "resolution_%02d_%04d"%(iref, total_iter)), myid, main_node, index=iref, npad=npad, finfo=frec)
			if(myid == 0):
				log.add( "Time to compute 3D: %d" % (time.time()-start_time) );start_time = time.time()
				volref.write_image(os.path.join(outdir, "vol%04d.hdf"%( total_iter)), iref)
				tmpref.append(volref)			
				if fourvar and runtype=="REFINEMENT": sumvol += volref
			#set_filter_parameters_from_adjusted_fsc(Tracker["constants"]["total_stack"],ngroup[iref],Tracker)
			if myid ==main_node:
				res = 0.5
				for ifreq in range(len(fscc[iref][0])-1,0,-1):
					if fscc[iref][1][ifreq] > 0.5 : # always use .5 as cutoff
						res = fscc[iref][0][ifreq]
						break

				Tracker["lowpass"] = min(0.45, res)
				#Tracker["lowpass"] = max(0.11, res)
				Tracker["falloff"] = 0.1
				log.add(" low pass filter is %f    %f   %d"%(Tracker["lowpass"], Tracker["falloff"], ngroup[iref]))
			else:
				Tracker["lowpass"] = 0.0
				Tracker["lowpass"] = 0.0
				res = 0.0
			res = sparx_utilities.bcast_number_to_all(res, main_node)
			highres.append(int(res*Tracker["nxinit"]+ 0.5))
			if myid ==main_node:
				log.add("%d highres                   %f"%(iref, highres[iref]))
			Tracker["lowpass"] = sparx_utilities.bcast_number_to_all(Tracker["lowpass"], main_node)
			Tracker["falloff"] = sparx_utilities.bcast_number_to_all(Tracker["falloff"], main_node)
			lowpass_tmp.append(Tracker["lowpass"])
		Tracker["lowpass"] = min(lowpass_tmp)
		if myid ==main_node:
			log.add(" the adopted  low pass filter is %f    %f   "%(Tracker["lowpass"], Tracker["falloff"]))
		for iref in range(numref):
			if myid ==main_node:
				volref = sparx_utilities.model_blank(nx,nx,nx)
				for j in range(-m, m+1):
					#if iref +j <0: k = numref+iref+j
					#else: k = (iref+j)%numref
					
					volref += tmpref[(j+iref)%numref]*weight[abs(j)] # weight the reference volumes 
			
				#volref = tmpref[iref]
				if Tracker["mask3D"]: 
					mask3D = sparx_utilities.get_im(Tracker["mask3D"])
					stat = EMAN2_cppwrap.Util.infomask(volref, mask3D, False)
					volref -= stat[0]
					EMAN2_cppwrap.Util.mul_scalar(volref, 1.0/stat[1])
					
				if(Tracker["constants"]["PWadjustment"]):
				
					rt = sparx_utilities.read_text_file(Tracker["PW_dict"][Tracker["constants"]["nxinit"]])
					ro = sparx_fundamentals.rops_table(volref)
					for i in range(1,len(ro)):  ro[i] = (rt[i]/ro[i])**Tracker["constants"]["upscale"]
					volref =sparx_filter.filt_table(volref,ro)
							
				if (Tracker["constants"]["low_pass_filter"]== -1.):  volref = sparx_filter.filt_tanl(volref, Tracker["lowpass"], Tracker["falloff"])                                       # low pass from resolution 
				else:                                                volref = sparx_filter.filt_tanl(volref, min(Tracker["constants"]["low_pass_filter"]/Tracker["shrinkage"],0.45), Tracker["falloff"]) # user define filter
				if Tracker["mask3D"]: EMAN2_cppwrap.Util.mul_img(volref, mask3D)
				volref.write_image( os.path.join(outdir,"volf%04d.hdf"%(total_iter)), iref)
				del volref
				
		"""Multiline Comment91"""
		#if(myid == main_node):
		"""Multiline Comment92"""
		mpi.mpi_barrier(mpi.MPI_COMM_WORLD)
		if terminate == 0: # headers are only updated when the program is going to terminate
			start_time = time.time()
			if nrefine!=0: par_str = ['xform.projection', 'ID', 'group']
			else: par_str = ['group', 'ID' ]
			"""Multiline Comment93"""
			if(myid == 0):log.add( "Time to write headers: %d\n" % (time.time()-start_time) );start_time = time.time()
		else:
			final_list = sparx_utilities.get_sorting_params_refine(Tracker, data, total_nima)
			group_list, ali3d_params_list = sparx_utilities.parsing_sorting_params(final_list)
			if myid ==main_node:
				group_list_saved_file =os.path.join(outdir, "list2.txt")
				sparx_utilities.write_text_file(group_list,group_list_saved_file)
			mpi.mpi_barrier(mpi.MPI_COMM_WORLD)
			Tracker["this_partition"]=group_list
			break
	if terminate != 1:
		final_list = sparx_utilities.get_sorting_params_refine(Tracker,data, total_nima)
		group_list, ali3d_params_list = sparx_utilities.parsing_sorting_params(final_list)  
		if myid ==main_node:
			group_list_saved_file =os.path.join(outdir, "list2.txt")
			sparx_utilities.write_text_file(group_list,group_list_saved_file)
		mpi.mpi_barrier(mpi.MPI_COMM_WORLD)
		Tracker["this_partition"] = group_list
	### program finishes
	if myid ==main_node:  log.add(" mref_ali3d_EQ_Kmeans finishes !")
		
######
 
 

			
			
			
