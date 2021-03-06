#!/usr/bin/env python
from __future__ import print_function
from __future__ import division

# This example script will extract CTF parameters from bdb:e2ctf.parms into a CSV file readable by a spreadsheet

from EMAN2 import *

db=db_open_dict("bdb:e2ctf.parms",True)

out=open("ctf_parms.txt","w")
for k in list(db.keys()):
	ctf=EMAN2Ctf()
	ctf.from_string(db[k][0])
	out.write( "%s,%1.3f,%1.1f\n"%(k,ctf.defocus,ctf.bfactor))
