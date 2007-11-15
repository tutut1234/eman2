#!/usr/bin/env python
#
# Author: Steven Ludtke, 11/01/2007 (sludtke@bcm.edu)
# Copyright (c) 2000-2006 Baylor College of Medicine
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
# Foundation, Inc., 59 Temple Place, Suite 330, Boston MA 02111-1307 USA
#
#

from OpenGL import GL,GLUT
from math import *

def initGL():
	"""Call this static function once to initialize necessary display lists"""
	if EMShape.dlists>=0: return
	EMShape.dlists=GL.glGenLists(1)
	GL.glNewList(EMShape.dlists,GL.GL_COMPILE)
	GL.glBegin(GL.GL_LINE_LOOP)
	d2r=pi/180.0
	for i in range(90): GL.glVertex(sin(i*d2r*4.0),cos(i*d2r*4.0))
	GL.glEnd()
	GL.glEndList()

def shidentity(x,y) : return x,y

class EMShape:
	"""This class represents a geometric shape which can be used to annotate
	the various data display widgets in EMAN2. The 'scr' shapes are in screen
	coordinates, and the others are in data coordinates. Shapes are initialized
	and read out in the form of short lists or tuples. Note that no validation is
	performed. A shape that is not understood will simply not be rendered, meaning
	the programmer may create 'invisible' shapes for out-of-band use. Colors
	are on the range 0-1.0 
	
		0            1  2  3  4  5     6     7     8
		"rect"       R  G  B  x0 y0    x1    y1    linew
		"line"       R  G  B  x0 y0    x1    y1    linew
		"label"      R  G  B  x0 y0    text  size	linew
		"circle"     R  G  B  x0 y0    r     linew
		"scrrect"    R  G  B  x0 y0    x1    y1    linew
		"scrline"    R  G  B  x0 y0    x1    y1    linew
		"scrlabel"   R  G  B  x0 y0    text  size	linew
		"scrcircle"  R  G  B  x0 y0    r     linew
"""
	def __init__(self,init=None) :
		"""init is a list/tuple containing the above parameters describing the shape"""
		if init : self.shape=list(init)
		else : self.shape=["None",0,0,0,0,0,0,0,0]
	
	dlists=-1
	
	
	def draw(self,d2s=None,col=None):
		"""This function causes the shape to render itself into the current GL context.
		d2s is a function of x,y which will convert data coordinates to screen
		coordinates. For data coordinate shapes, only the positional information
		is in data coordinates, font size, line width, etcare in screen units.
		col can be used to override the current shape's color."""
		s=self.shape
		
		if col==None: col=self.shape[1:4]
		if d2s==None : d2s=shidentity
		
		v=d2s(s[4],s[5])
		v2=d2s(s[4]+1,s[5]+1)
		sc=v2[0]-v[0]
		if s[0]=="rect":
			GL.glLineWidth(s[8])
			GL.glBegin(GL.GL_LINE_LOOP)
			GL.glColor(*col)
			GL.glVertex(*d2s(s[4],s[5]))
			GL.glVertex(*d2s(s[6],s[5]))
			GL.glVertex(*d2s(s[6],s[7]))
			GL.glVertex(*d2s(s[4],s[7]))
			GL.glEnd()
		elif s[0]=="line":
			GL.glColor(*col)
			GL.glLineWidth(s[8])
			GL.glBegin(GL.GL_LINES)
			GL.glVertex(*d2s(s[4],s[5]))
			GL.glVertex(*d2s(s[6],s[7]))
			GL.glEnd()
		elif s[0]=="label":
			GL.glPushMatrix()
			if s[8]<0 :
				GL.glColor(1.,1.,1.)
				GL.glTranslatef(v[0],v[1],0)
				GL.glScalef(s[7]/100.0/sc,s[7]/100.0/sc,s[7]/100.0/sc)
				GL.glLineWidth(-s[8])
				w=104.76*len(s[6])
				GL.glBegin(GL.GL_QUADS)
				GL.glVertex(-10.,-33.0)
				GL.glVertex(w+10.,-33.0)
				GL.glVertex(w+10.,119.05)
				GL.glVertex(-10.,119.05)
				GL.glEnd()
				GL.glColor(*col)
				for i in s[6]:
					GLUT.glutStrokeCharacter(GLUT.GLUT_STROKE_MONO_ROMAN,ord(i))
			else:
				GL.glColor(*col)
				GL.glTranslate(v[0],v[1],0)
#				GL.glScalef(s[7]/100.0,s[7]/100.0,s[7]/100.0)
				GL.glScalef(s[7]/100.0/sc,s[7]/100.0/sc,s[7]/100.0/sc)
				GL.glLineWidth(fabs(s[8]))
				for i in s[6]:
					GLUT.glutStrokeCharacter(GLUT.GLUT_STROKE_ROMAN,ord(i))
			GL.glPopMatrix()
		elif s[0]=="circle":
			GL.glPushMatrix()
			GL.glColor(*col)
			GL.glLineWidth(s[7])
			GL.glTranslate(v[0],v[1],0)
			GL.glScalef(s[6]*(v2[0]-v[0]),s[6]*(v2[1]-v[1]),1.0)
			GL.glCallList(EMShape.dlists)
			GL.glPopMatrix()
		else:
			mx=GL.glGetFloatv(GL.GL_MODELVIEW_MATRIX)
			GL.glPopMatrix()
			GL.glPushMatrix()
			if s[0]=="scrrect":
				GL.glLineWidth(s[8])
				GL.glBegin(GL.GL_LINE_LOOP)
				GL.glColor(*col)
				GL.glVertex(s[4],s[5])
				GL.glVertex(s[6],s[5])
				GL.glVertex(s[6],s[7])
				GL.glVertex(s[4],s[7])
				GL.glEnd()
			elif s[0]=="scrline":
				GL.glColor(*col)
				GL.glLineWidth(s[8])
				GL.glBegin(GL.GL_LINES)
				GL.glVertex(s[4],s[5])
				GL.glVertex(s[6],s[7])
				GL.glEnd()
			elif s[0]=="scrlabel":
				if s[8]<0 :
					GL.glColor(1.,1.,1.)
					GL.glTranslate(s[4],s[5],0)
					GL.glScalef(s[7]/1500.0/sc,s[7]/1500.0/sc,s[7]/1500.0/sc)
					GL.glLineWidth(-s[8])
					w=104.76*len(s[6])
					GL.glBegin(GL.GL_QUADS)
					GL.glVertex(-10.,-33.0)
					GL.glVertex(w+10.,-33.0)
					GL.glVertex(w+10.,119.05)
					GL.glVertex(-10.,119.05)
					GL.glEnd()
					GL.glColor(*col)
					for i in s[6]:
						GLUT.glutStrokeCharacter(GLUT.GLUT_STROKE_MONO_ROMAN,ord(i))
				else:
					GL.glColor(*col)
					GL.glTranslate(s[4],s[5],0)
	#				GL.glScalef(s[7]/100.0,s[7]/100.0,s[7]/100.0)
					GL.glScalef(s[7]/1500.0/sc,s[7]/1500.0/sc,s[7]/1500.0/sc)
					GL.glLineWidth(fabs(s[8]))
					for i in s[6]:
						GLUT.glutStrokeCharacter(GLUT.GLUT_STROKE_ROMAN,ord(i))
			elif s[0]=="scrcircle":
				GL.glColor(*col)
				GL.glLineWidth(s[7])
				GL.glTranslate(s[4],s[5],0)
				GL.glScalef(s[6],s[6],s[6])
				GL.glCallList(EMShape.dlists)
			GL.glLoadMatrixf(mx)

	def setShape(self,shape):
		"""sets the shape to a new tuple/list"""
		self.shape=list(shape)
	
	def setColor(self,r,g=-1,b=-1):
		"""Sets the color of the shape. You may call this either as s.setColor(r,g,b) or s.setcolor([r,g,b])"""
		
		try:
			if g<0 :
				self.shape[1]=r[0]
				self.shape[2]=r[1]
				self.shape[3]=r[2]
			else:
				self.shape[1]=r
				self.shape[2]=g
				self.shape[3]=b
		except:
			print "Invalid color set in shape ",self.shape
			return
			
	def setloc(self,x0,y0,x1=None,y1=None):
		"""This sets the coordinates of the shape"""
		self.shape[4]=x0
		self.shape[5]=y0
		if x1!=None : self.shape[6]=x1
		if y1!=None : self.shape[7]=y1
		
