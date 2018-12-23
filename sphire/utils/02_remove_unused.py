#!/usr/bin/env python
from __future__ import print_function
from __future__ import division

# Author: Markus Stabrin, 2018/11/20 (markus.stabrin@mpi-dortmund.mpg.de
# Copyright (c) 2018 Max Plank of molecular physiology Dortmund
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
import os
import importlib
import sys
import shutil
import re
import glob

USED_FOLDER = ('bin', 'libpy', 'templates')
USED_FOLDER_EMAN2 = ('libpyEM', 'libpyEM/qtgui')
BIN_FOLDER = ('bin', 'templates')
LIB_FOLDER = ('libpy', 'templates', 'libpyEM', 'libpyEM/qtgui')

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
LIB_DIR = os.path.join(CURRENT_DIR, '..', 'libpy')
NO_UNUSED = os.path.join(CURRENT_DIR, 'NO_IMPORTS')
try:
    shutil.rmtree(NO_UNUSED) 
except OSError:
    pass

FUNCDEF_RE = re.compile('^(?:def|class)\s+([^\(]+)')
SPARX_FUNC_RE = re.compile('(?:(?<=[^\w])|^)({0})\.([\w]+)'.format('|'.join([
    os.path.splitext(os.path.basename(entry))[0]
    for entry in glob.glob('{0}/*.py'.format(LIB_DIR))
    if 'sparx.py' not in entry
    ])))


def get_file_dict():
    file_dict = {}
    for folder_name in USED_FOLDER:
        files_dir = os.path.join(CURRENT_DIR, 'NO_WILDCARDS', folder_name, '*.py')
        file_dict[folder_name] = sorted(glob.glob(files_dir))
    for folder_name in USED_FOLDER_EMAN2:
        files_dir = os.path.join(CURRENT_DIR, '../..', folder_name, '*.py*')
        file_dict[folder_name] = [entry for entry in sorted(glob.glob(files_dir)) if not entry.endswith('.pyc')]
    return file_dict


def import_functions(file_dict):
    function_dict = {}
    for key, file_names in file_dict.items():
        for file_path in file_names:
            basename = os.path.basename(file_path)
            assert basename not in function_dict
            function_dict.setdefault(basename, {})['double_funcs'] = []
            function_dict.setdefault(basename, {})['used_funcs'] = []
            with open(file_path) as read:
                lines = read.readlines()
            current_func = None
            for line in lines:
                match_line = FUNCDEF_RE.match(line)
                if match_line:
                    current_func = match_line.group(1)
                    if current_func in function_dict[basename]:
                        function_dict[basename]['double_funcs'].append(current_func)
                    function_dict[basename][current_func] = []
                if current_func is None:
                    continue
                function_dict[basename][current_func].append(line)

    return function_dict


def recursive_used_search(module, function, function_dict):
    pass


def main():
    file_dict = get_file_dict()

    function_dict = import_functions(file_dict)

    for key in file_dict['bin']:
        basename = os.path.basename(key)
        try:
            function_dict[basename]['main']
        except:
            print(key)


if __name__ == '__main__':
    main()
