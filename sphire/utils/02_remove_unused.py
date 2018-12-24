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

GLOBAL_RE = re.compile('^\w')
FUNCDEF_RE = re.compile('^(?:def|class)\s+([^\(]+)')
SPARX_FUNC_RE = re.compile('(?:(?<=[^\w])|^)({0})\.([\w]+)'.format('|'.join([
    os.path.splitext(os.path.basename(entry))[0]
    for entry in glob.glob('{0}/*.py'.format(LIB_DIR))
    if 'sparx.py' not in entry
    ])))
COMMENT_RE = re.compile("^\s*#")
EMPTY_RE = re.compile("^\s*$")
BLOCK_STRING_SINGLE_RE = re.compile('\'\'\'')
BLOCK_STRING_DOUBLE_RE = re.compile('"""')


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
            module = os.path.splitext(os.path.basename(file_path))[0]
            assert module not in function_dict
            function_dict.setdefault(module, {})['double_funcs'] = []
            function_dict.setdefault(module, {})['global_lines'] = []
            function_dict.setdefault(module, {})['used_funcs'] = []
            with open(file_path) as read:
                lines = read.readlines()
            comment1 = False
            comment2 = False
            current_func = None
            for line in lines:
                if COMMENT_RE.match(line) or EMPTY_RE.match(line):
                    continue
                match_line = FUNCDEF_RE.match(line)
                match_global = GLOBAL_RE.match(line)
                match_block_1 = BLOCK_STRING_SINGLE_RE.findall(line)
                match_block_2 = BLOCK_STRING_DOUBLE_RE.findall(line)

                do_final = False
                if match_block_1 and not comment1 and not comment2:
                    if len(match_block_1) % 2 == 1:
                        comment1 = True
                elif match_block_2 and not comment1 and not comment2:
                    if len(match_block_2) % 2 == 1:
                        comment2 = True
                elif match_block_1:
                    comment1 = False
                    do_final = True
                elif match_block_2:
                    comment2 = False
                    do_final = True

                if not comment1 and not comment2 and not do_final:
                    if match_line:
                        current_func = match_line.group(1)
                        if current_func in function_dict[module]:
                            function_dict[module]['double_funcs'].append(current_func)
                        function_dict[module][current_func] = []
                    elif match_global:
                        current_func = 'global_lines'
                    assert current_func is not None, line
                    function_dict[module][current_func].append(line)

    return function_dict


def recursive_used_search(module, function, module_re_dict, function_dict):
    found_functions = []
    for line in function:
        sparx_match = SPARX_FUNC_RE.findall(line)
        if module_re_dict[module] is None:
            function_match = []
        else:
            function_match = module_re_dict[module].findall(line)

        for sparx_module, sparx_function in sparx_match:
            if sparx_function in function_dict[sparx_module]:
                found_functions.append((sparx_module, sparx_function))
            else:
                print('UNKNOWN FUNCTION', sparx_module, sparx_function)

        for module_function in function_match:
            if module_function in function_dict[module]:
                found_functions.append((module, module_function))
            else:
                print('UNKNOWN FUNCTION', module_function)

    found_functions = set(found_functions)
    for used_module, used_function in found_functions:
        if used_function in function_dict[used_module]['used_funcs']:
            continue
        else:
            function_dict[used_module]['used_funcs'].append(used_function)
        recursive_used_search(used_module, function_dict[used_module][used_function], module_re_dict, function_dict)


def main():
    file_dict = get_file_dict()

    function_dict = import_functions(file_dict)

    module_re_dict = {}
    for module, functions in function_dict.items():
        usable_functions = [
            entry
            for entry in functions
            if entry not in ('used_funcs', 'double_funcs', 'global_lines')
            ]
        if usable_functions:
            module_re_dict[module] = re.compile('(?:(?<=[^\w])|^)({0})(?=[^\w]|$)'.format('|'.join(usable_functions)))
        else:
            module_re_dict[module] = None

    for key in file_dict['bin']:
        module = os.path.splitext(os.path.basename(key))[0]
        recursive_used_search(module, function_dict[module]['global_lines'], module_re_dict, function_dict)

    for key in function_dict:
        print(function_dict[key]['used_funcs'])

if __name__ == '__main__':
    main()
