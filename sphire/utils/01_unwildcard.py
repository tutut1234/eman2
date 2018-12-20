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

from pyflakes import reporter as modReporter
from pyflakes.checker import Checker
import pyflakes.messages as pym
import pyflakes.api as pyfl

USED_FOLDER = ('bin', 'libpy', 'templates')
USED_FOLDER_EMAN2 = ('libpyEM', 'libpyEM/qtgui')
BIN_FOLDER = ('bin', 'templates')
LIB_FOLDER = ('libpy', 'templates', 'libpyEM', 'libpyEM/qtgui')

IGNORE_FILES = (
    'sparx.py'
    )
IGNORE_MODULES = (
    '__future__'
    )
EXTERNAL_LIBS = (
    'EMAN2_cppwrap',
    )
#EXTERNAL_LIBS = (
#    'subprocess',
#    'random',
#    'mpi',
#    'sys',
#    'traceback',
#    'time',
#    'numpy',
#    'numpy.random',
#    'math',
#    'operator',
#    'optparse',
#    'sets',
#    'copy',
#    'inspect',
#    'scipy',
#    'scipy.optimize',
#    'scipy.stats',
#    'datetime',
#    'heapq',
#    'matplotlib',
#    'os',
#    'string',
#    'builtins',
#    'shutil',
#    'glob',
#    'types',
#    'pickle',
#    'zlib',
#    'struct',
#    'fractions',
#    'socket',
#    )

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
NO_IMPORTS_DIR = os.path.join(CURRENT_DIR, 'NO_IMPORTS')
try:
    shutil.rmtree(NO_IMPORTS_DIR) 
except OSError:
    pass
NO_PLACEHOLD_DIR = os.path.join(CURRENT_DIR, 'NO_PLACEHOLD')
try:
    shutil.rmtree(NO_PLACEHOLD_DIR) 
except OSError:
    pass
NO_WILDCARDS_DIR = os.path.join(CURRENT_DIR, 'NO_WILDCARDS')
try:
    shutil.rmtree(NO_WILDCARDS_DIR) 
except OSError:
    pass

FUNCDEF_RE = re.compile('^(?:def|class)\s+([^\(]+)')
IMPORT_RE = re.compile('^(\s*)(import\s+([\w.]+)\s*.*|from\s+([\w.]+)\s+import.*)')
COMMENT_RE = re.compile("^\s*#")
BLOCK_STRING_SINGLE_RE = re.compile('\'\'\'')
BLOCK_STRING_DOUBLE_RE = re.compile('"""')
MULTILINE_RE = re.compile('.*\\\s*$')

FILE_NAME = None
PRINT_LINE = None
def print_all_info(**kwargs):
    if PRINT_LINE:
        if kwargs['line_idx'] in PRINT_LINE:
            for key in sorted(kwargs.keys()):
                print(key, ':', kwargs[key])
            print('')


def get_file_dict():
    file_dict = {}
    for folder_name in USED_FOLDER:
        files_dir = os.path.join(CURRENT_DIR, 'NO_COMMENTS', folder_name, '*.py')
        file_dict[folder_name] = sorted(glob.glob(files_dir))
    for folder_name in USED_FOLDER_EMAN2:
        files_dir = os.path.join(CURRENT_DIR, '../..', folder_name, '*.py*')
        file_dict[folder_name] = sorted(glob.glob(files_dir))
    return file_dict


def my_report(self, messageClass, *args, **kwargs):
    if pym.UndefinedName == messageClass:
        message = messageClass(self.filename, *args, **kwargs)
        self.undefined_names.append(message.to_list())
    elif pym.UnusedImport == messageClass:
        if self.filename not in IGNORE_FILES:
            message = messageClass(self.filename, *args, **kwargs)
            self.unused_imports.append(message.to_list())


def my_to_list(self):
    lineno = int(self.lineno) - 1
    col = int(self.col)
    name = re.match(".*'(.*)'", self.message % self.message_args).group(1)
    return (lineno, col, name)


def my_exception(self, filename, msg, lineno, offset, text):
    if msg == 'expected an indented block':
        print(filename, msg, lineno, offset, text.strip())
        self.undefined_names.append([int(lineno)-1, text])


def get_external_libs(external_libs, lib_modules=None):
    if lib_modules is None:
        lib_modules = []
    lib_modules_ext = {}
    for entry in external_libs:
        if entry in lib_modules:
            continue
        try:
            importlib.import_module(entry)
        except:
            print(entry)
            raise
        lib_modules_ext[entry] = dir(sys.modules[entry])
    return lib_modules_ext


def get_library_funcs(file_dict):
    lib_modules = {}
    for key in file_dict:
        if key not in LIB_FOLDER:
            continue
        for file_name in file_dict[key]:
            name = os.path.splitext(os.path.basename(file_name))[0].replace('.py', '')
            with open(file_name) as read:
                lines = read.readlines()
            lib_modules[name] = [
                FUNCDEF_RE.match(entry).group(1)
                for entry in lines
                if FUNCDEF_RE.match(entry)
                ]
            if 'global_def' in name:
                lib_modules[name].append('SPARXVERSION')
                lib_modules[name].append('CACHE_DISABLE')
                lib_modules[name].append('SPARX_MPI_TAG_UNIVERSAL')
                lib_modules[name].append('interpolation_method_2D')
                lib_modules[name].append('Eulerian_Angles')
                lib_modules[name].append('BATCH')
                lib_modules[name].append('MPI')
                lib_modules[name].append('LOGFILE')
                lib_modules[name].append('SPARX_DOCUMENTATION_WEBSITE')
            elif 'EMAN2_meta' == name:
                lib_modules[name].append('EMANVERSION')
                lib_modules[name].append('DATESTAMP')
            elif 'emapplication' in name:
                lib_modules[name].append('get_application')

            if name.startswith('sparx_'):
                transform_dict[name.replace('sparx_', '')] = name
    return lib_modules


def remove_imports(file_dict, lib_modules):
    local_imports = {}
    for key, file_names in file_dict.items():
        if key in USED_FOLDER_EMAN2:
            continue
        output_dir = os.path.join(NO_IMPORTS_DIR, key)
        try:
            os.makedirs(output_dir)
        except OSError:
            pass
        for file_path in file_names:
            basename = os.path.basename(file_path)
            output_file_name = os.path.join(output_dir, basename)
            with open(file_path) as read:
                lines = read.readlines()

            local_modules = []
            if basename not in IGNORE_FILES:
                comment1 = False
                comment2 = False
                multiline_idx = []
                for line_idx, line in enumerate(lines[:]):
                    if COMMENT_RE.match(line):
                        continue
                    match = IMPORT_RE.match(line)
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

                    if match and not comment1 and not comment2 and not do_final:
                        indent = match.group(1)
                        content = match.group(2)
                        module = None
                        if match.group(3) is not None:
                            module = match.group(3)
                        elif match.group(4) is not None:
                            module = match.group(4)
                        assert module is not None
                        if module in IGNORE_MODULES:
                            continue
                        new_line = '{0}pass#IMPORTIMPORTIMPORT {1}\n'.format(
                            indent,
                            '\n{0}'.format(indent).join(
                                [entry.strip() for entry in content.split(';')]
                                )
                            )
                        lines[line_idx] = new_line
                        idx = 0
                        while MULTILINE_RE.match(lines[line_idx+idx]):
                            idx += 1
                            multiline_idx.append(line_idx+idx)
                        local_modules.append(module)
                for idx in multiline_idx:
                    lines[idx] = '\n'
                try:
                    local_imports[file_path] = get_external_libs(set(local_modules), lib_modules)
                except:
                    print(file_path)
                    raise
            else:
                pass

            with open(output_file_name, 'w') as write:
                write.write(''.join(lines))
    return local_imports


def index_search(lines, index, no_index):
    if lines[index].strip().endswith('\\'):
        no_index.append(index+1)
        index_search(lines, index+1, no_index)


modReporter.Reporter.syntaxError = my_exception
reporter = modReporter._makeDefaultReporter()
reporter.undefined_names = []

Checker.report = my_report
Checker.undefined_names = []
Checker.unused_imports = []
pym.Message.to_list = my_to_list

def main():
    file_dict = get_file_dict()

    #Extract all function names from library files
    lib_modules = get_library_funcs(file_dict)

    # Look for content of external_modules
    lib_modules_ext = get_external_libs(EXTERNAL_LIBS)

    # Remove all imports from files
    lib_modules_local = remove_imports(file_dict, lib_modules)

if __name__ == '__main__':
    main()
