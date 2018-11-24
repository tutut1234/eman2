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
import shutil
import sys
import glob
import re
import importlib
import argparse

from pyflakes import reporter as modReporter
from pyflakes.checker import Checker
import pyflakes.messages as pym
import pyflakes.api as pyfl

parser = argparse.ArgumentParser()
parser.add_argument('--silent', action='store_true', help='Do not write any output to disc')
options = parser.parse_args()


def my_report(self, messageClass, *args, **kwargs):
    if pym.UndefinedName == messageClass:
        message = messageClass(self.filename, *args, **kwargs)
        self.okidoki.append(message.to_list())


def my_to_list(self):
    lineno = int(self.lineno) - 1
    col = int(self.col)
    name = re.match(".*'(.*)'", self.message % self.message_args).group(1)
    return [lineno, col, name]


def my_exception(self, filename, msg, lineno, offset, text):
    if msg == 'expected an indented block':
        print(filename, msg, lineno, offset, text)
        self.okidoki.append(lineno)


def index_search(lines, index, no_index):
    if lines[index].strip().endswith('\\'):
        no_index.append(index+1)
        index_search(lines, index+1, no_index)


folders = ['tmp', 'no_import', 'new']
for entry in folders:
    try:
        shutil.rmtree(entry)
    except Exception as e:
        print(e)
    try:
        os.mkdir(entry)
    except Exception as e:
        print(e)

IMPORT_DEF_RE = re.compile(r'^(?:def|class) ([^(]*).*')
IMPORT_IMPORTS_RE = re.compile(r'^(\s*).*from\s+([\w.]*)\s+import\s+([\w.]*)\s*(?:as\s+([\w.]*).*|)')
IMPORT_SINGLE_IMPORT_RE = re.compile(r'^(\s*)(?:from\s+[\w.]\s+|)import\s+([\w.,\s]*)\s*(?:as\s*([\w.]*).*|)')
IMPORT_LEADING_RE = re.compile("^(\s*)[^\s]*.*")
IMPORT_COMMENT_RE = re.compile("^(\s*)#[^\s]*.*")
IMPORT_MATPLOTLIB_RE = re.compile("^(\s*)matplotlib.use")


lib_files = glob.glob('../sparx/libpy/*.py')
lib_eman2_files = glob.glob('../libpyEM/*.py')
lib_eman2_files_2 = glob.glob('../libpyEM/qtgui/*.py')
lib_eman2_files_3 = glob.glob('../libpyEM/*.py.in')
bin_files = glob.glob('../sparx/bin/*.py')

qtgui_files = [os.path.splitext(os.path.basename(entry))[0] for entry in lib_eman2_files_2]


lib_modules = {}
lib_modules_ext = {}
for lib_file in lib_files + lib_eman2_files + lib_eman2_files_2 + lib_eman2_files_3:
    name = os.path.splitext(os.path.basename(lib_file))[0].replace('.py', '')
    with open(lib_file) as read:
        lib_modules[name] = [
            IMPORT_DEF_RE.match(entry).group(1)
            for entry in read.readlines()
            if IMPORT_DEF_RE.match(entry)
            ]
    if 'global_def' == name:
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


external_modules = [
    'subprocess',
    'random',
    'mpi',
    'sys',
    'traceback',
    'time',
    'numpy',
    'math',
    'operator',
    'optparse',
    'EMAN2_cppwrap',
    'sets',
    'copy',
    'inspect',
    'scipy',
    'scipy.optimize',
    'datetime',
    'heapq',
    'matplotlib',
    'os',
    'string',
    'builtins',
    'shutil',
    'glob',
    'types',
    'pickle',
    'zlib',
    'struct',
    'fractions',
    'socket',
    ]
for entry in external_modules:
    importlib.import_module(entry)
    lib_modules_ext[entry] = dir(sys.modules[entry])

python_files = bin_files + lib_files

modReporter.Reporter.syntaxError = my_exception
reporter = modReporter._makeDefaultReporter()
reporter.okidoki = []

Checker.report = my_report
Checker.okidoki = []
pym.Message.to_list = my_to_list


#python_files = glob.glob('../sparx/bin/sxsort3d_depth.py')
#python_files = glob.glob('../sparx/libpy/statistics.py')
rounds = 0
while True:
    rounds += 1
    ok = 0
    fatal = 0
    confusion = 0
    for file_name in python_files:
        print('######################################')
        print(file_name)
        Checker.okidoki = []
        reporter.okidoki = []

        with open(file_name, 'r') as read:
            lines = read.readlines()

        bad_index = []
        no_index = []
        file_modules = []
        local_func_import = {}
        for index, entry in enumerate(lines):
            match = IMPORT_IMPORTS_RE.match(entry)
            if IMPORT_COMMENT_RE.match(entry):
                continue

            elif match and \
                    'future' not in entry and \
                    'qt' not in entry.lower() and \
                    not 'builtins' in entry:
                string = match.group(1)
                local_func_import.setdefault(match.group(2), []).append(match.group(3))
                file_modules.append([index, string, [match.group(2)]])
                if match.group(4):
                    try:
                        lib_modules[match.group(2)].append(match.group(4))
                    except KeyError:
                        lib_modules_ext[match.group(2)].append(match.group(4))
                if match.group(2) in external_modules:
                    try:
                        lib_modules[match.group(2)].append(match.group(3))
                    except KeyError:
                        lib_modules_ext[match.group(2)].append(match.group(3))
                bad_index.append([index, string])
                if lines[index].strip().endswith('\\'):
                    no_index.append(index+1)
                    index_search(lines, index+1, no_index)

            else:
                match = IMPORT_SINGLE_IMPORT_RE.match(entry)
                if match and \
                        'future' not in entry and \
                        'qt' not in entry.lower() and \
                        not 'builtins' in entry:
                    string = match.group(1)
                    name = [loc_entry.strip() for loc_entry in match.group(2).split(',')]
                    file_modules.append([index, string, name])
                    if lines[index].strip().endswith('\\'):
                        no_index.append(index+1)
                        index_search(lines, index+1, no_index)

            match = IMPORT_MATPLOTLIB_RE.match(entry)
            if match:
                string = match.group(1)
                no_index.append(index)


        no_from_import_lines = lines[:]
        no_import_lines = lines[:]

        for idx, string in bad_index:
            if '#IMPORTIMPORTIMPORT' in lines[idx]:
                continue
            elif lines[idx].strip().startswith('#'):
                no_from_import_lines[idx] = '#{0}pass#IMPORTIMPORTIMPORT {1}\n'.format(string, lines[idx].strip())
                no_import_lines[idx] = '#{0}pass#IMPORTIMPORTIMPORT {1}\n'.format(string, lines[idx].strip())
            else:
                no_from_import_lines[idx] = '{0}pass#IMPORTIMPORTIMPORT {1}\n'.format(string, lines[idx].strip())
                no_import_lines[idx] = '{0}pass#IMPORTIMPORTIMPORT {1}\n'.format(string, lines[idx].strip())

        for idx in no_index:
            if '#IMPORTIMPORTIMPORT' in lines[idx]:
                continue
            else:
                string = IMPORT_LEADING_RE.match(lines[idx]).group(1)
                no_from_import_lines[idx] = '#{0}pass#IMPORTIMPORTIMPORT {1}\n'.format(string, lines[idx].strip())
                no_import_lines[idx] = '#{0}pass#IMPORTIMPORTIMPORT {1}\n'.format(string, lines[idx].strip())

        correct_imports = []
        for idx, string, module in file_modules:
            correct_imports.extend(module)
            if '#IMPORTIMPORTIMPORT' in lines[idx]:
                continue
            elif lines[idx].strip().startswith('#'):
                no_import_lines[idx] = '#{0}pass#IMPORTIMPORTIMPORT {1}\n'.format(string, lines[idx].strip())
            else:
                no_import_lines[idx] = '{0}pass#IMPORTIMPORTIMPORT {1}\n'.format(string, lines[idx].strip())
        correct_imports = list(set(correct_imports))

        file_content = ''.join(no_from_import_lines)
        pyfl.check(file_content, file_name, reporter)

        if not options.silent:
            with open(os.path.join('tmp', os.path.basename(file_name)), 'w') as write:
                write.write(file_content)

        if not options.silent:
            file_content = ''.join(no_import_lines)
            with open(os.path.join('no_import', os.path.basename(file_name)), 'w') as write:
                write.write(file_content)

        fatal_list = []
        ok_list = []
        confusion_list = []
        for line_number, column, name in Checker.okidoki:
            mod_list = []
            for key, values in lib_modules.items():
                for val in values:
                    if name == val:
                        mod_list.append(key)
            mod_list = list(set(mod_list))
            if len(mod_list) == 1:
                out_list = ok_list
            else:
                for key, values in lib_modules_ext.items():
                    for val in values:
                        if name == val:
                            mod_list.append(key)
                mod_list = list(set(mod_list))

                if not mod_list:
                    out_list = fatal_list
                elif len(mod_list) == 1:
                    out_list = ok_list
                #elif 'numpy' in mod_list:
                #    mod_list = ['numpy']
                #    out_list = ok_list
                elif ['math', 'numpy', 'scipy'] == sorted(mod_list):
                    mod_list = ['numpy']
                    out_list = ok_list
                elif ['numpy', 'scipy'] == sorted(mod_list):
                    mod_list = ['numpy']
                    out_list = ok_list
                elif ['numpy', 'scipy', 'scipy.optimize'] == sorted(mod_list):
                    mod_list = ['numpy']
                    out_list = ok_list
                elif name == 'os':
                    mod_list = ['os']
                    out_list = ok_list
                else:
                    local_imports = []
                    for key in local_func_import:
                        for module_name in local_func_import[key]:
                            if module_name == name:
                                local_imports.append(key)
                    local_imports = list(set(local_imports))

                    if len(local_imports) == 1:
                        mod_list = local_imports
                        out_list = ok_list
                    else:
                        local_imports = []
                        for entry in file_modules:
                            for module_name in entry[2]:
                                if module_name == name:
                                    local_imports.append(module_name)
                        local_imports = list(set(local_imports))

                        if len(local_imports) == 1:
                            mod_list = local_imports
                            out_list = ok_list
                        else:
                            #print(name, local_imports)
                            out_list = confusion_list
            out_list.append([line_number, column, name, mod_list])

        print('Typos that needs to be resolved:')
        template = 'name: {2:>25s}, line: {0: 6d}, column: {1: 6d}, module(s): {3}'
        fatal += len(fatal_list)
        for entry in fatal_list:
            print(template.format(*entry))

        print('')
        print('Confusion list:')
        confusion += len(confusion_list)
        for entry in confusion_list:
            print(template.format(*entry))
        print('')

        print('RESOLVED THINGS:')
        used_modules = []
        ok += len(ok_list)
        idx_line = 0
        idx_column = 1
        idx_name = 2
        idx_mod = 3
        len_dict = {}
        for entry in ok_list:
            print(template.format(*entry))
            used_modules.extend(entry[idx_mod])
            out = []

            new_line = []
            current_line = no_import_lines[entry[idx_line]]
            len_name = len(entry[idx_name])
            len_mod = len(entry[idx_mod][0])+1
            len_adjust = len_dict.setdefault(entry[idx_line], 0)

            new_line.append(current_line[:entry[idx_column] + len_adjust])
            new_line.append('{0}.{1}'.format(entry[idx_mod][0], entry[idx_name]))
            new_line.append(current_line[entry[idx_column]+len_name+len_adjust:])
            no_import_lines[entry[idx_line]] = ''.join(new_line)
            try:
                len_dict[entry[idx_line]] += len_mod
            except KeyError:
                len_dict[entry[idx_line]] = len_mod

        correct_imports_clean = []
        for entry in correct_imports:
            try:
                importlib.import_module(entry.split()[0])
            except ImportError:
                continue
            correct_imports_clean.append(entry)

        imports = ['import {0}\n'.format(entry) if entry not in qtgui_files else 'import eman2_gui.{0} as {0}\n'.format(entry) for entry in list(set(used_modules))]
        imports.extend(['import {0}\n'.format(entry) if entry.split('.')[-1] not in qtgui_files else 'import eman2_gui.{0} as {0}\n'.format(entry.split('.')[-1]) for entry in correct_imports_clean])
        imports = sorted(list(set(imports)))
        inserted = False
        for idx, entry in enumerate(imports[:]):
            if entry == 'import matplotlib\n' and not inserted:
                imports.insert(idx+1, 'matplotlib.use("Agg")\n')
                inserted = True
                break
            elif 'matplotlib' in entry and not inserted:
                imports.insert(idx, 'matplotlib.use("Agg")\n')
                imports.insert(idx, 'import matplotlib\n')
                inserted = True
                break

        imports = ''.join(imports)
        first_1 = False
        first_2 = False
        first_3 = False
        index = 3
        for idx, line in enumerate(no_import_lines[2:], 2):
            if line.startswith("class") or line.startswith('def'):
                    if not first_1 and not first_2 and not first_3:
                        index = idx
                        break
            elif line.startswith("'''"):
                if first_1:
                    index = idx+1
                    break
                else:
                    first_1 = True
            elif line.startswith('"""'):
                if first_2:
                    index = idx+1
                    break
                else:
                    first_2 = True
            elif line.startswith('#') and not first_1 and not first_2:
                first_3 = True
            elif first_3:
                index = idx+1
                break
        no_import_lines.insert(index, imports)

        remove_indices = []
        first_1 = False
        first_2 = False
        for idx, line in enumerate(no_import_lines[2:], 2):
            if line.startswith("'''"):
                if first_1:
                    first_1 = False
                else:
                    first_1 = True
            elif line.startswith('"""'):
                if first_2:
                    first_2 = False
                else:
                    first_2 = True
            elif line.startswith('#'):
                pass
            elif line.startswith("class") or line.startswith('def'):
                if not first_1 and not first_2:
                    pass
            if '#IMPORTIMPORTIMPORT' in line:
                remove_indices.append(idx)


        output_lines = []
        for idx, line in enumerate(no_import_lines):
            if idx in remove_indices:
                pass
            else:
                output_lines.append(line)

        if not options.silent:
            file_content = ''.join(output_lines)
            with open(os.path.join('new', os.path.basename(file_name)), 'w') as write:
                write.write(file_content)
            with open(file_name, 'w') as write:
                write.write(file_content)
        print('')

    print('FATAL:', fatal)
    print('CONFUSION:', confusion)
    print('RESOLVED:', ok)
    if ok == 0:
        print('Resolved after', rounds, 'rounds')
        break
