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
FUNCDEF_RE = re.compile('^(?:def|class)\s+([^\(]+)')

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

NO_USED = os.path.join(CURRENT_DIR, '02_NO_USED')
NO_UNUSED = os.path.join(CURRENT_DIR, '02_NO_UNUSED')
NO_WILDCARDS_DIR = os.path.join(CURRENT_DIR, '01_NO_WILDCARDS')

IMPORTS_SPHIRE_DIR = os.path.join(CURRENT_DIR, '03_SPHIRE_IMPORTS')
IMPORTS_SPARX_DIR = os.path.join(CURRENT_DIR, '03_SPARX_IMPORTS')
SPHIRE_DIR = os.path.join(CURRENT_DIR, '03_SPHIRE')
SPARX_DIR = os.path.join(CURRENT_DIR, '03_SPARX')

from pyflakes import reporter as modReporter
from pyflakes.checker import Checker
import pyflakes.messages as pym
import pyflakes.api as pyfl

def my_to_list(self):
    lineno = int(self.lineno)
    col = int(self.col)
    name = re.match(".*'(.*)'", self.message % self.message_args).group(1)
    return (lineno, col, name)


def my_exception(self, filename, msg, lineno, offset, text):
    if msg == 'expected an indented block':
        print(filename, msg, lineno, offset, text.strip())
        self.indent_error.append([int(lineno), text])


ERRORS = {}
def my_report(self, messageClass, *args, **kwargs):
    if pym.UndefinedName == messageClass:
        message = messageClass(self.filename, *args, **kwargs)
        self.undefined_names.append(message.to_list())
    elif pym.UnusedImport == messageClass:
        if self.filename not in IGNORE_FILES:
            message = messageClass(self.filename, *args, **kwargs)
            self.unused_imports.append(message.to_list())
    elif pym.UnusedVariable == messageClass:
        message = messageClass(self.filename, *args, **kwargs)
        self.unused_var.append(message.to_list())
    elif pym.RedefinedInListComp == messageClass:
        message = messageClass(self.filename, *args, **kwargs)
        self.shadowed_var.append(message.to_list())
    elif pym.RedefinedWhileUnused == messageClass:
        message = messageClass(self.filename, *args, **kwargs)
        self.dublicated_funcs.append(message.to_list())
    else:
        if self.filename not in IGNORE_FILES:
            print(messageClass(self.filename, *args, **kwargs))
            ERRORS[str(messageClass)] = messageClass(self.filename, *args, **kwargs).to_list()


def reset_lists():
    GLOBAL_CHECKER.undefined_names = []
    GLOBAL_CHECKER.unused_imports = []
    GLOBAL_CHECKER.unused_var = []
    GLOBAL_CHECKER.shadowed_var = []
    GLOBAL_CHECKER.dublicated_funcs = []

modReporter.Reporter.syntaxError = my_exception
GLOBAL_REPORTER = modReporter._makeDefaultReporter()
GLOBAL_REPORTER.indent_error = []

GLOBAL_CHECKER = Checker
GLOBAL_CHECKER.report = my_report
pym.Message.to_list = my_to_list


def get_file_dict():
    file_dict = {}
    for folder_name in USED_FOLDER:
        files_dir = os.path.join(CURRENT_DIR, NO_USED, folder_name, '*.py')
        file_dict.setdefault('no_used', {})[folder_name] = sorted(glob.glob(files_dir))
        files_dir = os.path.join(CURRENT_DIR, NO_UNUSED, folder_name, '*.py')
        file_dict.setdefault('no_unused', {})[folder_name] = sorted(glob.glob(files_dir))
    return file_dict


def put_imports(file_dict):
    for key in file_dict:
        if key == 'no_used':
            output_base = IMPORTS_SPARX_DIR
            second = True
        else:
            second = False
            output_base = IMPORTS_SPHIRE_DIR
        for dir_name in file_dict[key]:
            output_dir = os.path.join(output_base, dir_name)
            try:
                os.makedirs(output_dir)
            except OSError:
                pass
            for file_path in file_dict[key][dir_name]:
                basename = os.path.basename(file_path)
                imports_file = '{0}_imports'.format(file_path.replace(NO_USED, NO_WILDCARDS_DIR).replace(NO_UNUSED, NO_WILDCARDS_DIR))
                with open(imports_file, 'r') as read:
                    imports = ''.join(['import {0}'.format(entry) for entry in read.readlines()])
                with open(file_path, 'r') as read:
                    lines = read.readlines()
                if second:
                    lines.insert(1, imports)
                else:
                    for idx, line in enumerate(lines[:]):
                        if ('import' in line and '__future__' not in line) or FUNCDEF_RE.match(line):
                            lines.insert(idx, imports)
                            break
                with open(os.path.join(output_dir, basename), 'w') as write:
                    write.write(''.join(lines))



def main():
    file_dict = get_file_dict()

    put_imports(file_dict)

if __name__ == '__main__':
    main()
