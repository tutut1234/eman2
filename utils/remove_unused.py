import collections
import re
import glob
import shutil
import os

used_files = (
    'sx3dvariability.py',
    'sxcompute_isac_avg.py',
    'sxcter.py',
    'sxchains.py',
    'e2bdb.py',
    'e2boxer_old.py',
    'e2boxer.py',
    'e2display.py',
    'e2proc3d.py',
    'sxfilterlocal.py',
    'sxgui_cter.py',
    'sxgui_meridien.py',
    'sxgui_unblur.py',
    'sxheader.py',
    'sxisac2.py',
    'sxlocres.py',
    'sxmeridien.py',
    'sxpdb2em.py',
    'sxpipe.py',
    'sxgui.py',
    'sxcpy.py',
    'sxprocess.py',
    'sxproj_compare.py',
    'sxrelion2sphire.py',
    'sxsphire2relion.py',
    'sxrewindow.py',
    'sxrviper.py',
    'sxsort3d_depth.py',
    'sxsort3d.py',
    'sxsummovie.py',
    'sxunblur.py',
    'sxviper.py',
    'sxwindow.py',
    )

template_files = ['../sparx/templates/sxgui_template.py', '../sparx/templates/wikiparser.py']

bin_dir = '../sparx/bin'
lib_dir = '../sparx/libpy'

old_bin_dir = '../sparx/sparx_bin'
new_bin_dir = '../sparx/sphire_bin'
old_lib_dir = '../sparx/sparx_libpy'
new_lib_dir = '../sparx/sphire_libpy'

unused_bin_dir = '../sparx/sphire_bin_unused'
unused_lib_dir = '../sparx/sphire_libpy_unused'
#tmp_lib_dir = '../sparx/sphire_libpy_tmp'

IMPORT_RE = re.compile('^(?:def|class)\s+([^\(]+)')
END_RE = re.compile('^(?:\w|"|\')')
CONSTANT_RE = re.compile('^(\w+)')
START_RE = re.compile('^(?:def)')
SPARX_FUNC_RE = re.compile('[^\w](sparx_[\w_]+)\.([\w_]+)')
COMMENT_RE = re.compile('^\s*#')
MULTILINE1_RE = re.compile('^.*\'\'\'')
MULTILINE1_a_RE = re.compile('^.*\'\'\'.*\'\'\'')
MULTILINE2_RE = re.compile('^.*"""')
MULTILINE2_a_RE = re.compile('^.*""".*"""')

used_libs = []
used_libs_dict = {}

def check_comment(line, comment1, comment2, ignore, unused_functions):
    comment3 = False
    if COMMENT_RE.match(line):
        comment3 = True
    else:
        tmp_ignore = False
        for entry in unused_functions:
            if re.match('(?:def\s+|class\s+){0}'.format(entry), line):
                tmp_ignore = True
                ignore = True
                break
        if ignore and not tmp_ignore and (START_RE.match(line) or END_RE.match(line)):
            ignore = False

        if MULTILINE1_RE.match(line) and not MULTILINE1_a_RE.match(line) and comment1:
            comment1 = False
        elif MULTILINE2_RE.match(line) and not MULTILINE2_a_RE.match(line) and comment2:
            comment2 = False
        elif MULTILINE1_RE.match(line) and not MULTILINE1_a_RE.match(line) and not comment2:
            comment1 = True
        elif MULTILINE2_RE.match(line) and not MULTILINE2_a_RE.match(line) and not comment1:
            comment2 = True

    return comment1, comment2, comment3, ignore

for dirname in [new_bin_dir, new_lib_dir, unused_bin_dir, unused_lib_dir, old_bin_dir, old_lib_dir]:
    try:
        os.mkdir(dirname)
    except OSError as e:
        pass

bin_files = glob.glob('{0}/*'.format(bin_dir))
bin_files.extend(template_files)

# Copy all files to the specific folders
for file_path in sorted(bin_files):
    if os.path.basename(file_path) in used_files:
        if not os.path.exists('{0}/{1}'.format(new_bin_dir, os.path.basename(file_path))):
            shutil.copy2(file_path, new_bin_dir)
    else:
        if file_path in template_files:
            if not os.path.exists('{0}/{1}'.format(new_bin_dir, os.path.basename(file_path))):
                shutil.copy2(file_path, new_bin_dir)
        elif not os.path.exists('{0}/{1}'.format(old_bin_dir, os.path.basename(file_path))):
            shutil.copy2(file_path, old_bin_dir)


# Run through the binary files and identify used library files.
#bin_files = ['{0}/sxgui_meridien.py'.format(new_bin_dir)]
new_bin_files = glob.glob('{0}/*'.format(new_bin_dir))
used_libraries = {}
for file_path in sorted(new_bin_files):
    basename = os.path.basename(file_path)
    print(basename)
    with open(file_path) as read:
        lines = read.readlines()

    # Extract all available functions
    comment1 = False
    comment2 = False
    all_functions_and_classes = []
    for line in lines:
        comment1, comment2, comment3, _ = check_comment(line, comment1, comment2, False, [])
        match = IMPORT_RE.match(line)
        if match and not comment1 and not comment2 and not comment3:
            all_functions_and_classes.append(match.group(1))
        if comment3:
            comment3 = False

    print('DOUBLE_FUNCS:', [item for item, count in collections.Counter(all_functions_and_classes).items() if count > 1])

    # Extract all used functions in the file
    unused_functions = []
    len_unused = -999
    while len(unused_functions) != len_unused:
        len_unused = len(unused_functions)
        bad_idx = []
        tmp_used_functions = []
        for entry in all_functions_and_classes:
            comment1 = False
            comment2 = False
            ignore = False
            tmp_unused_functions = []
            for idx, line in enumerate(lines):
                comment1, comment2, comment3, ignore = check_comment(line, comment1, comment2, ignore, unused_functions)
                if ignore:
                    bad_idx.append(idx)
                re_string = '[^\w^\d^_]{0}[^\w^\d^_^%]'.format(entry)
                re_comp = re.compile(re_string)
                #if re_comp.search(line):
                #    print(line, comment1, comment2, comment3, ignore)
                if re_comp.search(line) and not START_RE.match(line) and not comment1 and not comment2 and not comment3 and not ignore:
                    tmp_used_functions.append(entry)
                if comment3:
                    comment3 = False
            tmp_unused_functions = [entry for entry in all_functions_and_classes if entry not in set(tmp_used_functions)]

        unused_functions.extend(tmp_unused_functions)
        unused_functions = list(set(unused_functions))
    used_functions = [entry for entry in all_functions_and_classes if entry not in set(unused_functions)]
    print('UNUSED FUNCS:', unused_functions)
    bad_idx_set = set(bad_idx)

    # Remove bad lines from file
    used_lines = []
    with open(file_path, 'w') as write1, open(os.path.join(unused_bin_dir, basename), 'w') as write2:
        for idx, line in enumerate(lines):
            if idx in bad_idx_set:
                write2.write(line)
            else:
                used_lines.append(line)
                write1.write(line)

    for line in used_lines:
        match = SPARX_FUNC_RE.findall(line)
        for module, func_name in match:
            used_libraries.setdefault(module, []).append(func_name)

#print('USED LIBRARIES:')
for key, item in used_libraries.items():
    used_libraries[key] = sorted(list(set(item)))
    #print('LIB', key, sorted(list(set(item))))

# Move unused libraries
lib_files = glob.glob('{0}/*.py'.format(lib_dir))
for file_path in sorted(lib_files):
    basename = os.path.basename(file_path)
    if basename.replace('.py', '') in used_libraries:
        if not os.path.exists('{0}/{1}'.format(new_lib_dir, basename)):
            shutil.copy2(file_path, new_lib_dir)
    else:
        if basename == 'sparx.py':
            if not os.path.exists('{0}/{1}'.format(new_lib_dir, basename)):
                shutil.copy2(file_path, new_lib_dir)
        elif not os.path.exists('{0}/{1}'.format(old_lib_dir, basename)):
            shutil.copy2(file_path, old_lib_dir)


# Extract the functions from the libraries
lib_files = glob.glob('{0}/*.py'.format(new_lib_dir))
#lib_files = glob.glob('{0}/sparx_global_def.py'.format(new_lib_dir))
function_dict = {}
tmp_line = 'CONSTANT'
for file_path in lib_files:
    basename = os.path.splitext(os.path.basename(file_path))[0]
    if basename in ('sparx'):
        continue

    with open(file_path) as read:
        lines = read.readlines()

    current_func = None
    comment1 = False
    comment2 = False
    for idx, line in enumerate(lines):
        tmp_func = None
        comment1, comment2, comment3, _ = check_comment(line, comment1, comment2, ignore, unused_functions)

        match_begin = IMPORT_RE.match(line)
        match_end = END_RE.match(line)
        match_const = CONSTANT_RE.match(line)
        if match_begin and not comment1 and not comment2 and not comment3:
            current_func = match_begin.group(1)
        elif match_end and match_const:
            current_func = None
            tmp_func = match_const.group(1)
            if tmp_func.startswith('class') or \
                    tmp_func.startswith('def') or \
                    tmp_func in ('import', 'from', 'global'):
                tmp_func = None
        elif match_end:
            current_func = None

        if current_func:
            function_dict.setdefault(basename, {}).setdefault(current_func, []).append(line)
        if tmp_func:
            function_dict.setdefault(basename, {}).setdefault(tmp_func, []).append(tmp_line)

        if comment3:
            comment3 = False

def recursive_check(used_libraries, function_dict, module, key):
    try:
        lines = function_dict[module][key]
    except KeyError:
        print(key)
        print(used_libraries)
        raise

    for line in lines:
        if line == tmp_line:
            continue
        match = SPARX_FUNC_RE.findall(line)
        for mod, name in match:
            if name in used_libraries[mod]:
                continue
            else:
                used_libraries[mod].append(name)
                recursive_check(used_libraries, function_dict, mod, name)

remove_idx = []
all_used_libraries = {}
all_used_libraries.update(used_libraries)
for module, items in used_libraries.items():
    #print('MODULE', module)
    try:
        function_dict[module]
    except KeyError:
        #print('Remove key:', module)
        remove_idx.append(module)
        continue
    if module == 'sparx_user_functions':
        items = list(function_dict[module].keys())

    #print(sorted(function_dict[module].keys()))
    for name in items:
        #print('NAME', name)
        recursive_check(all_used_libraries, function_dict, module, name)

#print('USED LIBRARIES:')
for key, item in all_used_libraries.items():
    used_libraries[key] = sorted(list(set(item)))
    #print('LIB', key, sorted(list(set(item))))

for file_path in lib_files:
    basename = os.path.splitext(os.path.basename(file_path))[0]
    if basename in ('sparx'):
        continue
    with open(file_path) as read:
        lines = read.readlines()

    with open(file_path, 'w') as write1, open(os.path.join(unused_lib_dir, basename + '.py'), 'w') as write2:
        out = write1
        for line in lines:
            match = IMPORT_RE.match(line)
            if match:
                if match.group(1) in all_used_libraries[basename.replace('.py', '')]:
                    out = write1
                else:
                    out = write2

            out.write(line)
