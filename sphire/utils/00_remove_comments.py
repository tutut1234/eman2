#!/usr/bin/env python
import os
import shutil
import re
import glob

USED_FOLDER = ('bin', 'libpy', 'templates')
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
NO_COMMENTS_DIR = os.path.join(CURRENT_DIR, 'NO_COMMENTS')
shutil.rmtree(NO_COMMENTS_DIR)

# Regular expressions
BLOCK_STRING_SINGLE_START_RE = re.compile('\s*\'\'\'')
BLOCK_STRING_DOUBLE_START_RE = re.compile('\s*"""')
BLOCK_STRING_SINGLE_RE = re.compile('\'\'\'')
BLOCK_STRING_DOUBLE_RE = re.compile('"""')
USAGE_RE = re.compile('\s*(?:usage|USAGE)\s*=\s*')
RETURN_RE = re.compile('\s*(?:usage|USAGE)\s*=\s*')
COMMENT_RE = re.compile('^\s*#')
FUNCDEF_RE = re.compile('^\s*(?:def|class)\s+([^\(]+)')
INDENT_RE = re.compile('^(\s*)')

FILE_NAME = None
PRINT_LINE = None
def print_all_info(**kwargs):
    if PRINT_LINE:
        if kwargs['line_idx'] in PRINT_LINE:
            for key in sorted(kwargs.keys()):
                print(key, ':', kwargs[key])
            print('')

def get_file_dict(file_name=None):
    file_dict = {}
    if file_name:
        file_dict[os.path.basename(file_name)] = [file_name]
    else:
        for folder_name in USED_FOLDER:
            files_dir = os.path.join(CURRENT_DIR, '..', folder_name, '*.py')
            file_dict[folder_name] = sorted(glob.glob(files_dir))
    return file_dict

def main():
    file_dict = get_file_dict(FILE_NAME)

    for dir_name, file_names in file_dict.items():
        output_dir = os.path.join(NO_COMMENTS_DIR, dir_name)
        output_dir_comment = os.path.join(NO_COMMENTS_DIR, dir_name, 'comment')
        os.makedirs(output_dir_comment)
        for file_path in file_names:
            basename = os.path.basename(file_path)
            output_file_name = os.path.join(output_dir, basename)
            output_file_name_comment = os.path.join(output_dir_comment, '{0}_comment.py'.format(os.path.splitext(basename)[0]))

            with open(file_path, 'r') as read:
                lines = read.readlines()
            comment1 = False
            comment2 = False
            usage = False
            docstring = False
            begin = False
            string_match = False
            comment = 0
            used_lines = []
            comments_lines = []
            for line_idx, line in enumerate(lines):
                do_write = False
                match_1_start = BLOCK_STRING_SINGLE_START_RE.match(line)
                match_1 = BLOCK_STRING_SINGLE_RE.findall(line)
                match_2_start = BLOCK_STRING_DOUBLE_START_RE.match(line)
                match_2 = BLOCK_STRING_DOUBLE_RE.findall(line)
                is_comment = bool(COMMENT_RE.match(line))
                is_usage = bool(USAGE_RE.match(line))
                is_doc = False
                is_first = False
                for i in range(6):
                    if line_idx - i < 0:
                        break
                    elif RETURN_RE.match(lines[line_idx-i]) and (match_1 or match_2):
                        break
                    elif FUNCDEF_RE.match(lines[line_idx-i]) and (match_1 or match_2):
                        is_doc = True
                        break
                is_string = bool((match_1 and not match_1_start) or (match_2 and not match_2_start) and not comment1 and not comment2)

                if FUNCDEF_RE.match(line) and not begin and not comment2 and not comment1:
                    begin = True
                elif FUNCDEF_RE.match(line) and not begin and (comment2 or comment1):
                    print("WOHOOO CHECK THIS OUT: MULTILINE COMMENT BEFORE FIRST FUNCTION", basename, line_idx, line)
                elif not begin:
                    pass
                print_all_info(
                    line_idx=line_idx,
                    line=line,
                    comment1=comment1,
                    comment2=comment2,
                    usage=usage,
                    do_write=do_write,
                    docstring=docstring,
                    string_match=string_match,
                    is_doc=is_doc,
                    is_usage=is_usage,
                    is_string=is_string,
                    is_comment=is_comment,
                    begin=begin
                    )

                if is_comment:
                    pass
                elif match_1 and not comment2 and not comment1:
                    if len(match_1) % 2 == 1:
                        is_first = True
                        if string_match:
                            string_match = False
                        elif is_string:
                            string_match = True
                        elif usage:
                            usage = False
                        elif is_usage:
                            usage = True
                        elif docstring:
                            docstring = False
                        elif is_doc:
                            docstring = True
                        else:
                            comment1 = True
                elif match_2 and not comment1 and not comment2:
                    if len(match_2) % 2 == 1:
                        is_first = True
                        if string_match:
                            string_match = False
                        elif is_string:
                            string_match = True
                        elif usage:
                            usage = False
                        elif is_usage:
                            usage = True
                        elif docstring:
                            docstring = False
                        elif is_doc:
                            docstring = True
                        else:
                            comment2 = True
                elif match_1:
                    if len(match_1) % 2 == 1:
                        comment1 = False
                        do_write = True
                elif match_2:
                    if len(match_2) % 2 == 1:
                        comment2 = False
                        do_write = True

                if (comment1 or comment2 or do_write) and begin:
                    if is_first:
                        string = INDENT_RE.match(line).group(1)
                        used_lines.append('{0}"""Multiline Comment{1}"""\n'.format(string, comment))
                        comments_lines.append('{0}{1}\n'.format(line.strip(), comment))
                        comment += 1
                        wrote_comment = True
                    else:
                        comments_lines.append(line)
                        wrote_comment = True
                else:
                    used_lines.append(line)

                print_all_info(
                    line_idx=line_idx,
                    line=line,
                    comment1=comment1,
                    comment2=comment2,
                    usage=usage,
                    docstring=docstring,
                    do_write=do_write,
                    string_match=string_match,
                    is_doc=is_doc,
                    is_usage=is_usage,
                    is_string=is_string,
                    is_comment=is_comment,
                    begin=begin
                    )

            if used_lines:
                with open(output_file_name, 'w') as write_out:
                    write_out.write(''.join(used_lines))
            if comments_lines:
                with open(output_file_name_comment, 'w') as write_comment:
                    write_comment.write(''.join(comments_lines))


if __name__ == '__main__':
    main()
