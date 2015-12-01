#!/usr/bin/env python

import sys
import os
import subprocess
import argparse

def split(input_file, test_file, grade_file, baseline_file):
    lines = []
    with open(input_file, 'r') as f:
        for line in f:
            lines.append(line)
    f.close()

    with open(test_file, 'w+') as t:
        for l in lines[:-60]:
            print >>t, l,
    t.close()

    with open(grade_file, 'w+') as g:
        for l in lines[-60:]:
            print >>g, l,
    g.close()

    print len(lines)
    pos = lines[-61]
    with open(baseline_file, 'w+') as b:
        for i in range(60):
            print >>b, pos,
    b.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Takes the input file and produces 3 output files: 1) all but the last 60 lines, 2) the last 60 lines, 3) the last line of #1 repeated 60 times')
    parser.add_argument('--input_file', action="store", help="The existing file to split up")
    parser.add_argument('--test_file', action="store", help="All but the last 60 lines will be written here")
    parser.add_argument('--grade_file', action="store", help="The last 60 lines will be written here")
    parser.add_argument('--baseline_file', action="store", help="The 61st-to-last line will be written here 60 times")

    results = parser.parse_args()
    print "Input File:   ", results.input_file
    print "Test File:    ", results.test_file
    print "Grade File:   ", results.grade_file
    print "Baseline File:", results.baseline_file
    if results.input_file is None or results.test_file is None or results.grade_file is None or results.baseline_file is None:
        print "Please supply all arguments. Run with -h for a list."
        exit(1)

    split(results.input_file, results.test_file, results.grade_file, results.baseline_file)
