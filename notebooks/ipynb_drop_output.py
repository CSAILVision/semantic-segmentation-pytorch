#!/usr/bin/env python

"""
Suppress output and prompt numbers in git version control.

This script will tell git to ignore prompt numbers and cell output
when looking at ipynb files UNLESS their metadata contains:

    "git" : { "keep_output" : true }

The notebooks themselves are not changed.

See also this blogpost: http://pascalbugnion.net/blog/ipython-notebooks-and-git.html.

Usage instructions
==================

1. Put this script in a directory that is on the system's path.
   For future reference, I will assume you saved it in 
   `~/scripts/ipynb_drop_output`.
2. Make sure it is executable by typing the command
   `chmod +x ~/scripts/ipynb_drop_output`.
3. Register a filter for ipython notebooks by
   putting the following line in `~/.config/git/attributes`:
   `*.ipynb  filter=clean_ipynb`
4. Connect this script to the filter by running the following 
   git commands:

   git config --global filter.clean_ipynb.clean ipynb_drop_output
   git config --global filter.clean_ipynb.smudge cat

To tell git NOT to ignore the output and prompts for a notebook,
open the notebook's metadata (Edit > Edit Notebook Metadata). A
panel should open containing the lines:

    {
        "name" : "",
        "signature" : "some very long hash"
    }

Add an extra line so that the metadata now looks like:

    {
        "name" : "",
        "signature" : "don't change the hash, but add a comma at the end of the line",
        "git" : { "keep_outputs" : true }
    }

You may need to "touch" the notebooks for git to actually register a change, if
your notebooks are already under version control.

Notes
=====

Changed by David Bau to make stripping output the default.

This script is inspired by http://stackoverflow.com/a/20844506/827862, but 
lets the user specify whether the ouptut of a notebook should be kept
in the notebook's metadata, and works for IPython v3.0.
"""

import sys
import json

nb = sys.stdin.read()

json_in = json.loads(nb)
nb_metadata = json_in["metadata"]
keep_output = False
if "git" in nb_metadata:
    if "keep_outputs" in nb_metadata["git"] and nb_metadata["git"]["keep_outputs"]:
        keep_output = True
if keep_output:
    sys.stdout.write(nb)
    exit() 


ipy_version = int(json_in["nbformat"])-1 # nbformat is 1 more than actual version.

def strip_output_from_cell(cell):
    if "outputs" in cell:
        cell["outputs"] = []
    if "prompt_number" in cell:
        del cell["prompt_number"]
    if "execution_count" in cell:
        cell["execution_count"] = None


if ipy_version == 2:
    for sheet in json_in["worksheets"]:
        for cell in sheet["cells"]:
            strip_output_from_cell(cell)
else:
    for cell in json_in["cells"]:
        strip_output_from_cell(cell)

json.dump(json_in, sys.stdout, sort_keys=True, indent=1, separators=(",",": "))
