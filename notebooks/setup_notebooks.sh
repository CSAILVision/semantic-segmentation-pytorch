#!/bin/bash

# Start from directory of script
cd "$(dirname "${BASH_SOURCE[0]}")"

# Set up git config filters so huge output of notebooks is not committed.
git config filter.clean_ipynb.clean "$(pwd)/ipynb_drop_output.py"
git config filter.clean_ipynb.smudge cat
git config filter.clean_ipynb.required true

# Set up symlinks for the example notebooks
for DIRNAME in ckpt mit_semseg config notebooks teaser data
do
  ln -sfn ../${DIRNAME} .
done
