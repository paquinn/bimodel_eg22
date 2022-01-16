# Identify the blender python directory, find the _exact_ python 
# version your blender install uses by running it with --version.

conda create --name blender python=3.7.7
conda activate blender
conda install numpy scipy sympy astunparse
conda install -c conda-forge pyperclip igl
pip3 install tetgen

# Rename blender's python dir to .old or something
# Symlink in the conda env's python

# For example:
# ln -s /Users/dan/miniconda3/envs/blender/ /Applications/Blender.app/Contents/Resources/2.90/python
