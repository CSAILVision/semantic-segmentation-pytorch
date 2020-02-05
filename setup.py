import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='mit_semseg',
    version='1.0.0',
    author='MIT CSAIL',
    description='Pytorch implementation for Semantic Segmentation/Scene Parsing on MIT ADE20K dataset',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/CSAILVision/semantic-segmentation-pytorch',
    packages=setuptools.find_packages(),
    classifiers=(
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
    ),
    install_requires=[
        'numpy',
        'torch>=0.4.1',
        'torchvision',
        'opencv-python',
        'yacs',
        'scipy',
        'tqdm'
    ]
)
