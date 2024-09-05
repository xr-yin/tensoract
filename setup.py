from setuptools import setup, find_packages

setup(
    name='tensoract',
    version='0.1.0',
    author='Xianrui Yin',
    author_email='xianrui.yin@tum.de',
    description='1D open quantum system simulation using tensor networks',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/xr-yin/tensoract',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache-2.0 License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
    install_requires=[
        'torch',
        'numpy',
        'scipy',
        'tqdm',
    ],
)