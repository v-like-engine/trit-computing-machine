"""
Setup script for ternary computing library
"""

import os
import sys
from pathlib import Path
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            import subprocess
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build extensions")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        import subprocess
        import shutil

        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}'
        ]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        cmake_args += [f'-DCMAKE_BUILD_TYPE={cfg}']
        build_args += ['--', '-j4']

        env = os.environ.copy()
        env['CXXFLAGS'] = f'{env.get("CXXFLAGS", "")} -DVERSION_INFO=\\"{self.distribution.get_version()}\\"'

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(
            ['cmake', ext.sourcedir] + cmake_args,
            cwd=self.build_temp,
            env=env
        )
        subprocess.check_call(
            ['cmake', '--build', '.'] + build_args,
            cwd=self.build_temp
        )


# Read README for long description
readme_file = Path(__file__).parent / 'README.md'
long_description = readme_file.read_text() if readme_file.exists() else ''

setup(
    name='ternary-computing',
    version='0.1.0',
    author='Ternary Computing Research',
    description='Simulation library for ternary computer systems',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(where='python'),
    package_dir={'': 'python'},
    ext_modules=[CMakeExtension('ternary._ternary_core')],
    cmdclass={'build_ext': CMakeBuild},
    python_requires='>=3.7',
    install_requires=[
        'numpy>=1.19.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov',
            'black',
            'mypy',
        ],
        'proto': [
            'protobuf>=3.19.0',
            'grpcio>=1.40.0',
            'grpcio-tools>=1.40.0',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: C++',
    ],
    keywords='ternary computing balanced-ternary setun simulation',
    project_urls={
        'Source': 'https://github.com/v-like-engine/trit-computing-machine',
    },
)
