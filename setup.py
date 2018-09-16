from pprint import pprint
from urllib.request import urlretrieve
import sys
import platform
import subprocess
import os
import pathlib

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion


class custom_build_ext(build_ext):
    def __init__(self, *args, **kwargs):
        self._swig_generated_modules = []
        super().__init__(*args, **kwargs)

    def run(self):
        super().run()
        self.distribution.py_modules.extend(self._swig_generated_modules)
        self.run_command('build_py')

    def build_extension(self, ext):
        env = os.environ.copy()
        cmake_args = []
        build_args = ['--target', 'ivfhnsw']
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        interface_temp = os.path.join(self.build_temp, 'interface')
        os.makedirs(interface_temp, exist_ok=True)

        urlretrieve('https://raw.githubusercontent.com/numpy/numpy/master/tools/swig/numpy.i',
                    os.path.join(interface_temp, 'numpy.i'),)
        subprocess.check_call(['cmake', os.path.abspath(os.curdir)] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)
        ext.library_dirs.append(os.path.join(self.build_temp, 'lib'))
        ext.swig_opts.append('-I' + os.path.join(self.build_temp, 'interface'))
        self._swig_generated_modules.append(ext.name.lstrip('_'))
        return super().build_extension(ext)


paths = ['interface/wrapper.i']

ext = [Extension(name='_' + os.path.splitext(os.path.basename(path))[0],
                 sources=[str(path)],
                 swig_opts=['-Iinclude', '-c++'],
                 include_dirs=['include', os.curdir],
                 libraries=['ivfhnsw', 'hnswlib', 'faiss', 'gomp', 'lapack',],
                 extra_compile_args=['-std=c++11', '-static'],)
                 for path in paths]

setup(
    name='ivfhnsw',
    version='0.1',
    ext_modules=ext,
    package_dir={'': 'interface'},
    py_modules=[],
    setup_requires=['pytest-runner'],
    install_requires=[
        'numpy',
    ],
    tests_require=['pytest>2.8'],
    include_package_data=True,
    cmdclass={
        'build_ext': custom_build_ext,
    }
)
