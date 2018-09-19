from pprint import pprint
from urllib.request import urlretrieve
import sys
import platform
import subprocess
import os

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion


python_src = 'python-src'


class custom_build_ext(build_ext):
    def run(self):
        super().run()
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

        ivfhnsw_package_path = os.path.join(self.build_lib, 'ivfhnsw')
        os.makedirs(ivfhnsw_package_path, exist_ok=True)
        ext.swig_opts.extend(['-outdir', ivfhnsw_package_path])

        import numpy
        ext.include_dirs.append(numpy.get_include())
        return super().build_extension(ext)


names = ['wrapper']
ext = [Extension(name='.'.join(['ivfhnsw', '_' + name]),
                 sources=[os.path.join('interface', '.'.join([name, 'i']))],
                 swig_opts=['-Iinclude', '-c++'],
                 include_dirs=['include', os.curdir],
                 libraries=['ivfhnsw', 'hnswlib', 'faiss', 'gomp', 'lapack',],
                 extra_compile_args=['-std=c++11', '-static'],)
                 for name in names]


setup(
    name='ivfhnsw',
    version='0.1',
    ext_modules=ext,
    package_dir={'': python_src},
    packages=find_packages(python_src),
    setup_requires=[
        'pytest-runner',
        'numpy',
    ],
    install_requires=[
        'numpy',
    ],
    tests_require=['pytest>2.8'],
    include_package_data=True,
    cmdclass={
        'build_ext': custom_build_ext,
    }
)
