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


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_SWIG_OUTDIR=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)


class custom_build_ext(build_ext):
    def run(self):
        super().run()

    def build_extension(self, ext):
        env = os.environ.copy()
        cmake_args = []
        build_args = []
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
        r = super().build_extension(ext)

        self.distribution.py_modules.append('ivfhnsw')
        self.run_command('build_py')
        return r


paths = ['interface/ivfhnsw.i']

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
    include_package_data=True,
    cmdclass={
        'build_ext': custom_build_ext,
    }
)
