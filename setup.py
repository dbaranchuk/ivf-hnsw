import contextlib
from pprint import pprint
from urllib.request import urlretrieve
import sys
import platform
import subprocess
import os

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.command.install_scripts import install_scripts
from distutils.version import LooseVersion


project_dir = os.path.dirname(os.path.abspath(__file__))
python_src = os.path.join(os.curdir, 'python-src')


class custom_install_scripts(install_scripts):
    def run(self):
        print(self.build_dir)
        super().run()


class custom_build_ext(build_ext):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._swig_outdir = None

    def run(self):
        # Store self.inplace flag because it gets overriden somehow
        # by `python setup.py test` pipeline
        self._real_inplace = self.inplace
        print('Inplace:', self.inplace)
        super().run()
        self.run_command('build_py')

    @contextlib.contextmanager
    def set_inplace(self, inplace):
        saved_inplace, self.inplace = self.inplace, inplace
        yield
        self.inplace = saved_inplace

    def build_extension(self, ext):
        interface_temp = os.path.join(self.build_temp, 'interface')
        os.makedirs(interface_temp, exist_ok=True)

        # Download numpy.i dependency to be used by swig
        urlretrieve('https://raw.githubusercontent.com/numpy/numpy/master/tools/swig/numpy.i',
                    os.path.join(interface_temp, 'numpy.i'),)

        # Build only ivfhnsw static library
        build_args = ['--target', 'ivfhnsw']
        cmake_args = []
        env = os.environ.copy()
        subprocess.check_call(['cmake', project_dir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', os.curdir] + build_args, cwd=self.build_temp)

        # Add path to the compiled static libraries
        ext.library_dirs.append(os.path.join(self.build_temp, 'lib'))
        # Add path to the temporary swig interface files directory
        ext.swig_opts.append('-I' + os.path.join(self.build_temp, 'interface'))

        with self.set_inplace(self._real_inplace):
            _swig_outdir = os.path.dirname(self.get_ext_fullpath(ext.name))
        os.makedirs(_swig_outdir, exist_ok=True)
        ext.swig_opts.extend(['-outdir', _swig_outdir])
        print('SWIG outdir:', _swig_outdir)

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
        'install_scripts': custom_install_scripts,
    }
)
