from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import sysconfig

try:
    import pybind11
except Exception as e:
    print("pybind11 is required to build the native extension. Please install it first.")
    raise


def get_openmp_args():
    c_args = []
    l_args = []
    if sys.platform.startswith('win'):
        # MSVC
        c_args.append('/openmp')
        # No special link flag needed typically for OpenMP on MSVC
    else:
        c_args.append('-fopenmp')
        l_args.append('-fopenmp')
    return c_args, l_args

c_args, l_args = get_openmp_args()

ext_modules = [
    Extension(
        'native_rolling',
        sources=['src/native/rolling_features.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=c_args + ['-std=c++17'] if not sys.platform.startswith('win') else c_args,
        extra_link_args=l_args,
    )
]

class BuildExt(build_ext):
    c_opts = {}

    def build_extensions(self):
        ct = self.compiler.compiler_type
        for ext in self.extensions:
            if ct == 'msvc':
                ext.extra_compile_args = ext.extra_compile_args + ['/std:c++17']
        build_ext.build_extensions(self)

setup(
    name='native_rolling',
    version='0.1.0',
    description='High-performance rolling feature computation (pybind11 + OpenMP)',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
)
