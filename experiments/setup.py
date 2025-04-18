from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy # Needed for include dirs if using numpy types (though not strictly required for the code above)

# Define compiler flags (optional but recommended)
compile_args = ["-O3", "-march=native"] # O3 optimization, optimize for local machine
# Optional: Add fast-math if needed, but test correctness carefully!
# compile_args.append("-ffast-math")

extensions = [
    Extension(
        "emd_1d_cython", # Name of the resulting module
        ["emd_1d_cython.pyx"], # List of .pyx source files
        include_dirs=[numpy.get_include()], # Include NumPy headers if using NumPy C-API features
        extra_compile_args=compile_args,
        # extra_link_args=[], # Add linker args if needed (e.g., OpenMP)
    )
]

setup(
    name="EMD Cython Module",
    ext_modules=cythonize(
        extensions,
        compiler_directives={'language_level': "3"} # Ensure Python 3 semantics
    ),
    zip_safe=False, # Generally good practice for Cython extensions
)