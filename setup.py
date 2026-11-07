from pathlib import Path
import os

from setuptools import setup

ROOT = Path(__file__).resolve().parent

def _src(name):
    """Return a source path relative to setup.py for setuptools >=77.
    
    setuptools >=77 rejects absolute paths in Extension() sources
    but allows them in include_dirs. This helper relativizes only
    the source file paths against the setup.py directory.
    """
    return str((ROOT / "csrc" / name).relative_to(ROOT))

ext_modules = []
cmdclass = {}


def _cb_flags() -> list[str]:
    """``-DP2B_CB=<n>`` from VLLM_EXL3_P2B_CB, for every compiler in the build."""
    cb = os.environ.get("VLLM_EXL3_P2B_CB")
    return [f"-DP2B_CB={int(cb)}"] if cb else []


def _cxx_std():
    """C++ standard for the extension: torch >= 2.14 headers refuse anything below C++20."""
    override = os.environ.get("VLLM_EXL3_CXX_STD")
    if override:
        return f"-std={override}"
    try:
        import torch

        major, minor = (int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
    except Exception:
        return "-std=c++17"
    return "-std=c++20" if (major, minor) >= (2, 14) else "-std=c++17"

if os.environ.get("VLLM_EXL3_NO_CUDA", "0") != "1":
    try:
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension

        ext_include = os.environ.get("EXL3_EXT_INCLUDE")
        if ext_include:
            ext_include = Path(ext_include)
        else:
            try:
                import exllamav3
                ext_include = Path(exllamav3.__file__).resolve().parent / "exllamav3_ext"
            except Exception:
                ext_include = None

        if not (ext_include and ext_include.is_dir()):
            candidates = [
                Path("/tmp/exllamav3/exllamav3/exllamav3_ext"),
                Path("/tmp/exllamav3_build/exllamav3/exllamav3_ext"),
                Path.home() / "src/glm53-exl3-runtime/exllamav3/exllamav3/exllamav3_ext",
                Path.home() / "exllamav3/exllamav3/exllamav3_ext",
            ]
            for c in candidates:
                if c.is_dir():
                    ext_include = c
                    break

        include_dirs = [str(ROOT / "csrc")]
        if ext_include and ext_include.is_dir():
            include_dirs.append(str(ext_include))
            if (ext_include / "quant").is_dir():
                include_dirs.append(str(ext_include / "quant"))

        ext_modules.append(
            CUDAExtension(
                name="vllm_exl3_c",
                sources=[
                    _src("bindings.cpp"),
                    _src("exl3_gemv.cu"),
                    _src("p2b_batched.cu"),
                    _src("p2b_moe.cu"),
                    _src("exl3_gemm.cu"),
                    _src("exl3_fat_gemm.cu"),
                ],
                include_dirs=include_dirs,
                extra_compile_args={
                    # Expert codebook: 1 = MCG (historical default), 2 = mul1.
                    # Both DeepSeek-V4.1 packs are mul1, so a build for them
                    # needs VLLM_EXL3_P2B_CB=2; the extension then exports
                    # P2B_CB so Python can gate the native path on it instead
                    # of assuming MCG. It has to go to BOTH compilers: the
                    # kernels live in .cu (nvcc) but bindings.cpp reports the
                    # value to Python and is compiled by the host compiler --
                    # defining it only for nvcc yields a mul1 kernel that
                    # advertises MCG.
                    "cxx": ["-O3", _cxx_std()] + _cb_flags(),
                    "nvcc": ["-O3", _cxx_std()] + _cb_flags(),
                },
            )
        )
        cmdclass["build_ext"] = BuildExtension
    except ImportError:
        pass

setup(
    name="vllm-exl3",
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
