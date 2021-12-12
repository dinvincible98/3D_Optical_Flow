from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='pointnet2',
    ext_modules=[
        CUDAExtension('pointnet2_cuda', [
            'cpp/pointnet2_api.cpp',
            
            'cpp/ball_query.cpp', 
            'cpp/ball_query_gpu.cu',
            'cpp/group_points.cpp', 
            'cpp/group_points_gpu.cu',
            'cpp/interpolate.cpp', 
            'cpp/interpolate_gpu.cu',
            'cpp/sampling.cpp', 
            'cpp/sampling_gpu.cu',
        ],
        extra_compile_args={'cxx': ['-g'],
                            'nvcc': ['-O2']})
    ],
    cmdclass={'build_ext': BuildExtension}
)
