
## HOW TO REPLACE CUDNN FILES

Below there is our folder structure whene our image was built.
The cudnn files wasn't included because of their weights but they could be downloaded
from https://developer.nvidia.com/rdp/cudnn-archive

Should be chosen the 7.6.3 version for CUDA 10.0, the right distribution is "cuDNN Library for Linux", after downloading it replace the empty folder cudnn-10.0-linux-x64-v7.6.3.30/

```
cudnn-10.0-linux-x64-v7.6.3.30/
.
└── cuda
    ├── include
    │   └── cudnn.h
    ├── lib64
    │   ├── libcudnn.so -> libcudnn.so.7
    │   ├── libcudnn.so.7 -> libcudnn.so.7.6.3
    │   ├── libcudnn.so.7.6.3
    │   └── libcudnn_static.a
    └── NVIDIA_SLA_cuDNN_Support.txt 
```