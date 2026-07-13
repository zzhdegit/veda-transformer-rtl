# ML-M2 Formal GPU Environment

- Python: `3.10.19 | packaged by Anaconda, Inc. | (main, Oct 21 2025, 16:41:31) [MSC v.1929 64 bit (AMD64)]`
- OS: `Windows-10-10.0.26100-SP0`
- PyTorch: `2.10.0.dev20251204+cu128`
- PyTorch CUDA runtime: `12.8`
- CUDA available: `True`
- GPU: `NVIDIA GeForce RTX 5080`
- Compute capability: `(12, 0)`
- BF16 supported: `True`
- nvidia-smi query: `NVIDIA GeForce RTX 5080, 576.88, 16303 MiB, 4301 MiB, 12.0`
- CUDA smoke: `{'status': 'ok', 'bf16_matmul': True, 'peak_allocated_bytes': 23855616}`

## nvidia-smi

```text
Mon Jul 13 20:29:10 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 576.88                 Driver Version: 576.88         CUDA Version: 12.9     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5080      WDDM  |   00000000:01:00.0  On |                  N/A |
| 63%   73C    P0            348W /  360W |   11589MiB /  16303MiB |     84%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            3528    C+G   ...8bbwe\PhoneExperienceHost.exe      N/A      |
|    0   N/A  N/A            6360    C+G   ...\Win64\AC2-Win64-Shipping.exe      N/A      |
|    0   N/A  N/A            8284    C+G   ...kyb3d8bbwe\EdgeGameAssist.exe      N/A      |
|    0   N/A  N/A            9596    C+G   ...\cef.win64\steamwebhelper.exe      N/A      |
|    0   N/A  N/A           11344    C+G   ...5n1h2txyewy\TextInputHost.exe      N/A      |
|    0   N/A  N/A           12460    C+G   ....0.4022.98\msedgewebview2.exe      N/A      |
|    0   N/A  N/A           13952    C+G   ...065.92.x64\msedgewebview2.exe      N/A      |
|    0   N/A  N/A           14236    C+G   ...lpaper_engine\wallpaper64.exe      N/A      |
|    0   N/A  N/A           15992    C+G   ...indows\System32\ShellHost.exe      N/A      |
|    0   N/A  N/A           19932    C+G   ..._cw5n1h2txyewy\SearchHost.exe      N/A      |
|    0   N/A  N/A           20200    C+G   ...IA App\CEF\NVIDIA Overlay.exe      N/A      |
|    0   N/A  N/A           22708    C+G   ...__2p2nqsd0c76g0\app\Codex.exe      N/A      |
|    0   N/A  N/A           25416    C+G   ...r\frontend\Docker Desktop.exe      N/A      |
|    0   N/A  N/A           26180    C+G   ...ice\root\Office16\WINWORD.EXE      N/A      |
|    0   N/A  N/A           27016    C+G   ...y\StartMenuExperienceHost.exe      N/A      |
|    0   N/A  N/A           30268    C+G   ....0.4022.98\msedgewebview2.exe      N/A      |
|    0   N/A  N/A           31712    C+G   ...IA App\CEF\NVIDIA Overlay.exe      N/A      |
|    0   N/A  N/A           34228    C+G   C:\Windows\explorer.exe               N/A      |
|    0   N/A  N/A           34956    C+G   ...App_cw5n1h2txyewy\LockApp.exe      N/A      |
|    0   N/A  N/A           35868    C+G   ...xyewy\ShellExperienceHost.exe      N/A      |
|    0   N/A  N/A           36132    C+G   ...t\Edge\Application\msedge.exe      N/A      |
+-----------------------------------------------------------------------------------------+
```

Official PyTorch install pages list CUDA 12.8 as a supported binary compute platform for Windows/Pip; this run used the existing CUDA 12.8 PyTorch environment rather than the base CPU-only environment.
