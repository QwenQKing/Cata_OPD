import os
from importlib.metadata import PackageNotFoundError, version

def get_version(pkg):
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None

package_name = "vllm"
package_version = get_version(package_name)

if "ROCM_PATH" in os.environ:
    import re

    package_version = version(package_name)
    package_version = re.match(r"(\d+\.\d+\.?\d*)", package_version).group(1)
else:
    package_version = get_version(package_name)

if package_version <= "0.6.3":
    vllm_mode = "customized"
    from .fire_vllm_rollout import FIREvLLMRollout  
    from .vllm_rollout import vLLMRollout  
else:
    vllm_mode = "spmd"
    from .vllm_rollout_spmd import vLLMAsyncRollout, vLLMRollout
