"""确认核心运行时依赖可导入，并报告 PyTorch 设备状态。"""

import numpy
import sklearn
import sentencepiece
import torch
import transformers
import yaml


def main() -> None:
    versions = {
        "numpy": numpy.__version__,
        "PyYAML": yaml.__version__,
        "scikit-learn": sklearn.__version__,
        "sentencepiece": sentencepiece.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    for name, version in versions.items():
        print(f"{name}: {version}")
    print(f"CUDA available: {torch.cuda.is_available()}")


if __name__ == "__main__":
    main()
