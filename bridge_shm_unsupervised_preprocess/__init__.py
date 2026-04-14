"""bridge_shm_unsupervised_preprocess package.

注意：这里避免在导入包时立即导入 `core`，防止 GUI 启动阶段因
`torch` 等重依赖缺失而直接崩溃。
"""

__all__ = ["BridgeSHMUnsupervisedPreprocessor", "main", "gui_main"]


def __getattr__(name: str):
    if name in {"BridgeSHMUnsupervisedPreprocessor", "main"}:
        from .core import BridgeSHMUnsupervisedPreprocessor, main

        mapping = {
            "BridgeSHMUnsupervisedPreprocessor": BridgeSHMUnsupervisedPreprocessor,
            "main": main,
        }
        return mapping[name]
    if name == "gui_main":
        from .gui_app import main as gui_main

        return gui_main
    raise AttributeError(name)
