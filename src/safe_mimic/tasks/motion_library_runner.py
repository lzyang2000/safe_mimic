"""Runner integration for tracking a library rather than one bundled motion."""

from mjlab.rl import MjlabOnPolicyRunner
from rsl_rl.env import VecEnv


class MotionLibraryOnPolicyRunner(MjlabOnPolicyRunner):
  """Run PPO without the single-motion ONNX exporter.

  mjlab passes ``registry_name`` to every tracking task. The base runner does
  not accept it, while the upstream tracking runner uses it when exporting one
  reference motion. A packed library has no single motion to bundle, so this
  runner accepts the registry metadata and deliberately leaves it unused.
  """

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
    registry_name: str | None = None,
  ) -> None:
    del registry_name
    super().__init__(env, train_cfg, log_dir, device)


__all__ = ["MotionLibraryOnPolicyRunner"]
