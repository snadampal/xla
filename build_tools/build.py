#!/usr/bin/python3
# Copyright 2024 The OpenXLA Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""XLA build script for use in CI.

This script is used for the Kokoro builds of XLA, but aims to be as agnostic to
the specifics of the VM as possible. The only Kokoro-specific things that are
assumed are:
  * that `KOKORO_JOB_NAME` is set, which is used to decide what build to run.
  * and all code ends up in `$PWD/github/$REPO_NAME`.
The script also assumes that the working directory never changes modulo `cd`ing
into the repo that should be built (mostly `github/xla`, but also JAX and TF).
"""
import dataclasses
import enum
import logging
import os
import subprocess
import sys
from typing import Any, Dict, List, Tuple


# TODO(ddunleavy): move this to the bazelrc
_DEFAULT_BAZEL_OPTIONS = dict(
    test_output="errors",
    verbose_failures=True,
    keep_going=True,
    nobuild_tests_only=True,
    profile="profile.json.gz",
    flaky_test_attempts=3,
    jobs=150,
    bes_upload_mode="fully_async",
)

_KW_ONLY_IF_PYTHON310 = {"kw_only": True} if sys.version_info >= (3, 10) else {}
_XLA_DEFAULT_TARGET_PATTERNS = (
    "//xla/...",
    "//build_tools/...",
    "@tsl//tsl/...",
)


def sh(args, check=True, **kwargs):
  logging.info("Starting process: %s", " ".join(args))
  return subprocess.run(args, check=check, **kwargs)


def _dict_to_cli_options(d: Dict[str, Any]) -> List[str]:
  # pylint: disable=g-bool-id-comparison
  return [f"--{k}" if v is True else f"--{k}={v}" for k, v in d.items()]


def _write_to_sponge_config(key, value) -> None:
  with open("custom_sponge_config.csv", "a") as f:
    f.write(f"{key},{value}\n")


class BuildType(enum.Enum):
  CPU_X86 = enum.auto()
  CPU_ARM64 = enum.auto()
  GPU = enum.auto()
  GPU_CONTINUOUS = enum.auto()

  JAX_CPU = enum.auto()
  JAX_GPU = enum.auto()

  TENSORFLOW_CPU = enum.auto()
  TENSORFLOW_GPU = enum.auto()


@dataclasses.dataclass(frozen=True, **_KW_ONLY_IF_PYTHON310)
class Build:
  """Class representing a build of XLA."""

  type_: BuildType
  repo: str
  image_url: str
  target_patterns: Tuple[str, ...]
  configs: Tuple[str, ...] = ()
  build_tag_filters: Tuple[str, ...] = ()
  test_tag_filters: Tuple[str, ...] = ()
  action_env: Dict[str, Any] = dataclasses.field(default_factory=dict)
  test_env: Dict[str, Any] = dataclasses.field(default_factory=dict)
  options: Dict[str, Any] = dataclasses.field(default_factory=dict)
  extra_setup_commands: Tuple[List[str], ...] = ()

  def bazel_test_command(self) -> List[str]:
    """Returns a bazel test command for this build.

    Returns: List of command line arguments
    """
    options = _dict_to_cli_options(self.options)
    configs = [f"--config={config}" for config in self.configs]
    build_tag_filters = (
        f"--build_tag_filters={','.join(self.build_tag_filters)}"
    )
    test_tag_filters = f"--test_tag_filters={','.join(self.test_tag_filters)}"
    action_env = [f"--action_env={k}={v}" for k, v in self.action_env.items()]
    test_env = [f"--test_env={k}={v}" for k, v in self.test_env.items()]

    tag_filters = [build_tag_filters, test_tag_filters]
    all_options = tag_filters + configs + action_env + test_env + options
    return ["bazel", "test", *all_options, "--", *self.target_patterns]

  def docker_run_command(self, *, command: str, **kwargs: Any) -> List[str]:
    options = _dict_to_cli_options(kwargs)

    return ["docker", "run", *options, self.image_url, command]

  def commands(self) -> List[List[str]]:
    """Returns list of commands for a build."""
    cmds = []
    cmds.append(["./github/xla/.kokoro/generate_index_html.sh", "index.html"])
    if self.repo != "openxla/xla":
      _, repo_name = self.repo.split("/")

      # pyformat:disable
      cmds.append(["git", "clone", "--depth=1",
                   f"https://github.com/{self.repo}", f"./github/{repo_name}"])
      # pyformat:enable

    cmds.extend(self.extra_setup_commands)

    # pyformat:disable

    if self.type_ == BuildType.CPU_ARM64:
      # We would need to install parallel, but `apt` hangs regularly on Kokoro
      # VMs due to yaqs/eng/q/4506961933928235008
      cmds.append(["docker", "pull", self.image_url])
    else:
      # This is a slightly odd use of parallel, we aren't doing anything besides
      # retrying after 15 seconds up to 3 times if `docker pull` fails.
      cmds.append(["parallel", "--ungroup", "--retries", "3", "--delay", "15",
                   "docker", "pull", ":::", self.image_url])

    container_name = "xla_ci"
    _, repo_name = self.repo.split("/")

    cmds.append(
        self.docker_run_command(command="bash", detach=True,
                                name=container_name, rm=True, interactive=True,
                                tty=True, volume="./github:/github",
                                workdir=f"/github/{repo_name}"))
    # pyformat:enable
    docker_exec = lambda cmd: ["docker", "exec", container_name, *cmd]
    cmds.append(docker_exec(self.bazel_test_command()))
    cmds.append(docker_exec(["bazel", "analyze-profile", "profile.json.gz"]))
    cmds.append(["docker", "stop", container_name])

    return cmds


def _tag_filters_for_compute_capability(
    compute_capability: int,
) -> Tuple[str, ...]:
  tag_filters = (f"requires-gpu-sm{compute_capability}-only",)
  for cc in (60, 70, 80, 90):
    if compute_capability >= cc:
      tag_filters += (f"requires-gpu-sm{cc}",)
    else:
      tag_filters += (f"-requires-gpu-sm{cc}",)
      tag_filters += (f"-requires-gpu-sm{cc}-only",)
  return tag_filters


_DEFAULT_IMAGE = "gcr.io/tensorflow-sigs/build:latest-python3.11"

# TODO(b/338885148): Remove this once the TF containers have cuDNN 9
_CUDNN_9_IMAGE = "gcr.io/tensorflow-sigs/build@sha256:0a9728e258d7e0e5830d1960a65968ffdc1d138af5441e30948918e0d50ab2c7"

_ARM64_JAX_MULTI_PYTHON_IMAGE = "us-central1-docker.pkg.dev/tensorflow-sigs/tensorflow/build-arm64:jax-latest-multi-python"


def nvidia_gpu_build_with_compute_capability(
    *, type_: BuildType, configs: Tuple[str, ...], compute_capability: int
) -> Build:
  extra_gpu_tags = _tag_filters_for_compute_capability(compute_capability)
  return Build(
      type_=type_,
      repo="openxla/xla",
      image_url=_CUDNN_9_IMAGE,
      target_patterns=_XLA_DEFAULT_TARGET_PATTERNS,
      configs=configs,
      test_tag_filters=("-no_oss", "requires-gpu-nvidia", "gpu")
      + extra_gpu_tags,
      build_tag_filters=("-no_oss", "requires-gpu-nvidia", "gpu"),
      options=dict(
          run_under="//tools/ci_build/gpu_build:parallel_gpu_execute",
          repo_env=f"TF_CUDA_COMPUTE_CAPABILITIES={compute_capability/10}",
          **_DEFAULT_BAZEL_OPTIONS,
      ),
      extra_setup_commands=(
          ["nvidia-smi"],
          # TODO(b/338885148): Remove this after TF was updated to cuDNN 9
          [
              "sed",
              "-i",
              r"s/@sigbuild-r2\.17-clang_/@sigbuild-r2.17-clang-cudnn9_/g",
              "github/xla/.bazelrc",
          ],
          [
              "sed",
              "-i",
              r"s/8\.9\.7\.29/9.1.1/g",
              "github/xla/.bazelrc",
          ],
      ),
  )


cpu_x86_tag_filter = (
    "-no_oss",
    "-gpu",
    "-requires-gpu-nvidia",
    "-requires-gpu-amd",
)
_CPU_X86_BUILD = Build(
    type_=BuildType.CPU_X86,
    repo="openxla/xla",
    image_url=_DEFAULT_IMAGE,
    configs=("warnings", "nonccl", "rbe_linux_cpu"),
    target_patterns=_XLA_DEFAULT_TARGET_PATTERNS,
    build_tag_filters=cpu_x86_tag_filter,
    test_tag_filters=cpu_x86_tag_filter,
    options=_DEFAULT_BAZEL_OPTIONS,
)

cpu_arm_tag_filter = (
    "-no_oss",
    "-gpu",
    "-requires-gpu-nvidia",
    "-requires-gpu-amd",
    "-not_run:arm",
)
_CPU_ARM64_BUILD = Build(
    type_=BuildType.CPU_ARM64,
    repo="openxla/xla",
    image_url=_ARM64_JAX_MULTI_PYTHON_IMAGE,
    configs=("warnings", "rbe_cross_compile_linux_arm64_xla", "nonccl"),
    target_patterns=_XLA_DEFAULT_TARGET_PATTERNS,
    options={**_DEFAULT_BAZEL_OPTIONS, "build_tests_only": True},
    build_tag_filters=cpu_arm_tag_filter,
    test_tag_filters=cpu_arm_tag_filter,
)
# TODO(ddunleavy): Setup additional build for a100 tests once L4 RBE is ready.
_GPU_BUILD = nvidia_gpu_build_with_compute_capability(
    type_=BuildType.GPU,
    configs=("warnings", "rbe_linux_cuda_nvcc"),
    compute_capability=75,
)

_JAX_CPU_BUILD = Build(
    type_=BuildType.JAX_CPU,
    repo="google/jax",
    image_url=_DEFAULT_IMAGE,
    configs=(
        "avx_posix",
        "mkl_open_source_only",
        "rbe_cpu_linux_py3.12",
        "tensorflow_testing_rbe_linux",
    ),
    target_patterns=("//tests:cpu_tests", "//tests:backend_independent_tests"),
    test_env=dict(
        JAX_NUM_GENERATED_CASES=25,
        JAX_SKIP_SLOW_TESTS=1,
    ),
    options=dict(
        **_DEFAULT_BAZEL_OPTIONS, override_repository="xla=/github/xla"
    ),
)

_JAX_GPU_BUILD = Build(
    type_=BuildType.JAX_GPU,
    repo="google/jax",
    image_url=_DEFAULT_IMAGE,
    configs=(
        "avx_posix",
        "mkl_open_source_only",
        "rbe_linux_cuda12.3_nvcc_py3.10",
        "tensorflow_testing_rbe_linux",
    ),
    target_patterns=("//tests:gpu_tests", "//tests:backend_independent_tests"),
    build_tag_filters=("-multiaccelerator",),
    test_tag_filters=("-multiaccelerator",),
    test_env=dict(
        JAX_SKIP_SLOW_TESTS=1,
        TF_CPP_MIN_LOG_LEVEL=0,
        JAX_EXCLUDE_TEST_TARGETS="PmapTest.testSizeOverflow",
    ),
    options=dict(
        **_DEFAULT_BAZEL_OPTIONS, override_repository="xla=/github/xla"
    ),
)

_TENSORFLOW_CPU_BUILD = Build(
    type_=BuildType.TENSORFLOW_CPU,
    repo="tensorflow/tensorflow",
    image_url=_DEFAULT_IMAGE,
    configs=(
        "release_cpu_linux",
        "rbe_linux_cpu",
        "linux_cpu_pycpp_test_filters",
    ),
    target_patterns=(
        "//tensorflow/compiler/...",
        "-//tensorflow/compiler/tf2tensorrt/...",
        "//tensorflow/python/...",
        "-//tensorflow/python/distribute/...",
        "-//tensorflow/python/compiler/tensorrt/...",
    ),
    options=dict(
        verbose_failures=True,
        test_output="errors",
        override_repository="xla=/github/xla",
        profile="profile.json.gz",
    ),
)

_TENSORFLOW_GPU_BUILD = Build(
    type_=BuildType.TENSORFLOW_GPU,
    repo="tensorflow/tensorflow",
    image_url=_DEFAULT_IMAGE,
    configs=(
        "release_gpu_linux",
        "rbe_linux_cuda",
        "linux_cuda_pycpp_test_filters",
    ),
    target_patterns=(
        "//tensorflow/compiler/...",
        "-//tensorflow/compiler/tf2tensorrt/...",
        "//tensorflow/python/...",
        "-//tensorflow/python/distribute/...",
        "-//tensorflow/python/compiler/tensorrt/...",
    ),
    build_tag_filters=("-no_oss", "+gpu"),
    test_tag_filters=("-no_oss", "+gpu"),
    options=dict(
        verbose_failures=True,
        test_output="errors",
        override_repository="xla=/github/xla",
        profile="profile.json.gz",
    ),
)

_KOKORO_JOB_NAME_TO_BUILD_MAP = {
    "tensorflow/xla/linux/arm64/build_cpu": _CPU_ARM64_BUILD,
    "tensorflow/xla/linux/cpu/build_cpu": _CPU_X86_BUILD,
    "tensorflow/xla/linux/gpu/build_gpu": _GPU_BUILD,
    "tensorflow/xla/linux/github_continuous/arm64/build_cpu": _CPU_ARM64_BUILD,
    "tensorflow/xla/linux/github_continuous/build_gpu": _GPU_BUILD,
    "tensorflow/xla/linux/github_continuous/build_cpu": _CPU_X86_BUILD,
    "tensorflow/xla/jax/cpu/build_cpu": _JAX_CPU_BUILD,
    "tensorflow/xla/jax/gpu/build_gpu": _JAX_GPU_BUILD,
    "tensorflow/xla/tensorflow/cpu/build_cpu": _TENSORFLOW_CPU_BUILD,
    "tensorflow/xla/tensorflow/gpu/build_gpu": _TENSORFLOW_GPU_BUILD,
}


def main():
  logging.basicConfig()
  logging.getLogger().setLevel(logging.INFO)
  kokoro_job_name = os.getenv("KOKORO_JOB_NAME")
  build = _KOKORO_JOB_NAME_TO_BUILD_MAP[kokoro_job_name]

  for cmd in build.commands():
    sh(cmd)


if __name__ == "__main__":
  main()
