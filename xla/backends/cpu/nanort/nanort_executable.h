/* Copyright 2024 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_BACKENDS_CPU_NANORT_NANORT_EXECUTABLE_H_
#define XLA_BACKENDS_CPU_NANORT_NANORT_EXECUTABLE_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/service/executable.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/chain.h"
#include "tsl/platform/threadpool.h"

namespace xla::cpu {

class NanoRtExecutable {
 public:
  // Creates a new instance of the NanoRtExecutable from compatible XLA
  // executable.
  static absl::StatusOr<std::unique_ptr<NanoRtExecutable>> Create(
      std::unique_ptr<Executable> executable,
      std::shared_ptr<tsl::thread::ThreadPool> thread_pool);

  // NanoRtExecutable can be asynchronous and return unavailable async value
  // that becomes available after the execution is complete. It is the caller's
  // responsibility to make sure that arguments, results and temp buffers are
  // alive during execution.
  using ExecuteEvent = tsl::Chain;

  // A non-owning read-only view into the XLA executable's argument buffer.
  class Argument {
   public:
    template <typename T>
    Argument(const T* data, int64_t size);

    template <typename T>
    explicit Argument(absl::Span<const T> data);

    absl::Span<const std::byte> data() const { return data_; }

   private:
    absl::Span<const std::byte> data_;
  };

  // A non-owning writable view into the XLA executable's result buffer.
  class Result {
   public:
    template <typename T>
    Result(T* data, int64_t size);

    template <typename T>
    explicit Result(absl::Span<T> data);

    absl::Span<std::byte> data() const { return data_; }

   private:
    absl::Span<std::byte> data_;
  };

  // A non-owning writable view into the XLA executable's temporary buffer (a
  // buffer that is used by the executable to store intermediate results).
  using PreallocatedTemp = absl::Span<std::byte>;

  tsl::AsyncValueRef<ExecuteEvent> Execute(absl::Span<const Argument> arguments,
                                           absl::Span<const Result> results,
                                           const PreallocatedTemp& temp);

 private:
  NanoRtExecutable(std::unique_ptr<Executable> executable,
                   std::shared_ptr<tsl::thread::ThreadPool> thread_pool,
                   size_t num_allocations,
                   std::vector<size_t> argument_to_allocation_index,
                   std::vector<size_t> result_to_allocation_index,
                   std::optional<size_t> temp_allocation_index);

  std::unique_ptr<Executable> executable_;
  std::shared_ptr<tsl::thread::ThreadPool> thread_pool_;

  size_t num_allocations_;

  // A mapping from the argument/result index to the index of the corresponding
  // allocation (defined by the executable's buffer assignment).
  std::vector<size_t> argument_to_allocation_index_;
  std::vector<size_t> result_to_allocation_index_;

  // Index of the temp allocation.
  std::optional<size_t> temp_allocation_index_;
};

template <typename T>
NanoRtExecutable::Argument::Argument(const T* data, int64_t size)
    : data_(reinterpret_cast<const std::byte*>(data), size * sizeof(T)) {}

template <typename T>
NanoRtExecutable::Argument::Argument(absl::Span<const T> data)
    : Argument(data.data(), data.size()) {}

template <typename T>
NanoRtExecutable::Result::Result(T* data, int64_t size)
    : data_(reinterpret_cast<std::byte*>(data), size * sizeof(T)) {}

template <typename T>
NanoRtExecutable::Result::Result(absl::Span<T> data)
    : Result(data.data(), data.size()) {}

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_NANORT_NANORT_EXECUTABLE_H_