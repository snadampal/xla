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

#include "xla/translate/mhlo_to_hlo/module_attributes_exporter.h"

#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "xla/service/hlo_module_config.h"

namespace mlir {
namespace mhlo {
namespace {

constexpr char kMhloNumPartitions[] = "mhlo.num_partitions";
constexpr char kMhloNumReplicas[] = "mhlo.num_replicas";

static std::vector<int64_t> ConvertDenseIntAttr(
    mlir::DenseIntElementsAttr attr) {
  auto values = attr.getValues<int64_t>();
  return {values.begin(), values.end()};
}

int FindRootInstructionIndex(xla::HloModuleProto& hlo_module, int& root_index) {
  for (const auto& [i, instruction] :
       llvm::enumerate(hlo_module.computations(0).instructions())) {
    if (instruction.id() == hlo_module.mutable_computations(0)->root_id()) {
      root_index = i;
      break;
    }
  }
  return root_index;
};

}  // namespace

void ExportHloModuleConfig(xla::HloModuleConfig& config,
                           mlir::ModuleOp module) {
  if (auto num_partitions =
          module->getAttrOfType<mlir::IntegerAttr>(kMhloNumPartitions)) {
    config.set_num_partitions(num_partitions.getInt());
  }
  if (auto num_replicas =
          module->getAttrOfType<mlir::IntegerAttr>(kMhloNumReplicas)) {
    config.set_replica_count(num_replicas.getInt());
  }
}

absl::Status ExportModuleEntryComputationParameterLayouts(
    const mlir::ArrayAttr& xla_entry_computation_parameter_layout,
    xla::HloModuleProto& hlo_module) {
  for (auto [i, parameter_layout] :
       llvm::enumerate(xla_entry_computation_parameter_layout)) {
    if (auto tuple_parameter_layout =
            mlir::dyn_cast<mlir::ArrayAttr>(parameter_layout)) {
      for (auto [j, tuple_element_parameter_layout] :
           llvm::enumerate(tuple_parameter_layout.getValue())) {
        auto parameter_layout_attr =
            mlir::dyn_cast_or_null<mlir::DenseIntElementsAttr>(
                tuple_element_parameter_layout);
        if (!parameter_layout_attr) {
          return absl::InvalidArgumentError(
              "Multi-level nested parameter layout is not supported.");
        }
        std::vector<int64_t> layout_dims =
            ConvertDenseIntAttr(parameter_layout_attr);
        hlo_module.mutable_host_program_shape()
            ->mutable_parameters()
            ->at(i)
            .mutable_tuple_shapes(j)
            ->mutable_layout()
            ->mutable_minor_to_major()
            ->Assign(layout_dims.begin(), layout_dims.end());
        hlo_module.mutable_computations(0)
            ->mutable_program_shape()
            ->mutable_parameters()
            ->at(i)
            .mutable_tuple_shapes(j)
            ->mutable_layout()
            ->mutable_minor_to_major()
            ->Assign(layout_dims.begin(), layout_dims.end());
        hlo_module.mutable_computations(0)
            ->mutable_instructions(i)
            ->mutable_shape()
            ->mutable_tuple_shapes(j)
            ->mutable_layout()
            ->mutable_minor_to_major()
            ->Assign(layout_dims.begin(), layout_dims.end());
      }
    } else {
      std::vector<int64_t> layout_dims = ConvertDenseIntAttr(
          mlir::cast<mlir::DenseIntElementsAttr>(parameter_layout));
      hlo_module.mutable_host_program_shape()
          ->mutable_parameters()
          ->at(i)
          .mutable_layout()
          ->mutable_minor_to_major()
          ->Assign(layout_dims.begin(), layout_dims.end());
      hlo_module.mutable_computations(0)
          ->mutable_program_shape()
          ->mutable_parameters()
          ->at(i)
          .mutable_layout()
          ->mutable_minor_to_major()
          ->Assign(layout_dims.begin(), layout_dims.end());
      hlo_module.mutable_computations(0)
          ->mutable_instructions(i)
          ->mutable_shape()
          ->mutable_layout()
          ->mutable_minor_to_major()
          ->Assign(layout_dims.begin(), layout_dims.end());
    }
  }
  return absl::OkStatus();
}

absl::Status ExportModuleEntryComputationParameterTiles(
    const mlir::ArrayAttr& xla_entry_computation_parameter_tiles,
    xla::HloModuleProto& hlo_module) {
  for (auto [i, parameter_tile_wrapper] :
       llvm::enumerate(xla_entry_computation_parameter_tiles)) {
    auto parameter_tile = mlir::cast<mlir::ArrayAttr>(parameter_tile_wrapper);

    // Skip empty parameter tile.
    if (parameter_tile.empty()) continue;

    if (auto tuple_parameter_tile =
            mlir::dyn_cast<mlir::ArrayAttr>(parameter_tile[0])) {
      for (auto [j, tuple_element_parameter_tile_wrapper] :
           llvm::enumerate(parameter_tile)) {
        if (!mlir::cast<mlir::ArrayAttr>(tuple_element_parameter_tile_wrapper)
                 .empty()) {
          auto parameter_tile_attr =
              mlir::dyn_cast_or_null<mlir::DenseIntElementsAttr>(
                  mlir::cast<mlir::ArrayAttr>(
                      tuple_element_parameter_tile_wrapper)[0]);
          if (!parameter_tile_attr) {
            return absl::InvalidArgumentError(
                "Multi-level nested parameter tile is not supported.");
          }
          std::vector<int64_t> tile_dims =
              ConvertDenseIntAttr(parameter_tile_attr);
          xla::TileProto tile;
          tile.mutable_dimensions()->Assign(tile_dims.begin(), tile_dims.end());
          *hlo_module.mutable_host_program_shape()
               ->mutable_parameters()
               ->at(i)
               .mutable_tuple_shapes(j)
               ->mutable_layout()
               ->mutable_tiles()
               ->Add() = tile;
          *hlo_module.mutable_computations(0)
               ->mutable_program_shape()
               ->mutable_parameters()
               ->at(i)
               .mutable_tuple_shapes(j)
               ->mutable_layout()
               ->mutable_tiles()
               ->Add() = tile;
          *hlo_module.mutable_computations(0)
               ->mutable_instructions(i)
               ->mutable_shape()
               ->mutable_tuple_shapes(j)
               ->mutable_layout()
               ->mutable_tiles()
               ->Add() = tile;
        }
      }
    } else {
      std::vector<int64_t> tile_dims = ConvertDenseIntAttr(
          mlir::cast<mlir::DenseIntElementsAttr>(parameter_tile[0]));
      xla::TileProto tile;
      tile.mutable_dimensions()->Assign(tile_dims.begin(), tile_dims.end());
      *hlo_module.mutable_host_program_shape()
           ->mutable_parameters()
           ->at(i)
           .mutable_layout()
           ->mutable_tiles()
           ->Add() = tile;
      *hlo_module.mutable_computations(0)
           ->mutable_program_shape()
           ->mutable_parameters()
           ->at(i)
           .mutable_layout()
           ->mutable_tiles()
           ->Add() = tile;
      *hlo_module.mutable_computations(0)
           ->mutable_instructions(i)
           ->mutable_shape()
           ->mutable_layout()
           ->mutable_tiles()
           ->Add() = tile;
    }
  }
  return absl::OkStatus();
}

void ExportModuleEntryComputationResultLayout(
    const mlir::DenseIntElementsAttr& xla_entry_computation_result_layout,
    xla::HloModuleProto& hlo_module, int root_index) {
  std::vector<int64_t> layout_dims =
      ConvertDenseIntAttr(xla_entry_computation_result_layout);
  hlo_module.mutable_host_program_shape()
      ->mutable_result()
      ->mutable_layout()
      ->mutable_minor_to_major()
      ->Assign(layout_dims.begin(), layout_dims.end());
  hlo_module.mutable_computations(0)
      ->mutable_program_shape()
      ->mutable_result()
      ->mutable_layout()
      ->mutable_minor_to_major()
      ->Assign(layout_dims.begin(), layout_dims.end());

  // ROOT is the result of the computation, so assign the layout to it.
  if (root_index == -1) {
    FindRootInstructionIndex(hlo_module, root_index);
  }
  hlo_module.mutable_computations(0)
      ->mutable_instructions(root_index)
      ->mutable_shape()
      ->mutable_layout()
      ->mutable_minor_to_major()
      ->Assign(layout_dims.begin(), layout_dims.end());
}

void ExportModuleEntryComputationResultTiles(
    const mlir::ArrayAttr& xla_entry_computation_result_tiles,
    xla::HloModuleProto& hlo_module, int root_index) {
  if (xla_entry_computation_result_tiles.empty()) return;

  std::vector<int64_t> arr =
      ConvertDenseIntAttr(mlir::cast<mlir::DenseIntElementsAttr>(
          xla_entry_computation_result_tiles[0]));
  xla::TileProto tile;
  tile.mutable_dimensions()->Assign(arr.begin(), arr.end());
  *hlo_module.mutable_host_program_shape()
       ->mutable_result()
       ->mutable_layout()
       ->mutable_tiles()
       ->Add() = tile;
  *hlo_module.mutable_computations(0)
       ->mutable_program_shape()
       ->mutable_result()
       ->mutable_layout()
       ->mutable_tiles()
       ->Add() = tile;

  // ROOT is the result of the computation, so assign the tile to it.
  if (root_index == -1) {
    FindRootInstructionIndex(hlo_module, root_index);
  }
  *hlo_module.mutable_computations(0)
       ->mutable_instructions(root_index)
       ->mutable_shape()
       ->mutable_layout()
       ->mutable_tiles()
       ->Add() = tile;
}

}  // namespace mhlo
}  // namespace mlir
