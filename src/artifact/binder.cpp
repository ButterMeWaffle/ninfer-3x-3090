#include "artifact/binder.h"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>

namespace ninfer::artifact {
namespace {

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
    const std::uint64_t mask = alignment - 1;
    if (value > std::numeric_limits<std::uint64_t>::max() - mask) {
        throw ArtifactError("materialization plan size overflows u64");
    }
    return (value + mask) & ~mask;
}

} // namespace

Binder::Binder(const Reader& reader)
    : reader_(reader), consumed_(reader.objects().size(), false),
      planned_(reader.objects().size(), false) {
    materialization_.object_count = reader.objects().size();
}

bool Binder::has_object(std::string_view name) const noexcept {
    return reader_.find(name) != nullptr;
}

ObjectHandle Binder::find_unconsumed(std::string_view name) {
    const auto& objects            = reader_.objects();
    const ObjectDescriptor* object = reader_.find(name);
    if (object == nullptr) {
        throw ArtifactError("required artifact object is missing: " + std::string(name));
    }
    const auto index = static_cast<std::size_t>(object - objects.data());
    if (consumed_[index]) {
        throw ArtifactError("artifact object was bound more than once: " + std::string(name));
    }
    consumed_[index] = true;
    return ObjectHandle{index};
}

ObjectHandle Binder::require_tensor(std::string_view name, NumericFormat format,
                                    StorageLayout layout, std::span<const std::uint64_t> shape) {
    const ObjectHandle handle = find_unconsumed(name);
    const auto* tensor        = std::get_if<TensorDescriptor>(&descriptor(handle));
    if (tensor == nullptr) {
        throw ArtifactError("required tensor is a resource: " + std::string(name));
    }
    if (tensor->format != format || tensor->layout != layout ||
        !std::equal(tensor->shape.begin(), tensor->shape.end(), shape.begin(), shape.end())) {
        throw ArtifactError("tensor descriptor does not match target contract: " +
                            std::string(name));
    }
    return handle;
}

ObjectHandle Binder::require_resource(std::string_view name, ResourceEncoding encoding) {
    const ObjectHandle handle = find_unconsumed(name);
    const auto* resource      = std::get_if<ResourceDescriptor>(&descriptor(handle));
    if (resource == nullptr) {
        throw ArtifactError("required resource is a tensor: " + std::string(name));
    }
    if (resource->encoding != encoding) {
        throw ArtifactError("resource encoding does not match target contract: " +
                            std::string(name));
    }
    return handle;
}

const ObjectDescriptor& Binder::descriptor(ObjectHandle handle) const {
    if (handle.index >= reader_.objects().size()) {
        throw ArtifactError("artifact object handle is out of range");
    }
    return reader_.objects()[handle.index];
}

PayloadSpan Binder::payload(ObjectHandle handle) const {
    return reader_.payload(descriptor(handle));
}

void Binder::materialize_on_device(ObjectHandle handle) {
    const auto* tensor = std::get_if<TensorDescriptor>(&descriptor(handle));
    if (tensor == nullptr) {
        throw ArtifactError("resource cannot be materialized as a device tensor");
    }
    if (planned_[handle.index]) {
        throw ArtifactError("artifact object has more than one materialization placement: " +
                            std::string(tensor->name));
    }
    const std::uint64_t alignment = tensor_alignment(tensor->layout);
    const std::uint64_t offset    = align_up(materialization_.device_capacity_bytes, alignment);
    if (tensor->bytes > std::numeric_limits<std::uint64_t>::max() - offset) {
        throw ArtifactError("materialization plan size overflows u64");
    }
    materialization_.device_objects.push_back(
        DeviceMaterialization{.object         = handle,
                              .offset         = offset,
                              .bytes          = tensor->bytes,
                              .alignment      = alignment,
                              .primary_offset = offset,
                              .primary_bytes  = tensor->bytes});
    materialization_.device_capacity_bytes = offset + tensor->bytes;
    materialization_.source_device_capacity_bytes =
        materialization_.device_capacity_bytes;
    planned_[handle.index]                 = true;
}

void Binder::shard_row_split_across_devices(ObjectHandle handle, RowSplitShardAxis axis,
                                            std::uint64_t split) {
    shard_row_split_across_devices(handle, axis, std::span<const std::uint64_t>(&split, 1));
}

void Binder::shard_row_split_across_devices(ObjectHandle handle, RowSplitShardAxis axis,
                                            std::span<const std::uint64_t> splits) {
    if (splits.empty() || splits.size() > 2) {
        throw ArtifactError("cross-device shard requires one or two split boundaries");
    }
    const auto* tensor = std::get_if<TensorDescriptor>(&descriptor(handle));
    if (tensor == nullptr || tensor->layout != StorageLayout::RowSplitK128V1 ||
        tensor->shape.size() != 2) {
        throw ArtifactError("only rank-two row-split tensors can be sharded across devices");
    }
    const auto placement = std::find_if(
        materialization_.device_objects.begin(), materialization_.device_objects.end(),
        [&](const DeviceMaterialization& item) { return item.object.index == handle.index; });
    if (placement == materialization_.device_objects.end()) {
        throw ArtifactError("cross-device shard tensor is not materialized on the device");
    }
    const auto duplicate = std::find_if(
        materialization_.row_split_shards.begin(), materialization_.row_split_shards.end(),
        [&](const RowSplitShardMaterialization& item) {
            return item.object.index == handle.index;
        });
    if (duplicate != materialization_.row_split_shards.end()) {
        throw ArtifactError("tensor was assigned more than one cross-device shard");
    }

    const std::uint64_t rows = tensor->shape[0];
    const std::uint64_t cols = tensor->shape[1];
    // Validate each boundary is strictly increasing, positive, and within the logical extent.
    std::uint64_t prev = 0;
    for (const std::uint64_t boundary : splits) {
        if (boundary <= prev) {
            throw ArtifactError("cross-device shard splits must be strictly increasing");
        }
        prev = boundary;
    }
    if (axis == RowSplitShardAxis::PairedRows) {
        if ((rows % 2) != 0 || prev >= rows / 2) {
            throw ArtifactError("paired-row shard splits are outside the logical row range");
        }
    } else {
        if (prev >= cols) {
            throw ArtifactError("column shard splits are outside the logical column range");
        }
        for (const std::uint64_t boundary : splits) {
            const std::array<std::uint64_t, 2> boundary_shape = {1, boundary};
            if (row_split_geometry(tensor->format, boundary_shape).padded_columns != boundary) {
                throw ArtifactError(
                    "column shard splits must preserve the row-split-k128 group boundary");
            }
        }
    }

    materialization_.row_split_shards.push_back(RowSplitShardMaterialization{
        .object = handle,
        .axis   = axis,
        .splits = std::vector<std::uint64_t>(splits.begin(), splits.end()),
    });
}

void Binder::retain_on_host(ObjectHandle handle) {
    const auto* resource = std::get_if<ResourceDescriptor>(&descriptor(handle));
    if (resource == nullptr) {
        throw ArtifactError("tensor cannot be retained as a host resource");
    }
    if (planned_[handle.index]) {
        throw ArtifactError("artifact object has more than one materialization placement: " +
                            std::string(resource->name));
    }
    materialization_.host_objects.push_back(HostMaterialization{handle});
    planned_[handle.index] = true;
}

void Binder::validate_only(ObjectHandle handle) {
    const ObjectDescriptor& object = descriptor(handle);
    if (planned_[handle.index]) {
        throw ArtifactError("artifact object has more than one materialization placement: " +
                            std::string(object_name(object)));
    }
    planned_[handle.index] = true;
}

MaterializationPlan Binder::finish() {
    const auto it = std::find(consumed_.begin(), consumed_.end(), false);
    if (it != consumed_.end()) {
        const auto index = static_cast<std::size_t>(it - consumed_.begin());
        throw ArtifactError("artifact object was not consumed by the selected target: " +
                            std::string(object_name(reader_.objects()[index])));
    }
    const auto unplanned = std::find(planned_.begin(), planned_.end(), false);
    if (unplanned != planned_.end()) {
        const auto index = static_cast<std::size_t>(unplanned - planned_.begin());
        throw ArtifactError("artifact object has no materialization placement: " +
                            std::string(object_name(reader_.objects()[index])));
    }

    if (!materialization_.row_split_shards.empty()) {
        materialization_.source_device_capacity_bytes =
            materialization_.device_capacity_bytes;
        std::uint64_t primary_capacity   = 0;
        std::uint64_t secondary_capacity = 0;
        std::uint64_t tertiary_capacity  = 0;
        for (DeviceMaterialization& placement : materialization_.device_objects) {
            placement.primary_offset = align_up(primary_capacity, placement.alignment);
            const auto shard = std::find_if(
                materialization_.row_split_shards.begin(),
                materialization_.row_split_shards.end(),
                [&](const RowSplitShardMaterialization& item) {
                    return item.object.index == placement.object.index;
                });
            if (shard == materialization_.row_split_shards.end()) {
                placement.primary_bytes = placement.bytes;
            } else {
                const auto& tensor = std::get<TensorDescriptor>(descriptor(placement.object));
                const std::uint64_t rows = tensor.shape[0];
                const std::uint64_t cols = tensor.shape[1];
                // Per-rank extent: rank 0 owns [0, splits[0]), rank i owns (splits[i-1],
                // splits[i]), the last rank owns (splits[last], extent).
                const auto rank_bounds = [&](std::size_t rank) -> std::uint64_t {
                    if (rank < shard->splits.size()) { return shard->splits[rank]; }
                    return (shard->axis == RowSplitShardAxis::PairedRows) ? rows / 2 : cols;
                };
                const std::size_t rank_count = shard->splits.size() + 1;
                auto rank_shape = [&](std::size_t rank) -> std::array<std::uint64_t, 2> {
                    const std::uint64_t begin = rank == 0 ? 0 : shard->splits[rank - 1];
                    const std::uint64_t end   = rank_bounds(rank);
                    if (shard->axis == RowSplitShardAxis::PairedRows) {
                        return {2 * (end - begin), cols};
                    }
                    return {rows, end - begin};
                };
                placement.primary_bytes =
                    row_split_geometry(tensor.format, rank_shape(0)).encoded_bytes;
                shard->secondary_offsets.resize(rank_count - 1);
                shard->secondary_bytes.resize(rank_count - 1);
                for (std::size_t rank = 1; rank < rank_count; ++rank) {
                    const std::uint64_t offset = (rank == 1)
                                                      ? align_up(secondary_capacity, placement.alignment)
                                                      : align_up(tertiary_capacity, placement.alignment);
                    const std::uint64_t bytes =
                        row_split_geometry(tensor.format, rank_shape(rank)).encoded_bytes;
                    if (bytes > std::numeric_limits<std::uint64_t>::max() - offset) {
                        throw ArtifactError("secondary materialization plan size overflows u64");
                    }
                    if (rank == 1) {
                        shard->secondary_offsets[rank - 1] = offset;
                        shard->secondary_bytes[rank - 1]   = bytes;
                        secondary_capacity = offset + bytes;
                    } else {
                        shard->secondary_offsets[rank - 1] = offset;
                        shard->secondary_bytes[rank - 1]   = bytes;
                        tertiary_capacity = offset + bytes;
                    }
                }
            }
            if (placement.primary_bytes >
                std::numeric_limits<std::uint64_t>::max() - placement.primary_offset) {
                throw ArtifactError("primary materialization plan size overflows u64");
            }
            primary_capacity = placement.primary_offset + placement.primary_bytes;
        }
        materialization_.device_capacity_bytes           = primary_capacity;
        materialization_.secondary_device_capacity_bytes = secondary_capacity;
        materialization_.tertiary_device_capacity_bytes  = tertiary_capacity;
    }
    return std::move(materialization_);
}

} // namespace ninfer::artifact
