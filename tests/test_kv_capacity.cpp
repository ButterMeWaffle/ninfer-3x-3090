#include "runtime/engine/kv_capacity.h"

#include <iostream>
#include <stdexcept>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int main() {
    int failures = 0;
    const ninfer::runtime::SequenceCapacityCurve curve{
        .main_page_tokens                     = 64,
        .minimum_main_page_groups             = 2,
        .maximum_main_page_groups             = 6,
        .minimum_device_reservation_bytes     = 1000,
        .bytes_per_additional_main_page_group = 128,
    };

    const auto automatic =
        ninfer::runtime::resolve_kv_capacity(ninfer::KvCapacityPolicy::automatic(50), curve, 1360);
    failures +=
        check(automatic.main_page_groups == 4 && automatic.resolved_tokens == 256 &&
                  automatic.runtime_reservation_bytes == 1256 &&
                  automatic.automatic_headroom_bytes == 50 && automatic.planned_slack_bytes == 104,
              "automatic KV capacity did not select the largest fitting page count");

    const auto capped =
        ninfer::runtime::resolve_kv_capacity(ninfer::KvCapacityPolicy::automatic(50), curve, 10000);
    failures += check(capped.main_page_groups == 6 && capped.resolved_tokens == 384,
                      "automatic KV capacity exceeded or missed the target maximum");

    const auto explicit_capacity = ninfer::runtime::resolve_kv_capacity(
        ninfer::KvCapacityPolicy::explicit_capacity(129), curve, 1200);
    failures +=
        check(explicit_capacity.main_page_groups == 3 && explicit_capacity.resolved_tokens == 192 &&
                  explicit_capacity.runtime_reservation_bytes == 1128,
              "explicit KV capacity did not use page-aligned token semantics");

    bool insufficient_rejected = false;
    try {
        (void)ninfer::runtime::resolve_kv_capacity(ninfer::KvCapacityPolicy::automatic(50), curve,
                                                   1049);
    } catch (const std::invalid_argument&) { insufficient_rejected = true; }
    failures += check(insufficient_rejected,
                      "automatic KV capacity accepted less than the minimum reservation");

    const ninfer::runtime::SequenceCapacityCurve sharded_curve{
        .main_page_tokens                                = 64,
        .minimum_main_page_groups                        = 2,
        .maximum_main_page_groups                        = 6,
        .minimum_device_reservation_bytes                = 1000,
        .bytes_per_additional_main_page_group            = 128,
        .minimum_secondary_device_reservation_bytes      = 600,
        .secondary_bytes_per_additional_main_page_group  = 256,
    };
    const auto sharded = ninfer::runtime::resolve_kv_capacity(
        ninfer::KvCapacityPolicy::automatic(50), sharded_curve, 1360, 1200);
    failures += check(
        sharded.main_page_groups == 4 && sharded.resolved_tokens == 256 &&
            sharded.runtime_reservation_bytes == 1256 &&
            sharded.secondary_runtime_reservation_bytes == 1112 &&
            sharded.secondary_available_after_weights_bytes == 1200 &&
            sharded.secondary_planned_slack_bytes == 88,
        "sharded automatic KV capacity did not enforce both device curves");

    const auto secondary_limited = ninfer::runtime::resolve_kv_capacity(
        ninfer::KvCapacityPolicy::automatic(50), sharded_curve, 1360, 1150);
    failures += check(secondary_limited.main_page_groups == 3 &&
                          secondary_limited.runtime_reservation_bytes == 1128 &&
                          secondary_limited.secondary_runtime_reservation_bytes == 856,
                      "sharded KV capacity did not honor the limiting secondary pool");

    bool insufficient_secondary_rejected = false;
    try {
        (void)ninfer::runtime::resolve_kv_capacity(
            ninfer::KvCapacityPolicy::explicit_capacity(129), sharded_curve, 1200, 855);
    } catch (const std::invalid_argument&) { insufficient_secondary_rejected = true; }
    failures += check(insufficient_secondary_rejected,
                      "explicit KV capacity accepted an undersized secondary pool");

    // 3-way: primary + secondary + tertiary pools (5/6/5 KV layer split)
    const ninfer::runtime::SequenceCapacityCurve triple_curve{
        .main_page_tokens                                = 64,
        .minimum_main_page_groups                        = 2,
        .maximum_main_page_groups                        = 6,
        .minimum_device_reservation_bytes                = 1000,
        .bytes_per_additional_main_page_group            = 128,
        .minimum_secondary_device_reservation_bytes      = 600,
        .secondary_bytes_per_additional_main_page_group  = 256,
        .minimum_tertiary_device_reservation_bytes       = 500,
        .tertiary_bytes_per_additional_main_page_group   = 200,
    };
    const auto triple = ninfer::runtime::resolve_kv_capacity(
        ninfer::KvCapacityPolicy::automatic(50), triple_curve, 1360, 1200, 1100);
    failures += check(
        triple.main_page_groups == 4 && triple.resolved_tokens == 256 &&
            triple.runtime_reservation_bytes == 1256 &&
            triple.secondary_runtime_reservation_bytes == 1112 &&
            triple.tertiary_runtime_reservation_bytes == 900 &&
            triple.tertiary_available_after_weights_bytes == 1100 &&
            triple.tertiary_planned_slack_bytes == 200,
        "3-way automatic KV capacity did not enforce all three device curves");

    // Tertiary-limited: tertiary pool is the bottleneck
    const auto triple_limited = ninfer::runtime::resolve_kv_capacity(
        ninfer::KvCapacityPolicy::automatic(50), triple_curve, 1360, 1200, 900);
    failures += check(triple_limited.main_page_groups == 3 &&
                          triple_limited.tertiary_runtime_reservation_bytes == 700,
                      "3-way KV capacity did not honor the limiting tertiary pool");

    bool insufficient_tertiary_rejected = false;
    try {
        (void)ninfer::runtime::resolve_kv_capacity(
            ninfer::KvCapacityPolicy::explicit_capacity(129), triple_curve, 1200, 1100, 655);
    } catch (const std::invalid_argument&) { insufficient_tertiary_rejected = true; }
    failures += check(insufficient_tertiary_rejected,
                      "explicit KV capacity accepted an undersized tertiary pool");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
