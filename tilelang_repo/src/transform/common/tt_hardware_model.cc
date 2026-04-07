/*!
 * \file tt_hardware_model.cc
 * \brief Minimal TT hardware snapshot intake and SpatialCapabilityModel derivation.
 */

#include "tt_hardware_model.h"

#include <tvm/runtime/logging.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include "spatial_vocab.h"

namespace tvm {
namespace tl {

namespace {

using tvm::ffi::Array;
using tvm::ffi::String;
using namespace tvm::tl::spatial;

std::string ReadFileIfExists(const std::filesystem::path& path) {
  if (path.empty() || !std::filesystem::exists(path) || !std::filesystem::is_regular_file(path)) {
    return "";
  }
  std::ifstream file(path);
  if (!file.is_open()) {
    return "";
  }
  std::ostringstream os;
  os << file.rdbuf();
  return os.str();
}

std::vector<std::filesystem::path> CandidateSoCDescriptorPaths() {
  std::vector<std::filesystem::path> candidates;
  auto push_env_path = [&](const char* env_name, const char* suffix) {
    if (const char* value = std::getenv(env_name)) {
      std::filesystem::path base(value);
      if (!base.empty()) {
        candidates.push_back(base / suffix);
      }
    }
  };
  push_env_path("TT_METAL_SIMULATOR_HOME", "soc_descriptor.yaml");
  push_env_path("TT_METAL_HOME", "sim/soc_descriptor.yaml");
  push_env_path("TT_METAL_RUNTIME_ROOT", "sim/soc_descriptor.yaml");
  push_env_path("TILELANG_HOME", "../tt_metal_repo/sim/soc_descriptor.yaml");
  candidates.push_back(std::filesystem::current_path() / "tt_metal_repo/sim/soc_descriptor.yaml");
  candidates.push_back(std::filesystem::current_path() / "../tt_metal_repo/sim/soc_descriptor.yaml");
  return candidates;
}

std::pair<std::string, std::string> LoadBlackholeSoCDescriptorText() {
  for (const auto& candidate : CandidateSoCDescriptorPaths()) {
    std::string text = ReadFileIfExists(candidate);
    if (!text.empty()) {
      return {text, candidate.lexically_normal().string()};
    }
  }
  return {"", ""};
}

std::string ExtractSection(const std::string& text, const std::string& key) {
  std::istringstream stream(text);
  std::ostringstream section;
  std::string line;
  bool in_section = false;
  while (std::getline(stream, line)) {
    const bool is_top_level =
        !line.empty() && line[0] != ' ' && line[0] != '\t' && line.find(':') != std::string::npos;
    if (!in_section) {
      if (line.rfind(key + ":", 0) == 0) {
        in_section = true;
      }
      continue;
    }
    if (is_top_level) {
      break;
    }
    section << line << '\n';
  }
  return section.str();
}

std::optional<int64_t> MatchInt(const std::string& text, const std::string& pattern) {
  std::smatch match;
  if (std::regex_search(text, match, std::regex(pattern))) {
    return std::stoll(match[1].str());
  }
  return std::nullopt;
}

std::optional<std::string> MatchString(const std::string& text, const std::string& pattern) {
  std::smatch match;
  if (std::regex_search(text, match, std::regex(pattern))) {
    return match[1].str();
  }
  return std::nullopt;
}

int64_t CountMatches(const std::string& text, const std::string& pattern) {
  int64_t count = 0;
  const std::regex regex(pattern);
  for (std::sregex_iterator it(text.begin(), text.end(), regex), end; it != end; ++it) {
    ++count;
  }
  return count;
}

int64_t GetTargetIntAttr(const Target& target, const char* key, int64_t fallback) {
  if (auto value = target->GetAttr<Integer>(key)) {
    return value.value()->value;
  }
  return fallback;
}

Array<String> MakeStringArray(std::initializer_list<const char*> values) {
  Array<String> result;
  for (const char* value : values) {
    result.push_back(String(value));
  }
  return result;
}

}  // namespace

TTHardwareModel::TTHardwareModel(ffi::String arch_name, ffi::String descriptor_path,
                                 int64_t logical_worker_grid_x, int64_t logical_worker_grid_y,
                                 int64_t functional_worker_count, int64_t router_only_count,
                                 int64_t dram_view_count, int64_t worker_l1_size,
                                 int64_t dram_view_size, bool noc_translation_id_enabled,
                                 int64_t unpacker_version, int64_t packer_version,
                                 int64_t overlay_version) {
  auto n = ffi::make_object<TTHardwareModelNode>();
  n->arch_name = std::move(arch_name);
  n->descriptor_path = std::move(descriptor_path);
  n->logical_worker_grid_x = logical_worker_grid_x;
  n->logical_worker_grid_y = logical_worker_grid_y;
  n->functional_worker_count = functional_worker_count;
  n->router_only_count = router_only_count;
  n->dram_view_count = dram_view_count;
  n->worker_l1_size = worker_l1_size;
  n->dram_view_size = dram_view_size;
  n->noc_translation_id_enabled = noc_translation_id_enabled;
  n->unpacker_version = unpacker_version;
  n->packer_version = packer_version;
  n->overlay_version = overlay_version;
  data_ = std::move(n);
}

TTHardwareModel BuildBlackholeTTHardwareModel(const Target& target) {
  const auto [descriptor_text, descriptor_path] = LoadBlackholeSoCDescriptorText();
  const int64_t logical_worker_grid_x = GetTargetIntAttr(target, "logical_worker_grid_x", 11);
  const int64_t logical_worker_grid_y = GetTargetIntAttr(target, "logical_worker_grid_y", 10);

  const std::string arch_name =
      MatchString(descriptor_text, R"(arch_name:\s*([A-Za-z0-9_]+))").value_or("BLACKHOLE");
  const int64_t worker_l1_size =
      MatchInt(descriptor_text, R"(worker_l1_size:\s*([0-9]+))").value_or(1572864);
  const int64_t dram_view_size =
      MatchInt(descriptor_text, R"(dram_view_size:\s*([0-9]+))").value_or(4278190080LL);
  const int64_t functional_worker_count =
      descriptor_text.empty() ? logical_worker_grid_x * logical_worker_grid_y
                              : CountMatches(ExtractSection(descriptor_text, "functional_workers"),
                                             R"(\b[0-9]+-[0-9]+\b)");
  const int64_t router_only_count =
      descriptor_text.empty() ? 0
                              : CountMatches(ExtractSection(descriptor_text, "router_only"),
                                             R"(\b[0-9]+-[0-9]+\b)");
  const int64_t dram_view_count =
      descriptor_text.empty() ? 8
                              : CountMatches(ExtractSection(descriptor_text, "dram_views"),
                                             R"(\bchannel:\s*[0-9]+\b)");
  const bool noc_translation_id_enabled =
      MatchString(descriptor_text,
                  R"(translation_id_enabled:\s*(True|False))")
              .value_or("True") == "True";
  const int64_t unpacker_version =
      MatchInt(descriptor_text, R"(unpacker:\s*\n\s*version:\s*([0-9]+))").value_or(2);
  const int64_t packer_version =
      MatchInt(descriptor_text, R"(packer:\s*\n\s*version:\s*([0-9]+))").value_or(2);
  const int64_t overlay_version =
      MatchInt(descriptor_text, R"(overlay:\s*\n\s*version:\s*([0-9]+))").value_or(2);

  return TTHardwareModel(String(arch_name), String(descriptor_path), logical_worker_grid_x,
                         logical_worker_grid_y, functional_worker_count, router_only_count,
                         dram_view_count, worker_l1_size, dram_view_size,
                         noc_translation_id_enabled, unpacker_version, packer_version,
                         overlay_version);
}

SpatialCapabilityModel DeriveSpatialCapabilityModel(const TTHardwareModel& hardware_model) {
  return SpatialCapabilityModel(
      hardware_model->arch_name, String("grid"), String("logical_worker_grid"),
      hardware_model->logical_worker_grid_x, hardware_model->logical_worker_grid_y,
      hardware_model->functional_worker_count, hardware_model->router_only_count,
      hardware_model->dram_view_count, hardware_model->worker_l1_size,
      hardware_model->dram_view_size,
      MakeStringArray({ToString(SpatialChannelKind::kPointToPoint),
                       ToString(SpatialChannelKind::kBroadcast),
                       ToString(SpatialChannelKind::kCarry),
                       ToString(SpatialChannelKind::kReduceMerge),
                       ToString(SpatialChannelKind::kGather),
                       ToString(SpatialChannelKind::kScatter)}),
      MakeStringArray({ToString(SpatialChannelPayloadKind::kTensor),
                       ToString(SpatialChannelPayloadKind::kStateVersion),
                       ToString(SpatialChannelPayloadKind::kIndex),
                       ToString(SpatialChannelPayloadKind::kPredicate),
                       ToString(SpatialChannelPayloadKind::kToken)}),
      MakeStringArray({ToString(SpatialChannelDeliveryKind::kOrdered),
                       ToString(SpatialChannelDeliveryKind::kCompletionVisible),
                       ToString(SpatialChannelDeliveryKind::kBufferedAsync),
                       ToString(SpatialChannelDeliveryKind::kPhaseBoundaryMaterialized)}),
      MakeStringArray({ToString(SpatialSyncKind::kDependency),
                       ToString(SpatialSyncKind::kBarrier),
                       ToString(SpatialSyncKind::kCompletion)}),
      MakeStringArray({"must_happen_before",
                       "carry_handoff",
                       "reduction_completion",
                       "selection_index_handoff",
                       "phase_boundary_materialization"}),
      MakeStringArray({"buffer_visibility",
                       "completion_visibility",
                       "phase_boundary",
                       "phase_boundary_materialization"}),
      MakeStringArray({ToString(SpatialLayoutKind::kRegular),
                       ToString(SpatialLayoutKind::kPacked),
                       ToString(SpatialLayoutKind::kIndexed)}),
      MakeStringArray({ToString(SpatialPartitionKind::kReplicated),
                       ToString(SpatialPartitionKind::kBlocked),
                       ToString(SpatialPartitionKind::kIndexed),
                       ToString(SpatialPartitionKind::kFiltered)}),
      MakeStringArray({ToString(SpatialResourceIntentKind::kBuffer),
                       ToString(SpatialResourceIntentKind::kStateResidency),
                       ToString(SpatialResourceIntentKind::kSynchronizationSupport),
                       ToString(SpatialResourceIntentKind::kPhaseBoundaryMaterialization),
                       ToString(SpatialResourceIntentKind::kLoweringSupport)}));
}

std::optional<TTHardwareModel> GetModuleTTHardwareModel(const IRModule& mod) {
  auto maybe_items = mod->global_infos.Get(attr::kTLTTHardwareModel);
  if (!maybe_items || maybe_items.value().empty()) {
    return std::nullopt;
  }
  return Downcast<TTHardwareModel>(maybe_items.value()[0]);
}

std::optional<SpatialCapabilityModel> GetModuleSpatialCapabilityModel(const IRModule& mod) {
  auto maybe_items = mod->global_infos.Get(attr::kTLSpatialCapabilityModel);
  if (!maybe_items || maybe_items.value().empty()) {
    return std::nullopt;
  }
  return Downcast<SpatialCapabilityModel>(maybe_items.value()[0]);
}

TVM_FFI_STATIC_INIT_BLOCK() { TTHardwareModelNode::RegisterReflection(); }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.TTHardwareModel",
                        [](ffi::String arch_name, ffi::String descriptor_path,
                           int64_t logical_worker_grid_x, int64_t logical_worker_grid_y,
                           int64_t functional_worker_count, int64_t router_only_count,
                           int64_t dram_view_count, int64_t worker_l1_size,
                           int64_t dram_view_size, bool noc_translation_id_enabled,
                           int64_t unpacker_version, int64_t packer_version,
                           int64_t overlay_version) {
                          return TTHardwareModel(
                              std::move(arch_name), std::move(descriptor_path),
                              logical_worker_grid_x, logical_worker_grid_y,
                              functional_worker_count, router_only_count, dram_view_count,
                              worker_l1_size, dram_view_size, noc_translation_id_enabled,
                              unpacker_version, packer_version, overlay_version);
                        });
}

}  // namespace tl
}  // namespace tvm
