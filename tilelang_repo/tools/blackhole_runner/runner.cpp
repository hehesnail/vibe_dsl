/*!
 * \file runner.cpp
 * \brief External runner for TileLang Blackhole kernels
 *
 * Protocol:
 *   tilelang_blackhole_runner <spec.json> <input.bin> <output.bin>
 */

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/buffer.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;
using namespace tt::tt_metal;

struct KernelArgSpec {
    std::string name;
    std::string kind;
    std::string dtype;
};

struct KernelSpec {
    std::string name;
    std::string kind;
    std::string core_type;
    std::string kernel_path;
    std::vector<uint32_t> compile_time_args;
    std::vector<KernelArgSpec> runtime_args;
};

struct CBConfig {
    uint32_t cb_id = 0;
    std::string name;
    std::string role;
    uint32_t num_pages = 1;
    uint32_t page_size_bytes = 2048;
    std::string data_format = "Float16_b";
};

struct CorePlan {
    struct PhysicalCore {
        uint32_t core_x = 0;
        uint32_t core_y = 0;
    };

    struct WorkPacket {
        uint32_t core_x = 0;
        uint32_t core_y = 0;
        uint32_t work_offset = 0;
        uint32_t work_count = 1;
    };

    uint32_t logical_grid_x = 1;
    uint32_t logical_grid_y = 1;
    std::string linearization = "row_major";
    std::vector<PhysicalCore> physical_cores;
    std::vector<WorkPacket> work_packets;
};

struct RunConfig {
    std::string entry_name;
    std::string spec_path;
    std::string input_path;
    std::string output_path;
    size_t input_size_bytes = 0;
    size_t output_size_bytes = 0;
    std::vector<uint32_t> scalar_args;
    std::vector<CBConfig> cb_configs;
    CorePlan core_plan;
    std::vector<KernelSpec> kernels;
};

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " <spec.json> <input.bin> <output.bin>\n";
}

json read_json_file(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Cannot open JSON file: " + path);
    }
    json j;
    file >> j;
    return j;
}

std::vector<uint8_t> read_file(const std::string& path, size_t size) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + path);
    }

    std::vector<uint8_t> data(size);
    if (!file.read(reinterpret_cast<char*>(data.data()), size)) {
        throw std::runtime_error("Cannot read file: " + path);
    }
    return data;
}

void write_file(const std::string& path, const std::vector<uint8_t>& data) {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot create file: " + path);
    }

    if (!file.write(reinterpret_cast<const char*>(data.data()), data.size())) {
        throw std::runtime_error("Cannot write file: " + path);
    }
}

tt::DataFormat parse_data_format(const std::string& value) {
    if (value == "Float16" || value == "Float16_b") return tt::DataFormat::Float16_b;
    if (value == "Float32") return tt::DataFormat::Float32;
    if (value == "UInt16") return tt::DataFormat::UInt16;
    if (value == "UInt32") return tt::DataFormat::UInt32;
    throw std::runtime_error("Unsupported data format in spec: " + value);
}

CBConfig parse_cb_config(const json& j) {
    CBConfig config;
    config.cb_id = j.at("cb_id").get<uint32_t>();
    config.name = j.value("name", "");
    config.role = j.value("role", "intermediate");
    config.num_pages = j.value("num_pages", 1u);
    config.page_size_bytes = j.value("page_size_bytes", 2048u);
    config.data_format = j.value("data_format", "Float16_b");
    return config;
}

KernelArgSpec parse_kernel_arg_spec(const json& j) {
    return KernelArgSpec{
        .name = j.value("name", ""),
        .kind = j.value("kind", ""),
        .dtype = j.value("dtype", "uint32")};
}

KernelSpec parse_kernel_spec(const json& j) {
    KernelSpec spec;
    spec.name = j.value("name", "");
    spec.kind = j.value("kind", "fused_dataflow");
    spec.core_type = j.value("core_type", "brisc");
    spec.kernel_path = j.at("kernel_path").get<std::string>();
    spec.compile_time_args = j.value("compile_time_args", std::vector<uint32_t>{});
    for (const auto& arg : j.value("runtime_args", json::array())) {
        spec.runtime_args.push_back(parse_kernel_arg_spec(arg));
    }
    return spec;
}

RunConfig parse_args(int argc, char* argv[]) {
    if (argc != 4) {
        print_usage(argv[0]);
        std::exit(1);
    }

    const std::string spec_path = argv[1];
    json spec_json = read_json_file(spec_path);

    RunConfig config;
    config.spec_path = spec_path;
    config.input_path = argv[2];
    config.output_path = argv[3];
    config.entry_name = spec_json.value("entry_name", "default");
    config.input_size_bytes = spec_json.value("input_size_bytes", 0u);
    config.output_size_bytes = spec_json.value("output_size_bytes", 0u);
    config.scalar_args = spec_json.value("scalar_args", std::vector<uint32_t>{});

    const auto& core = spec_json.value("core_plan", json::object());
    config.core_plan.logical_grid_x =
        core.value("logical_grid_x", core.value("grid_x", 1u));
    config.core_plan.logical_grid_y =
        core.value("logical_grid_y", core.value("grid_y", 1u));
    config.core_plan.linearization = core.value("linearization", "row_major");
    for (const auto& item : core.value("physical_cores", json::array())) {
        config.core_plan.physical_cores.push_back(CorePlan::PhysicalCore{
            .core_x = item.value("core_x", 0u),
            .core_y = item.value("core_y", 0u),
        });
    }
    for (const auto& item : core.value("work_packets", json::array())) {
        config.core_plan.work_packets.push_back(CorePlan::WorkPacket{
            .core_x = item.value("core_x", 0u),
            .core_y = item.value("core_y", 0u),
            .work_offset = item.value("work_offset", 0u),
            .work_count = item.value("work_count", 1u),
        });
    }

    for (const auto& cb : spec_json.value("cb_configs", json::array())) {
        config.cb_configs.push_back(parse_cb_config(cb));
    }
    for (const auto& kernel : spec_json.value("kernels", json::array())) {
        config.kernels.push_back(parse_kernel_spec(kernel));
    }

    if (config.kernels.empty()) {
        throw std::runtime_error("Spec contains no kernels");
    }
    return config;
}

uint32_t choose_page_size(const RunConfig& config, const std::string& role) {
    for (const auto& cb : config.cb_configs) {
        if (cb.role == role) {
            return cb.page_size_bytes;
        }
    }
    if (!config.cb_configs.empty()) {
        return config.cb_configs.front().page_size_bytes;
    }
    return 2048;
}

void create_circular_buffers(Program& program, const tt::tt_metal::CoreCoord& core, const RunConfig& config) {
    for (const auto& cb : config.cb_configs) {
        const uint32_t total_size = cb.num_pages * cb.page_size_bytes;
        CircularBufferConfig cb_config(
            total_size, {{static_cast<uint8_t>(cb.cb_id), parse_data_format(cb.data_format)}});
        cb_config.set_page_size(static_cast<uint8_t>(cb.cb_id), cb.page_size_bytes);
        CreateCircularBuffer(program, core, cb_config);
    }
}

KernelHandle create_kernel(
    Program& program, const tt::tt_metal::CoreCoord& core, const KernelSpec& kernel) {
    if (kernel.core_type == "trisc") {
        return CreateKernel(
            program,
            kernel.kernel_path,
            core,
            ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = false,
                .math_approx_mode = false,
                .compile_args = kernel.compile_time_args});
    }

    DataMovementProcessor processor = DataMovementProcessor::RISCV_0;
    NOC noc = NOC::RISCV_0_default;
    if (kernel.core_type == "ncrisc") {
        processor = DataMovementProcessor::RISCV_1;
        noc = NOC::RISCV_1_default;
    }

    return CreateKernel(
        program,
        kernel.kernel_path,
        core,
        DataMovementConfig{
            .processor = processor,
            .noc = noc,
            .compile_args = kernel.compile_time_args});
}

bool kernel_needs_scratch_l1(const KernelSpec& kernel) {
    for (const auto& arg : kernel.runtime_args) {
        if (arg.kind == "scratch_l1_buffer_addr32") {
            return true;
        }
    }
    return false;
}

std::vector<uint32_t> build_runtime_args(
    const KernelSpec& kernel,
    const RunConfig& config,
    uint32_t current_work_linear_id,
    const distributed::MeshBuffer& input_buffer,
    const distributed::MeshBuffer& output_buffer,
    const distributed::MeshBuffer* scratch_l1_buffer) {
    std::vector<uint32_t> args;
    size_t scalar_index = 0;
    const uint32_t tile_size = choose_page_size(config, "input");
    const uint64_t src_addr = input_buffer.address();
    const uint64_t dst_addr = output_buffer.address();

    for (const auto& arg : kernel.runtime_args) {
        if (arg.kind == "input_buffer_addr") {
            args.push_back(static_cast<uint32_t>(src_addr & 0xFFFFFFFF));
            args.push_back(static_cast<uint32_t>(src_addr >> 32));
        } else if (arg.kind == "input_buffer_addr32") {
            args.push_back(static_cast<uint32_t>(src_addr));
        } else if (arg.kind == "output_buffer_addr") {
            args.push_back(static_cast<uint32_t>(dst_addr & 0xFFFFFFFF));
            args.push_back(static_cast<uint32_t>(dst_addr >> 32));
        } else if (arg.kind == "output_buffer_addr32") {
            args.push_back(static_cast<uint32_t>(dst_addr));
        } else if (arg.kind == "tile_count") {
            args.push_back(tile_size == 0 ? 0 : static_cast<uint32_t>(config.input_size_bytes / tile_size));
        } else if (arg.kind == "current_work_linear_id") {
            args.push_back(current_work_linear_id);
        } else if (arg.kind == "scratch_l1_buffer_addr32") {
            if (scratch_l1_buffer == nullptr) {
                throw std::runtime_error("Spec requested scratch L1 buffer but none was allocated");
            }
            args.push_back(static_cast<uint32_t>(scratch_l1_buffer->address()));
        } else if (arg.kind == "scalar_u32") {
            if (scalar_index >= config.scalar_args.size()) {
                throw std::runtime_error("Spec requested more scalar args than provided");
            }
            args.push_back(config.scalar_args[scalar_index++]);
        } else {
            throw std::runtime_error("Unsupported runtime arg kind: " + arg.kind);
        }
    }

    return args;
}

int main(int argc, char* argv[]) {
    try {
        const RunConfig config = parse_args(argc, argv);

        std::cout << "[Runner] Initializing TT-Metal for " << config.entry_name << "\n";

        auto mesh_device = distributed::MeshDevice::create_unit_mesh(0);
        auto& cq = mesh_device->mesh_command_queue();
        constexpr tt::tt_metal::CoreCoord core = {0, 0};

        std::cout << "[Runner] Reading input: " << config.input_path
                  << " (" << config.input_size_bytes << " bytes)\n";
        std::vector<uint8_t> input_data = read_file(config.input_path, config.input_size_bytes);

        distributed::DeviceLocalBufferConfig input_dram_config{
            .page_size = choose_page_size(config, "input"),
            .buffer_type = BufferType::DRAM};
        distributed::DeviceLocalBufferConfig output_dram_config{
            .page_size = choose_page_size(config, "output"),
            .buffer_type = BufferType::DRAM};

        distributed::ReplicatedBufferConfig input_buffer_config{.size = config.input_size_bytes};
        auto input_buffer = distributed::MeshBuffer::create(
            input_buffer_config, input_dram_config, mesh_device.get());

        distributed::ReplicatedBufferConfig output_buffer_config{.size = config.output_size_bytes};
        auto output_buffer = distributed::MeshBuffer::create(
            output_buffer_config, output_dram_config, mesh_device.get());

        std::shared_ptr<distributed::MeshBuffer> scratch_l1_buffer;
        bool needs_scratch_l1 = false;
        for (const auto& kernel_spec : config.kernels) {
            if (kernel_needs_scratch_l1(kernel_spec)) {
                needs_scratch_l1 = true;
                break;
            }
        }
        if (needs_scratch_l1) {
            uint32_t scratch_size = choose_page_size(config, "input");
            for (const auto& cb : config.cb_configs) {
                scratch_size = std::max(scratch_size, cb.num_pages * cb.page_size_bytes);
            }
            distributed::DeviceLocalBufferConfig scratch_l1_config{
                .page_size = scratch_size,
                .buffer_type = BufferType::L1};
            distributed::ReplicatedBufferConfig scratch_l1_buffer_config{.size = scratch_size};
            scratch_l1_buffer = distributed::MeshBuffer::create(
                scratch_l1_buffer_config, scratch_l1_config, mesh_device.get());
        }

        EnqueueWriteMeshBuffer(cq, input_buffer, input_data, /*blocking=*/true);
        std::cout << "[Runner] Input transferred\n";

        std::vector<uint32_t> work_ids;
        for (const auto& packet : config.core_plan.work_packets) {
            for (uint32_t i = 0; i < packet.work_count; ++i) {
                work_ids.push_back(packet.work_offset + i);
            }
        }
        if (work_ids.empty()) {
            work_ids.push_back(0);
        }

        std::cout << "[Runner] Executing " << work_ids.size() << " logical work items\n";
        for (uint32_t work_id : work_ids) {
            Program program = CreateProgram();
            create_circular_buffers(program, core, config);

            for (const auto& kernel_spec : config.kernels) {
                std::cout << "[Runner] Loading kernel: " << kernel_spec.kernel_path
                          << " for work_id=" << work_id << "\n";
                KernelHandle kernel = create_kernel(program, core, kernel_spec);
                auto runtime_args = build_runtime_args(
                    kernel_spec,
                    config,
                    work_id,
                    *input_buffer,
                    *output_buffer,
                    scratch_l1_buffer ? scratch_l1_buffer.get() : nullptr);
                SetRuntimeArgs(program, kernel, core, runtime_args);
            }

            distributed::MeshWorkload workload;
            distributed::MeshCoordinateRange device_range(mesh_device->shape());
            workload.add_program(device_range, std::move(program));
            distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/true);
        }

        std::vector<uint8_t> output_data;
        distributed::EnqueueReadMeshBuffer(cq, output_data, output_buffer, /*blocking=*/true);
        std::cout << "[Runner] Output received (" << output_data.size() << " bytes)\n";

        write_file(config.output_path, output_data);
        std::cout << "[Runner] Output written to: " << config.output_path << "\n";
        std::cout << "[Runner] SUCCESS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[Runner] ERROR: " << e.what() << "\n";
        return 1;
    }
}
