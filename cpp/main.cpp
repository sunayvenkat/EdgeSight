


#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

constexpr std::size_t kElements = 3 * 640 * 640;
const std::vector<int64_t> kInputShape = {1, 3, 640, 640};

double percentile(std::vector<double> values, double p) {
    if (values.empty()) {
        throw std::runtime_error("No measurements");
    }

    std::sort(values.begin(), values.end());

    double position = (values.size() - 1) * p / 100.0;
    std::size_t low = static_cast<std::size_t>(position);
    std::size_t high = std::min(low + 1, values.size() - 1);

    return values[low] +
           (values[high] - values[low]) * (position - low);
}

double elapsed_ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(
        b - a
    ).count();
}

std::vector<float> load_inputs(
    const fs::path& path,
    std::size_t frame_count
) {
    std::uintmax_t expected =
        frame_count * kElements * sizeof(float);

    if (fs::file_size(path) != expected) {
        throw std::runtime_error(
            "Input binary size does not match frame count"
        );
    }

    std::vector<float> data(frame_count * kElements);

    std::ifstream file(path, std::ios::binary);

    file.read(
        reinterpret_cast<char*>(data.data()),
        static_cast<std::streamsize>(expected)
    );

    if (!file) {
        throw std::runtime_error("Could not read inputs");
    }

    return data;
}

void save_outputs(
    const std::vector<Ort::Value>& outputs,
    const fs::path& directory
) {
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        const auto info = outputs[i].GetTensorTypeAndShapeInfo();

        if (info.GetElementType() !=
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            throw std::runtime_error(
                "Expected FP32 model output"
            );
        }

        std::size_t count = info.GetElementCount();
        const float* data = outputs[i].GetTensorData<float>();

        fs::path path =
            directory / ("output_" + std::to_string(i) + ".bin");

        std::ofstream file(path, std::ios::binary);

        file.write(
            reinterpret_cast<const char*>(data),
            static_cast<std::streamsize>(count * sizeof(float))
        );

        if (!file) {
            throw std::runtime_error(
                "Could not save output tensor"
            );
        }
    }
}

int main(int argc, char* argv[]) {
    try {
        if (argc != 10) {
            std::cerr
                << "Usage: edgesight_cpp MODEL INPUTS OUTPUT_DIR "
                << "FRAMES THREADS GRAPH SPIN ITERATIONS WARMUP\n";
            return 1;
        }

        fs::path model_path = argv[1];
        fs::path inputs_path = argv[2];
        fs::path output_dir = argv[3];

        int frame_count = std::stoi(argv[4]);
        int threads = std::stoi(argv[5]);
        std::string graph = argv[6];
        int spin = std::stoi(argv[7]);
        int iterations = std::stoi(argv[8]);
        int warmup = std::stoi(argv[9]);

        if (frame_count < 1 || threads < 0 ||
            (graph != "all" && graph != "basic") ||
            (spin != 0 && spin != 1) ||
            iterations < 2 || warmup < 0) {
            throw std::invalid_argument(
                "Invalid benchmark arguments"
            );
        }

        // Load all inputs before measuring inference.
        std::vector<float> data =
            load_inputs(inputs_path, frame_count);

        Ort::Env env(
            ORT_LOGGING_LEVEL_WARNING,
            "EdgeSightNative"
        );

        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(threads);
        options.SetInterOpNumThreads(1);
        options.SetExecutionMode(
            ExecutionMode::ORT_SEQUENTIAL
        );

        options.SetGraphOptimizationLevel(
            graph == "all"
                ? GraphOptimizationLevel::ORT_ENABLE_ALL
                : GraphOptimizationLevel::ORT_ENABLE_BASIC
        );

        options.AddConfigEntry(
            "session.intra_op.allow_spinning",
            spin ? "1" : "0"
        );

        auto init_start = Clock::now();

        Ort::Session session(
            env,
            model_path.c_str(),
            options
        );

        auto init_end = Clock::now();

        // Verify the model input.
        if (session.GetInputCount() != 1) {
            throw std::runtime_error(
                "Expected one model input"
            );
        }

        auto input_info = session.GetInputTypeInfo(0)
            .GetTensorTypeAndShapeInfo();

        if (input_info.GetShape() != kInputShape ||
            input_info.GetElementType() !=
                ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            throw std::runtime_error(
                "Expected FP32 [1,3,640,640] input"
            );
        }

        Ort::AllocatorWithDefaultOptions allocator;

        // Copy names into strings so they remain valid.
        auto allocated_name =
            session.GetInputNameAllocated(0, allocator);

        std::string input_name = allocated_name.get();
        const char* input_names[] = {input_name.c_str()};

        std::vector<std::string> output_names;
        output_names.reserve(session.GetOutputCount());

        for (std::size_t i = 0;
             i < session.GetOutputCount(); ++i) {
            auto name =
                session.GetOutputNameAllocated(i, allocator);
            output_names.emplace_back(name.get());
        }

        std::vector<const char*> output_ptrs;

        for (const auto& name : output_names) {
            output_ptrs.push_back(name.c_str());
        }

        Ort::MemoryInfo memory_info =
            Ort::MemoryInfo::CreateCpu(
                OrtArenaAllocator,
                OrtMemTypeDefault
            );

        // Create input tensors outside timed inference.
        std::vector<Ort::Value> input_tensors;
        input_tensors.reserve(frame_count);

        for (int i = 0; i < frame_count; ++i) {
            float* pointer =
                data.data() + i * kElements;

            input_tensors.push_back(
                Ort::Value::CreateTensor<float>(
                    memory_info,
                    pointer,
                    kElements,
                    kInputShape.data(),
                    kInputShape.size()
                )
            );
        }

        Ort::RunOptions run_options;

        // Save reference outputs for the first image.
        fs::create_directories(output_dir);

        {
            auto outputs = session.Run(
                run_options,
                input_names,
                &input_tensors[0],
                1,
                output_ptrs.data(),
                output_ptrs.size()
            );

            save_outputs(outputs, output_dir);
        }

        // Warm up execution.
        for (int i = 0; i < warmup; ++i) {
            int index = i % frame_count;

            auto outputs = session.Run(
                run_options,
                input_names,
                &input_tensors[index],
                1,
                output_ptrs.data(),
                output_ptrs.size()
            );
        }

        std::vector<double> latencies;
        latencies.reserve(iterations);

        auto benchmark_start = Clock::now();

        for (int i = 0; i < iterations; ++i) {
            int index = i % frame_count;

            auto start = Clock::now();

            {
                auto outputs = session.Run(
                    run_options,
                    input_names,
                    &input_tensors[index],
                    1,
                    output_ptrs.data(),
                    output_ptrs.size()
                );
            }

            auto end = Clock::now();

            latencies.push_back(elapsed_ms(start, end));
        }

        auto benchmark_end = Clock::now();

        double total_ms = elapsed_ms(
            benchmark_start,
            benchmark_end
        );

        double mean = std::accumulate(
            latencies.begin(), latencies.end(), 0.0
        ) / latencies.size();

        double median = percentile(latencies, 50);
        double p95 = percentile(latencies, 95);
        double fps = iterations * 1000.0 / total_ms;

        double initialization_ms = elapsed_ms(
            init_start, init_end
        );

        fs::path report_path =
            output_dir / "benchmark.json";

        std::ofstream report(report_path);

        report << std::fixed << std::setprecision(6);
        report << "{\n";
        report << "  \"backend\": \"ONNX Runtime C++\",\n";
        report << "  \"ort_version\": \""
               << Ort::GetVersionString() << "\",\n";
        report << "  \"frames\": " << frame_count << ",\n";
        report << "  \"threads\": " << threads << ",\n";
        report << "  \"graph\": \"" << graph << "\",\n";
        report << "  \"spinning\": "
               << (spin ? "true" : "false") << ",\n";
        report << "  \"iterations\": " << iterations << ",\n";
        report << "  \"warmup\": " << warmup << ",\n";
        report << "  \"session_initialization_ms\": "
               << initialization_ms << ",\n";
        report << "  \"mean_ms\": " << mean << ",\n";
        report << "  \"median_ms\": " << median << ",\n";
        report << "  \"p95_ms\": " << p95 << ",\n";
        report << "  \"throughput_fps\": " << fps << "\n";
        report << "}\n";

        if (!report) {
            throw std::runtime_error(
                "Could not save benchmark report"
            );
        }

        std::cout << "\nEdgeSight C++ Inference\n";
        std::cout << "--------------------------\n";
        std::cout << "ONNX Runtime: "
                  << Ort::GetVersionString() << "\n";
        std::cout << "Median: " << median << " ms\n";
        std::cout << "P95: " << p95 << " ms\n";
        std::cout << "Throughput: " << fps << " FPS\n";
        std::cout << "Saved: " << report_path << "\n";

        return 0;

    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n";
        return 1;
    }
}
