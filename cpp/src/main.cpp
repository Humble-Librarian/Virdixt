#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include "text_preprocessor.hpp"
#include "laya_primitives.hpp"
#include "erp_advisor.hpp"
#include "inference_engine.hpp"

#ifdef _WIN32
#define ANSI_COLOR_RED ""
#define ANSI_COLOR_GREEN ""
#define ANSI_COLOR_YELLOW ""
#define ANSI_COLOR_RESET ""
#else
#define ANSI_COLOR_RED "\x1b[31m"
#define ANSI_COLOR_GREEN "\x1b[32m"
#define ANSI_COLOR_YELLOW "\x1b[33m"
#define ANSI_COLOR_RESET "\x1b[0m"
#endif

struct TestCase {
    std::string text;
    std::string description;
    std::vector<float> sample_logits;
};

void RunPipeline(InferenceEngine& engine, const TestCase& test, bool use_onnx_model = false) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::cout << ANSI_COLOR_YELLOW << "\nRunning Scenario: " << test.description << ANSI_COLOR_RESET << "\n";
    std::cout << "Original Text: " << test.text << "\n";
    
    // 1. Layer 1: Pruning
    auto t0 = std::chrono::high_resolution_clock::now();
    std::string pruned = TextPreprocessor::PruneToSignalSentences(test.text);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> prune_duration = t1 - t0;
    
    std::cout << "Pruned Text  : " << (pruned.empty() ? test.text : pruned) << "\n";
    std::cout << "Layer 1 Time : " << prune_duration.count() << " ms (Compression: "
              << TextPreprocessor::GetCompressionRatio() * 100.0f << "%)\n";
    
    // 2. Layer 2: Forward Pass & Logits
    std::vector<float> logits;
    if (use_onnx_model && engine.IsReady()) {
        std::vector<int64_t> input_ids = {101, 1000, 1001, 102}; // Mocked tokens for demo
        std::vector<int64_t> attention_mask = {1, 1, 1, 1};
        logits = engine.RunInference(input_ids, attention_mask);
    } else {
        logits = test.sample_logits;
    }
    
    // 3. Layer 3: Laya System-1 Primitive Evaluation
    LayaVerdict verdict = LayaSystem1Engine::Evaluate(logits.data(), static_cast<int>(logits.size()));
    verdict.action_recommendations = ERPAdvisor::GenerateRecommendations(verdict);
    
    // 4. Layer 4: Generate ERP Audit & Action Report
    std::string report = ERPAdvisor::FormatReport(verdict);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> total_duration = end_time - start_time;
    
    std::cout << (verdict.risk_grade == RiskGrade::CRITICAL ? ANSI_COLOR_RED : ANSI_COLOR_GREEN);
    std::cout << report << ANSI_COLOR_RESET;
    std::cout << "Total End-to-End Pipeline Time: " << total_duration.count() << " ms\n";
}

int main(int argc, char** argv) {
    std::string model_path = "../models/finbert.onnx";
    bool use_cuda = false;
    std::string text_input = "";
    bool run_live_onnx = false;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
            run_live_onnx = true;
        } else if (arg == "--cuda") {
            use_cuda = true;
        } else if (arg == "--live-onnx") {
            run_live_onnx = true;
        } else if (arg == "--text" && i + 1 < argc) {
            text_input = argv[++i];
        }
    }
    
    std::cout << "===================================================================\n";
    std::cout << "       VIRDIXT: HIGH-THROUGHPUT C++ DECISION ENGINE & ERP ADVISOR\n";
    std::cout << "===================================================================\n";
    
    InferenceEngine engine(model_path, use_cuda);
    
    // Test scenarios covering the financial spectrum
    std::vector<TestCase> scenarios = {
        {
            "The company experienced a severe decline in liquidity and breached its debt covenant, although revenue showed a slight 2% growth.",
            "Multi-Clause Distress with Revenue Growth Mask",
            {3.5f, -0.8f, -2.1f} // High Negative
        },
        {
            "Organic ARR grew by 45% and EBITDA margin expanded significantly over the fiscal year.",
            "High Growth & Margin Expansion",
            {-2.5f, 0.2f, 4.2f} // High Positive
        },
        {
            "The board filed a standard 8-K regarding the appointment of a new independent director.",
            "Neutral Governance Filing",
            {-1.2f, 3.4f, -0.9f} // High Neutral
        },
        {
            "Our primary supplier defaulted on their obligations, leading to $5M in inventory write-downs and margin contraction.",
            "Supplier Default & Severe Inventory Write-Downs",
            {4.1f, -0.5f, -2.4f} // High Negative
        },
        {
            "Despite severe FX headwinds reducing international profits, domestic revenue grew robustly by 15%.",
            "FX Headwinds with Resilient Domestic Growth",
            {-0.8f, 0.9f, 2.7f} // Positive
        }
    };
    
    try {
        if (!text_input.empty()) {
            TestCase custom_case;
            custom_case.text = text_input;
            custom_case.description = "User Document Input";
            custom_case.sample_logits = {0.0f, 1.0f, 0.0f};
            RunPipeline(engine, custom_case, run_live_onnx);
        } else {
            for (const auto& scenario : scenarios) {
                RunPipeline(engine, scenario, run_live_onnx);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << ANSI_COLOR_RED << "Fatal Error: " << e.what() << ANSI_COLOR_RESET << "\n";
        return 1;
    }
    
    return 0;
}
