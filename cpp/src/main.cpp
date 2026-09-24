#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include "text_preprocessor.hpp"
#include "laya_primitives.hpp"
#include "erp_advisor.hpp"
#include "inference_engine.hpp" // Header included for documentation

// Optional ONNX Runtime
// #include "inference_engine.hpp"

#ifdef _WIN32
// Basic colors for windows if supported, or empty
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
    std::vector<float> mock_logits;
};

void DemoWithRawLogits(const TestCase& test) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::cout << ANSI_COLOR_YELLOW << "\nRunning Test Case: " << test.description << ANSI_COLOR_RESET << "\n";
    std::cout << "Original Text: " << test.text << "\n";
    
    // 1. Prune
    std::string pruned = TextPreprocessor::PruneToSignalSentences(test.text);
    std::cout << "Pruned Text: " << pruned << "\n";
    std::cout << "Compression Ratio: " << TextPreprocessor::GetCompressionRatio() * 100.0f << "%\n";
    
    // 2. Tokenize (mocked)
    std::cout << "[NOTE] Tokenization requires the HuggingFace tokenizers C library. Using placeholder tokens.\n";
    std::vector<int64_t> input_ids = {101, 1000, 1001, 102}; // Mocked
    std::vector<int64_t> attention_mask = {1, 1, 1, 1};
    
    // 3. Inference (mocked with logits)
    // std::vector<float> logits = engine.RunInference(input_ids, attention_mask);
    const float* logits = test.mock_logits.data();
    
    // 4. Laya System-1 Evaluate
    LayaVerdict verdict = LayaSystem1Engine::Evaluate(logits, 3);
    verdict.action_recommendations = ERPAdvisor::GenerateRecommendations(verdict);
    
    // 5. Generate Report
    std::string report = ERPAdvisor::FormatReport(verdict);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end_time - start_time;
    
    std::cout << (verdict.risk_grade == RiskGrade::CRITICAL ? ANSI_COLOR_RED : ANSI_COLOR_GREEN);
    std::cout << report << ANSI_COLOR_RESET;
    std::cout << "Total Pipeline Time: " << duration.count() << " ms\n\n";
}

int main(int argc, char** argv) {
    std::string model_path = "../models/finbert.onnx";
    bool use_cuda = false;
    std::string text_input = "";
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--cuda") {
            use_cuda = true;
        } else if (arg == "--text" && i + 1 < argc) {
            text_input = argv[++i];
        }
    }
    
    // Tests: order Negative, Neutral, Positive
    std::vector<TestCase> tests = {
        {
            "Despite 10% revenue growth, the company breached its debt covenants. Severe liquidity distress.",
            "Multi-clause distress with revenue growth mask",
            {3.5f, 0.2f, -1.0f} // High Negative
        },
        {
            "Unprecedented EBITDA growth of 45%. Margin expansion is clearly visible.",
            "High growth & margin expansion",
            {-2.0f, 0.5f, 4.5f} // High Positive
        },
        {
            "The company filed its 10-Q report yesterday.",
            "Neutral governance filing",
            {-1.0f, 3.0f, -0.5f} // High Neutral
        },
        {
            "Supplier defaults led to massive inventory write-downs. We face significant bankruptcy risks.",
            "Supplier default and inventory write-downs",
            {4.0f, -0.5f, -2.0f} // High Negative
        },
        {
            "FX headwinds reduced GAAP earnings, but organic ARR growth remained strong.",
            "FX headwinds but organic ARR growth",
            {-0.5f, 1.0f, 2.5f} // Positive/Neutral
        }
    };
    
    try {
        if (!text_input.empty()) {
            TestCase t;
            t.text = text_input;
            t.description = "User Input";
            t.mock_logits = {0.0f, 1.0f, 0.0f}; // Default Neutral
            DemoWithRawLogits(t);
        } else {
            for (const auto& test : tests) {
                DemoWithRawLogits(test);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << ANSI_COLOR_RED << "Error: " << e.what() << ANSI_COLOR_RESET << "\n";
        return 1;
    }
    
    return 0;
}
