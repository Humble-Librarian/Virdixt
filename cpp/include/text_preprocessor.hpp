#pragma once
#include <string>
#include <vector>
#include <regex>
#include <iostream>

class TextPreprocessor {
private:
    static inline float last_compression_ratio_ = 0.0f;
    
public:
    static std::string PruneToSignalSentences(const std::string& raw_text) {
        if (raw_text.empty()) {
            last_compression_ratio_ = 1.0f;
            return "";
        }
        
        std::regex sentence_delims("([.!?]+)|(\\n+)");
        std::sregex_token_iterator iter(raw_text.begin(), raw_text.end(), sentence_delims, -1);
        std::sregex_token_iterator end;
        
        std::vector<std::string> sentences;
        for (; iter != end; ++iter) {
            std::string s = *iter;
            // Trim
            s.erase(0, s.find_first_not_of(" \t\r\n"));
            s.erase(s.find_last_not_of(" \t\r\n") + 1);
            if (!s.empty()) {
                sentences.push_back(s);
            }
        }
        
        std::regex signal_words_regex("(?i)\\b(revenue|profit|loss|margin|debt|covenant|EBITDA|impairment|growth|decline|default|bankruptcy|dividend|buyback|guidance|earnings|cash flow)\\b|[$%]");
        
        std::string pruned_text;
        for (const auto& s : sentences) {
            if (std::regex_search(s, signal_words_regex)) {
                pruned_text += s + ". ";
            }
        }
        
        if (!pruned_text.empty()) {
            pruned_text.pop_back(); // space
        }
        
        float orig_size = static_cast<float>(raw_text.size());
        float new_size = static_cast<float>(pruned_text.size());
        
        last_compression_ratio_ = (orig_size > 0.0f) ? (new_size / orig_size) : 1.0f;
        
        return pruned_text;
    }
    
    static float GetCompressionRatio() {
        return last_compression_ratio_;
    }
};
