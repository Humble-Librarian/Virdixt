#pragma once
#include "laya_primitives.hpp"
#include <string>
#include <vector>
#include <sstream>

class ERPAdvisor {
public:
    static std::vector<std::string> GenerateRecommendations(const LayaVerdict& verdict) {
        std::vector<std::string> recs;
        switch (verdict.risk_grade) {
            case RiskGrade::CRITICAL:
                recs.push_back("Freeze Purchase Orders");
                recs.push_back("Require Upfront Cash for new deals");
                recs.push_back("Request Covenant Compliance Certificate");
                break;
            case RiskGrade::WARNING:
                recs.push_back("Flag for Review");
                recs.push_back("Request Updated Financials");
                recs.push_back("Increase Monitoring Frequency");
                break;
            case RiskGrade::MONITOR:
                recs.push_back("Standard Due Diligence");
                recs.push_back("Note in CRM");
                break;
            case RiskGrade::MINIMAL:
                recs.push_back("Proceed Normal");
                recs.push_back("Eligible for Volume Credit Extension");
                break;
        }
        return recs;
    }
    
    static std::string FormatReport(const LayaVerdict& verdict) {
        std::stringstream ss;
        ss << "=========================================\n";
        ss << "           LAYA VERDICT REPORT           \n";
        ss << "=========================================\n";
        
        std::string risk_str;
        switch (verdict.risk_grade) {
            case RiskGrade::CRITICAL: risk_str = "CRITICAL"; break;
            case RiskGrade::WARNING: risk_str = "WARNING"; break;
            case RiskGrade::MONITOR: risk_str = "MONITOR"; break;
            case RiskGrade::MINIMAL: risk_str = "MINIMAL"; break;
        }
        
        std::string choice_str;
        switch (verdict.choice) {
            case SentimentChoice::NEGATIVE: choice_str = "NEGATIVE"; break;
            case SentimentChoice::NEUTRAL: choice_str = "NEUTRAL"; break;
            case SentimentChoice::POSITIVE: choice_str = "POSITIVE"; break;
        }
        
        ss << "Risk Grade      : " << risk_str << "\n";
        ss << "Sentiment       : " << choice_str << " (Confidence: " << verdict.confidence << ")\n";
        ss << "Distress Score  : " << verdict.distress_score << " / 100\n";
        ss << "\n--- NOUL Hypotheses ---\n";
        ss << "Liquidity Distress       : " << verdict.noul.liquidity_distress << "\n";
        ss << "Debt Covenant Breach     : " << verdict.noul.debt_covenant_breach << "\n";
        ss << "Growth/Expansion Momentum: " << verdict.noul.growth_expansion_momentum << "\n";
        ss << "Capital Return Sustain   : " << verdict.noul.capital_return_sustainable << "\n";
        ss << "\n--- Recommended ERP Actions ---\n";
        for (const auto& rec : verdict.action_recommendations) {
            ss << "- " << rec << "\n";
        }
        ss << "=========================================\n";
        return ss.str();
    }
};
