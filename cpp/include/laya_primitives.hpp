#pragma once
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

enum class SentimentChoice { NEGATIVE = 0, NEUTRAL = 1, POSITIVE = 2 };
enum class RiskGrade { MINIMAL, MONITOR, WARNING, CRITICAL };
enum class SAPActionFlag { PROCEED_NORMAL, FLAG_FOR_REVIEW, FREEZE_PURCHASE_ORDERS };

struct NoulHypotheses {
    float liquidity_distress;
    float debt_covenant_breach;
    float growth_expansion_momentum;
    float capital_return_sustainable;
};

struct LayaVerdict {
    SentimentChoice choice;
    float confidence;
    float distress_score;
    NoulHypotheses noul;
    RiskGrade risk_grade;
    SAPActionFlag sap_action;
    std::vector<std::string> action_recommendations;
};

class LayaSystem1Engine {
public:
    static LayaVerdict Evaluate(const float* raw_logits, int num_logits = 3, float temperature = 1.25f) {
        LayaVerdict verdict;
        
        // Temperature-scaled softmax
        float max_logit = -1e9f;
        for (int i = 0; i < num_logits; ++i) {
            if (raw_logits[i] > max_logit) max_logit = raw_logits[i];
        }
        
        float sum_exp = 0.0f;
        std::vector<float> probs(num_logits);
        for (int i = 0; i < num_logits; ++i) {
            probs[i] = std::exp((raw_logits[i] - max_logit) / temperature);
            sum_exp += probs[i];
        }
        
        for (int i = 0; i < num_logits; ++i) {
            probs[i] /= sum_exp;
        }
        
        // Assuming Standard Order: Negative(0), Neutral(1), Positive(2)
        float p_neg = probs[0];
        float p_neu = probs[1];
        float p_pos = probs[2];
        
        if (p_neg > p_neu && p_neg > p_pos) {
            verdict.choice = SentimentChoice::NEGATIVE;
            verdict.confidence = p_neg;
        } else if (p_pos > p_neg && p_pos > p_neu) {
            verdict.choice = SentimentChoice::POSITIVE;
            verdict.confidence = p_pos;
        } else {
            verdict.choice = SentimentChoice::NEUTRAL;
            verdict.confidence = p_neu;
        }
        
        // Distress score calculation: score = (p_neg * 100) + (p_neu * 15) - (p_pos * 35)
        float score = (p_neg * 100.0f) + (p_neu * 15.0f) - (p_pos * 35.0f);
        verdict.distress_score = std::max(0.0f, std::min(100.0f, score));
        
        // Noul hypothesis computation using sigmoid transforms of logits
        auto sigmoid = [](float x) { return 1.0f / (1.0f + std::exp(-x)); };
        verdict.noul.liquidity_distress = sigmoid(raw_logits[0]); 
        verdict.noul.debt_covenant_breach = sigmoid(raw_logits[0] * 1.2f);
        verdict.noul.growth_expansion_momentum = sigmoid(raw_logits[2]);
        verdict.noul.capital_return_sustainable = sigmoid(raw_logits[2] * 0.9f);
        
        // Asymmetric risk gate (35% negative threshold triggers WARNING)
        if (p_neg > 0.60f || verdict.distress_score > 70.0f) {
            verdict.risk_grade = RiskGrade::CRITICAL;
            verdict.sap_action = SAPActionFlag::FREEZE_PURCHASE_ORDERS;
        } else if (p_neg > 0.35f || verdict.distress_score > 40.0f) {
            verdict.risk_grade = RiskGrade::WARNING;
            verdict.sap_action = SAPActionFlag::FLAG_FOR_REVIEW;
        } else if (p_neu > 0.50f || verdict.distress_score > 20.0f) {
            verdict.risk_grade = RiskGrade::MONITOR;
            verdict.sap_action = SAPActionFlag::PROCEED_NORMAL;
        } else {
            verdict.risk_grade = RiskGrade::MINIMAL;
            verdict.sap_action = SAPActionFlag::PROCEED_NORMAL;
        }
        
        return verdict;
    }
};
