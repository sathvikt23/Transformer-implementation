"""
Target Token Budget: 100 Billion Tokens Baseline.
"""

# Total Pretraining Token Budget Baseline
TOTAL_TOKEN_BUDGET = 100_000_000_000  # 100 Billion Tokens

# Lane Percentage Allocations (% of total pretraining budget)
LANE_ALLOCATIONS = {
    "general_web": 0.42,      # 42% -> 42.0B Tokens
    "code": 0.23,             # 23% -> 23.0B Tokens
    "math_reasoning": 0.15,   # 15% -> 15.0B Tokens
    "indic": 0.10,            # 10% -> 10.0B Tokens (Protected Floor >= 8%)
    "agentic": 0.05,          # 5%  -> 5.0B Tokens  (Protected Floor >= 3%)
    "long_context": 0.05,     # 5%  -> 5.0B Tokens
}

# Internal General Web Sub-mix (Normalized from 100MB benchmark pipeline yield)
GENERAL_WEB_SUBMIX = {
    "D1_knowledge": 0.295,    # Wikipedia / Educational bridge (29.5%)
    "D3_general_web": 0.246,   # Diverse filtered web (24.6%)
    "D2_literature": 0.230,    # Quality books & documentation (23.0%)
    "D4_community": 0.230,     # News, forums, QA discussions (23.0%)
}

# Stage 2: Deduplication Settings
DEDUP_CONFIG = {
    "num_perm": 128,          # MinHash LSH permutations
    "jaccard_threshold": 0.90,# Jaccard similarity threshold for duplicates
}

# Stage 4 & 6: Heuristic & Task Filtering Thresholds
FILTER_CONFIG = {
    "min_word_count": 20,
    "max_word_count": 5000,
    "min_alpha_ratio": 0.60,
    "max_punc_ratio": 0.30,
    "max_digit_ratio": 0.40,
    "max_uppercase_ratio": 0.60,
    "min_unique_word_ratio": 0.30,
    "min_sentence_length": 5,
}

# Stage 5: ML Quality Classifier Threshold
FINEWEB_EDU_THRESHOLD = 0.50

# Stage 8: BART-Large-MNLI Target Domains
TARGET_DOMAINS = [
    "D1 Science (arXiv preprints, research papers)",
    "D1 Knowledge (Wikipedia, reference, educational text)",
    "D2 Code and Mathematics (source code, algorithms, math problems)",
    "D2 Literature (books, formal documentation, structured text)",
    "D3 General Web (diverse filtered web content, news, articles)",
    "D4 Community (forums, QA discussions, social commentary)",
]

# Curriculum Stage Target Ratios
CURRICULUM_STAGES = {
    "seed": {
        "general_web": 0.55,
        "code": 0.15,
        "math_reasoning": 0.14,
        "indic": 0.08,
        "agentic": 0.03,
        "long_context": 0.05,
    },
    "mid": {
        "general_web": 0.42,
        "code": 0.23,
        "math_reasoning": 0.15,
        "indic": 0.10,
        "agentic": 0.05,
        "long_context": 0.05,
    },
    "late": {
        "general_web": 0.25,
        "code": 0.30,
        "math_reasoning": 0.22,
        "indic": 0.11,
        "agentic": 0.07,
        "long_context": 0.05,
    },
    "anneal": {
        "general_web": 0.10,
        "code": 0.32,
        "math_reasoning": 0.25,
        "indic": 0.18,  # Spending verified Indic reserve
        "agentic": 0.10, # Spending top agentic reserve
        "long_context": 0.05,
    },
}
