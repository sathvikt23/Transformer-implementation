import os
import json
from pipeline_config import (
    TOTAL_TOKEN_BUDGET,
    LANE_ALLOCATIONS,
    GENERAL_WEB_SUBMIX,
    DEDUP_CONFIG,
    FILTER_CONFIG,
    FINEWEB_EDU_THRESHOLD,
    CURRICULUM_STAGES,
)
from cleaning import DataCurationPipeline
from mixture import CurriculumSampler

# Sample Benchmark Documents (dirty, raw HTML, PII, low quality, code, science, Indic)
SAMPLE_DOCUMENTS = [
    {
        "id": "doc_001",
        "text": """<html><body><h1>Buy Cheap Shoes Online!</h1><p>Contact sales at john_doe@shop.com for 50% discount offers! Free shipping http://spam.com/shoes free free free free free free!</p></body></html>"""
    },
    {
        "id": "doc_002",
        "text": """This article investigates quantum error correction codes in fault-tolerant quantum computation. We define the stabilizer formalism for surface codes and evaluate logical qubit decay rates under Pauli noise channels. Mathematical derivations demonstrate quadratic suppression of logical error rate with increasing lattice distance."""
    },
    {
        "id": "doc_003",
        "text": """def binary_search(arr, target):\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            low = mid + 1\n        else:\n            high = mid - 1\n    return -1"""
    },
    {
        "id": "doc_004",
        "text": """The History of Ancient Rome spans over a thousand years, beginning with its legendary founding in 753 BC by Romulus and Remus. Roman civilization expanded from a small kingdom on the Tiber River to an expansive empire encompassing the Mediterranean basin."""
    },
    {
        "id": "doc_005",
        "text": """Duplicate document check: The History of Ancient Rome spans over a thousand years, beginning with its legendary founding in 753 BC by Romulus and Remus. Roman civilization expanded from a small kingdom on the Tiber River to an expansive empire encompassing the Mediterranean basin."""
    },
    {
        "id": "doc_006",
        "text": """Check out this thread on Reddit: Posted by u/tech_user! What is the best framework for asynchronous Python web servers? FastAPI vs Sanic vs Tornado. Leave your comments below! Contact admin@forum.org for moderation inquiries."""
    },
    {
        "id": "doc_007",
        "text": """Buy cheap items 12345 67890 $$$$$$$$$$$ !!!!!!!!!!!!!!!!! buy now now now now now now now now now now now now now now now now now now now now now now now now now now now now now now"""
    },
    {
        "id": "doc_008",
        "text": """भारत एक विशाल देश है जिसकी सांस्कृतिक और भाषाई विविधता अद्वितीय है। भारतीय संविधान में 22 आधिकारिक भाषाओं को मान्यता दी गई है। संकेत संकलन और प्राकृतिक भाषा प्रसंस्करण में भारतीय भाषाओं का महत्व बढ़ रहा है।"""
    }
]


def main():
    print("=" * 70)
    print("      ERA5 Module-5: Data Curation Pipeline & Curriculum Mixture Engine")
    print(f"      Target Token Budget Baseline: {TOTAL_TOKEN_BUDGET / 1e9:.1f} Billion Tokens")
    print("=" * 70)

    # 1. Initialize Pipeline Config
    config = {
        "dedup": DEDUP_CONFIG,
        "filters": FILTER_CONFIG,
        "fineweb_threshold": FINEWEB_EDU_THRESHOLD,
    }

    pipeline = DataCurationPipeline(config)
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    retained_samples = []

    # 2. Run Documents Through 9-Stage Pipeline
    print("\n[Step 1] Executing 9-Stage Data Curation Pipeline...")
    for idx, doc in enumerate(SAMPLE_DOCUMENTS, 1):
        clean_text, meta = pipeline.process_document(doc["text"])

        if clean_text:
            print(f"  [PASS] Doc #{idx} ({doc['id']}) -> Domain: '{meta['domain']}' | Words: {meta['word_count']}")
            retained_samples.append({
                "id": doc["id"],
                "text": clean_text,
                "domain": meta["domain"],
            })
        else:
            print(f"  [DROP] Doc #{idx} ({doc['id']}) -> Filtered by pipeline stage")

    # 3. Print Funnel Analytics Summary
    summary = pipeline.get_funnel_summary()
    print("\n" + "=" * 70)
    print("                      PIPELINE FUNNEL ANALYTICS")
    print("=" * 70)
    print(f"  Total Input Documents : {summary['total_input_docs']}")
    print(f"  Retained Documents    : {summary['retained_docs']}")
    print(f"  Retained Yield Rate   : {summary['yield_percentage']}%")
    print(f"  Filtered Noise Rate   : {summary['noise_filtered_pct']}%")
    print(f"  PII Redacted Docs     : {summary['pii_redacted_docs']}")
    print("\n  Filter Drops by Stage:")
    for stage_name, drop_count in summary["drops"].items():
        print(f"    - {stage_name:<22}: {drop_count} dropped")

    # 4. Save Domain JSONL Output Files
    print("\n[Step 2] Exporting Cleaned Documents to Domain Pools...")
    domain_files = {}
    for item in retained_samples:
        filename = item["domain"].split()[0].lower() + "_pool.jsonl"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        domain_files[item["domain"]] = filepath

    for dom, path in domain_files.items():
        print(f"  Saved domain pool '{dom}' -> [output/{os.path.basename(path)}]")

    # 5. Demonstrate Curriculum Mixture Engine
    print("\n" + "=" * 70)
    print("             CURRICULUM MIXTURE ENGINE (100B BASELINE)")
    print("=" * 70)
    sampler = CurriculumSampler(CURRICULUM_STAGES, TOTAL_TOKEN_BUDGET)

    batch_token_budget = 100_000  # 100k token training step batch size
    for stage in ["seed", "mid", "late", "anneal"]:
        ratios = sampler.get_stage_ratios(stage)
        tokens = sampler.compute_token_allocations(stage, batch_token_budget)
        print(f"\n  Stage: [{stage.upper()}] (Batch Token Budget = {batch_token_budget:,} tokens)")
        for lane, share in ratios.items():
            print(f"    - {lane:<16}: {share * 100:>5.1f}% ({tokens[lane]:>6,} tokens)")

    print("\n" + "=" * 70)
    print("Pipeline & Curriculum Execution Complete Success!")
    print("=" * 70)


if __name__ == "__main__":
    main()
