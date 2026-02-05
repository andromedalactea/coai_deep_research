# Usage Guide: Running Computational Scientific Discovery

## Quick Answer to Your Question

**Why was it using gpt-4o?**

The scripts I initially created had **hardcoded defaults** that ignored your `.env` file. I've now fixed both scripts to:

✅ Read `RESEARCH_MODEL` from `.env` (you have: `openai:gpt-4.1`)  
✅ Read `FINAL_REPORT_MODEL` from `.env` (you have: `openai:gpt-5.2`)

Your models in `.env`:

```26:34:.env
# Use cheap model for summarization (high volume)
SUMMARIZATION_MODEL=openai:gpt-4.1-mini

# Use mid-tier for research (still good quality)
RESEARCH_MODEL=openai:gpt-4.1
COMPRESSION_MODEL=openai:gpt-4.1

# Use premium only for final report
FINAL_REPORT_MODEL=openai:gpt-5.2
```

**Now the scripts will automatically use these settings!**

---

## Running Discovery (3 Methods)

### Method 1: Command Line Script (Recommended)

```bash
# Uses your .env settings automatically
python run_discovery.py --file examples/biosignature-discovery-prompt.txt

# Or with a direct query
python run_discovery.py "Your research question here"

# Quick test (3 iterations)
python run_discovery.py --quick "Test question"

# Verbose mode (see all steps)
python run_discovery.py --verbose "Your question"
```

**What You'll See:**
```
📋 Environment Check:
   ✅ E2B_API_KEY is set
   ✅ OPENAI_API_KEY is set

⚙️  Configuration:
   Research model: openai:gpt-4.1        ← From your .env!
   Final report model: openai:gpt-5.2    ← From your .env!
   Max iterations: 8
   Verbose: True
   Output dir: outputs

📝 Query preview:
   Identify and computationally validate at least one novel...

Start discovery? [Y/n]: 

======================================================================
   COMPUTATIONAL SCIENTIFIC DISCOVERY - RUNNING
======================================================================
Started at: 2026-02-02 20:30:15
----------------------------------------------------------------------

[  0.8s] ❓ clarify_discovery_query      | Iter: 0 | Experiments: 0 | Outputs: 0
[  3.2s] 📋 generate_research_brief      | Iter: 0 | Experiments: 0 | Outputs: 0
[  7.5s] 🧠 discovery_supervisor         | Iter: 1 | Experiments: 0 | Outputs: 0
[ 15.2s] 📚 gather_knowledge             | Iter: 1 | Experiments: 0 | Outputs: 0
         └─ ArXiv search results...
[ 42.8s] 🧠 discovery_supervisor         | Iter: 2 | Experiments: 0 | Outputs: 0
[ 48.1s] 🔬 run_experiment               | Iter: 2 | Experiments: 1 | Outputs: 0  ← Code running!
         └─ Testing: Calculate Gibbs free energy for...
[ 54.3s] 📊 analyze_results              | Iter: 2 | Experiments: 1 | Outputs: 3  ← Figures generated!
...
```

### Method 2: Python Script (Programmatic)

```python
import asyncio
from interactive_discovery import discover

async def main():
    # Uses your .env models automatically
    result = await discover("Your research question")
    
    # View results
    result.summary()
    result.show_figures()
    result.save_report("my_report.md")

asyncio.run(main())
```

### Method 3: Jupyter Notebook

```python
from interactive_discovery import discover

# Run discovery
result = await discover(
    query="Your research question",
    verbose=True
)

# Results
result.summary()
result.show_figures()  # Displays inline in Jupyter
print(result.report)
```

---

## Model Selection Priority

The system follows this priority order:

1. **Command line argument** (highest priority)
   ```bash
   python run_discovery.py --model anthropic:claude-3-5-sonnet "Query"
   ```

2. **Environment variable** (.env file)
   ```bash
   RESEARCH_MODEL=openai:gpt-4.1
   FINAL_REPORT_MODEL=openai:gpt-5.2
   ```

3. **Fallback default** (lowest priority)
   ```
   openai:gpt-4o
   ```

**So if you don't specify `--model`, it will use your `.env` settings!**

---

## Your Current Setup

Based on your `.env`:

```
Research iterations  → openai:gpt-4.1      (cheaper, good quality)
Final report         → openai:gpt-5.2      (premium for synthesis)
```

This is a smart cost-optimization strategy:
- Saves money during iterative discovery
- Uses best model for final report only

---

## Running the Biosignature Discovery

### With Your .env Models (Recommended)

```bash
# This will use gpt-4.1 for discovery, gpt-5.2 for final report
python run_discovery.py \
    --file examples/biosignature-discovery-prompt.txt \
    --verbose \
    --iterations 8 \
    --output biosignature_results
```

### Override Models (If Needed)

```bash
# Use a different model just for this run
python run_discovery.py \
    --model anthropic:claude-3-5-sonnet \
    --final-model openai:gpt-5.2 \
    --file examples/biosignature-discovery-prompt.txt
```

---

## Monitoring Progress

With `--verbose`, you see detailed steps:

```
[  5.3s] 🧠 discovery_supervisor         | Iter: 1 | Experiments: 0 | Outputs: 0
[  8.2s] 📚 gather_knowledge             | Iter: 1 | Experiments: 0 | Outputs: 0
         └─ ArXiv search results...      ← What's happening
[ 15.6s] 🧠 discovery_supervisor         | Iter: 2 | Experiments: 0 | Outputs: 0
[ 22.1s] 🔬 run_experiment               | Iter: 2 | Experiments: 1 | Outputs: 0
         └─ Testing: Calculate Gibbs...  ← Hypothesis being tested
         └─ Code execution: ✅ Success   ← Code ran successfully!
[ 28.4s] 📊 analyze_results              | Iter: 2 | Experiments: 1 | Outputs: 3
         └─ Finding: Significant corr... ← What was found
```

**Key Metrics to Watch:**
- **Experiments count** - Should increase (means code is running)
- **Outputs count** - Should increase (means figures are being generated)
- **🔬 run_experiment** nodes - Confirm code is executing

---

## What's Saved

After completion, in `outputs/` (or your specified directory):

```
outputs/
├── report_20260202_203215.md          # Full research report
├── figure_a1b2c3.png                  # Generated figures
├── figure_d4e5f6.png
├── figure_g7h8i9.png
└── metadata_20260202_203215.json      # Run metadata
```

The report includes:
- All computational results
- Embedded figures: `![Caption](figure_abc123.png)`
- Statistical results (p-values, confidence intervals)
- Numerical calculations (ΔG values, S/N ratios)

---

## Testing the Fix

Quick test to confirm it uses your .env models:

```bash
python run_discovery.py \
    --quick \
    --verbose \
    "Calculate the mean and standard deviation of a random sample and visualize it"
```

Watch the output:
```
⚙️  Configuration:
   Research model: openai:gpt-4.1        ← Should show gpt-4.1 from your .env!
   Final report model: openai:gpt-5.2    ← Should show gpt-5.2 from your .env!
```

---

## All Command Line Options

```bash
python run_discovery.py [OPTIONS] "query"

Required (one of):
  "query text"          Direct query string
  --file, -f FILE       Load query from file

Optional:
  --verbose, -v         Show detailed progress
  --iterations, -i N    Max iterations (default: 8)
  --quick, -q           Quick mode (3 iterations)
  --output, -o DIR      Output directory (default: outputs)
  --model MODEL         Override RESEARCH_MODEL from .env
  --final-model MODEL   Override FINAL_REPORT_MODEL from .env
  --no-save             Don't save outputs to files
```

---

## Troubleshooting

### "Using wrong model"
Check that .env is loaded:
```python
import os
from dotenv import load_dotenv
load_dotenv()
print(os.getenv("RESEARCH_MODEL"))
```

### "Discovery freezes"
- Check iteration count (not hitting max?)
- Use `--verbose` to see where it's stuck
- Watch for `🔬 run_experiment` nodes (code execution)

### "No computational experiments"
The fixes I made should prevent this, but if it happens:
- Check that E2B_API_KEY is valid
- Review the supervisor prompt enforcement
- Look for error messages in verbose output

---

## Summary

✅ **Fixed**: Scripts now read models from your `.env`  
✅ **Your Setup**: gpt-4.1 for research, gpt-5.2 for reports  
✅ **Progress Monitoring**: Real-time updates with `--verbose`  
✅ **No LangGraph Studio Needed**: Runs directly in Python

**Run your biosignature discovery:**

```bash
python run_discovery.py \
    --file examples/biosignature-discovery-prompt.txt \
    --verbose \
    --output biosignature_results
```

This will use your .env models and show you every step of the process!
