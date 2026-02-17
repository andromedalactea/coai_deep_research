# Using DeepSeek Models with the Computational Discovery System

## Why DeepSeek?

DeepSeek models offer exceptional performance at a fraction of the cost of GPT-4/GPT-5:

| Model | Input Cost | Output Cost | vs GPT-4 |
|-------|-----------|-------------|----------|
| DeepSeek-Chat (V3) | $0.14/M tokens | $0.28/M tokens | **70x cheaper** |
| DeepSeek-Reasoner (R1) | $0.55/M tokens | $2.19/M tokens | **10x cheaper** |
| GPT-4 | $10/M tokens | $30/M tokens | baseline |

**For scientific discovery with lots of code execution and iterations, DeepSeek can save 90%+ on costs.**

---

## Setup Instructions

## Multi-Provider OpenAI-Compatible Routing (Recommended)

You can now use multiple OpenAI-compatible providers in one run.

Model syntax:
- `openai:model-name` -> default OpenAI credentials (`OPENAI_API_KEY`, optional `OPENAI_API_BASE`)
- `openai[alias]:model-name` -> aliased provider credentials

Alias environment variables:
- `OPENAI_COMPAT_<ALIAS>_API_KEY`
- `OPENAI_COMPAT_<ALIAS>_BASE_URL` (or `..._API_BASE`)

Example:
```bash
OPENAI_API_KEY=sk-proj-openai
OPENAI_COMPAT_DEEPSEEK_API_KEY=sk-deepseek
OPENAI_COMPAT_DEEPSEEK_BASE_URL=https://api.deepseek.com

SUPERVISOR_MODEL=openai:gpt-5.3
RESEARCH_MODEL=openai[deepseek]:deepseek-chat
FINAL_REPORT_MODEL=openai[deepseek]:deepseek-reasoner
```

### Step 1: Get DeepSeek API Key

1. Go to https://platform.deepseek.com/
2. Sign up/login
3. Navigate to API Keys: https://platform.deepseek.com/api_keys
4. Create a new API key
5. Copy the key (starts with `sk-`)

### Step 2: Configure Environment

Edit your `.env` file:

```bash
# Comment out or backup your OpenAI key
# OPENAI_API_KEY=sk-proj-...

# Add DeepSeek configuration
DEEPSEEK_API_KEY=sk-your-deepseek-key-here
OPENAI_API_BASE=https://api.deepseek.com
```

**IMPORTANT**: DeepSeek uses OpenAI-compatible API, so we set:
- API key in `DEEPSEEK_API_KEY` (but the system looks for `OPENAI_API_KEY`)
- Base URL in `OPENAI_API_BASE` to point to DeepSeek

### Step 3: Choose Models

In your `.env`, update model selections:

#### Option A: Use DeepSeek for Everything (Maximum Savings)

```bash
# Set the API key (DeepSeek key but use OPENAI_API_KEY variable)
OPENAI_API_KEY=sk-your-deepseek-key-here
OPENAI_API_BASE=https://api.deepseek.com

# Use DeepSeek Chat V3 for all tasks
SUMMARIZATION_MODEL=openai:deepseek-chat
RESEARCH_MODEL=openai:deepseek-chat
COMPRESSION_MODEL=openai:deepseek-chat
FINAL_REPORT_MODEL=openai:deepseek-chat
```

**Cost for typical biosignature run**: ~$0.50 (vs $35+ with GPT-4)

#### Option B: Use DeepSeek Reasoner for Final Report (Best Quality)

```bash
OPENAI_API_KEY=sk-your-deepseek-key-here
OPENAI_API_BASE=https://api.deepseek.com

# Use chat for iterations
SUMMARIZATION_MODEL=openai:deepseek-chat
RESEARCH_MODEL=openai:deepseek-chat
COMPRESSION_MODEL=openai:deepseek-chat

# Use R1 reasoner for complex final synthesis
FINAL_REPORT_MODEL=openai:deepseek-reasoner
```

**Cost for typical biosignature run**: ~$1.50 (vs $35+ with GPT-4)

#### Option C: Hybrid - DeepSeek for Iterations, GPT-5 for Final Report

```bash
# Native OpenAI (default openai:<model>)
OPENAI_API_KEY=sk-proj-your-openai-key

# DeepSeek as OpenAI-compatible alias
OPENAI_COMPAT_DEEPSEEK_API_KEY=sk-your-deepseek-key-here
OPENAI_COMPAT_DEEPSEEK_BASE_URL=https://api.deepseek.com

# Use DeepSeek for high-volume iterations
SUMMARIZATION_MODEL=openai[deepseek]:deepseek-chat
RESEARCH_MODEL=openai[deepseek]:deepseek-chat
COMPRESSION_MODEL=openai[deepseek]:deepseek-chat

# Use native OpenAI for final report
FINAL_REPORT_MODEL=openai:gpt-5.2
SUPERVISOR_MODEL=openai:gpt-5.3
```

This is fully supported: each OpenAI-compatible alias uses its own key and base URL.

---

## Model Details

### DeepSeek-Chat (V3)
- **Best for**: Iterations, code generation, analysis
- **Context**: 64K tokens
- **Strengths**: Fast, cheap, excellent code generation
- **Cost**: $0.14 input / $0.28 output per M tokens

### DeepSeek-Reasoner (R1)
- **Best for**: Complex reasoning, final synthesis
- **Context**: 64K tokens  
- **Strengths**: Chain-of-thought reasoning, scientific analysis
- **Cost**: $0.55 input / $2.19 output per M tokens
- **Note**: Outputs include visible reasoning chains

---

## Running Discovery with DeepSeek

Same commands work:

```bash
# Standard run
python run_discovery.py \
    --file examples/biosignature-discovery-prompt.txt \
    --verbose

# Quick test
python run_discovery.py \
    --quick \
    "Test DeepSeek: Calculate mean of random sample and plot histogram"
```

---

## Troubleshooting

### Issue: "Invalid API key"

**Solution**: Make sure you're using the right variable name:

```bash
# DeepSeek requires this exact setup:
OPENAI_API_KEY=sk-your-deepseek-key-here  # Use OPENAI_API_KEY, not DEEPSEEK_API_KEY
OPENAI_API_BASE=https://api.deepseek.com
```

The system looks for `OPENAI_API_KEY` even when using DeepSeek because we're using the OpenAI-compatible endpoint.

### Issue: "Model not found: deepseek-chat"

**Solution**: Check your model specification:

```bash
# Correct format (note the 'openai:' prefix):
RESEARCH_MODEL=openai:deepseek-chat

# Wrong (missing prefix):
RESEARCH_MODEL=deepseek-chat
```

### Issue: Want to switch back to OpenAI

**Solution**: Remove or comment out the base URL:

```bash
OPENAI_API_KEY=sk-proj-your-original-openai-key
# OPENAI_API_BASE=https://api.deepseek.com  # Comment this out

RESEARCH_MODEL=openai:gpt-4.1
FINAL_REPORT_MODEL=openai:gpt-5.2
```

---

## Performance Comparison

Based on biosignature discovery tests:

| Metric | GPT-4.1 | GPT-5.2 | DeepSeek-Chat | DeepSeek-R1 |
|--------|---------|---------|---------------|-------------|
| **Speed** | Fast | Slower | Very Fast | Medium |
| **Code Quality** | Excellent | Excellent | Excellent | Very Good |
| **Reasoning** | Very Good | Excellent | Good | Excellent |
| **Cost (typical run)** | $15 | $35 | $0.50 | $1.50 |
| **Best Use Case** | Balanced | Final reports | Iterations | Complex analysis |

**Recommendation for budget-conscious research**:
- Use `deepseek-chat` for everything during development/testing
- Switch to `deepseek-reasoner` or `gpt-5.2` for final production reports

---

## Example: Full DeepSeek Configuration

Complete `.env` setup for DeepSeek:

```bash
# ============================================
# DEEPSEEK CONFIGURATION (ACTIVE)
# ============================================
OPENAI_API_KEY=sk-your-deepseek-api-key-here
OPENAI_API_BASE=https://api.deepseek.com

# Search API (still need this)
TAVILY_API_KEY=tvly-your-tavily-key
SEARCH_API=tavily

# E2B for code execution (still need this)
E2B_API_KEY=your-e2b-key

# Model selection - all DeepSeek
SUMMARIZATION_MODEL=openai:deepseek-chat
RESEARCH_MODEL=openai:deepseek-chat
COMPRESSION_MODEL=openai:deepseek-chat
FINAL_REPORT_MODEL=openai:deepseek-reasoner  # Use R1 for final synthesis

# Research depth (same as before)
MAX_CONCURRENT_RESEARCH_UNITS=4
MAX_RESEARCHER_ITERATIONS=8
```

---

## Cost Savings Example

Biosignature discovery run (8 iterations, 3 experiments with code fixes):

**With GPT-4.1 + GPT-5.2**:
- Research iterations: ~500K tokens × $10/M = $5
- Code generation: ~200K tokens × $10/M = $2
- Final report: ~300K tokens × $30/M = $9
- **Total: ~$16**

**With DeepSeek-Chat + DeepSeek-R1**:
- Research iterations: ~500K tokens × $0.14/M = $0.07
- Code generation: ~200K tokens × $0.14/M = $0.03  
- Final report: ~300K tokens × $2.19/M = $0.66
- **Total: ~$0.76 (95% cheaper!)**

**For 100 research runs**: $1,600 → $76 💰

---

## Additional Notes

- DeepSeek models are very good at code generation (trained heavily on code)
- R1 (Reasoner) provides visible chain-of-thought, which can be useful for debugging
- DeepSeek-Chat V3 is comparable to GPT-4 on many benchmarks
- Rate limits on DeepSeek are generous (usually not an issue)
- DeepSeek's base URL is stable and reliable

---

## Next Steps

1. Get your DeepSeek API key
2. Update `.env` with the configuration above
3. Run a quick test:
   ```bash
   python run_discovery.py --quick "Calculate thermodynamics of H2S oxidation"
   ```
4. Monitor the output - should work identically to OpenAI
5. Check your DeepSeek dashboard to see usage and costs

**Questions?** The system treats DeepSeek exactly like OpenAI - same commands, same interface, just way cheaper!
