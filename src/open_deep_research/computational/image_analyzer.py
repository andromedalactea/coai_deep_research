"""Vision-based Image Analysis Module for Scientific Discovery.

This module provides AI-powered analysis of generated visualizations,
enabling the system to actually "see" and understand what plots show,
not just guess from code expectations.

Supports: Google Gemini, OpenAI GPT-4V, Anthropic Claude 3 Vision
"""

import asyncio
import base64
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class VisionProvider(Enum):
    """Supported vision model providers."""
    GOOGLE = "google"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


@dataclass
class ImageAnalysisResult:
    """Structured result from vision analysis."""
    # Basic description
    plot_type: str  # scatter, line, contour, heatmap, 3D surface, bar, histogram, etc.
    title_detected: Optional[str]
    axes_labels: Dict[str, str]  # {"x": "Temperature (K)", "y": "ΔG (kJ/mol)"}
    
    # Visual patterns
    main_pattern: str  # "positive correlation", "threshold behavior", "exponential decay", etc.
    key_features: List[str]  # List of notable visual features
    anomalies: List[str]  # Unexpected patterns or potential issues
    
    # Quantitative observations (estimated from visual)
    approximate_values: Dict[str, Any]  # {"max_y": "~150", "trend": "decreasing"}
    data_density: str  # "sparse", "moderate", "dense"
    
    # Quality assessment
    quality_score: float  # 0-1: how scientifically useful is this visualization
    quality_issues: List[str]  # ["missing axis labels", "unclear legend", etc.]
    
    # Scientific interpretation
    supports_hypothesis: Optional[bool]  # Based on visual evidence
    visual_evidence: str  # Explanation of what visual elements support/refute
    confidence: str  # "low", "medium", "high"
    
    # Raw description
    full_description: str


class VisionAnalyzer:
    """Analyze scientific visualizations using vision-capable AI models."""
    
    def __init__(
        self,
        provider: str = None,
        model: str = None,
        api_key: str = None
    ):
        """Initialize the vision analyzer.
        
        Args:
            provider: Vision provider (google, openai, anthropic). Auto-detected from env.
            model: Specific model to use. Defaults based on provider.
            api_key: API key. Defaults to environment variable.
        """
        # Auto-detect provider and model from environment
        self.provider = self._detect_provider(provider)
        self.model = model or self._get_default_model()
        self.api_key = api_key or self._get_api_key()
        
        logger.info(f"Vision analyzer initialized: provider={self.provider}, model={self.model}")
    
    def _detect_provider(self, provider: str = None) -> VisionProvider:
        """Detect vision provider from environment or explicit setting."""
        if provider:
            return VisionProvider(provider.lower())
        
        # Check environment for VISION_MODEL setting
        vision_model = os.getenv("VISION_MODEL", "")
        if vision_model:
            if vision_model.startswith("google:") or "gemini" in vision_model.lower():
                return VisionProvider.GOOGLE
            elif vision_model.startswith("openai:") or "gpt-4" in vision_model.lower():
                return VisionProvider.OPENAI
            elif vision_model.startswith("anthropic:") or "claude" in vision_model.lower():
                return VisionProvider.ANTHROPIC
        
        # Check which API keys are available
        if os.getenv("GOOGLE_API_KEY"):
            return VisionProvider.GOOGLE
        elif os.getenv("OPENAI_API_KEY"):
            return VisionProvider.OPENAI
        elif os.getenv("ANTHROPIC_API_KEY"):
            return VisionProvider.ANTHROPIC
        
        # Default to Google (best for scientific analysis)
        return VisionProvider.GOOGLE
    
    def _get_default_model(self) -> str:
        """Get default model for the provider."""
        vision_model = os.getenv("VISION_MODEL", "")
        if vision_model and ":" in vision_model:
            return vision_model.split(":", 1)[1]
        
        defaults = {
            VisionProvider.GOOGLE: "gemini-3-flash-preview",
            VisionProvider.OPENAI: "gpt-4o",
            VisionProvider.ANTHROPIC: "claude-3-5-sonnet-20241022"
        }
        return defaults.get(self.provider, "gemini-3-flash-preview")
    
    def _get_api_key(self) -> str:
        """Get API key for the provider."""
        keys = {
            VisionProvider.GOOGLE: os.getenv("GOOGLE_API_KEY"),
            VisionProvider.OPENAI: os.getenv("OPENAI_API_KEY"),
            VisionProvider.ANTHROPIC: os.getenv("ANTHROPIC_API_KEY")
        }
        return keys.get(self.provider, "")
    
    async def analyze_image(
        self,
        image_base64: str,
        image_format: str = "png",
        context: str = "",
        hypothesis: str = "",
        expected_content: str = "",
        code_that_generated: str = ""
    ) -> ImageAnalysisResult:
        """Analyze a scientific visualization with full context.
        
        Args:
            image_base64: Base64 encoded image data
            image_format: Image format (png, jpg, svg)
            context: General context about the experiment
            hypothesis: The hypothesis being tested
            expected_content: What the plot should show based on code
            code_that_generated: The code that created this visualization
            
        Returns:
            ImageAnalysisResult with detailed analysis
        """
        prompt = self._build_analysis_prompt(
            context=context,
            hypothesis=hypothesis,
            expected_content=expected_content,
            code_that_generated=code_that_generated
        )
        
        try:
            if self.provider == VisionProvider.GOOGLE:
                response = await self._analyze_with_google(image_base64, image_format, prompt)
            elif self.provider == VisionProvider.OPENAI:
                response = await self._analyze_with_openai(image_base64, image_format, prompt)
            elif self.provider == VisionProvider.ANTHROPIC:
                response = await self._analyze_with_anthropic(image_base64, image_format, prompt)
            else:
                raise ValueError(f"Unknown provider: {self.provider}")
            
            return self._parse_analysis_response(response)
            
        except Exception as e:
            logger.error(f"Vision analysis failed: {e}")
            # Return a basic result indicating failure
            return ImageAnalysisResult(
                plot_type="unknown",
                title_detected=None,
                axes_labels={},
                main_pattern="Analysis failed",
                key_features=[],
                anomalies=[f"Vision analysis error: {str(e)}"],
                approximate_values={},
                data_density="unknown",
                quality_score=0.0,
                quality_issues=["Could not analyze image"],
                supports_hypothesis=None,
                visual_evidence="Vision analysis failed - unable to determine",
                confidence="none",
                full_description=f"Analysis failed: {str(e)}"
            )
    
    def _build_analysis_prompt(
        self,
        context: str,
        hypothesis: str,
        expected_content: str,
        code_that_generated: str
    ) -> str:
        """Build a comprehensive prompt for image analysis."""
        prompt = """You are an expert scientific data analyst examining a visualization from a computational experiment.

## CONTEXT
"""
        if hypothesis:
            prompt += f"""
**Hypothesis Being Tested:**
{hypothesis}
"""
        
        if context:
            prompt += f"""
**Experiment Context:**
{context}
"""
        
        if expected_content:
            prompt += f"""
**What This Plot Should Show:**
{expected_content}
"""
        
        if code_that_generated:
            # Include relevant parts of the code
            code_preview = code_that_generated[:1500] if len(code_that_generated) > 1500 else code_that_generated
            prompt += f"""
**Code That Generated This:**
```python
{code_preview}
```
"""
        
        prompt += """
## YOUR TASK

Analyze this scientific visualization in detail. Provide a thorough, QUANTITATIVE assessment.

**IMPORTANT**: Be OBJECTIVE. Report exactly what you SEE, not what you expect or hope to see.
If the plot contradicts expectations, say so clearly. If the plot is unclear or problematic, say so.

## REQUIRED ANALYSIS

### 1. PLOT IDENTIFICATION
- What type of plot is this? (scatter, line, contour, heatmap, 3D surface, bar, histogram, box plot, etc.)
- What is the title (if visible)?
- What are the axis labels and their ranges?

### 2. VISUAL PATTERNS
- What is the main pattern or trend? Be specific (e.g., "linear increase from 200K to 400K", "exponential decay with τ ≈ 50 units")
- Are there multiple data series? Describe each.
- Any notable features? (peaks, valleys, asymptotes, inflection points)
- Any anomalies or unexpected features?

### 3. QUANTITATIVE OBSERVATIONS
Estimate key values from the visualization:
- Approximate min/max values on each axis
- Where does the trend cross zero or other significant thresholds?
- Slope or rate of change (if applicable)
- Spread or variance of data points

### 4. QUALITY ASSESSMENT
Rate from 0-100%:
- Are labels clear and readable?
- Is the scale appropriate?
- Are colors/markers distinguishable?
- Is the data well-presented for scientific communication?
List any quality issues.

### 5. HYPOTHESIS EVALUATION
Based ONLY on what you see in this visualization:
- Does this visual evidence support or contradict the hypothesis?
- What specific visual elements lead to this conclusion?
- Confidence level: low/medium/high

### 6. FULL DESCRIPTION
Provide a 2-3 paragraph scientific description of what this figure shows, as if writing for a research paper.

## RESPONSE FORMAT

Respond in this EXACT format:

PLOT_TYPE: [type]
TITLE: [title or "none visible"]
X_AXIS: [label] | RANGE: [min to max]
Y_AXIS: [label] | RANGE: [min to max]
Z_AXIS: [label if 3D] | RANGE: [min to max if applicable]

MAIN_PATTERN: [description]
KEY_FEATURES:
- [feature 1]
- [feature 2]
- [feature 3]

ANOMALIES:
- [anomaly 1 or "none"]

APPROXIMATE_VALUES:
- [key value 1]
- [key value 2]

DATA_DENSITY: [sparse/moderate/dense]

QUALITY_SCORE: [0-100]
QUALITY_ISSUES:
- [issue 1 or "none"]

SUPPORTS_HYPOTHESIS: [YES/NO/UNCLEAR]
VISUAL_EVIDENCE: [explanation]
CONFIDENCE: [LOW/MEDIUM/HIGH]

FULL_DESCRIPTION:
[Your detailed scientific description here]
"""
        return prompt
    
    async def _analyze_with_google(
        self,
        image_base64: str,
        image_format: str,
        prompt: str
    ) -> str:
        """Analyze image using Google Gemini."""
        try:
            from google import genai
            from google.genai import types
            
            # Create client with API key
            client = genai.Client(api_key=self.api_key)
            
            # Decode base64 to bytes
            image_bytes = base64.b64decode(image_base64)
            
            # Create image part using the correct API format
            image_part = types.Part.from_bytes(
                data=image_bytes,
                mime_type=f"image/{image_format}"
            )
            
            # Generate content using the new API
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=self.model,
                contents=[prompt, image_part]
            )
            
            return response.text
            
        except ImportError:
            logger.warning("google-genai not installed, trying langchain")
            return await self._analyze_with_langchain_google(image_base64, image_format, prompt)
        except Exception as e:
            logger.warning(f"google-genai direct API failed: {e}, trying langchain fallback")
            return await self._analyze_with_langchain_google(image_base64, image_format, prompt)
    
    async def _analyze_with_langchain_google(
        self,
        image_base64: str,
        image_format: str,
        prompt: str
    ) -> str:
        """Fallback: Use LangChain's Google integration."""
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage
        
        model = ChatGoogleGenerativeAI(
            model=self.model,
            google_api_key=self.api_key
        )
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": f"data:image/{image_format};base64,{image_base64}"
                }
            ]
        )
        
        response = await model.ainvoke([message])
        return response.content
    
    async def _analyze_with_openai(
        self,
        image_base64: str,
        image_format: str,
        prompt: str
    ) -> str:
        """Analyze image using OpenAI GPT-4V."""
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage
        
        model = ChatOpenAI(
            model=self.model,
            api_key=self.api_key,
            max_tokens=4096
        )
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{image_format};base64,{image_base64}"
                    }
                }
            ]
        )
        
        response = await model.ainvoke([message])
        return response.content
    
    async def _analyze_with_anthropic(
        self,
        image_base64: str,
        image_format: str,
        prompt: str
    ) -> str:
        """Analyze image using Anthropic Claude 3."""
        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import HumanMessage
        
        model = ChatAnthropic(
            model=self.model,
            api_key=self.api_key,
            max_tokens=4096
        )
        
        # Claude uses a specific format for images
        message = HumanMessage(
            content=[
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": f"image/{image_format}",
                        "data": image_base64
                    }
                },
                {"type": "text", "text": prompt}
            ]
        )
        
        response = await model.ainvoke([message])
        return response.content
    
    def _parse_analysis_response(self, response: str) -> ImageAnalysisResult:
        """Parse the structured response into ImageAnalysisResult."""
        
        def extract_value(key: str, default: str = "") -> str:
            """Extract a single-line value after a key."""
            if f"{key}:" in response:
                start = response.find(f"{key}:") + len(f"{key}:")
                end = response.find("\n", start)
                if end > 0:
                    return response[start:end].strip()
            return default
        
        def extract_list(key: str) -> List[str]:
            """Extract a list of items after a key."""
            items = []
            if f"{key}:" in response:
                start = response.find(f"{key}:") + len(f"{key}:")
                # Find next section (capitalized word followed by colon)
                import re
                remaining = response[start:]
                # Find where list ends (next section header or end)
                end_match = re.search(r'\n[A-Z_]+:', remaining)
                if end_match:
                    list_text = remaining[:end_match.start()]
                else:
                    list_text = remaining[:500]  # Limit
                
                for line in list_text.split("\n"):
                    line = line.strip()
                    if line.startswith("-"):
                        item = line[1:].strip()
                        if item and item.lower() != "none":
                            items.append(item)
            return items
        
        # Parse plot type
        plot_type = extract_value("PLOT_TYPE", "unknown")
        
        # Parse title
        title = extract_value("TITLE")
        if title.lower() in ["none visible", "none", "n/a"]:
            title = None
        
        # Parse axes
        axes_labels = {}
        x_axis = extract_value("X_AXIS")
        if x_axis and "|" in x_axis:
            axes_labels["x"] = x_axis.split("|")[0].strip()
        y_axis = extract_value("Y_AXIS")
        if y_axis and "|" in y_axis:
            axes_labels["y"] = y_axis.split("|")[0].strip()
        z_axis = extract_value("Z_AXIS")
        if z_axis and "|" in z_axis and z_axis.lower() not in ["n/a", "none"]:
            axes_labels["z"] = z_axis.split("|")[0].strip()
        
        # Parse main pattern
        main_pattern = extract_value("MAIN_PATTERN", "Not determined")
        
        # Parse lists
        key_features = extract_list("KEY_FEATURES")
        anomalies = extract_list("ANOMALIES")
        quality_issues = extract_list("QUALITY_ISSUES")
        
        # Parse approximate values
        approx_values_list = extract_list("APPROXIMATE_VALUES")
        approximate_values = {}
        for item in approx_values_list:
            if ":" in item or "=" in item:
                sep = ":" if ":" in item else "="
                parts = item.split(sep, 1)
                if len(parts) == 2:
                    approximate_values[parts[0].strip()] = parts[1].strip()
        
        # Parse data density
        data_density = extract_value("DATA_DENSITY", "moderate").lower()
        
        # Parse quality score
        quality_str = extract_value("QUALITY_SCORE", "50")
        try:
            quality_score = float(quality_str.replace("%", "")) / 100.0
        except:
            quality_score = 0.5
        
        # Parse hypothesis support
        supports_str = extract_value("SUPPORTS_HYPOTHESIS", "UNCLEAR").upper()
        if "YES" in supports_str:
            supports_hypothesis = True
        elif "NO" in supports_str:
            supports_hypothesis = False
        else:
            supports_hypothesis = None
        
        # Parse visual evidence
        visual_evidence = extract_value("VISUAL_EVIDENCE", "Unable to determine")
        
        # Parse confidence
        confidence = extract_value("CONFIDENCE", "medium").lower()
        
        # Parse full description
        full_description = ""
        if "FULL_DESCRIPTION:" in response:
            start = response.find("FULL_DESCRIPTION:") + len("FULL_DESCRIPTION:")
            full_description = response[start:].strip()
        
        return ImageAnalysisResult(
            plot_type=plot_type,
            title_detected=title,
            axes_labels=axes_labels,
            main_pattern=main_pattern,
            key_features=key_features,
            anomalies=anomalies,
            approximate_values=approximate_values,
            data_density=data_density,
            quality_score=quality_score,
            quality_issues=quality_issues,
            supports_hypothesis=supports_hypothesis,
            visual_evidence=visual_evidence,
            confidence=confidence,
            full_description=full_description
        )


# Global analyzer instance (lazy initialization)
_analyzer: Optional[VisionAnalyzer] = None


def get_vision_analyzer() -> VisionAnalyzer:
    """Get or create the global vision analyzer."""
    global _analyzer
    if _analyzer is None:
        _analyzer = VisionAnalyzer()
    return _analyzer


async def analyze_scientific_figure(
    image_base64: str,
    image_format: str = "png",
    hypothesis: str = "",
    experiment_context: str = "",
    expected_content: str = "",
    source_code: str = ""
) -> ImageAnalysisResult:
    """Convenience function to analyze a scientific figure.
    
    Args:
        image_base64: Base64 encoded image
        image_format: Image format (png, jpg)
        hypothesis: The hypothesis being tested
        experiment_context: Context about the experiment
        expected_content: What the visualization should show
        source_code: Code that generated the visualization
        
    Returns:
        ImageAnalysisResult with detailed analysis
    """
    analyzer = get_vision_analyzer()
    return await analyzer.analyze_image(
        image_base64=image_base64,
        image_format=image_format,
        context=experiment_context,
        hypothesis=hypothesis,
        expected_content=expected_content,
        code_that_generated=source_code
    )


def analysis_to_text(result: ImageAnalysisResult) -> str:
    """Convert ImageAnalysisResult to human-readable text for AI consumption."""
    parts = [
        "=== VISUAL ANALYSIS OF FIGURE ===",
        f"Plot Type: {result.plot_type}",
    ]
    
    if result.title_detected:
        parts.append(f"Title: {result.title_detected}")
    
    if result.axes_labels:
        axes_str = ", ".join([f"{k}: {v}" for k, v in result.axes_labels.items()])
        parts.append(f"Axes: {axes_str}")
    
    parts.append(f"\nMain Pattern: {result.main_pattern}")
    
    if result.key_features:
        parts.append("\nKey Features:")
        for f in result.key_features:
            parts.append(f"  • {f}")
    
    if result.anomalies:
        parts.append("\nAnomalies/Issues:")
        for a in result.anomalies:
            parts.append(f"  ⚠️ {a}")
    
    if result.approximate_values:
        parts.append("\nApproximate Values:")
        for k, v in result.approximate_values.items():
            parts.append(f"  • {k}: {v}")
    
    parts.append(f"\nQuality Score: {result.quality_score * 100:.0f}%")
    
    if result.quality_issues:
        parts.append("Quality Issues:")
        for i in result.quality_issues:
            parts.append(f"  • {i}")
    
    parts.append(f"\n--- HYPOTHESIS EVALUATION ---")
    support_str = "SUPPORTS" if result.supports_hypothesis else ("CONTRADICTS" if result.supports_hypothesis == False else "UNCLEAR")
    parts.append(f"Visual Evidence: {support_str} hypothesis")
    parts.append(f"Evidence: {result.visual_evidence}")
    parts.append(f"Confidence: {result.confidence}")
    
    parts.append(f"\n--- DETAILED DESCRIPTION ---")
    parts.append(result.full_description)
    
    return "\n".join(parts)
