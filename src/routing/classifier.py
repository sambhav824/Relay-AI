import re
from typing import List, Optional, Tuple
from src.core.message import Message
from src.routing.types import ClassificationResult, TaskType


class TaskClassifier:
    """Deterministic, zero-latency rule-based classifier for user prompts and conversation context."""

    DEFAULT_CONFIDENCE_THRESHOLD = 0.60

    # Coding patterns & keywords
    _CODE_BLOCK_REGEX = re.compile(r"```[\w+-]*\s*[\r\n]+(.*?)```", re.DOTALL)
    _INLINE_CODE_REGEX = re.compile(r"`[^`\n]{2,}`")
    _CODE_KEYWORDS = [
        "def ", "class ", "import ", "from ", "return ", "async ", "await ",
        "function", "const ", "let ", "var ", "public static void", "console.log",
        "print(", "struct ", "impl ", "fn ", "namespace ", "select ", "from ",
        "where ", "join ", "insert into ", "update ", "delete ", "git ", "npm ",
        "pip ", "dockerfile", "makefile", "traceback", "exception:", "syntaxerror",
        "typeerror", "nullpointer", "stack trace", "refactor", "unit test", "bug",
        "fix error", "compile", "breakpoint", "api endpoint", "payload",
        "python", "javascript", "typescript", "golang", "rust", "script",
        "write code", "coding", "algorithm", "regex", "sql query", "html", "css"
    ]
    _CODE_SYMBOLS = [r"\{[\s\S]*\}", r"=>", r"===?", r"!==?", r"\+\+", r"--", r"\/\/", r"#include"]

    # Mathematical reasoning patterns
    _MATH_KEYWORDS = [
        "calculate", "derivative", "integral", "differentiate", "equation",
        "solve for x", "matrix", "determinant", "eigenvalue", "vector space",
        "probability", "combinatorics", "permutation", "theorem", "proof",
        "geometric", "arithmetic", "logarithm", "prime number", "polynomial",
        "quadratic", "pythagorean", "standard deviation", "variance", "binomial"
    ]
    _MATH_SYMBOLS = [
        r"\\(sum|int|prod|sqrt|frac|partial|approx|neq|times|pm|le|ge)",
        r"\b\d+\s*[\+\-\*\/\^]\s*\d+\b",
        r"\bf\(x\)\s*=",
        r"x\^[0-9]+",
        r"\b[a-z]\s*=\s*[0-9]+",
    ]

    # Summarization patterns
    _SUMMARIZE_KEYWORDS = [
        "summarize", "summary", "tl;dr", "tldr", "brief overview", "condense",
        "bullet points of", "key takeaways", "digest", "recap", "in a nutshell",
        "short version", "give me the gist", "executive summary", "synopsis"
    ]

    # Creative writing patterns
    _CREATIVE_KEYWORDS = [
        "write a story", "write a poem", "poem", "poetry", "rhyme", "haiku",
        "creative writing", "fictional", "screenplay", "script for", "character dialogue",
        "song lyrics", "novel chapter", "metaphor", "fairytale", "creative prompt",
        "narrative arc", "roleplay as", "imagine you are"
    ]

    # Complex reasoning patterns
    _REASONING_KEYWORDS = [
        "step-by-step", "step by step", "chain of thought", "compare and contrast",
        "trade-offs", "tradeoffs", "root cause analysis", "pros and cons",
        "first principles", "philosophical implications", "deductive reasoning",
        "evaluate arguments", "deep dive analysis", "critical assessment",
        "counter-argument", "counterargument", "systematic review"
    ]

    # Document analysis indicators
    _DOC_KEYWORDS = [
        "attached document", "analyze the following text", "parse this log",
        "extract information from", "csv data", "json structure", "transcript analysis",
        "review this report", "document content", "table below"
    ]

    def __init__(self, confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD):
        self.confidence_threshold = confidence_threshold

    def classify(
        self,
        prompt: str,
        history: Optional[List[Message]] = None,
    ) -> ClassificationResult:
        """Classify user request into a TaskType with deterministic confidence scoring."""
        text = (prompt or "").strip()
        matched_cues = []
        scores: dict[TaskType, float] = {
            TaskType.CODING: 0.0,
            TaskType.MATH: 0.0,
            TaskType.SUMMARIZATION: 0.0,
            TaskType.DOCUMENT_ANALYSIS: 0.0,
            TaskType.CREATIVE: 0.0,
            TaskType.COMPLEX_REASONING: 0.0,
            TaskType.GENERAL: 0.1,  # baseline floor for general
        }

        lower_text = text.lower()

        # 1. Coding Analysis
        code_blocks = self._CODE_BLOCK_REGEX.findall(text)
        if code_blocks:
            scores[TaskType.CODING] += 3.0 * len(code_blocks)
            matched_cues.append(f"fenced_code_blocks({len(code_blocks)})")

        inline_codes = self._INLINE_CODE_REGEX.findall(text)
        if inline_codes:
            scores[TaskType.CODING] += 1.0 * min(3, len(inline_codes))
            matched_cues.append(f"inline_code_spans({len(inline_codes)})")

        for kw in self._CODE_KEYWORDS:
            if kw in lower_text:
                scores[TaskType.CODING] += 1.5
                matched_cues.append(f"code_kw:{kw.strip()}")

        for sym in self._CODE_SYMBOLS:
            if re.search(sym, text):
                scores[TaskType.CODING] += 1.2
                matched_cues.append("code_syntax_pattern")

        # 2. Mathematical Reasoning Analysis
        for kw in self._MATH_KEYWORDS:
            if kw in lower_text:
                scores[TaskType.MATH] += 1.8
                matched_cues.append(f"math_kw:{kw}")

        for sym in self._MATH_SYMBOLS:
            if re.search(sym, text):
                scores[TaskType.MATH] += 1.5
                matched_cues.append("math_symbol_pattern")

        # 3. Summarization Analysis
        for kw in self._SUMMARIZE_KEYWORDS:
            if kw in lower_text:
                scores[TaskType.SUMMARIZATION] += 2.5
                matched_cues.append(f"summarize_kw:{kw}")

        # 4. Creative Writing Analysis
        for kw in self._CREATIVE_KEYWORDS:
            if kw in lower_text:
                scores[TaskType.CREATIVE] += 2.5
                matched_cues.append(f"creative_kw:{kw}")

        # 5. Complex Reasoning Analysis
        for kw in self._REASONING_KEYWORDS:
            if kw in lower_text:
                scores[TaskType.COMPLEX_REASONING] += 2.0
                matched_cues.append(f"reasoning_kw:{kw}")

        # 6. Document Analysis
        word_count = len(text.split())
        if word_count >= 600:
            scores[TaskType.DOCUMENT_ANALYSIS] += min(5.0, word_count / 300)
            matched_cues.append(f"long_input_words({word_count})")

        for kw in self._DOC_KEYWORDS:
            if kw in lower_text:
                scores[TaskType.DOCUMENT_ANALYSIS] += 2.0
                matched_cues.append(f"doc_kw:{kw}")

        # 7. Context History Inspection (if prompt is brief follow-up)
        if history and word_count < 25:
            last_assistant_or_user = history[-1].content if history else ""
            if "```" in last_assistant_or_user or any(k in last_assistant_or_user.lower() for k in ["def ", "class ", "function"]):
                if any(k in lower_text for k in ["fix", "error", "debug", "run", "refactor", "why", "optimize"]):
                    scores[TaskType.CODING] += 3.0
                    matched_cues.append("context_history_coding_continuation")
            elif any(k in lower_text for k in ["summarize", "tldr", "shorter", "digest"]):
                scores[TaskType.SUMMARIZATION] += 3.0
                matched_cues.append("context_history_summarize_request")

        # Confidence Normalization
        total_score = sum(scores.values())
        sorted_tasks = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top_task, top_score = sorted_tasks[0]

        # Normalized confidence: proportion of top score relative to total evidence
        if total_score > 0.1:
            raw_confidence = top_score / total_score
            # Bound confidence between 0.20 and 0.99
            confidence = min(0.99, max(0.20, raw_confidence))
        else:
            confidence = 0.50

        # Safe fallback guard: if confidence is below threshold, default to GENERAL
        if confidence < self.confidence_threshold and top_task != TaskType.GENERAL:
            return ClassificationResult(
                task_type=TaskType.GENERAL,
                confidence=round(confidence, 2),
                is_fallback_general=True,
                matched_cues=matched_cues,
                raw_scores={k: round(v, 2) for k, v in scores.items()},
            )

        return ClassificationResult(
            task_type=top_task,
            confidence=round(confidence, 2),
            is_fallback_general=(top_task == TaskType.GENERAL and total_score <= 0.1),
            matched_cues=matched_cues,
            raw_scores={k: round(v, 2) for k, v in scores.items()},
        )

