import os, re, json, ast, traceback
from typing import List, Dict, Any, Optional, TypedDict, Tuple
from dataclasses import dataclass
from jsonschema import validate as jsonschema_validate, Draft202012Validator
from pydantic import BaseModel, Field
from functools import reduce

from pyspark.sql import DataFrame, functions as F
from pyspark.sql.types import *
from pyspark.sql.utils import AnalysisException
from pyspark.sql import SparkSession

# LangChain / LangGraph
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage
from langchain_community.chat_models import ChatDatabricks

spark = SparkSession.builder.appName("pyspark_agent").getOrCreate()

def strip_code_fences(code: str) -> str:
    """Return the inner code by removing triple-backtick fences.

    Supports both ```...``` and ```python ...``` styles and leaves input unchanged
    if no fences exist. This is intentionally minimal—just enough to clean LLM
    output before further processing.

    Args:
        code: Raw text that may contain fenced code.

    Returns:
        The unfenced code string. If `code` is falsy, returns it as-is.

    Examples:
        >>> strip_code_fences("```python\\nprint(1)\\n```")
        'print(1)'
    """
    if not code:
        return code
    m = re.search(r"```(?:python)?\s*(.*?)\s*```", code, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1) if m else code


def sanitize_pyspark_code(code: str) -> str:
    """Normalize/clean a PySpark snippet for safe parsing and execution.

    What this does:
    - Strips code fences and Markdown artifacts (bold, backticks).
    - Removes *all* import lines (we provide `df` and `F`).
    - Trims leading/trailing blank lines.
    - Normalizes indentation by removing the minimum common indent.

    This function is intentionally conservative; it does not rewrite logic or
    attempt to "fix" broken code—just gets it into a predictable, safe shape
    for AST checks and `exec`.

    Args:
        code: Arbitrary code block (often from an LLM).

    Returns:
        A cleaned, trimmed snippet suitable for static checks and execution.

    Examples:
        >>> sanitize_pyspark_code("```\\nfrom x import y\\n df_out = df\\n```")
        'df_out = df'
    """
    code = strip_code_fences(code or "")
    safe_lines: List[str] = []

    for line in code.splitlines():
        # Drop any import-style statements (including `from ... import ...`)
        if re.match(r"^\s*(import|from)\s+", line):
            continue

        # Remove Markdown bold **text** and inline code `text`
        line = re.sub(r'\*\*(.*?)\*\*', r'\1', line)
        line = re.sub(r'`([^`]+)`', r'\1', line)

        # Skip initial empty lines for a tidy block
        if not safe_lines and not line.strip():
            continue

        safe_lines.append(line)

    if not safe_lines:
        return ""

    # Remove trailing empties
    while safe_lines and not safe_lines[-1].strip():
        safe_lines.pop()

    # Normalize indentation by removing minimum common leading whitespace
    code_text = "\n".join(safe_lines)
    lines_with_content = [l for l in safe_lines if l.strip()]
    if lines_with_content:
        min_indent = min(len(l) - len(l.lstrip()) for l in lines_with_content)
        normalized_lines: List[str] = []
        for l in safe_lines:
            normalized_lines.append(l[min_indent:] if l.strip() and len(l) > min_indent else (l if l.strip() else ""))
        code_text = "\n".join(normalized_lines)

    return code_text.strip()


def _normalize_dbfs_path(p: str) -> str:
    """Normalize DBFS-style paths so Spark readers/writers accept them.

    Converts `/dbfs/...` to `dbfs:/...`. S3/ABFSS/etc. URIs pass through unchanged.

    Args:
        p: A local/DBFS/cloud path string.

    Returns:
        A normalized path string suitable for Spark IO.
    """
    if not p:
        return p
    p = p.strip()
    if p.startswith("/dbfs/"):
        return "dbfs:" + p[5:]
    return p


def _infer_format(path: str, user_fmt: str) -> str:
    """Infer input format from file suffix, honoring a user override.

    Args:
        path: Path that may carry a meaningful extension.
        user_fmt: Optional explicit format (e.g., "parquet", "delta").
                  If not "auto", it takes precedence.

    Returns:
        One of {"parquet","delta","json","csv"} with "parquet" as a safe default.
    """
    if user_fmt and user_fmt.lower() != "auto":
        return user_fmt.lower()
    p = (path or "").lower()
    if p.endswith(".parquet"):
        return "parquet"
    if p.endswith(".delta"):
        return "delta"
    if p.endswith(".json"):
        return "json"
    if p.endswith(".csv"):
        return "csv"
    return "parquet"  # default columnar choice


def read_df_from_path(path: str, fmt: str) -> DataFrame:
    """Read a dataset into a Spark DataFrame with light format handling.

    Notes:
        - Requires a live `spark` session in the current runtime.
        - CSV reads with header inference; other formats use Spark defaults.

    Args:
        path: Input path (DBFS/S3/ABFSS/etc.).
        fmt: Requested format or "auto" (will be inferred via `_infer_format`).

    Returns:
        A Spark DataFrame loaded from the provided path.
    """
    path = _normalize_dbfs_path(path)
    fmt = _infer_format(path, fmt)
    reader = spark.read  # type: ignore[name-defined]

    if fmt == "csv":
        return reader.option("header", True).option("inferSchema", True).csv(path)
    if fmt == "json":
        return reader.json(path)
    if fmt == "delta":
        return reader.format("delta").load(path)
    if fmt == "parquet":
        return reader.parquet(path)

    # Last resort: try parquet
    return reader.parquet(path)


def load_inputs(paths_csv: str, fmt: str) -> DataFrame:
    """Load one or more paths and union-by-name across a superset schema.

    Pragmatic behavior for real-world data drift:
    - Reads each path, collects the union of all columns.
    - Adds missing columns (nullable string) so unionByName succeeds.
    - Returns a single DataFrame of all rows.

    Args:
        paths_csv: Comma-separated list of input paths.
        fmt: Input format or "auto" for suffix-based inference.

    Returns:
        A Spark DataFrame combining all inputs.

    Raises:
        ValueError: If no usable paths are provided.
    """
    paths = [p.strip() for p in (paths_csv or "").split(",") if p.strip()]
    if not paths:
        raise ValueError("No input paths provided.")

    dfs = [read_df_from_path(p, fmt) for p in paths]
    if len(dfs) == 1:
        return dfs[0]

    # Build superset schema across ALL inputs, then unionByName with allowMissingColumns
    all_cols = sorted(list(set().union(*[set(df.columns) for df in dfs])))
    aligned: List[DataFrame] = []
    for df in dfs:
        missing = [c for c in all_cols if c not in df.columns]
        for c in missing:
            df = df.withColumn(c, F.lit(None).cast(StringType()))
        aligned.append(df.select(all_cols))

    return reduce(lambda a, b: a.unionByName(b, allowMissingColumns=True), aligned)


def schema_preview(df: DataFrame, max_cols: int = 40) -> str:
    """Return a human-friendly preview of `df.dtypes`, clipped for prompt context.

    Args:
        df: Input Spark DataFrame.
        max_cols: Maximum number of columns to include before clipping.

    Returns:
        A multi-line string of `col: dtype` lines (with an ellipsis sentinel if clipped).
    """
    dtypes = df.dtypes
    if len(dtypes) > max_cols:
        dtypes = dtypes[:max_cols] + [("…", "…")]
    return "\n".join([f"{c}: {t}" for c, t in dtypes])


def exec_pyspark_code(pyspark_code: str, df: DataFrame) -> DataFrame:
    """Execute a sanitized PySpark snippet with a constrained namespace.

    Contract:
      - Provides `df` (input DataFrame) and `F` (pyspark.sql.functions).
      - Snippet must assign the final result to `df_out`.
      - Raises if `df_out` is missing.

    Security posture:
      Assumes code was pre-validated (e.g., via AST checks) to forbid IO and
      dangerous calls. This function focuses on controlled execution, not policing.

    Args:
        pyspark_code: A cleaned, ready-to-exec PySpark snippet.
        df: The input DataFrame to operate on.

    Returns:
        The `df_out` DataFrame produced by the snippet.

    Raises:
        RuntimeError: If the snippet fails to set `df_out`.
    """
    local_env: Dict[str, Any] = {"df": df, "F": F}
    global_env: Dict[str, Any] = {}
    exec(pyspark_code, global_env, local_env)  # sandboxed dicts
    if "df_out" not in local_env:
        raise RuntimeError("Code did not produce `df_out`.")
    return local_env["df_out"]  # type: ignore[no-any-return]


def parse_llm_json(s: str) -> Dict[str, Any]:
    """Extract a JSON object from messy LLM text (fences and leading/trailing noise).

    This is a pragmatic parser: look for ```json fenced blocks first; if not present,
    find the first/last brace pair and parse that slice.

    Args:
        s: Raw model output possibly containing JSON.

    Returns:
        The parsed JSON object.

    Raises:
        json.JSONDecodeError: If the extracted slice cannot be parsed as JSON.
    """
    s = (s or "").strip()
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", s, flags=re.DOTALL | re.IGNORECASE)
    if m:
        s = m.group(1).strip()
    first, last = s.find("{"), s.rfind("}")
    if first != -1 and last != -1 and last > first:
        s = s[first:last+1]
    return json.loads(s)


def clean_dict_keys(d: Any) -> Any:
    """Recursively strip stray quotes from dict keys and string values.

    Some models produce objects like {'"verdict"': '"approve"'}—which is syntactically
    valid but annoying. This normalizes keys/values without altering structure.

    Args:
        d: Any nested structure (dict/list/scalars).

    Returns:
        The same structure with normalized keys and string values.
    """
    if isinstance(d, dict):
        cleaned: Dict[Any, Any] = {}
        for k, v in d.items():
            clean_k = k.strip().strip('"').strip("'") if isinstance(k, str) else k
            cleaned[clean_k] = clean_dict_keys(v)
        return cleaned
    if isinstance(d, list):
        return [clean_dict_keys(item) for item in d]
    if isinstance(d, str):
        return d.strip().strip('"').strip("'") if d.startswith(('"', "'")) else d
    return d


def normalize_validator_report(raw: Any) -> Dict[str, Any]:
    """Normalize a validator LLM response into a consistent, easy-to-route dict.

    Target shape:
        {
          "verdict": "approve" | "revise",
          "reasons": str,
          "suggested_edits": Optional[str],
          ... # passthrough keys preserved
        }

    Behavior:
    - Accepts dicts or strings (tries to parse embedded JSON inside strings).
    - Unwraps common nesting keys: result/output/data/response.
    - Normalizes key casing and strips stray quotes.
    - Falls back to a conservative "revise" verdict if parsing fails.

    Args:
        raw: Arbitrary LLM output (string or dict-like).

    Returns:
        A normalized dictionary suitable for downstream routing and logging.
    """
    default: Dict[str, Any] = {
        "verdict": "revise",
        "reasons": "Failed to parse validator response",
        "suggested_edits": None,
    }

    try:
        print(f"DEBUG normalize_validator_report: Input type: {type(raw)}")
        print(f"DEBUG normalize_validator_report: Input preview: {str(raw)[:200]}")

        # Parse strings; try fenced JSON first, then brace extraction
        if isinstance(raw, str):
            raw = raw.strip()
            if raw.startswith("```"):
                raw = re.sub(r"```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```\s*$", "", raw)
            try:
                raw = json.loads(raw)
                print("DEBUG: Successfully parsed JSON string")
            except json.JSONDecodeError:
                match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw)
                if match:
                    try:
                        raw = json.loads(match.group(0))
                        print("DEBUG: Extracted JSON object from text")
                    except Exception:
                        return default
                else:
                    return default

        if not isinstance(raw, dict):
            return default

        # Unwrap common nesting keys
        for key in ("result", "output", "data", "response"):
            for k in (key, f'"{key}"', f"'{key}'"):
                if k in raw and isinstance(raw[k], dict):
                    raw = raw[k]
                    break

        # Normalize keys (lowercase, strip quotes/whitespace)
        normalized: Dict[str, Any] = {}
        for k, v in list(raw.items()):
            if not isinstance(k, str):
                continue
            ck = k.strip()
            while ck and ck[0] in "\"'" and ck[-1] in "\"'":
                ck = ck[1:-1]
            normalized[ck.lower()] = v

        # Extract verdict
        verdict = None
        for vkey in ("verdict", "status", "result"):
            if vkey in normalized:
                verdict = normalized[vkey]
                break
        if isinstance(verdict, str):
            verdict = verdict.strip().strip('"').strip("'").lower()
        verdict = verdict if verdict in ("approve", "revise") else "revise"

        # Reasons + suggested edits
        reasons = normalized.get("reasons") or normalized.get("reason") or "No reasons provided"
        if not isinstance(reasons, str):
            reasons = str(reasons)
        suggested = normalized.get("suggested_edits") or normalized.get("suggested_edit")
        if suggested is not None and not isinstance(suggested, str):
            suggested = str(suggested)

        out: Dict[str, Any] = {
            "verdict": verdict,
            "reasons": reasons,
            "suggested_edits": suggested,
        }

        # Pass through any additional keys for debugging/telemetry
        for k, v in normalized.items():
            if k not in {"verdict", "status", "result", "reasons", "reason", "suggested_edits", "suggested_edit"}:
                out[k] = v

        return out

    except Exception as e:
        print(f"CRITICAL ERROR in normalize_validator_report: {e}")
        return default

def make_llm(provider: str, temperature: float) -> ChatDatabricks:
    """Return a Databricks-hosted chat model client for the requested provider.

    This helper abstracts over multiple model-serving endpoints (OpenAI, Anthropic, Llama)
    so the rest of the pipeline can stay provider-agnostic.

    Design intent:
        - Normalize provider labels to lowercase.
        - Route to a corresponding Databricks Model Serving endpoint.
        - Fail fast on unrecognized providers.

    Args:
        provider: Name of the LLM provider ("openai", "anthropic", or "llama").
        temperature: Sampling temperature to control model creativity.

    Returns:
        ChatDatabricks: A configured chat model client bound to the right endpoint.

    Raises:
        ValueError: If the provider is not recognized.

    Example:
        >>> llm = make_llm("openai", temperature=0.3)
        >>> llm.invoke("Hello, world!")
    """
    provider = (provider or "").lower()

    # Replace these with your actual Databricks Model Serving endpoint names
    if provider == "openai":
        return ChatDatabricks(endpoint="gpt-5-chat", temperature=temperature)
    if provider == "anthropic":
        return ChatDatabricks(endpoint="databricks-claude-sonnet-4", temperature=temperature)
    if provider == "llama":
        return ChatDatabricks(endpoint="databricks-llama-4-maverick", temperature=temperature)

    raise ValueError(f"Unknown provider: {provider}")

PLAN_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "pyspark_code": {"type": "string"},
        "expected_effects": {
            "type": "object",
            "properties": {
                "row_change": {
                    "type": "string",
                    "enum": ["increase", "decrease", "similar", "unknown"],
                },
                "new_columns": {"type": "array", "items": {"type": "string"}},
                "dropped_columns": {"type": "array", "items": {"type": "string"}},
                "column_type_changes": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                },
            },
            "required": ["row_change", "new_columns", "dropped_columns"],
            "additionalProperties": True,
        },
    },
    "required": ["summary", "pyspark_code", "expected_effects"],
    "additionalProperties": False,
}

#: Fast denylist to short-circuit obviously unsafe snippets *before* AST analysis.
#: Keep this blunt and conservative; we prefer false positives over risky code.
DENY_PATTERNS: List[str] = [
    r"\bdf\.write\b",
    r"\.saveAsTable\b",
    r"\binsertInto\b",
    r"\bspark\.sql\s*\(",
    r"\bdbutils\.",
    r"shutil",
    r"os\.remove",
    r"subprocess",
    r"__import__",
    r"\bopen\s*\(",
    r"SparkFiles",
    r"drop\s+table",
    r"delete\s+from",
]


def code_is_safe(pyspark_code: str) -> Tuple[bool, str]:
    """Static safety screening for proposed PySpark snippets.

    This is a two-stage filter:
      1) **Regex denylist** for quick wins (fast and opinionated).
      2) **AST walk** to block dangerous nodes/calls (imports, globals),
         plus attribute calls on sensitive objects (`spark`, `dbutils`),
         and any write/save-style attributes anywhere in the tree.

    We’re intentionally strict: these snippets execute in the same process,
    so we block IO and side-effects. If it’s not clearly safe, it’s a “no”.

    Args:
        pyspark_code: The sanitized code string to check (no imports/fences).

    Returns:
        (ok, reason):
            ok: True if the snippet passes all checks; False otherwise.
            reason: "ok" on success or a short explanation on failure.

    Examples:
        >>> code_is_safe("df_out = df.select('x')")
        (True, 'ok')
        >>> code_is_safe("spark.sql('select 1')")
        (False, 'Disallowed attribute call: spark.sql')
    """
    # 1) Quick regex denies
    for pat in DENY_PATTERNS:
        if re.search(pat, pyspark_code, flags=re.IGNORECASE):
            return False, f"Denied pattern matched: {pat}"

    # 2) AST inspection (imports, dangerous calls, attribute calls)
    try:
        tree = ast.parse(pyspark_code, mode="exec")
        for node in ast.walk(tree):
            # Ban import statements and scope trickery
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal)):
                return False, f"Disallowed AST node: {type(node).__name__}"

            # Ban dangerous builtins: eval/exec/compile/open/__import__
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in {
                    "eval",
                    "exec",
                    "compile",
                    "open",
                    "__import__",
                }:
                    return False, f"Disallowed call: {node.func.id}"

                # Ban attribute calls on sensitive roots (spark.*, dbutils.*)
                if isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name) and node.func.value.id in {
                        "spark",
                        "dbutils",
                    }:
                        return False, (
                            f"Disallowed attribute call: "
                            f"{node.func.value.id}.{node.func.attr}"
                        )

            # Ban write/save actions anywhere in the attribute chain
            if isinstance(node, ast.Attribute) and node.attr in {
                "write",
                "save",
                "saveAsTable",
                "insertInto",
            }:
                return False, f"Disallowed attribute: .{node.attr}"

    except Exception as e:
        # If we can’t confidently analyze the code, refuse it.
        return False, f"AST analysis error: {e}"

    return True, "ok"

PLANNER_SYS: str = """You are a senior PySpark data engineer. 
Given a user's natural-language instruction and a DataFrame schema preview, generate SAFE PySpark code that:
- uses the input `df` as the starting DataFrame,
- performs only in-memory transformations,
- NEVER writes to disk and NEVER uses spark.sql or dbutils,
- ends with a variable named `df_out` assigned to the resulting DataFrame.

Return JSON strictly matching this schema:
{schema}
"""

PLANNER_HUMAN: str = """User instruction:
```
{instruction}
```

DataFrame schema (dtypes and sample column names):
```
{schema_text}
```

Return ONLY the JSON (no prose)."""

VALIDATOR_SYS: str = """You are a meticulous code validator and QA agent.
Input: a proposed PySpark snippet and claimed effects.
Task:
1) Check safety (no IO, no spark.sql, only DataFrame ops).
2) Check that code likely compiles given the schema preview.
3) Predict whether expected effects are plausible.
Return a JSON:
{{"verdict":"approve"|"revise","reasons":"...", "suggested_edits":"(optional code if revise)"}}
No prose outside JSON.
"""

VALIDATOR_HUMAN: str = """Proposed plan JSON:
```
{plan_json}
```

Schema preview:
```
{schema_text}
```
"""

@dataclass
class ValidationResult:
    ok: bool
    warnings: List[str]
    details: Dict[str, Any]

def validate_transform(baseline: DataFrame, sample_out: DataFrame, plan: Dict[str,Any]) -> ValidationResult:
    warns = []
    details = {}
    
    try:
        base_cnt = baseline.count()
        out_cnt = sample_out.count()
        details["row_counts"] = {"baseline": base_cnt, "output": out_cnt}

        # Safely get expected_effects
        expected_effects = plan.get("expected_effects", {})
        if not isinstance(expected_effects, dict):
            print(f"WARNING: expected_effects is not a dict: {type(expected_effects)}")
            expected_effects = {}
        
        # Row count sanity
        row_change = expected_effects.get("row_change", "unknown")
        if isinstance(row_change, str):
            row_change = row_change.strip().strip('"').strip("'").lower()
        
        if out_cnt == 0 and row_change not in ["decrease","unknown"]:
            warns.append("Output sample row count is 0; this may be unintended.")
        if out_cnt > max(1, base_cnt) * 5:
            warns.append("Output sample exploded to >5x baseline rows; check joins/explodes.")

        # Column sanity
        base_cols = set(baseline.columns)
        out_cols = set(sample_out.columns)
        new_cols = out_cols - base_cols
        dropped = base_cols - out_cols
        details["columns"] = {
            "new": sorted(new_cols),
            "dropped": sorted(dropped),
            "final": sorted(out_cols)
        }

        # Compare with expected - safely handle the list
        claimed_new_raw = expected_effects.get("new_columns", [])
        if not isinstance(claimed_new_raw, list):
            claimed_new_raw = []
        # Clean each item in case they have quotes
        claimed_new = set()
        for item in claimed_new_raw:
            if isinstance(item, str):
                cleaned = item.strip().strip('"').strip("'")
                claimed_new.add(cleaned)
        
        if not claimed_new.issuperset(new_cols):
            warns.append(f"Plan under-reported new columns: {sorted(new_cols - claimed_new)}")

        # Null checks for new columns
        for c in new_cols:
            nulls = sample_out.filter(F.col(c).isNull()).count()
            if nulls == out_cnt and out_cnt > 0:
                warns.append(f"New column '{c}' is entirely null in sample.")

        ok = len(warns) == 0
        return ValidationResult(ok=ok, warnings=warns, details=details)
        
    except Exception as e:
        print(f"Error in validate_transform: {e}")
        traceback.print_exc()
        return ValidationResult(
            ok=False,
            warnings=[f"Validation error: {str(e)}"],
            details={"error": str(e)}
        )

class AgentState(TypedDict):
    instruction: str
    df_in: DataFrame
    df_sample: Optional[DataFrame]
    plan: Optional[Dict[str, Any]]
    code: Optional[str]
    validator_report: Optional[Dict[str, Any]]
    df_out: Optional[DataFrame]
    write_path: str
    write_format: str
    write_mode: str

from typing import Any, Dict
from jsonschema import Draft202012Validator

def node_plan(state: AgentState, llm: Any) -> Dict[str, Any]:
    """LLM-driven planning node — generate a safe PySpark transformation plan.

    This node is responsible for converting a user's natural-language
    instruction into a fully structured execution plan, including:
        - a summarized plan description,
        - generated PySpark transformation code, and
        - expected downstream effects on the DataFrame.

    Workflow:
        1. Render the planner prompt with schema context.
        2. Invoke the LLM to produce a JSON plan.
        3. Parse and validate the JSON structure against PLAN_SCHEMA.
        4. Sanitize and safety-check the generated PySpark code.

    Args:
        state: The current AgentState dictionary, containing the input DataFrame
               (`df_in`) and the user's instruction.
        llm: The planner LLM client (e.g., a Databricks-hosted Chat model).

    Returns:
        Updated AgentState including:
            - `plan`: Validated and cleaned plan JSON.
            - `code`: Sanitized PySpark code snippet ready for validation/execution.

    Raises:
        RuntimeError: If the LLM output cannot be parsed, validated, or fails safety checks.

    Example:
        >>> new_state = node_plan(state, planner_llm)
        >>> print(new_state["plan"]["summary"])
    """
    # Extract and format schema context for the prompt
    schema_text = schema_preview(state["df_in"])

    # Compose planner prompt with schema injection
    planner = ChatPromptTemplate.from_messages([
        ("system", PLANNER_SYS),
        ("human", PLANNER_HUMAN),
    ]).partial(
        schema=json.dumps(PLAN_SCHEMA, indent=2),
        schema_text=schema_text,
    )

    # Invoke the planner LLM with the user's natural-language instruction
    ai = llm.invoke(planner.format_messages(instruction=state["instruction"]))

    try:
        # Parse and clean LLM output
        plan = parse_llm_json(ai.content)
        plan = clean_dict_keys(plan)

        # Validate plan structure strictly against PLAN_SCHEMA
        Draft202012Validator(PLAN_SCHEMA).validate(plan)

    except Exception as e:
        print(f"Raw AI response: {ai.content[:1000]}")
        raise RuntimeError(
            f"Planner failed to return valid JSON plan: {e}\nRaw: {ai.content[:1000]}"
        )

    # --- Extract and sanitize PySpark code ---
    raw_code = plan.get("pyspark_code", "")
    if not raw_code:
        raise RuntimeError("Plan missing 'pyspark_code' field")

    print(f"DEBUG: Raw code from LLM:\n{raw_code}\n")

    code_clean = sanitize_pyspark_code(raw_code)
    print(f"DEBUG: Sanitized code:\n{code_clean}\n")

    # --- Safety check ---
    ok, why = code_is_safe(code_clean)
    if not ok:
        print(f"Failed safety check on code:\n{code_clean}")
        raise RuntimeError(f"Proposed code failed safety check: {why}")

    # Return updated state (immutably merged)
    return {**state, "plan": plan, "code": code_clean}

def escape_for_template(text: str) -> str:
    """Escape curly braces for safe inclusion in Python format strings.

    Many LLM prompt templates (and f-strings) treat braces `{}` as
    placeholders. This helper doubles them (`{{` and `}}`) so literal
    braces survive `.format()` substitution without triggering syntax errors.

    Args:
        text: Raw string that may contain literal `{` or `}` characters.

    Returns:
        The escaped string where all `{` and `}` have been safely doubled.

    Example:
        >>> escape_for_template("{plan_json}")
        '{{plan_json}}'
    """
    # Double braces so `.format()` interprets them literally.
    return text.replace("{", "{{").replace("}", "}}")

from typing import Any, Dict
import json
import traceback
from langchain_core.prompts import ChatPromptTemplate

def node_validate(state: AgentState, validator_llm: Any) -> Dict[str, Any]:
    """Validator node — sanity-check the plan and optionally dry-run on a sample.

    Responsibilities
    ----------------
    1) Build a validator prompt using the plan JSON and a compact schema preview.
    2) Invoke the validator LLM and normalize its (often messy) output.
    3) If the validator says "revise" *and* provides safe `suggested_edits`,
       substitute those for the dry-run.
    4) If a sample size is configured via a Databricks widget, run a dry-run
       validation to catch obvious issues (row explosions, null-only new columns, etc.).

    Behavior
    --------
    - Returns an updated state with `validator_report` and possibly updated `code`.
    - On any exception, returns a conservative 'revise' verdict with diagnostic info.
    - Does not persist or mutate external systems; purely in-memory checks.

    Args:
        state: Current AgentState dict (expects `df_in`, `plan`, and `code` at minimum).
        validator_llm: LLM client used to validate safety/compatibility/effects.

    Returns:
        A new AgentState dict with:
          - `validator_report`: normalized validator output (and optional dry-run results)
          - `code`: possibly updated code if safe suggested edits were provided
    """
    print("=" * 60)
    print("ENTERING node_validate")
    print("=" * 60)
    
    try:
        # --- Step 1: Schema preview for validator context ---
        print("Step 1: Getting schema preview...")
        schema_text = schema_preview(state["df_in"])
        print(f"Schema preview length: {len(schema_text)}")
        
        # --- Step 2: Build validator prompt ---
        print("Step 2: Building prompt...")
        vprompt = ChatPromptTemplate.from_messages([
            ("system", VALIDATOR_SYS),
            ("human", VALIDATOR_HUMAN),
        ])
        
        # --- Step 3: Format messages (escape braces in plan JSON) ---
        print("Step 3: Formatting messages...")
        plan_json_str = json.dumps(state["plan"], indent=2)
        plan_json_escaped = escape_for_template(plan_json_str)
        messages = vprompt.format_messages(
            plan_json=plan_json_escaped,
            schema_text=schema_text,
        )
        print(f"Formatted {len(messages)} messages")
        
        # --- Step 4: Invoke validator LLM ---
        print("Step 4: Invoking validator LLM...")
        ai = validator_llm.invoke(messages)
        print(f"LLM response received, length: {len(ai.content)}")
        print(f"DEBUG: Raw validator response: {ai.content[:500]}")
        
        # --- Step 5: Normalize/clean validator report ---
        print("Step 5: Normalizing validator report...")
        report = normalize_validator_report(ai.content)
        print("Step 6: Normalization complete")
        
        print("Step 7: Cleaning dict keys...")
        report = clean_dict_keys(report)
        print("Step 8: Cleaning complete")
        
        print(f"DEBUG: Normalized report keys: {list(report.keys())}")
        print(f"DEBUG: Verdict value: '{report.get('verdict')}' (type: {type(report.get('verdict'))})")
        
        # Ensure required keys exist
        if "verdict" not in report:
            print("WARNING: verdict missing from report, defaulting to 'revise'")
            report["verdict"] = "revise"
        if "reasons" not in report:
            report["reasons"] = "Validator response incomplete"
        if "suggested_edits" not in report:
            report["suggested_edits"] = None
        
        # Prefer safe suggested edits if validator requested a revision
        code_to_try = state["code"]
        if report["verdict"] == "revise" and report.get("suggested_edits"):
            candidate = sanitize_pyspark_code(report["suggested_edits"])
            ok, why = code_is_safe(candidate)
            if ok:
                code_to_try = candidate
        
        # --- Optional dry-run sample validation (Databricks widget controlled) ---
        # Keep the widget name exactly as used operationally.
        try:
            n = int(dbutils.widgets.get("Sample Rows") or "0")  # noqa: F821
        except Exception:
            n = 0
        
        df_sample = state["df_in"].limit(n) if n > 0 else None
        if df_sample is not None and df_sample.count() == 0:
            df_sample = None
        
        if df_sample is not None:
            try:
                sample_out = exec_pyspark_code(code_to_try, df_sample)
                vr = validate_transform(df_sample, sample_out, state["plan"])
                report["dry_run"] = {
                    "ok": vr.ok,
                    "warnings": vr.warnings,
                    "details": vr.details,
                }
                if not vr.ok:
                    report["verdict"] = "revise"
                    report["reasons"] = " ; ".join(vr.warnings) if vr.warnings else "Validation failed"
            except Exception as e:
                report["verdict"] = "revise"
                report["reasons"] = f"Code failed on sample: {e}"
                report["dry_run"] = {"ok": False, "error": str(e)}
        
        return {**state, "validator_report": report, "code": code_to_try}
        
    except Exception as e:
        print(f"CRITICAL ERROR in node_validate: {e}")
        traceback.print_exc()
        # Conservative fallback: return a structured "revise" verdict
        return {
            **state,
            "validator_report": {
                "verdict": "revise",
                "reasons": f"Validator node crashed: {str(e)}",
                "suggested_edits": None,
            },
        }

def node_execute(state: AgentState) -> Dict[str, Any]:
    """Execution node — run the validated PySpark code over the full dataset.

    Assumptions
    -----------
    - `state["code"]` is sanitized and has already passed safety checks.
    - `state["df_in"]` is a valid Spark DataFrame.
    - The executed snippet must assign its result to `df_out`.

    Behavior
    --------
    Executes the code in a constrained namespace (`df`, `F`) and returns a new
    state with `df_out` attached. If execution fails, we raise a `RuntimeError`
    with the original exception and the attempted code for fast debugging.

    Args:
        state: Current AgentState dict containing `code` and `df_in`.

    Returns:
        A new AgentState dict with `df_out` set to the execution result.

    Raises:
        RuntimeError: If the snippet fails to execute or does not yield `df_out`.
    """
    try:
        df_out = exec_pyspark_code(state["code"], state["df_in"])
        return {**state, "df_out": df_out}
    except Exception as e:
        # Fail loudly with enough context to debug quickly.
        raise RuntimeError(f"Execution failed: {e}\nCode:\n{state['code']}") from e

def node_write(state: AgentState) -> Dict[str, Any]:
    """Write the output DataFrame to persistent storage.

    This node handles the final persistence step in the LangGraph workflow.
    It takes the `df_out` DataFrame produced by the execution node and writes it
    to the configured output path using the specified format and mode.

    Design intent
    --------------
    - Enforce explicit control over write paths, formats, and modes.
    - Guard against missing outputs (we never silently skip writes).
    - Normalize DBFS paths so the same logic works for `/dbfs/` and `dbfs:/` URIs.
    - Keep format routing simple — just Spark-native writers (no spark.sql).

    Args:
        state: Current AgentState dict containing the keys:
            - `df_out`: DataFrame to write.
            - `write_path`: Destination path (DBFS/S3/ABFSS/etc.).
            - `write_format`: Output format ("delta", "parquet", "csv", "json").
            - `write_mode`: Write mode ("overwrite", "append", etc.).

    Returns:
        The unmodified AgentState dict after successful write.

    Raises:
        RuntimeError: If `df_out` is missing or write operation fails.

    Example:
        >>> node_write({
        ...     "df_out": df,
        ...     "write_path": "/dbfs/tmp/output",
        ...     "write_format": "delta",
        ...     "write_mode": "overwrite"
        ... })
        Successfully wrote output to dbfs:/tmp/output in delta format
    """
    # --- Sanity check ---
    if not state.get("df_out"):
        raise RuntimeError("No output DataFrame to write")

    # Normalize DBFS path so Spark understands both /dbfs/ and dbfs:/ conventions
    path = _normalize_dbfs_path(state["write_path"])
    fmt = state.get("write_format", "parquet")
    mode = state.get("write_mode", "overwrite")

    writer = state["df_out"].write.mode(mode)

    # --- Format routing ---
    # Keep this explicit; we don’t use dynamic evals for safety.
    if fmt == "delta":
        writer.format("delta").save(path)
    elif fmt == "parquet":
        writer.parquet(path)
    elif fmt == "json":
        writer.json(path)
    elif fmt == "csv":
        writer.option("header", True).csv(path)
    else:
        # Default fallback for unknown formats
        writer.parquet(path)

    print(f"Successfully wrote output to {path} in {fmt} format")
    return state

from typing import Any
import traceback

def safe_get_verdict(state: AgentState) -> str:
    """Extract a normalized verdict from the validator report for graph routing.

    This function is used by LangGraph’s conditional edge logic to determine
    whether the workflow should proceed to execution or stop for revision.

    Behavior
    --------
    - Returns `"approve"` only if the validator explicitly says so.
    - Defaults to `"revise"` in all other cases (including missing or malformed state).
    - Strips accidental quotes or whitespace that often appear in raw LLM outputs.

    Args:
        state: Current AgentState dictionary, expected to contain
               a `validator_report` with a `"verdict"` key.

    Returns:
        `"approve"` if the validator report explicitly approves the plan;
        otherwise `"revise"` (safe fallback).

    Example:
        >>> safe_get_verdict({"validator_report": {"verdict": "approve"}})
        'approve'
        >>> safe_get_verdict({})
        'revise'
    """
    try:
        report = state.get("validator_report")
        if not report:
            print("No validator_report in state, defaulting to 'revise'")
            return "revise"

        verdict = report.get("verdict", "revise")

        # Normalize verdict string for robustness
        if isinstance(verdict, str):
            verdict = verdict.strip().strip('"').strip("'").lower()

        result = "approve" if verdict == "approve" else "revise"
        print(f"Routing decision: {result}")
        return result

    except Exception as e:
        print(f"Error getting verdict: {e}")
        traceback.print_exc()
        return "revise"
    
def build_graph(planner_llm: ChatDatabricks, validator_llm: ChatDatabricks) -> Any:
    """Assemble and compile the LangGraph workflow (plan → validate → execute → write).
    
    Args:
        planner_llm: Chat model used to generate the PySpark plan/code.
        validator_llm: Chat model used to validate safety/compatibility/effects.

    Returns:
        A compiled LangGraph workflow object ready to `.invoke(initial_state)`.

    Example:
        >>> graph = build_graph(planner_llm, validator_llm)
        >>> result = graph.invoke(initial_state)
    """
    # Create the stateful workflow over the AgentState TypedDict contract.
    workflow = StateGraph(AgentState)

    # Nodes
    # Capture LLMs in closures so node functions stay pure wrt global state.
    workflow.add_node("plan",     lambda s: node_plan(s, planner_llm))
    workflow.add_node("validate", lambda s: node_validate(s, validator_llm))
    workflow.add_node("execute",  node_execute)
    workflow.add_node("write",    node_write)

    # Linear edges: START → plan → validate
    workflow.add_edge(START, "plan")
    workflow.add_edge("plan", "validate")

    # Conditional routing after validation:
    #   - approve → execute
    #   - revise  → END (could loop back to "plan" if you add retry semantics later)
    workflow.add_conditional_edges(
        "validate",
        safe_get_verdict,
        {
            "approve": "execute",
            "revise": END,  # intentional early stop on non-approved code
        },
    )

    # Final leg: execute → write → END
    workflow.add_edge("execute", "write")
    workflow.add_edge("write", END)

    # Compile to an executable graph.
    return workflow.compile()

def run_transformation(
    instruction: str,
    input_paths: str,
    output_path: str,
    input_format: str = "auto",
    output_format: str = "delta",
    write_mode: str = "overwrite",
    planner_provider: str = "openai",
    validator_provider: str = "openai",
    temperature: float = 0.0
) -> Optional[DataFrame]:
    """Plan, validate, execute, and persist a PySpark transformation.

    This is the single entry point that wires the whole workflow together:
    - Load inputs (tolerant to schema drift).
    - Spin up planner/validator LLM handles.
    - Build the LangGraph (plan → validate → execute → write).
    - Invoke the graph with a minimal initial state.
    - Print a concise run summary for quick inspection.

    Args:
        instruction: Natural-language description of the desired transformation.
        input_paths: Comma-separated list of input paths (DBFS/S3/ABFSS/etc.).
        output_path: Destination path for the final dataset.
        input_format: Input format or "auto" (parquet|delta|csv|json).
        output_format: Output format (parquet|delta|csv|json).
        write_mode: Spark write mode (e.g., "overwrite", "append").
        planner_provider: LLM provider for planning ("openai"|"anthropic"|"llama").
        validator_provider: LLM provider for validation.
        temperature: LLM decoding temperature.

    Returns:
        The final `DataFrame` (if execution reached the execute/write stages), else `None`.

    Notes:
        - Requires an active `spark` session in the runtime (Databricks notebooks inject it).
        - This function prints human-readable checkpoints by design for operational clarity.
    """
    # Load input data (handles multi-path union by name with missing columns filled)
    print(f"Loading data from: {input_paths}")
    df_in: DataFrame = load_inputs(input_paths, input_format)
    print(f"Loaded DataFrame with {df_in.count()} rows and {len(df_in.columns)} columns")

    # Create LLM clients (endpoints are configured in make_llm)
    planner_llm = make_llm(planner_provider, temperature)
    validator_llm = make_llm(validator_provider, temperature)

    # Build the orchestration graph
    graph = build_graph(planner_llm, validator_llm)

    # Initial state passed into the graph (intentionally minimal and explicit)
    initial_state: Dict[str, Any] = {
        "instruction": instruction,
        "df_in": df_in,
        "df_sample": None,
        "plan": None,
        "code": None,
        "validator_report": None,
        "df_out": None,
        "write_path": output_path,
        "write_format": output_format,
        "write_mode": write_mode,
    }

    # Execute the graph
    print("\n=== Starting transformation ===")
    result: Dict[str, Any] = graph.invoke(initial_state)

    # Summarize plan and validation for quick operator feedback
    print("\n=== Plan Summary ===")
    if result.get("plan"):
        print(result["plan"].get("summary", "No summary available"))

    print("\n=== Validator Report ===")
    if result.get("validator_report"):
        report = result["validator_report"]
        print(f"Verdict: {report.get('verdict', 'unknown')}")
        print(f"Reasons: {report.get('reasons', 'none')}")
        if report.get("dry_run"):
            print(f"Dry run: {report['dry_run']}")

    print("\n=== Execution Complete ===")
    if result.get("df_out"):
        df_out: DataFrame = result["df_out"]
        print(f"Output: {df_out.count()} rows, {len(df_out.columns)} columns")
        print(f"Schema: {df_out.printSchema()}")

    return result["df_out"] if result.get("df_out") else None

