# LangGraph-powered PySpark Transformation Agent

Turn plain English into **safe, validated** PySpark DataFrame transforms.  
Plans → validates (no IO, no `spark.sql`, AST checks) → executes in a constrained namespace (`df`, `F`) → optionally writes (Delta/Parquet/CSV/JSON).

---

## ✨ What it does

1. **Plan** — Generate PySpark from a natural-language instruction (LLM).
2. **Validate** — Safety + plausibility: denylist regex, AST walk, optional dry-run on a sample.
3. **Execute** — Run code in a constrained namespace (`df`, `F`).
4. **Write** — Persist via explicit formats/modes only.

```
instruction ──▶ plan (LLM)
              └▶ validate (LLM + static checks)
              └▶ execute (sandboxed)
              └▶ write (explicit IO)
```

---

## 🧱 Safety posture

- Strip imports, block IO (`df.write`, `saveAsTable`, `spark.sql`, `dbutils`, etc.).
- AST inspection to reject dangerous nodes (`import`, `global`, `eval/exec/compile/open/__import__`).
- Execution sandbox only receives `df` and `F` (no globals).
- Validator output normalization (messy JSON becomes predictable keys/values).

---

## 📦 Requirements

- A Spark runtime with **PySpark**.
  - **Databricks**: obviously.

---

## 🧩 Installation

Use the pinned versions below or whatever version that works for you.  My cluster is in Azure Databricks 16.4 LTS (includes Apache Spark 3.5.2, Scala 2.12):

langchain==0.3.27
langgraph==0.6.8
pydantic==2.11.10
langchain-openai==0.3.35
jsonschema==4.25.1
langchain-community==0.3.30
databricks-langchain==0.8.1

You can also drop these into `requirements.txt` the old fashion way:

```
langchain==0.3.27
langgraph==0.6.8
pydantic==2.11.10
langchain-openai==0.3.35
jsonschema==4.25.1
langchain-community==0.3.30
databricks-langchain==0.8.1
```

---

## ⚙️ Configuration

### LLM endpoints

`make_llm(provider, temperature)` routes to Databricks Model Serving via `ChatDatabricks`.  
Swap the placeholder endpoint names with your real endpoints:

```python
if provider == "openai":
    return ChatDatabricks(endpoint="your-gpt-endpoint", temperature=temperature)
if provider == "anthropic":
    return ChatDatabricks(endpoint="your-claude-endpoint", temperature=temperature)
if provider == "llama":
    return ChatDatabricks(endpoint="your-llama-endpoint", temperature=temperature)
```

## 🚀 Quickstart

### A) Databricks Notebook / Job

```python
from agent import run_transformation

out_df = run_transformation(
    instruction = "Select customer_id, sum(amount) as total by month for 2024 only.",
    input_paths = "dbfs:/datasets/transactions_2023.parquet, dbfs:/datasets/transactions_2024.parquet",
    output_path = "dbfs:/tmp/tx_agg_2024_delta",
    input_format = "auto",
    output_format = "delta",
    write_mode = "overwrite",
    planner_provider = "anthropic",   # or "openai" / "llama"
    validator_provider = "anthropic", # or "openai" / "llama"
    temperature = 0.0
)
```

## 🧪 Examples (instructions you can try)

- “Select `customer_id`, `sum(amount)` as `total` for 2024; group by month.”
- “Clean `email` (lowercase, trim) and drop duplicate rows by `user_id`.”
- “Join orders to customers on `customer_id`; keep only U.S. rows; write Parquet.”

---

## 🩺 Troubleshooting

- **`NameError: name 'spark' is not defined`**  
  The script creates a `SparkSession` (`SparkSession.builder.getOrCreate()`).  
  If you removed it for some reason, add:

  ```python
  from pyspark.sql import SparkSession
  spark = SparkSession.builder.getOrCreate()
  ```

- **Validator always returns “revise”**  
  Check the denylist and AST rules; ensure the generated code doesn’t call IO, `spark.sql`, or `dbutils`.  
  Increase `sample_rows` to catch logic issues earlier.

- **Mixed schemas across input files**  
  This is handled: missing columns are added as nullable strings so `unionByName` succeeds.

---

## 🗺️ Roadmap (nice to have)

- Column-level statistics & drift checks in the validator.
- Join cardinality and skew detection.
- Typed expectations (e.g., pydantic model for `expected_effects`).

---

## 📜 License

MIT.

---

## 🤝 Contributing

Small, surgical PRs welcome: doc improvements, new validators, better prompt hygiene.
