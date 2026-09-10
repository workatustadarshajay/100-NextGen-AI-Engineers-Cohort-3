# 98. PyTorch tensors and autograd

**Track:** DL  
**Companion notebook:** [../Practice/98-pytorch-tensors-and-autograd.ipynb](../Practice/98-pytorch-tensors-and-autograd.ipynb)

## Purpose

devices, shapes, computation graphs, gradients, and debugging.

This guide explains what the notebook demonstrates, why the technique matters, how its major functions behave, and how to recognize misuse. The notebook is the executable source of truth; this document is the conceptual and engineering reference.

## Topic-specific learning map

**PyTorch tensors and autograd** belongs to **Deep learning** and is prioritized as **P0**. Its practical scope is devices, shapes, computation graphs, gradients, and debugging. The lesson treats those phrases as mechanisms to understand and verify, not vocabulary to memorize.

1. **Devices:** Define it, locate it in the system flow, and identify one measurable behavior.
2. **Shapes:** Define it, locate it in the system flow, and identify one measurable behavior.
3. **Computation Graphs:** Define it, locate it in the system flow, and identify one measurable behavior.
4. **Gradients:** Define it, locate it in the system flow, and identify one measurable behavior.
5. **Debugging:** Define it, locate it in the system flow, and identify one measurable behavior.

**Study sequence:** Begin by drawing the input-to-output flow for PyTorch tensors and autograd. Next isolate each concept in a tiny example. Compare a baseline with one alternative, deliberately create a failure, and explain the observed evidence. Finally place the mechanism inside a realistic system with versioning, tests, monitoring, security, and a fallback.

**Mastery standard:** A learner has mastered PyTorch tensors and autograd when they can explain every listed concept without library jargon, implement or simulate the central mechanism, select an appropriate evaluation method, diagnose a counterexample, compare alternatives, and defend a production decision using explicit constraints.

## Before you begin

**Recommended prerequisites:** Vectors and matrices, supervised learning, loss functions, gradient-descent intuition, and Python array operations.

**Track foundation:** Deep learning composes differentiable layers that learn representations. Architecture creates an inductive bias; optimization updates parameters; regularization and validation control generalization; compute and data quality often dominate small syntax choices.

**Lifecycle:** `tensor preparation -> forward pass -> loss -> backpropagation -> optimizer step -> validation -> checkpoint -> inference`

**Evidence expected:** Learning curves, held-out metrics, ablations, seed variability, error examples, resource measurements, and comparison with simpler models.

A reader who is new to the topic should first learn the vocabulary, then trace one small example, then modify a single parameter, and only afterward apply the method to new data. Do not begin with a large project where data, algorithm, and deployment failures are mixed together.

## Teach it from zero

This lesson focuses on **PyTorch tensors and autograd**. In plain language: devices, shapes, computation graphs, gradients, and debugging. The topic belongs to the **DL** track, so interpret every operation through this larger foundation: Deep learning composes differentiable layers that learn representations. Architecture creates an inductive bias; optimization updates parameters; regularization and validation control generalization; compute and data quality often dominate small syntax choices.

The central learning question is not “Which line should I copy?” It is “What information enters, what operation happens, what state is created, what output leaves, and how can I prove that output is useful?” For **PyTorch tensors and autograd**, keep devices, shapes, computation graphs, gradients, and debugging. as the concrete goal while examining every function below.

### Concept map

1. **Input:** Define the object being processed and its semantic meaning.
2. **Representation:** Determine how the input is encoded in arrays, tables, text, vectors, or schemas.
3. **Operation:** Apply the deterministic rule or learned mapping.
4. **State:** Identify statistics, coefficients, weights, indexes, prompts, or configuration that must persist.
5. **Output:** Interpret values in domain language, including confidence and limitations.
6. **Evaluation:** Compare with a baseline and test realistic failures.
7. **Operation in context:** Place the feature inside the larger DL lifecycle.

## Learning outcomes

After completing the notebook and guide, you should be able to define the method, derive or narrate its core mechanism, identify required inputs and produced outputs, choose key parameters, build a baseline, evaluate results, diagnose failure modes, and explain how the component would be tested and monitored in production.

## Mental model

Start with a clearly defined input, apply one observable transformation or learning step, and inspect the output. A reliable workflow records shapes, data types, parameters, random seeds, and evaluation criteria. When a method learns from data, fit it only on training observations and reuse the learned state for validation, test, and production inputs.

Think in six layers:

1. **Problem:** What decision or user need does the output support?
2. **Data:** What does one row, sequence, image, document, or query represent?
3. **Method:** What fixed rules and learned parameters produce the output?
4. **Evidence:** Which baseline, metric, and diagnostic establish value?
5. **System:** How is the method versioned, served, secured, and monitored?
6. **Impact:** Who can be helped or harmed, and what controls are required?

## Theory and notation

A general learned mapping can be written as $\hat{y}=f_\theta(x)$, where $x$ is an input, $\theta$ is learned state, and $\hat{y}$ is an estimate. Training often minimizes

$$J(\theta)=\frac{1}{n}\sum_{i=1}^{n}\mathcal{L}(f_\theta(x_i), y_i)+\lambda R(\theta).$$

The loss $\mathcal{L}$ measures fit, regularization $R$ discourages undesirable complexity, and $\lambda$ balances them. For descriptive analysis and deterministic transformations, replace the optimization question with: which rule is applied, which information is discarded, and whether the output remains semantically valid.

## Internal mechanics

Each layer maps an input tensor to an output tensor. The forward pass composes those mappings and produces a loss. Automatic differentiation applies the chain rule backward through the computation graph to obtain gradients. An optimizer updates parameters; batches approximate the full-data gradient; repeated epochs expose the model to training examples. Architecture controls information flow, while activation functions make nonlinear composition possible.

For this topic, explicitly trace the representation entering the core operation, the configuration supplied by the developer, state learned from data or constructed from sources, the intermediate output, and the final decision or artifact. This trace reveals hidden dependencies and prevents confusion between fitting, transformation, inference, retrieval, and evaluation.

## Alternatives and method selection

Use dense networks for fixed-size feature vectors, convolutions for local spatial structure, recurrent or attention models for ordered sequences, autoencoders for representation and reconstruction, and transfer learning when a relevant pretrained representation exists. Always compare with a simpler statistical or classical ML model, especially on small tabular datasets.

A defensible selection compares at least one simpler option and one credible alternative. Compare data needs, assumptions, quality, uncertainty, interpretability, latency, memory, update cost, privacy, authorization, and maintenance. Document why the chosen method satisfies constraints rather than describing it as universally best.

## End-to-end workflow

1. Write the question and define success before examining results.
2. Specify input and output schemas, units, domains, and null behavior.
3. Inspect raw input, missingness, range, duplicates, labels, and provenance.
4. Establish a naive or rule-based baseline.
5. Choose a split strategy consistent with deployment, including time and group boundaries.
6. Apply the technique with explicit, versioned parameters.
7. Inspect intermediate values and fitted attributes.
8. Evaluate the primary metric and at least one complementary diagnostic.
9. Slice errors by meaningful cohort and inspect individual failures.
10. Test malformed, boundary, rare, and out-of-domain inputs.
11. Package all preprocessing and learned state together.
12. Define monitoring, alerts, fallback behavior, and rollback.

## Worked real-world scenario

For image defect detection, start with a labeled split grouped by production batch. Establish a classical image-feature baseline, then fine-tune a pretrained convolutional backbone. Track class imbalance, augmentation, learning curves, per-defect recall, calibration, inference latency, and examples of missed defects. A high aggregate accuracy is insufficient when rare safety-critical defects dominate cost.

Turn this scenario into a decision record containing the user, action, input unit, data source, baseline, expensive errors, metric, threshold or ranking policy, latency and privacy constraints, fallback, owner, and rollback condition. Then explain which evidence would cause you to reject the technique even if the demonstration works.

## Main functions

### `devices`
A core part of PyTorch tensors and autograd; study its mechanism, inputs, outputs, parameters, evidence, and failure modes. Check accepted input shapes and types, return values, defaults, and fitted attributes before using it in a larger workflow.

### `shapes`
A core part of PyTorch tensors and autograd; study its mechanism, inputs, outputs, parameters, evidence, and failure modes. Check accepted input shapes and types, return values, defaults, and fitted attributes before using it in a larger workflow.

### `computation graphs`
A core part of PyTorch tensors and autograd; study its mechanism, inputs, outputs, parameters, evidence, and failure modes. Check accepted input shapes and types, return values, defaults, and fitted attributes before using it in a larger workflow.

### `gradients`
A core part of PyTorch tensors and autograd; study its mechanism, inputs, outputs, parameters, evidence, and failure modes. Check accepted input shapes and types, return values, defaults, and fitted attributes before using it in a larger workflow.

### `debugging`
A core part of PyTorch tensors and autograd; study its mechanism, inputs, outputs, parameters, evidence, and failure modes. Check accepted input shapes and types, return values, defaults, and fitted attributes before using it in a larger workflow.

## Function-by-function learning sequence

1. **`devices`**: A core part of PyTorch tensors and autograd; study its mechanism, inputs, outputs, parameters, evidence, and failure modes.
2. **`shapes`**: A core part of PyTorch tensors and autograd; study its mechanism, inputs, outputs, parameters, evidence, and failure modes.
3. **`computation graphs`**: A core part of PyTorch tensors and autograd; study its mechanism, inputs, outputs, parameters, evidence, and failure modes.
4. **`gradients`**: A core part of PyTorch tensors and autograd; study its mechanism, inputs, outputs, parameters, evidence, and failure modes.
5. **`debugging`**: A core part of PyTorch tensors and autograd; study its mechanism, inputs, outputs, parameters, evidence, and failure modes.

For each item, create a tiny input and record: the exact Python type, dimensions, valid range, returned type, whether state changes, one important parameter, one failure case, and one realistic use. Then connect the functions in the same order used by the notebook. This process turns API names into a working mental model.

## Function contract checklist

For every function used in the notebook, document:

- Positional and keyword parameters, including defaults.
- Accepted scalar, vector, matrix, table, sequence, or mapping inputs.
- Expected dimensions and whether broadcasting occurs.
- Return type, dimensions, ordering, and units.
- Learned attributes ending in an underscore in scikit-learn-style APIs.
- Mutation behavior: in-place change versus returned copy.
- Exceptions, warnings, and behavior on empty or missing input.
- Numerical stability and dtype behavior.
- Runtime and memory implications at realistic scale.

## Inputs, outputs, and assumptions

- **Inputs:** Confirm dimensions, dtypes, units, missing values, provenance, and whether rows are independent.
- **Outputs:** Distinguish transformed data, labels, probabilities, scores, generated text, and learned parameters.
- **State:** Methods named `fit` learn state; `transform`, `predict`, and retrieval reuse it.
- **Randomness:** Set `random_state` or use a seeded generator for reproducible comparisons.
- **Assumptions:** Check distributional, independence, scale, stationarity, relevance, and access-control assumptions.
- **Availability:** Confirm every training feature or retrieved field exists at real inference time.

## Decision guide

Use this topic when its input assumptions hold, its output directly supports the decision, and evaluation shows meaningful improvement over a simpler baseline. Avoid it when the dataset is too small for stable estimation, the required input will not exist during real use, the objective does not match actual cost, or a deterministic rule can solve the problem more transparently.

Before selecting it, answer:

- What simpler method was tested?
- Which error is most costly?
- What sample, corpus, or traffic size supports the conclusion?
- Which assumption is least certain?
- What must happen when confidence is low?
- Who reviews harmful or high-impact outcomes?

## Complexity and scaling

Estimate cost in terms of observations $n$, features $d$, classes $c$, tokens $t$, or indexed vectors $v$. Ask whether the method scans all items, forms a dense $n \times d$ matrix, stores pairwise similarities, or performs repeated model calls. Benchmark representative sizes rather than extrapolating from tiny examples. Use batching, sparse representations, approximate search, caching, and incremental updates only after measurement identifies a bottleneck.

## Interpretation

Do not stop at a single score. Compare with a baseline, examine errors by meaningful segment, inspect confidence or ranking behavior, and study examples where the method fails. A technically correct output may still be operationally useless if the metric ignores cost, fairness, latency, freshness, explainability, citation quality, or human workflow.

Separate three claims:

- **Observed:** directly measured on specified data.
- **Inferred:** supported by analysis but dependent on assumptions.
- **Unknown:** not tested or not identifiable from current evidence.

## Evaluation design

Choose the metric after defining error costs. Use held-out data only for final estimates. Use cross-validation for model selection when observations are exchangeable; use chronological or grouped splits when they are not. Report central tendency and variability across folds, seeds, or bootstrap samples. For RAG and LLM systems, evaluate retrieval and generation separately, preserve human-reviewed examples, and record model and prompt versions.

## Error analysis

Create an error table containing identifiers, expected result, actual result, confidence or score, important features, source evidence, and an error category. Group failures into data quality, representation, model capacity, threshold, retrieval, prompt, unsupported claim, distribution shift, and system errors. Fix the dominant cause rather than tuning a metric blindly.

## Four-layer failure model

1. **Input failure:** malformed, stale, biased, mislabeled, unauthorized, or out-of-domain information.
2. **Method failure:** wrong assumptions, representation, objective, capacity, optimization, retrieval, or parameters.
3. **Evaluation failure:** leakage, weak references, wrong metric, hidden subgroup cost, or unsupported causal claims.
4. **System failure:** schema, dependency, service, permissions, index, timeout, logging, or monitoring defects.

Assign every important failure to a primary layer and collect evidence before changing the implementation. A model adjustment cannot repair a broken label definition, and a prompt adjustment cannot repair missing authorized evidence.

## Common pitfalls

- Data leakage from fitting before splitting.
- Silent dtype coercion or inconsistent category handling.
- Evaluating on observations used for learning or tuning.
- Using accuracy for severe class imbalance.
- Over-interpreting small datasets, coefficients, importance, or attention.
- Ignoring version-specific API behavior and warnings.
- Saving a model without preprocessing, metadata, or corpus version.
- Allowing untrusted text to override system instructions or access controls.
- Logging sensitive input without minimization and retention controls.
- Deploying without a fallback, owner, alert threshold, or rollback plan.

## Debugging playbook

1. Reproduce from a clean kernel or process.
2. Reduce to the smallest failing input.
3. Inspect types, shapes, ranges, schemas, and nulls.
4. Confirm the installed function signature and dependency version.
5. Separate data preparation, core operation, and evaluation.
6. Form one falsifiable hypothesis and run one discriminating check.
7. Add a regression test after finding the root cause.
8. Record whether the failure belongs to data, code, model, retrieval, prompt, permissions, infrastructure, or monitoring.

## Testing strategy

### Unit tests

Test pure transformations, validators, scoring functions, chunkers, parsers, and metric calculations with known examples. Include empty, one-row, missing, extreme, malformed, and duplicate inputs.

### Integration tests

Verify preprocessing, model or retriever, postprocessing, persistence, and schemas work together from raw input to final output. Confirm artifacts reload consistently.

### Regression tests

Freeze a small representative evaluation set and acceptable metric ranges. Investigate changes instead of updating expected values automatically.

### System tests

Measure latency, throughput, memory, cost, concurrency, timeout, retry, permission, and fallback behavior under realistic load.

## Production architecture

A production path usually includes input validation, feature or document preparation, a versioned model or index, postprocessing, output validation, structured logging, metrics, and a fallback. Use immutable artifact identifiers. Keep secrets outside notebooks and source control. Restrict tools and retrieval results by user authorization before content reaches a model.

## Monitoring plan

Track data quality, schema violations, missingness, category changes, feature drift, prediction or retrieval distributions, quality on delayed labels, latency percentiles, failure rates, token or compute cost, user feedback, and safety incidents. Set thresholds from historical behavior and operational tolerance. Every alert needs an owner and a response playbook.

## Responsible use

Document intended users, excluded uses, affected groups, known limitations, source licenses, privacy constraints, and human oversight. Group metric differences require context and do not alone prove fairness or unfairness. Generated or retrieved output must not bypass authorization. High-impact decisions need stronger validation, transparency, appeal, and human review.

## Practice checklist

- [ ] Run every notebook cell in order from a clean kernel.
- [ ] Explain every output and important parameter in your own words.
- [ ] Change one parameter and predict the effect first.
- [ ] Add realistic boundary and invalid cases.
- [ ] Write unit assertions and an integration check.
- [ ] Compare against a baseline.
- [ ] Add an error analysis table.
- [ ] Measure one scaling dimension.
- [ ] Record limitations and appropriate uses.
- [ ] Write monitoring and rollback notes.

## Extended exercises

1. Replace generated data with a small public or workplace-safe dataset.
2. Wrap the main operation in a typed function with a complete docstring.
3. Add validation that rejects malformed input with a useful error.
4. Compare at least three parameter settings in a table.
5. Add a diagnostic revealing information hidden by the aggregate metric.
6. Repeat evaluation across seeds, folds, or bootstrap samples.
7. Build a baseline and state exactly when the advanced method wins.
8. Persist and reload any learned state; verify identical predictions.
9. Simulate drift or an unseen input and observe behavior.
10. Write a production monitoring and incident-response checklist.

## Capstone specification

Build a reproducible project applying this topic to a real question. Include a problem statement, data contract, ethics and privacy review, EDA, baseline, implementation, parameter experiment, evaluation, error analysis, tests, saved artifacts, monitoring plan, and executive summary. The notebook must run from a clean kernel without hidden state.

### Suggested rubric

| Area | Weight | Evidence |
|---|---:|---|
| Problem framing | 10% | Decision, users, constraints, and success criteria |
| Data quality | 15% | Schema, provenance, leakage checks, and validation |
| Technical correctness | 20% | Correct implementation and explained functions |
| Evaluation | 20% | Baseline, metrics, uncertainty, and error analysis |
| Engineering | 15% | Reproducibility, tests, artifacts, and runtime |
| Responsible use | 10% | Risks, privacy, fairness, and safeguards |
| Communication | 10% | Clear narrative, visuals, and limitations |

## Review and interview questions

1. What does the method learn, if anything?
2. What input assumptions affect correctness?
3. Which function performs fitting, transformation, prediction, retrieval, or scoring?
4. How can leakage occur in this workflow?
5. Which baseline and metric provide a fair comparison?
6. What new behavior can appear on unseen data?
7. Which parameter controls complexity most directly?
8. How do runtime and memory scale?
9. How would you detect underfitting and overfitting?
10. How would you test deterministic and stochastic behavior?
11. Which errors are most expensive?
12. How would you choose a decision threshold?
13. What information must be versioned for reproduction?
14. How would you investigate a production quality drop?
15. What sensitive information could enter logs or artifacts?
16. How could this method disadvantage a user group?
17. Which access controls belong before inference or retrieval?
18. What is the fallback if the component is unavailable?
19. When is a simpler method preferable?
20. Which claim cannot be made from the current evidence?

## Advanced written and oral assessment

1. Draw the complete state-transition diagram for **PyTorch tensors and autograd**.
2. Derive or explain the core operation without naming a library.
3. Compare the chosen approach with two alternatives using explicit constraints.
4. Construct a minimal counterexample where the method gives a misleading result.
5. Design a split or evaluation set matching real deployment.
6. Explain how one parameter changes bias, variance, recall, precision, latency, or cost.
7. Propose a test suite covering contract, edge, regression, and system behavior.
8. Diagnose a hypothetical production degradation using the four-layer failure model.
9. Define an abstention, fallback, or human-review policy.
10. State one claim the current evidence supports and one it cannot support.

A strong answer connects theory, implementation, evidence, and operations. A weak answer lists API names without explaining the information flow or assumptions.

## Connections and next steps

Move from mechanics to a framework such as PyTorch or TensorFlow, while retaining the same shape, loss, validation, and monitoring discipline.

Return to earlier notebooks if input preparation, statistics, evaluation, or programming contracts are unclear. Continue to later notebooks when you can explain the complete lifecycle, reproduce the example, solve a modified problem, and identify at least one case where the method should not be used. The objective is transferable understanding across libraries and domains.

## Further reading

Use official Python, NumPy, pandas, Matplotlib, SciPy, scikit-learn, and Jupyter documentation for exact signatures and version behavior. For DL, LLM, agent, and RAG systems, consult the selected framework, model provider, vector database, evaluation tool, and deployment platform documentation because APIs, limits, safety controls, and model behavior change quickly. Prefer primary documentation and papers over copied snippets.
