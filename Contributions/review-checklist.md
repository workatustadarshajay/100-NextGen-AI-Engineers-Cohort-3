# Contribution Review Checklist

## Curriculum fit

- [ ] The topic fills a real gap and does not duplicate an existing lesson.
- [ ] Scope is one primary concept with explicit exclusions.
- [ ] Prerequisites and next topics are correct.
- [ ] Track and numbering are consistent.

## Learning quality

- [ ] The opening explains the topic in plain language.
- [ ] Internal mechanics are explained independently of a library.
- [ ] Inputs, outputs, state, and assumptions are explicit.
- [ ] At least two examples progress from minimal to realistic.
- [ ] Outputs are interpreted rather than merely printed.
- [ ] Exercises progress from observation to independent critique.
- [ ] A learner can demonstrate mastery through assertions, decisions, or artifacts.

## Technical correctness

- [ ] Code syntax and notebook JSON validate.
- [ ] Examples run top to bottom from a clean kernel.
- [ ] Randomness is seeded where relevant.
- [ ] Equations, terminology, and claims are correct.
- [ ] API behavior matches supported dependency versions.
- [ ] No hidden network, credential, GPU, or private-data requirement exists.
- [ ] Normal, boundary, invalid, and out-of-domain inputs are considered.

## Evaluation

- [ ] A meaningful baseline is included.
- [ ] Metrics match the decision and error costs.
- [ ] Leakage and split strategy are addressed.
- [ ] Uncertainty, seed sensitivity, or repeatability is considered.
- [ ] Error analysis includes meaningful examples or slices.
- [ ] Unsupported causal or universal claims are avoided.

## Alternatives and depth

- [ ] At least two credible alternatives are compared.
- [ ] Selection criteria include quality and operational constraints.
- [ ] Complexity and scaling behavior are explained.
- [ ] Four-layer failures cover input, method, evaluation, and system.
- [ ] Debugging checks target root causes.

## Engineering and operations

- [ ] Required artifacts and configuration are versioned.
- [ ] Unit, integration, regression, and system tests are described.
- [ ] Latency, throughput, memory, or cost is addressed where relevant.
- [ ] Monitoring signals and alert ownership are defined.
- [ ] Fallback, abstention, escalation, and rollback are explicit.

## Security and responsible use

- [ ] Sensitive data collection and logging are minimized.
- [ ] Authorization occurs before retrieval, inference, or action where required.
- [ ] Untrusted input and prompt-injection risks are addressed.
- [ ] Affected groups and harmful outcomes are considered.
- [ ] Human oversight is proportional to impact.
- [ ] Licenses and source provenance are acceptable.

## Documentation and maintainability

- [ ] Notebook and guide basenames match.
- [ ] The curriculum index and learning routes remain accurate.
- [ ] Durable changes are in the generator, not generated files only.
- [ ] The proposal records owner, reviewer, and status.
- [ ] `node validate_practice_library.js` passes.
