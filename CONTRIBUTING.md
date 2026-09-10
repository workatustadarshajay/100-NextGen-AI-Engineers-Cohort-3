# Contributing to the AI Practice Library

This library is designed to grow through focused, reviewable topic contributions. A contribution may add a notebook and guide, deepen an existing topic, add an exercise or case study, improve validation, or propose a new learning route.

## Start here

1. Read [PRACTICE_README.md](PRACTICE_README.md) to understand the current tracks.
2. Review [TOPIC_IDEAS.md](TOPIC_IDEAS.md) and select an unclaimed idea or propose a new one.
3. Copy [Contributions/topic-proposal-template.md](Contributions/topic-proposal-template.md) into `Contributions/Proposals/` and complete it.
4. Discuss overlap, prerequisites, scope, and expected learning evidence before implementation.
5. Add the topic to `generate_practice_library.js`; generated notebooks and guides should not become independent sources of truth.
6. Run `node generate_practice_library.js` only after preserving any intentional manual notebook changes.
7. Run `node validate_practice_library.js` and resolve all reported failures.
8. Review the generated notebook from a clean-kernel learner perspective.

## Important ownership rule

The generator is the canonical source for generated lessons. Direct notebook or guide edits can be overwritten the next time the generator runs. When improving a generated lesson, update its topic definition, track context, or shared template in `generate_practice_library.js`.

Before regeneration, check whether a learner has manually edited or executed notebooks. Move durable improvements into the generator and avoid overwriting unreviewed work.

## What makes a useful topic

A strong topic has one primary learning objective, explicit prerequisites, a realistic use case, executable examples, meaningful alternatives, and observable evidence of mastery. It should answer:

- What problem does this solve?
- What must the learner already understand?
- What information enters and leaves the method?
- What state is learned, constructed, or persisted?
- Which functions and parameters matter most?
- What baseline or alternative should be compared?
- How does the method fail?
- How is success evaluated?
- What changes at production scale?
- What should a learner build independently?

## Topic scope

Prefer one concept per notebook. Split a proposal when it contains multiple independently teachable mechanisms. For example, feature stores, model registries, and deployment strategies should be separate lessons even though all belong to MLOps.

Avoid creating a new notebook when the proposed material is only:

- Another dataset using the same method
- A vendor-specific rewrite without a new concept
- A list of links without an executable learning path
- A large end-to-end project with no isolated mechanism
- Repeated generic content already covered by the shared template

## Required notebook content

Every new notebook should include:

- Title, track, goal, prerequisites, and learning lifecycle
- Plain-language concept explanation
- Internal mechanics or algorithm steps
- Mathematical or computational view where relevant
- Inputs, outputs, assumptions, and pre-flight checks
- At least two executable examples
- A worked-example walkthrough
- Function-by-function API reference
- Important parameter experiments
- Alternatives and selection tradeoffs
- Realistic decision scenario
- Evaluation and error analysis
- Four-layer failure analysis
- Debugging procedure
- Production and responsible-use notes
- Progressive exercises and a learner practice cell
- Mini-project definition of done
- Knowledge checks, glossary, and next steps

## Required guide content

The matching guide must stand alone without requiring the reader to open the notebook. It should contain deeper theory, notation, function contracts, method selection, scaling, testing, monitoring, production architecture, risks, review questions, and a capstone rubric.

## Code standards

- Use deterministic local examples whenever practical.
- Set random seeds explicitly.
- Avoid mandatory paid APIs, credentials, network calls, or GPU access.
- Prefer generated, built-in, or clearly redistributable data.
- Keep secrets and personal data out of notebooks and outputs.
- Explain non-obvious functions, parameters, types, dimensions, and return values.
- Include normal, boundary, invalid, and out-of-domain cases.
- Use assertions for expected behavior.
- Do not hide warnings without explaining their cause.
- Keep code cells independently understandable while preserving top-to-bottom execution.

## Topic definition format

Each topic currently uses this structure in `generate_practice_library.js`:

```javascript
[
  "53",
  "Topic Title",
  "track-key",
  "One precise sentence describing the learning goal.",
  `# First executable Python example`,
  `# Second executable Python example`,
  [
    ["function_or_concept", "Exact responsibility and learning purpose."],
    ["important_parameter", "What it controls and why it matters."],
  ],
]
```

Use an existing track key from `trackContexts` or add a complete new track context with prerequisites, foundation, lifecycle, evidence, next steps, mechanics, alternatives, and scenario.

## Numbering and filenames

- Assign the next unused two-digit number.
- Use a concise title; the generator creates the lowercase hyphenated slug.
- Never reuse a retired number for a different topic.
- Keep notebook and guide basenames identical.
- Update learning routes when prerequisites or sequence change.

## Quality review

Reviewers should evaluate the contribution from four perspectives:

1. **Learner:** Can a motivated beginner follow the prerequisites and examples?
2. **Technical:** Are statements, code, equations, and API behavior correct?
3. **Instructional:** Does each exercise produce evidence of understanding?
4. **Operational:** Are testing, scaling, security, monitoring, and limitations realistic?

Use [Contributions/review-checklist.md](Contributions/review-checklist.md) for the complete review.

## Validation

Run:

```bash
node validate_practice_library.js
```

The validator checks notebook JSON, required metadata, Python syntax, numbering, companion guides, index links, and minimum content structure. Runtime validation additionally requires the packages listed in `Practice/requirements.txt`.

## Contribution types

- **New topic:** Adds one notebook, one guide, and index/route placement.
- **Depth improvement:** Adds topic-specific theory, examples, diagnostics, or exercises.
- **Correction:** Fixes inaccurate explanation, code, equation, or unsafe guidance.
- **Accessibility:** Improves vocabulary, sequencing, alternative explanations, or output interpretation.
- **Evaluation:** Adds stronger metrics, datasets, rubrics, or failure cases.
- **Engineering:** Improves validation, reproducibility, dependency handling, or navigation.

## Proposal acceptance criteria

A proposal is ready when it has a clear audience, narrow objective, prerequisite location, realistic example, comparison method, evaluation plan, failure cases, production connection, and named reviewer. Ideas can remain in the backlog until these details are available.
