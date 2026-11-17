# Request a Codex review

Codex can inspect changes to the heat-equation calculation and conductivity
inference on request. Reviews emphasize the physical model, adjoint derivatives,
likelihoods, priors, and the interpretation of inferred uncertainties.

## Enable

The workflow must be present on `master`, the repository's default branch.
Add an OpenAI API key as the repository secret `OPENAI_API_KEY` in
[Settings → Secrets and variables → Actions](https://github.com/MinhMPA/steady-2d-heat-inverse/settings/secrets/actions).
The action uses OpenAI API billing. A ChatGPT login alone does not configure it.

## Review a pull request

Write a new comment in the PR conversation:

```text
/codex review
```

You can supply a scientific focus:

```text
/codex review Check whether the adjoint gradient differentiates the same posterior used by HMC.
```

The requester needs write, maintain, or admin access. Editing an existing comment
does not start a review. Codex posts its findings as a new PR conversation comment
and records the exact head and base revisions examined. A subsequent push requires
a new request. The workflow can also inspect a fork PR when a maintainer requests it.

## Review code on a branch or at a commit

Open [Actions → Codex review](https://github.com/MinhMPA/steady-2d-heat-inverse/actions/workflows/codex-review.yml)
and choose **Run workflow**. Keep **Use workflow from** set to `master`.

- To review a PR, enter its **pr_number**; **ref** is ignored.
- To review code without a PR, leave **pr_number** empty and set **ref** to a
  branch, tag, or commit. Leaving it empty selects the current default branch.
- Use **focus** to specify a physical question or source files.

For a code review without a PR, read the report in the completed run's summary.
For a PR review, it also appears in the PR conversation.

From the GitHub command line, for example:

```bash
gh workflow run codex-review.yml --repo MinhMPA/steady-2d-heat-inverse --ref master \
  -f pr_number=2 -f focus='Check the discrete adjoint and posterior definition.'

gh workflow run codex-review.yml --repo MinhMPA/steady-2d-heat-inverse --ref master \
  -f ref=master -f focus='Check heat-equation signs and boundary conditions.'
```

## Interpretation and operation

This is source inspection. The review job does not install the scientific
environment or run numerical tests. Existing gradient, solver, and sampling
checks remain the evidence for numerical correctness; a clean Codex report
does not establish convergence or calibrated uncertainties.

Reviews run only when requested. The action uses its default model and a
30-minute job limit. The reviewed files are read-only, Git credentials are not
retained in the checkout, and a separate job publishes feedback. Action versions
are pinned to commit identifiers. The API key is passed only to its presence
check and the Codex action.

If a run reports a missing key, add `OPENAI_API_KEY` and request another review.
For API authentication, quota, or model-access failures, inspect the review job
log and the OpenAI API account. If no run appears after a comment, check that the
workflow is on `master`, the comment begins with `/codex review`, and you have
write access.

`/codex review` belongs to this workflow. The separate Codex GitHub integration
uses `@codex review`; enabling this workflow does not configure that integration.

[Claude reviews](CLAUDE_REVIEW.md) are also available through `/claude review`
or `@claude review`, using a separate Anthropic credential.

References: [OpenAI Codex action](https://github.com/openai/codex-action) and
[action security guidance](https://github.com/openai/codex-action/blob/main/docs/security.md).
