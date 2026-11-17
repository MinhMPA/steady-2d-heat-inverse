# Request a Claude review

Claude can inspect the heat-equation calculation and conductivity inference on
request. Reviews emphasize physical and statistical consequences: boundary
conditions, adjoint gradients, likelihoods, priors, and inferred uncertainties.

## Enable

The workflow must be present on `master`, the repository's default branch.
Add either credential in
[Settings → Secrets and variables → Actions](https://github.com/MinhMPA/steady-2d-heat-inverse/settings/secrets/actions):

- `ANTHROPIC_API_KEY`: an Anthropic API key, using API billing.
- `CLAUDE_CODE_OAUTH_TOKEN`: a Claude Code OAuth token. Eligible Claude Pro or Max
  subscribers can generate it locally with `claude setup-token`, then paste the
  resulting token directly into the GitHub secret form.

If both secrets exist, this workflow uses the API key. Do not paste credentials
into PRs, issues, or review requests. This workflow supplies GitHub's built-in
job token to the action, so it does not require a separate Claude GitHub App
installation or a personal GitHub token.

## Review a pull request

Write a new comment in the PR conversation:

```text
/claude review
```

`@claude review` works too. Either command can include a scientific focus:

```text
@claude review Check whether the adjoint gradient differentiates the same posterior used by HMC.
```

You need write, maintain, or admin access. Editing an existing comment does not
start a review. Claude posts its report as a PR conversation comment, identifying
the exact head and base revisions examined. A new push requires a new request.
Maintainers can also request reviews of fork PRs.

## Review code on a branch or at a commit

Open [Actions → Claude review](https://github.com/MinhMPA/steady-2d-heat-inverse/actions/workflows/claude-review.yml)
and choose **Run workflow**. Keep **Use workflow from** set to `master`.

- To review a PR, enter **pr_number**; **ref** is ignored.
- To review code without a PR, leave **pr_number** empty and enter a branch, tag,
  or commit in **ref**. Leaving it empty selects the current default branch.
- Use **focus** to specify a physical question or source files.

Read the report in the run's summary. PR reviews also appear in the PR conversation.

```bash
gh workflow run claude-review.yml --repo MinhMPA/steady-2d-heat-inverse --ref master \
  -f pr_number=2 -f focus='Check the discrete adjoint and posterior definition.'
```

## Interpretation and operation

The model has file-reading and search tools. The workflow prepares the PR diff
before Claude starts, and a separate job publishes the resulting report. Claude
does not run the numerical tests or change the calculation. The existing gradient,
solver, and sampling checks remain the numerical evidence; source inspection alone
does not establish convergence or calibrated uncertainties.

Reviews run only when requested, with a 30-minute job limit and at most 40 model
turns. The action is pinned to a specific revision, uses restricted execution,
and disables automatic discovery of project hooks, plugins, and external tools.
The review job's GitHub token has read access. Only the separate publication job
can comment on PRs.

If a run reports missing authentication, add one of the secrets above and request
another review. For authentication or quota failures, inspect the review job log
and your Anthropic account. If no run appears, check that the workflow is on
`master`, the command starts a new PR comment, and you have write access.

[Codex reviews](CODEX_REVIEW.md) remain available independently through `/codex review`.

References: [Anthropic setup guide](https://github.com/anthropics/claude-code-action/blob/main/docs/setup.md),
[provided GitHub token handling](https://github.com/anthropics/claude-code-action/blob/cc98b3ae241b0fcfb6a42a2690751afffecb4452/src/github/token.ts),
and [Claude command options](https://code.claude.com/docs/en/cli-reference).
