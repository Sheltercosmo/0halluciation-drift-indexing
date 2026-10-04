# GitHub description, topics and social cover

Use the following About description. It is also the package summary in `pyproject.toml` and the `description` field in [repository.json](../.github/repository.json).

> Reliable decisions with a purely statistical prior for document topic blocking. Jev-based indexing with zero generative LLM calls and optional embeddings.

The display name is **0halluciation drift indexing** and the repository slug is `0halluciation-drift-indexing`. The headline is **Reliable decisions with a purely statistical prior.** The selling point is topic blocking through Jev decisions and an explicit statistical prior. “0 generative LLM calls” applies to indexing; retrieval can use an LLM. Embeddings remain optional. The original opening introduction is preserved.

## Topics

The source of truth is [topics.json](../.github/topics.json). These 16 topics describe the actual implementation and its intended use:

```text
decision-models, bayesian-statistics, statistical-priors, topic-segmentation, document-chunking, semantic-chunking, jev, document-indexing, rag, embedding-free, python, typesafe-ai, retrieval-augmented-generation, bayesian-inference, extractive-summarization, reranking
```

GitHub permits up to 20 topics, each at most 50 characters, using lowercase letters, numbers and hyphens. Topics help repository discovery and are configured in the repository's **About** panel. [GitHub topic documentation](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/classifying-your-repository-with-topics)

## Cover

Use [github-social-preview.jpg](../assets/github-social-preview.jpg) for **Settings → General → Social preview → Edit → Upload an image**. The opaque ivory cover uses Folio, the original paper-owl mascot, with the repository name and the message “Reliable decisions with a purely statistical prior.” It carries no benchmark or zero-error claims.

The upload export is 1280 × 640 pixels and under 1 MB, matching [GitHub's social preview guidance](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/customizing-your-repositorys-social-media-preview). Committing a cover file or this metadata manifest does not automatically change GitHub's settings.

## Apply the About metadata

The public repository is [Sheltercosmo/0halluciation-drift-indexing](https://github.com/Sheltercosmo/0halluciation-drift-indexing). To reapply the checked-in description and topics, use an authenticated GitHub CLI from the repository root:

```powershell
$repoAbout = Get-Content -Raw .github/repository.json | ConvertFrom-Json
$repoTopics = Get-Content -Raw $repoAbout.topics_file | ConvertFrom-Json
gh repo edit Sheltercosmo/0halluciation-drift-indexing --description $repoAbout.description --add-topic ($repoTopics -join ',')
```

This command adds the selected topics and leaves existing topics in place; review the combined list against GitHub's 20-topic limit. See the [GitHub CLI reference](https://cli.github.com/manual/gh_repo_edit).

The README now explains decision-based topic blocking and the statistical boundary rule near its opening and links to the algorithm, usage and evaluation evidence. Package keywords match the topic manifest. These are discoverability improvements, not a promise of search rankings. Avoid adding unrelated project names, fabricated popularity badges, or unsupported performance claims.
