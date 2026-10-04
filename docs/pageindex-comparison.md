# Relationship to VectifyAI PageIndex

Checked against public documentation on **4 October 2026**.

The systems share a document-tree retrieval pattern: an LLM chooses branches and reads source content. PageIndex's [framework introduction](https://pageindex.ai/blog/pageindex-intro) describes recursive nodes linked to raw data and iterative section selection. That architectural overlap is substantial.

| Dimension | This project | PageIndex, as currently documented |
| --- | --- | --- |
| Retrieval | LLM proposes content, Jev reranks leaves, LLM reads upward | LLM reasons through the tree |
| Structure | Declared headings and contents links, then paragraph topic cuts | Flash derives structure from PDF layout |
| Index-time decisions | Jev same-topic and representativeness judgments | Index model summarizes and refines layout-derived sections |
| Boundary rule | Fixed-anchor comparisons with prior adjustment and probability drops | No equivalent rule described in the sources reviewed |
| Representative text | Exact source sentences, searched from the edges inward | Node summaries alongside titles and source ranges |
| Embeddings | Optional scorer; no vector database required | No vector database required |
| Current artifact scope | Small Markdown/plain-text implementation | Broader PDF-oriented indexing and retrieval system |

Sources: the [current repository README](https://github.com/VectifyAI/PageIndex/blob/main/README.md) describes indexing and retrieval models; the [Flash announcement](https://pageindex.ai/blog/pageindex-flash) explains layout extraction and summary generation.

**Conclusion:** the proposed indexing procedure differs from the documented PageIndex approach, while the retrieval concept overlaps. A documentation comparison does not establish research novelty or prove that no PageIndex implementation or related project uses similar techniques. The defensible contribution to evaluate is Jev-based paragraph segmentation plus extractive representatives, including its quality/cost tradeoffs.
