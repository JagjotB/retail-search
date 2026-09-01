# Experiment log

Model and feature choices were made on the validation split only. The frozen test split was evaluated after the configuration was selected.

| Experiment | Features | Validation NDCG@10 | Relative vs dense baseline |
|---|---|---:|---:|
| embedding-only baseline | dense LSA cosine only | 0.732127 | +0.00% |
| full LambdaMART | dense_similarity, word_title_tfidf_cosine, char_title_tfidf_cosine, word_richtext_tfidf_cosine, bm25_title, query_coverage, title_coverage, token_jaccard, exact_title_match, phrase_match, prefix_match, all_query_tokens_present, brand_match, color_match, numeric_match, query_token_count, title_token_count, token_length_ratio, character_length_ratio, description_coverage, character_similarity, partial_character_similarity, token_set_similarity, cross_encoder_rich_score | 0.822324 | +12.32% |
| ablation: no dense feature | word_title_tfidf_cosine, char_title_tfidf_cosine, word_richtext_tfidf_cosine, bm25_title, query_coverage, title_coverage, token_jaccard, exact_title_match, phrase_match, prefix_match, all_query_tokens_present, brand_match, color_match, numeric_match, query_token_count, title_token_count, token_length_ratio, character_length_ratio, description_coverage, character_similarity, partial_character_similarity, token_set_similarity, cross_encoder_rich_score | 0.822690 | +12.37% |
| ablation: semantic + length only | dense_similarity, query_token_count, title_token_count, token_length_ratio, character_length_ratio | 0.736832 | +0.64% |

Ablations are trained independently. They are diagnostic validation experiments, not alternate test-set attempts.
