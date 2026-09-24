mod tetes {
    use anyhow::{bail, Context, Result};
    use candle_core::{Device, Tensor};
    use qllama::context::params::LlamaPoolingType;
    use qllama::model::AddBos;
    use qllama_bench::split_data::QuerySummaries;
    use qllama_bench::{
        build_qwen3_reranker_prompts, build_simple_reranker_prompts, ensure_hf_model_file,
        init_backend, init_context, init_model, init_model_multi, init_reranker_context,
        load_query_summaries, process_splits_batch, rerank_qwen3, rerank_token_batches,
        SentenceScore,
    };
    use serial_test::serial;
    use std::fs;

    pub fn normalize_l2(ts: &Tensor) -> anyhow::Result<Tensor> {
        Ok(ts.broadcast_div(&ts.sqr()?.sum_keepdim(1)?.sqrt()?)?)
    }
    pub fn similarity_matrix(
        embeds1: &Tensor,
        embeds2: &Tensor,
        normalized: bool,
    ) -> anyhow::Result<Tensor> {
        if normalized {
            return Ok(embeds1.matmul(&embeds2.transpose(0, 1)?)?);
        }
        let embeds1_normed = normalize_l2(embeds1)?;
        let embeds2_normed = normalize_l2(embeds2)?;
        Ok(embeds1_normed.matmul(&embeds2_normed.transpose(0, 1)?)?)
    }

    #[test]
    #[serial(model)]
    fn test() -> Result<()> {
        let model_path = ensure_hf_model_file(
            "sabafallah/embeddinggemma-300m-sentence-transformers-gguf",
            "embeddinggemma-300m-sentence-transformers-q8_0.gguf",
            None,
        )?;

        let n_ctx = 2048;

        let backend = init_backend(true)?;
        let models = init_model_multi(&model_path, &backend, 4);
        let model = models.get(0).unwrap().as_ref().unwrap();

        //let mut ctx = init_context(&model, &backend, Some(3072), Some(3072), Some(3072))?;
        let mut ctx = init_context(model, &backend, Some(n_ctx), Some(n_ctx), Some(n_ctx))?;
        //let mut ctx = init_context(&model, &backend, None, None, None)?;

        let sentences1 = [
            "The new movie is awesome",
            "The cat sits outside",
            "A man is playing guitar",
            "I love pasta",
        ];
        let sentences1 = sentences1.map(|s| s.to_string()).to_vec();
        let gemma_prompt = "task: sentence similarity | query: ";

        println!("----------------------------------");
        for s in sentences1.iter() {
            println!("{}", s);
        }
        println!("----------------------------------");

        let sentences1_prompts = sentences1
            .iter()
            .map(|s| gemma_prompt.to_string() + s)
            .collect::<Vec<String>>();

        for prompt in sentences1_prompts.iter() {
            println!("{}", prompt);
        }

        println!("----------------------------------");

        let embeddings1 = process_splits_batch(model, &mut ctx, &sentences1_prompts)?;

        let embeddings1_ts = Tensor::new(embeddings1, &Device::Cpu)?;

        let sentences2 = [
            "The dog plays in the garden",
            "The new movie is so great",
            "A woman watches TV",
            "Do you like pizza?",
        ];

        let sentences2 = sentences2.map(|s| s.to_string()).to_vec();
        println!("----------------------------------");
        for s in sentences2.iter() {
            println!("{}", s);
        }

        let sentences2_prompts = sentences2
            .iter()
            .map(|s| gemma_prompt.to_string() + s)
            .collect::<Vec<String>>();

        for prompt in sentences2_prompts.iter() {
            println!("{}", prompt);
        }
        println!("----------------------------------");
        let embeddings2 = process_splits_batch(model, &mut ctx, &sentences2_prompts)?;

        let embeddings2_ts = Tensor::new(embeddings2, &Device::Cpu)?;

        let similarities = similarity_matrix(&embeddings1_ts, &embeddings2_ts, true)?;
        let similarities_vec: Vec<Vec<f32>> = similarities
            .to_vec2::<f32>()
            .with_context(|| "failed to convert similarities to vec")?;
        for (idx_i, sentence1) in sentences1.iter().enumerate() {
            println!("{}:", sentence1);
            let mut sts_similarities = Vec::new();
            for (idx_j, sentence2) in sentences2.iter().enumerate() {
                let score = similarities_vec.get(idx_i).unwrap().get(idx_j).unwrap();
                sts_similarities.push(SentenceScore {
                    sentence: sentence2.to_string(),
                    score: *score as f64,
                });
            }
            sts_similarities.sort();
            sts_similarities.reverse();
            for sts in sts_similarities.iter() {
                println!("\t {:?}", sts);
            }
        }
        Ok(())
    }

    #[test]
    fn query_summaries_data() -> Result<()> {
        let json_file_path = "tests/test_data/bert_paper_query_summaries.json";
        let input_str = fs::read_to_string(json_file_path)?;
        assert!(!input_str.is_empty());
        let query_summaries = serde_json::from_str::<QuerySummaries>(&input_str);
        assert!(query_summaries.is_ok());
        let query_summaries = query_summaries?;
        assert!(!query_summaries.query.is_empty());
        assert!(!query_summaries.summaries.is_empty());
        assert_eq!(query_summaries.summaries.len(), 20);
        Ok(())
    }

    #[test]
    #[serial(model)]
    fn bge_reranker_test() -> Result<()> {
        let query_summaries =
            load_query_summaries("tests/test_data/bert_paper_query_summaries.json")?;

        let model_path = ensure_hf_model_file(
            "sabafallah/bge-reranker-v2-m3-Q4_K_M-GGUF",
            "bge-reranker-v2-m3-Q4_K_M.gguf",
            None,
        )?;
        let backend = init_backend(true)?;
        let model = init_model(&model_path, &backend)?;
        let max_tokens = 2048;
        let pooling_type = Some(LlamaPoolingType::Rank);
        let mut ctx = init_reranker_context(&model, &backend, max_tokens, pooling_type)?;

        let prompt_lines =
            build_simple_reranker_prompts(&query_summaries.query, &query_summaries.summaries);

        // tokenize the prompt
        let tokens_lines_list = prompt_lines
            .iter()
            .map(|line| model.str_to_token(line, AddBos::Never))
            .collect::<Result<Vec<_>, _>>()
            .with_context(|| format!("failed to tokenize {:?}", prompt_lines))?;

        let n_ctx = ctx.n_ctx() as usize;

        if tokens_lines_list.iter().any(|tok| n_ctx < tok.len()) {
            bail!("One of the provided prompts exceeds the size of the context window");
        }

        let output = rerank_token_batches(
            &mut ctx,
            &tokens_lines_list,
            max_tokens as usize,
            true,
            "rank",
        )?;

        let scores = output
            .iter()
            .map(|embeddings| embeddings[0])
            .collect::<Vec<f32>>();
        let mut scores = scores.iter().enumerate().collect::<Vec<(usize, &f32)>>();
        // sort by score
        scores.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
        for (idx, score) in scores.iter() {
            println!("--------------- {} ---------------", idx);
            println!("score: {}", score);
            let summary = query_summaries.summaries.get(*idx).unwrap();
            println!("summary: {}", summary);
        }

        Ok(())
    }

    #[test]
    #[serial(model)]
    fn bge_reranker_simple_test() -> Result<()> {
        let model_path = ensure_hf_model_file(
            "sabafallah/bge-reranker-v2-m3-Q4_K_M-GGUF",
            "bge-reranker-v2-m3-q4_k_m.gguf",
            None,
        )?;
        let backend = init_backend(true)?;
        let model = init_model(&model_path, &backend)?;
        let max_tokens = 2048;
        let pooling_type = Some(LlamaPoolingType::Rank);
        let mut ctx = init_reranker_context(&model, &backend, max_tokens, pooling_type)?;

        let query = "What is machine learning?";
        let documents = vec![
            "Angela Merkel was the Chancellor of Germany",
            "Pizza is made with tomatoes and cheese",
            "Deep learning uses neural networks...",
            "The weather today is sunny and warm",
            "Machine learning is a subset of artificial intelligence",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<String>>();

        let prompt_lines = build_simple_reranker_prompts(query, &documents);

        // tokenize the prompt
        let tokens_lines_list = prompt_lines
            .iter()
            .map(|line| model.str_to_token(line, AddBos::Never))
            .collect::<Result<Vec<_>, _>>()
            .with_context(|| format!("failed to tokenize {:?}", prompt_lines))?;

        let n_ctx = ctx.n_ctx() as usize;

        if tokens_lines_list.iter().any(|tok| n_ctx < tok.len()) {
            bail!("One of the provided prompts exceeds the size of the context window");
        }

        let output = rerank_token_batches(
            &mut ctx,
            &tokens_lines_list,
            max_tokens as usize,
            true,
            "rank",
        )?;

        let scores = output
            .iter()
            .map(|embeddings| embeddings[0])
            .collect::<Vec<f32>>();
        let mut scores = scores.iter().enumerate().collect::<Vec<(usize, &f32)>>();
        // sort by score
        scores.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
        for (idx, score) in scores.iter() {
            println!("--------------- {} ---------------", idx);
            println!("score: {}", score);
            let doc = documents.get(*idx).unwrap();
            println!("doc: {}", doc);
        }

        Ok(())
    }

    #[test]
    #[serial(model)]
    fn qwen_reranker_test() -> Result<()> {
        let model_path = ensure_hf_model_file(
            "sabafallah/Qwen3-Reranker-0.6B-Q8_0-GGUF",
            "qwen3-reranker-0.6b-q8_0.gguf",
            None,
        )?;
        let backend = init_backend(true)?;
        let model = init_model(&model_path, &backend)?;
        let max_tokens = 8192;
        let mut ctx =
            init_reranker_context(&model, &backend, max_tokens, Some(LlamaPoolingType::Rank))?;

        let documents = [
            "Angela Merkel was the Chancellor of Germany",
            "Pizza is made with tomatoes and cheese",
            "Deep learning uses neural networks...",
            "The weather today is sunny and warm",
            "Machine learning is a subset of artificial intelligence",
        ];
        let query = "What is machine learning?";

        let tokens_lines_list = build_qwen3_reranker_prompts(query, &documents)
            .iter()
            .map(|prompt| model.str_to_token(prompt, AddBos::Never))
            .collect::<Result<Vec<_>, _>>()?;

        let n_ctx = ctx.n_ctx() as usize;
        if tokens_lines_list.iter().any(|tok| n_ctx < tok.len()) {
            bail!("One of the provided prompts exceeds the size of the context window");
        }

        let scores = rerank_qwen3(&mut ctx, &tokens_lines_list, max_tokens as usize)?;
        assert_eq!(scores.len(), documents.len());

        let mut ranked: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        for (idx, score) in &ranked {
            println!("{score:.4}  {}", documents[*idx]);
        }

        // Scores are P(yes), so they are probabilities.
        assert!(
            scores.iter().all(|s| (0.0..=1.0).contains(s)),
            "scores: {scores:?}"
        );
        // The machine-learning definition ranks first, the deep-learning line second.
        assert_eq!(ranked[0].0, 4, "ranking: {ranked:?}");
        assert_eq!(ranked[1].0, 2, "ranking: {ranked:?}");
        // The three unrelated documents are judged irrelevant.
        for idx in [0, 1, 3] {
            assert!(scores[idx] < 0.1, "document {idx} scored {}", scores[idx]);
        }
        assert!(
            scores[4] > 0.9,
            "machine-learning definition scored {}",
            scores[4]
        );

        Ok(())
    }

    #[test]
    fn raw_pointer_test() -> Result<()> {
        // Example array
        let arr = [1, 2, 3, 4, 5];

        // Get the raw pointer to the start of the array
        let ptr = arr.as_ptr();

        // Calculate the length of the slice
        let len = arr.len();

        // Define the range i..j
        let i = 1;
        let j = 4;

        // Ensure the range is within the bounds of the array
        assert!(i < j && j <= len);

        // Create the slice
        let slice = unsafe { std::slice::from_raw_parts(ptr.add(i), j - i) };

        // Print the slice
        println!("{:?}", slice);
        Ok(())
    }
}
