mod tetes {
    use anyhow::Result;
    use fast_text_splitter::config::SplitterLiteConfig;
    use fast_text_splitter::hf_tokenizer::HFTokenizer;
    use candle_core::{Device, Tensor};
    use llama_cpp_2::context::params::LlamaContextParams;
    use llama_cpp_2::context::LlamaContext;
    use llama_cpp_2::llama_backend::LlamaBackend;
    use llama_cpp_2::model::params::LlamaModelParams;
    use llama_cpp_2::model::LlamaModel;
    use llama_cpp_rs_bench::{get_splitter_config, process_single, process_splits_batch, to_llama_tokens, SentenceScore};
    use std::fs;
    use std::path::PathBuf;

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

    fn get_embeddings(ctx: &mut LlamaContext,
                      model: &LlamaModel,
                      splitter: &SplitterLiteConfig<HFTokenizer>,
                      sentences: &Vec<String>,
                      embeddings: &mut Vec<Vec<f32>>) -> Result<()> {
        for sentence in sentences {
            let splits = splitter.hf_splits(sentence.as_bytes());
            println!("tokens: {:?}", splits);
            let split = splits.first().unwrap();
            let tokens = to_llama_tokens(&split.tokens.clone(), model)?;
            let embedding = process_single(ctx, &tokens)?;
            embeddings.push(embedding);
        }
        Ok(())
    }

    fn get_embeddings2(ctx: &mut LlamaContext,
                      model: &LlamaModel,
                      sentences: &Vec<String>,
                      embeddings: &mut Vec<Vec<f32>>) -> Result<()> {
        process_splits_batch(model, ctx, sentences)?;
        Ok(())
    }

    #[test]
    fn test() {
        let backend = LlamaBackend::init().unwrap();
        //backend.void_logs();

        let model_params = if cfg!(any(feature = "cuda", feature = "vulkan", feature = "metal")) {
            LlamaModelParams::default().with_n_gpu_layers(1000)
        } else {
            LlamaModelParams::default()
        };

        let model_path = "/Users/sabafallah/dev/qimia_ai_dev/llama.cpp/models/all-MiniLM-L6-v2.gguf".to_string();
        //let model_path = "/Users/sabafallah/dev/qimia_ai_dev/llama.cpp/models/multilingual-e5-large-instruct-q4_k_m.gguf".to_string();
        let model_path = PathBuf::from(model_path);

        let model = LlamaModel::load_from_file(&backend, model_path, &model_params).unwrap();

        // initialize the context
        let ctx_params_default = LlamaContextParams::default();
        let parallelism = std::thread::available_parallelism().unwrap().get() as i32;
        println!("parallelism: {}", parallelism);
        let ctx_params = LlamaContextParams::default().with_n_threads_batch(parallelism)
            .with_embeddings(true);

        let mut ctx = model.new_context(&backend, ctx_params).unwrap();

        let splitter_config = get_splitter_config(None, Some(512)).unwrap();


        let sentences1 = [
            "The new movie is awesome",
            "The cat sits outside",
            "A man is playing guitar",
            "I love pasta",
        ];
        let sentences1 = sentences1.map(|s| s.to_string()).to_vec();
        let mut embeddings1 = Vec::new();
        get_embeddings(&mut ctx, &model, &splitter_config, &sentences1, &mut embeddings1).unwrap();

        get_embeddings2(&mut ctx, &model, &sentences1, &mut embeddings1).unwrap();

        let embeddings1_ts = Tensor::new(embeddings1, &Device::Cpu).unwrap();
        assert_eq!(embeddings1_ts.shape().dims2().unwrap(), (4usize, 384usize));


        let sentences2 = [
            "The dog plays in the garden",
            "The new movie is so great",
            "A woman watches TV",
            "Do you like pizza?",
        ];

        let sentences2 = sentences2.map(|s| s.to_string()).to_vec();
        let mut embeddings2 = Vec::new();
        get_embeddings(&mut ctx, &model, &splitter_config, &sentences2, &mut embeddings2).unwrap();
        let embeddings2_ts = Tensor::new(embeddings2, &Device::Cpu).unwrap();

        assert_eq!(embeddings2_ts.shape().dims2().unwrap(), (4usize, 384usize));

        let similarities = similarity_matrix(&embeddings1_ts, &embeddings2_ts, false).unwrap();
        let similarities_vec = similarities.to_vec2::<f32>().unwrap();
        for (idx_i, sentence1) in sentences1.iter().enumerate() {
            println!("{}:", sentence1);
            let mut sts_similarities = Vec::new();
            for (idx_j, sentence2) in sentences2.iter().enumerate() {
                let score = similarities_vec.get(idx_i).unwrap().get(idx_j).unwrap();
                sts_similarities.push(SentenceScore {
                    sentence: sentence2.to_string(),
                    score: *score,
                });
            }
            sts_similarities.sort();
            sts_similarities.reverse();
            for sts in sts_similarities.iter() {
                println!("\t {:?}", sts);
            }
        }


    }
}