mod tetes {
    use anyhow::{bail, Context, Result};
    use candle_core::{Device, Tensor};
    use fast_text_splitter::config::SplitterLiteConfig;
    use fast_text_splitter::hf_tokenizer::HFTokenizer;
    use fast_text_splitter::splitter::split_node::utils::SplitResultLite;
    use llama_cpp::context::LlamaContext;
    use llama_cpp::llama_backend::LlamaBackend;
    use llama_cpp::model::{AddBos, LlamaModel};
    use llama_cpp::token::LlamaToken;
    use llama_cpp_rs_bench::split_data::{QuerySummaries, SplitData, SummaryData};
    use llama_cpp_rs_bench::{batch_decode_rerank, get_embeddings, init_backend, init_context, init_model, init_reranker_context, init_splitter, llama_cpp_tokenize, process_batch, process_splits_batch, SentenceScore};
    use rayon::prelude::*;
    use std::fs;
    use std::path::Path;
    use llama_cpp::llama_batch::LlamaBatch;

    fn ensure_dir_exists(dir_path: &str) -> std::io::Result<()> {
        if !Path::new(dir_path).exists() {
            fs::create_dir_all(dir_path)?;
        }
        Ok(())
    }

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

    fn get_embeddings2(
        ctx: &mut LlamaContext,
        model: &LlamaModel,
        sentences: &Vec<String>,
    ) -> Result<Vec<Vec<f32>>> {
        process_splits_batch(model, ctx, sentences)
    }
    fn text_file_embeddings(
        model_path: &str,
        text_file_path: &str,
        out_dir: &str,
        hf_model: Option<String>,
        n_ctx: Option<u32>,
        n_batch: Option<u32>,
        n_ubatch: Option<u32>,
    ) -> Result<()> {
        let backend = init_backend(false)?;
        let model = init_model(&model_path, &backend)?;
        let mut ctx = init_context(&model, &backend, n_ctx, n_batch, n_ubatch)?;

        let n_ctx = ctx.n_ctx() as usize;

        let split_splitter = init_splitter(hf_model.clone(), None, Some(400), true)?;
        let sentence_splitter = init_splitter(hf_model, None, Some(400), false)?;

        let device = Device::Cpu;

        // chech if out_dir exists if not create it
        ensure_dir_exists(out_dir)?;

        let data_path = text_file_path;
        let binding = fs::read_to_string(data_path)?;
        let data = binding.as_bytes();

        let mx_tokens_splits = split_splitter.hf_splits(data);

        let mut splits_data = Vec::new();

        mx_tokens_splits
            .iter()
            .enumerate()
            .for_each(|(split_id, mx_tokens_split)| {
                process_split_data(
                    out_dir,
                    &mut ctx,
                    n_ctx,
                    &sentence_splitter,
                    &device,
                    &mut splits_data,
                    split_id,
                    mx_tokens_split,
                );
            });

        // save splits data json
        let splits_data_json = serde_json::to_string_pretty(&splits_data)?;
        let splits_data_file = format!("{}/splits_data.json", out_dir);
        fs::write(splits_data_file, splits_data_json)?;

        Ok(())
    }

    fn json_file_embeddings(
        model_path: &str,
        json_file_path: &str,
        out_dir: &str,
        hf_model: Option<String>,
    ) -> Result<()> {
        let backend = init_backend(false)?;
        let model = init_model(&model_path, &backend)?;
        let mut ctx = init_context(&model, &backend, Some(4096), None, None)?;

        let n_ctx = ctx.n_ctx() as usize;

        let split_splitter = init_splitter(hf_model.clone(), None, Some(4096), true)?;
        let sentence_splitter = init_splitter(hf_model, None, Some(4096), false)?;

        let device = Device::Cpu;

        let text_out_dir = format!("{}/text", out_dir);
        ensure_dir_exists(text_out_dir.as_str())?;

        let input_str = fs::read_to_string(json_file_path)?;
        let inputs_vec: Vec<SummaryData> = serde_json::from_str(&input_str)?;

        let mx_tokens_splits = inputs_vec
            .clone()
            .into_iter()
            .flat_map(|input| {
                let data = input.text.as_bytes();
                split_splitter.hf_splits(data)
            })
            .collect::<Vec<SplitResultLite>>();

        let mut splits_data = Vec::new();

        mx_tokens_splits
            .iter()
            .enumerate()
            .for_each(|(split_id, mx_tokens_split)| {
                process_split_data(
                    text_out_dir.as_str(),
                    &mut ctx,
                    n_ctx,
                    &sentence_splitter,
                    &device,
                    &mut splits_data,
                    split_id,
                    mx_tokens_split,
                );
            });

        // save splits data json
        let splits_data_json = serde_json::to_string_pretty(&splits_data)?;
        let splits_data_file = format!("{}/splits_data.json", text_out_dir);
        fs::write(splits_data_file, splits_data_json)?;

        let summary_out_dir = format!("{}/summary", out_dir);
        ensure_dir_exists(summary_out_dir.as_str())?;

        let mx_tokens_splits = inputs_vec
            .into_iter()
            .flat_map(|input| {
                let data = input.summary.as_bytes();
                split_splitter.hf_splits(data)
            })
            .collect::<Vec<SplitResultLite>>();

        let mut splits_data = Vec::new();
        mx_tokens_splits
            .iter()
            .enumerate()
            .for_each(|(split_id, mx_tokens_split)| {
                process_split_data(
                    summary_out_dir.as_str(),
                    &mut ctx,
                    n_ctx,
                    &sentence_splitter,
                    &device,
                    &mut splits_data,
                    split_id,
                    mx_tokens_split,
                );
            });

        // save splits data json
        let splits_data_json = serde_json::to_string_pretty(&splits_data)?;
        let splits_data_file = format!("{}/splits_data.json", summary_out_dir);
        fs::write(splits_data_file, splits_data_json)?;

        Ok(())
    }

    fn process_split_data(
        out_dir: &str,
        mut ctx: &mut LlamaContext,
        n_ctx: usize,
        sentence_splitter: &SplitterLiteConfig<HFTokenizer>,
        device: &Device,
        mut splits_data: &mut Vec<SplitData>,
        split_id: usize,
        mx_tokens_split: &SplitResultLite,
    ) {
        let split_id_str = format!("{:03}", split_id);

        let llama_tokens = llama_cpp_tokenize(&ctx.model, mx_tokens_split.split_string.as_str())
            .expect("unable to tokenize");
        let mut split_embedding_vec = Vec::new();
        get_embeddings(&llama_tokens, &mut split_embedding_vec, &mut ctx, n_ctx)
            .expect("embeddings failed");

        let split_embedding_tensor =
            Tensor::new(split_embedding_vec, &device).expect("unable to create tensor");
        let split_embedding_name = format!("split_embedding_{}", split_id_str);
        let split_embedding_file = format!("{}.safetensors", split_embedding_name);
        let split_embedding_path = format!("{}/{}", out_dir, split_embedding_file);

        split_embedding_tensor
            .save_safetensors(split_embedding_name.as_str(), split_embedding_path.as_str())
            .expect("unable to save tensors");

        let sentence_splits = sentence_splitter.hf_splits(mx_tokens_split.split_string.as_bytes());

        let sentence_splits_strs = sentence_splits
            .iter()
            .map(|sentence_split| sentence_split.split_string.clone())
            .collect::<Vec<String>>();

        let llama_tokens_list: Vec<Vec<LlamaToken>> = sentence_splits
            .iter()
            .map(|sentence_split| {
                let llama_tokens =
                    llama_cpp_tokenize(&ctx.model, sentence_split.split_string.as_str())
                        .expect("unable to convert to llama tokens");
                llama_tokens
            })
            .collect();
        let embds = process_batch(&mut ctx, &llama_tokens_list).expect("unable to process batch");
        let tensors = Tensor::new(embds, &device).expect("unable to create tensor");

        let sentence_embeddings_name = format!("sentence_embeddings_{}", split_id_str);
        let sentence_embeddings_file = format!("{}.safetensors", sentence_embeddings_name);
        let sentence_embeddings_path = format!("{}/{}", out_dir, sentence_embeddings_file);

        let split_data = SplitData {
            split_id,
            no_tokens: mx_tokens_split.tokens.len(),
            split_embedding_name: split_embedding_name.clone(),
            split_embedding_file: split_embedding_file.clone(),
            split_string: mx_tokens_split.split_string.clone(),
            sentence_embeddings_name: sentence_embeddings_name.clone(),
            sentence_embeddings_file: sentence_embeddings_file.clone(),
            sentences: sentence_splits_strs.clone(),
        };
        splits_data.push(split_data);

        tensors
            .save_safetensors(
                sentence_embeddings_name.as_str(),
                sentence_embeddings_path.as_str(),
            )
            .expect("unable to save tensors");
    }

    #[test]
    fn test_text_file_embeddings() -> Result<()> {
        let out_dir = "output/superlinear_embeddings/snowflake-arctic-embed-m-v1.5";
        //let out_dir = "output/superlinear_embeddings/gte-Qwen2-1.5B-instruct";
        //let out_dir = "output/superlinear_embeddings/multilingual-e5-large-instruct";
        //let out_dir = "output/superlinear_embeddings/bge-large-en";
        //let out_dir = "output/superlinear_embeddings/bge-m3";
        //let out_dir = "output/United_States/bge-m3";
        //let out_dir = "output/superlinear_embeddings/all-MiniLM-L6-v2";

        //let model_path = "models/all-MiniLM-L6-v2-Q4_K_M.gguf";
        //let model_path ="models/multilingual-e5-large-instruct-q8_0.gguf";
        //let model_path = "models/bge-large-en-v1.5-q8_0.gguf";
        //let model_path = "models/bge-m3-q4_k_m.gguf";
        //let model_path = "models/gemma-2-9b-it-Q4_K_M.gguf";
        let model_path = "models/snowflake-arctic-embed-m-v1.5-q4_k_m.gguf";
        //let model_path = "models/gte-qwen2-1.5b-instruct-q4_k_m.gguf";

        //let text_file_path = "tests/test_data/United_States.txt";
        let text_file_path = "tests/test_data/superlinear.txt";

        //let hf_model = Some("sentence-transformers/all-MiniLM-L6-v2".to_string());
        let hf_model = Some("Snowflake/snowflake-arctic-embed-m-v1.5".to_string());
        //let hf_model = Some("Alibaba-NLP/gte-Qwen2-1.5B-instruct".to_string());
        //let hf_model = Some("intfloat/multilingual-e5-large-instruct".to_string());
        //let hf_model = Some("BAAI/bge-large-en-v1.5".to_string());
        //let hf_model = Some("BAAI/bge-m3".to_string());
        //let hf_model = Some("google/gemma-2-9b-it".to_string());
        text_file_embeddings(
            &model_path,
            text_file_path,
            out_dir,
            hf_model,
            Some(512),
            Some(512),
            Some(512),
        )
    }

    #[test]
    fn test_json_file_embeddings() -> Result<()> {
        let out_dir = "output/gold_extractive/bge-m3";
        let model_path = "models/bge-m3-q4_k_m.gguf";
        let json_file_path = "tests/test_data/gold_extractive.json";
        let hf_model = Some("BAAI/bge-m3".to_string());
        json_file_embeddings(&model_path, json_file_path, out_dir, hf_model)
    }

    #[test]
    fn test() -> Result<()> {
        //let model_path = "models/all-MiniLM-L6-v2-Q4_K_M.gguf".to_string();
        let model_path = "models/snowflake-arctic-embed-m-v1.5-q4_k_m.gguf".to_string();

        //let model_path = "models/all-MiniLM-L6-v2-ggml-model-f16.gguf".to_string();
        //let model_path = "models/multilingual-e5-large-instruct-q4_k_m.gguf".to_string();

        //let model_id = "sentence-transformers/all-MiniLM-L6-v2".to_string();
        let model_id = "Snowflake/snowflake-arctic-embed-m-v1.5".to_string();

        let backend = init_backend(false)?;
        let model = init_model(&model_path, &backend)?;
        let mut ctx = init_context(&model, &backend, Some(3072), Some(3072), Some(3072))?;

        let sentences1 = [
            "The new movie is awesome",
            "The cat sits outside",
            "A man is playing guitar",
            "I love pasta",
        ];
        let sentences1 = sentences1.map(|s| s.to_string()).to_vec();
        let embeddings1 = process_splits_batch(&model, &mut ctx, &sentences1)?;

        let cache_used = ctx.get_kv_cache_used_cells();
        println!("cache_used: {}", cache_used);
        let kv_cache_size = ctx.get_kv_cache_token_count();
        println!("kv_cache_size: {}", kv_cache_size);

        let embeddings1_ts = Tensor::new(embeddings1, &Device::Cpu)?;

        let sentences2 = [
            "The dog plays in the garden",
            "The new movie is so great",
            "A woman watches TV",
            "Do you like pizza?",
        ];

        let sentences2 = sentences2.map(|s| s.to_string()).to_vec();
        let embeddings2 = process_splits_batch(&model, &mut ctx, &sentences2)?;

        let embeddings2_ts = Tensor::new(embeddings2, &Device::Cpu)?;

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
    fn bge_reranker_test() -> Result<()> {
        let json_file_path = "tests/test_data/bert_paper_query_summaries.json";
        let input_str = fs::read_to_string(json_file_path)?;
        let query_summaries = serde_json::from_str::<QuerySummaries>(&input_str)?;

        let model_path = "models/bge-m3-q4_k_m.gguf";
        let backend = init_backend(true)?;
        let model = init_model(model_path, &backend)?;
        let pooling = Some("rank");
        let mut ctx = init_reranker_context(
            &model,
            &backend,
            pooling,
            Some(2048),
            Some(2048),
            Some(2048),
        )?;

        let prompt_lines = {
            let query = query_summaries.query;
            let mut lines = Vec::new();
            for summary in query_summaries.summaries {
                // Todo!  update to get eos and sep from model instead of hardcoding
                lines.push(format!(
                    "{query}{eos}{sep}{summary}",
                    sep = "<s>",
                    eos = "</s>"
                ));
            }
            lines
        };

        // tokenize the prompt
        let tokens_lines_list = prompt_lines
            .iter()
            .map(|line| model.str_to_token(line, AddBos::Always))
            .collect::<Result<Vec<_>, _>>()
            .with_context(|| format!("failed to tokenize {:?}", prompt_lines))?;

        let n_ctx = ctx.n_ctx() as usize;
        let n_ctx_train = model.n_ctx_train();

        eprintln!("n_ctx = {n_ctx}, n_ctx_train = {n_ctx_train}");

        if tokens_lines_list.iter().any(|tok| n_ctx < tok.len()) {
            bail!("One of the provided prompts exceeds the size of the context window");
        }

        let n_embd = model.n_embd();


        // create a llama_batch with the size of the context
        // we use this object to submit token data for decoding
        let mut batch = LlamaBatch::new(2048, 1);

        let mut max_seq_id_batch = 0;
        let mut output = Vec::with_capacity(tokens_lines_list.len());
        let normalise = true;
        for tokens in &tokens_lines_list {
            // Flush the batch if the next prompt would exceed our batch size
            if (batch.n_tokens() as usize + tokens.len()) > 2048 {
                batch_decode_rerank(
                    &mut ctx,
                    &mut batch,
                    max_seq_id_batch,
                    &mut output,
                    normalise,
                    pooling.unwrap().to_string(),
                )?;
                max_seq_id_batch = 0;
                batch.clear();
            }

            batch.add_sequence(tokens, max_seq_id_batch, false)?;
            max_seq_id_batch += 1;
        }
        // Handle final batch
        batch_decode_rerank(
            &mut ctx,
            &mut batch,
            max_seq_id_batch,
            &mut output,
            normalise,
            pooling.unwrap().to_string(),
        )?;

        for (j, embeddings) in output.iter().enumerate() {
            if pooling.unwrap() == "none" {
                eprintln!("embedding {j}: ");
                for i in 0..n_embd as usize {
                    if !normalise {
                        eprint!("{:6.5} ", embeddings[i]);
                    } else {
                        eprint!("{:9.6} ", embeddings[i]);
                    }
                }
                eprintln!();
            } else if pooling.unwrap() == "rank" {
                eprintln!("rerank score {j}: {:8.3}", embeddings[0]);
            } else {
                eprintln!("embedding {j}: ");
                for i in 0..n_embd as usize {
                    if !normalise {
                        eprint!("{:6.5} ", embeddings[i]);
                    } else {
                        eprint!("{:9.6} ", embeddings[i]);
                    }
                }
                eprintln!();
            }
        }

        Ok(())
    }
}
