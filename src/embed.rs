#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use std::{fs::File, io::Read, path::Path};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use tokenizers::{PaddingParams, Tokenizer};


// use candle_transformers::models::bert::{BertModel, Config, HiddenAct, DTYPE};

use anyhow::{Error as E, Result};
// use candle::Tensor;
// use candle_nn::VarBuilder;
// use clap::Parser;
// use hf_hub::{api::sync::Api, Repo, RepoType};
// use tokenizers::{PaddingParams, Tokenizer};

// const WEIGHTS: &[u8] = include_bytes!("../models/model.safetensors");
// const CONFIG: &[u8] = include_bytes!("../models/config.json");
// const TOKENIZER: &[u8] = include_bytes!("../models/tokenizer.json");

// const WEIGHTS: &[u8] = include_bytes!(format!("{}/model.safetensors", env!("MODEL_DIR")));
// const CONFIG: &[u8] = include_bytes!(env!("MODEL_PATH"));
// const TOKENIZER: &[u8] = include_bytes!("../models/tokenizer.json");


pub struct EmbeddingModel {
    bert: BertModel,
    tokenizer: Tokenizer,
}

pub struct ModelPrams {
    pub(crate) weights: Vec<u8>,
    pub(crate) config: Vec<u8>,
    pub(crate) tokenizer: Vec<u8>
}

impl EmbeddingModel {
    pub fn new(params: ModelPrams) ->  Result<Self> {
        // let mut weights = Vec::new();
        // File::open(model_dir.join("/model.safetensors"))?.read_to_end(&mut weights)?;

        // let mut config = Vec::new();
        // File::open(model_dir.join("/config.json"))?.read_to_end(&mut config)?;

        // let mut tokenizer = Vec::new();
        // File::open(model_dir.join("/tokenizer.json"))?.read_to_end(&mut tokenizer)?;

        let device = &Device::Cpu;
        let vb = VarBuilder::from_slice_safetensors(&params.weights, DType::F32, device)?;
        let config: Config = serde_json::from_slice(&params.config)?;
        let tokenizer = Tokenizer::from_bytes(params.tokenizer).map_err(E::msg)?;
        let bert = BertModel::load(vb, &config)?;
    
        Ok(Self { bert, tokenizer })
    }

    pub fn get_embeddings(
        &mut self,
        sentences: &Vec<String>,
        normalize_embeddings: bool,
    ) -> Result<Vec<Vec<f32>>> {
        let device = &Device::Cpu;
        // set padding setting
        if let Some(pp) = self.tokenizer.get_padding_mut() {
            pp.strategy = tokenizers::PaddingStrategy::BatchLongest
        } else {
            let pp = PaddingParams {
                strategy: tokenizers::PaddingStrategy::BatchLongest,
                ..Default::default()
            };
            self.tokenizer.with_padding(Some(pp));
        }
        // set truncation setting TruncationParams
        let _ = self
            .tokenizer
            .with_truncation(Some(tokenizers::TruncationParams::default()));

        let tokens = self
            .tokenizer
            .encode_batch(sentences.to_vec(), true)
            .map_err(E::msg)?;

        let token_ids: Vec<Tensor> = tokens
            .iter()
            .map(|tokens| {
                let tokens = tokens.get_ids().to_vec();
                Tensor::new(tokens.as_slice(), device)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let token_ids = Tensor::stack(&token_ids, 0)?;
        let token_type_ids = token_ids.zeros_like()?;
        // console_log!("running inference on batch {:?}", token_ids.shape());
        let embeddings = self.bert.forward(&token_ids, &token_type_ids)?;
        // console_log!("generated embeddings {:?}", embeddings.shape());
        // Apply some avg-pooling by taking the mean embedding value for all tokens (including padding)
        let (_n_sentence, n_tokens, _hidden_size) = embeddings.dims3()?;
        let embeddings = (embeddings.sum(1)? / (n_tokens as f64))?;
        let embeddings = if normalize_embeddings {
            embeddings.broadcast_div(&embeddings.sqr()?.sum_keepdim(1)?.sqrt()?)?
        } else {
            embeddings
        };
        let embeddings_data = embeddings.to_vec2()?;
        Ok(embeddings_data)
    }
}
