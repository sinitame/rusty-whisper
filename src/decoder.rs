use ndarray::{s, ArrayBase, Dim, IxDynImpl, OwnedRepr};
use tract_core::value::TValue;

use crate::{runner::WhisperRunner, utils::Options, Tokenizer};

pub struct GreedyDecoder {
    decoded_tokens: Option<Vec<u32>>,
    options: Options,
    runner: WhisperRunner,
    tokenizer: Tokenizer,
    kv_cache: Vec<TValue>,
}

impl GreedyDecoder {
    pub fn new(runner: WhisperRunner, tokenizer: Tokenizer, options: Options) -> Self {
        let default_kv_cache = runner.default_kv_cache();
        Self {
            decoded_tokens: None,
            options,
            runner,
            tokenizer,
            kv_cache: default_kv_cache,
        }
    }

    pub fn get_result(&self) -> Option<String> {
        self.decoded_tokens.as_ref().map(|it| {
            self.tokenizer.decode(
                it.iter()
                    .map(|v| *v as usize)
                    .filter(|item| item < &50257)
                    .collect(),
            )
        })
    }

    pub fn process_audio(
        &mut self,
        audio: &[f32],
        language: &str,
        enable_multi_threaded_encoder: bool,
    ) -> bool {
        let audio_embeddings = self
            .runner
            .compute_audio_embedding(audio, enable_multi_threaded_encoder);

        // For each audio chunk, predict tokens
        for audio_embedding in audio_embeddings {
            let new_tokens = self.process_audio_chunk(&audio_embedding, language);
            if let Some(previous_tokens) = self.decoded_tokens.as_mut() {
                previous_tokens.extend(new_tokens.clone())
            } else {
                self.decoded_tokens = Some(new_tokens);
            }
        }
        true
    }

    fn process_audio_chunk(
        &mut self,
        audio_embedding: &ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>,
        language: &str,
    ) -> Vec<u32> {
        let mut tokens = Self::get_initial_tokens(
            &self.tokenizer,
            self.decoded_tokens.as_ref(),
            language,
            &self.options,
        );
        let initial_len = tokens.len();
        for _ in 0..224 {
            let kv = self.kv_cache.drain(..).map(|it| it.into()).collect();
            let (logits, new_kv) =
                self.runner
                    .inference_logits(tokens.as_slice(), &audio_embedding, kv, initial_len);
            self.kv_cache = new_kv;
            let next_token = logits
                .slice(s![.., -1, ..])
                .iter()
                .enumerate()
                .max_by(|(_, u), (_, v)| u.total_cmp(v))
                .map(|(i, _)| i as usize)
                .unwrap();

            if next_token == self.options.eot_token || tokens.len() > self.options.n_ctx {
                break;
            }

            tokens.push(next_token as u32);
        }

        tokens[initial_len..].to_vec()
    }

    pub fn get_initial_tokens(
        tokenizer: &Tokenizer,
        prompt: Option<&Vec<u32>>,
        language: &str,
        options: &Options,
    ) -> Vec<u32> {
        let init_tokens = tokenizer.get_initial_tokens(language);
        if let Some(prompt) = prompt {
            let prev_prompt_len = options.n_ctx / 2 - 1;
            let prompt_tokens: Vec<u32>;

            if prompt.len() > prev_prompt_len {
                prompt_tokens = prompt[prompt.len() - prev_prompt_len..].to_vec();
            } else {
                prompt_tokens = prompt.to_vec();
            }

            let tokens: Vec<_> = vec![options.sot_prev as u32]
                .into_iter()
                .chain(prompt_tokens.into_iter())
                .collect();
            let tokens: Vec<_> = tokens.into_iter().chain(init_tokens.into_iter()).collect();
            tokens
        } else {
            let tokens = vec![options.sot_prev as u32];
            let tokens: Vec<_> = tokens.into_iter().chain(init_tokens.into_iter()).collect();
            tokens
        }
    }
}
