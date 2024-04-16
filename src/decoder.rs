use ndarray::{s, ArrayBase, Dim, IxDynImpl, OwnedRepr};

use crate::{runner::WhisperRunner, utils::Options, Tokenizer};

pub struct GreedyDecoder {
    decoded_tokens: Option<Vec<u32>>,
    options: Options,
    runner: WhisperRunner,
    tokenizer: Tokenizer,
}

impl GreedyDecoder {
    pub fn new(runner: WhisperRunner, tokenizer: Tokenizer, options: Options) -> Self {
        Self {
            decoded_tokens: None,
            options,
            runner,
            tokenizer,
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
        // We always reset the KV cache between audio chunks even if we feed the previous chunk
        // tokens in the new prompt. The reason for that is because additional tokens are inserted.
        // The prompt becomes: [SOT_PREV, PROMPT (previous tokens), INIT_TOKENS].
        // TODO: this can be optimized
        let mut kv_cache = self.runner.default_kv_cache();
        let mut tokens = Self::get_initial_tokens(
            &self.tokenizer,
            self.decoded_tokens.as_ref(),
            language,
            &self.options,
        );
        let initial_len = tokens.len();

        // Loop while we don't exceed the max number of tokens that can be generated from an audio
        // chunk (30s of speech).
        for i in 0..224 {
            let kv = kv_cache.drain(..).map(|it| it.into()).collect();
            // If it's the first iteration, it means that we are processing a prompt,
            // so is_prompt should be true.
            let is_prompt = i == 0;
            let (logits, new_kv) =
                self.runner
                    .inference_logits(tokens.as_slice(), &audio_embedding, kv, is_prompt);

            // Update KV cache with new values
            kv_cache = new_kv;
            let next_token = logits
                .slice(s![.., -1, ..])
                .iter()
                .enumerate()
                .max_by(|(_, u), (_, v)| u.total_cmp(v))
                .map(|(i, _)| i as usize)
                .unwrap();

            // Stop if we reach end of transcript token.
            // Second condition is not very clear ..
            if next_token == self.options.eot_token || tokens.len() > self.options.n_ctx {
                break;
            }

            tokens.push(next_token as u32);
        }

        // Return only the tokens that are not part of the prompt
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
