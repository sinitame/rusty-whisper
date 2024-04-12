use ndarray::{s, Array2, ArrayBase, Dim, IxDynImpl, OwnedRepr};
use rayon::prelude::*;

use crate::{
    audio::{self, Featurizer},
    runner::WhisperRunner,
    utils::Options,
    Tokenizer,
};

pub struct GreedyDecoder {
    decoded_tokens: Option<Vec<i32>>,
    featurizer: Featurizer,
    options: Options,
    runner: WhisperRunner,
    tokenizer: Tokenizer,
}

impl GreedyDecoder {
    pub fn new(
        featurizer: Featurizer,
        runner: WhisperRunner,
        tokenizer: Tokenizer,
        options: Options,
    ) -> Self {
        Self {
            decoded_tokens: None,
            featurizer,
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

    pub fn process_audio(&mut self, audio: &[f32], language: &str) -> bool {
        let mel = self.featurizer.featurize(audio);

        let num_frames = mel.shape()[1];
        let mut seek = 0;
        let mut segments = vec![];

        // Compute audio embeddings in parallel
        while seek < num_frames {
            let segment: Array2<f32>;

            if seek + audio::N_FRAMES < mel.shape()[1] {
                segment = mel.slice(s![.., seek..seek + audio::N_FRAMES]).to_owned();
            } else {
                segment = mel.slice(s![.., seek..]).to_owned();
            }

            segments.push(audio::pad_or_trim(segment, audio::N_FRAMES));
            seek += audio::N_FRAMES;
        }
        let audio_embeddings: Vec<ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>> = segments
            .par_iter()
            .map(|segment| self.runner.compute_audio_embedding(segment.clone()))
            .collect();

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
    ) -> Vec<i32> {
        let mut tokens = self.get_initial_tokens(self.decoded_tokens.as_ref(), language);
        let initial_len = tokens.len();
        for _ in 0..224 {
            let logits =
                self.runner
                    .inference_logits(tokens.as_slice(), &audio_embedding, initial_len);

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

            tokens.push(next_token as i32);
        }

        tokens[initial_len..].to_vec()
    }

    fn get_initial_tokens(&self, prompt: Option<&Vec<i32>>, language: &str) -> Vec<i32> {
        let init_tokens = self.tokenizer.get_initial_tokens(language);
        if let Some(prompt) = prompt {
            let prev_prompt_len = self.options.n_ctx / 2 - 1;
            let prompt_tokens: Vec<i32>;

            if prompt.len() > prev_prompt_len {
                prompt_tokens = prompt[prompt.len() - prev_prompt_len..].to_vec();
            } else {
                prompt_tokens = prompt.to_vec();
            }

            let tokens: Vec<i32> = vec![self.options.sot_prev as i32]
                .into_iter()
                .chain(prompt_tokens.into_iter())
                .collect();
            let tokens: Vec<i32> = tokens.into_iter().chain(init_tokens.into_iter()).collect();
            tokens
        } else {
            let tokens = vec![self.options.sot_prev as i32];
            let tokens: Vec<i32> = tokens.into_iter().chain(init_tokens.into_iter()).collect();
            tokens
        }
    }
}
