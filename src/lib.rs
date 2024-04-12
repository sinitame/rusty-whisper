pub mod audio;
mod decoder;
mod runner;
mod tokenizers;
mod utils;

use audio::{read_audio, Featurizer};
use decoder::GreedyDecoder;
use runner::{load_model, load_pos_embedding, WhisperRunner};
use std::path::Path;
use tract_core::prelude::*;
use utils::Options;

pub use tokenizers::Tokenizer;

pub struct Whisper {
    decoder: GreedyDecoder,
}

impl Whisper {
    pub fn new<P: AsRef<Path>>(
        encoder_path: P,
        decoder_path: P,
        tokenizer_path: P,
        pos_emb_path: P,
        mel_filters_path: P,
    ) -> Whisper {
        // Load Tokenizer
        let tokenizer = Tokenizer::new(tokenizer_path.as_ref());

        // Load WhisperRunner
        let encoder = Arc::new(load_model(encoder_path.as_ref()));
        let decoder = Arc::new(load_model(decoder_path.as_ref()));
        let positional_embedding = load_pos_embedding(pos_emb_path.as_ref());
        let featurizer = Featurizer::new(mel_filters_path.as_ref());
        let runner = WhisperRunner::new(encoder, decoder, positional_embedding);

        // Create decoder
        let options = Options::new();
        let decoder = GreedyDecoder::new(featurizer, runner, tokenizer, options);

        Whisper { decoder }
    }

    pub fn predict<P: AsRef<Path>>(&mut self, audio_path: P, language: &str) -> Option<String> {
        let audio_data = read_audio(audio_path).unwrap();

        self.decoder.process_audio(&audio_data, language);
        self.decoder.get_result()
    }
}
