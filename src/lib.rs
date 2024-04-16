pub mod audio;
mod decoder;
mod runner;
mod tokenizers;
mod utils;

use audio::read_audio;
use std::path::Path;
use tract_core::prelude::*;

pub use audio::Featurizer;
pub use decoder::GreedyDecoder;
pub use runner::{load_model, load_pos_embedding, WhisperRunner};
pub use tokenizers::Tokenizer;
pub use utils::Options;

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
        enable_f16: bool,
    ) -> Whisper {
        // Load Tokenizer
        let tokenizer = Tokenizer::new(tokenizer_path.as_ref());

        // Load WhisperRunner
        let (encoder, encoder_float_precision) =
            load_model(encoder_path.as_ref(), "encoder", enable_f16).unwrap();
        let (decoder, decoder_float_precision) =
            load_model(decoder_path.as_ref(), "decoder", enable_f16).unwrap();
        let positional_embedding = load_pos_embedding(pos_emb_path.as_ref());
        let featurizer = Featurizer::new(mel_filters_path.as_ref());
        let runner = WhisperRunner::new(
            Arc::new(encoder),
            encoder_float_precision,
            Arc::new(decoder),
            decoder_float_precision,
            positional_embedding,
            featurizer,
        );

        // Create decoder
        let options = Options::default();
        let decoder = GreedyDecoder::new(runner, tokenizer, options);

        Whisper { decoder }
    }

    pub fn predict<P: AsRef<Path>>(&mut self, audio_path: P, language: &str) -> Option<String> {
        let audio_data = read_audio(audio_path).unwrap();
        self.decoder.process_audio(&audio_data, language, true);
        self.decoder.get_result()
    }
}
