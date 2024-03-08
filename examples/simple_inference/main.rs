use std::{path::PathBuf, str::FromStr};

use rusty_whisper::Whisper;
use serde::Deserialize;

#[derive(Deserialize)]
struct ModelConfig {
    pub encoder: String,
    pub decoder: String,
    pub tokenizer: String,
    pub positional_embedding: String,
    pub mel_filters: String,
}

fn main() {
    let args = std::env::args().collect::<Vec<String>>();
    let model_config_path = args
        .get(1)
        .expect("Please provide a path to the model configuration file.");
    let audio_file_path = args
        .get(2)
        .expect("Please provide a path to an audio file.");

    let model_path = PathBuf::from_str(&model_config_path).expect("Invalid config path");
    let model_dir = model_path
        .parent()
        .expect("Expected config file to be in model directory.");
    let config = std::fs::read_to_string(&model_config_path)
        .expect("Could not read model configuration file.");
    let model_config: ModelConfig = serde_json::from_str(&config).expect("Invalid config.");
    let whisper = Whisper::new(
        model_dir.join(model_config.encoder),
        model_dir.join(model_config.decoder),
        model_dir.join(model_config.tokenizer),
        model_dir.join(model_config.positional_embedding),
        model_dir.join(model_config.mel_filters),
    );
    let result = whisper.recognize_from_audio(&audio_file_path, "en");
    println!("{}", result);
}
