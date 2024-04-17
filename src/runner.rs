use std::{fs::File, path::Path, sync::Arc};

use ndarray::{s, Array2, Array3, ArrayD, Axis};
use ndarray_npy::NpzReader;
use rayon::prelude::*;
use tract_core::{
    anyhow::Result, floats::FloatPrecisionTranslator, prelude::translator::Translate,
    transform::get_transform,
};
use tract_onnx::prelude::*;

use crate::audio::{self, Featurizer};

pub struct WhisperRunner {
    encoder: Arc<TypedRunnableModel<TypedModel>>,
    encoder_float_precision: DatumType,
    decoder: Arc<TypedRunnableModel<TypedModel>>,
    decoder_float_precision: DatumType,
    positional_embedding: Array3<f32>,
    featurizer: Featurizer,
}

impl WhisperRunner {
    pub fn new(
        encoder: Arc<TypedRunnableModel<TypedModel>>,
        encoder_float_precision: DatumType,
        decoder: Arc<TypedRunnableModel<TypedModel>>,
        decoder_float_precision: DatumType,
        positional_embedding: Array3<f32>,
        featurizer: Featurizer,
    ) -> Self {
        Self {
            encoder,
            encoder_float_precision,
            decoder,
            decoder_float_precision,
            positional_embedding,
            featurizer,
        }
    }

    pub fn default_kv_cache(&self) -> Vec<TValue> {
        let default_tensor: TValue = if self.decoder_float_precision == DatumType::F16 {
            Tensor::zero::<f16>(&[1, 0, 512]).unwrap().into()
        } else {
            Tensor::zero::<f32>(&[1, 0, 512]).unwrap().into()
        };

        let mut default_kv_cache = Vec::new();
        for _ in 0..6 * 2 {
            default_kv_cache.push(default_tensor.clone())
        }
        default_kv_cache
    }

    pub fn compute_audio_embedding(
        &self,
        audio: &[f32],
        enable_multi_threading: bool,
    ) -> Vec<ArrayD<f32>> {
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
        let audio_embeddings: Vec<ArrayD<f32>> = if enable_multi_threading {
            segments
                .par_iter()
                .map(|segment| self.run_encoder(segment.clone()))
                .collect()
        } else {
            segments
                .iter()
                .map(|it| self.run_encoder(it.clone()))
                .collect()
        };

        audio_embeddings
    }

    fn run_encoder(&self, audio_features: Array2<f32>) -> ArrayD<f32> {
        let mel: Tensor = audio_features.insert_axis(Axis(0)).into();

        // Convert input to f16 if required
        let mel_float_type = mel.datum_type();
        let required_float_type = self.encoder_float_precision;
        let mel = match (mel_float_type, required_float_type) {
            (DatumType::F32, DatumType::F16) => mel.cast_to::<f16>().unwrap().into_owned(),
            _ => mel,
        };

        let inputs = tvec![mel.into_tvalue()];
        let encoder_out = self.encoder.run(inputs).unwrap().remove(0);
        let casted_output = encoder_out
            .cast_to::<f32>()
            .unwrap()
            .into_owned()
            .into_tvalue();
        casted_output.to_array_view::<f32>().unwrap().to_owned()
    }

    pub fn inference_logits(
        &mut self,
        tokens: &[u32],
        audio_embedding: &ArrayD<f32>,
        kv_cache: Vec<TValue>,
        is_prompt: bool,
    ) -> (ArrayD<f32>, Vec<TValue>) {
        let tokens = Array2::from_shape_vec(
            [1, tokens.len()],
            // Conversion is done here only because model input is I32 (but it should be U32)
            tokens.iter().map(|it| *it as i32).collect(),
        )
        .unwrap();
        let offset = kv_cache.get(0).map(|it| it.shape()[1]).unwrap_or(0);
        let mut tokens = tokens;

        if !is_prompt {
            // If we are not processing a prompt, we only feed the last token to the decoder
            // Previous tokens infos will be on the KV cache.
            tokens = tokens.slice(s![.., -1]).to_owned().insert_axis(Axis(0));
        }

        let mut audio_embedding_tensor = audio_embedding.clone().into_tensor();
        let mut pos_emb_tensor = self
            .positional_embedding
            .slice(s![.., offset..offset + tokens.shape()[1], ..])
            .into_owned()
            .into_tensor();

        // Cast inputs
        if self.decoder_float_precision == DatumType::F16 {
            audio_embedding_tensor = audio_embedding_tensor
                .cast_to::<f16>()
                .unwrap()
                .into_owned();
            pos_emb_tensor = pos_emb_tensor.cast_to::<f16>().unwrap().into_owned();
        }

        let mut inputs = tvec![
            tokens.into_tvalue(),
            audio_embedding_tensor.clone().into_tvalue(),
            pos_emb_tensor.into_tvalue(),
        ];
        inputs.extend(kv_cache);
        let mut out = self.decoder.run(inputs).unwrap();

        let logits_tensor = out.remove(0).cast_to::<f32>().unwrap().into_owned();
        let logits = logits_tensor.to_array_view::<f32>().unwrap().to_owned();
        let new_kv_cache = out.iter().map(|it| it.clone().into()).collect();
        (logits, new_kv_cache)
    }
}

pub fn load_pos_embedding<P: AsRef<Path>>(pos_emb_path: P) -> Array3<f32> {
    let file = File::open(pos_emb_path).expect("Failed to open file");
    let mut npz = NpzReader::new(file).expect("Failed to read NPZ file");
    let pos_emb: Array2<f32> = npz.by_index(0).unwrap();
    pos_emb.insert_axis(Axis(0))
}

pub fn load_model(
    model_path: &Path,
    model_name: &str,
    enable_f16: bool,
) -> Result<(TypedSimplePlan<TypedModel>, DatumType)> {
    let mut typed_model = tract_onnx::onnx()
        .model_for_path(model_path)
        .unwrap()
        .into_typed()
        .unwrap()
        .into_decluttered()
        .unwrap();

    if cfg!(any(
        feature = "accelerate",
        feature = "openblas",
        feature = "blis"
    )) {
        log::info!("Applying 'as-bas' transformation.");
        get_transform("as-blas")
            .unwrap()
            .transform(&mut typed_model)
            .unwrap();
    }

    // Convert model to f16 if enabled
    let inputs_datum = typed_model
        .input_outlets()?
        .iter()
        .map(|it| typed_model.outlet_fact(*it).map(|it| it.datum_type))
        .collect::<Result<Vec<DatumType>>>()?;

    let mut model_float_precision = inputs_datum
        .into_iter()
        .filter(|it| it.is_float())
        .next()
        .unwrap();

    if enable_f16 & (model_float_precision == DatumType::F32) {
        println!("Model {model_name} precision is in f32, converting from f32 to f16.");
        typed_model =
            FloatPrecisionTranslator::<f32, f16>::default().translate_model(&typed_model)?;
        model_float_precision = DatumType::F16;
    }

    Ok((
        typed_model.into_optimized()?.into_runnable()?,
        model_float_precision,
    ))
}
