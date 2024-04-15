use std::{fs::File, path::Path, sync::Arc};

use ndarray::{s, Array2, Array3, ArrayBase, Axis, Dim, IxDynImpl, OwnedRepr};
use ndarray_npy::NpzReader;
use rayon::prelude::*;
use tract_core::transform::get_transform;
use tract_ndarray::ArrayD;
use tract_onnx::prelude::*;

use crate::audio::{self, Featurizer};

pub struct WhisperRunner {
    encoder: Arc<TypedRunnableModel<TypedModel>>,
    decoder: Arc<TypedRunnableModel<TypedModel>>,
    positional_embedding: Array3<f32>,
    featurizer: Featurizer,
}

impl WhisperRunner {
    pub fn new(
        encoder: Arc<TypedRunnableModel<TypedModel>>,
        decoder: Arc<TypedRunnableModel<TypedModel>>,
        positional_embedding: Array3<f32>,
        featurizer: Featurizer,
    ) -> Self {
        Self {
            encoder,
            decoder,
            positional_embedding,
            featurizer,
        }
    }

    pub fn default_kv_cache(&self) -> Vec<TValue> {
        let mut default_kv_cache = Vec::new();
        for _ in 0..6 * 2 {
            default_kv_cache.push(Tensor::zero::<f32>(&[1, 0, 512]).unwrap().into())
        }
        default_kv_cache
    }

    pub fn compute_audio_embedding(
        &self,
        audio: &[f32],
        enable_multi_threading: bool,
    ) -> Vec<ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>> {
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
        let audio_embeddings: Vec<ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>> =
            if enable_multi_threading {
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

    fn run_encoder(
        &self,
        audio_features: Array2<f32>,
    ) -> ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>> {
        let mel: Tensor = audio_features.insert_axis(Axis(0)).into();
        let inputs = tvec![mel.into_tvalue()];
        let encoder_out = self.encoder.run(inputs).unwrap()[0]
            .to_array_view::<f32>()
            .unwrap()
            .to_owned();
        encoder_out
    }

    pub fn inference_logits(
        &mut self,
        tokens: &[u32],
        audio_embedding: &ArrayD<f32>,
        kv_cache: Vec<TValue>,
        initial_token_length: usize,
    ) -> (ArrayD<f32>, Vec<TValue>) {
        let tokens = Array2::from_shape_vec(
            [1, tokens.len()],
            // Conversion is done here only because model input is I32 (but it should be U32)
            tokens.iter().map(|it| *it as i32).collect(),
        )
        .unwrap();
        let offset = kv_cache.get(0).map(|it| it.shape()[1]).unwrap_or(0);
        let mut tokens = tokens;

        if tokens.shape()[1] > initial_token_length {
            tokens = tokens.slice(s![.., -1]).to_owned().insert_axis(Axis(0));
        }

        let pos_emb = self
            .positional_embedding
            .slice(s![.., offset..offset + tokens.shape()[1], ..])
            .to_owned();

        let mut inputs = tvec![
            tokens.into_tvalue(),
            audio_embedding.clone().into_tvalue(),
            pos_emb.into_tvalue(),
        ];

        inputs.extend(kv_cache);
        let out = self.decoder.run(inputs).unwrap();
        let logits = out[0].to_array_view::<f32>().unwrap().to_owned();
        let new_kv_cache = out[1..].iter().map(|it| it.clone().into()).collect();
        (logits, new_kv_cache)
    }
}

pub fn load_pos_embedding<P: AsRef<Path>>(pos_emb_path: P) -> Array3<f32> {
    let file = File::open(pos_emb_path).expect("Failed to open file");
    let mut npz = NpzReader::new(file).expect("Failed to read NPZ file");
    let pos_emb: Array2<f32> = npz.by_index(0).unwrap();
    pos_emb.insert_axis(Axis(0))
}

pub fn load_model(model_path: &Path) -> TypedSimplePlan<TypedModel> {
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

    typed_model
        .into_optimized()
        .unwrap()
        .into_runnable()
        .unwrap()
}
