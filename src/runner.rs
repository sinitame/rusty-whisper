use std::{fs::File, path::Path, sync::Arc};

use ndarray::{s, Array2, Array3, ArrayBase, Axis, Dim, IxDynImpl, OwnedRepr};
use ndarray_npy::NpzReader;
use tract_core::transform::get_transform;
use tract_onnx::prelude::*;

pub struct WhisperRunner {
    encoder: Arc<TypedRunnableModel<TypedModel>>,
    decoder: Arc<TypedRunnableModel<TypedModel>>,
    positional_embedding: Array3<f32>,
    kv_cache: Vec<Tensor>,
}

impl WhisperRunner {
    pub fn new(
        encoder: Arc<TypedRunnableModel<TypedModel>>,
        decoder: Arc<TypedRunnableModel<TypedModel>>,
        positional_embedding: Array3<f32>,
    ) -> Self {
        let mut default_kv_cache = Vec::new();
        for _ in 0..6 * 2 {
            default_kv_cache.push(Tensor::zero::<f32>(&[1, 0, 512]).unwrap())
        }
        Self {
            encoder,
            decoder,
            positional_embedding,
            kv_cache: default_kv_cache,
        }
    }

    pub fn compute_audio_embedding(
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
        tokens: &[i32],
        audio_embedding: &ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>,
        initial_token_length: usize,
    ) -> ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>> {
        let tokens = Array2::from_shape_vec([1, tokens.len()], tokens.to_vec()).unwrap();
        let offset = self.kv_cache.get(0).map(|it| it.shape()[1]).unwrap_or(0);
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

        let kv = self.kv_cache.drain(..).map(|it| it.into());
        inputs.extend(kv);
        let out = self.decoder.run(inputs).unwrap();
        let logits = out[0].to_array_view::<f32>().unwrap().to_owned();
        self.kv_cache = out[1..].iter().map(|it| it.clone().into_tensor()).collect();
        logits
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
