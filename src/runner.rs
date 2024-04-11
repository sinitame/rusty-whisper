use std::{fs::File, path::Path, sync::Arc};

use ndarray::{Array2, Array3, ArrayBase, Axis, Dim, IxDynImpl, OwnedRepr};
use ndarray_npy::NpzReader;
use tract_core::{
    prelude::{Framework, Tensor, TypedModel, TypedRunnableModel, TypedSimplePlan},
    tract_data::tvec,
    transform::get_transform,
};
use tract_onnx::prelude::InferenceModelExt;

pub struct WhisperRunner {
    encoder: Arc<TypedRunnableModel<TypedModel>>,
    pub decoder: Arc<TypedRunnableModel<TypedModel>>,
    pub positional_embedding: Array3<f32>,
}

impl WhisperRunner {
    pub fn new(
        encoder: Arc<TypedRunnableModel<TypedModel>>,
        decoder: Arc<TypedRunnableModel<TypedModel>>,
        positional_embedding: Array3<f32>,
    ) -> Self {
        Self {
            encoder,
            decoder,
            positional_embedding,
        }
    }

    pub fn compute_audio_features(
        &self,
        audio_features: Array2<f32>,
    ) -> ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>> {
        let mel: Tensor = audio_features.insert_axis(Axis(0)).into();
        let inputs = tvec!(mel.into());
        let encoder_out = self.encoder.run(inputs).unwrap()[0]
            .to_array_view::<f32>()
            .unwrap()
            .to_owned();

        encoder_out
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
