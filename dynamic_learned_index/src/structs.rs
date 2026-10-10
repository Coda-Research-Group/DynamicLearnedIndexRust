use serde::{Deserialize, Serialize};
use std::{borrow::Cow, path::PathBuf, time::Duration};

use crate::{
    constants::{
        DEFAULT_ARITY, DEFAULT_BUCKET_SIZE, DEFAULT_BUFFER_SIZE, DEFAULT_INPUT_SHAPE,
        DEFAULT_SEARCH_K, DEFAULT_SEARCH_N_CANDIDATES,
    },
    model::{InsertTimings, TrainTimings},
    ModelConfig, ModelDevice,
};

use half::f16;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LevelIndexConfig {
    pub model: ModelConfig,
    pub bucket_size: usize,
}

impl Default for LevelIndexConfig {
    fn default() -> Self {
        Self {
            model: ModelConfig::default(),
            bucket_size: DEFAULT_BUCKET_SIZE,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub enum RebuildStrategy {
    #[default]
    #[serde(rename = "no_rebuild")]
    NoRebuild,
    #[serde(rename = "basic_rebuild")]
    BasicRebuild,
    #[serde(rename = "greedy_rebuild")]
    GreedyRebuild,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type", content = "rebuild_strategy")]
pub enum CompactionStrategyConfig {
    #[serde(rename = "bentley_saxe")]
    BentleySaxe(RebuildStrategy),
}

impl Default for CompactionStrategyConfig {
    fn default() -> Self {
        CompactionStrategyConfig::BentleySaxe(Default::default())
    }
}

impl From<&str> for CompactionStrategyConfig {
    fn from(val: &str) -> Self {
        match val {
            "bentley_saxe:no_rebuild" => {
                CompactionStrategyConfig::BentleySaxe(RebuildStrategy::NoRebuild)
            }
            "bentley_saxe:basic_rebuild" => {
                CompactionStrategyConfig::BentleySaxe(RebuildStrategy::BasicRebuild)
            }
            "bentley_saxe:greedy_rebuild" => {
                CompactionStrategyConfig::BentleySaxe(RebuildStrategy::GreedyRebuild)
            }
            _ => panic!("Unknown compaction strategy: {val}"),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct IndexConfig {
    pub compaction_strategy: CompactionStrategyConfig,
    pub levels: LevelIndexConfig,
    pub buffer_size: usize,
    pub input_shape: usize,
    pub arity: usize,
    pub device: ModelDevice,
    pub distance_fn: DistanceFn,
    pub delete_method: DeleteMethod,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            compaction_strategy: Default::default(),
            levels: Default::default(),
            buffer_size: DEFAULT_BUFFER_SIZE,
            input_shape: DEFAULT_INPUT_SHAPE,
            arity: DEFAULT_ARITY,
            device: Default::default(),
            distance_fn: Default::default(),
            delete_method: Default::default(),
        }
    }
}

#[derive(Default, Deserialize, Serialize, Debug, Clone, Copy)]
pub enum DistanceFn {
    #[serde(rename = "l2")]
    L2,
    #[default]
    #[serde(rename = "dot")]
    Dot,
}

impl From<DistanceFn> for LabelMethod {
    fn from(val: DistanceFn) -> Self {
        match val {
            DistanceFn::L2 => LabelMethod::KMeans,
            DistanceFn::Dot => LabelMethod::SphericalKMeans,
        }
    }
}

impl From<&str> for DistanceFn {
    fn from(val: &str) -> Self {
        match val {
            "l2" => DistanceFn::L2,
            "dot" => DistanceFn::Dot,
            _ => panic!("Unknown distance function: {val}"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum LabelMethod {
    KMeans,
    SphericalKMeans,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum DeleteMethod {
    #[default]
    #[serde(rename = "oid_to_bucket")]
    OidToBucket,
}

impl From<&str> for DeleteMethod {
    fn from(val: &str) -> Self {
        match val {
            "oid_to_bucket" => DeleteMethod::OidToBucket,
            _ => panic!("Unknown delete method: {val}"),
        }
    }
}

pub struct SearchParams {
    pub k: usize,
    pub search_strategy: SearchStrategy,
}

pub trait SearchParamsT {
    fn into_search_params(self) -> SearchParams;
}

impl SearchParamsT for SearchParams {
    fn into_search_params(self) -> SearchParams {
        self
    }
}

impl SearchParamsT for () {
    fn into_search_params(self) -> SearchParams {
        SearchParams {
            k: DEFAULT_SEARCH_K,
            search_strategy: SearchStrategy::default(),
        }
    }
}

impl SearchParamsT for (usize, SearchStrategy) {
    fn into_search_params(self) -> SearchParams {
        SearchParams {
            k: self.0,
            search_strategy: self.1,
        }
    }
}

impl SearchParamsT for usize {
    fn into_search_params(self) -> SearchParams {
        SearchParams {
            k: self,
            search_strategy: SearchStrategy::default(),
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub enum SearchStrategy {
    ModelDriven(usize),
}

impl Default for SearchStrategy {
    fn default() -> Self {
        SearchStrategy::ModelDriven(DEFAULT_SEARCH_N_CANDIDATES)
    }
}

#[cfg(any(feature = "candle", feature = "mix"))]
pub trait CandleFloat: candle_core::WithDType {}
#[cfg(not(any(feature = "candle", feature = "mix")))]
pub trait CandleFloat {}

// Blanket impls
#[cfg(any(feature = "candle", feature = "mix"))]
impl<T: candle_core::WithDType> CandleFloat for T {}
#[cfg(not(any(feature = "candle", feature = "mix")))]
impl<T> CandleFloat for T {}

#[cfg(any(feature = "tch", feature = "mix"))]
pub trait TchFloat: tch::kind::Element {}
#[cfg(not(any(feature = "tch", feature = "mix")))]
pub trait TchFloat {}

#[cfg(any(feature = "tch", feature = "mix"))]
impl<T: tch::kind::Element> TchFloat for T {}
#[cfg(not(any(feature = "tch", feature = "mix")))]
impl<T> TchFloat for T {}

pub trait FloatElement: bytemuck::Pod + Default + CandleFloat + TchFloat {
    fn to_f32_slice(slice: &[Self]) -> Cow<'_, [f32]>;

    #[cfg(any(feature = "candle", feature = "mix"))]
    fn to_candle_dtype() -> candle_core::DType;

    fn zero() -> Self;
}

impl FloatElement for f32 {
    fn to_f32_slice(slice: &[Self]) -> Cow<'_, [f32]> {
        Cow::Borrowed(slice)
    }

    #[cfg(any(feature = "candle", feature = "mix"))]
    fn to_candle_dtype() -> candle_core::DType {
        candle_core::DType::F32
    }

    fn zero() -> Self {
        0.0_f32
    }
}

impl FloatElement for f16 {
    fn to_f32_slice(slice: &[Self]) -> Cow<'_, [f32]> {
        // HalfFloatVecExt::to_f32_slice is not zero-copy, it allocates a new Vec<f32> and converts each f16 to f32
        let mut v = Vec::with_capacity(slice.len());
        v.extend(slice.iter().map(|x| x.to_f32()));
        Cow::Owned(v)
    }

    #[cfg(any(feature = "candle", feature = "mix"))]
    fn to_candle_dtype() -> candle_core::DType {
        candle_core::DType::F16
    }

    fn zero() -> Self {
        f16::from_f32(0.0)
    }
}

pub(crate) type Records2Visit = Vec<(usize, usize, f32, usize)>;

/// What one search did, returned by `Index::search_with_stats`.
#[derive(Debug, Clone, Default)]
pub struct SearchStats {
    /// Evaluating the models of all levels.
    pub bucket_selection: Duration,
    /// Ordering the buckets and choosing the ones to visit.
    pub bucket_ordering: Duration,
    /// The k-NN search over the records of the visited buckets.
    pub scan: Duration,
    /// Number of visited buckets, the buffer included.
    pub visited_buckets: usize,
    /// Number of records in the visited buckets.
    pub visited_records: usize,
}

/// One insertion of data into a level, caused by a compaction or a rebuild.
#[derive(Debug, Clone, Copy, Default)]
pub struct CompactionEvent {
    /// Level the data was inserted into.
    pub level_idx: usize,
    pub n_inserted: usize,
    /// The level was created for this insertion.
    pub allocated_new_level: bool,
    /// The level existed and was empty, so its model is subject to the retrain strategy.
    pub is_retraining: bool,
    pub allocation: Duration,
    pub train: TrainTimings,
    pub insert: InsertTimings,
}

/// A rebuild triggered by a delete that left a level under-utilized.
#[derive(Debug, Clone, Default)]
pub struct RebuildEvent {
    /// The under-utilized level that triggered the rebuild.
    pub degenerated_level: usize,
    /// Levels whose data was moved. Empty when the buffer was flushed instead.
    pub source_levels: Vec<usize>,
    /// The buffer was flushed into the under-utilized level.
    pub from_buffer: bool,
    /// The insertion into the target level.
    pub moved: CompactionEvent,
    pub total: Duration,
}

/// What one delete did, returned by `Index::delete_with_stats`.
#[derive(Debug, Clone, Default)]
pub struct DeleteOutcome {
    pub deleted: bool,
    /// Level the record was deleted from; `None` if it was in the buffer or not found.
    pub level: Option<usize>,
    /// Finding and removing the record.
    pub lookup: Duration,
    /// `None` if no rebuild was needed or the rebuild strategy is `no_rebuild`.
    pub rebuild: Option<RebuildEvent>,
    pub total: Duration,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DiskBucket {
    pub bucket_idx: usize,
    pub records_offset: u64,
    pub ids_offset: u64,
    pub count: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum DiskLevelIndex {
    Hot(HotDiskLevelIndex),
    Cold(ColdDiskLevelIndex),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HotDiskLevelIndex {
    pub records_path: PathBuf,
    pub ids_path: PathBuf,
    pub buckets: Vec<DiskBucket>,
    pub config: LevelIndexConfig,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ColdDiskLevelIndex {
    pub config: LevelIndexConfig,
    /// Path to the cold storage data file for this level, if it is cold.
    pub cold_data_path: PathBuf,
    pub ids: Vec<u32>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DiskBuffer {
    pub records_path: PathBuf,
    pub ids_path: PathBuf,
    pub data: DiskBucket,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DiskIndex {
    pub compaction_strategy: CompactionStrategyConfig,
    pub levels_config: LevelIndexConfig,
    pub buffer_size: usize,
    pub input_shape: usize,
    pub arity: usize,
    pub distance_fn: DistanceFn,
    pub delete_method: DeleteMethod,
    pub levels: Vec<DiskLevelIndex>,
    pub disk_buffer: DiskBuffer,
    /// Byte budget for the index-level LRU bucket cache.
    pub cold_cache_size_bytes: u64,
    /// Base directory for per-level cold storage files (`{dir}/cold_level_{i}.bin`).
    pub cold_storage_dir: Option<PathBuf>,
    /// Index levels >= this threshold are stored cold. usize::MAX means no cold levels.
    pub cold_threshold_level: Option<usize>,
    // pub cold_storage_data_path: Option<PathBuf>,
}
