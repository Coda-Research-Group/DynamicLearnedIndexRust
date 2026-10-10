mod bucket;
mod clustering;
pub mod cold_storage;
// mod compaction_strategy;
mod constants;
mod errors;
pub mod index;
mod level_index;
pub mod model;
mod sampling;
pub mod structs;
pub mod types;
pub use bucket::{Bucket, BucketKind, BufferKind, StorageContainer};
pub use errors::{DliError, DliResult};
pub use index::{CompactionStrategy, Index, IndexBuilder};
pub use model::{ModelConfig, ModelDevice, ModelLayer, TrainParams};
pub use structs::{
    CompactionEvent, DeleteMethod, DeleteOutcome, DistanceFn, IndexConfig, RebuildEvent,
    RebuildStrategy, SearchParams, SearchParamsT, SearchStats, SearchStrategy,
};
pub use types::{Array, ArrayNumType, ArraySlice, Id};

// Always measures: it is only used on the compaction and training path, and the
// durations are returned to the caller in `CompactionEvent`.
macro_rules! timed {
    ($e:expr) => {{
        let __start = std::time::Instant::now();
        (($e), __start.elapsed())
    }};
}
pub(crate) use timed;
