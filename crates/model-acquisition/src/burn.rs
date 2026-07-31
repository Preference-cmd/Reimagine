//! Burn backend support for model acquisition and conversion.

pub mod checkpoint;

pub use checkpoint::{
    BurnCheckpointConverter, BurnConversionComponent, BurnConversionComponentRole,
    BurnConversionReport,
};
