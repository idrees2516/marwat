pub mod fast;
pub mod slow;
pub mod aggregator;
pub mod manipulation;
pub mod volatility;

pub use self::fast::FastOracle;
pub use self::slow::SlowOracle;
pub use self::aggregator::PriceAggregator;
pub use self::manipulation::ManipulationDetector;
pub use self::volatility::VolatilityOracle;
