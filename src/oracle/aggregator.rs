use super::{FastOracle, SlowOracle};
use crate::types::{Pool, Result, PanopticError};
use ethers::types::{U256, Address};
use std::collections::HashMap;
use tokio::sync::RwLock;
use std::sync::Arc;

pub struct PriceAggregator {
    fast_oracle: Arc<RwLock<FastOracle>>,
    slow_oracle: Arc<RwLock<SlowOracle>>,
    sources: HashMap<Address, Vec<Box<dyn PriceSource>>>,
    weights: HashMap<Address, Vec<f64>>,
    min_sources: usize,
    max_deviation: U256,
}

#[async_trait::async_trait]
pub trait PriceSource: Send + Sync {
    async fn get_price(&self, pool: &Pool) -> Result<U256>;
    async fn get_confidence(&self) -> f64;
    fn get_source_type(&self) -> SourceType;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceType {
    DEX,
    CEX,
    ChainLink,
    Custom,
}

impl PriceAggregator {
    pub fn new(
        fast_oracle: Arc<RwLock<FastOracle>>,
        slow_oracle: Arc<RwLock<SlowOracle>>,
        min_sources: usize,
        max_deviation: U256,
    ) -> Self {
        Self {
            fast_oracle,
            slow_oracle,
            sources: HashMap::new(),
            weights: HashMap::new(),
            min_sources,
            max_deviation,
        }
    }

    pub fn add_source(
        &mut self,
        pool: &Pool,
        source: Box<dyn PriceSource>,
        weight: f64,
    ) -> Result<()> {
        let sources = self.sources.entry(pool.address).or_insert_with(Vec::new);
        let weights = self.weights.entry(pool.address).or_insert_with(Vec::new);
        
        sources.push(source);
        weights.push(weight);

        Ok(())
    }

    pub async fn aggregate_price(&self, pool: &Pool) -> Result<U256> {
        let sources = self.sources.get(&pool.address)
            .ok_or(PanopticError::NoDataSources)?;
        let weights = self.weights.get(&pool.address)
            .ok_or(PanopticError::NoDataSources)?;

        if sources.len() < self.min_sources {
            return Err(PanopticError::InsufficientSources);
        }

        let mut prices = Vec::new();
        let mut total_weight = 0.0;
        let mut weighted_sum = U256::zero();

        for (source, &weight) in sources.iter().zip(weights.iter()) {
            let price = source.get_price(pool).await?;
            let confidence = source.get_confidence().await;
            let adjusted_weight = weight * confidence;

            prices.push(price);
            total_weight += adjusted_weight;
            weighted_sum += price * U256::from((adjusted_weight * 1e18) as u128);
        }

        // Validate price consistency
        let mean_price = weighted_sum / U256::from((total_weight * 1e18) as u128);
        for price in prices {
            let deviation = if price > mean_price {
                price - mean_price
            } else {
                mean_price - price
            };

            if deviation > self.max_deviation {
                return Err(PanopticError::PriceDeviation);
            }
        }

        Ok(mean_price)
    }

    pub async fn update_fast_oracle(&self, pool: &Pool) -> Result<()> {
        let price = self.aggregate_price(pool).await?;
        let timestamp = chrono::Utc::now().timestamp() as u64;
        
        self.fast_oracle.write().await
            .update_price(pool, price, timestamp).await
    }

    pub async fn update_slow_oracle(&self, pool: &Pool) -> Result<()> {
        let price = self.aggregate_price(pool).await?;
        
        self.slow_oracle.write().await
            .update_price(pool, price).await
    }

    pub async fn get_price(&self, pool: &Pool) -> Result<U256> {
        let fast_price = self.fast_oracle.read().await.get_price(pool).await?;
        let slow_price = self.slow_oracle.read().await.get_twap(pool).await?;

        let deviation = if fast_price > slow_price {
            fast_price - slow_price
        } else {
            slow_price - fast_price
        };

        if deviation > self.max_deviation {
            Ok(slow_price)
        } else {
            Ok(fast_price)
        }
    }

    pub async fn get_volatility(&self, pool: &Pool) -> Result<f64> {
        let fast_vol = self.fast_oracle.read().await
            .calculate_volatility(pool, 3600)?;
        let slow_vol = self.slow_oracle.read().await
            .get_volatility(pool).await?;

        Ok((fast_vol + slow_vol) / 2.0)
    }

    pub async fn detect_manipulation(&self, pool: &Pool) -> Result<bool> {
        let fast_manipulation = self.fast_oracle.read().await
            .detect_manipulation(pool, 3600)?;
        
        if fast_manipulation {
            return Ok(true);
        }

        let price_confidence = self.slow_oracle.read().await
            .calculate_confidence(pool).await?;
        
        Ok(price_confidence < 0.5)
    }
}
