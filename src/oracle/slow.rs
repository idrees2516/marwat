use crate::types::{Pool, Result, PanopticError};
use ethers::types::{U256, Address};
use std::collections::HashMap;
use tokio::sync::RwLock;
use std::sync::Arc;

pub struct SlowOracle {
    prices: Arc<RwLock<HashMap<Address, Vec<U256>>>>,
    window_size: usize,
    min_observations: usize,
    max_gap: u64,
    weights: Vec<f64>,
}

impl SlowOracle {
    pub fn new(
        window_size: usize,
        min_observations: usize,
        max_gap: u64,
    ) -> Self {
        let weights = (0..window_size)
            .map(|i| (i + 1) as f64 / window_size as f64)
            .collect();

        Self {
            prices: Arc::new(RwLock::new(HashMap::new())),
            window_size,
            min_observations,
            max_gap,
            weights,
        }
    }

    pub async fn update_price(&mut self, pool: &Pool, price: U256) -> Result<()> {
        let mut prices = self.prices.write().await;
        let history = prices.entry(pool.address).or_insert_with(Vec::new);
        
        history.push(price);
        if history.len() > self.window_size {
            history.remove(0);
        }

        Ok(())
    }

    pub async fn get_twap(&self, pool: &Pool) -> Result<U256> {
        let prices = self.prices.read().await;
        let history = prices.get(&pool.address)
            .ok_or(PanopticError::PriceNotFound)?;

        if history.len() < self.min_observations {
            return Err(PanopticError::InsufficientData);
        }

        let weighted_sum = history.iter().zip(&self.weights)
            .fold(U256::zero(), |acc, (price, weight)| {
                acc + (*price as f64 * weight) as u128
            });

        let total_weight: f64 = self.weights[..history.len()].iter().sum();
        Ok(weighted_sum / U256::from((total_weight * 1e18) as u128))
    }

    pub async fn get_volatility(&self, pool: &Pool) -> Result<f64> {
        let prices = self.prices.read().await;
        let history = prices.get(&pool.address)
            .ok_or(PanopticError::PriceNotFound)?;

        if history.len() < self.min_observations {
            return Err(PanopticError::InsufficientData);
        }

        let returns: Vec<f64> = history.windows(2)
            .map(|window| {
                (window[1].as_u128() as f64 / window[0].as_u128() as f64).ln()
            })
            .collect();

        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>()
            / (returns.len() - 1) as f64;

        Ok((variance * 365.0).sqrt())
    }

    pub async fn validate_price(&self, pool: &Pool, price: U256) -> Result<bool> {
        let twap = self.get_twap(pool).await?;
        let deviation = if price > twap {
            price - twap
        } else {
            twap - price
        };

        let max_deviation = twap / U256::from(10); // 10% deviation threshold
        Ok(deviation <= max_deviation)
    }

    pub async fn calculate_confidence(&self, pool: &Pool) -> Result<f64> {
        let prices = self.prices.read().await;
        let history = prices.get(&pool.address)
            .ok_or(PanopticError::PriceNotFound)?;

        if history.len() < self.min_observations {
            return Ok(0.0);
        }

        let gaps = history.windows(2)
            .map(|window| {
                (window[1].as_u128() as f64 / window[0].as_u128() as f64).abs()
            })
            .collect::<Vec<_>>();

        let mean_gap = gaps.iter().sum::<f64>() / gaps.len() as f64;
        let confidence = 1.0 / (1.0 + mean_gap.ln().abs());

        Ok(confidence.min(1.0).max(0.0))
    }
}
