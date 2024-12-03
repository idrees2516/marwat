use crate::types::{Pool, Result, PanopticError};
use ethers::types::{U256, Address};
use std::collections::HashMap;
use tokio::sync::RwLock;
use std::sync::Arc;

pub struct FastOracle {
    prices: Arc<RwLock<HashMap<Address, U256>>>,
    update_threshold: U256,
    last_updates: HashMap<Address, u64>,
    heartbeat: u64,
    deviation_threshold: U256,
    min_observations: usize,
    price_history: HashMap<Address, Vec<(U256, u64)>>,
}

impl FastOracle {
    pub fn new(
        update_threshold: U256,
        heartbeat: u64,
        deviation_threshold: U256,
        min_observations: usize,
    ) -> Self {
        Self {
            prices: Arc::new(RwLock::new(HashMap::new())),
            update_threshold,
            last_updates: HashMap::new(),
            heartbeat,
            deviation_threshold,
            min_observations,
            price_history: HashMap::new(),
        }
    }

    pub async fn update_price(&mut self, pool: &Pool, price: U256, timestamp: u64) -> Result<()> {
        let current_price = self.prices.read().await.get(&pool.address).copied();
        
        if let Some(current) = current_price {
            let price_change = if price > current {
                price - current
            } else {
                current - price
            };
            
            let percentage_change = price_change
                .checked_mul(U256::from(10000))
                .ok_or(PanopticError::MultiplicationOverflow)?
                .checked_div(current)
                .ok_or(PanopticError::DivisionByZero)?;
                
            if percentage_change < self.deviation_threshold {
                return Ok(());
            }
        }

        let last_update = self.last_updates.get(&pool.address).copied().unwrap_or(0);
        if timestamp - last_update < self.heartbeat {
            return Ok(());
        }

        self.prices.write().await.insert(pool.address, price);
        self.last_updates.insert(pool.address, timestamp);
        
        let history = self.price_history.entry(pool.address).or_insert_with(Vec::new);
        history.push((price, timestamp));
        
        if history.len() > 1000 {
            history.remove(0);
        }

        Ok(())
    }

    pub async fn get_price(&self, pool: &Pool) -> Result<U256> {
        self.prices
            .read()
            .await
            .get(&pool.address)
            .copied()
            .ok_or(PanopticError::PriceNotFound)
    }

    pub fn get_historical_prices(&self, pool: &Pool, lookback: u64) -> Vec<(U256, u64)> {
        self.price_history
            .get(&pool.address)
            .map(|prices| {
                prices
                    .iter()
                    .filter(|(_, timestamp)| *timestamp >= lookback)
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn calculate_volatility(&self, pool: &Pool, window: u64) -> Result<f64> {
        let prices = self.get_historical_prices(pool, window);
        if prices.len() < self.min_observations {
            return Err(PanopticError::InsufficientData);
        }

        let returns: Vec<f64> = prices
            .windows(2)
            .map(|window| {
                let (price1, _) = window[0];
                let (price2, _) = window[1];
                (price2.as_u128() as f64 / price1.as_u128() as f64).ln()
            })
            .collect();

        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns
            .iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>()
            / (returns.len() - 1) as f64;

        Ok((variance * (365.0 * 24.0 * 60.0 * 60.0 / window as f64)).sqrt())
    }

    pub fn detect_manipulation(&self, pool: &Pool, window: u64) -> Result<bool> {
        let prices = self.get_historical_prices(pool, window);
        if prices.len() < self.min_observations {
            return Ok(false);
        }

        let returns: Vec<f64> = prices
            .windows(2)
            .map(|window| {
                let (price1, _) = window[0];
                let (price2, _) = window[1];
                (price2.as_u128() as f64 / price1.as_u128() as f64).ln()
            })
            .collect();

        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let std_dev = (returns
            .iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>()
            / (returns.len() - 1) as f64)
            .sqrt();

        let z_scores: Vec<f64> = returns.iter().map(|r| (r - mean) / std_dev).collect();
        let max_z_score = z_scores.iter().map(|z| z.abs()).fold(0.0, f64::max);

        Ok(max_z_score > 3.0)
    }
}
