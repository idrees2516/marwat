use ethers::prelude::*;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use tokio::sync::RwLock;
use std::time::{SystemTime, UNIX_EPOCH};

use super::GameError;

#[derive(Debug, Clone)]
pub struct OracleConfig {
    pub twap_window: u64,
    pub min_observations: u32,
    pub max_deviation: f64,
    pub volatility_window: u64,
    pub manipulation_threshold: f64,
}

#[derive(Debug, Clone)]
pub struct PriceObservation {
    pub timestamp: u64,
    pub price: U256,
    pub liquidity: u128,
    pub volatility: f64,
}

#[derive(Debug)]
pub struct AdvancedOracleManager {
    config: OracleConfig,
    price_observations: Arc<RwLock<BTreeMap<(Address, Address), Vec<PriceObservation>>>>,
    oracle_weights: Arc<RwLock<HashMap<Address, f64>>>,
    cross_chain_prices: Arc<RwLock<HashMap<(Address, Address, u64), PriceObservation>>>,
    compressed_history: Arc<RwLock<CompressedPriceHistory>>,
}

#[derive(Debug)]
struct CompressedPriceHistory {
    compression_ratio: f64,
    max_compressed_points: usize,
    compressed_points: BTreeMap<u64, CompressedPoint>,
}

#[derive(Debug, Clone)]
struct CompressedPoint {
    min_price: U256,
    max_price: U256,
    avg_price: U256,
    total_liquidity: u128,
    observation_count: u32,
}

impl AdvancedOracleManager {
    pub fn new(config: OracleConfig) -> Self {
        Self {
            config,
            price_observations: Arc::new(RwLock::new(BTreeMap::new())),
            oracle_weights: Arc::new(RwLock::new(HashMap::new())),
            cross_chain_prices: Arc::new(RwLock::new(HashMap::new())),
            compressed_history: Arc::new(RwLock::new(CompressedPriceHistory {
                compression_ratio: 0.1,  // 10:1 compression
                max_compressed_points: 1000,
                compressed_points: BTreeMap::new(),
            })),
        }
    }

    pub async fn add_price_observation(
        &self,
        token0: Address,
        token1: Address,
        price: U256,
        liquidity: u128,
    ) -> Result<(), GameError> {
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let volatility = self.calculate_recent_volatility(token0, token1).await?;

        let observation = PriceObservation {
            timestamp: current_time,
            price,
            liquidity,
            volatility,
        };

        let mut observations = self.price_observations.write().await;
        let token_observations = observations
            .entry((token0, token1))
            .or_insert_with(Vec::new);

        // Add new observation
        token_observations.push(observation.clone());

        // Remove old observations
        token_observations.retain(|obs| 
            current_time - obs.timestamp <= self.config.twap_window);

        // Compress old data
        self.compress_historical_data(token0, token1, observation).await?;

        Ok(())
    }

    async fn compress_historical_data(
        &self,
        token0: Address,
        token1: Address,
        new_observation: PriceObservation,
    ) -> Result<(), GameError> {
        let mut compressed = self.compressed_history.write().await;
        let compression_window = (self.config.twap_window as f64 * compressed.compression_ratio) as u64;
        let window_timestamp = new_observation.timestamp - (new_observation.timestamp % compression_window);

        let point = compressed.compressed_points
            .entry(window_timestamp)
            .or_insert(CompressedPoint {
                min_price: new_observation.price,
                max_price: new_observation.price,
                avg_price: new_observation.price,
                total_liquidity: new_observation.liquidity,
                observation_count: 1,
            });

        // Update compressed point
        point.min_price = point.min_price.min(new_observation.price);
        point.max_price = point.max_price.max(new_observation.price);
        point.avg_price = (point.avg_price * U256::from(point.observation_count) + new_observation.price) 
            / U256::from(point.observation_count + 1);
        point.total_liquidity += new_observation.liquidity;
        point.observation_count += 1;

        // Remove old compressed points
        let oldest_allowed = window_timestamp - (self.config.twap_window * 10);  // Keep 10x window worth of compressed data
        compressed.compressed_points.retain(|&ts, _| ts >= oldest_allowed);

        // Limit total compressed points
        while compressed.compressed_points.len() > compressed.max_compressed_points {
            if let Some((&oldest_key, _)) = compressed.compressed_points.iter().next() {
                compressed.compressed_points.remove(&oldest_key);
            }
        }

        Ok(())
    }

    pub async fn calculate_twap(
        &self,
        token0: Address,
        token1: Address,
        window: Option<u64>,
    ) -> Result<U256, GameError> {
        let observations = self.price_observations.read().await;
        let token_observations = observations
            .get(&(token0, token1))
            .ok_or(GameError::InsufficientData)?;

        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let window = window.unwrap_or(self.config.twap_window);
        let cutoff_time = current_time - window;

        let relevant_observations: Vec<_> = token_observations.iter()
            .filter(|obs| obs.timestamp >= cutoff_time)
            .collect();

        if relevant_observations.len() < self.config.min_observations as usize {
            return Err(GameError::InsufficientData);
        }

        // Calculate time-weighted average
        let mut weighted_sum = U256::zero();
        let mut total_weight = U256::zero();

        for window in relevant_observations.windows(2) {
            let time_weight = window[1].timestamp - window[0].timestamp;
            let price_avg = (window[0].price + window[1].price) / U256::from(2);
            
            weighted_sum += price_avg * U256::from(time_weight);
            total_weight += U256::from(time_weight);
        }

        if total_weight == U256::zero() {
            return Ok(relevant_observations.last().unwrap().price);
        }

        Ok(weighted_sum / total_weight)
    }

    pub async fn calculate_geometric_mean_price(
        &self,
        token0: Address,
        token1: Address,
        window: u64,
    ) -> Result<U256, GameError> {
        let observations = self.price_observations.read().await;
        let token_observations = observations
            .get(&(token0, token1))
            .ok_or(GameError::InsufficientData)?;

        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let cutoff_time = current_time - window;

        let relevant_observations: Vec<_> = token_observations.iter()
            .filter(|obs| obs.timestamp >= cutoff_time)
            .collect();

        if relevant_observations.len() < self.config.min_observations as usize {
            return Err(GameError::InsufficientData);
        }

        // Calculate geometric mean
        let mut product = U256::from(1000000);  // Start with 1.0 in fixed point
        let count = relevant_observations.len() as u32;

        for obs in relevant_observations {
            product = product
                .saturating_mul(obs.price.integer_sqrt())
                .saturating_div(U256::from(1000));  // Maintain fixed point
        }

        // Apply power of 1/n
        let result = product.integer_sqrt();
        Ok(result)
    }

    pub async fn verify_cross_chain_price(
        &self,
        token0: Address,
        token1: Address,
        chain_id: u64,
        price: U256,
        timestamp: u64,
    ) -> Result<bool, GameError> {
        let cross_chain_prices = self.cross_chain_prices.read().await;
        
        if let Some(stored_price) = cross_chain_prices.get(&(token0, token1, chain_id)) {
            let price_diff = if price > stored_price.price {
                price - stored_price.price
            } else {
                stored_price.price - price
            };

            let price_deviation = price_diff
                .saturating_mul(U256::from(1000000))
                .saturating_div(stored_price.price);

            if price_deviation > U256::from((self.config.max_deviation * 1000000.0) as u64) {
                return Ok(false);
            }

            let time_diff = if timestamp > stored_price.timestamp {
                timestamp - stored_price.timestamp
            } else {
                stored_price.timestamp - timestamp
            };

            if time_diff > self.config.twap_window {
                return Ok(false);
            }
        }

        Ok(true)
    }

    pub async fn update_oracle_weight(
        &self,
        oracle: Address,
        weight: f64,
    ) -> Result<(), GameError> {
        let mut weights = self.oracle_weights.write().await;
        weights.insert(oracle, weight);
        Ok(())
    }

    pub async fn calculate_recent_volatility(
        &self,
        token0: Address,
        token1: Address,
    ) -> Result<f64, GameError> {
        let observations = self.price_observations.read().await;
        let token_observations = observations
            .get(&(token0, token1))
            .ok_or(GameError::InsufficientData)?;

        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let cutoff_time = current_time - self.config.volatility_window;

        let relevant_observations: Vec<_> = token_observations.iter()
            .filter(|obs| obs.timestamp >= cutoff_time)
            .collect();

        if relevant_observations.len() < 2 {
            return Ok(0.0);
        }

        // Calculate returns
        let mut returns = Vec::new();
        for window in relevant_observations.windows(2) {
            let return_val = (window[1].price.as_u128() as f64 / 
                            window[0].price.as_u128() as f64) - 1.0;
            returns.push(return_val);
        }

        // Calculate volatility (standard deviation of returns)
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / returns.len() as f64;

        Ok(variance.sqrt())
    }

    pub async fn detect_manipulation(
        &self,
        token0: Address,
        token1: Address,
    ) -> Result<bool, GameError> {
        let observations = self.price_observations.read().await;
        let token_observations = observations
            .get(&(token0, token1))
            .ok_or(GameError::InsufficientData)?;

        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let cutoff_time = current_time - self.config.volatility_window;

        let relevant_observations: Vec<_> = token_observations.iter()
            .filter(|obs| obs.timestamp >= cutoff_time)
            .collect();

        if relevant_observations.len() < self.config.min_observations as usize {
            return Ok(false);
        }

        // Check for sudden price movements
        for window in relevant_observations.windows(2) {
            let price_change = (window[1].price.as_u128() as f64 / 
                              window[0].price.as_u128() as f64).abs() - 1.0;
            
            if price_change > self.config.manipulation_threshold {
                return Ok(true);
            }
        }

        // Check for price/liquidity correlation
        let price_liquidity_correlation = self.calculate_price_liquidity_correlation(
            &relevant_observations
        );

        if price_liquidity_correlation.abs() > self.config.manipulation_threshold {
            return Ok(true);
        }

        Ok(false)
    }

    fn calculate_price_liquidity_correlation(
        &self,
        observations: &[&PriceObservation],
    ) -> f64 {
        if observations.len() < 2 {
            return 0.0;
        }

        let n = observations.len() as f64;
        let prices: Vec<f64> = observations.iter()
            .map(|obs| obs.price.as_u128() as f64)
            .collect();
        let liquidities: Vec<f64> = observations.iter()
            .map(|obs| obs.liquidity as f64)
            .collect();

        let price_mean = prices.iter().sum::<f64>() / n;
        let liquidity_mean = liquidities.iter().sum::<f64>() / n;

        let mut covariance = 0.0;
        let mut price_variance = 0.0;
        let mut liquidity_variance = 0.0;

        for i in 0..observations.len() {
            let price_diff = prices[i] - price_mean;
            let liquidity_diff = liquidities[i] - liquidity_mean;
            
            covariance += price_diff * liquidity_diff;
            price_variance += price_diff * price_diff;
            liquidity_variance += liquidity_diff * liquidity_diff;
        }

        covariance /= n;
        price_variance /= n;
        liquidity_variance /= n;

        if price_variance == 0.0 || liquidity_variance == 0.0 {
            return 0.0;
        }

        covariance / (price_variance * liquidity_variance).sqrt()
    }
}
