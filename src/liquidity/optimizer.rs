use crate::types::{Pool, Result, PanopticError};
use crate::pricing::models::OptionPricing;
use ethers::types::{U256, Address};
use std::collections::HashMap;
use std::sync::Arc;

pub struct LiquidityOptimizer {
    pricing: Arc<dyn OptionPricing>,
    target_utilization: f64,
    min_yield: f64,
    max_slippage: f64,
    rebalance_interval: u64,
    pool_metrics: HashMap<Address, PoolMetrics>,
}

struct PoolMetrics {
    historical_utilization: Vec<(f64, u64)>,
    yield_metrics: Vec<(f64, u64)>,
    slippage_metrics: Vec<(f64, u64)>,
    last_optimization: u64,
}

impl LiquidityOptimizer {
    pub fn new(
        pricing: Arc<dyn OptionPricing>,
        target_utilization: f64,
        min_yield: f64,
        max_slippage: f64,
        rebalance_interval: u64,
    ) -> Self {
        Self {
            pricing,
            target_utilization,
            min_yield,
            max_slippage,
            rebalance_interval,
            pool_metrics: HashMap::new(),
        }
    }

    pub fn update_metrics(
        &mut self,
        pool: &Pool,
        utilization: f64,
        yield_rate: f64,
        slippage: f64,
        timestamp: u64,
    ) -> Result<()> {
        let metrics = self.pool_metrics.entry(pool.address)
            .or_insert_with(|| PoolMetrics {
                historical_utilization: Vec::new(),
                yield_metrics: Vec::new(),
                slippage_metrics: Vec::new(),
                last_optimization: 0,
            });

        metrics.historical_utilization.push((utilization, timestamp));
        metrics.yield_metrics.push((yield_rate, timestamp));
        metrics.slippage_metrics.push((slippage, timestamp));

        // Keep only last 30 days of data
        let cutoff = timestamp.saturating_sub(30 * 24 * 3600);
        metrics.historical_utilization.retain(|(_, t)| *t > cutoff);
        metrics.yield_metrics.retain(|(_, t)| *t > cutoff);
        metrics.slippage_metrics.retain(|(_, t)| *t > cutoff);

        Ok(())
    }

    pub fn optimize_liquidity(
        &mut self,
        pool: &Pool,
        current_liquidity: U256,
        timestamp: u64,
    ) -> Result<U256> {
        let metrics = self.pool_metrics.get_mut(&pool.address)
            .ok_or(PanopticError::PoolNotFound)?;

        if timestamp - metrics.last_optimization < self.rebalance_interval {
            return Ok(current_liquidity);
        }

        // Calculate average metrics
        let avg_utilization = metrics.historical_utilization.iter()
            .map(|(u, _)| u)
            .sum::<f64>() / metrics.historical_utilization.len() as f64;

        let avg_yield = metrics.yield_metrics.iter()
            .map(|(y, _)| y)
            .sum::<f64>() / metrics.yield_metrics.len() as f64;

        let avg_slippage = metrics.slippage_metrics.iter()
            .map(|(s, _)| s)
            .sum::<f64>() / metrics.slippage_metrics.len() as f64;

        // Check if optimization is needed
        if avg_yield < self.min_yield || avg_slippage > self.max_slippage {
            return Ok(current_liquidity);
        }

        // Calculate optimal liquidity based on target utilization
        let utilization_ratio = self.target_utilization / avg_utilization;
        let optimal_liquidity = (current_liquidity.as_u128() as f64 * utilization_ratio) as u128;

        // Apply constraints from pricing model
        let model_optimal = self.pricing.calculate_optimal_liquidity(pool)?;
        let final_liquidity = U256::from(optimal_liquidity.min(model_optimal.as_u128()));

        metrics.last_optimization = timestamp;

        Ok(final_liquidity)
    }

    pub fn calculate_yield_metrics(
        &self,
        pool: &Pool,
        period: u64,
    ) -> Result<(f64, f64, f64)> {
        let metrics = self.pool_metrics.get(&pool.address)
            .ok_or(PanopticError::PoolNotFound)?;

        let recent_yields: Vec<f64> = metrics.yield_metrics.iter()
            .filter(|(_, t)| *t > period)
            .map(|(y, _)| *y)
            .collect();

        if recent_yields.is_empty() {
            return Ok((0.0, 0.0, 0.0));
        }

        let avg_yield = recent_yields.iter().sum::<f64>() / recent_yields.len() as f64;
        let min_yield = recent_yields.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_yield = recent_yields.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        Ok((avg_yield, min_yield, max_yield))
    }

    pub fn calculate_slippage_metrics(
        &self,
        pool: &Pool,
        period: u64,
    ) -> Result<(f64, f64, f64)> {
        let metrics = self.pool_metrics.get(&pool.address)
            .ok_or(PanopticError::PoolNotFound)?;

        let recent_slippage: Vec<f64> = metrics.slippage_metrics.iter()
            .filter(|(_, t)| *t > period)
            .map(|(s, _)| *s)
            .collect();

        if recent_slippage.is_empty() {
            return Ok((0.0, 0.0, 0.0));
        }

        let avg_slippage = recent_slippage.iter().sum::<f64>() / recent_slippage.len() as f64;
        let min_slippage = recent_slippage.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_slippage = recent_slippage.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        Ok((avg_slippage, min_slippage, max_slippage))
    }

    pub fn predict_optimal_liquidity(
        &self,
        pool: &Pool,
        timestamp: u64,
    ) -> Result<U256> {
        let metrics = self.pool_metrics.get(&pool.address)
            .ok_or(PanopticError::PoolNotFound)?;

        // Use recent data for prediction
        let recent_period = 7 * 24 * 3600; // 7 days
        let recent_utilization: Vec<(f64, u64)> = metrics.historical_utilization.iter()
            .filter(|(_, t)| *t > timestamp - recent_period)
            .cloned()
            .collect();

        if recent_utilization.is_empty() {
            return self.pricing.calculate_optimal_liquidity(pool);
        }

        // Simple linear regression for prediction
        let n = recent_utilization.len() as f64;
        let sum_x: f64 = recent_utilization.iter().map(|(_, t)| *t as f64).sum();
        let sum_y: f64 = recent_utilization.iter().map(|(u, _)| *u).sum();
        let sum_xy: f64 = recent_utilization.iter().map(|(u, t)| u * *t as f64).sum();
        let sum_xx: f64 = recent_utilization.iter().map(|(_, t)| (*t as f64).powi(2)).sum();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x.powi(2));
        let intercept = (sum_y - slope * sum_x) / n;

        // Predict utilization
        let predicted_utilization = slope * timestamp as f64 + intercept;
        let clamped_utilization = predicted_utilization.max(0.0).min(1.0);

        // Calculate optimal liquidity based on predicted utilization
        let current_optimal = self.pricing.calculate_optimal_liquidity(pool)?;
        let adjustment_factor = self.target_utilization / clamped_utilization;
        
        Ok(current_optimal * U256::from((adjustment_factor * 1e18) as u128) / U256::from(1e18 as u128))
    }
}
