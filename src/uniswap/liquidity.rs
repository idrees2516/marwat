use ethers::types::{U256, Address};
use std::sync::Arc;
use crate::types::{Result, PanopticError};
use crate::math::{sqrt_price_to_tick, tick_to_sqrt_price};
use super::pool::{UniswapV3Pool, Position};

/// Manages concentrated liquidity positions
pub struct ConcentratedLiquidityManager {
    min_tick: i32,
    max_tick: i32,
    tick_spacing: i32,
}

impl ConcentratedLiquidityManager {
    pub fn new(min_tick: i32, max_tick: i32, tick_spacing: i32) -> Self {
        Self {
            min_tick,
            max_tick,
            tick_spacing,
        }
    }

    /// Calculates optimal tick ranges for liquidity provision
    pub fn calculate_optimal_tick_ranges(
        &self,
        current_tick: i32,
        volatility: f64,
        target_range: f64,
    ) -> Result<Vec<(i32, i32)>> {
        let tick_range = (volatility * target_range * (self.tick_spacing as f64)).round() as i32;
        let tick_range = tick_range - (tick_range % self.tick_spacing);

        let lower_bound = (current_tick - tick_range).max(self.min_tick);
        let upper_bound = (current_tick + tick_range).min(self.max_tick);

        let mut ranges = Vec::new();
        let mut tick_lower = lower_bound;

        while tick_lower < upper_bound {
            let tick_upper = (tick_lower + tick_range).min(self.max_tick);
            if tick_upper - tick_lower >= self.tick_spacing {
                ranges.push((tick_lower, tick_upper));
            }
            tick_lower += self.tick_spacing;
        }

        Ok(ranges)
    }

    /// Calculates optimal liquidity distribution across tick ranges
    pub fn calculate_liquidity_distribution(
        &self,
        total_liquidity: U256,
        ranges: &[(i32, i32)],
        weights: Option<Vec<f64>>,
    ) -> Result<Vec<U256>> {
        if ranges.is_empty() {
            return Ok(vec![]);
        }

        let weights = weights.unwrap_or_else(|| {
            let weight = 1.0 / ranges.len() as f64;
            vec![weight; ranges.len()]
        });

        if weights.len() != ranges.len() {
            return Err(PanopticError::InvalidParameter("Weights length mismatch".into()));
        }

        let total_weight: f64 = weights.iter().sum();
        if (total_weight - 1.0).abs() > 1e-6 {
            return Err(PanopticError::InvalidParameter("Weights must sum to 1".into()));
        }

        let distribution = weights.iter()
            .map(|&weight| {
                let amount = (total_liquidity.as_u128() as f64 * weight).round() as u128;
                U256::from(amount)
            })
            .collect();

        Ok(distribution)
    }

    /// Calculates impermanent loss for a given price range
    pub fn calculate_impermanent_loss(
        &self,
        initial_price: U256,
        current_price: U256,
        tick_lower: i32,
        tick_upper: i32,
    ) -> Result<f64> {
        let sqrt_price_lower = tick_to_sqrt_price(tick_lower)?;
        let sqrt_price_upper = tick_to_sqrt_price(tick_upper)?;

        if current_price < sqrt_price_lower || current_price > sqrt_price_upper {
            return Ok(0.0);
        }

        let price_ratio = current_price.as_u128() as f64 / initial_price.as_u128() as f64;
        let sqrt_ratio = price_ratio.sqrt();

        // IL = 2 * sqrt(P1/P0) / (1 + P1/P0) - 1
        let il = 2.0 * sqrt_ratio / (1.0 + price_ratio) - 1.0;

        Ok(il)
    }

    /// Optimizes liquidity position based on fee income and impermanent loss
    pub fn optimize_liquidity_position(
        &self,
        pool: &UniswapV3Pool,
        position: &Position,
        tick_lower: i32,
        tick_upper: i32,
        fee_income: (U256, U256),
        time_horizon: u64,
    ) -> Result<OptimizationResult> {
        let (fee0, fee1) = fee_income;
        let current_tick = pool.tick;

        // Calculate fee APR
        let fee_apr = self.calculate_fee_apr(pool, position, fee0, fee1, time_horizon)?;

        // Calculate IL
        let il = self.calculate_impermanent_loss(
            pool.sqrt_price_x96,
            pool.sqrt_price_x96,
            tick_lower,
            tick_upper,
        )?;

        // Calculate optimal liquidity based on fee APR and IL
        let optimal_liquidity = if fee_apr > -il {
            // Increase liquidity if fee income outweighs IL
            position.liquidity.saturating_mul(U256::from(12) / U256::from(10))
        } else {
            // Decrease liquidity if IL outweighs fee income
            position.liquidity.saturating_mul(U256::from(8) / U256::from(10))
        };

        Ok(OptimizationResult {
            optimal_liquidity,
            fee_apr,
            impermanent_loss: il,
            should_rebalance: (fee_apr + il).abs() > 0.05, // 5% threshold
        })
    }

    /// Calculates fee APR for a position
    fn calculate_fee_apr(
        &self,
        pool: &UniswapV3Pool,
        position: &Position,
        fee0: U256,
        fee1: U256,
        time_horizon: u64,
    ) -> Result<f64> {
        if time_horizon == 0 {
            return Err(PanopticError::InvalidParameter("Time horizon cannot be zero".into()));
        }

        let total_value = position.liquidity;
        if total_value == U256::zero() {
            return Ok(0.0);
        }

        let fee_value = fee0.saturating_add(fee1);
        let annualized_fee = fee_value.saturating_mul(U256::from(365 * 24 * 3600)) / 
                            U256::from(time_horizon);

        Ok((annualized_fee.as_u128() as f64) / (total_value.as_u128() as f64))
    }
}

/// Result of liquidity position optimization
#[derive(Debug)]
pub struct OptimizationResult {
    pub optimal_liquidity: U256,
    pub fee_apr: f64,
    pub impermanent_loss: f64,
    pub should_rebalance: bool,
}
