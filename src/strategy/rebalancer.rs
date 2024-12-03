use ethers::types::{U256, Address};
use std::sync::Arc;
use crate::types::{Result, PanopticError};
use crate::uniswap::{UniswapV3Pool, Position, ConcentratedLiquidityManager};
use super::amm_options::{AMMOptionStrategy, StrategyType};

/// Configuration for position rebalancing
#[derive(Debug)]
pub struct RebalanceConfig {
    pub min_rebalance_interval: u64,
    pub price_deviation_threshold: f64,
    pub delta_deviation_threshold: f64,
    pub min_profit_threshold: f64,
    pub max_slippage: f64,
}

/// Manages position rebalancing
pub struct PositionRebalancer {
    config: RebalanceConfig,
    liquidity_manager: Arc<ConcentratedLiquidityManager>,
    last_rebalance: std::collections::HashMap<Address, u64>,
}

impl PositionRebalancer {
    pub fn new(
        config: RebalanceConfig,
        liquidity_manager: Arc<ConcentratedLiquidityManager>,
    ) -> Self {
        Self {
            config,
            liquidity_manager,
            last_rebalance: std::collections::HashMap::new(),
        }
    }

    /// Checks if a position needs rebalancing
    pub fn needs_rebalancing(
        &self,
        pool: &UniswapV3Pool,
        strategy: &AMMOptionStrategy,
        current_time: u64,
    ) -> Result<bool> {
        // Check minimum time interval
        if let Some(last_time) = self.last_rebalance.get(&pool.address) {
            if current_time - last_time < self.config.min_rebalance_interval {
                return Ok(false);
            }
        }

        // Check price deviation
        let current_price = pool.sqrt_price_x96;
        let strike_price = strategy.strike_price;
        let price_deviation = self.calculate_price_deviation(current_price, strike_price)?;
        
        if price_deviation > self.config.price_deviation_threshold {
            return Ok(true);
        }

        // Check delta deviation
        let delta_deviation = self.calculate_delta_deviation(pool, strategy)?;
        if delta_deviation > self.config.delta_deviation_threshold {
            return Ok(true);
        }

        Ok(false)
    }

    /// Rebalances a position
    pub fn rebalance_position(
        &mut self,
        pool: &UniswapV3Pool,
        strategy: &mut AMMOptionStrategy,
        current_time: u64,
    ) -> Result<RebalanceAction> {
        // Calculate optimal ranges
        let (new_tick_lower, new_tick_upper) = self.calculate_optimal_range(pool, strategy)?;

        // Calculate required actions
        let actions = if new_tick_lower != strategy.tick_lower || new_tick_upper != strategy.tick_upper {
            // Need to shift position
            let liquidity_delta = self.calculate_liquidity_adjustment(
                pool,
                strategy,
                new_tick_lower,
                new_tick_upper,
            )?;

            RebalanceAction::Shift {
                old_lower: strategy.tick_lower,
                old_upper: strategy.tick_upper,
                new_lower: new_tick_lower,
                new_upper: new_tick_upper,
                liquidity_delta,
            }
        } else {
            // Only need to adjust liquidity
            let liquidity_delta = self.calculate_liquidity_adjustment(
                pool,
                strategy,
                strategy.tick_lower,
                strategy.tick_upper,
            )?;

            RebalanceAction::AdjustLiquidity {
                liquidity_delta,
            }
        };

        // Update last rebalance time
        self.last_rebalance.insert(pool.address, current_time);

        // Update strategy ticks
        strategy.tick_lower = new_tick_lower;
        strategy.tick_upper = new_tick_upper;

        Ok(actions)
    }

    /// Calculates optimal range for the position
    fn calculate_optimal_range(
        &self,
        pool: &UniswapV3Pool,
        strategy: &AMMOptionStrategy,
    ) -> Result<(i32, i32)> {
        let current_price = pool.sqrt_price_x96;
        let current_tick = pool.tick;

        match strategy.strategy_type {
            StrategyType::CoveredCall => {
                // For covered calls, adjust range above current price
                let lower_tick = current_tick;
                let upper_tick = current_tick + (pool.tick_spacing * 10); // 10 spacing width
                Ok((lower_tick, upper_tick))
            },
            StrategyType::ProtectedPut => {
                // For protected puts, adjust range below current price
                let lower_tick = current_tick - (pool.tick_spacing * 10);
                let upper_tick = current_tick;
                Ok((lower_tick, upper_tick))
            },
            StrategyType::BullCallSpread | StrategyType::BearPutSpread => {
                // For spreads, create wider range around current price
                let lower_tick = current_tick - (pool.tick_spacing * 15);
                let upper_tick = current_tick + (pool.tick_spacing * 15);
                Ok((lower_tick, upper_tick))
            },
            _ => {
                // For other strategies, use symmetric range
                let range = pool.tick_spacing * 20;
                let lower_tick = current_tick - range;
                let upper_tick = current_tick + range;
                Ok((lower_tick, upper_tick))
            }
        }
    }

    /// Calculates required liquidity adjustment
    fn calculate_liquidity_adjustment(
        &self,
        pool: &UniswapV3Pool,
        strategy: &AMMOptionStrategy,
        new_lower: i32,
        new_upper: i32,
    ) -> Result<i128> {
        let current_price = pool.sqrt_price_x96;
        let target_liquidity = self.calculate_target_liquidity(
            pool,
            strategy,
            new_lower,
            new_upper,
        )?;

        let liquidity_delta = target_liquidity.as_i128()
            .saturating_sub(strategy.size.as_i128());

        Ok(liquidity_delta)
    }

    /// Calculates target liquidity for the position
    fn calculate_target_liquidity(
        &self,
        pool: &UniswapV3Pool,
        strategy: &AMMOptionStrategy,
        tick_lower: i32,
        tick_upper: i32,
    ) -> Result<U256> {
        let (sqrt_price_lower, sqrt_price_upper) = self.liquidity_manager
            .get_tick_range(tick_lower, tick_upper)?;

        self.liquidity_manager.calculate_liquidity_for_amounts(
            pool.sqrt_price_x96,
            sqrt_price_lower,
            sqrt_price_upper,
            strategy.size,
            strategy.collateral,
        )
    }

    /// Calculates price deviation
    fn calculate_price_deviation(
        &self,
        current_price: U256,
        strike_price: U256,
    ) -> Result<f64> {
        let current = current_price.as_u128() as f64;
        let strike = strike_price.as_u128() as f64;

        Ok(((current - strike) / strike).abs())
    }

    /// Calculates delta deviation
    fn calculate_delta_deviation(
        &self,
        pool: &UniswapV3Pool,
        strategy: &AMMOptionStrategy,
    ) -> Result<f64> {
        // This is a simplified calculation
        let current_tick = pool.tick;
        let target_tick = (strategy.strike_price.as_u128() as f64).log2() as i32;
        
        let deviation = (current_tick - target_tick).abs() as f64 / 
                       pool.tick_spacing as f64;

        Ok(deviation)
    }
}

/// Represents a rebalancing action
#[derive(Debug)]
pub enum RebalanceAction {
    Shift {
        old_lower: i32,
        old_upper: i32,
        new_lower: i32,
        new_upper: i32,
        liquidity_delta: i128,
    },
    AdjustLiquidity {
        liquidity_delta: i128,
    },
}
