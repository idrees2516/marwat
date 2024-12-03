use ethers::types::{U256, Address};
use std::sync::Arc;
use crate::types::{Result, PanopticError};
use crate::uniswap::{UniswapV3Pool, Position, ConcentratedLiquidityManager};
use crate::options::engine::{OptionsEngine, Greeks, MarketParameters};

/// Represents an AMM option strategy
#[derive(Debug)]
pub struct AMMOptionStrategy {
    pub strategy_type: StrategyType,
    pub strike_price: U256,
    pub expiry: u64,
    pub size: U256,
    pub collateral: U256,
    pub pool_address: Address,
    pub tick_lower: i32,
    pub tick_upper: i32,
}

#[derive(Debug, Clone, Copy)]
pub enum StrategyType {
    CoveredCall,
    ProtectedPut,
    BullCallSpread,
    BearPutSpread,
    IronCondor,
    Strangle,
    Straddle,
}

/// Manager for AMM-based option strategies
pub struct AMMStrategyManager {
    options_engine: Arc<OptionsEngine>,
    liquidity_manager: Arc<ConcentratedLiquidityManager>,
    min_collateral_ratio: f64,
    max_leverage: f64,
}

impl AMMStrategyManager {
    pub fn new(
        options_engine: Arc<OptionsEngine>,
        liquidity_manager: Arc<ConcentratedLiquidityManager>,
        min_collateral_ratio: f64,
        max_leverage: f64,
    ) -> Self {
        Self {
            options_engine,
            liquidity_manager,
            min_collateral_ratio,
            max_leverage,
        }
    }

    /// Creates a new option strategy position
    pub fn create_strategy(
        &self,
        pool: &UniswapV3Pool,
        strategy_type: StrategyType,
        strike_price: U256,
        expiry: u64,
        size: U256,
        collateral: U256,
    ) -> Result<AMMOptionStrategy> {
        // Validate strategy parameters
        self.validate_strategy_params(pool, strike_price, expiry, size, collateral)?;

        // Calculate strike ranges based on strategy type
        let (tick_lower, tick_upper) = self.calculate_strategy_ticks(
            strategy_type,
            strike_price,
            pool.tick_spacing,
        )?;

        // Create the strategy
        let strategy = AMMOptionStrategy {
            strategy_type,
            strike_price,
            expiry,
            size,
            collateral,
            pool_address: pool.address,
            tick_lower,
            tick_upper,
        };

        Ok(strategy)
    }

    /// Calculates optimal strike prices for a strategy
    pub fn calculate_optimal_strikes(
        &self,
        pool: &UniswapV3Pool,
        strategy_type: StrategyType,
        target_delta: f64,
    ) -> Result<Vec<U256>> {
        let current_price = pool.sqrt_price_x96;
        let volatility = self.options_engine.get_implied_volatility(
            current_price.as_u128() as u64,
            0, // current timestamp
        )?;

        match strategy_type {
            StrategyType::CoveredCall => {
                // For covered calls, find strikes with delta around 0.3
                let optimal_strike = self.find_strike_by_delta(
                    pool,
                    target_delta,
                    true, // is_call
                    volatility,
                )?;
                Ok(vec![optimal_strike])
            },
            StrategyType::ProtectedPut => {
                // For protected puts, find strikes with delta around -0.3
                let optimal_strike = self.find_strike_by_delta(
                    pool,
                    -target_delta,
                    false, // is_call
                    volatility,
                )?;
                Ok(vec![optimal_strike])
            },
            StrategyType::BullCallSpread | StrategyType::BearPutSpread => {
                // For spreads, find two strikes with desired delta spread
                let lower_strike = self.find_strike_by_delta(
                    pool,
                    target_delta,
                    true,
                    volatility,
                )?;
                let upper_strike = self.find_strike_by_delta(
                    pool,
                    target_delta * 0.5,
                    true,
                    volatility,
                )?;
                Ok(vec![lower_strike, upper_strike])
            },
            StrategyType::IronCondor => {
                // For iron condors, find four strikes
                let put_strike1 = self.find_strike_by_delta(
                    pool,
                    -target_delta,
                    false,
                    volatility,
                )?;
                let put_strike2 = self.find_strike_by_delta(
                    pool,
                    -target_delta * 0.5,
                    false,
                    volatility,
                )?;
                let call_strike1 = self.find_strike_by_delta(
                    pool,
                    target_delta * 0.5,
                    true,
                    volatility,
                )?;
                let call_strike2 = self.find_strike_by_delta(
                    pool,
                    target_delta,
                    true,
                    volatility,
                )?;
                Ok(vec![put_strike1, put_strike2, call_strike1, call_strike2])
            },
            StrategyType::Strangle | StrategyType::Straddle => {
                // For strangles and straddles
                let put_strike = self.find_strike_by_delta(
                    pool,
                    -target_delta,
                    false,
                    volatility,
                )?;
                let call_strike = self.find_strike_by_delta(
                    pool,
                    target_delta,
                    true,
                    volatility,
                )?;
                Ok(vec![put_strike, call_strike])
            },
        }
    }

    /// Finds a strike price that gives the desired delta
    fn find_strike_by_delta(
        &self,
        pool: &UniswapV3Pool,
        target_delta: f64,
        is_call: bool,
        volatility: f64,
    ) -> Result<U256> {
        let current_price = pool.sqrt_price_x96;
        let mut low = current_price.saturating_mul(U256::from(50)) / U256::from(100);
        let mut high = current_price.saturating_mul(U256::from(150)) / U256::from(100);
        
        for _ in 0..32 {  // Binary search with max 32 iterations
            let mid = (low + high) / U256::from(2);
            
            let params = MarketParameters {
                spot_price: current_price.as_u128() as f64,
                strike_price: mid.as_u128() as f64,
                time_to_expiry: 1.0, // normalized time
                risk_free_rate: 0.05, // assumption
                volatility,
                is_call,
            };

            let greeks = self.options_engine.calculate_greeks(&params)?;
            
            if (greeks.delta - target_delta).abs() < 0.01 {
                return Ok(mid);
            }

            if greeks.delta > target_delta {
                low = mid;
            } else {
                high = mid;
            }
        }

        Err(PanopticError::ConvergenceError)
    }

    /// Calculates required collateral for a strategy
    pub fn calculate_required_collateral(
        &self,
        strategy_type: StrategyType,
        strike_price: U256,
        size: U256,
    ) -> Result<U256> {
        let collateral = match strategy_type {
            StrategyType::CoveredCall => {
                // 100% collateral of underlying
                size
            },
            StrategyType::ProtectedPut => {
                // Strike price * size
                strike_price.saturating_mul(size) / U256::from(10).pow(18.into())
            },
            StrategyType::BullCallSpread | StrategyType::BearPutSpread => {
                // Difference between strikes * size
                size.saturating_mul(U256::from(self.max_leverage as u64))
            },
            StrategyType::IronCondor | StrategyType::Strangle | StrategyType::Straddle => {
                // Maximum loss * size
                strike_price.saturating_mul(size) / U256::from(10).pow(18.into())
            },
        };

        Ok(collateral)
    }

    /// Validates strategy parameters
    fn validate_strategy_params(
        &self,
        pool: &UniswapV3Pool,
        strike_price: U256,
        expiry: u64,
        size: U256,
        collateral: U256,
    ) -> Result<()> {
        // Check expiry
        let current_timestamp = 0; // TODO: Get current timestamp
        if expiry <= current_timestamp {
            return Err(PanopticError::InvalidExpiry);
        }

        // Check size
        if size == U256::zero() {
            return Err(PanopticError::InvalidSize);
        }

        // Check collateral ratio
        let required_collateral = self.calculate_required_collateral(
            StrategyType::CoveredCall, // Most conservative
            strike_price,
            size,
        )?;
        
        if collateral < required_collateral {
            return Err(PanopticError::InsufficientCollateral);
        }

        Ok(())
    }

    /// Calculates strategy ticks
    fn calculate_strategy_ticks(
        &self,
        strategy_type: StrategyType,
        strike_price: U256,
        tick_spacing: i32,
    ) -> Result<(i32, i32)> {
        let strike_tick = (strike_price.as_u128() as f64).log2() as i32 * tick_spacing;
        
        let (lower_offset, upper_offset) = match strategy_type {
            StrategyType::CoveredCall => (-2 * tick_spacing, 2 * tick_spacing),
            StrategyType::ProtectedPut => (-2 * tick_spacing, 2 * tick_spacing),
            StrategyType::BullCallSpread => (-tick_spacing, 3 * tick_spacing),
            StrategyType::BearPutSpread => (-3 * tick_spacing, tick_spacing),
            StrategyType::IronCondor => (-4 * tick_spacing, 4 * tick_spacing),
            StrategyType::Strangle => (-3 * tick_spacing, 3 * tick_spacing),
            StrategyType::Straddle => (-2 * tick_spacing, 2 * tick_spacing),
        };

        Ok((
            strike_tick + lower_offset,
            strike_tick + upper_offset,
        ))
    }
}
