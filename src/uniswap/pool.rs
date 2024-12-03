use ethers::types::{U256, Address, H256};
use std::sync::Arc;
use crate::types::{Result, PanopticError};
use crate::math::{sqrt_price_to_tick, tick_to_sqrt_price};

/// Represents a Uniswap V3 pool state
#[derive(Debug, Clone)]
pub struct UniswapV3Pool {
    pub address: Address,
    pub token0: Address,
    pub token1: Address,
    pub fee: u32,
    pub tick_spacing: i32,
    pub liquidity: U256,
    pub sqrt_price_x96: U256,
    pub tick: i32,
    pub observation_index: u16,
    pub observation_cardinality: u16,
    pub observation_cardinality_next: u16,
    pub fee_growth_global0_x128: U256,
    pub fee_growth_global1_x128: U256,
    pub protocol_fees: (U256, U256),
    pub unlocked: bool,
}

/// Represents a position in a Uniswap V3 pool
#[derive(Debug, Clone)]
pub struct Position {
    pub liquidity: U256,
    pub fee_growth_inside0_last_x128: U256,
    pub fee_growth_inside1_last_x128: U256,
    pub tokens_owed0: U256,
    pub tokens_owed1: U256,
}

/// Represents swap parameters
#[derive(Debug)]
pub struct SwapParams {
    pub zero_for_one: bool,
    pub amount_specified: i128,
    pub sqrt_price_limit_x96: U256,
}

impl UniswapV3Pool {
    pub fn new(
        address: Address,
        token0: Address,
        token1: Address,
        fee: u32,
        tick_spacing: i32,
    ) -> Self {
        Self {
            address,
            token0,
            token1,
            fee,
            tick_spacing,
            liquidity: U256::zero(),
            sqrt_price_x96: U256::zero(),
            tick: 0,
            observation_index: 0,
            observation_cardinality: 1,
            observation_cardinality_next: 1,
            fee_growth_global0_x128: U256::zero(),
            fee_growth_global1_x128: U256::zero(),
            protocol_fees: (U256::zero(), U256::zero()),
            unlocked: true,
        }
    }

    /// Simulates a swap in the pool
    pub fn simulate_swap(&self, params: SwapParams) -> Result<(i128, i128, U256, i32)> {
        let (amount0, amount1) = if params.zero_for_one {
            (-params.amount_specified, 0)
        } else {
            (0, -params.amount_specified)
        };

        let next_sqrt_price = self.calculate_next_sqrt_price(
            params.zero_for_one,
            params.amount_specified,
            self.liquidity,
        )?;

        let next_tick = sqrt_price_to_tick(next_sqrt_price)?;

        Ok((amount0, amount1, next_sqrt_price, next_tick))
    }

    /// Calculates the next sqrt price after a swap
    fn calculate_next_sqrt_price(
        &self,
        zero_for_one: bool,
        amount_specified: i128,
        liquidity: U256,
    ) -> Result<U256> {
        if liquidity == U256::zero() {
            return Err(PanopticError::InsufficientLiquidity);
        }

        let abs_amount = amount_specified.unsigned_abs();
        let current_sqrt_price = self.sqrt_price_x96;

        if zero_for_one {
            let price_delta = (abs_amount << 96) / liquidity;
            Ok(current_sqrt_price - price_delta)
        } else {
            let price_delta = (abs_amount << 96) / liquidity;
            Ok(current_sqrt_price + price_delta)
        }
    }

    /// Updates pool state after a swap
    pub fn update_state_after_swap(
        &mut self,
        sqrt_price_x96: U256,
        tick: i32,
        liquidity_delta: i128,
    ) -> Result<()> {
        self.sqrt_price_x96 = sqrt_price_x96;
        self.tick = tick;
        
        if liquidity_delta != 0 {
            let new_liquidity = if liquidity_delta > 0 {
                self.liquidity.saturating_add(U256::from(liquidity_delta as u128))
            } else {
                self.liquidity.saturating_sub(U256::from((-liquidity_delta) as u128))
            };
            self.liquidity = new_liquidity;
        }

        Ok(())
    }

    /// Calculates fees for a swap
    pub fn calculate_swap_fees(
        &self,
        amount_in: U256,
        amount_out: U256,
        zero_for_one: bool,
    ) -> Result<(U256, U256)> {
        let fee_amount = amount_in.saturating_mul(U256::from(self.fee)) / U256::from(1_000_000);
        
        let fee0 = if zero_for_one { fee_amount } else { U256::zero() };
        let fee1 = if zero_for_one { U256::zero() } else { fee_amount };

        Ok((fee0, fee1))
    }

    /// Gets the current tick range for a position
    pub fn get_tick_range(&self, tick_lower: i32, tick_upper: i32) -> Result<(U256, U256)> {
        if tick_lower >= tick_upper {
            return Err(PanopticError::InvalidTickRange);
        }

        let sqrt_price_lower = tick_to_sqrt_price(tick_lower)?;
        let sqrt_price_upper = tick_to_sqrt_price(tick_upper)?;

        Ok((sqrt_price_lower, sqrt_price_upper))
    }

    /// Calculates the amount of liquidity needed for a given position
    pub fn calculate_liquidity_for_amounts(
        &self,
        sqrt_price_current: U256,
        sqrt_price_lower: U256,
        sqrt_price_upper: U256,
        amount0_desired: U256,
        amount1_desired: U256,
    ) -> Result<U256> {
        if sqrt_price_current < sqrt_price_lower || sqrt_price_current > sqrt_price_upper {
            return Err(PanopticError::PriceOutOfRange);
        }

        let liquidity0 = amount0_desired.saturating_mul(sqrt_price_current) / 
                        (sqrt_price_upper - sqrt_price_current);
        
        let liquidity1 = amount1_desired / 
                        (sqrt_price_current - sqrt_price_lower);

        Ok(liquidity0.min(liquidity1))
    }
}

/// Manager for Uniswap V3 positions
pub struct PositionManager {
    positions: std::collections::HashMap<H256, Position>,
}

impl PositionManager {
    pub fn new() -> Self {
        Self {
            positions: std::collections::HashMap::new(),
        }
    }

    /// Creates a new position
    pub fn create_position(
        &mut self,
        pool: &UniswapV3Pool,
        tick_lower: i32,
        tick_upper: i32,
        amount0_desired: U256,
        amount1_desired: U256,
    ) -> Result<(H256, Position)> {
        let (sqrt_price_lower, sqrt_price_upper) = pool.get_tick_range(tick_lower, tick_upper)?;
        
        let liquidity = pool.calculate_liquidity_for_amounts(
            pool.sqrt_price_x96,
            sqrt_price_lower,
            sqrt_price_upper,
            amount0_desired,
            amount1_desired,
        )?;

        let position = Position {
            liquidity,
            fee_growth_inside0_last_x128: U256::zero(),
            fee_growth_inside1_last_x128: U256::zero(),
            tokens_owed0: U256::zero(),
            tokens_owed1: U256::zero(),
        };

        let position_key = H256::random(); // In practice, this would be deterministic
        self.positions.insert(position_key, position.clone());

        Ok((position_key, position))
    }

    /// Updates a position's liquidity
    pub fn update_position_liquidity(
        &mut self,
        position_key: H256,
        liquidity_delta: i128,
    ) -> Result<()> {
        let position = self.positions.get_mut(&position_key)
            .ok_or(PanopticError::PositionNotFound)?;

        let new_liquidity = if liquidity_delta > 0 {
            position.liquidity.saturating_add(U256::from(liquidity_delta as u128))
        } else {
            position.liquidity.saturating_sub(U256::from((-liquidity_delta) as u128))
        };

        position.liquidity = new_liquidity;
        Ok(())
    }

    /// Collects fees for a position
    pub fn collect_fees(
        &mut self,
        position_key: H256,
        pool: &UniswapV3Pool,
    ) -> Result<(U256, U256)> {
        let position = self.positions.get_mut(&position_key)
            .ok_or(PanopticError::PositionNotFound)?;

        let fees0 = position.tokens_owed0;
        let fees1 = position.tokens_owed1;

        position.tokens_owed0 = U256::zero();
        position.tokens_owed1 = U256::zero();

        Ok((fees0, fees1))
    }
}
