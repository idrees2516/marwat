use super::*;
use crate::pricing::Greeks;
use crate::math::{calculate_premium, calculate_collateral};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy)]
pub struct PositionMetrics {
    pub value: U256,
    pub unrealized_pnl: i128,
    pub realized_pnl: i128,
    pub liquidation_price: U256,
    pub margin_ratio: U256,
    pub greeks: Greeks,
    pub funding_rate: i128,
    pub implied_volatility: f64,
}

#[derive(Debug, Clone)]
pub struct PositionAdjustment {
    pub size_delta: i128,
    pub collateral_delta: i128,
    pub strike_delta: i32,
    pub expiry_delta: i64,
}

impl PositionManager {
    pub fn adjust_position(
        &mut self,
        pool: &Pool,
        owner: Address,
        token_id: TokenId,
        adjustment: PositionAdjustment,
        current_block: U256,
    ) -> Result<()> {
        let key = PositionKey { owner, token_id };
        let position = self.positions.get_mut(&key)
            .ok_or(PanopticError::PositionNotFound)?;

        // Validate position is not expired
        if position.is_expired(current_block) {
            return Err(PanopticError::PositionExpired);
        }

        // Apply size adjustment
        if adjustment.size_delta != 0 {
            let new_amount = if adjustment.size_delta > 0 {
                position.amount.checked_add(U256::from(adjustment.size_delta as u128))
            } else {
                position.amount.checked_sub(U256::from((-adjustment.size_delta) as u128))
            }.ok_or(PanopticError::InvalidAmount)?;

            // Validate new size
            if new_amount.is_zero() {
                return Err(PanopticError::InvalidAmount);
            }
            position.amount = new_amount;
        }

        // Apply collateral adjustment
        if adjustment.collateral_delta != 0 {
            let new_collateral = if adjustment.collateral_delta > 0 {
                position.collateral.checked_add(U256::from(adjustment.collateral_delta as u128))
            } else {
                position.collateral.checked_sub(U256::from((-adjustment.collateral_delta) as u128))
            }.ok_or(PanopticError::InvalidCollateral)?;

            // Validate new collateral
            self.collateral_manager.validate_collateral(pool, position, new_collateral)?;
            position.collateral = new_collateral;
        }

        // Apply strike adjustment
        if adjustment.strike_delta != 0 {
            let new_strike = Strike(
                if adjustment.strike_delta > 0 {
                    position.strike.0.checked_add(U256::from(adjustment.strike_delta as u128))
                } else {
                    position.strike.0.checked_sub(U256::from((-adjustment.strike_delta) as u128))
                }.ok_or(PanopticError::InvalidStrike)?
            );

            // Validate new strike
            let time_to_expiry = (position.expiry - current_block).as_u64() as f64 / 31_536_000.0; // Convert to years
            let _ = self.pricing_engine.calculate_option_price(pool, position.option_type, new_strike, time_to_expiry)?;
            position.strike = new_strike;
        }

        // Apply expiry adjustment
        if adjustment.expiry_delta != 0 {
            let new_expiry = if adjustment.expiry_delta > 0 {
                position.expiry.checked_add(U256::from(adjustment.expiry_delta as u128))
            } else {
                position.expiry.checked_sub(U256::from((-adjustment.expiry_delta) as u128))
            }.ok_or(PanopticError::InvalidExpiry)?;

            // Validate new expiry
            if new_expiry <= current_block {
                return Err(PanopticError::InvalidExpiry);
            }
            position.expiry = new_expiry;
        }

        Ok(())
    }

    pub fn get_position_metrics(
        &self,
        pool: &Pool,
        owner: Address,
        token_id: TokenId,
        current_block: U256,
    ) -> Result<PositionMetrics> {
        let position = self.get_position(owner, token_id)
            .ok_or(PanopticError::PositionNotFound)?;

        let time_to_expiry = (position.expiry - current_block).as_u64() as f64 / 31_536_000.0;

        // Calculate current value
        let value = self.calculate_position_value(pool, owner, token_id, time_to_expiry)?;

        // Calculate unrealized PnL
        let entry_value = calculate_premium(pool, position.option_type, position.strike, position.amount);
        let unrealized_pnl = value.as_u128() as i128 - entry_value.as_u128() as i128;

        // Get liquidation price
        let liquidation_price = self.collateral_manager
            .calculate_liquidation_price(pool, position)?;

        // Calculate margin ratio
        let required_collateral = self.collateral_manager
            .calculate_required_collateral(pool, position)?;
        let margin_ratio = position.collateral
            .checked_mul(U256::from(100))
            .ok_or(PanopticError::MultiplicationOverflow)?
            / required_collateral;

        // Calculate Greeks
        let greeks = self.pricing_engine.calculate_greeks(
            pool,
            position.option_type,
            position.strike,
            time_to_expiry,
        )?;

        // Calculate funding rate (simplified)
        let funding_rate = self.calculate_funding_rate(pool, position)?;

        // Calculate implied volatility
        let implied_volatility = self.pricing_engine.calculate_implied_volatility(pool)?;

        Ok(PositionMetrics {
            value,
            unrealized_pnl,
            realized_pnl: 0, // Tracked separately
            liquidation_price,
            margin_ratio,
            greeks,
            funding_rate,
            implied_volatility,
        })
    }

    fn calculate_funding_rate(
        &self,
        pool: &Pool,
        position: &Position,
    ) -> Result<i128> {
        let utilization = pool.liquidity.as_u128() as f64 / U256::MAX.as_u128() as f64;
        let base_rate = 0.01; // 1% base rate
        let multiplier = if utilization > 0.8 {
            2.0
        } else if utilization > 0.5 {
            1.5
        } else {
            1.0
        };
        
        let funding_rate = (base_rate * multiplier * utilization * 1e6) as i128;
        Ok(funding_rate)
    }

    pub fn simulate_adjustment(
        &self,
        pool: &Pool,
        owner: Address,
        token_id: TokenId,
        adjustment: &PositionAdjustment,
        current_block: U256,
    ) -> Result<PositionMetrics> {
        let position = self.get_position(owner, token_id)
            .ok_or(PanopticError::PositionNotFound)?;

        // Create a copy of the position with adjustments
        let mut adjusted_position = position.clone();

        // Apply adjustments to the copy
        if adjustment.size_delta != 0 {
            adjusted_position.amount = if adjustment.size_delta > 0 {
                adjusted_position.amount.checked_add(U256::from(adjustment.size_delta as u128))
            } else {
                adjusted_position.amount.checked_sub(U256::from((-adjustment.size_delta) as u128))
            }.ok_or(PanopticError::InvalidAmount)?;
        }

        if adjustment.strike_delta != 0 {
            adjusted_position.strike = Strike(
                if adjustment.strike_delta > 0 {
                    adjusted_position.strike.0.checked_add(U256::from(adjustment.strike_delta as u128))
                } else {
                    adjusted_position.strike.0.checked_sub(U256::from((-adjustment.strike_delta) as u128))
                }.ok_or(PanopticError::InvalidStrike)?
            );
        }

        if adjustment.expiry_delta != 0 {
            adjusted_position.expiry = if adjustment.expiry_delta > 0 {
                adjusted_position.expiry.checked_add(U256::from(adjustment.expiry_delta as u128))
            } else {
                adjusted_position.expiry.checked_sub(U256::from((-adjustment.expiry_delta) as u128))
            }.ok_or(PanopticError::InvalidExpiry)?;
        }

        // Calculate metrics for adjusted position
        let time_to_expiry = (adjusted_position.expiry - current_block).as_u64() as f64 / 31_536_000.0;
        
        let value = self.pricing_engine.calculate_option_price(
            pool,
            adjusted_position.option_type,
            adjusted_position.strike,
            time_to_expiry,
        )?.checked_mul(adjusted_position.amount)
            .ok_or(PanopticError::MultiplicationOverflow)?;

        let entry_value = calculate_premium(
            pool,
            adjusted_position.option_type,
            adjusted_position.strike,
            adjusted_position.amount,
        );

        let unrealized_pnl = value.as_u128() as i128 - entry_value.as_u128() as i128;

        let liquidation_price = self.collateral_manager
            .calculate_liquidation_price(pool, &adjusted_position)?;

        let required_collateral = self.collateral_manager
            .calculate_required_collateral(pool, &adjusted_position)?;
            
        let margin_ratio = adjusted_position.collateral
            .checked_mul(U256::from(100))
            .ok_or(PanopticError::MultiplicationOverflow)?
            / required_collateral;

        let greeks = self.pricing_engine.calculate_greeks(
            pool,
            adjusted_position.option_type,
            adjusted_position.strike,
            time_to_expiry,
        )?;

        let funding_rate = self.calculate_funding_rate(pool, &adjusted_position)?;
        let implied_volatility = self.pricing_engine.calculate_implied_volatility(pool)?;

        Ok(PositionMetrics {
            value,
            unrealized_pnl,
            realized_pnl: 0,
            liquidation_price,
            margin_ratio,
            greeks,
            funding_rate,
            implied_volatility,
        })
    }
}

use crate::types::{Position, Pool, OptionType, TokenId};
use crate::pricing::PricingEngine;
use crate::risk::RiskManager;
use crate::errors::{PanopticError, Result};
use ethers::types::U256;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct PositionAdjustment {
    pub size_change: i128,
    pub strike_change: Option<U256>,
    pub collateral_change: i128,
    pub option_type_change: Option<OptionType>,
}

#[derive(Debug, Clone)]
pub struct PositionMetrics {
    pub intrinsic_value: U256,
    pub time_value: U256,
    pub implied_volatility: f64,
    pub delta: f64,
    pub gamma: f64,
    pub vega: f64,
    pub theta: f64,
    pub rho: f64,
    pub leverage: f64,
    pub funding_rate: f64,
    pub liquidation_price: U256,
}

pub struct AdvancedPositionManager {
    pricing_engine: PricingEngine,
    risk_manager: RiskManager,
    position_metrics: HashMap<TokenId, PositionMetrics>,
}

impl AdvancedPositionManager {
    pub fn new(pricing_engine: PricingEngine, risk_manager: RiskManager) -> Self {
        Self {
            pricing_engine,
            risk_manager,
            position_metrics: HashMap::new(),
        }
    }

    pub fn adjust_position(
        &mut self,
        pool: &Pool,
        position: &mut Position,
        adjustment: PositionAdjustment,
        current_block: U256,
    ) -> Result<()> {
        // Validate adjustment
        self.validate_adjustment(pool, position, &adjustment)?;

        // Calculate new position parameters
        let new_size = position.amount.as_u128() as i128 + adjustment.size_change;
        if new_size < 0 {
            return Err(PanopticError::Custom("Invalid position size".into()));
        }
        position.amount = U256::from(new_size as u128);

        if let Some(new_strike) = adjustment.strike_change {
            position.strike = new_strike;
        }

        let new_collateral = position.collateral.as_u128() as i128 + adjustment.collateral_change;
        if new_collateral < 0 {
            return Err(PanopticError::Custom("Invalid collateral amount".into()));
        }
        position.collateral = U256::from(new_collateral as u128);

        if let Some(new_option_type) = adjustment.option_type_change {
            position.option_type = new_option_type;
        }

        // Validate new position
        self.risk_manager.validate_position(pool, position, current_block)?;

        // Update position metrics
        self.update_position_metrics(pool, position, current_block)?;

        Ok(())
    }

    pub fn validate_adjustment(
        &self,
        pool: &Pool,
        position: &Position,
        adjustment: &PositionAdjustment,
    ) -> Result<()> {
        // Check if size change is within limits
        let new_size = position.amount.as_u128() as i128 + adjustment.size_change;
        if new_size < 0 {
            return Err(PanopticError::Custom("Position size cannot be negative".into()));
        }

        // Validate strike price change
        if let Some(new_strike) = adjustment.strike_change {
            if new_strike > U256::MAX / 2 {
                return Err(PanopticError::Custom("Invalid strike price".into()));
            }
        }

        // Check collateral requirements
        let new_collateral = position.collateral.as_u128() as i128 + adjustment.collateral_change;
        if new_collateral < 0 {
            return Err(PanopticError::Custom("Collateral cannot be negative".into()));
        }

        Ok(())
    }

    pub fn update_position_metrics(
        &mut self,
        pool: &Pool,
        position: &Position,
        current_block: U256,
    ) -> Result<()> {
        let time_to_expiry = (position.expiry - current_block).as_u64() as f64 / 31_536_000.0;
        
        let spot_price = pool.sqrt_price_x96.pow(2.into()) / U256::from(2).pow(96.into());
        let strike_price = position.strike;

        // Calculate option values
        let intrinsic_value = match position.option_type {
            OptionType::Call => spot_price.saturating_sub(strike_price),
            OptionType::Put => strike_price.saturating_sub(spot_price),
        };

        let option_price = self.pricing_engine.calculate_option_price(
            pool,
            position.option_type,
            strike_price,
            time_to_expiry,
        )?;

        let time_value = option_price.saturating_sub(intrinsic_value);

        // Calculate Greeks
        let greeks = self.pricing_engine.calculate_greeks(
            pool,
            position.option_type,
            strike_price,
            time_to_expiry,
        )?;

        // Calculate implied volatility
        let implied_vol = self.pricing_engine.calculate_implied_volatility(pool)?;

        // Calculate leverage
        let notional_value = position.amount.checked_mul(spot_price)
            .ok_or(PanopticError::MultiplicationOverflow)?;
        let leverage = notional_value.as_u128() as f64 / position.collateral.as_u128() as f64;

        // Calculate funding rate
        let funding_rate = self.calculate_funding_rate(pool, position)?;

        // Calculate liquidation price
        let liquidation_price = self.calculate_liquidation_price(pool, position)?;

        let metrics = PositionMetrics {
            intrinsic_value,
            time_value,
            implied_volatility: implied_vol,
            delta: greeks.delta,
            gamma: greeks.gamma,
            vega: greeks.vega,
            theta: greeks.theta,
            rho: greeks.rho,
            leverage,
            funding_rate,
            liquidation_price,
        };

        self.position_metrics.insert(position.token_id, metrics);
        Ok(())
    }

    pub fn calculate_funding_rate(
        &self,
        pool: &Pool,
        position: &Position,
    ) -> Result<f64> {
        let utilization = pool.liquidity.as_u128() as f64 / U256::MAX.as_u128() as f64;
        let base_rate = 0.01; // 1% base rate
        
        // Funding rate increases with utilization
        let funding_rate = base_rate * (1.0 + utilization.powi(2));
        
        // Adjust for option type
        let adjusted_rate = match position.option_type {
            OptionType::Call => funding_rate,
            OptionType::Put => -funding_rate,
        };
        
        Ok(adjusted_rate)
    }

    pub fn calculate_liquidation_price(
        &self,
        pool: &Pool,
        position: &Position,
    ) -> Result<U256> {
        let spot_price = pool.sqrt_price_x96.pow(2.into()) / U256::from(2).pow(96.into());
        let min_collateral_ratio = U256::from(120); // 120% minimum collateral ratio
        
        let required_collateral = position.amount
            .checked_mul(spot_price)
            .ok_or(PanopticError::MultiplicationOverflow)?
            .checked_mul(min_collateral_ratio)
            .ok_or(PanopticError::MultiplicationOverflow)?
            .checked_div(U256::from(100))
            .ok_or(PanopticError::DivisionByZero)?;
            
        match position.option_type {
            OptionType::Call => {
                let upward_move = required_collateral
                    .checked_div(position.amount)
                    .ok_or(PanopticError::DivisionByZero)?;
                spot_price.checked_add(upward_move)
                    .ok_or(PanopticError::AdditionOverflow)
            },
            OptionType::Put => {
                let downward_move = required_collateral
                    .checked_div(position.amount)
                    .ok_or(PanopticError::DivisionByZero)?;
                spot_price.checked_sub(downward_move)
                    .ok_or(PanopticError::SubtractionUnderflow)
            },
        }
    }

    pub fn get_position_metrics(&self, token_id: &TokenId) -> Option<&PositionMetrics> {
        self.position_metrics.get(token_id)
    }

    pub fn simulate_adjustment(
        &self,
        pool: &Pool,
        position: &Position,
        adjustment: &PositionAdjustment,
        current_block: U256,
    ) -> Result<PositionMetrics> {
        let mut simulated_position = position.clone();
        
        // Apply adjustment to simulated position
        let new_size = simulated_position.amount.as_u128() as i128 + adjustment.size_change;
        if new_size < 0 {
            return Err(PanopticError::Custom("Invalid position size".into()));
        }
        simulated_position.amount = U256::from(new_size as u128);

        if let Some(new_strike) = adjustment.strike_change {
            simulated_position.strike = new_strike;
        }

        let new_collateral = simulated_position.collateral.as_u128() as i128 + adjustment.collateral_change;
        if new_collateral < 0 {
            return Err(PanopticError::Custom("Invalid collateral amount".into()));
        }
        simulated_position.collateral = U256::from(new_collateral as u128);

        if let Some(new_option_type) = adjustment.option_type_change {
            simulated_position.option_type = new_option_type;
        }

        // Calculate metrics for simulated position
        let time_to_expiry = (simulated_position.expiry - current_block).as_u64() as f64 / 31_536_000.0;
        let spot_price = pool.sqrt_price_x96.pow(2.into()) / U256::from(2).pow(96.into());

        let intrinsic_value = match simulated_position.option_type {
            OptionType::Call => spot_price.saturating_sub(simulated_position.strike),
            OptionType::Put => simulated_position.strike.saturating_sub(spot_price),
        };

        let option_price = self.pricing_engine.calculate_option_price(
            pool,
            simulated_position.option_type,
            simulated_position.strike,
            time_to_expiry,
        )?;

        let time_value = option_price.saturating_sub(intrinsic_value);
        let greeks = self.pricing_engine.calculate_greeks(
            pool,
            simulated_position.option_type,
            simulated_position.strike,
            time_to_expiry,
        )?;

        let implied_vol = self.pricing_engine.calculate_implied_volatility(pool)?;
        let notional_value = simulated_position.amount.checked_mul(spot_price)
            .ok_or(PanopticError::MultiplicationOverflow)?;
        let leverage = notional_value.as_u128() as f64 / simulated_position.collateral.as_u128() as f64;

        let funding_rate = self.calculate_funding_rate(pool, &simulated_position)?;
        let liquidation_price = self.calculate_liquidation_price(pool, &simulated_position)?;

        Ok(PositionMetrics {
            intrinsic_value,
            time_value,
            implied_volatility: implied_vol,
            delta: greeks.delta,
            gamma: greeks.gamma,
            vega: greeks.vega,
            theta: greeks.theta,
            rho: greeks.rho,
            leverage,
            funding_rate,
            liquidation_price,
        })
    }
}
