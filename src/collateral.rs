use crate::types::{OptionType, Pool, Position, Strike};
use crate::errors::{PanopticError, Result};
use ethers::types::U256;
use std::cmp;

pub struct CollateralManager {
    min_collateral_ratio: U256,
    liquidation_threshold: U256,
}

impl CollateralManager {
    pub fn new(min_collateral_ratio: U256, liquidation_threshold: U256) -> Self {
        Self {
            min_collateral_ratio,
            liquidation_threshold,
        }
    }

    pub fn validate_collateral(
        &self,
        pool: &Pool,
        position: &Position,
        collateral: U256,
    ) -> Result<()> {
        let required_collateral = self.calculate_required_collateral(pool, position)?;
        
        if collateral < required_collateral {
            return Err(PanopticError::InsufficientCollateral);
        }
        
        Ok(())
    }

    pub fn calculate_required_collateral(
        &self,
        pool: &Pool,
        position: &Position,
    ) -> Result<U256> {
        let spot_price = pool.sqrt_price_x96.pow(2.into()) / U256::from(2).pow(96.into());
        
        let required = match position.option_type {
            OptionType::Call => {
                // For calls, required collateral is max(spot_price - strike, 0) * amount * min_collateral_ratio
                if spot_price > position.strike.0 {
                    (spot_price - position.strike.0)
                        .checked_mul(position.amount)
                        .ok_or(PanopticError::MultiplicationOverflow)?
                        .checked_mul(self.min_collateral_ratio)
                        .ok_or(PanopticError::MultiplicationOverflow)?
                        / U256::from(100)
                } else {
                    U256::zero()
                }
            }
            OptionType::Put => {
                // For puts, required collateral is strike * amount * min_collateral_ratio
                position.strike.0
                    .checked_mul(position.amount)
                    .ok_or(PanopticError::MultiplicationOverflow)?
                    .checked_mul(self.min_collateral_ratio)
                    .ok_or(PanopticError::MultiplicationOverflow)?
                    / U256::from(100)
            }
        };
        
        Ok(required)
    }

    pub fn check_liquidation(
        &self,
        pool: &Pool,
        position: &Position,
    ) -> Result<bool> {
        let required_collateral = self.calculate_required_collateral(pool, position)?;
        
        if position.collateral == U256::zero() {
            return Ok(true);
        }
        
        let collateral_ratio = position.collateral
            .checked_mul(U256::from(100))
            .ok_or(PanopticError::MultiplicationOverflow)?
            / required_collateral;
            
        Ok(collateral_ratio < self.liquidation_threshold)
    }

    pub fn calculate_liquidation_price(
        &self,
        pool: &Pool,
        position: &Position,
    ) -> Result<U256> {
        match position.option_type {
            OptionType::Call => {
                // For calls, liquidation price is when collateral becomes insufficient
                let collateral_per_token = position.collateral
                    .checked_mul(U256::from(100))
                    .ok_or(PanopticError::MultiplicationOverflow)?
                    / (position.amount.checked_mul(self.min_collateral_ratio)
                        .ok_or(PanopticError::MultiplicationOverflow)?);
                
                Ok(position.strike.0.checked_add(collateral_per_token)
                    .ok_or(PanopticError::AdditionOverflow)?)
            }
            OptionType::Put => {
                // For puts, liquidation price is when position value exceeds collateral
                let collateral_per_token = position.collateral
                    .checked_mul(U256::from(100))
                    .ok_or(PanopticError::MultiplicationOverflow)?
                    / (position.amount.checked_mul(self.min_collateral_ratio)
                        .ok_or(PanopticError::MultiplicationOverflow)?);
                
                Ok(position.strike.0.checked_sub(collateral_per_token)
                    .ok_or(PanopticError::SubtractionOverflow)?)
            }
        }
    }

    pub fn calculate_max_leverage(
        &self,
        pool: &Pool,
        option_type: OptionType,
        strike: Strike,
    ) -> Result<U256> {
        let spot_price = pool.sqrt_price_x96.pow(2.into()) / U256::from(2).pow(96.into());
        
        match option_type {
            OptionType::Call => {
                if spot_price <= strike.0 {
                    return Ok(U256::from(100)); // 100x leverage when OTM
                }
                
                // For ITM calls, max leverage is determined by the collateral ratio
                let value_per_contract = spot_price
                    .checked_sub(strike.0)
                    .ok_or(PanopticError::SubtractionOverflow)?;
                
                Ok(U256::from(100)
                    .checked_mul(value_per_contract)
                    .ok_or(PanopticError::MultiplicationOverflow)?
                    / self.min_collateral_ratio)
            }
            OptionType::Put => {
                if spot_price >= strike.0 {
                    return Ok(U256::from(100)); // 100x leverage when OTM
                }
                
                // For ITM puts, max leverage is determined by the collateral ratio
                let value_per_contract = strike.0
                    .checked_sub(spot_price)
                    .ok_or(PanopticError::SubtractionOverflow)?;
                
                Ok(U256::from(100)
                    .checked_mul(value_per_contract)
                    .ok_or(PanopticError::MultiplicationOverflow)?
                    / self.min_collateral_ratio)
            }
        }
    }

    pub fn get_min_collateral_ratio(&self) -> U256 {
        self.min_collateral_ratio
    }

    pub fn get_liquidation_threshold(&self) -> U256 {
        self.liquidation_threshold
    }

    pub fn set_min_collateral_ratio(&mut self, ratio: U256) {
        self.min_collateral_ratio = ratio;
    }

    pub fn set_liquidation_threshold(&mut self, threshold: U256) {
        self.liquidation_threshold = threshold;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Tick;

    fn create_test_pool() -> Pool {
        Pool::new(
            Address::zero(),
            Address::zero(),
            Address::zero(),
            3000,
            60,
            U256::from(2).pow(96.into()), // sqrt_price_x96 = 1.0
            U256::from(1000000),
            Tick(0),
        )
    }

    fn create_test_position(option_type: OptionType, strike: U256, amount: U256, collateral: U256) -> Position {
        Position::new(
            Address::zero(),
            TokenId(U256::zero()),
            option_type,
            Strike(strike),
            amount,
            collateral,
            U256::from(1000), // expiry
        )
    }

    #[test]
    fn test_validate_collateral() {
        let manager = CollateralManager::new(U256::from(150), U256::from(120)); // 150% min ratio, 120% liquidation
        let pool = create_test_pool();
        
        // Test sufficient collateral
        let position = create_test_position(
            OptionType::Call,
            U256::from(1000),
            U256::from(1),
            U256::from(2000),
        );
        assert!(manager.validate_collateral(&pool, &position, U256::from(2000)).is_ok());
        
        // Test insufficient collateral
        let position = create_test_position(
            OptionType::Call,
            U256::from(1000),
            U256::from(1),
            U256::from(100),
        );
        assert!(manager.validate_collateral(&pool, &position, U256::from(100)).is_err());
    }

    #[test]
    fn test_check_liquidation() {
        let manager = CollateralManager::new(U256::from(150), U256::from(120));
        let pool = create_test_pool();
        
        // Test position above liquidation threshold
        let position = create_test_position(
            OptionType::Put,
            U256::from(1000),
            U256::from(1),
            U256::from(2000),
        );
        assert!(!manager.check_liquidation(&pool, &position).unwrap());
        
        // Test position below liquidation threshold
        let position = create_test_position(
            OptionType::Put,
            U256::from(1000),
            U256::from(1),
            U256::from(100),
        );
        assert!(manager.check_liquidation(&pool, &position).unwrap());
    }
}
