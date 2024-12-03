use crate::types::{OptionType, Pool, Position, PositionKey, Strike, TokenId};
use crate::errors::{PanopticError, Result};
use crate::pricing::PricingEngine;
use crate::collateral::CollateralManager;
use ethers::types::{Address, U256};
use std::collections::HashMap;

pub struct PositionManager {
    positions: HashMap<PositionKey, Position>,
    pricing_engine: PricingEngine,
    collateral_manager: CollateralManager,
}

impl PositionManager {
    pub fn new(
        pricing_engine: PricingEngine,
        collateral_manager: CollateralManager,
    ) -> Self {
        Self {
            positions: HashMap::new(),
            pricing_engine,
            collateral_manager,
        }
    }

    pub fn create_position(
        &mut self,
        pool: &Pool,
        owner: Address,
        option_type: OptionType,
        strike: Strike,
        amount: U256,
        collateral: U256,
        expiry: U256,
    ) -> Result<TokenId> {
        // Validate inputs
        if amount.is_zero() {
            return Err(PanopticError::InvalidAmount);
        }

        // Calculate required collateral
        let position = Position::new(
            owner,
            TokenId(U256::zero()), // Temporary token ID
            option_type,
            strike,
            amount,
            collateral,
            expiry,
        );

        // Validate collateral
        self.collateral_manager.validate_collateral(pool, &position, collateral)?;

        // Generate token ID
        let token_id = TokenId(U256::from(self.positions.len()));

        // Create position key
        let key = PositionKey { owner, token_id };

        // Ensure position doesn't exist
        if self.positions.contains_key(&key) {
            return Err(PanopticError::Custom("Position already exists".into()));
        }

        // Update position with actual token ID
        let position = Position::new(
            owner,
            token_id,
            option_type,
            strike,
            amount,
            collateral,
            expiry,
        );

        // Store position
        self.positions.insert(key, position);

        Ok(token_id)
    }

    pub fn close_position(
        &mut self,
        owner: Address,
        token_id: TokenId,
    ) -> Result<()> {
        let key = PositionKey { owner, token_id };

        // Ensure position exists and caller is owner
        if !self.positions.contains_key(&key) {
            return Err(PanopticError::PositionNotFound);
        }

        let position = self.positions.get(&key).unwrap();
        if position.owner != owner {
            return Err(PanopticError::Unauthorized);
        }

        // Remove position
        self.positions.remove(&key);

        Ok(())
    }

    pub fn exercise_position(
        &mut self,
        pool: &Pool,
        owner: Address,
        token_id: TokenId,
        amount: U256,
    ) -> Result<U256> {
        let key = PositionKey { owner, token_id };

        // Ensure position exists and caller is owner
        let position = self.positions.get_mut(&key)
            .ok_or(PanopticError::PositionNotFound)?;

        if position.owner != owner {
            return Err(PanopticError::Unauthorized);
        }

        if amount > position.amount {
            return Err(PanopticError::InvalidAmount);
        }

        // Calculate settlement amount
        let spot_price = pool.sqrt_price_x96.pow(2.into()) / U256::from(2).pow(96.into());
        let settlement = match position.option_type {
            OptionType::Call => {
                if spot_price > position.strike.0 {
                    (spot_price - position.strike.0)
                        .checked_mul(amount)
                        .ok_or(PanopticError::MultiplicationOverflow)?
                } else {
                    U256::zero()
                }
            }
            OptionType::Put => {
                if position.strike.0 > spot_price {
                    (position.strike.0 - spot_price)
                        .checked_mul(amount)
                        .ok_or(PanopticError::MultiplicationOverflow)?
                } else {
                    U256::zero()
                }
            }
        };

        // Update position
        position.amount = position.amount
            .checked_sub(amount)
            .ok_or(PanopticError::SubtractionOverflow)?;

        // Calculate released collateral
        let released_collateral = self.collateral_manager
            .calculate_required_collateral(pool, position)?
            .checked_mul(amount)
            .ok_or(PanopticError::MultiplicationOverflow)?
            / position.amount;

        position.collateral = position.collateral
            .checked_sub(released_collateral)
            .ok_or(PanopticError::SubtractionOverflow)?;

        // Remove position if fully exercised
        if position.amount.is_zero() {
            self.positions.remove(&key);
        }

        Ok(settlement)
    }

    pub fn add_collateral(
        &mut self,
        pool: &Pool,
        owner: Address,
        token_id: TokenId,
        additional_collateral: U256,
    ) -> Result<()> {
        let key = PositionKey { owner, token_id };

        // Ensure position exists and caller is owner
        let position = self.positions.get_mut(&key)
            .ok_or(PanopticError::PositionNotFound)?;

        if position.owner != owner {
            return Err(PanopticError::Unauthorized);
        }

        // Update collateral
        position.collateral = position.collateral
            .checked_add(additional_collateral)
            .ok_or(PanopticError::AdditionOverflow)?;

        // Validate new collateral amount
        self.collateral_manager.validate_collateral(pool, position, position.collateral)?;

        Ok(())
    }

    pub fn remove_collateral(
        &mut self,
        pool: &Pool,
        owner: Address,
        token_id: TokenId,
        collateral_to_remove: U256,
    ) -> Result<()> {
        let key = PositionKey { owner, token_id };

        // Ensure position exists and caller is owner
        let position = self.positions.get_mut(&key)
            .ok_or(PanopticError::PositionNotFound)?;

        if position.owner != owner {
            return Err(PanopticError::Unauthorized);
        }

        // Calculate new collateral amount
        let new_collateral = position.collateral
            .checked_sub(collateral_to_remove)
            .ok_or(PanopticError::SubtractionOverflow)?;

        // Validate new collateral amount
        self.collateral_manager.validate_collateral(pool, position, new_collateral)?;

        // Update collateral
        position.collateral = new_collateral;

        Ok(())
    }

    pub fn get_position(
        &self,
        owner: Address,
        token_id: TokenId,
    ) -> Option<&Position> {
        let key = PositionKey { owner, token_id };
        self.positions.get(&key)
    }

    pub fn get_all_positions(&self) -> &HashMap<PositionKey, Position> {
        &self.positions
    }

    pub fn check_liquidations(
        &self,
        pool: &Pool,
    ) -> Vec<PositionKey> {
        self.positions
            .iter()
            .filter_map(|(key, position)| {
                match self.collateral_manager.check_liquidation(pool, position) {
                    Ok(true) => Some(key.clone()),
                    _ => None,
                }
            })
            .collect()
    }

    pub fn calculate_position_value(
        &self,
        pool: &Pool,
        owner: Address,
        token_id: TokenId,
        time_to_expiry: f64,
    ) -> Result<U256> {
        let position = self.get_position(owner, token_id)
            .ok_or(PanopticError::PositionNotFound)?;

        let option_price = self.pricing_engine.calculate_option_price(
            pool,
            position.option_type,
            position.strike,
            time_to_expiry,
        )?;

        Ok(option_price.checked_mul(position.amount)
            .ok_or(PanopticError::MultiplicationOverflow)?)
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

    fn create_test_managers() -> (PricingEngine, CollateralManager) {
        let pricing_engine = PricingEngine::new(0.05, 1.0);
        let collateral_manager = CollateralManager::new(
            U256::from(150), // 150% min collateral ratio
            U256::from(120), // 120% liquidation threshold
        );
        (pricing_engine, collateral_manager)
    }

    #[test]
    fn test_create_position() {
        let (pricing_engine, collateral_manager) = create_test_managers();
        let mut position_manager = PositionManager::new(pricing_engine, collateral_manager);
        let pool = create_test_pool();

        let result = position_manager.create_position(
            &pool,
            Address::zero(),
            OptionType::Call,
            Strike(U256::from(1000)),
            U256::from(1),
            U256::from(2000),
            U256::from(1000), // expiry
        );

        assert!(result.is_ok());
    }

    #[test]
    fn test_exercise_position() {
        let (pricing_engine, collateral_manager) = create_test_managers();
        let mut position_manager = PositionManager::new(pricing_engine, collateral_manager);
        let pool = create_test_pool();

        // Create position
        let token_id = position_manager.create_position(
            &pool,
            Address::zero(),
            OptionType::Call,
            Strike(U256::from(1000)),
            U256::from(1),
            U256::from(2000),
            U256::from(1000), // expiry
        ).unwrap();

        // Exercise position
        let result = position_manager.exercise_position(
            &pool,
            Address::zero(),
            token_id,
            U256::from(1),
        );

        assert!(result.is_ok());
    }

    #[test]
    fn test_liquidations() {
        let (pricing_engine, collateral_manager) = create_test_managers();
        let mut position_manager = PositionManager::new(pricing_engine, collateral_manager);
        let pool = create_test_pool();

        // Create position with minimal collateral
        let token_id = position_manager.create_position(
            &pool,
            Address::zero(),
            OptionType::Call,
            Strike(U256::from(1000)),
            U256::from(1),
            U256::from(1000), // Minimal collateral
            U256::from(1000), // expiry
        ).unwrap();

        let liquidatable_positions = position_manager.check_liquidations(&pool);
        assert!(!liquidatable_positions.is_empty());
    }
}
