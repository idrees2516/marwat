use crate::types::{OptionType, Pool, Position, PositionKey, Strike, TokenId};
use crate::math::{calculate_premium, calculate_collateral};
use ethers::types::{Address, U256};
use std::collections::HashMap;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum PoolError {
    #[error("Insufficient collateral")]
    InsufficientCollateral,
    #[error("Invalid strike price")]
    InvalidStrike,
    #[error("Invalid amount")]
    InvalidAmount,
    #[error("Position not found")]
    PositionNotFound,
    #[error("Position already exists")]
    PositionAlreadyExists,
    #[error("Unauthorized")]
    Unauthorized,
}

pub struct PoolManager {
    pool: Pool,
    positions: HashMap<PositionKey, Position>,
}

impl PoolManager {
    pub fn new(pool: Pool) -> Self {
        Self {
            pool,
            positions: HashMap::new(),
        }
    }

    pub fn create_position(
        &mut self,
        owner: Address,
        option_type: OptionType,
        strike: Strike,
        amount: U256,
        collateral: U256,
        expiry: U256,
    ) -> Result<TokenId, PoolError> {
        // Validate inputs
        if amount.is_zero() {
            return Err(PoolError::InvalidAmount);
        }

        // Calculate required collateral
        let required_collateral = calculate_collateral(
            &self.pool,
            option_type,
            strike,
            amount,
        );

        if collateral < required_collateral {
            return Err(PoolError::InsufficientCollateral);
        }

        // Generate token ID
        let token_id = TokenId(U256::from(self.positions.len()));

        // Create position key
        let key = PositionKey { owner, token_id };

        // Ensure position doesn't already exist
        if self.positions.contains_key(&key) {
            return Err(PoolError::PositionAlreadyExists);
        }

        // Create new position
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
    ) -> Result<(), PoolError> {
        let key = PositionKey { owner, token_id };

        // Ensure position exists and caller is owner
        let position = self.positions.get(&key)
            .ok_or(PoolError::PositionNotFound)?;

        if position.owner != owner {
            return Err(PoolError::Unauthorized);
        }

        // Remove position
        self.positions.remove(&key);

        Ok(())
    }

    pub fn exercise_position(
        &mut self,
        owner: Address,
        token_id: TokenId,
        amount: U256,
    ) -> Result<U256, PoolError> {
        let key = PositionKey { owner, token_id };

        // Ensure position exists and caller is owner
        let position = self.positions.get_mut(&key)
            .ok_or(PoolError::PositionNotFound)?;

        if position.owner != owner {
            return Err(PoolError::Unauthorized);
        }

        // Calculate settlement amount
        let settlement = match position.option_type {
            OptionType::Call => {
                if self.pool.sqrt_price_x96 > position.strike.0 {
                    (self.pool.sqrt_price_x96 - position.strike.0) * amount
                } else {
                    U256::zero()
                }
            }
            OptionType::Put => {
                if position.strike.0 > self.pool.sqrt_price_x96 {
                    (position.strike.0 - self.pool.sqrt_price_x96) * amount
                } else {
                    U256::zero()
                }
            }
        };

        // Update position
        position.amount -= amount;
        position.collateral -= calculate_collateral(
            &self.pool,
            position.option_type,
            position.strike,
            amount,
        );

        // Remove position if fully exercised
        if position.amount.is_zero() {
            self.positions.remove(&key);
        }

        Ok(settlement)
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

    pub fn calculate_premium(
        &self,
        option_type: OptionType,
        strike: Strike,
        amount: U256,
    ) -> U256 {
        calculate_premium(
            &self.pool,
            option_type,
            strike,
            amount,
        )
    }
}
