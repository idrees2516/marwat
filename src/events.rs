use crate::types::{OptionType, Strike, TokenId};
use ethers::types::{Address, U256};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionCreated {
    pub owner: Address,
    pub token_id: TokenId,
    pub option_type: OptionType,
    pub strike: Strike,
    pub amount: U256,
    pub collateral: U256,
    pub expiry: U256,
    pub timestamp: U256,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionClosed {
    pub owner: Address,
    pub token_id: TokenId,
    pub timestamp: U256,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionExercised {
    pub owner: Address,
    pub token_id: TokenId,
    pub amount: U256,
    pub settlement: U256,
    pub timestamp: U256,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolCreated {
    pub token0: Address,
    pub token1: Address,
    pub fee: u32,
    pub tick_spacing: i32,
    pub sqrt_price_x96: U256,
    pub timestamp: U256,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Swap {
    pub sender: Address,
    pub recipient: Address,
    pub amount0: i128,
    pub amount1: i128,
    pub sqrt_price_x96: U256,
    pub liquidity: U256,
    pub tick: i32,
    pub timestamp: U256,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollateralAdded {
    pub owner: Address,
    pub token_id: TokenId,
    pub amount: U256,
    pub timestamp: U256,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollateralRemoved {
    pub owner: Address,
    pub token_id: TokenId,
    pub amount: U256,
    pub timestamp: U256,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeeTierEnabled {
    pub fee: u32,
    pub tick_spacing: i32,
    pub timestamp: U256,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OwnershipTransferred {
    pub previous_owner: Address,
    pub new_owner: Address,
    pub timestamp: U256,
}

pub trait EventEmitter {
    fn emit_position_created(&self, event: PositionCreated);
    fn emit_position_closed(&self, event: PositionClosed);
    fn emit_position_exercised(&self, event: PositionExercised);
    fn emit_pool_created(&self, event: PoolCreated);
    fn emit_swap(&self, event: Swap);
    fn emit_collateral_added(&self, event: CollateralAdded);
    fn emit_collateral_removed(&self, event: CollateralRemoved);
    fn emit_fee_tier_enabled(&self, event: FeeTierEnabled);
    fn emit_ownership_transferred(&self, event: OwnershipTransferred);
}

#[derive(Default)]
pub struct LogEventEmitter;

impl EventEmitter for LogEventEmitter {
    fn emit_position_created(&self, event: PositionCreated) {
        tracing::info!(?event, "Position created");
    }

    fn emit_position_closed(&self, event: PositionClosed) {
        tracing::info!(?event, "Position closed");
    }

    fn emit_position_exercised(&self, event: PositionExercised) {
        tracing::info!(?event, "Position exercised");
    }

    fn emit_pool_created(&self, event: PoolCreated) {
        tracing::info!(?event, "Pool created");
    }

    fn emit_swap(&self, event: Swap) {
        tracing::info!(?event, "Swap executed");
    }

    fn emit_collateral_added(&self, event: CollateralAdded) {
        tracing::info!(?event, "Collateral added");
    }

    fn emit_collateral_removed(&self, event: CollateralRemoved) {
        tracing::info!(?event, "Collateral removed");
    }

    fn emit_fee_tier_enabled(&self, event: FeeTierEnabled) {
        tracing::info!(?event, "Fee tier enabled");
    }

    fn emit_ownership_transferred(&self, event: OwnershipTransferred) {
        tracing::info!(?event, "Ownership transferred");
    }
}
