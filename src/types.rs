use ethers::types::{Address, U256};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OptionType {
    Call,
    Put,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TokenId(pub U256);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Strike(pub U256);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Tick(pub i32);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Position {
    pub owner: Address,
    pub token_id: TokenId,
    pub option_type: OptionType,
    pub strike: Strike,
    pub amount: U256,
    pub collateral: U256,
    pub expiry: U256,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Pool {
    pub address: Address,
    pub token0: Address,
    pub token1: Address,
    pub fee: u32,
    pub tick_spacing: i32,
    pub sqrt_price_x96: U256,
    pub liquidity: U256,
    pub tick: Tick,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PositionKey {
    pub owner: Address,
    pub token_id: TokenId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PoolKey {
    pub token0: Address,
    pub token1: Address,
    pub fee: u32,
}

impl Position {
    pub fn new(
        owner: Address,
        token_id: TokenId,
        option_type: OptionType,
        strike: Strike,
        amount: U256,
        collateral: U256,
        expiry: U256,
    ) -> Self {
        Self {
            owner,
            token_id,
            option_type,
            strike,
            amount,
            collateral,
            expiry,
        }
    }

    pub fn is_expired(&self, current_block: U256) -> bool {
        current_block >= self.expiry
    }
}

impl Pool {
    pub fn new(
        address: Address,
        token0: Address,
        token1: Address,
        fee: u32,
        tick_spacing: i32,
        sqrt_price_x96: U256,
        liquidity: U256,
        tick: Tick,
    ) -> Self {
        Self {
            address,
            token0,
            token1,
            fee,
            tick_spacing,
            sqrt_price_x96,
            liquidity,
            tick,
        }
    }

    pub fn key(&self) -> PoolKey {
        PoolKey {
            token0: self.token0,
            token1: self.token1,
            fee: self.fee,
        }
    }
}
