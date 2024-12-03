use crate::types::{Pool, PoolKey};
use crate::pool::PoolManager;
use ethers::types::{Address, U256};
use std::collections::HashMap;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum FactoryError {
    #[error("Pool already exists")]
    PoolAlreadyExists,
    #[error("Pool not found")]
    PoolNotFound,
    #[error("Invalid fee tier")]
    InvalidFeeTier,
    #[error("Invalid tokens")]
    InvalidTokens,
}

pub struct PanopticFactory {
    owner: Address,
    pools: HashMap<PoolKey, PoolManager>,
    fee_tiers: Vec<u32>,
}

impl PanopticFactory {
    pub fn new(owner: Address) -> Self {
        Self {
            owner,
            pools: HashMap::new(),
            fee_tiers: vec![100, 500, 3000, 10000], // 0.01%, 0.05%, 0.3%, 1%
        }
    }

    pub fn create_pool(
        &mut self,
        token0: Address,
        token1: Address,
        fee: u32,
        sqrt_price_x96: U256,
    ) -> Result<&PoolManager, FactoryError> {
        // Validate inputs
        if token0 >= token1 {
            return Err(FactoryError::InvalidTokens);
        }

        if !self.fee_tiers.contains(&fee) {
            return Err(FactoryError::InvalidFeeTier);
        }

        let key = PoolKey {
            token0,
            token1,
            fee,
        };

        // Ensure pool doesn't exist
        if self.pools.contains_key(&key) {
            return Err(FactoryError::PoolAlreadyExists);
        }

        // Calculate tick spacing based on fee tier
        let tick_spacing = match fee {
            100 => 1,    // 0.01%
            500 => 10,   // 0.05%
            3000 => 60,  // 0.3%
            10000 => 200,// 1%
            _ => return Err(FactoryError::InvalidFeeTier),
        };

        // Create pool
        let pool = Pool::new(
            Address::zero(), // Pool address will be deterministically generated
            token0,
            token1,
            fee,
            tick_spacing,
            sqrt_price_x96,
            U256::zero(),
            crate::types::Tick(0),
        );

        // Create pool manager
        let pool_manager = PoolManager::new(pool);
        
        // Store pool
        self.pools.insert(key, pool_manager);

        Ok(self.pools.get(&key).unwrap())
    }

    pub fn get_pool(
        &self,
        token0: Address,
        token1: Address,
        fee: u32,
    ) -> Option<&PoolManager> {
        let key = PoolKey {
            token0,
            token1,
            fee,
        };
        self.pools.get(&key)
    }

    pub fn get_pool_mut(
        &mut self,
        token0: Address,
        token1: Address,
        fee: u32,
    ) -> Option<&mut PoolManager> {
        let key = PoolKey {
            token0,
            token1,
            fee,
        };
        self.pools.get_mut(&key)
    }

    pub fn set_fee_tier(
        &mut self,
        fee: u32,
        enabled: bool,
    ) -> Result<(), FactoryError> {
        if enabled {
            if !self.fee_tiers.contains(&fee) {
                self.fee_tiers.push(fee);
            }
        } else {
            if let Some(pos) = self.fee_tiers.iter().position(|&x| x == fee) {
                self.fee_tiers.remove(pos);
            }
        }
        Ok(())
    }

    pub fn get_fee_tiers(&self) -> &[u32] {
        &self.fee_tiers
    }

    pub fn get_owner(&self) -> Address {
        self.owner
    }

    pub fn set_owner(&mut self, new_owner: Address) {
        self.owner = new_owner;
    }
}
