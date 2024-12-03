use ethers::types::{Address, U256};
use std::collections::HashMap;
use std::sync::Arc;
use crate::types::{Result, PanopticError};
use super::pool::UniswapV3Pool;

/// Factory for creating and managing Uniswap V3 pools
pub struct UniswapV3Factory {
    pools: HashMap<Address, Arc<UniswapV3Pool>>,
    supported_fee_tiers: Vec<u32>,
    owner: Address,
}

impl UniswapV3Factory {
    pub fn new(owner: Address, supported_fee_tiers: Vec<u32>) -> Self {
        Self {
            pools: HashMap::new(),
            supported_fee_tiers,
            owner,
        }
    }

    /// Creates a new pool
    pub fn create_pool(
        &mut self,
        token0: Address,
        token1: Address,
        fee: u32,
    ) -> Result<Address> {
        if !self.supported_fee_tiers.contains(&fee) {
            return Err(PanopticError::UnsupportedFeeTier);
        }

        if token0 >= token1 {
            return Err(PanopticError::InvalidTokenOrder);
        }

        let pool_address = self.compute_pool_address(token0, token1, fee);
        
        if self.pools.contains_key(&pool_address) {
            return Err(PanopticError::PoolAlreadyExists);
        }

        let tick_spacing = match fee {
            500 => 10,    // 0.05%
            3000 => 60,   // 0.3%
            10000 => 200, // 1%
            _ => return Err(PanopticError::UnsupportedFeeTier),
        };

        let pool = UniswapV3Pool::new(
            pool_address,
            token0,
            token1,
            fee,
            tick_spacing,
        );

        self.pools.insert(pool_address, Arc::new(pool));
        Ok(pool_address)
    }

    /// Gets a pool by its address
    pub fn get_pool(&self, pool_address: Address) -> Result<Arc<UniswapV3Pool>> {
        self.pools.get(&pool_address)
            .cloned()
            .ok_or(PanopticError::PoolNotFound)
    }

    /// Gets a pool by token pair and fee
    pub fn get_pool_by_tokens(
        &self,
        token0: Address,
        token1: Address,
        fee: u32,
    ) -> Result<Arc<UniswapV3Pool>> {
        let pool_address = self.compute_pool_address(token0, token1, fee);
        self.get_pool(pool_address)
    }

    /// Computes the deterministic address for a pool
    fn compute_pool_address(
        &self,
        token0: Address,
        token1: Address,
        fee: u32,
    ) -> Address {
        // This is a simplified version. In practice, you'd use create2 derivation
        let mut bytes = [0u8; 20];
        let token0_bytes = token0.as_bytes();
        let token1_bytes = token1.as_bytes();
        let fee_bytes = fee.to_be_bytes();

        for i in 0..12 {
            bytes[i] = token0_bytes[i] ^ token1_bytes[i] ^ fee_bytes[i % 4];
        }

        Address::from_slice(&bytes)
    }

    /// Lists all pools
    pub fn list_pools(&self) -> Vec<(Address, Address, Address, u32)> {
        self.pools.values()
            .map(|pool| (pool.address, pool.token0, pool.token1, pool.fee))
            .collect()
    }

    /// Gets supported fee tiers
    pub fn get_supported_fee_tiers(&self) -> &[u32] {
        &self.supported_fee_tiers
    }

    /// Checks if a fee tier is supported
    pub fn is_fee_tier_supported(&self, fee: u32) -> bool {
        self.supported_fee_tiers.contains(&fee)
    }

    /// Gets the owner address
    pub fn get_owner(&self) -> Address {
        self.owner
    }

    /// Sets a new owner (only current owner can call)
    pub fn set_owner(&mut self, caller: Address, new_owner: Address) -> Result<()> {
        if caller != self.owner {
            return Err(PanopticError::Unauthorized);
        }
        self.owner = new_owner;
        Ok(())
    }

    /// Enables a new fee tier (only owner can call)
    pub fn enable_fee_tier(&mut self, caller: Address, fee: u32) -> Result<()> {
        if caller != self.owner {
            return Err(PanopticError::Unauthorized);
        }
        if !self.supported_fee_tiers.contains(&fee) {
            self.supported_fee_tiers.push(fee);
        }
        Ok(())
    }
}
