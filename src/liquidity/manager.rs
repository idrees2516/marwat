use crate::types::{Pool, Result, PanopticError};
use crate::pricing::models::OptionPricing;
use ethers::types::{U256, Address};
use std::collections::HashMap;
use tokio::sync::RwLock;
use std::sync::Arc;

pub struct LiquidityManager {
    pools: HashMap<Address, PoolLiquidity>,
    pricing: Arc<dyn OptionPricing>,
    min_liquidity: U256,
    max_liquidity: U256,
    rebalance_threshold: f64,
    concentration_limit: f64,
}

struct PoolLiquidity {
    total_liquidity: U256,
    active_liquidity: U256,
    reserved_liquidity: U256,
    utilization: f64,
    concentration: HashMap<Address, f64>,
    last_rebalance: u64,
}

impl LiquidityManager {
    pub fn new(
        pricing: Arc<dyn OptionPricing>,
        min_liquidity: U256,
        max_liquidity: U256,
        rebalance_threshold: f64,
        concentration_limit: f64,
    ) -> Self {
        Self {
            pools: HashMap::new(),
            pricing,
            min_liquidity,
            max_liquidity,
            rebalance_threshold,
            concentration_limit,
        }
    }

    pub fn add_liquidity(
        &mut self,
        pool: &Pool,
        amount: U256,
        provider: Address,
    ) -> Result<()> {
        let liquidity = self.pools.entry(pool.address)
            .or_insert_with(|| PoolLiquidity {
                total_liquidity: U256::zero(),
                active_liquidity: U256::zero(),
                reserved_liquidity: U256::zero(),
                utilization: 0.0,
                concentration: HashMap::new(),
                last_rebalance: 0,
            });

        if liquidity.total_liquidity + amount > self.max_liquidity {
            return Err(PanopticError::ExcessiveLiquidity);
        }

        liquidity.total_liquidity += amount;
        liquidity.active_liquidity += amount;

        let provider_concentration = liquidity.concentration
            .entry(provider)
            .or_insert(0.0);
        *provider_concentration = (amount.as_u128() as f64) / 
            (liquidity.total_liquidity.as_u128() as f64);

        if *provider_concentration > self.concentration_limit {
            return Err(PanopticError::ConcentrationLimit);
        }

        Ok(())
    }

    pub fn remove_liquidity(
        &mut self,
        pool: &Pool,
        amount: U256,
        provider: Address,
    ) -> Result<()> {
        let liquidity = self.pools.get_mut(&pool.address)
            .ok_or(PanopticError::PoolNotFound)?;

        if amount > liquidity.total_liquidity {
            return Err(PanopticError::InsufficientLiquidity);
        }

        if liquidity.total_liquidity - amount < self.min_liquidity {
            return Err(PanopticError::InsufficientLiquidity);
        }

        let available = liquidity.active_liquidity - liquidity.reserved_liquidity;
        if amount > available {
            return Err(PanopticError::LiquidityLocked);
        }

        liquidity.total_liquidity -= amount;
        liquidity.active_liquidity -= amount;

        if let Some(concentration) = liquidity.concentration.get_mut(&provider) {
            *concentration = ((*concentration * liquidity.total_liquidity.as_u128() as f64) - 
                amount.as_u128() as f64) / liquidity.total_liquidity.as_u128() as f64;
        }

        Ok(())
    }

    pub fn reserve_liquidity(
        &mut self,
        pool: &Pool,
        amount: U256,
    ) -> Result<()> {
        let liquidity = self.pools.get_mut(&pool.address)
            .ok_or(PanopticError::PoolNotFound)?;

        let available = liquidity.active_liquidity - liquidity.reserved_liquidity;
        if amount > available {
            return Err(PanopticError::InsufficientLiquidity);
        }

        liquidity.reserved_liquidity += amount;
        liquidity.utilization = liquidity.reserved_liquidity.as_u128() as f64 / 
            liquidity.active_liquidity.as_u128() as f64;

        Ok(())
    }

    pub fn release_liquidity(
        &mut self,
        pool: &Pool,
        amount: U256,
    ) -> Result<()> {
        let liquidity = self.pools.get_mut(&pool.address)
            .ok_or(PanopticError::PoolNotFound)?;

        if amount > liquidity.reserved_liquidity {
            return Err(PanopticError::InvalidAmount);
        }

        liquidity.reserved_liquidity -= amount;
        liquidity.utilization = liquidity.reserved_liquidity.as_u128() as f64 / 
            liquidity.active_liquidity.as_u128() as f64;

        Ok(())
    }

    pub fn rebalance_liquidity(
        &mut self,
        pool: &Pool,
        timestamp: u64,
    ) -> Result<()> {
        let liquidity = self.pools.get_mut(&pool.address)
            .ok_or(PanopticError::PoolNotFound)?;

        if timestamp - liquidity.last_rebalance < 3600 {
            return Ok(());
        }

        if liquidity.utilization < self.rebalance_threshold {
            return Ok(());
        }

        let optimal_active = self.pricing.calculate_optimal_liquidity(pool)?;
        let current_active = liquidity.active_liquidity;

        if optimal_active > current_active {
            let available = liquidity.total_liquidity - liquidity.active_liquidity;
            let to_activate = (optimal_active - current_active).min(available);
            liquidity.active_liquidity += to_activate;
        } else {
            let min_required = liquidity.reserved_liquidity + 
                (liquidity.reserved_liquidity * U256::from(10) / U256::from(100));
            let to_deactivate = (current_active - optimal_active)
                .min(current_active - min_required);
            liquidity.active_liquidity -= to_deactivate;
        }

        liquidity.utilization = liquidity.reserved_liquidity.as_u128() as f64 / 
            liquidity.active_liquidity.as_u128() as f64;
        liquidity.last_rebalance = timestamp;

        Ok(())
    }

    pub fn get_liquidity_info(
        &self,
        pool: &Pool,
    ) -> Result<(U256, U256, U256, f64)> {
        let liquidity = self.pools.get(&pool.address)
            .ok_or(PanopticError::PoolNotFound)?;

        Ok((
            liquidity.total_liquidity,
            liquidity.active_liquidity,
            liquidity.reserved_liquidity,
            liquidity.utilization,
        ))
    }

    pub fn get_provider_concentration(
        &self,
        pool: &Pool,
        provider: Address,
    ) -> Result<f64> {
        let liquidity = self.pools.get(&pool.address)
            .ok_or(PanopticError::PoolNotFound)?;

        Ok(*liquidity.concentration.get(&provider).unwrap_or(&0.0))
    }
}
