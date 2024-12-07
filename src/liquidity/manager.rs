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

#[derive(Debug)]
struct TickLiquidity {
    active: U256,
    reserved: U256,
    positions: HashMap<Address, U256>,
    last_update: u64,
}

#[derive(Debug)]
struct PoolLiquidity {
    total_liquidity: U256,
    active_liquidity: U256,
    reserved_liquidity: U256,
    utilization: f64,
    concentration: HashMap<Address, f64>,
    last_rebalance: u64,
    tick_liquidity: HashMap<i32, TickLiquidity>,
    volatility_index: f64,
    rebalance_cooldown: u64,
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
                tick_liquidity: HashMap::new(),
                volatility_index: 0.0,
                rebalance_cooldown: 3600,
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

        // Check rebalance cooldown
        if timestamp - liquidity.last_rebalance < liquidity.rebalance_cooldown {
            return Ok(());
        }

        // Calculate volatility-adjusted threshold
        let adjusted_threshold = self.rebalance_threshold * 
            (1.0 + liquidity.volatility_index.min(1.0));

        if liquidity.utilization < adjusted_threshold {
            return Ok(());
        }

        // Get optimal liquidity based on current market conditions
        let optimal_active = self.pricing.calculate_optimal_liquidity(pool)?;
        let current_active = liquidity.active_liquidity;

        // Calculate tick range for liquidity distribution
        let current_tick = pool.tick.0;
        let tick_spacing = pool.tick_spacing;
        let range_width = ((liquidity.volatility_index * 100.0) as i32).max(10);
        
        let lower_tick = current_tick - (range_width * tick_spacing);
        let upper_tick = current_tick + (range_width * tick_spacing);

        if optimal_active > current_active {
            let available = liquidity.total_liquidity - liquidity.active_liquidity;
            let to_activate = (optimal_active - current_active).min(available);
            
            // Distribute new liquidity across ticks
            self.distribute_liquidity(
                liquidity,
                to_activate,
                lower_tick,
                upper_tick,
                tick_spacing,
                timestamp
            )?;
            
            liquidity.active_liquidity += to_activate;
        } else {
            let min_required = liquidity.reserved_liquidity + 
                (liquidity.reserved_liquidity * U256::from(20) / U256::from(100));
            let to_deactivate = (current_active - optimal_active)
                .min(current_active - min_required);
            
            // Remove liquidity from outer ticks first
            self.concentrate_liquidity(
                liquidity,
                to_deactivate,
                current_tick,
                tick_spacing,
                timestamp
            )?;
            
            liquidity.active_liquidity -= to_deactivate;
        }

        // Update pool metrics
        liquidity.utilization = liquidity.reserved_liquidity.as_u128() as f64 / 
            liquidity.active_liquidity.as_u128() as f64;
        liquidity.last_rebalance = timestamp;
        
        // Adjust rebalance cooldown based on volatility
        liquidity.rebalance_cooldown = if liquidity.volatility_index > 0.5 {
            1800 // 30 minutes for high volatility
        } else {
            3600 // 1 hour for normal conditions
        };

        Ok(())
    }

    fn distribute_liquidity(
        &mut self,
        liquidity: &mut PoolLiquidity,
        amount: U256,
        lower_tick: i32,
        upper_tick: i32,
        tick_spacing: i32,
        timestamp: u64,
    ) -> Result<()> {
        let mut remaining = amount;
        let num_ticks = ((upper_tick - lower_tick) / tick_spacing) as u32;
        
        // Calculate base amount per tick
        let base_amount = amount / U256::from(num_ticks);
        
        // Distribute with concentration around current price
        for tick in (lower_tick..=upper_tick).step_by(tick_spacing as usize) {
            if remaining == U256::zero() {
                break;
            }

            let tick_entry = liquidity.tick_liquidity
                .entry(tick)
                .or_insert(TickLiquidity {
                    active: U256::zero(),
                    reserved: U256::zero(),
                    positions: HashMap::new(),
                    last_update: timestamp,
                });

            let distance_from_center = (tick - lower_tick - (upper_tick - lower_tick) / 2).abs();
            let weight = 1.0 - (distance_from_center as f64 / (num_ticks as f64 / 2.0));
            
            let tick_amount = if tick == upper_tick {
                remaining // Use all remaining for last tick
            } else {
                let weighted = (base_amount.as_u128() as f64 * weight) as u128;
                let amount = U256::from(weighted.max(1));
                amount.min(remaining)
            };

            tick_entry.active += tick_amount;
            remaining -= tick_amount;
        }

        Ok(())
    }

    fn concentrate_liquidity(
        &mut self,
        liquidity: &mut PoolLiquidity,
        amount: U256,
        current_tick: i32,
        tick_spacing: i32,
        timestamp: u64,
    ) -> Result<()> {
        let mut remaining = amount;
        let mut ticks: Vec<_> = liquidity.tick_liquidity.keys().cloned().collect();
        ticks.sort_by_key(|t| (t - current_tick).abs());

        // Remove liquidity from outer ticks first
        for tick in ticks.into_iter().rev() {
            if remaining == U256::zero() {
                break;
            }

            if let Some(tick_entry) = liquidity.tick_liquidity.get_mut(&tick) {
                let available = tick_entry.active - tick_entry.reserved;
                let remove_amount = remaining.min(available);
                
                if remove_amount > U256::zero() {
                    tick_entry.active -= remove_amount;
                    remaining -= remove_amount;
                    tick_entry.last_update = timestamp;

                    // Remove tick entry if empty
                    if tick_entry.active == U256::zero() && 
                       tick_entry.reserved == U256::zero() {
                        liquidity.tick_liquidity.remove(&tick);
                    }
                }
            }
        }

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
