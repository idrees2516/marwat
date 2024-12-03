use crate::types::{Pool, Result, PanopticError, OptionType};
use crate::pricing::models::OptionPricing;
use crate::risk::manager::RiskManager;
use ethers::types::{U256, Address};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

pub struct MarketMaker {
    pricing: Arc<dyn OptionPricing>,
    risk_manager: Arc<RiskManager>,
    pools: HashMap<Address, PoolMaker>,
    min_spread: f64,
    max_position: U256,
    inventory_target: f64,
    rebalance_threshold: f64,
}

struct PoolMaker {
    positions: HashMap<OptionType, Position>,
    inventory: f64,
    last_rebalance: u64,
    volume_24h: U256,
    revenue_24h: U256,
}

struct Position {
    size: U256,
    entry_price: U256,
    current_price: U256,
    unrealized_pnl: i64,
    delta: f64,
    gamma: f64,
}

impl MarketMaker {
    pub fn new(
        pricing: Arc<dyn OptionPricing>,
        risk_manager: Arc<RiskManager>,
        min_spread: f64,
        max_position: U256,
        inventory_target: f64,
        rebalance_threshold: f64,
    ) -> Self {
        Self {
            pricing,
            risk_manager,
            pools: HashMap::new(),
            min_spread,
            max_position,
            inventory_target,
            rebalance_threshold,
        }
    }

    pub async fn update_quotes(
        &mut self,
        pool: &Pool,
        timestamp: u64,
    ) -> Result<Vec<Quote>> {
        let maker = self.pools.entry(pool.address)
            .or_insert_with(|| PoolMaker {
                positions: HashMap::new(),
                inventory: 0.0,
                last_rebalance: 0,
                volume_24h: U256::zero(),
                revenue_24h: U256::zero(),
            });

        let mut quotes = Vec::new();
        let base_volatility = self.pricing.get_implied_volatility(pool).await?;
        let risk_metrics = self.risk_manager.get_pool_metrics(pool).await?;

        // Adjust volatility based on inventory and risk
        let inventory_skew = (maker.inventory - self.inventory_target).abs();
        let vol_adjustment = 1.0 + (inventory_skew * 0.1).min(0.5);
        let adjusted_volatility = base_volatility * vol_adjustment;

        for option_type in [OptionType::Call, OptionType::Put] {
            let position = maker.positions.entry(option_type)
                .or_insert_with(|| Position {
                    size: U256::zero(),
                    entry_price: U256::zero(),
                    current_price: U256::zero(),
                    unrealized_pnl: 0,
                    delta: 0.0,
                    gamma: 0.0,
                });

            // Calculate base price
            let base_price = self.pricing.calculate_option_price(
                pool,
                option_type,
                adjusted_volatility,
            )?;

            // Calculate spread based on risk metrics
            let base_spread = self.min_spread * (1.0 + risk_metrics.total_risk_score);
            let inventory_spread = base_spread * (1.0 + inventory_skew);
            
            // Generate bid-ask quotes
            let bid_price = base_price * U256::from((1.0 - inventory_spread) as u128);
            let ask_price = base_price * U256::from((1.0 + inventory_spread) as u128);

            // Calculate maximum sizes
            let remaining_capacity = self.max_position - position.size;
            let risk_adjusted_size = self.risk_manager.calculate_max_position_size(
                pool,
                option_type,
                base_price,
            )?;
            
            let max_bid_size = remaining_capacity.min(risk_adjusted_size);
            let max_ask_size = position.size.min(risk_adjusted_size);

            quotes.push(Quote {
                option_type,
                bid_price,
                ask_price,
                bid_size: max_bid_size,
                ask_size: max_ask_size,
                timestamp,
            });

            // Update position metrics
            position.current_price = base_price;
            position.delta = self.pricing.calculate_delta(
                pool,
                option_type,
                adjusted_volatility,
            )?;
            position.gamma = self.pricing.calculate_gamma(
                pool,
                option_type,
                adjusted_volatility,
            )?;
        }

        Ok(quotes)
    }

    pub async fn execute_trade(
        &mut self,
        pool: &Pool,
        option_type: OptionType,
        size: U256,
        price: U256,
        is_buy: bool,
        timestamp: u64,
    ) -> Result<()> {
        let maker = self.pools.get_mut(&pool.address)
            .ok_or(PanopticError::PoolNotFound)?;

        let position = maker.positions.get_mut(&option_type)
            .ok_or(PanopticError::PositionNotFound)?;

        // Validate trade
        if is_buy {
            if position.size + size > self.max_position {
                return Err(PanopticError::ExcessivePosition);
            }
        } else {
            if size > position.size {
                return Err(PanopticError::InsufficientPosition);
            }
        }

        // Update position
        if is_buy {
            let total_cost = position.size * position.entry_price + size * price;
            position.size += size;
            position.entry_price = total_cost / position.size;
            maker.inventory += size.as_u128() as f64;
        } else {
            position.size -= size;
            maker.inventory -= size.as_u128() as f64;
        }

        // Update metrics
        maker.volume_24h += size * price;
        let pnl = if is_buy {
            -(price.as_u128() as i64)
        } else {
            price.as_u128() as i64
        };
        position.unrealized_pnl += pnl;

        // Check if rebalance is needed
        if (maker.inventory - self.inventory_target).abs() > self.rebalance_threshold {
            self.rebalance_inventory(pool, timestamp).await?;
        }

        Ok(())
    }

    async fn rebalance_inventory(
        &mut self,
        pool: &Pool,
        timestamp: u64,
    ) -> Result<()> {
        let maker = self.pools.get_mut(&pool.address)
            .ok_or(PanopticError::PoolNotFound)?;

        if timestamp - maker.last_rebalance < 3600 {
            return Ok(());
        }

        let inventory_imbalance = maker.inventory - self.inventory_target;
        if inventory_imbalance.abs() <= self.rebalance_threshold {
            return Ok(());
        }

        let rebalance_size = U256::from((inventory_imbalance.abs() * 0.5) as u128);
        let is_buy = inventory_imbalance < 0.0;

        for option_type in [OptionType::Call, OptionType::Put] {
            let position = maker.positions.get_mut(&option_type)
                .ok_or(PanopticError::PositionNotFound)?;

            if is_buy {
                if position.size + rebalance_size <= self.max_position {
                    position.size += rebalance_size;
                    maker.inventory += rebalance_size.as_u128() as f64;
                }
            } else {
                if rebalance_size <= position.size {
                    position.size -= rebalance_size;
                    maker.inventory -= rebalance_size.as_u128() as f64;
                }
            }
        }

        maker.last_rebalance = timestamp;
        Ok(())
    }

    pub fn get_position_metrics(
        &self,
        pool: &Pool,
        option_type: OptionType,
    ) -> Result<PositionMetrics> {
        let maker = self.pools.get(&pool.address)
            .ok_or(PanopticError::PoolNotFound)?;
        let position = maker.positions.get(&option_type)
            .ok_or(PanopticError::PositionNotFound)?;

        Ok(PositionMetrics {
            size: position.size,
            entry_price: position.entry_price,
            current_price: position.current_price,
            unrealized_pnl: position.unrealized_pnl,
            delta: position.delta,
            gamma: position.gamma,
        })
    }

    pub fn get_pool_metrics(
        &self,
        pool: &Pool,
    ) -> Result<PoolMetrics> {
        let maker = self.pools.get(&pool.address)
            .ok_or(PanopticError::PoolNotFound)?;

        Ok(PoolMetrics {
            inventory: maker.inventory,
            volume_24h: maker.volume_24h,
            revenue_24h: maker.revenue_24h,
        })
    }
}

#[derive(Debug)]
pub struct Quote {
    pub option_type: OptionType,
    pub bid_price: U256,
    pub ask_price: U256,
    pub bid_size: U256,
    pub ask_size: U256,
    pub timestamp: u64,
}

#[derive(Debug)]
pub struct PositionMetrics {
    pub size: U256,
    pub entry_price: U256,
    pub current_price: U256,
    pub unrealized_pnl: i64,
    pub delta: f64,
    pub gamma: f64,
}

#[derive(Debug)]
pub struct PoolMetrics {
    pub inventory: f64,
    pub volume_24h: U256,
    pub revenue_24h: U256,
}
