use ethers::prelude::*;
use std::collections::{HashMap, BTreeMap};
use std::sync::Arc;
use tokio::sync::RwLock;
use std::time::{SystemTime, UNIX_EPOCH};

use super::GameError;

#[derive(Debug, Clone)]
pub struct RiskConfig {
    pub max_position_size: U256,
    pub min_collateral_ratio: f64,
    pub max_leverage: f64,
    pub circuit_breaker_threshold: f64,
    pub cooldown_period: u64,
    pub max_slippage: f64,
}

#[derive(Debug, Clone)]
pub struct PositionRisk {
    pub position_id: U256,
    pub risk_score: f64,
    pub liquidation_price: U256,
    pub health_factor: f64,
    pub max_drawdown: f64,
    pub last_update: u64,
}

#[derive(Debug)]
pub struct RiskManager {
    config: RiskConfig,
    position_risks: Arc<RwLock<HashMap<U256, PositionRisk>>>,
    circuit_breakers: Arc<RwLock<HashMap<(Address, Address), CircuitBreaker>>>,
    flash_protection: Arc<RwLock<FlashProtection>>,
    mev_protection: Arc<RwLock<MEVProtection>>,
}

#[derive(Debug)]
struct CircuitBreaker {
    triggered: bool,
    trigger_time: u64,
    trigger_price: U256,
    volume_threshold: U256,
    price_threshold: f64,
    cooldown_end: u64,
}

#[derive(Debug)]
struct FlashProtection {
    price_impact_threshold: f64,
    min_block_delay: u64,
    max_position_ratio: f64,
    recent_trades: BTreeMap<u64, Vec<Trade>>,
}

#[derive(Debug, Clone)]
struct Trade {
    block_number: u64,
    price: U256,
    amount: U256,
    direction: bool,  // true for buy, false for sell
}

#[derive(Debug)]
struct MEVProtection {
    sandwich_threshold: f64,
    backrun_detection_window: u64,
    priority_gas_price: U256,
    protected_functions: Vec<[u8; 4]>,  // Function selectors
    recent_transactions: Vec<Transaction>,
}

#[derive(Debug, Clone)]
struct Transaction {
    hash: H256,
    block_number: u64,
    gas_price: U256,
    method_id: [u8; 4],
    timestamp: u64,
}

impl RiskManager {
    pub fn new(config: RiskConfig) -> Self {
        Self {
            config,
            position_risks: Arc::new(RwLock::new(HashMap::new())),
            circuit_breakers: Arc::new(RwLock::new(HashMap::new())),
            flash_protection: Arc::new(RwLock::new(FlashProtection {
                price_impact_threshold: 0.03,  // 3%
                min_block_delay: 1,
                max_position_ratio: 0.1,  // 10% of pool
                recent_trades: BTreeMap::new(),
            })),
            mev_protection: Arc::new(RwLock::new(MEVProtection {
                sandwich_threshold: 0.02,  // 2%
                backrun_detection_window: 3,  // blocks
                priority_gas_price: U256::from(50000000000u64),  // 50 gwei
                protected_functions: vec![[0; 4]; 0],
                recent_transactions: Vec::new(),
            })),
        }
    }

    pub async fn calculate_position_risk(
        &self,
        position_id: U256,
        current_price: U256,
        volatility: f64,
        collateral: U256,
    ) -> Result<PositionRisk, GameError> {
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Calculate risk metrics
        let leverage = collateral.as_u128() as f64 / self.config.min_collateral_ratio;
        let liquidation_price = current_price
            .saturating_mul(U256::from((1.0 - self.config.min_collateral_ratio) as u64))
            .saturating_div(U256::from(1000000));

        let health_factor = collateral.as_u128() as f64 / 
            (current_price.as_u128() as f64 * self.config.min_collateral_ratio);

        let max_drawdown = volatility * leverage * 2.0;  // 2-sigma event

        // Calculate composite risk score
        let risk_score = (leverage / self.config.max_leverage) * 0.4 +
            (volatility * 252.0).sqrt() * 0.3 +  // Annualized volatility
            (1.0 / health_factor) * 0.3;

        let position_risk = PositionRisk {
            position_id,
            risk_score,
            liquidation_price,
            health_factor,
            max_drawdown,
            last_update: current_time,
        };

        let mut risks = self.position_risks.write().await;
        risks.insert(position_id, position_risk.clone());

        Ok(position_risk)
    }

    pub async fn check_circuit_breaker(
        &self,
        token0: Address,
        token1: Address,
        current_price: U256,
        volume: U256,
    ) -> Result<bool, GameError> {
        let mut breakers = self.circuit_breakers.write().await;
        let breaker = breakers.entry((token0, token1)).or_insert(CircuitBreaker {
            triggered: false,
            trigger_time: 0,
            trigger_price: U256::zero(),
            volume_threshold: U256::from(1000000000000u64),  // 1M units
            price_threshold: 0.1,  // 10%
            cooldown_end: 0,
        });

        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Check if in cooldown
        if breaker.triggered && current_time < breaker.cooldown_end {
            return Ok(true);
        }

        // Reset if cooldown ended
        if breaker.triggered && current_time >= breaker.cooldown_end {
            breaker.triggered = false;
        }

        // Check volume threshold
        if volume > breaker.volume_threshold {
            breaker.triggered = true;
            breaker.trigger_time = current_time;
            breaker.trigger_price = current_price;
            breaker.cooldown_end = current_time + self.config.cooldown_period;
            return Ok(true);
        }

        // Check price movement
        if breaker.trigger_price > U256::zero() {
            let price_change = if current_price > breaker.trigger_price {
                current_price.saturating_sub(breaker.trigger_price)
                    .saturating_mul(U256::from(1000000))
                    .saturating_div(breaker.trigger_price)
            } else {
                breaker.trigger_price.saturating_sub(current_price)
                    .saturating_mul(U256::from(1000000))
                    .saturating_div(breaker.trigger_price)
            };

            if price_change > U256::from((breaker.price_threshold * 1000000.0) as u64) {
                breaker.triggered = true;
                breaker.cooldown_end = current_time + self.config.cooldown_period;
                return Ok(true);
            }
        }

        Ok(false)
    }

    pub async fn detect_flash_attacks(
        &self,
        block_number: u64,
        price: U256,
        amount: U256,
        direction: bool,
    ) -> Result<bool, GameError> {
        let mut flash_protection = self.flash_protection.write().await;
        
        // Add trade to recent trades
        flash_protection.recent_trades
            .entry(block_number)
            .or_insert_with(Vec::new)
            .push(Trade {
                block_number,
                price,
                amount,
                direction,
            });

        // Clean old trades
        flash_protection.recent_trades.retain(|&k, _| 
            k >= block_number.saturating_sub(flash_protection.min_block_delay));

        // Check for suspicious patterns
        let mut total_buy_volume = U256::zero();
        let mut total_sell_volume = U256::zero();
        let mut price_changes = Vec::new();
        let mut prev_price = price;

        for trades in flash_protection.recent_trades.values() {
            for trade in trades {
                if trade.direction {
                    total_buy_volume += trade.amount;
                } else {
                    total_sell_volume += trade.amount;
                }

                let price_change = if trade.price > prev_price {
                    trade.price.saturating_sub(prev_price)
                        .saturating_mul(U256::from(1000000))
                        .saturating_div(prev_price)
                } else {
                    prev_price.saturating_sub(trade.price)
                        .saturating_mul(U256::from(1000000))
                        .saturating_div(prev_price)
                };

                price_changes.push(price_change);
                prev_price = trade.price;
            }
        }

        // Check volume imbalance
        let volume_ratio = if total_sell_volume > U256::zero() {
            total_buy_volume.as_u128() as f64 / total_sell_volume.as_u128() as f64
        } else {
            f64::INFINITY
        };

        // Check price impact
        let total_price_impact: f64 = price_changes.iter()
            .map(|x| x.as_u128() as f64 / 1000000.0)
            .sum();

        Ok(volume_ratio > (1.0 / flash_protection.max_position_ratio) ||
           total_price_impact > flash_protection.price_impact_threshold)
    }

    pub async fn protect_from_mev(
        &self,
        transaction: Transaction,
    ) -> Result<bool, GameError> {
        let mut mev_protection = self.mev_protection.write().await;
        
        // Add transaction to recent list
        mev_protection.recent_transactions.push(transaction.clone());

        // Clean old transactions
        let min_block = transaction.block_number
            .saturating_sub(mev_protection.backrun_detection_window);
        mev_protection.recent_transactions.retain(|tx| 
            tx.block_number >= min_block);

        // Check for sandwich patterns
        let mut potential_sandwich = false;
        let mut similar_transactions = mev_protection.recent_transactions.iter()
            .filter(|tx| tx.method_id == transaction.method_id)
            .collect::<Vec<_>>();
        similar_transactions.sort_by_key(|tx| tx.gas_price);

        if similar_transactions.len() >= 3 {
            for window in similar_transactions.windows(3) {
                if window[0].gas_price < transaction.gas_price &&
                   window[2].gas_price > transaction.gas_price {
                    potential_sandwich = true;
                    break;
                }
            }
        }

        // Check for backrunning
        let high_gas_transactions = mev_protection.recent_transactions.iter()
            .filter(|tx| tx.gas_price > mev_protection.priority_gas_price)
            .count();
        let backrun_risk = high_gas_transactions as f64 / 
            mev_protection.recent_transactions.len() as f64;

        Ok(potential_sandwich || backrun_risk > mev_protection.sandwich_threshold)
    }

    pub async fn add_protected_function(
        &self,
        function_selector: [u8; 4],
    ) -> Result<(), GameError> {
        let mut mev_protection = self.mev_protection.write().await;
        mev_protection.protected_functions.push(function_selector);
        Ok(())
    }

    pub async fn update_circuit_breaker_config(
        &self,
        token0: Address,
        token1: Address,
        volume_threshold: U256,
        price_threshold: f64,
    ) -> Result<(), GameError> {
        let mut breakers = self.circuit_breakers.write().await;
        if let Some(breaker) = breakers.get_mut(&(token0, token1)) {
            breaker.volume_threshold = volume_threshold;
            breaker.price_threshold = price_threshold;
        }
        Ok(())
    }
}
