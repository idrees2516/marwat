use crate::types::{Pool, Result, PanopticError, OptionType, Settlement};
use crate::pricing::models::OptionPricing;
use crate::risk::manager::RiskManager;
use ethers::types::{U256, Address, H256};
use std::collections::{HashMap, BTreeMap};
use std::sync::Arc;
use tokio::sync::RwLock;

pub struct SettlementEngine {
    pricing: Arc<dyn OptionPricing>,
    risk_manager: Arc<RiskManager>,
    settlements: RwLock<BTreeMap<u64, Vec<Settlement>>>,
    position_states: HashMap<Address, PositionState>,
    netting_groups: HashMap<Address, Vec<Address>>,
    settlement_queue: Vec<PendingSettlement>,
    min_collateral_ratio: f64,
    max_settlement_delay: u64,
    gas_price_threshold: U256,
}

struct PositionState {
    collateral: U256,
    margin_requirement: U256,
    unrealized_pnl: i64,
    pending_settlements: Vec<H256>,
    last_settlement: u64,
}

struct PendingSettlement {
    id: H256,
    pool: Address,
    option_type: OptionType,
    size: U256,
    strike_price: U256,
    settlement_price: U256,
    timestamp: u64,
    priority: u8,
}

impl SettlementEngine {
    pub fn new(
        pricing: Arc<dyn OptionPricing>,
        risk_manager: Arc<RiskManager>,
        min_collateral_ratio: f64,
        max_settlement_delay: u64,
        gas_price_threshold: U256,
    ) -> Self {
        Self {
            pricing,
            risk_manager,
            settlements: RwLock::new(BTreeMap::new()),
            position_states: HashMap::new(),
            netting_groups: HashMap::new(),
            settlement_queue: Vec::new(),
            min_collateral_ratio,
            max_settlement_delay,
            gas_price_threshold,
        }
    }

    pub async fn add_settlement(
        &mut self,
        pool: &Pool,
        option_type: OptionType,
        size: U256,
        strike_price: U256,
        timestamp: u64,
    ) -> Result<H256> {
        let settlement_price = self.pricing.get_settlement_price(pool)?;
        let priority = self.calculate_settlement_priority(pool, size, timestamp)?;
        
        let id = self.generate_settlement_id(pool, option_type, timestamp);
        
        let pending = PendingSettlement {
            id,
            pool: pool.address,
            option_type,
            size,
            strike_price,
            settlement_price,
            timestamp,
            priority,
        };

        self.settlement_queue.push(pending);
        self.sort_settlement_queue();

        Ok(id)
    }

    pub async fn process_settlements(&mut self, current_timestamp: u64) -> Result<Vec<H256>> {
        let mut processed = Vec::new();
        let gas_price = self.get_current_gas_price()?;

        if gas_price > self.gas_price_threshold {
            return Ok(processed);
        }

        while let Some(settlement) = self.get_next_settlement() {
            if !self.should_process_settlement(&settlement, current_timestamp)? {
                break;
            }

            let position = self.position_states.get_mut(&settlement.pool)
                .ok_or(PanopticError::PositionNotFound)?;

            // Calculate settlement amount
            let amount = self.calculate_settlement_amount(
                &settlement.option_type,
                settlement.size,
                settlement.strike_price,
                settlement.settlement_price,
            )?;

            // Update position state
            position.unrealized_pnl += amount;
            position.pending_settlements.retain(|&id| id != settlement.id);
            position.last_settlement = current_timestamp;

            // Add to settlements history
            let mut settlements = self.settlements.write().await;
            settlements.entry(current_timestamp)
                .or_insert_with(Vec::new)
                .push(Settlement {
                    id: settlement.id,
                    pool: settlement.pool,
                    option_type: settlement.option_type,
                    size: settlement.size,
                    strike_price: settlement.strike_price,
                    settlement_price: settlement.settlement_price,
                    amount,
                    timestamp: current_timestamp,
                });

            processed.push(settlement.id);
        }

        Ok(processed)
    }

    pub async fn net_positions(&mut self, addresses: &[Address]) -> Result<U256> {
        let mut total_collateral = U256::zero();
        let mut total_requirement = U256::zero();

        for &address in addresses {
            let position = self.position_states.get(&address)
                .ok_or(PanopticError::PositionNotFound)?;

            total_collateral += position.collateral;
            total_requirement += position.margin_requirement;
        }

        if total_collateral < total_requirement {
            return Err(PanopticError::InsufficientCollateral);
        }

        self.netting_groups.insert(
            addresses[0],
            addresses.to_vec(),
        );

        Ok(total_collateral - total_requirement)
    }

    pub async fn reconcile_settlements(
        &mut self,
        start_timestamp: u64,
        end_timestamp: u64,
    ) -> Result<Vec<Settlement>> {
        let settlements = self.settlements.read().await;
        let mut reconciled = Vec::new();

        for (&timestamp, settlements) in settlements.range(start_timestamp..=end_timestamp) {
            for settlement in settlements {
                let position = self.position_states.get(&settlement.pool)
                    .ok_or(PanopticError::PositionNotFound)?;

                if position.last_settlement < timestamp {
                    reconciled.push(settlement.clone());
                }
            }
        }

        Ok(reconciled)
    }

    fn calculate_settlement_priority(
        &self,
        pool: &Pool,
        size: U256,
        timestamp: u64,
    ) -> Result<u8> {
        let position = self.position_states.get(&pool.address)
            .ok_or(PanopticError::PositionNotFound)?;

        let age = timestamp - position.last_settlement;
        let value = size * self.pricing.get_spot_price(pool)?;
        
        let priority = if age > self.max_settlement_delay {
            0 // Highest priority
        } else if value > U256::from(1000000000000000000u128) { // > 1 ETH
            1 // High value positions
        } else if position.unrealized_pnl < 0 {
            2 // Negative PnL positions
        } else if position.collateral < position.margin_requirement {
            3 // Under-collateralized positions
        } else {
            4 // Normal priority
        };

        Ok(priority)
    }

    fn should_process_settlement(
        &self,
        settlement: &PendingSettlement,
        current_timestamp: u64,
    ) -> Result<bool> {
        // Check if settlement is too old
        if current_timestamp < settlement.timestamp {
            return Ok(false);
        }

        let age = current_timestamp - settlement.timestamp;
        if age > self.max_settlement_delay {
            return Ok(true);
        }

        // Check position state
        let position = self.position_states.get(&settlement.pool)
            .ok_or(PanopticError::PositionNotFound)?;

        // Process if position is under-collateralized
        let collateral_ratio = position.collateral.as_u128() as f64 / 
            position.margin_requirement.as_u128() as f64;
        if collateral_ratio < self.min_collateral_ratio {
            return Ok(true);
        }

        // Process if significant unrealized PnL
        if position.unrealized_pnl.abs() > 1000000000 { // > 1 GWEI
            return Ok(true);
        }

        // Process based on priority
        Ok(settlement.priority <= 2) // Process high priority settlements
    }

    fn calculate_settlement_amount(
        &self,
        option_type: &OptionType,
        size: U256,
        strike_price: U256,
        settlement_price: U256,
    ) -> Result<i64> {
        let amount = match option_type {
            OptionType::Call => {
                if settlement_price > strike_price {
                    // Call option is in the money
                    let profit = settlement_price.saturating_sub(strike_price);
                    size.saturating_mul(profit)
                } else {
                    U256::zero()
                }
            },
            OptionType::Put => {
                if strike_price > settlement_price {
                    // Put option is in the money
                    let profit = strike_price.saturating_sub(settlement_price);
                    size.saturating_mul(profit)
                } else {
                    U256::zero()
                }
            },
            OptionType::CallSpread(upper_strike) => {
                if settlement_price > strike_price {
                    // Call spread payoff is capped at upper strike
                    let profit = settlement_price
                        .min(*upper_strike)
                        .saturating_sub(strike_price);
                    size.saturating_mul(profit)
                } else {
                    U256::zero()
                }
            },
            OptionType::PutSpread(lower_strike) => {
                if strike_price > settlement_price {
                    // Put spread payoff is capped at lower strike
                    let profit = strike_price
                        .saturating_sub(settlement_price.max(*lower_strike));
                    size.saturating_mul(profit)
                } else {
                    U256::zero()
                }
            },
        };

        // Convert to signed integer for PnL tracking
        Ok(amount.as_u64() as i64)
    }

    fn generate_settlement_id(
        &self,
        pool: &Pool,
        option_type: OptionType,
        timestamp: u64,
    ) -> H256 {
        let mut bytes = [0u8; 32];
        
        // First 20 bytes: pool address
        bytes[0..20].copy_from_slice(pool.address.as_bytes());
        
        // Next 1 byte: option type
        bytes[20] = match option_type {
            OptionType::Call => 0,
            OptionType::Put => 1,
            OptionType::CallSpread(_) => 2,
            OptionType::PutSpread(_) => 3,
        };
        
        // Last 8 bytes: timestamp
        bytes[24..32].copy_from_slice(&timestamp.to_be_bytes());
        
        H256::from(bytes)
    }

    fn sort_settlement_queue(&mut self) {
        self.settlement_queue.sort_by_key(|s| (s.priority, s.timestamp));
    }

    fn get_next_settlement(&mut self) -> Option<PendingSettlement> {
        self.settlement_queue.pop()
    }

    fn get_current_gas_price(&self) -> Result<U256> {
        // In a real implementation, this would query the network
        // For now, return a reasonable default
        Ok(U256::from(50000000000u64)) // 50 GWEI
    }
}
