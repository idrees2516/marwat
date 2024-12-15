use ethers::prelude::*;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use tokio::sync::RwLock;
use std::time::{SystemTime, UNIX_EPOCH};

use super::GameError;

#[derive(Debug, Clone)]
pub struct FeeConfig {
    pub base_fee: u32,
    pub max_fee: u32,
    pub min_fee: u32,
    pub volume_threshold: U256,
    pub volatility_multiplier: f64,
    pub time_decay: f64,
}

#[derive(Debug, Clone)]
pub struct FeeRecipient {
    pub address: Address,
    pub share: u32,  // Base points (1/10000)
    pub min_amount: U256,
    pub last_claim: u64,
}

#[derive(Debug)]
pub struct DynamicFeeSystem {
    config: FeeConfig,
    volume_tracker: Arc<RwLock<BTreeMap<Address, VolumeData>>>,
    fee_recipients: Arc<RwLock<Vec<FeeRecipient>>>,
    fee_curves: Arc<RwLock<HashMap<(Address, Address), FeeCurve>>>,
    rebate_system: Arc<RwLock<RebateManager>>,
}

#[derive(Debug, Clone)]
struct VolumeData {
    volume_24h: U256,
    last_update: u64,
    fee_paid: U256,
}

#[derive(Debug, Clone)]
struct FeeCurve {
    points: Vec<(U256, u32)>,  // (volume threshold, fee in base points)
    volatility_adjustment: f64,
    time_weights: Vec<(u64, f64)>,  // (time window, weight)
}

#[derive(Debug)]
struct RebateManager {
    trader_volumes: HashMap<Address, U256>,
    rebate_tiers: Vec<(U256, u32)>,  // (volume threshold, rebate in base points)
    unclaimed_rebates: HashMap<Address, U256>,
}

impl DynamicFeeSystem {
    pub fn new(config: FeeConfig) -> Self {
        Self {
            config,
            volume_tracker: Arc::new(RwLock::new(BTreeMap::new())),
            fee_recipients: Arc::new(RwLock::new(Vec::new())),
            fee_curves: Arc::new(RwLock::new(HashMap::new())),
            rebate_system: Arc::new(RwLock::new(RebateManager {
                trader_volumes: HashMap::new(),
                rebate_tiers: Vec::new(),
                unclaimed_rebates: HashMap::new(),
            })),
        }
    }

    pub async fn calculate_dynamic_fee(
        &self,
        token0: Address,
        token1: Address,
        amount: U256,
        volatility: f64,
    ) -> Result<u32, GameError> {
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Get custom fee curve if exists
        let fee_curves = self.fee_curves.read().await;
        if let Some(curve) = fee_curves.get(&(token0, token1)) {
            return self.calculate_fee_from_curve(curve, amount, volatility, current_time).await;
        }

        // Calculate based on volume
        let mut volume_data = self.volume_tracker.write().await;
        let volume = volume_data
            .entry(token0)
            .or_insert(VolumeData {
                volume_24h: U256::zero(),
                last_update: current_time,
                fee_paid: U256::zero(),
            });

        // Update 24h volume
        if current_time - volume.last_update > 86400 {
            volume.volume_24h = amount;
        } else {
            volume.volume_24h = volume.volume_24h.saturating_add(amount);
        }
        volume.last_update = current_time;

        // Calculate fee based on volume and volatility
        let volume_factor = if volume.volume_24h >= self.config.volume_threshold {
            0.8  // 20% discount for high volume
        } else {
            1.0
        };

        let volatility_factor = 1.0 + (volatility * self.config.volatility_multiplier);
        let base_fee = self.config.base_fee as f64;

        let dynamic_fee = (base_fee * volume_factor * volatility_factor) as u32;

        Ok(dynamic_fee
            .max(self.config.min_fee)
            .min(self.config.max_fee))
    }

    async fn calculate_fee_from_curve(
        &self,
        curve: &FeeCurve,
        amount: U256,
        volatility: f64,
        current_time: u64,
    ) -> Result<u32, GameError> {
        // Find applicable fee tier
        let mut base_fee = self.config.base_fee;
        for (threshold, fee) in curve.points.iter() {
            if amount >= *threshold {
                base_fee = *fee;
                break;
            }
        }

        // Apply volatility adjustment
        let volatility_fee = (base_fee as f64 * volatility * curve.volatility_adjustment) as u32;

        // Apply time-based weights
        let mut time_weight = 1.0;
        for (window, weight) in curve.time_weights.iter() {
            if current_time % window == 0 {
                time_weight *= weight;
            }
        }

        let final_fee = (base_fee as f64 * time_weight) as u32 + volatility_fee;
        
        Ok(final_fee
            .max(self.config.min_fee)
            .min(self.config.max_fee))
    }

    pub async fn add_fee_recipient(
        &self,
        recipient: FeeRecipient,
    ) -> Result<(), GameError> {
        let mut recipients = self.fee_recipients.write().await;
        
        // Validate total shares don't exceed 100%
        let total_share: u32 = recipients.iter()
            .map(|r| r.share)
            .sum::<u32>() + recipient.share;
            
        if total_share > 10000 {
            return Err(GameError::InvalidFeeShare);
        }
        
        recipients.push(recipient);
        Ok(())
    }

    pub async fn distribute_fees(
        &self,
        amount: U256,
    ) -> Result<Vec<(Address, U256)>, GameError> {
        let recipients = self.fee_recipients.read().await;
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let mut distributions = Vec::new();
        
        for recipient in recipients.iter() {
            let share_amount = amount * U256::from(recipient.share) / U256::from(10000);
            
            if share_amount >= recipient.min_amount {
                distributions.push((recipient.address, share_amount));
            }
        }

        Ok(distributions)
    }

    pub async fn update_fee_curve(
        &self,
        token0: Address,
        token1: Address,
        curve: FeeCurve,
    ) -> Result<(), GameError> {
        let mut fee_curves = self.fee_curves.write().await;
        fee_curves.insert((token0, token1), curve);
        Ok(())
    }

    pub async fn process_rebate(
        &self,
        trader: Address,
        volume: U256,
    ) -> Result<U256, GameError> {
        let mut rebate_manager = self.rebate_system.write().await;
        
        // Update trader volume
        *rebate_manager.trader_volumes.entry(trader).or_insert(U256::zero()) += volume;
        
        // Calculate rebate
        let trader_volume = rebate_manager.trader_volumes.get(&trader).unwrap();
        let mut rebate_rate = 0u32;
        
        for (threshold, rate) in rebate_manager.rebate_tiers.iter() {
            if trader_volume >= threshold {
                rebate_rate = *rate;
            }
        }
        
        if rebate_rate > 0 {
            let rebate_amount = volume * U256::from(rebate_rate) / U256::from(10000);
            *rebate_manager.unclaimed_rebates.entry(trader).or_insert(U256::zero()) += rebate_amount;
            Ok(rebate_amount)
        } else {
            Ok(U256::zero())
        }
    }

    pub async fn claim_rebate(
        &self,
        trader: Address,
    ) -> Result<U256, GameError> {
        let mut rebate_manager = self.rebate_system.write().await;
        
        if let Some(amount) = rebate_manager.unclaimed_rebates.remove(&trader) {
            Ok(amount)
        } else {
            Ok(U256::zero())
        }
    }

    pub async fn add_rebate_tier(
        &self,
        volume_threshold: U256,
        rebate_rate: u32,
    ) -> Result<(), GameError> {
        let mut rebate_manager = self.rebate_system.write().await;
        rebate_manager.rebate_tiers.push((volume_threshold, rebate_rate));
        rebate_manager.rebate_tiers.sort_by(|a, b| b.0.cmp(&a.0));
        Ok(())
    }
}
