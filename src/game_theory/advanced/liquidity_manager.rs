use ethers::prelude::*;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use tokio::sync::RwLock;
use std::time::{SystemTime, UNIX_EPOCH};

use super::GameError;

#[derive(Debug, Clone)]
pub struct LiquidityPosition {
    pub owner: Address,
    pub token0: Address,
    pub token1: Address,
    pub tick_lower: i32,
    pub tick_upper: i32,
    pub liquidity: u128,
    pub fee_growth_inside0_last_x128: U256,
    pub fee_growth_inside1_last_x128: U256,
    pub tokens_owed0: u128,
    pub tokens_owed1: u128,
    pub created_at: u64,
    pub last_update: u64,
}

#[derive(Debug, Clone)]
pub struct RangeStrategy {
    pub base_range: i32,
    pub volatility_multiplier: f64,
    pub rebalance_threshold: f64,
    pub min_time_between_rebalance: u64,
    pub max_positions_per_range: u32,
}

#[derive(Debug)]
pub struct AdvancedLiquidityManager {
    positions: Arc<RwLock<BTreeMap<U256, LiquidityPosition>>>,
    range_strategies: Arc<RwLock<HashMap<(Address, Address), RangeStrategy>>>,
    utilization_metrics: Arc<RwLock<HashMap<(Address, Address), UtilizationMetrics>>>,
    il_protection: Arc<RwLock<ImpermanentLossProtection>>,
}

#[derive(Debug, Clone)]
struct UtilizationMetrics {
    liquidity_distribution: BTreeMap<i32, u128>,
    volume_per_tick: BTreeMap<i32, U256>,
    fees_earned_per_tick: BTreeMap<i32, U256>,
    last_update: u64,
}

#[derive(Debug)]
struct ImpermanentLossProtection {
    coverage_ratio: f64,
    min_hold_time: u64,
    max_protection: U256,
    protection_curve: Vec<(u64, f64)>,  // (hold time, protection ratio)
    claims: HashMap<Address, Vec<ILClaim>>,
}

#[derive(Debug, Clone)]
struct ILClaim {
    position_id: U256,
    entry_price0: U256,
    entry_price1: U256,
    claim_time: u64,
    protection_amount: U256,
}

impl AdvancedLiquidityManager {
    pub fn new() -> Self {
        Self {
            positions: Arc::new(RwLock::new(BTreeMap::new())),
            range_strategies: Arc::new(RwLock::new(HashMap::new())),
            utilization_metrics: Arc::new(RwLock::new(HashMap::new())),
            il_protection: Arc::new(RwLock::new(ImpermanentLossProtection {
                coverage_ratio: 0.5,
                min_hold_time: 7 * 24 * 3600,  // 1 week
                max_protection: U256::from(1000000),  // 1M units
                protection_curve: vec![
                    (7 * 24 * 3600, 0.3),   // 30% after 1 week
                    (30 * 24 * 3600, 0.5),  // 50% after 1 month
                    (90 * 24 * 3600, 0.8),  // 80% after 3 months
                ],
                claims: HashMap::new(),
            })),
        }
    }

    pub async fn create_multi_range_position(
        &self,
        owner: Address,
        token0: Address,
        token1: Address,
        current_tick: i32,
        total_liquidity: u128,
        num_ranges: u32,
    ) -> Result<Vec<U256>, GameError> {
        let strategy = self.range_strategies.read().await
            .get(&(token0, token1))
            .cloned()
            .ok_or(GameError::StrategyNotFound)?;

        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let mut position_ids = Vec::new();
        let liquidity_per_range = total_liquidity / num_ranges as u128;

        for i in 0..num_ranges {
            let range_multiplier = 1.0 + (i as f64 * strategy.volatility_multiplier);
            let tick_range = (strategy.base_range as f64 * range_multiplier) as i32;
            
            let tick_lower = current_tick - tick_range;
            let tick_upper = current_tick + tick_range;

            let position = LiquidityPosition {
                owner,
                token0,
                token1,
                tick_lower,
                tick_upper,
                liquidity: liquidity_per_range,
                fee_growth_inside0_last_x128: U256::zero(),
                fee_growth_inside1_last_x128: U256::zero(),
                tokens_owed0: 0,
                tokens_owed1: 0,
                created_at: current_time,
                last_update: current_time,
            };

            let position_id = U256::from(current_time as u64)
                + U256::from(i)
                + U256::from(total_liquidity);

            let mut positions = self.positions.write().await;
            positions.insert(position_id, position);
            position_ids.push(position_id);
        }

        Ok(position_ids)
    }

    pub async fn optimize_position_ranges(
        &self,
        token0: Address,
        token1: Address,
        current_tick: i32,
        current_price: U256,
    ) -> Result<Vec<(i32, i32)>, GameError> {
        let metrics = self.utilization_metrics.read().await
            .get(&(token0, token1))
            .cloned()
            .ok_or(GameError::MetricsNotFound)?;

        let mut optimal_ranges = Vec::new();
        let mut total_volume = U256::zero();
        let mut volume_weighted_ticks = Vec::new();

        // Calculate volume-weighted tick ranges
        for (&tick, &volume) in metrics.volume_per_tick.iter() {
            total_volume += volume;
            volume_weighted_ticks.push((tick, volume));
        }

        volume_weighted_ticks.sort_by(|a, b| b.1.cmp(&a.1));

        // Generate optimal ranges based on volume distribution
        let mut covered_volume = U256::zero();
        let target_coverage = total_volume * U256::from(80) / U256::from(100);  // Target 80% volume coverage

        for (tick, volume) in volume_weighted_ticks {
            covered_volume += volume;
            
            let range_size = (volume * U256::from(100) / total_volume).as_u32() as i32;
            let tick_lower = tick - range_size;
            let tick_upper = tick + range_size;
            
            optimal_ranges.push((tick_lower, tick_upper));

            if covered_volume >= target_coverage {
                break;
            }
        }

        Ok(optimal_ranges)
    }

    pub async fn calculate_rebalance_need(
        &self,
        position_id: U256,
        current_tick: i32,
    ) -> Result<bool, GameError> {
        let positions = self.positions.read().await;
        let position = positions.get(&position_id).ok_or(GameError::PositionNotFound)?;

        let strategy = self.range_strategies.read().await
            .get(&(position.token0, position.token1))
            .cloned()
            .ok_or(GameError::StrategyNotFound)?;

        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Check minimum time between rebalances
        if current_time - position.last_update < strategy.min_time_between_rebalance {
            return Ok(false);
        }

        // Calculate position utilization
        let range_center = (position.tick_lower + position.tick_upper) / 2;
        let distance_from_center = (current_tick - range_center).abs() as f64;
        let range_size = (position.tick_upper - position.tick_lower) as f64;
        
        let utilization_ratio = distance_from_center / range_size;

        Ok(utilization_ratio > strategy.rebalance_threshold)
    }

    pub async fn calculate_impermanent_loss(
        &self,
        position_id: U256,
        current_price0: U256,
        current_price1: U256,
    ) -> Result<U256, GameError> {
        let positions = self.positions.read().await;
        let position = positions.get(&position_id).ok_or(GameError::PositionNotFound)?;

        let il_protection = self.il_protection.read().await;
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Find applicable IL claim
        let claims = il_protection.claims.get(&position.owner)
            .ok_or(GameError::NoILClaims)?;

        let claim = claims.iter()
            .find(|c| c.position_id == position_id)
            .ok_or(GameError::NoILClaims)?;

        // Calculate price changes
        let price0_change = current_price0
            .saturating_mul(U256::from(1000000))
            .saturating_div(claim.entry_price0);
        let price1_change = current_price1
            .saturating_mul(U256::from(1000000))
            .saturating_div(claim.entry_price1);

        // Calculate IL using standard formula
        let sqrt_k = (price0_change * price1_change).integer_sqrt();
        let il_ratio = U256::from(2000000)  // 2 * 1000000 for precision
            .saturating_sub(sqrt_k)
            .saturating_div(U256::from(1000000));

        // Calculate protection amount based on hold time
        let hold_time = current_time - position.created_at;
        let mut protection_ratio = 0.0;

        for &(time_threshold, ratio) in il_protection.protection_curve.iter() {
            if hold_time >= time_threshold {
                protection_ratio = ratio;
            }
        }

        let protection_amount = il_ratio
            .saturating_mul(U256::from((protection_ratio * 1000000.0) as u64))
            .saturating_div(U256::from(1000000));

        Ok(protection_amount.min(il_protection.max_protection))
    }

    pub async fn update_utilization_metrics(
        &self,
        token0: Address,
        token1: Address,
        tick: i32,
        volume: U256,
        fees: U256,
    ) -> Result<(), GameError> {
        let mut metrics = self.utilization_metrics.write().await;
        let entry = metrics.entry((token0, token1)).or_insert(UtilizationMetrics {
            liquidity_distribution: BTreeMap::new(),
            volume_per_tick: BTreeMap::new(),
            fees_earned_per_tick: BTreeMap::new(),
            last_update: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        });

        *entry.volume_per_tick.entry(tick).or_insert(U256::zero()) += volume;
        *entry.fees_earned_per_tick.entry(tick).or_insert(U256::zero()) += fees;

        Ok(())
    }
}
