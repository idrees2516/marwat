use ethers::prelude::*;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use tokio::sync::RwLock;
use std::time::{SystemTime, UNIX_EPOCH};

use super::GameError;

#[derive(Debug, Clone)]
pub struct TickAnalytics {
    pub crossings_count: u64,
    pub time_spent: u64,
    pub liquidity_concentration: u128,
    pub volatility_score: f64,
    pub last_cross_timestamp: u64,
    pub gas_used_historical: Vec<u64>,
}

#[derive(Debug)]
pub struct DynamicTickManager {
    base_spacing: i32,
    volatility_window: u64,
    tick_analytics: Arc<RwLock<BTreeMap<i32, TickAnalytics>>>,
    spacing_tiers: Arc<RwLock<BTreeMap<u64, i32>>>,
    custom_pair_spacing: Arc<RwLock<HashMap<(Address, Address), i32>>>,
    heat_map: Arc<RwLock<BTreeMap<i32, u128>>>,
}

#[derive(Debug, Clone)]
pub struct VolatilityConfig {
    pub base_window: u64,
    pub max_window: u64,
    pub min_observations: u32,
    pub volatility_threshold: f64,
    pub adjustment_speed: f64,
}

impl DynamicTickManager {
    pub fn new(base_spacing: i32, volatility_window: u64) -> Self {
        Self {
            base_spacing,
            volatility_window,
            tick_analytics: Arc::new(RwLock::new(BTreeMap::new())),
            spacing_tiers: Arc::new(RwLock::new(BTreeMap::new())),
            custom_pair_spacing: Arc::new(RwLock::new(HashMap::new())),
            heat_map: Arc::new(RwLock::new(BTreeMap::new())),
        }
    }

    pub async fn calculate_dynamic_spacing(
        &self,
        current_tick: i32,
        token0: Address,
        token1: Address,
    ) -> Result<i32, GameError> {
        // Check for custom pair spacing
        if let Some(&custom_spacing) = self.custom_pair_spacing.read().await.get(&(token0, token1)) {
            return Ok(custom_spacing);
        }

        let analytics = self.tick_analytics.read().await;
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Calculate recent volatility
        let mut recent_volatility = 0.0;
        let mut observation_count = 0;
        
        for (&tick, analytics) in analytics.range((current_tick - 1000)..(current_tick + 1000)) {
            if current_time - analytics.last_cross_timestamp <= self.volatility_window {
                recent_volatility += analytics.volatility_score;
                observation_count += 1;
            }
        }

        if observation_count > 0 {
            recent_volatility /= observation_count as f64;
        }

        // Determine spacing tier based on volatility
        let spacing_tiers = self.spacing_tiers.read().await;
        let mut selected_spacing = self.base_spacing;

        for (&volatility_threshold, &spacing) in spacing_tiers.iter() {
            if recent_volatility >= volatility_threshold as f64 {
                selected_spacing = spacing;
                break;
            }
        }

        Ok(selected_spacing)
    }

    pub async fn record_tick_crossing(
        &self,
        tick: i32,
        gas_used: u64,
        liquidity: u128,
    ) -> Result<(), GameError> {
        let mut analytics = self.tick_analytics.write().await;
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let tick_analytics = analytics.entry(tick).or_insert(TickAnalytics {
            crossings_count: 0,
            time_spent: 0,
            liquidity_concentration: 0,
            volatility_score: 0.0,
            last_cross_timestamp: current_time,
            gas_used_historical: Vec::new(),
        });

        tick_analytics.crossings_count += 1;
        tick_analytics.gas_used_historical.push(gas_used);
        if tick_analytics.gas_used_historical.len() > 1000 {
            tick_analytics.gas_used_historical.remove(0);
        }

        // Update volatility score
        let time_since_last = current_time - tick_analytics.last_cross_timestamp;
        tick_analytics.volatility_score = self.calculate_volatility_score(
            tick_analytics.crossings_count,
            time_since_last,
            tick_analytics.gas_used_historical.as_slice(),
        );

        tick_analytics.last_cross_timestamp = current_time;
        tick_analytics.liquidity_concentration = liquidity;

        // Update heat map
        let mut heat_map = self.heat_map.write().await;
        *heat_map.entry(tick).or_insert(0) += 1;

        Ok(())
    }

    fn calculate_volatility_score(
        &self,
        crossings: u64,
        time_period: u64,
        gas_history: &[u64],
    ) -> f64 {
        let crossing_rate = if time_period > 0 {
            crossings as f64 / time_period as f64
        } else {
            0.0
        };

        let avg_gas = if !gas_history.is_empty() {
            gas_history.iter().sum::<u64>() as f64 / gas_history.len() as f64
        } else {
            0.0
        };

        // Combine metrics into volatility score
        let base_volatility = crossing_rate * 100.0;
        let gas_factor = (avg_gas / 1_000_000.0).min(1.0);
        
        base_volatility * (1.0 + gas_factor)
    }

    pub async fn get_tick_analytics(&self, tick: i32) -> Result<Option<TickAnalytics>, GameError> {
        let analytics = self.tick_analytics.read().await;
        Ok(analytics.get(&tick).cloned())
    }

    pub async fn get_heat_map(
        &self,
        start_tick: i32,
        end_tick: i32,
    ) -> Result<BTreeMap<i32, u128>, GameError> {
        let heat_map = self.heat_map.read().await;
        Ok(heat_map
            .range(start_tick..=end_tick)
            .map(|(&k, &v)| (k, v))
            .collect())
    }

    pub async fn set_custom_pair_spacing(
        &self,
        token0: Address,
        token1: Address,
        spacing: i32,
    ) -> Result<(), GameError> {
        let mut custom_spacing = self.custom_pair_spacing.write().await;
        custom_spacing.insert((token0, token1), spacing);
        Ok(())
    }

    pub async fn add_spacing_tier(
        &self,
        volatility_threshold: u64,
        spacing: i32,
    ) -> Result<(), GameError> {
        let mut spacing_tiers = self.spacing_tiers.write().await;
        spacing_tiers.insert(volatility_threshold, spacing);
        Ok(())
    }

    pub async fn optimize_tick_spacing(
        &self,
        token0: Address,
        token1: Address,
        observation_period: u64,
    ) -> Result<i32, GameError> {
        let analytics = self.tick_analytics.read().await;
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let mut total_crossings = 0u64;
        let mut total_gas = 0u64;
        let mut active_ticks = 0u32;

        for analytics in analytics.values() {
            if current_time - analytics.last_cross_timestamp <= observation_period {
                total_crossings += analytics.crossings_count;
                total_gas += analytics.gas_used_historical.iter().sum::<u64>();
                active_ticks += 1;
            }
        }

        // Calculate optimal spacing based on usage patterns
        let avg_crossings_per_tick = if active_ticks > 0 {
            total_crossings as f64 / active_ticks as f64
        } else {
            0.0
        };

        let avg_gas_per_tick = if active_ticks > 0 {
            total_gas as f64 / active_ticks as f64
        } else {
            0.0
        };

        // Optimize spacing based on metrics
        let base_adjustment = (avg_crossings_per_tick * 0.1) as i32;
        let gas_adjustment = (avg_gas_per_tick / 1_000_000.0) as i32;
        
        let optimal_spacing = self.base_spacing + base_adjustment + gas_adjustment;
        
        // Ensure spacing remains within reasonable bounds
        let final_spacing = optimal_spacing.max(1).min(self.base_spacing * 4);
        
        // Update custom pair spacing
        self.set_custom_pair_spacing(token0, token1, final_spacing).await?;
        
        Ok(final_spacing)
    }
}
