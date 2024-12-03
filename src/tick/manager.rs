use crate::types::{Pool, Result, PanopticError};
use ethers::types::{U256, Address, H256};
use std::collections::{HashMap, BTreeMap};
use std::sync::Arc;
use tokio::sync::RwLock;

use super::{
    TickMath,
    TickBitmap,
    TickState,
    TickOracle,
    TickRange,
};

pub struct TickManager {
    math: Arc<TickMath>,
    bitmap: RwLock<TickBitmap>,
    state: Arc<TickState>,
    oracle: Arc<TickOracle>,
    ranges: RwLock<BTreeMap<(i32, i32), TickRange>>,
    config: TickConfig,
}

#[derive(Clone, Debug)]
pub struct TickConfig {
    pub min_tick: i32,
    pub max_tick: i32,
    pub tick_spacing: i32,
    pub fee_protocol: u32,
    pub max_liquidity_per_tick: U256,
}

#[derive(Clone, Debug)]
pub struct TickUpdate {
    pub tick: i32,
    pub liquidity_delta: i128,
    pub fee_growth: U256,
    pub seconds_per_liquidity: U256,
    pub tick_cumulative: U256,
}

#[derive(Clone, Debug)]
pub struct TickSnapshot {
    pub tick: i32,
    pub liquidity_gross: U256,
    pub liquidity_net: i128,
    pub fee_growth_outside: U256,
    pub seconds_per_liquidity_outside: U256,
    pub tick_cumulative_outside: U256,
    pub initialized: bool,
}

impl TickManager {
    pub fn new(
        config: TickConfig,
        state_manager: Arc<dyn StateManager>,
    ) -> Self {
        Self {
            math: Arc::new(TickMath::new(
                config.min_tick,
                config.max_tick,
                config.tick_spacing,
            )),
            bitmap: RwLock::new(TickBitmap::new(config.tick_spacing)),
            state: Arc::new(TickState::new(config.tick_spacing, state_manager)),
            oracle: Arc::new(TickOracle::new(config.tick_spacing)),
            ranges: RwLock::new(BTreeMap::new()),
            config,
        }
    }

    pub async fn initialize_tick(
        &self,
        tick: i32,
        liquidity_delta: i128,
        current_tick: i32,
        current_time: u32,
        current_liquidity: U256,
    ) -> Result<bool, &'static str> {
        // Validate tick
        if tick < self.config.min_tick || tick > self.config.max_tick {
            return Err("Tick out of range");
        }
        if tick % self.config.tick_spacing != 0 {
            return Err("Invalid tick spacing");
        }

        // Update tick state
        let flipped = self.state.update_tick(
            tick,
            liquidity_delta,
            tick > current_tick,
            current_tick,
            current_time,
            current_liquidity,
        ).await?;

        // Update bitmap if necessary
        if flipped {
            self.bitmap.write().await.flip_tick(tick)?;
        }

        // Update oracle
        self.oracle.update_tick(
            tick,
            current_tick,
            current_time,
            current_liquidity,
        ).await?;

        Ok(flipped)
    }

    pub async fn cross_tick(
        &self,
        tick: i32,
        current_time: u32,
        current_liquidity: U256,
    ) -> Result<i128, &'static str> {
        // Verify tick is initialized
        if !self.bitmap.read().await.is_initialized(tick)? {
            return Err("Tick not initialized");
        }

        // Cross the tick
        let liquidity_net = self.state.cross(
            tick,
            current_time,
            current_liquidity,
        ).await?;

        // Update oracle
        self.oracle.cross_tick(
            tick,
            current_time,
            current_liquidity,
        ).await?;

        Ok(liquidity_net)
    }

    pub async fn get_next_initialized_tick(
        &self,
        tick: i32,
        lte: bool,
    ) -> Result<(i32, bool), &'static str> {
        self.bitmap.read().await.next_initialized_tick(tick, lte)
    }

    pub async fn update_range(
        &self,
        tick_lower: i32,
        tick_upper: i32,
        liquidity_delta: i128,
        current_tick: i32,
        current_time: u32,
        current_liquidity: U256,
    ) -> Result<(), &'static str> {
        // Validate range
        if tick_lower >= tick_upper {
            return Err("Invalid tick range");
        }
        if tick_lower < self.config.min_tick || tick_upper > self.config.max_tick {
            return Err("Tick range out of bounds");
        }

        // Initialize or update ticks
        let flipped_lower = self.initialize_tick(
            tick_lower,
            liquidity_delta,
            current_tick,
            current_time,
            current_liquidity,
        ).await?;

        let flipped_upper = self.initialize_tick(
            tick_upper,
            liquidity_delta,
            current_tick,
            current_time,
            current_liquidity,
        ).await?;

        // Update range tracking
        let mut ranges = self.ranges.write().await;
        let range = ranges.entry((tick_lower, tick_upper)).or_insert_with(|| {
            TickRange::new(tick_lower, tick_upper)
        });

        range.update(
            liquidity_delta,
            current_tick,
            current_time,
            current_liquidity,
        )?;

        Ok(())
    }

    pub async fn get_fee_growth_inside(
        &self,
        tick_lower: i32,
        tick_upper: i32,
        current_tick: i32,
    ) -> Result<U256, &'static str> {
        self.state.get_fee_growth_inside(tick_lower, tick_upper, current_tick).await
    }

    pub async fn get_seconds_per_liquidity_inside(
        &self,
        tick_lower: i32,
        tick_upper: i32,
        current_tick: i32,
        current_time: u32,
        current_liquidity: U256,
    ) -> Result<U256, &'static str> {
        self.state.get_seconds_per_liquidity_inside(
            tick_lower,
            tick_upper,
            current_tick,
            current_time,
            current_liquidity,
        ).await
    }

    pub async fn get_tick_snapshot(&self, tick: i32) -> Result<Option<TickSnapshot>, &'static str> {
        if let Some(info) = self.state.get_tick_info(tick).await? {
            Ok(Some(TickSnapshot {
                tick,
                liquidity_gross: info.liquidity_gross,
                liquidity_net: info.liquidity_net,
                fee_growth_outside: info.fee_growth_outside,
                seconds_per_liquidity_outside: info.seconds_per_liquidity_outside,
                tick_cumulative_outside: info.tick_cumulative_outside,
                initialized: info.initialized,
            }))
        } else {
            Ok(None)
        }
    }

    pub fn get_sqrt_ratio_at_tick(&self, tick: i32) -> Result<U256, &'static str> {
        self.math.get_sqrt_ratio_at_tick(tick)
    }

    pub fn get_tick_at_sqrt_ratio(&self, sqrt_ratio: U256) -> Result<i32, &'static str> {
        self.math.get_tick_at_sqrt_ratio(sqrt_ratio)
    }

    pub async fn get_range_data(
        &self,
        tick_lower: i32,
        tick_upper: i32,
    ) -> Result<Option<&TickRange>, &'static str> {
        Ok(self.ranges.read().await.get(&(tick_lower, tick_upper)))
    }

    pub async fn clear_range(
        &self,
        tick_lower: i32,
        tick_upper: i32,
    ) -> Result<(), &'static str> {
        // Remove range tracking
        self.ranges.write().await.remove(&(tick_lower, tick_upper));

        // Clear tick state if no other ranges use these ticks
        let mut should_clear_lower = true;
        let mut should_clear_upper = true;

        for &(lower, upper) in self.ranges.read().await.keys() {
            if lower == tick_lower || upper == tick_lower {
                should_clear_lower = false;
            }
            if lower == tick_upper || upper == tick_upper {
                should_clear_upper = false;
            }
        }

        if should_clear_lower {
            self.state.clear_tick(tick_lower).await?;
            self.bitmap.write().await.flip_tick(tick_lower)?;
        }

        if should_clear_upper {
            self.state.clear_tick(tick_upper).await?;
            self.bitmap.write().await.flip_tick(tick_upper)?;
        }

        Ok(())
    }

    pub async fn update_global_metrics(
        &self,
        fee_growth_delta: U256,
        time_delta: u32,
        current_liquidity: U256,
    ) -> Result<(), &'static str> {
        self.state.update_global_metrics(
            fee_growth_delta,
            time_delta,
            current_liquidity,
        ).await
    }
}
