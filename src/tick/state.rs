use ethers::types::{U256, Address};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

pub struct TickState {
    tick_info: RwLock<HashMap<i32, TickInfo>>,
    fee_growth_global: RwLock<U256>,
    liquidity_growth_global: RwLock<U256>,
    tick_spacing: i32,
    state_manager: Arc<dyn StateManager>,
}

#[derive(Clone, Debug)]
pub struct TickInfo {
    pub liquidity_gross: U256,
    pub liquidity_net: i128,
    pub fee_growth_outside: U256,
    pub fee_growth_inside: U256,
    pub seconds_outside: u32,
    pub seconds_per_liquidity_outside: U256,
    pub tick_cumulative_outside: U256,
    pub initialized: bool,
}

#[async_trait::async_trait]
pub trait StateManager: Send + Sync {
    async fn load_tick_info(&self, tick: i32) -> Result<Option<TickInfo>, &'static str>;
    async fn save_tick_info(&self, tick: i32, info: &TickInfo) -> Result<(), &'static str>;
    async fn update_global_metrics(
        &self,
        fee_growth: U256,
        liquidity_growth: U256,
    ) -> Result<(), &'static str>;
}

impl TickState {
    pub fn new(tick_spacing: i32, state_manager: Arc<dyn StateManager>) -> Self {
        Self {
            tick_info: RwLock::new(HashMap::new()),
            fee_growth_global: RwLock::new(U256::zero()),
            liquidity_growth_global: RwLock::new(U256::zero()),
            tick_spacing,
            state_manager,
        }
    }

    pub async fn update_tick(
        &self,
        tick: i32,
        liquidity_delta: i128,
        upper: bool,
        current_tick: i32,
        current_time: u32,
        current_liquidity: U256,
    ) -> Result<bool, &'static str> {
        if tick % self.tick_spacing != 0 {
            return Err("Tick not aligned with spacing");
        }

        let mut tick_map = self.tick_info.write().await;
        let fee_growth_global = *self.fee_growth_global.read().await;
        let liquidity_growth_global = *self.liquidity_growth_global.read().await;

        let info = tick_map.entry(tick).or_insert_with(|| TickInfo {
            liquidity_gross: U256::zero(),
            liquidity_net: 0,
            fee_growth_outside: U256::zero(),
            fee_growth_inside: U256::zero(),
            seconds_outside: 0,
            seconds_per_liquidity_outside: U256::zero(),
            tick_cumulative_outside: U256::zero(),
            initialized: false,
        });

        let liquidity_gross_before = info.liquidity_gross;

        // Update liquidity tracking
        info.liquidity_gross = if liquidity_delta > 0 {
            info.liquidity_gross.saturating_add(U256::from(liquidity_delta as u128))
        } else {
            info.liquidity_gross.saturating_sub(U256::from((-liquidity_delta) as u128))
        };

        info.liquidity_net = if upper {
            info.liquidity_net.saturating_sub(liquidity_delta)
        } else {
            info.liquidity_net.saturating_add(liquidity_delta)
        };

        // Update fee growth tracking
        if current_tick >= tick {
            info.fee_growth_outside = fee_growth_global;
            info.seconds_outside = current_time;
            if current_liquidity > U256::zero() {
                info.seconds_per_liquidity_outside = liquidity_growth_global
                    .saturating_div(current_liquidity);
            }
            info.tick_cumulative_outside = U256::from(current_tick)
                .saturating_mul(U256::from(current_time));
        }

        let flipped = liquidity_gross_before == U256::zero() && info.liquidity_gross > U256::zero()
            || info.liquidity_gross == U256::zero() && liquidity_gross_before > U256::zero();

        info.initialized = info.liquidity_gross > U256::zero();

        // Persist state
        self.state_manager.save_tick_info(tick, info).await?;

        Ok(flipped)
    }

    pub async fn cross(
        &self,
        tick: i32,
        current_time: u32,
        current_liquidity: U256,
    ) -> Result<i128, &'static str> {
        let mut tick_map = self.tick_info.write().await;
        let fee_growth_global = *self.fee_growth_global.read().await;
        let liquidity_growth_global = *self.liquidity_growth_global.read().await;

        let info = if let Some(info) = tick_map.get_mut(&tick) {
            info
        } else {
            return Ok(0);
        };

        // Update fee growth tracking
        info.fee_growth_outside = fee_growth_global
            .saturating_sub(info.fee_growth_outside);
        
        info.seconds_outside = current_time
            .saturating_sub(info.seconds_outside);

        if current_liquidity > U256::zero() {
            info.seconds_per_liquidity_outside = liquidity_growth_global
                .saturating_div(current_liquidity)
                .saturating_sub(info.seconds_per_liquidity_outside);
        }

        info.tick_cumulative_outside = U256::from(tick)
            .saturating_mul(U256::from(current_time))
            .saturating_sub(info.tick_cumulative_outside);

        // Persist state
        self.state_manager.save_tick_info(tick, info).await?;

        Ok(info.liquidity_net)
    }

    pub async fn get_fee_growth_inside(
        &self,
        tick_lower: i32,
        tick_upper: i32,
        current_tick: i32,
    ) -> Result<U256, &'static str> {
        if tick_lower > tick_upper {
            return Err("Invalid tick range");
        }

        let tick_map = self.tick_info.read().await;
        let fee_growth_global = *self.fee_growth_global.read().await;

        let lower_info = tick_map.get(&tick_lower)
            .ok_or("Lower tick not initialized")?;
        let upper_info = tick_map.get(&tick_upper)
            .ok_or("Upper tick not initialized")?;

        let fee_growth_below = if current_tick >= tick_lower {
            lower_info.fee_growth_outside
        } else {
            fee_growth_global.saturating_sub(lower_info.fee_growth_outside)
        };

        let fee_growth_above = if current_tick >= tick_upper {
            upper_info.fee_growth_outside
        } else {
            fee_growth_global.saturating_sub(upper_info.fee_growth_outside)
        };

        Ok(fee_growth_global
            .saturating_sub(fee_growth_below)
            .saturating_sub(fee_growth_above))
    }

    pub async fn update_global_metrics(
        &self,
        fee_growth_delta: U256,
        time_delta: u32,
        current_liquidity: U256,
    ) -> Result<(), &'static str> {
        let mut fee_growth = self.fee_growth_global.write().await;
        let mut liquidity_growth = self.liquidity_growth_global.write().await;

        *fee_growth = fee_growth.saturating_add(fee_growth_delta);
        
        if current_liquidity > U256::zero() {
            *liquidity_growth = liquidity_growth.saturating_add(
                U256::from(time_delta).saturating_mul(current_liquidity)
            );
        }

        self.state_manager.update_global_metrics(*fee_growth, *liquidity_growth).await?;

        Ok(())
    }

    pub async fn get_tick_info(&self, tick: i32) -> Result<Option<TickInfo>, &'static str> {
        // Try memory cache first
        if let Some(info) = self.tick_info.read().await.get(&tick) {
            return Ok(Some(info.clone()));
        }

        // Load from persistent storage
        self.state_manager.load_tick_info(tick).await
    }

    pub async fn clear_tick(&self, tick: i32) -> Result<(), &'static str> {
        let mut tick_map = self.tick_info.write().await;
        tick_map.remove(&tick);

        // Clear from persistent storage
        self.state_manager.save_tick_info(tick, &TickInfo {
            liquidity_gross: U256::zero(),
            liquidity_net: 0,
            fee_growth_outside: U256::zero(),
            fee_growth_inside: U256::zero(),
            seconds_outside: 0,
            seconds_per_liquidity_outside: U256::zero(),
            tick_cumulative_outside: U256::zero(),
            initialized: false,
        }).await
    }

    pub async fn get_seconds_per_liquidity_inside(
        &self,
        tick_lower: i32,
        tick_upper: i32,
        current_tick: i32,
        current_time: u32,
        current_liquidity: U256,
    ) -> Result<U256, &'static str> {
        if tick_lower > tick_upper {
            return Err("Invalid tick range");
        }

        let tick_map = self.tick_info.read().await;
        let liquidity_growth_global = *self.liquidity_growth_global.read().await;

        let lower_info = tick_map.get(&tick_lower)
            .ok_or("Lower tick not initialized")?;
        let upper_info = tick_map.get(&tick_upper)
            .ok_or("Upper tick not initialized")?;

        let time_above = if current_tick >= tick_upper {
            current_time.saturating_sub(upper_info.seconds_outside)
        } else {
            upper_info.seconds_outside
        };

        let time_below = if current_tick >= tick_lower {
            current_time.saturating_sub(lower_info.seconds_outside)
        } else {
            lower_info.seconds_outside
        };

        let time_inside = current_time
            .saturating_sub(time_below)
            .saturating_sub(time_above);

        if current_liquidity == U256::zero() {
            return Ok(U256::zero());
        }

        Ok(U256::from(time_inside).saturating_mul(current_liquidity))
    }
}
