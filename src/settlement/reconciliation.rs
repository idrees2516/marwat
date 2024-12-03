use crate::types::{Pool, Result, PanopticError, Settlement};
use ethers::types::{U256, Address, H256};
use std::collections::{HashMap, BTreeMap};
use std::sync::Arc;
use chrono::{DateTime, Utc};

pub struct Reconciliation {
    settlements: BTreeMap<u64, Vec<Settlement>>,
    position_history: HashMap<Address, Vec<PositionSnapshot>>,
    reconciliation_window: u64,
    min_check_interval: u64,
    max_deviation: f64,
}

struct PositionSnapshot {
    timestamp: u64,
    size: U256,
    collateral: U256,
    unrealized_pnl: i64,
    realized_pnl: i64,
    funding_payments: i64,
}

#[derive(Debug)]
pub struct ReconciliationReport {
    pub position: Address,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub initial_state: PositionState,
    pub final_state: PositionState,
    pub settlements: Vec<Settlement>,
    pub discrepancies: Vec<Discrepancy>,
    pub total_deviation: f64,
}

#[derive(Debug)]
pub struct PositionState {
    pub size: U256,
    pub collateral: U256,
    pub unrealized_pnl: i64,
    pub realized_pnl: i64,
    pub funding_payments: i64,
}

#[derive(Debug)]
pub struct Discrepancy {
    pub timestamp: DateTime<Utc>,
    pub field: String,
    pub expected: String,
    pub actual: String,
    pub deviation: f64,
}

impl Reconciliation {
    pub fn new(
        reconciliation_window: u64,
        min_check_interval: u64,
        max_deviation: f64,
    ) -> Self {
        Self {
            settlements: BTreeMap::new(),
            position_history: HashMap::new(),
            reconciliation_window,
            min_check_interval,
            max_deviation,
        }
    }

    pub fn add_settlement(
        &mut self,
        settlement: Settlement,
    ) -> Result<()> {
        self.settlements.entry(settlement.timestamp)
            .or_insert_with(Vec::new)
            .push(settlement);
        Ok(())
    }

    pub fn record_position_snapshot(
        &mut self,
        position: Address,
        size: U256,
        collateral: U256,
        unrealized_pnl: i64,
        realized_pnl: i64,
        funding_payments: i64,
        timestamp: u64,
    ) -> Result<()> {
        let snapshot = PositionSnapshot {
            timestamp,
            size,
            collateral,
            unrealized_pnl,
            realized_pnl,
            funding_payments,
        };

        self.position_history.entry(position)
            .or_insert_with(Vec::new)
            .push(snapshot);

        Ok(())
    }

    pub fn reconcile_position(
        &self,
        position: Address,
        start_time: u64,
        end_time: u64,
    ) -> Result<ReconciliationReport> {
        let history = self.position_history.get(&position)
            .ok_or(PanopticError::PositionNotFound)?;

        let initial_snapshot = history.iter()
            .find(|s| s.timestamp >= start_time)
            .ok_or(PanopticError::NoDataAvailable)?;

        let final_snapshot = history.iter()
            .rev()
            .find(|s| s.timestamp <= end_time)
            .ok_or(PanopticError::NoDataAvailable)?;

        let mut settlements: Vec<Settlement> = Vec::new();
        let mut discrepancies: Vec<Discrepancy> = Vec::new();
        let mut total_deviation = 0.0;

        // Collect all settlements in the time range
        for (&timestamp, settlement_list) in self.settlements.range(start_time..=end_time) {
            for settlement in settlement_list {
                if settlement.pool == position {
                    settlements.push(settlement.clone());
                }
            }
        }

        // Calculate expected final state
        let mut expected_state = PositionState {
            size: initial_snapshot.size,
            collateral: initial_snapshot.collateral,
            unrealized_pnl: initial_snapshot.unrealized_pnl,
            realized_pnl: initial_snapshot.realized_pnl,
            funding_payments: initial_snapshot.funding_payments,
        };

        // Apply settlements to expected state
        for settlement in &settlements {
            expected_state.realized_pnl += settlement.amount;
            expected_state.unrealized_pnl -= settlement.amount;
        }

        // Compare with actual final state
        let actual_state = PositionState {
            size: final_snapshot.size,
            collateral: final_snapshot.collateral,
            unrealized_pnl: final_snapshot.unrealized_pnl,
            realized_pnl: final_snapshot.realized_pnl,
            funding_payments: final_snapshot.funding_payments,
        };

        // Check for discrepancies
        self.check_discrepancy(
            "size",
            expected_state.size,
            actual_state.size,
            end_time,
            &mut discrepancies,
            &mut total_deviation,
        );

        self.check_value_discrepancy(
            "unrealized_pnl",
            expected_state.unrealized_pnl,
            actual_state.unrealized_pnl,
            end_time,
            &mut discrepancies,
            &mut total_deviation,
        );

        self.check_value_discrepancy(
            "realized_pnl",
            expected_state.realized_pnl,
            actual_state.realized_pnl,
            end_time,
            &mut discrepancies,
            &mut total_deviation,
        );

        Ok(ReconciliationReport {
            position,
            start_time: DateTime::from_timestamp(start_time as i64, 0)
                .unwrap_or_default(),
            end_time: DateTime::from_timestamp(end_time as i64, 0)
                .unwrap_or_default(),
            initial_state: PositionState {
                size: initial_snapshot.size,
                collateral: initial_snapshot.collateral,
                unrealized_pnl: initial_snapshot.unrealized_pnl,
                realized_pnl: initial_snapshot.realized_pnl,
                funding_payments: initial_snapshot.funding_payments,
            },
            final_state: actual_state,
            settlements,
            discrepancies,
            total_deviation,
        })
    }

    fn check_discrepancy(
        &self,
        field: &str,
        expected: U256,
        actual: U256,
        timestamp: u64,
        discrepancies: &mut Vec<Discrepancy>,
        total_deviation: &mut f64,
    ) {
        if expected != actual {
            let max_val = expected.max(actual);
            let min_val = expected.min(actual);
            let deviation = if max_val > U256::zero() {
                (max_val.saturating_sub(min_val)).as_u128() as f64 / max_val.as_u128() as f64
            } else {
                0.0
            };

            if deviation > self.max_deviation {
                discrepancies.push(Discrepancy {
                    timestamp: DateTime::from_timestamp(timestamp as i64, 0)
                        .unwrap_or_default(),
                    field: field.to_string(),
                    expected: expected.to_string(),
                    actual: actual.to_string(),
                    deviation,
                });

                *total_deviation += deviation;
            }
        }
    }

    fn check_value_discrepancy(
        &self,
        field: &str,
        expected: i64,
        actual: i64,
        timestamp: u64,
        discrepancies: &mut Vec<Discrepancy>,
        total_deviation: &mut f64,
    ) {
        if expected != actual {
            let max_val = expected.abs().max(actual.abs());
            let deviation = if max_val > 0 {
                (expected - actual).abs() as f64 / max_val as f64
            } else {
                0.0
            };

            if deviation > self.max_deviation {
                discrepancies.push(Discrepancy {
                    timestamp: DateTime::from_timestamp(timestamp as i64, 0)
                        .unwrap_or_default(),
                    field: field.to_string(),
                    expected: expected.to_string(),
                    actual: actual.to_string(),
                    deviation,
                });

                *total_deviation += deviation;
            }
        }
    }

    pub fn cleanup_old_data(&mut self, current_time: u64) {
        let cutoff_time = current_time.saturating_sub(self.reconciliation_window);

        // Cleanup settlements
        self.settlements.retain(|&timestamp, _| timestamp > cutoff_time);

        // Cleanup position history
        for snapshots in self.position_history.values_mut() {
            snapshots.retain(|snapshot| snapshot.timestamp > cutoff_time);
        }

        // Remove empty position histories
        self.position_history.retain(|_, snapshots| !snapshots.is_empty());
    }

    pub fn get_position_snapshots(
        &self,
        position: Address,
        start_time: u64,
        end_time: u64,
    ) -> Result<Vec<PositionSnapshot>> {
        let snapshots = self.position_history.get(&position)
            .ok_or(PanopticError::PositionNotFound)?;

        Ok(snapshots.iter()
            .filter(|s| s.timestamp >= start_time && s.timestamp <= end_time)
            .cloned()
            .collect())
    }

    pub fn get_settlement_history(
        &self,
        position: Address,
        start_time: u64,
        end_time: u64,
    ) -> Vec<Settlement> {
        let mut settlements = Vec::new();
        
        for settlement_list in self.settlements.range(start_time..=end_time) {
            for settlement in settlement_list.1 {
                if settlement.pool == position {
                    settlements.push(settlement.clone());
                }
            }
        }

        settlements
    }

    pub fn calculate_pnl_metrics(
        &self,
        position: Address,
        start_time: u64,
        end_time: u64,
    ) -> Result<PnLMetrics> {
        let snapshots = self.get_position_snapshots(position, start_time, end_time)?;
        if snapshots.is_empty() {
            return Err(PanopticError::NoDataAvailable);
        }

        let first = snapshots.first().unwrap();
        let last = snapshots.last().unwrap();

        let realized_pnl_change = last.realized_pnl - first.realized_pnl;
        let unrealized_pnl_change = last.unrealized_pnl - first.unrealized_pnl;
        let total_funding = last.funding_payments - first.funding_payments;

        let settlements = self.get_settlement_history(position, start_time, end_time);
        let settlement_pnl: i64 = settlements.iter()
            .map(|s| s.amount)
            .sum();

        Ok(PnLMetrics {
            realized_pnl: realized_pnl_change,
            unrealized_pnl: unrealized_pnl_change,
            settlement_pnl,
            funding_pnl: total_funding,
            total_pnl: realized_pnl_change + unrealized_pnl_change + total_funding,
            position_value_change: if last.size > U256::zero() {
                last.unrealized_pnl as f64 / last.size.as_u128() as f64
            } else {
                0.0
            },
        })
    }
}

#[derive(Debug)]
pub struct PnLMetrics {
    pub realized_pnl: i64,
    pub unrealized_pnl: i64,
    pub settlement_pnl: i64,
    pub funding_pnl: i64,
    pub total_pnl: i64,
    pub position_value_change: f64,
}

impl Clone for PositionSnapshot {
    fn clone(&self) -> Self {
        Self {
            timestamp: self.timestamp,
            size: self.size,
            collateral: self.collateral,
            unrealized_pnl: self.unrealized_pnl,
            realized_pnl: self.realized_pnl,
            funding_payments: self.funding_payments,
        }
    }
}
