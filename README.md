# Panoptic Protocol - Rust Implementation

A high-performance, secure implementation of the Panoptic options protocol in Rust.

## Overview

Panoptic is a decentralized options protocol that enables users to create, trade, and exercise options on any ERC20 token pair. This implementation focuses on performance, security, and complete functionality.

## Features

- Complete options trading functionality (create, close, exercise positions)
- Advanced pricing engine using Black-Scholes model
- Robust collateral management system
- Real-time position monitoring and liquidation checks
- High-performance swap calculations
- Comprehensive Greeks calculations (delta, gamma, theta, vega, rho)
- Event emission system for tracking protocol activity

## Architecture

The protocol is organized into several core modules:

- `types.rs`: Core data structures and types
- `pool.rs`: Pool management and swap functionality
- `factory.rs`: Pool creation and management
- `position.rs`: Position creation and management
- `pricing.rs`: Option pricing and Greeks calculations
- `collateral.rs`: Collateral management and liquidation checks
- `errors.rs`: Error handling
- `events.rs`: Event emission system
- `math.rs`: Core mathematical calculations
- `utils.rs`: Utility functions

## Getting Started

1. Install Rust and Cargo:
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

2. Clone the repository:
```bash
git clone https://github.com/yourusername/panoptic-rs.git
cd panoptic-rs
```

3. Build the project:
```bash
cargo build --release
```

4. Run tests:
```bash
cargo test
```

## Usage

```rust
use panoptic_rs::{
    factory::PanopticFactory,
    types::{OptionType, Strike},
    pricing::PricingEngine,
    collateral::CollateralManager,
};

// Create factory
let factory = PanopticFactory::new(owner_address);

// Create pool
let pool = factory.create_pool(token0, token1, fee, sqrt_price_x96)?;

// Create position
let token_id = pool.create_position(
    owner,
    OptionType::Call,
    Strike(strike_price),
    amount,
    collateral,
    expiry,
)?;

// Exercise position
let settlement = pool.exercise_position(owner, token_id, amount)?;
```

## Security Features

- Comprehensive input validation
- Overflow protection using checked arithmetic
- Robust error handling
- Access control checks
- Liquidation protection
- Collateral validation

## Performance Optimizations

- Efficient data structures
- Minimal copying of data
- Optimized mathematical calculations
- Memory-efficient event handling
- Fast lookup using hashmaps

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
