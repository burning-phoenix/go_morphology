# GO_MSAE Unit Test Requirements

> **Purpose**: Document which `src/` implementations require unit tests for edge cases and validity testing
>
> **Based on**: Analysis of source code and notebook usage patterns
>
> **Priority Levels**:
> - 🔴 **Critical**: Bugs here cause silent data corruption or wrong scientific conclusions
> - 🟠 **High**: Edge cases that could fail silently in production
> - 🟡 **Medium**: Important for correctness but failures would be visible
> - 🟢 **Low**: Nice to have, defensive testing

---

## Table of Contents

1. [src/models/](#1-srcmodels)
2. [src/data/](#2-srcdata)
3. [src/analysis/](#3-srcanalysis)
4. [src/training/](#4-srctraining)
5. [src/utils/](#5-srcutils)
6. [src/visualization/](#6-srcvisualization)
7. [Summary: Recommended Test Suite](#7-summary-recommended-test-suite)

---

## 1. src/models/

### 1.1 `msae.py` - MatryoshkaSAE

#### 🔴 Critical: TopK Implementation

**Location**: `TopK` class (lines 23-54)

**Why Critical**: The straight-through estimator is the core of the sparsity mechanism. Incorrect gradient flow breaks training.

**Edge Cases to Test**:
```python
def test_topk_k_equals_hidden_dim():
    """k == hidden_dim should return all values unchanged."""
    x = torch.randn(10, 100)
    result = topk_activation(x, k=100)
    assert torch.allclose(result, x)

def test_topk_k_equals_1():
    """k=1 should keep only the maximum value."""
    x = torch.tensor([[1.0, 5.0, 2.0]])
    result = topk_activation(x, k=1)
    assert result[0, 1] == 5.0
    assert result[0, 0] == 0.0
    assert result[0, 2] == 0.0

def test_topk_gradient_straight_through():
    """Gradient should flow through selected features."""
    x = torch.randn(10, 100, requires_grad=True)
    result = topk_activation(x, k=10)
    loss = result.sum()
    loss.backward()
    # Gradient should be 1 for top-k, 0 for others
    assert (x.grad != 0).sum() == 10 * 10  # batch * k

def test_topk_negative_values():
    """TopK should work correctly with negative values."""
    x = torch.tensor([[-5.0, -1.0, -3.0]])  # All negative
    result = topk_activation(x, k=1)
    assert result[0, 1] == -1.0  # Largest (least negative)
```

**Validity Tests**:
```python
def test_topk_sparsity():
    """Output should have exactly k non-zeros per sample."""
    x = torch.randn(100, 4096)
    for k in [16, 32, 64, 128]:
        result = topk_activation(x, k=k)
        non_zeros = (result != 0).sum(dim=1)
        assert (non_zeros == k).all()
```

---

#### 🔴 Critical: forward_hierarchical Method

**Location**: `MatryoshkaSAE.forward_hierarchical()` (lines 199-236)

**Why Critical**: Returns reconstructions at all k levels - wrong computation produces invalid hierarchy metrics.

**Edge Cases to Test**:
```python
def test_forward_hierarchical_k_levels_ordering():
    """Reconstructions should improve as k increases."""
    model = create_msae(input_dim=256, k_levels=[16, 32, 64, 128])
    x = torch.randn(100, 256)
    reconstructions = model.forward_hierarchical(x)
    
    mses = {k: F.mse_loss(reconstructions[k], x).item() for k in [16, 32, 64, 128]}
    # MSE should decrease (reconstruction improves) as k increases
    assert mses[16] >= mses[32] >= mses[64] >= mses[128]

def test_forward_hierarchical_matches_forward():
    """forward_hierarchical(k) should match forward(k)."""
    model = create_msae()
    x = torch.randn(50, 256)
    
    hierarchical = model.forward_hierarchical(x)
    for k in [16, 32, 64, 128]:
        x_hat_hier = hierarchical[k]
        x_hat_single, _, _ = model.forward(x, k=k)
        assert torch.allclose(x_hat_hier, x_hat_single, atol=1e-5)
```

---

#### 🟠 High: Auxiliary Loss Computation

**Location**: `MatryoshkaSAE._compute_aux_loss()` (lines 303-350)

**Why High**: Used to revive dead latents - if broken, dead features accumulate silently.

**Edge Cases to Test**:
```python
def test_aux_loss_no_dead_latents():
    """Aux loss should be 0 when no dead latents exist."""
    model = create_msae()
    x = torch.randn(5000, 256)
    
    # Warm up activation counts
    for _ in range(10):
        model.compute_loss(x, update_stats=True)
    
    # If all latents active, aux_loss should be small/zero
    losses = model.compute_loss(x)
    # (May not be exactly 0 due to threshold)

def test_aux_loss_with_dead_latents():
    """Aux loss should be non-zero when dead latents exist."""
    model = create_msae()
    # Force some latents to be dead
    model.activation_counts[:100] = 0
    model.total_samples = torch.tensor(100000.0)
    
    x = torch.randn(100, 256)
    losses = model.compute_loss(x, update_stats=False)
    assert losses['aux_loss'] > 0
```

---

#### 🟠 High: Decoder Normalization

**Location**: `MatryoshkaSAE.normalize_decoder()` (lines 135-138)

**Edge Cases to Test**:
```python
def test_decoder_columns_unit_norm():
    """Each decoder column should have unit norm after normalization."""
    model = create_msae()
    model.normalize_decoder()
    
    norms = model.W_dec.norm(dim=0)  # Norm of each column
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

def test_decoder_normalization_after_update():
    """normalize_decoder should be called after optimizer step."""
    model = create_msae()
    optimizer = torch.optim.Adam(model.parameters())
    x = torch.randn(100, 256)
    
    losses = model.compute_loss(x)
    losses['loss'].backward()
    optimizer.step()
    model.normalize_decoder()
    
    norms = model.W_dec.norm(dim=0)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
```

---

### 1.2 `baseline_sae.py` - BaselineSAE

#### 🟡 Medium: Consistency with MatryoshkaSAE

**Validity Tests**:
```python
def test_baseline_sae_uses_same_topk():
    """Baseline should use same TopK mechanism as MSAE."""
    from src.models.msae import topk_activation as msae_topk
    from src.models.baseline_sae import BaselineSAE
    
    # Verify it imports/uses the same function
    baseline = BaselineSAE(input_dim=256, hidden_dim=4096, k=64)
    # Internal implementation should match

def test_baseline_vs_msae_at_same_k():
    """Identically initialized models should produce same output at k=64."""
    # Initialize both with same weights
    msae = create_msae()
    baseline = BaselineSAE(input_dim=256, hidden_dim=4096, k=64)
    
    # Copy weights
    baseline.W_enc.data = msae.W_enc.data.clone()
    baseline.W_dec.data = msae.W_dec.data.clone()
    baseline.b_pre.data = msae.b_pre.data.clone()
    
    x = torch.randn(100, 256)
    x_hat_msae, _, _ = msae.forward(x, k=64)
    x_hat_baseline, _, _ = baseline.forward(x)
    
    assert torch.allclose(x_hat_msae, x_hat_baseline, atol=1e-5)
```

---

### 1.3 `leela_zero.py` - LeelaZero

#### 🟠 High: forward_with_activations

**Location**: `LeelaZero.forward_with_activations()` (lines 135-175)

**Why High**: Used by all activation extraction - wrong block indices means wrong layer analysis.

**Edge Cases to Test**:
```python
def test_forward_with_activations_block_indices():
    """Should extract correct block activations."""
    model = create_leela_zero(weights_path)
    x = torch.randn(1, 18, 19, 19)
    
    outputs, acts = model.forward_with_activations(x, [0, 5, 20, 35])
    
    assert 0 in acts
    assert 5 in acts
    assert 20 in acts
    assert 35 in acts
    
    # Each should have shape (1, 256, 19, 19)
    for block_idx in [0, 5, 20, 35]:
        assert acts[block_idx].shape == (1, 256, 19, 19)

def test_forward_with_activations_boundary_blocks():
    """Test first and last block indices."""
    model = create_leela_zero(weights_path)
    x = torch.randn(1, 18, 19, 19)
    
    # First block
    _, acts = model.forward_with_activations(x, [0])
    assert 0 in acts
    
    # Last block (39 for 40-block model)
    _, acts = model.forward_with_activations(x, [39])
    assert 39 in acts

def test_forward_with_activations_invalid_block():
    """Should handle invalid block indices gracefully."""
    model = create_leela_zero(weights_path)
    x = torch.randn(1, 18, 19, 19)
    
    # Block 40 doesn't exist
    with pytest.raises((IndexError, KeyError)):
        model.forward_with_activations(x, [40])
```

---

#### 🔴 Critical: Weight Loading

**Location**: `load_leela_weights()` (lines 235-328)

**Why Critical**: Incorrect weight loading produces a non-functional model that still runs.

**Validity Tests**:
```python
def test_load_leela_weights_smoketest():
    """Loaded model should produce non-trivial outputs."""
    model = create_leela_zero(weights_path)
    model.eval()
    
    # Random input
    x = torch.randn(1, 18, 19, 19)
    policy, value = model(x)
    
    # Policy should be non-uniform (trained model behavior)
    entropy = -(F.softmax(policy, dim=-1) * F.log_softmax(policy, dim=-1)).sum()
    assert entropy < 5.0  # Far from uniform (max entropy ~5.89)
    
    # Value should be in range
    assert -1 <= value.item() <= 1

def test_load_leela_weights_architecture_match():
    """Loaded weights should match expected architecture."""
    model = create_leela_zero(weights_path)
    
    assert len(model.residual_tower) == 40
    assert model.residual_channels == 256
    assert model.in_channels == 18
```

---

## 2. src/data/

### 2.1 `position_encoder.py`

#### 🔴 Critical: Plane Encoding

**Location**: `PositionEncoder.encode()` (lines 126-165)

**Why Critical**: Wrong encoding produces activations that don't match training data format.

**Edge Cases to Test**:
```python
def test_encode_empty_board():
    """Empty board should have all zeros in stone planes."""
    state = BoardState()
    encoder = PositionEncoder()
    planes = encoder.encode(state)
    
    # Planes 0-15 should be all zeros
    assert (planes[:16] == 0).all()
    # One of planes 16/17 should be all 1s
    assert (planes[16].sum() + planes[17].sum()) == 19 * 19

def test_encode_color_to_play():
    """Plane 16/17 should correctly indicate color to play."""
    state = BoardState()
    encoder = PositionEncoder()
    
    # Black to play initially
    planes_black = encoder.encode(state)
    assert planes_black[16].sum() == 19 * 19
    assert planes_black[17].sum() == 0
    
    # After black plays, white to play
    state.play(3, 3, 'b')
    planes_white = encoder.encode(state)
    assert planes_white[16].sum() == 0
    assert planes_white[17].sum() == 19 * 19

def test_encode_stone_placement():
    """Stone positions should appear in correct planes."""
    state = BoardState()
    state.play(3, 3, 'b')  # Black at (3, 3)
    state.play(15, 15, 'w')  # White at (15, 15)
    
    encoder = PositionEncoder()
    planes = encoder.encode(state)
    
    # Now white to play: plane 0 = current (white), plane 8 = opponent (black)
    # Wait - after white plays, black is to play
    # So plane 0 = black (current), plane 8 = white (opponent)
    assert planes[0, 3, 3] == 1  # Black stone in current player plane
    assert planes[8, 15, 15] == 1  # White stone in opponent plane
```

**Validity Tests**:
```python
def test_encode_matches_leela_format():
    """Encoding should match Leela Zero expected format."""
    # From leela_zero_pytorch reference:
    # - Planes 0-7: current player history
    # - Planes 8-15: opponent history
    # - Plane 16: all 1s if black to play
    # - Plane 17: all 1s if white to play
    
    state = BoardState()
    encoder = PositionEncoder()
    planes = encoder.encode(state)
    
    assert planes.shape == (18, 19, 19)
    assert planes.dtype == torch.float32
```

---

#### 🟠 High: History Tracking

**Location**: `BoardState.play()` (lines 63-82)

**Edge Cases to Test**:
```python
def test_history_length_limit():
    """History should be capped at 8 positions."""
    state = BoardState(history_length=8)
    
    # Play more than 8 moves
    for i in range(15):
        state.play(i // 19, i % 19, 'b' if i % 2 == 0 else 'w')
    
    assert len(state.history) <= 8

def test_history_order():
    """Most recent positions should be first in history."""
    state = BoardState()
    
    state.play(0, 0, 'b')  # Move 1
    state.play(1, 1, 'w')  # Move 2
    state.play(2, 2, 'b')  # Move 3
    
    # history[0] should be state BEFORE move 3 (after move 2)
    assert state.history[0][1, 1] == 2  # White stone from move 2
```

---

### 2.2 `sgf_parser.py`

#### 🟠 High: Move Parsing

**Location**: `SGFParser._parse_move()` and related functions

**Edge Cases to Test**:
```python
def test_parse_pass_move():
    """Pass moves should be handled without error."""
    sgf = "(;GM[1]SZ[19];B[dd];W[tt])"  # tt = pass
    parser = SGFParser()
    positions = parser.parse(sgf)
    # Should not raise

def test_parse_resign():
    """Games ending in resign should parse correctly."""
    sgf = "(;GM[1]SZ[19]RE[B+R];B[dd];W[pp])"
    parser = SGFParser()
    positions = parser.parse(sgf)
    assert len(positions) >= 2

def test_parse_handcap_game():
    """Handicap games with AB/AW properties."""
    sgf = "(;GM[1]SZ[19]HA[2]AB[dd][pd];W[pp])"
    parser = SGFParser()
    positions = parser.parse(sgf)
    # Initial position should have 2 black stones
```

---

### 2.3 `activation_extractor.py`

#### 🟡 Medium: Chunked Extraction

**Edge Cases to Test**:
```python
def test_extractor_chunk_boundaries():
    """Chunks should contain correct positions at boundaries."""
    extractor = ActivationExtractor(model, chunk_size=100)
    
    # Extract from 250 positions
    positions = torch.randn(250, 18, 19, 19)
    chunks = list(extractor.extract_all(positions))
    
    assert len(chunks) == 3  # 100 + 100 + 50
    assert chunks[0].shape[0] == 100 * 361
    assert chunks[1].shape[0] == 100 * 361
    assert chunks[2].shape[0] == 50 * 361

def test_extractor_single_position():
    """Should work with just 1 position."""
    extractor = ActivationExtractor(model, chunk_size=100)
    positions = torch.randn(1, 18, 19, 19)
    chunks = list(extractor.extract_all(positions))
    
    assert len(chunks) == 1
    assert chunks[0].shape[0] == 361
```

---

## 3. src/analysis/

### 3.1 `concepts.py` - ConceptLabeler

#### 🔴 Critical: is_cutting_point (KNOWN BUG)

**Location**: `is_cutting_point()` (lines 299-331)

**Why Critical**: Currently has 100% positive rate bug - invalidates causal analysis.

**Edge Cases to Test**:
```python
def test_is_cutting_point_empty_board():
    """Empty board should have no cutting points."""
    labeler = ConceptLabeler()
    board = np.zeros((19, 19), dtype=np.int32)
    
    result = labeler.is_cutting_point(board)
    assert result.sum() == 0

def test_is_cutting_point_single_group():
    """Single group cannot have cutting points."""
    labeler = ConceptLabeler()
    board = np.zeros((19, 19), dtype=np.int32)
    # Connected L-shape
    board[0, 0] = 1
    board[0, 1] = 1
    board[1, 1] = 1
    
    result = labeler.is_cutting_point(board)
    assert result.sum() == 0  # No cutting points in single group

def test_is_cutting_point_actual_cut():
    """Known cutting point position should be detected."""
    labeler = ConceptLabeler()
    board = np.zeros((19, 19), dtype=np.int32)
    # Two black groups that share a liberty
    #   B . B
    #   B X B
    # X is cutting point (shared liberty of both groups)
    board[0, 0] = 1  # Group 1
    board[1, 0] = 1
    board[0, 2] = 1  # Group 2
    board[1, 2] = 1
    
    result = labeler.is_cutting_point(board)
    # (1, 1) should be the cutting point
    assert result[1, 1] == True

def test_is_cutting_point_not_100_percent():
    """Regression test: should not be 100% positive rate."""
    labeler = ConceptLabeler()
    # Random semi-empty board
    board = np.zeros((19, 19), dtype=np.int32)
    board[5:10, 5:10] = np.random.randint(0, 3, (5, 5))
    
    result = labeler.is_cutting_point(board)
    positive_rate = result.sum() / (19 * 19)
    assert positive_rate < 0.5  # Should be much less than 100%
```

---

#### 🟠 High: is_eye_shape (False Eye Detection)

**Location**: `is_eye_shape()` (lines 225-287)

**Why Important**: False eye detection is subtle and affects life/death concept accuracy.

**Edge Cases to Test**:
```python
def test_is_eye_shape_corner_eye():
    """Corner eye surrounded by own stones should be detected."""
    labeler = ConceptLabeler()
    board = np.zeros((19, 19), dtype=np.int32)
    # Corner eye at (0, 0)
    board[0, 1] = 1  # Black
    board[1, 0] = 1
    board[1, 1] = 1  # Only 1 diagonal, which is off-board
    
    eyes_black = labeler.is_eye_shape(board, StoneColor.BLACK)
    assert eyes_black[0, 0] == True

def test_is_eye_shape_false_eye():
    """False eye (enemy on diagonal) should NOT be detected."""
    labeler = ConceptLabeler()
    board = np.zeros((19, 19), dtype=np.int32)
    # False eye at (5, 5)
    board[4, 5] = 1  # Black
    board[6, 5] = 1
    board[5, 4] = 1
    board[5, 6] = 1
    board[4, 4] = 2  # White on diagonal - makes it false eye
    board[6, 6] = 2  # Two enemy diagonals
    
    eyes_black = labeler.is_eye_shape(board, StoneColor.BLACK)
    assert eyes_black[5, 5] == False  # False eye, not real

def test_is_eye_shape_edge_true_eye():
    """Edge eye with no enemy diagonals should be detected."""
    labeler = ConceptLabeler()
    board = np.zeros((19, 19), dtype=np.int32)
    # Edge eye at (0, 5)
    board[0, 4] = 1
    board[0, 6] = 1
    board[1, 5] = 1
    
    eyes_black = labeler.is_eye_shape(board, StoneColor.BLACK)
    assert eyes_black[0, 5] == True
```

---

#### 🟠 High: territory_estimate

**Location**: `estimate_territory()` (lines 358-417)

**Edge Cases to Test**:
```python
def test_territory_empty_board():
    """Empty board should have no territory for either side."""
    labeler = ConceptLabeler()
    board = np.zeros((19, 19), dtype=np.int32)
    
    territory = labeler.estimate_territory(board)
    # All empty - neither player has enclosed territory
    assert territory['black_territory'].sum() == 0
    assert territory['white_territory'].sum() == 0

def test_territory_fully_enclosed():
    """Fully enclosed region should be assigned."""
    labeler = ConceptLabeler()
    board = np.zeros((19, 19), dtype=np.int32)
    # Black surrounds corner:
    # B B B .
    # B . B .
    # B B B .
    board[0, 0:3] = 1
    board[1, 0] = 1
    board[1, 2] = 1
    board[2, 0:3] = 1
    
    territory = labeler.estimate_territory(board)
    assert territory['black_territory'][1, 1] == 1.0

def test_territory_dame():
    """Points touching both colors should be neutral (dame)."""
    labeler = ConceptLabeler()
    board = np.zeros((19, 19), dtype=np.int32)
    # B . W
    board[5, 5] = 1  # Black
    board[5, 7] = 2  # White
    # (5, 6) is dame
    
    territory = labeler.estimate_territory(board)
    assert territory['black_territory'][5, 6] == 0
    assert territory['white_territory'][5, 6] == 0
```

---

#### 🟡 Medium: Group Finding

**Location**: `_find_group()` (lines 107-142)

**Edge Cases to Test**:
```python
def test_find_group_large_group():
    """Should handle large connected groups efficiently."""
    labeler = ConceptLabeler()
    board = np.zeros((19, 19), dtype=np.int32)
    board[:, :] = 1  # Entire board is one black group
    
    group = labeler._find_group(board, 0, 0)
    assert len(group.stones) == 19 * 19
    assert group.liberty_count == 0  # No liberties

def test_find_group_diagonal_not_connected():
    """Diagonal stones should NOT be in same group."""
    labeler = ConceptLabeler()
    board = np.zeros((19, 19), dtype=np.int32)
    board[0, 0] = 1
    board[1, 1] = 1  # Diagonal, not connected
    
    group = labeler._find_group(board, 0, 0)
    assert len(group.stones) == 1  # Only (0, 0)
```

---

### 3.2 `probes.py` - ProbeEvaluator

#### 🔴 Critical: Position-Level Split

**Location**: `_split_data_by_position()` (lines 253-304)

**Why Critical**: Wrong splitting causes train/test contamination and inflated AUCs.

**Validity Tests**:
```python
def test_position_level_split_no_leakage():
    """No position should appear in both train and test."""
    evaluator = ProbeEvaluator()
    
    n_positions = 1000
    n_points = 361
    features = np.random.randn(n_positions * n_points, 256)
    labels = np.random.randint(0, 2, n_positions * n_points)
    position_ids = np.repeat(np.arange(n_positions), n_points)
    
    X_train, X_val, X_test, y_train, y_val, y_test = evaluator._split_data(
        features, labels, position_ids
    )
    
    # Reconstruct position IDs for each split
    train_pos_ids = np.unique(position_ids[:len(y_train)])
    test_pos_ids = np.unique(position_ids[:len(y_test)])
    
    # No overlap
    assert len(set(train_pos_ids) & set(test_pos_ids)) == 0

def test_position_level_split_proportions():
    """Splits should approximately match requested proportions."""
    evaluator = ProbeEvaluator(test_size=0.15, val_size=0.15)
    
    n_positions = 1000
    n_points = 361
    features = np.random.randn(n_positions * n_points, 256)
    labels = np.random.randint(0, 2, n_positions * n_points)
    position_ids = np.repeat(np.arange(n_positions), n_points)
    
    X_train, X_val, X_test, y_train, y_val, y_test = evaluator._split_data(
        features, labels, position_ids
    )
    
    total = len(y_train) + len(y_val) + len(y_test)
    test_ratio = len(y_test) / total
    assert 0.10 < test_ratio < 0.20  # Approximately 15%
```

---

#### 🟠 High: Bootstrap Confidence Intervals

**Location**: `evaluate_probe()` (lines 127-185)

**Edge Cases to Test**:
```python
def test_evaluate_probe_single_class():
    """Should return 0.5 when only one class present."""
    probe = LinearProbe(n_features=100)
    features = np.random.randn(100, 100)
    labels = np.ones(100)  # All positive
    
    auc, ci_low, ci_high = evaluate_probe(probe, features, labels)
    assert auc == 0.5
    assert ci_low == 0.5
    assert ci_high == 0.5

def test_evaluate_probe_few_samples():
    """Should handle very few samples."""
    probe = LinearProbe(n_features=10)
    features = np.random.randn(20, 10)
    labels = np.array([0] * 10 + [1] * 10)
    
    auc, ci_low, ci_high = evaluate_probe(probe, features, labels)
    # Should not crash, CI may be wide
    assert 0 <= auc <= 1
```

---

### 3.3 `hierarchy.py`

#### 🟠 High: Nestedness Computation

**Location**: `compute_nestedness()` (lines 37-104)

**Validity Tests**:
```python
def test_nestedness_perfect_matryoshka():
    """Perfect Matryoshka structure should have nestedness 1.0."""
    # Create mock model where top-k features are always subset
    model = MockMatryoshkaSAE()
    # Force features to have strict ranking
    activations = torch.randn(1000, 256)
    
    nestedness = compute_nestedness(model, activations, [16, 32, 64, 128])
    
    # Should be 1.0 if k=16 is always subset of k=32
    assert nestedness['k16_in_k32'] >= 0.99

def test_nestedness_random_features():
    """Random features should have partial overlap."""
    model = create_msae()
    model.eval()
    
    # Random activations
    activations = torch.randn(1000, 256)
    
    nestedness = compute_nestedness(model, activations, [16, 32, 64, 128])
    
    # k=16 should have ~50% overlap with k=32 (16/32)
    assert 0.3 < nestedness['k16_in_k32'] < 0.7
```

---

#### 🟡 Medium: Reconstruction R² Computation

**Location**: `compute_reconstruction_r2()` (lines 107-166)

**Edge Cases to Test**:
```python
def test_reconstruction_r2_perfect():
    """R² should be 1.0 for perfect reconstruction."""
    model = create_msae()
    
    # Use model's own output as "activations" (perfect decode)
    with torch.no_grad():
        x = torch.randn(100, 256)
        x_hat, _, _ = model.forward(x, k=128)
    
    # If model perfectly reconstructed, R² = 1
    # (won't actually be 1 due to sparsity loss)

def test_reconstruction_r2_range():
    """R² should be between 0 and 1 for reasonable reconstructions."""
    model = create_msae()
    activations = torch.randn(1000, 256)
    
    r2_scores = compute_reconstruction_r2(model, activations, [16, 32, 64, 128])
    
    for k, r2 in r2_scores.items():
        assert 0 <= r2 <= 1
```

---

### 3.4 `causal.py`

#### 🔴 Critical: Cached Forward Consistency

**Location**: `_cache_base_forward()` and `ablate_feature_cached()` (lines 217-424)

**Why Critical**: Inconsistency between cached and fresh computation invalidates causal results.

**Validity Tests**:
```python
def test_cached_vs_fresh_ablation():
    """Cached ablation should match fresh ablation."""
    analyzer = CausalAnalyzer(leela_model, sae_model, layer_idx=5)
    positions = torch.randn(10, 18, 19, 19)
    
    # Fresh ablation
    fresh_result = analyzer.ablate_feature(
        feature_idx=42, positions=positions, k=64
    )
    
    # Cached ablation
    analyzer._cache_base_forward(positions, k=64)
    cached_result = analyzer.ablate_feature_cached(
        feature_idx=42, layer_name='block5'
    )
    
    # Results should be very close
    assert abs(fresh_result.policy_kl_divergence - cached_result.policy_kl_divergence) < 0.01
```

---

#### 🟠 High: KL Divergence Computation

**Location**: `compute_policy_kl_divergence()` (lines 110-135)

**Edge Cases to Test**:
```python
def test_kl_div_identical_policies():
    """KL divergence of identical policies should be 0."""
    policy = torch.randn(10, 362)
    kl = compute_policy_kl_divergence(policy, policy)
    assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-5)

def test_kl_div_numerical_stability():
    """Should not produce NaN or inf for extreme probabilities."""
    # Policy with near-zero probabilities
    policy1 = torch.zeros(10, 362)
    policy1[:, 0] = 100  # Very high logit
    
    policy2 = torch.zeros(10, 362)
    policy2[:, 1] = 100  # Different high logit
    
    kl = compute_policy_kl_divergence(policy1, policy2)
    assert not torch.isnan(kl).any()
    assert not torch.isinf(kl).any()
```

---

### 3.5 `controls.py`

#### 🟠 High: Feature Permutation

**Location**: `permute_features()` (lines in controls.py)

**Why Important**: Must permute correctly to test geometric structure.

**Edge Cases to Test**:
```python
def test_permute_features_changes_features():
    """Permuted features should differ from original."""
    features = np.random.randn(1000, 4096)
    permuted = permute_features(features)
    
    # Should not be identical
    assert not np.allclose(features, permuted)
    
    # But should have same distribution
    assert np.allclose(features.mean(axis=0).mean(), permuted.mean(axis=0).mean(), atol=0.1)

def test_permute_features_per_sample():
    """Each sample should be permuted independently."""
    features = np.random.randn(100, 4096)
    permuted = permute_features(features)
    
    # Different samples should have different permutations
    # (statistically very unlikely to be same)
    perm_patterns = []
    for i in range(10):
        pattern = np.argsort(permuted[i, :10])
        perm_patterns.append(tuple(pattern))
    
    assert len(set(perm_patterns)) > 1  # Not all same
```

---

## 4. src/training/

### 4.1 `train_msae.py`

#### 🟡 Medium: create_activation_dataloader

**Location**: `create_activation_dataloader()` (lines 599-712)

**Edge Cases to Test**:
```python
def test_dataloader_normalization():
    """Loaded activations should be normalized."""
    train_loader, val_loader, stats = create_activation_dataloader(
        activations_dir='outputs/data/activations',
        block_idx=5,
        normalize=True
    )
    
    batch = next(iter(train_loader))
    # Mean should be ~0, std should be ~1
    assert abs(batch.mean()) < 0.5
    assert 0.5 < batch.std() < 1.5

def test_dataloader_chunk_ordering():
    """Chunks should be loaded in deterministic order."""
    loader1, _, _ = create_activation_dataloader(
        activations_dir='path', block_idx=5
    )
    loader2, _, _ = create_activation_dataloader(
        activations_dir='path', block_idx=5
    )
    
    batch1 = next(iter(loader1))
    batch2 = next(iter(loader2))
    # Should be identical (deterministic loading)
    assert torch.allclose(batch1, batch2)
```

---

## 5. src/utils/

### 5.1 `streaming_stats.py`

#### 🟠 High: Welford's Algorithm Accuracy

**Location**: `StreamingStats.update_batch()` (lines 44-79)

**Validity Tests**:
```python
def test_streaming_stats_matches_numpy():
    """Streaming stats should match numpy computation."""
    data = np.random.randn(10000, 256)
    
    # Streaming computation
    stats = StreamingStats(n_features=256)
    for i in range(0, 10000, 1000):
        stats.update_batch(data[i:i+1000])
    
    # Numpy computation
    np_mean = data.mean(axis=0)
    np_std = data.std(axis=0)
    
    assert np.allclose(stats.mean, np_mean, atol=1e-3)
    assert np.allclose(stats.std, np_std, atol=1e-3)

def test_streaming_stats_single_sample():
    """Should handle single sample correctly."""
    stats = StreamingStats(n_features=10)
    sample = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=np.float32)
    stats.update_batch(sample)
    
    assert np.allclose(stats.mean, sample[0])
    # Variance undefined for n=1, should be 0
    assert np.allclose(stats.variance, np.zeros(10))
```

---

## 6. src/visualization/

### 6.1 `go_board.py`

#### 🟡 Medium: Board Coordinate Conversion

**Edge Cases to Test**:
```python
def test_board_coordinate_boundaries():
    """Coordinates at board edges should render correctly."""
    renderer = GoBoardRenderer()
    
    # Corner positions
    for pos in [(0, 0), (0, 18), (18, 0), (18, 18)]:
        fig = renderer.render_board()
        renderer.add_markers([pos], marker='x')
        # Should not raise

def test_board_heatmap_shape():
    """Heatmap must be 19x19."""
    renderer = GoBoardRenderer()
    renderer.render_board()
    
    # Wrong shape should raise
    with pytest.raises((ValueError, AssertionError)):
        renderer.add_heatmap_overlay(np.zeros((20, 20)))
```

---

## 7. Summary: Recommended Test Suite

### Phase 1: Critical (Must Have Before Publication)

| Test Module | Functions to Test | Priority |
|-------------|------------------|----------|
| `test_models/test_topk.py` | TopK gradient, sparsity, edge cases | 🔴 |
| `test_models/test_msae.py` | forward_hierarchical consistency | 🔴 |
| `test_data/test_encoder.py` | 18-plane encoding correctness | 🔴 |
| `test_analysis/test_concepts.py` | is_cutting_point bug, eye detection | 🔴 |
| `test_analysis/test_probes.py` | Position-level split no leakage | 🔴 |
| `test_analysis/test_causal.py` | Cached vs fresh ablation | 🔴 |
| `test_models/test_leela.py` | Weight loading produces valid outputs | 🔴 |

### Phase 2: High (Required for Reliability)

| Test Module | Functions to Test | Priority |
|-------------|------------------|----------|
| `test_models/test_msae.py` | Aux loss, decoder normalization | 🟠 |
| `test_data/test_encoder.py` | History tracking, color to play | 🟠 |
| `test_analysis/test_hierarchy.py` | Nestedness computation correctness | 🟠 |
| `test_analysis/test_probes.py` | Bootstrap CI, edge cases | 🟠 |
| `test_utils/test_streaming.py` | Welford's accuracy | 🟠 |

### Running Tests

```bash
# Install pytest
pip install pytest pytest-cov

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run only critical tests
pytest tests/ -m critical -v
```

### Suggested Directory Structure

```
tests/
├── conftest.py           # Fixtures (mock models, sample data)
├── test_models/
│   ├── test_topk.py
│   ├── test_msae.py
│   ├── test_baseline_sae.py
│   └── test_leela_zero.py
├── test_data/
│   ├── test_encoder.py
│   ├── test_sgf_parser.py
│   └── test_activation_extractor.py
├── test_analysis/
│   ├── test_concepts.py
│   ├── test_probes.py
│   ├── test_hierarchy.py
│   ├── test_causal.py
│   └── test_controls.py
├── test_training/
│   └── test_train_msae.py
├── test_utils/
│   └── test_streaming_stats.py
└── test_visualization/
    └── test_go_board.py
```

---

*Last updated: 2026-01-04*
