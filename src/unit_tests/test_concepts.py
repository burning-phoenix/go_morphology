"""
Unit tests for Go concept labeling functions.

These tests validate SCIENTIFIC CORRECTNESS of Go concept detection.
Each test uses known Go positions with ground truth labels.

Priority: 🔴 Critical - wrong concept labels invalidate probe results.

Reference: docs/go_concepts.md (Sensei's Library)
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.concepts import ConceptLabeler, StoneColor, Group, create_concept_dataset


# =============================================================================
# Test Fixtures - Known Board Positions
# =============================================================================

def create_empty_board():
    """Create empty 19x19 board."""
    return np.zeros((19, 19), dtype=np.int32)


def create_corner_eye_position():
    """
    Create position with known true eye in corner.
    
    Board (corner):
      0 1 2
    0 . B .
    1 B . B
    2 . B .
    
    (0,0) is NOT an eye (only 2 adjacent stones needed in corner, but diagonals matter)
    Actually, let's make a proper corner eye:
    
      0 1
    0 . B
    1 B B
    
    (0,0) is a true eye: surrounded by own stones on all adjacent points.
    """
    board = create_empty_board()
    board[0, 1] = StoneColor.BLACK
    board[1, 0] = StoneColor.BLACK
    board[1, 1] = StoneColor.BLACK  # Diagonal - ensures true eye
    return board


def create_false_eye_position():
    """
    Create position with a false eye.
    
    A false eye has enemy stones on too many diagonals.
    
    Center false eye:
        3 4 5 6
    3   . . . .
    4   . B . .
    5   B . B .
    6   . B . .
    
    (5,5) surrounded by black orthogonally, but:
    Add white on 2 diagonals to make it false.
    """
    board = create_empty_board()
    # Surround (5,5) with black
    board[4, 5] = StoneColor.BLACK
    board[6, 5] = StoneColor.BLACK
    board[5, 4] = StoneColor.BLACK
    board[5, 6] = StoneColor.BLACK
    # Add 2 enemy diagonals (makes false eye in center)
    board[4, 4] = StoneColor.WHITE
    board[6, 6] = StoneColor.WHITE
    return board


def create_atari_position():
    """
    Create position with a known atari (1 liberty).
    
    Black stone at (5,5) with white on 3 sides:
    
        4 5 6
    4   . W .
    5   W B .   <- Black at (5,5) has only (5,6) as liberty
    6   . W .
    """
    board = create_empty_board()
    board[5, 5] = StoneColor.BLACK
    board[4, 5] = StoneColor.WHITE
    board[6, 5] = StoneColor.WHITE
    board[5, 4] = StoneColor.WHITE
    return board


def create_cutting_point_position():
    """
    Create position with a clear cutting point.
    
    Two black groups that share a liberty at (4,4):
    
        3 4 5
    3   . B .
    4   B . B   <- (4,4) is cutting point (shared liberty of 4 groups)
    5   . B .
    
    Each stone at (3,4), (4,3), (4,5), (5,4) is a separate single-stone group.
    Point (4,4) is adjacent to all four, making it a shared liberty.
    """
    board = create_empty_board()
    # Four separate single-stone groups surrounding (4,4)
    board[3, 4] = StoneColor.BLACK  # North of (4,4)
    board[4, 3] = StoneColor.BLACK  # West of (4,4)
    board[4, 5] = StoneColor.BLACK  # East of (4,4)
    board[5, 4] = StoneColor.BLACK  # South of (4,4)
    return board


def create_enclosed_territory():
    """
    Create position with clear enclosed territory.
    
    Black encloses corner (0-2, 0-2):
    
        0 1 2 3
    0   . . B .
    1   . . B .
    2   B B B .
    3   . . . .
    
    Points (0,0), (0,1), (1,0), (1,1) are black territory.
    """
    board = create_empty_board()
    board[0, 2] = StoneColor.BLACK
    board[1, 2] = StoneColor.BLACK
    board[2, 0] = StoneColor.BLACK
    board[2, 1] = StoneColor.BLACK
    board[2, 2] = StoneColor.BLACK
    return board


# =============================================================================
# Geometric Concept Tests - Position-Based (Static)
# =============================================================================

class TestGeometricConcepts:
    """Tests for geometric concepts that depend only on board position."""

    def test_edge_mask_exact_count(self):
        """
        INVARIANT: Edge mask must have exactly 72 points.
        Edge = first/last row (19+19) + first/last column excluding corners (17+17) = 72
        """
        labeler = ConceptLabeler(board_size=19)
        edges = labeler.is_edge()
        
        assert edges.sum() == 72, f"Edge count should be 72, got {edges.sum()}"

    def test_edge_mask_boundary_values(self):
        """
        INVARIANT: All boundary points must be marked as edge.
        """
        labeler = ConceptLabeler(board_size=19)
        edges = labeler.is_edge()
        
        # Check all boundary points
        for i in range(19):
            assert edges[0, i], f"Top edge ({0},{i}) not marked"
            assert edges[18, i], f"Bottom edge ({18},{i}) not marked"
            assert edges[i, 0], f"Left edge ({i},{0}) not marked"
            assert edges[i, 18], f"Right edge ({i},{18}) not marked"
        
        # Check interior points are NOT edges
        for r in range(1, 18):
            for c in range(1, 18):
                assert not edges[r, c], f"Interior point ({r},{c}) wrongly marked as edge"

    def test_corner_mask_exact_count(self):
        """
        INVARIANT: Corner regions = 4 corners × 4×4 = 64 points.
        """
        labeler = ConceptLabeler(board_size=19)
        corners = labeler.is_corner_region()
        
        assert corners.sum() == 64, f"Corner count should be 64, got {corners.sum()}"

    def test_corner_mask_regions(self):
        """
        INVARIANT: 4×4 region in each corner must be marked.
        """
        labeler = ConceptLabeler(board_size=19)
        corners = labeler.is_corner_region()
        
        # Top-left 4×4
        assert corners[0:4, 0:4].all(), "Top-left corner region incomplete"
        # Top-right 4×4
        assert corners[0:4, 15:19].all(), "Top-right corner region incomplete"
        # Bottom-left 4×4
        assert corners[15:19, 0:4].all(), "Bottom-left corner region incomplete"
        # Bottom-right 4×4
        assert corners[15:19, 15:19].all(), "Bottom-right corner region incomplete"
        
        # Center should NOT be corner
        assert not corners[9, 9], "Center marked as corner"

    def test_star_points_19x19(self):
        """
        INVARIANT: 19×19 has exactly 9 star points at known positions.
        """
        labeler = ConceptLabeler(board_size=19)
        stars = labeler.is_star_point()
        
        expected = [(3, 3), (3, 9), (3, 15),
                    (9, 3), (9, 9), (9, 15),
                    (15, 3), (15, 9), (15, 15)]
        
        assert stars.sum() == 9, f"Star point count should be 9, got {stars.sum()}"
        
        for r, c in expected:
            assert stars[r, c], f"Star point at ({r},{c}) not marked"


# =============================================================================
# Stone Color Concepts - Mutual Exclusivity
# =============================================================================

class TestStoneColorConcepts:
    """Tests for stone presence concepts."""

    def test_stone_colors_mutually_exclusive(self):
        """
        INVARIANT: Each point is exactly one of {empty, black, white}.
        No overlapping labels allowed.
        """
        labeler = ConceptLabeler()
        # Random board
        np.random.seed(42)
        board = np.random.randint(0, 3, (19, 19), dtype=np.int32)
        
        colors = labeler.stone_color(board)
        
        total = colors['black'].astype(int) + colors['white'].astype(int) + colors['empty'].astype(int)
        
        assert (total == 1).all(), "Stone colors not mutually exclusive"

    def test_stone_colors_match_board(self):
        """
        INVARIANT: Labels must exactly match input board.
        """
        labeler = ConceptLabeler()
        board = create_empty_board()
        board[5, 5] = StoneColor.BLACK
        board[10, 10] = StoneColor.WHITE
        
        colors = labeler.stone_color(board)
        
        # Specific checks
        assert colors['black'][5, 5] == True
        assert colors['white'][10, 10] == True
        assert colors['empty'][0, 0] == True
        
        # Count checks
        assert colors['black'].sum() == 1
        assert colors['white'].sum() == 1
        assert colors['empty'].sum() == 361 - 2


# =============================================================================
# Group Finding - Connectivity
# =============================================================================

class TestGroupFinding:
    """Tests for connected stone group detection."""

    def test_single_stone_group(self):
        """
        INVARIANT: Single stone is a group of size 1 with correct liberties.
        """
        labeler = ConceptLabeler()
        board = create_empty_board()
        board[9, 9] = StoneColor.BLACK  # Center stone
        
        group = labeler._find_group(board, 9, 9)
        
        assert len(group.stones) == 1
        assert (9, 9) in group.stones
        assert group.liberty_count == 4  # Center has 4 liberties

    def test_corner_stone_liberties(self):
        """
        INVARIANT: Corner stone has exactly 2 liberties.
        """
        labeler = ConceptLabeler()
        board = create_empty_board()
        board[0, 0] = StoneColor.BLACK
        
        group = labeler._find_group(board, 0, 0)
        
        assert group.liberty_count == 2, f"Corner stone should have 2 liberties, got {group.liberty_count}"

    def test_edge_stone_liberties(self):
        """
        INVARIANT: Edge (non-corner) stone has exactly 3 liberties.
        """
        labeler = ConceptLabeler()
        board = create_empty_board()
        board[0, 9] = StoneColor.BLACK  # Top edge, middle
        
        group = labeler._find_group(board, 0, 9)
        
        assert group.liberty_count == 3, f"Edge stone should have 3 liberties, got {group.liberty_count}"

    def test_connected_group(self):
        """
        INVARIANT: Orthogonally connected stones form ONE group.
        """
        labeler = ConceptLabeler()
        board = create_empty_board()
        # L-shape: 3 connected stones
        board[5, 5] = StoneColor.BLACK
        board[5, 6] = StoneColor.BLACK
        board[6, 5] = StoneColor.BLACK
        
        group = labeler._find_group(board, 5, 5)
        
        assert len(group.stones) == 3
        assert (5, 5) in group.stones
        assert (5, 6) in group.stones
        assert (6, 5) in group.stones

    def test_diagonal_not_connected(self):
        """
        INVARIANT: Diagonal stones are NOT connected (different groups).
        """
        labeler = ConceptLabeler()
        board = create_empty_board()
        board[5, 5] = StoneColor.BLACK
        board[6, 6] = StoneColor.BLACK  # Diagonal only
        
        group = labeler._find_group(board, 5, 5)
        
        assert len(group.stones) == 1, "Diagonal stone should not be in same group"


# =============================================================================
# Atari Detection - Liberty Counting
# =============================================================================

class TestAtariConcept:
    """Tests for atari (1 liberty) detection."""

    def test_atari_ground_truth(self):
        """
        GROUND TRUTH TEST: Known atari position must be detected.
        """
        labeler = ConceptLabeler()
        board = create_atari_position()
        
        atari = labeler.is_atari(board)
        
        # The black stone at (5,5) should be in atari
        assert atari[5, 5], "Black stone at (5,5) should be in atari"
        
        # White stones are NOT in atari (they have more liberties)
        assert not atari[4, 5], "White stone not in atari"
        assert not atari[6, 5], "White stone not in atari"
        assert not atari[5, 4], "White stone not in atari"

    def test_not_atari_many_liberties(self):
        """
        INVARIANT: Stone with 4 liberties is NOT in atari.
        """
        labeler = ConceptLabeler()
        board = create_empty_board()
        board[9, 9] = StoneColor.BLACK  # Lone center stone
        
        atari = labeler.is_atari(board)
        
        assert not atari[9, 9], "Center stone with 4 liberties should not be atari"

    def test_empty_not_atari(self):
        """
        INVARIANT: Empty points can never be "in atari".
        """
        labeler = ConceptLabeler()
        board = create_atari_position()
        
        atari = labeler.is_atari(board)
        
        # Check all empty points
        empty_mask = (board == StoneColor.EMPTY)
        atari_on_empty = atari & empty_mask
        
        assert atari_on_empty.sum() == 0, "Empty points cannot be in atari"


# =============================================================================
# Eye Shape Detection - Critical for Life/Death
# =============================================================================

class TestEyeShapeConcept:
    """Tests for eye shape detection."""

    def test_true_corner_eye(self):
        """
        GROUND TRUTH: Corner eye with proper diagonal coverage is true eye.
        """
        labeler = ConceptLabeler()
        board = create_corner_eye_position()
        
        eyes = labeler.is_eye_shape(board, StoneColor.BLACK)
        
        assert eyes[0, 0], "Corner point (0,0) should be detected as eye"

    def test_false_eye_detected(self):
        """
        GROUND TRUTH: Eye with 2 enemy diagonals is FALSE eye.
        
        From Sensei's Library: A false eye has too many enemy diagonals.
        In center: 2+ enemy diagonals makes false eye.
        """
        labeler = ConceptLabeler()
        board = create_false_eye_position()
        
        eyes = labeler.is_eye_shape(board, StoneColor.BLACK)
        
        assert not eyes[5, 5], \
            "Point (5,5) with 2 enemy diagonals should be FALSE eye, not detected as eye"

    def test_not_eye_without_surrounding(self):
        """
        INVARIANT: Empty point without full surrounding is not eye.
        """
        labeler = ConceptLabeler()
        board = create_empty_board()
        board[5, 5] = StoneColor.BLACK
        board[5, 6] = StoneColor.BLACK
        # (5, 4) is empty but only has 2 adjacent black stones
        
        eyes = labeler.is_eye_shape(board, StoneColor.BLACK)
        
        assert not eyes[5, 4], "Point without full surrounding is not eye"


# =============================================================================
# Cutting Point Detection - KNOWN BUG AREA
# =============================================================================

class TestCuttingPointConcept:
    """
    Tests for cutting point detection.
    
    NOTE: There is a known bug where is_cutting_point has 100% positive rate.
    These tests verify correct behavior - failures indicate the bug.
    """

    def test_cutting_point_empty_board(self):
        """
        INVARIANT: Empty board has NO cutting points.
        """
        labeler = ConceptLabeler()
        board = create_empty_board()
        
        cutting = labeler.is_cutting_point(board)
        
        assert cutting.sum() == 0, \
            f"Empty board should have 0 cutting points, got {cutting.sum()}"

    def test_cutting_point_single_group(self):
        """
        INVARIANT: Single connected group has no cutting points.
        """
        labeler = ConceptLabeler()
        board = create_empty_board()
        # Solid 3×3 black group
        board[5:8, 5:8] = StoneColor.BLACK
        
        cutting = labeler.is_cutting_point(board)
        
        assert cutting.sum() == 0, \
            f"Single connected group should have 0 cutting points, got {cutting.sum()}"

    def test_cutting_point_ground_truth(self):
        """
        GROUND TRUTH: Known cutting point position.
        
        Four separate stones with shared liberty at (4,4):
        This should be a cutting point.
        """
        labeler = ConceptLabeler()
        board = create_cutting_point_position()
        
        cutting = labeler.is_cutting_point(board)
        
        # (4, 4) is shared liberty of groups at (3,3), (3,5), (5,3), (5,5)
        # Should be cutting point (separates some groups)
        assert cutting[4, 4], \
            "Point (4,4) shared by multiple groups should be a cutting point"

    def test_cutting_point_not_100_percent(self):
        """
        REGRESSION TEST: Cutting point rate should NOT be 100%.
        
        This test catches the known bug.
        """
        labeler = ConceptLabeler()
        # Semi-random board
        np.random.seed(42)
        board = create_empty_board()
        for _ in range(30):
            r, c = np.random.randint(0, 19, 2)
            board[r, c] = np.random.choice([StoneColor.BLACK, StoneColor.WHITE])
        
        cutting = labeler.is_cutting_point(board)
        
        positive_rate = cutting.sum() / 361
        
        assert positive_rate < 0.5, \
            f"BUG DETECTED: Cutting point rate {positive_rate:.2%} is way too high"


# =============================================================================
# Territory Estimation
# =============================================================================

class TestTerritoryEstimation:
    """Tests for territory estimation."""

    def test_territory_empty_board(self):
        """
        INVARIANT: Empty board = no territory for either side.
        """
        labeler = ConceptLabeler()
        board = create_empty_board()
        
        territory = labeler.estimate_territory(board)
        
        assert territory['black_territory'].sum() == 0
        assert territory['white_territory'].sum() == 0

    def test_territory_enclosed_area(self):
        """
        GROUND TRUTH: Corner enclosed by black is black territory.
        """
        labeler = ConceptLabeler()
        board = create_enclosed_territory()
        
        territory = labeler.estimate_territory(board)
        
        # Points (0,0), (0,1), (1,0), (1,1) should be black territory
        assert territory['black_territory'][0, 0] > 0.5
        assert territory['black_territory'][0, 1] > 0.5
        assert territory['black_territory'][1, 0] > 0.5
        assert territory['black_territory'][1, 1] > 0.5

    def test_territory_mutual_exclusivity(self):
        """
        INVARIANT: A point cannot be both black and white territory.
        """
        labeler = ConceptLabeler()
        np.random.seed(42)
        board = create_empty_board()
        # Add some stones
        for _ in range(50):
            r, c = np.random.randint(0, 19, 2)
            board[r, c] = np.random.choice([StoneColor.BLACK, StoneColor.WHITE])
        
        territory = labeler.estimate_territory(board)
        
        overlap = (territory['black_territory'] > 0.5) & (territory['white_territory'] > 0.5)
        
        assert overlap.sum() == 0, "Territory cannot belong to both players"


# =============================================================================
# Label Aggregation
# =============================================================================

class TestComputeAllLabels:
    """Tests for label aggregation function."""

    def test_all_labels_present(self):
        """
        INVARIANT: Must compute all expected concept labels.
        """
        labeler = ConceptLabeler()
        board = create_empty_board()
        
        labels = labeler.compute_all_labels(board)
        
        required_labels = [
            'is_edge', 'is_corner', 'is_center', 'is_star_point',
            'has_black_stone', 'has_white_stone', 'is_empty',
            'is_atari', 'is_low_liberty', 'is_cutting_point',
            'black_eye_shape', 'white_eye_shape',
            'black_contact', 'white_contact',
            'black_territory', 'white_territory',
            'is_large_group'
        ]
        
        for label in required_labels:
            assert label in labels, f"Missing required label: {label}"

    def test_all_labels_correct_shape(self):
        """
        INVARIANT: All labels must have shape (19, 19).
        """
        labeler = ConceptLabeler()
        board = create_empty_board()
        
        labels = labeler.compute_all_labels(board)
        
        for name, arr in labels.items():
            assert arr.shape == (19, 19), f"Label '{name}' has wrong shape: {arr.shape}"

    def test_flatten_labels_shape(self):
        """
        INVARIANT: Flattened labels must have shape (361,).
        """
        labeler = ConceptLabeler()
        board = create_empty_board()
        
        labels = labeler.compute_all_labels(board)
        flat = labeler.flatten_labels(labels)
        
        for name, arr in flat.items():
            assert arr.shape == (361,), f"Flattened '{name}' has wrong shape: {arr.shape}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
