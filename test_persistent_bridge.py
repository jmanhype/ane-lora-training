"""Tests for persistent ANE bridge implementation."""

import os
import multiprocessing as mp
import numpy as np
import pytest
import time
from unittest.mock import Mock, patch, MagicMock

# Import the module under test
import sys
sys.path.insert(0, '/Users/speed/ane-lora-training')

from ane_lora_kernels import (
    PersistentANEBridge,
    _pad_spatial,
    ANE_SPATIAL_ALIGN,
    ANE_SPATIAL_MIN,
)


class TestSpatialPadding:
    """Test spatial dimension padding for ANE constraints."""

    def test_pad_spatial_below_minimum(self):
        """Values below ANE_SPATIAL_MIN (16) should round up to 16."""
        assert _pad_spatial(1) == 16
        assert _pad_spatial(8) == 16
        assert _pad_spatial(15) == 16

    def test_pad_spatial_at_minimum(self):
        """Value exactly at minimum should remain unchanged."""
        assert _pad_spatial(16) == 16

    def test_pad_spatial_alignment(self):
        """Values should be aligned to ANE_SPATIAL_ALIGN (16) boundary."""
        assert _pad_spatial(17) == 32
        assert _pad_spatial(24) == 32
        assert _pad_spatial(32) == 32
        assert _pad_spatial(48) == 48  # Already aligned

    def test_pad_spatial_large_values(self):
        """Large values should still align correctly."""
        assert _pad_spatial(2048) == 2048
        assert _pad_spatial(2050) == 2064


class TestPersistentANEBridgeInit:
    """Test PersistentANEBridge initialization and lifecycle."""

    @patch('ane_lora_kernels.mp.Process')
    @patch('ane_lora_kernels.mp.Event')
    def test_init_creates_worker_process(self, mock_event, mock_process):
        """Bridge should spawn a worker process on initialization."""
        mock_proc = Mock()
        mock_proc.start = Mock()
        mock_process.return_value = mock_proc

        # Mock the ready event to avoid timeout
        mock_event_instance = Mock()
        mock_event_instance.wait = Mock(return_value=True)
        mock_event.return_value = mock_event_instance

        bridge = PersistentANEBridge("/fake/path.dylib")

        # Verify process was created and started
        mock_process.assert_called_once()
        mock_proc.start.assert_called_once()

    @patch('ane_lora_kernels.mp.Process')
    @patch('ane_lora_kernels.mp.Event')
    def test_init_timeout_on_worker_not_ready(self, mock_event, mock_process):
        """Should raise RuntimeError if worker fails to start within timeout."""
        mock_proc = Mock()
        mock_proc.start = Mock()
        mock_process.return_value = mock_proc

        # Mock ready_event to timeout
        mock_event_instance = Mock()
        mock_event_instance.wait = Mock(return_value=False)
        mock_event.return_value = mock_event_instance

        with pytest.raises(RuntimeError, match="failed to start"):
            PersistentANEBridge("/fake/path.dylib")

    @patch('ane_lora_kernels.mp.Process')
    @patch('ane_lora_kernels.mp.Event')
    def test_init_creates_communication_channels(self, mock_event, mock_process):
        """Bridge should create Queue and Event for communication."""
        mock_event_instance = Mock()
        mock_event_instance.wait = Mock(return_value=True)
        mock_event.return_value = mock_event_instance

        bridge = PersistentANEBridge("/fake/path.dylib")

        assert hasattr(bridge, '_cmd_queue')
        assert hasattr(bridge, '_result_queue')
        assert hasattr(bridge, '_ready_event')


class TestPersistentANEBridgeCompute:
    """Test gradient computation via persistent bridge."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create sample LoRA module data
        self.seq_len = 64
        self.in_dim = 768
        self.out_dim = 768
        self.rank = 8

        self.dy = np.random.randn(self.seq_len, self.out_dim).astype(np.float32)
        self.x = np.random.randn(self.seq_len, self.in_dim).astype(np.float32)
        self.lora_a = np.random.randn(self.in_dim, self.rank).astype(np.float32)
        self.lora_b = np.random.randn(self.rank, self.out_dim).astype(np.float32)

        self.modules = [(self.dy, self.x, self.lora_a, self.lora_b)]

    @patch('ane_lora_kernels.shared_memory.SharedMemory')
    def test_compute_creates_shared_memory_buffers(self, mock_shm):
        """Compute should create shared memory buffers for inputs."""
        mock_shm_instance = Mock()
        mock_shm_instance.name = "test_buffer"
        mock_shm_instance.buf = bytearray(self.dy.nbytes)
        mock_shm.return_value = mock_shm_instance

        bridge = Mock()
        bridge._cmd_queue = mp.Queue()
        bridge._result_queue = mp.Queue()
        bridge._total_compiles = 0
        bridge._total_dispatches = 0
        bridge._total_steps = 0

        # Mock result queue to return success
        bridge._result_queue.put({
            "status": "ok",
            "d_a_shm": "d_a_test",
            "d_b_shm": "d_b_test",
            "d_a_shape": (self.in_dim, self.rank),
            "d_b_shape": (self.rank, self.out_dim),
            "compiles": 4,
            "dispatches": 4
        })

        # Call the real compute method
        from ane_lora_kernels import PersistentANEBridge
        result = PersistentANEBridge.compute_lora_gradients(bridge, self.modules)

        # Verify shared memory was created for 6 buffers (4 input + 2 output)
        assert mock_shm.call_count == 6

    @patch('ane_lora_kernels.shared_memory.SharedMemory')
    def test_compute_sends_correct_command(self, mock_shm):
        """Compute should send properly formatted command to worker."""
        mock_shm_instance = Mock()
        mock_shm_instance.name = "test_buffer"
        mock_shm_instance.buf = bytearray(self.dy.nbytes)
        mock_shm.return_value = mock_shm_instance

        bridge = Mock()
        bridge._cmd_queue = mp.Queue()
        bridge._result_queue = mp.Queue()
        bridge._total_compiles = 0
        bridge._total_dispatches = 0
        bridge._total_steps = 0

        # Mock result queue to return success
        bridge._result_queue.put({
            "status": "ok",
            "d_a_shm": "d_a_test",
            "d_b_shm": "d_b_test",
            "d_a_shape": (self.in_dim, self.rank),
            "d_b_shape": (self.rank, self.out_dim),
            "compiles": 4,
            "dispatches": 4
        })

        from ane_lora_kernels import PersistentANEBridge
        PersistentANEBridge.compute_lora_gradients(bridge, self.modules)

        # Verify command was sent
        assert not bridge._cmd_queue.empty()

    @patch('ane_lora_kernels.shared_memory.SharedMemory')
    def test_compute_raises_on_worker_error(self, mock_shm):
        """Compute should raise RuntimeError if worker returns error."""
        mock_shm_instance = Mock()
        mock_shm_instance.name = "test_buffer"
        mock_shm_instance.buf = bytearray(self.dy.nbytes)
        mock_shm.return_value = mock_shm_instance

        bridge = Mock()
        bridge._cmd_queue = mp.Queue()
        bridge._result_queue = mp.Queue()
        bridge._total_compiles = 0
        bridge._total_dispatches = 0
        bridge._total_steps = 0

        # Mock result queue to return error
        bridge._result_queue.put({
            "status": "error",
            "msg": "Test error message"
        })

        from ane_lora_kernels import PersistentANEBridge

        with pytest.raises(RuntimeError, match="Test error message"):
            PersistentANEBridge.compute_lora_gradients(bridge, self.modules)

    @patch('ane_lora_kernels.shared_memory.SharedMemory')
    def test_compute_pads_sequence_to_ane_constraints(self, mock_shm):
        """Compute should pad sequence to meet ANE spatial constraints."""
        # Use sequence length that needs padding
        short_seq = 10
        dy = np.random.randn(short_seq, self.out_dim).astype(np.float32)
        x = np.random.randn(short_seq, self.in_dim).astype(np.float32)
        modules = [(dy, x, self.lora_a, self.lora_b)]

        mock_shm_instance = Mock()
        mock_shm_instance.name = "test_buffer"
        # Buffer should be sized for padded sequence
        expected_padded = _pad_spatial(short_seq)
        mock_shm_instance.buf = bytearray(expected_padded * self.out_dim * 4)
        mock_shm.return_value = mock_shm_instance

        bridge = Mock()
        bridge._cmd_queue = mp.Queue()
        bridge._result_queue = mp.Queue()
        bridge._total_compiles = 0
        bridge._total_dispatches = 0
        bridge._total_steps = 0

        bridge._result_queue.put({
            "status": "ok",
            "d_a_shm": "d_a_test",
            "d_b_shm": "d_b_test",
            "d_a_shape": (self.in_dim, self.rank),
            "d_b_shape": (self.rank, self.out_dim),
            "compiles": 4,
            "dispatches": 4
        })

        from ane_lora_kernels import PersistentANEBridge
        PersistentANEBridge.compute_lora_gradients(bridge, modules)

        # Verify padded_seq was passed in command
        cmd = bridge._cmd_queue.get()
        assert cmd["padded_seq"] == expected_padded


class TestPersistentANEBridgeStats:
    """Test statistics tracking."""

    @patch('ane_lora_kernels.mp.Process')
    @patch('ane_lora_kernels.mp.Event')
    def test_stats_counters_initialized_to_zero(self, mock_event, mock_process):
        """Stats counters should start at zero."""
        mock_event_instance = Mock()
        mock_event_instance.wait = Mock(return_value=True)
        mock_event.return_value = mock_event_instance

        bridge = PersistentANEBridge("/fake/path.dylib")

        assert bridge.total_compiles == 0
        assert bridge.total_dispatches == 0
        assert bridge.total_steps == 0


class TestPersistentANEBridgeShutdown:
    """Test clean shutdown of persistent bridge."""

    @patch('ane_lora_kernels.mp.Process')
    @patch('ane_lora_kernels.mp.Event')
    def test_shutdown_sends_command_to_worker(self, mock_event, mock_process):
        """Shutdown should send shutdown command to worker process."""
        mock_proc = Mock()
        mock_proc.start = Mock()
        mock_proc.is_alive = Mock(return_value=True)
        mock_process.return_value = mock_proc

        mock_event_instance = Mock()
        mock_event_instance.wait = Mock(return_value=True)
        mock_event.return_value = mock_event_instance

        bridge = PersistentANEBridge("/fake/path.dylib")

        # Mock result queue to acknowledge shutdown
        bridge._result_queue.put({"status": "shutdown_ack"})

        bridge.shutdown()

        # Verify shutdown command was sent
        assert not bridge._cmd_queue.empty()
        cmd = bridge._cmd_queue.get()
        assert cmd["cmd"] == "shutdown"


class TestIntegrationScenarios:
    """Integration tests for common usage scenarios."""

    def test_multiple_sequential_compute_calls(self):
        """Test that multiple sequential compute calls work correctly."""
        # This would require a real or well-mocked worker
        # Placeholder for future integration test
        pass

    def test_compute_with_multiple_modules(self):
        """Test computing gradients for multiple LoRA modules at once."""
        # This would test the batch processing capability
        # Placeholder for future integration test
        pass


# ---------------------------------------------------------------------------
# Integration Tests (require real ANE bridge)
# ---------------------------------------------------------------------------

def find_bridge_path():
    """Find libane_bridge.dylib in standard locations."""
    candidates = [
        "./libane_bridge.dylib",
        "./build/libane_bridge.dylib",
        "../build/libane_bridge.dylib",
        os.path.expanduser("~/Projects/ane-lora-training/build/libane_bridge.dylib"),
        "/Users/speed/Projects/ane-lora-training/build/libane_bridge.dylib",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


@pytest.mark.integration
class TestRealPersistentBridge:
    """Integration tests requiring actual ANE bridge."""

    @pytest.fixture(autouse=True)
    def check_bridge_available(self):
        """Skip these tests if bridge is not available."""
        self.bridge_path = find_bridge_path()
        if self.bridge_path is None:
            pytest.skip("libane_bridge.dylib not found")

    @pytest.fixture
    def bridge(self):
        """Create a real bridge instance for testing."""
        b = PersistentANEBridge(self.bridge_path)
        yield b
        b.shutdown()

    def test_init_real_bridge(self, bridge):
        """Test that real bridge initializes successfully."""
        assert bridge._proc.is_alive()
        assert bridge.total_compiles == 0
        assert bridge.total_dispatches == 0
        assert bridge.total_steps == 0

    def test_shutdown_real_bridge(self, bridge):
        """Test clean shutdown of real bridge."""
        assert bridge._proc.is_alive()
        bridge.shutdown()
        assert not bridge._proc.is_alive()

    def test_double_shutdown_safe(self, bridge):
        """Test that double shutdown is safe."""
        bridge.shutdown()
        bridge.shutdown()  # Should not raise

    def _numpy_lora_gradient(self, dy, x, lora_a, lora_b):
        """Reference numpy implementation for validation."""
        tmp = dy @ lora_b.T
        d_lora_a = x.T @ tmp
        ax = x @ lora_a
        d_lora_b = ax.T @ dy
        return d_lora_a, d_lora_b

    @pytest.mark.parametrize("seq,out_dim,in_dim,rank", [
        (16, 16, 16, 8),
        (32, 32, 32, 16),
        (64, 64, 64, 8),
    ])
    def test_gradient_correctness_vs_numpy(self, bridge, seq, out_dim, in_dim, rank):
        """Test that ANE gradients match numpy reference."""
        np.random.seed(42)

        # Create test data
        dy = np.random.randn(seq, out_dim).astype(np.float32) * 0.1
        x = np.random.randn(seq, in_dim).astype(np.float32) * 0.1
        lora_a = np.random.randn(in_dim, rank).astype(np.float32) * 0.01
        lora_b = np.random.randn(rank, out_dim).astype(np.float32) * 0.01

        # Compute with ANE
        modules = [(dy, x, lora_a, lora_b)]
        ane_results = bridge.compute_lora_gradients(modules)
        d_a_ane, d_b_ane = ane_results[0]

        # Compute with numpy
        d_a_np, d_b_np = self._numpy_lora_gradient(dy, x, lora_a, lora_b)

        # Compare (allow fp16 tolerance from ANE fp16 computation)
        rtol = 1e-3
        atol = 1e-5

        np.testing.assert_allclose(d_a_ane, d_a_np, rtol=rtol, atol=atol,
                                   err_msg="d_lora_a mismatch")
        np.testing.assert_allclose(d_b_ane, d_b_np, rtol=rtol, atol=atol,
                                   err_msg="d_lora_b mismatch")

    def test_gradient_correctness_with_padding(self, bridge):
        """Test gradients with sequences requiring padding (not multiple of 16)."""
        np.random.seed(43)

        # Sequence that requires padding (10 -> 16)
        seq = 10
        out_dim = 16
        in_dim = 16
        rank = 8

        dy = np.random.randn(seq, out_dim).astype(np.float32) * 0.1
        x = np.random.randn(seq, in_dim).astype(np.float32) * 0.1
        lora_a = np.random.randn(in_dim, rank).astype(np.float32) * 0.01
        lora_b = np.random.randn(rank, out_dim).astype(np.float32) * 0.01

        modules = [(dy, x, lora_a, lora_b)]
        ane_results = bridge.compute_lora_gradients(modules)
        d_a_ane, d_b_ane = ane_results[0]

        # Numpy reference
        d_a_np, d_b_np = self._numpy_lora_gradient(dy, x, lora_a, lora_b)

        # Should still match despite padding
        np.testing.assert_allclose(d_a_ane, d_a_np, rtol=1e-3, atol=1e-5,
                                   err_msg="d_lora_a mismatch with padding")
        np.testing.assert_allclose(d_b_ane, d_b_np, rtol=1e-3, atol=1e-5,
                                   err_msg="d_lora_b mismatch with padding")

    def test_multiple_modules(self, bridge):
        """Test gradient computation for multiple modules in one call."""
        np.random.seed(44)

        modules = []
        for i in range(3):
            seq = 16 + i * 8
            out_dim = 16
            in_dim = 16
            rank = 8

            dy = np.random.randn(seq, out_dim).astype(np.float32) * 0.1
            x = np.random.randn(seq, in_dim).astype(np.float32) * 0.1
            lora_a = np.random.randn(in_dim, rank).astype(np.float32) * 0.01
            lora_b = np.random.randn(rank, out_dim).astype(np.float32) * 0.01

            modules.append((dy, x, lora_a, lora_b))

        # Compute with ANE
        ane_results = bridge.compute_lora_gradients(modules)

        # Verify each module
        for i, (dy, x, lora_a, lora_b) in enumerate(modules):
            d_a_ane, d_b_ane = ane_results[i]
            d_a_np, d_b_np = self._numpy_lora_gradient(dy, x, lora_a, lora_b)

            np.testing.assert_allclose(d_a_ane, d_a_np, rtol=1e-3, atol=1e-5,
                                       err_msg=f"Module {i}: d_lora_a mismatch")
            np.testing.assert_allclose(d_b_ane, d_b_np, rtol=1e-3, atol=1e-5,
                                       err_msg=f"Module {i}: d_lora_b mismatch")

    def test_sequential_steps(self, bridge):
        """Test multiple sequential training steps."""
        np.random.seed(45)

        n_steps = 5
        for step in range(n_steps):
            dy = np.random.randn(16, 16).astype(np.float32) * 0.1
            x = np.random.randn(16, 16).astype(np.float32) * 0.1
            lora_a = np.random.randn(16, 8).astype(np.float32) * 0.01
            lora_b = np.random.randn(8, 16).astype(np.float32) * 0.01

            modules = [(dy, x, lora_a, lora_b)]
            results = bridge.compute_lora_gradients(modules)

            assert len(results) == 1
            d_a, d_b = results[0]
            assert d_a.shape == (16, 8)
            assert d_b.shape == (8, 16)

        # Verify stats
        assert bridge.total_steps == n_steps
        # Each step does 4 compiles (tmp, d_a, ax, d_b)
        assert bridge.total_compiles == n_steps * 4
        assert bridge.total_dispatches == n_steps * 4

    def test_stats_tracking(self, bridge):
        """Test that compile/dispatch/step stats are tracked correctly."""
        np.random.seed(46)

        # First step with 2 modules
        modules = []
        for _ in range(2):
            dy = np.random.randn(16, 16).astype(np.float32) * 0.1
            x = np.random.randn(16, 16).astype(np.float32) * 0.1
            lora_a = np.random.randn(16, 8).astype(np.float32) * 0.01
            lora_b = np.random.randn(8, 16).astype(np.float32) * 0.01
            modules.append((dy, x, lora_a, lora_b))

        bridge.compute_lora_gradients(modules)

        assert bridge.total_steps == 1
        assert bridge.total_compiles == 8  # 2 modules * 4 compiles
        assert bridge.total_dispatches == 8


@pytest.mark.integration
class TestSharedMemory:
    """Integration tests for SharedMemory functionality."""

    def test_shared_memory_roundtrip(self):
        """Test that shared memory transfers data correctly."""
        # Create test array
        original = np.random.randn(16, 32).astype(np.float32)

        # Create shared memory
        from multiprocessing import shared_memory
        shm = shared_memory.SharedMemory(create=True, size=original.nbytes)

        # Write
        arr = np.ndarray(original.shape, dtype=np.float32, buffer=shm.buf)
        np.copyto(arr, original)

        # Read back
        copy = arr.copy()

        # Verify
        np.testing.assert_array_equal(original, copy)

        # Cleanup
        shm.close()
        shm.unlink()

    def test_shared_memory_padding(self):
        """Test shared memory with padded spatial dimensions."""
        from multiprocessing import shared_memory

        seq = 10
        out_dim = 16
        padded_seq = _pad_spatial(seq)  # Should be 16

        # Create padded array
        padded = np.zeros((padded_seq, out_dim), dtype=np.float32)
        original = np.random.randn(seq, out_dim).astype(np.float32)
        padded[:seq] = original

        # Create shared memory
        shm = shared_memory.SharedMemory(create=True, size=padded.nbytes)
        arr = np.ndarray(padded.shape, dtype=np.float32, buffer=shm.buf)
        np.copyto(arr, padded)

        # Read back and trim
        read_back = arr.copy()
        trimmed = read_back[:seq]

        # Verify trimmed matches original
        np.testing.assert_array_equal(original, trimmed)

        shm.close()
        shm.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
