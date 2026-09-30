"""分布式训练辅助逻辑的离线回归测试。"""

import os
import sys
from pathlib import Path
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from script.lib import distributed_training as distributed
from script.lib import finetune_deepseek_mira_lora, finetune_qwen3_vl_lora


class DistributedTrainingTests(unittest.TestCase):
    def test_local_rank_argument_sets_environment_for_legacy_launcher(self):
        with patch.dict(os.environ, {"LOCAL_RANK": "1"}, clear=False):
            distributed.set_local_rank_from_argument(1)
            self.assertEqual(distributed.local_rank(), 1)

    def test_local_rank_argument_rejects_conflicting_environment(self):
        with patch.dict(os.environ, {"LOCAL_RANK": "0"}, clear=False):
            with self.assertRaisesRegex(ValueError, "不一致"):
                distributed.set_local_rank_from_argument(1)

    def test_backend_parsers_accept_torchrun_local_rank_spellings(self):
        for backend in (finetune_deepseek_mira_lora, finetune_qwen3_vl_lora):
            with self.subTest(backend=backend.__name__):
                for option in ("--local-rank", "--local_rank"):
                    args = backend.build_parser().parse_args([option, "1"])
                    self.assertEqual(args.local_rank, 1)

    def test_too_many_local_workers_are_rejected_before_process_group_setup(self):
        with self.assertRaisesRegex(RuntimeError, "LOCAL_WORLD_SIZE=2"):
            distributed.validate_local_gpu_count(local_size=2, device_count=1)
        distributed.validate_local_gpu_count(local_size=2, device_count=2)

    def test_single_process_main_action_runs_and_returns_its_result(self):
        with patch.object(distributed, "is_distributed", return_value=False):
            result = distributed.run_on_main_process(lambda: "saved", "test action")
        self.assertEqual(result, "saved")

    def test_distributed_main_action_failure_is_broadcast_as_an_error(self):
        with patch.object(distributed, "is_distributed", return_value=True), \
                patch.object(distributed, "is_main_process", return_value=True), \
                patch.object(distributed, "broadcast_from_main", side_effect=lambda value: value):
            with self.assertRaisesRegex(RuntimeError, "写入失败"):
                distributed.run_on_main_process(
                    lambda: (_ for _ in ()).throw(ValueError("写入失败")),
                    "测试文件写入",
                )


if __name__ == "__main__":
    unittest.main()
