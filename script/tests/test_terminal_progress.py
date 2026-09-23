"""轻量终端进度条的离线测试。"""

import io
import unittest

from script.lib.terminal_progress import TerminalProgress


class TerminalProgressTests(unittest.TestCase):
    def test_updates_refresh_one_line_and_finish_with_newline(self):
        stream = io.StringIO()
        with TerminalProgress("Checking", 2, stream=stream, min_interval=0) as progress:
            progress.update(1)
            progress.update(2)

        output = stream.getvalue()
        self.assertEqual(output.count("\r"), 2)
        self.assertEqual(output.count("\n"), 1)
        self.assertIn("50.0% 1/2", output)
        self.assertIn("100.0% 2/2", output)
        self.assertIn("ETA", output)

    def test_exception_closes_progress_line(self):
        stream = io.StringIO()
        with self.assertRaisesRegex(RuntimeError, "failed"):
            with TerminalProgress("Checking", 2, stream=stream) as progress:
                progress.update(1)
                raise RuntimeError("failed")
        self.assertTrue(stream.getvalue().endswith("\n"))


if __name__ == "__main__":
    unittest.main()
