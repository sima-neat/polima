"""Run on a DevKit: POLIMA_NATIVE_CLI=/path/to/polima_cli python test_cli.py."""
import os
import subprocess
import unittest


@unittest.skipUnless(os.environ.get('POLIMA_NATIVE_CLI'), 'requires a built DevKit CLI')
class NativeCommands(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([os.environ['POLIMA_NATIVE_CLI'], *args], input='',
                              text=True, capture_output=True, timeout=30, check=False)

    def test_version(self):
        result = self.run_cli('--version')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertRegex(result.stdout.strip(), r'^polima \d+\.\d+\.\d+$')

    def test_python_command_help(self):
        for command in ('doctor', 'workbench', 'compile', 'deploy', 'run', 'studio'):
            with self.subTest(command=command):
                result = self.run_cli(command, '--help')
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('usage: polima ' + command, result.stdout)

    def test_unknown_command(self):
        result = self.run_cli('not-a-command')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('unknown command', result.stderr)
        self.assertNotIn('interactive', result.stdout)


if __name__ == '__main__':
    unittest.main()
