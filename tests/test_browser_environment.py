"""Desktop recovery for service-launched browsers; no network or model training."""
import asyncio
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from scratchllm.crawl import CrawlFailed, browser_environment, browser_session
from scratchllm.progress import CrawlProgress


@unittest.skipUnless(sys.platform == 'linux', 'Linux desktop environment recovery')
class BrowserEnvironmentTests(unittest.TestCase):
    def test_boot_service_recovers_paired_display_and_auth_without_importing_other_settings(self):
        original = {'PATH': '/usr/bin', 'XDG_RUNTIME_DIR': '/run/user/1000', 'XAUTHORITY': '/old/auth', 'HF_TOKEN': 'original'}
        result = subprocess.CompletedProcess([], 0, stdout="DISPLAY=:1\nXAUTHORITY='/desktop session/auth'\nPATH=/other\nHF_TOKEN=other\n")
        def lookup_environment(*args, **kwargs):
            self.assertEqual(kwargs['env'], original)
            return result
        with patch('scratchllm.crawl.subprocess.run', side_effect=lookup_environment) as lookup:
            environment = browser_environment(True, original)
        self.assertEqual(environment['DISPLAY'], ':1')
        self.assertEqual(environment['XAUTHORITY'], '/desktop session/auth')
        self.assertEqual(environment['PATH'], original['PATH'])
        self.assertEqual(environment['HF_TOKEN'], 'original')
        self.assertNotIn('DISPLAY', original)
        self.assertEqual(original['XAUTHORITY'], '/old/auth')
        self.assertEqual(lookup.call_args.args[0], ['systemctl', '--user', 'show-environment'])
        self.assertFalse(lookup.call_args.kwargs.get('shell', False))

    def test_explicit_display_and_headless_launch_do_not_query_desktop(self):
        with patch('scratchllm.crawl.subprocess.run', side_effect=AssertionError('Unexpected desktop lookup')):
            explicit = {'DISPLAY': 'localhost:10.0', 'XAUTHORITY': '/ssh/auth'}
            self.assertEqual(browser_environment(True, explicit), explicit)
            self.assertEqual(browser_environment(False, {}), {})

    def test_wayland_only_session_enables_native_wayland(self):
        result = subprocess.CompletedProcess([], 0, stdout='WAYLAND_DISPLAY=wayland-0\nXDG_RUNTIME_DIR=/run/user/1000\n')
        with patch('scratchllm.crawl.subprocess.run', return_value=result):
            environment = browser_environment(True, {'XAUTHORITY': '/old/auth'})
        self.assertEqual(environment['WAYLAND_DISPLAY'], 'wayland-0')
        self.assertEqual(environment['MOZ_ENABLE_WAYLAND'], '1')
        self.assertNotIn('XAUTHORITY', environment)

    def test_no_desktop_gives_actionable_error_before_launch(self):
        for error in (None, FileNotFoundError(), subprocess.TimeoutExpired('systemctl', 3)):
            with self.subTest(error=error), patch.dict('os.environ', {}, clear=True), \
                    patch('scratchllm.crawl.subprocess.run', side_effect=error,
                          return_value=subprocess.CompletedProcess([], 0, stdout='PATH=/usr/bin\n')), \
                    patch('scratchllm.crawl.AsyncCamoufox') as launch:
                async def attempt():
                    async with browser_session(SimpleNamespace(headed=True, profile='unused'), CrawlProgress(None, 0)):
                        self.fail('Browser opened without a desktop')
                with self.assertRaisesRegex(CrawlFailed, 'uncheck.*Visible browser'):
                    asyncio.run(attempt())
                launch.assert_not_called()

    def test_recovered_environment_reaches_browser_and_text_guard_is_installed(self):
        environment = {'DISPLAY': ':2', 'XAUTHORITY': '/desktop/auth'}
        manager, context = MagicMock(), AsyncMock()
        manager.__aenter__ = AsyncMock(return_value=context)
        manager.__aexit__ = AsyncMock()
        with patch('scratchllm.crawl.browser_environment', return_value=environment), \
                patch('scratchllm.crawl.AsyncCamoufox', return_value=manager) as launch:
            async def attempt():
                async with browser_session(SimpleNamespace(headed=True, profile='unused'), CrawlProgress(None, 0)) as opened:
                    self.assertIs(opened, context)
            asyncio.run(attempt())
        self.assertEqual(launch.call_args.kwargs['env'], environment)
        self.assertFalse(launch.call_args.kwargs['headless'])
        self.assertFalse(launch.call_args.kwargs['accept_downloads'])
        context.route.assert_awaited_once()
        self.assertEqual(context.route.call_args.args[0], '**/*')
        manager.__aexit__.assert_awaited_once()


if __name__ == '__main__':
    unittest.main()
