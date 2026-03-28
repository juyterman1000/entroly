import asyncio
import os
import shutil
from pathlib import Path


def _profile_candidates() -> list[Path]:
    if os.name == "nt":
        local_appdata = os.environ.get("LOCALAPPDATA")
        if not local_appdata:
            return []
        base = Path(local_appdata)
        return [
            base / "Google" / "Chrome" / "User Data",
            base / "Chromium" / "User Data",
        ]

    home = Path.home()
    return [
        home / ".config" / "google-chrome",
        home / ".config" / "chromium",
    ]


async def run() -> None:
    from playwright.async_api import async_playwright

    src = next((path for path in _profile_candidates() if path.exists()), None)
    if src is None:
        raise RuntimeError("No local Chrome/Chromium profile found for Playwright auth smoke test.")

    dst = src.parent / f"{src.name}-bot"
    if dst.exists():
        shutil.rmtree(dst)

    try:
        shutil.copytree(src, dst, ignore_dangling_symlinks=True)
    except Exception as exc:
        print(f"Warning on copy: {exc}")

    async with async_playwright() as p:
        browser = await p.chromium.launch_persistent_context(
            user_data_dir=str(dst),
            headless=True,
        )
        page = await browser.new_page()
        print("Navigating to HN...")
        await page.goto("https://news.ycombinator.com/submit")
        await page.screenshot(path="hn_auth_test.png")
        print("Taking HN screenshot.")

        await page.goto("https://old.reddit.com/r/LocalLLaMA/submit")
        await page.screenshot(path="reddit_auth_test.png")
        print("Taking Reddit screenshot.")

        await browser.close()


if __name__ == "__main__":
    asyncio.run(run())
