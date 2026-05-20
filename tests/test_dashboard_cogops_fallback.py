from entroly.dashboard import _cogops_unavailable_snapshot


def test_cogops_unavailable_snapshot_is_dashboard_safe():
    snap = _cogops_unavailable_snapshot("No module named 'entroly_core'")

    assert snap["engine"] == "unavailable"
    assert snap["status"] == "native_module_missing"
    assert snap["total_beliefs"] == 0
    assert "entroly-core" in snap["hint"]


def test_missing_cogops_core_does_not_surface_dashboard_error(monkeypatch):
    import builtins

    from entroly import dashboard

    class DummyEngine:
        def stats(self):
            return {}

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "entroly_core":
            raise ModuleNotFoundError("No module named 'entroly_core'", name="entroly_core")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(dashboard, "_engine", DummyEngine())

    snap = dashboard._get_full_snapshot()

    assert snap["cogops"]["status"] == "native_module_missing"
    assert not any(error.get("section") == "cogops" for error in snap["errors"])
