def test_package_imports():
    import jbct

    assert isinstance(jbct.__version__, str)
