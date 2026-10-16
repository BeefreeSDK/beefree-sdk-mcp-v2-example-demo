import unittest

from app.sessions import ExpiringStore


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class ExpiringStoreTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.store = ExpiringStore(ttl_seconds=3600, sweep_every=60, clock=self.clock)

    def test_entry_expires_after_inactivity(self):
        self.store["a"] = {"x": 1}
        self.clock.now += 3599
        self.assertEqual(self.store.get("a"), {"x": 1})
        self.clock.now += 3601
        self.assertIsNone(self.store.get("a"))
        self.assertNotIn("a", self.store)

    def test_reading_keeps_an_entry_alive(self):
        self.store["a"] = 1
        for _ in range(5):
            self.clock.now += 3000
            self.assertEqual(self.store["a"], 1)

    def test_sweep_frees_memory_for_entries_never_read_again(self):
        for i in range(100):
            self.store[f"s{i}"] = i
        self.clock.now += 3700
        self.store["fresh"] = "new"  # any access triggers the sweep
        self.assertEqual(len(self.store._data), 1)

    def test_behaves_like_a_dict(self):
        self.store["a"] = 1
        self.assertEqual(self.store.pop("a"), 1)
        self.assertIsNone(self.store.pop("missing", None))
        self.store.update({"b": 2})
        self.assertEqual(self.store.copy(), {"b": 2})
        self.store.clear()
        self.assertEqual(len(self.store), 0)


if __name__ == "__main__":
    unittest.main()
