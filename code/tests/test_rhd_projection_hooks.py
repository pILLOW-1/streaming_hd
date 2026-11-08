import unittest
from types import SimpleNamespace
from common.native_rhd import NativeRHD


class FakeValue:
    def to(self, device):
        return self

    def float(self):
        return self


class ProjectionHookTests(unittest.TestCase):
    def backend(self, fail=False):
        value = NativeRHD.__new__(NativeRHD)
        value.count = 32
        value.config = SimpleNamespace(final_logit_softcapping=None)
        value.engine = SimpleNamespace(active={28, 32}, spans=[[3, 6]], raw_span=(0, 6), stop=True)
        def norm(hidden):
            self.assertEqual(value.engine.active, set())
            self.assertFalse(value.engine.stop)
            self.assertIsNone(value.engine.raw_span)
            self.assertEqual(value.engine.spans, [])
            if fail:
                raise RuntimeError('independent projection failure')
            return hidden
        value.norm = norm
        value.head = lambda hidden: hidden
        return value

    def test_projection_disables_extraction_then_restores_identical_state(self):
        backend = self.backend()
        original = vars(backend.engine).copy()
        hidden = FakeValue()
        self.assertIs(backend.logits(hidden, 28), hidden)
        self.assertEqual(vars(backend.engine), original)
        self.assertIs(backend.engine.active, original['active'])
        self.assertIs(backend.engine.spans, original['spans'])

    def test_projection_exception_still_restores_capture_and_stop(self):
        backend = self.backend(fail=True)
        original = vars(backend.engine).copy()
        with self.assertRaises(RuntimeError):
            backend.logits(FakeValue(), 28)
        self.assertEqual(vars(backend.engine), original)

    def test_already_final_norm_state_is_not_normalized_twice(self):
        backend = self.backend(fail=True)
        hidden = FakeValue()
        self.assertIs(backend.logits(hidden, 32), hidden)
        self.assertTrue(backend.engine.stop)


if __name__ == '__main__':
    unittest.main()
