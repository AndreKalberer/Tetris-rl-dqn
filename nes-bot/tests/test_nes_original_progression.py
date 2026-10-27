import unittest
from nes_original.progression import after_clear,MarathonProgress


class ProgressionTests(unittest.TestCase):
    def test_native_18_transition_and_level29_completion(self):
        self.assertEqual(after_clear(18,129,1),(19,130))
        self.assertEqual(after_clear(29,239,1),(30,240))

    def test_810_line_level_and_rebirth_are_not_linear(self):
        level=18
        milestones={}
        for line in range(1,6001):
            new,_=after_clear(level,line-1,1)
            if new!=level:
                milestones[line]=new
            level=new
        self.assertEqual(milestones[2290],235)
        self.assertFalse(any(line in milestones for line in range(2300,3100)))
        self.assertEqual(milestones[3100],236)
        self.assertEqual(milestones[3290],255)
        self.assertEqual(milestones[3300],0)
        self.assertEqual(milestones[5190],189)
        self.assertEqual(milestones[6000],190)

    def test_raw_line_counter_wrap_is_separate_from_level_wrap(self):
        self.assertEqual(after_clear(29,25599,1),(29,0))
        tracker=MarathonProgress(255,3299,1000)
        state=tracker.update(0,3300,1001)
        self.assertEqual((state['absolute_level'],state['absolute_lines']),(256,3300))
        self.assertEqual(state['line_counter_cycles'],0)
        tracker=MarathonProgress(29,25599,10000)
        state=tracker.update(29,0,10001)
        self.assertEqual(state['absolute_lines'],25600)
        self.assertEqual(state['level_cycles'],0)

    def test_torn_counter_and_unannounced_reset_are_rejected(self):
        tracker=MarathonProgress(18,99,10)
        with self.assertRaises(ValueError):
            tracker.update(18,199,11)
        with self.assertRaises(ValueError):
            tracker.update(0,0,0)


if __name__=='__main__':
    unittest.main()
