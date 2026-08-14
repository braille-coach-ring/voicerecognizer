import re
import unittest

from voicerecognizer.utils.machine_id import get_machine_id


class TestMachineId(unittest.TestCase):
    def test_get_machine_id_format(self):
        machine_id = get_machine_id()
        self.assertTrue(machine_id.startswith("pc_"))
        self.assertTrue(re.match(r"^pc_[0-9a-f]{8}$", machine_id))

    def test_get_machine_id_consistency(self):
        id1 = get_machine_id()
        id2 = get_machine_id()
        self.assertEqual(id1, id2)


if __name__ == "__main__":
    unittest.main()
