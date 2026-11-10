import unittest
import tempfile
import shutil
from pathlib import Path
from src.database.db_manager import PatientDatabaseManager


class TestPatientDatabaseManager(unittest.TestCase):

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.db = PatientDatabaseManager(db_dir=self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_default_seeding(self):
        patients = self.db.list_patients()
        self.assertGreaterEqual(len(patients), 2)
        p1 = self.db.get_patient("PATIENT_01_TCGA_4944")
        self.assertIsNotNone(p1)
        self.assertEqual(p1["patient_code"], "PAC-2026-4944")

    def test_create_and_delete_patient(self):
        new_p = {
            "patient_id": "TEST_CASE_999",
            "patient_code": "PAC-999",
            "full_name": "Test Paciente",
            "volume_cm3": 50.0,
            "risk_stage": "Grado III"
        }
        res_save = self.db.save_patient(new_p, user_id="MED_PRINCIPAL")
        self.assertTrue(res_save)

        fetched = self.db.get_patient("TEST_CASE_999")
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched["volume_cm3"], 50.0)

        # Delete
        res_del = self.db.delete_patient("TEST_CASE_999", user_id="MED_PRINCIPAL")
        self.assertTrue(res_del)
        self.assertIsNone(self.db.get_patient("TEST_CASE_999"))

    def test_rbac_and_audit(self):
        users = self.db.list_users()
        self.assertGreaterEqual(len(users), 3)

        can_delete = self.db.check_permission("MED_PRINCIPAL", "DELETE")
        self.assertTrue(can_delete)

        logs = self.db.get_audit_logs()
        self.assertGreaterEqual(len(logs), 1)
        self.assertIn("crc32_checksum", logs[0])


if __name__ == "__main__":
    unittest.main()

