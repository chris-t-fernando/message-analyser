import os
import sys
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import config


def test_load_db_config(monkeypatch):
    params = {
        'PGHOST': 'localhost',
        'PGPORT': '5432',
        'PGUSER': '"user"',
        'PGPASSWORD': '"pass"',
        'PGDATABASE': '"db"',
    }

    class FakeSSM:
        def get_parameter(self, Name, WithDecryption):
            key = Name.split('/')[-1]
            return {'Parameter': {'Value': params[key]}}

    def fake_client(name):
        assert name == 'ssm'
        return FakeSSM()

    monkeypatch.setattr(config.boto3, 'client', fake_client)

    cfg = config.load_db_config('env')
    assert cfg.host == 'localhost'
    assert cfg.port == 5432
    assert cfg.user == 'user'
    assert cfg.password == 'pass'
    assert cfg.name == 'db'


def test_get_settings(monkeypatch):
    for k in ["DATABASE__HOST","DATABASE__PORT","DATABASE__USER","DATABASE__PASSWORD","DATABASE__NAME"]:
        monkeypatch.delenv(k, raising=False)
    db_cfg = config.DatabaseConfig(host='h', port=1, user='u', password='p', name='n')

    def fake_load(env):
        assert env == 'devtest'
        return db_cfg

    monkeypatch.setattr(config, 'load_db_config', fake_load)

    settings = config.get_settings()
    assert settings.database == db_cfg
