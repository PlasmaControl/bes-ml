import yaml
from pathlib import Path


def get_db_url() -> str:
    db_connection_file = Path.home() / 'db.yaml'
    print(f"Getting DB connection data from: {db_connection_file}")
    assert db_connection_file.exists()
    with db_connection_file.open() as yaml_file:
        data = yaml.safe_load(yaml_file)
    for key in [
        'user',
        'password',
        'host',
        'database',
    ]:
        assert key in data, f"File {db_connection_file} is missing key {key}"
    url = f"mysql+pymysql://{data['user']}:{data['password']}@{data['host']}/{data['database']}"
    return url
