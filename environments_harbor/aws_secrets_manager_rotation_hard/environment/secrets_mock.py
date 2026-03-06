import json


def get_secret_value(secret_arn, version_id=None):
    """
    Simulates AWS Secrets Manager GetSecretValue API call.
    Returns a dictionary with SecretString, VersionId, and VersionStages.
    """
    # Default values
    username = "dbuser"
    host = "db.example.com"
    port = 5432
    password = "valid_password"
    version_stages = ["AWSCURRENT"]
    
    # Determine password and version stages based on version_id
    if version_id:
        if 'pending' in version_id.lower() or version_id.endswith('-pending'):
            version_stages = ["AWSPENDING"]
            password = "new_pending_password"
        elif 'invalid' in version_id.lower():
            version_stages = ["AWSPENDING"]
            password = "invalid_password"
        elif 'nolabel' in version_id.lower():
            version_stages = []
            password = "nolabel_password"
        elif 'current' in version_id.lower():
            version_stages = ["AWSCURRENT"]
            password = "valid_password"
        else:
            # Default for unrecognized patterns
            version_stages = ["AWSPENDING"]
            password = "test_password"
    
    # Build secret data structure
    secret_data = {
        "username": username,
        "password": password,
        "host": host,
        "port": port
    }
    
    # Return response in AWS Secrets Manager format
    response = {
        "SecretString": json.dumps(secret_data),
        "VersionId": version_id if version_id else "default-version",
        "VersionStages": version_stages
    }
    
    return response